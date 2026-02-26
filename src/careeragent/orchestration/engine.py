
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import httpx

from careeragent.config import artifacts_root, get_settings
from careeragent.core.state import AgentState
from careeragent.services.db_service import SqliteStateStore
from careeragent.services.health_service import HealthService
from careeragent.services.notification_service import NotificationService
from careeragent.services.learning_resource_service import LearningResourceService

from careeragent.agents.security_agent import SanitizeAgent
from careeragent.agents.parser_agent_service import ParserAgentService, ExtractedResume
from careeragent.agents.parser_evaluator_service import ParserEvaluatorService


@dataclass(frozen=True)
class JobBoard:
    name: str
    domain: str


DEFAULT_JOB_BOARDS: Tuple[JobBoard, ...] = (
    JobBoard("LinkedIn Jobs", "linkedin.com/jobs"),
    JobBoard("Indeed", "indeed.com"),
    JobBoard("Glassdoor", "glassdoor.com"),
    JobBoard("ZipRecruiter", "ziprecruiter.com"),
    JobBoard("Monster", "monster.com"),
    JobBoard("Dice", "dice.com"),
    JobBoard("Lever", "jobs.lever.co"),
    JobBoard("Greenhouse", "boards.greenhouse.io"),
)

VISA_NEGATIVE = ("unable to sponsor","cannot sponsor","no sponsorship","do not sponsor","not sponsor","without sponsorship","no visa","cannot provide visa")
VISA_POSITIVE = ("visa sponsorship","h1b","opt","cpt","stem opt","work visa")

COMMON_SKILL_DICTIONARY = [
    "python","sql","fastapi","docker","kubernetes","mlflow","dvc","aws","azure","gcp","pytorch",
    "tensorflow","scikit-learn","pandas","numpy","spark","databricks","snowflake","airflow","kafka",
    "langchain","langgraph","rag","faiss","chroma","terraform","github actions","postgres","redis"
]


def _run_dir(run_id: str) -> Path:
    d = artifacts_root() / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


class LiveFeed:
    @staticmethod
    def emit(st: AgentState, *, layer: str, agent: str, message: str) -> None:
        st.meta.setdefault("live_feed", [])
        st.meta["live_feed"].append({"layer": layer, "agent": agent, "message": message})
        st.touch()


class LocalResumeExtractor:
    @staticmethod
    def extract_text(*, filename: str, data: bytes) -> str:
        name = (filename or "").lower()
        if name.endswith(".txt"):
            return data.decode("utf-8", errors="replace")
        if name.endswith(".pdf"):
            from pypdf import PdfReader  # type: ignore
            import io
            reader = PdfReader(io.BytesIO(data))
            return "\n".join([(pg.extract_text() or "") for pg in reader.pages])
        if name.endswith(".docx"):
            import docx  # type: ignore
            import io
            d = docx.Document(io.BytesIO(data))
            return "\n".join([p.text for p in d.paragraphs if p.text])
        return data.decode("utf-8", errors="replace")


class SerperDiscovery:
    SERPER_URL = "https://google.serper.dev/search"

    def __init__(self, *, api_key: str, health: HealthService) -> None:
        self._key = api_key
        self._health = health

    def search(self, *, st: AgentState, step_id: str, query: str, num: int = 10, tbs: Optional[str] = None) -> List[Dict[str, Any]]:
        headers = {"X-API-KEY": self._key, "Content-Type": "application/json"}
        body: Dict[str, Any] = {"q": query, "num": num}
        if tbs:
            body["tbs"] = tbs
        with httpx.Client(timeout=30.0) as client:
            r = client.post(self.SERPER_URL, headers=headers, json=body)

        if self._health.quota.handle_serper_response(
            state=st, step_id=step_id, status_code=r.status_code, tool_name="serper.search", error_detail=r.text[:200]
        ):
            return []

        if r.status_code >= 400:
            st.status = "api_failure"
            st.meta["run_failure_code"] = "API_FAILURE"
            st.meta["run_failure_provider"] = "serper"
            st.meta["run_failure_detail"] = r.text[:200]
            return []

        organic = (r.json().get("organic") or [])
        out = []
        for it in organic:
            out.append({"title": it.get("title") or "", "link": it.get("link") or "", "snippet": it.get("snippet") or ""})
        return out


class Scraper:
    @staticmethod
    def fetch_text(*, url: str, snippet: str) -> str:
        if not url:
            return snippet or ""
        try:
            with httpx.Client(timeout=18.0, follow_redirects=True) as client:
                r = client.get(url, headers={"User-Agent":"Mozilla/5.0"})
            if r.status_code >= 400:
                return snippet or ""
            html = r.text
            txt = re.sub(r"<(script|style)[^>]*>.*?</\\1>", " ", html, flags=re.S|re.I)
            txt = re.sub(r"<[^>]+>", " ", txt)
            txt = re.sub(r"\\s+", " ", txt).strip()
            return txt[:16000] if txt else (snippet or "")
        except Exception:
            return snippet or ""


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9\\+\\#\\.-]{1,}", (text or "").lower())


def cosine(a: Dict[str,int], b: Dict[str,int]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k,0) for k,v in a.items())
    na = sum(v*v for v in a.values()) ** 0.5
    nb = sum(v*v for v in b.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return float(dot/(na*nb))


def recency_hours(snippet: str) -> Optional[float]:
    s = (snippet or "").lower()
    if "today" in s:
        return 6.0
    if "yesterday" in s:
        return 24.0
    m = re.search(r"(\\d+)\\s*hours?\\s*ago", s)
    if m:
        return float(m.group(1))
    m = re.search(r"(\\d+)\\s*days?\\s*ago", s)
    if m:
        return float(m.group(1))*24.0
    return None


def ats_score(resume_text: str) -> float:
    t = (resume_text or "").lower()
    score = 0.0
    if re.search(r"[\\w\\.-]+@[\\w\\.-]+\\.\\w+", t): score += 0.20
    if re.search(r"\\+?\\d[\\d\\-\\s\\(\\)]{8,}\\d", t): score += 0.10
    for h in ["summary","skills","experience","education","projects"]:
        if h in t: score += 0.12
    if "-" in resume_text or "•" in resume_text: score += 0.10
    if len(resume_text) > 1200: score += 0.12
    return max(0.0, min(1.0, score))


def compute_interview_chance(skill_overlap: float, exp_align: float, ats_s: float, market: float) -> float:
    market = max(1.0, float(market))
    score = (0.45*skill_overlap + 0.35*exp_align + 0.20*ats_s) / market
    return max(0.0, min(1.0, float(score)))


def extract_job_skills(job_text: str, resume_skills: List[str]) -> List[str]:
    low = (job_text or "").lower()
    pool = list(dict.fromkeys([*(resume_skills or []), *COMMON_SKILL_DICTIONARY]))
    found = []
    for s in pool:
        s2 = str(s).lower().strip()
        if s2 and s2 in low:
            found.append(s2)
    return list(dict.fromkeys(found))[:40]


class OneClickAutomationEngine:
    """
    Description: Full Career Operating System loop (L0-L9).
    Layer: L0-L9
    """

    def __init__(self) -> None:
        self._s = get_settings()
        self._store = SqliteStateStore()
        self._health = HealthService()
        self._notifier = NotificationService(dry_run=not bool(getattr(self._s, "ntfy_topic", None)))
        self._sanitize = SanitizeAgent()
        self._parser = ParserAgentService()
        self._parser_eval = ParserEvaluatorService()
        self._learn = LearningResourceService(serper_api_key=getattr(self._s, "serper_api_key", None))

    def _persist(self, st: AgentState) -> None:
        d = _run_dir(st.run_id)
        _save_json(d / "state.json", st.model_dump())
        self._store.upsert_state(run_id=st.run_id, status=st.status, state=st.model_dump(), updated_at_utc=st.updated_at_utc)

    def load(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get_state(run_id=run_id)

    def start_run(self, *, filename: str, data: bytes, prefs: Dict[str, Any]) -> AgentState:
        st = AgentState.new(env=self._s.environment, mode="agentic", git_sha=None)
        st.meta["preferences"] = prefs
        st.meta.setdefault("job_scores", {})
        st.meta.setdefault("job_components", {})
        st.meta.setdefault("job_meta", {})
        LiveFeed.emit(st, layer="L1", agent="Dashboard", message="Run created. Starting autonomous pipeline…")
        self._persist(st)
        t = threading.Thread(target=self._run, args=(st.run_id, filename, data), daemon=True)
        t.start()
        return st

    def submit_action(self, *, run_id: str, action_type: str, payload: Dict[str, Any]) -> AgentState:
        raw = self.load(run_id)
        if not raw:
            raise ValueError("run_id not found")
        st = AgentState(**raw)
        st.meta["last_user_action"] = {"type": action_type, "payload": payload}
        LiveFeed.emit(st, layer="L5", agent="HITL", message=f"User action received: {action_type}")
        self._persist(st)
        t = threading.Thread(target=self._continue, args=(run_id,), daemon=True)
        t.start()
        return st

    # ---------------- main loop ----------------
    def _run(self, run_id: str, filename: str, data: bytes) -> None:
        raw = self.load(run_id)
        if not raw:
            return
        st = AgentState(**raw)
        run_dir = _run_dir(run_id)
        prefs = st.meta.get("preferences", {}) or {}

        # L2 extract
        st.start_step("l2_extract", layer_id="L2", tool_name="ResumeExtractor", input_ref={"filename": filename})
        LiveFeed.emit(st, layer="L2", agent="ParserAgent", message="Extracting resume text from upload…")
        resume_text = LocalResumeExtractor.extract_text(filename=filename, data=data)
        (run_dir / "resume_raw.txt").write_text(resume_text, encoding="utf-8")
        st.add_artifact("resume_raw", str(run_dir / "resume_raw.txt"), content_type="text/plain")
        st.end_step("l2_extract", status="ok", output_ref={"artifact_key":"resume_raw"}, message="extracted")

        # L0 sanitize
        st.start_step("l0_sanitize", layer_id="L0", tool_name="SanitizeAgent", input_ref={})
        safe = self._sanitize.sanitize_before_llm(
            state=st, step_id="l0_sanitize", tool_name="sanitize_before_llm", user_text=resume_text, context="resume"
        )
        if safe is None:
            st.status = "blocked"
            LiveFeed.emit(st, layer="L0", agent="SanitizeAgent", message="Prompt injection detected. Run blocked.")
            self._persist(st)
            return
        st.end_step("l0_sanitize", status="ok", output_ref={"sanitized": True}, message="pass")
        LiveFeed.emit(st, layer="L0", agent="SanitizeAgent", message="Security check passed.")

        # L2 intake bundle
        st.start_step("l2_parse", layer_id="L2", tool_name="ParserAgentService", input_ref={})
        extracted = self._parser.parse(raw_text=safe, orchestration_state=st, feedback=[])
        _save_json(run_dir / "intake_bundle.json", extracted.to_json_dict())
        st.add_artifact("intake_bundle", str(run_dir / "intake_bundle.json"), content_type="application/json")
        st.end_step("l2_parse", status="ok", output_ref={"artifact_key":"intake_bundle"}, message="parsed")

        # L3 eval (soft)
        ev = self._parser_eval.evaluate(orchestration_state=st, raw_text=safe, extracted=extracted, target_id="resume_main", threshold=float(prefs.get("resume_threshold", 0.55)))
        score = float(getattr(ev, "evaluation_score", 0.0))
        LiveFeed.emit(st, layer="L3", agent="ParserEvaluator", message=f"Intake quality={score:.2f}. Continuing automation.")

        # Do NOT hard stop. Only mark optional cleanup.
        if score < 0.55:
            st.meta["pending_action"] = "resume_cleanup_optional"
            LiveFeed.emit(st, layer="L5", agent="HITL", message="Optional: improve resume text for better ATS + scoring. Pipeline continues.")

        self._persist(st)

        # L3 job hunt across 8 boards for multiple roles
        roles = prefs.get("target_roles") or []
        if isinstance(roles, str):
            roles = [r.strip() for r in roles.splitlines() if r.strip()]
        if not roles:
            roles = [str(prefs.get("target_role","Data Scientist"))]

        location = str(prefs.get("location","United States"))
        remote = bool(prefs.get("remote", True))
        wfo_ok = bool(prefs.get("wfo_ok", True))
        salary = str(prefs.get("salary","")).strip()
        visa_required = bool(prefs.get("visa_sponsorship_required", False))
        recency_limit = float(prefs.get("recency_hours", 36))
        max_jobs = int(prefs.get("max_jobs", 40))
        discovery_threshold = float(prefs.get("discovery_threshold", 0.70))
        max_refinements = int(prefs.get("max_refinements", 3))

        serper_key = getattr(self._s, "serper_api_key", None)
        if not serper_key:
            st.status = "needs_human_approval"
            st.meta["pending_action"] = "missing_serper_key"
            LiveFeed.emit(st, layer="L3", agent="DiscoveryAgent", message="Missing SERPER_API_KEY in .env.")
            self._persist(st)
            return

        discovery = SerperDiscovery(api_key=serper_key, health=self._health)

        resume_skills = [s.lower() for s in (extracted.skills or [])]
        resume_ats = ats_score(safe)
        resume_tokens = {}
        for t in tokenize(safe):
            resume_tokens[t] = resume_tokens.get(t, 0) + 1
        exp_text = " ".join([" ".join(x.bullets) for x in (extracted.experience or [])]) or safe
        exp_tokens = {}
        for t in tokenize(exp_text):
            exp_tokens[t] = exp_tokens.get(t, 0) + 1

        # query builder
        def build_query(role: str) -> str:
            intent = []
            if remote: intent.append("remote")
            if wfo_ok: intent.append("hybrid")
            if not intent: intent.append("on-site")
            salary_part = f'"{salary}"' if salary else ""
            visa_part = '"visa sponsorship" OR h1b OR opt OR cpt' if visa_required else ""
            skill_hint = " ".join(resume_skills[:6])
            return f'{role} {location} {" ".join(intent)} {salary_part} {skill_hint} ({visa_part}) apply'

        # tbs recency (best-effort)
        tbs = "qdr:d" if recency_limit <= 36 else None

        # refinement loop
        query_variants = [build_query(r) for r in roles[:4]]
        all_results: List[Dict[str, Any]] = []

        for attempt in range(max_refinements + 1):
            LiveFeed.emit(st, layer="L3", agent="DiscoveryAgent", message=f"Hunt attempt {attempt+1}: searching 8 boards for {len(query_variants)} roles…")
            seen = set()
            all_results = []
            for qi, q in enumerate(query_variants):
                for b in DEFAULT_JOB_BOARDS:
                    step_id = f"l3_serper_{attempt+1}_{qi}_{b.domain.replace('/','_')}"
                    st.start_step(step_id, layer_id="L3", tool_name="serper.search", input_ref={"role_query": q, "board": b.name})
                    items = discovery.search(st=st, step_id=step_id, query=f"{q} site:{b.domain}", num=10, tbs=tbs)
                    st.end_step(step_id, status="ok", output_ref={"count": len(items)}, message=b.name)
                    for it in items:
                        link = it.get("link") or ""
                        if not link or link in seen:
                            continue
                        seen.add(link)
                        it["board"] = b.name
                        it["role_hint"] = roles[min(qi, len(roles)-1)]
                        all_results.append(it)

            # L4 scrape+score max_jobs
            LiveFeed.emit(st, layer="L4", agent="ScraperAgent", message=f"Scraping + scoring up to {max_jobs} jobs…")
            ranked: List[Dict[str, Any]] = []
            for idx, it in enumerate(all_results[:max_jobs]):
                url = it.get("link") or ""
                snippet = it.get("snippet") or ""
                title = it.get("title") or ""
                board = it.get("board") or "unknown"
                rh = recency_hours(snippet)
                if rh is not None and rh > recency_limit:
                    continue

                step_id = f"l4_scrape_{attempt+1}_{idx+1}"
                st.start_step(step_id, layer_id="L4", tool_name="Scraper", input_ref={"url": url, "board": board})
                text = Scraper.fetch_text(url=url, snippet=snippet)
                st.end_step(step_id, status="ok", output_ref={"chars": len(text)}, message="scraped")

                low = text.lower()
                visa_ok = not any(x in low for x in VISA_NEGATIVE)
                if visa_required and not visa_ok:
                    continue

                job_skills = extract_job_skills(text, resume_skills)
                overlap = len(set(job_skills) & set(resume_skills)) / max(1, len(set(job_skills))) if job_skills else 0.0

                job_tokens = {}
                for t in tokenize(text):
                    job_tokens[t] = job_tokens.get(t, 0) + 1
                exp_align = cosine(exp_tokens, job_tokens)

                market = 1.0
                if "applicants" in snippet.lower():
                    m = re.search(r"(\\d+)\\+?\\s*applicants", snippet.lower())
                    if m:
                        n = int(m.group(1))
                        market = 1.0 + min(1.5, n/200.0)

                score = compute_interview_chance(overlap, exp_align, resume_ats, market)
                if visa_required and any(x in low for x in VISA_POSITIVE):
                    score = min(1.0, score + 0.05)

                missing = [s for s in job_skills if s not in resume_skills][:12]
                jid = url or uuid4().hex

                ranked.append({
                    "job_id": jid,
                    "rank": 0,
                    "role_hint": it.get("role_hint"),
                    "title": title,
                    "board": board,
                    "url": url,
                    "recency_hours": rh,
                    "visa_ok": visa_ok,
                    "matched_skills": list(set(job_skills) & set(resume_skills))[:12],
                    "missing_skills": missing,
                    "components": {
                        "skill_overlap": overlap,
                        "experience_alignment": exp_align,
                        "ats_score": resume_ats,
                        "market_competition_factor": market,
                    },
                    "interview_chance_score": score,
                    "overall_match_percent": round(score*100.0, 2),
                    "rationale": [
                        f"SkillOverlap={overlap:.2f}",
                        f"ExperienceAlignment={exp_align:.2f}",
                        f"ATS={resume_ats:.2f}",
                        f"MarketFactor={market:.2f}",
                        ("VisaOK" if visa_ok else "NoSponsorship"),
                    ],
                })

                st.meta["job_scores"][jid] = float(score)
                st.meta["job_components"][jid] = ranked[-1]["components"]
                st.meta["job_meta"][jid] = {"role_title": title, "company": board, "url": url, "source": board}

            ranked.sort(key=lambda x: float(x["interview_chance_score"]), reverse=True)
            for i, r in enumerate(ranked, start=1):
                r["rank"] = i

            _save_json(run_dir / "ranking.json", ranked)
            st.add_artifact("ranking", str(run_dir / "ranking.json"), content_type="application/json")
            LiveFeed.emit(st, layer="L5", agent="EvaluatorAgent", message=f"Ranked {len(ranked)} jobs. Top={ranked[0]['overall_match_percent'] if ranked else 'n/a'}%")

            # Gate decision
            top_score = float(ranked[0]["interview_chance_score"]) if ranked else 0.0
            if top_score > 0.85 and ranked:
                LiveFeed.emit(st, layer="L5", agent="EvaluatorAgent", message="Auto-pilot engaged (top score > 0.85). Skipping HITL ranking gate.")
                self._generate_drafts_from_ranking(st=st, run_dir=run_dir, ranked=ranked, intake=extracted)
                st.status = "completed"
                st.meta["pending_action"] = None
                LiveFeed.emit(st, layer="L7", agent="ApplyExecutor", message="Auto-pilot completed draft generation and simulated submission.")
                self._persist(st)
                return

            if top_score >= discovery_threshold and len(ranked) >= min(20, max_jobs):
                st.status = "Pending"
                st.current_layer = 5
                st.meta["pending_action"] = "review_ranking"
                LiveFeed.emit(st, layer="L5", agent="EvaluatorAgent", message="Ranking ready for approval (HITL).")
                self._persist(st)
                return

            if attempt >= max_refinements:
                st.status = "Pending"
                st.current_layer = 5
                st.meta["pending_action"] = "low_confidence_discovery"
                LiveFeed.emit(st, layer="L5", agent="EvaluatorAgent", message="Low confidence after retries. Needs guidance.")
                self._persist(st)
                return

            # refine queries (simple deterministic refinement)
            if ranked:
                top = ranked[0]
                hint = " ".join((top.get("matched_skills") or [])[:6])
            else:
                hint = " ".join(resume_skills[:6])
            query_variants = [f"{r} {location} {hint} apply" for r in roles[:4]]
            self._persist(st)

    def _continue(self, run_id: str) -> None:
        raw = self.load(run_id)
        if not raw:
            return
        st = AgentState(**raw)
        run_dir = _run_dir(run_id)
        pending = st.meta.get("pending_action")
        action = (st.meta.get("last_user_action") or {}).get("type")
        payload = (st.meta.get("last_user_action") or {}).get("payload") or {}

        # Optional resume improvement loop
        if pending == "resume_cleanup_optional" and action == "resume_cleanup_submit":
            new_text = str(payload.get("resume_text","")).strip()
            if new_text:
                (run_dir / "resume_raw.txt").write_text(new_text, encoding="utf-8")
                st.add_artifact("resume_raw", str(run_dir / "resume_raw.txt"), content_type="text/plain")
                st.status = "running"
                st.meta["pending_action"] = None
                LiveFeed.emit(st, layer="L2", agent="ParserAgent", message="Resume updated. Restarting run…")
                self._persist(st)
                self._run(run_id, "resume.txt", new_text.encode("utf-8"))
            return

        # Approve ranking -> generate drafts + learning plan for missing skills
        if pending == "review_ranking" and action == "approve_ranking":
            ranking_path = run_dir / "ranking.json"
            intake_path = run_dir / "intake_bundle.json"
            if not (ranking_path.exists() and intake_path.exists()):
                st.status = "needs_human_approval"
                st.meta["pending_action"] = "missing_artifacts"
                self._persist(st)
                return

            ranked = json.loads(ranking_path.read_text(encoding="utf-8"))
            intake = ExtractedResume(**json.loads(intake_path.read_text(encoding="utf-8")))
            self._generate_drafts_from_ranking(st=st, run_dir=run_dir, ranked=ranked, intake=intake)

            st.status = "needs_human_approval"
            st.meta["pending_action"] = "review_drafts"
            LiveFeed.emit(st, layer="L6", agent="DraftAgent", message=f"Generated {len(drafts)} resume+cover packages + learning plans.")
            self._persist(st)
            return

        # Approve drafts -> mark completed (apply simulation can be added later)
        if pending == "review_drafts" and action == "approve_drafts":
            st.status = "completed"
            st.meta["pending_action"] = None
            LiveFeed.emit(st, layer="L7", agent="ApplyExecutor", message="Drafts approved. (Simulated) submission completed.")
            self._persist(st)
            return

        # Reject ranking -> rerun discovery with refined hint
        if pending == "review_ranking" and action == "reject_ranking":
            reason = str(payload.get("reason","")).strip()
            st.meta.setdefault("user_refinement_notes", [])
            if reason:
                st.meta["user_refinement_notes"].append(reason)
            st.status = "running"
            st.meta["pending_action"] = None
            LiveFeed.emit(st, layer="L5", agent="HITL", message="Ranking rejected. Re-running discovery…")
            self._persist(st)

            resume_path = run_dir / "resume_raw.txt"
            if resume_path.exists():
                self._run(run_id, "resume.txt", resume_path.read_bytes())
            return

        # Reject drafts -> go back to ranking approval
        if pending == "review_drafts" and action == "reject_drafts":
            st.status = "needs_human_approval"
            st.meta["pending_action"] = "review_ranking"
            LiveFeed.emit(st, layer="L6", agent="HITL", message="Drafts rejected. Returning to ranking review.")
            self._persist(st)
            return

        self._persist(st)

    def _generate_drafts_from_ranking(self, *, st: AgentState, run_dir: Path, ranked: List[Dict[str, Any]], intake: ExtractedResume) -> None:
        top_n = int((st.meta.get("preferences", {}) or {}).get("draft_count", 10))
        ranked = (ranked or [])[:top_n]

        drafts: List[Dict[str, Any]] = []
        learning: Dict[str, Any] = {}

        base_resume = self._build_ats_resume(intake)
        (run_dir / "ats_resume_base.md").write_text(base_resume, encoding="utf-8")
        st.add_artifact("ats_resume_base", str(run_dir / "ats_resume_base.md"), content_type="text/markdown")

        for j in ranked:
            jid = j["job_id"]
            title = j.get("title") or "Role"
            company = j.get("board") or "Company"
            missing = j.get("missing_skills") or []

            tailored = self._tailor_resume(base_resume, title, company, j.get("matched_skills") or [], missing)
            resume_path = run_dir / f"resume_{jid[:10]}.md"
            resume_path.write_text(tailored, encoding="utf-8")
            st.add_artifact(f"resume_{jid[:10]}", str(resume_path), content_type="text/markdown")

            cover = self._cover_letter(intake, title, company, j.get("matched_skills") or [])
            cover_path = run_dir / f"cover_{jid[:10]}.md"
            cover_path.write_text(cover, encoding="utf-8")
            st.add_artifact(f"cover_{jid[:10]}", str(cover_path), content_type="text/markdown")

            if missing:
                learning[jid] = self._learn.build_learning_plan(missing_skills=missing)

            drafts.append({
                "job_id": jid,
                "title": title,
                "company": company,
                "url": j.get("url"),
                "resume_path": str(resume_path),
                "cover_path": str(cover_path),
                "missing_skills": missing,
            })

        _save_json(run_dir / "drafts_bundle.json", {"drafts": drafts, "learning_plan": learning})
        st.add_artifact("drafts_bundle", str(run_dir / "drafts_bundle.json"), content_type="application/json")

    # -------- ATS Resume generation (no user ATS needed) --------
    def _build_ats_resume(self, intake: ExtractedResume) -> str:
        email = intake.contact.email or ""
        phone = intake.contact.phone or ""
        linkedin = intake.contact.linkedin or ""
        github = intake.contact.github or ""
        name = intake.name or "Candidate"

        skills = ", ".join((intake.skills or [])[:25])
        bullets = []
        if intake.experience and intake.experience[0].bullets:
            bullets = intake.experience[0].bullets[:8]

        bullets_md = "\n".join([f"- {b}" for b in bullets]) if bullets else "- Add 4–6 bullets with measurable impact (metrics, scope, tools)."

        return f"""# {name}
{email} | {phone} | {linkedin} | {github}

## Summary
{intake.summary or "AI/ML professional focused on production-grade GenAI and MLOps systems."}

## Skills
{skills}

## Experience
{bullets_md}

## Education
- (Auto-filled from resume intake)
"""

    def _tailor_resume(self, base: str, title: str, company: str, matched: List[str], missing: List[str]) -> str:
        matched_str = ", ".join(matched[:12])
        missing_str = ", ".join(missing[:8])
        return base + f"\n\n## Target Role Alignment\n- Target Role: {title} @ {company}\n- Matched Keywords: {matched_str}\n- Gap Keywords (learning plan attached): {missing_str}\n"

    def _cover_letter(self, intake: ExtractedResume, title: str, company: str, matched: List[str]) -> str:
        name = intake.name or "Candidate"
        email = intake.contact.email or ""
        kw = ", ".join(matched[:8])
        return f"""{name}
{email}

Dear Hiring Manager,

I’m applying for the {title} role. My background includes production-grade AI/ML delivery, GenAI application development, and MLOps practices (CI/CD, reproducible pipelines, evaluation, and governance).

I’m especially aligned to this role through: {kw}.

I’d welcome a quick conversation on how I can help {company} deliver measurable AI impact.

Sincerely,  
{name}
"""


ENGINE = OneClickAutomationEngine()
