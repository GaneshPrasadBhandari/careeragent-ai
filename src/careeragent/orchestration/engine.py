
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import httpx

from careeragent.config import artifacts_root, get_settings
from careeragent.orchestration.state import OrchestrationState
from careeragent.services.db_service import SqliteStateStore
from careeragent.services.notification_service import NotificationService

# HealthService is optional; if missing, we degrade gracefully
try:
    from careeragent.services.health_service import HealthService  # type: ignore
except Exception:
    HealthService = None  # type: ignore

from careeragent.agents.security_agent import SanitizeAgent
from careeragent.agents.parser_agent_service import ParserAgentService, ExtractedResume
from careeragent.agents.parser_evaluator_service import ParserEvaluatorService

# --------------------------- helpers ---------------------------
def _run_dir(run_id: str) -> Path:
    d = artifacts_root() / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


class LiveFeed:
    """
    Description: Live Agent Feed logger.
    Layer: L1
    """
    @staticmethod
    def emit(st: OrchestrationState, *, layer: str, agent: str, message: str) -> None:
        st.meta.setdefault("live_feed", [])
        st.meta["live_feed"].append({"layer": layer, "agent": agent, "message": message})
        st.touch()


class LocalResumeExtractor:
    """
    Description: Extract resume text from PDF/TXT/DOCX.
    Layer: L2
    """
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


# -------- Job discovery (Serper across 8 boards) --------
JOB_BOARDS = [
    ("LinkedIn Jobs", "linkedin.com/jobs"),
    ("Indeed", "indeed.com"),
    ("Glassdoor", "glassdoor.com"),
    ("ZipRecruiter", "ziprecruiter.com"),
    ("Monster", "monster.com"),
    ("Dice", "dice.com"),
    ("Lever", "jobs.lever.co"),
    ("Greenhouse", "boards.greenhouse.io"),
]

VISA_NEGATIVE = ("unable to sponsor","cannot sponsor","no sponsorship","do not sponsor","not sponsor","without sponsorship","no visa")
VISA_POSITIVE = ("visa sponsorship","h1b","opt","cpt","stem opt","work visa")


def _parse_recency_hours(snippet: str) -> Optional[float]:
    s = (snippet or "").lower()
    if "today" in s: return 6.0
    if "yesterday" in s: return 24.0
    m = re.search(r"(\d+)\s*hours?\s*ago", s)
    if m: return float(m.group(1))
    m = re.search(r"(\d+)\s*days?\s*ago", s)
    if m: return float(m.group(1)) * 24.0
    return None


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9\+\#\.-]{1,}", (text or "").lower())


def _cosine(a: Dict[str,int], b: Dict[str,int]) -> float:
    if not a or not b: return 0.0
    dot = sum(v * b.get(k,0) for k,v in a.items())
    na = sum(v*v for v in a.values()) ** 0.5
    nb = sum(v*v for v in b.values()) ** 0.5
    if na == 0 or nb == 0: return 0.0
    return float(dot/(na*nb))


def _ats_score(resume_text: str) -> float:
    t = (resume_text or "").lower()
    score = 0.0
    if re.search(r"[\w\.-]+@[\w\.-]+\.\w+", t): score += 0.20
    if re.search(r"\+?\d[\d\-\s\(\)]{8,}\d", t): score += 0.10
    for h in ["summary","skills","experience","education","projects"]:
        if h in t: score += 0.12
    if "-" in resume_text or "•" in resume_text: score += 0.10
    if len(resume_text) > 1200: score += 0.12
    return max(0.0, min(1.0, score))


def _compute_interview_chance(skill_overlap: float, exp_align: float, ats: float, market: float) -> float:
    market = max(1.0, float(market))
    score = (0.45*skill_overlap + 0.35*exp_align + 0.20*ats) / market
    return max(0.0, min(1.0, float(score)))


class SerperClient:
    SERPER_URL = "https://google.serper.dev/search"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def search(self, query: str, num: int = 10, tbs: Optional[str] = None) -> List[Dict[str, Any]]:
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        body: Dict[str, Any] = {"q": query, "num": num}
        if tbs:
            body["tbs"] = tbs
        with httpx.Client(timeout=30.0) as client:
            r = client.post(self.SERPER_URL, headers=headers, json=body)
        if r.status_code >= 400:
            return []
        organic = (r.json().get("organic") or [])
        return [{"title": it.get("title") or "", "link": it.get("link") or "", "snippet": it.get("snippet") or ""} for it in organic]


class Scraper:
    @staticmethod
    def fetch_text(url: str, snippet: str) -> str:
        if not url:
            return snippet or ""
        try:
            with httpx.Client(timeout=18.0, follow_redirects=True) as client:
                r = client.get(url, headers={"User-Agent":"Mozilla/5.0"})
            if r.status_code >= 400:
                return snippet or ""
            html = r.text
            txt = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S|re.I)
            txt = re.sub(r"<[^>]+>", " ", txt)
            txt = re.sub(r"\s+", " ", txt).strip()
            return txt[:16000] if txt else (snippet or "")
        except Exception:
            return snippet or ""


class OneClickAutomationEngine:
    """
    Description: Full one-click engine with soft-gated parser + full discovery/matching/ranking.
    Layer: L0-L9
    """

    def __init__(self) -> None:
        self._s = get_settings()
        self._store = SqliteStateStore()
        self._notifier = NotificationService(dry_run=not bool(self._s.twilio_account_sid))

        self._sanitize = SanitizeAgent()
        self._parser = ParserAgentService()
        self._parser_eval = ParserEvaluatorService()

        self._health = HealthService() if HealthService else None

    def _persist(self, st: OrchestrationState) -> None:
        d = _run_dir(st.run_id)
        _save_json(d / "state.json", st.model_dump())
        self._store.upsert_state(run_id=st.run_id, status=st.status, state=st.model_dump(), updated_at_utc=st.updated_at_utc)

    def load(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get_state(run_id=run_id)

    def start_run(self, *, filename: str, data: bytes, prefs: Dict[str, Any]) -> OrchestrationState:
        st = OrchestrationState.new(env=self._s.environment, mode="agentic", git_sha=None)
        st.meta["preferences"] = prefs
        st.meta.setdefault("live_feed", [])
        st.meta.setdefault("job_scores", {})
        st.meta.setdefault("job_components", {})
        st.meta.setdefault("job_meta", {})
        LiveFeed.emit(st, layer="L1", agent="Dashboard", message="Run created. Starting autonomous pipeline…")
        self._persist(st)
        t = threading.Thread(target=self._run, args=(st.run_id, filename, data), daemon=True)
        t.start()
        return st

    def submit_action(self, *, run_id: str, action_type: str, payload: Dict[str, Any]) -> OrchestrationState:
        raw = self.load(run_id)
        if not raw:
            raise ValueError("run_id not found")
        st = OrchestrationState(**raw)
        st.meta["last_user_action"] = {"type": action_type, "payload": payload}
        LiveFeed.emit(st, layer="L5", agent="HITL", message=f"User action received: {action_type}")
        self._persist(st)
        t = threading.Thread(target=self._continue, args=(run_id,), daemon=True)
        t.start()
        return st

    def _run(self, run_id: str, filename: str, data: bytes) -> None:
        raw = self.load(run_id)
        if not raw:
            return
        st = OrchestrationState(**raw)
        run_dir = _run_dir(run_id)
        prefs = st.meta.get("preferences", {}) or {}

        # ---- L2 Extract ----
        st.start_step("l2_extract", layer_id="L2", tool_name="ResumeExtractor", input_ref={"filename": filename})
        LiveFeed.emit(st, layer="L2", agent="ParserAgent", message="Extracting resume text from upload…")
        try:
            resume_text = LocalResumeExtractor.extract_text(filename=filename, data=data)
        except Exception as e:
            st.end_step("l2_extract", status="failed", output_ref={}, message=str(e))
            st.status = "needs_human_approval"
            st.meta["pending_action"] = "resume_cleanup"
            LiveFeed.emit(st, layer="L2", agent="ParserAgent", message=f"Resume extraction failed: {e}")
            self._persist(st)
            return

        (run_dir / "resume_raw.txt").write_text(resume_text, encoding="utf-8")
        st.add_artifact("resume_raw", str(run_dir / "resume_raw.txt"), content_type="text/plain")
        st.end_step("l2_extract", status="ok", output_ref={"artifact_key":"resume_raw"}, message="extracted")
        self._persist(st)

        # ---- L0 Sanitize ----
        st.start_step("l0_sanitize", layer_id="L0", tool_name="SanitizeAgent", input_ref={})
        safe = self._sanitize.sanitize_before_llm(state=st, step_id="l0_sanitize", tool_name="sanitize_before_llm", user_text=resume_text, context="resume")
        if safe is None:
            st.status = "blocked"
            LiveFeed.emit(st, layer="L0", agent="SanitizeAgent", message="Prompt injection detected. Run blocked.")
            self._persist(st)
            return
        st.end_step("l0_sanitize", status="ok", output_ref={"sanitized": True}, message="pass")
        LiveFeed.emit(st, layer="L0", agent="SanitizeAgent", message="Security check passed.")
        self._persist(st)

        # ---- L2 Parse (single strong pass) ----
        st.start_step("l2_parse", layer_id="L2", tool_name="ParserAgentService", input_ref={"attempt": 1})
        extracted = self._parser.parse(raw_text=safe, orchestration_state=st, feedback=[])
        p = run_dir / "extracted_resume.json"
        _save_json(p, extracted.to_json_dict())
        st.add_artifact("extracted_resume", str(p), content_type="application/json")
        st.end_step("l2_parse", status="ok", output_ref={"artifact_key":"extracted_resume"}, message="parsed")

        # ---- L3 Evaluate (SOFT gate) ----
        # strict gate optional; default False for automation
        strict_gate = bool(prefs.get("resume_strict_gate", False))
        threshold = 0.80 if strict_gate else float(prefs.get("resume_threshold", 0.55))

        ev = self._parser_eval.evaluate(
            orchestration_state=st,
            raw_text=safe,
            extracted=extracted,
            target_id="resume_main",
            threshold=threshold,
            retry_count=0,
            max_retries=0,
        )

        score = float(getattr(ev, "evaluation_score", 0.0))
        LiveFeed.emit(st, layer="L3", agent="ParserEvaluator", message=f"Resume quality={score:.2f} (threshold={threshold:.2f}).")

        # If truly broken, stop; otherwise continue but allow optional cleanup
        if score < 0.30:
            st.status = "needs_human_approval"
            st.meta["pending_action"] = "resume_cleanup"
            LiveFeed.emit(st, layer="L3", agent="ParserEvaluator", message="Resume parsing too weak. Needs manual cleanup.")
            self._persist(st)
            return
        elif score < threshold:
            st.meta["pending_action"] = "resume_cleanup_optional"
            LiveFeed.emit(st, layer="L3", agent="ParserEvaluator", message="Proceeding automatically, but resume cleanup is recommended.")

        self._persist(st)

        # ---- L3 Discovery ----
        if not self._s.serper_api_key:
            st.status = "needs_human_approval"
            st.meta["pending_action"] = "missing_serper_key"
            LiveFeed.emit(st, layer="L3", agent="DiscoveryAgent", message="Missing SERPER_API_KEY in .env.")
            self._persist(st)
            return

        target_role = str(prefs.get("target_role","Data Scientist"))
        location = str(prefs.get("location","United States"))
        remote = bool(prefs.get("remote", True))
        wfo_ok = bool(prefs.get("wfo_ok", True))
        salary = str(prefs.get("salary","")).strip()
        visa_required = bool(prefs.get("visa_sponsorship_required", False))
        recency_hours = float(prefs.get("recency_hours", 36))
        max_jobs = int(prefs.get("max_jobs", 40))
        discovery_threshold = float(prefs.get("discovery_threshold", 0.70))
        max_refinements = int(prefs.get("max_refinements", 3))

        resume_skills = [s.lower() for s in (extracted.skills or [])]
        ats = _ats_score(safe)
        st.meta["ats_score"] = ats

        serper = SerperClient(self._s.serper_api_key)
        tbs = "qdr:d" if recency_hours <= 36 else None

        base_query = self._build_query(target_role, location, remote, wfo_ok, salary, visa_required, resume_skills)

        for attempt in range(max_refinements + 1):
            LiveFeed.emit(st, layer="L3", agent="DiscoveryAgent", message=f"Hunt attempt {attempt+1}: searching 8 boards…")
            results = self._discover_all(serper, base_query, tbs=tbs)

            # ---- L4 Scrape + score ----
            ranked = self._score_jobs(
                resume_text=safe,
                extracted=extracted,
                ats=ats,
                visa_required=visa_required,
                recency_hours=recency_hours,
                results=results[:max_jobs],
            )
            _save_json(run_dir / "ranking.json", ranked)
            st.add_artifact("ranking", str(run_dir / "ranking.json"), content_type="application/json")
            self._persist(st)

            if not ranked:
                conf = 0.0
            else:
                top = float(ranked[0]["interview_chance_score"])
                avg = sum(float(x["interview_chance_score"]) for x in ranked[:20]) / max(1, min(20, len(ranked)))
                conf = min(1.0, 0.65*top + 0.35*avg)

            LiveFeed.emit(st, layer="L5", agent="EvaluatorAgent", message=f"Discovery confidence={conf:.2f} (need ≥ {discovery_threshold:.2f}).")

            if ranked and float(ranked[0]["interview_chance_score"]) >= discovery_threshold:
                st.status = "needs_human_approval"
                st.meta["pending_action"] = "review_ranking"
                LiveFeed.emit(st, layer="L1", agent="Dashboard", message="Ranking ready for review (HITL).")
                self._persist(st)
                return

            if attempt >= max_refinements:
                st.status = "needs_human_approval"
                st.meta["pending_action"] = "low_confidence_discovery"
                LiveFeed.emit(st, layer="L5", agent="EvaluatorAgent", message="Low confidence after retries. Needs guidance.")
                self._persist(st)
                return

            base_query = self._refine_query(base_query, ranked, resume_skills, visa_required, target_role, location)
            LiveFeed.emit(st, layer="L3", agent="DiscoveryAgent", message=f"Refining query → {base_query[:160]}")
            self._persist(st)

    def _continue(self, run_id: str) -> None:
        raw = self.load(run_id)
        if not raw:
            return
        st = OrchestrationState(**raw)
        run_dir = _run_dir(run_id)
        pending = st.meta.get("pending_action")
        action = (st.meta.get("last_user_action") or {}).get("type")
        payload = (st.meta.get("last_user_action") or {}).get("payload") or {}

        # NEW: Resume cleanup submit
        if pending in ("resume_cleanup", "resume_cleanup_optional") and action == "resume_cleanup_submit":
            new_text = str(payload.get("resume_text","")).strip()
            if not new_text:
                LiveFeed.emit(st, layer="L5", agent="HITL", message="Resume cleanup submitted but text was empty.")
                self._persist(st)
                return

            (run_dir / "resume_raw.txt").write_text(new_text, encoding="utf-8")
            st.add_artifact("resume_raw", str(run_dir / "resume_raw.txt"), content_type="text/plain")
            st.meta["pending_action"] = None
            st.status = "running"
            LiveFeed.emit(st, layer="L2", agent="ParserAgent", message="Resume updated by user. Restarting pipeline…")
            self._persist(st)

            # restart the run using updated text as txt bytes (no need for original upload)
            self._run(run_id, "resume.txt", new_text.encode("utf-8"))
            return

        # Ranking approval/reject can be wired next (your next step).
        LiveFeed.emit(st, layer="L5", agent="HITL", message=f"No handler for pending={pending}, action={action}")
        self._persist(st)

    def _build_query(self, target_role: str, location: str, remote: bool, wfo_ok: bool, salary: str, visa_required: bool, skills: List[str]) -> str:
        intent = []
        if remote: intent.append("remote")
        if wfo_ok: intent.append("hybrid")
        if not intent: intent.append("on-site")
        skill_str = " ".join(skills[:6])
        salary_part = f'"{salary}"' if salary else ""
        visa_part = '"visa sponsorship" OR h1b OR opt OR cpt' if visa_required else ""
        return f'{target_role} {location} {" ".join(intent)} {salary_part} {skill_str} ({visa_part}) apply'

    def _discover_all(self, serper: SerperClient, base_query: str, tbs: Optional[str]) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        for name, domain in JOB_BOARDS:
            q = f"{base_query} site:{domain}"
            items = serper.search(q, num=10, tbs=tbs)
            for it in items:
                link = it.get("link") or ""
                if not link or link in seen:
                    continue
                seen.add(link)
                it["board"] = name
                out.append(it)
        return out

    def _score_jobs(self, *, resume_text: str, extracted: ExtractedResume, ats: float, visa_required: bool, recency_hours: float, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        res_tokens = {}
        for t in _tokenize(resume_text):
            res_tokens[t] = res_tokens.get(t, 0) + 1
        exp_text = " ".join([" ".join(x.bullets) for x in (extracted.experience or [])]) if extracted.experience else resume_text
        exp_tokens = {}
        for t in _tokenize(exp_text):
            exp_tokens[t] = exp_tokens.get(t, 0) + 1
        resume_skills = [s.lower() for s in (extracted.skills or [])]

        for it in results:
            snippet = it.get("snippet") or ""
            rh = _parse_recency_hours(snippet)
            if rh is not None and rh > recency_hours:
                continue

            url = it.get("link") or ""
            board = it.get("board") or "unknown"
            title = it.get("title") or ""

            job_text = Scraper.fetch_text(url, snippet)
            low = job_text.lower()

            # visa filter
            v_ok = not any(x in low for x in VISA_NEGATIVE)
            if visa_required and not v_ok:
                continue

            # skills overlap (from resume skills present in job text)
            present = [s for s in resume_skills if s and s in low]
            overlap = len(set(present)) / max(1, len(set(resume_skills))) if resume_skills else 0.0

            # exp alignment cosine
            job_tokens = {}
            for t in _tokenize(job_text):
                job_tokens[t] = job_tokens.get(t, 0) + 1
            exp_align = _cosine(exp_tokens, job_tokens)

            market = 1.0
            if "applicants" in snippet.lower():
                m = re.search(r"(\d+)\+?\s*applicants", snippet.lower())
                if m:
                    n = int(m.group(1))
                    market = 1.0 + min(1.5, n/200.0)

            score = _compute_interview_chance(overlap, exp_align, ats, market)
            if visa_required and any(x in low for x in VISA_POSITIVE):
                score = min(1.0, score + 0.05)

            ranked.append({
                "title": title,
                "board": board,
                "url": url,
                "recency_hours": rh,
                "visa_ok": v_ok,
                "skill_overlap": overlap,
                "experience_alignment": exp_align,
                "ats_score": ats,
                "market_factor": market,
                "interview_chance_score": score,
                "overall_match_percent": round(score*100.0, 2),
                "matched_skills": present[:12],
                "rationale": [
                    f"SkillOverlap={overlap:.2f}",
                    f"ExperienceAlignment={exp_align:.2f}",
                    f"ATS={ats:.2f}",
                    f"MarketFactor={market:.2f}",
                    ("VisaOK" if v_ok else "NoSponsorship"),
                ]
            })

        ranked.sort(key=lambda x: float(x["interview_chance_score"]), reverse=True)
        for i, r in enumerate(ranked, start=1):
            r["rank"] = i
        return ranked

    def _refine_query(self, base_query: str, ranked: List[Dict[str, Any]], resume_skills: List[str], visa_required: bool, target_role: str, location: str) -> str:
        top = ranked[0] if ranked else {}
        top_skills = (top.get("matched_skills") or [])[:6]
        hint = " ".join(list(dict.fromkeys([*top_skills, *resume_skills]))[:8])
        visa = '("visa sponsorship" OR h1b OR opt)' if visa_required else ""
        return f"{target_role} {location} {hint} {visa} apply"


ENGINE = OneClickAutomationEngine()
