
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from careeragent.agents.parser_agent_service import ParserAgentService
from careeragent.agents.parser_evaluator_service import ParserEvaluatorService

# L6-L9 nodes (tool-resilient)
from careeragent.langgraph.nodes_l6_l9 import (
    l6_draft_node, l6_evaluator_node,
    l7_apply_node, l7_evaluator_node,
    l8_tracker_node, l8_evaluator_node,
    l9_analytics_node,
)

# Optional artifacts root
try:
    from careeragent.core.settings import artifacts_root  # type: ignore
except Exception:
    def artifacts_root() -> Path:
        return Path("src/careeragent/artifacts").resolve()

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

VISA_NEGATIVE = ("unable to sponsor","cannot sponsor","no sponsorship","do not sponsor","not sponsor","without sponsorship","no visa","cannot provide visa")
RARE_SKILL_SIGNALS = ("langgraph", "mcp", "agentic", "vector db", "faiss", "chroma", "rlhf")

class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    SERPER_API_KEY: Optional[str] = None
    MCP_SERVER_URL: Optional[str] = None
    MCP_AUTH_TOKEN: Optional[str] = None
    MCP_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: Optional[str] = None
    OLLAMA_MODEL: str = "llama3.2"

def mcp_token(s: RuntimeSettings) -> Optional[str]:
    return s.MCP_API_KEY or s.MCP_AUTH_TOKEN

def feed(st: Dict[str, Any], layer: str, agent: str, message: str) -> None:
    st.setdefault("live_feed", [])
    st["live_feed"].append({"layer": layer, "agent": agent, "message": message})

def inc_retry(st: Dict[str, Any], layer: str) -> int:
    st.setdefault("layer_retry_count", {})
    st["layer_retry_count"][layer] = int(st["layer_retry_count"].get(layer, 0)) + 1
    return int(st["layer_retry_count"][layer])

def threshold(st: Dict[str, Any], key: str, default: float = 0.70) -> float:
    t = (st.get("thresholds") or {})
    return float(t.get(key, t.get("default", default)))

def gate_decision(score: float, threshold_v: float, retries: int, max_retries: int) -> str:
    if score >= threshold_v:
        return "pass"
    if retries < max_retries:
        return "retry"
    return "hitl"

def log_attempt(st: Dict[str, Any], *, layer: str, agent: str, tool: str, model: Optional[str], status: str, confidence: float, error: Optional[str]) -> None:
    st.setdefault("attempts", [])
    st["attempts"].append({"layer_id": layer, "agent": agent, "tool": tool, "model": model, "status": status, "confidence": float(confidence), "error": error})

def log_gate(st: Dict[str, Any], *, layer: str, target: str, score: float, threshold_v: float, decision: str, feedback: List[str], reasoning_chain: Optional[List[str]] = None) -> None:
    st.setdefault("gates", [])
    st["gates"].append({"layer_id": layer, "target": target, "score": float(score), "threshold": float(threshold_v), "decision": decision, "feedback": feedback, "reasoning_chain": reasoning_chain or []})

    # UI compatibility: Mission Control reads `evaluations`.
    st.setdefault("evaluations", [])
    st["evaluations"].append(
        {
            "layer_id": layer,
            "target_id": target,
            "evaluation_score": float(score),
            "threshold": float(threshold_v),
            "decision": decision,
            "feedback": feedback,
            "reasoning_chain": reasoning_chain or [],
        }
    )

    # Engineer view transparency: keep a plain decision log list.
    st.setdefault("evaluation_logs", [])
    st["evaluation_logs"].append(
        {
            "layer": layer,
            "target": target,
            "score": float(score),
            "threshold": float(threshold_v),
            "decision": decision,
            "feedback": feedback,
        }
    )

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9\\+\\#\\.-]{1,}", (text or "").lower())

def cosine(a: Dict[str,int], b: Dict[str,int]) -> float:
    if not a or not b: return 0.0
    dot = sum(v*b.get(k,0) for k,v in a.items())
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    if na == 0 or nb == 0: return 0.0
    return float(dot/(na*nb))

def ats_score(text: str) -> float:
    t = (text or "").lower()
    s = 0.0
    if re.search(r"[\\w\\.-]+@[\\w\\.-]+\\.\\w+", t): s += 0.20
    if re.search(r"\\+?\\d[\\d\\-\\s\\(\\)]{8,}\\d", t): s += 0.10
    for h in ["summary","skills","experience","education","projects"]:
        if h in t: s += 0.12
    if "-" in text or "•" in text: s += 0.10
    if len(text) > 1200: s += 0.12
    return max(0.0, min(1.0, s))

def compute_interview_chance(skill_overlap: float, exp_align: float, ats: float, market: float) -> float:
    market = max(1.0, float(market))
    score = (0.45*skill_overlap + 0.35*exp_align + 0.20*ats) / market
    return max(0.0, min(1.0, float(score)))

async def serper_search(s: RuntimeSettings, query: str, num: int = 10, tbs: Optional[str] = None) -> Tuple[bool, float, Any, Optional[str]]:
    if not s.SERPER_API_KEY:
        return False, 0.0, None, "SERPER_API_KEY missing"
    headers = {"X-API-KEY": s.SERPER_API_KEY, "Content-Type": "application/json"}
    body: Dict[str, Any] = {"q": query, "num": num}
    if tbs: body["tbs"] = tbs
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post("https://google.serper.dev/search", headers=headers, json=body)
    if r.status_code == 403: return False, 0.0, None, "Serper 403 quota"
    if r.status_code >= 400: return False, 0.0, None, f"Serper {r.status_code}"
    organic = (r.json().get("organic") or [])
    out = [{"title": x.get("title") or "", "link": x.get("link") or "", "snippet": x.get("snippet") or ""} for x in organic]
    conf = 0.75 if out else 0.25
    return True, conf, out, None

async def mcp_invoke(s: RuntimeSettings, tool: str, payload: Dict[str, Any]) -> Tuple[bool, float, Any, Optional[str]]:
    if not (s.MCP_SERVER_URL and mcp_token(s)):
        return False, 0.0, None, "MCP not configured"
    url = s.MCP_SERVER_URL.rstrip("/") + "/invoke"
    headers = {"Authorization": f"Bearer {mcp_token(s)}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=35.0) as client:
        r = await client.post(url, headers=headers, json={"tool": tool, "payload": payload})
    if r.status_code >= 400:
        return False, 0.0, None, f"MCP {r.status_code}: {r.text[:150]}"
    return True, 0.85, r.json(), None

async def scrape_http(url: str) -> Tuple[bool, float, Any, Optional[str]]:
    try:
        async with httpx.AsyncClient(timeout=18.0, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code >= 400:
            return False, 0.0, None, f"HTTP {r.status_code}"
        html = r.text
        txt = re.sub(r"<(script|style)[^>]*>.*?</\\1>", " ", html, flags=re.S|re.I)
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = re.sub(r"\\s+", " ", txt).strip()
        conf = 0.65 if len(txt) > 1200 else 0.35
        return True, conf, {"text": txt[:20000]}, None
    except Exception as e:
        return False, 0.0, None, str(e)

# ---------------- L0 ----------------
async def L0_security(st: Dict[str, Any]) -> Dict[str, Any]:
    txt = st.get("resume_text") or ""
    inj = "ignore previous instructions" in txt.lower()
    ok = not inj
    log_attempt(st, layer="L0", agent="SanitizeAgent", tool="local.injection_heuristic", model=None, status=("ok" if ok else "failed"), confidence=(0.9 if ok else 0.1), error=None if ok else "prompt_injection")
    feed(st, "L0", "SanitizeAgent", "Security passed." if ok else "Blocked: prompt injection detected.")
    if not ok:
        st["status"] = "blocked"
        st["pending_action"] = "security_blocked"
    return st

# ---------------- L2 ----------------
async def L2_parse(st: Dict[str, Any]) -> Dict[str, Any]:
    parser = ParserAgentService()
    txt = st.get("resume_text") or ""
    prof = parser.parse(raw_text=txt, orchestration_state=None, feedback=[])
    prof_d = prof.to_json_dict()

    # --- Hardening: fill missing fields from raw resume text ---
    # The UI + ATS drafting depend on these being present.
    contact = (prof_d.get("contact") or {}) if isinstance(prof_d.get("contact"), dict) else {}

    # Email
    if not contact.get("email"):
        m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", txt)
        if m:
            contact["email"] = m.group(0)

    # Phone (US/international)
    if not contact.get("phone"):
        m = re.search(r"(\+?\d{1,3}[-\s]?)?(\(?\d{3}\)?[-\s]?)?\d{3}[-\s]?\d{4}", txt)
        if m and len(m.group(0).strip()) >= 10:
            contact["phone"] = m.group(0).strip()

    # Location
    if not contact.get("location"):
        if re.search(r"\b(united states|usa|u\.s\.)\b", txt.lower()):
            contact["location"] = "United States"

    # Links → normalize linkedin/github/medium where possible
    links = contact.get("links")
    if not isinstance(links, list):
        links = []
    # Extract links from raw text too
    for u in re.findall(r"https?://\S+", txt):
        u = u.strip().rstrip(").,]")
        if u and u not in links:
            links.append(u)
    contact["links"] = links
    # Convenience keys (downstream nodes sometimes expect these)
    if not contact.get("linkedin"):
        for u in links:
            if "linkedin.com" in u:
                contact["linkedin"] = u
                break
    if not contact.get("github"):
        for u in links:
            if "github.com" in u:
                contact["github"] = u
                break

    prof_d["contact"] = contact
    st["profile"] = prof_d
    log_attempt(st, layer="L2", agent="ParserAgent", tool="local.regex_parser", model=None, status="ok", confidence=0.65, error=None)
    feed(st, "L2", "ParserAgent", "Intake bundle created.")
    return st

async def EVAL_parser(st: Dict[str, Any]) -> Dict[str, Any]:
    prefs = st.get("preferences") or {}
    th = float((st.get("thresholds") or {}).get("parser", prefs.get("resume_threshold", 0.70)))
    prof = st.get("profile") or {}
    skills = len(prof.get("skills") or [])
    contact = prof.get("contact") or {}
    has_contact = bool(contact.get("email") or contact.get("phone"))
    score = min(1.0, 0.35 + (skills / 30.0) + (0.15 if has_contact else 0.0))
    fb = []
    if not has_contact: fb.append("Missing contact info. Continue, but add email/phone improves ATS.")
    if skills < 8: fb.append("Skill density low. Add more relevant tools/keywords.")
    retries = int((st.get("layer_retry_count") or {}).get("L2", 0))
    decision = gate_decision(score, th, retries, int(st.get("max_retries", 3)))
    log_gate(st, layer="L2", target="parser", score=score, threshold_v=th, decision=decision, feedback=fb)
    feed(st, "L3", "EvaluatorAgent", f"Parser score={score:.2f} decision={decision}")
    if decision == "hitl":
        st["status"] = "needs_human_approval"
        st["pending_action"] = "resume_cleanup_optional"
        st["hitl_reason"] = "Parser threshold not met"
        st["hitl_payload"] = {"feedback": fb, "score": score, "threshold": th}
    elif decision == "retry":
        inc_retry(st, "L2")
    return st

# ---------------- L3 ----------------
async def L3_discovery(st: Dict[str, Any]) -> Dict[str, Any]:
    s = RuntimeSettings()
    prefs = st.get("preferences") or {}
    roles = prefs.get("target_roles") or [prefs.get("target_role") or "Data Scientist"]
    roles = [r.strip() for r in roles if str(r).strip()][:4]
    location = str(prefs.get("location","United States"))
    visa_req = bool(prefs.get("visa_sponsorship_required", False))
    recency_h = float(prefs.get("recency_hours", 36))
    tbs = "qdr:d" if recency_h <= 36 else None

    prof = st.get("profile") or {}
    skills_hint = " ".join((prof.get("skills") or [])[:6])

    # Phase-2 refinement support (Evaluator → Discovery loop)
    qmods = st.get("query_modifiers") or {}
    neg_terms = [str(x).strip() for x in (qmods.get("neg_terms") or []) if str(x).strip()]
    must_include = [str(x).strip() for x in (qmods.get("must_include") or []) if str(x).strip()]
    refinement_feedback = str(st.get("refinement_feedback") or "").strip()
    strategy = str(st.get("search_strategy") or "default").strip().lower()

    country = str(prefs.get("country") or "US").strip().upper()
    if country == "US":
        # Common India/foreign noise exclusions
        neg_terms.extend(["India", "Bangalore", "Nashik", "Pune", "Hyderabad", "Chennai", "Mumbai", "Shine", "Naukri"])

    neg_clause = " ".join([f"-{t}" for t in list(dict.fromkeys(neg_terms))[:18]])
    include_clause = " ".join(list(dict.fromkeys(must_include))[:10])

    ats_sites = "(site:greenhouse.io OR site:lever.co OR site:workdayjobs.com OR site:myworkdayjobs.com OR site:icims.com OR site:successfactors.com OR site:jobvite.com OR site:smartrecruiters.com)"

    def build_query(role: str) -> str:
        visa_part = '"visa sponsorship" OR h1b OR opt OR cpt' if visa_req else ""
        # Strong geo intent: if US, force US synonyms in query.
        geo = location
        if country == "US":
            geo = f"{location} (United States OR USA OR \"United States\")"

        base = f"{role} {geo} {skills_hint} {include_clause}".strip()
        if visa_part:
            base += f" ({visa_part})"

        # Recency intent (Serper tbs handles most of it; keep explicit hint)
        if recency_h <= 36:
            base += " posted today OR posted in last 1 day OR posted in last 24 hours"

        if strategy in ("ats_only", "ats_strict"):
            base = f"{base} {ats_sites}"

        if refinement_feedback:
            base = f"{base} {refinement_feedback}"

        return f"{base} {neg_clause} apply".strip()

    queries = [build_query(r) for r in roles]
    st["discovery_queries"] = queries

    all_hits: List[Dict[str, Any]] = []
    for q in queries:
        ok, conf, data, err = await serper_search(s, q, num=20, tbs=tbs)
        log_attempt(st, layer="L3", agent="DiscoveryAgent", tool="serper.search", model=None,
                    status=("ok" if ok and conf >= 0.55 else ("low_conf" if ok else "failed")),
                    confidence=conf, error=err)
        if ok:
            for it in data:
                it["query"] = q
                it["source"] = "serper"
            all_hits.extend(data)

    # MCP fallback
    if (not all_hits):
        ok2, conf2, data2, err2 = await mcp_invoke(s, "jobs.search", {"queries": queries, "recency_hours": recency_h})
        log_attempt(st, layer="L3", agent="DiscoveryAgent", tool="mcp.jobs.search", model=None,
                    status=("ok" if ok2 and conf2 >= 0.55 else ("low_conf" if ok2 else "failed")),
                    confidence=conf2, error=err2)
        if ok2 and isinstance(data2, dict):
            hits = data2.get("results") or data2.get("jobs") or []
            for it in hits:
                it["source"] = "mcp"
            all_hits.extend(hits)

    seen = set()
    uniq = []
    for it in all_hits:
        link = it.get("link") or it.get("url") or ""
        if link and link not in seen:
            seen.add(link)
            uniq.append({"title": it.get("title") or "", "link": link, "snippet": it.get("snippet") or "", "source": it.get("source") or "unknown"})

    # Strategy enforcement: when evaluator sets ats_only, filter to ATS/career pages.
    strategy = str(st.get("search_strategy") or "default").strip().lower()
    if strategy in ("ats_only", "ats_strict"):
        allow = (
            "greenhouse.io",
            "jobs.lever.co",
            "workdayjobs.com",
            "myworkdayjobs.com",
            "icims.com",
            "successfactors.com",
            "jobvite.com",
            "smartrecruiters.com",
            "careers.",
        )
        uniq = [j for j in uniq if any(a in str(j.get("link") or "").lower() for a in allow)]
    st["jobs_raw"] = uniq
    feed(st, "L3", "DiscoveryAgent", f"Found {len(uniq)} unique jobs.")
    return st

async def EVAL_discovery(st: Dict[str, Any]) -> Dict[str, Any]:
    prefs = st.get("preferences") or {}
    th = float((st.get("thresholds") or {}).get("discovery", prefs.get("discovery_threshold", 0.70)))
    n = len(st.get("jobs_raw") or [])
    score = 0.2 if n < 8 else (0.6 if n < 20 else 0.85)
    fb = []
    if n < 20:
        fb.append("Low job volume. Broaden role titles, widen recency, or loosen visa filter.")
    retries = int((st.get("layer_retry_count") or {}).get("L3", 0))
    decision = gate_decision(score, th, retries, int(st.get("max_retries", 3)))
    log_gate(st, layer="L3", target="discovery", score=score, threshold_v=th, decision=decision, feedback=fb)
    feed(st, "L5", "EvaluatorAgent", f"Discovery score={score:.2f} decision={decision}")
    if decision == "hitl":
        st["status"] = "needs_human_approval"
        st["pending_action"] = "discovery_low_confidence"
        st["hitl_reason"] = "Discovery threshold not met"
        st["hitl_payload"] = {"feedback": fb, "score": score, "threshold": th}
    elif decision == "retry":
        inc_retry(st, "L3")
    return st

# ---------------- L4 ----------------
async def L4_match(st: Dict[str, Any]) -> Dict[str, Any]:
    s = RuntimeSettings()
    prefs = st.get("preferences") or {}
    max_jobs = int(prefs.get("max_jobs", 40))
    visa_req = bool(prefs.get("visa_sponsorship_required", False))

    prof = st.get("profile") or {}
    resume_text = st.get("resume_text") or ""
    resume_skills = [str(x).lower() for x in (prof.get("skills") or [])]
    ats = ats_score(resume_text)

    exp_text = resume_text
    exp = prof.get("experience") or []
    if isinstance(exp, list) and exp:
        e0 = exp[0] if isinstance(exp[0], dict) else {}
        bullets = e0.get("bullets") if isinstance(e0, dict) else []
        if isinstance(bullets, list) and bullets:
            exp_text = " ".join(bullets)
    exp_tokens = {}
    for t in tokenize(exp_text):
        exp_tokens[t] = exp_tokens.get(t, 0) + 1

    jobs = (st.get("jobs_raw") or [])[:max_jobs]
    scored: List[Dict[str, Any]] = []

    for idx, j in enumerate(jobs):
        url = j.get("link") or ""
        snippet = j.get("snippet") or ""
        title = j.get("title") or ""
        source = j.get("source") or "unknown"

        # quick domain hygiene
        low_url = str(url).lower()
        if any(bad in low_url for bad in ["in.talent.com", "shine.com", "naukri", ".in/"]):
            continue

        ok, conf, data, err = await scrape_http(url)
        log_attempt(st, layer="L4", agent="ScraperAgent", tool="httpx.scrape", model=None,
                    status=("ok" if ok and conf >= 0.45 else ("low_conf" if ok else "failed")),
                    confidence=conf, error=err)
        text = (data or {}).get("text") if ok else ""

        if (not ok) or conf < 0.45 or not text:
            ok2, conf2, data2, err2 = await mcp_invoke(s, "web.scrape", {"url": url})
            log_attempt(st, layer="L4", agent="ScraperAgent", tool="mcp.web.scrape", model=None,
                        status=("ok" if ok2 and conf2 >= 0.45 else ("low_conf" if ok2 else "failed")),
                        confidence=conf2, error=err2)
            if ok2 and isinstance(data2, dict):
                text = data2.get("text") or data2.get("content") or text

        text = text or snippet
        low = text.lower()
        visa_ok = not any(x in low for x in VISA_NEGATIVE)
        if visa_req and not visa_ok:
            continue

        matched = [s for s in resume_skills if s and s in low][:30]
        skill_overlap = (len(set(matched)) / max(1, len(set(resume_skills)))) if resume_skills else 0.0

        job_tokens = {}
        for t in tokenize(text):
            job_tokens[t] = job_tokens.get(t, 0) + 1
        exp_align = cosine(exp_tokens, job_tokens)

        market = 1.0
        score = compute_interview_chance(skill_overlap, exp_align, ats, market)

        missing = []
        for kw in ["langgraph","mcp","kubernetes","mlflow","dvc","airflow","kafka","terraform","databricks","snowflake","faiss","chroma"]:
            if kw in low and kw not in resume_skills:
                missing.append(kw)

        scored.append({
            "job_id": url or f"job_{idx}",
            "title": title,
            "url": url,
            "source": source,
            "snippet": snippet,
            "full_text": text[:8000],
            "visa_ok": visa_ok,
            "matched_skills": matched[:12],
            "missing_skills": missing[:12],
            "components": {
                "skill_overlap": float(skill_overlap),
                "experience_alignment": float(exp_align),
                "ats_score": float(ats),
                "market_competition_factor": float(market),
            },
            "score": float(score),
            "match_percent": round(float(score) * 100.0, 2),
        })

    scored.sort(key=lambda x: float(x["score"]), reverse=True)
    st["jobs_scored"] = scored
    feed(st, "L4", "MatcherAgent", f"Scored {len(scored)} jobs.")
    return st

async def EVAL_match(st: Dict[str, Any]) -> Dict[str, Any]:
    prefs = st.get("preferences") or {}
    th = float((st.get("thresholds") or {}).get("match", prefs.get("discovery_threshold", 0.70)))
    jobs = st.get("jobs_scored") or []
    top = float(jobs[0]["score"]) if jobs else 0.0
    fb = []
    if top < th:
        fb.append("Top match low. Try alternate role titles or improve resume keyword alignment.")
    retries = int((st.get("layer_retry_count") or {}).get("L4", 0))
    decision = gate_decision(top, th, retries, int(st.get("max_retries", 3)))
    log_gate(st, layer="L4", target="match", score=top, threshold_v=th, decision=decision, feedback=fb)
    feed(st, "L5", "EvaluatorAgent", f"Match top={top:.2f} decision={decision}")
    if decision == "hitl":
        st["status"] = "needs_human_approval"
        st["pending_action"] = "review_ranking"
    elif decision == "retry":
        inc_retry(st, "L4")
    return st

# ---------------- L5 ----------------
async def L5_rank(st: Dict[str, Any]) -> Dict[str, Any]:
    prefs = st.get("preferences") or {}
    top_n = int(prefs.get("top_n", 30))
    min_accept = float(prefs.get("min_match", 0.50))
    jobs = st.get("jobs_scored") or []
    if not jobs:
        st["status"] = "needs_human_approval"
        st["pending_action"] = "no_jobs"
        return st

    ranked = []
    flagged = []
    reasoning_chain = []

    for j in jobs[:max(50, top_n)]:
        s = float(j["score"])
        if s >= min_accept:
            ranked.append(j)
        else:
            text = (j.get("full_text") or "").lower()
            if any(x in text for x in RARE_SKILL_SIGNALS):
                reasoning_chain = [
                    f"Match score {s:.2f} < 0.50, but rare-skill signal detected.",
                    "Rare skills reduce applicant pool and increase interview odds.",
                    "Bypassing rejection → flagged for HITL review."
                ]
                flagged.append(j)

    ranked.sort(key=lambda x: float(x["score"]), reverse=True)
    st["ranking"] = ranked[:top_n]
    st.setdefault("hitl_payload", {})
    st["hitl_payload"]["flagged_low_score_high_potential"] = flagged[:20]
    st["hitl_payload"]["reasoning_chain"] = reasoning_chain
    st["status"] = "needs_human_approval"
    st["pending_action"] = "review_ranking"
    feed(st, "L5", "Ranker", f"Ranking ready: {len(st['ranking'])} jobs. Flagged={len(flagged)}")
    return st

# ---------------- PUBLIC RUNNERS ----------------
async def run_full_pipeline(st: Dict[str, Any]) -> Dict[str, Any]:
    st["status"] = "running"
    st["pending_action"] = None
    feed(st, "L1", "Orchestrator", "Automation active: running L0→L5…")

    st = await L0_security(st)
    if st.get("status") in ("blocked", "needs_human_approval"):
        return st

    st = await L2_parse(st)
    st = await EVAL_parser(st)
    if st.get("status") == "blocked":
        return st

    st = await L3_discovery(st)
    st = await EVAL_discovery(st)
    if st.get("status") in ("blocked", "needs_human_approval"):
        return st

    st = await L4_match(st)
    st = await EVAL_match(st)
    st = await L5_rank(st)
    return st

async def run_single_layer(st: Dict[str, Any], layer: str) -> Dict[str, Any]:
    layer = layer.upper()
    if layer == "L0": return await L0_security(st)
    if layer == "L2": st = await L2_parse(st); return await EVAL_parser(st)
    if layer == "L3": st = await L3_discovery(st); return await EVAL_discovery(st)
    if layer == "L4": st = await L4_match(st); return await EVAL_match(st)
    if layer == "L5": return await L5_rank(st)

    # NEW: L6–L9
    if layer == "L6":
        cg = await l6_draft_node(st)  # reuse state dict as CareerGraphState compatible
        st.update(cg)
        ev = await l6_evaluator_node(st)
        st.update(ev)
        return st
    if layer == "L7":
        cg = await l7_apply_node(st)
        st.update(cg)
        ev = await l7_evaluator_node(st)
        st.update(ev)
        return st
    if layer == "L8":
        cg = await l8_tracker_node(st)
        st.update(cg)
        ev = await l8_evaluator_node(st)
        st.update(ev)
        return st
    if layer == "L9":
        cg = await l9_analytics_node(st)
        st.update(cg)
        return st

    feed(st, "L1", "Engineer", f"Layer {layer} not implemented.")
    return st

# ---------------- HITL FLOWS (now run real L6–L9 nodes) ----------------
async def approve_ranking_flow(st: Dict[str, Any]) -> Dict[str, Any]:
    st["status"] = "running"
    st["pending_action"] = None
    feed(st, "L6", "HITL", "Ranking approved. Generating drafts…")

    st.update(await l6_draft_node(st))
    st.update(await l6_evaluator_node(st))

    # Pause for draft approval
    st["status"] = "needs_human_approval"
    st["pending_action"] = "review_drafts"
    return st

async def approve_drafts_flow(st: Dict[str, Any]) -> Dict[str, Any]:
    st["status"] = "running"
    st["pending_action"] = None
    feed(st, "L7", "HITL", "Drafts approved. Applying + tracking + analytics…")

    st.update(await l7_apply_node(st))
    st.update(await l7_evaluator_node(st))
    if st.get("status") == "needs_human_approval":
        return st

    st.update(await l8_tracker_node(st))
    st.update(await l8_evaluator_node(st))
    if st.get("status") == "needs_human_approval":
        return st

    st.update(await l9_analytics_node(st))

    st["status"] = "completed"
    st["pending_action"] = None
    feed(st, "L9", "Analytics", "Finalized: completed.")
    return st
