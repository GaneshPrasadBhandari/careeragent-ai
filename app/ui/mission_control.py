# app/ui/mission_control.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st


# -----------------------------
# UI constants + styling
# -----------------------------
LAYER_ORDER = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]

LAYER_TITLES = {
    "L0": "Security & Guardrails",
    "L1": "Mission Control (UI)",
    "L2": "Intake Bundle (Parsing/Profile)",
    "L3": "Discovery (Hunt / Job Boards)",
    "L4": "Scrape + Match + Score",
    "L5": "Evaluator + Ranking + HITL",
    "L6": "Drafting (ATS Resume + Cover)",
    "L7": "Apply Executor + Notifications",
    "L8": "Tracking (DB + Status)",
    "L9": "Analytics + Learning Center + XAI",
}

DEFAULT_API = os.getenv("API_URL", "http://127.0.0.1:8000")


def _inject_css() -> None:
    """Description: Inject CSS for cards/badges. Layer: L1 Input: None Output: styled UI"""
    st.markdown(
        """
<style>
/* layout polish */
.block-container { padding-top: 1.2rem; }

/* badges */
.badge {
  display:inline-block; padding: 0.22rem 0.55rem; border-radius: 999px;
  font-size: 0.82rem; font-weight: 600; margin-right: 0.4rem;
  border: 1px solid rgba(255,255,255,0.12);
}
.badge-green { background: rgba(30, 180, 90, 0.18); color: #b7ffd1; }
.badge-amber { background: rgba(255, 180, 0, 0.18); color: #ffe0a6; }
.badge-red   { background: rgba(255, 80, 80, 0.18); color: #ffb8b8; }
.badge-blue  { background: rgba(90, 160, 255, 0.18); color: #cfe4ff; }

/* job cards */
.card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 0.85rem;
  margin-bottom: 0.65rem;
  background: rgba(255,255,255,0.03);
}
.card-title { font-size: 1.02rem; font-weight: 700; margin-bottom: 0.25rem; }
.card-sub { opacity: 0.85; font-size: 0.9rem; margin-bottom: 0.4rem; }
.card-meta { opacity: 0.8; font-size: 0.82rem; }

.score-pill {
  display:inline-block; padding: 0.18rem 0.45rem; border-radius: 999px;
  font-size: 0.8rem; font-weight: 700; border: 1px solid rgba(255,255,255,0.12);
}
.score-hi { background: rgba(30, 180, 90, 0.18); color: #b7ffd1; }
.score-md { background: rgba(255, 180, 0, 0.18); color: #ffe0a6; }
.score-lo { background: rgba(255, 80, 80, 0.18); color: #ffb8b8; }

/* clickable link */
a { text-decoration: none !important; }
</style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# API helpers
# -----------------------------
def api_get(api_base: str, path: str, timeout: int = 30) -> requests.Response:
    """Description: GET wrapper. Layer: L1 Input: url Output: response"""
    return requests.get(f"{api_base}{path}", timeout=timeout)


def api_post(api_base: str, path: str, timeout: int = 60, **kwargs) -> requests.Response:
    """Description: POST wrapper. Layer: L1 Input: url + payload Output: response"""
    return requests.post(f"{api_base}{path}", timeout=timeout, **kwargs)


def safe_json(resp: requests.Response) -> Dict[str, Any]:
    """Description: Safe JSON parse. Layer: L1 Input: response Output: dict"""
    try:
        return resp.json()
    except Exception:
        return {"_raw": resp.text[:2500], "_status_code": resp.status_code}


def is_backend_online(api_base: str) -> Tuple[bool, str]:
    """Description: Ping backend. Layer: L1 Input: api_base Output: ok + msg"""
    try:
        r = api_get(api_base, "/health", timeout=4)
        if r.status_code == 200:
            return True, "API Online"
        return False, f"API issue {r.status_code}"
    except Exception as e:
        return False, str(e)


# -----------------------------
# State normalization
# Supports both:
# - OrchestrationState-like (steps, artifacts, meta.live_feed, evaluations)
# - LangGraph state (attempts, gates, ranking, jobs_scored, bridge_docs)
# -----------------------------
@dataclass
class Normalized:
    run_id: str
    status: str
    pending: Optional[str]
    live_feed: List[Dict[str, Any]]
    attempts: List[Dict[str, Any]]
    gates: List[Dict[str, Any]]
    evaluations: List[Dict[str, Any]]
    artifacts: Dict[str, Any]
    profile: Dict[str, Any]
    jobs_raw: List[Dict[str, Any]]
    jobs_scraped: List[Dict[str, Any]]
    jobs_scored: List[Dict[str, Any]]
    ranking: List[Dict[str, Any]]
    drafts: Dict[str, Any]
    bridge_docs: Dict[str, Any]
    meta: Dict[str, Any]
    steps: List[Dict[str, Any]]


def normalize_state(st_json: Dict[str, Any]) -> Normalized:
    """Description: Normalize backend state into consistent structure. Layer: L1"""
    run_id = str(st_json.get("run_id") or st_json.get("runId") or "")
    status = str(st_json.get("status") or "unknown")

    meta = st_json.get("meta") or st_json.get("metadata") or {}
    pending = meta.get("pending_action") or st_json.get("pending_action")

    # Live feed can live in meta.live_feed or top-level live_feed
    live_feed = (meta.get("live_feed") or st_json.get("live_feed") or []) or []

    # Attempts: for LangGraph it’s top-level attempts; for OrchestrationState might not exist
    attempts = st_json.get("attempts") or meta.get("attempts") or []
    # Gates: for LangGraph it’s top-level gates; otherwise in evaluations
    gates = st_json.get("gates") or meta.get("gates") or []
    evaluations = st_json.get("evaluations") or meta.get("evaluations") or []

    artifacts = st_json.get("artifacts") or {}

    profile = st_json.get("profile") or meta.get("profile") or {}

    jobs_raw = st_json.get("jobs_raw") or []
    jobs_scraped = st_json.get("jobs_scraped") or []
    jobs_scored = st_json.get("jobs_scored") or []
    ranking = st_json.get("ranking") or []

    drafts = st_json.get("drafts") or {}
    bridge_docs = st_json.get("bridge_docs") or st_json.get("bridgeDocs") or {}

    steps = st_json.get("steps") or []

    # If ranking is stored as artifact on OrchestrationState, load it
    if not ranking and isinstance(artifacts, dict):
        rk = artifacts.get("ranking")
        if isinstance(rk, dict) and rk.get("path") and Path(rk["path"]).exists():
            try:
                ranking = json.loads(Path(rk["path"]).read_text(encoding="utf-8"))
            except Exception:
                ranking = []

    # If drafts bundle is stored as artifact
    if not drafts and isinstance(artifacts, dict):
        db = artifacts.get("drafts_bundle") or artifacts.get("drafts_bundle_json") or artifacts.get("drafts")
        if isinstance(db, dict) and db.get("path") and Path(db["path"]).exists():
            try:
                drafts = json.loads(Path(db["path"]).read_text(encoding="utf-8"))
            except Exception:
                drafts = {}

    return Normalized(
        run_id=run_id,
        status=status,
        pending=pending,
        live_feed=live_feed,
        attempts=[a if isinstance(a, dict) else getattr(a, "__dict__", {}) for a in attempts],
        gates=[g if isinstance(g, dict) else getattr(g, "__dict__", {}) for g in gates],
        evaluations=[e if isinstance(e, dict) else getattr(e, "__dict__", {}) for e in evaluations],
        artifacts=artifacts if isinstance(artifacts, dict) else {},
        profile=profile if isinstance(profile, dict) else {},
        jobs_raw=jobs_raw if isinstance(jobs_raw, list) else [],
        jobs_scraped=jobs_scraped if isinstance(jobs_scraped, list) else [],
        jobs_scored=jobs_scored if isinstance(jobs_scored, list) else [],
        ranking=ranking if isinstance(ranking, list) else [],
        drafts=drafts if isinstance(drafts, dict) else {},
        bridge_docs=bridge_docs if isinstance(bridge_docs, dict) else {},
        meta=meta if isinstance(meta, dict) else {},
        steps=steps if isinstance(steps, list) else [],
    )


# -----------------------------
# Progress & badges
# -----------------------------
def compute_progress(n: Normalized) -> float:
    """Description: Compute progress heuristic. Layer: L1 Input: normalized Output: 0..1"""
    # Prefer steps if available
    if n.steps:
        total = max(1, len(n.steps))
        done = sum(1 for s in n.steps if s.get("finished_at_utc"))
        return min(1.0, done / total)

    # Fallback: key presence
    score = 0
    if n.profile:
        score += 2
    if n.jobs_raw:
        score += 2
    if n.jobs_scored:
        score += 2
    if n.ranking:
        score += 2
    if n.drafts:
        score += 2
    return min(1.0, score / 10.0)


def workflow_badge(status: str, pending: Optional[str]) -> str:
    """Description: Render workflow status badge html. Layer: L1"""
    if status == "needs_human_approval" or (pending and "hitl" in str(pending)):
        return '<span class="badge badge-amber">Human Approval Pending</span>'
    if status in ("blocked", "failed"):
        return '<span class="badge badge-red">Blocked / Failed</span>'
    if status == "completed":
        return '<span class="badge badge-green">Completed</span>'
    return '<span class="badge badge-green">Automation Active</span>'


def score_pill(pct: float) -> str:
    """Description: Score pill html. Layer: L1"""
    if pct >= 75:
        cls = "score-pill score-hi"
    elif pct >= 55:
        cls = "score-pill score-md"
    else:
        cls = "score-pill score-lo"
    return f'<span class="{cls}">{pct:.1f}%</span>'


# -----------------------------
# Glass-box layer details
# -----------------------------
def attempts_for_layer(attempts: List[Dict[str, Any]], layer_id: str) -> List[Dict[str, Any]]:
    """Description: Filter attempts by layer_id. Layer: L1"""
    out = [a for a in attempts if str(a.get("layer_id", "")).upper() == layer_id.upper()]
    return out


def gate_for_layer(gates: List[Dict[str, Any]], layer_id: str) -> Optional[Dict[str, Any]]:
    """Description: Get last gate for layer. Layer: L1"""
    items = [g for g in gates if str(g.get("layer_id", "")).upper() == layer_id.upper()]
    return items[-1] if items else None


def eval_for_layer(evals: List[Dict[str, Any]], layer_id: str) -> Optional[Dict[str, Any]]:
    """Description: Get last evaluation for layer. Layer: L1"""
    items = [e for e in evals if str(e.get("layer_id", "")).upper() == layer_id.upper()]
    return items[-1] if items else None


def render_tool_audit(attempts: List[Dict[str, Any]]) -> None:
    """Description: Render tool audit list. Layer: L1"""
    if not attempts:
        st.caption("No tool attempts recorded yet.")
        return
    for i, a in enumerate(attempts[-30:], start=1):
        tool = a.get("tool")
        model = a.get("model")
        status = a.get("status")
        conf = float(a.get("confidence", 0.0) or 0.0)
        err = a.get("error")
        line = f"Attempt {i}: {model + ' + ' if model else ''}{tool} — {status} (conf={conf:.2f})"
        if err:
            line += f" — err={str(err)[:120]}"
        st.write(line)


def render_eval_reasoning(gate: Optional[Dict[str, Any]], ev: Optional[Dict[str, Any]]) -> None:
    """Description: Render evaluator reasoning. Layer: L1"""
    if gate:
        st.write(f"**Decision:** {gate.get('decision')} | **Score:** {float(gate.get('score',0))*100:.1f}% | **Threshold:** {float(gate.get('threshold',0))*100:.1f}%")
        fb = gate.get("feedback") or []
        if fb:
            st.write("**Feedback:**")
            for x in fb[:8]:
                st.write(f"- {x}")
        rc = gate.get("reasoning_chain") or []
        if rc:
            st.write("**ReasoningChain (Bypass/Explain):**")
            for x in rc[:10]:
                st.write(f"- {x}")
        return

    if ev:
        st.write(f"**Score:** {float(ev.get('evaluation_score',0))*100:.1f}% | **Threshold:** {float(ev.get('threshold',0))*100:.1f}%")
        fb = ev.get("feedback") or []
        if fb:
            st.write("**Feedback:**")
            for x in fb[:8]:
                st.write(f"- {x}")
        return

    st.caption("No evaluator output for this layer yet.")


def render_layer_panel(n: Normalized, layer_id: str) -> None:
    """Description: Glass-box layer panel. Layer: L1"""
    title = LAYER_TITLES.get(layer_id, layer_id)
    with st.expander(f"{layer_id} — {title}", expanded=False):
        st.markdown(workflow_badge(n.status, n.pending), unsafe_allow_html=True)

        st.markdown("#### Tool Audit")
        render_tool_audit(attempts_for_layer(n.attempts, layer_id))

        st.markdown("#### Evaluation Reasoning")
        render_eval_reasoning(gate_for_layer(n.gates, layer_id), eval_for_layer(n.evaluations, layer_id))

        st.markdown("#### Layer Output Snapshot")
        if layer_id == "L2":
            st.json(n.profile or {"note": "profile not available"})
        elif layer_id == "L3":
            st.write(f"Queries: {n.meta.get('discovery_queries') or n.meta.get('queries') or []}")
            st.write(f"Raw jobs: {len(n.jobs_raw)}")
        elif layer_id == "L4":
            st.write(f"Scored jobs: {len(n.jobs_scored)}")
            if n.jobs_scored:
                st.json(n.jobs_scored[0])
        elif layer_id == "L5":
            st.write(f"Ranking size: {len(n.ranking)}")
            flagged = (n.meta.get("hitl_payload") or {}).get("flagged_low_score_high_potential") or n.meta.get("flagged_low_score_high_potential") or []
            if flagged:
                st.warning(f"Flagged high-potential low-score jobs: {len(flagged)}")
        elif layer_id == "L6":
            st.json(n.drafts or {"note": "drafts not ready"})
        elif layer_id == "L9":
            st.json(n.bridge_docs or {"note": "bridge docs not ready"})
        else:
            st.caption("No structured snapshot configured for this layer yet.")


# -----------------------------
# Approval Grid + Preview
# -----------------------------
def job_list(n: Normalized) -> List[Dict[str, Any]]:
    """Description: Select best jobs list from state. Layer: L1"""
    if n.ranking:
        return n.ranking
    if n.jobs_scored:
        # sort by score if needed
        items = sorted(n.jobs_scored, key=lambda x: float(x.get("score", 0.0) or x.get("interview_chance_score", 0.0)), reverse=True)
        return items
    return []


def job_score_pct(job: Dict[str, Any]) -> float:
    """Description: Extract score percent from job dict. Layer: L1"""
    if "overall_match_percent" in job:
        return float(job.get("overall_match_percent") or 0.0)
    if "match_percent" in job:
        return float(job.get("match_percent") or 0.0)
    if "score" in job:
        return float(job.get("score") or 0.0) * 100.0
    if "interview_chance_score" in job:
        return float(job.get("interview_chance_score") or 0.0) * 100.0
    return 0.0


def job_id(job: Dict[str, Any]) -> str:
    """Description: Resolve a stable job id. Layer: L1"""
    return str(job.get("job_id") or job.get("url") or job.get("link") or job.get("title") or "")


def render_job_card(job: Dict[str, Any], selected: bool) -> None:
    """Description: Render a single job card. Layer: L1"""
    title = job.get("title") or job.get("role_title") or "Role"
    company = job.get("company") or job.get("board") or job.get("source") or "Company"
    url = job.get("url") or job.get("link") or ""
    pct = job_score_pct(job)
    pill = score_pill(pct)

    sel = " (selected)" if selected else ""
    st.markdown(
        f"""
<div class="card">
  <div class="card-title">{title}{sel}</div>
  <div class="card-sub">{company}</div>
  <div>{pill}</div>
  <div class="card-meta">
    <div>URL: <a href="{url}" target="_blank">{url[:80] + ('…' if len(url)>80 else '')}</a></div>
    <div>Missing skills: {", ".join((job.get("missing_skills") or job.get("missing_required_skills") or [])[:6])}</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def extract_job_text(job: Dict[str, Any]) -> str:
    """Description: Get job description text for preview. Layer: L1"""
    return (job.get("full_text") or job.get("requirements_text") or job.get("snippet") or "").strip()


def extract_draft_texts(n: Normalized, job: Dict[str, Any]) -> Tuple[str, str]:
    """
    Description: Retrieve tailored resume + cover text for a job.
    Layer: L1
    """
    jid = job_id(job)
    drafts_bundle = n.drafts or {}

    # common structure: {"drafts":[{job_id,resume_path,cover_path,...}], "learning_plan": {...}}
    drafts = drafts_bundle.get("drafts") if isinstance(drafts_bundle, dict) else None
    if isinstance(drafts, list):
        for d in drafts:
            if str(d.get("job_id")) == jid:
                rp = d.get("resume_path")
                cp = d.get("cover_path")
                resume_txt = Path(rp).read_text(encoding="utf-8") if rp and Path(rp).exists() else ""
                cover_txt = Path(cp).read_text(encoding="utf-8") if cp and Path(cp).exists() else ""
                return resume_txt, cover_txt

    # fallback: no drafts yet
    return "", ""


# -----------------------------
# Learning Center + Analytics
# -----------------------------
def render_learning_center(n: Normalized) -> None:
    """Description: Learning center view. Layer: L9"""
    st.subheader("Learning Center — Skill Gaps → Tutorials/Docs")
    jobs = job_list(n)
    if not jobs:
        st.info("No jobs ranked yet. Run discovery + matching first.")
        return

    # Aggregate missing skills
    missing = {}
    for j in jobs[:30]:
        for s in (j.get("missing_skills") or j.get("missing_required_skills") or []):
            missing[str(s).strip()] = missing.get(str(s).strip(), 0) + 1
    missing_sorted = sorted(missing.items(), key=lambda x: x[1], reverse=True)[:20]

    if not missing_sorted:
        st.success("No missing skills detected (or not provided by backend).")
        return

    st.write("Top gaps across ranked jobs:")
    for skill, cnt in missing_sorted:
        st.write(f"- **{skill}** (appears in {cnt} jobs)")

    # Show bridge docs / learning links if present
    bridge = n.bridge_docs or {}
    drafts = n.drafts or {}
    learning_plan = {}
    if isinstance(drafts, dict):
        learning_plan = drafts.get("learning_plan") or {}

    st.markdown("### Tutorials & Docs")
    # If backend keyed by job_id -> plan, show for selected job first
    sel = st.session_state.get("selected_job_id")
    if sel and isinstance(learning_plan, dict) and sel in learning_plan:
        st.info(f"Showing learning plan for selected job: {sel}")
        st.json(learning_plan.get(sel))
        return

    # Otherwise show first available skill plans
    # (bridge docs might be keyed by skill)
    if isinstance(bridge, dict) and bridge:
        st.json(bridge)
    elif isinstance(learning_plan, dict) and learning_plan:
        # flatten
        any_job = next(iter(learning_plan.keys()))
        st.info(f"Showing learning plan sample for job: {any_job}")
        st.json(learning_plan.get(any_job))
    else:
        st.caption("Learning links not available yet. Generate drafts to produce bridge documents.")


def render_analytics(n: Normalized) -> None:
    """Description: Analytics view with applied tracker + interview calendar board. Layer: L9"""
    st.subheader("Analytics — Applied Jobs + Interview Calendar")

    applied = (n.meta.get("applied_jobs") or n.meta.get("submissions") or []) or []
    interviews = (n.meta.get("interviews") or []) or []

    st.markdown("### Applied Jobs Tracker")
    if applied:
        st.dataframe(applied, use_container_width=True)
    else:
        st.caption("No applied jobs tracked yet (backend should store submissions into state.meta).")

    st.markdown("### Interview Calendar (board view)")
    # Expect interview items like: {"date":"2026-03-02","time":"10:00","company":"X","role":"Y","stage":"Phone"}
    if not interviews:
        st.caption("No interviews scheduled yet.")
        return

    # Group by date
    by_date: Dict[str, List[Dict[str, Any]]] = {}
    for it in interviews:
        d = str(it.get("date") or "unknown")
        by_date.setdefault(d, []).append(it)

    dates = sorted(by_date.keys())
    cols = st.columns(min(7, max(1, len(dates))))
    for i, d in enumerate(dates[:7]):
        with cols[i]:
            st.markdown(f"**{d}**")
            for ev in by_date[d]:
                st.markdown(
                    f"- {ev.get('time','')} **{ev.get('company','')}** — {ev.get('role','')} ({ev.get('stage','')})"
                )


# -----------------------------
# Engineer controls (per layer)
# -----------------------------
def engineer_controls(api_base: str, run_id: str) -> None:
    """Description: Render engineer mode buttons for each layer. Layer: L1"""
    st.markdown("### Engineer View — Run Any Layer (L0 → L9)")
    st.caption("These buttons call /action with action_type=execute_layer. Backend must implement it.")

    cols = st.columns(5)
    for i, lid in enumerate(LAYER_ORDER):
        with cols[i % 5]:
            if st.button(f"Run {lid}", use_container_width=True):
                r = api_post(api_base, f"/action/{run_id}", json={"action_type": "execute_layer", "payload": {"layer": lid}}, timeout=45)
                if r.status_code >= 400:
                    st.error(f"{lid} execute failed: {r.status_code} {r.text[:200]}")
                else:
                    st.success(f"{lid} triggered")


# -----------------------------
# Main UI
# -----------------------------
def main() -> None:
    """
    Description: Streamlit Mission Control UI for multi-agent LangGraph.
    Layer: L1
    Input: backend state via API
    Output: interactive control + visualization
    """
    st.set_page_config(page_title="CareerAgent-AI Mission Control", layout="wide")
    _inject_css()

    # Session defaults
    st.session_state.setdefault("run_id", "")
    st.session_state.setdefault("view_mode", "Pilot View")
    st.session_state.setdefault("selected_job_id", None)
    st.session_state.setdefault("auto_refresh", True)
    st.session_state.setdefault("refresh_secs", 2)

    # Sidebar
    with st.sidebar:
        st.header("Mission Control")
        api_base = st.text_input("API Base URL", value=DEFAULT_API)

        ok, msg = is_backend_online(api_base)
        if ok:
            st.success("🟢 Backend Online")
        else:
            st.error("🔴 Backend Offline")
            st.caption(msg)

        st.divider()
        st.session_state["view_mode"] = st.selectbox("View Mode", ["Pilot View", "Engineer View"], index=0)
        st.session_state["auto_refresh"] = st.checkbox("Live Update", value=st.session_state["auto_refresh"])
        st.session_state["refresh_secs"] = st.slider("Refresh interval (sec)", 1, 10, int(st.session_state["refresh_secs"]), 1)

        st.divider()
        st.subheader("Start Hunt")
        resume_file = st.file_uploader("Upload Resume (PDF/TXT/DOCX)", type=["pdf", "txt", "docx"])

        roles_text = st.text_area("Target Roles (1 per line, up to 4)", value="Data Scientist\nML Engineer\nGenAI Engineer", height=90)
        target_roles = [r.strip() for r in roles_text.splitlines() if r.strip()][:4]

        country = st.text_input("Country", value="US")
        location = st.text_input("Location", value="United States")
        remote = st.checkbox("Remote preferred", value=True)
        wfo_ok = st.checkbox("On-site/WFO acceptable", value=True)
        salary = st.text_input("Salary target (optional)", value="")
        visa_required = st.checkbox("Visa sponsorship required (F1/OPT)", value=False)

        recency_hours = st.slider("Only jobs posted within (hours)", 12, 168, 36, 6)
        max_jobs = st.slider("Scrape/score jobs per run", 20, 60, 40, 5)

        st.subheader("Threshold Overrides")
        th_global = st.slider("Default threshold", 0.50, 0.90, 0.70, 0.05)
        th_parser = st.slider("Parser threshold", 0.40, 0.90, 0.70, 0.05)
        th_discovery = st.slider("Discovery threshold", 0.40, 0.90, 0.70, 0.05)
        th_match = st.slider("Match threshold", 0.40, 0.90, 0.70, 0.05)
        max_refinements = st.slider("Max retries / refinements", 1, 6, 3, 1)

        user_phone = st.text_input("Phone for SMS (optional)", value="")

        start_btn = st.button("🚀 Start Hunt", type="primary", use_container_width=True, disabled=(resume_file is None))

        st.divider()
        st.subheader("Existing Run")
        run_id_in = st.text_input("Run ID", value=st.session_state["run_id"])

    # Start run
    if start_btn:
        prefs = {
            "target_roles": target_roles,
            "country": country,
            "location": location,
            "remote": remote,
            "wfo_ok": wfo_ok,
            "salary": salary,
            "visa_sponsorship_required": visa_required,
            "recency_hours": float(recency_hours),
            "max_jobs": int(max_jobs),
            "max_refinements": int(max_refinements),
            "user_phone": user_phone.strip() or None,
            # thresholds (backend should honor these)
            "thresholds": {"default": float(th_global), "parser": float(th_parser), "discovery": float(th_discovery), "match": float(th_match)},
        }

        files = {"resume": (resume_file.name, resume_file.getvalue())}
        data = {"preferences_json": json.dumps(prefs)}

        try:
            r = api_post(api_base, "/analyze", files=files, data=data, timeout=180)
            if r.status_code >= 400:
                st.error(f"/analyze failed: {r.status_code} {r.text[:800]}")
            else:
                out = safe_json(r)
                st.session_state["run_id"] = out.get("run_id", "")
                st.success(f"Run started: {st.session_state['run_id']}")
        except Exception as e:
            st.error(f"API not reachable: {e}")

    # Determine run id
    run_id = (st.session_state.get("run_id") or run_id_in or "").strip()
    if not run_id:
        st.info("Upload resume and click Start Hunt, or paste an existing Run ID in the sidebar.")
        return

    # Real-time placeholders
    header_box = st.empty()
    progress_box = st.empty()
    feed_box = st.empty()
    tabs_box = st.empty()

    def render_once() -> None:
        # Fetch state
        r = api_get(api_base, f"/status/{run_id}", timeout=30)
        if r.status_code != 200:
            st.error(f"Run not found: {r.status_code} {r.text[:300]}")
            return
        raw = safe_json(r)
        n = normalize_state(raw)

        # Header
        header_box.markdown(
            f"""
<div style="display:flex; align-items:center; justify-content:space-between;">
  <div>
    <h2 style="margin:0;">CareerAgent-AI — Glass-Box Mission Control</h2>
    <div style="opacity:0.85;">Run: <b>{n.run_id}</b></div>
  </div>
  <div>{workflow_badge(n.status, n.pending)}</div>
</div>
            """,
            unsafe_allow_html=True,
        )

        # Progress
        prog = compute_progress(n)
        with progress_box.container():
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.metric("Status", n.status)
            with c2:
                st.metric("Pending", str(n.pending))
            with c3:
                st.metric("Progress", f"{prog*100:.0f}%")
            st.progress(prog)

        # Live Agent Feed
        with feed_box.container():
            st.markdown("### Live Agent Feed")
            if not n.live_feed:
                st.caption("No feed yet.")
            else:
                for ev in n.live_feed[-120:]:
                    st.write(f"**[{ev.get('layer')} {ev.get('agent')}]** {ev.get('message')}")

        # Tabs
        with tabs_box.container():
            t1, t2, t3, t4 = st.tabs(["Pilot/Engineer", "Approval Grid", "Learning Center", "Analytics"])

            with t1:
                if st.session_state["view_mode"] == "Pilot View":
                    st.subheader("Pilot View")
                    st.caption("One-click monitoring. Use Start Hunt on the left.")
                else:
                    st.subheader("Engineer View")
                    engineer_controls(api_base, n.run_id)

                st.markdown("### Glass-Box Layers (L0–L9)")
                for lid in LAYER_ORDER:
                    render_layer_panel(n, lid)

            with t2:
                st.subheader("Approval Grid (20–50 jobs)")
                jobs = job_list(n)
                if not jobs:
                    st.info("No jobs available yet. Wait for discovery + scoring.")
                else:
                    # grid controls
                    cols = st.columns([1, 1, 1])
                    with cols[0]:
                        show_n = st.slider("Show jobs", 10, min(50, len(jobs)), min(20, len(jobs)), 5)
                    with cols[1]:
                        min_pct = st.slider("Min score %", 0, 100, 50, 5)
                    with cols[2]:
                        only_visa_ok = st.checkbox("Only visa-ok", value=False)

                    filtered = []
                    for j in jobs:
                        pct = job_score_pct(j)
                        if pct < min_pct:
                            continue
                        if only_visa_ok and j.get("visa_ok") is False:
                            continue
                        filtered.append(j)

                    # job cards
                    for j in filtered[:show_n]:
                        jid = job_id(j)
                        selected = (st.session_state.get("selected_job_id") == jid)
                        colA, colB = st.columns([0.88, 0.12])
                        with colA:
                            render_job_card(j, selected=selected)
                        with colB:
                            if st.button("Select", key=f"sel_{jid}", use_container_width=True):
                                st.session_state["selected_job_id"] = jid

                    # Preview / Action panel
                    st.divider()
                    st.subheader("Preview — Job Description vs Tailored Draft")
                    sel_id = st.session_state.get("selected_job_id")
                    sel_job = None
                    for j in jobs:
                        if job_id(j) == sel_id:
                            sel_job = j
                            break

                    if not sel_job:
                        st.info("Select a job to preview.")
                    else:
                        left, right = st.columns(2)
                        with left:
                            st.markdown("#### Original Job Description")
                            st.text_area("JD", value=extract_job_text(sel_job)[:12000], height=360)
                        with right:
                            resume_txt, cover_txt = extract_draft_texts(n, sel_job)
                            st.markdown("#### Agent-Tailored Resume (preview)")
                            st.text_area("Tailored Resume", value=resume_txt[:12000], height=170)
                            st.markdown("#### Cover Letter (preview)")
                            st.text_area("Cover Letter", value=cover_txt[:12000], height=170)

                        # Action buttons
                        a1, a2, a3 = st.columns([1, 1, 1])
                        with a1:
                            if st.button("✅ Approve", type="primary", use_container_width=True):
                                rr = api_post(api_base, f"/action/{n.run_id}", json={"action_type": "approve_job", "payload": {"job_id": sel_id}}, timeout=45)
                                if rr.status_code >= 400:
                                    st.error(rr.text[:800])
                                else:
                                    st.success("Approved job.")
                        with a2:
                            if st.button("✍️ Approve/Edit", use_container_width=True):
                                rr = api_post(api_base, f"/action/{n.run_id}", json={"action_type": "approve_edit_job", "payload": {"job_id": sel_id}}, timeout=45)
                                if rr.status_code >= 400:
                                    st.error(rr.text[:800])
                                else:
                                    st.success("Sent to edit flow (backend must implement).")
                        with a3:
                            if st.button("❌ Reject", use_container_width=True):
                                rr = api_post(api_base, f"/action/{n.run_id}", json={"action_type": "reject_job", "payload": {"job_id": sel_id}}, timeout=45)
                                if rr.status_code >= 400:
                                    st.error(rr.text[:800])
                                else:
                                    st.warning("Rejected job.")

            with t3:
                render_learning_center(n)

            with t4:
                render_analytics(n)

    # Render once (always)
    render_once()

    # Auto-refresh loop (safe-ish)
    # Streamlit reruns script; we emulate "realtime" by sleeping and rerunning while running.
    if st.session_state["auto_refresh"]:
        # Only auto-refresh if run is active or pending HITL
        # (avoid infinite loops when user just wants to browse)
        time.sleep(float(st.session_state["refresh_secs"]))
        # st.experimental_rerun()
        st.rerun()


if __name__ == "__main__":
    main()