"""
app/ui/mission_control.py
==========================
CareerAgent-AI — Mission Control Dashboard
Fixes applied:
  1. Empty selectbox label (line 1307 in old file) → "View Mode" with label_visibility="collapsed"
  2. Start Hunt properly calls POST /hunt/start with resume upload
  3. Progress bar polls GET /hunt/{run_id}/status and auto-refreshes
  4. Layer cards update status in real-time (running/ok/error/waiting)
  5. Live Agent Feed shows per-agent messages
  6. All stat cards update from live state
"""

from __future__ import annotations

import json
import time
from html import escape
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

import requests
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="CareerAgent-AI — Mission Control",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# STYLING
# ══════════════════════════════════════════════════════════════════════════════

def _inject_css() -> None:
    st.markdown("""
    <style>
    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', 'SF Pro Display', -apple-system, sans-serif;
        background-color: #F8F9FA;
        color: #1B263B;
    }
    .stApp { background-color: #F8F9FA; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #1B263B !important;
        border-right: 1px solid #1e1e2e;
    }
    section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    /* ── Stat card ── */
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #D9DEE5;
        border-radius: 10px;
        padding: 16px 20px;
        min-height: 80px;
    }
    .stat-label { font-size: 11px; color: #6e7681; text-transform: uppercase; letter-spacing: 0.08em; }
    .stat-value { font-size: 28px; font-weight: 700; color: #1B263B; margin: 4px 0 2px; }
    .stat-sub   { font-size: 12px; color: #5C677D; }
    .stat-value.green { color: #2D6A4F; }
    .stat-value.orange { color: #f0883e; }

    /* ── Progress bar container ── */
    .progress-wrap {
        position: sticky; top: 0.5rem; z-index: 20;
        background: #FFFFFF;
        border: 1px solid #D9DEE5;
        border-radius: 10px;
        padding: 16px 20px 20px;
        margin: 12px 0;
    }
    .progress-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .progress-title  { font-size: 12px; color: #6e7681; text-transform: uppercase; letter-spacing: 0.1em; }
    .progress-pct    { font-size: 20px; font-weight: 700; color: #1B263B; }
    .progress-track  { background: #1e1e2e; border-radius: 6px; height: 8px; width: 100%; }
    .progress-fill   { height: 8px; border-radius: 6px; transition: width 0.5s ease;
                        background: linear-gradient(90deg, #1B263B 0%, #2D6A4F 100%); }

    /* ── Layer card ── */
    .layer-card {
        background: #FFFFFF;
        border: 1px solid #D9DEE5;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }
    .layer-card.running { border-left: 3px solid #388bfd; }
    .layer-card.ok      { border-left: 3px solid #3fb950; }
    .layer-card.error   { border-left: 3px solid #f85149; }
    .layer-card.waiting { border-left: 3px solid #21262d; }
    .layer-card.skipped { border-left: 3px solid #8b949e; }

    .layer-header   { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .layer-name     { font-size: 14px; font-weight: 600; color: #1B263B; }
    .layer-status-badge {
        font-size: 11px; padding: 2px 8px; border-radius: 20px; font-weight: 500;
    }
    .badge-waiting  { background: #21262d; color: #8b949e; }
    .badge-running  { background: #1c2d3f; color: #388bfd; }
    .badge-ok       { background: #E6F4EA; color: #2D6A4F; }
    .badge-error    { background: #2d1a1a; color: #f85149; }
    .badge-skipped  { background: #21262d; color: #8b949e; }

    .layer-meta { display: flex; gap: 24px; margin: 8px 0; }
    .meta-item  { font-size: 12px; color: #8b949e; }
    .meta-key   { color: #6e7681; }
    .meta-val   { color: #c9d1d9; }
    .layer-desc    { font-size: 12px; color: #6e7681; margin: 4px 0; }
    .layer-output  { font-size: 12px; color: #8b949e; margin-top: 8px; padding-top: 8px;
                     border-top: 1px solid #1e1e2e; }
    .output-label  { color: #6e7681; font-size: 11px; text-transform: uppercase; }
    .output-val    { color: #c9d1d9; margin-top: 2px; }

    /* ── Agent Feed ── */
    .feed-wrap {
        background: #0d1117;
        border: 1px solid #D9DEE5;
        border-radius: 10px;
        padding: 14px 18px;
        max-height: 200px;
        overflow-y: auto;
        margin-top: 12px;
    }
    .feed-title { font-size: 11px; color: #3fb950; text-transform: uppercase;
                  letter-spacing: 0.1em; margin-bottom: 8px; }
    .feed-entry { font-size: 12px; color: #8b949e; padding: 2px 0; }
    .feed-ts    { color: #3b4a5a; font-size: 11px; margin-right: 8px; }
    .feed-msg   { color: #b1bac4; }
    .feed-empty { color: #3b4a5a; font-size: 12px; font-style: italic; }

    /* ── Section header ── */
    .section-header {
        font-size: 11px; color: #6e7681; text-transform: uppercase;
        letter-spacing: 0.1em; margin: 16px 0 8px; padding-bottom: 4px;
        border-bottom: 1px solid #1e1e2e;
    }

    /* ── Status badge ── */
    .run-status {
        font-size: 12px; padding: 4px 12px; border-radius: 20px;
        background: #21262d; color: #8b949e; font-weight: 500;
    }
    .run-status.running { background: #1c2d3f; color: #388bfd; }
    .run-status.completed { background: #E6F4EA; color: #2D6A4F; }
    .run-status.error { background: #FDECEC; color: #C92A2A; }
    .run-status.pending_human_input { background:#FFF4E5; color:#B26A00; }

    /* ── Job table ── */
    .job-row {
        background: #FFFFFF; border: 1px solid #D9DEE5; border-radius: 8px;
        padding: 12px 16px; margin-bottom: 6px;
        display: flex; justify-content: space-between; align-items: center;
    }
    .job-title   { font-size: 14px; font-weight: 600; color: #1B263B; }
    .job-company { font-size: 12px; color: #8b949e; }
    .job-score   { font-size: 16px; font-weight: 700; color: #3fb950; }
    .job-badge   { font-size: 11px; padding: 2px 8px; border-radius: 20px;
                   background: #1c2d3f; color: #388bfd; }

    /* ── Tab content ── */
    .empty-state {
        text-align: center; padding: 60px 20px; color: #6e7681;
    }
    .empty-icon  { font-size: 48px; margin-bottom: 12px; }
    .empty-title { font-size: 16px; font-weight: 600; color: #8b949e; margin-bottom: 6px; }
    .empty-sub   { font-size: 13px; color: #6e7681; }

    /* ── Pipeline node icons ── */
    .pipeline-nodes { display: flex; justify-content: space-between; margin: 12px 0 4px; }
    .node {
        width: 28px; height: 28px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 12px; flex-shrink: 0;
    }
    .node-waiting   { background: #21262d; color: #6e7681; border: 1px solid #30363d; }
    .node-running   { background: #1c2d3f; color: #388bfd; border: 1px solid #388bfd;
                      animation: pulse 1.5s infinite; }
    .node-ok        { background: #1a2e1a; color: #3fb950; border: 1px solid #3fb950; }
    .node-error     { background: #2d1a1a; color: #f85149; border: 1px solid #f85149; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }

    /* ── Sidebar button ── */
    .stButton > button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 20px !important;
        width: 100% !important;
        cursor: pointer !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%) !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER DEFINITIONS  (mirror backend)
# ══════════════════════════════════════════════════════════════════════════════

LAYERS = [
    {"id": 0, "icon": "🔒", "name": "Security & Guardrails",          "weight": 5,  "agent": "GuardAgent",      "desc": "Sanitizes input, runs guardrail checks, validates API tokens"},
    {"id": 1, "icon": "🖥️", "name": "Mission Control (UI)",            "weight": 5,  "agent": "UIAgent",         "desc": "Initializes UI state, loads run configuration"},
    {"id": 2, "icon": "📄", "name": "Intake Bundle (Parsing/Profile)", "weight": 15, "agent": "ParseAgent",      "desc": "Parses resume via LLM+regex, extracts skills/experience/education, builds search personas"},
    {"id": 3, "icon": "🔍", "name": "Discovery (Hunt / Job Boards)",   "weight": 25, "agent": "HuntAgent",       "desc": "Scrapes LinkedIn & Indeed with Playwright, deduplicates, geo-fences results"},
    {"id": 4, "icon": "⚖️", "name": "Scrape + Match + Score",          "weight": 15, "agent": "MatchAgent",      "desc": "Extracts full JD text, runs semantic + keyword scoring against your profile"},
    {"id": 5, "icon": "🏆", "name": "Evaluator + Ranking + HITL",      "weight": 10, "agent": "EvalAgent",       "desc": "Phase-2 evaluation, ranks by interview probability, triggers HITL gate"},
    {"id": 6, "icon": "✍️", "name": "Drafting (ATS Resume + Cover)",   "weight": 10, "agent": "DraftAgent",      "desc": "Generates tailored ATS resume + cover letter per approved job using LLM"},
    {"id": 7, "icon": "🚀", "name": "Apply Executor + Notifications",  "weight": 5,  "agent": "ApplyAgent",      "desc": "Auto-applies to approved jobs, sends SMS/email notifications"},
    {"id": 8, "icon": "🗄️", "name": "Tracking (DB + Status)",          "weight": 5,  "agent": "TrackAgent",      "desc": "Records applications to DB, updates deduplication memory"},
    {"id": 9, "icon": "📊", "name": "Analytics + Learning Center + XAI","weight": 5, "agent": "AnalyticsAgent",  "desc": "Analytics, self-learning from outcomes, career roadmap, XAI explanations"},
]

DEFAULT_OUTPUTS = [
    "Layer not yet executed.",
    "Layer not yet executed.",
    "Layer not yet executed.",
    "0 raw jobs fetched",
    "0 jobs scored",
    "0 jobs ranked",
    "0 draft packages generated",
    "0 applications submitted",
    "Layer not yet executed.",
    "Bridge docs appear after L9 completes.",
]


# ══════════════════════════════════════════════════════════════════════════════
# API HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _api_get(api_base: str, path: str, timeout: int = 5) -> Optional[dict]:
    try:
        r = requests.get(f"{api_base.rstrip('/')}{path}", timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _api_health(api_base: str) -> bool:
    resp = _api_get(api_base, "/health", timeout=3)
    return resp is not None and resp.get("status") == "ok"


def _api_start_hunt(api_base: str, resume_bytes: bytes, filename: str, config: dict) -> Optional[str]:
    try:
        r = requests.post(
            f"{api_base.rstrip('/')}/hunt/start",
            files={"resume": (filename, resume_bytes, "application/octet-stream")},
            data={"config": json.dumps(config)},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json().get("run_id")
        st.error(f"Backend error {r.status_code}: {r.text[:200]}")
    except requests.exceptions.ConnectionError:
        st.error("🔴 Cannot connect to backend. Make sure `uvicorn careeragent.api.main:app` is running on port 8000.")
    except Exception as exc:
        st.error(f"Start hunt error: {exc}")
    return None


def _api_get_status(api_base: str, run_id: str) -> Optional[dict]:
    raw = _api_get(api_base, f"/hunt/{run_id}/status", timeout=5)
    if not raw:
        return None

    # Backward/alternate backend compatibility: normalize common field variants.
    if "progress_pct" not in raw and "progress_percent" in raw:
        raw["progress_pct"] = raw.get("progress_percent", 0)

    pending = str(raw.get("pending_action") or "").strip().lower() or None
    alias_map = {
        "review_ranking": "approve_ranking",
        "rankings_review": "approve_ranking",
        "review_drafts": "approve_drafts",
        "drafts_review": "approve_drafts",
        "gap_analysis": "update_profile_skills",
        "review_followups": "approve_followups",
    }
    if pending in alias_map:
        raw["pending_action"] = alias_map[pending]
        pending = raw["pending_action"]
    if raw.get("status") in ("needs_human_approval", "pending_human_input") and not pending:
        layers = raw.get("layers") or []
        l5 = layers[5] if len(layers) > 5 else {}
        l6 = layers[6] if len(layers) > 6 else {}
        l7 = layers[7] if len(layers) > 7 else {}
        if l5.get("status") == "ok" and l6.get("status") == "waiting":
            raw["pending_action"] = "approve_ranking"
        elif l6.get("status") == "ok" and l7.get("status") == "waiting":
            raw["pending_action"] = "approve_drafts"
        elif any(str(x.get("draft_status") or "").lower().startswith("pending") for x in (raw.get("followup_queue") or [])):
            raw["pending_action"] = "approve_followups"

    return raw


def _api_get_jobs(api_base: str, run_id: str) -> list[dict]:
    resp = _api_get(api_base, f"/hunt/{run_id}/jobs", timeout=5)
    return resp.get("jobs", []) if resp else []

def _api_get_artifacts(api_base: str, run_id: str) -> dict:
    resp = _api_get(api_base, f"/hunt/{run_id}/artifacts", timeout=8)
    return resp.get("artifacts", {}) if resp else {}


def _api_action(api_base: str, run_id: str, action: str, payload: Optional[dict] = None) -> bool:
    try:
        body = {"action": action, "action_type": action}
        if payload:
            body.update(payload)
        r = requests.post(f"{api_base.rstrip('/')}/hunt/{run_id}/action", json=body, timeout=20)
        if r.status_code == 200:
            return True
        st.error(f"Action failed ({r.status_code}): {r.text[:200]}")
    except Exception as exc:
        st.error(f"Action request failed: {exc}")
    return False


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════

def _init_session():
    defaults = {
        "run_id":         None,
        "run_status":     None,   # full status dict from API
        "view_mode":      "Pilot View",
        "live_update":    True,
        "refresh_sec":    5,
        "api_base":       "http://localhost:8000",
        "last_poll":      0.0,
        "active_tab":     "Pipeline Layers",
        "hunt_running":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def _status_class(status: str) -> str:
    return {"waiting": "badge-waiting", "running": "badge-running",
            "ok": "badge-ok", "error": "badge-error", "skipped": "badge-skipped"}.get(status, "badge-waiting")


def _status_label(status: str) -> str:
    return {"waiting": "○ Waiting", "running": "⟳ Running", "ok": "✓ Done",
            "error": "✗ Error", "skipped": "— Skipped"}.get(status, "○ Waiting")


def _node_class(status: str) -> str:
    return {"waiting": "node-waiting", "running": "node-running",
            "ok": "node-ok", "error": "node-error", "skipped": "node-waiting"}.get(status, "node-waiting")


def render_stat_cards(status: Optional[dict]) -> None:
    """4-column stat cards row."""
    jobs_disc  = status.get("jobs_discovered",  0) if status else 0
    jobs_score = status.get("jobs_scored",       0) if status else 0
    top_match  = status.get("top_match_score",   0.0) if status else 0.0
    approved   = status.get("jobs_approved",     0) if status else 0
    cand_name  = status.get("candidate_name",    "—") if status else "—"
    skills_n   = status.get("skills_extracted",  0) if status else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Jobs Discovered</div>
            <div class="stat-value {'green' if jobs_disc > 0 else ''}">{jobs_disc}</div>
            <div class="stat-sub">{jobs_score} ranked &amp; scored</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        match_color = "green" if top_match >= 70 else ("orange" if top_match >= 45 else "")
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Top Match Score</div>
            <div class="stat-value {match_color}">{top_match:.0f}%</div>
            <div class="stat-sub">Best alignment found</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Approved</div>
            <div class="stat-value {'orange' if approved > 0 else ''}">{approved}</div>
            <div class="stat-sub">Jobs ready to apply</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Candidate</div>
            <div class="stat-value" style="font-size:18px;margin-top:6px">{cand_name}</div>
            <div class="stat-sub">{skills_n} skills extracted</div>
        </div>""", unsafe_allow_html=True)


def render_progress_bar(status: Optional[dict], layers_data: list[dict]) -> None:
    """Pipeline progress bar with node icons."""
    pct = status.get("progress_pct", 0.0) if status else 0.0

    # Node icons HTML
    nodes_html = '<div class="pipeline-nodes">'
    for ld in LAYERS:
        layer_status = layers_data[ld["id"]]["status"] if layers_data else "waiting"
        nodes_html += f'<div class="node {_node_class(layer_status)}" title="L{ld["id"]}: {ld["name"]}">{ld["icon"]}</div>'
    nodes_html += "</div>"

    st.markdown(f"""
    <div class="progress-wrap">
        <div class="progress-header">
            <span class="progress-title">Pipeline Progress — L0 → L9</span>
            <span class="progress-pct">{pct:.1f}%</span>
        </div>
        {nodes_html}
        <div class="progress-track">
            <div class="progress-fill" style="width:{pct}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if layers_data:
        for ld in LAYERS:
            layer_state = layers_data[ld["id"]]
            meta = layer_state.get("meta") or {}
            tools = meta.get("tools_used") or []
            attempts = int(meta.get("attempt_count") or 1)
            tool_txt = " & ".join([str(t) for t in tools]) if tools else "No explicit tools recorded"
            st.caption(f"Step {ld['id']}: Used {tool_txt} | {attempts} attempts")


def render_layer_card(ld: dict, layer_state: dict, expanded: bool = False) -> None:
    """Render one layer card with expandable details."""
    layer_id    = ld["id"]
    status      = layer_state.get("status", "waiting")
    meta        = layer_state.get("meta", {})
    error       = layer_state.get("error", "")
    output      = layer_state.get("output") or DEFAULT_OUTPUTS[layer_id]
    agent       = ld["agent"]
    started_at  = layer_state.get("started_at", "")
    finished_at = layer_state.get("finished_at", "")

    # Compute elapsed time
    time_str = "—"
    if started_at and finished_at:
        try:
            from datetime import datetime, timezone
            t0 = datetime.fromisoformat(started_at)
            t1 = datetime.fromisoformat(finished_at)
            elapsed = (t1 - t0).total_seconds()
            time_str = f"{elapsed:.1f}s"
        except Exception:
            time_str = "—"
    elif started_at and status == "running":
        time_str = "running…"

    badge_cls  = _status_class(status)
    card_cls   = f"layer-card {status}"
    status_lbl = _status_label(status)

    error_html = ""
    if error:
        error_html = f'<div style="color:#f85149;font-size:12px;margin-top:4px">⚠ {escape(str(error))}</div>'

    output_html = f"""
    <div class="layer-output">
        <div class="output-label">Output Snapshot</div>
        <div class="output-val">{escape(str(output))}</div>
    </div>
    """

    with st.expander(f"{ld['icon']}  L{layer_id} · {ld['name']}  {status_lbl}", expanded=expanded):
        st.markdown(f"""
        <div class="{card_cls}">
            <div class="layer-meta">
                <span class="meta-item"><span class="meta-key">Agent:</span> <span class="meta-val">{agent if status != 'waiting' else '—'}</span></span>
                <span class="meta-item"><span class="meta-key">Weight:</span> <span class="meta-val">{ld['weight']}% of total</span></span>
                <span class="meta-item"><span class="meta-key">Status:</span> <span class="meta-val layer-status-badge {badge_cls}">{status}</span></span>
                <span class="meta-item"><span class="meta-key">Time:</span> <span class="meta-val">{time_str}</span></span>
            </div>
            <div class="layer-desc">{ld['desc']}</div>
            {error_html}
            {output_html}
        </div>
        """, unsafe_allow_html=True)


def render_hitl_controls(api_base: str, run_id: Optional[str], status: Optional[dict]) -> None:
    if not run_id or not status:
        return
    pending = str(status.get("pending_action") or "").strip().lower() or None
    alias_map = {
        "review_ranking": "approve_ranking",
        "rankings_review": "approve_ranking",
        "review_drafts": "approve_drafts",
        "drafts_review": "approve_drafts",
        "gap_analysis": "update_profile_skills",
        "review_followups": "approve_followups",
    }
    if pending in alias_map:
        pending = alias_map[pending]
    if pending in {"human_approval", "approval", "rank_approval"}:
        pending = "approve_ranking"
    if pending in {"draft_approval", "approve_documents"}:
        pending = "approve_drafts"
    if pending in {"followup_approval", "review_followups"}:
        pending = "approve_followups"

    waiting_for_human = status.get("status") in ("needs_human_approval", "pending_human_input") or bool(pending)
    if not waiting_for_human:
        return

    if not pending:
        st.info("Run is waiting for approval. Approval type was inferred from layer state/job outputs.")
        layers = status.get("layers") or []
        l5 = layers[5] if len(layers) > 5 else {}
        l6 = layers[6] if len(layers) > 6 else {}
        l7 = layers[7] if len(layers) > 7 else {}
        if l5.get("status") == "ok" and l6.get("status") == "waiting":
            pending = "approve_ranking"
        elif l6.get("status") == "ok" and l7.get("status") == "waiting":
            pending = "approve_drafts"
        else:
            st.warning("Approval state is missing from backend response. Open Full run JSON below to inspect.")
            return

    if not pending:
        return

    st.markdown('<div class="section-header">Human-in-the-Loop Approval Required</div>', unsafe_allow_html=True)

    if pending == "approve_ranking":
        st.warning("Ranking evaluator is waiting for your decision. Select recommended jobs and approve, or reject to re-plan from intake.")
        ranked_jobs = (status.get("layer_debug") or {}).get("L5", {}).get("qualified_jobs", []) or status.get("approved_jobs_preview", [])
        if ranked_jobs:
            options = {
                f"{j.get('title','Role')} · {j.get('company','')} "
                f"(match {j.get('score',0)*100:.0f}% | interview {j.get('interview_probability_percent',0):.0f}%)": j.get("id")
                for j in ranked_jobs
            }
            selected_labels = st.multiselect("Recommended jobs for approval", list(options.keys()), default=list(options.keys()))
            selected_ids = [options[x] for x in selected_labels]
            selected_urls = [
                j.get("url")
                for j in ranked_jobs
                if j.get("id") in selected_ids and j.get("url")
            ]
            st.caption(f"Selected {len(selected_ids)} jobs for downstream drafting/apply layers.")
            with st.expander("Why these jobs are recommended"):
                for j in ranked_jobs[:8]:
                    st.markdown(
                        f"- **{j.get('title','')} @ {j.get('company','')}** — "
                        f"match `{j.get('score',0)*100:.1f}%`, interview `{j.get('interview_probability_percent',0):.1f}%`  \n"
                        f"  reasoning: {j.get('llm_reasoning') or 'Skill overlap + ATS alignment'}  \n"
                        f"  link: {j.get('url') or 'N/A'}"
                    )
        else:
            selected_ids = []
            selected_urls = []

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Approve Ranked Jobs", key="approve_ranking_btn"):
                if _api_action(api_base, run_id, "approve_ranking", {"selected_job_ids": selected_ids, "selected_job_urls": selected_urls}):
                    st.success("Ranking approved. Continuing to drafting layer...")
                    st.rerun()
        with c2:
            if st.button("↩️ Reject & Re-plan from L2", key="reject_ranking_btn"):
                if _api_action(api_base, run_id, "reject_ranking"):
                    st.success("Ranking rejected. Pipeline looped back to intake/planning.")
                    st.rerun()


    elif pending == "update_profile_skills":
        gap = ((status.get("layer_debug") or {}).get("L5") or {}).get("gap_analysis") or {}
        checklist = gap.get("missing_skills_checklist") or []
        st.warning("GapAnalysisAgent found near-threshold matches. Confirm skills you already have to update your profile and re-run from L4.")
        if checklist:
            selected = st.multiselect("Missing Skills Checklist", options=checklist, default=checklist[:3], key="gap_skill_selection")
        else:
            selected = st.text_input("Enter skills (comma separated)", key="gap_skill_text")
            selected = [x.strip() for x in selected.split(",") if x.strip()]
        if st.button("I have these skills, update my profile.", key="gap_update_profile_btn", type="primary"):
            if _api_action(api_base, run_id, "update_profile_skills", {"skills": selected}):
                st.success("Profile updated. Re-running scoring from L4.")
                st.rerun()

    elif pending == "approve_drafts":
        st.warning("Draft resumes/cover letters are ready. Approve to continue auto-apply or reject to return to ranking review.")
        artifacts = _api_get_artifacts(api_base, run_id)
        if artifacts:
            for job_id, files in artifacts.items():
                st.markdown(f"**{job_id}**")
                resume = files.get("resume_docx")
                cover = files.get("cover_docx")
                if resume:
                    st.markdown(f"- [Preview Resume]({api_base.rstrip('/')}/artifact/download?path={quote_plus(resume)})")
                if cover:
                    st.markdown(f"- [Preview Cover Letter]({api_base.rstrip('/')}/artifact/download?path={quote_plus(cover)})")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Approve Drafts & Continue Apply", key="approve_drafts_btn"):
                if _api_action(api_base, run_id, "approve_drafts"):
                    st.success("Drafts approved. Continuing to apply layers...")
                    st.rerun()
        with c2:
            if st.button("↩️ Reject Drafts", key="reject_drafts_btn"):
                if _api_action(api_base, run_id, "reject_drafts"):
                    st.success("Drafts rejected. Returned to ranking approval.")
                    st.rerun()

    elif pending == "approve_followups":
        st.warning("Follow-up emails are drafted. Approve to send and complete tracking/analytics.")
        drafts = (((status.get("layer_debug") or {}).get("L7") or {}).get("email_drafts") or [])
        if drafts:
            for draft in drafts[:10]:
                with st.expander(f"📧 {draft.get('subject','Follow-up draft')} — {draft.get('job_id','')}", expanded=False):
                    st.caption(f"Status: {draft.get('status','drafted')}")
                    st.code(draft.get("body", ""), language="markdown")
        else:
            st.caption("No follow-up drafts found in layer output.")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Approve & Send Follow-ups", key="approve_followups_btn"):
                if _api_action(api_base, run_id, "approve_followups"):
                    st.success("Follow-up emails approved and sent. Continuing run...")
                    st.rerun()
        with c2:
            if st.button("↩️ Reject Follow-ups", key="reject_followups_btn"):
                if _api_action(api_base, run_id, "reject_followups"):
                    st.success("Follow-up drafts rejected. Waiting for your revised approval.")
                    st.rerun()


def render_stepwise_details(status: Optional[dict]) -> None:
    """Detailed layer-by-layer parsed/debug payloads."""
    if not status:
        return
    st.markdown('<div class="section-header">Stepwise Outputs & Evaluator Checks</div>', unsafe_allow_html=True)

    profile = status.get("profile", {})
    layer_debug = status.get("layer_debug", {})
    evaluations = status.get("evaluations", [])

    c1, c2 = st.columns(2)
    with c1:
        with st.expander("🧾 Parsed Resume (L2)", expanded=False):
            st.json(profile if profile else {"info": "Waiting for parse output"})
        with st.expander("🔎 Discovery + Match Details (L3/L4)", expanded=False):
            st.json({
                "L3": layer_debug.get("L3", {}),
                "L4": layer_debug.get("L4", {}),
            })
    with c2:
        with st.expander("🏆 Evaluator Decisions", expanded=False):
            st.json(evaluations if evaluations else [{"info": "No evaluator entries yet"}])
        with st.expander("✍️ Draft + Apply Details (L6/L7)", expanded=False):
            st.json({
                "L6": layer_debug.get("L6", {}),
                "L7": layer_debug.get("L7", {}),
            })

    missing = []
    top_jobs = (layer_debug.get("L4", {}) or {}).get("top_jobs", [])
    for job in top_jobs:
        missing.extend(job.get("missing_skills") or [])
    missing = list(dict.fromkeys([m for m in missing if m]))[:20]

    full_report = {
        "run_id": status.get("run_id"),
        "uploaded_resume": {
            "candidate_name": status.get("candidate_name"),
            "skills_extracted": status.get("skills_extracted"),
            "resume_path": (status.get("profile") or {}).get("source_resume_path", "stored server-side"),
        },
        "parsed_profile": profile,
        "missing_skills_detected": missing,
        "job_scraping": {
            "jobs_discovered": status.get("jobs_discovered", 0),
            "source_urls": [j.get("url") for j in (status.get("raw_job_leads_preview") or []) if j.get("url")],
        },
        "ranking_and_predictions": [
            {
                "title": j.get("title"),
                "company": j.get("company"),
                "url": j.get("url"),
                "match_percent": round(float(j.get("score") or 0.0) * 100, 2),
                "interview_probability_percent": j.get("interview_probability_percent", 0),
                "reasoning": j.get("llm_reasoning"),
            }
            for j in ((layer_debug.get("L5", {}) or {}).get("qualified_jobs") or [])[:20]
        ],
        "layer_debug": layer_debug,
    }
    with st.expander("🧱 Layer Debug Logs (per layer)", expanded=False):
        for lid in range(10):
            with st.expander(f"L{lid} debug", expanded=False):
                st.json((layer_debug.get(f"L{lid}", {}) if isinstance(layer_debug, dict) else {}))

    with st.expander("📜 One-click full pipeline report (scrollable)", expanded=False):
        st.caption("Includes uploaded resume metadata, parsed content, missing skills, job scraping links, ranking reasons, and all layer outputs.")
        st.text_area("Pipeline report", value=json.dumps(full_report, indent=2, default=str), height=420)
        st.download_button(
            "⬇️ Download full_pipeline_report.json",
            data=json.dumps(full_report, indent=2, default=str),
            file_name="full_pipeline_report.json",
            mime="application/json",
            use_container_width=True,
        )


def render_json_downloads(status: Optional[dict]) -> None:
    if not status:
        return

    st.markdown('<div class="section-header">JSON Exports (Layer by Layer)</div>', unsafe_allow_html=True)
    payloads = {
        "L2_parsed_profile.json": status.get("profile", {}),
        "L3_discovery.json": (status.get("layer_debug") or {}).get("L3", {}),
        "L4_matching_scoring.json": (status.get("layer_debug") or {}).get("L4", {}),
        "L5_evaluator_ranking.json": {
            "L5": (status.get("layer_debug") or {}).get("L5", {}),
            "evaluations": status.get("evaluations", []),
        },
        "L6_drafts.json": (status.get("layer_debug") or {}).get("L6", {}),
        "L7_apply_results.json": (status.get("layer_debug") or {}).get("L7", {}),
        "run_status.json": status,
    }

    full_run_status_json = json.dumps(status, indent=2, default=str)
    st.download_button(
        label="📥 Download Full Operational Trace (JSON)",
        data=full_run_status_json,
        file_name="careeros_trace.json",
        mime="application/json",
        use_container_width=True,
        type="primary",
    )

    cols = st.columns(3)
    for i, (filename, payload) in enumerate(payloads.items()):
        with cols[i % 3]:
            st.download_button(
                label=f"⬇️ {filename}",
                data=json.dumps(payload or {"info": "No data yet"}, indent=2, default=str),
                file_name=filename,
                mime="application/json",
                use_container_width=True,
            )


def render_agent_feed(status: Optional[dict]) -> None:
    """Live Agent Feed section."""
    feed = status.get("agent_log", []) if status else []

    if not feed:
        feed_content = '<div class="feed-empty">Waiting for agent activity…</div>'
    else:
        entries = ""
        for entry in reversed(feed[-20:]):  # newest first
            ts  = entry.get("ts", "")[:19].replace("T", " ")
            msg = entry.get("msg", "")
            entries += f'<div class="feed-entry"><span class="feed-ts">{escape(str(ts))}</span><span class="feed-msg">{escape(str(msg))}</span></div>'
        feed_content = entries

    st.markdown(f"""
    <div class="feed-wrap">
        <div class="feed-title">+ Live Agent Feed</div>
        {feed_content}
    </div>
    """, unsafe_allow_html=True)


def render_job_board(api_base: str, run_id: Optional[str], status: Optional[dict]) -> None:
    """Job Board tab."""
    jobs = []
    if run_id and status and status.get("jobs_discovered", 0) > 0:
        jobs = _api_get_jobs(api_base, run_id)

    if not jobs:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <div class="empty-title">No jobs discovered yet</div>
            <div class="empty-sub">Upload a resume and click Start Hunt to begin</div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown(f'<div class="section-header">{len(jobs)} Jobs Found</div>', unsafe_allow_html=True)
    min_score = st.slider("Job board score filter", 0.0, 1.0, 0.45, 0.05)
    min_interview = st.slider("Interview call prediction filter (%)", 0, 100, 35, 5)
    only_remote = st.checkbox("Show remote only in board", value=False)

    filtered = [
        j for j in jobs
        if j.get("score", 0) >= min_score
        and float(j.get("interview_probability_percent") or 0.0) >= float(min_interview)
        and (not only_remote or j.get("remote"))
    ]
    st.caption(f"Showing {len(filtered)} / {len(jobs)} jobs")

    for job in filtered[:40]:
        score = job.get("score", 0)
        score_c = "green" if score >= 0.7 else ("orange" if score >= 0.45 else "")
        remote_b = "🌐 Remote" if job.get("remote") else f"📍 {job.get('location','')}"
        why = ", ".join(job.get("matched_skills", [])[:4]) or "Keyword overlap + semantic fit"
        st.markdown(f"""
        <div class="job-row">
            <div>
                <div class="job-title">{job.get('title','')}</div>
                <div class="job-company">{job.get('company','')}  ·  {remote_b}</div>
                <div style="font-size:11px;color:#5C677D;margin-top:2px">
                    LLM reasoning: {job.get('llm_reasoning') or why}
                </div>
                <div style="font-size:11px;color:#58a6ff;margin-top:2px">🔗 {job.get('url','')}</div>
            </div>
            <div style="text-align:right">
                <div class="job-score" style="color:{'#3fb950' if score_c=='green' else '#f0883e' if score_c=='orange' else '#8b949e'}">{score*100:.0f}%</div>
                <div class="job-badge">{job.get('source','').upper()}</div>
                <div style="font-size:11px;color:#58a6ff;margin-top:4px">Interview {job.get('interview_probability_percent',0):.0f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)




def render_match_analysis(status: Optional[dict]) -> None:
    if not status:
        st.info("Run pipeline to see match analysis.")
        return
    layer_debug = status.get("layer_debug") or {}
    l5 = layer_debug.get("L5") or {}
    gap = l5.get("gap_analysis") or {}
    qualified = l5.get("qualified_jobs") or []
    top_jobs = ((layer_debug.get("L4") or {}).get("top_jobs") or [])
    source_jobs = qualified if qualified else top_jobs
    matched = []
    missing = list(gap.get("missing_skills_checklist") or [])
    for j in source_jobs[:8]:
        matched.extend(j.get("matched_skills") or [])
        missing.extend(j.get("missing_skills") or [])
    matched = list(dict.fromkeys([m for m in matched if m]))[:12]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ✅ Matched Skills")
        if matched:
            st.bar_chart({"matched": [len(matched)]})
            st.caption(" | ".join(matched))
        else:
            st.caption("No matched skills yet")
    with c2:
        st.markdown("#### ⚠️ Missing Skills")
        if missing:
            st.bar_chart({"missing": [len(missing)]})
            st.caption(" | ".join(missing))
        else:
            st.caption("No missing skills identified")

def render_analytics(api_base: str, run_id: Optional[str], status: Optional[dict]) -> None:
    """Analytics tab."""
    if not status or status.get("progress_pct", 0) < 90:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📊</div>
            <div class="empty-title">Analytics available after L9 completes</div>
            <div class="empty-sub">Run the full pipeline to see career insights</div>
        </div>
        """, unsafe_allow_html=True)
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Jobs Discovered", status.get("jobs_discovered", 0))
    with c2:
        st.metric("Applied To", status.get("jobs_applied", 0))
    with c3:
        st.metric("Top Match", f"{status.get('top_match_score',0):.0f}%")
    with c4:
        st.metric("Interview Calls (Predicted)", len(status.get("interviews", []) or []))

    st.markdown("#### 🤖 LLM + Agent Tooling in this run")
    llm_stack = status.get("llm_stack") or {}
    if llm_stack:
        stack_rows = []
        for purpose, detail in llm_stack.items():
            stack_rows.append({
                "Purpose": purpose,
                "Provider": detail.get("provider", "-"),
                "Model": detail.get("model", "-"),
                "Reason": detail.get("why", ""),
            })
        st.dataframe(stack_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No LLM stack metadata captured yet.")

    lcol1, lcol2 = st.columns(2)
    with lcol1:
        langsmith = status.get("langsmith", {}) or {}
        st.markdown("**LangSmith tracing**")
        if langsmith.get("enabled") and langsmith.get("dashboard_url"):
            st.success("Active")
            st.markdown(f"[Open LangSmith run trace]({langsmith.get('dashboard_url')})")
        else:
            st.caption("LangSmith disabled. Set LANGCHAIN_TRACING_V2 and LANGSMITH_API_KEY.")
    with lcol2:
        langgraph = status.get("langgraph", {}) or {}
        st.markdown("**LangGraph tracing**")
        if langgraph.get("enabled") and langgraph.get("dashboard_url"):
            st.markdown(f"[Open LangGraph run trace]({langgraph.get('dashboard_url')})")
        else:
            st.caption(langgraph.get("note") or "LangGraph trace URL is not configured.")

    applications = status.get("apply_results") or []
    st.markdown("#### 📌 Application tracking")
    if applications:
        st.dataframe([
            {
                "Job ID": row.get("job_id"),
                "Company": row.get("company"),
                "Title": row.get("title"),
                "Status": row.get("status"),
                "Applied At": row.get("applied_at"),
                "Channel": row.get("apply_channel"),
                "Next Action": row.get("next_action"),
                "Apply URL": row.get("url"),
            }
            for row in applications
        ], use_container_width=True, hide_index=True)
    else:
        st.caption("No application data yet.")

    st.markdown("#### 📈 Analytics Dashboard")
    applied = len(applications)
    interview_1 = sum(1 for row in applications if "interview" in str(row.get("status") or "").lower())
    final_round = sum(1 for row in applications if "final" in str(row.get("status") or "").lower())
    offer = sum(1 for row in applications if any(k in str(row.get("status") or "").lower() for k in ("offer", "selected")))
    ac1, ac2, ac3, ac4 = st.columns(4)
    ac1.metric("Applied", applied)
    ac2.metric("Interview 1", interview_1, f"{(interview_1 / max(1, applied)) * 100:.1f}%")
    ac3.metric("Final Round", final_round, f"{(final_round / max(1, applied)) * 100:.1f}%")
    ac4.metric("Offer", offer, f"{(offer / max(1, applied)) * 100:.1f}%")

    feedback_loop = (((status.get("layer_debug") or {}).get("L9") or {}).get("analytics_summary") or {}).get("feedback_loop") or {}
    st.markdown("#### 🧠 Self-Learning insights")
    st.json(feedback_loop or {"info": "No feedback insights yet."})

    st.markdown("#### 💬 Feedback ingestion")
    if run_id:
        with st.form("feedback_form", clear_on_submit=True):
            fb_source = st.selectbox("Feedback source", options=["user", "employer"], index=0)
            fb_text = st.text_area("Feedback text", placeholder="Share what worked / failed, interview updates, rejection reason, bugs, etc.")
            fb_submitted = st.form_submit_button("Submit feedback")
        if fb_submitted:
            if fb_text.strip():
                try:
                    _api_post(api_base, f"/hunt/{run_id}/feedback", json={"source": fb_source, "text": fb_text.strip()}, timeout=25)
                    st.success("Feedback saved and added to self-learning signals.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to submit feedback: {e}")
            else:
                st.warning("Feedback text is required.")
    else:
        st.caption("Start a run to submit feedback.")

    feedback_events = status.get("feedback_events") or []
    if feedback_events:
        st.dataframe(feedback_events[-20:], use_container_width=True, hide_index=True)
    else:
        st.caption("No feedback captured yet. Submit feedback above to improve future runs.")

    c5, c6 = st.columns(2)
    with c5:
        st.markdown("#### 📅 Interview queue")
        interviews = status.get("interviews") or []
        if interviews:
            st.dataframe(interviews, use_container_width=True, hide_index=True)
        else:
            st.caption("No high-probability interview calls predicted yet.")
    with c6:
        st.markdown("#### ✉️ Employer follow-up drafts")
        followups = status.get("followup_queue") or []
        if followups:
            st.dataframe(followups, use_container_width=True, hide_index=True)
        else:
            st.caption("No follow-up drafts in queue.")

    st.markdown("#### 🔔 Notification delivery log")
    notification_log = status.get("notification_log") or []
    if notification_log:
        st.dataframe(notification_log, use_container_width=True, hide_index=True)
        st.caption("Notification results include provider-level delivery attempts and responses.")
    else:
        st.caption("No notifications attempted yet.")

    errors = status.get("errors", [])
    if errors:
        st.warning("**Pipeline Errors:**\n" + "\n".join(f"- {e}" for e in errors))


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> tuple[str, Optional[bytes], Optional[str], Optional[str], dict]:
    """
    Returns (api_base, resume_bytes, resume_filename, run_id, config)
    """
    with st.sidebar:
        st.markdown("""
        <div style="padding:12px 0 20px">
            <div style="font-size:18px;font-weight:700;color:#1B263B">🎯 CareerAgent-AI</div>
            <div style="font-size:12px;color:#5C677D;margin-top:2px">Autonomous Job Hunt Engine</div>
        </div>
        """, unsafe_allow_html=True)

        # ── API Base URL ──────────────────────────────────────────────────────
        api_base = st.text_input("Backend URL", value=st.session_state["api_base"], key="api_base_input")
        st.session_state["api_base"] = api_base

        # ── Health indicator ──────────────────────────────────────────────────
        is_healthy = _api_health(api_base)
        color  = "#3fb950" if is_healthy else "#f85149"
        label  = "Backend Online" if is_healthy else "Backend Offline"
        dot    = "●"
        st.markdown(f'<div style="font-size:13px;color:{color}">{dot} {label}</div>',
                    unsafe_allow_html=True)

        st.divider()

        # ── View Mode  ← FIX: non-empty label + label_visibility="collapsed" ──
        st.caption("VIEW MODE")
        st.session_state["view_mode"] = st.selectbox(
            "View Mode",                             # ← was "" (empty) — FIXED
            ["Pilot View", "Engineer View"],
            index=["Pilot View", "Engineer View"].index(st.session_state["view_mode"]),
            label_visibility="collapsed",            # hides label visually
        )

        # ── Live Update ───────────────────────────────────────────────────────
        st.session_state["live_update"] = st.checkbox(
            "🔴  Live Update", value=st.session_state["live_update"]
        )
        if st.session_state["live_update"]:
            st.session_state["refresh_sec"] = st.slider(
                "Refresh interval (sec)",
                min_value=2, max_value=30,
                value=st.session_state["refresh_sec"],
            )

        st.divider()

        # ── Target Roles ──────────────────────────────────────────────────────
        st.caption("TARGET ROLES")
        roles_input = st.text_area(
            "Target Roles",
            value="Software Engineer\nBackend Developer\nPlatform Engineer",
            height=80,
            label_visibility="collapsed",
            help="One role per line",
        )
        target_roles = [r.strip() for r in roles_input.split("\n") if r.strip()]

        # ── Options ───────────────────────────────────────────────────────────
        remote_only = st.checkbox("Remote Only", value=True)
        threshold   = st.slider("Match Threshold", 0.30, 0.90, 0.45, 0.05,
                                help="Minimum score for a job to qualify")
        posted_hours = st.selectbox(
            "Posted within",
            [1, 3, 6, 12, 24, 48, 72, 168],
            index=7,
            format_func=lambda x: f"Last {x} hour{'s' if x != 1 else ''}",
        )
        max_jobs = st.slider("How many jobs to scrape today", 20, 150, 80, 5)
        salary_min, salary_max = st.slider("Salary range (USD)", 0, 400000, (80000, 220000), step=10000)

        require_ranking_approval = st.checkbox("Require ranking approval (HITL)", value=True)
        require_draft_approval = st.checkbox("Require draft approval before apply", value=True)
        require_followup_approval = st.checkbox("Require follow-up email approval", value=True)

        st.caption("Notifications")
        notif_email = st.text_input("Gmail for notifications", value="")
        notif_phone = st.text_input("Phone number for SMS", value="", placeholder="+1 415 555 0100")
        profile_links = st.text_input("Profile links (LinkedIn/GitHub)", value="", help="Comma-separated URLs used by auto-apply forms")
        enable_email = st.checkbox("Enable email notifications", value=False)
        enable_sms = st.checkbox("Enable SMS notifications", value=False)

        config = {
            "target_roles":             target_roles,
            "match_threshold":          threshold,
            "geo_preferences":          {"remote": remote_only, "locations": []},
            "require_ranking_approval": require_ranking_approval,
            "require_draft_approval":   require_draft_approval,
            "require_followup_approval": require_followup_approval,
            "posted_within_hours":      posted_hours,
            "max_jobs":                 max_jobs,
            "salary_min":               salary_min,
            "salary_max":               salary_max,
            "work_modes":               ["remote"] if remote_only else ["remote", "hybrid", "onsite"],
            "notifications": {
                "email": notif_email,
                "phone": " ".join(notif_phone.split()),
                "links": [u.strip() for u in profile_links.split(",") if u.strip()],
                "enable_email": enable_email,
                "enable_sms": enable_sms,
            },
        }

        st.divider()

        # ── Resume Upload ─────────────────────────────────────────────────────
        st.caption("RESUME")
        resume_file = st.file_uploader(
            "Resume Upload",
            type=["pdf", "txt", "docx", "md"],
            label_visibility="collapsed",
            help="Upload your resume (PDF, TXT, or DOCX)",
        )

        resume_bytes    = resume_file.read() if resume_file else None
        resume_filename = resume_file.name   if resume_file else None
        if resume_file:
            st.caption(f"Uploaded: {resume_filename} ({round(len(resume_bytes or b'')/1024,1)}KB)")
            if resume_filename.lower().endswith((".txt", ".md")) and resume_bytes:
                with st.expander("Preview uploaded resume"):
                    st.code((resume_bytes.decode("utf-8", errors="ignore"))[:4000])

        # ── Start Hunt button ─────────────────────────────────────────────────
        start_clicked = st.button("🚀  Start Hunt", disabled=(resume_bytes is None or not is_healthy))

        if not is_healthy:
            st.caption("⚠ Start backend first:\n`uv run uvicorn careeragent.api.main:app --app-dir src --host 127.0.0.1 --port 8000 --reload`")
        elif resume_bytes is None:
            st.caption("Upload your resume to begin.")

        # ── Handle Start Hunt ─────────────────────────────────────────────────
        if start_clicked and resume_bytes and is_healthy:
            with st.spinner("Launching pipeline…"):
                run_id = _api_start_hunt(api_base, resume_bytes, resume_filename or "resume.pdf", config)
            if run_id:
                st.session_state["run_id"]       = run_id
                st.session_state["run_status"]   = None
                st.session_state["hunt_running"] = True
                st.session_state["last_poll"]    = 0.0
                st.success(f"✓ Run started: `{run_id}`")
            else:
                st.error("Failed to start run — check backend logs.")

        # ── Show current run ID ───────────────────────────────────────────────
        if st.session_state.get("run_id"):
            st.caption(f"Run ID: `{st.session_state['run_id']}`")

    return api_base, resume_bytes, resume_filename, st.session_state.get("run_id"), config


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    _init_session()
    _inject_css()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    api_base, _resume_bytes, _filename, run_id, _config = render_sidebar()

    # ── Poll backend for status ───────────────────────────────────────────────
    status = st.session_state.get("run_status")
    now    = time.time()

    if run_id and (now - st.session_state["last_poll"] > 1.5):   # max 1 poll per 1.5s
        fresh = _api_get_status(api_base, run_id)
        if fresh:
            st.session_state["run_status"] = fresh
            status = fresh
            st.session_state["last_poll"] = now
            # Stop auto-refresh when done
            if fresh.get("status") in ("completed", "error"):
                st.session_state["hunt_running"] = False

    # ── Extract layer data ────────────────────────────────────────────────────
    layers_data = []
    if status and "layers" in status:
        layers_data = status["layers"]
    else:
        layers_data = [{"status": "waiting", "meta": {}, "output": None, "error": None,
                        "started_at": None, "finished_at": None} for _ in LAYERS]

    # ── Header ────────────────────────────────────────────────────────────────
    run_label  = f"Run: `{run_id}`  |  L0→L9 Planner-Director Pipeline" if run_id else "No active run"
    run_state  = (status or {}).get("status", "idle")
    state_cls  = f"run-status {run_state}" if run_state in ("running","completed","error","pending_human_input") else "run-status"

    hcol1, hcol2 = st.columns([8, 2])
    with hcol1:
        st.markdown(f"""
        <h2 style="margin:0 0 4px;font-size:22px;font-weight:700;color:#1B263B">
            🎯 CareerAgent-AI — Mission Control
        </h2>
        <div style="font-size:12px;color:#5C677D">{run_label}</div>
        """, unsafe_allow_html=True)
    with hcol2:
        st.markdown(f"""
        <div style="text-align:right;padding-top:10px">
            <span class="{state_cls}">{'— Idle' if run_state == 'idle' else ('Pending Human Input' if run_state in ('pending_human_input','needs_human_approval') else run_state.title())}</span>
        </div>
        """, unsafe_allow_html=True)
        langsmith = (status or {}).get("langsmith", {}) if status else {}
        fallback_url = f"https://smith.langchain.com/projects/p/{langsmith.get('project') or 'careeragent-ai'}"
        if langsmith.get("enabled") and (langsmith.get("dashboard_url") or langsmith.get("project")):
            link = langsmith.get("dashboard_url") or fallback_url
            st.markdown(f"[🧭 LangSmith dashboard]({link})")

    st.markdown("<hr style='border:none;border-top:1px solid #1e1e2e;margin:12px 0'>", unsafe_allow_html=True)

    # ── Stat cards ────────────────────────────────────────────────────────────
    render_stat_cards(status)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Progress bar ─────────────────────────────────────────────────────────
    render_progress_bar(status, layers_data)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_pipeline, tab_jobs, tab_match, tab_learn, tab_analytics = st.tabs([
        "📋  Pipeline Layers",
        "💼  Job Board",
        "🧩  Match Analysis",
        "🎓  Learning Center",
        "📊  Analytics",
    ])

    with tab_pipeline:
        st.markdown('<div class="section-header">Layer Details — click to expand</div>',
                    unsafe_allow_html=True)

        running_layer = next(
            (i for i, ls in enumerate(layers_data) if ls.get("status") == "running"), None
        )
        for ld in LAYERS:
            layer_state = layers_data[ld["id"]] if layers_data else {"status": "waiting"}
            # Auto-expand the currently-running layer
            is_expanded = (ld["id"] == running_layer)
            render_layer_card(ld, layer_state, expanded=is_expanded)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        render_agent_feed(status)
        render_hitl_controls(api_base, run_id, status)
        render_stepwise_details(status)
        render_json_downloads(status)
        with st.expander("🧠 Full run JSON / tools / API traces", expanded=False):
            st.json(status or {"info": "No run status yet"})

    with tab_jobs:
        render_job_board(api_base, run_id, status)

    with tab_match:
        render_match_analysis(status)

    with tab_learn:
        if not status or status.get("progress_pct", 0) < 50:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🎓</div>
                <div class="empty-title">Learning Center</div>
                <div class="empty-sub">Personalized career coaching appears after pipeline completes</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            skills = status.get("profile", {}).get("skills", []) if isinstance(status.get("profile"), dict) else []
            st.markdown(f"""
            <div style="color:#c9d1d9">
                <h4 style="color:#1B263B">Skills Profile</h4>
                <p>{', '.join(skills[:15]) if skills else 'Run pipeline to extract skills'}</p>
            </div>
            """, unsafe_allow_html=True)

    with tab_analytics:
        render_analytics(api_base, run_id, status)

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if st.session_state.get("live_update") and run_id:
        run_state_now = (status or {}).get("status", "")
        if run_state_now not in ("completed", "error", "pending_human_input", "needs_human_approval"):
            refresh_sec = max(1, int(st.session_state.get("refresh_sec", 2)))
            tick = st.empty()
            tick.caption(f"Auto-refreshing every {refresh_sec}s…")
            time.sleep(refresh_sec)
            st.rerun()


if __name__ == "__main__":
    main()
