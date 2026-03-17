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
import logging
import os
import sqlite3
import threading
import time
import uuid
import hashlib
from html import escape
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus
from urllib.parse import urlparse, urlunparse

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "https://careeragent-api.onrender.com")


class LiveFeedHandler(logging.Handler):
    """Capture orchestrator/agent info logs and mirror them into Mission Control feed."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno != logging.INFO:
                return
            msg = str(record.getMessage() or "").strip()
            if not msg:
                return
            if "live_feed_log" not in st.session_state:
                st.session_state["live_feed_log"] = []
            st.session_state["live_feed_log"].append(
                {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "msg": msg,
                }
            )
            st.session_state["live_feed_log"] = st.session_state["live_feed_log"][-120:]
        except Exception:
            return


def _install_live_feed_logger() -> None:
    if st.session_state.get("live_feed_handler_installed"):
        return
    handler = LiveFeedHandler()
    handler.setLevel(logging.INFO)
    for logger_name in ("careeragent.orchestrator", "orchestrator", "careeragent.agents", "leadscout"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    st.session_state["live_feed_handler_installed"] = True


def _default_api_base() -> str:
    api_base = os.getenv("API_BASE_URL")
    if api_base:
        return _resolve_api_base(api_base)

    default_backend = str(os.getenv("BACKEND_URL") or "https://careeragent-api.onrender.com").strip()
    api_url = os.getenv("API_URL") or default_backend
    if api_url:
        return _resolve_api_base(api_url)

    api_hostport = os.getenv("API_HOSTPORT")
    if api_hostport:
        return f"http://{api_hostport}"

    render_external_url = (os.getenv("RENDER_EXTERNAL_URL") or "").strip()
    render_service_name = (os.getenv("RENDER_SERVICE_NAME") or "").strip()

    def _infer_render_api_name(name: str) -> Optional[str]:
        clean = str(name or "").strip().lower()
        if not clean:
            return None
        replacements = (
            ("-dashboard", "-api"),
            ("-frontend", "-api"),
            ("-front", "-api"),
            ("-ui", "-api"),
            ("-web", "-api"),
        )
        for suffix, mapped in replacements:
            if clean.endswith(suffix):
                return clean[: -len(suffix)] + mapped
        return None

    inferred_name = _infer_render_api_name(render_service_name)
    if inferred_name:
        inferred = inferred_name
        return f"https://{inferred}.onrender.com"

    if render_external_url and "-dashboard.onrender.com" in render_external_url:
        return render_external_url.replace("-dashboard.onrender.com", "-api.onrender.com").rstrip("/")

    if render_external_url and ".onrender.com" in render_external_url:
        host = urlparse(render_external_url).netloc or render_external_url
        subdomain = host.split(".", 1)[0].strip()
        inferred = _infer_render_api_name(subdomain)
        if inferred:
            return f"https://{inferred}.onrender.com"

    return _resolve_api_base(default_backend)


def _resolve_api_base(raw_value: str) -> str:
    """Normalize backend URL and recover common Render dashboard/API mixups."""
    clean = str(raw_value or "").strip()
    fallback = str(os.getenv("BACKEND_URL") or "https://careeragent-api.onrender.com").strip() or "http://localhost:8000"
    if not clean:
        clean = fallback

    if not clean.startswith(("http://", "https://")):
        host_hint = clean.split("/", 1)[0].split(":", 1)[0].strip().lower()
        local_hosts = {"localhost", "127.0.0.1", "0.0.0.0"}
        scheme = "http" if host_hint in local_hosts else "https"
        clean = f"{scheme}://{clean.lstrip('/')}"

    parsed = urlparse(clean)
    host = (parsed.netloc or "").strip().lower()
    path = (parsed.path or "").strip()

    if host.endswith("-dashboard.onrender.com"):
        host = host.replace("-dashboard.onrender.com", "-api.onrender.com")

    known_endpoint_prefixes = ("/health", "/docs", "/openapi", "/hunt")
    if any(path.startswith(prefix) for prefix in known_endpoint_prefixes):
        path = ""

    normalized = urlunparse((parsed.scheme or "https", host, path.rstrip("/"), "", "", "")).rstrip("/")
    return normalized or "http://localhost:8000"


def _candidate_api_bases(raw_value: str) -> list[str]:
    """Build de-duplicated backend URL candidates for Render/local deployments."""
    primary = _resolve_api_base(raw_value)
    candidates: list[str] = []

    def _push(url: str) -> None:
        clean = _resolve_api_base(url)
        if clean and clean not in candidates:
            candidates.append(clean)

    _push(primary)
    raw = str(raw_value or "").strip().rstrip("/")
    if raw:
        if not raw.startswith(("http://", "https://")):
            host_hint = raw.split("/", 1)[0].split(":", 1)[0].strip().lower()
            local_hosts = {"localhost", "127.0.0.1", "0.0.0.0"}
            raw_scheme = "http" if host_hint in local_hosts else "https"
            raw = f"{raw_scheme}://{raw.lstrip('/')}"
        if raw not in candidates:
            candidates.append(raw)

    parsed = urlparse(primary)
    host = (parsed.netloc or "").strip().lower()
    scheme = parsed.scheme or "https"
    if host.endswith(".onrender.com"):
        subdomain = host.split(".", 1)[0]
        render_swaps = (
            ("-dashboard", "-api"),
            ("-frontend", "-api"),
            ("-front", "-api"),
            ("-ui", "-api"),
            ("-web", "-api"),
            ("-dashboard", "-backend"),
            ("-frontend", "-backend"),
            ("-front", "-backend"),
            ("-ui", "-backend"),
            ("-web", "-backend"),
        )
        for src, dest in render_swaps:
            if subdomain.endswith(src):
                _push(f"{scheme}://{subdomain[:-len(src)]}{dest}.onrender.com")

    return candidates


def _normalize_clickable_url(url: str) -> str:
    clean = str(url or "").strip()
    if not clean:
        return ""
    if clean.startswith("http://"):
        # Many job boards block plain-http pages with certificate/privacy warnings.
        return "https://" + clean[len("http://"):]
    if clean.startswith("https://"):
        return clean
    return f"https://{clean.lstrip('/')}"


def _safe_url_text(url: str, limit: int = 120) -> str:
    text = str(url or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…"


BETA_DB_PATH = Path("analytics/feedback_archive.db")


def _ensure_beta_feedback_db() -> Path:
    BETA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(BETA_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS public_beta_feedback (
                timestamp TEXT NOT NULL,
                user_identifier TEXT NOT NULL,
                user_role TEXT NOT NULL,
                feedback_text TEXT NOT NULL,
                rating INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS linkedin_user_sessions (
                first_seen TEXT NOT NULL,
                session_id TEXT NOT NULL UNIQUE,
                user_identifier TEXT NOT NULL,
                source TEXT NOT NULL
            )
            """
        )
    return BETA_DB_PATH


def _release_feedback_vault_locks() -> None:
    """Best-effort unlock/checkpoint so a new hunt isn't blocked by feedback writes."""
    db_paths = [
        _ensure_beta_feedback_db(),
        Path("logs/careeragent_tracking.db"),
    ]
    for db_path in db_paths:
        if not db_path.exists():
            continue
        try:
            with sqlite3.connect(db_path, timeout=1.0, isolation_level=None) as con:
                con.execute("PRAGMA busy_timeout = 500")
                con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                con.execute("PRAGMA optimize")
        except Exception:
            continue


def _reset_start_hunt_state() -> None:
    """Clear previous run/session artifacts before launching a new hunt."""
    st.session_state["run_id"] = None
    st.session_state["run_status"] = None
    st.session_state["hunt_running"] = False
    st.session_state["last_poll"] = 0.0
    st.session_state["live_feed_log"] = []

    for key in list(st.session_state.keys()):
        low = str(key).lower()
        if "discovery" in low or "job_cache" in low:
            st.session_state.pop(key, None)

    _release_feedback_vault_locks()


def _submit_feedback_background(api_base: str, run_id: str, source: str, text: str) -> None:
    """Send feedback asynchronously so UI responds instantly."""

    def _worker() -> None:
        try:
            _api_post(api_base, f"/hunt/{run_id}/feedback", json={"source": source, "text": text}, timeout=25)
        except Exception as exc:
            log.warning("Background feedback submission failed for run %s: %s", run_id, exc)

    threading.Thread(target=_worker, daemon=True).start()


@st.cache_data(show_spinner=False)
def _cached_resume_parse(resume_bytes: bytes, resume_filename: str) -> dict:
    """Cache lightweight resume parsing metadata by file content hash."""
    digest = hashlib.sha256(resume_bytes).hexdigest()[:16] if resume_bytes else ""
    preview = ""
    if resume_filename.lower().endswith((".txt", ".md")):
        preview = (resume_bytes.decode("utf-8", errors="ignore"))[:4000]
    return {
        "digest": digest,
        "size_kb": round(len(resume_bytes or b"") / 1024, 1),
        "preview": preview,
    }


def _track_public_beta_session() -> dict:
    db_path = _ensure_beta_feedback_db()
    if not st.session_state.get("public_beta_session_id"):
        st.session_state["public_beta_session_id"] = uuid.uuid4().hex
    session_id = str(st.session_state["public_beta_session_id"])
    qp = st.query_params
    source = str(qp.get("source", "direct")).lower()
    is_linkedin = source == "linkedin" or "linkedin" in str(qp.get("utm_source", "")).lower()
    user_identifier = str(qp.get("user", "public-linkedin-user" if is_linkedin else "public-user"))
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO linkedin_user_sessions(first_seen, session_id, user_identifier, source) VALUES (datetime('now'), ?, ?, ?)",
            (session_id, user_identifier, "linkedin" if is_linkedin else source),
        )
        total = conn.execute("SELECT COUNT(DISTINCT session_id) FROM linkedin_user_sessions").fetchone()[0]
        li_total = conn.execute("SELECT COUNT(DISTINCT session_id) FROM linkedin_user_sessions WHERE source='linkedin'").fetchone()[0]
    return {"session_id": session_id, "user_identifier": user_identifier, "source": ("linkedin" if is_linkedin else source), "total_sessions": total, "linkedin_sessions": li_total}


def _insert_beta_feedback(*, user_identifier: str, user_role: str, feedback_text: str, rating: int) -> None:
    db_path = _ensure_beta_feedback_db()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO public_beta_feedback(timestamp, user_identifier, user_role, feedback_text, rating) VALUES (datetime('now'), ?, ?, ?, ?)",
            (user_identifier.strip() or "anonymous", user_role.strip() or "public", feedback_text.strip(), int(rating)),
        )


def _read_beta_feedback() -> list[dict]:
    db_path = _ensure_beta_feedback_db()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT timestamp, user_identifier, user_role, feedback_text, rating FROM public_beta_feedback ORDER BY timestamp DESC"
        ).fetchall()
    out = []
    for ts, uid, role, txt, rating in rows:
        rv = int(rating)
        if rv >= 4:
            sentiment = "🟢 Positive"
        elif rv >= 3:
            sentiment = "🟡 Neutral"
        else:
            sentiment = "🔴 Issues"
        out.append({
            "Timestamp": ts,
            "User": uid,
            "Role": role,
            "Rating": rv,
            "Visual Sentiment": sentiment,
            "Feedback": txt,
        })
    return out


def _clear_test_feedback_data() -> int:
    """Delete obvious test/demo feedback rows while keeping real beta data."""
    db_path = _ensure_beta_feedback_db()
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            DELETE FROM public_beta_feedback
            WHERE lower(user_identifier) LIKE '%test%'
               OR lower(user_identifier) LIKE '%demo%'
               OR lower(user_identifier) LIKE '%sample%'
               OR lower(user_identifier) LIKE '%qa%'
               OR lower(user_identifier) LIKE '%dummy%'
               OR lower(user_identifier) = 'public-user'
               OR lower(feedback_text) LIKE '%test%'
               OR lower(feedback_text) LIKE '%dummy%'
               OR lower(feedback_text) LIKE '%lorem ipsum%'
            """
        )
        deleted = int(cur.rowcount or 0)
    return deleted



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
        background-color: #F8FAFC;
        color: #0F172A;
    }
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background-color: #F8FAFC;
        color: #0F172A;
    }

    /* Force readable text in main area (prevents white-on-white in dark browser mode) */
    [data-testid="stMain"] [data-testid="stMarkdownContainer"],
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] span,
    [data-testid="stMain"] label,
    [data-testid="stMain"] .stCaption {
        color: #000000 !important;
    }

    [data-testid="stMain"] [data-baseweb="tab-list"] button {
        color: #475569 !important;
    }
    [data-testid="stMain"] [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #DC2626 !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #101216 !important;
        border-right: 1px solid #1F2937;
    }
    section[data-testid="stSidebar"] * { color: #E5E7EB !important; }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6 {
        background: transparent !important;
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] .stTextInput > div > div > input,
    section[data-testid="stSidebar"] .stTextArea textarea,
    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"],
    section[data-testid="stSidebar"] .stFileUploader,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
    section[data-testid="stSidebar"] .stSlider,
    section[data-testid="stSidebar"] .stTextInput > div,
    section[data-testid="stSidebar"] .stTextArea > div,
    section[data-testid="stSidebar"] .stSelectbox > div {
        background: transparent !important;
        border-color: #334155 !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        border: 1px dashed #334155 !important;
        border-radius: 10px !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #1F2937 !important;
    }
    .sidebar-warmup {
        border: 1px solid #334155;
        background: rgba(17, 24, 39, 0.92);
        color: #F9FAFB;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 12px;
        font-size: 13px;
        font-weight: 600;
    }

    /* ── Stat card ── */
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #D9DEE5;
        border-radius: 10px;
        padding: 16px 20px;
        min-height: 80px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
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
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
    }
    .progress-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .progress-title  { font-size: 12px; color: #6e7681; text-transform: uppercase; letter-spacing: 0.1em; }
    .progress-pct    { font-size: 20px; font-weight: 700; color: #1B263B; }
    .progress-track  { background: #E2E8F0; border-radius: 6px; height: 8px; width: 100%; }
    .progress-fill   { height: 8px; border-radius: 6px; transition: width 0.5s ease;
                        background: linear-gradient(90deg, #1B263B 0%, #2D6A4F 100%); }

    /* ── Layer card ── */
    .layer-card {
        background: #FFFFFF;
        border: 1px solid #D9DEE5;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 8px;
        box-shadow: 0 1px 8px rgba(15, 23, 42, 0.04);
    }
    .layer-card.running { border-left: 3px solid #388bfd; }
    .layer-card.ok      { border-left: 3px solid #3fb950; }
    .layer-card.error   { border-left: 3px solid #f85149; }
    .layer-card.waiting { border-left: 3px solid #94A3B8; }
    .layer-card.skipped { border-left: 3px solid #8b949e; }

    .layer-header   { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .layer-name     { font-size: 14px; font-weight: 600; color: #1B263B; }
    .layer-status-badge {
        font-size: 11px; padding: 2px 8px; border-radius: 20px; font-weight: 500;
    }
    .badge-waiting  { background: #EEF2F7; color: #334155; }
    .badge-running  { background: #1c2d3f; color: #388bfd; }
    .badge-ok       { background: #E6F4EA; color: #2D6A4F; }
    .badge-error    { background: #2d1a1a; color: #f85149; }
    .badge-skipped  { background: #EEF2F7; color: #475569; }

    .layer-meta { display: flex; gap: 24px; margin: 8px 0; }
    .meta-item  { font-size: 12px; color: #8b949e; }
    .meta-key   { color: #6e7681; }
    .meta-val   { color: #334155; }
    .layer-desc    { font-size: 12px; color: #6e7681; margin: 4px 0; }
    .layer-output  { font-size: 12px; color: #64748B; margin-top: 8px; padding-top: 8px;
                     border-top: 1px solid #E2E8F0; }
    .output-label  { color: #6e7681; font-size: 11px; text-transform: uppercase; }
    .output-val    { color: #334155; margin-top: 2px; }

    /* ── Agent Feed ── */
    .feed-wrap {
        background: #030712;
        border: 1px solid #D9DEE5;
        border-radius: 10px;
        padding: 14px 18px;
        min-height: 140px;
        max-height: 240px;
        overflow-y: auto;
        margin-top: 12px;
    }
    .feed-title { font-size: 12px; color: #FACC15 !important; text-transform: uppercase;
                  letter-spacing: 0.12em; margin-bottom: 8px; font-weight: 800; }
    .feed-entry { font-size: 13px; color: #FDE047 !important; padding: 4px 0; line-height: 1.5; font-weight: 700;
                  font-family: "JetBrains Mono", "SFMono-Regular", Menlo, monospace; }
    .feed-ts    { color: #FBBF24 !important; font-size: 12px; margin-right: 8px; font-weight: 800; }
    .feed-msg   { color: #FFF59D !important; font-weight: 700; }
    .feed-empty { color: #FCD34D !important; font-size: 12px; font-style: italic; }
    .feed-wrap, .feed-wrap *, .feed-wrap [data-testid="stMarkdownContainer"], .feed-wrap p, .feed-wrap span {
        color: #FACC15 !important;
    }

    /* ── Code/log readability override ── */
    .stCode, .live-feed-log {
        color: #1B263B !important;
        background-color: #FFFFFF !important;
        font-family: 'Courier New', monospace;
        border: 1px solid #D9DEE5;
    }
    .live-feed-log *, .stCode * {
        color: #1B263B !important;
        -webkit-text-fill-color: #1B263B !important;
        font-family: 'Courier New', monospace !important;
    }

    /* ── Section header ── */
    .section-header {
        font-size: 11px; color: #334155; text-transform: uppercase;
        letter-spacing: 0.1em; margin: 16px 0 8px; padding-bottom: 4px;
        border-bottom: 1px solid #D9DEE5;
    }

    /* ── Status badge ── */
    .run-status {
        font-size: 12px; padding: 4px 12px; border-radius: 20px;
        background: #EEF2F7; color: #0F172A !important; font-weight: 700; border: 1px solid #CBD5E1;
    }
    .run-status.running { background: #DBEAFE; color: #1D4ED8 !important; border-color: #93C5FD; }
    .run-status.completed { background: #DCFCE7; color: #166534 !important; border-color: #86EFAC; }
    .run-status.error { background: #FEE2E2; color: #B91C1C !important; border-color: #FCA5A5; }
    .run-status.pending_human_input { background:#FEF3C7; color:#92400E !important; border-color:#FCD34D; }

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

    /* Global readability for markdown/code blocks */
    div[data-testid="stMarkdownContainer"] pre,
    div[data-testid="stMarkdownContainer"] code,
    .stCodeBlock,
    pre,
    pre *,
    code,
    code * {
        color: #1B263B !important;
        -webkit-text-fill-color: #1B263B !important;
        background-color: #FFFFFF !important;
    }
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
    last_error = ""
    for candidate in _candidate_api_bases(api_base):
        try:
            r = requests.get(f"{candidate.rstrip('/')}{path}", timeout=timeout)
            if r.status_code == 200:
                st.session_state["last_backend_error"] = ""
                return r.json()
            body = (r.text or "")[:500]
            last_error = f"HTTP {r.status_code} from {candidate}{path}: {body}"
        except Exception as exc:
            last_error = f"Request failed for {candidate}{path}: {exc}"
            continue
    if last_error:
        st.session_state["last_backend_error"] = last_error
    return None


def _api_post(api_base: str, path: str, timeout: int = 20, **kwargs) -> requests.Response:
    return requests.post(f"{api_base.rstrip('/')}{path}", timeout=timeout, **kwargs)


def _api_health(api_base: str) -> bool:
    candidates = _candidate_api_bases(api_base)
    health_paths = ("/health", "/ready", "/")

    # Connection guard budget: do not block health checks beyond 5s per probe.
    for candidate in candidates:
        for path in health_paths:
            resp = _api_get(candidate, path, timeout=5)
            if resp is not None and (
                resp.get("status") in {"ok", "healthy", "online"}
                or resp.get("ok") is True
                or str(resp.get("service") or "").strip().lower() == "careeragent-api"
            ):
                return True
            try:
                if requests.get(f"{candidate}{path}", timeout=5).status_code == 200:
                    return True
            except Exception:
                pass
    return False


def _show_connection_guard() -> None:
    st.info("🚀 Agent is waking up... Please wait 30 seconds.")


def _api_start_hunt(api_base: str, resume_bytes: bytes, filename: str, config: dict) -> Optional[str]:
    candidates = _candidate_api_bases(api_base)
    last_err = None
    for resolved_base in candidates:
        try:
            endpoint = f"{resolved_base.rstrip('/')}/hunt/start"

            # Be patient with free-tier cold starts during L1 parsing/profile extraction.
            # Retry up to 10 times when backend returns 503 or explicit
            # "backend_unavailable" sentinel.
            for attempt in range(1, 11):
                r = requests.post(
                    endpoint,
                    files={"resume": (filename, resume_bytes, "application/octet-stream")},
                    data={"config": json.dumps(config)},
                    timeout=30,
                )
                if r.status_code == 200:
                    return r.json().get("run_id")
                payload = {}
                try:
                    payload = r.json() if r.text else {}
                except Exception:
                    payload = {}
                status_value = str(payload.get("status") or "").strip().lower()
                is_initializing = status_value == "initializing"
                retry_after = int(payload.get("retry_after") or 5) if is_initializing else 5
                if is_initializing:
                    st.info("⏳ AI Engine Warming Up...")
                    last_err = "Backend initializing"
                else:
                    last_err = f"Backend error {r.status_code}: {r.text[:200]}"
                body_text = (r.text or "").lower()
                should_warm_retry = is_initializing or r.status_code == 503 or "backend_unavailable" in body_text
                if should_warm_retry and attempt < 10:
                    st.toast("Waking up AI Engine...")
                    # Opportunistic warm-up probe before retrying.
                    try:
                        requests.get(f"{resolved_base.rstrip('/')}/health", timeout=5)
                    except Exception:
                        pass
                    time.sleep(max(1, retry_after))
                    continue
                if r.status_code in {502, 504} and attempt < 10:
                    time.sleep(min(2.5 * attempt, 10.0))
                    continue
                break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            last_err = f"Cannot connect to backend candidate: {resolved_base}"
            continue
        except Exception as exc:
            st.warning(f"Start hunt request issue: {exc}")
            return None

    if last_err and ("Cannot connect" in last_err):
        _show_connection_guard()
    st.error(last_err or "Backend error: no response payload.")
    return None


def _api_get_status(api_base: str, run_id: str) -> Optional[dict]:
    last_hb = str(st.session_state.get("last_heartbeat_at") or "")
    path = f"/hunt/{run_id}/status?wait_for_heartbeat=1&max_wait_seconds=12&since_heartbeat={quote_plus(last_hb)}"
    raw = _api_get(api_base, path, timeout=15)
    if not raw:
        return None
    if raw.get("last_heartbeat_at"):
        incoming_hb = str(raw.get("last_heartbeat_at") or "")
        st.session_state["last_heartbeat_at"] = incoming_hb

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
    resp = _api_get(api_base, f"/hunt/{run_id}/jobs?limit=200", timeout=8)
    return resp.get("jobs", []) if resp else []

def _api_get_artifacts(api_base: str, run_id: str) -> dict:
    resp = _api_get(api_base, f"/hunt/{run_id}/artifacts", timeout=8)
    return resp.get("artifacts", {}) if resp else {}


def _api_action(api_base: str, run_id: str, action: str, payload: Optional[dict] = None) -> bool:
    try:
        request_token = uuid.uuid4().hex
        body = {"action": action, "action_type": action, "request_token": request_token}
        if payload:
            body.update(payload)
        last_err = None
        for candidate in _candidate_api_bases(api_base):
            endpoint = f"{candidate.rstrip('/')}/hunt/{run_id}/action"
            for attempt in range(1, 7):
                r = requests.post(endpoint, json=body, timeout=75)
                if r.status_code == 200:
                    return True
                last_err = f"Action failed ({r.status_code}): {r.text[:200]}"
                if r.status_code in {502, 503, 504} and attempt < 6:
                    time.sleep(1.5 * attempt)
                    continue
                break
        st.error(last_err or "Action failed due to unknown backend response.")
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
        "api_base":       _default_api_base(),
        "last_poll":      0.0,
        "last_heartbeat_at": "",
        "last_heartbeat_received_at": 0.0,
        "last_backend_error": "",
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
        preview = (status.get("approved_jobs_preview") or [])
        ranked_preview = (((status.get("layer_debug") or {}).get("L5") or {}).get("qualified_jobs") or [])
        if any(bool(job.get("is_demo")) for job in [*preview, *ranked_preview[:10]]):
            diagnostics = status.get("discovery_diagnostics") or {}
            fallback_reason = diagnostics.get("fallback_reason")
            if fallback_reason:
                st.info(f"Live providers fell back to demo jobs: {fallback_reason}")
                st.caption("To get live jobs, set SERPER_API_KEY and/or TAVILY_API_KEY in the API environment and redeploy/restart.")
            else:
                st.info("Live providers returned no jobs, so fallback demo results are shown. Demo links may open search pages when direct postings are unavailable.")
        st.warning("Ranking evaluator is waiting for your decision. Select recommended jobs and approve, or reject to re-plan from intake.")
        ranked_jobs = (status.get("layer_debug") or {}).get("L5", {}).get("qualified_jobs", []) or status.get("approved_jobs_preview", [])
        if ranked_jobs:
            options = {
                f"{j.get('title','Role')} · {j.get('company','')} "
                f"(match {j.get('score',0)*100:.0f}% | interview {j.get('interview_probability_percent',0):.0f}% | {j.get('id','no-id')})": j.get("id")
                for j in ranked_jobs
            }
            selected_labels = st.multiselect("Recommended jobs for approval", list(options.keys()), default=list(options.keys()))
            selected_ids = [options[x] for x in selected_labels]
            selected_urls = [
                _normalize_clickable_url(j.get("url", ""))
                for j in ranked_jobs
                if j.get("id") in selected_ids and _normalize_clickable_url(j.get("url", ""))
            ]
            st.caption(f"Selected {len(selected_ids)} jobs for downstream drafting/apply layers out of {len(ranked_jobs)} ranked jobs.")
            with st.expander("Why these jobs are recommended"):
                st.caption(f"Showing {len(ranked_jobs)} ranked jobs with explanation and direct links.")
                for j in ranked_jobs:
                    job_url = _normalize_clickable_url(j.get('url') or '')
                    display_url = _safe_url_text(job_url)
                    st.markdown(
                        f"- **{j.get('title','')} @ {j.get('company','')}** — "
                        f"match `{j.get('score',0)*100:.1f}%`, interview `{j.get('interview_probability_percent',0):.1f}%`  \n"
                        f"  reasoning: {j.get('llm_reasoning') or 'Skill overlap + ATS alignment'}  \
"
                        f"  rationale: {' '.join((j.get('recommendation_rationale') or [])[:3]) or 'Model found strong profile-to-role evidence across skills, recency, and delivery signals.'}  \
"
                        f"  link: {'[Open job posting](' + job_url + ')' if job_url else 'N/A'}  \
"
                        f"  url: `{display_url}`"
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
    container = st.empty()
    feed = status.get("agent_log", []) if status else []
    live_logger_feed = st.session_state.get("live_feed_log", [])
    merged_feed = [*feed, *live_logger_feed][-25:]

    if not merged_feed:
        container.markdown(
            '<div class="feed-wrap live-feed-log"><div class="feed-title">+ Live Agent Feed</div><div class="feed-empty">Waiting for agent activity…</div></div>',
            unsafe_allow_html=True,
        )
        return

    lines: list[str] = []
    for entry in reversed(merged_feed):
        ts = str(entry.get("ts", ""))[:19].replace("T", " ")
        msg = str(entry.get("msg", "")).strip()
        lines.append(
            f'<div class="feed-entry"><span class="feed-ts">[{escape(ts)}]</span>'
            f'<span style="color: #FFD700; font-family: monospace;">{escape(msg)}</span></div>'
        )
    container.markdown(
        f'<div class="feed-wrap live-feed-log"><div class="feed-title">+ Live Agent Feed</div>{"".join(lines)}</div>',
        unsafe_allow_html=True,
    )


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

    allowed_statuses = {"ready", "approved"}
    filtered = [
        j for j in jobs
        if j.get("score", 0) >= min_score
        and float(j.get("interview_probability_percent") or 0.0) >= float(min_interview)
        and (not only_remote or j.get("remote"))
        and (
            not str(j.get("status") or "").strip()
            or str(j.get("status") or "").strip().lower() in allowed_statuses
        )
    ]
    st.caption(f"Showing {len(filtered)} / {len(jobs)} jobs")

    non_direct = [j for j in filtered if not bool(j.get("is_direct_job_url", True))]
    if non_direct:
        st.warning(
            f"{len(non_direct)} listings are board/search URLs (not direct posting URLs). "
            "They are shown for discovery context, but should be opened and replaced with direct ATS job pages before applying."
        )

    for job in filtered[:40]:
        score = job.get("score", 0)
        score_c = "green" if score >= 0.7 else ("orange" if score >= 0.45 else "")
        remote_b = "🌐 Remote" if job.get("remote") else f"📍 {job.get('location','')}"
        why = ", ".join(job.get("matched_skills", [])[:4]) or "Keyword overlap + semantic fit"
        direct_badge = "✅ Direct job URL" if bool(job.get("is_direct_job_url", True)) else "⚠️ Search/board URL"
        st.markdown(f"""
        <div class="job-row">
            <div>
                <div class="job-title">{job.get('title','')}</div>
                <div class="job-company">{job.get('company','')}  ·  {remote_b}</div>
                <div style="font-size:11px;color:#5C677D;margin-top:2px">
                    LLM reasoning: {job.get('llm_reasoning') or why}<br/>Rationale: {' '.join((job.get('recommendation_rationale') or [])[:2]) or 'Role fit inferred from project/experience signals + skill match.'}
                </div>
                <div style="font-size:11px;color:#58a6ff;margin-top:2px">🔗 <a href="{_normalize_clickable_url(job.get('url',''))}" target="_blank" rel="noopener noreferrer">Open posting</a></div>
                <div style="font-size:10px;color:#5C677D;word-break:break-all">{escape(_safe_url_text(_normalize_clickable_url(job.get('url',''))))}</div>
                <div style="font-size:11px;color:{'#2D6A4F' if bool(job.get('is_direct_job_url', True)) else '#B45309'};margin-top:2px">{direct_badge}</div>
            </div>
            <div style="text-align:right">
                <div class="job-score" style="color:{'#3fb950' if score_c=='green' else '#f0883e' if score_c=='orange' else '#8b949e'}">{score*100:.0f}%</div>
                <div class="job-badge">{job.get('source','').upper()}</div>
                <div style="font-size:11px;color:#58a6ff;margin-top:4px">Interview {job.get('interview_probability_percent',0):.0f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)




def _dedupe_skills(skills: list[str], *, exclude: list[str] | None = None, limit: int = 20) -> list[str]:
    ex = {str(s).strip().lower() for s in (exclude or []) if str(s).strip()}
    out: list[str] = []
    seen = set()
    for skill in skills or []:
        s = str(skill).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen or key in ex:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def _learning_resources_for_skill(skill: str) -> list[tuple[str, str]]:
    k = str(skill or "").lower()
    curated = {
        "langgraph": [("LangGraph docs", "https://python.langchain.com/docs/langgraph"), ("LangGraph quickstart", "https://langchain-ai.github.io/langgraph/tutorials/introduction/")],
        "langsmith": [("LangSmith observability docs", "https://docs.smith.langchain.com/"), ("Tracing quickstart", "https://docs.smith.langchain.com/observability/quick-start")],
        "rag": [("RAG conceptual guide", "https://python.langchain.com/docs/concepts/rag/"), ("DeepLearning.AI RAG short course", "https://www.deeplearning.ai/short-courses/retrieval-augmented-generation-rag/")],
        "llm": [("OpenAI cookbook", "https://cookbook.openai.com/"), ("Prompt engineering guide", "https://www.promptingguide.ai/")],
        "azure": [("Microsoft Learn: Azure AI", "https://learn.microsoft.com/en-us/training/azure/"), ("Azure AI Foundry", "https://learn.microsoft.com/en-us/azure/ai-studio/")],
        "python": [("Python official tutorial", "https://docs.python.org/3/tutorial/"), ("Real Python paths", "https://realpython.com/learning-paths/")],
        "tableau": [("Tableau learning", "https://www.tableau.com/learn/training"), ("Tableau docs", "https://help.tableau.com/")],
        "power bi": [("Power BI learning path", "https://learn.microsoft.com/en-us/training/powerplatform/power-bi/"), ("Power BI docs", "https://learn.microsoft.com/en-us/power-bi/")],
    }
    for key, links in curated.items():
        if key in k:
            return links
    slug = skill.strip().replace(" ", "+")
    return [("Roadmap.sh", f"https://roadmap.sh/search?q={slug}"), ("YouTube practical tutorials", f"https://www.youtube.com/results?search_query={slug}+hands+on+tutorial")]


def _render_learning_recommendations(missing_skills: list[str]) -> None:
    if not missing_skills:
        st.caption("No high-confidence skill gaps detected yet.")
        return
    for skill in missing_skills[:8]:
        links = _learning_resources_for_skill(skill)
        st.markdown(f"- **{skill}**: " + " · ".join([f"[{label}]({url})" for label, url in links]))




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

    matched: list[str] = []
    missing: list[str] = list(gap.get("missing_skills_checklist") or [])
    for j in source_jobs[:8]:
        matched.extend(j.get("matched_jd_skills") or j.get("matched_skills") or [])
        missing.extend(j.get("missing_jd_skills") or j.get("missing_skills") or [])

    profile_skills = ((status.get("profile") or {}).get("skills") or []) if isinstance(status.get("profile"), dict) else []
    matched = _dedupe_skills(matched, limit=18)
    missing = _dedupe_skills(missing, exclude=matched + profile_skills, limit=20)

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

    st.markdown("#### 📚 Best learning resources for the missing skills")
    _render_learning_recommendations(missing)


def render_analytics(api_base: str, run_id: Optional[str], status: Optional[dict], *, is_admin: bool) -> None:
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

    if is_admin:
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
            if langsmith.get("enabled"):
                st.success("Active")
            else:
                st.warning("Tracing currently not active (missing key or tracing env flag).")
            if langsmith.get("dashboard_url"):
                st.markdown(f"[Open LangSmith project]({langsmith.get('dashboard_url')})")
            if langsmith.get("run_url"):
                st.link_button("View Trace", langsmith.get("run_url"), use_container_width=False)
            if langsmith.get("note"):
                st.caption(langsmith.get("note"))
        with lcol2:
            langgraph = status.get("langgraph", {}) or {}
            st.markdown("**LangGraph tracing**")
            if langgraph.get("enabled") and langgraph.get("dashboard_url"):
                st.markdown(f"[Open LangGraph run trace]({langgraph.get('dashboard_url')})")
            else:
                st.caption(langgraph.get("note") or "LangGraph trace URL is not configured.")
                st.code("Set LANGGRAPH_STUDIO_URL=https://smith.langchain.com/o/<workspace>/projects/<project>")

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
                "Screenshot Path": row.get("screenshot_path") or row.get("submission_proof"),
            }
            for row in applications
        ], use_container_width=True, hide_index=True)
        with st.expander("Open application links"):
            for row in applications:
                job_url = _normalize_clickable_url(row.get("url", ""))
                if job_url:
                    st.markdown(f"- **{row.get('title','Role')}** @ {row.get('company','')} — [Open job page]({job_url})")
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
                _submit_feedback_background(api_base, run_id, fb_source, fb_text.strip())
                st.success("Thank you — your feedback was received and is saving in the background.")
            else:
                st.warning("Feedback text is required.")
    else:
        st.caption("Start a run to submit feedback.")

    feedback_events = status.get("feedback_events") or []
    if feedback_events:
        st.caption(f"Feedback signals captured for this run: {len(feedback_events)}")
        preview_rows = [
            {
                "time": str(e.get("ts") or "")[:19].replace("T", " "),
                "source": e.get("source"),
                "accepted": (e.get("evaluation") or {}).get("is_genuine"),
                "confidence": (e.get("evaluation") or {}).get("confidence"),
                "reason": (e.get("evaluation") or {}).get("reason"),
                "feedback": str(e.get("text") or "")[:180],
            }
            for e in reversed(feedback_events[-12:])
        ]
        st.dataframe(preview_rows, use_container_width=True, hide_index=True)
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
        unresolved = []
        for row in notification_log:
            result = row.get("result") or {}
            if result.get("sent"):
                continue
            reason = result.get("reason") or result.get("error") or "provider_not_configured"
            unresolved.append(f"{row.get('event', 'notification')}: {reason}")
        if unresolved:
            st.warning("Notification issues detected:\n" + "\n".join(f"- {x}" for x in unresolved[:6]))
        st.caption("Notification results include provider-level delivery attempts and responses.")
    else:
        st.caption("No notifications attempted yet.")

    errors = status.get("errors", [])
    if errors:
        st.warning("**Pipeline Errors:**\n" + "\n".join(f"- {e}" for e in errors))


def render_executive_summary(*, is_admin: bool) -> None:
    if not is_admin:
        st.warning("🔒 Executive Summary is restricted to Admin Mode.")
        return

    st.markdown("### Executive Summary — Public Beta Feedback")
    rows = _read_beta_feedback()
    st.markdown("#### Feedback Database")
    st.caption("Admin-only visibility for archive management and records.")
    st.markdown("#### Feedback Review")
    st.caption("Live review dashboard backed by analytics/feedback_archive.db for beta-demo triage.")
    clear_col, info_col = st.columns([1, 2])
    with clear_col:
        if st.button("🗂️ Archive Management: Clear Test Data", use_container_width=True):
            deleted = _clear_test_feedback_data()
            st.success(f"Archive cleaned. Removed {deleted} test/demo feedback rows.")
            rows = _read_beta_feedback()
    with info_col:
        st.caption("Use archive cleanup before final demos to keep only real beta-tester feedback.")

    if not rows:
        st.info("No public beta feedback has been submitted yet.")
        return
    st.caption("Sortable table for leadership demo reviews. Visual Sentiment uses rating-based color coding.")
    st.markdown("#### 🏦 Beta Feedback Vault")
    st.dataframe(rows, use_container_width=True, hide_index=True)
    total = len(rows)
    positive = sum(1 for r in rows if str(r.get("Visual Sentiment","")).startswith("🟢"))
    neutral = sum(1 for r in rows if str(r.get("Visual Sentiment","")).startswith("🟡"))
    issues = sum(1 for r in rows if str(r.get("Visual Sentiment","")).startswith("🔴"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Feedback", total)
    c2.metric("Positive", positive)
    c3.metric("Neutral", neutral)
    c4.metric("Issues", issues)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> tuple[str, Optional[bytes], Optional[str], Optional[str], dict, bool]:
    """
    Returns (api_base, resume_bytes, resume_filename, run_id, config, is_admin)
    """
    with st.sidebar:
        st.markdown("""
        <div style="padding:12px 0 20px;background:transparent">
            <div style="font-size:18px;font-weight:700;color:#FFFFFF">🎯 CareerAgent-AI</div>
            <div style="font-size:12px;color:#CBD5E1;margin-top:2px">Autonomous Job Hunt Engine</div>
        </div>
        """, unsafe_allow_html=True)

        admin_key = st.text_input("🔐 Hidden Admin Gate", type="password", help="Enter admin key to unlock protected controls")
        is_admin = admin_key == "ganesh2026"

        # ── API Base URL ──────────────────────────────────────────────────────
        api_base = st.text_input(
            "Backend URL",
            value=st.session_state["api_base"],
            key="api_base_input",
            disabled=not is_admin,
        )
        resolved_api_base = _resolve_api_base(api_base)
        st.session_state["api_base"] = resolved_api_base
        api_base = resolved_api_base

        # ── Health indicator ──────────────────────────────────────────────────
        is_healthy = _api_health(api_base)
        if (not is_healthy) and st.session_state.get("run_id"):
            is_healthy = _api_get_status(api_base, st.session_state.get("run_id")) is not None
        color  = "#3fb950" if is_healthy else "#f85149"
        label  = "Backend Online" if is_healthy else "Backend Offline"
        dot    = "●"
        st.markdown(f'<div style="font-size:13px;color:{color}">{dot} {label}</div>',
                    unsafe_allow_html=True)

        if "warmup_started_at" not in st.session_state:
            st.session_state["warmup_started_at"] = time.time()
        warmup_remaining = max(0.0, 10.0 - (time.time() - float(st.session_state.get("warmup_started_at") or 0.0)))
        if warmup_remaining > 0:
            st.markdown(
                f'<div class="sidebar-warmup">⚡ System Warming Up — initial backend handshake in progress ({int(warmup_remaining)}s).</div>',
                unsafe_allow_html=True,
            )
        elif not is_healthy:
            st.markdown(
                '<div class="sidebar-warmup">⚡ System Warming Up — backend cold start detected. Please wait while services wake up.</div>',
                unsafe_allow_html=True,
            )

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
        top_sources = [
            "linkedin.com", "indeed.com", "glassdoor.com", "ziprecruiter.com",
            "greenhouse.io", "lever.co", "workday.com", "myworkdayjobs.com",
        ]
        source_domains = st.multiselect(
            "Preferred job sites",
            options=top_sources,
            default=top_sources,
            help="Discovery/matching keeps only jobs from selected job boards/career sites.",
        )

        require_ranking_approval = st.checkbox("Require ranking approval (HITL)", value=True)
        require_draft_approval = st.checkbox("Require draft approval before apply", value=True)
        require_followup_approval = st.checkbox("Require follow-up email approval", value=True)

        st.caption("Notifications")
        notif_email = st.text_input("Gmail for notifications", value="")
        notif_phone = st.text_input("Phone number for SMS", value="", placeholder="+1 415 555 0100")
        profile_links = st.text_input("Profile links (LinkedIn/GitHub)", value="", help="Comma-separated URLs used by auto-apply forms")
        additional_skills_raw = st.text_area("Skills you already have (comma/newline separated)", value="", height=70)
        additional_skills = [x.strip() for x in additional_skills_raw.replace("\n", ",").split(",") if x.strip()]
        enable_email = st.checkbox("Enable email notifications", value=True)
        enable_sms = st.checkbox("Enable SMS notifications", value=True)

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
            "allowed_job_domains":      source_domains,
            "notifications": {
                "email": notif_email,
                "phone": " ".join(notif_phone.split()),
                "links": [u.strip() for u in profile_links.split(",") if u.strip()],
                "enable_email": enable_email,
                "enable_sms": enable_sms,
            },
            "additional_skills": additional_skills,
        }

        st.divider()

        st.caption("PUBLIC BETA FEEDBACK")
        tracker = st.session_state.get("beta_tracker") or _track_public_beta_session()
        st.session_state["beta_tracker"] = tracker
        st.caption(f"LinkedIn testers: {tracker.get('linkedin_sessions', 0)} | Total public sessions: {tracker.get('total_sessions', 0)}")
        with st.form("beta_feedback_sidebar", clear_on_submit=True):
            beta_role = st.selectbox("Your role", ["LinkedIn User", "Recruiter", "Hiring Manager", "Engineer", "Other"], index=0)
            beta_rating = st.slider("Rating", min_value=1, max_value=5, value=4)
            beta_text = st.text_area("What should we improve?", height=90, placeholder="Share your thoughts…")
            beta_submit = st.form_submit_button("Submit Beta Feedback")
            if beta_submit:
                if beta_text.strip():
                    _insert_beta_feedback(
                        user_identifier=str(tracker.get("user_identifier") or tracker.get("session_id") or "public-user"),
                        user_role=beta_role,
                        feedback_text=beta_text.strip(),
                        rating=int(beta_rating),
                    )
                    st.success("Thanks — feedback saved to beta archive.")
                else:
                    st.warning("Please enter feedback text before submitting.")

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
        resume_meta = _cached_resume_parse(resume_bytes, resume_filename or "resume.pdf") if resume_file and resume_bytes else None
        if resume_file and resume_meta:
            st.caption(f"Uploaded: {resume_filename} ({resume_meta.get('size_kb', 0)}KB)")
            if resume_filename.lower().endswith((".txt", ".md")) and resume_meta.get("preview"):
                with st.expander("Preview uploaded resume"):
                    st.code(resume_meta.get("preview") or "")

        # ── Start Hunt button ─────────────────────────────────────────────────
        start_clicked = st.button("🚀  Start Hunt", disabled=(resume_bytes is None))

        if not is_healthy:
            _show_connection_guard()
        elif resume_bytes is None:
            st.caption("Upload your resume to begin.")

        # ── Handle Start Hunt ─────────────────────────────────────────────────
        if start_clicked and resume_bytes:
            _reset_start_hunt_state()
            with st.spinner("Launching pipeline…"):
                run_id = _api_start_hunt(api_base, resume_bytes, resume_filename or "resume.pdf", config)
            if run_id:
                st.session_state["run_id"]       = run_id
                st.session_state["run_status"]   = None
                st.session_state["hunt_running"] = True
                st.session_state["last_poll"]    = 0.0
                st.success(f"✓ Run started: `{run_id}`")
                st.rerun()
            else:
                _show_connection_guard()

        # ── Show current run ID ───────────────────────────────────────────────
        if st.session_state.get("run_id"):
            st.caption(f"Run ID: `{st.session_state['run_id']}`")

    return api_base, resume_bytes, resume_filename, st.session_state.get("run_id"), config, is_admin


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    _init_session()
    _install_live_feed_logger()
    _inject_css()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    api_base, _resume_bytes, _filename, run_id, _config, is_admin = render_sidebar()

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

    if run_id and not status:
        backend_err = st.session_state.get("last_backend_error") or "Run started, but live status is not available yet."
        st.error(backend_err)

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
        fallback_url = f"https://smith.langchain.com/projects?name={langsmith.get('project') or 'careeragent-ai-beta'}"
        if is_admin and langsmith.get("enabled") and (langsmith.get("dashboard_url") or langsmith.get("project")):
            link = langsmith.get("dashboard_url") or fallback_url
            st.markdown(f"[🧭 LangSmith dashboard]({link})")

    st.markdown("<hr style='border:none;border-top:1px solid #1e1e2e;margin:12px 0'>", unsafe_allow_html=True)

    intro_left, intro_right = st.columns([7, 3])
    with intro_left:
        st.markdown(
            """
            ### What CareerAgent-AI does
            - Finds high-fit, relevant roles aligned to your profile and preferences.
            - Generates **custom ATS resume + cover letter** drafts per approved job.
            - Runs human approvals, then can **auto-fill/apply** for approved jobs.
            - Sends employer notifications/follow-up emails and tracks outcomes.
            - Recommends tutorials and learning paths when skills are missing.

            **Why use it:** reduce manual application effort, improve quality, and create a measurable interview funnel.
            """
        )
    with intro_right:
        st.markdown("### How to use")
        st.markdown("1. Upload resume in sidebar")
        st.markdown("2. Start Hunt → approve ranked jobs")
        st.markdown("3. Review custom resume/cover drafts → approve apply/follow-ups")
        st.markdown("4. Track analytics and use Learning Center for missing skills")
        st.caption("Beta users should use a deployed API URL so the backend stays online for everyone.")

    # ── Stat cards ────────────────────────────────────────────────────────────
    render_stat_cards(status)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Progress bar ─────────────────────────────────────────────────────────
    render_progress_bar(status, layers_data)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_labels = [
        "📋  Pipeline Layers",
        "💼  Job Board",
        "🧩  Match Analysis",
        "🎓  Learning Center",
        "📊  Analytics",
    ]
    if is_admin:
        tab_labels.append("🧾  Executive Summary")
    tabs = st.tabs(tab_labels)
    tab_pipeline, tab_jobs, tab_match, tab_learn, tab_analytics = tabs[:5]
    tab_exec = tabs[5] if is_admin else None

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
        if is_admin:
            st.markdown("#### Raw JSON Logs")
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
                <p>{', '.join(skills[:20]) if skills else 'Run pipeline to extract skills'}</p>
            </div>
            """, unsafe_allow_html=True)
            missing_for_learning = []
            l5 = ((status.get("layer_debug") or {}).get("L5") or {})
            gap = l5.get("gap_analysis") or {}
            missing_for_learning.extend(gap.get("missing_skills_checklist") or [])
            for j in (l5.get("qualified_jobs") or [])[:8]:
                missing_for_learning.extend(j.get("missing_jd_skills") or j.get("missing_skills") or [])
            missing_for_learning = _dedupe_skills(missing_for_learning, exclude=skills, limit=15)
            st.markdown("#### Personalized upskilling path")
            _render_learning_recommendations(missing_for_learning)

    with tab_analytics:
        render_analytics(api_base, run_id, status, is_admin=is_admin)

    if is_admin and tab_exec is not None:
        with tab_exec:
            render_executive_summary(is_admin=is_admin)

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
