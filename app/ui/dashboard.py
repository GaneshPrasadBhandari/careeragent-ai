
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_API = os.getenv("API_URL", "http://127.0.0.1:8000")


def _safe_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {"_raw": resp.text[:1500], "_status_code": resp.status_code}


def _api_get(api: str, path: str, timeout: int = 25) -> requests.Response:
    return requests.get(f"{api}{path}", timeout=timeout)


def _api_post(api: str, path: str, timeout: int = 30, **kwargs) -> requests.Response:
    return requests.post(f"{api}{path}", timeout=timeout, **kwargs)


def _exists(path: Optional[str]) -> bool:
    return bool(path) and Path(path).exists()


def main() -> None:
    st.set_page_config(page_title="CareerAgent-AI Mission Control", layout="wide")
    st.title("CareerAgent-AI — Mission Control (One-Click Automation)")
    st.caption("Upload resume → autonomous ingestion + discovery + ranking → HITL approvals → drafts + dossier downloads")

    api = st.sidebar.text_input("API Base URL", value=DEFAULT_API)

    # Backend status
    st.sidebar.divider()
    st.sidebar.subheader("Backend")
    try:
        h = _api_get(api, "/health", timeout=3)
        if h.status_code == 200:
            st.sidebar.success("🟢 API Online")
        else:
            st.sidebar.warning(f"🟠 API issue ({h.status_code})")
    except Exception as e:
        st.sidebar.error("🔴 API Offline")
        st.sidebar.caption(str(e))
        st.stop()

    # Inputs
    st.sidebar.divider()
    st.sidebar.subheader("Resume Upload")
    resume_file = st.sidebar.file_uploader("Upload Resume (PDF/TXT/DOCX)", type=["pdf", "txt", "docx"])

    st.sidebar.subheader("Preferences")
    target_role = st.sidebar.text_input("Target role", value="Data Scientist")
    country = st.sidebar.text_input("Country", value="US")
    location = st.sidebar.text_input("Location", value="United States")
    remote = st.sidebar.checkbox("Remote preferred", value=True)
    wfo_ok = st.sidebar.checkbox("On-site/WFO acceptable", value=True)
    salary = st.sidebar.text_input("Salary target (optional)", value="")
    visa_required = st.sidebar.checkbox("Visa sponsorship required (F1/OPT)", value=False)

    recency_hours = st.sidebar.slider("Only jobs posted within (hours)", 12, 168, 36, 6)
    max_jobs = st.sidebar.slider("Max jobs to score per run", 10, 80, 40, 5)

    st.sidebar.subheader("Autonomy Controls")
    discovery_threshold = st.sidebar.slider("Discovery confidence threshold", 0.50, 0.90, 0.70, 0.05)
    max_refinements = st.sidebar.slider("Max query refinements", 1, 6, 3, 1)

    # NEW: soft gate controls
    resume_strict_gate = st.sidebar.checkbox("Strict resume gate (stop on low parse)", value=False)
    resume_threshold = st.sidebar.slider("Resume quality threshold (soft gate)", 0.35, 0.90, 0.55, 0.05)

    run_btn = st.sidebar.button("🚀 RUN ONE-CLICK", type="primary", use_container_width=True, disabled=(resume_file is None))

    st.sidebar.divider()
    st.sidebar.subheader("Existing Run")
    run_id_in = st.sidebar.text_input("Run ID", value=st.session_state.get("run_id", ""))

    if run_btn:
        prefs = {
            "target_role": target_role.strip() or "Data Scientist",
            "country": country.strip() or "US",
            "location": location.strip() or "United States",
            "remote": bool(remote),
            "wfo_ok": bool(wfo_ok),
            "salary": salary.strip(),
            "visa_sponsorship_required": bool(visa_required),
            "recency_hours": float(recency_hours),
            "max_jobs": int(max_jobs),
            "discovery_threshold": float(discovery_threshold),
            "max_refinements": int(max_refinements),
            "resume_strict_gate": bool(resume_strict_gate),
            "resume_threshold": float(resume_threshold),
        }
        files = {"resume": (resume_file.name, resume_file.getvalue())}
        data = {"preferences_json": json.dumps(prefs)}
        r = _api_post(api, "/analyze", files=files, data=data, timeout=180)
        if r.status_code >= 400:
            st.error(f"/analyze failed: {r.status_code}\n\n{r.text[:1500]}")
            st.stop()
        out = _safe_json(r)
        st.session_state["run_id"] = out["run_id"]
        st.success(f"Run started: {out['run_id']} (status: {out.get('status')})")

    run_id = st.session_state.get("run_id") or run_id_in.strip()
    if not run_id:
        st.info("Upload resume and click RUN, or paste a run_id.")
        return

    # Poll
    _ = st.button("🔄 Refresh", use_container_width=True)
    r = _api_get(api, f"/status/{run_id}", timeout=25)
    if r.status_code != 200:
        st.warning(f"Run not found yet ({r.status_code}).")
        return
    state = _safe_json(r)

    status = state.get("status", "unknown")
    meta = state.get("meta", {}) or {}
    pending = meta.get("pending_action")
    feed = meta.get("live_feed", []) or []
    steps = state.get("steps", []) or []
    artifacts = state.get("artifacts", {}) or {}
    evals = state.get("evaluations", []) or []

    c1, c2, c3 = st.columns([1,1,1])
    with c1: st.metric("Run ID", run_id)
    with c2: st.metric("Status", status)
    with c3: st.metric("Pending", str(pending))

    # Layer map (aligned with your architecture)
    with st.expander("Architecture Layer Map (L0–L9)", expanded=False):
        st.markdown("""
- **L0** Security & Guardrails  
- **L1** Mission Control (UI)  
- **L2** Parsing & Profile Extraction  
- **L3** Discovery / Connectors  
- **L4** Scrape + Matching / Vectorization  
- **L5** Evaluator + Ranking + HITL gates  
- **L6** Drafting (Resume/Cover/Answers)  
- **L7** Execution (Apply/Notifications)  
- **L8** Tracking (Status/CRM/DB)  
- **L9** Analytics + XAI + Dossier Export
        """)

    # Feed
    st.markdown("### Live Agent Feed")
    for ev in feed[-180:]:
        st.write(f"**[{ev.get('layer')} {ev.get('agent')}]** {ev.get('message')}")

    # Evaluations
    with st.expander("Evaluations (Scores + Reasons)", expanded=False):
        if not evals:
            st.caption("No evaluations yet.")
        else:
            for e in evals[-10:]:
                st.write(f"**{e.get('layer_id')} / {e.get('target_id')}** score={e.get('evaluation_score'):.2f} threshold={e.get('threshold')}")
                fb = e.get("feedback") or []
                if fb:
                    st.write("- " + "\n- ".join(fb[:6]))

    # Resume cleanup HITL
    if status == "needs_human_approval" and pending in ("resume_cleanup", "resume_cleanup_optional"):
        st.markdown("## Human-in-the-Loop: Resume Cleanup")
        resume_ref = (artifacts.get("resume_raw") or {}).get("path")
        current_text = ""
        if resume_ref and Path(resume_ref).exists():
            current_text = Path(resume_ref).read_text(encoding="utf-8", errors="ignore")

        st.info("Your resume is parseable, but quality is below target. You can paste an improved version (add email/phone + clearer headings) and continue automatically.")
        new_text = st.text_area("Paste improved resume text (ATS headings + contact)", value=current_text, height=220)

        if st.button("✅ Submit cleaned resume & continue", type="primary"):
            rr = _api_post(api, f"/action/{run_id}", json={"action_type": "resume_cleanup_submit", "payload": {"resume_text": new_text}}, timeout=60)
            if rr.status_code >= 400:
                st.error(rr.text[:1200])
            else:
                st.success("Submitted. Refresh in a few seconds.")

    # Ranking
    st.markdown("### Ranking")
    ranking_ref = artifacts.get("ranking")
    if ranking_ref and _exists(ranking_ref.get("path")):
        ranking = json.loads(Path(ranking_ref["path"]).read_text(encoding="utf-8"))
        st.dataframe(
            [{
                "rank": x.get("rank"),
                "score_%": x.get("overall_match_percent"),
                "title": x.get("title"),
                "board": x.get("board"),
                "recency_h": x.get("recency_hours"),
                "visa_ok": x.get("visa_ok"),
                "url": x.get("url"),
            } for x in ranking],
            use_container_width=True
        )
    else:
        st.caption("Ranking not available yet.")

    with st.expander("Full State JSON", expanded=False):
        st.json(state)


if __name__ == "__main__":
    main()
