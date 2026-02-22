
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


def _api_post(api: str, path: str, timeout: int = 60, **kwargs) -> requests.Response:
    return requests.post(f"{api}{path}", timeout=timeout, **kwargs)


def _exists(path: Optional[str]) -> bool:
    return bool(path) and Path(path).exists()


def main() -> None:
    st.set_page_config(page_title="CareerAgent-AI Mission Control", layout="wide")
    st.title("CareerAgent-AI — Mission Control (One-Click Automation)")
    st.caption("L0→L9: Intake → Hunt → Scrape/Match → Rank → Draft → Learning Plan → HITL approvals")

    api = st.sidebar.text_input("API Base URL", value=DEFAULT_API)

    # Backend status
    st.sidebar.divider()
    st.sidebar.subheader("Backend")
    try:
        h = _api_get(api, "/health", timeout=3)
        st.sidebar.success("🟢 API Online" if h.status_code == 200 else f"🟠 API issue ({h.status_code})")
    except Exception as e:
        st.sidebar.error("🔴 API Offline")
        st.sidebar.caption(str(e))
        st.stop()

    # Upload
    st.sidebar.divider()
    st.sidebar.subheader("Resume Upload")
    resume_file = st.sidebar.file_uploader("Upload Resume (PDF/TXT/DOCX)", type=["pdf","txt","docx"])

    # Preferences
    st.sidebar.subheader("Targets")
    roles_text = st.sidebar.text_area(
        "Target Roles (1 per line, up to 4)",
        value="Data Scientist\nML Engineer\nGenAI Engineer",
        height=90
    )
    target_roles = [r.strip() for r in roles_text.splitlines() if r.strip()][:4]

    country = st.sidebar.text_input("Country", value="US")
    location = st.sidebar.text_input("Location", value="United States")
    remote = st.sidebar.checkbox("Remote preferred", value=True)
    wfo_ok = st.sidebar.checkbox("On-site/WFO acceptable", value=True)
    salary = st.sidebar.text_input("Salary target (optional)", value="")
    visa_required = st.sidebar.checkbox("Visa sponsorship required (F1/OPT)", value=False)

    st.sidebar.subheader("Job Hunt Controls")
    recency_hours = st.sidebar.slider("Only jobs posted within (hours)", 12, 168, 36, 6)
    max_jobs = st.sidebar.slider("Jobs to score per run", 20, 60, 40, 5)

    st.sidebar.subheader("Quality Gates")
    discovery_threshold = st.sidebar.slider("Min top-score to accept ranking", 0.50, 0.90, 0.70, 0.05)
    max_refinements = st.sidebar.slider("Max query refinements", 1, 6, 3, 1)
    resume_threshold = st.sidebar.slider("Resume quality threshold (soft)", 0.35, 0.90, 0.55, 0.05)
    draft_count = st.sidebar.slider("Draft packages (top jobs)", 3, 20, 10, 1)

    st.sidebar.subheader("Notifications")
    user_phone = st.sidebar.text_input("Phone for SMS (optional)", value="")

    run_btn = st.sidebar.button("🚀 RUN ONE-CLICK", type="primary", use_container_width=True, disabled=(resume_file is None))

    st.sidebar.divider()
    st.sidebar.subheader("Existing Run")
    run_id_in = st.sidebar.text_input("Run ID", value=st.session_state.get("run_id", ""))

    if run_btn:
        prefs = {
            "target_roles": target_roles,
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
            "resume_threshold": float(resume_threshold),
            "draft_count": int(draft_count),
            "user_phone": user_phone.strip() or None,
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
        st.info("Upload resume and click RUN, or paste a run_id to monitor.")
        return

    st.button("🔄 Refresh", use_container_width=True)

    r = _api_get(api, f"/status/{run_id}", timeout=25)
    if r.status_code != 200:
        st.warning(f"Run not found yet ({r.status_code}).")
        return
    state = _safe_json(r)

    status = state.get("status","unknown")
    meta = state.get("meta", {}) or {}
    pending = meta.get("pending_action")
    feed = meta.get("live_feed", []) or []
    artifacts = state.get("artifacts", {}) or {}
    evals = state.get("evaluations", []) or []

    c1,c2,c3 = st.columns([1,1,1])
    with c1: st.metric("Run ID", run_id)
    with c2: st.metric("Status", status)
    with c3: st.metric("Pending", str(pending))

    with st.expander("Architecture Layers (L0–L9)"):
        st.markdown("""
- **L0** Security & Guardrails  
- **L1** Mission Control (UI)  
- **L2** Intake Bundle (Profile extraction)  
- **L3** Discovery (8 job boards)  
- **L4** Scrape + Match + Score (20–40 jobs)  
- **L5** Ranking + Evaluator + HITL gates  
- **L6** ATS Resume + Cover Letter generation  
- **L7** Apply (simulated for now)  
- **L8** Tracking (SQLite snapshots)  
- **L9** Learning Plan + Analytics  
        """)

    st.markdown("### Live Agent Feed")
    for ev in feed[-220:]:
        st.write(f"**[{ev.get('layer')} {ev.get('agent')}]** {ev.get('message')}")

    with st.expander("Evaluations (scores + reasons)"):
        if not evals:
            st.caption("No evaluations yet.")
        else:
            for e in evals[-10:]:
                st.write(f"**{e.get('layer_id')} / {e.get('target_id')}** score={e.get('evaluation_score'):.2f} threshold={e.get('threshold')}")
                fb = e.get("feedback") or []
                if fb:
                    st.write("- " + "\n- ".join(fb[:6]))

    # Optional resume improvement
    if pending == "resume_cleanup_optional":
        st.markdown("## Optional Resume Cleanup (improves ATS + scoring)")
        resume_ref = (artifacts.get("resume_raw") or {}).get("path")
        current = ""
        if resume_ref and Path(resume_ref).exists():
            current = Path(resume_ref).read_text(encoding="utf-8", errors="ignore")
        new_text = st.text_area("Paste improved resume text (add email/phone + headings + bullets)", value=current, height=220)
        if st.button("✅ Submit improved resume & rerun"):
            rr = _api_post(api, f"/action/{run_id}", json={"action_type":"resume_cleanup_submit","payload":{"resume_text":new_text}}, timeout=60)
            st.success("Submitted. Refresh in a few seconds.")

    # Ranking
    st.markdown("### Ranking (Top 20–40)")
    ranking_ref = artifacts.get("ranking")
    if ranking_ref and _exists(ranking_ref.get("path")):
        ranking = json.loads(Path(ranking_ref["path"]).read_text(encoding="utf-8"))
        st.dataframe(
            [{
                "rank": x.get("rank"),
                "score_%": x.get("overall_match_percent"),
                "role_hint": x.get("role_hint"),
                "title": x.get("title"),
                "board": x.get("board"),
                "recency_h": x.get("recency_hours"),
                "visa_ok": x.get("visa_ok"),
                "url": x.get("url"),
                "missing_skills": ", ".join((x.get("missing_skills") or [])[:6]),
            } for x in ranking],
            use_container_width=True
        )
    else:
        st.caption("Ranking not available yet.")

    # HITL controls
    st.markdown("### Human-in-the-Loop Controls")
    if status == "needs_human_approval" and pending == "review_ranking":
        c1,c2 = st.columns(2)
        with c1:
            if st.button("✅ Approve Ranking → Generate Draft Packages", type="primary", use_container_width=True):
                _api_post(api, f"/action/{run_id}", json={"action_type":"approve_ranking","payload":{}}, timeout=60)
                st.success("Approved. Generating ATS resume + cover letters + learning plans…")
        with c2:
            reason = st.text_input("Reason (optional) to refine ranking", value="")
            if st.button("❌ Reject Ranking → Re-run Discovery", use_container_width=True):
                _api_post(api, f"/action/{run_id}", json={"action_type":"reject_ranking","payload":{"reason":reason}}, timeout=60)
                st.warning("Rejected. Discovery rerun started.")

    if status == "needs_human_approval" and pending == "review_drafts":
        c1,c2 = st.columns(2)
        with c1:
            if st.button("✅ Approve Drafts → Complete (Simulated Apply)", type="primary", use_container_width=True):
                _api_post(api, f"/action/{run_id}", json={"action_type":"approve_drafts","payload":{}}, timeout=60)
                st.success("Approved. Completing run…")
        with c2:
            reason = st.text_input("Reason (optional) to revise drafts", value="")
            if st.button("❌ Reject Drafts → Back to Ranking", use_container_width=True):
                _api_post(api, f"/action/{run_id}", json={"action_type":"reject_drafts","payload":{"reason":reason}}, timeout=60)
                st.warning("Rejected drafts. Returning to ranking.")

    # Drafts bundle preview
    st.markdown("### Drafts + Learning Plan")
    drafts_ref = artifacts.get("drafts_bundle")
    if drafts_ref and _exists(drafts_ref.get("path")):
        bundle = json.loads(Path(drafts_ref["path"]).read_text(encoding="utf-8"))
        st.json(bundle)
    else:
        st.caption("Draft bundle not available yet.")

    with st.expander("Full State JSON"):
        st.json(state)


if __name__ == "__main__":
    main()
