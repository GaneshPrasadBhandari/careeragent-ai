from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# --- Path bootstrap
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_API = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")


def _safe_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {"_raw": resp.text[:2500], "_status_code": resp.status_code}


def _api_get(api: str, path: str, timeout: int = 25) -> requests.Response:
    return requests.get(f"{api}{path}", timeout=timeout)


def _api_post(api: str, path: str, timeout: int = 60, **kwargs) -> requests.Response:
    return requests.post(f"{api}{path}", timeout=timeout, **kwargs)


def _exists(path: Optional[str]) -> bool:
    return bool(path) and Path(path).exists()


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def _read_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def _load_json(path: str) -> Any:
    return json.loads(_read_text(path))


def _get_pending(state: Dict[str, Any]) -> Optional[str]:
    if state.get("pending_action") is not None:
        return state.get("pending_action")
    meta = state.get("meta", {}) or {}
    return meta.get("pending_action")


def _is_pending_status(status: Any) -> bool:
    s = str(status or "").strip().lower()
    return s in {"pending", "needs_human_approval"}


def _layer_is_l5(state: Dict[str, Any]) -> bool:
    cur = str(state.get("current_layer") or "").strip().upper()
    return cur in {"5", "L5"}


def _get_feed(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(state.get("live_feed"), list):
        return state.get("live_feed") or []
    meta = state.get("meta", {}) or {}
    if isinstance(meta.get("live_feed"), list):
        return meta.get("live_feed") or []
    return []


def _get_plan_layers(state: Dict[str, Any]) -> List[str]:
    meta = state.get("meta", {}) or {}
    plan = meta.get("plan_layers")
    if isinstance(plan, list) and plan:
        return [str(x) for x in plan]
    return ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]


def _progress(state: Dict[str, Any]) -> Tuple[int, str]:
    # Prefer explicit progress_percent/current_layer from backend
    if state.get("progress_percent") is not None:
        pct = int(float(state.get("progress_percent") or 0))
        cur = state.get("current_layer")
        label = f"Current: {cur}" if cur else "Running"
        if state.get("status") == "needs_human_approval":
            label = f"Waiting / HITL ({_get_pending(state)})"
        return max(0, min(100, pct)), label

    # fallback derived
    steps = state.get("steps") or []
    plan = _get_plan_layers(state)
    total = max(1, len(plan))
    ok = 0
    current = None
    for s in steps:
        if s.get("status") == "ok":
            ok += 1
        if s.get("status") == "running":
            current = s.get("layer_id")
    pct = int(round((ok / total) * 100))
    label = f"Current: {current}" if current else "Queued"
    return pct, label


def _job_url(job: Dict[str, Any]) -> str:
    return str(job.get("url") or job.get("job_id") or "")


def _job_title(job: Dict[str, Any]) -> str:
    return str(job.get("title") or job.get("role_title") or "(untitled)")


def _job_match_percent(job: Dict[str, Any]) -> float:
    if job.get("overall_match_percent") is not None:
        return float(job.get("overall_match_percent") or 0.0)
    if job.get("match_percent") is not None:
        return float(job.get("match_percent") or 0.0)
    return 0.0


def _job_components(job: Dict[str, Any]) -> Dict[str, float]:
    comp = job.get("components") or {}
    # v2 matcher uses jd_alignment + ats_proxy
    jd_align = float(job.get("jd_alignment_percent") or 0.0)
    exp = float(comp.get("experience_alignment") or 0.0) * 100.0
    ats = float(comp.get("ats_proxy") or 0.0) * 100.0
    return {"jd_align": jd_align, "exp": exp, "ats": ats}


def _artifact_btn(col, artifacts: Dict[str, Any], key: str, label: str, filename: str, mime: str) -> None:
    ref = artifacts.get(key)
    if not ref:
        return
    p = ref.get("path") if isinstance(ref, dict) else getattr(ref, "path", None)
    if not p or not Path(p).exists():
        return
    col.download_button(label, data=_read_bytes(p), file_name=filename, mime=mime, use_container_width=True)


def _list_downloads(artifacts: Dict[str, Any]) -> None:
    st.markdown("### Downloads")
    cols = st.columns(6)
    _artifact_btn(cols[0], artifacts, "jobs_raw", "⬇️ jobs_raw.json", "jobs_raw.json", "application/json")
    _artifact_btn(cols[1], artifacts, "jobs_scored", "⬇️ jobs_scored.json", "jobs_scored.json", "application/json")
    _artifact_btn(cols[2], artifacts, "ranking", "⬇️ ranking.json", "ranking.json", "application/json")
    _artifact_btn(cols[3], artifacts, "extracted_profile", "⬇️ profile.json", "extracted_profile.json", "application/json")
    _artifact_btn(cols[4], artifacts, "evaluation", "⬇️ evaluation.json", "evaluation.json", "application/json")
    _artifact_btn(cols[5], artifacts, "career_roadmap", "⬇️ roadmap.md", "career_roadmap.md", "text/markdown")


def _draft_picker(artifacts: Dict[str, Any]) -> None:
    keys = sorted([k for k in artifacts.keys() if k.startswith("resume_") or k.startswith("cover_")])
    if not keys:
        st.caption("No draft artifacts yet.")
        return

    st.markdown("### Draft Outputs")
    pick = st.selectbox("Select file", keys)
    ref = artifacts[pick]
    p = ref.get("path") if isinstance(ref, dict) else getattr(ref, "path", None)
    if not p or not Path(p).exists():
        st.warning(f"Missing file: {p}")
        return

    if p.endswith(".pdf"):
        st.download_button("⬇️ Download PDF", data=_read_bytes(p), file_name=Path(p).name, mime="application/pdf", use_container_width=True)
        return
    if p.endswith(".docx"):
        st.download_button(
            "⬇️ Download DOCX",
            data=_read_bytes(p),
            file_name=Path(p).name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
        return

    txt = _read_text(p)
    st.code(txt[:12000], language="markdown")
    st.download_button("⬇️ Download", data=txt.encode("utf-8"), file_name=Path(p).name, mime="text/plain", use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="CareerAgent-AI Mission Control", layout="wide")
    st.title("CareerAgent-AI — Mission Control")
    st.caption("L0→L9: Planner/Director soft-fencing • Personas • HITL • ATS drafts • Analytics & Memory")

    if "run_id" not in st.session_state:
        st.session_state["run_id"] = ""
    if "selected_urls" not in st.session_state:
        st.session_state["selected_urls"] = set()

    api = st.sidebar.text_input("API Base URL", value=DEFAULT_API).rstrip("/")

    # health
    try:
        h = _api_get(api, "/health", timeout=4)
        st.sidebar.success("🟢 API Online" if h.status_code == 200 else f"🟠 API issue {h.status_code}")
    except Exception as e:
        st.sidebar.error("🔴 API Offline")
        st.sidebar.caption(str(e))
        st.stop()

    # upload
    st.sidebar.subheader("Resume Upload")
    resume_file = st.sidebar.file_uploader("Upload Resume (PDF/TXT/DOCX)", type=["pdf", "txt", "docx"])

    st.sidebar.subheader("Targets")
    roles_text = st.sidebar.text_area("Target Roles", value="Solution Architect\nGenAI Architect\nML Engineer", height=90)
    target_roles = [r.strip() for r in roles_text.splitlines() if r.strip()][:4]

    country = st.sidebar.text_input("Country", value="US")
    location = st.sidebar.text_input("Location", value="United States")
    remote = st.sidebar.checkbox("Remote preferred", value=True)
    wfo_ok = st.sidebar.checkbox("On-site/WFO acceptable", value=True)
    visa_required = st.sidebar.checkbox("Visa sponsorship required", value=False)

    recency_hours = st.sidebar.slider("Only jobs posted within (hours)", 12, 168, 36, 6)
    max_jobs = st.sidebar.slider("Jobs to score", 20, 60, 40, 5)

    st.sidebar.subheader("Quality")
    discovery_threshold = st.sidebar.slider("Discovery threshold", 0.50, 0.90, 0.70, 0.05)
    max_refinements = st.sidebar.slider("Max refinements", 1, 6, 3, 1)
    resume_threshold = st.sidebar.slider("Intake threshold", 0.35, 0.90, 0.55, 0.05)
    draft_count = st.sidebar.slider("Draft packages", 3, 20, 10, 1)

    run_btn = st.sidebar.button("🚀 RUN ONE-CLICK", type="primary", use_container_width=True, disabled=(resume_file is None))

    if run_btn:
        prefs = {
            "target_roles": target_roles,
            "country": country,
            "location": location,
            "remote": remote,
            "wfo_ok": wfo_ok,
            "visa_sponsorship_required": visa_required,
            "recency_hours": float(recency_hours),
            "max_jobs": int(max_jobs),
            "discovery_threshold": float(discovery_threshold),
            "max_refinements": int(max_refinements),
            "resume_threshold": float(resume_threshold),
            "draft_count": int(draft_count),
        }
        files = {"resume": (resume_file.name, resume_file.getvalue())}
        data = {"preferences_json": json.dumps(prefs)}
        r = _api_post(api, "/analyze", files=files, data=data, timeout=180)
        out = _safe_json(r)
        if r.status_code >= 400:
            st.error(out)
            st.stop()
        st.session_state["run_id"] = out["run_id"]
        st.session_state["selected_urls"] = set()

    run_id = st.session_state.get("run_id")
    if not run_id:
        st.info("Upload resume and run.")
        return

    view = st.radio("View mode", ["Pilot View", "Engineer View (Fire Engine)"], horizontal=True)
    st.button("🔄 Refresh", use_container_width=True)
    st.caption("Auto-refresh runs every 5s while waiting for HITL states.")

    r = _api_get(api, f"/status/{run_id}", timeout=25)
    state = _safe_json(r)

    status = state.get("status")
    pending = _get_pending(state)

    # Emergency auto-refresh for Waiting/HITL states so controls appear immediately.
    is_waiting_hitl = _is_pending_status(status)
    if is_waiting_hitl:
        st_autorefresh(interval=5000, key=f"hitl_refresh_{run_id}")
    feed = _get_feed(state)
    artifacts = state.get("artifacts", {}) or {}
    evals = state.get("evaluations", []) or []

    pct, label = _progress(state)
    a, b, c, d = st.columns([1.6, 1.0, 1.2, 1.0])
    a.markdown(f"**Run ID**: `{run_id}`")
    b.markdown(f"**Status**: `{status}`")
    c.markdown(f"**Pending**: `{pending}`")
    d.markdown(f"**Progress**: `{pct}%`")
    st.progress(pct / 100.0, text=label)

    with st.expander("Live Agent Feed", expanded=True):
        for ev in feed[-200:]:
            st.write(f"**[{ev.get('layer')} {ev.get('agent')}]** {ev.get('message')}")

    with st.expander("Evaluations", expanded=False):
        if not evals:
            st.caption("No evaluations yet.")
        else:
            for e in evals[-12:]:
                st.write(f"**{e.get('layer_id')}** score={e.get('evaluation_score')} decision={e.get('decision')}")
                fb = e.get("feedback") or []
                if fb:
                    st.caption("\n".join(["- " + str(x) for x in fb[:6]]))

    if view.startswith("Engineer"):
        st.markdown("## Engineer Controls")
        cols = st.columns(6)
        for i, layer in enumerate(["L0", "L1", "L2", "L3", "L4", "L5"]):
            if cols[i].button(f"Run {layer}", use_container_width=True):
                _api_post(api, f"/action/{run_id}", json={"action_type": "execute_layer", "payload": {"layer": layer}}, timeout=60)
                st.success("Triggered")

    _list_downloads(artifacts)

    # Ranking
    st.markdown("### Ranking")
    ranking_ref = artifacts.get("ranking")
    ranking: List[Dict[str, Any]] = []
    if ranking_ref:
        p = ranking_ref.get("path") if isinstance(ranking_ref, dict) else getattr(ranking_ref, "path", None)
        if p and Path(p).exists():
            ranking = _load_json(p)
    if ranking:
        rows = []
        for idx, x in enumerate(ranking[:40], start=1):
            mp = _job_match_percent(x)
            comp = _job_components(x)
            rows.append(
                {
                    "rank": idx,
                    "interview_%": float(x.get("interview_probability_percent") or 0.0),
                    "missing_gap_%": float(x.get("missing_skills_gap_percent") or 0.0),
                    "match_%": mp,
                    "jd_align_%": comp["jd_align"],
                    "ats_proxy_%": comp["ats"],
                    "title": _job_title(x),
                    "url": _job_url(x),
                }
            )
        st.dataframe(rows, use_container_width=True)

    # HITL ranking
    st.markdown("### Human-in-the-Loop")
    # Show scorecard summary (top probability + best gap)
    hitl = (state.get("meta") or {}).get("hitl_summary") or {}
    if hitl:
        st.caption(
            f"Interview Probability (best): {hitl.get('top_interview_probability_percent', 0)}%  |  "
            f"Missing Skills Gap (best): {hitl.get('best_missing_skills_gap_percent', 0)}%"
        )

    if status == "needs_human_approval" and pending == "relax_constraints":
        st.warning("Director recommends relaxing constraints to improve discovery volume.")
        proposal = (state.get("meta") or {}).get("relax_proposal") or {}
        if proposal:
            st.json(proposal)
        c1, c2 = st.columns(2)
        if c1.button("✅ Approve Relax Constraints (Widen Search)", type="primary", use_container_width=True):
            _api_post(api, f"/action/{run_id}", json={"action_type": "relax_constraints", "payload": {}}, timeout=120)
            st.success("Relax applied. Refresh.")
        if c2.button("➡️ Continue Without Relax (Review Current Ranking)", use_container_width=True):
            # Just switch to ranking review by reloading; backend already has ranking.
            st.info("Proceed to ranking review below.")


    jobs_raw_len = len(state.get("jobs_raw") or [])
    jobs_scored_len = len(state.get("jobs_scored") or [])
    retry_count = int(state.get("retry_count") or 0)
    force_manual = bool((state.get("meta") or {}).get("force_manual_job_link"))
    show_manual = (
        (jobs_raw_len == 0 and retry_count >= 2)
        or jobs_scored_len == 0
        and (
            force_manual
            or retry_count >= 2
            or _is_pending_status(status)
            or pending in ("review_ranking", "relax_constraints")
        )
    )
    if show_manual:
        with st.expander("⚡ Manual Job Link (HITL Fallback)", expanded=_is_pending_status(status)):
            st.warning("No scored jobs are currently available. Manual link input is now prioritized for HITL recovery.")
            manual_url = st.text_input("Manual Job Link", key=f"manual_job_url_{run_id}", placeholder="https://...")
            if st.button("🚀 Submit Manual Job Link", use_container_width=True, type="primary"):
                clean = (manual_url or "").strip()
                if not clean:
                    st.warning("Enter a valid job URL before submitting.")
                else:
                    _api_post(api, f"/action/{run_id}", json={"action_type": "approve_ranking", "payload": {"selected_job_urls": [clean]}}, timeout=120)
                    st.success("Manual URL submitted. Refresh to continue L6/L7 flow.")

    if _is_pending_status(status) and pending in ("review_ranking", "relax_constraints"):
        if not ranking:
            st.warning("Ranking not loaded yet. Refresh.")
        else:
            if not st.session_state["selected_urls"]:
                for j in ranking[:6]:
                    if _job_url(j):
                        st.session_state["selected_urls"].add(_job_url(j))

            for i, j in enumerate(ranking[:20], start=1):
                url = _job_url(j)
                sel = st.checkbox(f"{i}. {_job_title(j)}", value=(url in st.session_state["selected_urls"]), key=f"sel_{i}")
                if sel:
                    st.session_state["selected_urls"].add(url)
                else:
                    st.session_state["selected_urls"].discard(url)

            if st.button("✅ Approve Selected → Generate Drafts", type="primary", use_container_width=True):
                _api_post(api, f"/action/{run_id}", json={"action_type": "approve_ranking", "payload": {"selected_job_urls": sorted(list(st.session_state["selected_urls"]))}}, timeout=120)
                st.success("Approved. Refresh.")

    if _is_pending_status(status) and pending == "review_drafts":
        st.info("Drafts ready. Review and approve to finalize L7–L9.")
        _draft_picker(artifacts)
        if st.button("✅ Approve Drafts → Finalize", type="primary", use_container_width=True):
            _api_post(api, f"/action/{run_id}", json={"action_type": "approve_drafts", "payload": {}}, timeout=120)
            st.success("Finalizing. Refresh.")

    hard_override = _layer_is_l5(state) and _is_pending_status(status)
    if hard_override and pending not in {"review_ranking", "review_drafts", "relax_constraints"}:
        st.warning("L5 pending with unrecognized pending_action format. Hard-override controls enabled.")
        c1, c2 = st.columns(2)
        if c1.button("✅ Approve (Hard Override)", type="primary", use_container_width=True):
            payload = {"selected_job_urls": sorted(list(st.session_state.get("selected_urls") or []))}
            _api_post(api, f"/action/{run_id}", json={"action_type": "approve_ranking", "payload": payload}, timeout=120)
            st.success("Approval submitted. Refresh.")
        if c2.button("❌ Reject (Hard Override)", use_container_width=True):
            _api_post(api, f"/action/{run_id}", json={"action_type": "request_refine", "payload": {"reason": "Manual reject via L5 override"}}, timeout=120)
            st.info("Rejection submitted. Refresh.")

    with st.expander("Full State JSON"):
        st.json(state)


if __name__ == "__main__":
    main()
