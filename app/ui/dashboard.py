from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

# --- Path bootstrap (keep stable imports no matter how streamlit is launched)
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_API = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")


# ----------------------------
# HTTP helpers
# ----------------------------
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


# ----------------------------
# State compatibility helpers (critical fix)
# ----------------------------
def _get_pending(state: Dict[str, Any]) -> Optional[str]:
    # New backend writes pending_action top-level; old UI read meta only.
    if state.get("pending_action") is not None:
        return state.get("pending_action")
    meta = state.get("meta", {}) or {}
    return meta.get("pending_action")


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
    # fallback
    return ["L0", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]


def _progress_from_steps(state: Dict[str, Any]) -> Tuple[int, str]:
    """
    Returns (percent, current_label)
    """
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

    # If no steps yet but last_layer exists, approximate
    if not steps:
        meta = state.get("meta", {}) or {}
        last_layer = meta.get("last_layer")
        if last_layer and last_layer in plan:
            idx = plan.index(last_layer)
            ok = max(ok, idx)  # approximate
            current = last_layer

    pct = int(round((ok / total) * 100))
    label = f"Current: {current}" if current else ("Waiting / HITL" if state.get("status") == "needs_human_approval" else "Queued")
    return pct, label


# ----------------------------
# Job rendering helpers
# ----------------------------
def _job_url(job: Dict[str, Any]) -> str:
    return str(job.get("url") or job.get("job_id") or "")


def _job_title(job: Dict[str, Any]) -> str:
    return str(job.get("title") or job.get("role_title") or "(untitled)")


def _job_match_percent(job: Dict[str, Any]) -> float:
    # support both schemas
    if job.get("overall_match_percent") is not None:
        return float(job.get("overall_match_percent") or 0.0)
    if job.get("match_percent") is not None:
        return float(job.get("match_percent") or 0.0)
    if job.get("interview_chance_score") is not None:
        return float(job.get("interview_chance_score") or 0.0) * 100.0
    return 0.0


def _job_components(job: Dict[str, Any]) -> Dict[str, float]:
    comp = job.get("components") or {}
    # normalized to 0–100
    skill = float(comp.get("skill_overlap") or 0.0) * 100.0
    exp = float(comp.get("experience_alignment") or 0.0) * 100.0
    ats = float(comp.get("ats_score") or 0.0) * 100.0
    return {"skill": skill, "exp": exp, "ats": ats}


def _list_artifacts_downloads(artifacts: Dict[str, Any]) -> None:
    st.markdown("### Downloads")
    cols = st.columns(4)

    def btn(i: int, key: str, fname: str, mime: str) -> None:
        ref = artifacts.get(key)
        if not ref:
            return
        p = ref.get("path")
        if not p or not Path(p).exists():
            return
        cols[i].download_button(
            f"⬇️ {fname}",
            data=_read_bytes(p),
            file_name=fname,
            mime=mime,
            use_container_width=True,
        )

    btn(0, "jobs_raw", "jobs_raw.json", "application/json")
    btn(1, "jobs_scored", "jobs_scored.json", "application/json")
    btn(2, "ranking", "ranking.json", "application/json")
    btn(3, "extracted_profile", "extracted_profile.json", "application/json")


def _preview_drafts(artifacts: Dict[str, Any]) -> None:
    # Show resume_*/cover_* artifacts
    resume_keys = sorted([k for k in artifacts.keys() if k.startswith("resume_")])
    cover_keys = sorted([k for k in artifacts.keys() if k.startswith("cover_")])
    all_keys = resume_keys + cover_keys

    if not all_keys:
        st.caption("No draft files yet (expected resume_* / cover_* artifacts).")
        return

    st.markdown("### Draft Preview (ATS Resume + Cover Letter)")
    pick = st.selectbox("Select draft file", all_keys)
    p = artifacts[pick]["path"]
    if not p or not Path(p).exists():
        st.warning(f"Missing file: {p}")
        return

    txt = _read_text(p)
    st.code(txt[:12000], language="markdown")
    st.download_button(
        "⬇️ Download selected draft",
        data=txt.encode("utf-8"),
        file_name=Path(p).name,
        mime="text/markdown",
        use_container_width=True,
    )


# ----------------------------
# UI
# ----------------------------
def main() -> None:
    st.set_page_config(page_title="CareerAgent-AI Mission Control", layout="wide")
    st.title("CareerAgent-AI — Mission Control (One-Click Automation)")
    st.caption("L0→L9: Intake → Hunt → Scrape/Match → Rank → Draft → Learning Plan → HITL approvals")

    # session state
    if "run_id" not in st.session_state:
        st.session_state["run_id"] = ""
    if "selected_urls" not in st.session_state:
        st.session_state["selected_urls"] = set()

    api = st.sidebar.text_input("API Base URL", value=DEFAULT_API).rstrip("/")

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
    resume_file = st.sidebar.file_uploader("Upload Resume (PDF/TXT/DOCX)", type=["pdf", "txt", "docx"])

    # Preferences
    st.sidebar.subheader("Targets")
    roles_text = st.sidebar.text_area(
        "Target Roles (1 per line, up to 4)",
        value="Data Scientist\nML Engineer\nGenAI Engineer",
        height=90,
    )
    target_roles = [r.strip() for r in roles_text.splitlines() if r.strip()][:4]

    country = st.sidebar.text_input("Country", value="US")
    location = st.sidebar.text_input("Location", value="United States")
    remote = st.sidebar.checkbox("Remote preferred", value=True)
    wfo_ok = st.sidebar.checkbox("On-site/WFO acceptable", value=True)
    salary = st.sidebar.text_input("Salary target (optional)", value="")

    # ✅ keep your visa option
    visa_required = st.sidebar.checkbox("Visa sponsorship required (F1/OPT/H1B)", value=False)

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

    # ✅ red button (Streamlit "primary" respects theme; your screenshot shows it red)
    run_btn = st.sidebar.button(
        "🚀 RUN ONE-CLICK",
        type="primary",
        use_container_width=True,
        disabled=(resume_file is None),
    )

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
        st.session_state["selected_urls"] = set()
        st.success(f"Run started: {out['run_id']} (status: {out.get('status')})")

    run_id = st.session_state.get("run_id") or run_id_in.strip()
    if not run_id:
        st.info("Upload resume and click RUN, or paste a run_id to monitor.")
        return

    # --------- Two views (Pilot vs Fire Engine)
    view = st.radio("View mode", ["Pilot View", "Engineer View (Fire Engine)"], horizontal=True)

    refresh = st.button("🔄 Refresh", use_container_width=True)

    r = _api_get(api, f"/status/{run_id}", timeout=25)
    if r.status_code != 200:
        st.warning(f"Run not found yet ({r.status_code}).")
        return

    state = _safe_json(r)
    status = state.get("status", "unknown")
    pending = _get_pending(state)
    feed = _get_feed(state)
    artifacts = state.get("artifacts", {}) or {}
    evals = state.get("evaluations", []) or []

    # --------- Top status bar + progress
    pct, label = _progress_from_steps(state)
    c1, c2, c3, c4 = st.columns([1.6, 1.2, 1.2, 1.0])
    with c1:
        st.markdown(f"**Run ID**: `{run_id}`")
    with c2:
        st.markdown(f"**Status**: `{status}`")
    with c3:
        st.markdown(f"**Pending**: `{pending}`")
    with c4:
        st.markdown(f"**Progress**: `{pct}%`")
    st.progress(pct / 100.0, text=label)

    # --------- Step timeline / bar (shows what’s running)
    with st.expander("Steps timeline (what ran, what’s running now)", expanded=(view.startswith("Engineer"))):
        steps = state.get("steps") or []
        if not steps:
            st.caption("No steps recorded yet.")
        else:
            st.dataframe(
                [
                    {
                        "layer": s.get("layer_id"),
                        "status": s.get("status"),
                        "started": s.get("started_at_utc"),
                        "finished": s.get("finished_at_utc"),
                    }
                    for s in steps
                ],
                use_container_width=True,
            )

    # --------- Live feed
    st.markdown("### Live Agent Feed")
    for ev in feed[-220:]:
        st.write(f"**[{ev.get('layer')} {ev.get('agent')}]** {ev.get('message')}")

    # --------- Evaluations
    with st.expander("Evaluations (scores + reasons)"):
        if not evals:
            st.caption("No evaluations yet.")
        else:
            for e in evals[-12:]:
                score = e.get("evaluation_score")
                thr = e.get("threshold")
                st.write(f"**{e.get('layer_id')} / {e.get('target_id')}** score={score} threshold={thr}")
                fb = e.get("feedback") or []
                if fb:
                    st.write("- " + "\n- ".join([str(x) for x in fb[:6]]))

    # --------- Engineer view: run layers manually
    if view.startswith("Engineer"):
        st.markdown("## Engineer Controls (Run layers manually)")
        cols = st.columns(6)
        layers = ["L0", "L2", "L3", "L4", "L5", "L6"]
        for i, layer in enumerate(layers):
            if cols[i].button(f"Run {layer}", use_container_width=True):
                rr = _api_post(
                    api,
                    f"/action/{run_id}",
                    json={"action_type": "execute_layer", "payload": {"layer": layer}},
                    timeout=90,
                )
                if rr.status_code >= 400:
                    st.error(rr.text[:1200])
                else:
                    st.success(f"Triggered {layer}. Refresh to see updates.")

    # --------- Downloads
    _list_artifacts_downloads(artifacts)

    # --------- Discovered job links
    st.markdown("### Discovered Job Links (jobs_raw)")
    jobs_raw_ref = artifacts.get("jobs_raw")
    if jobs_raw_ref and _exists(jobs_raw_ref.get("path")):
        jobs_raw = _load_json(jobs_raw_ref["path"])
        st.caption(f"Saved at: `{jobs_raw_ref['path']}`")
        for j in jobs_raw[:40]:
            st.markdown(f"- [{_job_title(j)}]({_job_url(j)})")
    else:
        st.caption("jobs_raw not available yet.")

    # --------- Ranking
    st.markdown("### Ranking (Top 20–40)")
    ranking_ref = artifacts.get("ranking")
    ranking: List[Dict[str, Any]] = []
    if ranking_ref and _exists(ranking_ref.get("path")):
        ranking = _load_json(ranking_ref["path"])
        st.caption(f"Saved at: `{ranking_ref['path']}`")

        # show table for both schemas
        rows = []
        for idx, x in enumerate(ranking[:60], start=1):
            mp = _job_match_percent(x)
            comp = _job_components(x)
            rows.append(
                {
                    "rank": x.get("rank", idx),
                    "match_%": round(mp, 2),
                    "skill_%": round(comp["skill"], 0),
                    "exp_%": round(comp["exp"], 0),
                    "ats_%": round(comp["ats"], 0),
                    "title": _job_title(x),
                    "board/source": x.get("board") or x.get("source"),
                    "url": _job_url(x),
                    "missing_skills": ", ".join((x.get("missing_skills") or [])[:6]),
                }
            )
        st.dataframe(rows, use_container_width=True)
    else:
        st.caption("Ranking not available yet.")

    # --------- Optional resume cleanup (keep your old behavior)
    if pending == "resume_cleanup_optional":
        st.markdown("## Optional Resume Cleanup (improves ATS + scoring)")
        resume_ref = (artifacts.get("resume_raw") or {}).get("path")
        current = _read_text(resume_ref) if resume_ref and Path(resume_ref).exists() else ""
        new_text = st.text_area(
            "Paste improved resume text (add email/phone + headings + bullets)",
            value=current,
            height=220,
        )
        if st.button("✅ Submit improved resume & rerun"):
            rr = _api_post(
                api,
                f"/action/{run_id}",
                json={"action_type": "resume_cleanup_submit", "payload": {"resume_text": new_text}},
                timeout=90,
            )
            if rr.status_code >= 400:
                st.error(rr.text[:1200])
            else:
                st.success("Submitted. Refresh in a few seconds.")

    # --------- HITL controls (now actually works + selection + previews)
    st.markdown("### Human-in-the-Loop Controls")

    if status == "needs_human_approval" and pending == "review_ranking":
        st.info("Ranking is ready. Select jobs and approve to generate tailored ATS resume + cover letter drafts.")

        # default select top draft_count
        if ranking and not st.session_state["selected_urls"]:
            for j in ranking[: min(int(draft_count), len(ranking))]:
                u = _job_url(j)
                if u:
                    st.session_state["selected_urls"].add(u)

        if not ranking:
            st.warning("No ranking loaded yet. Refresh.")
        else:
            # approval grid with checkboxes + score breakdown
            for idx, job in enumerate(ranking[:30], start=1):
                url = _job_url(job)
                title = _job_title(job)
                mp = _job_match_percent(job)
                comp = _job_components(job)

                with st.expander(f"{idx}. {title} — {mp:.1f}%"):
                    st.markdown(f"[Open job link]({url})")
                    a, b, c = st.columns(3)
                    a.metric("Skill", f"{comp['skill']:.0f}%")
                    b.metric("Exp", f"{comp['exp']:.0f}%")
                    c.metric("ATS", f"{comp['ats']:.0f}%")

                    sel = st.checkbox(
                        "Select this job",
                        value=(url in st.session_state["selected_urls"]),
                        key=f"sel_job_{idx}",
                    )
                    if sel and url:
                        st.session_state["selected_urls"].add(url)
                    if (not sel) and url in st.session_state["selected_urls"]:
                        st.session_state["selected_urls"].remove(url)

                    ms = job.get("missing_skills") or []
                    if ms:
                        st.caption("Missing skills: " + ", ".join([str(x) for x in ms[:12]]))

            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Approve Selected → Generate Draft Packages", type="primary", use_container_width=True):
                    payload = {"selected_job_urls": sorted(list(st.session_state["selected_urls"]))}
                    rr = _api_post(api, f"/action/{run_id}", json={"action_type": "approve_ranking", "payload": payload}, timeout=120)
                    if rr.status_code >= 400:
                        st.error(f"approve_ranking failed: {rr.status_code}\n\n{rr.text[:1500]}")
                    else:
                        st.success("Approved. Draft generation started. Refresh and review drafts below.")
            with c2:
                reason = st.text_input("Reason (optional) to refine ranking", value="")
                if st.button("❌ Reject Ranking → Re-run Discovery", use_container_width=True):
                    rr = _api_post(api, f"/action/{run_id}", json={"action_type": "reject_ranking", "payload": {"reason": reason}}, timeout=90)
                    if rr.status_code >= 400:
                        st.error(f"reject_ranking failed: {rr.status_code}\n\n{rr.text[:1500]}")
                    else:
                        st.warning("Rejected. Discovery rerun started.")

    if status == "needs_human_approval" and pending == "review_drafts":
        st.info("Drafts are ready. Preview/download, then approve to finalize.")

        _preview_drafts(artifacts)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Approve Drafts → Complete (Simulated Apply)", type="primary", use_container_width=True):
                rr = _api_post(api, f"/action/{run_id}", json={"action_type": "approve_drafts", "payload": {}}, timeout=120)
                if rr.status_code >= 400:
                    st.error(f"approve_drafts failed: {rr.status_code}\n\n{rr.text[:1500]}")
                else:
                    st.success("Approved. Completing run… Refresh to see status=completed.")
        with c2:
            reason = st.text_input("Reason (optional) to revise drafts", value="")
            if st.button("❌ Reject Drafts → Back to Ranking", use_container_width=True):
                rr = _api_post(api, f"/action/{run_id}", json={"action_type": "reject_drafts", "payload": {"reason": reason}}, timeout=90)
                if rr.status_code >= 400:
                    st.error(f"reject_drafts failed: {rr.status_code}\n\n{rr.text[:1500]}")
                else:
                    st.warning("Rejected drafts. Returning to ranking.")

    # --------- Drafts + learning plan (keep old section + add new visibility)
    st.markdown("### Drafts + Learning Plan")

    drafts_ref = artifacts.get("drafts_bundle")
    if drafts_ref and _exists(drafts_ref.get("path")):
        bundle = _load_json(drafts_ref["path"])
        st.json(bundle)
    else:
        # if resume_*/cover_* exist, show them
        resume_keys = sorted([k for k in artifacts.keys() if k.startswith("resume_")])
        cover_keys = sorted([k for k in artifacts.keys() if k.startswith("cover_")])
        if resume_keys or cover_keys:
            st.caption("Draft files created (resume_*/cover_*). Use the HITL Draft Review above to preview/download.")
        else:
            st.caption("Draft bundle not available yet.")

    with st.expander("Full State JSON"):
        st.json(state)


if __name__ == "__main__":
    main()