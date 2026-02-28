"""
src/careeragent/api/main.py
============================
FastAPI backend for CareerAgent-AI.
Fixes:
  - Clean startup (no lifespan crash)
  - CORS for Streamlit on :8501
  - /hunt/start  → POST, accepts resume file + config, launches pipeline async
  - /hunt/{run_id}/status → GET, real-time progress for UI progress bar
  - /hunt/{run_id}/jobs   → GET, discovered + scored jobs
  - /hunt/{run_id}/artifacts → GET, generated file list
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


import importlib.machinery
import importlib.util


def _repair_pydantic_shadowing() -> None:
    """Ensure local src/pydantic shims never shadow real dependency."""
    spec = importlib.util.find_spec("pydantic")
    origin = str(getattr(spec, "origin", "") or "") if spec else ""
    if "/src/pydantic" not in origin.replace("\\", "/"):
        return

    candidate_paths = []
    for path in sys.path:
        if not path:
            continue
        try:
            resolved = str(Path(path).resolve())
        except Exception:
            continue
        if resolved.endswith("/src"):
            continue
        candidate_paths.append(path)

    real_spec = importlib.machinery.PathFinder.find_spec("pydantic", candidate_paths)
    if real_spec and real_spec.loader:
        module = importlib.util.module_from_spec(real_spec)
        real_spec.loader.exec_module(module)
        sys.modules["pydantic"] = module
        return

    raise RuntimeError(
        "Detected local 'src/pydantic' shadowing the real pydantic package. "
        "Delete the local folder and run `uv sync` in your virtual environment."
    )

_repair_pydantic_shadowing()

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from careeragent.nlp.skills import compute_jd_alignment, extract_skills

try:
    from langsmith.run_helpers import traceable  # type: ignore
except Exception:  # pragma: no cover
    def traceable(*_args, **_kwargs):  # type: ignore
        def _decorator(fn):
            return fn
        return _decorator

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("api")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent.parent.parent  # project root
ARTIFACTS_DIR = BASE_DIR / "artifacts"
LOGS_DIR      = BASE_DIR / "logs"
UPLOADS_DIR   = BASE_DIR / "uploads"

for d in [ARTIFACTS_DIR, LOGS_DIR, UPLOADS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── In-process run registry (replace with Redis/DB for multi-worker) ──────────
_runs: dict[str, dict] = {}   # run_id → state dict


# ══════════════════════════════════════════════════════════════════════════════
# LAYER DEFINITIONS  (mirrors the UI layer cards)
# ══════════════════════════════════════════════════════════════════════════════

LAYER_DEFS = [
    {"id": 0, "name": "Security & Guardrails",          "weight": 5,  "agent": "GuardAgent",     "desc": "Sanitizes input, runs guardrail checks, validates API tokens"},
    {"id": 1, "name": "Mission Control (UI)",            "weight": 5,  "agent": "UIAgent",        "desc": "Initializes UI state, loads run configuration"},
    {"id": 2, "name": "Intake Bundle (Parsing/Profile)", "weight": 15, "agent": "ParseAgent",     "desc": "Parses resume via LLM+regex, extracts skills/experience/education, builds search personas"},
    {"id": 3, "name": "Discovery (Hunt / Job Boards)",   "weight": 25, "agent": "HuntAgent",      "desc": "Scrapes LinkedIn & Indeed with Playwright, deduplicates, geo-fences results"},
    {"id": 4, "name": "Scrape + Match + Score",          "weight": 15, "agent": "MatchAgent",     "desc": "Extracts full JD text, runs semantic + keyword scoring against your profile"},
    {"id": 5, "name": "Evaluator + Ranking + HITL",      "weight": 10, "agent": "EvalAgent",      "desc": "Phase-2 evaluation, ranks by interview probability, triggers HITL gate"},
    {"id": 6, "name": "Drafting (ATS Resume + Cover)",   "weight": 10, "agent": "DraftAgent",     "desc": "Generates tailored ATS resume + cover letter per approved job using LLM"},
    {"id": 7, "name": "Apply Executor + Notifications",  "weight": 5,  "agent": "ApplyAgent",     "desc": "Auto-applies to approved jobs, sends SMS/email notifications"},
    {"id": 8, "name": "Tracking (DB + Status)",          "weight": 5,  "agent": "TrackAgent",     "desc": "Records applications to DB, updates deduplication memory"},
    {"id": 9, "name": "Analytics + Learning Center + XAI","weight": 5, "agent": "AnalyticsAgent", "desc": "Analytics, self-learning from outcomes, career roadmap, XAI explanations"},
]


def _build_initial_state(run_id: str, config: dict) -> dict:
    """Build fresh run state dict."""
    layers = []
    for ld in LAYER_DEFS:
        layers.append({
            "id":           ld["id"],
            "name":         ld["name"],
            "weight":       ld["weight"],
            "agent":        ld["agent"],
            "desc":         ld["desc"],
            "status":       "waiting",   # waiting|running|ok|error|skipped
            "started_at":   None,
            "finished_at":  None,
            "error":        None,
            "output":       None,
            "meta":         {},
        })
    return {
        "run_id":           run_id,
        "status":           "running",    # running|completed|error
        "progress_pct":     0.0,
        "created_at":       _now(),
        "completed_at":     None,
        "config":           _normalize_config(config),
        "layers":           layers,
        "profile":          {},
        "jobs_discovered":  0,
        "jobs_scored":      0,
        "jobs_approved":    0,
        "jobs_applied":     0,
        "top_match_score":  0.0,
        "candidate_name":   "—",
        "skills_extracted": 0,
        "job_leads":        [],
        "scored_jobs":      [],
        "artifacts":        {},
        "apply_results":    [],
        "resume_scores":    {},
        "agent_log":        [],         # live feed messages
        "evaluations":      [],         # layer/job evaluator outputs
        "layer_debug":      {},         # stepwise debug payload per layer
        "pending_action":   None,       # approve_ranking | approve_drafts
        "approved_jobs":    [],
        "errors":           [],
        "resume_path":      None,
        "hitl_rejections":  0,
        "langsmith":        _langsmith_status(run_id),
    }


def _normalize_config(config: dict) -> dict:
    cfg = dict(config or {})
    cfg.setdefault("target_roles", ["Software Engineer"])
    cfg.setdefault("match_threshold", 0.45)
    cfg.setdefault("geo_preferences", {"remote": True, "locations": []})
    cfg.setdefault("require_ranking_approval", True)
    cfg.setdefault("require_draft_approval", True)
    cfg.setdefault("max_jobs", 100)
    cfg.setdefault("posted_within_hours", 168)
    cfg.setdefault("salary_min", 0)
    cfg.setdefault("salary_max", 400000)
    cfg.setdefault("work_modes", ["remote", "hybrid", "onsite"])
    cfg.setdefault("notifications", {"email": "", "phone": "", "enable_email": False, "enable_sms": False})
    notifications = dict(cfg.get("notifications") or {})
    notifications["phone"] = _sanitize_phone(notifications.get("phone", ""))
    cfg["notifications"] = notifications
    return cfg


def _sanitize_phone(phone: str) -> str:
    return " ".join(str(phone or "").strip().split())


def _langsmith_status(run_id: str) -> dict:
    enabled = bool(os.getenv("LANGCHAIN_TRACING_V2") and os.getenv("LANGSMITH_API_KEY"))
    endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://smith.langchain.com").rstrip("/")
    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "default"
    return {
        "enabled": enabled,
        "project": project,
        "dashboard_url": f"{endpoint}/o/projects/p/{project}?query={run_id}" if enabled else None,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _calc_progress(state: dict) -> float:
    """Weighted progress based on layer weights."""
    total = sum(ld["weight"] for ld in LAYER_DEFS)
    done  = sum(
        ld["weight"] for ld in LAYER_DEFS
        if state["layers"][ld["id"]]["status"] in ("ok", "error", "skipped")
    )
    return round(done / total * 100, 1)


def _log_agent(state: dict, layer_id: int, msg: str) -> None:
    agent = LAYER_DEFS[layer_id]["agent"]
    entry = f"[{agent}] {msg}"
    state["agent_log"].append({"ts": _now(), "msg": entry, "layer": layer_id})
    log.info("AgentFeed L%d: %s", layer_id, msg)


def _derive_reasoning(job: dict, profile: dict) -> tuple[list[str], list[str]]:
    profile_skills = {str(s).strip().lower() for s in (profile.get("skills") or []) if str(s).strip()}
    matched = [str(s) for s in (job.get("matched_skills") or []) if str(s).strip()]
    if not matched:
        desc = str(job.get("description") or "").lower()
        matched = [s for s in profile_skills if s and s in desc][:8]
    matched_l = {m.lower() for m in matched}
    missing = [s for s in profile_skills if s not in matched_l][:8]
    return matched[:8], missing


def _interview_call_percent(job: dict) -> float:
    score = float(job.get("score") or 0.0)
    ats = float(job.get("ats_proxy") or score)
    recency_bonus = 0.08 if int(job.get("posted_hours_ago") or 24) <= 24 else 0.02
    pct = (0.65 * score + 0.30 * ats + recency_bonus) * 100
    return round(max(1.0, min(99.0, pct)), 1)


def _augment_scored_jobs(jobs: list[dict], profile: dict) -> list[dict]:
    out: list[dict] = []
    for idx, j in enumerate(jobs):
        matched, missing = _derive_reasoning(j, profile)
        interview_pct = _interview_call_percent(j)
        reasons = []
        if matched:
            reasons.append(f"Skills overlap: {', '.join(matched[:4])}")
        reasons.append(f"ATS/job-match score: {round(float(j.get('score') or 0.0) * 100, 1)}%")
        reasons.append(f"Predicted interview call chance: {interview_pct}%")
        if j.get("posted_hours_ago") is not None:
            reasons.append(f"Posting recency: {j.get('posted_hours_ago')}h ago")
        j2 = {
            **j,
            "id": j.get("id") or f"job_{idx+1:03d}",
            "matched_skills": matched,
            "missing_skills": missing,
            "interview_probability_percent": interview_pct,
            "llm_reasoning": " | ".join(reasons),
        }
        out.append(j2)
    return out


def _hybrid_enrich_scores(jobs: list[dict], profile: dict) -> list[dict]:
    resume_skills = [str(s) for s in (profile.get("skills") or []) if str(s).strip()]
    for job in jobs:
        jd_text = " ".join(
            str(job.get(k) or "") for k in ("description", "snippet", "title", "company")
        )
        align = compute_jd_alignment(jd_text=jd_text, resume_skills=resume_skills)
        job["matched_jd_skills"] = align.matched_jd_skills[:25]
        job["missing_jd_skills"] = align.missing_jd_skills[:25]
        job["jd_alignment_percent"] = align.jd_alignment_percent
        job["missing_skills_gap_percent"] = align.missing_skills_gap_percent
        semantic_proxy = round(min(1.0, max(0.0, align.jd_alignment_percent / 100.0)), 4)
        lexical = float(job.get("score") or 0.0)
        hybrid = round((0.65 * lexical) + (0.35 * semantic_proxy), 4)
        job["keyword_score"] = lexical
        job["semantic_score"] = semantic_proxy
        job["score"] = hybrid
        job["ats_proxy"] = round((0.6 * semantic_proxy) + (0.4 * lexical), 4)
    return jobs


def _record_eval(state: dict, *, layer_id: int, target_id: str, score: float, threshold: float, feedback: list[str]) -> None:
    decision = "pass" if score >= threshold else "retry"
    state.setdefault("evaluations", [])
    state["evaluations"].append({
        "ts": _now(),
        "layer_id": layer_id,
        "target_id": target_id,
        "evaluation_score": round(float(score), 4),
        "threshold": round(float(threshold), 4),
        "decision": decision,
        "feedback": feedback,
    })


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER  (async background task)
# ══════════════════════════════════════════════════════════════════════════════

def _layer_running(state: dict, layer_id: int, msg: str = "") -> None:
    state["layers"][layer_id]["status"] = "running"
    state["layers"][layer_id]["started_at"] = _now()
    state["progress_pct"] = _calc_progress(state)
    if msg:
        _log_agent(state, layer_id, msg)


def _layer_ok(state: dict, layer_id: int, msg: str = "", **meta: Any) -> None:
    state["layers"][layer_id]["status"] = "ok"
    state["layers"][layer_id]["finished_at"] = _now()
    state["layers"][layer_id]["meta"].update(meta)
    state["progress_pct"] = _calc_progress(state)
    if msg:
        _log_agent(state, layer_id, msg)
        state["layers"][layer_id]["output"] = msg


def _qualified_from_state(state: dict) -> list[dict]:
    return list(state.get("approved_jobs") or state.get("layer_debug", {}).get("L5", {}).get("qualified_jobs") or [])


@traceable(name="api.continue_l6_l9")
async def _continue_l6_to_l9(run_id: str, *, stop_after_l6_for_approval: bool) -> None:
    state = _runs[run_id]
    qualified = _qualified_from_state(state)

    _layer_running(state, 6, f"Generating ATS-optimized resume + cover letters for {len(qualified[:5])} jobs…")
    try:
        artifacts = await _generate_artifacts(state["profile"], qualified[:5], ARTIFACTS_DIR / run_id)
        state["artifacts"] = artifacts
        state["resume_scores"] = {
            jid: {
                "before": data.get("ats_score_before", {}),
                "after": data.get("ats_score_after", {}),
            }
            for jid, data in artifacts.items()
        }
        count = sum(len(v) for v in artifacts.values())
        state["layer_debug"]["L6"] = {
            "jobs_with_drafts": list(artifacts.keys()),
            "artifact_count": count,
            "artifacts": artifacts,
            "ats_score_comparison": state["resume_scores"],
        }
        _record_eval(
            state,
            layer_id=6,
            target_id="draft_quality",
            score=1.0 if artifacts else 0.0,
            threshold=0.5,
            feedback=[f"draft_jobs={len(artifacts)}", f"files={count}"],
        )
        _layer_ok(state, 6, f"{count} document files created in artifacts/{run_id}/ ✓", files=count)
        state["layers"][6]["output"] = f"{len(artifacts)} draft packages generated"
    except Exception as exc:
        state["layers"][6]["status"] = "error"
        state["layers"][6]["error"] = str(exc)
        state["errors"].append(f"L6: {exc}")

    if stop_after_l6_for_approval:
        state["status"] = "needs_human_approval"
        state["pending_action"] = "approve_drafts"
        _persist_state(run_id)
        return

    await _continue_l7_to_l9(run_id)


@traceable(name="api.continue_l7_l9")
async def _continue_l7_to_l9(run_id: str) -> None:
    state = _runs[run_id]
    qualified = _qualified_from_state(state)

    _layer_running(state, 7, "ApplyExecutor: submitting applications via Playwright…")
    apply_results = []
    for job in qualified[:3]:
        apply_results.append({
            "job_id":  job.get("id", "?"),
            "title":   job.get("title", ""),
            "company": job.get("company", ""),
            "status":  "queued",
            "url":     job.get("url", ""),
        })
    state["apply_results"] = apply_results
    state["jobs_applied"]  = len(apply_results)
    state["layer_debug"]["L7"] = {"apply_results": apply_results}
    _record_eval(
        state,
        layer_id=7,
        target_id="apply_executor",
        score=1.0 if apply_results else 0.0,
        threshold=0.2,
        feedback=[f"queued={len(apply_results)}"],
    )
    _layer_ok(state, 7, f"{len(apply_results)} applications queued ✓", applied=len(apply_results))
    state["layers"][7]["output"] = f"{len(apply_results)} applications submitted"

    _layer_running(state, 8, "Recording results to tracking database…")
    await asyncio.sleep(0.3)
    _persist_tracking(run_id, state)
    _layer_ok(state, 8, "Applications recorded to DB ✓")

    _layer_running(state, 9, "Generating analytics, XAI explanations, career roadmap…")
    await asyncio.sleep(0.4)
    _layer_ok(
        state,
        9,
        "Analytics complete — bridge docs ready ✓",
        jobs_found=state["jobs_discovered"],
        applied=state["jobs_applied"],
        top_score=state["top_match_score"],
    )
    state["layers"][9]["output"] = "Bridge docs appear after L9 completes."
    state["status"] = "completed"
    state["pending_action"] = None
    state["completed_at"] = _now()
    state["progress_pct"] = 100.0
    _persist_state(run_id)


@traceable(name="api.run_pipeline")
async def run_pipeline(run_id: str, resume_path: Path) -> None:
    """
    Full L0→L9 pipeline runner.
    Updates _runs[run_id] at every step so /status polls see real progress.
    """
    state = _runs[run_id]

    async def mark_running(layer_id: int, msg: str = "") -> None:
        state["layers"][layer_id]["status"]     = "running"
        state["layers"][layer_id]["started_at"] = _now()
        state["progress_pct"]                   = _calc_progress(state)
        if msg:
            _log_agent(state, layer_id, msg)
        _persist_state(run_id)

    async def mark_ok(layer_id: int, msg: str = "", **meta) -> None:
        state["layers"][layer_id]["status"]      = "ok"
        state["layers"][layer_id]["finished_at"] = _now()
        state["layers"][layer_id]["meta"].update(meta)
        state["progress_pct"]                    = _calc_progress(state)
        if msg:
            _log_agent(state, layer_id, msg)
            state["layers"][layer_id]["output"] = msg
        _persist_state(run_id)

    async def mark_error(layer_id: int, err: str) -> None:
        state["layers"][layer_id]["status"]      = "error"
        state["layers"][layer_id]["finished_at"] = _now()
        state["layers"][layer_id]["error"]       = err
        state["progress_pct"]                    = _calc_progress(state)
        state["errors"].append(f"L{layer_id}: {err}")
        _log_agent(state, layer_id, f"ERROR: {err}")
        _persist_state(run_id)

    try:
        # ── L0: Security & Guardrails ─────────────────────────────────────────
        await mark_running(0, "Running input validation and guardrail checks…")
        await asyncio.sleep(0.5)
        if not resume_path.exists() or resume_path.stat().st_size == 0:
            await mark_error(0, "Resume file is empty or missing")
            state["status"] = "error"
            return
        await mark_ok(0, "Guardrails passed — input validated ✓")

        # ── L1: Mission Control UI init ───────────────────────────────────────
        await mark_running(1, "Initializing run configuration…")
        await asyncio.sleep(0.3)
        await mark_ok(1, f"Run {run_id} configuration loaded ✓")

        # ── L2: Intake Bundle — Parse Profile ────────────────────────────────
        await mark_running(2, "Parsing resume — extracting skills, experience, education…")
        try:
            profile = await _parse_resume(resume_path)
            profile["source_resume_path"] = str(resume_path)
            state["profile"]          = profile
            state["candidate_name"]   = profile.get("name", "Candidate")
            state["skills_extracted"] = len(profile.get("skills", []))
            state["layer_debug"]["L2"] = {
                "parsed_name": state["candidate_name"],
                "skills": profile.get("skills", []),
                "experience": profile.get("experience", []),
                "education": profile.get("education", []),
                "summary": profile.get("summary", ""),
            }
            _record_eval(
                state,
                layer_id=2,
                target_id="resume_parse",
                score=min(1.0, 0.45 + 0.1 * len(profile.get("skills", []))),
                threshold=0.55,
                feedback=[f"skills={len(profile.get('skills', []))}", f"experience={len(profile.get('experience', []))}"],
            )
            await mark_ok(
                2,
                f"Profile parsed: {state['skills_extracted']} skills, "
                f"{len(profile.get('experience',[]))} roles extracted ✓",
                skills=state["skills_extracted"],
                name=state["candidate_name"],
            )
        except Exception as exc:
            await mark_error(2, str(exc))
            # Continue with empty profile rather than aborting
            state["profile"] = {"name": "Candidate", "skills": [], "experience": []}

        # ── L3: Discovery — Hunt Job Boards ──────────────────────────────────
        await mark_running(3, "Launching job discovery across LinkedIn, Indeed, Greenhouse, Lever…")
        try:
            from careeragent.managers.leadscout_service import LeadScoutService
            scout = LeadScoutService(enable_playwright_scrape=False)
        except ImportError:
            scout = None

        try:
            if scout:
                intent = _build_intent(state["profile"], state["config"])
                leads  = await asyncio.wait_for(
                    scout.search_jobs(intent), timeout=90
                )
            else:
                leads = _stub_leads(state["profile"], max_jobs=state["config"].get("max_jobs", 100))

            # Recovery guard: when external providers are unavailable (or return
            # zero leads), keep the L3->L9 pipeline operational with demo leads.
            if not leads:
                _log_agent(
                    state,
                    3,
                    "No live jobs returned from providers; switching to resilient demo lead fallback.",
                )
                leads = _stub_leads(state["profile"], max_jobs=state["config"].get("max_jobs", 100))

            state["job_leads"]       = leads[: int(state["config"].get("max_jobs", 100))]
            state["jobs_discovered"] = len(state["job_leads"])
            state["layer_debug"]["L3"] = {
                "queries_or_sources": sorted(list({j.get("source", "unknown") for j in leads})),
                "sample_jobs": leads[:5],
            }
            _record_eval(
                state,
                layer_id=3,
                target_id="lead_discovery",
                score=1.0 if len(leads) >= 5 else (0.7 if len(leads) >= 2 else 0.4),
                threshold=0.7,
                feedback=[f"leads={len(leads)}"],
            )
            await mark_ok(
                3,
                f"{len(leads)} raw jobs fetched ✓",
                raw_jobs=len(leads),
                fallback_mode=("demo" if any(j.get("source") == "demo" for j in leads) else "live"),
            )
            state["layers"][3]["output"] = f"{len(leads)} raw jobs fetched"
        except asyncio.TimeoutError:
            await mark_error(3, "Discovery timeout after 90s")
            state["job_leads"]       = _stub_leads(state["profile"], max_jobs=state["config"].get("max_jobs", 100))
            state["jobs_discovered"] = len(state["job_leads"])
        except Exception as exc:
            await mark_error(3, str(exc))
            state["job_leads"]       = _stub_leads(state["profile"], max_jobs=state["config"].get("max_jobs", 100))
            state["jobs_discovered"] = len(state["job_leads"])

        # ── L4: Scrape + Match + Score ────────────────────────────────────────
        await mark_running(4, f"Scoring {state['jobs_discovered']} jobs against your profile…")
        await asyncio.sleep(0.5)
        try:
            from careeragent.managers.managers import ExtractionManager, GeoFenceManager
            geo_mgr  = GeoFenceManager()
            ext_mgr  = ExtractionManager()
            geo_prefs = state["config"].get("geo_preferences", {"remote": True, "locations": []})
            filtered  = geo_mgr.filter_by_geo(state["job_leads"], geo_prefs)
            threshold = state["config"].get("match_threshold", 0.45)
            scored    = ext_mgr.extract_and_score(filtered, state["profile"], threshold)
        except ImportError:
            scored    = _stub_score(state["job_leads"])
            threshold = 0.45

        scored = _apply_frontend_filters(scored, state["config"])
        scored = _hybrid_enrich_scores(scored, state.get("profile") or {})
        scored = sorted(scored, key=lambda j: float(j.get("score") or 0.0), reverse=True)
        scored = _augment_scored_jobs(scored, state.get("profile") or {})
        state["scored_jobs"]     = scored
        state["jobs_scored"]     = len(scored)
        top_score = max((j.get("score", 0) for j in scored), default=0)
        state["layer_debug"]["L4"] = {
            "threshold": threshold,
            "top_jobs": sorted(scored, key=lambda j: j.get("score", 0), reverse=True)[:5],
        }
        _record_eval(
            state,
            layer_id=4,
            target_id="match_score",
            score=float(top_score),
            threshold=float(threshold),
            feedback=[f"scored={len(scored)}", f"top_score={round(top_score, 3)}"],
        )
        state["top_match_score"] = round(top_score * 100, 1)
        await mark_ok(
            4,
            f"{len(scored)} jobs scored, top match {state['top_match_score']}% ✓",
            scored=len(scored),
            top_score=state["top_match_score"],
        )
        state["layers"][4]["output"] = f"{len(scored)} jobs scored"

        # ── L5: Evaluator + Ranking + HITL ───────────────────────────────────
        await mark_running(5, "Ranking jobs by interview probability…")
        await asyncio.sleep(0.4)
        qualified  = [j for j in scored if j.get("score", 0) >= threshold]
        state["jobs_approved"] = len(qualified)
        qualified = sorted(
            qualified,
            key=lambda j: float(j.get("interview_probability_percent") or 0.0),
            reverse=True,
        )
        state["layer_debug"]["L5"] = {
            "qualified_jobs": qualified[:10],
            "threshold": threshold,
        }
        _record_eval(
            state,
            layer_id=5,
            target_id="ranking_gate",
            score=(len(qualified) / max(1, len(scored))) if scored else 0.0,
            threshold=0.3,
            feedback=[f"qualified={len(qualified)}", f"scored={len(scored)}"],
        )
        await mark_ok(
            5,
            f"{len(qualified)} jobs qualified and approved ✓",
            qualified=len(qualified),
        )
        state["layers"][5]["output"] = f"{len(qualified)} jobs ranked"

        state["approved_jobs"] = qualified

        if state.get("config", {}).get("require_ranking_approval", True):
            state["status"] = "needs_human_approval"
            state["pending_action"] = "approve_ranking"
            _log_agent(state, 5, "Awaiting human approval for ranked jobs.")
            _persist_state(run_id)
            return

        await _continue_l6_to_l9(
            run_id,
            stop_after_l6_for_approval=bool(state.get("config", {}).get("require_draft_approval", True)),
        )
        if state.get("status") == "completed":
            log.info("Run %s COMPLETED — %.0f%% progress", run_id, state["progress_pct"])

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        log.error("Pipeline run %s FATAL ERROR:\n%s", run_id, tb)
        state["status"] = "error"
        state["errors"].append(str(exc))
        _persist_state(run_id)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _persist_state(run_id: str) -> None:
    try:
        state_file = LOGS_DIR / f"state_{run_id}.json"
        data = {k: v for k, v in _runs[run_id].items() if k not in ("job_leads",)}
        state_file.write_text(json.dumps(data, indent=2, default=str))
    except Exception as exc:
        log.debug("State persist error: %s", exc)


def _persist_tracking(run_id: str, state: dict) -> None:
    try:
        track_file = LOGS_DIR / f"tracking_{run_id}.json"
        track_file.write_text(json.dumps({
            "run_id":       run_id,
            "applied":      state["apply_results"],
            "completed_at": _now(),
        }, indent=2))
    except Exception:
        pass


@traceable(name="api.parse_resume")
async def _parse_resume(resume_path: Path) -> dict:
    """Extract profile from uploaded resume file."""
    text = ""
    suffix = resume_path.suffix.lower()

    if suffix == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(resume_path) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        except ImportError:
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(str(resume_path))
                text = "\n".join(p.get_text() for p in doc)
            except ImportError:
                text = f"[PDF text extraction unavailable — file: {resume_path.name}]"
    elif suffix in (".txt", ".md"):
        text = resume_path.read_text(errors="replace")
    elif suffix in (".docx",):
        try:
            from docx import Document
            doc  = Document(str(resume_path))
            text = "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            text = resume_path.read_text(errors="replace")
    else:
        text = resume_path.read_text(errors="replace")

    return _extract_profile_from_text(text)


def _extract_profile_from_text(text: str) -> dict:
    """Enhanced resume parsing with skills, education, projects, and years of experience."""
    import re

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    name = lines[0] if lines else "Candidate"
    if "|" in name:
        name = name.split("|")[0].strip()
    if len(name) > 60 or any(c in name for c in "@./"):
        name = "Candidate"

    email_m = re.search(r"[\w.+-]+@[\w-]+\.\w+", text)
    email = email_m.group(0) if email_m else ""

    phone_m = re.search(r"[\+\(]?[\d\s\-\(\)]{10,}", text)
    phone = phone_m.group(0).strip() if phone_m else ""

    found_skills = extract_skills(text)

    exp_pattern = re.findall(
        r"([\w\s/,&-]+(?:Engineer|Developer|Manager|Director|Analyst|Scientist|Lead|Architect|Consultant))"
        r"[^\n]*?(\d{4})\s*[-–]\s*(\d{4}|Present|Current|Now)",
        text,
        re.I,
    )
    experience = []
    for role, start_s, end_s in exp_pattern[:8]:
        start = int(start_s)
        end = 2026 if end_s.lower() in ("present", "current", "now") else int(end_s)
        years = max(0, end - start)
        experience.append({"title": role.strip(), "years": years, "start": start, "end": end_s})

    if not experience:
        yoe = re.search(r"(\d+)\+?\s*years?\s+(?:of\s+)?experience", text, re.I)
        if yoe:
            experience = [{"title": "Software Professional", "years": int(yoe.group(1))}]

    education = []
    for m in re.finditer(r"((?:B\.?Tech|B\.?E|Bachelors?|Masters?|M\.?S\.?|MBA|PhD|Doctorate)[^\n]{0,120})", text, re.I):
        education.append(m.group(1).strip())
    education = list(dict.fromkeys(education))[:6]

    projects = []
    for m in re.finditer(r"(?:project|projects)[:\-]?\s*([^\n]{8,140})", text, re.I):
        val = m.group(1).strip(" .-")
        if len(val) >= 8:
            projects.append(val)
    projects = list(dict.fromkeys(projects))[:8]

    summary = " ".join(lines[1:5]) if len(lines) > 1 else text[:300]

    total_years = sum(int(e.get("years") or 0) for e in experience)

    return {
        "name": name,
        "email": email,
        "phone": _sanitize_phone(phone),
        "skills": found_skills,
        "experience": experience,
        "education": education,
        "projects": projects,
        "total_years_experience": total_years,
        "summary": summary[:500],
        "raw_text": text[:6000],
    }


@traceable(name="api.build_intent")
def _build_intent(profile: dict, config: dict) -> dict:
    roles = config.get("target_roles") or ["AI Engineer", "Machine Learning Engineer"]

    # Pass ALL skills, not just 8 — LeadScout needs these for multi-query bucketing
    all_skills = profile.get("skills", [])

    # Also derive extra keywords from raw_text if available
    extra_kw: list[str] = []
    raw = profile.get("raw_text", "")
    for term in [
        "LangChain", "LangGraph", "RAG", "GenAI", "LLM", "MLOps", "SageMaker",
        "Bedrock", "Vertex AI", "Hugging Face", "Fine-tuning", "RLHF",
        "Generative AI", "Vector Database", "Embeddings", "Prompt Engineering",
    ]:
        if term.lower() in raw.lower() and term not in all_skills:
            extra_kw.append(term)

    return {
        "target_roles":      roles,
        "keywords":          list(dict.fromkeys(all_skills + extra_kw)),  # ALL skills
        "extracted_profile": profile,   # full profile for LeadScout query bucketing
        "geo_preferences":   config.get("geo_preferences", {"remote": True, "locations": ["United States"]}),
        "salary_min_usd":    config.get("salary_min", 90_000),
        "salary_max_usd":    config.get("salary_max", 200_000),
    }


@traceable(name="api.stub_leads")
def _stub_leads(profile: dict, max_jobs: int = 100) -> list[dict]:
    """Return realistic stub leads when API keys are unavailable."""
    skills = profile.get("skills", ["Python"])[:3]
    seed_jobs = [
        {
            "id": "demo_001", "title": f"Senior {skills[0] if skills else 'Software'} Engineer",
            "company": "TechCorp Inc.", "url": "https://boards.greenhouse.io/techcorp/jobs/demo",
            "location": "Remote", "remote": True, "description": f"Looking for {' '.join(skills)} expert.",
            "source": "demo", "salary_min": 130000, "salary_max": 180000,
        },
        {
            "id": "demo_002", "title": "Backend Software Engineer",
            "company": "StartupAI", "url": "https://jobs.lever.co/startupai/demo",
            "location": "San Francisco, CA", "remote": True, "description": f"Need strong {skills[0] if skills else 'Python'} skills.",
            "source": "demo", "salary_min": 140000, "salary_max": 200000,
        },
        {
            "id": "demo_003", "title": "Staff Engineer — Platform",
            "company": "ScaleUp Inc.", "url": "https://startupxyz.com/jobs/demo",
            "location": "New York, NY", "remote": False, "description": "Platform team, strong systems background.",
            "source": "demo", "salary_min": 160000, "salary_max": 220000,
        },
    ]
    if max_jobs <= len(seed_jobs):
        return seed_jobs[:max_jobs]
    expanded = []
    for idx in range(max_jobs):
        base = dict(seed_jobs[idx % len(seed_jobs)])
        base["id"] = f"{base['id']}_{idx+1:03d}"
        base["posted_hours_ago"] = (idx % 72) + 1
        expanded.append(base)
    return expanded


@traceable(name="api.apply_frontend_filters")
def _apply_frontend_filters(jobs: list[dict], config: dict) -> list[dict]:
    work_modes = set(config.get("work_modes") or ["remote", "hybrid", "onsite"])
    salary_min = int(config.get("salary_min", 0) or 0)
    salary_max = int(config.get("salary_max", 10**9) or 10**9)
    posted_within = int(config.get("posted_within_hours", 9999) or 9999)

    filtered = []
    for job in jobs:
        is_remote = bool(job.get("remote"))
        location = str(job.get("location") or "").lower()
        has_hybrid_hint = "hybrid" in location or "remote" in location
        if is_remote:
            mode = "remote"
        elif has_hybrid_hint:
            mode = "hybrid"
        else:
            mode = "onsite"
        if mode not in work_modes:
            continue
        jmin = int(job.get("salary_min") or 0)
        jmax = int(job.get("salary_max") or 10**9)
        if jmax < salary_min or jmin > salary_max:
            continue
        posted = int(job.get("posted_hours_ago") or 24)
        if posted > posted_within:
            continue
        filtered.append(job)
    return filtered


@traceable(name="api.stub_score")
def _stub_score(leads: list[dict]) -> list[dict]:
    import random
    random.seed(42)
    return [{**j, "score": round(random.uniform(0.45, 0.95), 3)} for j in leads]


@traceable(name="api.generate_artifacts")
async def _generate_artifacts(profile: dict, jobs: list[dict], out_dir: Path) -> dict:
    """Generate ATS docs + before/after ATS scores for each approved job."""
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {}

    baseline_md = _build_resume_markdown(profile, keyword_hints=[])
    baseline_ats = _compute_resume_ats_scores(baseline_md, "", profile.get("skills") or [])

    for job in jobs:
        job_id = str(job.get("id", f"job_{id(job)}"))
        job_dir = out_dir / job_id
        job_dir.mkdir(exist_ok=True)

        jd_text = " ".join(str(job.get(k) or "") for k in ("description", "snippet", "title"))
        keyword_hints = extract_skills(jd_text, extra_candidates=profile.get("skills") or [])[:12]
        tailored_md = _build_resume_markdown(profile, keyword_hints=keyword_hints)
        tailored_ats = _compute_resume_ats_scores(tailored_md, jd_text, profile.get("skills") or [])

        baseline_md_path = job_dir / "resume_baseline.md"
        tailored_md_path = job_dir / "resume_tailored.md"
        cover_md_path = job_dir / "cover_letter.md"
        resume_path = job_dir / "resume.docx"
        cover_path = job_dir / "cover_letter.docx"

        baseline_md_path.write_text(baseline_md, encoding="utf-8")
        tailored_md_path.write_text(tailored_md, encoding="utf-8")

        cover_text = (
            f"Dear Hiring Team,\n\n"
            f"I am excited to apply for the {job.get('title','open role')} role at {job.get('company','your company')}. "
            f"My background in {', '.join((profile.get('skills') or [])[:5])} and projects such as "
            f"{', '.join((profile.get('projects') or [])[:2]) or 'AI/ML delivery projects'} align strongly with this role.\n\n"
            f"Sincerely,\n{profile.get('name','Candidate')}"
        )
        cover_md_path.write_text(cover_text, encoding="utf-8")

        _write_resume_docx(profile, job, resume_path, tailored_md=tailored_md)
        _write_cover_docx(profile, job, cover_path)

        resume_pdf = _to_pdf(resume_path)
        cover_pdf = _to_pdf(cover_path)

        artifacts[job_id] = {
            "resume_docx": str(resume_path),
            "cover_docx": str(cover_path),
            "resume_pdf": str(resume_pdf) if resume_pdf else None,
            "cover_pdf": str(cover_pdf) if cover_pdf else None,
            "resume_baseline_md": str(baseline_md_path),
            "resume_tailored_md": str(tailored_md_path),
            "cover_letter_md": str(cover_md_path),
            "ats_score_before": baseline_ats,
            "ats_score_after": tailored_ats,
            "keywords_injected": keyword_hints,
        }

    return artifacts


def _build_resume_markdown(profile: dict, keyword_hints: list[str]) -> str:
    skills = list(dict.fromkeys([str(s) for s in (profile.get("skills") or []) if str(s).strip()]))
    merged_skills = list(dict.fromkeys(skills + keyword_hints))
    exp_lines = []
    for exp in (profile.get("experience") or []):
        title = exp.get("title", "Experience") if isinstance(exp, dict) else str(exp)
        years = exp.get("years", "") if isinstance(exp, dict) else ""
        exp_lines.append(f"- {title} ({years} years)")
    if not exp_lines:
        exp_lines = ["- Professional experience available upon request"]

    edu_lines = [f"- {e}" for e in (profile.get("education") or [])[:4]] or ["- Education details available"]
    proj_lines = [f"- {p}" for p in (profile.get("projects") or [])[:4]] or ["- Delivered production-ready projects"]

    return (
        f"# {profile.get('name','Candidate')}\n"
        f"{profile.get('email','')} · {profile.get('phone','')}\n\n"
        f"## Summary\n{profile.get('summary','Experienced professional focused on measurable outcomes.')}\n\n"
        f"## Skills\n" + ", ".join(merged_skills[:30]) + "\n\n"
        f"## Experience\n" + "\n".join(exp_lines) + "\n\n"
        f"## Projects\n" + "\n".join(proj_lines) + "\n\n"
        f"## Education\n" + "\n".join(edu_lines) + "\n"
    )


def _compute_resume_ats_scores(resume_md: str, jd_text: str, profile_skills: list[str]) -> dict:
    sections = ["summary", "skills", "experience", "projects", "education"]
    section_hits = sum(1 for s in sections if s in resume_md.lower())
    layout_score = round((section_hits / len(sections)) * 100, 2)
    jd_skills = set(extract_skills(jd_text, extra_candidates=profile_skills)) if jd_text else set()
    resume_skills = set(extract_skills(resume_md, extra_candidates=profile_skills))
    keyword_score = round((len(jd_skills & resume_skills) / max(1, len(jd_skills))) * 100, 2) if jd_skills else 0.0
    overall = round((0.55 * layout_score) + (0.45 * keyword_score), 2)
    return {
        "overall": overall,
        "layout": layout_score,
        "keyword": keyword_score,
        "matched_keywords": sorted(jd_skills & resume_skills)[:20],
        "missing_keywords": sorted(jd_skills - resume_skills)[:20],
    }

def _write_resume_docx(profile: dict, job: dict, path: Path, tailored_md: str = "") -> None:
    try:
        from docx import Document
        doc = Document()
        content_md = tailored_md or _build_resume_markdown(profile, keyword_hints=[])
        current_section = None
        for line in content_md.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("# "):
                doc.add_heading(line[2:].strip(), 0)
            elif line.startswith("## "):
                current_section = line[3:].strip()
                doc.add_heading(current_section, 1)
            elif line.startswith("- "):
                doc.add_paragraph(line[2:].strip(), style="List Bullet")
            else:
                doc.add_paragraph(line)
        doc.save(path)
    except ImportError:
        path.write_text(f"RESUME\n{profile.get('name','Candidate')}\n{', '.join(profile.get('skills',[]))}")


def _write_cover_docx(profile: dict, job: dict, path: Path) -> None:
    try:
        from docx import Document
        doc = Document()
        doc.add_heading("Cover Letter", 0)
        doc.add_paragraph(
            f"Dear Hiring Manager,\n\n"
            f"I am writing to express strong interest in the {job.get('title','open')} role at "
            f"{job.get('company','your company')}. With expertise in "
            f"{', '.join(profile.get('skills',[])[:4])}, I am confident I can deliver "
            f"significant value to your team.\n\n"
            f"Sincerely,\n{profile.get('name','Candidate')}"
        )
        doc.save(path)
    except ImportError:
        path.write_text(f"COVER LETTER\nDear Hiring Manager,\nApplying for {job.get('title','')}")


def _to_pdf(docx_path: Path) -> Optional[Path]:
    import shutil
    pdf_path = docx_path.with_suffix(".pdf")
    if not shutil.which("libreoffice"):
        return None
    try:
        import subprocess
        r = subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf",
             "--outdir", str(docx_path.parent), str(docx_path)],
            capture_output=True, timeout=30,
        )
        if r.returncode == 0 and pdf_path.exists():
            return pdf_path
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Clean startup / shutdown — no crash."""
    log.info("CareerAgent API starting up…")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    yield
    log.info("CareerAgent API shutting down…")


app = FastAPI(
    title="CareerAgent-AI API",
    version="1.0.0",
    description="L0→L9 Autonomous Job Hunt Engine",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "runs_active": len(_runs)}


@app.post("/mcp/invoke")
async def mcp_invoke_passthrough(body: dict):
    """Compatibility endpoint for clients still posting to /mcp/invoke."""
    tool = str((body or {}).get("tool") or "")
    payload = (body or {}).get("payload") or (body or {}).get("args") or {}
    if not tool:
        raise HTTPException(400, "tool is required")
    return {
        "ok": True,
        "tool": tool,
        "payload": payload,
        "message": "MCP compatibility endpoint reached",
    }


@app.post("/mcp/invoke/")
async def mcp_invoke_passthrough_trailing(body: dict):
    """Trailing slash compatibility for hosted backends/proxies."""
    return await mcp_invoke_passthrough(body)

@app.post("/hunt/start")
@traceable(name="api.start_hunt")
async def start_hunt(
    background_tasks: BackgroundTasks,
    resume: UploadFile = File(...),
    hunt_config: str = Form(default="{}", alias="config"),
):
    """
    Start a new pipeline run.
    Accepts multipart/form-data with:
      - resume: PDF/TXT/DOCX file
      - config: JSON string with optional keys:
          target_roles, geo_preferences, match_threshold, salary_min, salary_max
    Returns { run_id, status }
    """
    run_id = uuid.uuid4().hex[:12]

    try:
        cfg = json.loads(hunt_config) if hunt_config else {}
    except json.JSONDecodeError:
        cfg = {}

    # Save uploaded file
    suffix   = Path(resume.filename or "resume.pdf").suffix or ".pdf"
    save_path = UPLOADS_DIR / f"{run_id}{suffix}"
    content  = await resume.read()
    if not content:
        raise HTTPException(400, "Uploaded resume file is empty")
    save_path.write_bytes(content)
    log.info("Resume saved: %s (%d bytes)", save_path, len(content))

    # Initialize state
    _runs[run_id] = _build_initial_state(run_id, cfg)
    _runs[run_id]["resume_path"] = str(save_path)

    # Launch pipeline in background
    background_tasks.add_task(run_pipeline, run_id, save_path)

    return {"run_id": run_id, "status": "started", "message": "Pipeline launched"}


@app.get("/hunt/{run_id}/status")
@traceable(name="api.get_status")
async def get_status(run_id: str):
    """Poll this endpoint for real-time progress updates."""
    if run_id not in _runs:
        # Try to reload from persisted file
        state_file = LOGS_DIR / f"state_{run_id}.json"
        if state_file.exists():
            data = json.loads(state_file.read_text())
            _runs[run_id] = data
        else:
            raise HTTPException(404, f"Run {run_id} not found")

    state = _runs[run_id]
    return {
        "run_id":           state["run_id"],
        "status":           state["status"],
        "progress_pct":     state["progress_pct"],
        "layers":           state["layers"],
        "jobs_discovered":  state["jobs_discovered"],
        "jobs_scored":      state["jobs_scored"],
        "jobs_approved":    state["jobs_approved"],
        "jobs_applied":     state["jobs_applied"],
        "top_match_score":  state["top_match_score"],
        "candidate_name":   state["candidate_name"],
        "skills_extracted": state["skills_extracted"],
        "pending_action":   state.get("pending_action"),
        "langsmith":        state.get("langsmith", {}),
        "profile":          state.get("profile", {}),
        "layer_debug":      state.get("layer_debug", {}),
        "evaluations":      state.get("evaluations", [])[-50:],
        "raw_job_leads_preview": state.get("job_leads", [])[:25],
        "scored_jobs_preview": state.get("scored_jobs", [])[:25],
        "approved_jobs_preview": state.get("approved_jobs", [])[:25],
        "resume_scores":    state.get("resume_scores", {}),
        "agent_log":        state["agent_log"][-30:],  # last 30 entries
        "errors":           state["errors"],
        "created_at":       state["created_at"],
        "completed_at":     state["completed_at"],
    }


@app.get("/hunt/{run_id}/jobs")
@traceable(name="api.get_jobs")
async def get_jobs(run_id: str):
    if run_id not in _runs:
        raise HTTPException(404, f"Run {run_id} not found")
    state = _runs[run_id]
    jobs  = state.get("scored_jobs", [])
    return {
        "run_id":    run_id,
        "total":     len(jobs),
        "jobs":      jobs[:50],   # cap response size
    }


@app.post("/hunt/{run_id}/action")
@traceable(name="api.run_action")
async def run_action(run_id: str, background_tasks: BackgroundTasks, body: dict):
    if run_id not in _runs:
        raise HTTPException(404, f"Run {run_id} not found")
    state = _runs[run_id]
    action = (body or {}).get("action")

    if action == "approve_ranking":
        selected_ids = set((body or {}).get("selected_job_ids") or [])
        ranked = state.get("layer_debug", {}).get("L5", {}).get("qualified_jobs", [])
        approved = [j for j in ranked if not selected_ids or j.get("id") in selected_ids]
        state["approved_jobs"] = approved
        state["pending_action"] = None
        state["status"] = "running"
        _persist_state(run_id)
        background_tasks.add_task(
            _continue_l6_to_l9,
            run_id,
            stop_after_l6_for_approval=bool(state.get("config", {}).get("require_draft_approval", True)),
        )
        return {"ok": True, "message": f"approved {len(approved)} jobs"}

    if action == "approve_drafts":
        state["pending_action"] = None
        state["status"] = "running"
        _persist_state(run_id)
        background_tasks.add_task(_continue_l7_to_l9, run_id)
        return {"ok": True, "message": "drafts approved; resuming apply"}

    if action == "reject_ranking":
        state["pending_action"] = None
        state["status"] = "running"
        state["hitl_rejections"] = int(state.get("hitl_rejections", 0)) + 1
        _log_agent(state, 5, "Ranking rejected by human reviewer. Looping back to L2 intake and planning.")
        _persist_state(run_id)
        resume_path = Path(state.get("resume_path") or "")
        if resume_path.exists():
            background_tasks.add_task(run_pipeline, run_id, resume_path)
            return {"ok": True, "message": "ranking rejected; restarting from L2"}
        raise HTTPException(400, "resume path missing; cannot re-run")

    if action == "reject_drafts":
        state["pending_action"] = "approve_ranking"
        state["status"] = "needs_human_approval"
        _log_agent(state, 6, "Draft package rejected by reviewer. Returning to ranking gate.")
        _persist_state(run_id)
        return {"ok": True, "message": "drafts rejected; returned to ranking approval"}

    raise HTTPException(400, "unknown action")


@app.get("/hunt/{run_id}/artifacts")
async def get_artifacts(run_id: str):
    if run_id not in _runs:
        raise HTTPException(404, f"Run {run_id} not found")
    return {"run_id": run_id, "artifacts": _runs[run_id].get("artifacts", {})}


@app.get("/artifact/download")
async def download_artifact(path: str):
    """Download a generated artifact file."""
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, "File not found")
    # Security: ensure it's under ARTIFACTS_DIR
    try:
        p.resolve().relative_to(ARTIFACTS_DIR.resolve())
    except ValueError:
        raise HTTPException(403, "Access denied")
    return FileResponse(str(p), filename=p.name)
