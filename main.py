"""CareerAgent-AI Orchestrator (Phase 1 + Phase 2 Evaluator).

What this file fixes (per your request):
  1) Progress Bar UI Sync
     - Every node explicitly sets `state['progress_percent']` at node START.
     - Mapping: L1:10, L2:20, L3:40, L4:50, L5:60, L6:75, L7:85, L8:95, L9:100.

  2) Discovery ↔ Evaluator infinite loop
     - Evaluator MUST produce `refinement_feedback` on RETRY_SEARCH.
     - Discovery ingests that feedback and rewrites the next Serper/Tavily query using
       negative operators (e.g., "-India -Bangalore") and stronger must-include terms.
     - Retry loops are bounded and each retry forces a *strategy shift* (query changes),
       preventing “same query, same rejection” loops.

  3) Dynamic constraint validation
     - Evaluator compares job metadata against dynamic `state['preferences']`
       (e.g., country=US, recency_hours=36). Non-matching jobs get score=0 with reason.
     - If 100% of a batch is rejected, evaluator emits strategy-shift feedback (not a duplicate retry).

  4) Engineer View transparency
     - Every decision and its reasoning is appended to `state['evaluation_logs']`.
       Your Fire Engine/Engineer View can render it live.

Layer: L0-L9 (LangGraph state machine)

Run examples (CLI):
  python main.py --resume data/demo/resume_sample_backend_ml.txt --roles "Solution Architect" --location "United States" --top-n 8

Notes:
  - This orchestrator is dependency-tolerant: if Phase 2 fails, it falls back to Phase 1 artifacts.
  - LangSmith tracing is enabled via environment variables, and evaluator decisions are traceable.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

# Ensure `src/` is importable when running as `python main.py`
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

# LangSmith tracing (EvaluatorAgent uses @traceable)
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "careeragent-ai-phase2")
os.environ.setdefault("LANGCHAIN_ENDPOINT", os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"))

try:
    from langgraph.graph import StateGraph, START, END

    _LANGGRAPH_AVAILABLE = True
except Exception:
    StateGraph = None  # type: ignore
    START = END = None  # type: ignore
    _LANGGRAPH_AVAILABLE = False

from pydantic import BaseModel, Field

from careeros.jobs.service import build_jobpost_from_text, write_jobpost
from careeros.matching.service import compute_match, write_match_result
from careeros.parsing.service import build_profile_from_text, write_profile
from careeros.ranking.service import rank_all_jobs, write_shortlist
from careeros.evidence.vector_store import VectorRecord, index_records
from careeros.integrations.job_boards.sources import discover_job_urls_for_roles, fetch_job_page_text

from evaluator import evaluate_ranked_jobs


# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------


class AgentStateModel(BaseModel):
    """Shared state across nodes."""

    run_id: str

    # Required by you (Prompt 1)
    evaluation: Dict[str, Any] = Field(default_factory=lambda: {"score": 0.0, "reason": "", "action": ""})
    phase1_results: Dict[str, Any] = Field(default_factory=dict)
    messages: List[Dict[str, Any]] = Field(default_factory=list)

    # Fixes requested in this prompt
    progress_percent: int = 0
    current_layer: str = "L0"
    evaluation_logs: List[Dict[str, Any]] = Field(default_factory=list)
    refinement_feedback: str = ""

    # Dynamic constraints (no hardcoding)
    preferences: Dict[str, Any] = Field(
        default_factory=lambda: {
            "country": "US",
            "recency_hours": 36,
        }
    )

    # Loop controls
    iteration: int = 0
    max_iterations: int = 3

    # Fault tolerance
    phase2_failed: bool = False
    phase2_error: str = ""


class AgentState(TypedDict, total=False):
    run_id: str
    evaluation: Dict[str, Any]
    phase1_results: Dict[str, Any]
    messages: List[Dict[str, Any]]

    progress_percent: int
    current_layer: str
    evaluation_logs: List[Dict[str, Any]]
    refinement_feedback: str
    preferences: Dict[str, Any]

    iteration: int
    max_iterations: int

    phase2_failed: bool
    phase2_error: str


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


_PROGRESS_MAP: dict[str, int] = {
    "L0": 0,
    "L1": 10,
    "L2": 20,
    "L3": 40,
    "L4": 50,
    "L5": 60,
    "L6": 75,
    "L7": 85,
    "L8": 95,
    "L9": 100,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: str | Path) -> Path:
    pp = Path(p)
    pp.mkdir(parents=True, exist_ok=True)
    return pp


def _write_json(path: str | Path, payload: Any) -> str:
    fp = Path(path)
    _ensure_dir(fp.parent)
    fp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(fp)


def _copy_to_current(versioned_path: str, current_path: str) -> None:
    src = Path(versioned_path)
    dst = Path(current_path)
    _ensure_dir(dst.parent)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _db_connect(database_url: str):
    """Create a DB-API connection.

    Supports:
      - sqlite:///path.db
      - sqlitecloud://... (requires `sqlitecloud` package)
    """
    if database_url.startswith("sqlitecloud://"):
        try:
            import sqlitecloud  # type: ignore

            return sqlitecloud.connect(database_url)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"sqlitecloud connection failed (install sqlitecloud?): {e}")

    if database_url.startswith("sqlite:///"):
        db_path = database_url.replace("sqlite:///", "", 1)
        return sqlite3.connect(db_path)

    return sqlite3.connect(database_url)


def persist_evaluation(
    *,
    run_id: str,
    evaluation: dict[str, Any],
    phase1_results: dict[str, Any],
    database_url: str,
) -> dict[str, Any]:
    """Persist evaluation to DB for every run."""

    conn = None
    try:
        conn = _db_connect(database_url)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluator_runs (
              run_id TEXT PRIMARY KEY,
              created_at_utc TEXT,
              score REAL,
              action TEXT,
              reason TEXT,
              evaluation_json TEXT,
              profile_path TEXT,
              ranking_path TEXT
            )
            """
        )
        cur.execute(
            """
            INSERT OR REPLACE INTO evaluator_runs
              (run_id, created_at_utc, score, action, reason, evaluation_json, profile_path, ranking_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                str(evaluation.get("ts") or _utc_now_iso()),
                float(evaluation.get("score") or 0.0),
                str(evaluation.get("action") or ""),
                str(evaluation.get("reason") or ""),
                json.dumps(evaluation),
                str(phase1_results.get("profile_path") or ""),
                str(phase1_results.get("ranking_path") or ""),
            ),
        )
        conn.commit()
        return {"status": "ok"}
    except Exception as e:  # noqa: BLE001
        out = {"status": "degraded", "error": str(e), "run_id": run_id, "ts": _utc_now_iso()}
        _write_json(Path("outputs/phase2") / f"db_persist_failed_{run_id}_{_stamp()}.json", out)
        return out
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


def _emit_snapshot(state: AgentState, *, status: str = "running") -> None:
    """Write a stable snapshot for UI polling.

    This is the missing piece when you want progress to move at node START.
    Even if a node later blocks, UI can still show the last emitted layer and %.
    """
    run_id = state.get("run_id") or "unknown"
    payload = {
        "status": status,
        "run_id": run_id,
        "current_layer": state.get("current_layer"),
        "progress_percent": state.get("progress_percent"),
        "evaluation": state.get("evaluation"),
        "refinement_feedback": state.get("refinement_feedback"),
        "evaluation_logs": state.get("evaluation_logs") or [],
        "messages": state.get("messages") or [],
        "phase1": state.get("phase1_results") or {},
        "ts": _utc_now_iso(),
    }
    _write_json("outputs/state/current_run.json", payload)
    _write_json(Path("outputs/state/runs") / f"{run_id}.json", payload)


def _start_node(
    state: AgentState,
    *,
    layer: str,
    note: str,
) -> dict[str, Any]:
    """Update progress at node START + emit UI snapshot."""
    pct = int(_PROGRESS_MAP.get(layer, 0))
    msgs = list(state.get("messages") or [])
    evlogs = list(state.get("evaluation_logs") or [])

    msgs.append({"role": "system", "content": f"{layer} start: {note}", "ts": _utc_now_iso()})
    evlogs.append(
        {
            "ts": _utc_now_iso(),
            "layer": layer,
            "event": "START",
            "progress_percent": pct,
            "note": note,
        }
    )

    updates = {
        "current_layer": layer,
        "progress_percent": pct,
        "messages": msgs,
        "evaluation_logs": evlogs,
    }

    snap = dict(state)
    snap.update(updates)
    _emit_snapshot(snap, status="running")
    return updates


def _fail_safe_evaluation(*, run_id: str, reason: str) -> dict[str, Any]:
    return {"score": 0.0, "reason": reason, "action": "FALLBACK_PHASE1", "run_id": run_id, "ts": _utc_now_iso()}


# -----------------------------------------------------------------------------
# Nodes (L0-L9)
# -----------------------------------------------------------------------------


def node_l0_bootstrap(state: AgentState) -> AgentState:
    updates = _start_node(state, layer="L0", note="Bootstrap run")

    run_id = state.get("run_id") or uuid.uuid4().hex

    # Normalize base fields so later nodes never KeyError.
    base = {
        "run_id": run_id,
        "evaluation": state.get("evaluation") or {"score": 0.0, "reason": "", "action": ""},
        "phase1_results": state.get("phase1_results") or {},
        "messages": updates["messages"],
        "evaluation_logs": updates["evaluation_logs"],
        "progress_percent": updates["progress_percent"],
        "current_layer": updates["current_layer"],
        "preferences": state.get("preferences")
        or {
            "country": "US",
            "recency_hours": 36,
        },
        "refinement_feedback": str(state.get("refinement_feedback") or ""),
        "iteration": int(state.get("iteration") or 0),
        "max_iterations": int(state.get("max_iterations") or 3),
        "phase2_failed": bool(state.get("phase2_failed") or False),
        "phase2_error": str(state.get("phase2_error") or ""),
    }

    return base


def node_l1_intake(state: AgentState) -> AgentState:
    updates = _start_node(state, layer="L1", note="Load resume text & user constraints")

    p1 = dict(state.get("phase1_results") or {})
    resume_path = str(p1.get("resume_path") or "").strip()

    if resume_path and Path(resume_path).exists():
        resume_text = Path(resume_path).read_text(encoding="utf-8")
    else:
        resume_text = str(p1.get("resume_text") or "")

    p1["resume_text"] = resume_text
    p1["intake_ts"] = _utc_now_iso()

    return {"phase1_results": p1, **updates}


def node_l2_profile_extract(state: AgentState) -> AgentState:
    updates = _start_node(state, layer="L2", note="Extract evidence profile from resume")

    run_id = state["run_id"]
    p1 = dict(state.get("phase1_results") or {})
    resume_text = str(p1.get("resume_text") or "")

    if not resume_text.strip():
        ev = _fail_safe_evaluation(run_id=run_id, reason="Missing resume text; cannot extract profile")
        msgs = list(updates["messages"])
        msgs.append({"role": "system", "content": ev["reason"], "ts": ev["ts"]})
        return {"evaluation": ev, "messages": msgs, "phase2_failed": True, "phase2_error": ev["reason"], **updates}

    try:
        profile = build_profile_from_text(resume_text, candidate_name=str(p1.get("candidate_name") or "") or None)
        profile_path = str(write_profile(profile))

        out_dir = _ensure_dir("outputs/phase2")
        ts = _stamp()
        extracted_versioned = str(out_dir / f"extracted_profile_{run_id}_{ts}.json")
        extracted_current = str(out_dir / "extracted_profile.json")

        _write_json(extracted_versioned, profile.model_dump())
        _copy_to_current(extracted_versioned, extracted_current)

        p1["profile_path"] = profile_path
        p1["extracted_profile_path"] = extracted_versioned
        p1["extracted_profile_current"] = extracted_current

        return {"phase1_results": p1, **updates}
    except Exception as e:  # noqa: BLE001
        ev = _fail_safe_evaluation(run_id=run_id, reason=f"Profile extraction failed: {e}")
        return {"evaluation": ev, "phase2_failed": True, "phase2_error": str(e), **updates}


def _normalize_negative_operators(feedback: str) -> str:
    """Convert refinement feedback into negative operators suitable for Serper/Tavily queries."""
    if not feedback:
        return ""

    # Already contains explicit negatives? Keep them.
    if "-" in feedback:
        negs = [tok for tok in feedback.split() if tok.strip().startswith("-") and len(tok.strip()) > 1]
        return " ".join(dict.fromkeys(negs))

    # Light heuristic: extract proper nouns after "exclude/avoid/not" phrases.
    fb = feedback.replace(";", " ").replace(",", " ")
    tokens = [t.strip() for t in fb.split() if t.strip()]

    stop = {"exclude", "avoiding", "avoid", "not", "in", "within", "last", "hours", "hour", "posted", "roles", "role", "jobs", "job"}

    cand: list[str] = []
    for i, t in enumerate(tokens):
        low = t.lower()
        if low in {"exclude", "avoid", "avoiding", "not"}:
            # grab next few tokens as candidates
            for nxt in tokens[i + 1 : i + 6]:
                if nxt.lower() in stop:
                    continue
                # prefer capitalized or country-like tokens
                if nxt[:1].isupper() and nxt.isalpha():
                    cand.append(nxt)

    # Deduplicate, keep small
    cand = list(dict.fromkeys(cand))[:6]
    return " ".join([f"-{c}" for c in cand])


def _extract_recency_hours(feedback: str, default_hours: int) -> int:
    if not feedback:
        return default_hours
    m = None
    for pat in (r"last\s+(\d{1,3})\s*h", r"last\s+(\d{1,3})\s*hours", r"within\s+(\d{1,3})\s*hours"):
        m = __import__("re").search(pat, feedback.lower())
        if m:
            break
    if not m:
        return default_hours
    try:
        v = int(m.group(1))
        return max(1, min(168, v))
    except Exception:
        return default_hours


def node_l3_discovery_and_extraction(state: AgentState) -> AgentState:
    updates = _start_node(state, layer="L3", note="Discovery (Serper/Tavily) + extraction")

    run_id = state["run_id"]
    p1 = dict(state.get("phase1_results") or {})
    prefs = dict(state.get("preferences") or {})

    # Enforce strategy shift per retry loop
    it = int(state.get("iteration") or 0)
    prev_action = str((state.get("evaluation") or {}).get("action") or "").upper().strip()
    if prev_action == "RETRY_SEARCH":
        it += 1

    # Feedback must be applied on retries
    feedback = str(state.get("refinement_feedback") or "").strip()
    if not feedback:
        # fallback to last evaluator message feedback
        msgs = updates.get("messages") or []
        if msgs:
            last_fb = str(msgs[-1].get("feedback") or "").strip()
            feedback = last_fb

    neg_ops = _normalize_negative_operators(feedback)

    # Recency: prefer dynamic preference, override if feedback includes a number
    pref_recency = int(prefs.get("recency_hours") or p1.get("recent_hours") or 72)
    recency = _extract_recency_hours(feedback, pref_recency)
    prefs["recency_hours"] = recency

    roles = p1.get("roles") or []
    if isinstance(roles, str):
        roles = [roles]
    roles = [r.strip() for r in roles if isinstance(r, str) and r.strip()]
    if not roles:
        roles = ["Solution Architect"]

    # Must-include terms from feedback (keep safe; don't hallucinate)
    must_terms: list[str] = []
    for kw in ["Solution Architecture", "Solution Architect", "GenAI", "LLM", "Architecture"]:
        if kw.lower() in feedback.lower():
            must_terms.append(f'"{kw}"' if " " in kw else kw)

    # Strategy shift: on each retry, increase breadth and ensure query changes.
    max_per_source = 2 + min(3, it)  # 2→3→4→5
    top_n = int(p1.get("top_n") or 8)
    daily_limit = max(top_n * (1 + min(2, it)), 10)

    # Location is dynamic preference; keep it US-focused by default.
    location = str(p1.get("location") or "United States")
    if prefs.get("country"):
        # If your UI stores country=US, force location string to include US/United States
        if str(prefs["country"]).upper() in {"US", "USA", "UNITED STATES"} and "united" not in location.lower():
            location = "United States"

    # Build query-injected roles. Since underlying discovery API only accepts `role`, we embed operators there.
    query_suffix = " ".join([*must_terms, neg_ops]).strip()

    refined_roles = [f"{r} {query_suffix}".strip() if query_suffix else r for r in roles]

    # Persist what we are about to query (Engineer View transparency)
    evlogs = list(updates.get("evaluation_logs") or [])
    evlogs.append(
        {
            "ts": _utc_now_iso(),
            "layer": "L3",
            "event": "QUERY_BUILD",
            "iteration": it,
            "location": location,
            "recent_hours": recency,
            "max_per_source": max_per_source,
            "daily_limit": daily_limit,
            "roles": refined_roles,
            "refinement_feedback": feedback,
        }
    )

    p1["recent_hours"] = recency
    p1["discovery_feedback"] = feedback
    p1["iteration"] = it

    try:
        out = discover_job_urls_for_roles(
            roles=refined_roles,
            location=location,
            max_per_source=max_per_source,
            daily_limit=daily_limit,
            recent_hours=recency,
        )

        urls = out.get("urls") or []
        job_texts = out.get("job_texts") or []

        job_paths: list[str] = []
        job_items: list[dict[str, Any]] = []

        for i, url in enumerate(urls[: max(5, top_n)]):
            fetched = fetch_job_page_text(url, timeout_s=6)
            raw = (fetched.get("text") or "").strip()
            if not raw:
                raw = str(job_texts[i]) if i < len(job_texts) else ""
            job = build_jobpost_from_text(raw_text=raw, url=url)
            jp = str(write_jobpost(job))
            job_paths.append(jp)
            job_items.append(
                {
                    "job_path": jp,
                    "url": url,
                    "fetch_status": fetched.get("status"),
                    "notes": fetched.get("notes"),
                }
            )

        disc_payload = {
            "run_id": run_id,
            "ts": _utc_now_iso(),
            "roles": refined_roles,
            "location": location,
            "recent_hours": recency,
            "max_per_source": max_per_source,
            "daily_limit": daily_limit,
            "urls": urls,
            "job_paths": job_paths,
            "notes": out.get("errors") or [],
            "refinement_feedback": feedback,
        }
        disc_path = _write_json(Path("outputs/phase2") / f"discovery_{run_id}_{_stamp()}.json", disc_payload)

        p1["discovery_path"] = disc_path
        p1["job_paths"] = job_paths
        p1["discovery_urls"] = urls
        p1["job_items"] = job_items

        return {
            "phase1_results": p1,
            "preferences": prefs,
            "iteration": it,
            "evaluation_logs": evlogs,
            **updates,
        }

    except Exception as e:  # noqa: BLE001
        evlogs.append(
            {
                "ts": _utc_now_iso(),
                "layer": "L3",
                "event": "ERROR",
                "error": str(e),
                "traceback": traceback.format_exc(limit=3),
            }
        )
        # Fail-safe: no discovery results, but keep run alive.
        p1["job_paths"] = []
        p1["discovery_error"] = str(e)
        return {
            "phase1_results": p1,
            "preferences": prefs,
            "evaluation_logs": evlogs,
            "phase2_failed": True,
            "phase2_error": f"L3 discovery failed: {e}",
            **updates,
        }


def node_l4_identity_rag(state: AgentState) -> AgentState:
    updates = _start_node(state, layer="L4", note="Index identity RAG (Qdrant/Chroma) based on .env")

    run_id = state["run_id"]
    p1 = dict(state.get("phase1_results") or {})

    try:
        profile_path = str(p1.get("profile_path") or "")
        job_paths = p1.get("job_paths") or []
        profile = json.loads(Path(profile_path).read_text(encoding="utf-8")) if profile_path else {}

        records: list[VectorRecord] = []
        if profile:
            records.append(
                VectorRecord(
                    item_id="profile",
                    item_type="profile",
                    text=str(profile.get("raw_text") or ""),
                    metadata={"run_id": run_id, "candidate": profile.get("candidate_name")},
                )
            )
        for jp in job_paths:
            try:
                job = json.loads(Path(jp).read_text(encoding="utf-8"))
                records.append(
                    VectorRecord(
                        item_id=Path(jp).stem,
                        item_type="job",
                        text=str(job.get("raw_text") or ""),
                        metadata={"run_id": run_id, "url": job.get("url"), "location": job.get("location")},
                    )
                )
            except Exception:
                continue

        idx = index_records(run_id, records) if records else {"status": "degraded", "note": "no_records"}
        out_path = _write_json(Path("outputs/phase2") / f"identity_rag_{run_id}_{_stamp()}.json", idx)
        p1["identity_rag"] = idx
        p1["identity_rag_path"] = out_path

        return {"phase1_results": p1, **updates}
    except Exception as e:  # noqa: BLE001
        # Phase 2 failure should not kill Phase 1.
        evlogs = list(updates.get("evaluation_logs") or [])
        evlogs.append({"ts": _utc_now_iso(), "layer": "L4", "event": "ERROR", "error": str(e)})
        return {
            "phase1_results": p1,
            "evaluation_logs": evlogs,
            "phase2_failed": True,
            "phase2_error": f"L4 identity RAG failed: {e}",
            **updates,
        }


def node_l5_matcher_and_ranker(state: AgentState) -> AgentState:
    updates = _start_node(state, layer="L5", note="Matcher + Ranker")

    run_id = state["run_id"]
    p1 = dict(state.get("phase1_results") or {})

    profile_path = str(p1.get("profile_path") or "")
    job_paths = p1.get("job_paths") or []
    top_n = int(p1.get("top_n") or 8)

    try:
        shortlist = rank_all_jobs(profile_path=profile_path, top_n=top_n, run_id=run_id, job_paths=list(job_paths))
        shortlist_path = str(write_shortlist(shortlist))

        items: list[dict[str, Any]] = []
        for it in shortlist.items:
            try:
                job_data = json.loads(Path(it.job_path).read_text(encoding="utf-8"))
            except Exception:
                job_data = {}
            items.append(
                {
                    "job_path": it.job_path,
                    "score": round(float(it.score) / 100.0, 4) if float(it.score) > 1.0 else round(float(it.score), 4),
                    "matched_skills": it.overlap_skills,
                    "missing_skills": it.missing_skills,
                    "url": job_data.get("url"),
                    "job_text": (job_data.get("raw_text") or "")[:12000],
                }
            )

        ranking_payload = {
            "run_id": run_id,
            "ts": _utc_now_iso(),
            "profile_path": profile_path,
            "shortlist_path": shortlist_path,
            "items": items,
        }

        out_dir = _ensure_dir("outputs/phase2")
        ts = _stamp()
        ranking_versioned = str(out_dir / f"ranking_{run_id}_{ts}.json")
        ranking_current = str(out_dir / "ranking.json")

        _write_json(ranking_versioned, ranking_payload)
        _copy_to_current(ranking_versioned, ranking_current)

        # Persist a match_result artifact for UI compatibility (top job)
        top_job_path = shortlist.items[0].job_path if shortlist.items else (job_paths[0] if job_paths else "")
        overlap = shortlist.items[0].overlap_skills if shortlist.items else []
        if top_job_path and profile_path:
            try:
                from careeros.jobs.schema import JobPost
                from careeros.parsing.schema import EvidenceProfile

                prof = EvidenceProfile.model_validate_json(Path(profile_path).read_text(encoding="utf-8"))
                job = JobPost.model_validate_json(Path(top_job_path).read_text(encoding="utf-8"))
                mr = compute_match(prof, job, run_id, profile_path=profile_path, job_path=top_job_path)
                mr_path = str(write_match_result(mr))
                p1["top_match_result_path"] = mr_path
            except Exception:
                pass

        p1["shortlist_path"] = shortlist_path
        p1["ranking_path"] = ranking_versioned
        p1["ranking_current"] = ranking_current
        p1["top_job_path"] = top_job_path
        p1["top_overlap_skills"] = overlap

        return {"phase1_results": p1, **updates}

    except Exception as e:  # noqa: BLE001
        evlogs = list(updates.get("evaluation_logs") or [])
        evlogs.append({"ts": _utc_now_iso(), "layer": "L5", "event": "ERROR", "error": str(e)})
        p1["ranking_error"] = str(e)
        return {
            "phase1_results": p1,
            "evaluation": _fail_safe_evaluation(run_id=run_id, reason=f"Ranker failed: {e}"),
            "evaluation_logs": evlogs,
            "phase2_failed": True,
            "phase2_error": f"L5 matcher/ranker failed: {e}",
            **updates,
        }


def node_l5_evaluator(state: AgentState) -> AgentState:
    # Still L5 but logically Phase 2.
    updates = _start_node(state, layer="L5", note="EvaluatorAgent (Gemini-1.5-Flash + CRAG)")

    run_id = state["run_id"]
    p1 = dict(state.get("phase1_results") or {})

    try:
        out = evaluate_ranked_jobs(
            run_id=run_id,
            extracted_profile_path=str(p1.get("extracted_profile_current") or ""),
            ranking_path=str(p1.get("ranking_current") or ""),
            pass_threshold=float(p1.get("pass_threshold") or 0.7),
            model=str(p1.get("evaluator_model") or "gemini-1.5-flash"),
            preferences=dict(state.get("preferences") or {}),
        )

        # Merge evaluation + refinement feedback + evaluation logs
        evaluation = out.get("evaluation") or {}
        refinement_feedback = str(out.get("refinement_feedback") or evaluation.get("refinement_feedback") or "").strip()

        msgs = list(updates.get("messages") or [])
        msgs.extend(out.get("messages") or [])

        evlogs = list(updates.get("evaluation_logs") or [])
        evlogs.extend(out.get("evaluation_logs") or [])

        # Persist decision immediately (even if we retry)
        try:
            db_url = os.getenv("DATABASE_URL", "sqlite:///career_os.db")
            persist_evaluation(run_id=run_id, evaluation=evaluation, phase1_results=p1, database_url=db_url)
        except Exception:
            pass

        # Ensure feedback exists on RETRY_SEARCH to avoid no-op loops
        action = str(evaluation.get("action") or "").upper().strip()
        if action == "RETRY_SEARCH" and not refinement_feedback:
            refinement_feedback = "Tighten query: US-only roles, last 36 hours, include Solution Architecture, exclude India (use -India)."
            evaluation["refinement_feedback"] = refinement_feedback

        # Log final evaluator decision for Engineer View
        evlogs.append(
            {
                "ts": _utc_now_iso(),
                "layer": "L5_Evaluator",
                "event": "DECISION",
                "score": evaluation.get("score"),
                "action": evaluation.get("action"),
                "reason": evaluation.get("reason"),
                "refinement_feedback": refinement_feedback,
            }
        )

        return {
            "evaluation": evaluation,
            "refinement_feedback": refinement_feedback,
            "messages": msgs,
            "evaluation_logs": evlogs,
            **updates,
        }

    except Exception as e:  # noqa: BLE001
        # Hard fallback to Phase 1
        ev = _fail_safe_evaluation(run_id=run_id, reason=f"Phase 2 evaluator failed: {e}")
        msgs = list(updates.get("messages") or [])
        msgs.append({"role": "evaluator", "content": ev["reason"], "ts": ev["ts"], "feedback": ""})
        evlogs = list(updates.get("evaluation_logs") or [])
        evlogs.append({"ts": _utc_now_iso(), "layer": "L5_Evaluator", "event": "ERROR", "error": str(e)})
        return {
            "evaluation": ev,
            "messages": msgs,
            "evaluation_logs": evlogs,
            "phase2_failed": True,
            "phase2_error": str(e),
            **updates,
        }


def node_l6_checkpoint(state: AgentState) -> AgentState:
    updates = _start_node(state, layer="L6", note="Persist checkpoint + fallback manifest")

    run_id = state["run_id"]
    p1 = dict(state.get("phase1_results") or {})
    evaluation = dict(state.get("evaluation") or {})

    evaluation.setdefault("score", 0.0)
    evaluation.setdefault("reason", "")
    evaluation.setdefault("action", "")
    evaluation.setdefault("run_id", run_id)
    evaluation.setdefault("ts", _utc_now_iso())

    # Persist to DB
    db_url = os.getenv("DATABASE_URL", "sqlite:///career_os.db")
    p1["db_persist"] = persist_evaluation(run_id=run_id, evaluation=evaluation, phase1_results=p1, database_url=db_url)

    # Fallback manifest (manual One-Click generation)
    top_job_path = str(p1.get("top_job_path") or "")
    fallback_manifest = {
        "run_id": run_id,
        "ts": _utc_now_iso(),
        "profile_path": p1.get("profile_path"),
        "job_path": top_job_path,
        "overlap_skills": p1.get("top_overlap_skills") or [],
        "shortlist_path": p1.get("shortlist_path"),
        "ranking_path": p1.get("ranking_path"),
        "phase2_failed": bool(state.get("phase2_failed") or False),
        "phase2_error": str(state.get("phase2_error") or ""),
    }
    _write_json(Path("outputs/phase2") / f"fallback_manifest_{run_id}_{_stamp()}.json", fallback_manifest)
    p1["fallback_manifest"] = fallback_manifest

    return {"phase1_results": p1, **updates}


def node_l7_postprocess(state: AgentState) -> AgentState:
    updates = _start_node(state, layer="L7", note="Post-process (UI readiness)")
    return {**updates}


def node_l8_export(state: AgentState) -> AgentState:
    updates = _start_node(state, layer="L8", note="Export (no-op placeholder for now)")
    return {**updates}


def node_l9_done(state: AgentState) -> AgentState:
    updates = _start_node(state, layer="L9", note="Finalize run")

    # Emit final snapshot with status ok (even if Phase2 failed; Phase1 fallback exists)
    final_state = dict(state)
    final_state.update(updates)

    # Mark completion
    _emit_snapshot(final_state, status="ok")

    return {**updates}


# -----------------------------------------------------------------------------
# Routing
# -----------------------------------------------------------------------------


def route_after_evaluator(state: AgentState) -> str:
    """Quality gate + loop bounds.

    Returns:
      - "retry_discovery" to go back to L3
      - "checkpoint" to continue forward
    """
    ev = state.get("evaluation") or {}
    action = str(ev.get("action") or "").upper().strip()

    try:
        score = float(ev.get("score") or 0.0)
    except Exception:
        score = 0.0

    it = int(state.get("iteration") or 0)
    mx = int(state.get("max_iterations") or 3)

    # Hard fallback if Phase 2 already failed
    if state.get("phase2_failed"):
        return "checkpoint"

    # Retry path
    if action == "RETRY_SEARCH" or score < 0.7:
        if it >= mx:
            # clamp, proceed with fallback artifacts
            ev["action"] = "FALLBACK_PHASE1"
            ev["reason"] = (str(ev.get("reason") or "") + " | max refinement iterations reached").strip(" |")
            return "checkpoint"
        return "retry_discovery"

    return "checkpoint"


# -----------------------------------------------------------------------------
# Graph
# -----------------------------------------------------------------------------


def build_graph():
    if not _LANGGRAPH_AVAILABLE:
        raise RuntimeError("langgraph is not installed")

    g = StateGraph(AgentState)

    g.add_node("L0_bootstrap", node_l0_bootstrap)
    g.add_node("L1_intake", node_l1_intake)
    g.add_node("L2_profile", node_l2_profile_extract)
    g.add_node("L3_discovery", node_l3_discovery_and_extraction)
    g.add_node("L4_identity_rag", node_l4_identity_rag)

    g.add_node("L5_matcher", node_l5_matcher_and_ranker)
    g.add_node("L5_evaluator", node_l5_evaluator)

    g.add_node("L6_checkpoint", node_l6_checkpoint)
    g.add_node("L7_postprocess", node_l7_postprocess)
    g.add_node("L8_export", node_l8_export)
    g.add_node("L9_done", node_l9_done)

    g.add_edge(START, "L0_bootstrap")
    g.add_edge("L0_bootstrap", "L1_intake")
    g.add_edge("L1_intake", "L2_profile")
    g.add_edge("L2_profile", "L3_discovery")
    g.add_edge("L3_discovery", "L4_identity_rag")
    g.add_edge("L4_identity_rag", "L5_matcher")
    g.add_edge("L5_matcher", "L5_evaluator")

    g.add_conditional_edges(
        "L5_evaluator",
        route_after_evaluator,
        {
            "retry_discovery": "L3_discovery",
            "checkpoint": "L6_checkpoint",
        },
    )

    g.add_edge("L6_checkpoint", "L7_postprocess")
    g.add_edge("L7_postprocess", "L8_export")
    g.add_edge("L8_export", "L9_done")
    g.add_edge("L9_done", END)

    return g.compile()


def run_pipeline(initial: AgentState) -> dict[str, Any]:
    """Execute pipeline with hard failure protection."""

    init = AgentStateModel.model_validate(
        {
            "run_id": initial.get("run_id") or uuid.uuid4().hex,
            "evaluation": initial.get("evaluation") or {"score": 0.0, "reason": "", "action": ""},
            "phase1_results": initial.get("phase1_results") or {},
            "messages": initial.get("messages") or [],
            "progress_percent": int(initial.get("progress_percent") or 0),
            "current_layer": str(initial.get("current_layer") or "L0"),
            "evaluation_logs": initial.get("evaluation_logs") or [],
            "refinement_feedback": str(initial.get("refinement_feedback") or ""),
            "preferences": initial.get("preferences")
            or {
                "country": "US",
                "recency_hours": 36,
            },
            "iteration": int(initial.get("iteration") or 0),
            "max_iterations": int(initial.get("max_iterations") or 3),
            "phase2_failed": bool(initial.get("phase2_failed") or False),
            "phase2_error": str(initial.get("phase2_error") or ""),
        }
    )

    state: AgentState = init.model_dump()  # type: ignore[assignment]

    try:
        if _LANGGRAPH_AVAILABLE:
            app = build_graph()
            out = app.invoke(state)
            return dict(out)

        # Deterministic fallback
        for fn in (
            node_l0_bootstrap,
            node_l1_intake,
            node_l2_profile_extract,
            node_l3_discovery_and_extraction,
            node_l4_identity_rag,
            node_l5_matcher_and_ranker,
            node_l5_evaluator,
        ):
            state.update(fn(state))
            if fn is node_l5_evaluator:
                if route_after_evaluator(state) == "retry_discovery":
                    # bounded retries
                    while True:
                        state["iteration"] = int(state.get("iteration") or 0) + 1
                        state.update(node_l3_discovery_and_extraction(state))
                        state.update(node_l5_matcher_and_ranker(state))
                        state.update(node_l5_evaluator(state))
                        if route_after_evaluator(state) != "retry_discovery":
                            break

        state.update(node_l6_checkpoint(state))
        state.update(node_l7_postprocess(state))
        state.update(node_l8_export(state))
        state.update(node_l9_done(state))
        return dict(state)

    except Exception as e:  # noqa: BLE001
        # Absolute fail-safe: never throw to caller
        run_id = state.get("run_id") or "unknown"
        ev = _fail_safe_evaluation(run_id=run_id, reason=f"Pipeline crashed: {e}")
        state["evaluation"] = ev
        state["phase2_failed"] = True
        state["phase2_error"] = str(e)

        evlogs = list(state.get("evaluation_logs") or [])
        evlogs.append({"ts": _utc_now_iso(), "layer": state.get("current_layer"), "event": "FATAL", "error": str(e)})
        state["evaluation_logs"] = evlogs

        # Emit final snapshot so UI stops spinning
        _emit_snapshot(state, status="error")
        _write_json(Path("outputs/phase2") / f"fatal_{run_id}_{_stamp()}.json", {"error": str(e), "trace": traceback.format_exc()})
        return dict(state)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", dest="resume_path", required=True, help="Path to resume text file")
    p.add_argument("--roles", nargs="+", default=["Solution Architect"], help="Target roles")
    p.add_argument("--location", default="United States", help="Location for discovery")
    p.add_argument("--top-n", dest="top_n", type=int, default=8, help="Top-N jobs to shortlist")
    p.add_argument("--max-iterations", dest="max_iterations", type=int, default=3, help="Max evaluator retry loops")
    p.add_argument("--pass-threshold", dest="pass_threshold", type=float, default=0.7, help="Evaluator pass threshold")
    p.add_argument("--evaluator-model", dest="evaluator_model", default="gemini-1.5-flash", help="Gemini model")
    p.add_argument("--country", default="US", help="Preference: country (e.g., US)")
    p.add_argument("--recency-hours", dest="recency_hours", type=int, default=36, help="Preference: recency hours")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    run_id = uuid.uuid4().hex

    initial: AgentState = {
        "run_id": run_id,
        "evaluation": {"score": 0.0, "reason": "", "action": ""},
        "phase1_results": {
            "resume_path": args.resume_path,
            "roles": args.roles,
            "location": args.location,
            "top_n": int(args.top_n),
            "max_iterations": int(args.max_iterations),
            "pass_threshold": float(args.pass_threshold),
            "evaluator_model": str(args.evaluator_model),
        },
        "preferences": {"country": args.country, "recency_hours": int(args.recency_hours)},
        "messages": [],
        "evaluation_logs": [],
        "iteration": 0,
        "max_iterations": int(args.max_iterations),
        "progress_percent": 0,
        "current_layer": "L0",
        "refinement_feedback": "",
        "phase2_failed": False,
        "phase2_error": "",
    }

    out = run_pipeline(initial)

    ev = out.get("evaluation") or {}
    print(
        json.dumps(
            {"run_id": run_id, "score": ev.get("score"), "action": ev.get("action"), "reason": ev.get("reason")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
