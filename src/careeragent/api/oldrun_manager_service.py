# src/careeragent/api/run_manager_service.py
from __future__ import annotations

import asyncio
import concurrent.futures as cf
import copy
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from careeragent.services.db_service import SqliteStateStore
from careeragent.langgraph.runtime_nodes import run_single_layer
from careeragent.langgraph.hitl_flows import approve_ranking_flow, approve_drafts_flow


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def artifacts_root() -> Path:
    return Path("src/careeragent/artifacts").resolve()


class RunManagerService:
    """
    Description: Background runner + HITL action router.
    Layer: L8
    """

    def __init__(self) -> None:
        self._store = SqliteStateStore()

    def _runs_dir(self, run_id: str) -> Path:
        d = artifacts_root() / "runs" / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_state(self, *, run_id: str, state: Dict[str, Any]) -> None:
        state.setdefault("meta", {})
        state["meta"]["heartbeat_utc"] = utc_now()
        self._store.upsert_state(run_id=run_id, status=str(state.get("status", "unknown")), state=state, updated_at_utc=utc_now())

    def get_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get_state(run_id=run_id)

    def create_run(self, *, resume_filename: str, resume_text: str, resume_bytes: bytes, preferences: Dict[str, Any]) -> Dict[str, Any]:
        run_id = uuid4().hex
        run_dir = self._runs_dir(run_id)
        (run_dir / "resume_upload.bin").write_bytes(resume_bytes)
        (run_dir / "resume_raw.txt").write_text(resume_text, encoding="utf-8")

        state: Dict[str, Any] = {
            "run_id": run_id,
            "status": "running",
            "pending_action": None,
            "preferences": preferences,
            "resume_filename": resume_filename,
            "resume_text": resume_text,
            "profile": {},
            "jobs_raw": [],
            "jobs_scored": [],
            "ranking": [],
            "drafts": {},
            "bridge_docs": {},
            "meta": {
                "created_at_utc": utc_now(),
                "heartbeat_utc": utc_now(),
                "last_layer": None,
                "plan_layers": ["L0", "L2", "L3", "L4", "L5"],
                "approved_job_urls": [],
            },
            "steps": [],
            "live_feed": [{"layer": "L1", "agent": "API", "message": "Run created. Starting background pipeline…"}],
            "attempts": [],
            "evaluations": [],
            "artifacts": {
                "resume_raw": {"path": str(run_dir / "resume_raw.txt"), "content_type": "text/plain"},
                "resume_upload": {"path": str(run_dir / "resume_upload.bin"), "content_type": "application/octet-stream"},
            },
        }

        self.save_state(run_id=run_id, state=state)
        return state

    def start_background(self, run_id: str) -> None:
        t = threading.Thread(target=self._bg, args=(run_id,), daemon=True)
        t.start()

    def _call_layer(self, state: Dict[str, Any], layer: str) -> Dict[str, Any]:
        st_copy = copy.deepcopy(state)
        return asyncio.run(run_single_layer(st_copy, layer))

    def _bg(self, run_id: str) -> None:
        state = self.get_state(run_id)
        if not state:
            return

        plan = [("L0", 10), ("L2", 20), ("L3", 45), ("L4", 120), ("L5", 15)]

        for layer, tmo in plan:
            if state.get("status") in ("blocked", "needs_human_approval", "failed", "completed"):
                self.save_state(run_id=run_id, state=state)
                return

            state.setdefault("meta", {})
            state["meta"]["last_layer"] = layer
            state.setdefault("steps", []).append({"layer_id": layer, "status": "running", "started_at_utc": utc_now()})
            state.setdefault("live_feed", []).append({"layer": layer, "agent": "Orchestrator", "message": f"Running {layer}…"})
            self.save_state(run_id=run_id, state=state)

            with cf.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(self._call_layer, state, layer)
                try:
                    state = fut.result(timeout=tmo)
                    step_status = "ok"
                except cf.TimeoutError:
                    state["status"] = "needs_human_approval"
                    state["pending_action"] = f"timeout_{layer.lower()}"
                    state.setdefault("live_feed", []).append({"layer": layer, "agent": "TimeoutGuard", "message": f"{layer} timed out after {tmo}s"})
                    step_status = "failed"
                except Exception as e:
                    state["status"] = "failed"
                    state["pending_action"] = f"error_{layer.lower()}"
                    state.setdefault("live_feed", []).append({"layer": layer, "agent": "CrashGuard", "message": f"{layer} crashed: {e}"})
                    step_status = "failed"

            state["steps"][-1]["status"] = step_status
            state["steps"][-1]["finished_at_utc"] = utc_now()
            self.save_state(run_id=run_id, state=state)

            if state.get("pending_action") == "review_ranking":
                state["status"] = "needs_human_approval"
                self.save_state(run_id=run_id, state=state)
                return

        if state.get("status") == "running":
            state["status"] = "needs_human_approval"
            state["pending_action"] = "review_ranking"
            state.setdefault("live_feed", []).append({"layer": "L5", "agent": "Orchestrator", "message": "Ranking ready for review."})
            self.save_state(run_id=run_id, state=state)

    async def handle_action(self, *, run_id: str, action_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = self.get_state(run_id)
        if not state:
            raise ValueError("run_id not found")

        if action_type == "approve_ranking":
            selected = payload.get("selected_job_urls") or []
            if isinstance(selected, list):
                state.setdefault("meta", {})["approved_job_urls"] = [str(u).strip() for u in selected if str(u).strip()]
            state = await asyncio.to_thread(lambda: asyncio.run(approve_ranking_flow(copy.deepcopy(state))))
            self.save_state(run_id=run_id, state=state)
            return state

        if action_type == "reject_ranking":
            reason = str(payload.get("reason", "")).strip()
            state.setdefault("meta", {}).setdefault("ranking_reject_reasons", []).append(reason or "no_reason")
            state["status"] = "running"
            state["pending_action"] = None
            state.setdefault("live_feed", []).append({"layer": "L5", "agent": "HITL", "message": f"Ranking rejected. Reason: {reason[:140]}"})
            self.save_state(run_id=run_id, state=state)
            self.start_background(run_id)
            return state

        if action_type == "approve_drafts":
            state = await asyncio.to_thread(lambda: asyncio.run(approve_drafts_flow(copy.deepcopy(state))))
            self.save_state(run_id=run_id, state=state)
            return state

        if action_type == "reject_drafts":
            reason = str(payload.get("reason", "")).strip()
            state.setdefault("meta", {}).setdefault("draft_reject_reasons", []).append(reason or "no_reason")
            state["status"] = "needs_human_approval"
            state["pending_action"] = "review_ranking"
            state.setdefault("live_feed", []).append({"layer": "L6", "agent": "HITL", "message": f"Drafts rejected. Reason: {reason[:140]}"})
            self.save_state(run_id=run_id, state=state)
            return state

        if action_type == "execute_layer":
            layer = str(payload.get("layer", "")).upper()
            state = await asyncio.to_thread(lambda: self._call_layer(state, layer))
            self.save_state(run_id=run_id, state=state)
            return state

        # Optional resume cleanup submission (if your backend supports it)
        if action_type == "resume_cleanup_submit":
            new_text = str(payload.get("resume_text", "")).strip()
            if new_text:
                state["resume_text"] = new_text
                state["status"] = "running"
                state["pending_action"] = None
                state.setdefault("live_feed", []).append({"layer": "L2", "agent": "HITL", "message": "Resume updated by user. Re-running pipeline."})
                self.save_state(run_id=run_id, state=state)
                self.start_background(run_id)
            return state

        state.setdefault("live_feed", []).append({"layer": "L5", "agent": "HITL", "message": f"Unhandled action_type={action_type}"})
        self.save_state(run_id=run_id, state=state)
        return state