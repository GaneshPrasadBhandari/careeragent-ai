from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from careeragent.core.mcp_client import MCPClient
from careeragent.core.settings import Settings
from careeragent.core.state import AgentState, Preferences
from careeragent.core.state_store import StateStore
from careeragent.orchestration.orchestrator import Orchestrator


class RunManagerService:
    """Description: Creates and manages runs and background execution.
    Layer: L8
    Input: API requests
    Output: state transitions
    """

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.mcp = MCPClient(settings)
        self.store = StateStore(settings, self.mcp)
        self.orch = Orchestrator(settings, self.store, self.mcp)

        self._threads: Dict[str, threading.Thread] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._resume_blobs: Dict[str, Dict[str, Any]] = {}

    def _lock(self, run_id: str) -> threading.Lock:
        if run_id not in self._locks:
            self._locks[run_id] = threading.Lock()
        return self._locks[run_id]

    def create_run(self, *, resume_filename: str, resume_bytes: bytes, preferences: Dict[str, Any]) -> Dict[str, Any]:
        run_id = f"run_{uuid.uuid4().hex[:10]}"

        prefs = Preferences.model_validate(preferences or {})
        state = AgentState(run_id=run_id, status="queued", preferences=prefs)
        state.meta["plan_layers"] = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]

        # Save resume bytes in memory for this run (simple)
        self._resume_blobs[run_id] = {"filename": resume_filename, "bytes": resume_bytes}

        self.store.save(state)
        self._spawn_phase1(run_id)

        return {"run_id": run_id, "status": "running"}

    def get_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        st = self.store.load(run_id)
        if not st:
            return None
        return st.model_dump(mode="json")

    def handle_action(self, run_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        action_type = body.get("action_type")
        payload = body.get("payload") or {}

        st = self.store.load(run_id)
        if not st:
            return {"error": "run_not_found"}

        with self._lock(run_id):
            if action_type == "execute_layer":
                layer = str(payload.get("layer") or "")
                if layer in ("L0", "L1", "L2", "L3", "L4", "L5"):
                    self._spawn_phase1(run_id)
                    return {"ok": True, "message": f"restarted phase1 (requested {layer})"}
                if layer == "L6":
                    self._spawn_phase2(run_id)
                    return {"ok": True, "message": "started phase2"}
                return {"error": "invalid_layer"}

            if action_type == "approve_ranking":
                urls = payload.get("selected_job_urls") or []
                st.approved_job_urls = [str(u) for u in urls if u]
                st.status = "running"
                st.pending_action = None
                self.store.save(st)
                self._spawn_phase2(run_id)
                return {"ok": True}

            if action_type == "reject_ranking":
                reason = str(payload.get("reason") or "")
                st.refinement_feedback = (st.refinement_feedback or "") + "\n" + reason
                st.retry_count += 1
                st.status = "running"
                st.pending_action = None
                self.store.save(st)
                self._spawn_phase1(run_id)
                return {"ok": True}

            if action_type == "approve_drafts":
                st.status = "running"
                st.pending_action = None
                self.store.save(st)
                self._spawn_finalize(run_id)
                return {"ok": True}

            if action_type == "reject_drafts":
                st.status = "needs_human_approval"
                st.pending_action = "review_ranking"
                self.store.save(st)
                return {"ok": True}

            if action_type == "resume_cleanup_submit":
                txt = str(payload.get("resume_text") or "")
                # overwrite resume_raw
                run_dir = self.store.run_dir(run_id)
                (run_dir / "resume_raw.txt").write_text(txt, encoding="utf-8")
                st.status = "running"
                st.pending_action = None
                self.store.save(st)
                self._spawn_phase1(run_id)
                return {"ok": True}

            if action_type == "relax_constraints":
                # widen recency to 7d
                st.preferences.recency_hours = max(168.0, st.preferences.recency_hours)
                st.retry_count += 1
                st.status = "running"
                st.pending_action = None
                self.store.save(st)
                self._spawn_phase1(run_id)
                return {"ok": True}

        return {"error": "unknown_action"}

    # --------------------
    # Background execution
    # --------------------
    def _spawn_phase1(self, run_id: str) -> None:
        if run_id in self._threads and self._threads[run_id].is_alive():
            return

        def worker() -> None:
            st = self.store.load(run_id)
            if not st:
                return
            blob = self._resume_blobs.get(run_id)
            if not blob:
                st.status = "failed"
                st.pending_action = None
                st.log_eval("resume blob missing")
                self.store.save(st)
                return
            out = self.orch.run_phase1_to_hitl(st, resume_filename=blob["filename"], resume_bytes=blob["bytes"]) 
            self.store.save(out)

        t = threading.Thread(target=worker, daemon=True)
        self._threads[run_id] = t
        t.start()

    def _spawn_phase2(self, run_id: str) -> None:
        if run_id in self._threads and self._threads[run_id].is_alive():
            return

        def worker() -> None:
            st = self.store.load(run_id)
            if not st:
                return
            out = self.orch.run_phase2_after_ranking(st)
            self.store.save(out)

        t = threading.Thread(target=worker, daemon=True)
        self._threads[run_id] = t
        t.start()

    def _spawn_finalize(self, run_id: str) -> None:
        if run_id in self._threads and self._threads[run_id].is_alive():
            return

        def worker() -> None:
            st = self.store.load(run_id)
            if not st:
                return
            out = self.orch.run_finalize_after_drafts(st, dry_run_apply=True)
            self.store.save(out)

        t = threading.Thread(target=worker, daemon=True)
        self._threads[run_id] = t
        t.start()
