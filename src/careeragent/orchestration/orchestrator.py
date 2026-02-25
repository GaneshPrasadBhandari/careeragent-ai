from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from careeragent.agents.auto_applier_agent_service import AutoApplierAgentService
from careeragent.agents.drafting_agent_service import DraftingAgentService
from careeragent.agents.governance_auditor_service import GovernanceAuditor
from careeragent.agents.l2_intake_evaluator_service import L2IntakeEvaluatorService
from careeragent.agents.l7_analytics_learning_service import CareerCoach, DashboardManager, SelfLearningAgent
from careeragent.agents.matcher_agent_service import MatcherAgentService
from careeragent.agents.memory_manager_service import MemoryManager
from careeragent.agents.parser_agent_service import ParserAgentService
from careeragent.agents.phase2_evaluator_agent_service import Phase2EvaluatorAgentService
from careeragent.agents.ranker_agent_service import RankerAgentService
from careeragent.core.mcp_client import MCPClient
from careeragent.core.settings import Settings
from careeragent.core.state import AgentState, ArtifactRef, PROGRESS_MAP
from careeragent.core.state_store import StateStore
from careeragent.managers.l3_managers import ExtractionManager, GeoFenceManager, LeadScout
from careeragent.orchestration.planner_director import Director, Planner
from careeragent.nlp.skills import compute_jd_alignment
from careeragent.services.notification_service import NotificationService


class Orchestrator:
    """Description: Core orchestrator implementing L0-L9 with Planner-Director and soft-fencing.
    Layer: L2
    """

    def __init__(self, settings: Settings, store: StateStore, mcp: MCPClient) -> None:
        self.s = settings
        self.store = store
        self.mcp = mcp

        self.planner = Planner()
        self.director = Director()

        self.parser = ParserAgentService(settings)
        self.intake_gate = L2IntakeEvaluatorService()

        self.scout = LeadScout(settings)
        self.geo = GeoFenceManager()
        self.extract = ExtractionManager(settings)

        self.matcher = MatcherAgentService(settings)
        self.ranker = RankerAgentService()
        self.eval2 = Phase2EvaluatorAgentService(settings)

        self.drafter = DraftingAgentService(settings)
        self.applier = AutoApplierAgentService(settings)

        self.dashboard_mgr = DashboardManager(settings, mcp)
        self.learner = SelfLearningAgent(settings, mcp)
        self.coach = CareerCoach(settings)
        self.memory = MemoryManager(settings, mcp)
        self.gov = GovernanceAuditor(settings, mcp)
        self.notify = NotificationService(settings=settings)

    # --------------------------
    # High-level entrypoints
    # --------------------------
    def run_phase1_to_hitl(self, state: AgentState, *, resume_filename: str, resume_bytes: bytes) -> AgentState:
        """Run L0-L5 and stop at ranking HITL."""
        state.meta.setdefault("plan_layers", ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"])
        state.status = "running"
        self.store.save(state)

        try:
            # L0
            state.start_step("L0", "SanitizeAgent", "Starting run", PROGRESS_MAP["L0"])  # type: ignore
            state.end_step_ok("L0", "ok")
            self.store.save(state)

            # L1
            state.start_step("L1", "IntakeAgent", "Loading resume + preferences", PROGRESS_MAP["L1"])  # type: ignore
            raw_text = self._extract_text(resume_filename, resume_bytes)
            # save raw resume
            run_dir = Path("outputs/runs") / state.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            resume_raw_path = run_dir / "resume_raw.txt"
            resume_raw_path.write_text(raw_text, encoding="utf-8")
            state.artifacts["resume_raw"] = ArtifactRef(path=str(resume_raw_path), mime="text/plain")
            state.end_step_ok("L1", "ok")
            self.store.save(state)

            # L2 Parser + Planner/Director
            state.start_step("L2", "Parser+Planner", "Extracting profile and building search personas", PROGRESS_MAP["L2"])  # type: ignore
            prof, used_text = self.parser.parse_from_upload(filename=resume_filename, file_bytes=resume_bytes, raw_text=raw_text)
            state.extracted_profile = prof.model_dump(mode="json")

            # persist profile
            profile_path = run_dir / "extracted_profile.json"
            profile_path.write_text(json.dumps(state.extracted_profile, indent=2), encoding="utf-8")
            state.artifacts["extracted_profile"] = ArtifactRef(path=str(profile_path), mime="application/json")

            # L2 evaluator gate
            score, reason, fb = self.intake_gate.evaluate(state)
            self.intake_gate.write_to_state(state, score, reason, fb)

            if score < state.preferences.resume_threshold:
                state.pending_action = "resume_cleanup_optional"
                state.status = "needs_human_approval"
                state.end_step_ok("L2", "needs resume cleanup")
                self.store.save(state)
                return state

            # build personas
            state.search_personas = self.planner.build_personas(state)
            state.active_persona_id = state.search_personas[0].persona_id

            state.end_step_ok("L2", "ok")
            self.store.save(state)

            # L3-L5 loop with soft fencing
            state = self._discovery_match_rank_loop(state)

            # If the loop already paused for a HITL gate (e.g., relax_constraints), respect it.
            if state.status == "needs_human_approval" and state.pending_action:
                self.store.save(state)
                return state

            # Stop at HITL ranking
            # Write shortlist snapshot for dashboards even if we pause here.
            try:
                self.dashboard_mgr.record_shortlist(state, status="awaiting_ranking_approval")
            except Exception:
                pass
            state.pending_action = "review_ranking"
            state.status = "needs_human_approval"
            self.notify.send_alert(message=f"Run {state.run_id}: HITL required at L5 ranking review.")
            self.store.save(state)
            return state

        except Exception as e:
            state.status = "failed"
            state.pending_action = None
            state.end_step_error(state.current_layer or "L0", str(e))  # type: ignore
            state.log_eval(traceback.format_exc())
            self.store.save(state)
            return state

    def run_phase2_after_ranking(self, state: AgentState) -> AgentState:
        """Run L6-L9 after ranking approval."""
        state.status = "running"
        state.pending_action = None
        self.store.save(state)

        try:
            # L6 drafting
            state.start_step("L6", "DraftingAgent", "Generating ATS resume + cover letter", PROGRESS_MAP["L6"])  # type: ignore
            # prevent duplicates
            state.approved_job_urls = self.memory.filter_duplicates(state, state.approved_job_urls)
            draft_meta = self.drafter.generate_for_jobs(state)

            # ATS evaluator gate for auto-apply readiness
            # Entity-based match is a percent now.
            below = [d for d in draft_meta if float(d.get("ats_keyword_match_percent") or 0.0) < 70.0]
            if below:
                state.log_eval(f"[L6 ATS] {len(below)} drafts below 70% entity keyword match; HITL review required")

            state.end_step_ok("L6", f"drafts={len(draft_meta)}")
            self.store.save(state)

            state.pending_action = "review_drafts"
            state.status = "needs_human_approval"
            self.notify.send_alert(message=f"Run {state.run_id}: HITL required at L6 draft review.")
            self.store.save(state)
            return state

        except Exception as e:
            state.status = "failed"
            state.pending_action = None
            state.end_step_error("L6", str(e))
            state.log_eval(traceback.format_exc())
            self.store.save(state)
            return state

    def run_finalize_after_drafts(self, state: AgentState, *, dry_run_apply: bool = True) -> AgentState:
        """Complete L7-L9."""
        state.status = "running"
        state.pending_action = None
        self.store.save(state)

        try:
            # L7 auto-apply + analytics
            state.start_step("L7", "AutoApplier+Analytics", "Applying and recording learning", PROGRESS_MAP["L7"])  # type: ignore
            self.notify.send_alert(message=f"Run {state.run_id}: final submission gate reached before apply.")
            apply_results = self.applier.apply(state, dry_run=dry_run_apply)
            state.meta["apply_results"] = [r.__dict__ for r in apply_results]
            self.dashboard_mgr.record_approved(state)
            self.learner.ingest_failure(state)
            roadmap = self.coach.build_roadmap(state)
            self.coach.save_roadmap(state, roadmap)
            state.end_step_ok("L7", "ok")
            self.store.save(state)

            # L8 memory
            state.start_step("L8", "MemoryManager", "Marking applied and dedupe memory", PROGRESS_MAP["L8"])  # type: ignore
            for url in state.approved_job_urls:
                self.memory.mark_applied(url)
            state.end_step_ok("L8", "ok")
            self.store.save(state)

            # L9 governance
            state.start_step("L9", "GovernanceAuditor", "Final governance summary", PROGRESS_MAP["L9"])  # type: ignore
            self.gov.finalize(state)
            state.end_step_ok("L9", "ok")

            state.status = "completed"
            state.pending_action = None
            self.store.save(state)
            return state

        except Exception as e:
            state.status = "failed"
            state.pending_action = None
            state.end_step_error(state.current_layer or "L7", str(e))  # type: ignore
            state.log_eval(traceback.format_exc())
            self.store.save(state)
            return state

    # --------------------------
    # Internal helpers
    # --------------------------
    def _extract_text(self, filename: str, resume_bytes: bytes) -> str:
        if filename.lower().endswith(".txt"):
            return resume_bytes.decode("utf-8", errors="ignore")
        if filename.lower().endswith(".docx"):
            # parser extracts text itself, but we use it for storage and LLM backfill
            prof, text = self.parser.parse_from_upload(filename=filename, file_bytes=resume_bytes, raw_text=None)
            return text
        if filename.lower().endswith(".pdf"):
            # best-effort
            try:
                from io import BytesIO

                from PyPDF2 import PdfReader

                reader = PdfReader(BytesIO(resume_bytes))
                parts: List[str] = []
                for page in reader.pages[:12]:
                    parts.append(page.extract_text() or "")
                return "\n".join([p.strip() for p in parts if p.strip()])
            except Exception:
                return ""
        return resume_bytes.decode("utf-8", errors="ignore")

    def _persist_json_artifact(self, state: AgentState, key: str, payload: Any) -> None:
        run_dir = Path("outputs/runs") / state.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"{key}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        state.artifacts[key] = ArtifactRef(path=str(path), mime="application/json")

    def _decorate_hitl_scorecard(self, state: AgentState) -> None:
        """Compute UI scorecard fields for HITL.

        Description:
          Adds Interview Probability % and Missing Skills Gap % to each ranked job.
          These are rendered by the dashboard before approvals.

        Layer: L5
        Input: state.ranking + profile skills
        Output: mutated ranking entries + state.meta['hitl_summary']
        """

        prof = state.extracted_profile or {}
        resume_skills = prof.get("skills") or []

        best_prob = 0.0
        best_gap = 100.0

        for job in state.ranking[:60]:
            match_pct = float(job.get("overall_match_percent") or 0.0)

            # Ensure JD alignment exists (L4 should provide it; this is a safety net)
            align_pct = job.get("jd_alignment_percent")
            gap_pct = job.get("missing_skills_gap_percent")
            if align_pct is None or gap_pct is None:
                jd_text = (job.get("full_text_md") or job.get("snippet") or "")
                sc = compute_jd_alignment(jd_text=jd_text, resume_skills=resume_skills)
                align_pct = sc.jd_alignment_percent
                gap_pct = sc.missing_skills_gap_percent
                job["jd_alignment_percent"] = align_pct
                job["missing_skills_gap_percent"] = gap_pct
                job["matched_jd_skills"] = sc.matched_jd_skills[:40]
                job["missing_jd_skills"] = sc.missing_jd_skills[:40]

            align_pct_f = float(align_pct or 0.0)
            gap_pct_f = float(gap_pct or (100.0 - align_pct_f))

            # Interview probability is a calibrated heuristic for UX.
            # It is *not* a claim about any employer decision.
            prob = 0.60 * match_pct + 0.40 * align_pct_f

            # guardrails / penalties for weak alignment
            if align_pct_f < 35:
                prob -= 10
            if match_pct < 50:
                prob -= 6

            prob = max(0.0, min(100.0, prob))
            job["interview_probability_percent"] = round(prob, 2)

            best_prob = max(best_prob, prob)
            best_gap = min(best_gap, gap_pct_f)

        state.meta["hitl_summary"] = {
            "top_interview_probability_percent": round(best_prob, 2),
            "best_missing_skills_gap_percent": round(best_gap, 2),
            "note": "Heuristic scorecard for prioritization (not a guarantee).",
        }

    def _discovery_match_rank_loop(self, state: AgentState) -> AgentState:
        min_viable = 3

        while state.retry_count <= state.max_retries:
            # L3
            state.start_step("L3", "ManagerCluster", "Searching jobs (personas + geo fence + extraction)", PROGRESS_MAP["L3"])  # type: ignore
            raw = self.scout.search(state, limit=max(20, state.preferences.max_jobs))
            kept, rejected = self.geo.filter(state, raw)
            enriched, notes = self.extract.enrich(state, kept, max_jobs=state.preferences.max_jobs)
            state.jobs_raw = enriched
            self._persist_json_artifact(state, "jobs_raw", state.jobs_raw)
            state.end_step_ok("L3", f"jobs_raw={len(enriched)} rejected={len(rejected)}")
            self.store.save(state)

            # L4
            state.start_step("L4", "MatcherAgent", "Scoring matches", PROGRESS_MAP["L4"])  # type: ignore
            state.jobs_scored = self.matcher.score_jobs(state)
            self._persist_json_artifact(state, "jobs_scored", state.jobs_scored)
            state.end_step_ok("L4", f"jobs_scored={len(state.jobs_scored)}")
            self.store.save(state)

            if state.meta.get("l4_recursive_loop_required"):
                state.retry_count += 1
                state.log_eval("[L4] Top interview chance below 0.70; looping back to L3 discovery.")
                state.active_persona_id = self.director.next_persona(state)
                self.director.relax_constraints(state)
                self.store.save(state)
                continue

            # L5
            state.start_step("L5", "Ranker+Evaluator", "Ranking and evaluating", PROGRESS_MAP["L5"])  # type: ignore
            state.ranking = self.ranker.rank(state)
            self._persist_json_artifact(state, "ranking", state.ranking)

            # Build HITL scorecard fields used by the UI (Interview Probability + Missing Skills Gap)
            self._decorate_hitl_scorecard(state)
            self._persist_json_artifact(state, "hitl_scorecard", {"top": state.ranking[:12]})

            # evaluate
            score, reason, action, feedback = self.eval2.evaluate(state)
            self.eval2.write_to_state(state, score, reason, action, feedback)
            self._persist_json_artifact(state, "evaluation", state.evaluation)

            # update best so far if ranking non-empty
            top_score = float(state.ranking[0].get("overall_match_percent") or 0.0) if state.ranking else 0.0
            if state.ranking and top_score > state.best_so_far.score:
                state.best_so_far.score = top_score
                state.best_so_far.ranking_path = state.artifacts["ranking"].path
                state.best_so_far.jobs_scored_path = state.artifacts["jobs_scored"].path
                state.best_so_far.persona_id = state.active_persona_id

            state.end_step_ok("L5", f"ranking={len(state.ranking)} action={action}")
            self.store.save(state)

            # Soft-fencing director decision
            viable = sum(1 for j in state.ranking[:25] if float(j.get("phase2_score") or 0.0) >= 0.55)
            should_retry, why = self.director.soft_fence(state, viable_count=viable, batch_score=score)

            if action == "PROCEED" and not should_retry:
                return state

            # If we have a best-so-far ranking but current loop collapsed, use best-so-far.
            if not state.ranking and state.best_so_far.ranking_path:
                try:
                    state.ranking = json.loads(Path(state.best_so_far.ranking_path).read_text(encoding="utf-8"))
                    state.jobs_scored = json.loads(Path(state.best_so_far.jobs_scored_path or "").read_text(encoding="utf-8")) if state.best_so_far.jobs_scored_path else state.jobs_scored
                    self._persist_json_artifact(state, "ranking", state.ranking)
                    self._persist_json_artifact(state, "jobs_scored", state.jobs_scored)
                    state.log_eval("[Director] Restored best-so-far ranking to avoid wipeout")
                    return state
                except Exception:
                    pass

            # Retry path: if we’re going to relax constraints repeatedly, force a HITL gate.
            # This prevents the UI from looking like it is "stuck" at L5 while refinements churn.
            max_auto = max(1, int(state.preferences.max_refinements))

            if state.retry_count >= max_auto:
                proposal = {
                    "reason": why,
                    "current_persona": state.active_persona_id,
                    "next_persona": self.director.next_persona(state),
                    "current_recency_hours": float(state.preferences.recency_hours),
                    "proposed_recency_hours": float(min(168.0, max(state.preferences.recency_hours, 72.0))),
                }
                state.meta["relax_proposal"] = proposal
                state.pending_action = "relax_constraints"
                state.status = "needs_human_approval"
                state.log_eval(f"[HITL] Approval required to relax constraints: {proposal}")
                self.notify.send_alert(message=f"Run {state.run_id}: HITL needed to relax constraints at L5.")

                # Write shortlist snapshot even though we’re paused.
                try:
                    self.dashboard_mgr.record_shortlist(state, status="awaiting_relax_approval")
                except Exception:
                    pass
                self.store.save(state)
                return state

            # Auto-retry path (bounded)
            state.retry_count += 1
            state.log_eval(f"[Director] retry={state.retry_count} reason={why}")
            state.active_persona_id = self.director.next_persona(state)
            self.director.relax_constraints(state)
            self.store.save(state)

        # Exit after budget: restore best-so-far if exists
        if state.best_so_far.ranking_path:
            state.ranking = json.loads(Path(state.best_so_far.ranking_path).read_text(encoding="utf-8"))
            self._persist_json_artifact(state, "ranking", state.ranking)
        return state
