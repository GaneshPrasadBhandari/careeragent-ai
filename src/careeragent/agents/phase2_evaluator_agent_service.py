from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from careeragent.core.settings import Settings
from careeragent.core.state import AgentState, EvaluationEntry
from careeragent.tools.llm_tools import GeminiClient
from careeragent.tools.web_tools import extract_explicit_location, is_non_us_location
from careeragent.tools.web_tools import TavilyClient


def _detect_posted_hours(text: str) -> Optional[float]:
    """Try to detect 'x hours ago' or 'x days ago' from snippet/header."""
    if not text:
        return None
    m = re.search(r"(\d{1,3})\s*(hour|hours)\s*ago", text, flags=re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d{1,3})\s*(day|days)\s*ago", text, flags=re.I)
    if m:
        return float(m.group(1)) * 24.0
    return None


class Phase2EvaluatorAgentService:
    """Description: L5_Evaluator with soft-fencing + CRAG.
    Layer: L5
    Input: extracted_profile + ranking
    Output: state.evaluation + routing action
    """

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.gemini = GeminiClient(settings)
        self.tavily = TavilyClient(settings)

    def evaluate(self, state: AgentState) -> Tuple[float, str, str, str]:
        prefs = state.preferences
        ranking = list(state.ranking)

        if not ranking:
            score = 0.0
            reason = "no ranking to evaluate"
            action = "RETRY_SEARCH"
            feedback = "Widen search: broaden to ATS-preferred and 7 days; keep US intent."
            return score, reason, action, feedback

        accepted: List[Dict[str, Any]] = []
        rejected_reasons: List[str] = []

        for job in ranking[: min(len(ranking), 25)]:
            url = str(job.get("url") or job.get("job_id") or "")
            snippet = str(job.get("snippet") or "")
            title = str(job.get("title") or "")

            # Geo-Fence: reject only if explicit location metadata is non-US
            loc = job.get("location_hint") or extract_explicit_location(snippet, title, "")
            if prefs.country.upper() == "US" and loc and is_non_us_location(loc):
                job["phase2_score"] = 0.0
                job["phase2_reason"] = f"Rejected: explicit non-US location '{loc}'"
                rejected_reasons.append(job["phase2_reason"])
                continue

            # Recency: only hard reject if explicitly detected
            hours = _detect_posted_hours(snippet)
            if hours is not None and hours > float(prefs.recency_hours):
                job["phase2_score"] = 0.0
                job["phase2_reason"] = f"Rejected: posted {hours:.0f}h ago (> {prefs.recency_hours}h)"
                rejected_reasons.append(job["phase2_reason"])
                continue

            # Match quality: use Gemini if available, else soft deterministic score
            base = float(job.get("overall_match_percent") or 0.0) / 100.0
            llm_score = None
            llm_reason = None
            if self.s.GEMINI_API_KEY:
                llm_score, llm_reason = self._gemini_judge(state, job)
            score = float(llm_score if llm_score is not None else base)

            # Soft fencing: do NOT zero out unless hard violation
            job["phase2_score"] = round(score, 3)
            job["phase2_reason"] = llm_reason or f"Score from matcher: {base:.2f}"

            # Accept if reasonable
            if score >= 0.55:
                accepted.append(job)
            else:
                rejected_reasons.append(f"Low fit ({score:.2f}): {url}")

        # Batch decision
        accepted_rate = len(accepted) / max(1, min(len(ranking), 25))
        batch_score = round(accepted_rate, 3)

        if len(accepted) >= 3:
            return batch_score, f"Accepted {len(accepted)} jobs", "PROCEED", "Proceed to HITL ranking approval."

        # Soft-fence: if some accepted but not enough, do NOT collapse to 0; proceed but advise relax.
        if len(accepted) > 0:
            feedback = "Low viable count. Suggest widen recency to 7 days and ATS-preferred remote roles." \
                       " Exclude low-signal aggregator boards and widen ATS-preferred sources."
            return batch_score, f"Only {len(accepted)} viable jobs", "PROCEED", feedback

        # None accepted: retry with strategy shift (not hard fail)
        feedback = (
            "No viable jobs after strict filtering. Shift strategy: ATS-preferred (not ATS-only), widen to 7 days, "
            "keep role-specific keywords, and suppress low-signal aggregator board noise."
        )
        return 0.2, "All jobs rejected (soft-fence) — shifting strategy", "RETRY_SEARCH", feedback

    def _gemini_judge(self, state: AgentState, job: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
        prof = state.extracted_profile
        prefs = state.preferences
        jd = (job.get("full_text_md") or job.get("snippet") or "")[:9000]

        prompt = (
            "You are an evaluator for job-to-candidate fit.\n"
            "Rules:\n"
            "- Do NOT reject just because the company has global offices; only location metadata matters.\n"
            "- If location is unknown, do not assume.\n"
            "- Return STRICT JSON: {score: float 0-1, reason: string, dealbreakers: [string], refinement_feedback: string}.\n"
            "- Score should reflect role alignment with the candidate profile and preferences.\n\n"
            f"PREFERENCES: country={prefs.country}, recency_hours={prefs.recency_hours}, remote={prefs.remote}, wfo_ok={prefs.wfo_ok}\n"
            f"CANDIDATE_PROFILE_JSON: {prof}\n\n"
            f"JOB_TITLE: {job.get('title')}\nJOB_URL: {job.get('url')}\nJOB_TEXT:\n{jd}\n"
        )
        j = self.gemini.generate_json(prompt, temperature=0.15, max_tokens=800)
        if not isinstance(j, dict):
            return None, None
        try:
            score = float(j.get("score"))
        except Exception:
            score = None
        reason = str(j.get("reason") or "").strip() or None

        # If Gemini suggests missing company info, do CRAG once
        if reason and "unknown" in reason.lower() and self.s.TAVILY_API_KEY:
            cr = self._crag_company(job)
            if cr:
                prompt2 = prompt + "\n\nCOMPANY_CONTEXT_FROM_WEB:\n" + cr
                j2 = self.gemini.generate_json(prompt2, temperature=0.15, max_tokens=800)
                if isinstance(j2, dict) and j2.get("score") is not None:
                    try:
                        score = float(j2.get("score"))
                        reason = str(j2.get("reason") or reason)
                    except Exception:
                        pass

        return score, reason

    def _crag_company(self, job: Dict[str, Any]) -> Optional[str]:
        title = str(job.get("title") or "")
        url = str(job.get("url") or "")
        # heuristically extract company name from title like "X - Solution Architect"
        company = title.split("-")[0].strip() if "-" in title else ""
        q = f"{company} company overview remote policy" if company else f"company info {url}"
        res = self.tavily.search(q, max_results=3)
        if not res:
            return None
        lines = []
        for r in res:
            lines.append(f"- {r.get('title')}: {r.get('url')}\n  {str(r.get('snippet') or '')[:240]}")
        return "\n".join(lines)

    def write_to_state(self, state: AgentState, score: float, reason: str, action: str, feedback: str) -> None:
        state.evaluation = {"score": float(score), "reason": reason, "action": action, "refinement_feedback": feedback}
        state.refinement_feedback = feedback if action == "RETRY_SEARCH" else state.refinement_feedback

        entry = EvaluationEntry(
            layer_id="L5",
            target_id="ranking_batch",
            evaluation_score=float(score),
            threshold=float(state.preferences.discovery_threshold),
            decision="pass" if action == "PROCEED" else "retry",
            feedback=[reason, feedback][:6],
        )
        state.evaluations.append(entry)
        state.log_eval(f"[L5_Evaluator] action={action} score={score:.2f} reason={reason}")
