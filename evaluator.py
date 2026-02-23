"""Phase 2 Evaluator (L5_Evaluator) — fixed for loop safety + transparency.

Key changes vs your stuck build:
  - Always returns `refinement_feedback` on RETRY_SEARCH.
  - Enforces dynamic constraints from `preferences` (country, recency_hours) using Gemini-1.5-Flash.
    *If a job violates constraints → job score is forced to 0 with logged reason.*
  - If 100% of the batch is rejected → generates a strategy-shift feedback (not the same retry).
  - Streams "thinking" to Engineer View via `evaluation_logs` (per-job + overall decision).
  - Corrective RAG (CRAG) with Tavily for missing company context, then re-evaluates.

Layer: L5
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx


# -----------------------------------------------------------------------------
# LangSmith tracing (best-effort)
# -----------------------------------------------------------------------------
try:
    from langsmith import traceable  # type: ignore
except Exception:  # pragma: no cover

    def traceable(*_args: Any, **_kwargs: Any):  # type: ignore
        def _decorator(fn):
            return fn

        return _decorator


JSON_RE = re.compile(r"\{[\s\S]*\}")


_RE_POSTED_HOURS = re.compile(r"\b(?:posted|updated)\s*(?:\:|\-)?\s*(\d{1,3})\s*(?:h|hr|hrs|hour|hours)\s*ago\b", re.IGNORECASE)
_RE_POSTED_DAYS = re.compile(r"\b(?:posted|updated)\s*(?:\:|\-)?\s*(\d{1,3})\s*(?:d|day|days)\s*ago\b", re.IGNORECASE)
_RE_BARE_DAYS = re.compile(r"\b(\d{1,3})\s*(?:d|day|days)\s*ago\b", re.IGNORECASE)
_RE_BARE_HOURS = re.compile(r"\b(\d{1,3})\s*(?:h|hr|hrs|hour|hours)\s*ago\b", re.IGNORECASE)


def _recency_violation(job_text: str, recency_hours: int) -> str | None:
    """Return a reason string if the posting is clearly older than recency_hours."""
    if not job_text or not recency_hours:
        return None

    txt = job_text[:4000]

    m = _RE_POSTED_HOURS.search(txt) or _RE_BARE_HOURS.search(txt)
    if m:
        try:
            hrs = int(m.group(1))
            if hrs > recency_hours:
                return f"Job appears older than recency constraint ({hrs}h > {recency_hours}h)."
        except Exception:
            return None

    m = _RE_POSTED_DAYS.search(txt) or _RE_BARE_DAYS.search(txt)
    if m:
        try:
            days = int(m.group(1))
            hrs = days * 24
            if hrs > recency_hours:
                return f"Job appears older than recency constraint ({days}d > {max(1, int(recency_hours/24))}d)."
        except Exception:
            return None

    return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _safe_extract_json(text: str) -> dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass
    m = JSON_RE.search(text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _get_gemini_key() -> str:
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY"):
        v = os.getenv(k)
        if v and v.strip():
            return v.strip()
    raise RuntimeError("Gemini API key missing. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env")


def _gemini_generate_json(
    *,
    prompt: str,
    model: str = "gemini-1.5-flash",
    temperature: float = 0.2,
    timeout_s: int = 45,
) -> dict[str, Any]:
    """Call Gemini and return parsed JSON.

    Tries google-generativeai SDK first, falls back to REST.
    """
    api_key = _get_gemini_key()

    # SDK path
    try:  # pragma: no cover
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(model)
        resp = m.generate_content(prompt, generation_config={"temperature": temperature})
        txt = "".join([p.text for p in (resp.candidates[0].content.parts or []) if getattr(p, "text", None)])  # type: ignore
        return _safe_extract_json(txt)
    except Exception:
        pass

    # REST fallback
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, params={"key": api_key}, json=payload)
        r.raise_for_status()
        data = r.json()
        try:
            txt = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            txt = json.dumps(data)
        return _safe_extract_json(txt)


def _tavily_search(*, query: str, max_results: int = 5, timeout_s: int = 20) -> dict[str, Any]:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return {"status": "degraded", "results": [], "errors": ["TAVILY_API_KEY not configured"]}
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post("https://api.tavily.com/search", json={"api_key": key, "query": query, "max_results": max_results})
        r.raise_for_status()
        data = r.json()
        return {"status": "ok", "query": query, "results": data.get("results") or []}


def _brief_from_tavily(results: list[dict[str, Any]], max_chars: int = 1600) -> str:
    bullets: list[str] = []
    for r in results[:6]:
        title = str(r.get("title") or "").strip()
        url = str(r.get("url") or "").strip()
        content = str(r.get("content") or "").strip()
        if not (title or content):
            continue
        bullets.append(f"- {title} ({url}) — {content}")
    brief = "\n".join(bullets).strip()
    return brief[:max_chars]


@dataclass
class EvaluatorDecision:
    score: float
    reason: str
    action: str
    refinement_feedback: str
    meta: dict[str, Any]
    evaluation_logs: list[dict[str, Any]]


class EvaluatorAgent:
    """Gemini-based evaluator with dynamic constraints and CRAG."""

    def __init__(
        self,
        *,
        pass_threshold: float = 0.7,
        model: str = "gemini-1.5-flash",
        max_jobs_to_judge: int = 6,
    ) -> None:
        self.pass_threshold = float(pass_threshold)
        self.model = model
        self.max_jobs_to_judge = int(max_jobs_to_judge)

    def _pick_jobs(self, ranking: dict[str, Any]) -> list[dict[str, Any]]:
        items = ranking.get("items") or ranking.get("ranked") or []
        if not isinstance(items, list):
            return []
        return items[: max(1, self.max_jobs_to_judge)]

    def _thin_match(self, job_item: dict[str, Any]) -> bool:
        matched = job_item.get("matched_skills") or job_item.get("overlap_skills") or []
        missing = job_item.get("missing_skills") or []
        try:
            m = len(matched)
            req = m + len(missing)
            ratio = (m / req) if req else 0.0
        except Exception:
            return True
        return (m < 4) or (ratio < 0.30)

    def _build_prompt(
        self,
        *,
        profile: dict[str, Any],
        job: dict[str, Any],
        preferences: dict[str, Any],
        company_context: str | None = None,
    ) -> str:
        """Prompt that forces constraint validation."""

        pref_country = str(preferences.get("country") or "").strip()
        pref_recency = int(preferences.get("recency_hours") or 0)

        profile_skills = profile.get("skills") or []
        profile_titles = profile.get("titles") or []
        profile_domains = profile.get("domains") or []

        # Job metadata (from job_post artifact if available)
        job_title = job.get("title") or job.get("job_title") or ""
        company = job.get("company") or ""
        location = job.get("location") or ""
        remote = job.get("remote")
        url = job.get("url") or ""

        job_text = job.get("job_text") or ""
        matched = job.get("matched_skills") or job.get("overlap_skills") or []
        missing = job.get("missing_skills") or []

        ctx = company_context or ""
        if ctx:
            ctx = f"\n\nCompany context (CRAG, live sources):\n{ctx}\n"

        return (
            "You are a strict hiring evaluator for an AI/ML candidate.\n"
            "Your job: score candidate-to-job fit AND validate constraints from preferences.\n\n"
            "RULES:\n"
            "- NEVER invent facts. If unknown, say unknown.\n"
            "- Constraint validation is mandatory.\n"
            "- If the job violates preferences (country/recency), set score=0 and action=RETRY_SEARCH with reason.\n"
            "- If matched_skills is thin OR missing_skills contains deal-breakers, action=RETRY_SEARCH.\n"
            "- Return ONLY valid JSON matching schema.\n\n"
            "JSON schema to return:\n"
            "{\n"
            "  \"job_score\": 0.0,                 // float 0..1\n"
            "  \"job_reason\": \"...\",             // specific\n"
            "  \"constraint_ok\": true,\n"
            "  \"constraint_reason\": \"...\",       // if constraint_ok=false\n"
            "  \"deal_breakers\": [\"...\"],\n"
            "  \"need_company_context\": false,\n"
            "  \"company_name\": \"\",\n"
            "  \"query_refinement\": \"\"            // next search improvements with negative operators\n"
            "}\n\n"
            f"Preferences (dynamic):\n- country: {pref_country}\n- recency_hours: {pref_recency}\n\n"
            "Candidate profile (evidence):\n"
            f"- Titles: {profile_titles}\n"
            f"- Skills: {profile_skills}\n"
            f"- Domains: {profile_domains}\n\n"
            "Job metadata:\n"
            f"- title: {job_title}\n- company: {company}\n- location: {location}\n- remote: {remote}\n- url: {url}\n\n"
            "Job signals from matcher:\n"
            f"- matched_skills: {matched}\n"
            f"- missing_skills: {missing}\n\n"
            f"Job description text:\n{job_text[:9000]}\n"
            f"{ctx}"
            "\nEvaluate now." 
        )

    def _make_strategy_shift_feedback(self, preferences: dict[str, Any]) -> str:
        """Used when 100% of batch is rejected."""
        country = str(preferences.get("country") or "US")
        rec = int(preferences.get("recency_hours") or 36)
        # Keep it actionable and query-ready.
        return (
            f"Shift strategy: enforce {country}-only in query; ensure jobs posted within last {rec} hours; "
            "add explicit keywords \"Solution Architecture\" OR \"Solution Architect\"; "
            "exclude non-target locations using negative operators like -India -Bangalore; "
            "increase breadth (more sources/results) and include \"remote\" if acceptable." 
        )

    @traceable(name="careeragent-ai-phase2.evaluator")
    def evaluate(self, *, profile: dict[str, Any], ranking: dict[str, Any], preferences: dict[str, Any]) -> EvaluatorDecision:
        jobs = self._pick_jobs(ranking)
        logs: list[dict[str, Any]] = []

        if not jobs:
            fb = self._make_strategy_shift_feedback(preferences)
            logs.append({"ts": _utc_now_iso(), "layer": "L5_Evaluator", "event": "EMPTY_BATCH", "reason": "No ranked jobs"})
            return EvaluatorDecision(
                score=0.0,
                reason="No ranked jobs available to evaluate.",
                action="RETRY_SEARCH",
                refinement_feedback=fb,
                meta={"status": "empty_ranking", "ts": _utc_now_iso()},
                evaluation_logs=logs,
            )

        # Stage A: deterministic guard for obviously weak matches
        thin_flags = [self._thin_match(j) for j in jobs]
        if all(thin_flags):
            fb = "Tighten query: include must-have skills/titles; exclude irrelevant locations with -operators; increase seniority match." 
            logs.append({"ts": _utc_now_iso(), "layer": "L5_Evaluator", "event": "THIN_MATCH", "checked": len(jobs)})
            return EvaluatorDecision(
                score=0.25,
                reason="Top matches are too weak (thin matched_skills vs missing_skills).",
                action="RETRY_SEARCH",
                refinement_feedback=fb,
                meta={"status": "thin_match", "checked": len(jobs), "ts": _utc_now_iso()},
                evaluation_logs=logs,
            )

        per_job: list[dict[str, Any]] = []
        best_idx = -1
        best_score = -1.0

        # Judge each job with Gemini
        for idx, j in enumerate(jobs):
            # Deterministic constraint guard: if text clearly violates recency, reject without LLM.
            pref_rec = int(preferences.get("recency_hours") or 0)
            viol = _recency_violation(str(j.get("job_text") or ""), pref_rec)
            if viol:
                out = {
                    "job_score": 0.0,
                    "job_reason": viol,
                    "constraint_ok": False,
                    "constraint_reason": viol,
                    "deal_breakers": [],
                    "need_company_context": False,
                    "company_name": str(j.get("company") or ""),
                    "query_refinement": f"Ensure jobs are posted within the last {pref_rec} hours.",
                }
            else:
                prompt = self._build_prompt(profile=profile, job=j, preferences=preferences)
                out = {}
                try:
                    out = _gemini_generate_json(prompt=prompt, model=self.model)
                except Exception as e:  # noqa: BLE001
                    # degrade: keep job score low, still log and continue
                    out = {
                        "job_score": 0.0,
                        "job_reason": f"Gemini call failed: {e}",
                        "constraint_ok": False,
                        "constraint_reason": "Unable to validate constraints due to model failure",
                        "deal_breakers": [],
                        "need_company_context": False,
                        "company_name": str(j.get("company") or ""),
                        "query_refinement": "",
                    }

            # Normalize output
            try:
                job_score = float(out.get("job_score", out.get("score", 0.0)))
            except Exception:
                job_score = 0.0
            job_score = max(0.0, min(1.0, job_score))

            constraint_ok = bool(out.get("constraint_ok", True))
            constraint_reason = str(out.get("constraint_reason") or "").strip()

            # If constraint mismatch, force score to 0 (per requirement)
            if not constraint_ok:
                job_score = 0.0

            deal_breakers = out.get("deal_breakers") or []
            if isinstance(deal_breakers, list) and deal_breakers:
                # treat blockers as failure
                job_score = min(job_score, 0.25)

            # CRAG trigger when missing company context
            need_ctx = bool(out.get("need_company_context"))
            company_name = str(out.get("company_name") or j.get("company") or "").strip()
            if need_ctx and company_name:
                queries = [
                    f"{company_name} company overview products services",
                    f"{company_name} remote work policy hybrid onsite",
                    f"{company_name} data science machine learning",
                ]
                results: list[dict[str, Any]] = []
                for q in queries[:3]:
                    try:
                        res = _tavily_search(query=q, max_results=4)
                        results.extend(res.get("results") or [])
                    except Exception:
                        continue

                brief = _brief_from_tavily(results)
                if brief:
                    prompt2 = self._build_prompt(profile=profile, job=j, preferences=preferences, company_context=brief)
                    try:
                        out2 = _gemini_generate_json(prompt=prompt2, model=self.model)
                        out2["_crag"] = {"queries": queries[:3], "results_count": len(results)}
                        out = out2
                        try:
                            job_score2 = float(out.get("job_score", out.get("score", 0.0)))
                        except Exception:
                            job_score2 = job_score
                        job_score = max(0.0, min(1.0, job_score2))
                        constraint_ok = bool(out.get("constraint_ok", constraint_ok))
                        if not constraint_ok:
                            job_score = 0.0
                    except Exception:
                        pass

            # Store job decision
            job_entry = {
                "idx": idx,
                "job_path": j.get("job_path"),
                "url": j.get("url"),
                "title": j.get("title"),
                "company": j.get("company"),
                "location": j.get("location"),
                "remote": j.get("remote"),
                "job_score": round(job_score, 4),
                "constraint_ok": bool(out.get("constraint_ok", constraint_ok)),
                "constraint_reason": str(out.get("constraint_reason") or constraint_reason),
                "job_reason": str(out.get("job_reason") or out.get("reason") or ""),
                "deal_breakers": out.get("deal_breakers") or [],
                "query_refinement": str(out.get("query_refinement") or "").strip(),
                "need_company_context": bool(out.get("need_company_context")),
            }
            per_job.append(job_entry)

            logs.append(
                {
                    "ts": _utc_now_iso(),
                    "layer": "L5_Evaluator",
                    "event": "JOB_EVAL",
                    "job_idx": idx,
                    "job_score": job_entry["job_score"],
                    "constraint_ok": job_entry["constraint_ok"],
                    "reason": job_entry["job_reason"] or job_entry["constraint_reason"],
                    "url": job_entry["url"],
                }
            )

            if job_score > best_score:
                best_score = job_score
                best_idx = idx

        # Aggregate decision
        best = per_job[best_idx] if best_idx >= 0 else None
        pass_threshold = self.pass_threshold

        any_pass = any(float(j.get("job_score") or 0.0) >= pass_threshold for j in per_job)
        all_rejected = all(float(j.get("job_score") or 0.0) <= 0.0 for j in per_job)

        if any_pass and best is not None:
            reason = best.get("job_reason") or "Best job meets constraints and fit threshold."
            logs.append({"ts": _utc_now_iso(), "layer": "L5_Evaluator", "event": "DECISION", "action": "PROCEED", "score": best_score, "reason": reason})
            return EvaluatorDecision(
                score=float(best_score),
                reason=reason,
                action="PROCEED",
                refinement_feedback="",
                meta={
                    "model": self.model,
                    "threshold": pass_threshold,
                    "per_job": per_job,
                    "best": best,
                    "ts": _utc_now_iso(),
                },
                evaluation_logs=logs,
            )

        # RETRY_SEARCH
        # Pick the most informative query_refinement among jobs
        refinements = [str(j.get("query_refinement") or "").strip() for j in per_job if str(j.get("query_refinement") or "").strip()]
        refinement_feedback = refinements[0] if refinements else ""

        # If everything got rejected, we must shift strategy (not same retry)
        if all_rejected:
            refinement_feedback = self._make_strategy_shift_feedback(preferences)

        # If still empty, generate a minimal actionable feedback
        if not refinement_feedback:
            country = str(preferences.get("country") or "US")
            rec = int(preferences.get("recency_hours") or 36)
            refinement_feedback = (
                f"Refine query: enforce {country}-only; ensure jobs posted within last {rec} hours; "
                "roles must mention Solution Architecture; exclude India using -India -Bangalore." 
            )

        # Overall score is best score (likely < threshold)
        reason = "No jobs met constraints + threshold." if all_rejected else "Best job below threshold or had blockers."
        if best is not None and best.get("constraint_ok") is False:
            reason = f"Rejected due to constraints: {best.get('constraint_reason') or best.get('job_reason')}".strip()

        logs.append(
            {
                "ts": _utc_now_iso(),
                "layer": "L5_Evaluator",
                "event": "DECISION",
                "action": "RETRY_SEARCH",
                "score": float(best_score if best_score >= 0 else 0.0),
                "reason": reason,
                "refinement_feedback": refinement_feedback,
                "all_rejected": all_rejected,
            }
        )

        return EvaluatorDecision(
            score=float(best_score if best_score >= 0 else 0.0),
            reason=reason,
            action="RETRY_SEARCH",
            refinement_feedback=refinement_feedback,
            meta={
                "model": self.model,
                "threshold": pass_threshold,
                "per_job": per_job,
                "best": best,
                "all_rejected": all_rejected,
                "ts": _utc_now_iso(),
            },
            evaluation_logs=logs,
        )


# -----------------------------------------------------------------------------
# Artifact loading + entrypoint
# -----------------------------------------------------------------------------


def load_profile_and_ranking(
    *,
    extracted_profile_path: str | None = None,
    ranking_path: str | None = None,
    base_dir: str = "outputs/phase2",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    bd = Path(base_dir)
    bd.mkdir(parents=True, exist_ok=True)

    def _latest(pattern: str) -> str | None:
        files = sorted(bd.glob(pattern))
        return str(files[-1]) if files else None

    prof = extracted_profile_path or str(bd / "extracted_profile.json")
    rank = ranking_path or str(bd / "ranking.json")

    if not Path(prof).exists():
        prof = _latest("extracted_profile_*.json") or prof
    if not Path(rank).exists():
        rank = _latest("ranking_*.json") or rank

    profile = _read_json(prof) if Path(prof).exists() else {}
    ranking = _read_json(rank) if Path(rank).exists() else {}
    return profile, ranking, {"extracted_profile_path": prof, "ranking_path": rank}


def _enrich_ranking_items_with_job_metadata(ranking: dict[str, Any]) -> dict[str, Any]:
    """Ensure each item includes title/company/location/remote when job_path is available."""
    items = ranking.get("items")
    if not isinstance(items, list):
        return ranking

    enriched: list[dict[str, Any]] = []
    for it in items:
        item = dict(it) if isinstance(it, dict) else {}
        jp = str(item.get("job_path") or "")
        if jp and Path(jp).exists():
            try:
                job = _read_json(jp)
                for k in ("title", "company", "location", "remote", "url"):
                    if job.get(k) is not None and item.get(k) in (None, ""):
                        item[k] = job.get(k)
            except Exception:
                pass
        enriched.append(item)

    ranking = dict(ranking)
    ranking["items"] = enriched
    return ranking


@traceable(name="careeragent-ai-phase2.evaluator_node")
def evaluate_ranked_jobs(
    *,
    run_id: str,
    extracted_profile_path: str | None = None,
    ranking_path: str | None = None,
    pass_threshold: float = 0.7,
    model: str = "gemini-1.5-flash",
    preferences: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """LangGraph node entrypoint for L5_Evaluator."""

    prefs = preferences or {"country": "US", "recency_hours": 36}

    profile, ranking, paths = load_profile_and_ranking(
        extracted_profile_path=extracted_profile_path,
        ranking_path=ranking_path,
    )
    ranking = _enrich_ranking_items_with_job_metadata(ranking)

    agent = EvaluatorAgent(pass_threshold=pass_threshold, model=model)
    decision = agent.evaluate(profile=profile, ranking=ranking, preferences=prefs)

    evaluation = {
        "score": round(float(decision.score), 4),
        "reason": decision.reason,
        "action": decision.action,
        "run_id": run_id,
        "ts": _utc_now_iso(),
        "refinement_feedback": decision.refinement_feedback,
        "meta": decision.meta,
        "inputs": {**paths, "preferences": prefs},
    }

    # This is what L3 consumes next.
    refinement_feedback = decision.refinement_feedback

    # For the live feed: keep short but informative
    msg_line = f"score={evaluation['score']} action={evaluation['action']} reason={evaluation['reason']}"

    return {
        "evaluation": evaluation,
        "refinement_feedback": refinement_feedback,
        "messages": [
            {
                "role": "evaluator",
                "content": msg_line,
                "feedback": refinement_feedback or evaluation["reason"],
                "ts": evaluation["ts"],
            }
        ],
        "evaluation_logs": decision.evaluation_logs,
        "phase2": {"status": "ok", "model": model},
    }
