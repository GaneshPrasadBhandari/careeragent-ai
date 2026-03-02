from __future__ import annotations

from typing import Any


def qualified_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve jobs for L6 drafting with resilient fallbacks."""
    approved = list(state.get("approved_jobs") or [])
    if approved:
        return approved

    qualified = list(state.get("layer_debug", {}).get("L5", {}).get("qualified_jobs") or [])
    if qualified:
        return qualified

    # Final resilience fallback: if strict thresholding produced zero qualified
    # roles, continue with top scored opportunities so downstream L6/L7 are not
    # blocked and users still receive tailored ATS drafts.
    return list(state.get("scored_jobs") or [])[:40]


def job_selection_keyset(job: dict[str, Any]) -> set[str]:
    """Return robust identifiers used by frontend selection payloads."""
    keys: set[str] = set()
    for candidate in (
        job.get("id"),
        job.get("job_id"),
        job.get("url"),
        f"{job.get('title','')}|{job.get('company','')}",
    ):
        value = str(candidate or "").strip()
        if value:
            keys.add(value)
    return keys


def pick_approved_jobs(ranked: list[dict[str, Any]], selected_values: list[str]) -> list[dict[str, Any]]:
    """Pick approved jobs, tolerating different frontend identifier formats."""
    if not selected_values:
        return list(ranked)

    selected = {str(v).strip() for v in selected_values if str(v).strip()}
    approved: list[dict[str, Any]] = []
    for job in ranked:
        if job_selection_keyset(job) & selected:
            approved.append(job)
    return approved
