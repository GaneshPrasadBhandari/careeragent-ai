from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit, urlunsplit


DEFAULT_APPROVAL_FALLBACK_COUNT = 12


def _normalize_selection_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "|" in text:
        left, right = text.split("|", 1)
        return f"{left.strip().lower()}|{right.strip().lower()}"
    if text.startswith("//"):
        text = f"https:{text}"
    elif text.startswith(("www.", "linkedin.com/", "jobs.", "boards.", "careers.")):
        text = f"https://{text}"
    if not text.startswith(("http://", "https://")):
        return text.lower()
    try:
        parts = urlsplit(text)
        query = parse_qs(parts.query, keep_blank_values=False)
        for key in ("url", "u", "q", "redirect", "redirect_url", "dest", "destination", "target"):
            nested = query.get(key, [""])[0]
            if nested.startswith(("http://", "https://", "www.")):
                return _normalize_selection_value(unquote(nested))
        clean_query = "&".join(
            f"{k.lower()}={v}"
            for k, values in sorted(query.items())
            if not (
                k.lower().startswith(("utm_", "trk", "ref", "fbclid", "gclid"))
                or k.lower().endswith("_src")
                or k.lower() == "source"
            )
            for v in values
        )
        netloc = parts.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return urlunsplit((parts.scheme.lower(), netloc, parts.path.rstrip("/"), clean_query, ""))
    except Exception:
        return text.lower()


def qualified_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve jobs for L6 drafting with resilient fallbacks."""
    approved = list(state.get("approved_jobs") or [])
    if approved:
        return approved

    qualified = list(state.get("layer_debug", {}).get("L5", {}).get("qualified_jobs") or [])
    if qualified:
        return qualified

    scored = list(state.get("scored_jobs") or [])
    if scored:
        return scored[:DEFAULT_APPROVAL_FALLBACK_COUNT]

    raw_preview = list(state.get("approved_jobs_preview") or [])
    if raw_preview:
        return raw_preview[:DEFAULT_APPROVAL_FALLBACK_COUNT]

    return []


def job_selection_keyset(job: dict[str, Any]) -> set[str]:
    """Return robust identifiers used by frontend selection payloads."""
    keys: set[str] = set()
    for candidate in (
        job.get("id"),
        job.get("job_id"),
        job.get("url"),
        job.get("direct_job_url"),
        job.get("redirect_url"),
        job.get("job_url"),
        job.get("application_url"),
        f"{job.get('title', '')}|{job.get('company', '')}",
    ):
        value = _normalize_selection_value(candidate)
        if value:
            keys.add(value)
    return keys


def pick_approved_jobs(ranked: list[dict[str, Any]], selected_values: list[str]) -> list[dict[str, Any]]:
    """Pick approved jobs, tolerating different frontend identifier formats."""
    if not selected_values:
        return list(ranked)

    selected = {_normalize_selection_value(v) for v in selected_values if _normalize_selection_value(v)}
    approved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for job in ranked:
        if not (job_selection_keyset(job) & selected):
            continue
        dedupe_key = _normalize_selection_value(job.get("direct_job_url") or job.get("url") or job.get("id") or job.get("job_id"))
        if dedupe_key and dedupe_key in seen:
            continue
        if dedupe_key:
            seen.add(dedupe_key)
        approved.append(job)
    return approved
