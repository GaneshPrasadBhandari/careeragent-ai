#!/usr/bin/env python3
"""CareerAgent environment + provider readiness diagnostics.

This script checks:
- key presence (LLMs, APIs, tools)
- light connectivity/auth probes where safe
- Qdrant + Chroma availability
- LangSmith configuration consistency

Usage:
  python check_env.py
"""
from __future__ import annotations

import json
import os
from urllib.parse import urlparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None
import urllib.error
import urllib.request

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

try:
    from qdrant_client import QdrantClient
except Exception:  # pragma: no cover
    QdrantClient = None


load_dotenv()


@dataclass
class CheckResult:
    name: str
    state: str  # active | warning | failed
    detail: str = ""


def _icon(state: str) -> str:
    if state == "active":
        return "✅"
    if state == "warning":
        return "⚠️"
    return "❌"


def _print_result(result: CheckResult) -> None:
    label = {"active": "Active", "warning": "Warning", "failed": "Failed/Missing"}.get(result.state, result.state)
    details = f" ({result.detail})" if result.detail else ""
    print(f"{result.name:<32} {_icon(result.state)} {label}{details}")


def _http_get(url: str, headers: Optional[dict[str, str]] = None, timeout: int = 8) -> tuple[Optional[int], str]:
    try:
        if requests is not None:
            r = requests.get(url, headers=headers or {}, timeout=timeout)
            return r.status_code, ""
        req = urllib.request.Request(url, headers=headers or {}, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return int(resp.status), ""
    except Exception as exc:
        return None, str(exc)


def _http_post(url: str, headers: Optional[dict[str, str]] = None, payload: Optional[dict[str, Any]] = None, timeout: int = 8) -> tuple[Optional[int], str]:
    try:
        if requests is not None:
            r = requests.post(url, headers=headers or {}, json=payload or {}, timeout=timeout)
            return r.status_code, ""
        req = urllib.request.Request(
            url,
            data=json.dumps(payload or {}).encode("utf-8"),
            headers=headers or {"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return int(resp.status), ""
    except Exception as exc:
        return None, str(exc)


def _state_from_http(code: Optional[int], *, ok: set[int], warn: set[int], err: str = "") -> tuple[str, str]:
    if code in ok:
        return "active", f"HTTP {code}"
    if code in warn:
        return "warning", f"HTTP {code}"
    if code is None:
        msg = err or "network_unreachable"
        return "warning", msg
    return "failed", f"HTTP {code}"


def _env(*keys: str) -> str:
    for key in keys:
        val = str(os.getenv(key, "")).strip()
        if val:
            return val
    return ""


def _bool_env(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _key_state(value: str, *, strict: bool) -> str:
    if value:
        return "active"
    return "failed" if strict else "warning"


def _normalize_qdrant_url(raw_url: str) -> str:
    """Return a Qdrant URL that is safe to probe.

    - Adds https:// for host-only inputs.
    - Removes known dashboard-only paths.
    - Keeps custom API base paths when they are likely intentional.
    """
    value = raw_url.strip().rstrip("/")
    if not value:
        return value

    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"

    parsed = urlparse(value)
    path = (parsed.path or "").rstrip("/")
    first_segment = path.lstrip("/").split("/")[0] if path else ""
    dashboard_segments = {"dashboard", "ui", "console", "cluster"}

    clean_path = "" if first_segment in dashboard_segments else path
    rebuilt = f"{parsed.scheme}://{parsed.netloc}{clean_path}"
    return rebuilt.rstrip("/")


def _join_url(base: str, tail: str) -> str:
    return f"{base.rstrip('/')}/{tail.lstrip('/')}"


def _probe_qdrant_http(qdrant_url: str, qdrant_key: str) -> tuple[str, str]:
    """Probe Qdrant with known collection endpoints."""
    base = _normalize_qdrant_url(qdrant_url)
    headers = {"api-key": qdrant_key}

    parsed = urlparse(base)
    base_path = (parsed.path or "").rstrip("/")
    candidates = ["collections"] if base_path.endswith("/v1") else ["collections", "v1/collections"]

    last_code: Optional[int] = None
    last_err = ""
    for path in candidates:
        url = _join_url(base, path)
        code, err = _http_get(url, headers=headers)
        last_code, last_err = code, err
        state, detail = _state_from_http(code, ok={200}, warn={401, 403, 429, 500, 502, 503, 504}, err=err)
        if state == "active":
            return state, f"{detail} ({urlparse(url).path})"
        if code in {401, 403, 429}:
            return "warning", f"{detail} ({urlparse(url).path})"

    if last_code == 404:
        return "failed", "HTTP 404 (QDRANT_URL points to a non-API endpoint; use your cluster API URL)"
    return _state_from_http(last_code, ok={200}, warn={401, 403, 429, 500, 502, 503, 504}, err=last_err)


def run() -> dict[str, Any]:
    print("\n--- CareerAgent-AI: Full Environment Diagnostic ---")
    results: list[CheckResult] = []

    # LLM keys
    openai_key = _env("OPENAI_API_KEY")
    anthropic_key = _env("ANTHROPIC_API_KEY")
    gemini_key = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
    hf_token = _env("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGINGFACEHUB_API_TOKEN")

    render_runtime = bool(_env("RENDER_SERVICE_NAME") or _env("RENDER_INSTANCE_ID"))
    strict_required_keys = _bool_env("CHECK_ENV_STRICT", default=render_runtime)

    results.append(CheckResult("OpenAI Key", _key_state(openai_key, strict=strict_required_keys)))
    results.append(CheckResult("Anthropic Key", "active" if anthropic_key else "warning"))
    results.append(CheckResult("Gemini/Google Key", "active" if gemini_key else "warning"))
    results.append(CheckResult("Hugging Face Token", "active" if hf_token else "warning"))

    # API keys/tools
    tavily_key = _env("TAVILY_API_KEY")
    serper_key = _env("SERPER_API_KEY")
    resend_key = _env("RESEND_API_KEY")
    sendgrid_key = _env("SENDGRID_API_KEY")

    results.append(CheckResult("Tavily Key", _key_state(tavily_key, strict=strict_required_keys)))
    results.append(CheckResult("Serper Key", _key_state(serper_key, strict=strict_required_keys)))
    results.append(CheckResult("Resend Key", "active" if resend_key else "warning"))
    results.append(CheckResult("SendGrid Key", "active" if sendgrid_key else "warning"))

    # LangSmith
    ls_key = _env("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
    ls_project = _env("LANGSMITH_PROJECT")
    lc_project = _env("LANGCHAIN_PROJECT")
    results.append(CheckResult("LangSmith API Key", "active" if ls_key else "warning"))

    if ls_project and lc_project and ls_project == lc_project:
        results.append(CheckResult("LangSmith Project Sync", "active", f"{ls_project}"))
    elif ls_project or lc_project:
        results.append(CheckResult("LangSmith Project Sync", "warning", f"ls={ls_project or 'unset'} lc={lc_project or 'unset'}"))
    else:
        results.append(CheckResult("LangSmith Project Sync", "warning", "projects unset"))

    # Qdrant + Chroma
    qdrant_url = _env("QDRANT_URL")
    qdrant_key = _env("QDRANT_API_KEY")
    qdrant_state = "warning"
    qdrant_detail = "not configured"
    if qdrant_url and qdrant_key and QdrantClient:
        try:
            normalized_qdrant_url = _normalize_qdrant_url(qdrant_url)
            client = QdrantClient(url=normalized_qdrant_url, api_key=qdrant_key, timeout=8)
            col = client.get_collections()
            qdrant_state = "active"
            count = len(getattr(col, "collections", []) or [])
            qdrant_detail = f"collections={count}"
        except Exception as exc:
            # fallback probe to separate network from auth/config issues
            qdrant_state, qdrant_detail = _probe_qdrant_http(qdrant_url, qdrant_key)
            if qdrant_state == "failed":
                qdrant_detail = qdrant_detail or str(exc)
    elif qdrant_url and qdrant_key:
        qdrant_state, qdrant_detail = _probe_qdrant_http(qdrant_url, qdrant_key)
    elif qdrant_url or qdrant_key:
        qdrant_state, qdrant_detail = "warning", "QDRANT_URL/QDRANT_API_KEY partially set"
    results.append(CheckResult("Qdrant Cloud", qdrant_state, qdrant_detail))

    chroma_dir = _env("CHROMA_PERSIST_DIR") or "outputs/phase3/chroma"
    results.append(CheckResult("Local Chroma Cache", "active" if Path(chroma_dir).exists() else "warning", chroma_dir))

    # Reachability probes
    if serper_key:
        code, err = _http_post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
            payload={"q": "AI Engineer remote", "num": 1, "gl": "us", "hl": "en"},
        )
        st, dt = _state_from_http(code, ok={200}, warn={400, 401, 403, 429, 500, 502, 503, 504}, err=err)
        results.append(CheckResult("Serper Reachability", st, dt))
    else:
        results.append(CheckResult("Serper Reachability", "warning", "not_checked (missing key)"))

    if tavily_key:
        code, err = _http_post(
            "https://api.tavily.com/search",
            payload={"api_key": tavily_key, "query": "AI jobs", "max_results": 1, "search_depth": "basic"},
        )
        st, dt = _state_from_http(code, ok={200}, warn={400, 401, 403, 429, 500, 502, 503, 504}, err=err)
        results.append(CheckResult("Tavily Reachability", st, dt))
    else:
        results.append(CheckResult("Tavily Reachability", "warning", "not_checked (missing key)"))

    if openai_key:
        code, err = _http_get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {openai_key}"})
        st, dt = _state_from_http(code, ok={200}, warn={401, 403, 429, 500, 502, 503, 504}, err=err)
        results.append(CheckResult("OpenAI Reachability", st, dt))
    else:
        results.append(CheckResult("OpenAI Reachability", "warning", "not_checked (missing key)"))

    if anthropic_key:
        code, err = _http_get(
            "https://api.anthropic.com/v1/models",
            headers={"x-api-key": anthropic_key, "anthropic-version": "2023-06-01"},
        )
        st, dt = _state_from_http(code, ok={200}, warn={400, 401, 403, 404, 429, 500, 502, 503, 504}, err=err)
        results.append(CheckResult("Anthropic Reachability", st, dt))
    else:
        results.append(CheckResult("Anthropic Reachability", "warning", "not_checked (missing key)"))

    # Render environment hint
    render_service = _env("RENDER_SERVICE_NAME")
    render_instance = _env("RENDER_INSTANCE_ID")
    if render_service or render_instance:
        results.append(CheckResult("Render Runtime", "active", f"service={render_service or 'unknown'}"))
    else:
        results.append(CheckResult("Render Runtime", "warning", "not running inside Render shell"))

    for item in results:
        _print_result(item)

    print("-" * 68)
    failed = [r.name for r in results if r.state == "failed"]
    warns = [r.name for r in results if r.state == "warning"]

    if failed:
        print("❌ Critical checks failed:", ", ".join(failed))
    if warns:
        print("⚠️ Warnings:", ", ".join(warns))
    if not failed:
        print("✅ No critical failures detected.")

    summary = {
        "critical_ok": not failed,
        "failed": failed,
        "warnings": warns,
        "results": [r.__dict__ for r in results],
    }
    print("\nJSON Summary:")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
