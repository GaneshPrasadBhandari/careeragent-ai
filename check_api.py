#!/usr/bin/env python3
"""Environment/API diagnostics for CareerAgent providers and LLM keys.

Usage:
  python check_api.py
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal envs
    httpx = None
import urllib.error
import urllib.request




def _warn(status: str, message: str, **extra: object) -> dict:
    payload = {"ok": None, "status": status, "warning": message}
    payload.update(extra)
    return payload


def _is_network_blocked_error(exc: Exception) -> bool:
    low = str(exc).lower()
    markers = (
        "tunnel connection failed",
        "403 forbidden",
        "temporary failure in name resolution",
        "name or service not known",
        "nodename nor servname provided",
        "network is unreachable",
        "connection refused",
        "timed out",
    )
    return any(marker in low for marker in markers)

def _mask(value: str) -> str:
    if not value:
        return "(missing)"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


async def _check_serper(key: str) -> dict:
    if not key:
        return _warn("missing_key", "API key is not set in this runtime environment.")
    try:
        if httpx is None:
            req = urllib.request.Request(
                "https://google.serper.dev/search",
                data=json.dumps({"q": "AI Engineer remote", "num": 3, "gl": "us", "hl": "en"}).encode("utf-8"),
                headers={"X-API-KEY": key, "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
                payload = json.loads(resp.read().decode("utf-8") or "{}")
                return {"ok": resp.status == 200, "status_code": resp.status, "organic_count": len(payload.get("organic", []))}
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": key, "Content-Type": "application/json"},
                json={"q": "AI Engineer remote", "num": 3, "gl": "us", "hl": "en"},
            )
        payload = {}
        try:
            payload = r.json() if r.text else {}
        except Exception:
            payload = {"raw": r.text[:200]}
        return {
            "ok": r.status_code == 200,
            "status_code": r.status_code,
            "organic_count": len(payload.get("organic", [])) if isinstance(payload, dict) else None,
            "error": payload.get("message") if isinstance(payload, dict) else None,
        }
    except Exception as exc:
        if _is_network_blocked_error(exc):
            return _warn("network_blocked", "Outbound network restrictions prevented provider health check.", error=str(exc))
        return {"ok": False, "status": f"error: {exc}"}


async def _check_tavily(key: str) -> dict:
    if not key:
        return _warn("missing_key", "API key is not set in this runtime environment.")
    try:
        if httpx is None:
            req = urllib.request.Request(
                "https://api.tavily.com/search",
                data=json.dumps({
                    "api_key": key,
                    "query": "AI Solution Architect remote jobs",
                    "search_depth": "basic",
                    "max_results": 3,
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
                payload = json.loads(resp.read().decode("utf-8") or "{}")
                return {"ok": resp.status == 200, "status_code": resp.status, "results_count": len(payload.get("results", []))}
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": key,
                    "query": "AI Solution Architect remote jobs",
                    "search_depth": "basic",
                    "max_results": 3,
                },
            )
        payload = {}
        try:
            payload = r.json() if r.text else {}
        except Exception:
            payload = {"raw": r.text[:200]}
        return {
            "ok": r.status_code == 200,
            "status_code": r.status_code,
            "results_count": len(payload.get("results", [])) if isinstance(payload, dict) else None,
            "error": payload.get("error") if isinstance(payload, dict) else None,
        }
    except Exception as exc:
        if _is_network_blocked_error(exc):
            return _warn("network_blocked", "Outbound network restrictions prevented provider health check.", error=str(exc))
        return {"ok": False, "status": f"error: {exc}"}


async def _check_remotive() -> dict:
    try:
        if httpx is None:
            with urllib.request.urlopen("https://remotive.com/api/remote-jobs?search=AI+Engineer", timeout=20) as resp:  # noqa: S310
                payload = json.loads(resp.read().decode("utf-8") or "{}")
                if resp.status == 200:
                    return {"ok": True, "status_code": resp.status, "jobs_count": len(payload.get("jobs", []))}
                return _warn("provider_unreachable", "Remotive returned non-200 response.", status_code=resp.status)
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get("https://remotive.com/api/remote-jobs", params={"search": "AI Engineer"})
        payload = r.json() if r.status_code == 200 else {}
        if r.status_code == 200:
            return {"ok": True, "status_code": r.status_code, "jobs_count": len(payload.get("jobs", []))}
        if r.status_code in {401, 403, 429, 500, 502, 503, 504}:
            return _warn("provider_unreachable", "Remotive returned non-200 response.", status_code=r.status_code)
        return {"ok": False, "status_code": r.status_code, "jobs_count": len(payload.get("jobs", []))}
    except Exception as exc:
        if _is_network_blocked_error(exc):
            return _warn("network_blocked", "Outbound network restrictions prevented provider health check.", error=str(exc))
        return {"ok": False, "status": f"error: {exc}"}


def _snapshot_env() -> dict:
    keys = [
        "SERPER_API_KEY",
        "TAVILY_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "LANGSMITH_API_KEY",
    ]
    return {k: _mask(str(os.getenv(k, ""))) for k in keys}


async def main() -> None:
    serper = str(os.getenv("SERPER_API_KEY", "")).strip()
    tavily = str(os.getenv("TAVILY_API_KEY", "")).strip()

    checks = {
        "serper": await _check_serper(serper),
        "tavily": await _check_tavily(tavily),
        "remotive": await _check_remotive(),
    }
    failing = [name for name, data in checks.items() if data.get("ok") is False]
    warnings = [name for name, data in checks.items() if data.get("ok") is None]

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "env_keys": _snapshot_env(),
        "summary": {
            "healthy": len(failing) == 0,
            "failing_checks": failing,
            "warning_checks": warnings,
        },
        "checks": checks,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
