"""Lightweight compatibility entrypoint for ``uvicorn api_main:app``.

Provides immediate health responses while lazily loading the full backend app.
"""

from __future__ import annotations

import asyncio
import gc
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"
for p in (str(_REPO_ROOT), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

_backend_app: Any | None = None
_backend_error: Exception | None = None
_backend_lock = asyncio.Lock()
_INIT_TIMEOUT_SECONDS = 180
logger = logging.getLogger(__name__)


def _is_health_path(path: str) -> bool:
    normalized = (path or "/").split("?", 1)[0].rstrip("/") or "/"
    return normalized in {"/", "/health", "/healthz", "/ready", "/readyz"}


def _json_response(status_code: int, payload: dict[str, Any]) -> tuple[list[list[bytes]], bytes]:
    body = json.dumps(payload).encode("utf-8")
    headers = [
        [b"content-type", b"application/json"],
        [b"cache-control", b"no-store, no-cache, must-revalidate"],
        [b"content-length", str(len(body)).encode("ascii")],
    ]
    return headers, body


async def _load_backend() -> Any:
    global _backend_app, _backend_error

    if _backend_app is not None:
        return _backend_app
    if _backend_error is not None:
        raise _backend_error

    async with _backend_lock:
        if _backend_app is not None:
            return _backend_app
        if _backend_error is not None:
            raise _backend_error

        try:
            gc.collect()
            _backend_app = await asyncio.wait_for(
                asyncio.to_thread(lambda: importlib.import_module("careeragent.api.main").app),
                timeout=_INIT_TIMEOUT_SECONDS,
            )
            return _backend_app
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to initialize careeragent.api.main: %s", exc)
            _backend_error = exc
            raise


async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:  # ASGI entrypoint
    if scope.get("type") != "http":
        return

    path = str(scope.get("path") or "/")
    if _is_health_path(path):
        headers, body = _json_response(200, {"status": "online"})
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        await send({"type": "http.response.body", "body": body})
        return

    try:
        backend_app = await _load_backend()
    except Exception as exc:  # noqa: BLE001
        error_name = type(exc).__name__
        headers, body = _json_response(
            503,
            {
                "status": "error",
                "error": "backend_unavailable",
                "message": "Backend app failed to initialize",
                "exception": error_name,
            },
        )
        await send({"type": "http.response.start", "status": 503, "headers": headers})
        await send({"type": "http.response.body", "body": body})
        return

    await backend_app(scope, receive, send)


if __name__ == "__main__":
    uvicorn = importlib.import_module("uvicorn")
    uvicorn.run(app, host="0.0.0.0", port=10000)
