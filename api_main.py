"""Lightweight compatibility entrypoint for ``uvicorn api_main:app``.

Provides immediate health responses while a global background initializer loads
``careeragent.api.main``.
"""

from __future__ import annotations

import gc
import importlib
import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"
for p in (str(_REPO_ROOT), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

_backend_app: Any | None = None
_backend_error: Exception | None = None
_backend_ready = False
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


def _init_backend_background() -> None:
    """Initialize the heavy backend app in a dedicated global background task."""
    global _backend_app, _backend_error, _backend_ready
    try:
        gc.collect()
        _backend_app = importlib.import_module("careeragent.api.main").app
        _backend_ready = True
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize careeragent.api.main: %s", exc)
        _backend_error = exc


_backend_bootstrap_thread = threading.Thread(target=_init_backend_background, daemon=True)
_backend_bootstrap_thread.start()


async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:  # ASGI entrypoint
    if scope.get("type") != "http":
        return

    path = str(scope.get("path") or "/")
    if _is_health_path(path):
        headers, body = _json_response(200, {"status": "online"})
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        await send({"type": "http.response.body", "body": body})
        return

    if not _backend_ready or _backend_app is None:
        if _backend_error is None:
            headers, body = _json_response(503, {"status": "initializing", "retry_after": 5})
            await send({"type": "http.response.start", "status": 503, "headers": headers})
            await send({"type": "http.response.body", "body": body})
            return
    if _backend_error is not None:
        exc = _backend_error
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

    await _backend_app(scope, receive, send)


if __name__ == "__main__":
    uvicorn = importlib.import_module("uvicorn")
    uvicorn.run(app, host="0.0.0.0", port=10000, workers=1)
