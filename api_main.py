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

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"
for p in (str(_REPO_ROOT), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

_backend_app: Any | None = None
_backend_error: Exception | None = None
_backend_ready = False
_backend_bootstrap_thread: threading.Thread | None = None
logger = logging.getLogger(__name__)


def _is_hunt_start_path(path: str) -> bool:
    normalized = (path or "/").split("?", 1)[0].rstrip("/") or "/"
    return normalized in {"/hunt/start", "/start_hunt"}


def _is_health_path(path: str) -> bool:
    normalized = (path or "/").split("?", 1)[0].rstrip("/") or "/"
    return normalized in {"/", "/health", "/healthz", "/ready", "/readyz"}


def _json_response(status_code: int, payload: dict[str, Any]) -> JSONResponse:
    return JSONResponse(
        content=payload,
        status_code=status_code,
        headers={"cache-control": "no-store, no-cache, must-revalidate"},
    )


def _init_backend_background() -> None:
    """Initialize the heavy backend app in a dedicated background task."""
    global _backend_app, _backend_error, _backend_ready
    try:
        gc.collect()
        _backend_app = importlib.import_module("careeragent.api.main").app
        _backend_ready = True
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize careeragent.api.main: %s", exc)
        _backend_error = exc


def _ensure_lazy_warmup() -> None:
    global _backend_bootstrap_thread
    if _backend_ready or _backend_error is not None:
        return
    if _backend_bootstrap_thread is not None and _backend_bootstrap_thread.is_alive():
        return
    _backend_bootstrap_thread = threading.Thread(target=_init_backend_background, daemon=True)
    _backend_bootstrap_thread.start()


@app.on_event("startup")
async def start_up() -> None:
    _ensure_lazy_warmup()


async def _proxy_to_backend(request: Request) -> Response:
    assert _backend_app is not None
    body = await request.body()
    messages: list[dict[str, Any]] = []
    sent_once = False

    async def receive() -> dict[str, Any]:
        nonlocal sent_once
        if sent_once:
            return {"type": "http.disconnect"}
        sent_once = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    await _backend_app(request.scope, receive, send)

    start = next((m for m in messages if m.get("type") == "http.response.start"), None)
    status_code = int((start or {}).get("status", 502))
    headers: dict[str, str] = {}
    for k, v in (start or {}).get("headers", []):
        key = k.decode("latin-1")
        if key.lower() == "content-length":
            continue
        headers[key] = v.decode("latin-1")

    body_chunks = [m.get("body", b"") for m in messages if m.get("type") == "http.response.body"]
    return Response(content=b"".join(body_chunks), status_code=status_code, headers=headers)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def gateway(path: str, request: Request) -> Response:
    request_path = str(request.url.path or "/")

    if _is_health_path(request_path):
        return _json_response(200, {"status": "online"})

    if not _backend_ready or _backend_app is None:
        if _backend_error is None:
            _ensure_lazy_warmup()
            if _is_hunt_start_path(request_path):
                return _json_response(202, {"status": "loading_models", "retry_after": 3})
            return _json_response(503, {"status": "initializing", "retry_after": 5})

    if _backend_error is not None:
        return _json_response(
            503,
            {
                "status": "error",
                "error": "backend_unavailable",
                "message": "Backend app failed to initialize",
                "exception": type(_backend_error).__name__,
            },
        )

    return await _proxy_to_backend(request)


if __name__ == "__main__":
    uvicorn = importlib.import_module("uvicorn")
    uvicorn.run(app, host="0.0.0.0", port=10000, workers=1)
