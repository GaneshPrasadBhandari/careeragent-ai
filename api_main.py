"""Lightweight compatibility entrypoint for ``uvicorn api_main:app``.

Provides immediate health responses while a global background initializer loads
``careeragent.api.main``.
"""

from __future__ import annotations

import gc
import importlib
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response
    _FASTAPI_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by unit tests
    FastAPI = None  # type: ignore[assignment]
    Request = Any  # type: ignore[assignment]
    JSONResponse = Any  # type: ignore[assignment]
    Response = Any  # type: ignore[assignment]
    _FASTAPI_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)


def _load_env_file() -> None:
    """Parse `.env` without importing heavy dotenv dependencies."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = value.strip().strip('"').strip("'")
    except Exception as exc:  # pragma: no cover - env parsing should never block startup
        logger.debug("Skipping .env bootstrap due to parse error: %s", exc)


_load_env_file()


def _disable_langchain_tracing_due_to_error(exc: Exception) -> None:
    """Fail-safe: disable tracing when LangSmith auth/ingest errors surface."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    if "langsmith" not in msg and "langchain" not in msg:
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    logger.warning("LangSmith/LangChain tracing disabled after initialization error: %s", exc)


def _dependency_missing_app(exc: ModuleNotFoundError) -> Callable[..., Any]:
    """ASGI fallback used when FastAPI dependency is unavailable."""

    async def _app(scope: dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]) -> None:
        path = str((scope or {}).get("path") or "/")
        is_health = path in {"/", "/health", "/healthz", "/ready", "/readyz"}
        status = 200 if is_health else 503
        payload: dict[str, Any] = {
            "status": "ok" if is_health else "error",
            "backend_dependency_missing": True,
            "dependency": "fastapi",
            "error": "backend_dependency_missing",
            "detail": str(exc),
        }
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": json.dumps(payload).encode("utf-8"),
            }
        )

    return _app


if FastAPI is None:
    app = _dependency_missing_app(_FASTAPI_IMPORT_ERROR or ModuleNotFoundError("fastapi"))
else:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "online"}

    _REPO_ROOT = Path(__file__).resolve().parent
    _SRC = _REPO_ROOT / "src"
    for p in (str(_REPO_ROOT), str(_SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)

    _backend_app: Any | None = None
    _backend_error: Exception | None = None
    _backend_ready = False
    _backend_bootstrap_thread: threading.Thread | None = None
    _health_server_thread: threading.Thread | None = None
    _pending_hunt_runs: dict[str, dict[str, Any]] = {}

    class _DedicatedHealthHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] not in {"/health", "/healthz", "/ready", "/readyz"}:
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps({"status": "online"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _start_dedicated_health_server() -> None:
        port = int(os.getenv("HEALTHCHECK_PORT", "10001"))
        server = ThreadingHTTPServer(("0.0.0.0", port), _DedicatedHealthHandler)
        logger.info("Dedicated health responder listening on :%s", port)
        server.serve_forever()

    def _is_hunt_start_path(path: str) -> bool:
        normalized = (path or "/").split("?", 1)[0].rstrip("/") or "/"
        return normalized in {"/hunt/start", "/start_hunt"}

    def _pending_status_path(path: str) -> str | None:
        normalized = (path or "/").split("?", 1)[0].rstrip("/")
        m = re.match(r"^/hunt/([A-Za-z0-9_-]{6,64})/status$", normalized)
        if not m:
            return None
        return m.group(1)

    def _extract_client_run_id(raw_body: bytes) -> str:
        try:
            text = raw_body.decode("utf-8", errors="ignore")
        except Exception:
            text = ""
        m = re.search(r'"client_run_id"\s*:\s*"([A-Za-z0-9_-]{6,64})"', text)
        return (m.group(1) if m else "").strip()[:64]

    def _register_lightweight_handshake(raw_body: bytes) -> str:
        run_id = _extract_client_run_id(raw_body) or uuid.uuid4().hex[:12]
        _pending_hunt_runs[run_id] = {
            "run_id": run_id,
            "status": "queued",
            "progress_pct": 0.0,
            "queued_at": time.time(),
            "message": "Lightweight handshake accepted while backend warms.",
        }
        return run_id

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
            try:
                _backend_app = importlib.import_module("careeragent.api.main").app
            except Exception as exc:  # noqa: BLE001
                _disable_langchain_tracing_due_to_error(exc)
                if os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "false":
                    _backend_app = importlib.import_module("careeragent.api.main").app
                else:
                    raise
            _backend_ready = True
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to initialize careeragent.api.main: %s", exc)
            _backend_error = exc

    def _ensure_lazy_warmup() -> None:
        global _backend_bootstrap_thread, _health_server_thread
        if _health_server_thread is None or not _health_server_thread.is_alive():
            _health_server_thread = threading.Thread(target=_start_dedicated_health_server, daemon=True)
            _health_server_thread.start()
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
                pending_run_id = _pending_status_path(request_path)
                if pending_run_id and pending_run_id in _pending_hunt_runs:
                    return _json_response(200, dict(_pending_hunt_runs[pending_run_id]))
                if _is_hunt_start_path(request_path):
                    raw_body = await request.body()
                    run_id = _register_lightweight_handshake(raw_body)
                    return _json_response(
                        200,
                        {
                            "run_id": run_id,
                            "task_id": run_id,
                            "status": "loading_models",
                            "queued": True,
                            "retry_after": 2,
                            "message": "Handshake OK. Backend is warming and will continue startup shortly.",
                        },
                    )
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
