"""Compatibility entrypoint for users running ``uvicorn api_main:app``.

Primary behavior:
- Forward to canonical FastAPI app at ``careeragent.api.main``.

Fallback behavior (dependency-missing environments):
- If FastAPI/uvicorn is missing, expose an importable ASGI ``app`` and provide
  a tiny stdlib HTTP server for local diagnostics.
"""

from __future__ import annotations

import importlib
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"
for p in (str(_REPO_ROOT), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)


_FALLBACK_ENABLED = False
_FALLBACK_ERROR: ModuleNotFoundError | None = None


def _fallback_payload(status_code: int) -> bytes:
    error_name = _FALLBACK_ERROR.name if _FALLBACK_ERROR else "unknown"
    return json.dumps(
        {
            "status": "ok" if status_code == 200 else "error",
            "error": None if status_code == 200 else "backend_dependency_missing",
            "mode": "fallback",
            "backend_dependency_missing": True,
            "missing_module": error_name,
            "hint": "Install backend dependencies (e.g. `pip install -e .` or `uv sync`).",
        }
    ).encode("utf-8")


def _dependency_missing_app(error: ModuleNotFoundError):
    async def _app(scope: dict[str, Any], receive: Any, send: Any) -> None:  # pragma: no cover - runtime guard
        if scope.get("type") != "http":
            return

        path = str(scope.get("path") or "/")
        status = 200 if path == "/health" else 503
        body = _fallback_payload(status)
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(body)).encode("ascii")],
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    return _app


def _run_fallback_http(host: str = "127.0.0.1", port: int = 8000) -> None:
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            status = 200 if self.path == "/health" else 503
            body = _fallback_payload(status)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"Fallback backend running on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - manual shutdown path
        print("Fallback backend stopped.")
    finally:
        server.server_close()


try:
    app = importlib.import_module("careeragent.api.main").app
except ModuleNotFoundError as exc:
    _FALLBACK_ENABLED = True
    _FALLBACK_ERROR = exc
    app = _dependency_missing_app(exc)


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    if _FALLBACK_ENABLED:
        _run_fallback_http(host=host, port=port)
    else:
        uvicorn = importlib.import_module("uvicorn")
        uvicorn.run("api_main:app", host=host, port=port, reload=False)
