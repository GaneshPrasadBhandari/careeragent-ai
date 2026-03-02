"""Compatibility entrypoint for users running ``uvicorn api_main:app``.

Primary behavior:
- Forward to canonical FastAPI app at ``careeragent.api.main``.

Fallback behavior (dependency-missing environments):
- If FastAPI is missing, still expose an importable ASGI ``app`` that returns
  a 503 JSON response with installation guidance instead of crashing at import.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"
for p in (str(_REPO_ROOT), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _dependency_missing_app(error: ModuleNotFoundError):
    async def _app(scope: dict[str, Any], receive: Any, send: Any) -> None:  # pragma: no cover - runtime guard
        if scope.get("type") != "http":
            return

        body = json.dumps(
            {
                "status": "error",
                "error": "backend_dependency_missing",
                "missing_module": error.name,
                "hint": "Install backend dependencies (e.g. `pip install -e .` or `uv sync`).",
            }
        ).encode("utf-8")

        await send(
            {
                "type": "http.response.start",
                "status": 503,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(body)).encode("ascii")],
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    return _app


try:
    app = importlib.import_module("careeragent.api.main").app
except ModuleNotFoundError as exc:
    app = _dependency_missing_app(exc)
