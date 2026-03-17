from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any


class TestClient:
    def __init__(self, app: Any) -> None:
        self.app = app

    def get(self, path: str):
        return self.request("GET", path)

    def request(self, method: str, path: str):
        async def _call():
            messages = []

            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                messages.append(message)

            scope = {"type": "http", "method": method, "path": path, "headers": []}
            await self.app(scope, receive, send)
            start = next((m for m in messages if m.get("type") == "http.response.start"), {})
            body = b"".join(m.get("body", b"") for m in messages if m.get("type") == "http.response.body")
            status = int(start.get("status", 500))
            return status, body

        status, body = asyncio.run(_call())

        def _json():
            try:
                return json.loads(body.decode("utf-8") or "{}")
            except Exception:
                return {}

        return SimpleNamespace(status_code=status, text=body.decode("utf-8", errors="ignore"), json=_json)
