from __future__ import annotations

import json
from typing import Any


class Response:
    def __init__(self, content: bytes | str = b"", status_code: int = 200, headers: dict[str, str] | None = None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}


class JSONResponse(Response):
    def __init__(self, content: Any = None, status_code: int = 200, headers: dict[str, str] | None = None):
        payload = json.dumps(content or {}).encode("utf-8")
        merged = {"content-type": "application/json", **(headers or {})}
        super().__init__(content=payload, status_code=status_code, headers=merged)


class FileResponse(Response):
    def __init__(self, path: str, filename: str | None = None):
        self.path = path
        self.filename = filename
        super().__init__(content=b"", status_code=200, headers={})
