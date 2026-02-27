"""Minimal httpx-compatible shim for offline test environments."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class HTTPError(Exception):
    pass


@dataclass
class Response:
    status_code: int
    text: str = ""

    def json(self) -> Any:
        return json.loads(self.text or "{}")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise HTTPError(f"HTTP {self.status_code}: {self.text}")


class Client:
    def __init__(self, timeout: float | None = None, **_: Any) -> None:
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url: str, **kwargs: Any) -> Response:
        return _request("GET", url, timeout=self.timeout, **kwargs)

    def post(self, url: str, **kwargs: Any) -> Response:
        return _request("POST", url, timeout=self.timeout, **kwargs)


class AsyncClient(Client):
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str, **kwargs: Any) -> Response:
        return super().get(url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> Response:
        return super().post(url, **kwargs)


def _request(method: str, url: str, timeout: float | None = None, **kwargs: Any) -> Response:
    headers = kwargs.get("headers") or {}
    payload = kwargs.get("json")
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers = {**headers, "Content-Type": "application/json"}
    req = urllib.request.Request(url, method=method, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            return Response(status_code=getattr(resp, "status", 200), text=body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
        return Response(status_code=e.code, text=body)
