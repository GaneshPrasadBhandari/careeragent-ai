"""Tiny local fallback for requests used in offline/constrained validation."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class _Response:
    status_code: int
    text: str

    def json(self) -> Any:
        try:
            return json.loads(self.text or "{}")
        except Exception:
            return {}


def _request(method: str, url: str, timeout: int = 5, **kwargs) -> _Response:
    headers = dict(kwargs.get("headers") or {})
    data = kwargs.get("data")
    json_payload = kwargs.get("json")
    if json_payload is not None:
        data = json.dumps(json_payload).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")
    if isinstance(data, str):
        data = data.encode("utf-8")
    req = urllib.request.Request(url=url, method=method.upper(), data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return _Response(status_code=int(getattr(resp, "status", 200)), text=body)
    except urllib.error.HTTPError as e:
        return _Response(status_code=int(e.code), text=e.read().decode("utf-8", errors="replace"))
    except Exception:
        return _Response(status_code=0, text="")


def get(url: str, timeout: int = 5, **kwargs) -> _Response:
    return _request("GET", url, timeout=timeout, **kwargs)


def post(url: str, timeout: int = 5, **kwargs) -> _Response:
    return _request("POST", url, timeout=timeout, **kwargs)
