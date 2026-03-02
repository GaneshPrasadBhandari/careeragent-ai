"""Tiny local fallback for requests used in offline/constrained validation.

This shim intentionally supports the subset used by Streamlit mission control:
- GET/POST
- `json=` payloads
- `data=` dict payloads (form-url-encoded)
- `files=` multipart upload payloads
- `requests.exceptions.ConnectionError`
"""
from __future__ import annotations

import json
import uuid
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple


class exceptions:  # pylint: disable=too-few-public-methods
    class ConnectionError(Exception):
        pass


@dataclass
class _Response:
    status_code: int
    text: str

    def json(self) -> Any:
        try:
            return json.loads(self.text or "{}")
        except Exception:
            return {}


def _encode_multipart(files: Dict[str, Tuple[str, bytes, str]], data: Dict[str, Any] | None = None) -> tuple[bytes, str]:
    boundary = f"----careeragent-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for key, value in (data or {}).items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
                str(value).encode(),
                b"\r\n",
            ]
        )

    for field, file_obj in (files or {}).items():
        filename = "upload.bin"
        content = b""
        content_type = "application/octet-stream"
        if isinstance(file_obj, tuple):
            if len(file_obj) >= 1:
                filename = str(file_obj[0] or filename)
            if len(file_obj) >= 2:
                c = file_obj[1]
                content = c if isinstance(c, (bytes, bytearray)) else str(c).encode()
            if len(file_obj) >= 3 and file_obj[2]:
                content_type = str(file_obj[2])

        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                content,
                b"\r\n",
            ]
        )

    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _request(method: str, url: str, timeout: int = 5, **kwargs) -> _Response:
    headers = dict(kwargs.get("headers") or {})
    data = kwargs.get("data")
    json_payload = kwargs.get("json")
    files = kwargs.get("files")

    if files:
        payload, content_type = _encode_multipart(files=files, data=(data if isinstance(data, dict) else None))
        data = payload
        headers.setdefault("Content-Type", content_type)
    elif json_payload is not None:
        data = json.dumps(json_payload).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")
    elif isinstance(data, dict):
        data = urllib.parse.urlencode({k: "" if v is None else str(v) for k, v in data.items()}).encode("utf-8")
        headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
    elif isinstance(data, str):
        data = data.encode("utf-8")

    req = urllib.request.Request(url=url, method=method.upper(), data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return _Response(status_code=int(getattr(resp, "status", 200)), text=body)
    except urllib.error.HTTPError as e:
        return _Response(status_code=int(e.code), text=e.read().decode("utf-8", errors="replace"))
    except Exception as e:
        raise exceptions.ConnectionError(str(e)) from e


def get(url: str, timeout: int = 5, **kwargs) -> _Response:
    return _request("GET", url, timeout=timeout, **kwargs)


def post(url: str, timeout: int = 5, **kwargs) -> _Response:
    return _request("POST", url, timeout=timeout, **kwargs)
