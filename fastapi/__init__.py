from __future__ import annotations

import json
from typing import Any, Callable


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str = ""):
        self.status_code = int(status_code)
        self.detail = detail
        super().__init__(detail)


class BackgroundTasks:
    def __init__(self) -> None:
        self.tasks: list[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]] = []

    def add_task(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self.tasks.append((fn, args, kwargs))


class UploadFile:
    def __init__(self, filename: str = "", file: Any | None = None) -> None:
        self.filename = filename
        self._file = file

    async def read(self) -> bytes:
        if self._file is None:
            return b""
        return self._file.read()


def File(*_args: Any, **_kwargs: Any) -> None:
    return None


def Form(*_args: Any, **_kwargs: Any) -> None:
    return None


class Request:
    def __init__(self, scope: dict[str, Any], body: bytes = b"") -> None:
        self.scope = scope
        self._body = body

    @property
    def url(self):
        class _URL:
            def __init__(self, path: str) -> None:
                self.path = path

        return _URL(self.scope.get("path", "/"))

    async def body(self) -> bytes:
        return self._body


class FastAPI:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._routes: list[tuple[set[str], str, Callable[..., Any]]] = []
        self._startup_handlers: list[Callable[..., Any]] = []

    def add_middleware(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def on_event(self, event: str):
        def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            if event == "startup":
                self._startup_handlers.append(fn)
            return fn

        return _decorator

    def api_route(self, path: str, methods: list[str] | set[str] | tuple[str, ...] | None = None):
        allowed = {m.upper() for m in (methods or ["GET"])}

        def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._routes.append((allowed, path, fn))
            return fn

        return _decorator

    def get(self, path: str):
        return self.api_route(path, methods=["GET"])

    def post(self, path: str):
        return self.api_route(path, methods=["POST"])

    async def __call__(self, scope: dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]) -> None:
        method = str(scope.get("method") or "GET").upper()
        path = str(scope.get("path") or "/")
        for allowed, route_path, fn in self._routes:
            if method not in allowed:
                continue
            if route_path != path and route_path != "/{path:path}":
                continue
            request = Request(scope)
            result = fn(request) if route_path == "/{path:path}" else fn()
            if hasattr(result, "__await__"):
                result = await result
            if isinstance(result, dict):
                body = json.dumps(result).encode("utf-8")
                headers = [[b"content-type", b"application/json"]]
                await send({"type": "http.response.start", "status": 200, "headers": headers})
                await send({"type": "http.response.body", "body": body})
                return
            status = int(getattr(result, "status_code", 200))
            content = getattr(result, "content", b"")
            if isinstance(content, str):
                content = content.encode("utf-8")
            headers = []
            for k, v in getattr(result, "headers", {}).items():
                headers.append([str(k).encode("latin-1"), str(v).encode("latin-1")])
            await send({"type": "http.response.start", "status": status, "headers": headers})
            await send({"type": "http.response.body", "body": content})
            return

        body = b'{"detail":"Not Found"}'
        await send({"type": "http.response.start", "status": 404, "headers": [[b"content-type", b"application/json"]]})
        await send({"type": "http.response.body", "body": body})
