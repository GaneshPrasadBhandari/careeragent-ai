import asyncio
import importlib.util
import json
import sys
from pathlib import Path


def _load_api_main_module():
    p = Path(__file__).resolve().parents[2] / "api_main.py"
    spec = importlib.util.spec_from_file_location("api_main_entry", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["api_main_entry"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_api_main_app_is_importable() -> None:
    api_main = _load_api_main_module()
    assert callable(api_main.app)


def _call_asgi(path: str) -> tuple[int, dict]:
    api_main = _load_api_main_module()
    fallback = api_main._dependency_missing_app(ModuleNotFoundError("fastapi"))
    sent = []

    async def _send(msg):
        sent.append(msg)

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    asyncio.run(fallback({"type": "http", "path": path}, _receive, _send))
    status = sent[0]["status"]
    payload = json.loads(sent[1]["body"].decode("utf-8"))
    return status, payload


def test_dependency_fallback_health_returns_200_json() -> None:
    status, payload = _call_asgi("/health")
    assert status == 200
    assert payload["status"] == "ok"
    assert payload["backend_dependency_missing"] is True


def test_dependency_fallback_non_health_returns_503_json() -> None:
    status, payload = _call_asgi("/")
    assert status == 503
    assert payload["error"] == "backend_dependency_missing"
