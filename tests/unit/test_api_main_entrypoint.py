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


def test_dependency_fallback_app_returns_503_json() -> None:
    api_main = _load_api_main_module()
    fallback = api_main._dependency_missing_app(ModuleNotFoundError("fastapi"))

    sent = []

    async def _send(msg):
        sent.append(msg)

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    asyncio.run(fallback({"type": "http"}, _receive, _send))

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 503
    payload = json.loads(sent[1]["body"].decode("utf-8"))
    assert payload["error"] == "backend_dependency_missing"
