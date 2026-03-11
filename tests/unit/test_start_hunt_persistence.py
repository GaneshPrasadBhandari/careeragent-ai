import importlib
import io
import json
import sys
import types

import pytest
from starlette.datastructures import UploadFile


def _import_api_main_with_stubs():
    try:
        return importlib.import_module("careeragent.api.main")
    except Exception:
        sys.modules.pop("careeragent.api.main", None)

        fastapi = types.ModuleType("fastapi")

        class _Dummy:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                return None

        class _FastAPI(_Dummy):
            def add_middleware(self, *args, **kwargs):
                return None

            def get(self, *args, **kwargs):
                def _decorator(fn):
                    return fn
                return _decorator

            post = get

        class _BackgroundTasks:
            def __init__(self):
                self.calls = []

            def add_task(self, fn, *args, **kwargs):
                self.calls.append((fn, args, kwargs))

        class _HTTPException(Exception):
            def __init__(self, status_code: int = 500, detail: str = ""):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        fastapi.BackgroundTasks = _BackgroundTasks
        fastapi.FastAPI = _FastAPI
        fastapi.File = lambda *args, **kwargs: None
        fastapi.Form = lambda *args, **kwargs: None
        fastapi.HTTPException = _HTTPException
        fastapi.UploadFile = _Dummy

        middleware = types.ModuleType("fastapi.middleware")
        cors = types.ModuleType("fastapi.middleware.cors")
        cors.CORSMiddleware = _Dummy

        responses = types.ModuleType("fastapi.responses")
        responses.FileResponse = _Dummy
        responses.JSONResponse = _Dummy

        sys.modules.setdefault("fastapi", fastapi)
        sys.modules.setdefault("fastapi.middleware", middleware)
        sys.modules.setdefault("fastapi.middleware.cors", cors)
        sys.modules.setdefault("fastapi.responses", responses)

        return importlib.import_module("careeragent.api.main")


api = _import_api_main_with_stubs()


class _BackgroundTasksForTest:
    def __init__(self) -> None:
        self.calls = []

    def add_task(self, fn, *args, **kwargs):
        self.calls.append((fn, args, kwargs))


@pytest.mark.asyncio
async def test_start_hunt_persists_initial_state(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api, "LOGS_DIR", tmp_path / "logs")
    api.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    api.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    api._runs.clear()

    background = _BackgroundTasksForTest()
    resume = UploadFile(filename="resume.txt", file=io.BytesIO(b"python\nml\n"))
    resp = await api.start_hunt(background, resume=resume, hunt_config=json.dumps({"target_roles": ["AI Engineer"]}))

    run_id = resp["run_id"]
    state_file = api.LOGS_DIR / f"state_{run_id}.json"
    assert state_file.exists()
    state = json.loads(state_file.read_text())
    assert state["run_id"] == run_id
    assert state["status"] == "running"
    assert background.calls
