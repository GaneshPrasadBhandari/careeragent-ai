import io
import json
import pytest

pytest.importorskip("fastapi")

from fastapi import BackgroundTasks
from starlette.datastructures import UploadFile

import careeragent.api.main as api


@pytest.mark.asyncio
async def test_start_hunt_persists_initial_state(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api, "LOGS_DIR", tmp_path / "logs")
    api.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    api.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    api._runs.clear()

    background = BackgroundTasks()
    resume = UploadFile(filename="resume.txt", file=io.BytesIO(b"python\nml\n"))
    resp = await api.start_hunt(background, resume=resume, hunt_config=json.dumps({"target_roles": ["AI Engineer"]}))

    run_id = resp["run_id"]
    state_file = api.LOGS_DIR / f"state_{run_id}.json"
    assert state_file.exists()
    state = json.loads(state_file.read_text())
    assert state["run_id"] == run_id
    assert state["status"] == "running"
