from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from careeragent.api import main as api


def _ts_ago(seconds: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


def test_is_stalled_early_run_detects_first_layer_hang():
    state = {
        "status": "running",
        "progress_pct": 10.0,
        "updated_at": _ts_ago(45),
        "layers": [
            {"status": "ok"},
            {"status": "running"},
            {"status": "waiting"},
            {"status": "waiting"},
        ],
    }
    assert api._is_stalled_early_run(state) is True


def test_is_stalled_early_run_detects_second_layer_hang() -> None:
    state = {
        "status": "running",
        "progress_pct": 10.0,
        "updated_at": _ts_ago(45),
        "layers": [
            {"status": "ok"},
            {"status": "ok"},
            {"status": "running"},
            {"status": "waiting"},
            {"status": "waiting"},
        ],
    }
    assert api._is_stalled_early_run(state) is True


def test_is_stalled_early_run_ignores_recent_activity() -> None:
    state = {
        "status": "running",
        "progress_pct": 10.0,
        "updated_at": _ts_ago(5),
        "layers": [
            {"status": "ok"},
            {"status": "running"},
            {"status": "waiting"},
            {"status": "waiting"},
        ],
    }
    assert api._is_stalled_early_run(state) is False


def test_try_recover_stalled_run_marks_once(tmp_path, monkeypatch):
    resume_path = tmp_path / "resume.txt"
    resume_path.write_text("python")
    monkeypatch.setattr(api, "LOGS_DIR", tmp_path / "logs")
    api.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    state = {
        "run_id": "abc123",
        "status": "running",
        "progress_pct": 10.0,
        "updated_at": _ts_ago(60),
        "resume_path": str(resume_path),
        "layers": [
            {"status": "ok", "meta": {}},
            {"status": "running", "meta": {}},
            {"status": "waiting", "meta": {}},
            {"status": "waiting", "meta": {}},
        ],
        "agent_log": [],
    }

    scheduled = []

    class _Loop:
        def create_task(self, coro):
            scheduled.append(coro)
            coro.close()

    async def _fake_run_pipeline(_run_id: str, _path: Path):
        return None

    monkeypatch.setattr(api.asyncio, "get_running_loop", lambda: _Loop())
    monkeypatch.setattr(api, "run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(api, "_persist_state", lambda _run_id: None)

    api._try_recover_stalled_run("abc123", state)
    assert state.get("recovery_attempted_at")
    assert scheduled, "expected recovery scheduler to enqueue pipeline"

    before = len(scheduled)
    api._try_recover_stalled_run("abc123", state)
    assert len(scheduled) == before, "recovery should only be attempted once"


def test_recovery_lock_released_when_scheduler_fails(tmp_path, monkeypatch):
    resume_path = tmp_path / "resume.txt"
    resume_path.write_text("python")
    monkeypatch.setattr(api, "LOGS_DIR", tmp_path / "logs")
    api.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    state = {
        "run_id": "abc124",
        "status": "running",
        "progress_pct": 10.0,
        "updated_at": _ts_ago(60),
        "resume_path": str(resume_path),
        "layers": [
            {"status": "ok", "meta": {}},
            {"status": "running", "meta": {}},
            {"status": "waiting", "meta": {}},
        ],
        "agent_log": [],
    }

    class _FailLoop:
        def create_task(self, coro):
            coro.close()
            raise RuntimeError("loop scheduling failed")

    class _OkLoop:
        def __init__(self):
            self.calls = 0

        def create_task(self, coro):
            self.calls += 1
            coro.close()

    persisted = []
    monkeypatch.setattr(api, "_persist_state", lambda _run_id: persisted.append(_run_id))

    monkeypatch.setattr(api.asyncio, "get_running_loop", lambda: _FailLoop())
    api._try_recover_stalled_run("abc124", state)
    assert not state.get("recovery_attempted_at")
    assert state.get("recovery_last_error")

    ok_loop = _OkLoop()
    monkeypatch.setattr(api.asyncio, "get_running_loop", lambda: ok_loop)

    async def _fake_run_pipeline(_run_id: str, _path: Path):
        return None

    monkeypatch.setattr(api, "run_pipeline", _fake_run_pipeline)
    api._try_recover_stalled_run("abc124", state)
    assert state.get("recovery_attempted_at")
    assert ok_loop.calls == 1
