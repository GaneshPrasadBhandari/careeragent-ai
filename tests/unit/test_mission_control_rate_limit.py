from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import app.ui.mission_control as mission_control


class DummyResponse:
    def __init__(self, status_code: int, payload=None, text: str = "", headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._payload


def test_api_start_hunt_retries_rate_limit(monkeypatch):
    session_state = {"start_hunt_error": None}
    errors = []

    monkeypatch.setattr(mission_control, "st", SimpleNamespace(session_state=session_state, error=errors.append))

    calls = {"count": 0}

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return DummyResponse(429, text="Too Many Requests", headers={"Retry-After": "1"})
        return DummyResponse(200, payload={"run_id": "run-123"})

    sleeps = []
    monkeypatch.setattr(mission_control.requests, "post", fake_post)
    monkeypatch.setattr(mission_control.time, "sleep", lambda seconds: sleeps.append(seconds))

    run_id = mission_control._api_start_hunt("https://api.example.com", b"resume", "resume.pdf", {})

    assert run_id == "run-123"
    assert calls["count"] == 2
    assert sleeps == [1]
    assert errors == []
    assert session_state["start_hunt_error"] is None


def test_api_get_status_uses_cached_state_on_rate_limit(monkeypatch):
    cached = {"status": "running", "progress_pct": 20}
    session_state = {"run_status": cached, "last_poll": 0.0, "backend_warning": None}

    monkeypatch.setattr(mission_control, "st", SimpleNamespace(session_state=session_state, warning=lambda _: None))
    monkeypatch.setattr(
        mission_control.requests,
        "get",
        lambda *args, **kwargs: DummyResponse(429, text="Too Many Requests", headers={"Retry-After": "7"}),
    )
    monkeypatch.setattr(mission_control.time, "time", lambda: 100.0)

    status = mission_control._api_get_status("https://api.example.com", "run-123")

    assert status == cached
    assert session_state["backend_warning"] == "Backend is throttling status checks. Waiting 7s before polling again."
    assert session_state["last_poll"] == 106.0
