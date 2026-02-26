from careeragent.api.run_manager_service import RunManagerService
from careeragent.core.settings import Settings
from careeragent.core.state import AgentState, Preferences


def test_reject_drafts_routes_back_to_l6_with_feedback(tmp_path, monkeypatch) -> None:
    settings = Settings(DATABASE_URL=f"sqlite:///{tmp_path}/test.db")
    rm = RunManagerService(settings)

    state = AgentState(run_id="run_test", status="needs_human_approval", pending_action="review_drafts", preferences=Preferences())
    rm.store.save(state)

    called = {"phase2": False}

    def _fake_spawn_phase2(run_id: str) -> None:
        called["phase2"] = run_id == "run_test"

    monkeypatch.setattr(rm, "_spawn_phase2", _fake_spawn_phase2)

    out = rm.handle_action("run_test", {"action_type": "reject_drafts", "payload": {"reason": "Need tighter ATS wording"}})
    assert out["ok"] is True

    updated = rm.store.load("run_test")
    assert updated is not None
    assert updated.status == "running"
    assert updated.pending_action is None
    assert updated.meta.get("re_draft_flag") is True
    assert updated.meta.get("draft_revision_feedback") == "Need tighter ATS wording"
    assert called["phase2"] is True
