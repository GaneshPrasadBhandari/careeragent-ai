import json

from careeragent.api import main as api_main


def test_resolve_action_request_accepts_legacy_and_nested_payload_shapes():
    action, payload = api_main._resolve_action_request(  # type: ignore[attr-defined]
        {
            "action_type": "approve_ranking",
            "payload": {"selected_job_ids": ["job-1", "job-2"]},
        }
    )
    assert action == "approve_ranking"
    assert payload["selected_job_ids"] == ["job-1", "job-2"]


def test_load_run_state_from_disk_hydrates_missing_memory_state(tmp_path, monkeypatch):
    run_id = "run_disk_reload"
    monkeypatch.setattr(api_main, "LOGS_DIR", tmp_path)
    api_main._runs.pop(run_id, None)
    payload = {"run_id": run_id, "status": "pending_human_input", "pending_action": "approve_ranking"}
    (tmp_path / f"state_{run_id}.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = api_main._load_run_state_from_disk(run_id)  # type: ignore[attr-defined]

    assert loaded == payload
    assert api_main._runs[run_id]["pending_action"] == "approve_ranking"
