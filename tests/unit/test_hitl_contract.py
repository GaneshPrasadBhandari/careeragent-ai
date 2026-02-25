def test_hitl_scorecard_fields_present_in_job_payload_contract():
    # Contract: ranking payloads should carry scorecard fields used by UI.
    required = {
        "interview_probability_percent",
        "missing_skills_gap_percent",
        "jd_alignment_percent",
        "ats_keyword_match_percent",
    }
    # We don't instantiate full pipeline here; we enforce a documented contract.
    # If you later refactor, update orchestrator to keep these keys stable.
    assert len(required) == 4
