from careeragent.api.approval_utils import pick_approved_jobs, qualified_from_state


def test_pick_approved_jobs_matches_multiple_identifier_formats() -> None:
    ranked = [
        {"id": "a1", "job_id": "x1", "url": "https://example.com/1", "title": "Eng", "company": "Acme"},
        {"id": "a2", "url": "https://example.com/2", "title": "DS", "company": "Beta"},
    ]

    assert len(pick_approved_jobs(ranked, [])) == 2
    assert pick_approved_jobs(ranked, ["a1"])[0]["id"] == "a1"
    assert pick_approved_jobs(ranked, ["x1"])[0]["id"] == "a1"
    assert pick_approved_jobs(ranked, ["https://example.com/2"])[0]["id"] == "a2"
    assert pick_approved_jobs(ranked, ["Eng|Acme"])[0]["id"] == "a1"


def test_pick_approved_jobs_matches_sanitized_redirect_urls() -> None:
    ranked = [
        {
            "id": "a1",
            "url": "https://jobs.example.com/redirect?url=https%3A%2F%2Fwww.linkedin.com%2Fjobs%2Fview%2F123%3Futm_source%3Dfoo&trk=public_jobs",
            "direct_job_url": "https://www.linkedin.com/jobs/view/123?utm_campaign=abc",
            "title": "AI Engineers",
            "company": "Acme",
        }
    ]

    approved = pick_approved_jobs(ranked, ["https://www.linkedin.com/jobs/view/123"])
    assert approved[0]["id"] == "a1"


def test_qualified_from_state_fallback_order() -> None:
    assert qualified_from_state({"approved_jobs": [{"id": "approved"}]}) == [{"id": "approved"}]

    state = {
        "approved_jobs": [],
        "layer_debug": {"L5": {"qualified_jobs": [{"id": "qualified"}]}},
    }
    assert qualified_from_state(state) == [{"id": "qualified"}]

    state = {
        "approved_jobs": [],
        "layer_debug": {"L5": {"qualified_jobs": []}},
        "scored_jobs": [{"id": "s1"}, {"id": "s2"}, {"id": "s3"}],
    }
    assert qualified_from_state(state) == [{"id": "s1"}, {"id": "s2"}, {"id": "s3"}]


def test_pick_approved_jobs_matches_q_redirect_and_application_url_shapes() -> None:
    ranked = [
        {
            "id": "a3",
            "redirect_url": "https://jobs.example.com/out?q=https%3A%2F%2Fboards.greenhouse.io%2Facme%2Fjobs%2F789%3Fgh_src%3Dtracker",
            "application_url": "www.boards.greenhouse.io/acme/jobs/789?utm_source=tracker",
            "title": "ML Engineer",
            "company": "Acme",
        }
    ]

    approved = pick_approved_jobs(ranked, ["https://boards.greenhouse.io/acme/jobs/789"])
    assert approved[0]["id"] == "a3"
