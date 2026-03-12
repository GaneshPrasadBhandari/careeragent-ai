from careeragent.managers.leadscout_service import _is_plausible_job_link, _normalize_result_url
from careeragent.managers.leadscout_service import JobLead, LeadScoutService


def test_normalize_google_redirect_url() -> None:
    url = "https://www.google.com/url?q=https://www.linkedin.com/jobs/view/12345/&sa=U"
    assert _normalize_result_url(url) == "https://www.linkedin.com/jobs/view/12345"


def test_filters_non_job_board_search_pages() -> None:
    assert not _is_plausible_job_link("https://boards.greenhouse.io/search#t=ml")
    assert _is_plausible_job_link("https://boards.greenhouse.io/company/jobs/123456")


def test_normalize_indeed_url_keeps_job_key() -> None:
    url = "https://www.indeed.com/viewjob?jk=abc123&from=serp"
    assert _normalize_result_url(url) == "https://www.indeed.com/viewjob?jk=abc123"


def test_filter_by_recency_removes_old_postings() -> None:
    svc = LeadScoutService(max_results_per_source=5)
    leads = [
        JobLead(id="1", title="A", company="X", url="https://www.indeed.com/viewjob?jk=1", posted_date="2 hours ago"),
        JobLead(id="2", title="B", company="Y", url="https://www.linkedin.com/jobs/view/2", posted_date="2 weeks ago"),
    ]
    out = svc._filter_by_recency(leads, recency_hours=48)
    assert len(out) == 1
    assert out[0].id == "1"


def test_parse_posted_datetime_supports_iso8601() -> None:
    svc = LeadScoutService(max_results_per_source=5)
    dt = svc._parse_posted_datetime("2026-03-10T12:30:00Z")
    assert dt is not None
    assert dt.year == 2026


def test_role_filter_keeps_target_titles_only() -> None:
    svc = LeadScoutService(max_results_per_source=5)
    leads = [
        JobLead(id="1", title="AI Engineer", company="Acme", url="https://example.com/job/1", description="genai platform"),
        JobLead(id="2", title="Backend Software Engineer", company="Acme", url="https://example.com/job/2", description="distributed systems"),
        JobLead(id="3", title="Principal Data Scientist", company="Acme", url="https://example.com/job/3", description="llm and ml"),
    ]
    out = svc._filter_by_role_relevance(
        leads,
        intent_plan={"target_roles": ["AI Engineer", "Principal Data Scientist"]},
    )
    ids = {x.id for x in out}
    assert "1" in ids
    assert "3" in ids
    assert "2" not in ids
