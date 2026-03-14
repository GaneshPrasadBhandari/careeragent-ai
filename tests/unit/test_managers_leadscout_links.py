from careeragent.managers.leadscout_service import (
    _curated_query_url,
    _infer_company_name,
    _is_plausible_job_link,
    _is_supported_mirror_board_url,
    _looks_like_blocked_portal_response,
    _normalize_result_url,
)
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


def test_backfill_curated_search_urls_reaches_target_count() -> None:
    svc = LeadScoutService(max_results_per_source=3)
    leads = [
        JobLead(id="1", title="AI Engineer", company="Acme", url="https://www.linkedin.com/jobs/view/1", description="ai ml")
    ]
    out = svc._backfill_curated_search_urls(
        leads,
        intent_plan={"target_roles": ["AI Engineer"], "keywords": ["python", "llm"], "geo_preferences": {"remote": True}},
        target_count=12,
    )
    assert len(out) >= 12
    assert any(x.source == "query_backfill" for x in out)


def test_hybrid_relevance_score_rewards_role_keyword_overlap() -> None:
    svc = LeadScoutService(max_results_per_source=3)
    lead = JobLead(
        id="x",
        title="Senior AI Engineer",
        company="Acme",
        url="https://example.com/job/x",
        description="python llm rag production systems",
        remote=True,
        posted_hours_ago=12,
    )
    score, reason = svc._hybrid_relevance_score(
        lead,
        {"target_roles": ["AI Engineer"], "keywords": ["python", "llm", "rag"]},
    )
    assert score > 0.45
    assert "role_hits" in reason


def test_curated_query_url_uses_valid_lever_host() -> None:
    url = _curated_query_url("jobs.lever.co", "platform+engineer")
    assert "google.com/search" in url
    assert "site%3Ajobs.lever.co" in url


def test_backfill_curated_search_urls_avoids_invalid_www_hosts() -> None:
    svc = LeadScoutService(max_results_per_source=3)
    out = svc._backfill_curated_search_urls(
        [],
        intent_plan={"target_roles": ["Platform Engineer"], "keywords": ["azure", "openai"], "geo_preferences": {"remote": True}},
        target_count=7,
    )
    backfilled = [x.url for x in out if x.source == "query_backfill"]
    assert any("jobs.lever.co" in u for u in backfilled)
    assert any("site%3Amyworkdayjobs.com" in u for u in backfilled)
    assert not any("www.jobs.lever.co" in u for u in backfilled)
    assert not any("www.myworkdayjobs.com" in u for u in backfilled)


def test_curated_query_url_uses_stable_endpoints_for_fragile_boards() -> None:
    glassdoor = _curated_query_url("glassdoor.com/Job/jobs.htm", "software+engineer")
    zipr = _curated_query_url("ziprecruiter.com/Jobs", "software+engineer")
    workday = _curated_query_url("myworkdayjobs.com", "platform+engineer")
    assert "/Job/index.htm?sc.keyword=" in glassdoor
    assert "/jobs-search?search=" in zipr
    assert "google.com/search" in workday and "site%3Amyworkdayjobs.com" in workday


def test_curated_query_url_avoids_double_jobs_path_for_ziprecruiter() -> None:
    zipr = _curated_query_url("ziprecruiter.com/Jobs", "platform+engineer")
    assert "jobs/jobs-search" not in zipr


def test_infer_company_name_extracts_from_title_suffix() -> None:
    company = _infer_company_name(
        title="Principal Data Scientist - Acme Labs",
        company_hint="",
        url="https://www.indeed.com/viewjob?jk=abc123",
    )
    assert company == "Acme Labs"


def test_infer_company_name_uses_tenant_slug_for_ats_hosts() -> None:
    company = _infer_company_name(
        title="AI Engineer",
        company_hint="",
        url="https://boards.greenhouse.io/paper/jobs/987654",
    )
    assert company == "Paper"


def test_supported_mirror_board_url_recognizes_supported_hosts() -> None:
    assert _is_supported_mirror_board_url("https://www.linkedin.com/jobs/view/123") is True
    assert _is_supported_mirror_board_url("https://example.com/careers/123") is False


def test_blocked_portal_detector_flags_access_denied_pages() -> None:
    assert _looks_like_blocked_portal_response(403, "Access denied by security policy") is True
    assert _looks_like_blocked_portal_response(200, "Regular job page content") is False
