from careeragent.managers.leadscout_service import _is_plausible_job_link, _normalize_result_url


def test_normalize_google_redirect_url() -> None:
    url = "https://www.google.com/url?q=https://www.linkedin.com/jobs/view/12345/&sa=U"
    assert _normalize_result_url(url) == "https://www.linkedin.com/jobs/view/12345"


def test_filters_non_job_board_search_pages() -> None:
    assert not _is_plausible_job_link("https://boards.greenhouse.io/search#t=ml")
    assert _is_plausible_job_link("https://boards.greenhouse.io/company/jobs/123456")


def test_normalize_indeed_url_keeps_job_key() -> None:
    url = "https://www.indeed.com/viewjob?jk=abc123&from=serp"
    assert _normalize_result_url(url) == "https://www.indeed.com/viewjob?jk=abc123"
