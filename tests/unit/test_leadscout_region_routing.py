from careeragent.managers.leadscout_service import JobLead, LeadScoutService


def test_country_selector_defaults_to_us_sources() -> None:
    svc = LeadScoutService()
    regions = svc._resolve_locations({"geo_preferences": {"country_selector": "US", "locations": []}})
    providers = svc._source_rotation_for_region(regions[0])
    assert [provider["label"] for provider in providers[:8]] == [
        "LinkedIn",
        "Glassdoor",
        "Indeed",
        "ZipRecruiter",
        "MyVisaJobs",
        "Greenhouse",
        "Lever",
        "Google Jobs",
    ]


def test_country_selector_prioritizes_india_sources() -> None:
    svc = LeadScoutService()
    regions = svc._resolve_locations({"geo_preferences": {"country_selector": "IN", "locations": []}})
    providers = svc._source_rotation_for_region(regions[0])
    assert [provider["label"] for provider in providers[:8]] == [
        "LinkedIn",
        "Indeed",
        "Glassdoor",
        "ZipRecruiter",
        "Google Jobs",
        "Naukri",
        "Wellfound",
        "Monster",
    ]


def test_leadscout_dedupes_same_job_across_similar_titles() -> None:
    svc = LeadScoutService()
    raw = [
        JobLead(id="1", title="Senior AI Engineer", company="Acme", location="Remote", url="https://linkedin.com/jobs/view/1", source="linkedin"),
        JobLead(id="2", title="Senior AI Engineer", company="Acme", location="United States", url="https://indeed.com/viewjob?jk=1", source="indeed"),
        JobLead(id="3", title="Principal ML Engineer", company="Beta", location="Remote", url="https://jobs.lever.co/beta/1", source="lever"),
    ]
    out = svc._dedupe_similar_jobs(raw)
    assert len(out) == 2
