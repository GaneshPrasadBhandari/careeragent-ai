from careeragent.managers.leadscout_service import LeadScoutService


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
    assert [provider["label"] for provider in providers[:3]] == ["Naukri", "Wellfound", "Monster"]
