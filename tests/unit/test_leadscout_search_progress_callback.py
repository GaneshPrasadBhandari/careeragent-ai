import asyncio

from careeragent.managers.leadscout_service import JobLead, LeadScoutService


def test_search_jobs_reports_mock_batch_to_progress_callback(monkeypatch) -> None:
    service = LeadScoutService(max_results_per_source=3, enable_playwright_scrape=False)

    async def fake_run_source_tasks(source_coros, timeout_seconds, batch_progress_callback=None, diagnostics=None):
        for coro in source_coros:
            try:
                coro.close()
            except Exception:
                pass
        lead = JobLead(
            id="job-1",
            title="ML Engineer",
            company="Acme",
            url="https://www.linkedin.com/jobs/view/123",
            source="serper_organic",
            posted_date="1 hour ago",
        )
        if batch_progress_callback:
            batch_progress_callback(1, [lead])
        return [[lead]]

    async def fake_validate(leads, serper_key=""):
        return leads

    monkeypatch.setattr(service, "_llm_expand_queries", lambda *args, **kwargs: [])
    monkeypatch.setattr(service, "_rank_leads_hybrid", lambda leads, intent: leads)
    monkeypatch.setattr(service, "_backfill_curated_search_urls", lambda leads, intent_plan, target_count: leads)
    monkeypatch.setattr(service, "_run_source_tasks", fake_run_source_tasks)
    monkeypatch.setattr(service, "_validate_and_retry_links", fake_validate)

    events = []

    def progress_callback(total, batch):
        events.append((total, batch))

    result = asyncio.run(
        service.search_jobs(
            {
                "target_roles": ["ML Engineer"],
                "keywords": ["python"],
                "geo_preferences": {"locations": ["United States"], "remote": True},
            },
            progress_callback=progress_callback,
        )
    )

    assert result
    assert events
    assert events[0][0] >= 1
    assert len(events[0][1]) >= 1



def test_backfill_curated_search_urls_can_reach_large_target_without_hanging():
    service = LeadScoutService(max_results_per_source=25, enable_playwright_scrape=False)
    leads = service._backfill_curated_search_urls(
        [],
        intent_plan={
            "target_roles": ["AI Engineer"],
            "keywords": ["Python"],
            "geo_preferences": {"remote": True},
        },
        target_count=80,
    )

    assert len(leads) == 80
    assert len({lead.url for lead in leads}) == 80
    assert all(lead.source == "query_backfill" for lead in leads)
