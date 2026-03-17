import json
from pathlib import Path

from careeragent.managers.leadscout_service import LeadScoutService
import careeragent.managers.leadscout_service as l3


def test_cached_injection_returns_requested_limit(tmp_path, monkeypatch):
    payload = [
        {"id": f"job{i}", "title": "AI Engineer", "company": "ACME", "url": f"https://example.com/jobs/{i}"}
        for i in range(1, 25)
    ]
    cache_file = tmp_path / 'cached_jobs.json'
    cache_file.write_text(json.dumps(payload), encoding='utf-8')
    monkeypatch.setattr(l3, 'CACHED_JOBS_FILE', cache_file)

    service = LeadScoutService(enable_playwright_scrape=False)
    jobs = service._load_cached_jobs({"target_roles": ["AI Engineer"], "keywords": ["python"]}, limit=10)
    assert len(jobs) == 10
    assert all(j.source == 'cached_jobs' for j in jobs)
