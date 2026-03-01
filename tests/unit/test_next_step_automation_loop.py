from careeragent.agents.analytics_service import AnalyticsService
from careeragent.agents.feedback_eval_service import FeedbackEvaluatorService, FeedbackItem
from careeragent.agents.leadscout_service import JobLead, LeadScoutService
from careeragent.core.state import AgentState, Preferences


def test_feedback_employer_outcome_updates_learning_loop() -> None:
    state = AgentState(run_id="r-loop", preferences=Preferences())
    svc = FeedbackEvaluatorService()
    item = FeedbackItem(
        source="employer",
        text="Thank you for applying. We would like to schedule an interview for next round.",
        meta={"job_url": "https://jobs.example.com/123", "company": "Example"},
    )
    result = svc.ingest(state=state, item=item)
    assert result.stored is True
    assert state.meta["learning_loop"]["interview"] == 1
    assert state.meta["status_updates"][-1]["status"] == "interviewing"


def test_leadscout_source_quota_prefers_diverse_sources() -> None:
    scout = LeadScoutService(max_results_per_source=3, enable_playwright_scrape=False)
    leads = [
        JobLead(id="1", title="A", company="C", url="https://www.linkedin.com/jobs/view/1"),
        JobLead(id="2", title="A", company="C", url="https://www.linkedin.com/jobs/view/2"),
        JobLead(id="3", title="A", company="C", url="https://www.indeed.com/viewjob?jk=1"),
        JobLead(id="4", title="A", company="C", url="https://boards.greenhouse.io/acme/jobs/1"),
        JobLead(id="5", title="A", company="C", url="https://jobs.lever.co/acme/1"),
    ]
    out = scout._enforce_source_quotas(leads, quota_targets=scout._build_source_quota_targets())
    counts = scout._count_sources(out)
    assert any("linkedin.com" in k for k in counts)
    assert any("indeed.com" in k for k in counts)
    assert any("greenhouse.io" in k for k in counts)


def test_analytics_includes_funnel_audit() -> None:
    st = AgentState(run_id="r-funnel", preferences=Preferences())
    st.jobs_raw = [{"url": "a"}, {"url": "b"}, {"url": "c"}]
    st.ranking = [{"url": "a"}, {"url": "b"}]
    st.approved_job_urls = ["a", "b"]
    st.meta["apply_attempts"] = [
        {"job_url": "a", "status": "submitted", "reason": "ok"},
        {"job_url": "b", "status": "skipped", "reason": "captcha required"},
    ]
    st.meta["submissions"] = {"sub_1": {"job_id": "a"}}
    st.meta["source_health"] = {
        "source_counts": {"linkedin.com": 2},
        "source_errors": {"_search_tavily": 1},
        "source_quota_targets": {"linkedin.com": 2},
    }

    report = AnalyticsService().build_report(orchestration_state=st)
    assert report.funnel_audit.discovered == 3
    assert report.funnel_audit.attempted == 2
    assert report.funnel_audit.submitted == 1
    assert report.funnel_audit.blocker_taxonomy["anti_bot_captcha"] == 1
    assert report.source_telemetry.source_counts["linkedin.com"] == 2
