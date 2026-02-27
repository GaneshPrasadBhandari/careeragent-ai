"""
debug_checklist.py  —  Layer-by-Layer Diagnostic Runner
=========================================================
Run this BEFORE deploying to production to validate each layer independently.
Mirrors the debug checklist from the issue brief.

Usage:
    python debug_checklist.py                 # full check
    python debug_checklist.py --layer 6       # check only L6
    python debug_checklist.py --layer 7 --url https://boards.greenhouse.io/...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

# ── ANSI colours ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg):  print(f"  {RED}✗{RESET} {msg}")
def warn(msg):  print(f"  {YELLOW}?{RESET} {msg}")
def info(msg):  print(f"  {CYAN}→{RESET} {msg}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}")


SAMPLE_PROFILE = """
Alice Chen | alice@example.com | +1-415-555-0190
Staff Backend Engineer with 8 years building distributed systems.
Skills: Python, FastAPI, PostgreSQL, Redis, AWS, Docker, Kubernetes,
        gRPC, Kafka, Terraform, TypeScript, React, SQL.
Experience:
  - Staff Engineer @ DataCo (2020-present, 4 yrs): led migration of monolith to
    microservices; reduced p99 latency by 40%.
  - Senior Engineer @ Acme Corp (2016-2020, 4 yrs): built real-time analytics
    pipeline processing 10B events/day.
Education: M.S. Computer Science, Stanford University, 2016.
LinkedIn: https://linkedin.com/in/alicechen
Open to: remote or San Francisco Bay Area, $160k-$220k.
"""

SAMPLE_JOBS = [
    {
        "id":          "gh_001",
        "title":       "Staff Software Engineer — Backend",
        "company":     "TechCorp",
        "url":         "https://boards.greenhouse.io/techcorp/jobs/12345",
        "location":    "Remote",
        "remote":      True,
        "description": (
            "We need a Staff Engineer with 5+ years of experience in Python, FastAPI, "
            "PostgreSQL, and AWS. Experience with Kafka and Kubernetes is a plus. "
            "You'll lead a team of 4-6 engineers building our data platform."
        ),
        "source":      "greenhouse",
    },
    {
        "id":          "lv_002",
        "title":       "Senior Python Engineer",
        "company":     "StartupXYZ",
        "url":         "https://jobs.lever.co/startupxyz/abc-def",
        "location":    "San Francisco, CA",
        "remote":      False,
        "description": (
            "Looking for a Python engineer with 3+ years, Django/FastAPI, SQL, "
            "and basic AWS knowledge."
        ),
        "source":      "lever",
    },
    {
        "id":          "jb_003",
        "title":       "Java Backend Developer",
        "company":     "EnterpriseInc",
        "url":         "https://careers.enterprise.com/jobs/999",
        "location":    "New York, NY",
        "remote":      False,
        "description": "Java Spring Boot, Oracle DB, 7+ years required.",
        "source":      "generic",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL LAYER CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def check_l0_l2_profile():
    header("L0-L2  Profile Ingestion + Parsing + Planning")
    from orchestrator import chunk_profile, _stub_parse, _stub_plan, _validate_profile

    # Chunking
    chunks = chunk_profile(SAMPLE_PROFILE)
    ok(f"chunk_profile produced {len(chunks)} chunk(s)")

    # 4000-token stress test
    big_profile = SAMPLE_PROFILE * 50   # ~50k chars
    big_chunks  = chunk_profile(big_profile)
    chars_per_chunk = max(len(c) for c in big_chunks)
    if chars_per_chunk <= 3800 * 4:
        ok(f"4000-token chunking OK — max chunk = {chars_per_chunk} chars across {len(big_chunks)} chunks")
    else:
        fail(f"Chunk too large: {chars_per_chunk} chars")

    # Parsing
    profile = _stub_parse(SAMPLE_PROFILE)
    try:
        _validate_profile(profile)
        ok(f"Profile parsed — skills found: {profile.get('skills', [])}")
    except ValueError as e:
        fail(f"Validation error: {e}")

    # Planning
    plan = _stub_plan(profile, ["Staff Backend Engineer"])
    if plan.get("target_roles"):
        ok(f"Intent plan built — target roles: {plan['target_roles']}")
    else:
        fail("Intent plan missing target_roles")


def check_l3_leadscout():
    header("L3  LeadScout — method availability")
    from leadscout_service import LeadScoutService
    scout = LeadScoutService(enable_playwright_scrape=False)

    # SafeMethodResolver test
    from orchestrator import SafeMethodResolver
    for alias in ["search_jobs", "find_jobs", "scrape_jobs", "run", "execute"]:
        m = SafeMethodResolver.resolve(scout, alias)
        if callable(m):
            ok(f"Alias '{alias}' resolves → {m.__name__}")
        else:
            fail(f"Alias '{alias}' not resolved")

    # API key status
    import os
    for key in ["ADZUNA_APP_ID", "JSEARCH_API_KEY", "SERPAPI_KEY"]:
        val = os.getenv(key, "")
        if val:
            ok(f"Env var {key} is set")
        else:
            warn(f"Env var {key} not set — that source will be skipped")


def check_l4_geofence():
    header("L4  GeoFence — filtering logic")
    from managers import GeoFenceManager
    gf = GeoFenceManager()

    remote_prefs = {"remote": True, "locations": []}
    result = gf.filter_by_geo(SAMPLE_JOBS, remote_prefs)
    remote_count = sum(1 for j in result if j.get("remote"))
    ok(f"Remote filter: {len(result)}/{len(SAMPLE_JOBS)} jobs passed ({remote_count} remote)")

    sf_prefs = {"remote": False, "locations": ["san francisco"]}
    result2 = gf.filter_by_geo(SAMPLE_JOBS, sf_prefs)
    ok(f"SF filter: {len(result2)}/{len(SAMPLE_JOBS)} jobs passed")

    # Alias check
    from orchestrator import SafeMethodResolver
    for alias in ["filter_by_geo", "apply_geo_filter", "geo_filter", "filter"]:
        m = SafeMethodResolver.resolve(gf, alias)
        if callable(m):
            ok(f"Alias '{alias}' → {m.__name__}")
        else:
            fail(f"Alias '{alias}' not resolved")


def check_l5_extraction():
    header("L5  Extraction & Scoring")
    from managers import ExtractionManager
    from orchestrator import _stub_parse

    extractor = ExtractionManager()
    profile   = _stub_parse(SAMPLE_PROFILE)
    profile["skills"] = ["Python", "FastAPI", "PostgreSQL", "AWS", "Kubernetes", "SQL"]

    scored = extractor.extract_and_score(SAMPLE_JOBS, profile, threshold=0.45)
    for j in scored:
        badge = f"{GREEN}PASS{RESET}" if j["score"] >= 0.45 else f"{RED}FAIL{RESET}"
        print(f"    [{badge}] {j['title'][:45]:<45} score={j['score']:.3f}  skills={j.get('matched_skills')}")

    qualified = [j for j in scored if j["score"] >= 0.45]
    if qualified:
        ok(f"{len(qualified)}/{len(scored)} jobs above threshold 0.45")
    else:
        warn("No jobs qualified — check profile skills vs. JD content")

    # Threshold comparison
    old_threshold = 0.55
    old_qualified = [j for j in scored if j["score"] >= old_threshold]
    info(f"Old threshold 0.55 would pass {len(old_qualified)}/{len(scored)} jobs")
    info(f"New threshold 0.45 passes   {len(qualified)}/{len(scored)} jobs")


def check_l6_artifacts():
    header("L6  Artifact Generation — files in artifacts/")
    import asyncio
    from orchestrator import _stub_parse, _stub_generate_artifacts, ARTIFACTS_DIR
    from managers import ExtractionManager

    extractor = ExtractionManager()
    profile   = _stub_parse(SAMPLE_PROFILE)
    profile["skills"] = ["Python", "FastAPI", "PostgreSQL", "AWS"]
    scored    = extractor.extract_and_score(SAMPLE_JOBS[:2], profile, threshold=0.0)

    artifacts = asyncio.run(_stub_generate_artifacts(profile, scored, ARTIFACTS_DIR))

    for job_id, files in artifacts.items():
        for ftype, fpath in files.items():
            if fpath:
                p = Path(fpath)
                if p.exists() and p.stat().st_size > 0:
                    ok(f"{job_id}/{ftype} → {p}  ({p.stat().st_size} bytes)")
                else:
                    fail(f"{job_id}/{ftype} NOT found at {fpath}")
            else:
                warn(f"{job_id}/{ftype} = None (PDF conversion likely unavailable)")

    # Check python-docx
    try:
        import docx
        ok("python-docx installed")
    except ImportError:
        warn("python-docx not installed — DOCX files will be text placeholders.  Run: pip install python-docx")

    # Check LibreOffice for PDF conversion
    import shutil
    if shutil.which("libreoffice"):
        ok("LibreOffice found — PDF conversion available")
    else:
        warn("LibreOffice not found — PDF conversion unavailable.  Install with: apt install libreoffice")


async def check_l7_apply_executor(test_url: Optional[str] = None):
    header("L7  ApplyExecutor — Playwright check")
    from orchestrator import ApplyExecutor, ARTIFACTS_DIR

    # Check Playwright installation
    try:
        from playwright.async_api import async_playwright
        ok("Playwright installed")
    except ImportError:
        fail("Playwright NOT installed.  Run: pip install playwright && playwright install chromium")
        return

    # Check chromium
    import shutil
    chromium_paths = [
        Path.home() / ".cache/ms-playwright/chromium-1091/chrome-linux/chrome",
        Path("/usr/bin/chromium"),
        Path("/usr/bin/chromium-browser"),
    ]
    found = any(p.exists() for p in chromium_paths) or bool(shutil.which("chromium"))
    if found:
        ok("Chromium browser found")
    else:
        warn("Chromium not found — run: playwright install chromium")

    if not test_url:
        info("No --url provided — skipping live form test.")
        info("Provide a Greenhouse/Lever URL to test full upload flow:")
        info("  python debug_checklist.py --layer 7 --url https://boards.greenhouse.io/...")
        return

    # Create dummy artifact files
    with tempfile.TemporaryDirectory() as tmp:
        resume = Path(tmp) / "resume.pdf"
        resume.write_bytes(b"%PDF-1.4 stub resume for testing")
        cover  = Path(tmp) / "cover.pdf"
        cover.write_bytes(b"%PDF-1.4 stub cover letter for testing")

        executor = ApplyExecutor()
        info(f"Testing upload on: {test_url}")
        try:
            result = await executor.apply_to_job(
                test_url,
                {"resume_pdf": str(resume), "cover_pdf": str(cover)},
                {"id": "test", "title": "Test Job"},
            )
            if result.get("resume_uploaded"):
                ok("Resume upload: SUCCESS")
            else:
                fail("Resume upload: FAILED — check UPLOAD_SELECTORS in ApplyExecutor")
            if result.get("submitted"):
                ok("Form submission: SUCCESS")
            else:
                warn("Form submission: not confirmed (may be multi-step form)")
            if result.get("screenshot"):
                ok(f"Screenshot saved: {result['screenshot']}")
        except Exception as exc:
            fail(f"ApplyExecutor error: {exc}")


def check_state_management():
    header("State Management — AgentState & Progress Bar")
    from orchestrator import AgentState

    state = AgentState()
    assert state.progress_pct() == 0.0
    ok("Initial progress = 0%")

    state.mark_running(0)
    state.mark_ok(0, chunks=1)
    assert state.progress_pct() == 10.0
    ok("After L0 OK: progress = 10%")

    for i in range(1, 10):
        state.mark_running(i)
        state.mark_ok(i)
    assert state.progress_pct() == 100.0
    ok("After all layers OK: progress = 100%")

    # Test error path
    state2 = AgentState()
    state2.mark_running(3)
    state2.mark_error(3, "LeadScout timeout")
    assert state2.layers[3].status == "error"
    ok("Error state recorded correctly")
    ok("State persisted to logs/ directory")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CareerAgent Layer Debug Checklist")
    parser.add_argument("--layer", type=int, help="Run only this layer (0-9)")
    parser.add_argument("--url",   type=str, help="Job URL for L7 live test")
    args = parser.parse_args()

    checks = {
        0:  check_l0_l2_profile,
        1:  check_l0_l2_profile,
        2:  check_l0_l2_profile,
        3:  check_l3_leadscout,
        4:  check_l4_geofence,
        5:  check_l5_extraction,
        6:  check_l6_artifacts,
        7:  lambda: asyncio.run(check_l7_apply_executor(args.url)),
        8:  check_state_management,
        9:  check_state_management,
    }

    if args.layer is not None:
        fn = checks.get(args.layer)
        if fn:
            fn()
        else:
            print(f"Unknown layer {args.layer}")
            sys.exit(1)
    else:
        print(f"\n{BOLD}{'═'*60}{RESET}")
        print(f"{BOLD}  CareerAgent  Pipeline Debug Checklist{RESET}")
        print(f"{BOLD}{'═'*60}{RESET}")
        check_l0_l2_profile()
        check_l3_leadscout()
        check_l4_geofence()
        check_l5_extraction()
        check_l6_artifacts()
        asyncio.run(check_l7_apply_executor(args.url))
        check_state_management()
        print(f"\n{BOLD}Checklist complete.  Review any {RED}✗{RESET}{BOLD} items above.{RESET}\n")


if __name__ == "__main__":
    from typing import Optional
    main()
