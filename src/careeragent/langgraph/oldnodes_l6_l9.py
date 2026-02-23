
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from careeragent.langgraph.state import CareerGraphState, GateEvent, utc_now
from careeragent.langgraph.tool_selector import ToolSelector
from careeragent.langgraph.tools import (
    MCPClient,
    ToolSettings,
    ToolResult,
    ollama_generate,
    serper_search,
    requests_scrape,
    firecrawl_scrape,
)


# ---------- helpers ----------
def _feed(layer: str, agent: str, msg: str) -> Dict[str, Any]:
    return {"live_feed": [{"layer": layer, "agent": agent, "message": msg}]}


def _threshold(state: CareerGraphState, key: str, default: float = 0.70) -> float:
    return float((state.get("thresholds") or {}).get(key, default))


def _gate(score: float, thresh: float, retries: int, max_retries: int) -> str:
    if score >= thresh:
        return "pass"
    if retries < max_retries:
        return "retry"
    return "hitl"


def _runs_dir(run_id: str) -> Path:
    # artifacts_root() exists in your repo; fallback to src/careeragent/artifacts if missing
    try:
        from careeragent.config import artifacts_root  # type: ignore
        root = artifacts_root()
    except Exception:
        root = Path("src/careeragent/artifacts").resolve()
    d = root / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _score_ats(text: str) -> float:
    t = (text or "").lower()
    score = 0.0
    if "summary" in t: score += 0.15
    if "skills" in t: score += 0.20
    if "experience" in t: score += 0.25
    if "education" in t: score += 0.10
    if "-" in text or "•" in text: score += 0.15
    if re.search(r"[\w\.-]+@[\w\.-]+\.\w+", t): score += 0.10
    return max(0.0, min(1.0, score))


def _draft_quality_score(resume_md: str, cover_md: str) -> Tuple[float, List[str]]:
    fb: List[str] = []
    ats = _score_ats(resume_md)
    tone = 0.7 if "Dear" in cover_md and "Sincerely" in cover_md else 0.4
    contact = 0.6 if re.search(r"[\w\.-]+@[\w\.-]+\.\w+", resume_md) else 0.3
    length = 0.7 if len(resume_md) > 900 else 0.4
    score = 0.40*ats + 0.25*tone + 0.20*contact + 0.15*length
    if ats < 0.6: fb.append("ATS structure weak: add headings and bullets.")
    if tone < 0.6: fb.append("Cover letter tone weak: add greeting + closing.")
    if contact < 0.5: fb.append("Contact missing: include email/phone/LinkedIn.")
    if length < 0.6: fb.append("Resume too short: add bullets with impact + tools.")
    return max(0.0, min(1.0, score)), fb


def _missing_skills_from_jobs(ranking: List[Dict[str, Any]]) -> List[str]:
    skills: List[str] = []
    for j in ranking[:30]:
        for s in (j.get("missing_skills") or j.get("missing_required_skills") or []):
            s2 = str(s).strip().lower()
            if s2 and s2 not in skills:
                skills.append(s2)
    return skills[:20]


# ============================================================
# L6 DRAFT NODE (3 tools): Local template -> Ollama -> MCP
# ============================================================
async def l6_draft_node(state: CareerGraphState) -> Dict[str, Any]:
    """
    Description: Generate ATS resume + cover letter bundle for ranked jobs with tool resilience.
    Layer: L6
    Input: state.profile + state.ranking
    Output: state.drafts + artifacts for each job
    """
    settings = ToolSettings()
    mcp = MCPClient(settings.MCP_SERVER_URL, settings.MCP_API_KEY)
    attempts = state.get("attempts", [])
    run_id = state.get("run_id") or "run"
    run_dir = _runs_dir(run_id)

    profile = state.get("profile") or {}
    ranking = state.get("ranking") or state.get("jobs_scored") or []
    prefs = state.get("preferences") or {}
    top_n = int(prefs.get("draft_count", 10))
    ranking = ranking[:top_n]

    name = str(profile.get("name") or "Candidate")
    contact = profile.get("contact") or {}
    email = str(contact.get("email") or "")
    phone = str(contact.get("phone") or "")
    linkedin = str(contact.get("linkedin") or "")
    github = str(contact.get("github") or "")
    skills = ", ".join((profile.get("skills") or [])[:25])

    base_resume = f"""# {name}
{email} | {phone} | {linkedin} | {github}

## Summary
AI/ML + GenAI builder focused on production-grade delivery (MLOps, evaluation, governance).

## Skills
{skills}

## Experience
- Add 4–6 bullets per role with measurable impact (metrics, scope, tools).

## Education
- (from intake)
"""

    def local_package(job: Dict[str, Any]) -> Dict[str, str]:
        title = str(job.get("title") or job.get("role_title") or "Role")
        company = str(job.get("company") or job.get("board") or job.get("source") or "Company")
        matched = ", ".join((job.get("matched_skills") or [])[:10])
        missing = ", ".join((job.get("missing_skills") or [])[:8])
        resume = base_resume + f"\n## Target Role Alignment\n- Target: {title} @ {company}\n- Matched keywords: {matched}\n- Gap keywords: {missing}\n"
        cover = f"""{name}
{email}

Dear Hiring Manager,

I’m applying for the {title} role at {company}. I bring production-grade AI/ML and GenAI experience, including reliable APIs, evaluation, and MLOps pipelines.

Aligned keywords: {matched}.

I’d welcome a quick conversation on how I can help {company} deliver measurable AI outcomes.

Sincerely,
{name}
"""
        return {"resume_md": resume, "cover_md": cover}

    drafts: List[Dict[str, Any]] = []
    artifacts_delta: Dict[str, Any] = {}

    for i, job in enumerate(ranking, start=1):
        jid = str(job.get("job_id") or job.get("url") or f"job_{i}")

        # --- Tool A: local template ---
        async def tool_a() -> ToolResult:
            pkg = local_package(job)
            conf = 0.65
            return ToolResult(ok=True, confidence=conf, data=pkg)

        # --- Tool B: Ollama ---
        async def tool_b() -> ToolResult:
            title = str(job.get("title") or "Role")
            company = str(job.get("company") or job.get("board") or "Company")
            jd = (job.get("full_text") or job.get("snippet") or "")[:3500]
            prompt = (
                "Generate ATS resume (markdown) and cover letter (markdown) tailored to this job.\n"
                "Return JSON with keys: resume_md, cover_md.\n\n"
                f"CANDIDATE_NAME: {name}\nCONTACT: {email} {phone} {linkedin} {github}\n"
                f"SKILLS: {(profile.get('skills') or [])[:25]}\n"
                f"JOB_TITLE: {title}\nCOMPANY: {company}\nJOB_DESC:\n{jd}\n"
            )
            r = await ollama_generate(settings, prompt)
            if not r.ok:
                return r
            try:
                j = json.loads(r.data.get("text") or "{}")
                resume_md = str(j.get("resume_md") or "")
                cover_md = str(j.get("cover_md") or "")
                conf = 0.70 if (len(resume_md) > 800 and len(cover_md) > 200) else 0.45
                return ToolResult(ok=True, confidence=conf, data={"resume_md": resume_md, "cover_md": cover_md})
            except Exception as e:
                return ToolResult(ok=False, confidence=0.0, error=str(e))

        # --- Tool C: MCP high-fidelity drafting ---
        async def tool_c() -> ToolResult:
            return await mcp.invoke(tool="draft.generate", payload={"profile": profile, "job": job})

        res = await ToolSelector.run(
            layer_id="L6",
            agent="DraftAgent",
            calls=[
                ("local.template_draft", None, tool_a),
                ("ollama.draft.generate", settings.OLLAMA_MODEL, tool_b),
                ("mcp.draft.generate", None, tool_c),
            ],
            min_conf=0.55,
            attempts_log=attempts,
        )

        if not res.ok:
            continue

        resume_md = str((res.data or {}).get("resume_md") or "")
        cover_md = str((res.data or {}).get("cover_md") or "")

        resume_path = Path(run_dir) / f"resume_{jid[:12]}.md"
        cover_path = Path(run_dir) / f"cover_{jid[:12]}.md"
        artifacts_delta[f"resume_{jid[:12]}"] = {"path": _safe_write(resume_path, resume_md), "content_type": "text/markdown"}
        artifacts_delta[f"cover_{jid[:12]}"] = {"path": _safe_write(cover_path, cover_md), "content_type": "text/markdown"}

        drafts.append({
            "job_id": jid,
            "title": job.get("title") or job.get("role_title"),
            "company": job.get("company") or job.get("board") or job.get("source"),
            "url": job.get("url") or job.get("link"),
            "resume_path": str(resume_path),
            "cover_path": str(cover_path),
            "missing_skills": job.get("missing_skills") or [],
        })

    bundle = {"drafts": drafts}
    return {
        "drafts": bundle,
        "attempts": attempts,
        "artifacts": artifacts_delta,
        **_feed("L6", "DraftAgent", f"Generated {len(drafts)} draft packages."),
    }


async def l6_evaluator_node(state: CareerGraphState) -> Dict[str, Any]:
    """
    Description: Evaluate draft quality with thresholds + retries.
    Layer: L6
    Input: drafts
    Output: gate event + HITL if needed
    """
    drafts = (state.get("drafts") or {}).get("drafts") or []
    if not drafts:
        # hard fail to HITL
        gate = GateEvent(layer_id="L6", target="draft", score=0.0, threshold=_threshold(state, "draft", 0.70),
                         decision="hitl", retries=int((state.get("layer_retry_count") or {}).get("L6", 0)),
                         feedback=["No drafts generated."], reasoning_chain=[], at_utc=utc_now())
        return {"gates": [gate], "status": "needs_human_approval", "pending_action": "review_drafts"}

    # evaluate first draft sample
    sample = drafts[0]
    rp = sample.get("resume_path")
    cp = sample.get("cover_path")
    resume_md = Path(rp).read_text(encoding="utf-8") if rp and Path(rp).exists() else ""
    cover_md = Path(cp).read_text(encoding="utf-8") if cp and Path(cp).exists() else ""

    score, fb = _draft_quality_score(resume_md, cover_md)
    retries = int((state.get("layer_retry_count") or {}).get("L6", 0))
    max_r = int(state.get("max_retries", 3))
    th = _threshold(state, "draft", 0.70)

    decision = _gate(score, th, retries, max_r)
    gate = GateEvent(layer_id="L6", target="draft", score=float(score), threshold=float(th), decision=decision,
                     retries=retries, feedback=fb, reasoning_chain=[], at_utc=utc_now())

    delta: Dict[str, Any] = {"gates": [gate], **_feed("L6", "EvaluatorAgent", f"Draft score={score:.2f} decision={decision}")}

    if decision == "retry":
        lrc = dict(state.get("layer_retry_count") or {})
        lrc["L6"] = retries + 1
        delta["layer_retry_count"] = lrc
        return delta

    if decision == "hitl":
        return {**delta, "status": "needs_human_approval", "pending_action": "review_drafts",
                "hitl_reason": "Draft quality below threshold", "hitl_payload": {"feedback": fb, "score": score, "threshold": th}}

    return delta


# ============================================================
# L7 APPLY NODE (3 tools): Simulated -> MCP -> Custom HTTP
# ============================================================
async def l7_apply_node(state: CareerGraphState) -> Dict[str, Any]:
    """
    Description: Apply executor (simulated by default; MCP optional).
    Layer: L7
    Input: drafts
    Output: meta.submissions + status
    """
    settings = ToolSettings()
    mcp = MCPClient(settings.MCP_SERVER_URL, settings.MCP_API_KEY)
    attempts = state.get("attempts", [])
    prefs = state.get("preferences") or {}

    drafts = (state.get("drafts") or {}).get("drafts") or []
    if not drafts:
        return {"status": "needs_human_approval", "pending_action": "review_drafts", **_feed("L7", "ApplyExecutor", "No drafts to apply.")}

    # apply top K
    k = int(prefs.get("apply_count", 5))
    to_apply = drafts[:k]

    submissions: List[Dict[str, Any]] = []
    for d in to_apply:
        jid = str(d.get("job_id"))
        url = str(d.get("url") or "")

        # Tool A: simulated apply
        async def tool_a() -> ToolResult:
            return ToolResult(ok=True, confidence=0.80, data={"submission_id": f"sim_{jid[:10]}", "status": "Applied", "url": url})

        # Tool B: MCP apply
        async def tool_b() -> ToolResult:
            return await mcp.invoke(tool="apply.submit", payload={"job": d, "preferences": prefs})

        # Tool C: custom HTTP apply gateway (placeholder)
        async def tool_c() -> ToolResult:
            # you can swap this to your own apply microservice
            return ToolResult(ok=False, confidence=0.0, error="custom apply gateway not configured")

        res = await ToolSelector.run(
            layer_id="L7",
            agent="ApplyExecutor",
            calls=[
                ("local.simulated_apply", None, tool_a),
                ("mcp.apply.submit", None, tool_b),
                ("custom.apply.gateway", None, tool_c),
            ],
            min_conf=0.55,
            attempts_log=attempts,
        )

        if res.ok:
            submissions.append({"job_id": jid, **(res.data or {})})

    meta = dict(state.get("meta") or {})
    meta.setdefault("submissions", [])
    meta["submissions"].extend(submissions)

    return {"meta": meta, "attempts": attempts, **_feed("L7", "ApplyExecutor", f"Submitted {len(submissions)} applications.")}


async def l7_evaluator_node(state: CareerGraphState) -> Dict[str, Any]:
    """
    Description: Evaluate apply stage success.
    Layer: L7
    """
    meta = state.get("meta") or {}
    subs = meta.get("submissions") or []
    score = 0.80 if len(subs) >= 1 else 0.0
    fb = [] if subs else ["No submissions created. Check apply tool credentials or use simulation."]
    retries = int((state.get("layer_retry_count") or {}).get("L7", 0))
    th = _threshold(state, "apply", 0.70)
    decision = _gate(score, th, retries, int(state.get("max_retries", 3)))
    gate = GateEvent(layer_id="L7", target="apply", score=float(score), threshold=float(th), decision=decision,
                     retries=retries, feedback=fb, reasoning_chain=[], at_utc=utc_now())
    if decision == "hitl":
        return {"gates": [gate], "status": "needs_human_approval", "pending_action": "apply_failed",
                "hitl_reason": "Apply stage failed", "hitl_payload": {"feedback": fb, "score": score, "threshold": th}}
    return {"gates": [gate], **_feed("L7", "EvaluatorAgent", f"Apply score={score:.2f} decision={decision}")}


# ============================================================
# L8 TRACKER NODE (3 tools): SQLite/meta -> MCP -> file log
# ============================================================
async def l8_tracker_node(state: CareerGraphState) -> Dict[str, Any]:
    """
    Description: Update application tracking statuses.
    Layer: L8
    Input: meta.submissions
    Output: meta.applied_jobs + statuses
    """
    settings = ToolSettings()
    mcp = MCPClient(settings.MCP_SERVER_URL, settings.MCP_API_KEY)
    attempts = state.get("attempts", [])

    meta = dict(state.get("meta") or {})
    subs = meta.get("submissions") or []

    # Tool A: local tracker
    async def tool_a() -> ToolResult:
        applied = []
        for s in subs:
            applied.append({"job_id": s.get("job_id"), "status": s.get("status","Applied"), "url": s.get("url")})
        return ToolResult(ok=True, confidence=0.75, data={"applied_jobs": applied})

    # Tool B: MCP tracker update
    async def tool_b() -> ToolResult:
        return await mcp.invoke(tool="tracker.sync", payload={"submissions": subs})

    # Tool C: file-based tracker fallback
    async def tool_c() -> ToolResult:
        run_id = state.get("run_id") or "run"
        path = _runs_dir(run_id) / "applied_jobs.json"
        path.write_text(json.dumps({"applied_jobs": subs}, indent=2), encoding="utf-8")
        return ToolResult(ok=True, confidence=0.55, data={"applied_jobs": subs, "path": str(path)})

    res = await ToolSelector.run(
        layer_id="L8",
        agent="TrackerAgent",
        calls=[
            ("local.tracker", None, tool_a),
            ("mcp.tracker.sync", None, tool_b),
            ("file.tracker", None, tool_c),
        ],
        min_conf=0.55,
        attempts_log=attempts,
    )

    if res.ok:
        meta["applied_jobs"] = (res.data or {}).get("applied_jobs") or subs

    return {"meta": meta, "attempts": attempts, **_feed("L8", "TrackerAgent", "Tracking updated.")}


async def l8_evaluator_node(state: CareerGraphState) -> Dict[str, Any]:
    meta = state.get("meta") or {}
    applied = meta.get("applied_jobs") or []
    score = 0.80 if len(applied) >= 1 else 0.0
    fb = [] if applied else ["No applied jobs tracked yet."]
    th = _threshold(state, "tracker", 0.70)
    retries = int((state.get("layer_retry_count") or {}).get("L8", 0))
    decision = _gate(score, th, retries, int(state.get("max_retries", 3)))
    gate = GateEvent(layer_id="L8", target="tracker", score=float(score), threshold=float(th), decision=decision,
                     retries=retries, feedback=fb, reasoning_chain=[], at_utc=utc_now())
    if decision == "hitl":
        return {"gates":[gate], "status":"needs_human_approval", "pending_action":"tracker_issue",
                "hitl_reason":"Tracker below threshold", "hitl_payload":{"feedback":fb,"score":score,"threshold":th}}
    return {"gates":[gate], **_feed("L8","EvaluatorAgent", f"Tracker score={score:.2f} decision={decision}")}


# ============================================================
# L9 ANALYTICS + BRIDGE DOCS (3 tools): Local -> Ollama -> ReportLab
# ============================================================
async def l9_analytics_node(state: CareerGraphState) -> Dict[str, Any]:
    """
    Description: Produce milestone report JSON + (optional) PDF + bridge docs for missing skills.
    Layer: L9
    """
    settings = ToolSettings()
    mcp = MCPClient(settings.MCP_SERVER_URL, settings.MCP_API_KEY)
    attempts = state.get("attempts", [])
    run_id = state.get("run_id") or "run"
    run_dir = _runs_dir(run_id)

    ranking = state.get("ranking") or []
    missing = _missing_skills_from_jobs(ranking)

    # Tool A: Serper learning links (bridge docs)
    async def bridge_a() -> ToolResult:
        if not settings.SERPER_API_KEY or not missing:
            return ToolResult(ok=True, confidence=0.55, data={})
        plan: Dict[str, Any] = {}
        for sk in missing[:12]:
            yt = await serper_search(settings, f"{sk} tutorial youtube", num=4)
            docs = await serper_search(settings, f"{sk} official documentation", num=4)
            plan[sk] = {
                "youtube": yt.data if yt.ok else [],
                "docs": docs.data if docs.ok else [],
            }
        return ToolResult(ok=True, confidence=0.70, data=plan)

    # Tool B: MCP learning links
    async def bridge_b() -> ToolResult:
        return await mcp.invoke(tool="learning.bridge_docs", payload={"skills": missing})

    # Tool C: static fallback
    async def bridge_c() -> ToolResult:
        plan = {sk: {"youtube": [], "docs": [], "note": "Add SERPER_API_KEY or MCP for links"} for sk in missing[:12]}
        return ToolResult(ok=True, confidence=0.55, data=plan)

    bridge = await ToolSelector.run(
        layer_id="L9",
        agent="LearningBridgeAgent",
        calls=[
            ("serper.learning_links", None, bridge_a),
            ("mcp.learning.bridge_docs", None, bridge_b),
            ("static.learning_fallback", None, bridge_c),
        ],
        min_conf=0.55,
        attempts_log=attempts,
    )

    bridge_docs = bridge.data if bridge.ok else {}

    # Tool A: local analytics JSON
    async def tool_a() -> ToolResult:
        meta = state.get("meta") or {}
        subs = meta.get("submissions") or []
        report = {
            "run_id": run_id,
            "top_scores": [float(j.get("score", 0.0)) for j in (state.get("ranking") or [])[:10]],
            "submissions": subs,
            "missing_skills_top": missing[:12],
            "tool_audit_count": len(state.get("attempts") or []),
        }
        return ToolResult(ok=True, confidence=0.75, data=report)

    # Tool B: Ollama narrative summary
    async def tool_b() -> ToolResult:
        prompt = (
            "Create a concise milestone report summary for a job-hunt automation run. "
            "Include: what was done, why the top jobs were selected, key skill gaps.\n\n"
            f"TOP_MISSING_SKILLS: {missing[:12]}\n"
            f"RANKING_TOP3: {(state.get('ranking') or [])[:3]}\n"
        )
        return await ollama_generate(settings, prompt)

    # Tool C: ReportLab PDF (best-effort; if missing, return low-conf but ok)
    async def tool_c() -> ToolResult:
        try:
            from reportlab.lib.pagesizes import LETTER
            from reportlab.pdfgen import canvas
        except Exception as e:
            return ToolResult(ok=True, confidence=0.55, data={"pdf": None, "note": "reportlab not installed"}, error=str(e))

        pdf_path = run_dir / "milestone_report.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=LETTER)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, 750, "CareerAgent-AI — Milestone Report")
        c.setFont("Helvetica", 10)
        c.drawString(72, 730, f"Run ID: {run_id}")
        y = 710
        c.setFont("Helvetica-Bold", 11)
        c.drawString(72, y, "Top Missing Skills")
        y -= 14
        c.setFont("Helvetica", 10)
        for sk in missing[:12]:
            c.drawString(80, y, f"- {sk}")
            y -= 12
            if y < 80:
                c.showPage()
                y = 750
        c.showPage()
        c.save()
        return ToolResult(ok=True, confidence=0.75, data={"pdf": str(pdf_path)})

    base = await ToolSelector.run(
        layer_id="L9",
        agent="AnalyticsAgent",
        calls=[
            ("local.analytics_json", None, tool_a),
            ("ollama.analytics_summary", settings.OLLAMA_MODEL, tool_b),
            ("reportlab.pdf_report", None, tool_c),
        ],
        min_conf=0.55,
        attempts_log=attempts,
    )

    report_json = base.data if isinstance(base.data, dict) else {}
    summary_text = ""
    if isinstance(base.data, dict) and "text" in base.data:
        summary_text = str(base.data.get("text") or "")

    # Persist JSON report
    report_path = run_dir / "milestone_report.json"
    report_path.write_text(json.dumps({"report": report_json, "summary": summary_text, "bridge_docs": bridge_docs}, indent=2), encoding="utf-8")

    artifacts = {
        "milestone_report_json": {"path": str(report_path), "content_type": "application/json"},
    }
    if isinstance(base.data, dict) and base.data.get("pdf"):
        artifacts["milestone_report_pdf"] = {"path": str(base.data["pdf"]), "content_type": "application/pdf"}

    return {
        "bridge_docs": bridge_docs,
        "attempts": attempts,
        "artifacts": artifacts,
        **_feed("L9", "AnalyticsAgent", "Analytics + Bridge Docs generated."),
    }
