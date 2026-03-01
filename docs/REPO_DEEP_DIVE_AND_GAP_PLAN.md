# CareerAgent-AI Deep Dive (Current Reality vs Blueprint)

## What the app currently does (verified from code)

- **Frontend (Streamlit mission control)**
  - Launches a hunt run, uploads resume, polls backend status, renders L0–L9 pipeline progress, evaluator/HITL controls, job board, and analytics tab.
- **Backend (FastAPI)**
  - Starts asynchronous runs via `/hunt/start`.
  - Simulates L0–L9 workflow with resume parse, job discovery/match scoring, approval gates, drafting artifacts, apply queue, tracking, and analytics summary.
  - Exposes run status endpoints, jobs endpoint, and artifact links.
- **Automation currently in place**
  - Resume parsing + skill extraction.
  - Job lead scoring + ranking.
  - Human approval loops for ranking and drafts.
  - Draft generation for tailored resume and cover letter artifacts.
  - Apply stage + tracking payload creation (simulated execution metadata).

## What was missing before this patch

- In-dashboard visibility of **which model/provider is used** for ATS writing, parsing, and reasoning.
- Analytics dashboard lacked **application-level detail** (company, URL, timestamp, channel, next action).
- No clear panel for **follow-up emails**, **interview queue**, or **notification logs**.
- LangGraph trace link visibility was missing/unclear if env vars were not configured.

## What this patch implements

1. **LLM/tool stack transparency**
   - Captures and exposes per-purpose LLM metadata in run state (`llm_stack`) for ATS writing, parser, and ranking reasoner.

2. **LangGraph + LangSmith run link diagnostics**
   - Adds `langgraph` state object with a direct run URL when configured.
   - Keeps explicit note when URL/env vars are missing.

3. **Richer application tracking**
   - Apply results now include:
     - company/title/job_id
     - apply URL
     - status (queued/submitted)
     - channel (auto_apply vs draft_email_review)
     - timestamps + next action + follow-up due time

4. **Follow-up + interview queues**
   - Adds `interviews` and `followup_queue` structures to run state.

5. **Notification delivery telemetry**
   - Adds `notification_log` captured during L7.
   - Uses existing notification service in dry-run mode unless provider credentials are configured.

6. **Analytics dashboard upgrade**
   - New analytics UI sections:
     - LLM/tool/model table
     - LangSmith/LangGraph trace links
     - application tracking table
     - interview queue table
     - employer follow-up queue
     - notification delivery log

## Remaining for production-complete blueprint

- Real job-board auto-apply integrations (platform-specific auth + anti-bot + legal constraints).
- Real outbound Gmail/Twilio sending in non-dry mode with secure key management.
- Real bidirectional inbox/calendar integrations (read replies + create Google Calendar events).
- Durable relational schema for applications/interviews/followups across sessions/users (beyond in-memory + log snapshots).
- Strong multi-user auth, RBAC, and tenancy boundaries.
- End-to-end LangGraph execution tracing tied to all L-layer agents (not just link plumbing).
