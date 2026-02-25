# CareerAgent-AI: Production Roadmap (L0–L9)
**Motive:** An end-to-end, agentic job-automation platform that bypasses bot detection and ensures 100% ATS-compliance for the US market.

## 🏗️ Architecture: The Recursive Gate (L0–L9)
| Layer | Agent/Service | Purpose & Logic |
| :--- | :--- | :--- |
| **L0-L1** | Sanitize & Intake | Security scrubbing + Resume/LinkedIn/Website ingestion. |
| **L2** | Parser & Planner | **FIX:** Move to Gemini 2.0. Generate 2-3 specific search personas. |
| **L3** | LeadScout | Multi-source discovery (Serper/Tavily/LinkedIn). **FIX:** Scrape full-text JD. |
| **L4** | Matcher | **FIX:** Use Human-Semantic Scoring (Growth/Fit) + ATS Keyword Match. |
| **L5** | Phase2 Evaluator | **HITL Gate:** Recursive loop; pause if Match Score < 0.8. |
| **L6** | Drafting Agent | **USA Standard:** Strictly ATS-compliant .docx (No columns/tables). |
| **L7** | Auto-Applier | **FIX:** Operationalize Playwright auto-fill + Gmail/Calendar integration. |
| **L8-L9** | Audit & Learning | Persistence in SQLite + LangSmith observability. |

## 🔴 Critical Fixes Required
1. **Unify State Models:** Merge `OrchestrationState` (legacy) into `AgentState` (core). All services must use `AgentState`.
2. **Operationalize L7:** Enable Playwright form-filling and map `AgentState` fields to common ATS (Workday, Greenhouse).
3. **SMS/Alerts:** Replace Twilio with `ntfy.sh` (free/OS) for "Needs Approval" push notifications.
4. **Google Integration:** Finish the OAuth flow for Gmail drafts and Calendar scheduling.