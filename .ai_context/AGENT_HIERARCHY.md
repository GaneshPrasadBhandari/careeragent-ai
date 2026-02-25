# 🤖 CareerOS: Final Agent Hierarchy & Logic

## Tier 1: The C-Suite (Layer 2 - Orchestration)
- **CEO Agent (Planner):** Consumes L1 Intake and L1 Evidence to create a "Weekly Search Strategy."
- **Director Agent (Reviewer):** Audits the CEO's strategy against L9 Guardrails (e.g., ensuring "Remote-only" constraints are met).

## Tier 2: The Tactical Managers (Layer 3 - Matching)
- **Sourcing Manager:** Triggers job ingestion (P3).
- **Ranking Manager:** Performs "Artifact Math" to score jobs (P5) without hallucination.

## Tier 3: The Creator Agents (Layer 4 - Generation)
- **Resume Agent:** Tailors bullets grounded in the Evidence Profile (P18).
- **Cover Letter Agent:** Drafts narratives based on job requirements.
- **Constraint:** These agents MUST pull from the L8 Knowledge Layer and are blocked from fabricating skills.

## Tier 4: The Safety Gate (Layer 5 - Human-in-the-Loop)
- **Status:** Add this line to Tier 4: "The system implements the pause by setting is_approved: false in the state file and terminating the current execution loop until a new request is received with is_approved: true."
- **Action:** User (Tanish/Sita) reviews generated artifacts. Progress to L6 is logically locked until a `Human_Approval_Artifact` exists.

## Tier 5: The Worker Agents (Layer 6 - Execution)
- **Application Worker:** Uses tools to fill forms or send emails (P8).
- **Tracking Worker:** Updates the application status and audit logs.

## Tier 6: The Analyst Agents (Layer 7 - Analytics)
- **Feedback Agent:** Monitors success metrics (interviews vs. rejections).
- **Learning Agent:** Detects "Market Drift" and feeds updates back to the CEO (L2) to adjust future strategies.

## Tier 7: The Governance Agents (Layer 9 - Platform Ops)
- **Guardrail Agent:** Silently validates every L4 output against the Evidence Profile.
- **Audit Agent:** Maintains the "Explain-My-Rank" log for every system decision.