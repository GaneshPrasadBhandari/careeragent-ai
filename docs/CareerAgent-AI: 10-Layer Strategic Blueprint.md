# CareerAgent-AI: 10-Layer Strategic Blueprint

## Purpose of This Blueprint

This document defines the strategic 10-layer architecture for **CareerAgent-AI** as a governed, agentic career operating system.

The goal of the platform is not just to automate isolated job-search tasks.  
It is to orchestrate the full workflow from **intake to discovery, matching, approval, execution, tracking, learning, and governance** while keeping the user in control of sensitive decisions.

A core architectural principle of CareerAgent-AI is that **every major layer includes an evaluator function**.  
The evaluator validates outputs before they are allowed to progress to the next layer. If the output is weak, unsafe, incomplete, or poorly grounded, the workflow can loop back for refinement or pause for human review.

---

# Layer 0 — User Layer (Interface and Intent Capture)

## Purpose
Capture the user’s goals, profile, constraints, documents, and preferences in a clean, structured form before the workflow begins.

This is the user-facing layer where the system learns:
- who the user is
- what roles they want
- where they want to work
- what constraints they have
- what resume or supporting documents they want the system to use

## Primary Inputs
- Resume or profile upload
- Target roles
- Salary expectations
- Preferred locations
- Visa or work-authorization constraints
- Job-search goals
- User approvals and edits during later phases

## Agents
- **Intake Assistant**  
  Collects user goals, role preferences, constraints, and resume/profile inputs.
- **Profile Capture Agent**  
  Converts raw user input into a structured candidate profile.
- **Constraint Capture Agent**  
  Captures hard constraints such as geography, visa needs, work mode, or timing requirements.

## Evaluator (L0)
The L0 evaluator validates:
- all required fields are present
- uploaded resume or profile is readable
- critical constraints are captured
- session state is initialized correctly

If L0 fails, the workflow does not begin.

## Current Implementation
- Streamlit-based UI
- Mission control / dashboard entry flow
- Resume upload and structured form handling

---

# Layer 1 — Entry and Security Layer (Sanitization, Session Integrity, Runtime Readiness)

## Purpose
Protect the system before orchestration begins by sanitizing inputs, reducing prompt-injection risk, validating runtime readiness, and preparing a secure execution session.

## Primary Inputs
- Raw user-provided text
- Resume content
- Session configuration
- Environment settings
- Runtime keys and integration availability

## Agents
- **Sanitize Agent**  
  Screens user-provided text and uploaded content for prompt injection patterns, malicious formatting, or malformed input.
- **PII Guard Agent**  
  Identifies and governs sensitive data handling where required.
- **Session Readiness Agent**  
  Confirms core runtime dependencies and execution prerequisites are available.

## Evaluator (L1)
The L1 evaluator validates:
- inputs are sanitized
- runtime configuration is usable
- critical services are reachable
- session security checks pass

If L1 fails, the pipeline stops or degrades gracefully before orchestration.

## Current Focus
- Input hygiene
- session readiness validation
- environment and runtime checks

---

# Layer 2 — Orchestrator Core Layer (Planner, Director, Profile Understanding)

## Purpose
Convert user intent into a valid execution plan and initialize stateful workflow control.

This is the operational brain of the system.

## Primary Inputs
- Structured user profile from L0
- sanitized inputs from L1
- workflow constraints
- runtime state

## Agents
- **Planner Agent**  
  Breaks the job-hunt workflow into executable sub-tasks such as intake parsing, search, extraction, ranking, drafting, approval, execution, and tracking.
- **Director Agent**  
  Reviews whether the plan obeys hard constraints and blocks invalid plans before downstream execution.
- **Parser Agent**  
  Extracts experience, skills, domain history, achievements, and role-fit signals from the uploaded resume/profile.
- **Profile Structuring Agent**  
  Converts extracted resume information into normalized internal state for matching and later RAG use.

## Evaluator (L2)
The L2 evaluator validates:
- parsing quality
- completeness of extracted user profile
- plan coherence
- hard constraint alignment
- execution readiness of the generated orchestration plan

If L2 fails, the plan is refined or rejected before discovery starts.

## Immediate Gap to Fix
- The Director must explicitly reject plans that violate hard constraints such as geography or role fit.
- The parsed profile should be scored for completeness before moving to discovery.

---

# Layer 3 — Manager Layer (Discovery, Search Operations, Competitive Intelligence)

## Purpose
Find, normalize, and filter high-quality job opportunities based on the user’s goals and constraints.

## Primary Inputs
- approved execution plan from L2
- user profile and constraints
- source configurations
- search parameters

## Manager Agents
- **Lead Scout Manager**  
  Generates role-specific search intents and negative filters.
- **Search Operations Manager**  
  Executes queries through supported web or API tools.
- **Geo-Fencing Manager**  
  Enforces hard geographic constraints where applicable.
- **Extraction Manager**  
  Pulls detailed job descriptions and structured job data.
- **Deduplication Manager**  
  Removes duplicate roles across sources and variants.
- **Source Reliability Manager**  
  Scores source quality and keeps weak or low-signal results from polluting the pool.

## Evaluator (L3)
The L3 evaluator validates:
- job pool quality
- source relevance
- geographic compliance
- deduplication success
- minimum useful pool size for downstream ranking

If L3 fails, the system loops back for refined search or broader but still policy-compliant retrieval.

## Current Focus
- discovery
- extraction
- URL persistence
- local-first safe workflow behavior

---

# Layer 4 — Agent Layer (Identity RAG, Matching, Gap Analysis, Recommendation Logic)

## Purpose
Determine how the user’s profile fits each discovered opportunity and generate grounded recommendation intelligence.

## Primary Inputs
- discovered jobs from L3
- structured candidate profile from L2
- stored memory / retrieval context
- prior project and experience evidence

## Agents
- **Skill Matcher Agent**  
  Maps candidate experience, skills, projects, and domain history against the job description.
- **Gap Analyst Agent**  
  Identifies missing requirements, weak signals, and ATS keyword gaps.
- **RAG Agent**  
  Retrieves user-specific project, experience, and profile context from structured memory/vector retrieval.
- **Recommendation Agent**  
  Produces fit logic, ranking rationale, and explainable job recommendations.
- **Shortlist Predictor Agent**  
  Estimates role-fit strength and potential shortlist / interview likelihood as a directional score.
- **ATS Insight Agent**  
  Flags likely ATS weaknesses and optimization opportunities.

## Evaluator (L4)
The L4 evaluator validates:
- quality of profile-to-job matching
- grounding quality of retrieved evidence
- explainability strength
- ATS relevance
- recommendation confidence

Jobs below threshold can be discarded, deprioritized, or routed for manual review.

## Current / Future Direction
- semantic and contextual matching
- RAG-grounded recommendation logic
- ATS improvement support
- fit scoring and shortlist prediction direction

---

# Layer 5 — Human-in-the-Loop Layer (Approval, Review, Trust Checkpoint)

## Purpose
Prevent black-box execution by ensuring the user approves sensitive or career-impacting decisions before automation proceeds.

## Primary Inputs
- ranked jobs
- generated explanation
- tailored application package drafts
- inferred skills or corrections
- proposed actions

## Agents
- **Liaison Agent**  
  Presents recommendations, rankings, evidence, and suggested next steps to the user.
- **Approval Routing Agent**  
  Determines which actions require user approval, edit, or rejection.
- **Explanation Agent**  
  Shows why a role was recommended, what changed in the resume, and what evidence was used.

## Evaluator (L5)
The L5 evaluator validates:
- that approval artifacts are complete
- the user’s decision state is captured correctly
- only approved items move forward
- rejected or edited outputs are routed properly

If the user rejects or edits outputs, the workflow loops back to the appropriate earlier layer.

## Critical Human-Control Actions
Examples include:
- final resume approval
- cover letter approval
- shortlist confirmation
- auto-apply approval
- interview scheduling approval
- recruiter communication approval
- acceptance or rejection of inferred missing skills

---

# Layer 6 — Execution Layer (Drafting, Application Actions, Outreach)

## Purpose
Take real-world action on approved opportunities through controlled execution.

## Primary Inputs
- approved shortlist from L5
- approved application assets
- approved communication actions
- workflow state and execution targets

## Agents
- **Executive Drafter Agent**  
  Produces ATS-oriented resume and cover letter versions tailored to each approved job.
- **Application Executor Agent**  
  Supports form-filling, document upload, and assisted or progressively automated application workflows.
- **Email Executive Agent**  
  Drafts recruiter emails, follow-ups, thank-you messages, rejection responses, and feedback requests.
- **Interview Coordination Agent**  
  Supports scheduling suggestions and calendar coordination with user approval.
- **Notification Agent**  
  Triggers reminders and approval prompts through supported channels.

## Evaluator (L6)
The L6 evaluator validates:
- ATS formatting and consistency
- draft quality
- hallucination risk in generated documents
- outbound communication quality
- execution-readiness of apply flows
- required approvals before real-world action

## Immediate Gaps to Fix
- stronger ATS-format validation after draft generation
- more robust safe execution checks before apply automation
- explicit separation of assisted execution vs approval-gated automation

---

# Layer 7 — Analytics and Learning Layer (Tracking, Feedback, Career Intelligence)

## Purpose
Track outcomes, measure workflow effectiveness, and improve future recommendations through feedback-aware learning.

## Primary Inputs
- application execution records
- recruiter replies
- interview signals
- user corrections
- follow-up outcomes
- skill-gap findings

## Agents
- **Dashboard Manager**  
  Tracks applications, dates, statuses, company progress, and workflow milestones.
- **Feedback Learning Agent**  
  Incorporates user feedback and downstream results to refine future search and ranking behavior.
- **Career Coach Agent**  
  Generates upskilling recommendations from skill gaps and missed role-fit signals.
- **Communication Outcome Agent**  
  Tracks whether outreach or follow-ups improved response quality.
- **Lifecycle Analytics Agent**  
  Measures funnel-level performance from discovery to interviews and offers.

## Evaluator (L7)
The L7 evaluator validates:
- analytics consistency
- status tracking accuracy
- feedback quality
- correctness of derived insights
- validity of recommendations surfaced to the user

## Immediate Gap to Fix
- dashboard persistence should be backed by stable structured storage so follow-ups, interview states, and offer-stage signals are not lost.

---

# Layer 8 — Infrastructure Layer (Data, Models, Memory, Tool Connectivity)

## Purpose
Provide the foundational runtime services that support state, memory, models, tools, and persistent storage.

## Components
- **Vector Database Layer**  
  Supports long-term retrieval and grounding for profile/project memory.
- **Structured Storage Layer**  
  Stores applications, workflow state, analytics, and evidence records.
- **Model Access Layer**  
  Provides local and hosted model execution paths.
- **MCP / Tool Connectivity Layer**  
  Supports structured tool invocation and external action patterns.
- **Runtime Configuration Layer**  
  Manages settings, environment variables, and service connectivity.

## Evaluator (L8)
The L8 evaluator validates:
- service availability
- latency health
- token / cost behavior
- storage consistency
- tool connectivity quality
- model/runtime readiness

## Current Direction
- SQLite current path
- PostgreSQL-ready direction
- vector retrieval support
- MCP-compatible tool patterns
- model and storage modularity

---

# Layer 9 — Governance, Ethics, and Operations Layer (Audit, Compliance, Observability)

## Purpose
Ensure the system remains governable, transparent, policy-aware, and operationally healthy.

## Agents
- **Compliance Auditor Agent**  
  Checks workflow behavior against safe automation and policy boundaries.
- **Governance Auditor Agent**  
  Validates approval gating, traceability, and execution discipline.
- **Observability Agent**  
  Surfaces traces, logs, and runtime visibility for debugging and trust.
- **Ops Health Agent**  
  Monitors execution health, failures, retries, and system reliability posture.

## Evaluator (L9)
The L9 evaluator provides the final audit check on:
- policy compliance
- observability completeness
- approval integrity
- workflow safety
- operational readiness

## Observability and Ops Direction
- LangSmith for end-to-end tracing
- evaluator result visibility across layers
- evidence-linked workflow logging
- future runtime drift and behavior monitoring
- auditability for sensitive automation paths

---

# Cross-Layer Design Principle: Evaluator at Every Layer

CareerAgent-AI is intentionally designed so that **each major layer has an evaluator function**.

This provides:
- output validation before progression
- retry or refinement loops when quality is weak
- stronger grounding and lower hallucination risk
- better policy enforcement
- more transparent workflow control
- higher trust for sensitive automation

This evaluator-per-layer design is one of the system’s defining characteristics.

---

# Cross-Layer Design Principle: Human Approval for Career-Critical Actions

No matter how capable the agents become, some decisions should remain explicitly user-controlled.

Examples:
- final resume submission
- cover letter submission
- job shortlist acceptance
- inferred skill approval
- recruiter messaging approval
- interview scheduling approval
- offer-stage decisions

This keeps the platform useful without turning it into an unsafe black box.

---

# Immediate Gap Analysis

## 1. L2 Director Hard-Constraint Enforcement
The Director layer should explicitly reject plans that violate hard constraints such as geography, job-type restrictions, or user-defined exclusions.

## 2. L6 ATS Validation
The drafting / execution pipeline should include a stronger ATS-format evaluator before final documents are approved or uploaded.

## 3. L7 Stable Persistence
Tracking and analytics should be backed by reliable structured persistence so application status, follow-up timing, and interview states remain stable.

## 4. Cross-Layer Trace Visibility
LangSmith and local evidence logging should make it easier to inspect exactly:
- which agent acted
- which evaluator passed or failed
- which loop retried
- which evidence was used
- which approval checkpoint was triggered

## 5. Missing-Skill Workflow Completion
The product should fully support:
- inferred skill confirmation
- rejected-skill correction
- upskilling recommendations
- later resume refresh and re-entry into the workflow

---

# Strategic Summary

CareerAgent-AI is not just an AI resume assistant.

It is being designed as a **governed, evaluator-driven, agentic career operating system** that can support the full journey from job discovery to application execution, communication support, interview coordination, analytics, and learning.

Its long-term strength comes from five ideas working together:

1. **Stateful agent orchestration**
2. **Evaluator-controlled progression**
3. **Human approval for sensitive actions**
4. **Explainability and evidence visibility**
5. **Operational traceability and governance**
