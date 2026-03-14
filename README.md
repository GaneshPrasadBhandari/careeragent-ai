# CareerAgent-AI
### An AI-Driven Career Operating System (Capstone → Deployable Beta Product)

CareerAgent-AI is a full-stack AI product designed to transform fragmented job-search tasks into one orchestrated workflow.

It helps users move from **planning and discovery** to **matching, ATS-tailored application assets, approvals, tracking, and continuous improvement** through a controlled, human-centered AI system.

This repository reflects a **capstone-to-product build** evolving toward a deployable beta platform for AI-assisted career workflow automation.

---


## Live Beta Demo (Phase 6)

CareerAgent-AI Phase 6 is now deployed for live beta testing.

### Public beta access
- **Dashboard UI:** [https://careeragent-dashboard.onrender.com](https://careeragent-dashboard.onrender.com)
- **API Docs:** [https://careeragent-api.onrender.com/docs](https://careeragent-api.onrender.com/docs)
- **Health Check:** [https://careeragent-api.onrender.com/health](https://careeragent-api.onrender.com/health)

### What can be tested in this beta
- Mission Control dashboard
- resume upload and job-hunt flow
- L0→L9 pipeline visibility
- live progress tracking
- backend/API connectivity
- deployed beta workflow validation

### Beta notes
- This is a **public beta deployment** for Phase 6 validation.
- The app is currently hosted on free-tier infrastructure, so the first request after inactivity may take extra time.
- Please avoid uploading highly sensitive personal or confidential data during beta testing.
- If the dashboard does not connect automatically, refresh once and verify the backend URL inside the UI.

### Suggested test flow
1. Open the dashboard UI
2. Upload a test resume
3. Start a hunt
4. Observe progress, layer updates, and generated outputs
5. Report bugs or edge cases with screenshots and reproduction steps

---


## Overview

Modern job searching is still broken.

Candidates jump across job boards, rewrite resumes manually, track applications in spreadsheets, lose context between follow-ups, and spend too much time on repetitive work that should be streamlined.

CareerAgent-AI addresses that problem by acting as a **career operating system**. Instead of solving one isolated task, it coordinates the end-to-end workflow:

**Plan → Discover → Match → Prepare → Approve → Apply → Track → Learn → Improve**

The platform is designed around:

- agentic workflow orchestration with **LangGraph**
- evaluator-controlled quality gates at each major layer
- **MCP-compatible tool access patterns** for structured external actions
- human approval for sensitive decisions
- explainability and evidence visibility
- **LangSmith-based tracing and observability**
- multi-tool and multi-model execution flexibility
- tracking, analytics, and feedback-driven improvement

The long-term vision is not just document generation. It is a governed, intelligent system that can assist across the full journey from initial search to interview scheduling, follow-ups, offer-stage support, and continuous upskilling.

---

## 1. Vision and Problem Statement

### The Problem

Job searching today is:

- fragmented across platforms
- manual and repetitive
- difficult to track consistently
- emotionally stressful and time-consuming
- often optimized for volume instead of quality

Most existing tools solve only one part of the workflow:

- job boards list jobs
- resume tools rewrite text
- trackers log applications
- auto-apply tools focus on speed

CareerAgent-AI is designed to orchestrate the full workflow while keeping the user in control of high-stakes decisions.

---

### The Solution

CareerAgent-AI acts as a **personal career operating system**, coordinating:

- job discovery
- intelligent job matching
- ATS-oriented resume and cover letter generation
- evaluator-based validation at every workflow layer
- assisted and semi-automated application workflows
- explainable recommendations with evidence and rationale
- tracking, analytics, and feedback-aware improvement
- recruiter communication support
- interview coordination and approval-driven scheduling
- skill-gap detection and guided upskilling support

The system is designed to automate as much of the workflow as possible while keeping **humans in control of critical and life-impacting decisions**.

That includes actions such as:

- final resume approval
- cover letter approval
- job shortlist confirmation
- auto-apply approval
- interview scheduling approval
- sensitive recruiter communication review

This makes the product closer to a **career execution operating system** than a narrow AI assistant.

---

## 2. Product Philosophy

CareerAgent-AI is built on **enterprise-minded AI principles**, not one-shot chatbot behavior:

- **Assisted automation first**
- **Human-in-the-loop for critical actions**
- **Explainability by design**
- **Agent orchestration, not monolithic AI**
- **Provider-agnostic model layer**
- **Composable, extensible architecture**
- **Deployment-ready product direction**

This makes it suitable for:

- real demos
- recruiter and evaluator review
- beta-user testing
- future startup packaging and commercialization

---

## 3. Evaluator-Driven Workflow Design

A core design principle in CareerAgent-AI is that **each major workflow layer can be paired with an evaluator agent**.

The evaluator reviews the output of that layer for:

- relevance
- completeness
- grounding quality
- policy alignment
- hallucination risk
- bias risk
- execution readiness
- user-value quality

If the output passes, it moves to the next layer.

If the output does not pass, the workflow can loop back to the same layer, refine the result, and try again until it reaches an acceptable threshold or requires human intervention.

This creates a more reliable and governed execution pattern than one-shot generation.

The broader design also supports the use of **multiple tools, LLMs, APIs, retrieval strategies, and fallback paths** so the system can choose the most suitable route for each task instead of depending on a single brittle component.

---

## 4. System Architecture

![CareerAgent-AI Architecture](./CareerAgent-AI_Architecture.png)

> *CareerAgent-AI architecture blueprint — user interface, orchestration core, managers, agent services, approval gates, tracking, analytics, memory, and governance.*

### High-Level Architecture

The platform is structured around a layered workflow model spanning:

1. **User and Entry Layer**
   - profile intake
   - resume upload
   - role and constraint capture

2. **Orchestration Core**
   - workflow control
   - sequencing
   - runtime coordination
   - state progression

3. **Manager Layer**
   - planning
   - prioritization
   - decision routing
   - escalation logic

4. **Agent Layer**
   - parsing
   - matching
   - ranking
   - drafting
   - package generation
   - apply execution
   - analytics support

5. **Human Approval Gates**
   - shortlist approval
   - resume and cover letter approval
   - sensitive action review
   - negotiation-sensitive decisions

6. **Execution and Tracking**
   - application records
   - timestamps
   - generated artifacts
   - evidence logs
   - workflow history

7. **Analytics and Learning**
   - outcome tracking
   - feedback signals
   - performance analysis
   - future optimization loops

8. **Memory and Models**
   - structured profile memory
   - semantic retrieval
   - ranking logic
   - LLM/model access

9. **Governance and Ops**
   - policy controls
   - auditability
   - observability
   - secure runtime handling

Detailed supporting docs:
- [docs/architecture.md](./docs/architecture.md)
- [docs/10_layer_strategic_roadmap.md](./docs/10_layer_strategic_roadmap.md)

---

## 5. Core Capabilities

### Current / Beta-Oriented Capabilities
- Resume/profile intake and parsing
- Role-aware job discovery and prioritization
- Matching and ranking workflows
- ATS-tailored resume generation
- Cover letter and application package generation
- Human approval gates
- Tracking and evidence logging
- Notification and workflow support
- Streamlit-based mission control and dashboard views

### Target Product Capabilities
- Evaluator agent at every layer to validate outcomes before passing to the next stage
- Retry and refinement loops when outputs fail quality thresholds
- Multi-tool, multi-LLM, and multi-API strategy for stronger output quality and resilience
- Explainable recommendations showing why jobs were chosen and what evidence supports the fit
- Missing-skill detection with user approval, rejection, or correction options
- ATS score comparison across resume versions
- Interview-shortlist likelihood prediction
- Auto-apply execution with form-filling, document upload, and evidence logs
- Recruiter email drafting for follow-ups, interview replies, thank-you notes, feedback requests, and offer-stage communication
- Approval-driven Google Calendar scheduling for interviews
- Analytics dashboards for applications, company tracking, replies, interviews, offers, and workflow performance
- Feedback-driven optimization and continuous improvement
- Guided upskilling support with learning resources such as documentation and video tutorials when skill gaps are detected

---

## 6. Explainability, Trust, and Human Control

CareerAgent-AI is designed to make recommendations visible and reviewable.

For major actions, the user should be able to see:

- why a role was recommended
- what evidence supports the match
- what changed in the resume or cover letter
- which skills were emphasized or inferred
- what confidence or fit logic influenced the recommendation
- what evaluator checks passed or failed before progression

This explainability layer is critical for reducing blind automation, hallucination risk, and hidden bias.

The system is intentionally designed so that sensitive decisions remain human-approved even when the surrounding workflow becomes highly automated.

---

## 7. Job Data Strategy

### Structured and Compliant Inputs

The system is designed to support compliant ingestion patterns such as:

- official APIs where available
- structured imports
- user-provided saved jobs
- workflow-safe assisted application paths
- approved connector-style integrations where policy allows

### Product Direction

The long-term design supports multi-source ingestion while respecting platform constraints, user consent, governance requirements, and safe automation boundaries.

---

## 8. End-to-End Operational Pipeline

### Step 1 — User Profile Setup
The user provides:

- resume or structured profile
- role targets
- domains
- constraints
- location preferences
- job search goals

This becomes the operating context for downstream workflow logic.

### Step 2 — Intake Validation and Skill Understanding
The system parses and validates the profile, identifies strengths, and checks for missing or implied skills.

If a likely skill is inferred but not clearly present in the resume, the user can:

- approve it
- reject it
- edit it

If real skill gaps exist, the system can later recommend learning resources and upskilling paths.

### Step 3 — Planning and Strategy
A planner/director layer creates an execution strategy:

- role priority
- search focus
- application plan
- follow-up direction
- document tailoring intent

### Step 4 — Job Ingestion
Jobs are brought in through supported sources and normalized into a common internal structure.

### Step 5 — Matching and Ranking
The system evaluates jobs using:

- deterministic filtering
- hybrid and semantic matching
- skill alignment
- context-aware ranking
- heuristic fit scoring

It can also explain why a job is being recommended and what evidence supports the match.

### Step 6 — Evaluator Validation
An evaluator layer checks whether the matching and recommendation output is strong enough.

If it fails quality thresholds, the workflow loops back for refinement.

### Step 7 — Application Package Generation
For shortlisted roles, the system prepares:

- tailored resume content
- cover letter drafts
- application answers
- supporting rationale

The design supports ATS improvement, stronger targeting, and future shortlist/interview prediction scoring.

### Step 8 — Human Approval Gate
The user reviews:

- what will be submitted
- why it was selected
- what changed
- What evidence supports it
- whether to approve, edit, or reject

This is essential because job applications affect real careers and should not be blindly automated.

### Step 9 — Application Execution
The system supports assisted and progressively automated application execution, including:

- form-filling
- document upload
- logging of actions
- evidence capture

### Step 10 — Tracking and Analytics
Each action can be recorded with:

- company
- role
- timestamps
- submitted assets
- workflow status
- recruiter responses
- interview updates
- offer-stage progress

### Step 11 — Communication Support
The system can draft communications for:

- follow-ups
- recruiter replies
- interview confirmations
- thank-you emails
- rejection responses
- feedback requests
- offer-stage interactions

Critical or sensitive outbound communication remains approval-based.

### Step 12 — Interview Scheduling Support
If an interview email is detected, the system can assist with scheduling and propose calendar actions, while waiting for user approval or edits before finalizing times.

### Step 13 — Learning and Improvement
The platform uses workflow outcomes, recruiter responses, feedback, and user corrections to improve future recommendations, document generation, prioritization logic, and decision support.

### Step 14 — Upskilling Support
When skills are missing or weak for target roles, the platform can recommend learning resources such as documentation, tutorials, and guided improvement paths so the user can strengthen the profile and re-enter the workflow with a better application package.

Detailed workflow reference:
- [docs/pipeline.md](./docs/pipeline.md)

---

## 9. Technology Stack

### Core Application Stack
- Python 3.11+
- FastAPI-style backend services
- Streamlit UI and mission-control views
- Pytest
- UV / modern Python dependency workflow
- Docker

### Agentic AI and Orchestration
- **LangGraph** for stateful agentic workflow orchestration
- manager/agent service architecture
- evaluator and guardrail services at each major workflow layer
- human-in-the-loop approval checkpoints
- retry and refinement loops for failed evaluations

### Tooling and Integration Layer
- **MCP-style / MCP-compatible tool access patterns**
- multi-tool routing for task-specific execution
- external API and automation support
- structured fallback paths when one tool or service is unavailable

### LLM and Reasoning Layer
- provider-flexible LLM integration
- support for local, hosted, and API-based model execution
- hybrid decision logic using rules, semantic retrieval, and LLM reasoning
- explainable recommendation generation

### Retrieval, Matching, and Intelligence
- RAG-oriented retrieval design
- hybrid search using keyword, semantic, and contextual matching
- skill-gap detection and role-fit analysis
- shortlist and interview-likelihood scoring direction

### Tracing, Observability, and Quality Control
- **LangSmith** for workflow tracing, debugging, and execution visibility
- evidence-linked step logging
- evaluator feedback loops
- explainability and audit support
- future data quality and drift monitoring direction

### Data and Storage
- SQLite-based current path
- PostgreSQL-ready direction
- vector retrieval design
- artifact and evidence storage
- tracking and analytics records

### Deployment and Platform Operations
- Docker-based packaging
- local container orchestration experiments with Minikube
- GitHub Actions CI/CD direction
- AWS deployment direction
- Azure deployment direction
- beta-hosting platform readiness
- environment-driven runtime configuration
- scalable deployment path for future public beta testing
  
### MLOps and Runtime Monitoring Direction
- MLflow
- DVC
- observability
- evaluation loops
- future-ready data drift and behavior drift monitoring direction
- deployment monitoring

---

## 10. Observability, Tracing, and Runtime Governance

CareerAgent-AI is designed to be observable, traceable, and reviewable at runtime.

That includes:

- **LangSmith tracing** for agents, workflow paths, tools, and model calls
- evaluator results at each major layer
- evidence-linked logs for recommendations and generated outputs
- approval-gate checkpoints for sensitive actions
- execution visibility across orchestration, generation, and apply flows
- future-ready monitoring for data drift, model behavior drift, and workflow reliability

This matters because agentic systems should not behave like black boxes.  
They should be inspectable, debuggable, and governable in production-style environments.



## 11. Repository Structure

```text
careeragent-ai/
├── .ai_context/                     # Internal architecture maps, phase references, agent hierarchy
├── .github/workflows/              # GitHub workflow scaffolding
├── .streamlit/                     # Streamlit secrets and runtime config
├── _patch_v5/                      # Patch workspace used during iterative fixes
├── _rollback/                      # Rollback archives and pre-fix snapshots
├── app/                            # App entry layer and UI-facing runtime
│   ├── ui/
│   │   ├── dashboard.py
│   │   └── mission_control.py
│   └── main.py
├── docs/                           # Project docs, portfolio, roadmap, architecture, deployment notes
│   ├── media/
│   ├── 10_layer_strategic_roadmap.md
│   ├── PATCH_NOTES.md
│   ├── PORTFOLIO.md
│   ├── README.md
│   ├── REPO_DEEP_DIVE_AND_GAP_PLAN.md
│   ├── SETUP_AND_VALIDATION.md
│   ├── architecture.md
│   ├── competitive-landscape.md
│   ├── deployment.md
│   ├── pipeline.md
│   ├── roadmap.md
│   └── vision.md
├── newfolder/                      # Temporary experimental runtime node work
├── notebooks/                      # Early notebooks and deployment/setup experiments
├── notebooks_v2/                   # Iterative fixes, stabilization, and workflow debugging notebooks
├── sqlitecloud:/                   # Local SQLiteCloud connection artifact
├── src/                            # Main product source code
│   ├── careeragent/
│   │   ├── agents/                 # Agents, evaluators, schemas, workflow services
│   │   ├── api/                    # API entrypoints, request models, run manager
│   │   ├── core/                   # Config, settings, state, state stores
│   │   ├── integrations/           # Integration layer
│   │   ├── langgraph/              # Graphs, nodes, HITL flows, runtime nodes
│   │   ├── managers/               # Planning and manager-layer logic
│   │   ├── nlp/                    # NLP utilities and skills processing
│   │   ├── orchestration/          # Planner, director, engine, orchestrator
│   │   ├── services/               # Analytics, notifications, DB, exporter, XAI
│   │   ├── tools/                  # LLM tools and web tools
│   │   └── ops_check.py
│   ├── pydantic/
│   ├── pydantic_bridge_backup/
│   ├── pydantic_fallback_backup/
│   ├── pydantic_settings/
│   └── httpx.py
├── streamlit/                      # Streamlit shim/runtime package
├── tests/                          # Unit and integration tests
│   ├── integration/
│   ├── unit/
│   ├── conftest.py
│   └── test_runtime_flow.py
├── uploads/                        # Local uploaded resumes and run-time artifacts
├── .dvcignore
├── .env
├── .env_example
├── .gitignore
├── .python-version
├── LICENSE
├── README.md
├── README_PHASE2.md
├── README_PHASE3.md
├── REPO_MAP.md
├── api_main.py
├── check_env.py
├── debug_checklist.py
├── evaluator.py
├── full_system_debug.txt
├── get-pip.py
├── ls
├── main.py
├── mission_control.py
├── pyproject.toml
├── requests.py
├── run_app.py
├── setup_folders.py
├── setup_repo.py
└── uv.lock


Without that, the rest of your README can render badly.

---


```md
## 12. Code Organization Notes
## 13. Main Runtime Areas
## 14. Demo Videos
## 15. Documentation
## 16. Market Context
## 17. Current State of the Repository
## 18. Quick Start



## 19. Beta Storage, Privacy, and Developer Access (Render)

- End-user dashboard views intentionally avoid exposing raw storage paths and full feedback payloads.
- Current beta persistence is **Render service local filesystem + SQLite in `logs/`**:
  - uploads: `uploads/`
  - generated artifacts: `artifacts/<run_id>/...`
  - dated feedback snapshots: `artifacts/feedback/YYYY-MM-DD/<run_id>/...`
  - tracking DB: `logs/careeragent_tracking.db`
  - run state snapshots: `logs/state_<run_id>.json`
- For developer-only inspection, use the protected endpoint:
  - `GET /dev/hunt/{run_id}/storage?token=<CAREERAGENT_DEV_TOKEN>`
  - Set `CAREERAGENT_DEV_TOKEN` on the API service env in Render.
- Important for free/beta Render tiers: local filesystem is tied to the service instance and may not be durable across full redeploy/rebuilds.


## 20. Deployment Direction
CareerAgent-AI is being prepared for:

- Docker packaging
- GitHub Actions CI/CD automation
- local container orchestration experiments with **Minikube**
- cloud deployment on **AWS**, **Azure**, or lightweight beta-hosting platforms
- public beta testing
- LangSmith-based runtime tracing and debugging
- observability improvements across agents, tools, and workflows
- MLflow / DVC alignment
- future data-drift and model-behavior monitoring
- stronger runtime traceability and governance controls

The product is being designed not just to run locally, but to evolve into a deployment-ready agentic AI platform with production-oriented visibility and control.

See:
- [docs/deployment.md](./docs/deployment.md)


## 20. Ethics, Security, and Trust
CareerAgent-AI is designed around:

- human approval gates
- no fabricated skills or unsupported claims
- explainability and evidence logging
- secure handling of sensitive configuration
- privacy-aware data handling
- governed automation instead of blind execution

This is essential when dealing with people’s careers and application materials.

---


## 21. Positioning
CareerAgent-AI began as a capstone-driven system and is being developed into a deployable beta product for AI-assisted career workflow automation.

It is intended to serve as:

- a flagship portfolio project
- a serious technical demo for recruiters and evaluators
- a startup-ready foundation for future beta testing and commercialization

---


## 22. License
This repository is available for evaluation and educational review. See the LICENSE
 file for usage terms.

---


## 23. Author
Ganesh Prasad Bhandari
AI Architect | GenAI Researcher | Founder – AIinovateHUB

GitHub: @ganeshprasadbhandari
