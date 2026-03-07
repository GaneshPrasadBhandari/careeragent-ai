# CareerAgent-AI
### An AI-Driven Career Operating System for Intelligent Job Search Automation

CareerAgent-AI is a full-stack AI product designed to transform fragmented job-search tasks into one orchestrated workflow.

It helps users move from **planning and discovery** to **matching, ATS-tailored application assets, approvals, tracking, and continuous improvement** through a controlled, human-centered AI system.

This repository reflects a **capstone-to-product build** evolving into a deployable beta platform for AI-assisted career workflow automation.

---

## Overview

Modern job searching is still broken.

Candidates jump across job boards, rewrite resumes manually, track applications in spreadsheets, lose follow-up context, and spend too much time on repetitive tasks that should be streamlined.

CareerAgent-AI addresses that problem by acting as a **career operating system**. Instead of solving one isolated task, it coordinates the end-to-end workflow:

**Plan → Discover → Match → Prepare → Approve → Apply → Track → Learn → Improve**

The platform is designed around:

- agentic workflow orchestration
- explainability
- human approval gates
- evidence capture
- operational traceability
- deployment-oriented system design

---

## The Problem

Job searching today is:

- fragmented across platforms
- manual and repetitive
- difficult to track consistently
- stressful and time-consuming
- often optimized for volume instead of quality

Most existing tools solve only one part of the workflow:
- job boards list jobs
- resume tools rewrite text
- trackers store statuses
- auto-apply tools focus on speed

CareerAgent-AI is designed to orchestrate the workflow end to end while keeping the user in control of high-stakes decisions.

---

## The Solution

CareerAgent-AI is built as a **career operating system** that coordinates:

- profile intake and resume parsing
- planning and strategy
- job discovery and prioritization
- matching and ranking
- ATS-oriented resume tailoring
- cover letter and package generation
- approval checkpoints
- application execution support
- tracking and evidence logging
- analytics and learning

The goal is not blind automation.

The goal is **controlled, explainable, high-quality career workflow execution**.

---

## Current Product Capabilities

### Core workflow capabilities
- Resume/profile intake and parsing
- Role-aware job discovery and prioritization
- Matching and ranking workflows
- ATS-tailored resume and cover letter generation
- Human-in-the-loop approval gates
- Application workflow orchestration
- Notification and tracking support
- Evidence-linked run visibility
- Streamlit-based mission control and dashboard views

### Technical design principles
- Assisted automation first
- Human approval for critical actions
- No fabricated skills or unsupported claims
- Explainability by design
- Modular services and orchestration layers
- Product-first architecture with beta deployment intent

---

## Why It Is Different

CareerAgent-AI is **not just a resume builder** and **not just an auto-apply bot**.

It is designed as an **agentic career workflow platform** that combines:

- planning
- discovery
- ranking
- document generation
- approval gates
- execution support
- tracking
- learning loops

That makes it closer to a **career operating system** than a single-purpose AI tool.

---

## Architecture Direction

CareerAgent-AI is structured around a layered workflow model spanning:

- user interaction and review
- application entry and mission control
- orchestration core
- manager and planning logic
- agent execution services
- approval gates
- execution and tracking
- analytics and learning
- memory, models, governance, and ops

Detailed architecture notes are documented in:
- [docs/10_layer_strategic_roadmap.md](./docs/10_layer_strategic_roadmap.md)
- [docs/REPO_DEEP_DIVE_AND_GAP_PLAN.md](./docs/REPO_DEEP_DIVE_AND_GAP_PLAN.md)

---

## Demo Videos

### Product Walkthroughs
- **Recruiter-friendly walkthrough:** https://youtu.be/_IpHNsKfmmE
- **Deep technical walkthrough:** https://www.youtube.com/watch?v=xI3dF-FLsy8&t=2255s

---

## Repository Structure

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
├── notebooks_v2/                   # Iterative fixes, stabilization, workflow debugging notebooks
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
