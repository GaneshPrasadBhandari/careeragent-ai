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
├── app/                         # App entrypoints and Streamlit-facing UI
│   ├── ui/
│   │   ├── dashboard.py
│   │   └── mission_control.py
│   └── main.py
├── src/careeragent/             # Main product source code
│   ├── agents/                  # Domain agents, evaluators, schemas
│   ├── api/                     # API entrypoints, request models, run manager
│   ├── core/                    # Config, settings, state, state stores
│   ├── langgraph/               # Graphs, nodes, HITL flows, runtime nodes
│   ├── managers/                # Manager-layer logic and coordination
│   ├── orchestration/           # Engine, planner, orchestrator
│   ├── services/                # Notifications, analytics, DB, exporter, XAI
│   ├── tools/                   # LLM tools and web tools
│   ├── nlp/                     # Skills and language utilities
│   └── integrations/            # External integration layer
├── docs/                        # Product, setup, validation, and portfolio docs
│   ├── PORTFOLIO.md
│   ├── SETUP_AND_VALIDATION.md
│   ├── REPO_DEEP_DIVE_AND_GAP_PLAN.md
│   ├── PATCH_NOTES.md
│   ├── 10_layer_strategic_roadmap.md
│   └── media/
├── notebooks/                   # Early experiments and setup notebooks
├── notebooks_v2/                # Iterative stabilization and workflow notebooks
├── tests/                       # Unit and integration test coverage
├── uploads/                     # Local uploaded artifacts for app runs
├── .ai_context/                 # Internal architecture and phase mapping notes
├── _patch_v5/                   # Patch workspace
├── _rollback/                   # Rollback snapshots and safety backups
├── pyproject.toml
├── .env_example
├── README.md
└── uv.lock
