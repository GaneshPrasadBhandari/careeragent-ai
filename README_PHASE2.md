# 🚀 CareerAgent-AI: Phase 2 (Stateful Operationalization)

## 📌 Phase 2 Overview
Phase 2 transitions CareerAgent-AI from a linear script into a **Stateful Multi-Agent System**. Using **LangGraph**, the system now maintains long-term memory, handles recursive evaluation loops, and executes production-grade browser automation.

### **Key Architectural Shift: L0 - L9 Pipeline**
We have codified the agent workflow into a 10-layer operational stack. Each layer is protected by an **Evaluator Gate** that validates output before moving the state forward.

---

## 🏗️ Layered Agent Architecture

| Layer | Agent Role | Logic & Operationalization | Tools / LLM |
| :--- | :--- | :--- | :--- |
| **L0-L2** | **Intake & Planner** | Extracts deep project technical stacks and builds search personas. | Gemini 1.5 Pro |
| **L3** | **Search Cluster** | Distributed search using Firecrawl with geofencing and anti-bot logic. | Firecrawl / Claude 3.5 |
| **L4** | **Semantic Matcher** | **Weighted Scoring:** 45% Skills / 35% Exp / 20% ATS. Hybrid keyword + semantic search. | Gemini 1.5 Pro |
| **L5** | **Ranker Evaluator** | Validates match quality. Triggers `RETRY_SEARCH` if score < 0.70. | Gemini 1.5 Pro |
| **L6** | **ATS Drafter** | Generates strict 1-column .docx resumes (no tables/images) to bypass parser filters. | Claude 3.5 Sonnet |
| **L7** | **Auto-Applier** | Browser automation for Workday/Greenhouse/Lever with randomized human-emulation delays. | Playwright |
| **L8** | **Comm Agent** | Real-time push notifications for HITL (Human-in-the-Loop) approval. | ntfy.sh |
| **L9** | **Governance** | Final audit of the run, recording learning signals to the database. | LangGraph / SQL |

---

## 🛠️ Operational Stack (Phase 2)
- **State Management:** `LangGraph` for persistent agent memory and thread-safe execution.
- **Validation:** `Pydantic v2` for strict schema enforcement at every layer.
- **Dependency Management:** `uv` with `dependency-groups` for optimized production environments.
- **Stealth Automation:** `Playwright` with fingerprinted browser sessions.
- **Monitoring:** `Loguru` for structured JSON logging and `ntfy.sh` for remote mobile alerts.

---

## 🚀 Getting Started (Operational Commands)
Verify your environment is synced with the Phase 2 requirements:

```bash
# Sync production dependencies
uv sync

# Install Playwright browser binaries
uv run playwright install chromium

# Run the Phase 2 Health Check
uv run python src/careeragent/ops_check.py