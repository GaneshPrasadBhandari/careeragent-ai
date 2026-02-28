# 🚀 CareerOS: Autonomous Career Intelligence Platform (v3.0)

**"Scaling resume precision with self-correcting agentic workflows."**

CareerOS is a production-grade, multi-agent system designed to automate the end-to-end job search lifecycle. Built for high-experience professionals (16+ years), it leverages a **Cognitive Double-Pass** architecture to move beyond simple keyword matching, ensuring deep semantic alignment between veteran experience and modern job requirements.

---

## 🧠 The 9-Layer Agentic Architecture
The platform operates as a sequential pipeline of specialized agents, each acting as a quality gate with autonomous retry capabilities.

| Layer | Agent | Mission | Automation Function |
| :--- | :--- | :--- | :--- |
| **L0** | **GuardAgent** | Security & Input Validation | PII scrubbing and API token verification. |
| **L1** | **UIAgent** | Mission Control Interface | Real-time state management and auto-refreshing UI. |
| **L2** | **ParserAgent** | **Cognitive Double-Pass** | Phase 1 (Regex) + Phase 2 (LLM Contextual Inference). |
| **L3** | **DiscoveryAgent**| Global Lead Ingestion | Multi-source scraping (Tavily, LinkedIn, Serper). |
| **L4** | **MatchAgent** | Hybrid Scoring Engine | 40% Keyword + 60% Semantic alignment. |
| **L5** | **EvalAgent** | Self-Correction Loop | **HITL Gate:** Audits match quality before drafting. |
| **L6** | **DraftAgent** | ATS-Optimized Tailoring | LLM-powered Markdown/LaTeX resume generation. |
| **L7** | **ApplyAgent** | Automated Submission | Playwright-based form filling (Workday/Greenhouse). |
| **L8** | **AnalyticsAgent**| ROI Dashboard | Funnel tracking: Applied → Interview → Offer. |

---


## 🛠️ Operational Guide

### 1. Prerequisites
* **Python:** 3.10+
* **Package Manager:** `uv` (Highly recommended for speed)
* **API Keys:** Gemini-1.5-Flash, Tavily, and LangSmith.

### 2. Local Setup
```bash
# Clone the repository
git clone [https://github.com/GaneshPrasadBhandari/careeragent-ai.git](https://github.com/GaneshPrasadBhandari/careeragent-ai.git)
cd careeragent-ai
```


# Sync the environment
```bash
uv sync

# Configure environment variables
cp .env_example .env
# Update .env with your specific API keys
```

## 3. Running the Platform
For local development, you must run the Backend (FastAPI) and the Frontend (Streamlit) in separate terminal tabs.

### Start Backend (API):

```Bash
python src/careeragent/api/main.py

for example:-
uv run uvicorn careeragent.api.main:app --app-dir src --host 127.0.0.1 --port 8000 --reload
```

### Start Frontend (UI):

```Bash
streamlit run app/ui/mission_control.py

for example:-
streamlit run app/main.py --server.port 8501
``` 

## 4. Verification & Logs
To ensure the agents are operating with "Self-Correction" active:

```Bash
# Run unit tests for the Cognitive Parser
pytest tests/unit/test_parser_dual_phase.py

# Tail the agent logs for real-time diagnostics
tail -f logs/careeragent.log

example:-
tail -n 200 logs/careeragent.log > full_system_debug.txt 2>/dev/null || history | tail -n 100 > full_system_debug.txt
```

---
# 📊 Phase 3 Capabilities (The "Functional Engine")
## Dual-Pass Logic:## 
Extracts "inferred skills" from project descriptions (e.g., identifies "Cloud Scaling" from architectural achievements).

## Hybrid Matching:##
 Improved match scores for senior candidates (16+ years) using vector embeddings to understand role seniority.

## Mission Control:##
 A unified Streamlit dashboard that tracks agent progress across all 9 layers with auto-incremental progress bars.

## Observability:##
 Deep-link integration with LangSmith for 100% trace transparency.
---

# 🚀 Roadmap: Phase 4 (Advanced Refinement)
## L7 Automation:
Expanding Playwright scripts for automated "Easy Apply" across multiple platforms.

## L8 Analytics Dashboard: 
A dedicated tracking page for application conversion rates and interview rounds.

## Learning Loop: 
Automated profile refinement based on employer feedback (rejection email analysis).

---
# Maintainer
Ganesh Prasad Bhandari AI & Agentic AI Developer | Operationalizing Agentic Workflows


***

### **How to Use This**
1. Open your project in VS Code.
2. Create/Open `README.md` at the root.
3. Paste the content above.

4. **Git Commands to Push:**
```bash
git add README.md
git commit -m "docs: finalize Phase 3 readme with 9-layer architecture"
git push origin feature/phase4_refinement_automation
```