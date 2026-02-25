# 🏗️ CareerOS: AI Context & Orchestration Map

## 🎯 1. Current Mission
- **Status:** Phase 1 (Deterministic) and Phase 2A (Agentic Foundation: P13, P14) Complete.
- **Active Focus:** **P15: Hybrid Matcher & Human Approval Gate**.
- [cite_start]**Current Technical Goal:** Ensuring the orchestrator pauses and waits for a user `approved_match_v2.json` before allowing L4 Creator Agents to generate content[cite: 78, 83, 84].
- **Philosophy:** "Explainability by Design". Every AI action must be traceable to the Evidence Profile.
**Current Stage:** Transitioning from Phase 1 (Deterministic) to Phase 2 (Agentic).
**Active Pipeline Point:** P15 (Human-in-the-loop Approval Gate).
**Primary Goal:** Implementing the L2 (C-Suite) Agent logic using LangGraph.


## 👤 Persona Context (The "Why")
- **Tanish (The Builder):** Needs speed but lacks tracking. The system must prevent him from "embellishing" skills to match jobs.
- **Sita (The Optimizer):** Strategic and disciplined. She needs repeatable quality and a system that learns from her outcomes.
- **Moment of Truth:** "Can I trust this system enough to submit it to a real employer?" Every piece of code must support this trust.


## 🏛️ The 9-Layer Architecture vs. Pipeline Stages
| Layer | Name | Status | Pipeline Stages | Agent Responsibility |
| :--- | :--- | :--- | :--- | :--- |
| **L1** | Entry/UX | ✅ DONE | P1 | Deterministic Logic |
| **L2** | Strategic Orch | 🏗️ BUSY | P12-P14 | **CEO (Planner) & Director (Reviewer)** |
| **L3** | Tactical Mgmt | ✅ DONE | P4-P5 | Manager Agents (Ranking/Matching) |
| **L4** | Generation | 🏗️ NEXT | P18 | Content Agents (Resume/Cover Letter) |
| **L5** | Human Gate | 🏗️ NEXT | P15 | **User Approval (The Brake)** |
| **L6** | Execution | 📅 TODO | P8 | Worker Agents (Apply/Submit) |
| **L7** | Analytics | 📅 TODO | P9, P31 | Feedback & Self-Learning |
| **L8** | Memory/Models | ✅ DONE | P3, P18 | RAG & Vector DB (Chroma) |
| **L9** | Governance | 🏗️ BUSY | P7, P16 | Guardrails & XAI (Explainability) |



## 🤖 Tiered Agent Rules for Codex
1. **L2 C-Suite:** The CEO must output a `SearchStrategy` JSON. The Director must validate it against `user_constraints`.
2. **Manager Agents (L3):** Execute specific tactics (e.g., scoring 50 jobs).
3. **L4 Creator Agents:** These agents MUST pull data from the `EvidenceProfile` (L2 artifact). [cite_start]They are strictly forbidden from inventing experience[cite: 21, 204].
4. **L5 Human-in-the-Loop:** No agent in L6 (Execution) can trigger an action unless the `StateModel` contains an `is_approved: true` flag from L5.
5. [cite_start]**Safety Brake:** L6 Worker agents are logically locked until an L5 human approval artifact exists[cite: 11, 24].
6. [cite_start]**L7 Feedback Loop:** After an L6 Worker completes an application, the Analyst agent must update the "Success Metrics" to inform next week's CEO strategy[cite: 2284].



## 🛠️ Tech Stack & Constraints
- **State Management:** LangGraph (Stateful Graph).
- **Communication:** Every step must produce a validated JSON artifact.
- **Grounding:** No agent can invent skills. Grounding source: `src/data/evidence_profile.json`.