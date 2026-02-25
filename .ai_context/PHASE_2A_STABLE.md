# 🛡️ Phase 2A: Agentic Foundation (P13 - P14)

## ✅ Completed Stages
| Stage | Artifact | Location | XAI Feature |
| :--- | :--- | :--- | :--- |
| **P13** | `StateModel` | `outputs/state/` | Stores "Decision History" in the JSON object. |
| **P14** | `tool_registry.py` | `src/api/orchestrator/` | Standardizes tools with docstrings for LLM "Explainability." |

## 🚦 Operational Rules for P15+ (The "Baton")
1. **State Path:** Use `outputs/state/current_run.json` for all P15+ state tracking.
2. **Human Intercept:** When `is_approved` is `false`, the Orchestrator must stop and return a `401_REQUIRES_HUMAN` status to the API.
3. **Black Box Prevention:** Every Match Result must now include a `match_explanation` dictionary containing:
   - `top_skills_found`: List[str]
   - `missing_skills`: List[str]
   - `confidence_score`: float