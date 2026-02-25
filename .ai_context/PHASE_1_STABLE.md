# 🛡️ Phase 1: The Deterministic Spine (Stable reference)

## 🏗️ Core Architecture: The "Artifact Contract"
Phase 1 (P1–P12) is built on **Deterministic Functions** (no LLMs/Agents). [cite_start]Every step must produce a validated JSON artifact or the system stops[cite: 4, 5, 7, 8].

## ✅ Completed Pipeline Stages (P1 - P12)
| Stage | Artifact Produced | Location | Logic Used |
| :--- | :--- | :--- | :--- |
| **P1** | `intake_bundle.json` | `outputs/intake/` | [cite_start]Pydantic validation of user constraints[cite: 9, 11, 12]. |
| **P2** | `profile.json` | `outputs/profile/` | [cite_start]Deterministic keyword extraction from resume[cite: 13, 14, 15]. |
| **P3** | `job_post.json` | `outputs/jobs/` | [cite_start]Deterministic keyword extraction from Job Description[cite: 16, 17]. |
| **P4/5**| `match_result.json` | `outputs/match/` | Set Math: `(Skills ∩ Keywords) / Total Keywords`. **(Note: In Phase 2+, this is wrapped into `outputs/state/current_run.json`)** |
| **P6/7**| `validated_gen.md` | `outputs/gen/` | [cite_start]Template filling + Guardrail claim-check[cite: 21, 22, 23]. |
| **P8** | `.docx / .pdf` export | `exports/` | [cite_start]Conversion of approved JSON to document[cite: 24]. |
| **P9** | `application_ledger` | `logs/` | [cite_start]Persistent log to prevent duplicate applications[cite: 25]. |
| **P10/11**| `action_reminders` | `logs/` | [cite_start]Date-math for follow-up notifications[cite: 26]. |
| **P12** | **One-Click Orch** | `src/orchestrator`| [cite_start]Sequential execution of P1 through P11[cite: 27, 28]. |

## 🛠️ Operational Truths for Codex
- [cite_start]**Validation:** All schemas are defined in `src/careeros/{module}/schema.py` using **Pydantic V2**[cite: 11, 30].
- **Grounding:** The `profile.json` is the **Single Source of Truth** for all generation. [cite_start]No agent may "invent" skills[cite: 15, 23].
- [cite_start]**Traceability:** Every run generates a unique `run_id` found in `logs/careeros.jsonl`[cite: 32, 445].