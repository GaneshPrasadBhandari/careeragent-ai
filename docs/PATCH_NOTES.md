# CareerAgent-AI Patch (Planner–Director Soft-Fencing + L0–L9)

## What was broken
- Evaluator treated constraints as **hard gates** → `score=0.0` → `RETRY_SEARCH` loops.
- Retry loops overwrote good results with empty `[]` artifacts.
- Jobs were rejected if the word "India" appeared anywhere in the job body (global office lists).
- DOCX hyperlinks (LinkedIn/GitHub) were invisible to regex parsers.
- UI showed "No evaluations yet" because decisions were not persisted to `state.evaluations`.

## What this patch changes
### L0–L2 Core Brain
- **Planner** builds three dynamic personas (ATS/strict → ATS-preferred/remote → broad) from preferences.
- **Director** enforces **soft-fencing**: relax constraints and shift persona instead of collapsing to zero.
- **Parser** uses deterministic extraction + DOCX relationship hyperlink extraction + Gemini backfill.
- **L2 evaluator gate** validates intake bundle quality before proceeding.

### L3 Manager Cluster
- Lead Scout executes persona query with negative operators (e.g., `-India`) but does not hard reject US jobs.
- Geo-fence rejects ONLY when explicit location metadata is non-US.
- Extraction uses **Jina Reader** (`https://r.jina.ai/<url>`) and checks robots.txt.

### L4–L5
- Matching + ranking run deterministically.
- L5 evaluator uses Gemini (if available) and CRAG (Tavily) for missing company context.

### L6 Execution
- Generates ATS-friendly resume + cover letter in **MD + DOCX + PDF** per approved job.
- Computes ATS keyword match; if below 0.90, forces HITL review before finalization.

### L7 Analytics + Learning
- SQLite tracker: applied_date, company, priority, interview_status.
- Self-learning agent stores retry-loop signals.
- Career coach generates a 6-month roadmap (Gemini if available).

### L8 Memory
- Prevent duplicate applications via SQLite URL-hash memory (Qdrant optional).

### L9 Governance
- robots.txt compliance checks.
- token usage estimate + budget summary.

## Rollback
Use `scripts/backup_and_apply_patch.sh` to create `_rollback/<timestamp>/` before overwriting.
