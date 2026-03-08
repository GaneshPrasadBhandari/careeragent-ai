# Patch Notes — Planner/Director Soft-Fencing and L0–L9 Stabilization

## What was broken

- Constraint evaluation behaved like a hard gate, which drove `score=0.0` outcomes and caused repeated `RETRY_SEARCH` loops.
- Retry loops sometimes overwrote valid results with empty `[]` artifacts.
- Jobs were incorrectly rejected when the word "India" appeared anywhere in the job description, even if the actual role location was in the United States.
- DOCX hyperlinks such as LinkedIn or GitHub links were not reliably visible to regex-only parsers.
- The UI sometimes showed "No evaluations yet" because decisions were not being persisted into `state.evaluations`.

## What this patch changed

### L0–L2 Core Brain
- **Planner** now builds dynamic search personas ranging from strict ATS-fit to broader role exploration based on user preferences.
- **Director** now uses **soft-fencing** so constraints can be relaxed intelligently instead of collapsing the workflow to zero-value failure states.
- **Parser** now combines deterministic extraction, DOCX relationship hyperlink extraction, and Gemini-assisted backfill where available.
- **L2 evaluator gate** now validates intake bundle quality before downstream progression.

### L3 Manager Cluster
- **Lead Scout** executes persona-aware queries with negative operators such as `-India` without incorrectly hard-rejecting valid U.S.-based roles.
- **Geo-fencing** now rejects jobs only when explicit structured location metadata is non-U.S.
- **Extraction** uses **Jina Reader** (`https://r.jina.ai/<url>`) and checks `robots.txt` before content retrieval.

### L4–L5
- Matching and ranking now run through a more deterministic path.
- **L5 evaluator** can use Gemini and CRAG/Tavily-based retrieval support when additional company context is needed.

### L6 Execution
- The system generates ATS-friendly resume and cover-letter outputs in **Markdown, DOCX, and PDF** for approved jobs.
- ATS keyword match is computed, and low-match outputs can be forced back into HITL review before finalization.

### L7 Analytics and Learning
- SQLite tracking now captures fields such as applied date, company, priority, and interview status.
- Retry-loop signals are recorded for future system improvement.
- The career-coach path can generate longer-horizon upskilling guidance when supporting models are available.

### L8 Memory
- Duplicate applications are reduced through SQLite-based URL-hash memory, with optional vector-memory extensions.

### L9 Governance
- `robots.txt` compliance checks were strengthened.
- Token usage estimation and budget-summary tracking were added.

## Rollback

Use `scripts/backup_and_apply_patch.sh` to create `_rollback/<timestamp>/` before overwriting files.
