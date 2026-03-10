# Phase 6 Launch Plan (Observability + Cloud + Beta)

## 1) LangSmith visibility debugging checklist

1. Set both tracing flags:
   - `LANGSMITH_TRACING=true`
   - `LANGCHAIN_TRACING_V2=true`
2. Set both project keys to same value:
   - `LANGSMITH_PROJECT=careeragent-ai-new`
   - `LANGCHAIN_PROJECT=careeragent-ai-new`
3. Ensure API key is present as either:
   - `LANGSMITH_API_KEY`, or
   - `LANGCHAIN_API_KEY`
4. Validate runtime status endpoint surfaces `langsmith.enabled=true` and a dashboard URL.

## 2) Render deployment approach (24x7 beta)

- Deploy two web services from `render.yaml`:
  - `careeragent-api` (FastAPI)
  - `careeragent-dashboard` (Streamlit)
- Add all secrets in Render Dashboard environment variables.
- Confirm API health endpoint before sharing dashboard URL.

## 3) Beta release execution

- Publish dashboard link via LinkedIn + friends cohort.
- Ask users to submit feedback through **Analytics → Feedback** in dashboard.
- Track recruiter responses via employer feedback events to strengthen self-learning loop.

## 4) Repository cleanup candidates (analysis first)

Potential legacy/non-runtime artifacts to archive after validation:
- `README1.md`
- `newfolder/`
- `ls` (empty file)
- old exploratory notebooks under `notebooks_v2/`

> Do not archive until each item is confirmed as unused by imports, scripts, or docs.
