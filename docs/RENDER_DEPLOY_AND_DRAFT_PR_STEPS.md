# Phase 6 — Draft PR, Render Setup, VS Code Pull, and UI Test (Step-by-step)

Use this checklist exactly to finish Phase 6 today.

## A) Push branch + create Draft PR

### 1) Verify you are on Phase 6 branch
```bash
git checkout feature/phase6_refinement_observability_langsmith
git status
```

### 2) Push branch to GitHub
```bash
git push -u origin feature/phase6_refinement_observability_langsmith
```

### 3) Create Draft PR
Option 1 (GitHub UI):
- Open repo → **Compare & pull request**
- Base: `main` (or your release branch policy)
- Compare: `feature/phase6_refinement_observability_langsmith`
- Click **Create draft pull request**

Option 2 (GitHub CLI):
```bash
gh pr create \
  --base main \
  --head feature/phase6_refinement_observability_langsmith \
  --title "Phase 6: LangSmith observability, Render deploy prep, beta UX" \
  --body-file docs/PHASE6_LAUNCH_PLAN.md \
  --draft
```

---

## B) Pull same branch in VS Code

In VS Code terminal:
```bash
git fetch origin
git checkout feature/phase6_refinement_observability_langsmith
git pull --ff-only
```

---

## C) Render setup (yes, you must configure secrets manually)

`render.yaml` defines services, but **secret values are not auto-filled**. Add them in Render dashboard.

### 1) Create Blueprint deploy
- Render → **New +** → **Blueprint**
- Select GitHub repo
- Render reads `render.yaml` and creates:
  - `careeragent-api`
  - `careeragent-dashboard`
- `buildFilter.paths` is configured so dashboard-only edits do not redeploy API and API-only edits do not redeploy dashboard.

### 2) Required secrets/env vars for `careeragent-api`
Set these in Render (Environment tab):
- `LANGSMITH_API_KEY` (or `LANGCHAIN_API_KEY`)
- `GEMINI_API_KEY` / other LLM keys you use
- `SERPER_API_KEY` (if using Serper)
- `TAVILY_API_KEY` (if using Tavily)
- `LANGGRAPH_STUDIO_URL` (optional but recommended; enables direct LangGraph run-trace link in dashboard analytics)
- `DATABASE_URL` (if not default sqlite path)
- any notification keys you use (SendGrid/Resend/Twilio)

Already defaulted in yaml (can keep):
- `LANGSMITH_TRACING=true`
- `LANGCHAIN_TRACING_V2=true`
- `LANGSMITH_PROJECT=careeragent-ai-new`
- `LANGCHAIN_PROJECT=careeragent-ai-new`

### 3) Dashboard API URL binding
`careeragent-dashboard` uses:
- `API_BASE_URL` from `careeragent-api` service `url`

This allows Streamlit sidebar to prefill the backend URL automatically.

---

## D) Post-deploy validation

### API checks
- Open: `https://<careeragent-api>.onrender.com/health`
- Confirm healthy JSON response.

### Dashboard checks
- Open: `https://<careeragent-dashboard>.onrender.com`
- Verify sidebar backend URL is auto-filled from `API_BASE_URL`
- Upload a resume and start a run
- Check top-right LangSmith link appears when tracing is enabled

### LangSmith checks
- Confirm project name: `careeragent-ai-new`
- Confirm new traces appear after real run starts

---

## E) Beta release checklist (LinkedIn + friends)

- Share dashboard URL + 3-line usage instruction.
- Ask each tester to submit feedback via Analytics tab.
- Track recruiter/user feedback trends before Phase 7 planning.

