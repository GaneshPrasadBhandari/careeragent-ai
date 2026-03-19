# Stepwise Layer Review

## Frontend / Mission Control
- The main UI risk area is duplicate Streamlit widget rendering when the same helper is mounted in multiple branches of a single run.
- Job-link rendering should stay centralized so every surface uses the same URL normalization and click behavior.
- Navigation changes should be introduced conservatively because tabs, sidebar navigation, and fallback rendering can easily conflict.

## L0 — Guardrails
- Current behavior is simple and stable.
- Main review point: keep failures explicit and avoid blocking the run on non-security parsing issues.

## L1 — Run Initialization
- Configuration hydration is mostly stable.
- Review point: preserve defaults consistently between API, UI, and persisted run state.

## L2 — Resume Parse
- Parsing is resilient because the pipeline continues on failure with a minimal profile.
- Review point: keep parser failures observable in `layer_debug` and `errors`.

## L3 — Discovery
- Discovery is the highest-risk layer for bad external data.
- Review points:
  - enforce region-aware source filtering after URL unwrapping,
  - prefer direct board URLs over intermediary Google links,
  - keep fallback/demo leads region-aware.

## L4 — Match / Score
- This is the most workflow-sensitive layer.
- Review points:
  - keep one shared scoring path for normal runs and reruns,
  - preserve original lead metadata and direct URLs through enrichment,
  - keep cognitive reasoning bounded by timeouts with heuristic fallback.

## L5 — Rank / HITL
- Ranking and approval logic is functional but tightly coupled to frontend expectations.
- Review point: ensure approval-gate state always lines up with what the dashboard renders.

## L6 — Draft Generation
- Draft generation is mostly isolated.
- Review point: keep artifact creation failures local so they do not corrupt upstream ranking state.

## L7 — Apply
- Apply execution depends heavily on contact/profile completeness.
- Review point: keep queued/submitted states distinct and preserve original job URLs for auditability.

## L8 — Tracking
- Tracking is lightweight and low risk.
- Review point: keep persistence failures non-fatal and visible.

## L9 — Analytics
- Analytics is useful but should not be allowed to destabilize the run.
- Review point: keep it as a terminal summarization layer with safe fallbacks.

## Recommended Fix Order
1. Frontend rendering stability.
2. L3 direct-link and region correctness.
3. L4 scoring/link preservation.
4. L5 approval-state/UI alignment.
5. L6–L9 hardening and observability cleanup.
