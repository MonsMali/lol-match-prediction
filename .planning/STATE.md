# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-24)

**Core value:** Users can simulate a professional draft as it happens live and instantly see the predicted win probability for each team
**Current focus:** Phase 1 - ML Adapter

## Current Position

Phase: 1 of 4 (ML Adapter)
Plan: 0 of 3 in current phase
Status: Ready to plan
Last activity: 2026-02-24 — Roadmap and state initialized

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: none yet
- Trend: -

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Init]: Single FastAPI service serves both REST API and compiled React SPA as static files — one Render deployment, no split cold starts
- [Init]: Riot Data Dragon CDN used browser-direct for champion portraits — no backend proxying, no bundled assets
- [Init]: Model upload endpoint (API-07) is included in v1 but must be protected with a secret token header due to joblib pickle RCE risk
- [Init]: Railway excluded (trial-only as of 2025); Render free tier is the deployment target

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 1]: Exact RAM footprint of loading only pre-serialized joblib artifacts (without CSV) is unmeasured — must profile with `memory_profiler` before first Render deploy; if over ~400 MB, upgrade to Render Starter ($7/month) is the mitigation
- [Phase 1]: scikit-learn version used during last Colab training run is not confirmed — must match the version pinned in production requirements.txt or model loading will silently break
- [Phase 5 risk, now Phase 3]: Step-by-step live draft mode requires `InteractiveLoLPredictor` to handle UNKNOWN champion tokens for partial drafts — MEDIUM confidence that this works end-to-end; validate with a test call before building the LIVE mode backend endpoint

## Session Continuity

Last session: 2026-02-24
Stopped at: Roadmap created, STATE.md initialized — ready to plan Phase 1
Resume file: None
