# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-24)

**Core value:** Users can simulate a professional draft as it happens live and instantly see the predicted win probability for each team
**Current focus:** Phase 1 - ML Adapter

## Current Position

Phase: 1 of 4 (ML Adapter)
Plan: 1 of 3 in current phase
Status: In progress
Last activity: 2026-02-24 — Completed 01-01-PLAN.md (Artifact Audit and Adapter Schemas)

Progress: [█░░░░░░░░░] 10%

## Performance Metrics

**Velocity:**
- Total plans completed: 1
- Average duration: 9 min
- Total execution time: 0.15 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-ml-adapter | 1/3 | 9 min | 9 min |

**Recent Trend:**
- Last 5 plans: 01-01 (9 min)
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
- [01-01]: Production model is VotingClassifier (RF+GBM+LR+SVM) with 27 features, NOT standalone LogisticRegression — adapter must use enhanced feature set
- [01-01]: scikit-learn pinned to 1.5.0 — model fails to load on newer versions
- [01-01]: ultimate_feature_engineering.joblib is corrupted — must use standalone lookup dicts (champion_meta_strength, champion_synergies, team_historical_performance)
- [01-01]: Patch format is float in lookup dicts but string in encoders — adapter must handle conversion

### Pending Todos

None.

### Blockers/Concerns

- [Phase 1]: Exact RAM footprint of loading only pre-serialized joblib artifacts (without CSV) is unmeasured — must profile with `memory_profiler` before first Render deploy; if over ~400 MB, upgrade to Render Starter ($7/month) is the mitigation
- [Phase 1 RESOLVED]: scikit-learn version confirmed as 1.5.0 via artifact inspection
- [Phase 1 NEW]: Feature computation for Plan 02 must target the 27-feature enhanced set (not 37 or 48 from research) — the feature engineering logic must be reverse-engineered from the enhanced training code
- [Phase 1 NEW]: ultimate_feature_engineering.joblib corruption means richer lookup dicts (champion_characteristics with multi-field values) are unavailable — standalone dicts have simpler structures (single float values)
- [Phase 5 risk, now Phase 3]: Step-by-step live draft mode requires `InteractiveLoLPredictor` to handle UNKNOWN champion tokens for partial drafts — MEDIUM confidence that this works end-to-end; validate with a test call before building the LIVE mode backend endpoint

## Session Continuity

Last session: 2026-02-24T12:47:01Z
Stopped at: Completed 01-01-PLAN.md — adapter schemas and normalization created, audit findings documented
Resume file: None
