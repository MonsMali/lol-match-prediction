# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-24)

**Core value:** Users can simulate a professional draft as it happens live and instantly see the predicted win probability for each team
**Current focus:** Phase 3 - React Draft Board (In progress)

## Current Position

Phase: 3 of 4 (React Draft Board)
Plan: 1 of 5 in current phase
Status: In progress
Last activity: 2026-02-25 -- Completed 03-01-PLAN.md (Frontend foundation: Vite + React + Zustand + TanStack Query)

Progress: [███████████░░░░░] 54%

## Performance Metrics

**Velocity:**
- Total plans completed: 7
- Average duration: 5 min
- Total execution time: 0.63 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-ml-adapter | 3/3 | 18 min | 6 min |
| 02-fastapi-backend | 3/3 | 16 min | 5 min |
| 03-react-draft-board | 1/5 | 4 min | 4 min |

**Recent Trend:**
- Last 5 plans: 02-01 (9 min), 02-03 (3 min), 02-02 (4 min), 03-01 (4 min)
- Trend: stable

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Init]: Single FastAPI service serves both REST API and compiled React SPA as static files -- one Render deployment, no split cold starts
- [Init]: Riot Data Dragon CDN used browser-direct for champion portraits -- no backend proxying, no bundled assets
- [Init]: Model upload endpoint (API-07) is included in v1 but must be protected with a secret token header due to joblib pickle RCE risk
- [Init]: Railway excluded (trial-only as of 2025); Render free tier is the deployment target
- [01-01]: Production model is VotingClassifier (RF+GBM+LR+SVM) with 27 features, NOT standalone LogisticRegression -- adapter must use enhanced feature set
- [01-01]: scikit-learn pinned to 1.5.0 -- model fails to load on newer versions
- [01-01]: ultimate_feature_engineering.joblib is corrupted -- must use standalone lookup dicts (champion_meta_strength, champion_synergies, team_historical_performance)
- [01-01]: Patch format is float in lookup dicts but string in encoders -- adapter must handle conversion
- [01-02]: Feature computation targets 27-feature enhanced set, reverse-engineered from scaler statistics (training code not in codebase)
- [01-02]: Unseen LabelEncoder values map to -1 (graceful degradation)
- [01-02]: League defaults to LCK and split to Summer when not inferrable -- minimal impact on predictions
- [01-03]: Memory footprint confirmed at ~168 MB RSS -- 33% of 512 MB Render budget, leaving 344 MB headroom for FastAPI
- [02-01]: Team names use canonical full names from training data (Gen.G not GenG, Hanwha Life Esports not HLE)
- [02-01]: DDragon version pinned to 14.24.1
- [02-01]: Health endpoint does not use get_adapter dependency -- always responds even during loading
- [02-02]: Champion validation at API layer before adapter call -- collects all invalid names in single 422 response
- [02-02]: Unknown teams silently fall back to first sorted valid team (no user-facing error)
- [02-02]: predict endpoint uses sync def for auto-threading of blocking ML inference
- [02-03]: Test prediction uses object.__new__ to bypass singleton for artifact validation in isolation
- [02-03]: Production requirements pin scikit-learn==1.5.0 and exclude training-only deps
- [03-01]: Vite dev proxy for /api and /health avoids CORS in development -- API_BASE defaults to empty string
- [03-01]: apiFetch generic wrapper centralizes error handling and Content-Type headers for all API calls
- [03-01]: Draft store uses getSlotArray helper for mode-switching recalculation

### Pending Todos

None.

### Blockers/Concerns

- [Phase 1 RESOLVED]: All INFER-01 through INFER-04 requirements validated with 13 passing tests
- [Phase 1 RESOLVED]: RAM footprint measured at ~168 MB for all artifacts -- well within Render free tier limits
- [Phase 1 RESOLVED]: scikit-learn version confirmed as 1.5.0 via artifact inspection
- [Phase 1 RESOLVED]: Feature computation for Plan 02 successfully targets the 27-feature enhanced set
- [Phase 1 RESOLVED]: ultimate_feature_engineering.joblib corruption handled -- standalone dicts sufficient for 27-feature model
- [Phase 1 MINOR]: League inference hardcoded to LCK -- low impact on predictions, worth noting in API documentation
- [Phase 3 risk]: Step-by-step live draft mode requires handling UNKNOWN champion tokens for partial drafts -- validate with a test call before building the LIVE mode backend endpoint

## Session Continuity

Last session: 2026-02-25T11:56:19Z
Stopped at: Completed 03-01-PLAN.md -- Frontend foundation (Vite + React + Zustand + TanStack Query)
Resume file: None
