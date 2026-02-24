# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-24)

**Core value:** Users can simulate a professional draft as it happens live and instantly see the predicted win probability for each team
**Current focus:** Phase 1 - ML Adapter

## Current Position

Phase: 1 of 4 (ML Adapter)
Plan: 2 of 3 in current phase
Status: In progress
Last activity: 2026-02-24 -- Completed 01-02-PLAN.md (Core Adapter with predict_from_draft)

Progress: [██░░░░░░░░] 20%

## Performance Metrics

**Velocity:**
- Total plans completed: 2
- Average duration: 7 min
- Total execution time: 0.23 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-ml-adapter | 2/3 | 14 min | 7 min |

**Recent Trend:**
- Last 5 plans: 01-01 (9 min), 01-02 (5 min)
- Trend: improving

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

### Pending Todos

None.

### Blockers/Concerns

- [Phase 1 RESOLVED]: RAM footprint measured at ~168 MB for all artifacts -- well within Render free tier limits
- [Phase 1 RESOLVED]: scikit-learn version confirmed as 1.5.0 via artifact inspection
- [Phase 1 RESOLVED]: Feature computation for Plan 02 successfully targets the 27-feature enhanced set, reverse-engineered from scaler.feature_names_in_ and mean/scale values
- [Phase 1 RESOLVED]: ultimate_feature_engineering.joblib corruption handled -- standalone dicts sufficient for 27-feature model
- [Phase 1 MINOR]: League inference hardcoded to LCK -- if this significantly affects predictions, a team-to-league mapping should be added in a future plan
- [Phase 3 risk]: Step-by-step live draft mode requires handling UNKNOWN champion tokens for partial drafts -- validate with a test call before building the LIVE mode backend endpoint

## Session Continuity

Last session: 2026-02-24T12:59:37Z
Stopped at: Completed 01-02-PLAN.md -- LoLDraftAdapter singleton with predict_from_draft created
Resume file: None
