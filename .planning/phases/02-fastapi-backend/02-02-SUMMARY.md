---
phase: 02-fastapi-backend
plan: 02
subsystem: api
tags: [fastapi, rest, prediction, champions, teams, ddragon]

# Dependency graph
requires:
  - phase: 01-ml-adapter
    provides: "LoLDraftAdapter with predict_from_draft, valid_champions, valid_teams"
  - phase: 02-fastapi-backend/01
    provides: "FastAPI app scaffold with lifespan, schemas, dependencies, champion mapping, team data"
provides:
  - "POST /api/predict endpoint with champion validation and silent team fallback"
  - "GET /api/champions endpoint returning model-known champions with DDragon URLs"
  - "GET /api/teams endpoint returning teams grouped by league"
affects: [03-react-frontend, 02-fastapi-backend/03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Champion validation at API layer before adapter call (fail fast)"
    - "Silent team fallback on unknown names (no 422 for teams)"
    - "Sync def for blocking ML inference (FastAPI auto-threads)"
    - "Async def for lightweight data endpoints"

key-files:
  created:
    - api/routers/predict.py
    - api/routers/champions.py
    - api/routers/teams.py
  modified:
    - api/main.py

key-decisions:
  - "Champion names validated at API layer before adapter to provide cleaner 422 errors"
  - "Unknown teams silently fall back to first sorted valid team (no user-facing error)"
  - "predict endpoint uses plain def (not async def) so FastAPI runs it in threadpool for blocking ML inference"
  - "Teams endpoint filters TEAMS_BY_LEAGUE against adapter.valid_teams and collects remainder under Other"

patterns-established:
  - "API-layer champion validation: normalize then reject unknowns with 422"
  - "Silent team fallback: catch ValueError on unknown team, substitute, retry"
  - "TestClient context manager required for lifespan events"

requirements-completed: [API-01, API-03, API-04]

# Metrics
duration: 4min
completed: 2026-02-24
---

# Phase 2 Plan 02: Core Data Endpoints Summary

**Three REST endpoints for draft prediction, champion listing with DDragon URLs, and league-grouped team listing, all wired into FastAPI app**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-24T14:36:40Z
- **Completed:** 2026-02-24T14:40:10Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- POST /api/predict accepts full draft payloads, validates 20 champion names, returns blue/red win probabilities
- GET /api/champions returns 167 model-known champions with DDragon CDN portrait URLs
- GET /api/teams returns teams grouped by LCK/LEC/LCS/LPL/Other, filtered to model-known teams
- All 6 smoke tests pass: health, champions, teams, predict, invalid champion 422, unknown team silent fallback

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement predict, champions, and teams routers** - `83b1885` (feat)
2. **Task 2: Register routers in app and smoke test** - already committed in `e45699d` (02-03 plan updated main.py with all routers)

**Plan metadata:** (pending)

## Files Created/Modified
- `api/routers/predict.py` - POST /api/predict with champion validation and silent team fallback
- `api/routers/champions.py` - GET /api/champions returning sorted champions with DDragon URLs
- `api/routers/teams.py` - GET /api/teams returning league-grouped teams filtered to model-known set
- `api/main.py` - Wired predict, champions, teams routers (already updated by 02-03 execution)

## Decisions Made
- Champion validation happens at API layer before calling adapter -- collects all invalid names and returns them in a single 422 response
- Unknown team names trigger a fallback to the first alphabetically sorted valid team (100 Thieves) with a server-side warning log, no error to user
- predict endpoint is sync (plain `def`) to let FastAPI auto-thread the blocking ML inference; champions and teams are `async def` since they are lightweight

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- TestClient must be used as context manager (`with TestClient(app) as client:`) for lifespan events to fire; without it, model_ready stays False and all adapter-dependent endpoints return 503
- main.py was already updated by a prior 02-03 plan execution, so Task 2 commit was a no-op for that file

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All three core data endpoints operational and verified via smoke tests
- Frontend can consume /api/champions, /api/teams, and /api/predict
- Admin upload endpoint from Plan 03 already exists alongside these routes

## Self-Check: PASSED

---
*Phase: 02-fastapi-backend*
*Completed: 2026-02-24*
