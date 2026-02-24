---
phase: 02-fastapi-backend
verified: 2026-02-24T14:46:07Z
status: gaps_found
score: 11/12 must-haves verified
gaps:
  - truth: "New model artifacts are validated with a test prediction before swapping the live model"
    status: failed
    reason: "Admin upload endpoint builds the test draft using champions as a list, but predict_from_draft requires blue_picks and red_picks as role-keyed dicts ({top, jungle, mid, bot, support}). The test prediction always raises AttributeError: 'list' object has no attribute 'keys' and the upload endpoint returns 422 before any model swap occurs."
    artifacts:
      - path: "api/routers/admin.py"
        issue: "Lines 105-111 build dummy_draft with blue_picks=champions[:5] (list) and red_picks=champions[10:15] (list). DraftInput requires dict[str, str] with role keys."
    missing:
      - "Change blue_picks and red_picks in dummy_draft to role-keyed dicts, e.g.: {'top': champions[0], 'jungle': champions[1], 'mid': champions[2], 'bot': champions[3], 'support': champions[4]}"
      - "Same fix for red_picks using champions[10:15]"
---

# Phase 2: FastAPI Backend Verification Report

**Phase Goal:** A running FastAPI service exposes the ML adapter as REST endpoints with correct schemas, CORS, model lifecycle management, and champion/team data endpoints
**Verified:** 2026-02-24T14:46:07Z
**Status:** gaps_found
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | FastAPI app starts with lifespan that loads LoLDraftAdapter singleton into app.state | VERIFIED | TestClient with context triggers lifespan; health returns status="ready", model_name="VotingClassifier" |
| 2 | GET /health returns JSON with status field even before model finishes loading | VERIFIED | Without lifespan context, GET /health returns 200 {status: "loading"}; with context returns {status: "ready", champion_count: 167} |
| 3 | CORS middleware allows requests from localhost:5173 (Vite dev server) | VERIFIED | CORSMiddleware registered with allow_origins=["http://localhost:5173", "http://localhost:3000"] |
| 4 | Champion name-to-DDragon ID mapping covers all known mismatches | VERIFIED | 12 entries in DDRAGON_ID_MAP including Wukong->MonkeyKing, Nunu & Willump->Nunu, Kai'Sa->Kaisa (correct DDragon ID; plan spec said KaiSa which would 404) |
| 5 | Team-to-league grouping dict provides LCK, LEC, LCS, LPL team lists | VERIFIED | TEAMS_BY_LEAGUE has 4 leagues x 10 teams each with canonical full names |
| 6 | POST /api/predict accepts a complete draft payload and returns blue_win_probability and red_win_probability | VERIFIED | Returns 200 {blue_win_probability: 0.292, red_win_probability: 0.708} for valid draft |
| 7 | POST /api/predict returns 422 with detail message when champion names are invalid | VERIFIED | Returns 422 {detail: "Invalid champion names: NotAChamp"} |
| 8 | POST /api/predict returns 503 when model is still loading | VERIFIED | Without lifespan, POST /api/predict returns 503 {detail: "Model is loading"} |
| 9 | Unknown team names are accepted with silent fallback (no 422 error for unknown teams) | VERIFIED | UNKNOWN_TEAM_XYZ accepted, warning logged, fallback team used, returns 200 with probabilities |
| 10 | GET /api/champions returns only champions the model knows about, each with DDragon image URL | VERIFIED | Returns 167 champions; Wukong has correct URL .../MonkeyKing.png |
| 11 | GET /api/teams returns professional teams grouped by league (LCK, LEC, LCS, LPL plus Other) | VERIFIED | Returns LCK, LEC, LCS, LPL, Other; Other contains remaining 258 valid teams not in hardcoded rosters |
| 12 | New model artifacts are validated with a test prediction before swapping the live model | FAILED | Test draft in admin.py passes picks as list; adapter requires role-keyed dict; test prediction always raises AttributeError |

**Score:** 11/12 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `api/main.py` | FastAPI app, lifespan, CORS, router inclusion | VERIFIED | 71 lines, lifespan context, CORSMiddleware, 5 routers included |
| `api/schemas.py` | Pydantic request/response models | VERIFIED | 77 lines, all 7 model classes defined |
| `api/dependencies.py` | get_adapter and require_admin_token | VERIFIED | 44 lines, both functions implemented |
| `api/champion_mapping.py` | DDragon ID mapping and URL builder | VERIFIED | 55 lines, 12 entries in DDRAGON_ID_MAP, get_ddragon_url function |
| `api/team_data.py` | Team-to-league grouping | VERIFIED | 64 lines, 4 leagues x 10 teams each |
| `api/routers/health.py` | GET /health endpoint | VERIFIED | 36 lines, always responds regardless of model state |
| `api/routers/predict.py` | POST /api/predict endpoint | VERIFIED | 111 lines, champion validation + team fallback + adapter call |
| `api/routers/champions.py` | GET /api/champions endpoint | VERIFIED | 37 lines, iterates valid_champions with DDragon URLs |
| `api/routers/teams.py` | GET /api/teams endpoint | VERIFIED | 43 lines, filters TEAMS_BY_LEAGUE against valid_teams |
| `api/routers/admin.py` | POST /api/admin/upload-model endpoint | PARTIAL | 154 lines, auth gate + singleton reset present, but test prediction format is broken |
| `requirements-prod.txt` | Production requirements with scikit-learn==1.5.0 | VERIFIED | 21 lines, scikit-learn==1.5.0 pinned, all training deps excluded |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `api/main.py` | `src.adapter.LoLDraftAdapter` | lifespan context manager | WIRED | LoLDraftAdapter() called in lifespan, stored to app.state.adapter |
| `api/dependencies.py` | `api/main.py` | `request.app.state.adapter` | WIRED | Checks model_ready on app.state, returns adapter or 503 |
| `api/routers/predict.py` | `src.adapter.LoLDraftAdapter` | `Depends(get_adapter)` -> `adapter.predict_from_draft` | WIRED | get_adapter dependency injected, predict_from_draft called with draft_dict |
| `api/routers/champions.py` | `api/champion_mapping.py` | `get_ddragon_url` per valid champion | WIRED | Iterates sorted(adapter.valid_champions), calls get_ddragon_url for each |
| `api/routers/teams.py` | `api/team_data.py` | `TEAMS_BY_LEAGUE` filtered against `adapter.valid_teams` | WIRED | Filters each league roster against valid, collects remainder as Other |
| `api/routers/admin.py` | `src.adapter.LoLDraftAdapter` | singleton reset after validation | PARTIAL | Reset logic (_instance=None) present and correct; test prediction validation is broken (list vs dict format) |
| `api/routers/admin.py` | `api/dependencies.py` | `Depends(require_admin_token)` | WIRED | require_admin_token in Depends, 401 returned for wrong token |

### Requirements Coverage

| Requirement | Description | Status | Blocking Issue |
|-------------|-------------|--------|---------------|
| API-01 | FastAPI REST endpoint accepts complete draft, returns win probabilities | SATISFIED | None -- POST /api/predict functional end-to-end |
| API-02 | ML model loads once at startup via lifespan, stored in app.state | SATISFIED | None -- lifespan pattern fully implemented |
| API-03 | Champion list endpoint returns valid champions with DDragon image URL | SATISFIED | None -- 167 champions with correct URLs |
| API-04 | Team list endpoint returns teams grouped by league | SATISFIED | None -- LCK/LEC/LCS/LPL + Other |
| API-05 | CORS configured to allow frontend requests | SATISFIED | None -- localhost:5173 and :3000 allowed |
| API-06 | Health/readiness endpoint returns model status and version info | SATISFIED | None -- always responds; full info when ready |
| API-07 | Model file upload endpoint with security token gate | BLOCKED | Test prediction in admin endpoint uses incorrect format (list instead of dict for picks); upload validation always fails before swap can occur |
| API-08 | Data Dragon champion name mapping handles mismatches | SATISFIED | None -- 12 entries covering Wukong, Nunu, all apostrophe names |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | - | - | - | No stub patterns, TODO/FIXME comments, placeholder text, or empty implementations found in any API file |

### Human Verification Required

No items require human verification. All observable behaviors were verified programmatically via TestClient integration tests.

### Gaps Summary

One gap blocks full goal achievement. The admin model upload endpoint at `api/routers/admin.py` constructs a dummy draft for test prediction validation using champion lists (`champions[:5]`) for `blue_picks` and `red_picks`. The `LoLDraftAdapter.predict_from_draft` method requires these fields as role-keyed dicts (`{top, jungle, mid, bot, support}`) per the `DraftInput` schema. The mismatch causes `AttributeError: 'list' object has no attribute 'keys'` on every test prediction attempt, meaning the upload endpoint will always return 422 and the hot-swap validation path is never reachable.

The fix is minimal: replace the list slices with explicit role dicts in the dummy_draft construction block.

All other 11 must-haves are fully verified. The primary user-facing endpoints (predict, champions, teams, health) are functional and wired correctly. The admin endpoint's security gate (Bearer token) works correctly for authentication. Only the internal validation logic for the test prediction is broken.

---

_Verified: 2026-02-24T14:46:07Z_
_Verifier: Claude (gsd-verifier)_
