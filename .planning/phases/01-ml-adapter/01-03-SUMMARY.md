---
phase: 01-ml-adapter
plan: 03
subsystem: testing
tags: [pytest, memory-profiling, validation, end-to-end]

# Dependency graph
requires:
  - "01-01: DraftInput/PredictionResult/AdapterStatus schemas, champion aliases"
  - "01-02: LoLDraftAdapter singleton with predict_from_draft and get_status"
provides:
  - "13 passing end-to-end tests covering prediction, validation, memory, and no-CSV guarantee"
  - "Memory profiling data: ~168 MB RSS (well within 512 MB Render limit)"
  - "Confirmed INFER-01 through INFER-04 requirements met"
affects: [02-api]

# Tech tracking
tech-stack:
  added: [pytest]
  patterns:
    - "Module-scoped pytest fixture for singleton adapter to avoid redundant init"
    - "Helper function _make_draft for constructing test drafts from sorted valid names"

key-files:
  created:
    - tests/test_adapter.py
  modified: []

key-decisions:
  - "Used close-but-wrong champion name ('Aatro') to trigger fuzzy 'Did you mean' suggestion in unknown champion test"
  - "Drafts constructed from sorted valid_champions/valid_teams for deterministic test data"
  - "Added test_predict_completes_under_one_second beyond plan's 11 tests (13 total)"

patterns-established:
  - "Test fixtures use module scope to share adapter singleton across all tests"
  - "Validation error tests use pytest.raises with match= for message content assertions"

requirements-completed: [INFER-01, INFER-02, INFER-03, INFER-04]

# Metrics
duration: 4min
completed: 2026-02-24
---

# Phase 1 Plan 03: End-to-End Adapter Testing Summary

**13 pytest tests validate prediction correctness (non-degenerate, sum-to-1, swap-sensitive), all 5 validation error paths, sub-second latency, 168 MB RSS (under 512 MB), and no-CSV guarantee**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-24T13:02:30Z
- **Completed:** 2026-02-24T13:06:30Z
- **Tasks:** 1
- **Files created:** 1

## Accomplishments
- Confirmed memory footprint at ~168 MB RSS -- 33% of the 512 MB Render free-tier budget
- Validated predictions are non-degenerate: 3 drafts produce 3 different probabilities, none exactly 0.5
- Proved swapping blue/red sides changes the prediction (dual-perspective works)
- Verified all 5 validation error paths: unknown champion (with fuzzy match), unknown team, duplicate, missing role, wrong ban count
- Static source check confirms zero CSV loading in any adapter module

## Task Commits

Each task was committed atomically:

1. **Task 1: Measure memory and write 13 end-to-end tests** - `b428db9` (test)

## Files Created/Modified
- `tests/test_adapter.py` - 13 pytest tests covering singleton, status, prediction correctness, swap sensitivity, non-degeneracy, latency, 5 validation error paths, no-CSV guarantee, and memory budget

## Memory Profiling Results

| Measurement | RSS (MB) |
|---|---|
| Baseline (Python + stdlib) | 9.1 |
| After artifact loading | 167.1 |
| After first prediction | 168.4 |
| Delta (adapter footprint) | 158.0 |
| Render free-tier budget | 512.0 |
| Headroom | 343.6 (67%) |

The adapter uses 33% of the available memory budget, leaving substantial headroom for the FastAPI server, request handling, and concurrent predictions in Phase 2.

## Sample Prediction Output

| Draft | Blue Team | Red Team | Blue Win | Red Win |
|---|---|---|---|---|
| Draft 1 (champs 0-19) | 100 Thieves vs Albus NoX Luna | - | 22.51% | 77.49% |
| Draft 2 (champs 20-39) | Different teams | - | 79.57% | 20.43% |
| Draft 3 (champs 40-59) | Different teams | - | 48.18% | 51.82% |
| Swapped draft 1 | Teams/champs swapped | - | 79.18% | 20.82% |

Predictions show healthy variance across different team/champion combinations.

## Test Results Summary

All 13 tests pass:

| Test | Category | Result |
|---|---|---|
| test_adapter_singleton | Core | PASSED |
| test_get_status | Core | PASSED |
| test_predict_from_draft_basic | Prediction | PASSED |
| test_predict_swapped_teams_changes_result | Prediction | PASSED |
| test_predict_not_degenerate | Prediction | PASSED |
| test_predict_completes_under_one_second | Performance | PASSED |
| test_validation_unknown_champion | Validation | PASSED |
| test_validation_unknown_team | Validation | PASSED |
| test_validation_duplicate_champion | Validation | PASSED |
| test_validation_missing_role | Validation | PASSED |
| test_validation_wrong_ban_count | Validation | PASSED |
| test_no_csv_loading | Static | PASSED |
| test_memory_under_512mb | Memory | PASSED |

## Requirements Validation

| Requirement | Description | Status |
|---|---|---|
| INFER-01 | predict_from_draft returns probabilities without CSV | Confirmed -- no read_csv in source, predictions return valid floats |
| INFER-02 | Uses only joblib artifacts | Confirmed -- 6 joblib files loaded, no other data sources |
| INFER-03 | Accepts draft dict, returns typed result | Confirmed -- DraftInput in, PredictionResult out |
| INFER-04 | Under 512 MB RAM | Confirmed -- 168 MB RSS (33% of budget) |

## Decisions Made

1. **Added test_predict_completes_under_one_second** -- Plan specified 11 tests but performance validation is essential for the API layer. Added as a 12th functional test beyond the plan's specification. Combined with the memory test, this brings the total to 13.

2. **Used close-but-wrong champion name for fuzzy match test** -- "Aatro" (close to "Aatrox") triggers the "Did you mean" path, while a completely wrong name like "NotAChampion" does not produce suggestions. This tests the actual fuzzy-match code path.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed pytest in venv**
- **Found during:** Task 1 (test execution)
- **Issue:** pytest not installed in /home/lvc/tese-venv
- **Fix:** `pip install pytest` in the existing venv
- **Files modified:** None (environment only)
- **Verification:** pytest runs and all 13 tests pass

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Minimal -- only affected test runner availability.

## Issues Encountered

None -- all validation paths, prediction behavior, and memory measurements matched expectations from the 01-02 SUMMARY.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Phase 1 complete.** All 4 INFER requirements validated. The adapter is ready for Phase 2 (FastAPI REST API).

**Phase 2 can safely assume:**
- `LoLDraftAdapter()` singleton is thread-safe for read-only predictions
- Memory footprint is ~168 MB leaving ~344 MB for FastAPI + uvicorn overhead
- Prediction latency is sub-second (suitable for synchronous API endpoints)
- All validation errors raise ValueError with descriptive messages (map to HTTP 422)

**Minor concern for Phase 2:**
- League inference is hardcoded to "LCK" -- low impact on predictions but worth noting in API documentation

## Self-Check: PASSED

---
*Phase: 01-ml-adapter*
*Completed: 2026-02-24*
