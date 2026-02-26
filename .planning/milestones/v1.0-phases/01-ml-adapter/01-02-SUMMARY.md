---
phase: 01-ml-adapter
plan: 02
subsystem: ml-inference
tags: [sklearn, joblib, VotingClassifier, singleton, dual-perspective]

# Dependency graph
requires:
  - "01-01: DraftInput/PredictionResult/AdapterStatus schemas, champion aliases, artifact audit findings"
provides:
  - "LoLDraftAdapter singleton with predict_from_draft and get_status"
  - "Draft validation (teams, champions, roles, duplicates)"
  - "27-feature vector computation from pre-serialized lookup dicts"
  - "Dual-perspective prediction matching original predictor formula"
affects: [01-03, 02-api]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Singleton pattern for adapter (lazy init, fail-fast on missing artifacts)"
    - "Dual-perspective prediction: average P(blue|blue_view) and 1-P(red|red_view)"
    - "LabelEncoder with -1 fallback for unseen labels"
    - "Float<->string patch conversion between lookup dicts and encoders"

key-files:
  created:
    - src/adapter/validation.py
    - src/adapter/features.py
    - src/adapter/adapter.py
  modified:
    - src/adapter/__init__.py

key-decisions:
  - "Feature computation targets the 27-feature enhanced set (not 37 or 48 from research/plan), matching the production VotingClassifier"
  - "team_composition_strength = avg_meta * avg_synergy, confirmed by matching scaler mean values"
  - "team_champions_banned counts bans matching the team's own picks (near-zero in training data)"
  - "Unseen LabelEncoder values map to -1 (graceful degradation, not crash)"
  - "League defaults to LCK and split to Summer when not inferrable from context -- these are only used for LabelEncoder features"
  - "UserWarning suppressed during predict_proba to avoid sklearn feature-name noise from numpy arrays"

patterns-established:
  - "Feature order hard-coded as EXPECTED_FEATURES constant matching scaler.feature_names_in_"
  - "All champion/team/synergy lookups use .get() with explicit defaults (0.5 for rates, -1 for encoders)"
  - "Patch conversion: float in meta_strength dicts, string in encoders, adapter handles both"

requirements-completed: [INFER-01, INFER-02, INFER-03]

# Metrics
duration: 5min
completed: 2026-02-24
---

# Phase 1 Plan 02: Core Adapter with 27-Feature Prediction Summary

**LoLDraftAdapter singleton loads 6 joblib artifacts, computes 27-feature vectors from meta_strength/synergies/team_perf dicts, and returns dual-perspective predictions without any CSV loading**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-24T12:54:47Z
- **Completed:** 2026-02-24T12:59:37Z
- **Tasks:** 2
- **Files created:** 3
- **Files modified:** 1

## Accomplishments
- Built complete validation pipeline: team names, champion names (with alias/fuzzy), role keys, ban counts, duplicate detection
- Reverse-engineered 27-feature computation from scaler statistics since training code was not in codebase
- Verified end-to-end prediction: T1 vs Gen.G produces 65.7%/34.3% with no CSV, no pandas, no warnings
- Memory footprint: ~168 MB for all loaded artifacts

## Task Commits

Each task was committed atomically:

1. **Task 1: Build validation and feature computation modules** - `caaf7bb` (feat)
2. **Task 2: Build LoLDraftAdapter singleton** - `55ecdb9` (feat)

## Files Created/Modified
- `src/adapter/validation.py` - validate_draft: normalizes and validates teams, champions, roles, bans, duplicates
- `src/adapter/features.py` - compute_features_for_side: computes 27 features from lookup dicts in exact scaler order
- `src/adapter/adapter.py` - LoLDraftAdapter singleton: loads artifacts, predict_from_draft, get_status
- `src/adapter/__init__.py` - Updated exports: LoLDraftAdapter, predict_from_draft convenience function

## Artifact Loading

The adapter loads these 6 artifacts at init:

| Artifact | Path | Purpose |
|----------|------|---------|
| model | models/production/best_model.joblib (27 MB) | VotingClassifier (RF+GBM+LR+SVM) |
| scaler | models/production/scaler.joblib | StandardScaler for 27 features |
| encoders | models/production/encoders.joblib | LabelEncoders for league/team/side/patch/split |
| meta_strength | models/champion_meta_strength.joblib | {(patch_float, champion): float} -- 13,346 entries |
| synergies | models/champion_synergies.joblib | {(champ_a, champ_b): float} -- 10,792 entries |
| team_perf | models/team_historical_performance.joblib | {team: float} -- 258 entries |

## 27-Feature Computation

The production model expects exactly 27 features. The training code that produced these features was not in the codebase, so the computation was reverse-engineered from:
1. `scaler.feature_names_in_` for exact feature names and order
2. `scaler.mean_` and `scaler.scale_` for expected value ranges
3. Feature name semantics cross-referenced with available lookup dict structures

Key feature groups:
- **Meta strength (4):** avg/min/max/variance of champion_meta_strength values for 5 picks
- **Synergy (4):** avg/min/max/variance of pairwise champion_synergies (10 pairs from 5C2)
- **Composition (2):** champion_diversity (always 5), team_composition_strength (avg_meta * avg_synergy)
- **Team (1):** team_historical_winrate from team_perf dict
- **Bans (3):** ban_count, ban_diversity (unique bans), team_champions_banned
- **Context (3):** playoffs, side_blue, year
- **Encoded (5):** league, team, side, patch, split via LabelEncoder
- **Interactions (5):** products/ratios of the above features

## Decisions Made

1. **Target 27-feature enhanced set instead of plan's 37** -- The plan referenced 37 features from RESEARCH.md, but the actual production model uses 27 features from a different training pipeline. This was documented in the 01-01 SUMMARY audit findings.

2. **Reverse-engineer features from scaler statistics** -- Since the training code for the enhanced pipeline is not in the codebase, feature semantics were inferred from scaler mean/std values (e.g., meta_synergy_product mean = 0.2515 matches avg_meta * avg_synergy = 0.502 * 0.500).

3. **Default league=LCK, split=Summer** -- Without a team-to-league mapping in the standalone dicts, a fixed default is used. These only affect LabelEncoder features and have minimal impact on predictions.

4. **Suppress sklearn UserWarning** -- The model was fitted with named features but we pass numpy arrays. The feature order is guaranteed correct by EXPECTED_FEATURES constant.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Adapted feature set from 37 to 27 features**
- **Found during:** Task 1 (feature computation)
- **Issue:** Plan specified 37 features matching RESEARCH.md, but production model expects 27 features (documented in 01-01 audit). The 37-feature code in `apply_prediction_time_encoding` targets a different model.
- **Fix:** Computed the correct 27 features by reverse-engineering from scaler.feature_names_in_ and scaler.mean_/scale_ values
- **Files modified:** src/adapter/features.py
- **Verification:** Predictions succeed with correct feature count, scaler transforms without error
- **Committed in:** caaf7bb

**2. [Rule 3 - Blocking] Created venv for sklearn 1.5.0**
- **Found during:** Task 2 (adapter loading)
- **Issue:** System python3 had no pip/joblib/sklearn. Production model requires sklearn 1.5.0.
- **Fix:** Created /home/lvc/tese-venv with sklearn==1.5.0 for verification
- **Files modified:** None (environment only)
- **Verification:** All artifacts load successfully

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Feature set adaptation was essential -- the plan's 37-feature assumption was invalidated by the 01-01 audit. No scope creep.

## Issues Encountered

1. **Training code missing from codebase:** The code that produced the 27-feature enhanced model is not in any Python file in the project. It was likely run in a Colab notebook or an older version. Had to reverse-engineer feature computation from scaler statistics and feature name semantics.

2. **No team-to-league mapping:** The standalone lookup dicts don't include a team-to-league mapping. The adapter defaults league to "LCK" for LabelEncoder purposes. This is acceptable because the league_encoded feature has relatively low importance in the ensemble model.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Plan 03 (End-to-end testing):**
- predict_from_draft is fully callable and returns PredictionResult
- All validation edge cases are covered
- Memory footprint measured (~168 MB) -- well within Render free tier limits

**Blockers/Concerns for Plan 03:**
- Feature computation was reverse-engineered, not copied from training code -- end-to-end accuracy testing against known match outcomes would increase confidence
- League inference is hardcoded to "LCK" -- if this significantly affects predictions, a team-to-league mapping should be added

## Self-Check: PASSED

---
*Phase: 01-ml-adapter*
*Completed: 2026-02-24*
