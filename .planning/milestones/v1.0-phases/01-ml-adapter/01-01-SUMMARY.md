---
phase: 01-ml-adapter
plan: 01
subsystem: ml-inference
tags: [sklearn, joblib, dataclass, normalization, difflib]

# Dependency graph
requires: []
provides:
  - "DraftInput, PredictionResult, AdapterStatus typed dataclasses"
  - "Champion alias map (49 entries) and normalization functions"
  - "Comprehensive audit of all serialized model artifacts"
affects: [01-02, 01-03, 02-api]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Stdlib dataclasses for typed schemas (no Pydantic yet)"
    - "difflib.get_close_matches for fuzzy champion suggestions"
    - "Case-insensitive alias map with lowercase keys"

key-files:
  created:
    - src/adapter/__init__.py
    - src/adapter/schemas.py
    - src/adapter/normalization.py
  modified: []

key-decisions:
  - "Production model is VotingClassifier (27 features), NOT standalone LogisticRegression (thesis reported LR as best, but deployed model is an ensemble)"
  - "ultimate_feature_engineering.joblib is corrupted (EOF on unpickle) -- must use standalone lookup dicts instead"
  - "ultimate_best_model.joblib requires catboost dependency -- cannot be used without installing catboost"
  - "sklearn version pinned to 1.5.0 based on successful load and multi_class=deprecated attribute"
  - "Patch format in lookup dicts is float (14.18), not string -- adapter must handle conversion"
  - "Standalone team_historical_performance has simple {team: float} structure, not the richer dict described in research"

patterns-established:
  - "Champion names use Oracle's Elixir display format: apostrophes (Kai'Sa), spaces (Lee Sin), full names (Nunu & Willump)"
  - "Normalize via lowercase alias map -> case-insensitive exact match -> fuzzy error"

requirements-completed: [INFER-01, INFER-02]

# Metrics
duration: 9min
completed: 2026-02-24
---

# Phase 1 Plan 01: Artifact Audit and Adapter Schemas Summary

**Audited all serialized model artifacts resolving 4 open questions; created src/adapter with typed DraftInput/PredictionResult/AdapterStatus dataclasses and 49-entry champion alias normalization**

## Performance

- **Duration:** 9 min
- **Started:** 2026-02-24T12:38:20Z
- **Completed:** 2026-02-24T12:47:01Z
- **Tasks:** 2
- **Files created:** 3

## Accomplishments
- Resolved all 4 open questions from RESEARCH.md with concrete runtime artifact inspection
- Created typed schemas matching CONTEXT.md contract (DraftInput, PredictionResult, AdapterStatus)
- Built champion alias map covering Riot API ids, abbreviations, apostrophe-less variants, and common misspellings
- Discovered critical deviation: production model is VotingClassifier (not standalone LR), feature engineering object is corrupted

## Task Commits

Each task was committed atomically:

1. **Task 1: Audit serialized artifacts** - No file artifacts (audit script was temporary and deleted)
2. **Task 2: Create adapter package** - `a2a4118` (feat)

## Files Created/Modified
- `src/adapter/__init__.py` - Package init exporting all public names
- `src/adapter/schemas.py` - DraftInput, PredictionResult, AdapterStatus dataclasses
- `src/adapter/normalization.py` - CHAMPION_ALIASES dict (49 entries) and normalize functions

## Audit Findings

### Open Question 1: Correct Model File

| File | Size | Type | Features | Loadable |
|------|------|------|----------|----------|
| `models/production/best_model.joblib` | 27 MB | VotingClassifier (RF+GBM+LR+SVM) | 27 | Yes (sklearn 1.5.0) |
| `models/enhanced_best_model.joblib` | 27 MB | VotingClassifier (identical to production) | 27 | Yes |
| `models/ultimate_best_model.joblib` | 231 KB | Unknown (requires catboost) | 48? | No |
| `models/ultimate_best_model_2302.joblib` | 231 KB | Unknown (requires catboost) | 48? | No |

**Decision:** Use `models/production/best_model.joblib`. It is a soft-voting VotingClassifier with 4 estimators (RandomForest, GradientBoosting, LogisticRegression, SVM) and near-equal weights (~0.25 each). This is the only loadable model without installing catboost. The 27-feature set matches the production scaler and encoders.

**Note:** The thesis documents LogisticRegression as the best model (82.97% AUC), but the deployed production artifact is an ensemble VotingClassifier using 27 features (not 37). The enhanced/production pipeline uses a different feature set than the "ultimate" pipeline (48 features). This is a significant finding.

### Open Question 2: scikit-learn Version

- `_sklearn_version` attribute is **NOT set** on any estimator (older serialization format)
- Model loads cleanly with sklearn **1.5.0**; fails with 1.8.0 due to `__pyx_unpickle_CyHalfBinomialLoss` removal
- `LogisticRegression.multi_class` = `deprecated` confirms sklearn >= 1.5.0
- **Pin:** `scikit-learn==1.5.0`

### Open Question 3: Lookup Dict Key Formats

**Standalone files (loadable):**

| Artifact | Type | Key Format | Count |
|----------|------|-----------|-------|
| `champion_meta_strength.joblib` | dict | `(float_patch, str_champion)` e.g. `(13.05, 'Kennen')` | 13,346 |
| `champion_synergies.joblib` | dict | `(str_champion, str_champion)` e.g. `('Kennen', 'Wukong')` | 10,792 |
| `team_historical_performance.joblib` | dict | `str_team` -> `float` (overall win rate only) | 258 |

**Patch format:** Patches are **floats** (e.g., `14.18`, `13.05`, `3.15`), NOT strings. The single exception is `"Unknown"` (string). Latest patch: **14.18**.

**Feature engineering object (corrupted, NOT loadable):**
- `ultimate_feature_engineering.joblib` -- EOFError during unpickle (likely binary corruption from OneDrive sync or git autocrlf on WSL cross-filesystem). Contains `\r\n` sequences and stray `\r` bytes in binary data.
- The richer lookup dict structures (champion_characteristics with win_rate/early_game_strength/etc., team_historical_performance with overall_winrate/recent_winrate/form_trend/games_played) described in RESEARCH.md are **only available inside this corrupted object**.

**Impact on Plan 02:** The adapter must either:
  (a) Use the standalone simpler lookup dicts (fewer features), matching the 27-feature production model, OR
  (b) Re-serialize the feature engineering object from a working environment (Colab) to fix the corruption

Option (a) is the pragmatic path since the production model expects 27 features anyway.

### Open Question 4: Champion Name Format

Champion names in all lookup dicts use **Oracle's Elixir display format**:
- Apostrophes preserved: `Kai'Sa`, `Kha'Zix`, `Cho'Gath`, `Rek'Sai`, `Vel'Koz`, `Kog'Maw`, `Bel'Veth`, `K'Sante`
- Spaces preserved: `Lee Sin`, `Twisted Fate`, `Miss Fortune`, `Xin Zhao`, `Jarvan IV`, `Dr. Mundo`, `Renata Glasc`, `Aurelion Sol`, `Tahm Kench`, `Master Yi`, `Nunu & Willump`, `Aurelion Sol`
- Title case: `LeBlanc` (not `Leblanc`)
- Uses display names: `Wukong` (NOT `MonkeyKing`), `Fiddlesticks` (NOT `FiddleSticks`)
- Total unique champions: **167**

### Encoder Structure

Production encoders (`models/production/encoders.joblib`):
- Type: `dict` of 5 `LabelEncoder` instances
- Keys: `league` (9 classes), `team` (258 classes), `side` (2: Blue/Red), `patch` (167 classes, string format like "10.01"), `split` (9 classes)
- Note: Encoder patches are **strings** ("10.01") while champion_meta_strength patches are **floats** (10.01) -- format mismatch to handle

### Scaler Comparison

| Scaler | Features | Matches Model |
|--------|----------|---------------|
| `production/scaler.joblib` | 27 (enhanced feature set) | YES - production VotingClassifier |
| `ultimate_scaler.joblib` | 48 (ultimate feature set) | NO - matches unloadable ultimate model |

**Production scaler 27 features:**
`team_avg_meta_strength`, `team_min_meta_strength`, `team_max_meta_strength`, `team_meta_variance`, `team_avg_synergy`, `team_min_synergy`, `team_max_synergy`, `team_synergy_variance`, `champion_diversity`, `team_composition_strength`, `team_historical_winrate`, `ban_count`, `ban_diversity`, `team_champions_banned`, `playoffs`, `side_blue`, `year`, `league_encoded`, `team_encoded`, `side_encoded`, `patch_encoded`, `split_encoded`, `meta_synergy_product`, `meta_synergy_ratio`, `historical_meta_product`, `composition_strength_gap`, `ban_pressure_ratio`

## Decisions Made

1. **Use production VotingClassifier (27 features) as the adapter's model** -- it is the only fully loadable model. The ultimate models (48 features) require catboost and the feature engineering object is corrupted. Rationale: ship with what works, upgrade later if needed.

2. **Pin scikit-learn==1.5.0** -- model will not load with newer versions due to internal Cython API changes.

3. **Champion names use display format** -- Wukong not MonkeyKing, apostrophes and spaces preserved. The alias map handles all known variants.

4. **Patch format is float in lookup dicts, string in encoders** -- the adapter must handle conversion between these formats when computing features.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed sklearn 1.5.0 instead of latest**
- **Found during:** Task 1 (artifact audit)
- **Issue:** Production model fails to load with sklearn 1.8.0 due to removed `__pyx_unpickle_CyHalfBinomialLoss`
- **Fix:** Installed sklearn==1.5.0 in audit environment
- **Files modified:** None (venv only)
- **Verification:** Model loads successfully
- **Committed in:** N/A (environment only)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Minimal -- only affected audit environment version. Production pinning was always planned.

## Issues Encountered

1. **Feature engineering object corrupted:** `ultimate_feature_engineering.joblib` has binary corruption (stray `\r` bytes, truncated data). Likely caused by OneDrive sync or git autocrlf on WSL cross-filesystem mount. This means the richer lookup dicts (champion_characteristics, etc.) are inaccessible from this file. The standalone lookup files work fine and contain sufficient data for the 27-feature production model.

2. **Ultimate models require catboost:** Both `ultimate_best_model.joblib` variants need `catboost` to unpickle. The production model (VotingClassifier) does not need catboost.

3. **RESEARCH.md assumptions partially incorrect:** Research described a 37-feature LogisticRegression model, but the actual production artifact is a 27-feature VotingClassifier. The 37-feature (or 48-feature "ultimate") pipeline represents a different training run. Plan 02 must use the 27-feature set, not the 37 described in research.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Plan 02 (Feature computation and prediction):**
- Schemas and normalization are complete
- All artifact paths and formats are documented
- The 27-feature production model + scaler + encoders are confirmed loadable
- Standalone lookup dicts (meta_strength, synergies, team_performance) are available

**Blockers/Concerns for Plan 02:**
- Feature computation must target the **27-feature enhanced set** (not the 37 or 48 from research)
- The feature computation logic must be reverse-engineered from the `enhanced_best_model` training code, not from `create_advanced_features_vectorized` (which produces the different ultimate feature set)
- Patch format mismatch (float in meta_strength dicts vs string in encoders) needs careful handling
- Memory profiling still needed (STATE.md blocker) -- should be measured after adapter is fully loaded

## Self-Check: PASSED

---
*Phase: 01-ml-adapter*
*Completed: 2026-02-24*
