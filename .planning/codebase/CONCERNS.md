# Codebase Concerns

**Analysis Date:** 2026-02-24

---

## Tech Debt

**Duplicate feature engineering modules (highest priority):**
- Issue: Two near-identical `AdvancedFeatureEngineering` classes coexist. The legacy module at `src/feature_engineering/advanced_feature_engineering.py` (1,857 lines) duplicates significant logic from the canonical module at `src/features/engineering.py` (2,150 lines). Most production model files import the legacy path directly.
- Files: `src/feature_engineering/advanced_feature_engineering.py`, `src/features/engineering.py`
- Impact: Bug fixes and new features applied to one file are silently missing from the other. Callers import inconsistent implementations depending on which `sys.path` is active.
- Fix approach: Remove `src/feature_engineering/advanced_feature_engineering.py`. Update all imports to use `src/features/engineering.py`. This requires changing `src/models/trainer.py`, `src/models/optimizer.py`, `src/models/enhanced_ultimate_predictor.py`, `src/models/comprehensive_logistic_regression_comparison.py`, `src/prediction/predictor.py`, and `src/prediction/interactive_match_predictor.py`.

**Duplicate model training modules:**
- Issue: Four largely overlapping model training classes exist: `UltimateLoLPredictor` in `src/models/ultimate_predictor.py` (744 lines), `UltimateLoLPredictor` in `src/models/trainer.py` (1,205 lines), `EnhancedUltimateLoLPredictor` in `src/models/enhanced_ultimate_predictor.py` (2,802 lines), and `src/models/optimizer.py` (2,802 lines — identical line count to `enhanced_ultimate_predictor.py`).
- Files: `src/models/ultimate_predictor.py`, `src/models/trainer.py`, `src/models/enhanced_ultimate_predictor.py`, `src/models/optimizer.py`
- Impact: It is unclear which class is authoritative for production training. Model saving paths differ across files (e.g., `trainer.py` saves to `models/ultimate_best_model.joblib` while `enhanced_ultimate_predictor.py` saves to `models/bayesian_optimized_models/`). Maintaining four parallel codebases multiplies defect surface.
- Fix approach: Designate `src/models/trainer.py` as the single training entry point. Archive or delete the others. Document the decision in CLAUDE.md.

**Inconsistent path resolution — no centralized config usage:**
- Issue: `src/config.py` provides a clean, centralized `DATASET_PATH`, `BEST_MODEL_PATH`, etc., but nearly every module constructs paths manually using `os.path.join(os.path.dirname(os.path.abspath(__file__)), ...)` chains. `src/models/trainer.py` line 78 constructs `data/complete_target_leagues_dataset.csv` (missing the `processed/` subdirectory that `src/config.py` specifies), causing a primary path miss and reliance on fallback search.
- Files: `src/models/trainer.py`, `src/prediction/predictor.py`, `src/prediction/interactive_match_predictor.py`, `src/feature_engineering/advanced_feature_engineering.py`
- Impact: Path searches silently fall back to legacy locations, making it hard to reason about which file is actually used at runtime.
- Fix approach: Replace all manual path construction with `from src.config import DATASET_PATH, BEST_MODEL_PATH, SCALER_PATH` and remove fallback search loops.

**`sys.path.append` used for all cross-module imports:**
- Issue: Every module in `src/models/`, `src/prediction/`, and `src/training/` calls `sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))` before importing sibling packages. This is a project setup problem; `setup.py` exists but is not used consistently.
- Files: All files under `src/models/`, `src/prediction/`, `src/training/`
- Impact: Import behaviour is fragile and order-dependent. Running scripts from different working directories produces `ModuleNotFoundError`.
- Fix approach: Install the package in editable mode (`pip install -e .`) and remove all `sys.path` manipulations. Ensure `setup.py` declares all packages correctly.

**Patch timeline stub — incomplete data:**
- Issue: `_create_patch_timeline()` in `src/features/engineering.py` (line 79) returns only 6 hardcoded patch-to-date entries covering patches 13.22–14.3. The comment states "Add more patch mappings as needed" but this was never completed. The dataset spans 2014–2024 with hundreds of patches.
- Files: `src/features/engineering.py` lines 79–86, `src/feature_engineering/advanced_feature_engineering.py` lines 79–86
- Impact: All patch-date lookups outside those 6 entries will silently return no data, degrading meta timing features.
- Fix approach: Either populate the full patch timeline from Oracle's Elixir patch notes, infer dates from the dataset itself (`df.groupby('patch')['date'].min()`), or remove the unused dictionary.

**`create_advanced_features()` uses row-iteration over 37,000 matches:**
- Issue: `create_advanced_features()` in `src/features/engineering.py` (line 595) iterates every row with `for idx, match in self.df.iterrows()`. The same class provides `create_advanced_features_vectorized()` which avoids this, but the legacy codepath is still called by `src/models/trainer.py` (line 138) when `use_enhanced_v2=False`.
- Files: `src/features/engineering.py` lines 588–800, `src/feature_engineering/advanced_feature_engineering.py` lines 584–900
- Impact: Training runtime is 10–50x longer than necessary for the non-vectorized path. The legacy module (`advanced_feature_engineering.py`) still has `iterrows()` loops at lines 195, 255, 318, 343, 428, 453, 546, 584, 1431, 1649, 1720.
- Fix approach: Make `create_advanced_features_vectorized()` the sole path. Remove the `iterrows()`-based `create_advanced_features()` or mark it deprecated.

**Bare `except:` clauses silently swallow errors:**
- Issue: Eight bare `except:` clauses exist that catch every exception type including `KeyboardInterrupt` and `SystemExit`, then continue silently.
- Files:
  - `src/features/engineering.py` line 1529
  - `src/feature_engineering/advanced_feature_engineering.py` line 1520
  - `src/data_processing/create_complete_target_dataset.py` line 79
  - `src/data/processor.py` line 79
  - `src/prediction/predictor.py` lines 1148, 1180
  - `src/prediction/interactive_match_predictor.py` lines 1047, 1079
- Impact: Corrupt feature calculations pass silently; AUC values used in model selection may be computed on incomplete feature sets without any warning.
- Fix approach: Replace `except:` with `except Exception as e:` and log the error at minimum. For the leakage detection path (`pass  # Skip features that cause errors`), log the exception at DEBUG level.

**`warnings.filterwarnings('ignore')` applied globally in 20 files:**
- Issue: Every module in `src/` suppresses all Python warnings at import time.
- Files: All 20 files identified — representative examples are `src/models/trainer.py`, `src/features/engineering.py`, `src/prediction/predictor.py`.
- Impact: Deprecation warnings from scikit-learn, pandas, and numpy are invisible. Future library upgrades may silently break computations.
- Fix approach: Remove global suppression. Apply targeted context managers (`with warnings.catch_warnings():`) only around known noisy third-party calls.

**Commented-out code left in production paths:**
- Issue: `src/features/engineering.py` line 98 contains `# self._clean_target_leagues_data()  # DISABLED - using pre-cleaned dataset`. Several other analysis steps are disabled inline without explanation of when or whether they should be re-enabled.
- Files: `src/features/engineering.py` lines 97–99
- Impact: Future maintainers cannot determine if disabled code is safe to delete or if it guards a correctness property.
- Fix approach: Delete dead code or add a documented conditional configuration flag.

---

## Known Bugs

**`UltimateLoLPredictor.__init__` in `src/models/trainer.py` looks for dataset at wrong primary path:**
- Symptoms: On a fresh clone the dataset is at `data/processed/complete_target_leagues_dataset.csv`. The constructor at line 78 builds the path as `data/complete_target_leagues_dataset.csv` (missing `processed/`), triggers the "not found" branch, then searches fallbacks and finds the correct path at index 2 of `alternative_paths`.
- Files: `src/models/trainer.py` lines 75–102
- Trigger: Any run of `src/models/trainer.py` on a clean project layout.
- Workaround: The fallback search succeeds so training completes, but a log warning is always emitted.

**`InteractiveLoLPredictor` hard-codes a specific Bayesian model path as default:**
- Symptoms: `src/prediction/predictor.py` line 41 sets the default `model_path` to `models/bayesian_optimized_models/bayesian_best_model_Logistic_Regression.joblib`. This directory does not exist in the repository; `models/production/best_model.joblib` is the actual production model. The class falls through a 8-path search list before finding the correct file.
- Files: `src/prediction/predictor.py` lines 36–42, 91–103
- Trigger: Running `python src/prediction/predictor.py` or importing `InteractiveLoLPredictor` with default arguments.
- Workaround: Correct model is eventually loaded via fallback, but startup is slower and confusing.

**Model-scaler mismatch risk in multi-path loading:**
- Symptoms: `load_model_components()` in `src/prediction/predictor.py` loads the first model file found and then independently loads the first scaler file found. There is no verification that the model and scaler were trained together, so a Bayesian-optimized model could be paired with a standard scaler from a different run.
- Files: `src/prediction/predictor.py` lines 105–168
- Trigger: Multiple `.joblib` scaler files exist in `models/` from different training runs.
- Workaround: None currently. Deployment packages that bundle model + scaler together (checked at line 123) avoid this, but simple model files do not.

---

## Security Considerations

**S3 URL is a hardcoded HTTP endpoint — no certificate pinning or integrity check:**
- Risk: The Oracle's Elixir S3 base URL is hardcoded in two places (`src/config.py` line 132 and `src/data/downloader.py` line 42). Downloaded CSV files are not verified against a checksum before being loaded into pandas.
- Files: `src/config.py` line 132, `src/data/downloader.py` lines 42–46
- Current mitigation: HTTPS is used. The downloader stores an `etag` per file but it is not validated against a server-provided expected hash.
- Recommendations: Validate file integrity using the `etag` or a SHA-256 hash after download. Raise on hash mismatch rather than silently loading potentially corrupted data.

**`joblib.load()` called on untrusted file paths without validation:**
- Risk: `joblib.load()` deserializes arbitrary Python objects. If a malicious `.joblib` file were placed in the `models/` directory (e.g., via a compromised S3 bucket or shared drive), it would execute code on load.
- Files: `src/prediction/predictor.py` lines 121, 147, 162; `src/prediction/interactive_match_predictor.py` equivalent lines
- Current mitigation: None. Models are loaded from local paths only; no remote model fetching exists currently.
- Recommendations: For a future web deployment, add path allowlist validation before loading any `.joblib` file. Consider model signing.

---

## Performance Bottlenecks

**`_analyze_champion_characteristics()` uses `iterrows()` over full dataset:**
- Problem: `src/feature_engineering/advanced_feature_engineering.py` lines 195–250 iterate all 37,000+ rows to build champion win rate statistics, a computation that pandas `groupby` + `agg` could perform in milliseconds.
- Files: `src/feature_engineering/advanced_feature_engineering.py` lines 183–250
- Cause: Early implementation choice preserved across refactors.
- Improvement path: Replace the loop with `df.groupby(['top_champion', ...])[['result', 'game_length']].agg(...)`. Estimated 50–200x speedup.

**Feature engineering loads entire dataset into memory twice:**
- Problem: `load_and_analyze_data()` reads the full CSV into `self.df`. `create_advanced_features_vectorized()` then creates a second full DataFrame (`features_data`) before converting to a DataFrame. With 37,502 rows and 33+ features, peak RAM usage approaches 2–4 GB.
- Files: `src/features/engineering.py` lines 88–130, 980–1300
- Cause: No streaming or chunking is implemented.
- Improvement path: Evaluate memory with `df.memory_usage(deep=True)`. If this becomes a constraint at scale, switch from CSV to Parquet and use chunked feature creation.

**`apply_advanced_encoding()` re-fits target encoders on the entire dataset:**
- Problem: `src/features/engineering.py`'s `apply_advanced_encoding()` fits `sklearn.preprocessing.TargetEncoder` on the full dataset (including test rows), which constitutes target leakage. The method `_investigate_target_leakage()` flags this risk but does not fix it in the encoding step.
- Files: `src/features/engineering.py` lines covering `apply_advanced_encoding`, `src/feature_engineering/advanced_feature_engineering.py` equivalent
- Cause: Encoding is done before train/test split in the pipeline.
- Improvement path: Move target encoding inside the training fold. Pass a `training_mask` into the encoder fitting step, or use `sklearn.pipeline.Pipeline` to ensure the encoder only sees training data.

---

## Fragile Areas

**Team roster database is a static hardcoded dict, not updated for 2025:**
- Files: `src/prediction/predictor.py` lines 55–60, `src/prediction/interactive_match_predictor.py` equivalent
- Why fragile: Teams in LCK, LEC, LCS, and LPL change rosters and organizations between splits. The dict shows the 2024 roster snapshot. New teams, relegated teams, or renamed organizations will fail validation ("Team not found") at prediction time.
- Safe modification: Add a `update_teams_db(year: int)` method that pulls current roster data, or replace with a lookup against the dataset's `teamname` column.
- Test coverage: No test checks that `teams_db` matches the dataset's actual team names.

**Champion pool is a static list fixed at "patch 14.1 example":**
- Files: `src/prediction/predictor.py` lines 63–74 (comment: "Current meta champions (patch 14.1 example)")
- Why fragile: The pool contains ~50 champions. League of Legends has 170+ champions. The `load_champion_database()` method overrides this with dataset-derived champions at runtime, but if that method fails, the fallback is the incomplete static list, silently restricting valid input.
- Safe modification: Remove the static fallback pool. Raise an explicit error if the champion database cannot be loaded from the dataset.
- Test coverage: No test validates champion database completeness.

**`_create_patch_timeline()` returns only 6 entries covering 2 months of patches:**
- Files: `src/features/engineering.py` lines 79–86
- Why fragile: Any patch outside the 6 hardcoded entries (`13.22`–`14.3`) returns no data. The dictionary is consumed by meta timing features, so missing patches produce default values silently.
- Safe modification: Generate the timeline from the dataset at load time using `df.groupby('patch')['date'].min()`.
- Test coverage: None.

**Leakage detection in `_investigate_target_leakage()` runs after encoding, not before:**
- Files: `src/features/engineering.py` lines 1412–1553
- Why fragile: The method detects suspicious features (AUC > 0.85) only after `apply_advanced_encoding()` has already run, which means target-leaked encodings are already embedded in the feature matrix. Detection output is printed but no automatic remediation occurs.
- Safe modification: Move `_investigate_target_leakage()` before encoding. Add a configurable threshold that raises or logs a structured warning rather than only printing.
- Test coverage: None.

---

## Scaling Limits

**No batch prediction API — predictor is interactive-only:**
- Current capacity: Single match prediction via stdin prompt loop.
- Limit: The `InteractiveLoLPredictor` in `src/prediction/predictor.py` is designed around `input()` calls. There is no programmatic API for batch predictions or integration into a web service.
- Scaling path: Extract prediction logic from the CLI loop into a `predict(match_features: dict) -> float` method, making the class callable without stdin interaction. The confidence and edge-case infrastructure in `src/prediction/confidence.py` and `src/features/edge_cases.py` is already designed as callable modules.

**Data pipeline has no incremental processing:**
- Current capacity: Full CSV reload on every training run.
- Limit: Each run re-reads and re-processes all 37,502 rows regardless of how many new matches exist.
- Scaling path: The `DataPipeline` class in `src/data/pipeline.py` has version tracking infrastructure. Extend it to checkpoint the processed feature matrix and only re-compute features for new rows.

---

## Dependencies at Risk

**`catboost` is an optional dependency with no version pin:**
- Risk: `src/models/trainer.py` lines 11–15 try-import CatBoost. `requirements.txt` does not pin a version. CatBoost has had breaking API changes across major versions.
- Impact: Silent degradation — the model suite trains without CatBoost if not installed, potentially excluding a competitive algorithm without the user knowing.
- Migration plan: Pin `catboost>=1.2` in `requirements.txt`. Log a WARNING (not just a print) when CatBoost is unavailable.

**`optuna` is optional but controls the best-performing training path:**
- Risk: Bayesian hyperparameter optimization (the path that produces the documented 82.97% AUC) requires Optuna. If not installed, the system silently falls back to grid search without informing the user.
- Files: `src/models/enhanced_ultimate_predictor.py` lines 18–25
- Impact: Users running the system without Optuna get substantially worse results with no warning.
- Migration plan: Add Optuna to `requirements.txt` as a required dependency since it is critical to the thesis results.

**`statsmodels` used for McNemar test without version pin:**
- Risk: `src/models/enhanced_ultimate_predictor.py` line 33 imports `statsmodels.stats.contingency_tables.mcnemar`. This is used in statistical significance testing. The `statsmodels` API for McNemar has changed in minor versions.
- Impact: Statistical validation tests may fail silently on newer statsmodels versions.
- Migration plan: Pin `statsmodels>=0.14` and add a test that runs the McNemar call.

---

## Test Coverage Gaps

**No unit tests for feature engineering correctness:**
- What is not tested: That `create_advanced_features_vectorized()` and `create_advanced_features()` produce the same values. That temporal momentum features use `.shift(1)` correctly to avoid look-ahead bias. That target encoding is not fitted on test data.
- Files: `src/features/engineering.py` (all feature creation methods), `src/feature_engineering/advanced_feature_engineering.py`
- Risk: Feature leakage bugs or off-by-one errors in rolling windows could inflate reported AUC without being detected.
- Priority: High

**No integration test for the full training pipeline:**
- What is not tested: End-to-end: data load -> feature engineering -> train/test split -> model training -> evaluation -> model save.
- Files: `tests/quick_test.py` tests only imports and class loading. `tests/test_model_features.py` tests initialization only.
- Risk: Breaking changes to any step in the pipeline are discovered only at full training time (30–60 minutes).
- Priority: High

**No tests for model-scaler consistency:**
- What is not tested: That the saved scaler was fitted on the same feature set and training data as the saved model.
- Files: `src/prediction/predictor.py` (model loading), `src/models/trainer.py` (model saving)
- Risk: Silent prediction errors when the wrong scaler is loaded.
- Priority: Medium

**Test files use emojis and informal output format:**
- What is not tested: N/A — this is a maintainability concern.
- Files: `tests/quick_test.py`, `tests/test_model_features.py`
- Risk: Tests are written as scripts rather than using a test framework (pytest/unittest). They cannot be discovered automatically, run in CI, or produce machine-readable pass/fail results.
- Priority: Medium
- Fix approach: Convert to `pytest` functions with `assert` statements. Remove emoji output.

**`test_directory_structure()` checks for directories that no longer exist:**
- What is not tested: The test at `tests/quick_test.py` lines 150–175 checks for `visualizations/` and `experiments/` at the project root. These have been moved to `outputs/visualizations/` and `models/experiments/`.
- Files: `tests/quick_test.py` lines 150–175
- Risk: Test always reports failure for directory structure even when the actual structure is correct.
- Priority: Low

---

*Concerns audit: 2026-02-24*
