# Architecture

**Analysis Date:** 2026-02-24

## Pattern Overview

**Overall:** Pipeline-based ML system with layered separation of concerns

**Key Characteristics:**
- Data flows unidirectionally: raw CSV download -> processing -> feature engineering -> model training -> inference
- No web framework; system is script/CLI-driven with an interactive terminal predictor
- `src/config.py` acts as a single source of truth for all file paths and constants
- Dual-path strategy: canonical paths under `src/data/`, `src/features/`, etc. coexist with legacy paths (`src/data_collection/`, `src/feature_engineering/`) for backward compatibility
- Models are serialized with `joblib` and promoted from `models/experiments/` to `models/production/`

## Layers

**Configuration Layer:**
- Purpose: Centralize all file path references and training/drift constants
- Location: `src/config.py`
- Contains: Path constants, `TARGET_LEAGUES` list, `TRAINING_DEFAULTS`, `DRIFT_THRESHOLDS`, `METRIC_WEIGHTS`
- Depends on: Python `pathlib` only
- Used by: Every other module that needs a file path or global constant

**Data Ingestion Layer:**
- Purpose: Download raw match CSVs from Oracle's Elixir S3 and validate their schema
- Location: `src/data/downloader.py`, `src/data/schema.py`
- Contains: `OraclesElixirDownloader`, `SchemaValidator`, `ValidationResult`
- Depends on: `src/config.py`, `requests`
- Used by: `src/data/pipeline.py`

**Data Processing Layer:**
- Purpose: Filter leagues, deduplicate, merge annual files, and build the canonical processed dataset
- Location: `src/data/pipeline.py`, `src/data/processor.py`, `src/data/filter.py`, `src/data/analyzer.py`, `src/data/extractor.py`, `src/data/quality.py`
- Contains: `DataPipeline`, `CompleteTargetDatasetCreator`, `TemporalWeighter`, `DataAugmenter`, `OutlierDetector`, `DataValidator`
- Depends on: Ingestion layer, `src/config.py`, `pandas`
- Used by: Feature engineering layer, training layer

**Feature Engineering Layer:**
- Purpose: Transform raw match rows into 33+ ML-ready features with zero data leakage (pre-match only)
- Location: `src/features/engineering.py` (canonical), `src/feature_engineering/advanced_feature_engineering.py` (legacy)
- Contains: `AdvancedFeatureEngineering` — computes champion meta strength, team win rates, pick/ban strategy scores, interaction features; applies target encoding
- Depends on: Processed dataset CSV, `sklearn`, `numpy`
- Used by: `src/models/trainer.py`, `src/prediction/predictor.py`

**Model Training Layer:**
- Purpose: Train, optimize, compare, and persist multiple classifier algorithms
- Location: `src/models/trainer.py` (main orchestrator), `src/models/optimizer.py` (Bayesian hyperparameter search), `src/models/robustness.py` (learning/validation curves), `src/models/explainability.py` (SHAP), `src/models/comprehensive_logistic_regression_comparison.py`
- Contains: `UltimateLoLPredictor`, `RobustnessAnalyzer`, `ModelExplainer`
- Depends on: Feature engineering layer, `sklearn`, `xgboost`, `lightgbm`, `catboost`, `optuna`, `shap`
- Used by: `scripts/run_training.py`, continuous training layer

**Continuous Training Layer:**
- Purpose: Automated retraining cycle with drift detection, model versioning, and production promotion
- Location: `src/training/trainer.py`, `src/training/drift.py`, `src/training/versioning.py`, `src/training/scheduler.py`, `src/training/config.py`
- Contains: `ContinuousTrainer`, `DriftDetector`, `ModelVersionManager`, `TrainingScheduler`
- Depends on: Model training layer, evaluation layer, `src/config.py`
- Used by: `scripts/refresh_data.py`, `scripts/run_training.py`

**Evaluation Layer:**
- Purpose: Multi-metric model assessment beyond accuracy (calibration, uncertainty, decision curves)
- Location: `src/evaluation/metrics.py`
- Contains: `MultiMetricEvaluator`, `ProbabilityCalibrator`, `UncertaintyQuantifier`; dataclasses `CalibrationResult`, `DecisionCurveResult`, `UncertaintyResult`
- Depends on: `sklearn`, `numpy`
- Used by: Model training layer, continuous training layer

**Prediction/Inference Layer:**
- Purpose: Load trained artifacts and run interactive pre-match predictions for real games
- Location: `src/prediction/predictor.py` (canonical), `src/prediction/interactive_match_predictor.py` (legacy), `src/prediction/confidence.py`
- Contains: `InteractiveLoLPredictor`, `ConfidenceEstimator`
- Depends on: `models/production/` artifacts, feature engineering layer, `src/models/explainability.py`, `src/features/edge_cases.py`
- Used by: End users running the CLI predictor

**Edge Case / Robustness Support:**
- Purpose: Detect out-of-distribution inputs and apply confidence penalties
- Location: `src/features/edge_cases.py`
- Contains: `EdgeCaseHandler` — detects unknown champions, new teams, new patches; returns confidence penalty multipliers
- Depends on: Training-time statistics
- Used by: `src/prediction/predictor.py`

## Data Flow

**Training Flow:**

1. `scripts/run_training.py` triggers `src/models/trainer.py`
2. `src/data/pipeline.py` downloads raw CSVs via `src/data/downloader.py`, validates with `src/data/schema.py`, filters leagues with `src/data/filter.py`, and writes `data/processed/complete_target_leagues_dataset.csv`
3. `src/features/engineering.py` (`AdvancedFeatureEngineering`) reads the CSV, computes champion meta, team history, and interaction features, applies target encoding
4. `UltimateLoLPredictor` in `src/models/trainer.py` performs temporal split (stratified or pure), trains multiple algorithms, runs Bayesian optimization via `src/models/optimizer.py`
5. `src/evaluation/metrics.py` evaluates each model; best model is serialized to `models/experiments/`
6. `src/training/versioning.py` (`ModelVersionManager`) compares to production baseline; if better, promotes to `models/production/` (best_model.joblib, scaler.joblib, encoders.joblib)

**Inference Flow:**

1. User runs `src/prediction/predictor.py`
2. `InteractiveLoLPredictor.__init__()` loads `models/production/best_model.joblib`, `scaler.joblib`, `encoders.joblib`
3. `setup_feature_engineering()` instantiates `AdvancedFeatureEngineering` and loads champion database
4. User enters team names and pick/ban sequences interactively in the terminal
5. `EdgeCaseHandler` checks for unknown champions/teams and applies confidence penalties
6. `ConfidenceEstimator` produces calibrated win probability with confidence interval
7. `ModelExplainer` (SHAP) returns top features driving the prediction

**Drift Detection / Continuous Update Flow:**

1. `DriftDetector` in `src/training/drift.py` monitors performance metrics (AUC, F1, log loss) and feature distributions against thresholds in `src/config.py` (`DRIFT_THRESHOLDS`)
2. When drift is detected, `TrainingScheduler` triggers `ContinuousTrainer`
3. `ContinuousTrainer` runs a rolling-window retrain and delegates promotion logic to `ModelVersionManager`

**State Management:**
- No in-memory state between runs; all state is persisted to disk via `joblib` serialization
- `src/config.py` provides `get_model_path()` which checks `models/production/` first, then `models/` root, then legacy paths — allowing safe fallback

## Key Abstractions

**AdvancedFeatureEngineering:**
- Purpose: Single class responsible for all feature computation; must be instantiated before both training and inference
- Examples: `src/features/engineering.py`, `src/feature_engineering/advanced_feature_engineering.py`
- Pattern: Stateful class — internal dicts (`champion_meta_strength`, `team_historical_performance`, etc.) are populated by `load_and_analyze_data()` then consumed by `create_advanced_features_vectorized()`; serialized via `__getstate__` to avoid `defaultdict` lambda pickling issues

**UltimateLoLPredictor:**
- Purpose: Orchestrates the full training experiment cycle for all algorithm families
- Examples: `src/models/trainer.py`
- Pattern: God-object trainer that wraps sklearn, XGBoost, LightGBM, CatBoost, and ensemble methods under one interface; exposes `split_data_temporally()` and `split_data_stratified_temporal()` for the novel temporal validation methodology

**DataPipeline:**
- Purpose: End-to-end data refresh orchestration
- Examples: `src/data/pipeline.py`
- Pattern: Returns typed `PipelineResult` dataclass with success flag, match counts, errors, and processing time — allows callers to handle partial failures

**ModelVersionManager:**
- Purpose: Lifecycle management for model artifacts
- Examples: `src/training/versioning.py`
- Pattern: Serializes `ModelMetadata` dataclass alongside each `.joblib`; supports `promote()`, `rollback()` by moving files between `models/experiments/` and `models/production/`

## Entry Points

**Training:**
- Location: `scripts/run_training.py`
- Triggers: Manual CLI execution
- Responsibilities: Accepts configuration flags (quick_mode, stratified_temporal, calibrate_probs), delegates to `src/models/trainer.py::main()`, saves and compares run results via `scripts/compare_results.py`

**Data Refresh:**
- Location: `scripts/refresh_data.py`, `scripts/process_new_data.py`
- Triggers: Manual CLI execution when new Oracle's Elixir data is available
- Responsibilities: Invoke `DataPipeline` to download and reprocess raw CSVs

**Interactive Prediction:**
- Location: `src/prediction/predictor.py` (also `src/prediction/interactive_match_predictor.py`)
- Triggers: Manual CLI execution
- Responsibilities: Load production models, run pick/ban session, return win probability with explanation

**Package Entry:**
- Location: `src/__init__.py`
- Triggers: `import src` or `pip install -e .`
- Responsibilities: Exposes package version `2.0.0`; `src/config.py` auto-creates all required directories on import via `ensure_dirs()`

## Error Handling

**Strategy:** Fail-fast with informative messages; no silent fallbacks on missing critical files

**Patterns:**
- `src/config.py` raises `FileNotFoundError` with all searched paths listed when dataset or model not found
- `AdvancedFeatureEngineering.__init__()` raises `ValueError` if the wrong dataset path is passed (guards against data contamination)
- Optional dependencies (`catboost`, `shap`, `matplotlib`) are wrapped in `try/except ImportError` with a warning; the system degrades gracefully by disabling those features
- `DataPipeline` returns a `PipelineResult` with `success=False` and populated `errors` list rather than raising on partial data failure

## Cross-Cutting Concerns

**Logging:** `print()` statements throughout; no structured logging framework. Output goes to stdout.

**Validation:** Schema validation in `src/data/schema.py` (`SchemaValidator`); input validation for champion names uses fuzzy matching inside `InteractiveLoLPredictor`

**Authentication:** None required. Oracle's Elixir S3 bucket is publicly accessible.

**Path Resolution:** All modules use `os.path.dirname(os.path.abspath(__file__))` to compute `project_root` and then append to `sys.path`. The canonical approach via `src/config.py` constants is used in newer modules; older modules replicate this pattern inline.

---

*Architecture analysis: 2026-02-24*
