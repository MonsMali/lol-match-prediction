# Codebase Structure

**Analysis Date:** 2026-02-24

## Directory Layout

```
tese/                                        # Project root
├── src/                                     # All Python source code
│   ├── __init__.py                          # Package root (version 2.0.0)
│   ├── config.py                            # Centralized path & constant config
│   ├── data/                                # Data ingestion and processing (canonical)
│   │   ├── downloader.py                    # Oracle's Elixir S3 downloader
│   │   ├── pipeline.py                      # End-to-end data pipeline orchestrator
│   │   ├── processor.py                     # Raw-to-processed dataset creator
│   │   ├── filter.py                        # League filtering logic
│   │   ├── analyzer.py                      # Oracle Elixir column analysis
│   │   ├── extractor.py                     # Feature extraction utilities
│   │   ├── schema.py                        # Schema validation
│   │   └── quality.py                       # Temporal weighting, augmentation, validation
│   ├── features/                            # Feature engineering (canonical)
│   │   ├── engineering.py                   # AdvancedFeatureEngineering (33+ features)
│   │   └── edge_cases.py                    # EdgeCaseHandler (unknown champs/teams)
│   ├── models/                              # Model training and analysis
│   │   ├── trainer.py                       # UltimateLoLPredictor (main trainer)
│   │   ├── optimizer.py                     # Bayesian hyperparameter optimization
│   │   ├── robustness.py                    # Learning/validation curve analysis
│   │   ├── explainability.py                # SHAP-based model explanations
│   │   ├── enhanced_ultimate_predictor.py   # Enhanced variant of trainer
│   │   ├── ultimate_predictor.py            # Legacy trainer entry point
│   │   └── comprehensive_logistic_regression_comparison.py
│   ├── prediction/                          # Inference interface
│   │   ├── predictor.py                     # InteractiveLoLPredictor (canonical)
│   │   ├── interactive_match_predictor.py   # Legacy predictor entry point
│   │   └── confidence.py                    # ConfidenceEstimator (calibrated probs)
│   ├── training/                            # Continuous training system
│   │   ├── trainer.py                       # ContinuousTrainer (auto retrain + promote)
│   │   ├── drift.py                         # DriftDetector (ADWIN, sliding window)
│   │   ├── versioning.py                    # ModelVersionManager (lifecycle)
│   │   ├── scheduler.py                     # TrainingScheduler / TrainingTrigger
│   │   └── config.py                        # TrainingConfig, DriftConfig dataclasses
│   ├── evaluation/                          # Multi-metric evaluation
│   │   └── metrics.py                       # MultiMetricEvaluator, ProbabilityCalibrator
│   ├── data_collection/                     # Legacy: data extraction scripts
│   │   ├── focused_feature_extractor.py
│   │   ├── oracle_elixir_analyzer.py
│   │   ├── filter_target_leagues.py
│   │   ├── analyze_focused_data.py
│   │   └── analyze_original_columns.py
│   ├── data_processing/                     # Legacy: dataset creation
│   │   └── create_complete_target_dataset.py
│   └── feature_engineering/                 # Legacy: feature engineering
│       └── advanced_feature_engineering.py
├── scripts/                                 # CLI runner scripts
│   ├── run_training.py                      # Training entry point
│   ├── refresh_data.py                      # Data refresh entry point
│   ├── process_new_data.py                  # Process new Oracle's Elixir data
│   └── compare_results.py                   # Compare training run metrics
├── data/                                    # Data storage (not committed to git)
│   ├── raw/                                 # Downloaded Oracle's Elixir CSV files
│   └── processed/                           # Cleaned/filtered dataset
│       ├── complete_target_leagues_dataset.csv   # PRIMARY dataset (37,502 matches)
│       └── backups/                         # Timestamped dataset backups
├── models/                                  # Serialized model artifacts
│   ├── production/                          # Active production models
│   │   ├── best_model.joblib                # Logistic Regression (82.97% AUC)
│   │   ├── scaler.joblib                    # StandardScaler for features
│   │   └── encoders.joblib                  # Categorical encoders
│   ├── experiments/                         # Candidate models awaiting promotion
│   ├── champion_meta_strength.joblib        # Legacy artifact
│   ├── champion_synergies.joblib            # Legacy artifact
│   ├── team_historical_performance.joblib   # Legacy artifact
│   └── *.joblib                             # Other legacy model versions
├── tests/                                   # Test suite
│   ├── __init__.py
│   ├── quick_test.py                        # Fast smoke tests
│   └── test_model_features.py              # Feature engineering tests
├── docs/                                    # Documentation
│   ├── API_DOCUMENTATION.md
│   ├── INSTALLATION_GUIDE.md
│   ├── MODEL_PERFORMANCE_REPORT.md
│   ├── THESIS_OVERVIEW.md
│   ├── FOCUSED_FEATURES_METHODOLOGY.md
│   └── SPLIT_STRATEGY_GUIDE.md
├── outputs/                                 # Generated output artifacts
│   ├── visualizations/                      # Matplotlib plots
│   └── results/                             # JSON/CSV results
├── results/                                 # Historical training run results
│   └── previous_run_20250530_011743/
├── notebooks/                               # Jupyter notebooks (exploratory)
├── visualizations/                          # Root-level visualization dumps
├── catboost_info/                           # CatBoost training metadata (auto-generated)
├── Thesis/                                  # Thesis source files
├── requirements.txt                         # Python dependencies
├── setup.py                                 # Editable package install
├── CLAUDE.md                                # AI context file
├── README.md
├── TODO.md
└── thesis.pdf                               # Final thesis document
```

## Directory Purposes

**`src/`:**
- Purpose: All application source code, organized by functional layer
- Contains: Python packages (each with `__init__.py`)
- Key files: `src/config.py` (imported by everything)

**`src/data/`:**
- Purpose: Complete data lifecycle from download to processed CSV
- Contains: Downloader, pipeline orchestrator, schema validator, quality utilities
- Key files: `src/data/pipeline.py`, `src/data/downloader.py`, `src/data/quality.py`

**`src/features/`:**
- Purpose: Feature engineering for ML — transforms raw match data into numeric features
- Contains: `AdvancedFeatureEngineering`, `EdgeCaseHandler`
- Key files: `src/features/engineering.py`

**`src/models/`:**
- Purpose: Multi-algorithm training, optimization, robustness analysis, and explainability
- Contains: Trainers, Bayesian optimizer, SHAP explainer, learning curve analyzer
- Key files: `src/models/trainer.py`, `src/models/optimizer.py`, `src/models/explainability.py`

**`src/prediction/`:**
- Purpose: Production inference interface — interactive CLI predictor
- Contains: `InteractiveLoLPredictor`, `ConfidenceEstimator`
- Key files: `src/prediction/predictor.py`, `src/prediction/confidence.py`

**`src/training/`:**
- Purpose: Continuous learning system — automated drift detection, versioned retraining, and model promotion
- Contains: `ContinuousTrainer`, `DriftDetector`, `ModelVersionManager`, `TrainingScheduler`
- Key files: `src/training/trainer.py`, `src/training/drift.py`, `src/training/versioning.py`

**`src/evaluation/`:**
- Purpose: Comprehensive model evaluation beyond accuracy
- Contains: AUC, log loss, Brier score, ECE, MCC, Cohen's Kappa, calibration, decision curves
- Key files: `src/evaluation/metrics.py`

**`src/data_collection/`, `src/data_processing/`, `src/feature_engineering/`:**
- Purpose: Legacy modules kept for backward compatibility and historical reference
- Imports still work; canonical equivalents live in `src/data/`, `src/features/`
- Do not add new code here; add to canonical paths instead

**`scripts/`:**
- Purpose: CLI entry points for training and data refresh workflows
- Contains: Runner scripts that import from `src/`
- Key files: `scripts/run_training.py`, `scripts/refresh_data.py`

**`data/processed/`:**
- Purpose: The single canonical processed dataset consumed by all training and evaluation code
- Generated: Yes (by `src/data/pipeline.py` or `src/data/processor.py`)
- Committed: No — CSV is too large for git; `.gitignore` excludes it

**`models/production/`:**
- Purpose: Active production artifacts loaded by `InteractiveLoLPredictor`
- Generated: Yes (promoted by `ModelVersionManager`)
- Committed: No — binary files excluded from git

**`models/experiments/`:**
- Purpose: Candidate model artifacts awaiting performance comparison and promotion
- Generated: Yes (output of each training run)
- Committed: No

**`outputs/`:**
- Purpose: Generated visualizations and result JSON/CSV from training runs
- Generated: Yes
- Committed: No

## Key File Locations

**Entry Points:**
- `scripts/run_training.py`: Start a training experiment
- `scripts/refresh_data.py`: Download and reprocess Oracle's Elixir data
- `src/prediction/predictor.py`: Launch the interactive match predictor

**Configuration:**
- `src/config.py`: All path constants, target leagues, training defaults, drift thresholds

**Core Logic:**
- `src/features/engineering.py`: `AdvancedFeatureEngineering` — primary feature computation class
- `src/models/trainer.py`: `UltimateLoLPredictor` — primary training orchestrator
- `src/data/pipeline.py`: `DataPipeline` — data refresh orchestrator

**Primary Dataset:**
- `data/processed/complete_target_leagues_dataset.csv`: 37,502 professional matches (2014-2024)

**Production Models:**
- `models/production/best_model.joblib`: Logistic Regression, 82.97% AUC-ROC
- `models/production/scaler.joblib`: StandardScaler
- `models/production/encoders.joblib`: Categorical encoders

**Testing:**
- `tests/quick_test.py`: Smoke tests
- `tests/test_model_features.py`: Feature engineering tests

## Naming Conventions

**Files:**
- Snake_case for all Python files: `advanced_feature_engineering.py`, `run_training.py`
- Descriptive names indicating the primary class or function: `downloader.py` contains `OraclesElixirDownloader`
- Legacy files prefixed with historical context: `interactive_match_predictor.py`, `ultimate_predictor.py`

**Classes:**
- PascalCase: `AdvancedFeatureEngineering`, `UltimateLoLPredictor`, `InteractiveLoLPredictor`, `DataPipeline`, `ContinuousTrainer`, `DriftDetector`, `ModelVersionManager`
- Dataclasses used extensively for typed return values: `PipelineResult`, `TrainingResult`, `ModelMetadata`, `DriftResult`, `CalibrationResult`

**Directories:**
- Lowercase snake_case: `data_collection/`, `feature_engineering/`, `data_processing/`
- Single-word canonical names for new modules: `data/`, `features/`, `models/`, `prediction/`, `training/`, `evaluation/`

**Artifacts:**
- `joblib` files named after their content: `best_model.joblib`, `scaler.joblib`, `champion_meta_strength.joblib`
- Legacy artifacts prefixed: `enhanced_best_model.joblib`, `ultimate_best_model.joblib`

## Where to Add New Code

**New Feature Engineering Logic:**
- Primary code: `src/features/engineering.py` inside `AdvancedFeatureEngineering`
- If it handles edge cases or unseen inputs: `src/features/edge_cases.py`

**New Model Algorithm:**
- Implementation: `src/models/trainer.py` inside `UltimateLoLPredictor`
- If it needs a dedicated optimizer: `src/models/optimizer.py`

**New Data Source or Data Field:**
- Downloader changes: `src/data/downloader.py`
- Schema changes: `src/data/schema.py`
- Processing changes: `src/data/processor.py` or `src/data/pipeline.py`

**New Evaluation Metric:**
- Implementation: `src/evaluation/metrics.py` inside `MultiMetricEvaluator`

**New CLI Script:**
- Location: `scripts/`
- Import from `src/` using `sys.path.insert(0, str(project_root))`

**New Tests:**
- Location: `tests/`
- Naming: `test_<module>.py`

**New Visualization Output:**
- Write to: `outputs/visualizations/` using path from `src/config.py` (`VISUALIZATIONS_DIR`)

**New Prediction Interface Feature:**
- Location: `src/prediction/predictor.py` inside `InteractiveLoLPredictor`
- Confidence/calibration logic: `src/prediction/confidence.py`

## Special Directories

**`catboost_info/`:**
- Purpose: CatBoost auto-generates this during training with learn logs and tmp files
- Generated: Yes (by CatBoost library automatically)
- Committed: No — should be in `.gitignore`

**`models/experiments/`:**
- Purpose: Staging area for new model versions before production promotion
- Generated: Yes
- Committed: No

**`data/processed/backups/`:**
- Purpose: Timestamped snapshots of the processed dataset for rollback
- Generated: Yes (by `DataPipeline` before overwriting)
- Committed: No

**`.planning/codebase/`:**
- Purpose: GSD codebase analysis documents consumed by AI planning tools
- Generated: Yes (by map-codebase command)
- Committed: Yes

---

*Structure analysis: 2026-02-24*
