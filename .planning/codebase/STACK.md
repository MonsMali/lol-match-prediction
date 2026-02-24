# Technology Stack

**Analysis Date:** 2026-02-24

## Languages

**Primary:**
- Python 3.8+ - All source code, ML pipeline, data processing, and prediction
  - Development environment runs Python 3.12.3
  - Minimum version enforced in `setup.py`: `python_requires=">=3.8"`

**Secondary:**
- None detected. No JavaScript, TypeScript, SQL schemas, or shell scripts in the core src tree.

## Runtime

**Environment:**
- CPython 3.12.3 (local development on Linux/WSL2)
- Google Colab (GPU-accelerated training via `notebooks/colab_training.ipynb`)
  - Confirmed Tesla T4 GPU used for Colab training runs

**Package Manager:**
- pip (standard)
- `requirements.txt` present and committed
- `setup.py` (setuptools) for installable package
- No lockfile (no `pip freeze` output or `poetry.lock` detected)

## Frameworks

**Core:**
- scikit-learn >=1.0.0 - Classification models, preprocessing, evaluation metrics, calibration
- XGBoost >=1.5.0 - Gradient boosted tree models
- LightGBM >=3.3.0 - Gradient boosted tree models (fast)
- CatBoost >=1.0.0 - Gradient boosted tree models with categorical support (optional import, gracefully degraded)

**Optimization:**
- Optuna >=3.0.0 - Bayesian hyperparameter optimization (optional import, gracefully degraded)

**Explainability:**
- SHAP >=0.41.0 - SHapley Additive exPlanations for feature importance and local predictions
  - Used in `src/models/explainability.py`
  - Supports TreeExplainer, LinearExplainer, KernelExplainer

**Neural Networks (optional):**
- PyTorch - Optional GPU-accelerated training (optional import, gracefully degraded)
  - Used only inside `try/except` blocks in `src/models/optimizer.py`, `src/models/enhanced_ultimate_predictor.py`
  - MLPClassifier from scikit-learn used as primary neural net alternative

**Testing (dev only):**
- pytest >=7.0.0 - Test runner
- pytest-cov >=3.0.0 - Coverage reporting

**Code Quality (dev only):**
- black >=22.0.0 - Code formatting
- flake8 >=4.0.0 - Linting
- mypy >=0.950 - Type checking

## Key Dependencies

**Critical:**
- `pandas >=1.3.0` - Primary data structure for all match data; CSV I/O; DataFrame operations throughout the entire pipeline
- `numpy >=1.21.0` - Numerical operations underpinning all feature engineering and model computations
- `joblib >=1.1.0` - Model persistence (`.joblib` files) and parallel computation; all trained artifacts saved/loaded via joblib
- `scikit-learn >=1.0.0` - StandardScaler, TargetEncoder, LabelEncoder, cross-validation, all evaluation metrics, calibration

**Infrastructure:**
- `requests >=2.28.0` - HTTP client for downloading data from Oracle's Elixir S3 bucket (`src/data/downloader.py`)
- `tqdm >=4.62.0` - Progress bars for long-running data and training loops
- `matplotlib >=3.4.0` - Visualization output to `outputs/visualizations/`
- `seaborn >=0.11.0` - Statistical plot styling on top of matplotlib

## Configuration

**Environment:**
- No `.env` file detected; no environment variables required for core functionality
- Paths are managed entirely in `src/config.py` using Python `pathlib.Path` relative to project root
- `ORACLES_ELIXIR_S3_BASE` URL is hardcoded in `src/config.py` (no auth required for S3 public bucket)
- No secrets or API keys needed for data download (Oracle's Elixir bucket is public)

**Build:**
- `setup.py` - Package installation config; entry point `lol-predict` maps to `src.prediction.predictor:main`
- `requirements.txt` - Runtime dependencies
- No `pyproject.toml`, `Makefile`, `Dockerfile`, or CI config detected

**Key config constants in `src/config.py`:**
- `DATASET_PATH` - `data/processed/complete_target_leagues_dataset.csv`
- `BEST_MODEL_PATH` - `models/production/best_model.joblib`
- `SCALER_PATH` - `models/production/scaler.joblib`
- `ENCODERS_PATH` - `models/production/encoders.joblib`
- `TRAINING_DEFAULTS` - Rolling window, validation size, test size, split strategy
- `DRIFT_THRESHOLDS` - AUC/F1/accuracy drop thresholds triggering retraining
- `METRIC_WEIGHTS` - Weighted composite scoring for multi-metric evaluation
- `TARGET_LEAGUES` - List of 10 league identifiers used to filter data

## Platform Requirements

**Development:**
- Python 3.8 or higher
- pip for dependency installation
- Sufficient disk space for raw CSV data (Oracle's Elixir files per year ~50-200 MB each)
- Internet access for S3 data downloads

**Production / Training:**
- Local: CPU-only supported
- Google Colab: GPU (Tesla T4 or similar) used for faster training, mounted via Google Drive
- No containerization or cloud deployment infrastructure detected

---

*Stack analysis: 2026-02-24*
