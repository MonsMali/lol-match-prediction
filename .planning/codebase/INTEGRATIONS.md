# External Integrations

**Analysis Date:** 2026-02-24

## APIs & External Services

**Data Source:**
- Oracle's Elixir S3 Bucket - Primary source of all professional LoL match data (2014-present)
  - SDK/Client: Python `requests` library (no SDK; direct HTTP)
  - Auth: None - the S3 bucket is publicly accessible, no credentials required
  - Base URL: `https://oracleselixir-downloadable-match-data.s3-us-west-2.amazonaws.com`
  - Bucket listing endpoint: `{base}?list-type=2` (S3 ListObjectsV2, XML response)
  - Implementation: `src/data/downloader.py` - `OraclesElixirDownloader` class
  - File format: CSV files per year, pattern `{year}.*match.*data.*\.csv`
  - Download strategy: ETag-based deduplication via local manifest (`data/raw/download_manifest.txt`)
  - Supports full download and incremental update modes

**Google Colab (training environment):**
- Platform: Google Colab (Jupyter-based cloud GPU environment)
- Auth: Google Drive OAuth via `from google.colab import drive; drive.mount('/content/drive')`
- Usage: Model training runs only; not part of core prediction pipeline
- Implementation: `notebooks/colab_training.ipynb`
- Project is mounted from Google Drive at `/content/drive/MyDrive/tese`

## Data Storage

**Databases:**
- None - No relational or NoSQL database is used
- All data is stored as flat CSV files on the local filesystem

**Primary Dataset:**
- Format: CSV
- Location: `data/processed/complete_target_leagues_dataset.csv`
- Access: `pandas.read_csv()` throughout all pipeline modules
- Size: ~40,945 match rows (as of 2026-02-04)

**Raw Data:**
- Format: CSV files per year
- Location: `data/raw/` directory
- Source: Downloaded from Oracle's Elixir S3 bucket

**Model Artifacts:**
- Format: `.joblib` (binary serialization via `joblib`)
- Primary location: `models/production/`
  - `best_model.joblib` - Best trained Logistic Regression model
  - `scaler.joblib` - StandardScaler fitted on training data
  - `encoders.joblib` - Categorical encoders (TargetEncoder, LabelEncoder)
- Additional artifacts in `models/`:
  - `champion_meta_strength.joblib` - Champion effectiveness by patch
  - `champion_synergies.joblib` - Champion synergy calculations
  - `team_historical_performance.joblib` - Team win rate histories

**Versioned Data:**
- Format: CSV files with version-tagged filenames
- Location: `data/processed/versions/`
- Version format: `dataset_v{YYYY}.{MM}.{DD}_{hash8}.csv`
- Version history: `data/processed/versions/version_history.txt` (pipe-delimited)

**File Storage:**
- Local filesystem only. No cloud blob storage used for artifact storage (Colab uses Google Drive mount as a local path proxy).

**Caching:**
- None - No Redis, Memcached, or in-memory caching layer. Data is re-read from CSV each run.

## Authentication & Identity

**Auth Provider:**
- None - No user authentication system exists
- The system is a local CLI/script-based tool, not a web application
- No login, sessions, JWT, or OAuth flows in the core codebase

**Google Colab Auth:**
- Handled automatically by `drive.mount()` in notebook context only

## Monitoring & Observability

**Error Tracking:**
- None - No Sentry, Datadog, or equivalent error tracking service is integrated

**Drift Detection:**
- Custom implementation in `src/training/drift.py` - `DriftDetector` class
- Triggered thresholds defined in `src/config.py` under `DRIFT_THRESHOLDS`:
  - AUC drop > 2%
  - F1 drop > 3%
  - Accuracy drop > 3%
  - Log loss increase > 5%

**Logs:**
- `print()` statements throughout all modules (no structured logging framework)
- No log files written; all output goes to stdout/stderr
- Pipeline results printed with separator lines (`=`*60 formatting)

**Model Versioning:**
- Custom implementation in `src/training/versioning.py` - `ModelVersionManager`
- Versions stored in `models/versions/` directory
- No MLflow, Weights & Biases, or similar MLOps platform

## CI/CD & Deployment

**Hosting:**
- None - no deployed service detected
- System runs locally or on Google Colab

**CI Pipeline:**
- None detected - no GitHub Actions, CircleCI, or equivalent configuration files found

**Package:**
- Installable as a Python package via `pip install -e .` (editable mode)
- Entry point `lol-predict` defined in `setup.py` for CLI access to predictor

## Environment Configuration

**Required env vars:**
- None - the system requires no environment variables for normal operation
- All configuration is via hardcoded paths in `src/config.py` relative to project root

**Optional / implicit:**
- `PYTHONPATH` may need to include project root (several modules use `sys.path.append()` as a workaround)

**Secrets location:**
- No secrets required; Oracle's Elixir S3 bucket is publicly accessible without credentials

## Webhooks & Callbacks

**Incoming:**
- None - no webhook endpoints; no web server

**Outgoing:**
- None - the system only makes outbound HTTP GET requests to Oracle's Elixir S3 for data downloads

---

*Integration audit: 2026-02-24*
