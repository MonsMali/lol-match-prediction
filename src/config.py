"""
Centralized path configuration for the LoL Match Prediction System.

This module provides a single source of truth for all paths in the project,
making it easy to update locations and ensuring consistency across all modules.
"""

from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASET_PATH = PROCESSED_DATA_DIR / "complete_target_leagues_dataset.csv"

# Model paths
PRODUCTION_MODELS_DIR = MODELS_DIR / "production"
EXPERIMENTS_DIR = MODELS_DIR / "experiments"

# Production model files
BEST_MODEL_PATH = PRODUCTION_MODELS_DIR / "best_model.joblib"
SCALER_PATH = PRODUCTION_MODELS_DIR / "scaler.joblib"
ENCODERS_PATH = PRODUCTION_MODELS_DIR / "encoders.joblib"

# Legacy model paths (for backward compatibility)
LEGACY_MODEL_PATHS = {
    "enhanced_best_model": MODELS_DIR / "enhanced_best_model.joblib",
    "enhanced_scaler": MODELS_DIR / "enhanced_scaler.joblib",
    "enhanced_encoders": MODELS_DIR / "enhanced_encoders.joblib",
    "champion_meta_strength": MODELS_DIR / "champion_meta_strength.joblib",
    "champion_synergies": MODELS_DIR / "champion_synergies.joblib",
    "team_historical_performance": MODELS_DIR / "team_historical_performance.joblib",
}

# Output paths
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"
RESULTS_DIR = OUTPUTS_DIR / "results"


def ensure_dirs():
    """Create all necessary directories if they don't exist."""
    for dir_path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        PRODUCTION_MODELS_DIR,
        EXPERIMENTS_DIR,
        VISUALIZATIONS_DIR,
        RESULTS_DIR,
        DOCS_DIR,
        TESTS_DIR,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_model_path(model_name: str) -> Path:
    """
    Get the path for a model file, checking production first then legacy.

    Args:
        model_name: Name of the model (e.g., 'best_model', 'scaler')

    Returns:
        Path to the model file

    Raises:
        FileNotFoundError: If model not found in any location
    """
    # Check production path first
    production_path = PRODUCTION_MODELS_DIR / f"{model_name}.joblib"
    if production_path.exists():
        return production_path

    # Check legacy paths
    if model_name in LEGACY_MODEL_PATHS:
        legacy_path = LEGACY_MODEL_PATHS[model_name]
        if legacy_path.exists():
            return legacy_path

    # Check root models directory
    root_path = MODELS_DIR / f"{model_name}.joblib"
    if root_path.exists():
        return root_path

    raise FileNotFoundError(
        f"Model '{model_name}' not found. Searched:\n"
        f"  - {production_path}\n"
        f"  - {LEGACY_MODEL_PATHS.get(model_name, 'N/A')}\n"
        f"  - {root_path}"
    )


def get_dataset_path() -> Path:
    """
    Get the path to the main dataset file.

    Returns:
        Path to the dataset

    Raises:
        FileNotFoundError: If dataset not found
    """
    if DATASET_PATH.exists():
        return DATASET_PATH

    # Check legacy location
    legacy_path = DATA_DIR / "complete_target_leagues_dataset.csv"
    if legacy_path.exists():
        return legacy_path

    raise FileNotFoundError(
        f"Dataset not found. Expected at:\n"
        f"  - {DATASET_PATH}\n"
        f"  - {legacy_path}\n"
        "Run: python src/data_processing/create_complete_target_dataset.py"
    )


# ============================================================================
# Pipeline Configuration
# ============================================================================

# Oracle's Elixir S3 configuration
ORACLES_ELIXIR_S3_BASE = "https://oracleselixir-downloadable-match-data.s3-us-west-2.amazonaws.com"

# Target leagues for the prediction system
TARGET_LEAGUES = [
    'CBLOL',     # Brazil
    'EU LCS',    # Europe (old name)
    'LEC',       # Europe (new name)
    'LCK',       # Korea
    'LCS',       # North America (new name)
    'LPL',       # China
    'MSI',       # Mid-Season Invitational
    'NA LCS',    # North America (old name)
    'WLDs',      # World Championship (Oracle's Elixir format)
    'Worlds',    # World Championship (alternative name)
]

# Data versioning directory
DATA_VERSIONS_DIR = PROCESSED_DATA_DIR / "versions"


# ============================================================================
# Training Configuration
# ============================================================================

# Model versioning directory
MODEL_VERSIONS_DIR = MODELS_DIR / "versions"

# Default training parameters
TRAINING_DEFAULTS = {
    'rolling_window_months': 12,
    'validation_size': 0.2,
    'test_size': 0.1,
    'quick_mode': False,
    'use_stratified_temporal': True,
}

# Drift detection thresholds
DRIFT_THRESHOLDS = {
    'auc_drop': 0.02,      # Retrain if AUC drops by 2%
    'f1_drop': 0.03,       # Retrain if F1 drops by 3%
    'accuracy_drop': 0.03,  # Retrain if accuracy drops by 3%
    'log_loss_increase': 0.05,  # Retrain if log loss increases by 5%
}

# Multi-metric evaluation weights
METRIC_WEIGHTS = {
    'auc': 0.30,       # Discrimination ability
    'log_loss': 0.25,  # Probability accuracy
    'brier': 0.20,     # Calibration quality
    'ece': 0.15,       # Expected calibration error
    'f1': 0.10,        # Classification performance
}


def ensure_all_dirs():
    """Create all necessary directories including new ones."""
    ensure_dirs()
    for dir_path in [DATA_VERSIONS_DIR, MODEL_VERSIONS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


# Ensure directories exist on import
ensure_dirs()
