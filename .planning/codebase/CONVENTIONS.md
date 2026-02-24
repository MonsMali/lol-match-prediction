# Coding Conventions

**Analysis Date:** 2026-02-24

## Naming Patterns

**Files:**
- Module files use `snake_case`: `advanced_feature_engineering.py`, `interactive_match_predictor.py`
- Class files named after their primary class: `trainer.py` contains `ContinuousTrainer`, `versioning.py` contains `ModelVersionManager`
- Script files are imperative verbs: `run_training.py`, `compare_results.py`, `refresh_data.py`
- Config files named `config.py` at module level

**Classes:**
- `PascalCase` throughout: `AdvancedFeatureEngineering`, `UltimateLoLPredictor`, `ContinuousTrainer`, `ModelVersionManager`, `EdgeCaseHandler`, `DriftDetector`, `SchemaValidator`
- Dataclasses follow the same pattern: `TrainingResult`, `PipelineResult`, `DriftResult`, `ModelMetadata`
- Custom exceptions use descriptive names ending in `Exception`: `RestartDraftException`

**Functions and Methods:**
- `snake_case` for all functions and methods: `load_and_analyze_data()`, `create_advanced_features()`, `split_data_temporally()`, `calculate_ece()`
- Private methods prefixed with single underscore: `_create_patch_timeline()`, `_convert_defaultdicts()`, `_generate_version()`
- Static helper methods use `@staticmethod` decorator: `_convert_defaultdicts`, `calculate_mcc`, `calculate_ece`
- Factory/builder functions named with verbs: `create_evaluation_report()`, `create_quality_report()`

**Variables:**
- `snake_case` for all variables: `data_path`, `champion_meta_strength`, `head_to_head_records`
- Constants use `UPPER_SNAKE_CASE`: `CATBOOST_AVAILABLE`, `SHAP_AVAILABLE`, `PLOTTING_AVAILABLE`, `HIGH_CONFIDENCE_THRESHOLD`, `MIN_TEAM_MATCHES`
- Boolean flags are descriptive: `is_calibrated`, `is_production`, `use_stratified_temporal`

**Module-Level Constants:**
- Path constants in `src/config.py` are `UPPER_SNAKE_CASE`: `PROJECT_ROOT`, `DATASET_PATH`, `BEST_MODEL_PATH`
- Configuration dictionaries use `UPPER_SNAKE_CASE` keys: `DRIFT_THRESHOLDS`, `METRIC_WEIGHTS`, `TRAINING_DEFAULTS`
- Class-level penalty/threshold dicts: `PENALTIES`, `MIN_TEAM_MATCHES`, `VERSION_PREFIX`

## Code Style

**Formatting:**
- No automated formatter (black is listed as a dev dependency in `setup.py` but not enforced)
- 4-space indentation throughout
- String quotes: mix of single and double quotes, no consistent preference
- Lines are generally kept readable but no enforced line length limit

**Linting:**
- `flake8` and `mypy` listed as dev dependencies in `setup.py` extras but no config files present (no `.flake8`, `setup.cfg`, or `pyproject.toml`)
- `warnings.filterwarnings('ignore')` used at module level in most files to suppress sklearn/numpy warnings: `src/features/engineering.py`, `src/models/robustness.py`, `src/features/edge_cases.py`, `src/data/quality.py`, `src/prediction/confidence.py`, `src/models/explainability.py`

## Import Organization

**Order observed across source files:**
1. Standard library imports (`os`, `sys`, `pathlib`, `datetime`, `typing`, `dataclasses`, `collections`, `hashlib`, `shutil`)
2. Third-party data/ML imports (`pandas`, `numpy`, `sklearn`, `xgboost`, `lightgbm`, `joblib`, `scipy`)
3. Visualization (`matplotlib`, `seaborn`)
4. Internal project imports (`from src.config import ...`, `from src.data... import ...`)

**Path manipulation for imports:**
- Legacy and standalone scripts add to `sys.path` manually at module load:
  ```python
  sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  ```
- Newer modules (`src/training/`, `src/data/pipeline.py`) use the package import style:
  ```python
  from src.config import DATASET_PATH, MODELS_DIR
  from .config import TrainingConfig, DEFAULT_CONFIG  # relative imports within package
  ```

**Conditional imports with availability flags:**
```python
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")
```
This pattern is used for: `catboost`, `shap`, `matplotlib`/`seaborn`, and internal optional modules (`evaluation.metrics`, `data.quality`).

**Path Aliases:**
- None configured. Imports are either package-relative or via `sys.path` manipulation.

## Error Handling

**Patterns:**
- `FileNotFoundError` raised with descriptive messages including the expected path and recovery instructions:
  ```python
  raise FileNotFoundError(
      f"Model '{model_name}' not found. Searched:\n"
      f"  - {production_path}\n"
      f"  - {root_path}"
  )
  ```
  Used in `src/config.py` (`get_model_path`, `get_dataset_path`) and `src/features/engineering.py`.

- `ValueError` raised for invalid input states:
  ```python
  raise ValueError(f"WRONG DATASET! Must use 'complete_target_leagues_dataset.csv', not the old contaminated dataset.")
  ```

- Broad `except Exception as e` with `traceback.print_exc()` used in test scripts (`tests/quick_test.py`, `tests/test_model_features.py`).

- Optional module handling uses try/except at module level, setting a boolean flag, then checking the flag before use:
  ```python
  try:
      from evaluation.metrics import MultiMetricEvaluator
      MULTI_METRIC_AVAILABLE = True
  except ImportError:
      MULTI_METRIC_AVAILABLE = False
  ```

- Prediction/inference code uses broad try/except to keep the interactive loop running: heavy use in `src/prediction/interactive_match_predictor.py` (22 try/except blocks) and `src/prediction/predictor.py` (26 blocks).

## Logging

**Framework:** `print()` - no logging library is used anywhere in the codebase.

**Patterns:**
- Section headers use `=` separators:
  ```python
  print("=" * 60)
  print("ROLLING WINDOW TRAINING")
  print("=" * 60)
  ```
- Status messages use f-strings with descriptive labels:
  ```python
  print(f"Loaded dataset: {self.df.shape}")
  print(f"Timestamp: {datetime.now().isoformat()}")
  ```
- No structured logging, no log levels, no timestamps in production code (only in `run_training.py` script)

## Comments

**Module Docstrings:**
- All newer modules (`src/training/`, `src/data/`, `src/evaluation/`) have module-level docstrings:
  ```python
  """
  Data Quality Module for LoL Match Prediction.

  Provides data quality improvements including:
  - Temporal weighting for training samples
  ...
  """
  ```

**Class Docstrings:**
- Classes have docstrings explaining purpose, key features, and attributes:
  ```python
  class EdgeCaseHandler:
      """
      Detects and handles edge cases in match prediction inputs.

      Attributes:
          known_champions: Set of champions seen during training
      """
  ```

**Method Docstrings (Google/NumPy style):**
- Newer `src/training/` and `src/data/` modules use Args/Returns/Raises format:
  ```python
  def get_model_path(model_name: str) -> Path:
      """
      Get the path for a model file, checking production first then legacy.

      Args:
          model_name: Name of the model

      Returns:
          Path to the model file

      Raises:
          FileNotFoundError: If model not found in any location
      """
  ```
- Legacy modules (`src/features/engineering.py`, `src/models/trainer.py`) have shorter or absent method docstrings.

**Inline Comments:**
- Used for data leakage warnings and critical business logic:
  ```python
  # CRITICAL: Verify we're using the correct clean dataset
  # SKIP: Clean and filter target leagues data (already cleaned in dataset creation)
  ```
- Section dividers in `src/config.py`:
  ```python
  # ============================================================================
  # Pipeline Configuration
  # ============================================================================
  ```

## Type Annotations

**Usage:**
- Newer modules use type hints consistently: `src/training/`, `src/data/`, `src/evaluation/`, `src/features/edge_cases.py`, `src/prediction/confidence.py`
- Legacy modules (`src/features/engineering.py`, `src/models/trainer.py`) have no type hints
- Standard `typing` module imports: `Dict`, `List`, `Tuple`, `Optional`, `Any`, `Union`, `Set`

**Example pattern from newer code:**
```python
def calculate_weights(self, dates: pd.Series,
                      reference_date: Optional[datetime] = None) -> np.ndarray:
```

## Dataclass Usage

**Pattern:** Dataclasses used for structured return values and configuration objects throughout `src/training/` and `src/data/`:
```python
@dataclass
class TrainingResult:
    success: bool
    version: str = ""
    promoted: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)
```
- `field(default_factory=...)` used for mutable defaults (dicts, lists)
- Dataclasses used as: `TrainingResult`, `PipelineResult`, `DriftResult`, `ModelMetadata`, `CalibrationResult`, `DataQualityReport`, `SchemaInfo`, `ValidationResult`

## Module Design

**Exports:**
- `__init__.py` files exist but are empty (no explicit `__all__` lists)
- Classes are imported directly from their module files

**Configuration Pattern:**
- Centralized path config in `src/config.py` using `pathlib.Path`
- Module-level default instances:
  ```python
  DEFAULT_CONFIG = ContinuousLearningConfig()
  ```
- Config classes use `@classmethod` factory methods (`from_dict`) and `to_dict` serialization

**Separation between legacy and new code:**
- Legacy paths (`src/feature_engineering/`, `src/data_collection/`, `src/data_processing/`) preserved for backward compatibility
- New canonical paths: `src/features/`, `src/data/`, `src/training/`, `src/evaluation/`

---

*Convention analysis: 2026-02-24*
