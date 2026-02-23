# League of Legends Match Prediction System

## Project Overview

**Author**: Luis Conceicao (luis.viegas.conceicao@gmail.com)
**Type**: Master's Thesis
**Title**: Novel Temporal Validation for Evolving Competitive Environments - A League of Legends Machine Learning Framework
**University**: Aarhus University, Business and Social Sciences

This is a machine learning system for predicting professional League of Legends match outcomes using pre-match information only (picks, bans, team data). The key innovation is the novel temporal validation methodology designed for evolving competitive environments where game meta continuously changes.

## Key Achievements

- **Best Model**: Logistic Regression with 82.97% AUC-ROC
- **Dataset**: 37,502 professional matches (2014-2024) from LPL, LCK, LCS, LEC, Worlds, MSI
- **Features**: 33+ advanced engineered features
- **Principle Validated**: "Feature Quality > Model Complexity" - linear models outperform complex ensembles with good feature engineering

## Project Structure

```
/mnt/d/Tese/
├── data/                              # Dataset storage
│   ├── raw/                           # Original Oracle's Elixir files
│   └── processed/                     # Processed datasets
│       └── complete_target_leagues_dataset.csv
├── models/                            # Trained models
│   ├── production/                    # Best models for deployment
│   │   ├── best_model.joblib
│   │   ├── scaler.joblib
│   │   └── encoders.joblib
│   ├── experiments/                   # Experimental model runs
│   ├── enhanced_best_model.joblib     # Legacy: Best Logistic Regression model
│   ├── enhanced_scaler.joblib         # Legacy: StandardScaler for features
│   ├── champion_meta_strength.joblib  # Champion meta data
│   ├── champion_synergies.joblib      # Synergy calculations
│   └── team_historical_performance.joblib  # Team performance data
├── src/
│   ├── __init__.py
│   ├── config.py                      # Centralized path configuration
│   ├── data/                          # Data processing (consolidated)
│   │   ├── __init__.py
│   │   ├── filter.py                  # League filtering
│   │   ├── extractor.py               # Feature extraction
│   │   ├── analyzer.py                # Oracle Elixir analysis
│   │   └── processor.py               # Dataset creation
│   ├── features/                      # Feature engineering
│   │   ├── __init__.py
│   │   └── engineering.py             # AdvancedFeatureEngineering
│   ├── models/                        # Model training
│   │   ├── __init__.py
│   │   ├── trainer.py                 # UltimateLoLPredictor
│   │   ├── optimizer.py               # Bayesian optimization
│   │   └── comprehensive_logistic_regression_comparison.py
│   ├── prediction/                    # Inference
│   │   ├── __init__.py
│   │   └── predictor.py               # InteractiveLoLPredictor
│   ├── data_collection/               # Legacy: Data extraction scripts
│   ├── data_processing/               # Legacy: Dataset creation
│   └── feature_engineering/           # Legacy: Feature engineering
├── docs/                              # Documentation
│   ├── API_DOCUMENTATION.md
│   ├── INSTALLATION_GUIDE.md
│   ├── MODEL_PERFORMANCE_REPORT.md
│   ├── THESIS_OVERVIEW.md
│   ├── FOCUSED_FEATURES_METHODOLOGY.md
│   └── SPLIT_STRATEGY_GUIDE.md
├── tests/                             # Test suite
│   └── __init__.py
├── outputs/                           # Generated outputs
│   ├── visualizations/
│   └── results/
├── notebooks/                         # Jupyter notebooks
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation
├── README.md
├── CLAUDE.md                          # AI context file
└── thesis.pdf                         # Complete thesis
```

## Core Classes

### AdvancedFeatureEngineering
**File**: `src/features/engineering.py` (or legacy: `src/feature_engineering/advanced_feature_engineering.py`)

Creates 33+ features including:
- **Meta Analysis** (9): Champion meta strength, pick/ban popularity, meta advantage
- **Team Performance** (8): Historical win rates, recent form, experience
- **Strategic** (7): Ban strategy, champion scaling, draft flexibility
- **Interactions** (9): Meta x Form, Scaling x Experience combinations

Key methods:
- `load_and_analyze_data()` - Load and analyze dataset
- `create_advanced_features()` - Generate all engineered features
- `create_advanced_features_vectorized()` - Optimized vectorized version
- `apply_advanced_encoding()` - Apply target encoding

### UltimateLoLPredictor
**File**: `src/models/trainer.py` (or legacy: `src/models/ultimate_predictor.py`)

Multi-algorithm training system supporting:
- Random Forest, XGBoost, LightGBM, CatBoost
- Logistic Regression, SVM, Neural Networks
- Ensemble methods with weighted voting

Key methods:
- `prepare_advanced_features()` - Prepare feature set
- `split_data_temporally()` - Pure temporal split
- `split_data_stratified_temporal()` - Year-wise stratified split (best results)
- `train_advanced_models()` - Train all models with hyperparameter optimization

### InteractiveLoLPredictor
**File**: `src/prediction/predictor.py` (or legacy: `src/prediction/interactive_match_predictor.py`)

Real-time match prediction with:
- Professional draft simulation (pick/ban phases)
- Team selection from major leagues
- Champion validation with fuzzy matching
- Best-of-series support

## Validation Strategies

Three novel temporal validation approaches:

1. **Stratified Temporal Split** (Best: 82.97% AUC)
   - Year-wise stratified temporal splitting
   - Balanced meta representation across years

2. **Stratified Random Temporal** (82.85% AUC)
   - Within-year random sampling + temporal ordering
   - Novel methodology reducing intra-year meta bias

3. **Pure Temporal Split** (82.78% AUC)
   - Chronological split (70/20/10)
   - Realistic temporal evolution simulation

## Tech Stack

- **Language**: Python 3.8+
- **ML**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Optimization**: Optuna (Bayesian optimization)
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn

## Running the System

### Install Dependencies
```bash
cd /mnt/d/Tese
pip install -r requirements.txt
# Or install in development mode:
pip install -e .
```

### Train Models
```bash
python src/models/trainer.py
# Or legacy path:
python src/models/ultimate_predictor.py
```

### Interactive Predictions
```bash
python src/prediction/predictor.py
# Or legacy path:
python src/prediction/interactive_match_predictor.py
```

### Create Dataset (if needed)
```bash
python src/data/processor.py
# Or legacy path:
python src/data_processing/create_complete_target_dataset.py
```

## Feature Engineering Details

The system uses ONLY pre-match information (zero data leakage):
- **Champion picks**: 5 per team (top, jungle, mid, bot, support)
- **Champion bans**: 5 per team
- **Team identities**: Names and leagues
- **Meta information**: Patch, year, split, playoffs status

NO in-game stats are used (kills, gold, objectives, etc.)

## Model Files

Pre-trained models in `models/production/` (with legacy copies in `models/`):
- `best_model.joblib` - Logistic Regression (best performer)
- `scaler.joblib` - Feature scaler
- `encoders.joblib` - Categorical encoders

Additional model artifacts in `models/`:
- `champion_meta_strength.joblib` - Champion effectiveness by patch
- `champion_synergies.joblib` - Synergy calculations
- `team_historical_performance.joblib` - Team win rate histories

## Future Work

The user wants to:
1. **Enhance the predictor** - Improve accuracy or add features
2. **Create a web app** - Deploy as a web application for real-time predictions

## Important Notes

- Do Not use emojis, make this look professional
- Dataset source: Oracle's Elixir professional LoL match data
- All predictions are for educational/analytical purposes only
- The system validates that simpler models (Logistic Regression) can outperform complex ones with proper feature engineering
- Temporal validation is crucial for evolving competitive environments

## Quick Reference Commands

```python
# Using centralized config
from src.config import DATASET_PATH, BEST_MODEL_PATH, SCALER_PATH

# Load feature engineering (new path)
from src.features.engineering import AdvancedFeatureEngineering
fe = AdvancedFeatureEngineering()  # Uses config paths automatically
df = fe.load_and_analyze_data()
features = fe.create_advanced_features_vectorized()

# Load trained model
import joblib
model = joblib.load("models/production/best_model.joblib")
scaler = joblib.load("models/production/scaler.joblib")

# Make prediction
from src.prediction.predictor import InteractiveLoLPredictor
predictor = InteractiveLoLPredictor()
```

### Legacy Paths (still supported)
```python
from src.feature_engineering.advanced_feature_engineering import AdvancedFeatureEngineering
from src.prediction.interactive_match_predictor import InteractiveLoLPredictor
```
