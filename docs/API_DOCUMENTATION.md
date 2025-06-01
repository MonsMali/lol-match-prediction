# API Documentation

## Overview

This document provides comprehensive API documentation for the League of Legends Match Prediction System developed by **Luís Conceição** (luis.viegas.conceicao@gmail.com).

## Core Classes

### `AdvancedFeatureEngineering`

**Location**: `src/feature_engineering/advanced_feature_engineering.py`

**Purpose**: Advanced feature engineering system for League of Legends match data with 33+ sophisticated features.

#### Constructor
```python
AdvancedFeatureEngineering(data_path=None)
```

**Parameters**:
- `data_path` (str, optional): Path to the dataset CSV file. If None, automatically locates in organized structure.

#### Key Methods

##### `load_and_analyze_data()`
```python
df = fe.load_and_analyze_data()
```
**Returns**: `pandas.DataFrame` - Loaded and analyzed dataset  
**Description**: Loads the dataset with comprehensive analysis and missing value handling.

##### `create_advanced_features()`
```python
features_df = fe.create_advanced_features()
```
**Returns**: `pandas.DataFrame` - Advanced engineered features  
**Description**: Creates 24 advanced features including meta strength, team dynamics, and champion synergies.

##### `apply_advanced_encoding()`
```python
encoded_df = fe.apply_advanced_encoding()
```
**Returns**: `pandas.DataFrame` - Final encoded feature matrix  
**Description**: Applies target encoding to categorical features and creates final feature set.

---

### `UltimateLoLPredictor`

**Location**: `src/models/ultimate_predictor.py`

**Purpose**: Comprehensive ML model training and evaluation system with multiple algorithms.

#### Constructor
```python
UltimateLoLPredictor(data_path=None)
```

#### Key Methods

##### `prepare_advanced_features()`
```python
X, y = predictor.prepare_advanced_features()
```
**Returns**: `tuple` - (features DataFrame, target Series)  
**Description**: Prepares complete advanced feature set for training.

##### `split_data_stratified_temporal()`
```python
predictor.split_data_stratified_temporal(train_size=0.6, val_size=0.2, test_size=0.2)
```
**Parameters**:
- `train_size` (float): Training set proportion
- `val_size` (float): Validation set proportion  
- `test_size` (float): Test set proportion

**Description**: Novel stratified temporal splitting maintaining year-wise stratification.

##### `train_advanced_models(quick_mode=False)`
```python
predictor.train_advanced_models(quick_mode=False)
```
**Parameters**:
- `quick_mode` (bool): If True, uses reduced hyperparameter search

**Description**: Trains comprehensive model suite with hyperparameter optimization.

##### `create_ultimate_ensemble()`
```python
predictor.create_ultimate_ensemble()
```
**Description**: Creates sophisticated ensemble using performance-weighted voting.

##### `save_ultimate_model(best_model_name)`
```python
predictor.save_ultimate_model(best_model_name)
```
**Parameters**:
- `best_model_name` (str): Name of the best performing model

**Description**: Saves the complete model system for deployment.

---

### `EnhancedUltimateLoLPredictor`

**Location**: `src/models/enhanced_ultimate_predictor.py`

**Purpose**: Enhanced prediction system with Bayesian optimization and advanced validation.

#### Constructor
```python
EnhancedUltimateLoLPredictor(data_path=None)
```

#### Key Methods

##### `split_data_stratified_random_temporal()`
```python
predictor.split_data_stratified_random_temporal(train_size=0.6, val_size=0.2, test_size=0.2, random_state=42)
```
**Parameters**:
- `train_size`, `val_size`, `test_size` (float): Split proportions
- `random_state` (int): Random seed for reproducibility

**Description**: **BREAKTHROUGH METHOD** - Stratified random temporal splitting reducing intra-year meta bias.

##### `comprehensive_statistical_evaluation(best_model_name, baseline_f1=None)`
```python
stats = predictor.comprehensive_statistical_evaluation(best_model_name, baseline_f1=0.7485)
```
**Parameters**:
- `best_model_name` (str): Name of best model
- `baseline_f1` (float, optional): Baseline F1 score for comparison

**Returns**: `dict` - Statistical analysis results including confidence intervals

**Description**: Comprehensive statistical evaluation with bootstrap confidence intervals and significance testing.

##### `create_results_visualization()`
```python
filepath = predictor.create_results_visualization()
```
**Returns**: `str` - Path to saved visualization file  
**Description**: Creates comprehensive results visualization for thesis presentation.

---

### `InteractiveLoLPredictor`

**Location**: `src/prediction/interactive_match_predictor.py`

**Purpose**: Interactive match prediction system with real-time draft simulation.

#### Constructor
```python
InteractiveLoLPredictor(model_path=None)
```

#### Key Methods

##### `load_model_components()`
```python
success = predictor.load_model_components()
```
**Returns**: `bool` - Success status  
**Description**: Loads trained model, scaler, and feature engineering components.

##### `get_picks_and_bans()`
```python
match_data = predictor.get_picks_and_bans()
```
**Returns**: `dict` - Complete match data with picks and bans  
**Description**: Interactive draft phase with professional pick/ban simulation.

##### `predict_match(match_data)`
```python
prediction = predictor.predict_match(match_data)
```
**Parameters**:
- `match_data` (dict): Match data from draft phase

**Returns**: `dict` - Prediction results with probabilities  
**Description**: Generates match outcome prediction with confidence metrics.

---

### `ComprehensiveLogisticRegressionComparison`

**Location**: `src/models/comprehensive_logistic_regression_comparison.py`

**Purpose**: Comprehensive comparison of validation strategies with Bayesian optimization.

#### Constructor
```python
ComprehensiveLogisticRegressionComparison(data_path=None)
```

#### Key Methods

##### `set_split_configuration(config_name)`
```python
comparison.set_split_configuration('optimized')  # 'standard', 'balanced', 'optimized'
```
**Parameters**:
- `config_name` (str): Split configuration name

**Description**: Sets train/validation/test split ratios.

##### `perform_nested_cv_for_strategy(strategy_name, n_calls=50)`
```python
results = comparison.perform_nested_cv_for_strategy('stratified_temporal', n_calls=50)
```
**Parameters**:
- `strategy_name` (str): Validation strategy name
- `n_calls` (int): Bayesian optimization trials

**Returns**: `dict` - Nested cross-validation results  
**Description**: Performs nested CV with Bayesian hyperparameter optimization.

---

## Usage Examples

### Quick Model Training
```python
from src.models.ultimate_predictor import UltimateLoLPredictor

# Initialize and train
predictor = UltimateLoLPredictor()
X, y = predictor.prepare_advanced_features()
predictor.split_data_stratified_temporal()
predictor.train_advanced_models(quick_mode=True)

# Evaluate
best_model, results = predictor.evaluate_models()
final_results = predictor.final_test_evaluation(best_model)
```

### Bayesian Optimization Training
```python
from src.models.enhanced_ultimate_predictor import main_enhanced

# Run with breakthrough validation
predictor, results = main_enhanced(
    use_bayesian=True, 
    save_models=True, 
    use_stratified_random=True
)
```

### Interactive Prediction
```python
from src.prediction.interactive_match_predictor import InteractiveLoLPredictor

predictor = InteractiveLoLPredictor()
if predictor.load_model_components():
    match_data = predictor.get_picks_and_bans()
    prediction = predictor.predict_match(match_data)
    print(f"Prediction: {prediction}")
```

## Error Handling

All classes implement comprehensive error handling:

- **FileNotFoundError**: When dataset files are missing
- **ImportError**: When optional dependencies are unavailable
- **ValueError**: When invalid parameters are provided
- **KeyError**: When required data fields are missing

## Performance Metrics

### Model Performance
- **Primary Metric**: AUC-ROC (Area Under ROC Curve)
- **Secondary Metrics**: F1 Score, Accuracy, Precision, Recall
- **Validation**: 5-fold Cross-Validation with confidence intervals

### Statistical Validation
- **Bootstrap Confidence Intervals**: 1000 samples
- **Significance Testing**: t-tests vs baseline
- **Calibration Analysis**: Probability calibration curves

## Dependencies

### Required
- pandas >= 1.3.0
- numpy >= 1.21.0  
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- joblib >= 1.1.0

### Optional (Enhanced Features)
- xgboost >= 1.5.0
- lightgbm >= 3.3.0
- catboost >= 1.0.0
- optuna >= 3.0.0 (Bayesian optimization)

## Contact

**Author**: Luís Conceição  
**Email**: luis.viegas.conceicao@gmail.com  
**Project**: Master's Thesis - League of Legends Match Prediction System

---

*This API documentation covers the core functionality. For implementation details, see the source code and methodology documentation.* 