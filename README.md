# League of Legends Match Prediction System

A comprehensive machine learning system for predicting League of Legends match outcomes using advanced feature engineering and multiple prediction models.

## Author

**Luís Conceição**  
📧 Email: [luis.viegas.conceicao@gmail.com](mailto:luis.viegas.conceicao@gmail.com)

## Project Overview

This thesis project focuses on developing advanced machine learning models to predict League of Legends match outcomes. The system incorporates sophisticated feature engineering, multiple validation strategies, and state-of-the-art ML algorithms.

### Key Features

- 🎮 **Advanced Feature Engineering**: 33+ engineered features including champion synergies, meta indicators, and team dynamics
- 🤖 **Multiple ML Models**: Random Forest, XGBoost, LightGBM, CatBoost, Logistic Regression, and Neural Networks
- 🔬 **Bayesian Optimization**: Intelligent hyperparameter tuning using Gaussian Processes
- 📊 **Multiple Validation Strategies**: Temporal, stratified temporal, and novel stratified random approaches
- 🎯 **Interactive Prediction**: Real-time match outcome prediction with draft simulation
- 📈 **Comprehensive Analysis**: Statistical significance testing and performance visualization

## Project Structure

```
📦 Tese/
├── 📁 src/                          # Source code
│   ├── 📁 models/                   # ML model implementations
│   │   ├── ultimate_predictor.py
│   │   ├── enhanced_ultimate_predictor.py
│   │   └── comprehensive_logistic_regression_comparison.py
│   ├── 📁 feature_engineering/      # Advanced feature engineering
│   ├── 📁 data_collection/          # Data processing scripts
│   └── 📁 prediction/               # Interactive prediction system
├── 📁 data/                         # Datasets
├── 📁 models/                       # Trained models
├── 📁 visualizations/               # Generated plots and charts
├── 📁 results/                      # Analysis results
├── 📁 tests/                        # Testing utilities
└── 📁 docs/                         # Documentation
```

## Quick Start

### 1. Dependencies Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm catboost optuna joblib
```

### 2. Test Everything Works
```bash
# Quick system check
python tests/quick_test.py

# Feature engineering test
python tests/test_model_features.py
```

### 3. Train Models
```bash
# Ultimate predictor with comprehensive features
python src/models/ultimate_predictor.py

# Enhanced predictor with Bayesian optimization
python src/models/enhanced_ultimate_predictor.py

# Logistic regression comparison
python src/models/comprehensive_logistic_regression_comparison.py
```

### 4. Interactive Predictions
```bash
# Real-time match prediction
python src/prediction/interactive_match_predictor.py
```

## Model Performance

The system achieves state-of-the-art performance in League of Legends match prediction:

- 🎯 **Primary Metric**: AUC-ROC > 0.77
- 📊 **F1 Score**: > 0.74
- 🔬 **Statistical Significance**: Validated with bootstrap confidence intervals
- 📈 **Validation Strategy**: Novel stratified random temporal approach

## Key Innovations

### 1. Advanced Feature Engineering
- Champion synergy analysis across 166+ champions
- Meta-evolution indicators with patch-aware features
- Team composition dynamics and historical matchups
- Player performance metrics and ban strategy analysis

### 2. Novel Validation Methodology
- **Stratified Random Temporal**: Breakthrough approach reducing intra-year meta bias
- Maintains temporal integrity while improving generalization
- Statistically significant improvement over baseline methods

### 3. Bayesian Optimization Framework
- Gaussian Process-based hyperparameter optimization
- Intelligent exploration of continuous parameter spaces
- ~50% faster training while maintaining optimal performance

## Testing Framework

Comprehensive testing utilities ensure code reliability:

```bash
# System-wide verification
python tests/quick_test.py

# Feature engineering validation  
python tests/test_model_features.py
```

## Results and Visualizations

All results are automatically saved to organized directories:

- 📊 **Visualizations**: `visualizations/` - Performance plots, confusion matrices, feature importance
- 📈 **Results**: `results/` - Detailed analysis, statistical tests, model comparisons
- 🤖 **Models**: `models/` - Trained models with metadata for deployment

## Documentation

Detailed methodology and implementation notes:

- 📚 `docs/FOCUSED_FEATURES_METHODOLOGY.md` - Feature engineering approach
- 📊 `docs/SPLIT_STRATEGY_GUIDE.md` - Validation methodology
- 🧪 `tests/README.md` - Testing framework documentation

## Academic Context

This work represents a comprehensive approach to esports match prediction, contributing:

1. **Novel validation methodology** for temporal gaming data
2. **Advanced feature engineering** for MOBA games  
3. **Bayesian optimization framework** for ML model tuning
4. **Statistical rigor** with significance testing and confidence intervals

## Contact

For questions, collaboration, or academic inquiries:

**Luís Conceição**  
📧 [luis.viegas.conceicao@gmail.com](mailto:luis.viegas.conceicao@gmail.com)

## License

This project is part of a Master's thesis. Please contact the author for usage permissions and citation requirements.

---

*Developed as part of Master's thesis research in Machine Learning applied to Esports Analytics*
