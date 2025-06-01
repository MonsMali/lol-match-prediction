# League of Legends Match Prediction - Master's Thesis

A comprehensive machine learning system for predicting professional League of Legends match outcomes using advanced feature engineering and multiple modeling approaches.

## 🎯 Project Overview

This thesis project develops a sophisticated prediction system for professional League of Legends matches using data from Oracle's Elixir. The system progresses through three main development stages, ultimately achieving optimal performance with logistic regression models.

## 📊 Key Results

- **Best Model**: Logistic Regression with focused feature engineering
- **Performance**: High accuracy on professional match prediction
- **Features**: 200+ engineered features including champion synergies, meta strength, and team performance metrics
- **Interactive System**: Real-time pick/ban simulation for match prediction

## 🏗️ Project Structure

```
📁 lol-match-prediction-thesis/
├── 📁 src/                           # Main source code
│   ├── 📁 data_collection/           # Data gathering & processing
│   │   ├── oracle_elixir_analyzer.py # Main data collection script
│   │   ├── filter_target_leagues.py  # League filtering
│   │   ├── analyze_original_columns.py
│   │   ├── analyze_focused_data.py
│   │   └── focused_feature_extractor.py
│   ├── 📁 feature_engineering/       # Feature creation
│   │   └── advanced_feature_engineering.py
│   ├── 📁 models/                    # Model development (3 stages)
│   │   ├── ultimate_predictor.py     # Stage 1: Initial system
│   │   ├── enhanced_ultimate_predictor.py  # Stage 2: Hyperparameter tuning
│   │   └── comprehensive_logistic_regression_comparison.py  # Stage 3: Final model
│   └── 📁 prediction/                # Interactive application
│       └── interactive_match_predictor.py
├── 📁 data/                          # Dataset files (gitignored)
│   └── target_leagues_dataset.csv    # Main dataset
├── 📁 models/                        # Saved model files (gitignored)
│   ├── enhanced_models/              # Enhanced model artifacts
│   ├── bayesian_optimized_models/    # Bayesian optimization results
│   └── *.joblib                      # Trained models and preprocessors
├── 📁 visualizations/                # All plots and charts
├── 📁 experiments/                   # Experimental analyses
│   └── README_comprehensive.md       # Detailed experiment documentation
├── 📁 results/                       # Results and outputs
├── 📁 docs/                          # Thesis documentation
│   ├── FOCUSED_FEATURES_METHODOLOGY.md  # Feature engineering methodology
│   └── SPLIT_STRATEGY_GUIDE.md       # Data splitting strategies
├── requirements.txt                  # Dependencies
├── README.md                         # This file
└── .gitignore                        # Git ignore file
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Models

#### Stage 1: Initial Prediction System
```bash
cd src/models
python ultimate_predictor.py
```

#### Stage 2: Enhanced System with Hyperparameter Tuning
```bash
cd src/models
python enhanced_ultimate_predictor.py
```

#### Stage 3: Final Logistic Regression Model
```bash
cd src/models
python comprehensive_logistic_regression_comparison.py
```

### Interactive Match Prediction
```bash
cd src/prediction
python interactive_match_predictor.py
```

## 📈 Model Development Progression

### Stage 1: Ultimate Predictor
- Initial baseline system
- Multiple algorithm comparison
- Basic feature engineering
- Cross-validation framework

### Stage 2: Enhanced Ultimate Predictor
- Bayesian hyperparameter optimization
- Advanced feature engineering
- Improved preprocessing pipeline
- Comprehensive model evaluation

### Stage 3: Comprehensive Logistic Regression
- **Final and best-performing model**
- Focused on logistic regression optimization
- Advanced feature selection
- Production-ready implementation

## 🎮 Interactive Prediction System

The interactive match predictor simulates a real League of Legends pick/ban phase:

1. **Team Selection**: Choose two professional teams
2. **Pick/Ban Simulation**: Interactive champion selection
3. **Real-time Prediction**: Live probability updates
4. **Feature Analysis**: See which factors influence the prediction

## 📊 Feature Engineering

The system includes 200+ engineered features:

- **Champion Meta Strength**: Current patch performance metrics
- **Team Synergies**: Champion combination effectiveness
- **Historical Performance**: Team and player statistics
- **Draft Analysis**: Pick/ban phase optimization
- **Temporal Features**: Time-based performance trends

## 📚 Documentation

- **[Feature Methodology](docs/FOCUSED_FEATURES_METHODOLOGY.md)**: Detailed feature engineering approach
- **[Split Strategy Guide](docs/SPLIT_STRATEGY_GUIDE.md)**: Data splitting methodologies
- **[Experiment Documentation](experiments/README_comprehensive.md)**: Comprehensive experiment results

## 🔬 Research Methodology

1. **Data Collection**: Professional match data from Oracle's Elixir
2. **Feature Engineering**: Advanced statistical and domain-specific features
3. **Model Development**: Iterative improvement across three stages
4. **Validation**: Rigorous cross-validation and temporal splitting
5. **Production**: Interactive prediction system

## 📊 Results & Visualizations

The project includes comprehensive visualizations:
- Model performance comparisons
- Feature importance analysis
- ROC curves and precision-recall plots
- Confusion matrices
- Learning curves
- Cross-validation results

## 🛠️ Technical Stack

- **Python 3.8+**
- **Scikit-learn**: Machine learning models
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Visualizations
- **Joblib**: Model persistence
- **CatBoost**: Gradient boosting (experimental)

## 📄 Citation

If you use this work in your research, please cite:

```
[Your Name] (2025). League of Legends Match Prediction using Advanced Machine Learning.
Master's Thesis, [Your University].
```

## 📧 Contact

[Your Name] - [Your Email]

## 📜 License

This project is part of a Master's thesis and is available for academic use.

---

**Note**: This is a research project for academic purposes. The prediction system is designed for educational and analytical use, not for gambling or commercial betting applications. 