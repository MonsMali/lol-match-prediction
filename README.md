# League of Legends Match Prediction System

> **Novel Temporal Validation for Evolving Competitive Environments - A League of Legends Machine Learning Framework**

## Author

**Luís Conceição**  
📧 Email: [luis.viegas.conceicao@gmail.com](mailto:luis.viegas.conceicao@gmail.com)  
🎓 **Master's Thesis** - Advanced Machine Learning for Esports Analytics  
📄 **Thesis**: [Novel Temporal Validation for Evolving Competitive Environments - A League of Legends Machine Learning Framework](Thesis/Novel%20Temporal%20Validation%20for%20Evolving%20Competitive%20Environments%20-%20A%20League%20of%20Legends%20Machine%20Learning%20Framework.pdf)

---

## 🏆 Project Highlights

- **🥇 State-of-the-Art Performance**: 82.97% AUC-ROC with novel validation methodology
- **🚀 Breakthrough Research**: "Feature Quality > Model Complexity" principle validation
- **🎯 Advanced Feature Engineering**: 33+ sophisticated features including meta analysis and team dynamics
- **📊 Comprehensive Analysis**: 9 ML algorithms + 3 validation strategies with Bayesian optimization
- **⚡ Production Ready**: Real-time interactive prediction system with professional UI
- **🔒 Zero Data Leakage**: Pure pre-match prediction using only picks, bans, and meta information

## 📊 Research Overview

This Master's thesis focuses on developing **novel temporal validation methodologies for evolving competitive environments**, specifically addressing the unique challenges of League of Legends match prediction where game meta continuously evolves.

### 🔬 **Key Research Contributions**

#### **1. Methodological Innovations**
- **Novel Temporal Validation Framework**: Three-strategy approach for evolving competitive environments
- **Stratified Random Temporal Validation**: Breakthrough approach reducing intra-year meta bias
- **Meta-Aware Validation**: First systematic approach accounting for game evolution in ML validation
- **Advanced Feature Engineering Framework**: 33+ engineered features for competitive gaming

#### **2. Performance Achievements**
- **Best Model**: Logistic Regression with 82.97% AUC-ROC
- **Dataset**: 37,502 professional matches (2014-2024) - **No Data Leakage**
- **Coverage**: Major leagues + international tournaments (LPL, LCK, LCS, LEC, Worlds, MSI)
- **Improvement**: +5.9% over previous state-of-the-art
- **Cross-Implementation Validation**: Consistent results across 3 independent systems

#### **3. Research Insights**
- **"Feature Quality > Model Complexity"** principle validation in competitive gaming
- Linear models can outperform complex ensembles with domain-specific feature engineering
- Temporal validation strategies crucial for evolving competitive domains
- Meta-game evolution significantly impacts prediction accuracy across patches

## 🏗️ System Architecture

```
📦 Novel Temporal Validation Framework
├── 🔧 src/
│   ├── 🧠 models/                    # Core ML models
│   │   ├── ultimate_predictor.py           # Comprehensive model suite
│   │   ├── enhanced_ultimate_predictor.py  # Bayesian optimization
│   │   └── comprehensive_logistic_regression_comparison.py
│   ├── ⚙️ feature_engineering/       # Advanced feature systems
│   │   └── advanced_feature_engineering.py
│   ├── 🎯 prediction/               # Interactive prediction
│   │   └── interactive_match_predictor.py
│   └── 📊 data_collection/          # Data processing
│       ├── oracle_elixir_analyzer.py
│       ├── filter_target_leagues.py
│       ├── analyze_focused_data.py
│       └── create_complete_target_dataset.py  # Comprehensive dataset creation
├── 📁 data/                         # Datasets
│   └── complete_target_leagues_dataset.csv  # Comprehensive dataset (37K matches)
├── 🤖 models/                       # Trained models
├── 📈 visualizations/               # Analysis plots
├── 📋 results/                      # Experiment results
├── 📚 docs/                         # Documentation
├── 🧪 tests/                        # Testing framework
└── 📄 Thesis/                       # Complete thesis document
    └── Novel Temporal Validation for Evolving Competitive Environments.pdf
```

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+, 8GB RAM (16GB recommended), 2GB storage
```

### Installation
```bash
# Clone repository
git clone https://github.com/MonsMali/lol-match-prediction.git
cd lol-match-prediction

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib
pip install xgboost lightgbm catboost optuna  # Enhanced features

# Create comprehensive dataset (optional - already provided)
python src/data_processing/create_complete_target_dataset.py

# Verify installation
python tests/quick_test.py
```

### Quick Training
```bash
# Test feature engineering
python tests/test_model_features.py

# Train models (quick mode)
python src/models/ultimate_predictor.py

# Interactive prediction
python src/prediction/interactive_match_predictor.py
```

## 📈 Performance Results

### Model Comparison
| Model | AUC-ROC | F1 Score | Key Strength |
|-------|---------|----------|--------------|
| **Logistic Regression** ⭐ | **0.8297** | **0.7485** | **Optimal with advanced features** |
| Random Forest | 0.8201 | 0.7398 | Feature importance analysis |
| XGBoost | 0.8189 | 0.7372 | Non-linear pattern detection |
| LightGBM | 0.8156 | 0.7341 | Efficient training |
| Ultimate Ensemble | 0.8312 | 0.7501 | Best overall performance |

### Research Validation
- **Statistical Significance**: p < 0.001 vs baseline methods
- **Bootstrap CI**: 0.8297 ± 0.0123 (1000 samples)
- **Cross-Validation**: 5-fold nested CV with temporal awareness

## 🎯 Feature Engineering Excellence

### Feature Categories (33 total features)

#### **Meta Analysis Features** (9 features)
- Champion meta strength by patch
- Pick/ban popularity metrics
- Meta advantage calculations
- Team composition synergies

#### **Team Performance Dynamics** (8 features)
- Historical win rates
- Recent form trends (last 10 games)
- Experience normalization
- Performance consistency

#### **Strategic Features** (7 features)
- Ban strategy analysis
- Champion scaling patterns
- Draft flexibility metrics
- High-priority target elimination

#### **Advanced Interactions** (9 features)
- Meta × Form interactions
- Scaling × Experience combinations
- Composition balance metrics

**Performance Impact**: Features contribute 41.7% improvement over baseline models.

## 🔬 Novel Validation Methodology

### Three Breakthrough Validation Strategies

#### **1. Stratified Temporal Split** ⭐ **BEST**
- **Method**: Year-wise stratified temporal splitting
- **AUC-ROC**: 0.8297
- **Innovation**: Balanced meta representation + temporal order

#### **2. Stratified Random Temporal** 🚀 **BREAKTHROUGH**
- **Method**: Within-year random sampling + temporal ordering
- **AUC-ROC**: 0.8285
- **Contribution**: Novel methodology reducing intra-year meta bias

#### **3. Pure Temporal Split**
- **Method**: Chronological split (70/20/10)
- **AUC-ROC**: 0.8278
- **Use Case**: Realistic temporal evolution simulation

## 🎮 Interactive Prediction System

### Real-Time Match Prediction
```bash
python src/prediction/interactive_match_predictor.py
```

**Features**:
- 🎯 Professional draft simulation
- ⚡ Real-time prediction (< 100ms)
- 📊 Confidence intervals
- 🔄 Interactive pick/ban phases
- 💡 Strategic insights

### Example Prediction Output
```
🏆 MATCH PREDICTION RESULTS 🏆
═══════════════════════════════════════

📈 Blue Team Win Probability: 67.4% (High Confidence)
📉 Red Team Win Probability: 32.6%

🔍 Key Factors:
✅ Blue team meta advantage: +12.3%
✅ Recent form trend: +8.7%
⚠️ Red team scaling potential: +5.2%

📊 Confidence Level: 94.2%
⏱️ Prediction Time: 87ms
```

## 📚 Documentation

### 📖 **Comprehensive Documentation Available**

| Document | Description | Link |
|----------|-------------|------|
| 🔧 **API Documentation** | Complete API reference for all classes | [`docs/API_DOCUMENTATION.md`](docs/API_DOCUMENTATION.md) |
| 🚀 **Installation Guide** | Detailed setup and troubleshooting | [`docs/INSTALLATION_GUIDE.md`](docs/INSTALLATION_GUIDE.md) |
| 📊 **Performance Report** | Model results and research insights | [`docs/MODEL_PERFORMANCE_REPORT.md`](docs/MODEL_PERFORMANCE_REPORT.md) |
| 🧠 **Feature Methodology** | Advanced feature engineering details | [`docs/FOCUSED_FEATURES_METHODOLOGY.md`](docs/FOCUSED_FEATURES_METHODOLOGY.md) |
| ⚖️ **Validation Strategy** | Temporal splitting methodology | [`docs/SPLIT_STRATEGY_GUIDE.md`](docs/SPLIT_STRATEGY_GUIDE.md) |

## 🔧 Technical Implementation

### Core Technologies
- **Language**: Python 3.8+
- **ML Framework**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Optimization**: Optuna (Bayesian optimization)
- **Validation**: Custom temporal strategies
- **Visualization**: matplotlib, seaborn
- **Data Processing**: pandas, numpy

### System Requirements
- **Development**: 4GB RAM, 2+ CPU cores
- **Training**: 16GB RAM, 4+ CPU cores (recommended)
- **Production**: 2GB RAM, minimal CPU requirements

## 🧪 Testing Framework

### Comprehensive Testing Suite
```bash
# Quick system verification
python tests/quick_test.py

# Feature engineering validation
python tests/test_model_features.py

# Full test suite
python -m pytest tests/
```

**Test Coverage**:
- ✅ Import validation
- ✅ Data loading verification
- ✅ Feature engineering pipeline
- ✅ Model instantiation
- ✅ Path resolution

## 📊 Dataset Information

### Professional Match Data
- **Total Matches**: 37,502 professional matches (**Zero Data Leakage**)
- **Time Span**: 2014-2024 (11 years of esports evolution)
- **Coverage**: 6 major leagues + international tournaments
- **Quality**: Pre-match information only (picks, bans, meta) - realistic prediction setup
- **Tournaments**: Includes prestigious Worlds Championship and MSI

### Data Distribution
| League/Tournament | Matches | Coverage |
|------------------|---------|----------|
| **LPL** (China) | 11,848 | 31.6% |
| **LCK** (Korea) | 8,842 | 23.6% |
| **LCS** (North America) | 6,790 | 18.1% |
| **LEC** (Europe) | 6,304 | 16.8% |
| **🌍 Worlds** (Championship) | 2,490 | 6.6% |
| **🌍 MSI** (Mid-Season) | 1,228 | 3.3% |

### Comprehensive Dataset Creation
Our dataset represents a **breakthrough in data quality** for LoL esports prediction:

#### **🔧 Advanced Processing Pipeline**
- **11 Years of Data**: Processed 996,864 raw records from Oracle's Elixir
- **Intelligent Filtering**: Extracted 225,012 target league records  
- **Quality Transformation**: Created 37,502 high-quality team-match records
- **Zero Leakage**: Strict pre-match information only (picks, bans, meta)

#### **🏆 Tournament Integration**
- **International Prestige**: Added Worlds Championship and MSI tournaments
- **Complete Coverage**: Every major region + premier international events
- **Historical Depth**: From 2014 Worlds to 2024 tournaments

#### **📊 Processing Statistics**
- **Extraction Rate**: 3.8% (quality over quantity approach)
- **Perfect Champion Data**: 100% complete picks/bans for all matches
- **Clean Structure**: Team-match format optimized for ML training

### Data Quality Features
- **🔒 No Data Leakage**: Only pre-match information (picks, bans, meta)
- **🏆 Complete Coverage**: All major regions + international tournaments  
- **📊 Perfect Champion Data**: 100% complete pick/ban information
- **⚡ Optimized Structure**: Team-match format for efficient processing

## 🤝 Contributing

### For Researchers and Developers

1. **Code Style**: Follow PEP 8 guidelines
2. **Testing**: All new features must include tests
3. **Documentation**: Update relevant docs for changes
4. **Performance**: Maintain or improve model performance

### Research Extensions
- **Real-time Learning**: Adaptive models for meta shifts
- **Cross-Region Analysis**: Multi-league transfer learning
- **Granular Predictions**: In-game state prediction
- **Advanced Interactions**: Deep learning feature combinations

## 📄 License & Citation

### Academic Use
This project represents the implementation of a Master's thesis. When using this work, please cite:

```bibtex
@mastersthesis{conceicao2024lol,
  title={Novel Temporal Validation for Evolving Competitive Environments: 
         A League of Legends Machine Learning Framework},
  author={Conceição, Luís},
  year={2024},
  school={Master's Thesis},
  email={luis.viegas.conceicao@gmail.com},
  note={Available at: https://github.com/MonsMali/lol-match-prediction}
}
```

### Thesis Reference
**Full Thesis Document**: [Novel Temporal Validation for Evolving Competitive Environments - A League of Legends Machine Learning Framework](Thesis/Novel%20Temporal%20Validation%20for%20Evolving%20Competitive%20Environments%20-%20A%20League%20of%20Legends%20Machine%20Learning%20Framework.pdf)

## 🌟 Acknowledgments

Special thanks to:
- **Oracle's Elixir** for comprehensive esports data
- **League of Legends Esports** community for domain insights
- **Scikit-learn** and **ML community** for excellent frameworks
- **Academic advisors** for research guidance and thesis supervision

## 📞 Contact

**Luís Conceição**  
📧 **Email**: [luis.viegas.conceicao@gmail.com](mailto:luis.viegas.conceicao@gmail.com)  
🔗 **GitHub**: [MonsMali/lol-match-prediction](https://github.com/MonsMali/lol-match-prediction)  
💼 **LinkedIn**: Connect for research collaboration  
📄 **Thesis**: [Complete Document Available](Thesis/Novel%20Temporal%20Validation%20for%20Evolving%20Competitive%20Environments%20-%20A%20League%20of%20Legends%20Machine%20Learning%20Framework.pdf)

---

## 🏅 Research Impact

> *"This thesis demonstrates that sophisticated feature engineering enables simpler models to achieve state-of-the-art performance, while introducing novel temporal validation methodologies applicable to any evolving competitive domain."*

**Key Contributions**: Novel temporal validation for competitive gaming • Advanced feature engineering • Production-ready system • Breakthrough insights for evolving environments

---

⭐ **Star this repository** if you find it useful for your research or projects!

🔄 **Watch** for updates on methodology improvements and new features.

🍴 **Fork** to build upon this research for your own esports analytics projects.

**Note**: This repository implements the complete methodology from the Master's thesis "Novel Temporal Validation for Evolving Competitive Environments". The system is designed for educational and analytical use, not for gambling or commercial betting applications.
