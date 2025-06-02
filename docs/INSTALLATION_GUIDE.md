# Installation Guide

## Prerequisites

- **Python**: 3.8+ (Recommended: 3.9 or 3.10)
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 8GB RAM (16GB recommended for full model training)
- **Storage**: At least 2GB free space for datasets and models

## Quick Setup

### 1. Clone the Repository
```bash
git clone https://github.com/MonsMali/lol-match-prediction.git
cd lol-match-prediction
```

### 2. Install Dependencies
```bash
# Required packages
pip install pandas numpy scikit-learn matplotlib seaborn joblib

# Enhanced ML packages (recommended)
pip install xgboost lightgbm catboost optuna

# Statistical analysis packages
pip install scipy statsmodels
```

### 3. Verify Installation
```bash
# Quick system check
python tests/quick_test.py

# Feature engineering test
python tests/test_model_features.py
```

## Detailed Installation

### Step 1: Python Environment Setup

#### Option A: Using Anaconda (Recommended)
```bash
# Create new environment
conda create -n lol-prediction python=3.9
conda activate lol-prediction

# Install packages
conda install pandas numpy scikit-learn matplotlib seaborn joblib
conda install -c conda-forge xgboost lightgbm optuna
pip install catboost
```

#### Option B: Using pip with virtual environment
```bash
# Create virtual environment
python -m venv lol-prediction
source lol-prediction/bin/activate  # On Windows: lol-prediction\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install packages
pip install -r requirements.txt  # If available, or install manually
```

### Step 2: Verify GPU Support (Optional)

#### For XGBoost GPU Support
```bash
# Check CUDA installation
nvidia-smi

# Install XGBoost with GPU support
pip install xgboost[gpu]
```

#### For LightGBM GPU Support
```bash
# Install LightGBM with GPU support
pip install lightgbm --config-settings=cmake.define.USE_GPU=ON
```

### Step 3: Project Structure Verification

Run the comprehensive test to ensure everything is set up correctly:

```bash
python tests/quick_test.py
```

Expected output:
```
🧪 COMPREHENSIVE CODE TESTING (NO MODEL TRAINING)
================================================================================
📦 TESTING DEPENDENCIES
   ✅ pandas
   ✅ numpy
   ✅ scikit-learn
   ✅ matplotlib
   ✅ seaborn
   ✅ joblib
   ✅ xgboost (optional)
   ✅ lightgbm (optional)
   ✅ catboost (optional)
   ✅ optuna (optional)

📁 TESTING DIRECTORY STRUCTURE
   ✅ All directories present

🔗 TESTING IMPORTS
   ✅ All classes imported successfully

🎯 TEST SUMMARY
   ✅ PASS Dependencies
   ✅ PASS Directory Structure
   ✅ PASS Imports
   ✅ PASS Data Loading
   ✅ PASS Model Classes

🎉 ALL TESTS PASSED! Your reorganized code is ready to use!
```

## Package Versions

### Core Requirements
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0
```

### Enhanced ML Packages
```
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
optuna>=3.0.0
```

### Statistical Analysis
```
scipy>=1.7.0
statsmodels>=0.13.0
```

## Troubleshooting

### Common Installation Issues

#### Issue: "ModuleNotFoundError"
```bash
# Solution: Ensure you're in the correct environment
conda activate lol-prediction  # or source lol-prediction/bin/activate

# Verify Python path
python -c "import sys; print(sys.path)"
```

#### Issue: "Permission denied" on Windows
```bash
# Solution: Run as administrator or use --user flag
pip install --user package_name
```

#### Issue: CatBoost installation fails
```bash
# Solution: Try alternative installation
conda install -c conda-forge catboost
# or
pip install catboost --no-cache-dir
```

#### Issue: GPU packages not working
```bash
# Solution: Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# For XGBoost GPU issues
pip uninstall xgboost
pip install xgboost==1.7.4  # Try specific version
```

### Performance Optimization

#### For Large Datasets
```python
# Increase memory limits if needed
import os
os.environ['PYTHONHASHSEED'] = '0'  # For reproducibility
```

#### For Faster Training
```bash
# Use multiple cores (already configured in code)
export OMP_NUM_THREADS=4  # Adjust based on your CPU
```

## Recent Improvements

### Simplified Dataset Path Resolution (Latest Update)
The codebase now uses a **single, deterministic path** for dataset loading:
- **Primary Path**: `project_root/Data/complete_target_leagues_dataset.csv`
- **No Fallbacks**: Eliminates confusion and ensures correct dataset usage
- **Clear Errors**: If dataset not found, run: `python src/data_processing/create_complete_target_dataset.py`

**Benefits:**
- ✅ More reliable and predictable
- ✅ Eliminates potential wrong dataset loading
- ✅ Simpler code maintenance
- ✅ Cleaner error messages

## Data Setup

### Option 1: Using Provided Dataset
If you have access to the prepared dataset:
```bash
# Place the dataset in the data directory
cp target_leagues_dataset.csv data/
```

### Option 2: Data Collection (If building from scratch)
```bash
# Run data collection scripts
python src/data_collection/filter_target_leagues.py
python src/data_collection/analyze_focused_data.py
```

## First Run

### Quick Training Test
```bash
# Test with small subset (recommended first)
python tests/test_model_features.py
```

### Full Model Training
```bash
# Ultimate predictor
python src/models/ultimate_predictor.py

# Enhanced predictor with Bayesian optimization
python src/models/enhanced_ultimate_predictor.py

# Comprehensive comparison
python src/models/comprehensive_logistic_regression_comparison.py
```

### Interactive Prediction
```bash
# After training models
python src/prediction/interactive_match_predictor.py
```

## Development Setup

### For Contributing or Modifying Code

#### Install Development Dependencies
```bash
pip install pytest black flake8 jupyter
```

#### Code Style
```bash
# Format code
black src/

# Check style
flake8 src/
```

#### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/quick_test.py
```

## Environment Variables

### Optional Configuration
```bash
# Set logging level
export PYTHONPATH="${PYTHONPATH}:./src"

# For reproducible results
export PYTHONHASHSEED=0

# For GPU memory management (if using GPU)
export CUDA_VISIBLE_DEVICES=0
```

## Docker Setup (Advanced)

### Using Docker (Optional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "tests/quick_test.py"]
```

Build and run:
```bash
docker build -t lol-prediction .
docker run lol-prediction
```

## System Requirements by Use Case

### Testing and Development
- **CPU**: 2+ cores
- **RAM**: 4GB minimum
- **Storage**: 1GB

### Full Model Training
- **CPU**: 4+ cores (8+ recommended)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 2GB for models and results
- **Time**: 30-120 minutes depending on configuration

### Production Deployment
- **CPU**: 2+ cores
- **RAM**: 2GB minimum
- **Storage**: 500MB for models only

## Support

### Getting Help

1. **Check Tests**: Run `python tests/quick_test.py` to identify issues
2. **Documentation**: Review `docs/API_DOCUMENTATION.md` for usage details
3. **Issues**: Check common troubleshooting above
4. **Contact**: luis.viegas.conceicao@gmail.com

### Reporting Issues

When reporting issues, please include:
- Python version (`python --version`)
- Operating system
- Error message
- Output of `python tests/quick_test.py`

---

**Author**: Luís Conceição  
**Email**: luis.viegas.conceicao@gmail.com  
**Project**: League of Legends Match Prediction System 