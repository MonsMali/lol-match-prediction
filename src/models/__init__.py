# Models module
"""
Machine learning model implementations.

Modules:
- trainer: UltimateLoLPredictor class for multi-model training
- optimizer: Enhanced predictor with Bayesian optimization
- comprehensive_logistic_regression_comparison: Strategy comparison
- robustness: RobustnessAnalyzer for learning/validation curves
- explainability: ModelExplainer for SHAP-based explanations
"""

from .robustness import RobustnessAnalyzer, get_default_param_configs
from .explainability import ModelExplainer

__all__ = ["RobustnessAnalyzer", "get_default_param_configs", "ModelExplainer"]
