# Prediction module
"""
Match prediction interface.

Modules:
- predictor: InteractiveLoLPredictor class for real-time predictions
- confidence: ConfidenceEstimator for calibrated predictions with uncertainty

Legacy aliases:
- interactive_match_predictor -> predictor
"""

from .confidence import ConfidenceEstimator

__all__ = ["ConfidenceEstimator"]
