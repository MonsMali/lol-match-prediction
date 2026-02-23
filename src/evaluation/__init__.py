"""
Evaluation module for multi-metric model assessment.

This module provides comprehensive evaluation capabilities including:
- Multiple metrics (AUC, Log Loss, Brier Score, ECE, F1, MCC, Cohen's Kappa)
- Composite scoring for model ranking
- Calibration analysis (ECE, MCE, Calibration Slope)
- Probability calibration (Platt Scaling, Isotonic Regression)
- Uncertainty quantification (Bootstrap CI)
- Decision Curve Analysis
- Model comparison with statistical tests
"""

from .metrics import (
    # Main evaluator
    MultiMetricEvaluator,

    # Calibration
    ProbabilityCalibrator,
    CalibrationResult,

    # Uncertainty
    UncertaintyQuantifier,
    UncertaintyResult,

    # Decision analysis
    DecisionCurveResult,

    # Model comparison
    ModelComparator,

    # Convenience functions
    evaluate_model,
    generate_reliability_diagram_data,
    create_evaluation_report
)

__all__ = [
    'MultiMetricEvaluator',
    'ProbabilityCalibrator',
    'CalibrationResult',
    'UncertaintyQuantifier',
    'UncertaintyResult',
    'DecisionCurveResult',
    'ModelComparator',
    'evaluate_model',
    'generate_reliability_diagram_data',
    'create_evaluation_report'
]
