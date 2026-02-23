"""
Training module for continuous learning and model management.

This module provides:
- Model versioning and lifecycle management
- Drift detection for performance and features
- ADWIN concept drift detection
- Performance decay rate analysis
- Scheduled training with configurable triggers
- Continuous training with automatic promotion
"""

from .versioning import ModelVersionManager
from .drift import (
    DriftDetector,
    DriftResult,
    DriftReport,
    DecayRateResult,
    ADWINResult,
    print_drift_report
)
from .scheduler import TrainingScheduler
from .trainer import ContinuousTrainer

__all__ = [
    'ModelVersionManager',
    'DriftDetector',
    'DriftResult',
    'DriftReport',
    'DecayRateResult',
    'ADWINResult',
    'print_drift_report',
    'TrainingScheduler',
    'ContinuousTrainer'
]
