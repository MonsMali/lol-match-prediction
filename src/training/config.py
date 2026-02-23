"""
Configuration for continuous learning and training.

Central configuration for drift detection thresholds, training schedules,
and model version management.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import MODELS_DIR, EXPERIMENTS_DIR


@dataclass
class DriftConfig:
    """Configuration for drift detection."""

    # Performance drift thresholds (absolute drops)
    auc_threshold: float = 0.02      # Retrain if AUC drops by 2%
    f1_threshold: float = 0.03       # Retrain if F1 drops by 3%
    accuracy_threshold: float = 0.03  # Retrain if accuracy drops by 3%
    log_loss_threshold: float = 0.05  # Retrain if log loss increases by 5%

    # Feature drift thresholds
    psi_threshold: float = 0.1       # Population Stability Index threshold
    ks_threshold: float = 0.1        # Kolmogorov-Smirnov threshold

    # Rolling window for baseline comparison
    baseline_window_days: int = 30   # Days of data for baseline

    # Minimum samples for drift detection
    min_samples: int = 100


@dataclass
class SchedulerConfig:
    """Configuration for training scheduler."""

    # Schedule options
    schedule_type: str = "weekly"    # 'daily', 'weekly', 'monthly', 'patch'
    day_of_week: int = 0             # Monday = 0, for weekly schedule
    day_of_month: int = 1            # For monthly schedule
    hour: int = 2                    # Hour to run (24h format)

    # Trigger conditions
    min_new_matches: int = 500       # Minimum new matches to trigger data-based retrain
    force_retrain_days: int = 30     # Force retrain after N days regardless

    # Cooldown
    min_hours_between_training: int = 4  # Minimum hours between training runs


@dataclass
class VersioningConfig:
    """Configuration for model versioning."""

    # Storage paths
    models_dir: Path = field(default_factory=lambda: MODELS_DIR)
    experiments_dir: Path = field(default_factory=lambda: EXPERIMENTS_DIR)

    # Version retention
    max_versions_to_keep: int = 10   # Maximum model versions to retain
    keep_production_history: int = 5  # Keep last N production models

    # Metadata
    include_training_data_hash: bool = True
    include_feature_stats: bool = True


@dataclass
class TrainingConfig:
    """Configuration for continuous training."""

    # Training parameters
    rolling_window_months: int = 12  # Training window for rolling retrain
    validation_size: float = 0.2     # Validation set proportion
    test_size: float = 0.1           # Test set proportion

    # Promotion criteria (new model must beat baseline by at least this much)
    promotion_auc_margin: float = 0.005   # New model must be 0.5% better on AUC
    promotion_f1_margin: float = 0.005    # New model must be 0.5% better on F1

    # Training options
    quick_mode: bool = False         # Use reduced hyperparameter search
    use_stratified_temporal: bool = True  # Use meta-aware splitting

    # Model selection
    algorithms: List[str] = field(default_factory=lambda: [
        'Logistic Regression',
        'LightGBM',
        'XGBoost',
        'Random Forest'
    ])


@dataclass
class ContinuousLearningConfig:
    """Master configuration combining all training configs."""

    drift: DriftConfig = field(default_factory=DriftConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    versioning: VersioningConfig = field(default_factory=VersioningConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Global settings
    enabled: bool = True
    verbose: bool = True
    log_to_file: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ContinuousLearningConfig':
        """Create config from dictionary."""
        drift = DriftConfig(**config_dict.get('drift', {}))
        scheduler = SchedulerConfig(**config_dict.get('scheduler', {}))
        versioning = VersioningConfig(**config_dict.get('versioning', {}))
        training = TrainingConfig(**config_dict.get('training', {}))

        return cls(
            drift=drift,
            scheduler=scheduler,
            versioning=versioning,
            training=training,
            enabled=config_dict.get('enabled', True),
            verbose=config_dict.get('verbose', True),
            log_to_file=config_dict.get('log_to_file', True)
        )

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'drift': {
                'auc_threshold': self.drift.auc_threshold,
                'f1_threshold': self.drift.f1_threshold,
                'accuracy_threshold': self.drift.accuracy_threshold,
                'log_loss_threshold': self.drift.log_loss_threshold,
                'psi_threshold': self.drift.psi_threshold,
                'ks_threshold': self.drift.ks_threshold,
                'baseline_window_days': self.drift.baseline_window_days,
                'min_samples': self.drift.min_samples,
            },
            'scheduler': {
                'schedule_type': self.scheduler.schedule_type,
                'day_of_week': self.scheduler.day_of_week,
                'day_of_month': self.scheduler.day_of_month,
                'hour': self.scheduler.hour,
                'min_new_matches': self.scheduler.min_new_matches,
                'force_retrain_days': self.scheduler.force_retrain_days,
                'min_hours_between_training': self.scheduler.min_hours_between_training,
            },
            'versioning': {
                'max_versions_to_keep': self.versioning.max_versions_to_keep,
                'keep_production_history': self.versioning.keep_production_history,
                'include_training_data_hash': self.versioning.include_training_data_hash,
                'include_feature_stats': self.versioning.include_feature_stats,
            },
            'training': {
                'rolling_window_months': self.training.rolling_window_months,
                'validation_size': self.training.validation_size,
                'test_size': self.training.test_size,
                'promotion_auc_margin': self.training.promotion_auc_margin,
                'promotion_f1_margin': self.training.promotion_f1_margin,
                'quick_mode': self.training.quick_mode,
                'use_stratified_temporal': self.training.use_stratified_temporal,
                'algorithms': self.training.algorithms,
            },
            'enabled': self.enabled,
            'verbose': self.verbose,
            'log_to_file': self.log_to_file,
        }


# Default configuration instance
DEFAULT_CONFIG = ContinuousLearningConfig()
