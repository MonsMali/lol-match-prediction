"""
Continuous Training for LoL Match Prediction System.

Provides automated training with validation, promotion, and rollback.
"""

import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import DATASET_PATH, MODELS_DIR
from src.evaluation.metrics import MultiMetricEvaluator

from .config import TrainingConfig, DEFAULT_CONFIG
from .versioning import ModelVersionManager
from .drift import DriftDetector
from .scheduler import TrainingScheduler, TrainingTrigger


@dataclass
class TrainingResult:
    """Result of a training run."""
    success: bool
    version: str = ""
    promoted: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)
    comparison: Dict = field(default_factory=dict)
    training_time_seconds: float = 0.0
    error: str = ""


class ContinuousTrainer:
    """
    Continuous training system with automatic validation and promotion.

    Features:
    - Rolling window training
    - Automatic comparison with production
    - Promotion if new model is better
    - Integration with drift detection and scheduling
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize continuous trainer.

        Args:
            config: Training configuration
        """
        self.config = config or DEFAULT_CONFIG.training
        self.version_manager = ModelVersionManager()
        self.evaluator = MultiMetricEvaluator()
        self.scheduler = TrainingScheduler()
        self.drift_detector = DriftDetector()

    def train_rolling_window(self, window_months: Optional[int] = None) -> TrainingResult:
        """
        Train model using rolling window of recent data.

        Args:
            window_months: Months of data to include

        Returns:
            TrainingResult with training outcome
        """
        start_time = datetime.now()
        window_months = window_months or self.config.rolling_window_months

        print(f"\n{'='*60}")
        print("ROLLING WINDOW TRAINING")
        print(f"{'='*60}")
        print(f"Window: {window_months} months")

        result = TrainingResult(success=False)

        try:
            # Load and filter data by date
            df = pd.read_csv(DATASET_PATH)

            if 'date' not in df.columns:
                result.error = "Dataset missing 'date' column"
                return result

            df['date'] = pd.to_datetime(df['date'])
            cutoff_date = datetime.now() - timedelta(days=window_months * 30)
            df_filtered = df[df['date'] >= cutoff_date].copy()

            print(f"Data after filtering: {len(df_filtered)} rows")
            print(f"Date range: {df_filtered['date'].min()} to {df_filtered['date'].max()}")

            if len(df_filtered) < 1000:
                result.error = f"Insufficient data: {len(df_filtered)} rows (need 1000+)"
                return result

            # Import training components
            from src.feature_engineering.advanced_feature_engineering import AdvancedFeatureEngineering
            from src.models.trainer import UltimateLoLPredictor

            # Save filtered data temporarily
            temp_path = MODELS_DIR / "temp_rolling_data.csv"
            df_filtered.to_csv(temp_path, index=False)

            # Train model
            predictor = UltimateLoLPredictor(str(temp_path))
            predictor.prepare_advanced_features()

            if self.config.use_stratified_temporal:
                predictor.split_data_stratified_temporal()
            else:
                predictor.split_data_temporally()

            predictor.train_advanced_models(quick_mode=self.config.quick_mode)
            best_model, _ = predictor.evaluate_models()

            # Get model and evaluate
            model = predictor.models[best_model]
            use_scaled = predictor.results[best_model].get('use_scaled', False)

            X_test = predictor.X_test_scaled if use_scaled else predictor.X_test
            y_test = predictor.y_test

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = self.evaluator.calculate_all_metrics(y_test, y_pred, y_proba)
            metrics['composite'] = self.evaluator.calculate_composite_score(metrics)

            # Calculate data hash
            data_hash = hashlib.md5(df_filtered.to_json().encode()).hexdigest()[:8]

            # Save version
            version = self.version_manager.save_model_version(
                model=model,
                metadata={
                    'algorithm': best_model,
                    'metrics': metrics,
                    'training_config': {
                        'window_months': window_months,
                        'use_stratified_temporal': self.config.use_stratified_temporal,
                        'quick_mode': self.config.quick_mode
                    },
                    'data_hash': data_hash,
                    'feature_count': predictor.X.shape[1],
                    'training_samples': len(predictor.X_train),
                    'validation_samples': len(predictor.X_val),
                    'test_samples': len(predictor.X_test)
                },
                scaler=predictor.scaler
            )

            result.success = True
            result.version = version
            result.metrics = metrics

            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

        except Exception as e:
            result.error = str(e)
            print(f"Training error: {e}")

        result.training_time_seconds = (datetime.now() - start_time).total_seconds()
        return result

    def train_with_validation(self) -> TrainingResult:
        """
        Train model and automatically promote if better than production.

        Returns:
            TrainingResult with training and promotion outcome
        """
        print(f"\n{'='*60}")
        print("CONTINUOUS TRAINING WITH VALIDATION")
        print(f"{'='*60}")

        # Train new model
        result = self.train_rolling_window()

        if not result.success:
            return result

        # Get production metrics for comparison
        production_version = self.version_manager.get_production_version()

        if production_version and production_version != "legacy":
            # Compare with production
            comparison = self.version_manager.compare_versions(
                production_version, result.version
            )
            result.comparison = comparison

            # Check if new model is better
            auc_diff = comparison['metric_comparison'].get('auc', {}).get('difference', 0)
            f1_diff = comparison['metric_comparison'].get('f1', {}).get('difference', 0)

            should_promote = (
                auc_diff >= self.config.promotion_auc_margin and
                f1_diff >= self.config.promotion_f1_margin
            )

            print(f"\nComparison with production ({production_version}):")
            print(f"  AUC difference: {auc_diff:+.4f} (threshold: {self.config.promotion_auc_margin})")
            print(f"  F1 difference: {f1_diff:+.4f} (threshold: {self.config.promotion_f1_margin})")

            if should_promote:
                print(f"\nNew model is better - promoting to production")
                self.version_manager.promote_to_production(result.version)
                result.promoted = True
            else:
                print(f"\nNew model not significantly better - keeping current production")
                result.promoted = False

        else:
            # No production model - promote this one
            print(f"\nNo production model found - promoting new model")
            self.version_manager.promote_to_production(result.version)
            result.promoted = True

        # Record training completion
        trigger = TrainingTrigger(
            trigger_type='validation',
            reason='Continuous training with validation'
        )
        job_id = self.scheduler.schedule_training(trigger)
        self.scheduler.record_training_complete(
            job_id, result.success, result.metrics
        )

        return result

    def run_scheduled_training(self) -> Optional[TrainingResult]:
        """
        Run training if scheduled trigger is met.

        Returns:
            TrainingResult if training ran, None otherwise
        """
        should_train, reason = self.scheduler.should_retrain()

        if not should_train:
            print(f"No training needed: {reason}")
            return None

        print(f"Training triggered: {reason}")

        trigger = TrainingTrigger(
            trigger_type='scheduled',
            reason=reason
        )
        job_id = self.scheduler.schedule_training(trigger)

        result = self.train_with_validation()

        self.scheduler.record_training_complete(
            job_id, result.success, result.metrics
        )

        return result

    def run_drift_triggered_training(self, current_metrics: Dict[str, float]) -> Optional[TrainingResult]:
        """
        Run training if drift is detected.

        Args:
            current_metrics: Current model performance metrics

        Returns:
            TrainingResult if training ran, None otherwise
        """
        # Set baseline from production model
        production_version = self.version_manager.get_production_version()

        if production_version and production_version != "legacy":
            _, prod_metadata = self.version_manager.load_model_version(production_version)
            self.drift_detector.set_baseline_metrics(prod_metadata.metrics)

        # Detect drift
        drift_result = self.drift_detector.detect_performance_drift(current_metrics)

        if not drift_result.is_drifting:
            print(f"No drift detected: {drift_result.recommendation}")
            return None

        print(f"Drift detected ({drift_result.severity}): {drift_result.recommendation}")

        trigger = TrainingTrigger(
            trigger_type='drift',
            reason=f"Drift detected: {drift_result.severity}"
        )
        job_id = self.scheduler.schedule_training(trigger)

        result = self.train_with_validation()

        self.scheduler.record_training_complete(
            job_id, result.success, result.metrics
        )

        return result

    def get_training_status(self) -> Dict:
        """Get current training system status."""
        scheduler_status = self.scheduler.get_scheduler_status()
        production_version = self.version_manager.get_production_version()

        production_metrics = {}
        if production_version and production_version != "legacy":
            try:
                _, metadata = self.version_manager.load_model_version(production_version)
                production_metrics = metadata.metrics
            except Exception:
                pass

        versions = self.version_manager.list_versions()

        return {
            'scheduler': scheduler_status,
            'production_version': production_version,
            'production_metrics': production_metrics,
            'total_versions': len(versions),
            'recent_versions': versions[-5:] if versions else []
        }


def print_training_result(result: TrainingResult) -> None:
    """Print formatted training result."""
    print(f"\n{'='*60}")
    print("TRAINING RESULT")
    print(f"{'='*60}")
    print(f"Success: {result.success}")
    print(f"Version: {result.version}")
    print(f"Promoted: {result.promoted}")
    print(f"Training Time: {result.training_time_seconds:.1f}s")

    if result.error:
        print(f"Error: {result.error}")

    if result.metrics:
        print(f"\nMetrics:")
        for metric, value in result.metrics.items():
            print(f"  {metric}: {value:.4f}")

    if result.comparison:
        print(f"\nComparison:")
        for metric, values in result.comparison.get('metric_comparison', {}).items():
            diff = values.get('difference', 0)
            print(f"  {metric}: {diff:+.4f}")


def main():
    """CLI interface for continuous training."""
    import argparse

    parser = argparse.ArgumentParser(description="LoL Match Prediction Continuous Training")
    parser.add_argument('--train', action='store_true', help='Run training with validation')
    parser.add_argument('--scheduled', action='store_true', help='Run scheduled training check')
    parser.add_argument('--status', action='store_true', help='Show training status')
    parser.add_argument('--window', type=int, default=12, help='Rolling window months')
    parser.add_argument('--quick', action='store_true', help='Use quick mode')

    args = parser.parse_args()

    config = TrainingConfig(
        rolling_window_months=args.window,
        quick_mode=args.quick
    )
    trainer = ContinuousTrainer(config)

    if args.train:
        result = trainer.train_with_validation()
        print_training_result(result)
    elif args.scheduled:
        result = trainer.run_scheduled_training()
        if result:
            print_training_result(result)
    elif args.status:
        status = trainer.get_training_status()
        print("\nTraining System Status:")
        print(f"  Production Version: {status['production_version']}")
        print(f"  Total Versions: {status['total_versions']}")
        print(f"  Last Training: {status['scheduler'].get('last_training')}")
        print(f"  Next Scheduled: {status['scheduler'].get('next_scheduled')}")

        if status['production_metrics']:
            print(f"\nProduction Metrics:")
            for metric, value in status['production_metrics'].items():
                print(f"  {metric}: {value:.4f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
