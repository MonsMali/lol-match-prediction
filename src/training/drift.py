"""
Drift Detection for LoL Match Prediction System.

Detects performance drift and feature drift to trigger model retraining.

Enhanced with:
- Performance Decay Rate tracking
- ADWIN (Adaptive Windowing) concept drift detection
- Sliding window analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from scipy import stats
from collections import deque

from .config import DriftConfig, DEFAULT_CONFIG


@dataclass
class DriftResult:
    """Result of drift detection."""
    is_drifting: bool
    drift_type: str  # 'performance', 'feature', 'concept', 'none'
    severity: str    # 'low', 'medium', 'high', 'critical'
    details: Dict = field(default_factory=dict)
    recommendation: str = ""
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DecayRateResult:
    """Result of performance decay rate analysis."""
    decay_rate: float  # Slope of performance over time (negative = declining)
    decay_rate_per_month: float  # Normalized to monthly rate
    r_squared: float  # How well the linear fit explains the decay
    is_significant: bool  # Statistical significance of decay
    p_value: float
    projected_months_to_threshold: Optional[float]  # Time until performance drops below threshold
    trend: str  # 'stable', 'declining', 'improving'
    details: Dict = field(default_factory=dict)


@dataclass
class ADWINResult:
    """Result of ADWIN concept drift detection."""
    drift_detected: bool
    drift_points: List[int]  # Indices where drift was detected
    window_sizes: List[int]  # Window sizes at each point
    confidence: float
    details: Dict = field(default_factory=dict)


@dataclass
class DriftReport:
    """Comprehensive drift detection report."""
    performance_drift: Optional[DriftResult] = None
    feature_drift: Optional[DriftResult] = None
    concept_drift: Optional[ADWINResult] = None
    decay_analysis: Optional[DecayRateResult] = None
    overall_status: str = "OK"
    should_retrain: bool = False
    urgency: str = "low"  # 'low', 'medium', 'high', 'critical'
    reasons: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class DriftDetector:
    """
    Detects model drift through performance and feature monitoring.

    Types of drift detected:
    - Performance drift: Model metrics degrading over time
    - Feature drift: Input feature distributions changing
    """

    def __init__(self, baseline_metrics: Optional[Dict[str, float]] = None,
                 config: Optional[DriftConfig] = None):
        """
        Initialize drift detector.

        Args:
            baseline_metrics: Baseline performance metrics to compare against
            config: Drift detection configuration
        """
        self.config = config or DEFAULT_CONFIG.drift
        self.baseline_metrics = baseline_metrics or {}
        self.metric_history: List[Dict] = []
        self.feature_baseline: Optional[pd.DataFrame] = None

    def set_baseline_metrics(self, metrics: Dict[str, float]) -> None:
        """Set baseline metrics for comparison."""
        self.baseline_metrics = metrics.copy()

    def set_feature_baseline(self, df: pd.DataFrame) -> None:
        """Set baseline feature distributions."""
        self.feature_baseline = df.describe().to_dict()

    def detect_performance_drift(self, current_metrics: Dict[str, float]) -> DriftResult:
        """
        Detect performance drift by comparing current metrics to baseline.

        Args:
            current_metrics: Current model performance metrics

        Returns:
            DriftResult with drift detection outcome
        """
        if not self.baseline_metrics:
            return DriftResult(
                is_drifting=False,
                drift_type='none',
                severity='low',
                details={'error': 'No baseline metrics set'},
                recommendation='Set baseline metrics before drift detection'
            )

        # Track metric drops
        drops = {}
        critical_drops = []
        high_drops = []
        medium_drops = []

        # Check AUC (higher is better)
        if 'auc' in self.baseline_metrics and 'auc' in current_metrics:
            baseline = self.baseline_metrics['auc']
            current = current_metrics['auc']
            drop = baseline - current

            drops['auc'] = {
                'baseline': baseline,
                'current': current,
                'drop': drop,
                'threshold': self.config.auc_threshold
            }

            if drop >= self.config.auc_threshold * 2:
                critical_drops.append(('auc', drop))
            elif drop >= self.config.auc_threshold:
                high_drops.append(('auc', drop))
            elif drop >= self.config.auc_threshold * 0.5:
                medium_drops.append(('auc', drop))

        # Check F1 (higher is better)
        if 'f1' in self.baseline_metrics and 'f1' in current_metrics:
            baseline = self.baseline_metrics['f1']
            current = current_metrics['f1']
            drop = baseline - current

            drops['f1'] = {
                'baseline': baseline,
                'current': current,
                'drop': drop,
                'threshold': self.config.f1_threshold
            }

            if drop >= self.config.f1_threshold * 2:
                critical_drops.append(('f1', drop))
            elif drop >= self.config.f1_threshold:
                high_drops.append(('f1', drop))
            elif drop >= self.config.f1_threshold * 0.5:
                medium_drops.append(('f1', drop))

        # Check accuracy (higher is better)
        if 'accuracy' in self.baseline_metrics and 'accuracy' in current_metrics:
            baseline = self.baseline_metrics['accuracy']
            current = current_metrics['accuracy']
            drop = baseline - current

            drops['accuracy'] = {
                'baseline': baseline,
                'current': current,
                'drop': drop,
                'threshold': self.config.accuracy_threshold
            }

            if drop >= self.config.accuracy_threshold:
                high_drops.append(('accuracy', drop))

        # Check log loss (lower is better)
        if 'log_loss' in self.baseline_metrics and 'log_loss' in current_metrics:
            baseline = self.baseline_metrics['log_loss']
            current = current_metrics['log_loss']
            increase = current - baseline

            drops['log_loss'] = {
                'baseline': baseline,
                'current': current,
                'increase': increase,
                'threshold': self.config.log_loss_threshold
            }

            if increase >= self.config.log_loss_threshold:
                high_drops.append(('log_loss', increase))

        # Determine severity
        is_drifting = bool(critical_drops or high_drops)
        if critical_drops:
            severity = 'critical'
        elif high_drops:
            severity = 'high'
        elif medium_drops:
            severity = 'medium'
        else:
            severity = 'low'

        # Generate recommendation
        if critical_drops:
            recommendation = f"URGENT: Critical drift detected in {[d[0] for d in critical_drops]}. Immediate retraining required."
        elif high_drops:
            recommendation = f"Retrain recommended: Significant drift in {[d[0] for d in high_drops]}."
        elif medium_drops:
            recommendation = "Monitor closely. Minor performance degradation detected."
        else:
            recommendation = "Performance is within acceptable range."

        # Add to history
        self.metric_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': current_metrics.copy(),
            'is_drifting': is_drifting
        })

        return DriftResult(
            is_drifting=is_drifting,
            drift_type='performance' if is_drifting else 'none',
            severity=severity,
            details={
                'metric_drops': drops,
                'critical_drops': critical_drops,
                'high_drops': high_drops,
                'medium_drops': medium_drops
            },
            recommendation=recommendation
        )

    def detect_feature_drift(self, reference: pd.DataFrame,
                             current: pd.DataFrame,
                             columns: Optional[List[str]] = None) -> DriftResult:
        """
        Detect feature drift between reference and current data.

        Uses Population Stability Index (PSI) and Kolmogorov-Smirnov test.

        Args:
            reference: Reference/baseline data
            current: Current data to compare
            columns: Specific columns to check (None = all numeric)

        Returns:
            DriftResult with feature drift detection
        """
        if columns is None:
            # Use common numeric columns
            numeric_cols = reference.select_dtypes(include=[np.number]).columns
            common_cols = set(numeric_cols) & set(current.columns)
            columns = list(common_cols)

        if not columns:
            return DriftResult(
                is_drifting=False,
                drift_type='none',
                severity='low',
                details={'error': 'No common numeric columns found'},
                recommendation='Check data compatibility'
            )

        drift_details = {}
        drifted_features = []

        for col in columns:
            if col not in reference.columns or col not in current.columns:
                continue

            ref_data = reference[col].dropna()
            cur_data = current[col].dropna()

            if len(ref_data) < self.config.min_samples or len(cur_data) < self.config.min_samples:
                continue

            # Calculate PSI
            psi = self._calculate_psi(ref_data, cur_data)

            # KS Test
            ks_stat, ks_pvalue = stats.ks_2samp(ref_data, cur_data)

            drift_details[col] = {
                'psi': psi,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'ref_mean': float(ref_data.mean()),
                'cur_mean': float(cur_data.mean()),
                'ref_std': float(ref_data.std()),
                'cur_std': float(cur_data.std())
            }

            # Check thresholds
            if psi > self.config.psi_threshold or ks_stat > self.config.ks_threshold:
                drifted_features.append({
                    'feature': col,
                    'psi': psi,
                    'ks_stat': ks_stat
                })

        # Determine severity based on number and magnitude of drifted features
        n_drifted = len(drifted_features)
        n_total = len(columns)
        drift_ratio = n_drifted / n_total if n_total > 0 else 0

        is_drifting = n_drifted > 0

        if drift_ratio > 0.3:
            severity = 'critical'
        elif drift_ratio > 0.15:
            severity = 'high'
        elif drift_ratio > 0.05:
            severity = 'medium'
        else:
            severity = 'low'

        # Generate recommendation
        if severity == 'critical':
            recommendation = f"URGENT: {n_drifted}/{n_total} features show significant drift. Data pipeline review and retraining required."
        elif severity == 'high':
            recommendation = f"Retrain recommended: {n_drifted} features drifted significantly."
        elif severity == 'medium':
            recommendation = f"Monitor: {n_drifted} features showing minor drift."
        else:
            recommendation = "Feature distributions stable."

        return DriftResult(
            is_drifting=is_drifting,
            drift_type='feature' if is_drifting else 'none',
            severity=severity,
            details={
                'n_features_checked': n_total,
                'n_features_drifted': n_drifted,
                'drift_ratio': drift_ratio,
                'drifted_features': drifted_features,
                'feature_details': drift_details
            },
            recommendation=recommendation
        )

    def _calculate_psi(self, reference: pd.Series, current: pd.Series,
                       n_bins: int = 10) -> float:
        """
        Calculate Population Stability Index.

        PSI measures how much the distribution has shifted.
        PSI < 0.1: No significant change
        PSI 0.1-0.2: Moderate change
        PSI > 0.2: Significant change

        Args:
            reference: Reference distribution
            current: Current distribution
            n_bins: Number of bins

        Returns:
            PSI value
        """
        # Create bins from reference data
        _, bin_edges = np.histogram(reference, bins=n_bins)

        # Calculate proportions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Add small value to avoid division by zero
        ref_props = (ref_counts + 1) / (len(reference) + n_bins)
        cur_props = (cur_counts + 1) / (len(current) + n_bins)

        # Calculate PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))

        return float(psi)

    def calculate_decay_rate(self, metric_name: str = 'auc',
                            min_threshold: float = 0.70,
                            use_history: bool = True,
                            time_series: Optional[List[Tuple[datetime, float]]] = None) -> DecayRateResult:
        """
        Calculate the rate of performance decay over time.

        Uses linear regression to estimate performance trend and project
        when the model might fall below acceptable thresholds.

        Args:
            metric_name: Which metric to analyze ('auc', 'f1', 'accuracy')
            min_threshold: Minimum acceptable performance threshold
            use_history: Whether to use internal metric history
            time_series: Optional external time series data [(datetime, value), ...]

        Returns:
            DecayRateResult with decay analysis
        """
        # Gather data points
        if time_series is not None:
            dates = [t[0] for t in time_series]
            values = [t[1] for t in time_series]
        elif use_history and self.metric_history:
            dates = []
            values = []
            for entry in self.metric_history:
                if metric_name in entry.get('metrics', {}):
                    dates.append(datetime.fromisoformat(entry['timestamp']))
                    values.append(entry['metrics'][metric_name])
        else:
            return DecayRateResult(
                decay_rate=0.0,
                decay_rate_per_month=0.0,
                r_squared=0.0,
                is_significant=False,
                p_value=1.0,
                projected_months_to_threshold=None,
                trend='unknown',
                details={'error': 'Insufficient data for decay analysis'}
            )

        if len(values) < 3:
            return DecayRateResult(
                decay_rate=0.0,
                decay_rate_per_month=0.0,
                r_squared=0.0,
                is_significant=False,
                p_value=1.0,
                projected_months_to_threshold=None,
                trend='unknown',
                details={'error': f'Need at least 3 data points, got {len(values)}'}
            )

        # Convert dates to numeric (days since first observation)
        base_date = min(dates)
        days = np.array([(d - base_date).total_seconds() / 86400 for d in dates])
        values = np.array(values)

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(days, values)
        r_squared = r_value ** 2

        # Convert to monthly rate
        decay_rate_per_month = slope * 30  # days to months

        # Determine trend
        if p_value > 0.05:
            trend = 'stable'
        elif slope < -0.001:  # Significant negative slope
            trend = 'declining'
        elif slope > 0.001:  # Significant positive slope
            trend = 'improving'
        else:
            trend = 'stable'

        # Project time to threshold
        projected_months = None
        if slope < 0 and values[-1] > min_threshold:
            current_value = intercept + slope * days[-1]
            days_to_threshold = (min_threshold - current_value) / slope
            if days_to_threshold > 0:
                projected_months = days_to_threshold / 30

        return DecayRateResult(
            decay_rate=float(slope),
            decay_rate_per_month=float(decay_rate_per_month),
            r_squared=float(r_squared),
            is_significant=p_value < 0.05,
            p_value=float(p_value),
            projected_months_to_threshold=projected_months,
            trend=trend,
            details={
                'n_observations': len(values),
                'time_span_days': float(days[-1] - days[0]),
                'intercept': float(intercept),
                'std_error': float(std_err),
                'first_value': float(values[0]),
                'last_value': float(values[-1]),
                'min_value': float(values.min()),
                'max_value': float(values.max())
            }
        )

    def detect_concept_drift_adwin(self, data_stream: np.ndarray,
                                   delta: float = 0.002,
                                   min_window: int = 30) -> ADWINResult:
        """
        Detect concept drift using ADWIN (Adaptive Windowing) algorithm.

        ADWIN maintains a window of recent items and detects drift by
        finding optimal cut points where the mean of subwindows differs
        significantly.

        Particularly useful for LoL data where meta shifts can cause
        sudden changes in model performance.

        Args:
            data_stream: Stream of performance values (e.g., rolling accuracy)
            delta: Confidence parameter (smaller = more sensitive to drift)
            min_window: Minimum window size before checking for drift

        Returns:
            ADWINResult with drift detection results
        """
        n = len(data_stream)
        if n < min_window * 2:
            return ADWINResult(
                drift_detected=False,
                drift_points=[],
                window_sizes=[],
                confidence=1.0 - delta,
                details={'error': f'Need at least {min_window * 2} points for ADWIN'}
            )

        drift_points = []
        window_sizes = []
        window_start = 0

        for i in range(min_window, n):
            window = data_stream[window_start:i+1]
            window_size = len(window)

            # Check for optimal cut point
            drift_at = self._find_adwin_cut(window, delta)

            if drift_at is not None:
                # Drift detected, record point and shrink window
                absolute_drift_point = window_start + drift_at
                drift_points.append(absolute_drift_point)
                window_sizes.append(window_size)

                # Move window start past the drift point
                window_start = absolute_drift_point + 1

        return ADWINResult(
            drift_detected=len(drift_points) > 0,
            drift_points=drift_points,
            window_sizes=window_sizes,
            confidence=1.0 - delta,
            details={
                'n_drifts': len(drift_points),
                'total_observations': n,
                'final_window_size': n - window_start,
                'drift_rate': len(drift_points) / n if n > 0 else 0
            }
        )

    def _find_adwin_cut(self, window: np.ndarray, delta: float) -> Optional[int]:
        """
        Find optimal cut point in window using ADWIN criterion.

        Args:
            window: Current window of observations
            delta: Confidence parameter

        Returns:
            Cut point index if drift detected, None otherwise
        """
        n = len(window)
        if n < 10:  # Minimum subwindow size
            return None

        window_mean = window.mean()
        window_var = window.var() + 1e-10  # Avoid division by zero

        best_cut = None
        best_epsilon = 0

        # Check potential cut points
        for cut in range(5, n - 5):  # Leave at least 5 on each side
            w0 = window[:cut]
            w1 = window[cut:]

            n0, n1 = len(w0), len(w1)
            m0, m1 = w0.mean(), w1.mean()
            v0, v1 = w0.var() + 1e-10, w1.var() + 1e-10

            # Harmonic mean of sizes
            m = (n0 * n1) / (n0 + n1)

            # ADWIN bound calculation
            epsilon_cut = np.sqrt((2 / m) * window_var * np.log(2 / delta))
            epsilon_cut += (2 / (3 * m)) * np.log(2 / delta)

            # Check if means differ significantly
            mean_diff = abs(m0 - m1)
            if mean_diff > epsilon_cut and mean_diff > best_epsilon:
                best_cut = cut
                best_epsilon = mean_diff

        return best_cut

    def detect_sliding_window_drift(self, data: np.ndarray,
                                    window_size: int = 50,
                                    step_size: int = 10,
                                    threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect drift using sliding window comparison.

        Compares consecutive windows to detect gradual or sudden shifts.

        Args:
            data: Array of values to analyze
            window_size: Size of each window
            step_size: Step between windows
            threshold: Threshold for mean difference to flag drift

        Returns:
            Dictionary with sliding window analysis results
        """
        n = len(data)
        if n < window_size * 2:
            return {
                'drift_detected': False,
                'error': 'Insufficient data for sliding window analysis'
            }

        windows = []
        means = []
        stds = []
        drift_points = []

        for i in range(0, n - window_size + 1, step_size):
            window = data[i:i + window_size]
            windows.append((i, i + window_size))
            means.append(window.mean())
            stds.append(window.std())

        # Compare consecutive windows
        for i in range(1, len(means)):
            diff = abs(means[i] - means[i-1])
            pooled_std = np.sqrt((stds[i]**2 + stds[i-1]**2) / 2)

            # Normalized difference (effect size)
            if pooled_std > 0:
                effect_size = diff / pooled_std
            else:
                effect_size = 0

            if effect_size > threshold * 5 or diff > threshold:
                drift_points.append({
                    'window_index': i,
                    'position': windows[i][0],
                    'mean_before': means[i-1],
                    'mean_after': means[i],
                    'effect_size': effect_size
                })

        return {
            'drift_detected': len(drift_points) > 0,
            'n_drift_points': len(drift_points),
            'drift_points': drift_points,
            'window_means': means,
            'window_stds': stds,
            'overall_trend': 'declining' if len(means) > 1 and means[-1] < means[0] else 'stable'
        }

    def generate_drift_report(self, current_metrics: Optional[Dict[str, float]] = None,
                              reference_data: Optional[pd.DataFrame] = None,
                              current_data: Optional[pd.DataFrame] = None,
                              performance_stream: Optional[np.ndarray] = None,
                              include_decay_analysis: bool = True,
                              include_concept_drift: bool = True) -> DriftReport:
        """
        Generate comprehensive drift detection report.

        Args:
            current_metrics: Current performance metrics
            reference_data: Reference feature data
            current_data: Current feature data
            performance_stream: Optional stream of performance values for ADWIN
            include_decay_analysis: Whether to include decay rate analysis
            include_concept_drift: Whether to include ADWIN concept drift detection

        Returns:
            DriftReport with all drift detection results
        """
        report = DriftReport()
        report.reasons = []
        report.recommendations = []
        severity_scores = []  # Track severity for overall urgency

        # Check performance drift
        if current_metrics and self.baseline_metrics:
            perf_drift = self.detect_performance_drift(current_metrics)
            report.performance_drift = perf_drift

            if perf_drift.is_drifting:
                report.reasons.append(f"Performance drift: {perf_drift.severity}")
                report.recommendations.append(perf_drift.recommendation)
                severity_scores.append(
                    {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}.get(perf_drift.severity, 0)
                )

        # Check feature drift
        if reference_data is not None and current_data is not None:
            feat_drift = self.detect_feature_drift(reference_data, current_data)
            report.feature_drift = feat_drift

            if feat_drift.is_drifting:
                report.reasons.append(f"Feature drift: {feat_drift.severity}")
                report.recommendations.append(feat_drift.recommendation)
                severity_scores.append(
                    {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}.get(feat_drift.severity, 0)
                )

        # Check concept drift with ADWIN
        if include_concept_drift and performance_stream is not None:
            adwin_result = self.detect_concept_drift_adwin(performance_stream)
            report.concept_drift = adwin_result

            if adwin_result.drift_detected:
                n_drifts = len(adwin_result.drift_points)
                if n_drifts > 3:
                    severity_scores.append(4)
                    report.reasons.append(f"Concept drift: {n_drifts} drift points detected (critical)")
                    report.recommendations.append("Multiple concept drift points detected. Meta shift likely. Immediate retraining with recent data recommended.")
                elif n_drifts > 1:
                    severity_scores.append(3)
                    report.reasons.append(f"Concept drift: {n_drifts} drift points detected (high)")
                    report.recommendations.append("Concept drift detected. Consider retraining with weighted recent samples.")
                else:
                    severity_scores.append(2)
                    report.reasons.append(f"Concept drift: {n_drifts} drift point detected (medium)")
                    report.recommendations.append("Minor concept drift detected. Monitor performance closely.")

        # Perform decay rate analysis
        if include_decay_analysis and self.metric_history:
            decay_result = self.calculate_decay_rate()
            report.decay_analysis = decay_result

            if decay_result.trend == 'declining' and decay_result.is_significant:
                if decay_result.projected_months_to_threshold and decay_result.projected_months_to_threshold < 2:
                    severity_scores.append(4)
                    report.reasons.append(f"Performance decay: model projected to fall below threshold in {decay_result.projected_months_to_threshold:.1f} months")
                    report.recommendations.append("URGENT: Performance declining rapidly. Schedule retraining immediately.")
                elif decay_result.projected_months_to_threshold and decay_result.projected_months_to_threshold < 6:
                    severity_scores.append(3)
                    report.reasons.append(f"Performance decay: declining trend detected")
                    report.recommendations.append(f"Performance declining. Plan retraining within {decay_result.projected_months_to_threshold:.0f} months.")
                else:
                    severity_scores.append(2)
                    report.reasons.append("Performance decay: gradual decline detected")
                    report.recommendations.append("Gradual performance decline. Monitor and plan periodic retraining.")

        # Determine overall urgency
        if severity_scores:
            max_severity = max(severity_scores)
            report.urgency = {4: 'critical', 3: 'high', 2: 'medium', 1: 'low'}.get(max_severity, 'low')
        else:
            report.urgency = 'low'

        # Determine overall status
        if report.urgency in ['critical', 'high']:
            report.overall_status = 'DRIFT_DETECTED'
            report.should_retrain = True
        elif report.urgency == 'medium':
            report.overall_status = 'WARNING'
            report.should_retrain = False
        else:
            report.overall_status = 'OK'
            report.should_retrain = False

        # Add general recommendation if no specific issues
        if not report.recommendations:
            report.recommendations.append("Model performance and data distributions are stable. No action needed.")

        return report


def print_drift_report(report: DriftReport) -> None:
    """Print formatted drift report."""
    print(f"\n{'='*60}")
    print("DRIFT DETECTION REPORT")
    print(f"{'='*60}")
    print(f"Generated: {report.generated_at}")
    print(f"Overall Status: {report.overall_status}")
    print(f"Urgency: {report.urgency.upper()}")
    print(f"Should Retrain: {report.should_retrain}")

    if report.performance_drift:
        print(f"\nPerformance Drift:")
        print(f"  Is Drifting: {report.performance_drift.is_drifting}")
        print(f"  Severity: {report.performance_drift.severity}")
        print(f"  Recommendation: {report.performance_drift.recommendation}")

        if report.performance_drift.details.get('metric_drops'):
            print(f"  Metric Analysis:")
            for metric, values in report.performance_drift.details['metric_drops'].items():
                baseline = values.get('baseline', values.get('current', 0))
                current = values.get('current', 0)
                change = values.get('drop', values.get('increase', 0))
                print(f"    {metric}: {baseline:.4f} -> {current:.4f} (change: {change:+.4f})")

    if report.feature_drift:
        print(f"\nFeature Drift:")
        print(f"  Is Drifting: {report.feature_drift.is_drifting}")
        print(f"  Severity: {report.feature_drift.severity}")
        print(f"  Features Checked: {report.feature_drift.details.get('n_features_checked', 0)}")
        print(f"  Features Drifted: {report.feature_drift.details.get('n_features_drifted', 0)}")
        print(f"  Recommendation: {report.feature_drift.recommendation}")

    if report.concept_drift:
        print(f"\nConcept Drift (ADWIN):")
        print(f"  Drift Detected: {report.concept_drift.drift_detected}")
        print(f"  Drift Points: {len(report.concept_drift.drift_points)}")
        print(f"  Confidence: {report.concept_drift.confidence:.3f}")
        if report.concept_drift.drift_points:
            print(f"  Drift Locations: {report.concept_drift.drift_points[:5]}{'...' if len(report.concept_drift.drift_points) > 5 else ''}")

    if report.decay_analysis:
        print(f"\nPerformance Decay Analysis:")
        print(f"  Trend: {report.decay_analysis.trend}")
        print(f"  Decay Rate (per month): {report.decay_analysis.decay_rate_per_month:+.4f}")
        print(f"  R-squared: {report.decay_analysis.r_squared:.4f}")
        print(f"  Statistically Significant: {report.decay_analysis.is_significant}")
        if report.decay_analysis.projected_months_to_threshold:
            print(f"  Projected Months to Threshold: {report.decay_analysis.projected_months_to_threshold:.1f}")

    if report.reasons:
        print(f"\nReasons:")
        for reason in report.reasons:
            print(f"  - {reason}")

    if report.recommendations:
        print(f"\nRecommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")
