"""
Multi-Metric Evaluation System for LoL Match Prediction.

Provides comprehensive model evaluation beyond F1 Score, including
probability quality metrics essential for prediction confidence.

Enhanced with:
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- Balanced Accuracy
- Calibration Slope and Intercept
- Net Benefit (Decision Curve Analysis)
- Reliability Diagram Generation
- Probability Calibration (Platt Scaling, Isotonic Regression)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, log_loss, brier_score_loss,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import warnings


@dataclass
class CalibrationResult:
    """Results from probability calibration."""
    calibrated_proba: np.ndarray
    method: str
    calibration_slope: float
    calibration_intercept: float
    ece_before: float
    ece_after: float
    improvement: float


@dataclass
class DecisionCurveResult:
    """Results from decision curve analysis."""
    thresholds: np.ndarray
    net_benefit_model: np.ndarray
    net_benefit_all: np.ndarray
    net_benefit_none: np.ndarray


@dataclass
class UncertaintyResult:
    """Results from uncertainty quantification."""
    mean_prediction: np.ndarray
    std_prediction: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    confidence_level: float


class MultiMetricEvaluator:
    """
    Comprehensive model evaluation with multiple metrics and composite scoring.

    Metrics included:
    - AUC-ROC: Discrimination ability (30% default weight)
    - Log Loss: Probability accuracy (25% default weight)
    - Brier Score: Calibration quality (20% default weight)
    - ECE: Expected Calibration Error (15% default weight)
    - F1 Score: Classification performance (10% default weight)

    Enhanced metrics:
    - Matthews Correlation Coefficient (MCC)
    - Cohen's Kappa
    - Balanced Accuracy
    - Calibration Slope
    - Maximum Calibration Error (MCE)
    """

    DEFAULT_WEIGHTS = {
        'auc': 0.30,
        'log_loss': 0.25,
        'brier': 0.20,
        'ece': 0.15,
        'f1': 0.10
    }

    ENHANCED_WEIGHTS = {
        'auc': 0.25,
        'log_loss': 0.20,
        'brier': 0.15,
        'ece': 0.10,
        'f1': 0.10,
        'mcc': 0.10,
        'calibration_slope': 0.10
    }

    def __init__(self, metric_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the evaluator with optional custom metric weights.

        Args:
            metric_weights: Dictionary of metric weights. Must sum to 1.0.
                           Keys: 'auc', 'log_loss', 'brier', 'ece', 'f1'
        """
        if metric_weights is None:
            self.weights = self.DEFAULT_WEIGHTS.copy()
        else:
            # Validate weights
            if abs(sum(metric_weights.values()) - 1.0) > 1e-6:
                raise ValueError("Metric weights must sum to 1.0")
            self.weights = metric_weights

    @staticmethod
    def calculate_ece(y_true: np.ndarray, y_proba: np.ndarray,
                      n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        ECE measures the difference between predicted probabilities and
        actual outcomes across probability bins.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities for positive class
            n_bins: Number of bins for calibration

        Returns:
            ECE value (lower is better, 0 is perfect calibration)
        """
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            # Find samples in this bin
            bin_mask = (y_proba > bin_boundaries[i]) & (y_proba <= bin_boundaries[i + 1])
            bin_size = np.sum(bin_mask)

            if bin_size > 0:
                # Calculate accuracy and confidence for this bin
                bin_accuracy = np.mean(y_true[bin_mask])
                bin_confidence = np.mean(y_proba[bin_mask])

                # Weight by bin size
                ece += (bin_size / len(y_true)) * abs(bin_accuracy - bin_confidence)

        return ece

    @staticmethod
    def calculate_mce(y_true: np.ndarray, y_proba: np.ndarray,
                      n_bins: int = 10) -> float:
        """
        Calculate Maximum Calibration Error (MCE).

        MCE is the maximum difference between predicted probability and
        actual accuracy across all bins.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            n_bins: Number of bins

        Returns:
            MCE value (lower is better)
        """
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mce = 0.0

        for i in range(n_bins):
            bin_mask = (y_proba > bin_boundaries[i]) & (y_proba <= bin_boundaries[i + 1])
            bin_size = np.sum(bin_mask)

            if bin_size > 0:
                bin_accuracy = np.mean(y_true[bin_mask])
                bin_confidence = np.mean(y_proba[bin_mask])
                mce = max(mce, abs(bin_accuracy - bin_confidence))

        return mce

    @staticmethod
    def calculate_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Matthews Correlation Coefficient (MCC).

        MCC is a balanced measure that can be used even if classes are
        imbalanced. It returns a value between -1 and +1.

        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels

        Returns:
            MCC value (-1 to +1, higher is better)
        """
        return matthews_corrcoef(y_true, y_pred)

    @staticmethod
    def calculate_cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Cohen's Kappa coefficient.

        Measures agreement between predictions and actual values,
        accounting for agreement by chance.

        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels

        Returns:
            Kappa value (0 = chance agreement, 1 = perfect agreement)
        """
        return cohen_kappa_score(y_true, y_pred)

    @staticmethod
    def calculate_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Balanced Accuracy.

        The average of recall for each class, useful for imbalanced datasets.

        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels

        Returns:
            Balanced accuracy (0 to 1)
        """
        return balanced_accuracy_score(y_true, y_pred)

    @staticmethod
    def calculate_calibration_slope(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
        """
        Calculate calibration slope and intercept using logistic regression.

        A slope of 1 and intercept of 0 indicates perfect calibration.
        Slope > 1 indicates underconfidence, slope < 1 indicates overconfidence.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities

        Returns:
            Tuple of (slope, intercept)
        """
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        # Clip probabilities to avoid log(0)
        y_proba_clipped = np.clip(y_proba, 1e-10, 1 - 1e-10)

        # Convert to log-odds
        log_odds = np.log(y_proba_clipped / (1 - y_proba_clipped))

        # Fit logistic regression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr = LogisticRegression(solver='lbfgs', max_iter=1000)
            lr.fit(log_odds.reshape(-1, 1), y_true)

        slope = lr.coef_[0][0]
        intercept = lr.intercept_[0]

        return slope, intercept

    @staticmethod
    def calculate_net_benefit(y_true: np.ndarray, y_proba: np.ndarray,
                              thresholds: Optional[np.ndarray] = None) -> DecisionCurveResult:
        """
        Calculate Net Benefit for Decision Curve Analysis.

        Net benefit quantifies the clinical/practical utility of a model
        at different decision thresholds.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            thresholds: Probability thresholds to evaluate (default: 0.01 to 0.99)

        Returns:
            DecisionCurveResult with net benefit curves
        """
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 99)

        n = len(y_true)
        prevalence = np.mean(y_true)

        net_benefit_model = []
        net_benefit_all = []
        net_benefit_none = []

        for threshold in thresholds:
            # Model net benefit
            y_pred = (y_proba >= threshold).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))

            nb_model = (tp / n) - (fp / n) * (threshold / (1 - threshold))
            net_benefit_model.append(nb_model)

            # Treat all net benefit
            nb_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
            net_benefit_all.append(nb_all)

            # Treat none net benefit (always 0)
            net_benefit_none.append(0)

        return DecisionCurveResult(
            thresholds=thresholds,
            net_benefit_model=np.array(net_benefit_model),
            net_benefit_all=np.array(net_benefit_all),
            net_benefit_none=np.array(net_benefit_none)
        )

    @staticmethod
    def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate specificity (true negative rate).

        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels

        Returns:
            Specificity value (0 to 1)
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    @staticmethod
    def calculate_sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate sensitivity (true positive rate / recall).

        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels

        Returns:
            Sensitivity value (0 to 1)
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_proba: np.ndarray,
                              include_enhanced: bool = True) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.

        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_proba: Predicted probabilities for positive class
            include_enhanced: Whether to include enhanced metrics (MCC, Kappa, etc.)

        Returns:
            Dictionary containing all metrics
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_proba = np.asarray(y_proba)

        metrics = {
            # Classification metrics
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),

            # Probability quality metrics
            'auc': roc_auc_score(y_true, y_proba),
            'log_loss': log_loss(y_true, y_proba),
            'brier': brier_score_loss(y_true, y_proba),

            # Calibration metrics
            'ece': self.calculate_ece(y_true, y_proba),
            'mce': self.calculate_mce(y_true, y_proba)
        }

        if include_enhanced:
            # Enhanced classification metrics
            metrics['mcc'] = self.calculate_mcc(y_true, y_pred)
            metrics['cohens_kappa'] = self.calculate_cohens_kappa(y_true, y_pred)
            metrics['balanced_accuracy'] = self.calculate_balanced_accuracy(y_true, y_pred)
            metrics['sensitivity'] = self.calculate_sensitivity(y_true, y_pred)
            metrics['specificity'] = self.calculate_specificity(y_true, y_pred)

            # Enhanced calibration metrics
            try:
                slope, intercept = self.calculate_calibration_slope(y_true, y_proba)
                metrics['calibration_slope'] = slope
                metrics['calibration_intercept'] = intercept
            except Exception:
                metrics['calibration_slope'] = 1.0
                metrics['calibration_intercept'] = 0.0

        return metrics

    def calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate weighted composite score for model ranking.

        Higher is better. Metrics where lower is better (log_loss, brier, ece)
        are inverted using (1 - normalized_value).

        Args:
            metrics: Dictionary of metric values

        Returns:
            Composite score between 0 and 1
        """
        # Normalize metrics to 0-1 range where higher is always better
        normalized = {}

        # AUC: already 0-1, higher is better
        normalized['auc'] = metrics['auc']

        # F1: already 0-1, higher is better
        normalized['f1'] = metrics['f1']

        # Log Loss: typically 0-1+ range, lower is better
        # Use sigmoid-like transformation to bound it
        normalized['log_loss'] = 1 / (1 + metrics['log_loss'])

        # Brier Score: 0-1 range, lower is better
        normalized['brier'] = 1 - metrics['brier']

        # ECE: 0-1 range, lower is better
        normalized['ece'] = 1 - metrics['ece']

        # Calculate weighted sum
        composite = sum(
            self.weights[metric] * normalized[metric]
            for metric in self.weights.keys()
        )

        return composite

    def rank_models(self, model_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Rank models by composite score.

        Args:
            model_results: Dictionary mapping model names to their metrics

        Returns:
            DataFrame with models ranked by composite score
        """
        rankings = []

        for model_name, metrics in model_results.items():
            composite = self.calculate_composite_score(metrics)

            rankings.append({
                'Model': model_name,
                'Composite': composite,
                'AUC': metrics.get('auc', 0),
                'Log Loss': metrics.get('log_loss', 0),
                'Brier': metrics.get('brier', 0),
                'ECE': metrics.get('ece', 0),
                'F1': metrics.get('f1', 0),
                'Accuracy': metrics.get('accuracy', 0)
            })

        df = pd.DataFrame(rankings)
        df = df.sort_values('Composite', ascending=False).reset_index(drop=True)
        df.insert(0, 'Rank', range(1, len(df) + 1))

        return df

    def get_calibration_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                              n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate calibration curve data for plotting.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            n_bins: Number of bins

        Returns:
            Tuple of (mean_predicted_value, fraction_of_positives, bin_counts)
        """
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mean_predicted = []
        fraction_positives = []
        bin_counts = []

        for i in range(n_bins):
            bin_mask = (y_proba > bin_boundaries[i]) & (y_proba <= bin_boundaries[i + 1])
            bin_size = np.sum(bin_mask)

            if bin_size > 0:
                mean_predicted.append(np.mean(y_proba[bin_mask]))
                fraction_positives.append(np.mean(y_true[bin_mask]))
                bin_counts.append(bin_size)

        return (
            np.array(mean_predicted),
            np.array(fraction_positives),
            np.array(bin_counts)
        )

    def print_evaluation_report(self, metrics: Dict[str, float],
                                model_name: str = "Model") -> None:
        """
        Print a formatted evaluation report.

        Args:
            metrics: Dictionary of metric values
            model_name: Name of the model being evaluated
        """
        composite = self.calculate_composite_score(metrics)

        print(f"\n{'='*60}")
        print(f"EVALUATION REPORT: {model_name}")
        print(f"{'='*60}")

        print(f"\nCOMPOSITE SCORE: {composite:.4f}")

        print(f"\nClassification Metrics:")
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall:    {metrics.get('recall', 0):.4f}")
        print(f"  F1 Score:  {metrics.get('f1', 0):.4f}")

        print(f"\nProbability Quality Metrics:")
        print(f"  AUC-ROC:   {metrics.get('auc', 0):.4f}")
        print(f"  Log Loss:  {metrics.get('log_loss', 0):.4f}")
        print(f"  Brier:     {metrics.get('brier', 0):.4f}")

        print(f"\nCalibration Metrics:")
        print(f"  ECE:       {metrics.get('ece', 0):.4f}")
        print(f"  MCE:       {metrics.get('mce', 0):.4f}")

        print(f"\nMetric Weights Used:")
        for metric, weight in self.weights.items():
            print(f"  {metric}: {weight:.0%}")


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray,
                   metric_weights: Optional[Dict[str, float]] = None,
                   include_enhanced: bool = True) -> Dict[str, float]:
    """
    Convenience function to evaluate a model with all metrics.

    Args:
        model: Trained model with predict and predict_proba methods
        X: Feature matrix
        y: True labels
        metric_weights: Optional custom metric weights
        include_enhanced: Whether to include enhanced metrics

    Returns:
        Dictionary of all metrics including composite score
    """
    evaluator = MultiMetricEvaluator(metric_weights)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = evaluator.calculate_all_metrics(y, y_pred, y_proba, include_enhanced)
    metrics['composite'] = evaluator.calculate_composite_score(metrics)

    return metrics


class ProbabilityCalibrator:
    """
    Probability calibration using Platt Scaling or Isotonic Regression.

    Improves the reliability of predicted probabilities without changing
    the discrimination (AUC) of the model.
    """

    def __init__(self, method: str = 'isotonic'):
        """
        Initialize the calibrator.

        Args:
            method: Calibration method - 'platt' (sigmoid) or 'isotonic'
        """
        if method not in ['platt', 'isotonic', 'sigmoid']:
            raise ValueError("Method must be 'platt', 'sigmoid', or 'isotonic'")
        self.method = 'sigmoid' if method == 'platt' else method
        self.calibrator = None
        self.is_fitted = False

    def fit(self, y_true: np.ndarray, y_proba: np.ndarray) -> 'ProbabilityCalibrator':
        """
        Fit the calibrator to training data.

        Args:
            y_true: True binary labels
            y_proba: Uncalibrated predicted probabilities

        Returns:
            Self for chaining
        """
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        if self.method == 'sigmoid':
            # Platt scaling using logistic regression
            y_proba_clipped = np.clip(y_proba, 1e-10, 1 - 1e-10)
            log_odds = np.log(y_proba_clipped / (1 - y_proba_clipped))
            self.calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
            self.calibrator.fit(log_odds.reshape(-1, 1), y_true)
        else:
            # Isotonic regression
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_proba, y_true)

        self.is_fitted = True
        return self

    def calibrate(self, y_proba: np.ndarray) -> np.ndarray:
        """
        Calibrate probabilities using the fitted calibrator.

        Args:
            y_proba: Uncalibrated predicted probabilities

        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before calibrating")

        y_proba = np.asarray(y_proba)

        if self.method == 'sigmoid':
            y_proba_clipped = np.clip(y_proba, 1e-10, 1 - 1e-10)
            log_odds = np.log(y_proba_clipped / (1 - y_proba_clipped))
            return self.calibrator.predict_proba(log_odds.reshape(-1, 1))[:, 1]
        else:
            return self.calibrator.predict(y_proba)

    def fit_calibrate(self, y_true: np.ndarray, y_proba: np.ndarray) -> CalibrationResult:
        """
        Fit calibrator and return calibrated probabilities with metrics.

        Args:
            y_true: True binary labels
            y_proba: Uncalibrated predicted probabilities

        Returns:
            CalibrationResult with calibrated probabilities and metrics
        """
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        # Calculate ECE before calibration
        ece_before = MultiMetricEvaluator.calculate_ece(y_true, y_proba)

        # Fit and calibrate
        self.fit(y_true, y_proba)
        calibrated_proba = self.calibrate(y_proba)

        # Calculate ECE after calibration
        ece_after = MultiMetricEvaluator.calculate_ece(y_true, calibrated_proba)

        # Calculate calibration slope
        try:
            slope, intercept = MultiMetricEvaluator.calculate_calibration_slope(
                y_true, calibrated_proba
            )
        except Exception:
            slope, intercept = 1.0, 0.0

        return CalibrationResult(
            calibrated_proba=calibrated_proba,
            method=self.method,
            calibration_slope=slope,
            calibration_intercept=intercept,
            ece_before=ece_before,
            ece_after=ece_after,
            improvement=ece_before - ece_after
        )


class UncertaintyQuantifier:
    """
    Uncertainty quantification for model predictions using bootstrap.

    Provides confidence intervals and prediction uncertainty estimates.
    """

    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95,
                 random_state: int = 42):
        """
        Initialize the uncertainty quantifier.

        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)
            random_state: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state

    def quantify_prediction_uncertainty(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        clone_model_fn: Optional[callable] = None
    ) -> UncertaintyResult:
        """
        Quantify prediction uncertainty using bootstrap resampling.

        Args:
            model: Base model to use as template
            X_train: Training features
            y_train: Training labels
            X_test: Test features to predict on
            clone_model_fn: Function to clone model (default: sklearn clone)

        Returns:
            UncertaintyResult with confidence intervals
        """
        from sklearn.base import clone

        np.random.seed(self.random_state)

        predictions = []
        n_samples = len(X_train)

        for i in range(self.n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X_train[indices] if isinstance(X_train, np.ndarray) else X_train.iloc[indices]
            y_boot = y_train[indices] if isinstance(y_train, np.ndarray) else y_train.iloc[indices]

            # Clone and fit model
            if clone_model_fn:
                boot_model = clone_model_fn(model)
            else:
                boot_model = clone(model)

            try:
                boot_model.fit(X_boot, y_boot)
                pred = boot_model.predict_proba(X_test)[:, 1]
                predictions.append(pred)
            except Exception:
                continue

        if len(predictions) == 0:
            raise RuntimeError("All bootstrap iterations failed")

        predictions = np.array(predictions)

        # Calculate statistics
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(predictions, alpha / 2 * 100, axis=0)
        ci_upper = np.percentile(predictions, (1 - alpha / 2) * 100, axis=0)

        return UncertaintyResult(
            mean_prediction=np.mean(predictions, axis=0),
            std_prediction=np.std(predictions, axis=0),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level
        )

    def quantify_metric_uncertainty(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric_fn: callable
    ) -> Dict[str, float]:
        """
        Quantify uncertainty in a metric using bootstrap.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            metric_fn: Function to calculate metric (takes y_true, y_proba)

        Returns:
            Dictionary with mean, std, ci_lower, ci_upper
        """
        np.random.seed(self.random_state)

        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        metrics = []
        n_samples = len(y_true)

        for _ in range(self.n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            try:
                metric_value = metric_fn(y_true[indices], y_proba[indices])
                metrics.append(metric_value)
            except Exception:
                continue

        metrics = np.array(metrics)
        alpha = 1 - self.confidence_level

        return {
            'mean': np.mean(metrics),
            'std': np.std(metrics),
            'ci_lower': np.percentile(metrics, alpha / 2 * 100),
            'ci_upper': np.percentile(metrics, (1 - alpha / 2) * 100)
        }


class ModelComparator:
    """
    Statistical comparison of multiple models.

    Provides significance testing and ranking with confidence intervals.
    """

    def __init__(self, evaluator: Optional[MultiMetricEvaluator] = None):
        """
        Initialize the comparator.

        Args:
            evaluator: MultiMetricEvaluator instance (creates default if None)
        """
        self.evaluator = evaluator or MultiMetricEvaluator()

    def compare_models(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int = 1000
    ) -> pd.DataFrame:
        """
        Compare multiple models with bootstrap confidence intervals.

        Args:
            models: Dictionary mapping model names to fitted models
            X: Feature matrix
            y: True labels
            n_bootstrap: Number of bootstrap samples for CI

        Returns:
            DataFrame with metrics and confidence intervals for each model
        """
        quantifier = UncertaintyQuantifier(n_bootstrap=n_bootstrap)
        results = []

        for name, model in models.items():
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]

            # Calculate point estimates
            metrics = self.evaluator.calculate_all_metrics(y, y_pred, y_proba)

            # Calculate AUC confidence interval
            auc_ci = quantifier.quantify_metric_uncertainty(
                y, y_proba, roc_auc_score
            )

            results.append({
                'Model': name,
                'AUC': metrics['auc'],
                'AUC_CI_Lower': auc_ci['ci_lower'],
                'AUC_CI_Upper': auc_ci['ci_upper'],
                'F1': metrics['f1'],
                'MCC': metrics.get('mcc', 0),
                'ECE': metrics['ece'],
                'Brier': metrics['brier'],
                'Composite': self.evaluator.calculate_composite_score(metrics)
            })

        df = pd.DataFrame(results)
        df = df.sort_values('AUC', ascending=False).reset_index(drop=True)
        df.insert(0, 'Rank', range(1, len(df) + 1))

        return df

    @staticmethod
    def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray,
                     y_pred_b: np.ndarray) -> Dict[str, float]:
        """
        Perform McNemar's test to compare two classifiers.

        Tests whether the two classifiers have significantly different
        error rates.

        Args:
            y_true: True labels
            y_pred_a: Predictions from model A
            y_pred_b: Predictions from model B

        Returns:
            Dictionary with test statistic and p-value
        """
        # Build contingency table
        correct_a = (y_pred_a == y_true)
        correct_b = (y_pred_b == y_true)

        # b = A wrong, B right; c = A right, B wrong
        b = np.sum(~correct_a & correct_b)
        c = np.sum(correct_a & ~correct_b)

        # McNemar's test with continuity correction
        if b + c == 0:
            return {'statistic': 0.0, 'p_value': 1.0}

        statistic = (abs(b - c) - 1) ** 2 / (b + c)

        # Chi-squared distribution with 1 df
        from scipy import stats
        p_value = 1 - stats.chi2.cdf(statistic, df=1)

        return {'statistic': statistic, 'p_value': p_value}


def generate_reliability_diagram_data(y_true: np.ndarray, y_proba: np.ndarray,
                                      n_bins: int = 10) -> Dict[str, np.ndarray]:
    """
    Generate data for reliability diagram visualization.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        n_bins: Number of bins

    Returns:
        Dictionary with bin_centers, accuracy, confidence, counts
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # Use sklearn's calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

    # Calculate bin counts
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    counts = []
    for i in range(n_bins):
        mask = (y_proba > bin_boundaries[i]) & (y_proba <= bin_boundaries[i + 1])
        counts.append(np.sum(mask))

    return {
        'bin_centers': prob_pred,
        'accuracy': prob_true,
        'confidence': prob_pred,
        'counts': np.array(counts),
        'perfect_calibration': np.linspace(0, 1, 100)
    }


def create_evaluation_report(model: Any, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray,
                             model_name: str = "Model",
                             include_calibration: bool = True,
                             include_uncertainty: bool = False) -> Dict[str, Any]:
    """
    Create a comprehensive evaluation report for a model.

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        include_calibration: Whether to include calibration analysis
        include_uncertainty: Whether to include uncertainty quantification

    Returns:
        Dictionary with comprehensive evaluation results
    """
    evaluator = MultiMetricEvaluator()

    # Get predictions
    y_pred_train = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    train_metrics = evaluator.calculate_all_metrics(y_train, y_pred_train, y_proba_train)
    test_metrics = evaluator.calculate_all_metrics(y_test, y_pred_test, y_proba_test)

    report = {
        'model_name': model_name,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'train_composite': evaluator.calculate_composite_score(train_metrics),
        'test_composite': evaluator.calculate_composite_score(test_metrics),
        'generalization_gap': {
            'auc': train_metrics['auc'] - test_metrics['auc'],
            'f1': train_metrics['f1'] - test_metrics['f1'],
            'mcc': train_metrics.get('mcc', 0) - test_metrics.get('mcc', 0)
        }
    }

    # Add calibration analysis
    if include_calibration:
        calibrator = ProbabilityCalibrator(method='isotonic')
        calibration_result = calibrator.fit_calibrate(y_test, y_proba_test)
        report['calibration'] = {
            'ece_before': calibration_result.ece_before,
            'ece_after': calibration_result.ece_after,
            'improvement': calibration_result.improvement,
            'slope': calibration_result.calibration_slope,
            'intercept': calibration_result.calibration_intercept
        }

    # Add reliability diagram data
    report['reliability_diagram'] = generate_reliability_diagram_data(y_test, y_proba_test)

    # Add decision curve data
    report['decision_curve'] = MultiMetricEvaluator.calculate_net_benefit(y_test, y_proba_test)

    # Add uncertainty quantification
    if include_uncertainty:
        quantifier = UncertaintyQuantifier(n_bootstrap=500)
        auc_ci = quantifier.quantify_metric_uncertainty(
            y_test, y_proba_test, roc_auc_score
        )
        report['uncertainty'] = {
            'auc_mean': auc_ci['mean'],
            'auc_std': auc_ci['std'],
            'auc_ci_lower': auc_ci['ci_lower'],
            'auc_ci_upper': auc_ci['ci_upper']
        }

    return report
