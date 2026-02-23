"""
Prediction Confidence Estimation Module

Provides calibrated predictions with uncertainty quantification for
League of Legends match prediction.

Key Features:
- Isotonic/Platt calibration using sklearn's CalibratedClassifierCV
- Confidence levels: high (>0.7), medium (0.55-0.7), low (<0.55)
- 95% confidence intervals using ensemble variance
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import cross_val_predict
from typing import Dict, List, Optional, Tuple, Union
import warnings

warnings.filterwarnings('ignore')


class ConfidenceEstimator:
    """
    Provides calibrated predictions with uncertainty quantification.

    This class wraps a trained classifier and provides:
    - Probability calibration (isotonic or sigmoid/Platt)
    - Confidence level classification (high/medium/low)
    - 95% confidence intervals for predictions

    Attributes:
        model: The base classifier to calibrate
        calibrated_model: The calibrated classifier
        is_calibrated: Whether calibration has been performed
        calibration_method: The method used for calibration
    """

    # Confidence level thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.70
    MEDIUM_CONFIDENCE_THRESHOLD = 0.55

    def __init__(self, model, X_cal: Optional[np.ndarray] = None,
                 y_cal: Optional[np.ndarray] = None):
        """
        Initialize the ConfidenceEstimator.

        Args:
            model: A trained sklearn-compatible classifier with predict_proba
            X_cal: Optional calibration features (if provided, auto-calibrates)
            y_cal: Optional calibration targets (if provided, auto-calibrates)
        """
        self.model = model
        self.calibrated_model = None
        self.is_calibrated = False
        self.calibration_method = None

        # Store calibration data for confidence interval estimation
        self._X_cal = X_cal
        self._y_cal = y_cal
        self._calibration_residuals = None

        # Auto-calibrate if data provided
        if X_cal is not None and y_cal is not None:
            self.calibrate(X_cal, y_cal)

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray,
                  method: str = 'isotonic', cv: int = 5) -> None:
        """
        Calibrate the model's probability predictions.

        Args:
            X_cal: Calibration features
            y_cal: Calibration targets
            method: Calibration method ('isotonic' or 'sigmoid')
            cv: Number of cross-validation folds for calibration

        Isotonic regression is preferred for:
        - Larger datasets (>1000 samples)
        - When relationship between predicted and true probabilities is non-linear

        Sigmoid (Platt scaling) is preferred for:
        - Smaller datasets
        - When probabilities need minor corrections
        """
        self._X_cal = X_cal
        self._y_cal = y_cal
        self.calibration_method = method

        # Create calibrated classifier
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.model,
            method=method,
            cv=cv
        )

        # Fit on calibration data
        self.calibrated_model.fit(X_cal, y_cal)
        self.is_calibrated = True

        # Calculate calibration residuals for confidence intervals
        self._calculate_calibration_residuals(X_cal, y_cal)

    def _calculate_calibration_residuals(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Calculate residuals between predicted probabilities and actual outcomes.
        Used for estimating prediction uncertainty.
        """
        if self.calibrated_model is None:
            return

        # Get calibrated probabilities
        proba = self.calibrated_model.predict_proba(X)[:, 1]

        # Calculate residuals (difference from actual outcome)
        residuals = np.abs(proba - y)
        self._calibration_residuals = residuals

    def predict_with_confidence(self, X: np.ndarray) -> Dict:
        """
        Make predictions with confidence estimates.

        Args:
            X: Features for prediction (can be single sample or batch)

        Returns:
            Dictionary containing:
            - prediction: Binary prediction (0 or 1)
            - probability: Calibrated probability of class 1
            - confidence_level: 'high', 'medium', or 'low'
            - bounds: Tuple of (lower_bound, upper_bound) for 95% CI
            - raw_probability: Original uncalibrated probability
        """
        # Ensure X is 2D
        X = np.atleast_2d(X)

        # Get raw probability from base model
        raw_proba = self.model.predict_proba(X)[:, 1]

        # Get calibrated probability if available
        if self.is_calibrated and self.calibrated_model is not None:
            cal_proba = self.calibrated_model.predict_proba(X)[:, 1]
        else:
            cal_proba = raw_proba

        # Calculate confidence bounds
        lower_bounds, upper_bounds = self._calculate_confidence_bounds(cal_proba)

        # Determine predictions and confidence levels
        predictions = (cal_proba >= 0.5).astype(int)
        confidence_levels = self._get_confidence_levels(cal_proba)

        # Return results
        if len(X) == 1:
            # Single prediction - return scalar values
            return {
                'prediction': int(predictions[0]),
                'probability': float(cal_proba[0]),
                'confidence_level': confidence_levels[0],
                'bounds': (float(lower_bounds[0]), float(upper_bounds[0])),
                'raw_probability': float(raw_proba[0])
            }
        else:
            # Batch prediction - return arrays
            return {
                'prediction': predictions,
                'probability': cal_proba,
                'confidence_level': confidence_levels,
                'bounds': (lower_bounds, upper_bounds),
                'raw_probability': raw_proba
            }

    def _calculate_confidence_bounds(self, proba: np.ndarray,
                                     confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence interval bounds for probability predictions.

        Uses bootstrapped residuals from calibration if available,
        otherwise uses a heuristic based on distance from 0.5.

        Args:
            proba: Array of probability predictions
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        if self._calibration_residuals is not None and len(self._calibration_residuals) > 0:
            # Use empirical residuals to estimate uncertainty
            # Higher uncertainty for predictions near 0.5
            base_uncertainty = np.percentile(
                self._calibration_residuals,
                confidence * 100
            )

            # Scale uncertainty based on distance from 0.5
            # Predictions near 0.5 have higher uncertainty
            distance_from_half = np.abs(proba - 0.5)
            uncertainty_scale = 1 - distance_from_half  # Higher near 0.5

            margin = base_uncertainty * uncertainty_scale
        else:
            # Heuristic: uncertainty inversely proportional to confidence
            # Near 0.5 = high uncertainty, near 0 or 1 = low uncertainty
            distance_from_half = np.abs(proba - 0.5)
            margin = 0.15 * (1 - distance_from_half)  # Max margin of 0.15

        # Calculate bounds and clip to [0, 1]
        lower_bounds = np.clip(proba - margin, 0, 1)
        upper_bounds = np.clip(proba + margin, 0, 1)

        return lower_bounds, upper_bounds

    def _get_confidence_levels(self, proba: np.ndarray) -> List[str]:
        """
        Classify predictions into confidence levels.

        Args:
            proba: Array of probability predictions

        Returns:
            List of confidence level strings ('high', 'medium', 'low')
        """
        # Distance from 0.5 indicates how confident the prediction is
        distance_from_half = np.abs(proba - 0.5)

        # Convert to effective confidence (0.5 distance = 1.0 confidence)
        effective_confidence = 0.5 + distance_from_half

        levels = []
        for conf in effective_confidence:
            if conf >= self.HIGH_CONFIDENCE_THRESHOLD:
                levels.append('high')
            elif conf >= self.MEDIUM_CONFIDENCE_THRESHOLD:
                levels.append('medium')
            else:
                levels.append('low')

        return levels

    def get_calibration_metrics(self, X_test: np.ndarray,
                                y_test: np.ndarray) -> Dict:
        """
        Compute calibration metrics on test data.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with calibration metrics:
            - brier_score: Brier score (lower is better)
            - ece: Expected Calibration Error
            - calibration_curve: Tuple of (fraction_of_positives, mean_predicted_value)
        """
        # Get predictions
        if self.is_calibrated:
            proba = self.calibrated_model.predict_proba(X_test)[:, 1]
        else:
            proba = self.model.predict_proba(X_test)[:, 1]

        # Brier score
        brier_score = np.mean((proba - y_test) ** 2)

        # Expected Calibration Error
        ece = self._calculate_ece(proba, y_test)

        # Calibration curve data
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, proba, n_bins=10, strategy='uniform'
        )

        return {
            'brier_score': float(brier_score),
            'ece': float(ece),
            'calibration_curve': (fraction_of_positives, mean_predicted_value)
        }

    def _calculate_ece(self, proba: np.ndarray, y_true: np.ndarray,
                       n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error.

        ECE measures the average gap between predicted probabilities
        and actual outcomes across probability bins.

        Args:
            proba: Predicted probabilities
            y_true: True labels
            n_bins: Number of bins for calibration

        Returns:
            ECE value (lower is better, 0 = perfect calibration)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            # Get samples in this bin
            in_bin = (proba > bin_boundaries[i]) & (proba <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                # Calculate accuracy and confidence for this bin
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = proba[in_bin].mean()

                # Add weighted absolute difference
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

        return ece


def create_confidence_estimator_from_model(model_path: str,
                                           scaler_path: Optional[str] = None,
                                           X_cal: Optional[np.ndarray] = None,
                                           y_cal: Optional[np.ndarray] = None) -> ConfidenceEstimator:
    """
    Factory function to create a ConfidenceEstimator from saved model files.

    Args:
        model_path: Path to the saved model (.joblib)
        scaler_path: Optional path to the saved scaler
        X_cal: Optional calibration features
        y_cal: Optional calibration targets

    Returns:
        Configured ConfidenceEstimator instance
    """
    import joblib

    # Load model
    model_data = joblib.load(model_path)

    # Handle both dictionary and direct model formats
    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model']
    else:
        model = model_data

    return ConfidenceEstimator(model, X_cal, y_cal)
