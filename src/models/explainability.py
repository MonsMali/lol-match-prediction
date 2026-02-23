"""
Model Explainability Module

Provides global and local model explanations using SHAP
(SHapley Additive exPlanations) for League of Legends match prediction.

Key Features:
- Global SHAP summary plots for feature importance
- Per-prediction force plots and waterfall charts
- Top features pushing toward win/loss for each prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import warnings
import os

warnings.filterwarnings('ignore')

# Check if SHAP is available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn(
        "SHAP not available. Install with: pip install shap>=0.41.0\n"
        "Explainability features will be limited."
    )


class ModelExplainer:
    """
    Provides global and local model explanations using SHAP.

    This class supports multiple model types:
    - Tree-based models (XGBoost, LightGBM, Random Forest): Uses TreeExplainer
    - Linear models (Logistic Regression): Uses LinearExplainer
    - Other models: Uses KernelExplainer (slower but universal)

    Attributes:
        model: The trained model to explain
        feature_names: List of feature names
        explainer: The SHAP explainer instance
        shap_values: Cached SHAP values for background data
    """

    def __init__(self, model, X_background: np.ndarray,
                 feature_names: Optional[List[str]] = None):
        """
        Initialize the ModelExplainer.

        Args:
            model: A trained sklearn-compatible model
            X_background: Background data for SHAP (typically training data sample)
            feature_names: Optional list of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self._X_background = X_background

        if not SHAP_AVAILABLE:
            warnings.warn("SHAP not available. Explainability features disabled.")
            return

        # Initialize appropriate explainer based on model type
        self._initialize_explainer(X_background)

    def _initialize_explainer(self, X_background: np.ndarray) -> None:
        """
        Initialize the appropriate SHAP explainer for the model type.
        """
        if not SHAP_AVAILABLE:
            return

        model_type = type(self.model).__name__

        # Tree-based models
        tree_models = [
            'RandomForestClassifier', 'ExtraTreesClassifier',
            'GradientBoostingClassifier', 'XGBClassifier',
            'LGBMClassifier', 'CatBoostClassifier'
        ]

        # Linear models
        linear_models = ['LogisticRegression', 'LinearSVC', 'SGDClassifier']

        try:
            if model_type in tree_models:
                # Use TreeExplainer for tree-based models (fast)
                self.explainer = shap.TreeExplainer(self.model)
            elif model_type in linear_models:
                # Use LinearExplainer for linear models
                # For classification, we need to handle the link function
                self.explainer = shap.LinearExplainer(
                    self.model,
                    X_background,
                    feature_perturbation='interventional'
                )
            else:
                # Use KernelExplainer as fallback (slower but universal)
                # Sample background data to speed up computation
                if len(X_background) > 100:
                    background_sample = shap.sample(X_background, 100)
                else:
                    background_sample = X_background

                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    background_sample
                )
        except Exception as e:
            warnings.warn(f"Failed to initialize SHAP explainer: {e}")
            self.explainer = None

    def global_importance(self) -> pd.DataFrame:
        """
        Calculate global feature importance using SHAP values.

        Returns:
            DataFrame with features ranked by mean absolute SHAP value
            Columns: feature, importance, direction (positive/negative impact)
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return self._fallback_importance()

        try:
            # Calculate SHAP values for background data
            shap_values = self.explainer.shap_values(self._X_background)

            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multi-class: use positive class
                shap_values = shap_values[1]
            elif hasattr(shap_values, 'values'):
                # Explanation object
                shap_values = shap_values.values

            # Calculate mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            mean_shap = np.mean(shap_values, axis=0)

            # Create importance DataFrame
            if self.feature_names is not None:
                feature_names = self.feature_names
            else:
                feature_names = [f'feature_{i}' for i in range(len(mean_abs_shap))]

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_abs_shap,
                'mean_impact': mean_shap
            })

            # Add direction indicator
            importance_df['direction'] = importance_df['mean_impact'].apply(
                lambda x: 'positive' if x > 0 else 'negative'
            )

            # Sort by importance
            importance_df = importance_df.sort_values(
                'importance', ascending=False
            ).reset_index(drop=True)

            return importance_df

        except Exception as e:
            warnings.warn(f"Failed to calculate SHAP importance: {e}")
            return self._fallback_importance()

    def _fallback_importance(self) -> pd.DataFrame:
        """
        Fallback feature importance when SHAP is not available.
        Uses model's built-in feature importance if available.
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_).flatten()
        else:
            # No feature importance available
            n_features = self._X_background.shape[1] if self._X_background is not None else 0
            return pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(n_features)],
                'importance': [0.0] * n_features,
                'direction': ['unknown'] * n_features
            })

        if self.feature_names is not None:
            feature_names = self.feature_names
        else:
            feature_names = [f'feature_{i}' for i in range(len(importances))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'direction': 'unknown'
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        return importance_df

    def explain_prediction(self, X_single: np.ndarray,
                           top_n: int = 10) -> Dict:
        """
        Explain a single prediction using SHAP values.

        Args:
            X_single: Single sample features (1D or 2D with shape (1, n_features))
            top_n: Number of top contributing features to return

        Returns:
            Dictionary containing:
            - top_features: List of dicts with feature contributions
            - prediction_value: The base prediction value
            - expected_value: The expected (base) value
            - total_contribution: Sum of all SHAP values
        """
        X_single = np.atleast_2d(X_single)

        if not SHAP_AVAILABLE or self.explainer is None:
            return self._fallback_explain_prediction(X_single, top_n)

        try:
            # Calculate SHAP values for this sample
            shap_values = self.explainer.shap_values(X_single)

            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multi-class: use positive class
                shap_values = shap_values[1]
            elif hasattr(shap_values, 'values'):
                shap_values = shap_values.values

            shap_values = shap_values.flatten()

            # Get expected value
            if hasattr(self.explainer, 'expected_value'):
                expected_value = self.explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
            else:
                expected_value = 0.5

            # Get feature names
            if self.feature_names is not None:
                feature_names = self.feature_names
            else:
                feature_names = [f'feature_{i}' for i in range(len(shap_values))]

            # Create feature contributions list
            contributions = []
            for i, (name, value) in enumerate(zip(feature_names, shap_values)):
                contributions.append({
                    'feature': name,
                    'shap_value': float(value),
                    'feature_value': float(X_single[0, i]) if i < X_single.shape[1] else None,
                    'direction': 'win' if value > 0 else 'loss'
                })

            # Sort by absolute SHAP value and get top N
            contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            top_features = contributions[:top_n]

            # Separate win and loss factors
            win_factors = [c for c in top_features if c['direction'] == 'win']
            loss_factors = [c for c in top_features if c['direction'] == 'loss']

            return {
                'top_features': top_features,
                'win_factors': win_factors,
                'loss_factors': loss_factors,
                'expected_value': float(expected_value),
                'total_contribution': float(sum(shap_values)),
                'prediction_value': float(expected_value + sum(shap_values))
            }

        except Exception as e:
            warnings.warn(f"Failed to explain prediction: {e}")
            return self._fallback_explain_prediction(X_single, top_n)

    def _fallback_explain_prediction(self, X_single: np.ndarray,
                                     top_n: int) -> Dict:
        """
        Fallback explanation when SHAP is not available.
        Uses feature importance and feature values.
        """
        importance_df = self.global_importance()

        # Get top features by importance
        top_features = []
        for _, row in importance_df.head(top_n).iterrows():
            feature_idx = importance_df[importance_df['feature'] == row['feature']].index[0]
            feature_value = float(X_single[0, feature_idx]) if feature_idx < X_single.shape[1] else None

            top_features.append({
                'feature': row['feature'],
                'importance': float(row['importance']),
                'feature_value': feature_value,
                'direction': 'unknown'
            })

        return {
            'top_features': top_features,
            'win_factors': [],
            'loss_factors': [],
            'expected_value': 0.5,
            'total_contribution': 0.0,
            'prediction_value': 0.5,
            'note': 'SHAP not available - showing feature importance only'
        }

    def plot_summary(self, output_path: Optional[str] = None,
                     max_display: int = 20) -> None:
        """
        Generate and optionally save a SHAP summary plot.

        Args:
            output_path: Path to save the plot (if None, displays interactively)
            max_display: Maximum number of features to display
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            print("SHAP not available - cannot generate summary plot")
            return

        try:
            import matplotlib.pyplot as plt

            # Calculate SHAP values
            shap_values = self.explainer.shap_values(self._X_background)

            # Handle different formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif hasattr(shap_values, 'values'):
                shap_values = shap_values.values

            # Create summary plot
            plt.figure(figsize=(12, 10))
            shap.summary_plot(
                shap_values,
                self._X_background,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
            plt.title('SHAP Feature Importance Summary')
            plt.tight_layout()

            if output_path:
                # Ensure directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Summary plot saved to: {output_path}")
            else:
                plt.show()

        except Exception as e:
            warnings.warn(f"Failed to generate summary plot: {e}")

    def plot_waterfall(self, X_single: np.ndarray,
                       output_path: Optional[str] = None,
                       max_display: int = 15) -> None:
        """
        Generate a waterfall plot for a single prediction explanation.

        Args:
            X_single: Single sample to explain
            output_path: Path to save the plot
            max_display: Maximum number of features to display
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            print("SHAP not available - cannot generate waterfall plot")
            return

        try:
            import matplotlib.pyplot as plt

            X_single = np.atleast_2d(X_single)

            # Get SHAP explanation
            explanation = self.explainer(X_single)

            # Handle multi-output
            if len(explanation.shape) == 3:
                explanation = explanation[:, :, 1]

            plt.figure(figsize=(12, 8))
            shap.plots.waterfall(explanation[0], max_display=max_display, show=False)
            plt.title('SHAP Waterfall - Prediction Explanation')
            plt.tight_layout()

            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Waterfall plot saved to: {output_path}")
            else:
                plt.show()

        except Exception as e:
            warnings.warn(f"Failed to generate waterfall plot: {e}")

    def format_explanation_text(self, explanation: Dict,
                                team_name: str = "Blue Team") -> str:
        """
        Format a prediction explanation as human-readable text.

        Args:
            explanation: Dictionary from explain_prediction()
            team_name: Name of the team being predicted to win

        Returns:
            Formatted string explanation
        """
        lines = []
        lines.append(f"Top Factors Favoring {team_name}:")

        # Win factors
        for factor in explanation.get('win_factors', [])[:5]:
            value = factor['shap_value']
            feature = factor['feature']
            lines.append(f"  + {feature}: +{abs(value):.1%} impact")

        lines.append(f"\nTop Factors Against {team_name}:")

        # Loss factors
        for factor in explanation.get('loss_factors', [])[:5]:
            value = factor['shap_value']
            feature = factor['feature']
            lines.append(f"  - {feature}: -{abs(value):.1%} impact")

        return '\n'.join(lines)


def create_explainer_from_model(model_path: str,
                                X_background: np.ndarray,
                                feature_names: Optional[List[str]] = None) -> ModelExplainer:
    """
    Factory function to create a ModelExplainer from saved model files.

    Args:
        model_path: Path to the saved model (.joblib)
        X_background: Background data for SHAP
        feature_names: Optional list of feature names

    Returns:
        Configured ModelExplainer instance
    """
    import joblib

    # Load model
    model_data = joblib.load(model_path)

    # Handle both dictionary and direct model formats
    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model']
        if feature_names is None and 'feature_names' in model_data:
            feature_names = model_data['feature_names']
    else:
        model = model_data

    return ModelExplainer(model, X_background, feature_names)
