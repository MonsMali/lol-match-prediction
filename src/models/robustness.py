"""
Model Robustness Analysis Module

Provides learning curves and validation curves to understand model behavior
and diagnose overfitting/underfitting for League of Legends match prediction.

Key Features:
- Learning curves: train/val performance vs training size
- Validation curves: performance vs hyperparameter values
- Saves visualizations to outputs/visualizations/
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score
from sklearn.base import clone
from typing import Dict, List, Optional, Tuple, Union
import warnings
import os

warnings.filterwarnings('ignore')

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available. Plotting disabled.")


class RobustnessAnalyzer:
    """
    Analyzes model robustness through learning and validation curves.

    This class helps diagnose:
    - Overfitting: Large gap between train and validation performance
    - Underfitting: Both train and validation performance are low
    - Optimal training size: Point where adding more data doesn't help
    - Optimal hyperparameters: Best values for key parameters

    Attributes:
        output_dir: Directory to save visualizations
        scoring: Scoring metric to use (default 'roc_auc')
    """

    def __init__(self, output_dir: Optional[str] = None,
                 scoring: str = 'roc_auc'):
        """
        Initialize the RobustnessAnalyzer.

        Args:
            output_dir: Directory to save plots (default: outputs/visualizations/)
            scoring: Scoring metric for evaluation
        """
        if output_dir is None:
            # Default to project's output directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            output_dir = os.path.join(project_root, "outputs", "visualizations")

        self.output_dir = output_dir
        self.scoring = scoring

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_learning_curves(self, model, X: np.ndarray, y: np.ndarray,
                                  cv: int = 5,
                                  train_sizes: Optional[np.ndarray] = None,
                                  n_jobs: int = -1) -> Dict:
        """
        Generate learning curves showing train/validation performance vs training size.

        Learning curves help identify:
        - If more data would improve performance
        - Signs of overfitting (large train-val gap)
        - Signs of underfitting (both scores are low)

        Args:
            model: sklearn-compatible model
            X: Feature matrix
            y: Target vector
            cv: Number of cross-validation folds
            train_sizes: Training sizes to evaluate (default: 10% to 100%)
            n_jobs: Number of parallel jobs

        Returns:
            Dictionary containing:
            - train_sizes: Actual training sizes used
            - train_scores_mean: Mean training scores
            - train_scores_std: Std of training scores
            - val_scores_mean: Mean validation scores
            - val_scores_std: Std of validation scores
            - diagnosis: String describing the model's behavior
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        print(f"Generating learning curves with {len(train_sizes)} training sizes...")

        try:
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y,
                train_sizes=train_sizes,
                cv=cv,
                scoring=self.scoring,
                n_jobs=n_jobs,
                shuffle=True,
                random_state=42
            )

            # Calculate statistics
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)

            # Diagnose the model
            diagnosis = self._diagnose_learning_curves(train_mean, val_mean)

            result = {
                'train_sizes': train_sizes_abs,
                'train_scores_mean': train_mean,
                'train_scores_std': train_std,
                'val_scores_mean': val_mean,
                'val_scores_std': val_std,
                'diagnosis': diagnosis
            }

            print(f"Learning curves generated successfully")
            print(f"Diagnosis: {diagnosis}")

            return result

        except Exception as e:
            warnings.warn(f"Failed to generate learning curves: {e}")
            return {
                'train_sizes': np.array([]),
                'train_scores_mean': np.array([]),
                'train_scores_std': np.array([]),
                'val_scores_mean': np.array([]),
                'val_scores_std': np.array([]),
                'diagnosis': f'Error: {str(e)}'
            }

    def _diagnose_learning_curves(self, train_scores: np.ndarray,
                                  val_scores: np.ndarray) -> str:
        """
        Diagnose model behavior from learning curves.
        """
        final_train = train_scores[-1]
        final_val = val_scores[-1]
        gap = final_train - final_val

        # Check convergence
        val_improvement = val_scores[-1] - val_scores[len(val_scores)//2]
        converged = abs(val_improvement) < 0.01

        if gap > 0.1:
            if final_val < 0.6:
                return "High variance (overfitting) - Consider regularization or more data"
            else:
                return "Moderate overfitting - Model generalizes but has room for improvement"
        elif final_train < 0.7 and final_val < 0.7:
            return "High bias (underfitting) - Consider more features or complex model"
        elif converged and final_val > 0.75:
            return "Well-fitted - Good generalization, training size appears sufficient"
        elif not converged:
            return "Not converged - More training data may improve performance"
        else:
            return "Acceptable fit - Consider ensemble methods for further improvement"

    def generate_validation_curves(self, model, X: np.ndarray, y: np.ndarray,
                                   param_name: str,
                                   param_range: Union[List, np.ndarray],
                                   cv: int = 5,
                                   n_jobs: int = -1) -> Dict:
        """
        Generate validation curves showing performance vs hyperparameter values.

        Validation curves help identify:
        - Optimal hyperparameter values
        - Overfitting at high parameter values
        - Underfitting at low parameter values

        Args:
            model: sklearn-compatible model
            X: Feature matrix
            y: Target vector
            param_name: Name of the hyperparameter to vary
            param_range: Range of values to test
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs

        Returns:
            Dictionary containing:
            - param_range: Parameter values tested
            - train_scores_mean: Mean training scores
            - train_scores_std: Std of training scores
            - val_scores_mean: Mean validation scores
            - val_scores_std: Std of validation scores
            - best_param: Optimal parameter value
            - best_score: Score at optimal parameter
        """
        print(f"Generating validation curves for {param_name}...")

        try:
            train_scores, val_scores = validation_curve(
                model, X, y,
                param_name=param_name,
                param_range=param_range,
                cv=cv,
                scoring=self.scoring,
                n_jobs=n_jobs
            )

            # Calculate statistics
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)

            # Find optimal parameter
            best_idx = np.argmax(val_mean)
            best_param = param_range[best_idx]
            best_score = val_mean[best_idx]

            result = {
                'param_name': param_name,
                'param_range': np.array(param_range),
                'train_scores_mean': train_mean,
                'train_scores_std': train_std,
                'val_scores_mean': val_mean,
                'val_scores_std': val_std,
                'best_param': best_param,
                'best_score': best_score
            }

            print(f"Validation curves generated successfully")
            print(f"Optimal {param_name}: {best_param} (score: {best_score:.4f})")

            return result

        except Exception as e:
            warnings.warn(f"Failed to generate validation curves: {e}")
            return {
                'param_name': param_name,
                'param_range': np.array(param_range),
                'train_scores_mean': np.array([]),
                'train_scores_std': np.array([]),
                'val_scores_mean': np.array([]),
                'val_scores_std': np.array([]),
                'best_param': None,
                'best_score': None,
                'error': str(e)
            }

    def plot_learning_curves(self, curves_data: Dict,
                             output_path: Optional[str] = None,
                             title: str = "Learning Curves") -> None:
        """
        Plot learning curves.

        Args:
            curves_data: Dictionary from generate_learning_curves()
            output_path: Path to save plot (if None, saves to default location)
            title: Plot title
        """
        if not PLOTTING_AVAILABLE:
            print("Matplotlib not available - cannot generate plot")
            return

        if len(curves_data.get('train_sizes', [])) == 0:
            print("No data to plot")
            return

        plt.figure(figsize=(10, 6))

        train_sizes = curves_data['train_sizes']
        train_mean = curves_data['train_scores_mean']
        train_std = curves_data['train_scores_std']
        val_mean = curves_data['val_scores_mean']
        val_std = curves_data['val_scores_std']

        # Plot training scores
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                         alpha=0.1, color='blue')
        plt.plot(train_sizes, train_mean, 'o-', color='blue',
                 label='Training Score')

        # Plot validation scores
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                         alpha=0.1, color='orange')
        plt.plot(train_sizes, val_mean, 'o-', color='orange',
                 label='Validation Score')

        plt.xlabel('Training Size')
        plt.ylabel(f'Score ({self.scoring})')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        # Add diagnosis text
        diagnosis = curves_data.get('diagnosis', '')
        if diagnosis:
            plt.figtext(0.5, 0.02, f"Diagnosis: {diagnosis}",
                       ha='center', fontsize=9, style='italic')

        plt.tight_layout()

        if output_path is None:
            output_path = os.path.join(self.output_dir, 'learning_curves.png')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Learning curves saved to: {output_path}")

    def plot_validation_curves(self, curves_data: Dict,
                               output_path: Optional[str] = None,
                               title: Optional[str] = None) -> None:
        """
        Plot validation curves.

        Args:
            curves_data: Dictionary from generate_validation_curves()
            output_path: Path to save plot (if None, saves to default location)
            title: Plot title (if None, auto-generated)
        """
        if not PLOTTING_AVAILABLE:
            print("Matplotlib not available - cannot generate plot")
            return

        if len(curves_data.get('param_range', [])) == 0:
            print("No data to plot")
            return

        plt.figure(figsize=(10, 6))

        param_range = curves_data['param_range']
        param_name = curves_data.get('param_name', 'parameter')
        train_mean = curves_data['train_scores_mean']
        train_std = curves_data['train_scores_std']
        val_mean = curves_data['val_scores_mean']
        val_std = curves_data['val_scores_std']

        # Handle log scale for certain parameters
        use_log = param_name in ['C', 'alpha', 'learning_rate', 'gamma']

        # Plot training scores
        if use_log:
            plt.semilogx(param_range, train_mean, 'o-', color='blue',
                        label='Training Score')
        else:
            plt.plot(param_range, train_mean, 'o-', color='blue',
                    label='Training Score')

        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                        alpha=0.1, color='blue')

        # Plot validation scores
        if use_log:
            plt.semilogx(param_range, val_mean, 'o-', color='orange',
                        label='Validation Score')
        else:
            plt.plot(param_range, val_mean, 'o-', color='orange',
                    label='Validation Score')

        plt.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                        alpha=0.1, color='orange')

        # Mark optimal point
        best_param = curves_data.get('best_param')
        best_score = curves_data.get('best_score')
        if best_param is not None and best_score is not None:
            plt.axvline(x=best_param, color='green', linestyle='--',
                       label=f'Optimal: {best_param}')

        plt.xlabel(param_name)
        plt.ylabel(f'Score ({self.scoring})')

        if title is None:
            title = f'Validation Curve for {param_name}'
        plt.title(title)

        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if output_path is None:
            output_path = os.path.join(
                self.output_dir,
                f'validation_curve_{param_name}.png'
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Validation curve saved to: {output_path}")

    def plot_curves(self, curves_data: Dict,
                    output_path: Optional[str] = None) -> None:
        """
        Plot either learning or validation curves based on data structure.

        Args:
            curves_data: Dictionary from generate_*_curves()
            output_path: Path to save plot
        """
        if 'train_sizes' in curves_data:
            self.plot_learning_curves(curves_data, output_path)
        elif 'param_range' in curves_data:
            self.plot_validation_curves(curves_data, output_path)
        else:
            print("Unknown curves data format")

    def comprehensive_analysis(self, model, X: np.ndarray, y: np.ndarray,
                               param_configs: Optional[List[Dict]] = None,
                               cv: int = 5) -> Dict:
        """
        Perform comprehensive robustness analysis.

        Args:
            model: sklearn-compatible model
            X: Feature matrix
            y: Target vector
            param_configs: List of dicts with 'name' and 'range' for validation curves
            cv: Number of cross-validation folds

        Returns:
            Dictionary containing all analysis results
        """
        print("=" * 60)
        print("COMPREHENSIVE ROBUSTNESS ANALYSIS")
        print("=" * 60)

        results = {
            'learning_curves': None,
            'validation_curves': {},
            'cross_validation': None
        }

        # 1. Learning curves
        print("\n1. Generating Learning Curves...")
        results['learning_curves'] = self.generate_learning_curves(model, X, y, cv=cv)
        self.plot_learning_curves(results['learning_curves'])

        # 2. Cross-validation baseline
        print("\n2. Cross-Validation Baseline...")
        try:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=self.scoring)
            results['cross_validation'] = {
                'mean': float(cv_scores.mean()),
                'std': float(cv_scores.std()),
                'scores': cv_scores.tolist()
            }
            print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        except Exception as e:
            warnings.warn(f"Cross-validation failed: {e}")

        # 3. Validation curves for specified parameters
        if param_configs:
            print("\n3. Generating Validation Curves...")
            for config in param_configs:
                param_name = config['name']
                param_range = config['range']
                try:
                    vc_result = self.generate_validation_curves(
                        model, X, y, param_name, param_range, cv=cv
                    )
                    results['validation_curves'][param_name] = vc_result
                    self.plot_validation_curves(vc_result)
                except Exception as e:
                    warnings.warn(f"Validation curve for {param_name} failed: {e}")

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)

        return results


def get_default_param_configs(model) -> List[Dict]:
    """
    Get default parameter configurations for validation curves based on model type.

    Args:
        model: sklearn-compatible model

    Returns:
        List of parameter configuration dictionaries
    """
    model_type = type(model).__name__

    configs = {
        'LogisticRegression': [
            {'name': 'C', 'range': np.logspace(-3, 3, 7)}
        ],
        'RandomForestClassifier': [
            {'name': 'max_depth', 'range': [3, 5, 7, 10, 15, 20, None]},
            {'name': 'n_estimators', 'range': [50, 100, 200, 300, 500]}
        ],
        'XGBClassifier': [
            {'name': 'max_depth', 'range': [3, 5, 7, 9, 11]},
            {'name': 'learning_rate', 'range': [0.01, 0.05, 0.1, 0.2, 0.3]}
        ],
        'LGBMClassifier': [
            {'name': 'max_depth', 'range': [3, 5, 7, 9, 11, -1]},
            {'name': 'num_leaves', 'range': [15, 31, 50, 75, 100]}
        ],
        'SVC': [
            {'name': 'C', 'range': np.logspace(-2, 2, 5)},
            {'name': 'gamma', 'range': np.logspace(-3, 1, 5)}
        ]
    }

    return configs.get(model_type, [])
