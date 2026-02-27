import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score,
    log_loss, brier_score_loss, matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import sys
import os

# Add the parent directory to Python path to import feature engineering
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.engineering import AdvancedFeatureEngineering
from models.robustness import RobustnessAnalyzer, get_default_param_configs
from models.explainability import ModelExplainer

# Import new evaluation module
try:
    from evaluation.metrics import (
        MultiMetricEvaluator,
        ProbabilityCalibrator,
        UncertaintyQuantifier,
        create_evaluation_report
    )
    MULTI_METRIC_AVAILABLE = True
except ImportError:
    MULTI_METRIC_AVAILABLE = False

# Import data quality module
try:
    from data.quality import (
        TemporalWeighter,
        DataAugmenter,
        OutlierDetector,
        DataValidator,
        create_quality_report
    )
    DATA_QUALITY_AVAILABLE = True
except ImportError:
    DATA_QUALITY_AVAILABLE = False

warnings.filterwarnings('ignore')

# GPU Detection
GPU_AVAILABLE = False
GPU_DEVICE_NAME = "N/A"
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        GPU_DEVICE_NAME = torch.cuda.get_device_name(0)
        print(f"GPU detected: {GPU_DEVICE_NAME}")
    else:
        print("No GPU detected. Training will use CPU.")
except ImportError:
    # Try XGBoost's built-in GPU check as fallback
    try:
        _test = xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=1)
        _test.fit(np.array([[0]]), np.array([0]))
        GPU_AVAILABLE = True
        GPU_DEVICE_NAME = "CUDA (detected via XGBoost)"
        print(f"GPU detected via XGBoost: CUDA available")
    except Exception:
        print("No GPU detected. Training will use CPU.")


class UltimateLoLPredictor:
    """
    Ultimate League of Legends match prediction system combining:
    - Advanced feature engineering
    - Multiple state-of-the-art ML algorithms
    - Rigorous train/validation/test split
    - Hyperparameter optimization
    - Ensemble methods
    """
    
    def __init__(self, data_path=None):
        if data_path is None:
            # Build path to dataset in the new organized structure
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            data_path = os.path.join(project_root, "data", "complete_target_leagues_dataset.csv")
        
        self.data_path = data_path
        
        # Verify the file exists
        if not os.path.exists(self.data_path):
            print(f" Dataset file not found at: {self.data_path}")
            print("Looking for alternative paths...")
            # Try alternative paths
            alternative_paths = [
                os.path.join(os.path.dirname(os.path.dirname(script_dir)), "data", "processed", "complete_target_leagues_dataset.csv"),
                os.path.join(os.path.dirname(os.path.dirname(script_dir)), "data", "complete_target_leagues_dataset.csv"),
                os.path.join(project_root, "data", "processed", "complete_target_leagues_dataset.csv"),
                os.path.join(project_root, "data", "target_leagues_dataset.csv"),  # Fallback to old dataset
                "../../data/processed/complete_target_leagues_dataset.csv",
                "../../data/complete_target_leagues_dataset.csv",
                "../data/complete_target_leagues_dataset.csv"
            ]
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    self.data_path = alt_path
                    print(f" Found dataset at: {self.data_path}")
                    break
            else:
                raise FileNotFoundError(f"Dataset file not found. Please ensure complete_target_leagues_dataset.csv is in the data/ directory.")
        
        print(f"Using dataset: {self.data_path}")
        self.feature_engineering = AdvancedFeatureEngineering(self.data_path)
        
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
        # Data splits
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
    def prepare_advanced_features(self, use_enhanced_v2: bool = False):
        """Prepare the complete advanced feature set.

        Args:
            use_enhanced_v2: Whether to use enhanced v2 features including
                             side selection, patch transition, and extended interactions
        """
        print("ULTIMATE LOL MATCH PREDICTION SYSTEM")
        print("=" * 80)

        # Load and engineer features
        df = self.feature_engineering.load_and_analyze_data()

        # Use enhanced v2 features if requested (skips slow iterrows-based create_advanced_features)
        if use_enhanced_v2 and hasattr(self.feature_engineering, 'create_enhanced_features_v2'):
            print("\nApplying Enhanced Feature Engineering v2...")
            advanced_features = self.feature_engineering.create_enhanced_features_v2()
            print(f"   Enhanced features added: side selection, patch transition, extended interactions")
        else:
            advanced_features = self.feature_engineering.create_advanced_features()

        final_features = self.feature_engineering.apply_advanced_encoding()

        # Get target variable
        self.X = final_features
        self.y = df['result']

        # Run data quality checks if available
        if DATA_QUALITY_AVAILABLE:
            print("\nRunning Data Quality Checks...")
            validator = DataValidator()
            quality_report = validator.validate_dataset(df)
            print(f"   Quality Score: {quality_report.quality_score:.1f}/100")
            print(f"   Outliers: {quality_report.outlier_count}")
            print(f"   Duplicates: {quality_report.duplicate_count}")

        print(f"\nFINAL FEATURE SUMMARY:")
        print(f"   Total features: {self.X.shape[1]}")
        print(f"   Total samples: {self.X.shape[0]}")
        print(f"   Class distribution: {self.y.value_counts().to_dict()}")

        return self.X, self.y

    def apply_temporal_weighting(self, decay_type: str = 'exponential',
                                 half_life_days: int = 180) -> np.ndarray:
        """Apply temporal weighting to training samples.

        More recent matches get higher weights to account for meta evolution.

        Args:
            decay_type: Type of decay ('exponential', 'linear', 'step')
            half_life_days: Days until weight is halved

        Returns:
            Array of sample weights for training
        """
        if not DATA_QUALITY_AVAILABLE:
            print("   Warning: Data quality module not available, skipping temporal weighting")
            return None

        print(f"\nApplying Temporal Weighting (decay={decay_type}, half_life={half_life_days}d)...")

        weighter = TemporalWeighter(decay_type=decay_type, half_life_days=half_life_days)
        df = self.feature_engineering.df

        # Get weights for training data
        train_dates = df.loc[self.X_train.index, 'date']
        weights = weighter.calculate_weights(train_dates)

        print(f"   Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
        print(f"   Mean weight: {weights.mean():.3f}")

        self.sample_weights = weights
        return weights

    def _calculate_ece(self, y_true, y_proba, n_bins=10):
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
            bin_mask = (y_proba > bin_boundaries[i]) & (y_proba <= bin_boundaries[i + 1])
            bin_size = np.sum(bin_mask)

            if bin_size > 0:
                bin_accuracy = np.mean(y_true[bin_mask])
                bin_confidence = np.mean(y_proba[bin_mask])
                ece += (bin_size / len(y_true)) * abs(bin_accuracy - bin_confidence)

        return ece

    def _calculate_composite_score(self, auc, log_loss_val, brier, ece, f1,
                                    mcc=None, kappa=None, weights=None):
        """
        Calculate weighted composite score for model ranking.

        Default weights:
        - AUC: 25% (discrimination ability)
        - Log Loss: 20% (probability accuracy)
        - Brier: 15% (calibration quality)
        - ECE: 10% (expected calibration error)
        - F1: 10% (classification performance)
        - MCC: 10% (balanced classification metric)
        - Kappa: 10% (agreement beyond chance)

        Args:
            auc: AUC-ROC score
            log_loss_val: Log loss value
            brier: Brier score
            ece: Expected calibration error
            f1: F1 score
            mcc: Matthews Correlation Coefficient (optional)
            kappa: Cohen's Kappa (optional)
            weights: Optional custom weights dict

        Returns:
            Composite score between 0 and 1 (higher is better)
        """
        if weights is None:
            if mcc is not None and kappa is not None:
                weights = {
                    'auc': 0.25, 'log_loss': 0.20, 'brier': 0.15,
                    'ece': 0.10, 'f1': 0.10, 'mcc': 0.10, 'kappa': 0.10
                }
            else:
                weights = {'auc': 0.30, 'log_loss': 0.25, 'brier': 0.20, 'ece': 0.15, 'f1': 0.10}

        # Normalize metrics to 0-1 range where higher is always better
        normalized = {
            'auc': auc,  # Already 0-1, higher is better
            'f1': f1,    # Already 0-1, higher is better
            'log_loss': 1 / (1 + log_loss_val),  # Transform: lower is better
            'brier': 1 - brier,  # 0-1, lower is better
            'ece': 1 - ece       # 0-1, lower is better
        }

        # Add optional metrics if provided
        if mcc is not None:
            normalized['mcc'] = (mcc + 1) / 2  # MCC is -1 to 1, normalize to 0-1
        if kappa is not None:
            normalized['kappa'] = (kappa + 1) / 2  # Kappa can be negative, normalize

        composite = sum(weights.get(k, 0) * normalized.get(k, 0) for k in weights.keys())
        return composite

    def _calculate_all_metrics(self, y_true, y_pred, y_proba):
        """Calculate all evaluation metrics for a model.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities

        Returns:
            Dictionary of all metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_proba),
            'log_loss': log_loss(y_true, y_proba),
            'brier': brier_score_loss(y_true, y_proba),
            'ece': self._calculate_ece(y_true, y_proba),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'kappa': cohen_kappa_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'sensitivity': recall_score(y_true, y_pred),  # Same as recall
            'specificity': recall_score(y_true, y_pred, pos_label=0)
        }

        # Calculate calibration slope if evaluator available
        if MULTI_METRIC_AVAILABLE:
            try:
                evaluator = MultiMetricEvaluator()
                detailed = evaluator.evaluate(y_true, y_pred, y_proba)
                metrics['calibration_slope'] = detailed.get('calibration_slope', None)
            except Exception:
                metrics['calibration_slope'] = None
        else:
            metrics['calibration_slope'] = None

        return metrics

    def split_data_temporally(self, train_size=0.6, val_size=0.2, test_size=0.2):
        """Split data temporally for realistic evaluation."""
        print(f"\n TEMPORAL DATA SPLITTING")
        
        # Ensure sizes sum to 1.0
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split sizes must sum to 1.0"
        
        # Get the original dataframe for date sorting
        df = self.feature_engineering.df.copy()
        df_sorted = df.sort_values('date')
        
        total_samples = len(df_sorted)
        train_end = int(total_samples * train_size)
        val_end = int(total_samples * (train_size + val_size))
        
        # Split indices
        train_indices = df_sorted.index[:train_end]
        val_indices = df_sorted.index[train_end:val_end]
        test_indices = df_sorted.index[val_end:]
        
        # Split features and target
        self.X_train = self.X.loc[train_indices]
        self.X_val = self.X.loc[val_indices]
        self.X_test = self.X.loc[test_indices]
        self.y_train = self.y.loc[train_indices]
        self.y_val = self.y.loc[val_indices]
        self.y_test = self.y.loc[test_indices]
        
        print(f"    Training: {self.X_train.shape} ({train_size:.1%})")
        print(f"    Validation: {self.X_val.shape} ({val_size:.1%})")
        print(f"    Test: {self.X_test.shape} ({test_size:.1%})")
        
        # Print date ranges
        print(f"    Train: {df_sorted.loc[train_indices, 'date'].min()} to {df_sorted.loc[train_indices, 'date'].max()}")
        print(f"    Val: {df_sorted.loc[val_indices, 'date'].min()} to {df_sorted.loc[val_indices, 'date'].max()}")
        print(f"    Test: {df_sorted.loc[test_indices, 'date'].min()} to {df_sorted.loc[test_indices, 'date'].max()}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Convert back to DataFrames
        self.X_train_scaled = pd.DataFrame(self.X_train_scaled, columns=self.X.columns, index=self.X_train.index)
        self.X_val_scaled = pd.DataFrame(self.X_val_scaled, columns=self.X.columns, index=self.X_val.index)
        self.X_test_scaled = pd.DataFrame(self.X_test_scaled, columns=self.X.columns, index=self.X_test.index)
        
        # Print class distributions
        print(f"\n CLASS DISTRIBUTIONS:")
        print(f"   Train: {self.y_train.value_counts(normalize=True).round(3).to_dict()}")
        print(f"   Val: {self.y_val.value_counts(normalize=True).round(3).to_dict()}")
        print(f"   Test: {self.y_test.value_counts(normalize=True).round(3).to_dict()}")
    
    def split_data_stratified_temporal(self, train_size=0.6, val_size=0.2, test_size=0.2):
        """Split data with stratified temporal approach - handling meta evolution."""
        print(f"\n STRATIFIED TEMPORAL DATA SPLITTING (META-AWARE)")
        print("   Strategy: Balance training across all years to handle meta evolution")
        
        # Ensure sizes sum to 1.0
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split sizes must sum to 1.0"
        
        # Get the original dataframe with year information
        df = self.feature_engineering.df.copy()
        df_sorted = df.sort_values('date')
        
        # Get unique years
        years = sorted(df_sorted['year'].unique())
        print(f"    Years available: {years}")
        
        # Initialize split indices
        train_indices = []
        val_indices = []
        test_indices = []
        
        # Split each year proportionally
        for year in years:
            year_data = df_sorted[df_sorted['year'] == year].sort_values('date')
            year_size = len(year_data)
            
            if year_size < 10:  # Skip years with too few matches
                print(f"    Skipping {year}: only {year_size} matches")
                continue
            
            # Calculate split points for this year
            train_end = int(year_size * train_size)
            val_end = int(year_size * (train_size + val_size))
            
            # Get indices for this year
            year_indices = year_data.index.tolist()
            train_indices.extend(year_indices[:train_end])
            val_indices.extend(year_indices[train_end:val_end])
            test_indices.extend(year_indices[val_end:])
            
            print(f"   {year}: {train_end} train, {val_end-train_end} val, {year_size-val_end} test")
        
        # Convert to sets for indexing
        train_indices = list(set(train_indices))
        val_indices = list(set(val_indices))
        test_indices = list(set(test_indices))
        
        # Split features and target
        self.X_train = self.X.loc[train_indices]
        self.X_val = self.X.loc[val_indices]
        self.X_test = self.X.loc[test_indices]
        self.y_train = self.y.loc[train_indices]
        self.y_val = self.y.loc[val_indices]
        self.y_test = self.y.loc[test_indices]
        
        print(f"\n    FINAL STRATIFIED SPLIT:")
        print(f"    Training: {self.X_train.shape}")
        print(f"    Validation: {self.X_val.shape}")
        print(f"    Test: {self.X_test.shape}")
        
        # Print year distributions
        train_years = df_sorted.loc[train_indices, 'year'].value_counts().sort_index()
        val_years = df_sorted.loc[val_indices, 'year'].value_counts().sort_index()
        test_years = df_sorted.loc[test_indices, 'year'].value_counts().sort_index()
        
        print(f"\n    Training year distribution: {train_years.to_dict()}")
        print(f"    Validation year distribution: {val_years.to_dict()}")
        print(f"    Test year distribution: {test_years.to_dict()}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Convert back to DataFrames
        self.X_train_scaled = pd.DataFrame(self.X_train_scaled, columns=self.X.columns, index=self.X_train.index)
        self.X_val_scaled = pd.DataFrame(self.X_val_scaled, columns=self.X.columns, index=self.X_val.index)
        self.X_test_scaled = pd.DataFrame(self.X_test_scaled, columns=self.X.columns, index=self.X_test.index)
        
        # Print class distributions
        print(f"\n CLASS DISTRIBUTIONS:")
        print(f"   Train: {self.y_train.value_counts(normalize=True).round(3).to_dict()}")
        print(f"   Val: {self.y_val.value_counts(normalize=True).round(3).to_dict()}")
        print(f"   Test: {self.y_test.value_counts(normalize=True).round(3).to_dict()}")
    
    def train_advanced_models(self, quick_mode=False):
        """Train multiple advanced ML models."""
        print(f"\n TRAINING ADVANCED MODEL SUITE")
        if quick_mode:
            print(" QUICK MODE: Reduced hyperparameter search for faster training")
        print("=" * 60)
        
        # Define comprehensive model suite
        if quick_mode:
            # Reduced parameter grids for faster training
            models_config = {
                'Random Forest': {
                    'model': RandomForestClassifier(
                        random_state=42, n_jobs=-1, class_weight='balanced'
                    ),
                    'params': {
                        'n_estimators': [200],
                        'max_depth': [10],
                        'min_samples_split': [10]
                    },
                    'use_scaled': False
                },
                'XGBoost': {
                    'model': xgb.XGBClassifier(
                        random_state=42, n_jobs=-1, eval_metric='logloss',
                        tree_method='gpu_hist' if GPU_AVAILABLE else 'hist',
                        gpu_id=0 if GPU_AVAILABLE else None
                    ),
                    'params': {
                        'n_estimators': [200],
                        'max_depth': [6],
                        'learning_rate': [0.1]
                    },
                    'use_scaled': False
                },
                'LightGBM': {
                    'model': lgb.LGBMClassifier(
                        random_state=42, n_jobs=-1, class_weight='balanced', verbose=-1,
                        device='gpu' if GPU_AVAILABLE else 'cpu'
                    ),
                    'params': {
                        'n_estimators': [200],
                        'max_depth': [8],
                        'learning_rate': [0.1]
                    },
                    'use_scaled': False
                },
                'Logistic Regression': {
                    'model': LogisticRegression(
                        random_state=42, max_iter=1000, class_weight='balanced'
                    ),
                    'params': {
                        'C': [0.1, 1.0, 10.0],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear']
                    },
                    'use_scaled': True
                }
            }
        else:
            # Full parameter grids for comprehensive search
            models_config = {
                'Random Forest': {
                    'model': RandomForestClassifier(
                        random_state=42, n_jobs=-1, class_weight='balanced'
                    ),
                    'params': {
                        'n_estimators': [200, 300],
                        'max_depth': [8, 12],
                        'min_samples_split': [10, 15],
                        'min_samples_leaf': [4, 6]
                    },
                    'use_scaled': False
                },
                'Extra Trees': {
                    'model': ExtraTreesClassifier(
                        random_state=42, n_jobs=-1, class_weight='balanced'
                    ),
                    'params': {
                        'n_estimators': [200, 300],
                        'max_depth': [10, 15],
                        'min_samples_split': [8, 12],
                        'min_samples_leaf': [3, 5]
                    },
                    'use_scaled': False
                },
                'XGBoost': {
                    'model': xgb.XGBClassifier(
                        random_state=42, n_jobs=-1, eval_metric='logloss',
                        tree_method='gpu_hist' if GPU_AVAILABLE else 'hist',
                        gpu_id=0 if GPU_AVAILABLE else None
                    ),
                    'params': {
                        'n_estimators': [200, 300],
                        'max_depth': [6, 8],
                        'learning_rate': [0.1, 0.15],
                        'subsample': [0.8, 0.9],
                        'colsample_bytree': [0.8, 0.9]
                    },
                    'use_scaled': False
                },
                'LightGBM': {
                    'model': lgb.LGBMClassifier(
                        random_state=42, n_jobs=-1, class_weight='balanced', verbose=-1,
                        device='gpu' if GPU_AVAILABLE else 'cpu'
                    ),
                    'params': {
                        'n_estimators': [200, 300],
                        'max_depth': [8, 12],
                        'learning_rate': [0.1, 0.15],
                        'num_leaves': [31, 50],
                        'subsample': [0.8, 0.9]
                    },
                    'use_scaled': False
                },
                'Logistic Regression': {
                    'model': LogisticRegression(
                        random_state=42, max_iter=1000, class_weight='balanced'
                    ),
                    'params': {
                        'C': [0.1, 1.0, 10.0],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear', 'saga']
                    },
                    'use_scaled': True
                },
                'SVM': {
                    'model': SVC(
                        random_state=42, probability=True, class_weight='balanced'
                    ),
                    'params': {
                        'C': [1.0, 10.0],
                        'kernel': ['rbf', 'poly'],
                        'gamma': ['scale', 'auto']
                    },
                    'use_scaled': True
                },
                'MLP': {
                    'model': MLPClassifier(
                        random_state=42, max_iter=1000
                    ),
                    'params': {
                        'hidden_layer_sizes': [(100,), (150,), (200,)],
                        'activation': ['relu', 'tanh'],
                        'solver': ['adam', 'sgd']
                    },
                    'use_scaled': True
                }
            }
        
        # Add CatBoost if available (excellent for categorical features)
        if CATBOOST_AVAILABLE:
            models_config['CatBoost'] = {
                'model': cb.CatBoostClassifier(
                    random_state=42, verbose=False, auto_class_weights='Balanced',
                    task_type='GPU' if GPU_AVAILABLE else 'CPU'
                ),
                'params': {
                    'iterations': [200, 300],
                    'depth': [6, 8],
                    'learning_rate': [0.1, 0.15],
                    'l2_leaf_reg': [3, 5]
                },
                'use_scaled': False
            }
        
        if GPU_AVAILABLE:
            print(f"\n  GPU Acceleration: Enabled ({GPU_DEVICE_NAME})")
        else:
            print(f"\n  GPU Acceleration: Disabled (using CPU)")

        # Train each model with hyperparameter optimization
        for name, config in models_config.items():
            print(f"\n Training {name}...")

            model = config['model']
            params = config['params']
            use_scaled = config['use_scaled']

            # Select appropriate data
            X_train_data = self.X_train_scaled if use_scaled else self.X_train
            X_val_data = self.X_val_scaled if use_scaled else self.X_val

            # Hyperparameter optimization with GPU fallback
            grid_search = GridSearchCV(
                model, params, cv=3, scoring='f1', n_jobs=-1, verbose=0
            )

            try:
                grid_search.fit(X_train_data, self.y_train)
            except Exception as gpu_error:
                if GPU_AVAILABLE and name in ('XGBoost', 'LightGBM', 'CatBoost'):
                    print(f"   GPU training failed for {name}, falling back to CPU: {gpu_error}")
                    if name == 'XGBoost':
                        model.set_params(tree_method='hist', gpu_id=None)
                    elif name == 'LightGBM':
                        model.set_params(device='cpu')
                    elif name == 'CatBoost':
                        model.set_params(task_type='CPU')
                    grid_search = GridSearchCV(
                        model, params, cv=3, scoring='f1', n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X_train_data, self.y_train)
                else:
                    raise

            best_model = grid_search.best_estimator_
            
            print(f"   Best params: {grid_search.best_params_}")
            
            # Evaluate on validation set
            y_pred = best_model.predict(X_val_data)
            y_pred_proba = best_model.predict_proba(X_val_data)[:, 1]
            
            # Calculate all metrics using new method
            metrics = self._calculate_all_metrics(self.y_val, y_pred, y_pred_proba)

            # Cross-validation
            cv_scores = cross_val_score(best_model, X_train_data, self.y_train, cv=5, scoring='f1')

            # Calculate composite score with new metrics
            composite = self._calculate_composite_score(
                metrics['auc'], metrics['log_loss'], metrics['brier'],
                metrics['ece'], metrics['f1'], metrics['mcc'], metrics['kappa']
            )

            # Store results with all metrics
            self.models[name] = best_model
            self.results[name] = {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'auc': metrics['auc'],
                'log_loss': metrics['log_loss'],
                'brier': metrics['brier'],
                'ece': metrics['ece'],
                'mcc': metrics['mcc'],
                'kappa': metrics['kappa'],
                'balanced_accuracy': metrics['balanced_accuracy'],
                'sensitivity': metrics['sensitivity'],
                'specificity': metrics['specificity'],
                'calibration_slope': metrics['calibration_slope'],
                'composite': composite,
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'best_params': grid_search.best_params_,
                'use_scaled': use_scaled
            }

            print(f"   Validation F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f} | Composite: {composite:.4f}")
            print(f"   Log Loss: {metrics['log_loss']:.4f} | Brier: {metrics['brier']:.4f} | ECE: {metrics['ece']:.4f}")
            print(f"   MCC: {metrics['mcc']:.4f} | Kappa: {metrics['kappa']:.4f} | Bal.Acc: {metrics['balanced_accuracy']:.4f}")
            print(f"   CV F1: {cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})")
    
    def create_ultimate_ensemble(self):
        """Create sophisticated ensemble of best models."""
        print(f"\n CREATING ULTIMATE ENSEMBLE")
        
        # Get models that support probability prediction
        prob_models = [(name, model) for name, model in self.models.items() 
                      if hasattr(model, 'predict_proba')]
        
        if len(prob_models) < 2:
            print("    Not enough models for ensemble")
            return
        
        # Calculate performance-based weights
        weights = []
        model_names = []
        
        for name, _ in prob_models:
            f1_performance = self.results[name]['f1']
            cv_stability = 1 / (1 + self.results[name]['cv_f1_std'])  # Reward stability
            combined_score = f1_performance * cv_stability
            
            weights.append(combined_score)
            model_names.append(name)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        print("    Ensemble composition:")
        for name, weight in zip(model_names, weights):
            print(f"      {name}: {weight:.3f}")
        
        # Compute weighted soft-voting predictions using already-trained models
        # (avoids re-training all models which can take hours)
        y_pred_proba_ensemble = np.zeros(len(self.y_val))
        for (name, model), weight in zip(prob_models, weights):
            use_scaled = self.results[name].get('use_scaled', False)
            X_val_data = self.X_val_scaled if use_scaled else self.X_val
            y_pred_proba_ensemble += weight * model.predict_proba(X_val_data)[:, 1]

        y_pred_ensemble = (y_pred_proba_ensemble >= 0.5).astype(int)

        # Calculate all metrics
        metrics = self._calculate_all_metrics(self.y_val, y_pred_ensemble, y_pred_proba_ensemble)
        composite = self._calculate_composite_score(
            metrics['auc'], metrics['log_loss'], metrics['brier'],
            metrics['ece'], metrics['f1'], metrics['mcc'], metrics['kappa']
        )

        # Store ensemble config for inference and test evaluation
        self.ensemble_config = {
            'model_names': model_names,
            'weights': weights
        }
        self.results['Ultimate Ensemble'] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'auc': metrics['auc'],
            'log_loss': metrics['log_loss'],
            'brier': metrics['brier'],
            'ece': metrics['ece'],
            'mcc': metrics['mcc'],
            'kappa': metrics['kappa'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'sensitivity': metrics['sensitivity'],
            'specificity': metrics['specificity'],
            'calibration_slope': metrics['calibration_slope'],
            'composite': composite,
            'cv_f1_mean': 0,
            'cv_f1_std': 0,
            'best_params': {'type': 'weighted_soft_vote', 'n_models': len(model_names)},
            'use_scaled': False
        }

        print(f"   Validation F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f} | Composite: {composite:.4f}")
    
    def evaluate_models(self):
        """Comprehensive model evaluation on validation set with composite ranking."""
        print(f"\n{'='*80}")
        print("MODEL EVALUATION (Composite Ranking)")
        print(f"{'='*80}")

        # Create results summary with all metrics
        results_summary = []
        for name, results in self.results.items():
            results_summary.append({
                'Model': name,
                'Composite': results.get('composite', 0),
                'AUC': results['auc'],
                'Log Loss': results.get('log_loss', 0),
                'Brier': results.get('brier', 0),
                'ECE': results.get('ece', 0),
                'F1': results['f1'],
                'MCC': results.get('mcc', 0),
                'Kappa': results.get('kappa', 0),
                'Bal.Acc': results.get('balanced_accuracy', 0),
                'Accuracy': results['accuracy'],
                'CV F1 Mean': results.get('cv_f1_mean', 0),
                'CV F1 Std': results.get('cv_f1_std', 0)
            })

        # Sort by composite score (higher is better)
        results_df = pd.DataFrame(results_summary).sort_values('Composite', ascending=False)
        results_df.insert(0, 'Rank', range(1, len(results_df) + 1))

        # Display ranking table
        print("\nVALIDATION PERFORMANCE RANKING:")
        display_cols = ['Rank', 'Model', 'Composite', 'AUC', 'Log Loss', 'Brier', 'ECE', 'F1', 'MCC', 'Kappa']
        print(results_df[display_cols].round(4).to_string(index=False))

        # Best model by composite score
        best_model_name = results_df.iloc[0]['Model']

        print(f"\nBEST MODEL: {best_model_name}")
        print(f"  Composite Score:    {results_df.iloc[0]['Composite']:.4f}")
        print(f"  AUC-ROC:            {results_df.iloc[0]['AUC']:.4f}")
        print(f"  Log Loss:           {results_df.iloc[0]['Log Loss']:.4f}")
        print(f"  Brier Score:        {results_df.iloc[0]['Brier']:.4f}")
        print(f"  ECE:                {results_df.iloc[0]['ECE']:.4f}")
        print(f"  F1 Score:           {results_df.iloc[0]['F1']:.4f}")
        print(f"  MCC:                {results_df.iloc[0]['MCC']:.4f}")
        print(f"  Cohen's Kappa:      {results_df.iloc[0]['Kappa']:.4f}")
        print(f"  Balanced Accuracy:  {results_df.iloc[0]['Bal.Acc']:.4f}")

        # Print metric weights used
        print(f"\nMetric Weights: AUC=25%, Log Loss=20%, Brier=15%, ECE=10%, F1=10%, MCC=10%, Kappa=10%")

        return best_model_name, results_df
    
    def final_test_evaluation(self, best_model_name, calibrate_probs: bool = True,
                               quantify_uncertainty: bool = True):
        """Final evaluation on completely unseen test set.

        Args:
            best_model_name: Name of the best model to evaluate
            calibrate_probs: Whether to apply probability calibration
            quantify_uncertainty: Whether to compute bootstrap confidence intervals
        """
        print(f"\nFINAL TEST SET EVALUATION")
        print("=" * 60)
        print(f"EVALUATING ON COMPLETELY UNSEEN TEST DATA")
        print(f"Best Model: {best_model_name}")

        # Handle ensemble separately since it uses pre-trained models directly
        if best_model_name == 'Ultimate Ensemble' and hasattr(self, 'ensemble_config'):
            cfg = self.ensemble_config
            y_pred_proba_test = np.zeros(len(self.y_test))
            for name, weight in zip(cfg['model_names'], cfg['weights']):
                use_scaled = self.results[name].get('use_scaled', False)
                X_data = self.X_test_scaled if use_scaled else self.X_test
                y_pred_proba_test += weight * self.models[name].predict_proba(X_data)[:, 1]
            y_pred_test = (y_pred_proba_test >= 0.5).astype(int)
            best_model = None
            use_scaled = False
            X_test_data = self.X_test
        else:
            best_model = self.models[best_model_name]
            use_scaled = self.results[best_model_name]['use_scaled']
            X_test_data = self.X_test_scaled if use_scaled else self.X_test
            y_pred_test = best_model.predict(X_test_data)
            y_pred_proba_test = best_model.predict_proba(X_test_data)[:, 1]

        # Calculate all metrics
        test_metrics = self._calculate_all_metrics(self.y_test, y_pred_test, y_pred_proba_test)

        print(f"\nFINAL TEST RESULTS:")
        print(f"   Accuracy:          {test_metrics['accuracy']:.4f}")
        print(f"   Precision:         {test_metrics['precision']:.4f}")
        print(f"   Recall:            {test_metrics['recall']:.4f}")
        print(f"   F1 Score:          {test_metrics['f1']:.4f}")
        print(f"   AUC:               {test_metrics['auc']:.4f}")
        print(f"   MCC:               {test_metrics['mcc']:.4f}")
        print(f"   Cohen's Kappa:     {test_metrics['kappa']:.4f}")
        print(f"   Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        print(f"   Log Loss:          {test_metrics['log_loss']:.4f}")
        print(f"   Brier Score:       {test_metrics['brier']:.4f}")
        print(f"   ECE:               {test_metrics['ece']:.4f}")

        # Probability Calibration
        calibration_result = None
        if calibrate_probs and MULTI_METRIC_AVAILABLE:
            print(f"\nPROBABILITY CALIBRATION:")
            try:
                X_val_data = self.X_val_scaled if use_scaled else self.X_val
                val_proba = best_model.predict_proba(X_val_data)[:, 1]

                calibrator = ProbabilityCalibrator()
                calibration_result = calibrator.calibrate(
                    self.y_val.values, val_proba,
                    self.y_test.values, y_pred_proba_test
                )

                print(f"   Method: {calibration_result.method}")
                print(f"   ECE Before: {calibration_result.ece_before:.4f}")
                print(f"   ECE After:  {calibration_result.ece_after:.4f}")
                print(f"   Improvement: {calibration_result.improvement:.1%}")
                print(f"   Calibration Slope: {calibration_result.calibration_slope:.4f}")
            except Exception as e:
                print(f"   Calibration skipped: {e}")

        # Uncertainty Quantification
        uncertainty_result = None
        if quantify_uncertainty and MULTI_METRIC_AVAILABLE:
            print(f"\nUNCERTAINTY QUANTIFICATION:")
            try:
                quantifier = UncertaintyQuantifier(n_bootstrap=100)
                uncertainty_result = quantifier.bootstrap_confidence_intervals(
                    best_model, X_test_data, self.y_test
                )

                print(f"   Confidence Level: {uncertainty_result.confidence_level:.0%}")
                print(f"   Mean Prediction Std: {uncertainty_result.std_prediction.mean():.4f}")

                # Calculate metric confidence intervals
                metric_cis = quantifier.calculate_metric_confidence_intervals(
                    best_model, X_test_data, self.y_test
                )
                print(f"   AUC 95% CI: [{metric_cis['auc']['ci_lower']:.4f}, {metric_cis['auc']['ci_upper']:.4f}]")
                print(f"   F1 95% CI:  [{metric_cis['f1']['ci_lower']:.4f}, {metric_cis['f1']['ci_upper']:.4f}]")
            except Exception as e:
                print(f"   Uncertainty quantification skipped: {e}")

        # Compare with validation performance
        val_results = self.results[best_model_name]
        print(f"\nVALIDATION vs TEST COMPARISON:")
        print(f"   F1:       {val_results['f1']:.4f} -> {test_metrics['f1']:.4f} (delta: {val_results['f1'] - test_metrics['f1']:+.4f})")
        print(f"   AUC:      {val_results['auc']:.4f} -> {test_metrics['auc']:.4f} (delta: {val_results['auc'] - test_metrics['auc']:+.4f})")
        print(f"   MCC:      {val_results.get('mcc', 0):.4f} -> {test_metrics['mcc']:.4f}")
        print(f"   Accuracy: {val_results['accuracy']:.4f} -> {test_metrics['accuracy']:.4f} (delta: {val_results['accuracy'] - test_metrics['accuracy']:+.4f})")

        # Classification report
        print(f"\nDETAILED CLASSIFICATION REPORT:")
        print(classification_report(self.y_test, y_pred_test))

        # Confusion matrix visualization
        cm = confusion_matrix(self.y_test, y_pred_test)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Final Test Confusion Matrix - {best_model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Save to visualizations directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        viz_dir = os.path.join(project_root, "outputs", "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        plt.savefig(os.path.join(viz_dir, 'ultimate_test_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Feature importance analysis
        self.analyze_feature_importance(best_model_name)

        return {
            'model': best_model_name,
            'test_accuracy': test_metrics['accuracy'],
            'test_f1': test_metrics['f1'],
            'test_auc': test_metrics['auc'],
            'test_mcc': test_metrics['mcc'],
            'test_kappa': test_metrics['kappa'],
            'test_balanced_accuracy': test_metrics['balanced_accuracy'],
            'test_log_loss': test_metrics['log_loss'],
            'test_brier': test_metrics['brier'],
            'test_ece': test_metrics['ece'],
            'generalization_gap': val_results['f1'] - test_metrics['f1'],
            'calibration_result': calibration_result,
            'uncertainty_result': uncertainty_result
        }
    
    def analyze_feature_importance(self, model_name):
        """Analyze feature importance for interpretability."""
        print(f"\n FEATURE IMPORTANCE ANALYSIS")
        print("=" * 40)
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n TOP 20 FEATURES - {model_name}:")
            for i, (_, row) in enumerate(importance_df.head(20).iterrows(), 1):
                print(f"   {i:2d}. {row['feature']}: {row['importance']:.4f}")
            
            # Save feature importance plot
            plt.figure(figsize=(12, 10))
            top_features = importance_df.head(20)
            sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
            plt.title(f'Top 20 Feature Importance - {model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            
            # Save to visualizations directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            viz_dir = os.path.join(project_root, "outputs", "visualizations")
            os.makedirs(viz_dir, exist_ok=True)

            plt.savefig(os.path.join(viz_dir, 'ultimate_feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Save feature importance data to results directory
            results_dir = os.path.join(project_root, "outputs", "results")
            os.makedirs(results_dir, exist_ok=True)
            importance_df.to_csv(os.path.join(results_dir, 'ultimate_feature_importance.csv'), index=False)
            
        else:
            print(f"    {model_name} does not provide feature importances")
    
    def save_ultimate_model(self, best_model_name):
        """Save the ultimate model and all preprocessing components."""
        print(f"\n SAVING ULTIMATE MODEL SYSTEM")

        # Create models directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        models_dir = os.path.join(project_root, "models")

        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)

        # Save best model
        if best_model_name in self.models:
            best_model = self.models[best_model_name]
            joblib.dump(best_model, os.path.join(models_dir, 'ultimate_best_model.joblib'))
            print(f"    Saved best model ({best_model_name}): ultimate_best_model.joblib")

        # Save all individual trained models with clear naming
        saved_models = {}
        for name, model in self.models.items():
            safe_name = name.lower().replace(' ', '_')
            filename = f'model_{safe_name}.joblib'
            joblib.dump(model, os.path.join(models_dir, filename))
            saved_models[filename] = name

        # Save shared components
        joblib.dump(self.scaler, os.path.join(models_dir, 'ultimate_scaler.joblib'))
        joblib.dump(self.feature_engineering, os.path.join(models_dir, 'ultimate_feature_engineering.joblib'))

        # Save ensemble config if available
        if hasattr(self, 'ensemble_config'):
            joblib.dump(self.ensemble_config, os.path.join(models_dir, 'ensemble_config.joblib'))

        # Save metadata with model name mapping
        metadata = {
            'best_model_name': best_model_name,
            'feature_names': list(self.X.columns),
            'num_features': len(self.X.columns),
            'model_performance': self.results[best_model_name],
            'all_results': {name: res for name, res in self.results.items()},
            'saved_models': saved_models,
            'feature_engineering_config': {
                'champion_characteristics_count': len(self.feature_engineering.champion_characteristics),
                'meta_strength_combinations': len(self.feature_engineering.champion_meta_strength),
                'team_compositions': len(self.feature_engineering.team_compositions)
            }
        }

        joblib.dump(metadata, os.path.join(models_dir, 'ultimate_model_metadata.joblib'))

        # Print summary of all saved files
        print(f"\n    Saved files:")
        print(f"      ultimate_best_model.joblib  -> {best_model_name}")
        for filename, model_name in saved_models.items():
            print(f"      {filename:<35s} -> {model_name}")
        print(f"      ultimate_scaler.joblib")
        print(f"      ultimate_feature_engineering.joblib")
        print(f"      ultimate_model_metadata.joblib")

    def analyze_robustness(self, best_model_name, run_validation_curves=True):
        """Analyze model robustness with learning and validation curves."""
        print(f"\n{'='*60}")
        print("ROBUSTNESS ANALYSIS")
        print(f"{'='*60}")

        best_model = self.models[best_model_name]
        use_scaled = self.results[best_model_name]['use_scaled']

        # Select appropriate data
        X_data = self.X_train_scaled if use_scaled else self.X_train
        y_data = self.y_train

        # Initialize robustness analyzer
        analyzer = RobustnessAnalyzer()

        # Generate learning curves
        print("\nGenerating learning curves...")
        learning_curves = analyzer.generate_learning_curves(
            best_model, X_data, y_data, cv=5
        )

        # Plot learning curves
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        viz_dir = os.path.join(project_root, "outputs", "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        analyzer.plot_learning_curves(
            learning_curves,
            output_path=os.path.join(viz_dir, 'learning_curves.png')
        )

        # Generate validation curves for appropriate parameters
        if run_validation_curves:
            param_configs = get_default_param_configs(best_model)
            if param_configs:
                print("\nGenerating validation curves...")
                for config in param_configs[:2]:  # Limit to 2 parameters
                    try:
                        val_curves = analyzer.generate_validation_curves(
                            best_model, X_data, y_data,
                            config['name'], config['range'], cv=5
                        )
                        analyzer.plot_validation_curves(
                            val_curves,
                            output_path=os.path.join(
                                viz_dir,
                                f"validation_curve_{config['name']}.png"
                            )
                        )
                    except Exception as e:
                        print(f"   Skipping {config['name']}: {e}")

        # SHAP-based explainability
        print("\nGenerating SHAP explanations...")
        try:
            # Sample background data for SHAP
            sample_size = min(500, len(X_data))
            if hasattr(X_data, 'sample'):
                X_background = X_data.sample(n=sample_size, random_state=42)
            else:
                indices = np.random.choice(len(X_data), sample_size, replace=False)
                X_background = X_data[indices]

            feature_names = list(self.X.columns)
            explainer = ModelExplainer(best_model, X_background.values if hasattr(X_background, 'values') else X_background, feature_names)

            # Generate global importance
            importance_df = explainer.global_importance()
            print("\nTop 10 SHAP Feature Importance:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                print(f"   {i}. {row['feature']}: {row['importance']:.4f}")

            # Save SHAP summary plot
            explainer.plot_summary(output_path=os.path.join(viz_dir, 'shap_summary.png'))

        except Exception as e:
            print(f"   SHAP analysis skipped: {e}")

        print(f"\nRobustness analysis complete. Visualizations saved to: {viz_dir}")

        return {
            'learning_curves': learning_curves,
            'diagnosis': learning_curves.get('diagnosis', 'Unknown')
        }

def main(quick_mode=False, use_stratified_temporal=False, use_enhanced_v2=False,
         use_temporal_weighting=False, calibrate_probs=True, quantify_uncertainty=True):
    """Main execution function.

    Args:
        quick_mode: Reduce hyperparameter search for faster training
        use_stratified_temporal: Use stratified temporal split (meta-aware)
        use_enhanced_v2: Use enhanced v2 features (side selection, patch transition, etc.)
        use_temporal_weighting: Apply temporal sample weighting during training
        calibrate_probs: Apply probability calibration on test set
        quantify_uncertainty: Compute bootstrap confidence intervals
    """
    print("ULTIMATE LEAGUE OF LEGENDS MATCH PREDICTION SYSTEM")
    if quick_mode:
        print("RUNNING IN QUICK MODE - Reduced training time")
    if use_stratified_temporal:
        print("USING STRATIFIED TEMPORAL SPLIT - Meta-aware methodology")
    else:
        print("USING PURE TEMPORAL SPLIT - Academic rigor methodology")
    if use_enhanced_v2:
        print("USING ENHANCED V2 FEATURES - Side selection, patch transition, extended interactions")
    if use_temporal_weighting:
        print("USING TEMPORAL WEIGHTING - Recent matches weighted higher")
    print("=" * 80)

    # Initialize ultimate predictor
    predictor = UltimateLoLPredictor()

    # Prepare advanced features (with optional v2 enhancements)
    X, y = predictor.prepare_advanced_features(use_enhanced_v2=use_enhanced_v2)

    # Split data using chosen methodology
    if use_stratified_temporal:
        predictor.split_data_stratified_temporal()
    else:
        predictor.split_data_temporally()

    # Apply temporal weighting if requested
    if use_temporal_weighting:
        sample_weights = predictor.apply_temporal_weighting()
    else:
        sample_weights = None

    # Train advanced model suite
    predictor.train_advanced_models(quick_mode=quick_mode)

    # Evaluate models
    best_model, results = predictor.evaluate_models()

    # Final test evaluation with calibration and uncertainty
    final_results = predictor.final_test_evaluation(
        best_model,
        calibrate_probs=calibrate_probs,
        quantify_uncertainty=quantify_uncertainty
    )

    # Save ultimate model
    predictor.save_ultimate_model(best_model)

    # Robustness analysis (optional - can be time consuming)
    if not quick_mode:
        robustness_results = predictor.analyze_robustness(best_model, run_validation_curves=True)

    print(f"\n{'='*60}")
    print("ULTIMATE SYSTEM TRAINING COMPLETE!")
    print(f"Best Model: {best_model}")
    print(f"Final Test F1: {final_results['test_f1']:.4f}")
    print(f"Final Test AUC: {final_results['test_auc']:.4f}")
    print(f"Final Test MCC: {final_results['test_mcc']:.4f}")
    print(f"Final Test Accuracy: {final_results['test_accuracy']:.4f}")
    print(f"Generalization Gap: {final_results['generalization_gap']:.4f}")

    # Print calibration summary if available
    if final_results.get('calibration_result'):
        cal = final_results['calibration_result']
        print(f"Calibration Improvement: {cal.improvement:.1%}")

    return predictor, final_results

if __name__ == "__main__":
    # Configuration options:
    #
    # Option 1: Pure temporal (academic rigor)
    # predictor, results = main(quick_mode=False, use_stratified_temporal=False)
    #
    # Option 2: Stratified temporal (meta-aware)
    # predictor, results = main(quick_mode=False, use_stratified_temporal=True)
    #
    # Option 3: Enhanced v2 features with temporal weighting
    # predictor, results = main(
    #     quick_mode=False,
    #     use_stratified_temporal=True,
    #     use_enhanced_v2=True,
    #     use_temporal_weighting=True
    # )
    #
    # Current: Stratified temporal with all enhancements
    predictor, results = main(
        quick_mode=False,
        use_stratified_temporal=True,
        use_enhanced_v2=True,
        use_temporal_weighting=False,  # Can be enabled
        calibrate_probs=True,
        quantify_uncertainty=True
    ) 