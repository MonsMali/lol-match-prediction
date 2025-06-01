import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import sys
import os
from datetime import datetime

# Bayesian Optimization imports
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.acquisition import gaussian_ei

# Add parent directory to path to import the feature engineering
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advanced_feature_engineering import AdvancedFeatureEngineering

warnings.filterwarnings('ignore')

class ComprehensiveLogisticRegressionComparison:
    """
    Comprehensive Logistic Regression Comparison with:
    - Three different data splitting strategies
    - Bayesian optimization for hyperparameter tuning
    - Nested Cross-Validation for robust evaluation
    - Individual visualizations for each approach
    - Comparative analysis
    """
    
    def __init__(self, data_path="../Dataset collection/target_leagues_dataset.csv"):
        # Fix path resolution to work from Experiment folder
        if not os.path.exists(data_path):
            # Try alternative path if running from different directory
            alternative_path = "Dataset collection/target_leagues_dataset.csv"
            if os.path.exists(alternative_path):
                data_path = alternative_path
            else:
                # Construct absolute path
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir)
                data_path = os.path.join(project_root, "Dataset collection", "target_leagues_dataset.csv")
        
        self.data_path = data_path
        self.feature_engineering = AdvancedFeatureEngineering(data_path)
        
        # Create organized output directories
        self.setup_output_directories()
        
        # Split ratio configurations for testing
        self.split_configs = {
            'standard': (0.70, 0.15, 0.15),  # Current approach
            'balanced': (0.60, 0.20, 0.20),  # More validation & test data
            'optimized': (0.70, 0.20, 0.10)  # Max training + robust validation
        }
        
        # Bayesian optimization search space
        self.bayesian_space = [
            Real(1e-4, 100, prior='log-uniform', name='C'),  # Regularization strength
            Categorical(['l1', 'l2', 'elasticnet'], name='penalty'),  # Penalty type
            Categorical(['liblinear', 'saga', 'lbfgs'], name='solver'),  # Solver
            Integer(2000, 6000, name='max_iter'),  # Enhanced iteration range
            Categorical(['balanced', None], name='class_weight'),  # Class balancing
            Real(0.1, 0.9, name='l1_ratio')  # For elasticnet penalty
        ]
        
        # Results storage for each strategy
        self.strategies = {
            'pure_temporal': {
                'name': 'Pure Temporal Split',
                'description': 'Original chronological split approach',
                'scaler': StandardScaler(),
                'results': {},
                'model': None,
                'nested_cv_results': {},
                'bayesian_history': []
            },
            'stratified_temporal': {
                'name': 'Stratified Temporal Split', 
                'description': '70/15/15 with year stratification',
                'scaler': StandardScaler(),
                'results': {},
                'model': None,
                'nested_cv_results': {},
                'bayesian_history': []
            },
            'stratified_random_temporal': {
                'name': 'Stratified Random Temporal Split',
                'description': 'Random stratified with temporal awareness',
                'scaler': StandardScaler(),
                'results': {},
                'model': None,
                'nested_cv_results': {},
                'bayesian_history': []
            }
        }
        
        # Data storage
        self.X = None
        self.y = None
        self.data_splits = {}
        self.current_split_config = 'standard'  # Default split configuration
        
    def setup_output_directories(self):
        """Create organized directory structure for outputs."""
        print("📁 SETTING UP OUTPUT DIRECTORIES")
        
        # Create main directories
        self.dirs = {
            'visualizations': 'Visualizations',
            'models': 'bayesian_optimized_models',
            'results': 'results'
        }
        
        for dir_name, dir_path in self.dirs.items():
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"   ✅ Created: {dir_path}/")
            else:
                print(f"   📁 Using existing: {dir_path}/")
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.dirs['results'], f'bayesian_comparison_{self.timestamp}')
        
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)
            print(f"   🆕 Created run directory: {self.run_dir}/")
        
        print("   📋 Directory structure ready!")
    
    def prepare_advanced_features(self):
        """Prepare the complete advanced feature set."""
        print("🚀 BAYESIAN OPTIMIZED LOGISTIC REGRESSION COMPARISON")
        print("=" * 80)
        print("🧠 Three Splitting Strategies + Bayesian Optimization + Nested CV")
        print("=" * 80)
        
        # Load and engineer features
        df = self.feature_engineering.load_and_analyze_data()
        advanced_features = self.feature_engineering.create_advanced_features_vectorized()
        final_features = self.feature_engineering.apply_advanced_encoding_optimized()
        
        # Get target variable
        self.X = final_features
        self.y = df['result']
        
        print(f"\n📊 FINAL FEATURE SUMMARY:")
        print(f"   Total features: {self.X.shape[1]}")
        print(f"   Total samples: {self.X.shape[0]}")
        print(f"   Class distribution: {self.y.value_counts().to_dict()}")
        
        return self.X, self.y
    
    def set_split_configuration(self, config_name):
        """Set the split configuration to use."""
        if config_name in self.split_configs:
            self.current_split_config = config_name
            train_ratio, val_ratio, test_ratio = self.split_configs[config_name]
            
            print(f"\n📊 SPLIT CONFIGURATION: {config_name.upper()}")
            print(f"   Train: {train_ratio:.0%} | Val: {val_ratio:.0%} | Test: {test_ratio:.0%}")
            
            # Check if X is available and get sample count estimates
            if self.X is not None:
                total_samples = len(self.X)
                train_samples = int(total_samples * train_ratio)
                val_samples = int(total_samples * val_ratio)
                test_samples = int(total_samples * test_ratio)
                print(f"   Estimated samples: ~{train_samples:,} | ~{val_samples:,} | ~{test_samples:,}")
            else:
                print(f"   Estimated samples: TBD (features not loaded yet)")
            
            # Update strategy descriptions
            for strategy in self.strategies.values():
                strategy['description'] = f"{strategy['description'].split(' with')[0]} with {train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%} split"
        else:
            print(f"❌ Unknown split configuration: {config_name}")
    
    def split_data_pure_temporal(self):
        """Pure temporal split with configurable ratios."""
        train_ratio, val_ratio, test_ratio = self.split_configs[self.current_split_config]
        
        print(f"\n📊 STRATEGY 1: PURE TEMPORAL SPLIT ({self.current_split_config.upper()})")
        print(f"   Strategy: Chronological order, {train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%} split")
        
        df = self.feature_engineering.df.copy()
        df_sorted = df.sort_values('date')
        
        total_samples = len(df_sorted)
        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))
        
        # Get indices
        train_indices = df_sorted.index[:train_end].tolist()
        val_indices = df_sorted.index[train_end:val_end].tolist()
        test_indices = df_sorted.index[val_end:].tolist()
        
        # Store splits
        self.data_splits['pure_temporal'] = {
            'X_train': self.X.loc[train_indices],
            'X_val': self.X.loc[val_indices],
            'X_test': self.X.loc[test_indices],
            'y_train': self.y.loc[train_indices],
            'y_val': self.y.loc[val_indices],
            'y_test': self.y.loc[test_indices]
        }
        
        # Print info
        splits = self.data_splits['pure_temporal']
        actual_train_ratio = len(splits['X_train']) / len(self.X)
        actual_val_ratio = len(splits['X_val']) / len(self.X)
        actual_test_ratio = len(splits['X_test']) / len(self.X)
        
        print(f"   📈 Training: {splits['X_train'].shape} ({actual_train_ratio:.1%})")
        print(f"   📊 Validation: {splits['X_val'].shape} ({actual_val_ratio:.1%})")
        print(f"   📉 Test: {splits['X_test'].shape} ({actual_test_ratio:.1%})")
        
        # Print date ranges
        print(f"   📅 Train: {df_sorted.loc[train_indices, 'date'].min()} to {df_sorted.loc[train_indices, 'date'].max()}")
        print(f"   📅 Val: {df_sorted.loc[val_indices, 'date'].min()} to {df_sorted.loc[val_indices, 'date'].max()}")
        print(f"   📅 Test: {df_sorted.loc[test_indices, 'date'].min()} to {df_sorted.loc[test_indices, 'date'].max()}")
    
    def split_data_stratified_temporal(self):
        """Stratified temporal split with configurable ratios."""
        train_ratio, val_ratio, test_ratio = self.split_configs[self.current_split_config]
        
        print(f"\n📊 STRATEGY 2: STRATIFIED TEMPORAL SPLIT ({self.current_split_config.upper()})")
        print(f"   Strategy: Year-wise stratification, {train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%} split")
        
        df = self.feature_engineering.df.copy()
        df_sorted = df.sort_values('date')
        years = sorted(df_sorted['year'].unique())
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for year in years:
            year_data = df_sorted[df_sorted['year'] == year].sort_values('date')
            year_size = len(year_data)
            
            if year_size < 10:
                continue
            
            train_end = int(year_size * train_ratio)
            val_end = int(year_size * (train_ratio + val_ratio))
            
            year_indices = year_data.index.tolist()
            train_indices.extend(year_indices[:train_end])
            val_indices.extend(year_indices[train_end:val_end])
            test_indices.extend(year_indices[val_end:])
        
        # Store splits
        self.data_splits['stratified_temporal'] = {
            'X_train': self.X.loc[train_indices],
            'X_val': self.X.loc[val_indices],
            'X_test': self.X.loc[test_indices],
            'y_train': self.y.loc[train_indices],
            'y_val': self.y.loc[val_indices],
            'y_test': self.y.loc[test_indices]
        }
        
        # Print info
        splits = self.data_splits['stratified_temporal']
        actual_train_ratio = len(splits['X_train']) / len(self.X)
        actual_val_ratio = len(splits['X_val']) / len(self.X)
        actual_test_ratio = len(splits['X_test']) / len(self.X)
        
        print(f"   📈 Training: {splits['X_train'].shape} ({actual_train_ratio:.1%})")
        print(f"   📊 Validation: {splits['X_val'].shape} ({actual_val_ratio:.1%})")
        print(f"   📉 Test: {splits['X_test'].shape} ({actual_test_ratio:.1%})")
        
        # Print year distributions
        train_years = df_sorted.loc[train_indices, 'year'].value_counts().sort_index()
        print(f"   📅 Year distribution: {train_years.to_dict()}")
    
    def split_data_stratified_random_temporal(self):
        """Stratified random temporal split with configurable ratios."""
        train_ratio, val_ratio, test_ratio = self.split_configs[self.current_split_config]
        
        print(f"\n📊 STRATEGY 3: STRATIFIED RANDOM TEMPORAL SPLIT ({self.current_split_config.upper()})")
        print(f"   Strategy: Random {train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%} within each year (patch-aware)")
        
        df = self.feature_engineering.df.copy()
        df_sorted = df.sort_values('date')
        years = sorted(df_sorted['year'].unique())
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        print(f"   📅 Years available: {years}")
        
        for year in years:
            year_data = df_sorted[df_sorted['year'] == year]
            year_size = len(year_data)
            
            if year_size < 10:
                print(f"   ⚠️ Skipping {year}: only {year_size} matches")
                continue
            
            # Get year indices and targets for stratified split
            year_indices = year_data.index.tolist()
            year_targets = self.y.loc[year_indices]
            
            # Try stratified split, fall back to regular split if not enough samples per class
            try:
                # First split: train+val vs test
                temp_indices, test_year_indices = train_test_split(
                    year_indices, 
                    test_size=test_ratio, 
                    stratify=year_targets,
                    random_state=42
                )
                
                # Second split: train vs val 
                temp_targets = self.y.loc[temp_indices]
                val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)  # Adjust for remaining data
                
                train_year_indices, val_year_indices = train_test_split(
                    temp_indices,
                    test_size=val_ratio_adjusted,
                    stratify=temp_targets,
                    random_state=42
                )
                
            except ValueError:
                # Fallback to regular random split if stratification fails
                print(f"   ⚠️ {year}: Using random split (insufficient samples per class)")
                temp_indices, test_year_indices = train_test_split(
                    year_indices, test_size=test_ratio, random_state=42
                )
                val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
                train_year_indices, val_year_indices = train_test_split(
                    temp_indices, test_size=val_ratio_adjusted, random_state=42
                )
            
            train_indices.extend(train_year_indices)
            val_indices.extend(val_year_indices)
            test_indices.extend(test_year_indices)
            
            print(f"   {year}: {len(train_year_indices)} train, {len(val_year_indices)} val, {len(test_year_indices)} test")
        
        # Store splits
        self.data_splits['stratified_random_temporal'] = {
            'X_train': self.X.loc[train_indices],
            'X_val': self.X.loc[val_indices],
            'X_test': self.X.loc[test_indices],
            'y_train': self.y.loc[train_indices],
            'y_val': self.y.loc[val_indices],
            'y_test': self.y.loc[test_indices]
        }
        
        # Print info
        splits = self.data_splits['stratified_random_temporal']
        actual_train_ratio = len(splits['X_train']) / len(self.X)
        actual_val_ratio = len(splits['X_val']) / len(self.X)
        actual_test_ratio = len(splits['X_test']) / len(self.X)
        
        print(f"\n   📊 FINAL STRATIFIED RANDOM TEMPORAL SPLIT:")
        print(f"   📈 Training: {splits['X_train'].shape} ({actual_train_ratio:.1%})")
        print(f"   📊 Validation: {splits['X_val'].shape} ({actual_val_ratio:.1%})")
        print(f"   📉 Test: {splits['X_test'].shape} ({actual_test_ratio:.1%})")
        
        # Print class distributions
        print(f"   📋 Train classes: {splits['y_train'].value_counts(normalize=True).round(3).to_dict()}")
        print(f"   📋 Val classes: {splits['y_val'].value_counts(normalize=True).round(3).to_dict()}")
        print(f"   📋 Test classes: {splits['y_test'].value_counts(normalize=True).round(3).to_dict()}")
        
        # Print year distributions
        train_years = df_sorted.loc[train_indices, 'year'].value_counts().sort_index()
        print(f"   📅 Year distribution: {train_years.to_dict()}")
    
    def validate_hyperparameters(self, params):
        """Validate hyperparameter combinations for LogisticRegression."""
        # Handle solver constraints
        if params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']:
            params['solver'] = 'liblinear'
        elif params['penalty'] == 'l2' and params['solver'] not in ['lbfgs', 'liblinear', 'saga']:
            params['solver'] = 'lbfgs'
        elif params['penalty'] == 'elasticnet' and params['solver'] != 'saga':
            params['solver'] = 'saga'
        
        # Remove l1_ratio if not using elasticnet
        if params['penalty'] != 'elasticnet':
            params = {k: v for k, v in params.items() if k != 'l1_ratio'}
        
        return params
    
    def bayesian_objective(self, params_list, X_train, y_train, cv_folds, strategy_name):
        """Objective function for Bayesian optimization."""
        # Convert parameter list to dictionary
        param_names = ['C', 'penalty', 'solver', 'max_iter', 'class_weight', 'l1_ratio']
        params = dict(zip(param_names, params_list))
        
        # Validate parameters
        params = self.validate_hyperparameters(params)
        
        try:
            # Create model with parameters
            model = LogisticRegression(random_state=42, **params)
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, 
                                      scoring='roc_auc', n_jobs=-1)
            
            # Return negative score (minimization)
            mean_score = np.mean(cv_scores)
            
            # Store history for analysis
            self.strategies[strategy_name]['bayesian_history'].append({
                'params': params.copy(),
                'auc_mean': mean_score,
                'auc_std': np.std(cv_scores),
                'auc_scores': cv_scores.tolist()
            })
            
            return -mean_score  # Negative because gp_minimize minimizes
            
        except Exception as e:
            print(f"   ⚠️ Error with params {params}: {e}")
            return 0.5  # Return poor score for invalid combinations
    
    def perform_nested_cv_for_strategy(self, strategy_name, outer_cv_folds=5, n_calls=50):
        """Perform nested CV with Bayesian optimization for a specific strategy."""
        print(f"\n🧠 BAYESIAN NESTED CV FOR {self.strategies[strategy_name]['name'].upper()}")
        print("=" * 60)
        
        # Get data splits
        splits = self.data_splits[strategy_name]
        scaler = self.strategies[strategy_name]['scaler']
        
        # Scale data
        X_train_scaled = scaler.fit_transform(splits['X_train'])
        X_val_scaled = scaler.transform(splits['X_val'])
        X_test_scaled = scaler.transform(splits['X_test'])
        
        # Combine train and val for nested CV
        X_nested = np.vstack([X_train_scaled, X_val_scaled])
        y_nested = pd.concat([splits['y_train'], splits['y_val']])
        
        # Setup CV
        outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=42)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # Storage
        nested_scores = {
            'f1': [], 'accuracy': [], 'auc': [], 'precision': [], 'recall': []
        }
        best_params_per_fold = []
        fold_results = []
        
        print(f"🧠 Running Bayesian Nested CV with {n_calls} optimization calls per fold...")
        print(f"🎯 Primary metric: AUC (optimized via Gaussian Process)")
        
        # Outer CV loop
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_nested, y_nested), 1):
            print(f"\n   Fold {fold_idx}/{outer_cv_folds}:")
            
            X_train_fold = X_nested[train_idx]
            X_test_fold = X_nested[test_idx]
            y_train_fold = y_nested.iloc[train_idx]
            y_test_fold = y_nested.iloc[test_idx]
            
            # Create objective function for this fold
            def objective(params_list):
                return self.bayesian_objective(params_list, X_train_fold, y_train_fold, 
                                             inner_cv, strategy_name)
            
            # Run Bayesian optimization
            print(f"      🔄 Running Bayesian optimization...")
            
            result = gp_minimize(
                func=objective,
                dimensions=self.bayesian_space,
                n_calls=n_calls,
                random_state=42,
                acq_func='EI',  # Expected Improvement
                n_initial_points=10
            )
            
            # Get best parameters
            param_names = ['C', 'penalty', 'solver', 'max_iter', 'class_weight', 'l1_ratio']
            best_params = dict(zip(param_names, result.x))
            best_params = self.validate_hyperparameters(best_params)
            best_params_per_fold.append(best_params)
            
            # Train best model and evaluate
            best_model = LogisticRegression(random_state=42, **best_params)
            best_model.fit(X_train_fold, y_train_fold)
            
            y_pred_fold = best_model.predict(X_test_fold)
            y_pred_proba_fold = best_model.predict_proba(X_test_fold)[:, 1]
            
            fold_metrics = {
                'fold': fold_idx,
                'accuracy': accuracy_score(y_test_fold, y_pred_fold),
                'precision': precision_score(y_test_fold, y_pred_fold),
                'recall': recall_score(y_test_fold, y_pred_fold),
                'f1': f1_score(y_test_fold, y_pred_fold),
                'auc': roc_auc_score(y_test_fold, y_pred_proba_fold),
                'best_params': best_params,
                'bayesian_score': -result.fun  # Convert back to positive AUC
            }
            
            fold_results.append(fold_metrics)
            
            # Store scores for statistics
            for metric in nested_scores.keys():
                nested_scores[metric].append(fold_metrics[metric])
            
            print(f"      Best params: {best_params}")
            print(f"      🎯 AUC: {fold_metrics['auc']:.4f} | F1: {fold_metrics['f1']:.4f} | Accuracy: {fold_metrics['accuracy']:.4f}")
            print(f"      🧠 Bayesian opt score: {fold_metrics['bayesian_score']:.4f}")
        
        # Calculate statistics for all metrics
        nested_cv_results = {
            'scores': nested_scores,
            'fold_results': fold_results,
            'best_params_per_fold': best_params_per_fold
        }
        
        # Calculate means, stds, and confidence intervals for each metric
        for metric in nested_scores.keys():
            scores = nested_scores[metric]
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            nested_cv_results[f'mean_{metric}'] = mean_score
            nested_cv_results[f'std_{metric}'] = std_score
            nested_cv_results[f'{metric}_confidence_interval_95'] = (
                mean_score - 1.96 * std_score / np.sqrt(outer_cv_folds),
                mean_score + 1.96 * std_score / np.sqrt(outer_cv_folds)
            )
        
        self.strategies[strategy_name]['nested_cv_results'] = nested_cv_results
        
        print(f"\n📊 {self.strategies[strategy_name]['name'].upper()} BAYESIAN NESTED CV RESULTS:")
        print(f"   🎯 Mean AUC: {nested_cv_results['mean_auc']:.4f} ± {nested_cv_results['std_auc']:.4f} (PRIMARY METRIC)")
        print(f"   📈 Mean F1: {nested_cv_results['mean_f1']:.4f} ± {nested_cv_results['std_f1']:.4f}")
        print(f"   📊 Mean Accuracy: {nested_cv_results['mean_accuracy']:.4f} ± {nested_cv_results['std_accuracy']:.4f}")
        print(f"   📊 AUC 95% CI: [{nested_cv_results['auc_confidence_interval_95'][0]:.4f}, {nested_cv_results['auc_confidence_interval_95'][1]:.4f}]")
        print(f"   🎯 Individual AUC scores: {[f'{score:.4f}' for score in nested_scores['auc']]}")
        
        # Bayesian optimization analysis
        print(f"\n🧠 BAYESIAN OPTIMIZATION ANALYSIS:")
        max_iter_used = []
        penalties_used = []
        for params in best_params_per_fold:
            max_iter_used.append(params.get('max_iter', 'unknown'))
            penalties_used.append(params.get('penalty', 'unknown'))
        
        print(f"   🔄 Max iterations used per fold: {max_iter_used}")
        print(f"   📊 Penalties chosen: {penalties_used}")
        print(f"   🧠 Total evaluations: {len(self.strategies[strategy_name]['bayesian_history'])}")
        
        # Show best discovered parameters
        history = self.strategies[strategy_name]['bayesian_history']
        if history:
            best_eval = max(history, key=lambda x: x['auc_mean'])
            print(f"   🏆 Best discovered AUC: {best_eval['auc_mean']:.4f}")
            print(f"   🏆 Best params: {best_eval['params']}")
        
        return nested_cv_results
    
    def train_final_model_for_strategy(self, strategy_name):
        """Train final model for a strategy using Bayesian optimization results."""
        print(f"\n🏆 TRAINING FINAL MODEL: {self.strategies[strategy_name]['name'].upper()}")
        
        # Get most common parameters from Bayesian optimization
        param_combinations = self.strategies[strategy_name]['nested_cv_results']['best_params_per_fold']
        
        # Find most frequent parameter combination or use best performing one
        if len(param_combinations) > 1:
            # Use parameter combination that appears most frequently
            param_strings = [str(sorted(params.items())) for params in param_combinations]
            from collections import Counter
            most_common_params_str = Counter(param_strings).most_common(1)[0][0]
            most_common_params = dict(eval(most_common_params_str))
        else:
            most_common_params = param_combinations[0]
        
        # Alternative: use the parameter set with best average performance from history
        history = self.strategies[strategy_name]['bayesian_history']
        if history:
            best_from_history = max(history, key=lambda x: x['auc_mean'])
            print(f"   🧠 Best from Bayesian history: AUC = {best_from_history['auc_mean']:.4f}")
            
            # Use best from history if significantly better
            cv_mean = self.strategies[strategy_name]['nested_cv_results']['mean_auc']
            if best_from_history['auc_mean'] > cv_mean + 0.005:  # 0.5% improvement threshold
                most_common_params = best_from_history['params']
                print(f"   🎯 Using best Bayesian params instead of most common")
        
        print(f"   🧠 Final parameters: {most_common_params}")
        
        # Check convergence settings
        max_iter = most_common_params.get('max_iter', 2000)
        penalty = most_common_params.get('penalty', 'l2')
        print(f"   📊 Penalty: {penalty}, Max iterations: {max_iter}")
        
        if max_iter >= 4000:
            print(f"   ✅ High iteration count ({max_iter}) - excellent convergence potential")
        elif max_iter >= 2000:
            print(f"   📊 Medium iteration count ({max_iter}) - good convergence balance")
        else:
            print(f"   ⚠️ Low iteration count ({max_iter}) - may need monitoring")
        
        # Get data
        splits = self.data_splits[strategy_name]
        scaler = self.strategies[strategy_name]['scaler']
        
        # Scale all data splits
        X_train_scaled = scaler.fit_transform(splits['X_train'])
        X_val_scaled = scaler.transform(splits['X_val'])
        X_test_scaled = scaler.transform(splits['X_test'])
        
        # Train model on training data only (for validation evaluation)
        temp_model = LogisticRegression(random_state=42, **most_common_params)
        temp_model.fit(X_train_scaled, splits['y_train'])
        
        # Check for convergence warnings
        if hasattr(temp_model, 'n_iter_'):
            actual_iterations = temp_model.n_iter_[0] if hasattr(temp_model.n_iter_, '__len__') else temp_model.n_iter_
            print(f"   📊 Actual iterations used: {actual_iterations}/{max_iter}")
            if actual_iterations >= max_iter * 0.9:
                print(f"   ⚠️ Near max iterations - may need higher limit")
            else:
                print(f"   ✅ Converged efficiently")
        
        # Evaluate on validation set
        y_pred_val = temp_model.predict(X_val_scaled)
        y_pred_proba_val = temp_model.predict_proba(X_val_scaled)[:, 1]
        
        val_metrics = {
            'val_accuracy': accuracy_score(splits['y_val'], y_pred_val),
            'val_precision': precision_score(splits['y_val'], y_pred_val),
            'val_recall': recall_score(splits['y_val'], y_pred_val),
            'val_f1': f1_score(splits['y_val'], y_pred_val),
            'val_auc': roc_auc_score(splits['y_val'], y_pred_proba_val)
        }
        
        # Train final model on combined train+val data
        X_final_train = pd.concat([splits['X_train'], splits['X_val']])
        y_final_train = pd.concat([splits['y_train'], splits['y_val']])
        
        X_final_train_scaled = scaler.fit_transform(X_final_train)
        X_test_scaled = scaler.transform(splits['X_test'])
        
        final_model = LogisticRegression(random_state=42, **most_common_params)
        final_model.fit(X_final_train_scaled, y_final_train)
        
        # Check final model convergence
        if hasattr(final_model, 'n_iter_'):
            final_iterations = final_model.n_iter_[0] if hasattr(final_model.n_iter_, '__len__') else final_model.n_iter_
            print(f"   📊 Final model iterations: {final_iterations}/{max_iter}")
        
        # Evaluate on test set
        y_pred_test = final_model.predict(X_test_scaled)
        y_pred_proba_test = final_model.predict_proba(X_test_scaled)[:, 1]
        
        test_metrics = {
            'accuracy': accuracy_score(splits['y_test'], y_pred_test),
            'precision': precision_score(splits['y_test'], y_pred_test),
            'recall': recall_score(splits['y_test'], y_pred_test),
            'f1': f1_score(splits['y_test'], y_pred_test),
            'auc': roc_auc_score(splits['y_test'], y_pred_proba_test)
        }
        
        # Combine validation and test metrics
        all_metrics = {**val_metrics, **test_metrics}
        
        self.strategies[strategy_name]['model'] = final_model
        self.strategies[strategy_name]['results'] = all_metrics
        self.strategies[strategy_name]['X_test_scaled'] = X_test_scaled
        
        print(f"   ✅ Validation - AUC: {val_metrics['val_auc']:.4f}, Accuracy: {val_metrics['val_accuracy']:.4f}")
        print(f"   ✅ Test - AUC: {test_metrics['auc']:.4f}, Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")
        
        return final_model, all_metrics
    
    def create_individual_visualizations(self, strategy_name):
        """Create individual visualizations for a strategy."""
        print(f"\n📊 Creating visualizations for {self.strategies[strategy_name]['name']}")
        
        strategy = self.strategies[strategy_name]
        nested_cv = strategy['nested_cv_results']
        model = strategy['model']
        results = strategy['results']
        splits = self.data_splits[strategy_name]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{strategy["name"]} - Comprehensive Analysis (AUC-Focused)', fontsize=16, fontweight='bold')
        
        # 1. AUC Scores Distribution
        ax1 = axes[0, 0]
        auc_scores = nested_cv['scores']['auc']
        ax1.hist(auc_scores, bins=5, alpha=0.7, color='gold', edgecolor='black')
        ax1.axvline(np.mean(auc_scores), color='red', linestyle='--', label=f'Mean: {np.mean(auc_scores):.4f}')
        ax1.set_xlabel('AUC Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Nested CV AUC Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. AUC Scores by Fold
        ax2 = axes[0, 1]
        folds = range(1, len(auc_scores) + 1)
        ax2.plot(folds, auc_scores, 'o-', color='darkorange', linewidth=2, markersize=8)
        ax2.fill_between(folds, 
                        [np.mean(auc_scores) - np.std(auc_scores)] * len(folds),
                        [np.mean(auc_scores) + np.std(auc_scores)] * len(folds),
                        alpha=0.2, color='orange')
        ax2.set_xlabel('CV Fold')
        ax2.set_ylabel('AUC Score')
        ax2.set_title('AUC Score by CV Fold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Comprehensive Performance Metrics
        ax3 = axes[0, 2]
        metrics = ['Val_AUC', 'Test_AUC', 'Val_Acc', 'Test_Acc', 'Val_F1', 'Test_F1']
        values = [
            results['val_auc'], results['auc'], 
            results['val_accuracy'], results['accuracy'],
            results['val_f1'], results['f1']
        ]
        colors = ['gold', 'orange', 'lightblue', 'skyblue', 'lightgreen', 'green']
        
        bars = ax3.bar(metrics, values, color=colors, alpha=0.7)
        ax3.set_ylim(0, 1)
        ax3.set_ylabel('Score')
        ax3.set_title('Validation vs Test Performance')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Confusion Matrix
        ax4 = axes[1, 0]
        y_pred_test = model.predict(strategy['X_test_scaled'])
        cm = confusion_matrix(splits['y_test'], y_pred_test)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax4)
        ax4.set_title('Test Set Confusion Matrix')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        
        # 5. Feature Importance (Top 15)
        ax5 = axes[1, 1]
        if hasattr(model, 'coef_'):
            feature_importance = pd.DataFrame({
                'feature': self.X.columns,
                'coefficient': model.coef_[0],
                'abs_coefficient': np.abs(model.coef_[0])
            }).sort_values('abs_coefficient', ascending=False)
            
            top_features = feature_importance.head(15)
            colors = ['red' if coef < 0 else 'blue' for coef in top_features['coefficient']]
            
            ax5.barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)
            ax5.set_yticks(range(len(top_features)))
            ax5.set_yticklabels(top_features['feature'], fontsize=8)
            ax5.set_xlabel('Coefficient Value')
            ax5.set_title('Top 15 Feature Coefficients')
            ax5.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax5.invert_yaxis()
        
        # 6. AUC Performance Comparison (CV vs Val vs Test)
        ax6 = axes[1, 2]
        auc_comparison_data = {
            'CV Mean': nested_cv['mean_auc'],
            'Validation': results['val_auc'],
            'Test': results['auc']
        }
        bars = ax6.bar(auc_comparison_data.keys(), auc_comparison_data.values(), 
                      color=['orange', 'gold', 'darkorange'], alpha=0.7)
        ax6.set_ylabel('AUC Score')
        ax6.set_title('AUC: CV vs Validation vs Test')
        ax6.set_ylim(0, 1)
        
        # Add error bars for CV
        ax6.errorbar([0], [nested_cv['mean_auc']], yerr=[nested_cv['std_auc']], 
                    fmt='none', color='black', capsize=5)
        
        for bar, value in zip(bars, auc_comparison_data.values()):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save to organized directory
        viz_path = os.path.join(self.dirs['visualizations'], f'{strategy_name}_comprehensive_analysis.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {viz_path}")
    
    def create_comparative_analysis(self):
        """Create comprehensive comparative analysis visualizations."""
        print(f"\n📊 CREATING COMPARATIVE ANALYSIS")
        
        # Check if all strategies have completed nested CV
        completed_strategies = {}
        for strategy_key, strategy in self.strategies.items():
            if 'nested_cv_results' in strategy and 'mean_auc' in strategy['nested_cv_results']:
                completed_strategies[strategy_key] = strategy
            else:
                print(f"   ⚠️ Skipping {strategy['name']} - nested CV not completed")
        
        if len(completed_strategies) < 2:
            print(f"   ❌ Need at least 2 completed strategies for comparison. Found: {len(completed_strategies)}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Strategy Comparison', fontsize=16, fontweight='bold')
        
        # 1. F1 Score Comparison
        ax1 = axes[0, 0]
        strategies_names = [completed_strategies[k]['name'] for k in completed_strategies.keys()]
        cv_means = [completed_strategies[k]['nested_cv_results']['mean_auc'] for k in completed_strategies.keys()]
        cv_stds = [completed_strategies[k]['nested_cv_results']['std_auc'] for k in completed_strategies.keys()]
        test_f1s = [completed_strategies[k]['results']['auc'] for k in completed_strategies.keys()]
        
        x_pos = np.arange(len(strategies_names))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, cv_means, width, label='CV Mean', 
                       color='skyblue', alpha=0.7, yerr=cv_stds, capsize=5)
        bars2 = ax1.bar(x_pos + width/2, test_f1s, width, label='Test AUC', 
                       color='lightcoral', alpha=0.7)
        
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('AUC')
        ax1.set_title('AUC Comparison: CV vs Test')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([name.replace(' ', '\n') for name in strategies_names], fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars2, test_f1s):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. All Metrics Comparison
        ax2 = axes[0, 1]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        strategy_keys = list(completed_strategies.keys())
        
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, strategy_key in enumerate(strategy_keys):
            values = [completed_strategies[strategy_key]['results'][m.lower()] for m in metrics]
            ax2.bar(x + i*width, values, width, 
                   label=completed_strategies[strategy_key]['name'], alpha=0.7)
        
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Score')
        ax2.set_title('Comprehensive Metrics Comparison')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Generalization Gap Analysis
        ax3 = axes[1, 0]
        gaps = [completed_strategies[k]['nested_cv_results']['mean_auc'] - completed_strategies[k]['results']['auc'] 
                for k in completed_strategies.keys()]
        colors = ['green' if gap >= 0 else 'red' for gap in gaps]
        
        bars = ax3.bar(range(len(strategies_names)), gaps, color=colors, alpha=0.7)
        ax3.set_xlabel('Strategy')
        ax3.set_ylabel('Generalization Gap (CV - Test)')
        ax3.set_title('Generalization Gap Analysis')
        ax3.set_xticks(range(len(strategies_names)))
        ax3.set_xticklabels([name.replace(' ', '\n') for name in strategies_names], fontsize=10)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.grid(True, alpha=0.3)
        
        for bar, gap in zip(bars, gaps):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.002 if gap >= 0 else -0.005),
                    f'{gap:+.3f}', ha='center', va='bottom' if gap >= 0 else 'top', fontsize=9)
        
        # 4. Confidence Intervals
        ax4 = axes[1, 1]
        cis_lower = [completed_strategies[k]['nested_cv_results']['auc_confidence_interval_95'][0] for k in completed_strategies.keys()]
        cis_upper = [completed_strategies[k]['nested_cv_results']['auc_confidence_interval_95'][1] for k in completed_strategies.keys()]
        
        for i, (strategy_key, lower, upper, mean) in enumerate(zip(strategy_keys, cis_lower, cis_upper, cv_means)):
            ax4.errorbar(i, mean, yerr=[[mean - lower], [upper - mean]], 
                        fmt='o', capsize=10, capthick=2, markersize=8,
                        label=completed_strategies[strategy_key]['name'])
        
        ax4.set_xlabel('Strategy')
        ax4.set_ylabel('AUC')
        ax4.set_title('95% Confidence Intervals')
        ax4.set_xticks(range(len(strategies_names)))
        ax4.set_xticklabels([name.replace(' ', '\n') for name in strategies_names], fontsize=10)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to organized directory
        viz_path = os.path.join(self.dirs['visualizations'], 'comprehensive_strategy_comparison.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {viz_path}")
        print(f"   📊 Compared {len(completed_strategies)} completed strategies")
    
    def save_comprehensive_results(self):
        """Save all results to organized directories."""
        print(f"\n💾 SAVING BAYESIAN OPTIMIZATION RESULTS")
        
        # Create summary dataframe
        summary_data = []
        for strategy_key, strategy in self.strategies.items():
            nested_cv = strategy['nested_cv_results']
            results = strategy['results']
            
            summary_data.append({
                'Strategy': strategy['name'],
                'Description': strategy['description'],
                'CV_AUC_Mean': nested_cv['mean_auc'],
                'CV_AUC_Std': nested_cv['std_auc'],
                'CV_F1_Mean': nested_cv['mean_f1'],
                'CV_F1_Std': nested_cv['std_f1'],
                'CV_Accuracy_Mean': nested_cv['mean_accuracy'],
                'CV_Accuracy_Std': nested_cv['std_accuracy'],
                'AUC_CI_Lower': nested_cv['auc_confidence_interval_95'][0],
                'AUC_CI_Upper': nested_cv['auc_confidence_interval_95'][1],
                'Val_Accuracy': results['val_accuracy'],
                'Val_AUC': results['val_auc'],
                'Val_F1': results['val_f1'],
                'Test_Accuracy': results['accuracy'],
                'Test_Precision': results['precision'],
                'Test_Recall': results['recall'],
                'Test_F1': results['f1'],
                'Test_AUC': results['auc'],
                'Generalization_Gap_AUC': nested_cv['mean_auc'] - results['auc'],
                'Generalization_Gap_F1': nested_cv['mean_f1'] - results['f1'],
                'Bayesian_Evaluations': len(strategy['bayesian_history'])
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to results directory
        summary_path = os.path.join(self.run_dir, 'bayesian_comparison_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Save full results including Bayesian history
        full_results = {
            'experiment_info': {
                'date': datetime.now().isoformat(),
                'strategies_tested': len(self.strategies),
                'dataset_size': len(self.X),
                'feature_count': len(self.X.columns),
                'optimization_method': 'Bayesian (Gaussian Process)',
                'hyperparameter_search_space': str(self.bayesian_space),
                'convergence_monitoring': 'enabled',
                'iteration_range': '2000-6000'
            },
            'strategies': self.strategies
        }
        
        results_path = os.path.join(self.run_dir, 'bayesian_comparison_results.joblib')
        joblib.dump(full_results, results_path)
        
        # Save individual models to bayesian_optimized_models directory
        best_strategy = summary_df.loc[summary_df['Test_AUC'].idxmax()]
        best_strategy_key = [k for k, v in self.strategies.items() if v['name'] == best_strategy['Strategy']][0]
        
        for strategy_key, strategy in self.strategies.items():
            model_path = os.path.join(self.dirs['models'], f'{strategy_key}_bayesian_model.joblib')
            scaler_path = os.path.join(self.dirs['models'], f'{strategy_key}_bayesian_scaler.joblib')
            history_path = os.path.join(self.dirs['models'], f'{strategy_key}_bayesian_history.joblib')
            
            joblib.dump(strategy['model'], model_path)
            joblib.dump(strategy['scaler'], scaler_path)
            joblib.dump(strategy['bayesian_history'], history_path)
            
            # Save best model with special name
            if strategy_key == best_strategy_key:
                bayesian_best_path = os.path.join(self.dirs['models'], 'bayesian_best_model_Logistic_Regression.joblib')
                bayesian_package = {
                    'model': strategy['model'],
                    'scaler': strategy['scaler'],
                    'model_name': f"Bayesian Optimized Logistic Regression - {strategy['name']}",
                    'strategy': strategy['name'],
                    'performance': strategy['results'],
                    'feature_columns': list(self.X.columns),
                    'training_date': datetime.now().isoformat(),
                    'optimization_method': 'Bayesian (Gaussian Process)',
                    'bayesian_history': strategy['bayesian_history'],
                    'total_evaluations': len(strategy['bayesian_history']),
                    'convergence_optimized': True
                }
                joblib.dump(bayesian_package, bayesian_best_path)
                print(f"   🏆 Bayesian best model: {bayesian_best_path}")
        
        print(f"   ✅ Saved: {summary_path}")
        print(f"   ✅ Saved: {results_path}")
        print(f"   ✅ Saved: Individual models + Bayesian histories in {self.dirs['models']}/")
        print(f"   📁 All results organized in: {self.run_dir}/")
        
        return summary_df

def main():
    """Main execution function."""
    print("🎯 BAYESIAN OPTIMIZED LOGISTIC REGRESSION COMPARISON")
    print("🧠 Advanced ML with Bayesian Hyperparameter Optimization")
    print("=" * 80)
    
    # Ask user for execution mode
    print("\n🎮 EXECUTION MODE SELECTION:")
    print("   1️⃣  Full Comparison (All 3 strategies) - Recommended")
    print("   2️⃣  Focus Mode (Top 2 performers only)")
    print("   3️⃣  Winner Focus (Stratified Temporal only)")
    print("   4️⃣  Split Strategy Analysis (Test different split ratios)")
    
    mode = input("\nSelect mode (1/2/3/4, default 1): ").strip()
    
    # Initialize comparison
    comparison = ComprehensiveLogisticRegressionComparison()
    
    # Step 1: Prepare features
    X, y = comparison.prepare_advanced_features()
    
    # Handle split strategy analysis mode
    if mode == "4":
        return run_split_strategy_analysis(comparison)
    
    # Ask for split configuration
    print("\n📊 SPLIT RATIO SELECTION:")
    print("   🔸 Standard (70/15/15): Balanced approach")
    print("   🔸 Balanced (60/20/20): More robust validation & test")
    print("   🔸 Optimized (70/20/10): Max training + robust validation [RECOMMENDED]")
    
    split_choice = input("\nSelect split (standard/balanced/optimized, default optimized): ").strip().lower()
    
    if split_choice in ['balanced', 'standard']:
        comparison.set_split_configuration(split_choice)
    else:
        comparison.set_split_configuration('optimized')  # Default to recommended
    
    # Ask for Bayesian optimization settings
    print("\n🧠 BAYESIAN OPTIMIZATION SETTINGS:")
    print("   🔸 Quick (30 calls per fold): ~10 minutes per strategy")
    print("   🔸 Balanced (50 calls per fold): ~15 minutes per strategy [RECOMMENDED]")
    print("   🔸 Thorough (100 calls per fold): ~30 minutes per strategy")
    print("   🔸 Extensive (200 calls per fold): ~60 minutes per strategy")
    
    opt_choice = input("\nSelect optimization depth (quick/balanced/thorough/extensive, default balanced): ").strip().lower()
    
    n_calls_map = {
        'quick': 30,
        'balanced': 50,
        'thorough': 100,
        'extensive': 200
    }
    n_calls = n_calls_map.get(opt_choice, 50)  # Default to balanced
    
    # Step 2: Create all data splits
    comparison.split_data_pure_temporal()
    comparison.split_data_stratified_temporal()
    comparison.split_data_stratified_random_temporal()
    
    # Determine which strategies to run
    if mode == "3":
        strategies_to_run = ['stratified_temporal']
        print(f"\n🎯 WINNER FOCUS MODE: Running Stratified Temporal only")
    elif mode == "2":
        strategies_to_run = ['stratified_temporal', 'stratified_random_temporal']
        print(f"\n🥇 TOP PERFORMERS MODE: Running Stratified Temporal + Stratified Random")
    else:
        strategies_to_run = list(comparison.strategies.keys())
        print(f"\n🎯 FULL COMPARISON MODE: Running all 3 strategies")
    
    print(f"   🧠 Bayesian optimization: {n_calls} calls per fold")
    print(f"   📊 Search space: Continuous C, Categorical penalties, 2000-6000 iterations")
    print(f"   ⏱️  Estimated time: {len(strategies_to_run) * (n_calls//10 + 5)} minutes")
    
    # Ask for confirmation
    confirm = input(f"\nProceed with {len(strategies_to_run)} strategy(ies)? (y/n, default y): ").strip().lower()
    if confirm == 'n':
        print("Cancelled by user")
        return None, None
    
    # Step 3: Train and evaluate selected strategies
    for strategy_name in strategies_to_run:
        if strategy_name not in comparison.strategies:
            continue
            
        print(f"\n{'='*60}")
        print(f"🧠 PROCESSING STRATEGY: {comparison.strategies[strategy_name]['name'].upper()}")
        print(f"{'='*60}")
        
        # Nested CV with Bayesian optimization
        comparison.perform_nested_cv_for_strategy(strategy_name, n_calls=n_calls)
        
        # Train final model with best Bayesian parameters
        comparison.train_final_model_for_strategy(strategy_name)
        
        # Create individual visualizations
        comparison.create_individual_visualizations(strategy_name)
    
    # Step 4: Comparative analysis (only if multiple strategies)
    if len(strategies_to_run) > 1:
        comparison.create_comparative_analysis()
    else:
        print(f"\n📊 Single strategy mode - skipping comparative analysis")
    
    # Step 5: Save results
    summary_df = comparison.save_comprehensive_results()
    
    # Final summary
    print(f"\n🎉 BAYESIAN OPTIMIZATION COMPLETE!")
    print("=" * 80)
    
    if len(strategies_to_run) > 1:
        best_strategy = summary_df.loc[summary_df['Test_AUC'].idxmax()]
        print(f"🏆 BEST STRATEGY: {best_strategy['Strategy']}")
        print(f"📊 Test AUC: {best_strategy['Test_AUC']:.4f}")
        print(f"🎯 CV AUC: {best_strategy['CV_AUC_Mean']:.4f} ± {best_strategy['CV_AUC_Std']:.4f}")
        print(f"📈 Generalization Gap: {best_strategy['Generalization_Gap_AUC']:+.4f}")
        
        print(f"\n📋 BAYESIAN OPTIMIZATION SUMMARY:")
        for _, row in summary_df.iterrows():
            print(f"   {row['Strategy']}: Test AUC = {row['Test_AUC']:.4f}")
        
        print(f"\n🧠 BAYESIAN VS GRID SEARCH BENEFITS:")
        print(f"   ✅ Intelligent exploration of continuous parameter space")
        print(f"   ✅ Automatic convergence to optimal regions")
        print(f"   ✅ Efficient use of computational budget ({n_calls} calls vs 1000+ grid combinations)")
        print(f"   ✅ Gaussian Process learns from previous evaluations")
            
    else:
        strategy_result = summary_df.iloc[0]
        print(f"🎯 BAYESIAN FOCUSED ANALYSIS: {strategy_result['Strategy']}")
        print(f"📊 Test AUC: {strategy_result['Test_AUC']:.4f}")
        print(f"🎯 CV AUC: {strategy_result['CV_AUC_Mean']:.4f} ± {strategy_result['CV_AUC_Std']:.4f}")
    
    return comparison, summary_df

def run_split_strategy_analysis(comparison):
    """Run comprehensive analysis of different split strategies with Bayesian optimization."""
    print("\n🧪 SPLIT STRATEGY ANALYSIS MODE (Bayesian Optimized)")
    print("=" * 50)
    print("Testing all split configurations on the winning strategy (Stratified Temporal)")
    
    # Ask for optimization depth
    print("\n🧠 BAYESIAN OPTIMIZATION DEPTH:")
    print("   🔸 Quick (30 calls): ~10 minutes per split config")
    print("   🔸 Balanced (50 calls): ~15 minutes per split config")
    print("   🔸 Thorough (100 calls): ~30 minutes per split config")
    
    opt_choice = input("\nSelect optimization depth (quick/balanced/thorough, default balanced): ").strip().lower()
    n_calls_map = {'quick': 30, 'balanced': 50, 'thorough': 100}
    n_calls = n_calls_map.get(opt_choice, 50)
    
    # Only test the winning strategy with different splits
    strategy_to_test = 'stratified_temporal'
    split_configs = ['standard', 'balanced', 'optimized']
    
    all_results = []
    
    for split_config in split_configs:
        print(f"\n" + "="*60)
        print(f"🧪 TESTING SPLIT CONFIGURATION: {split_config.upper()}")
        print(f"="*60)
        
        # Set split configuration
        comparison.set_split_configuration(split_config)
        
        # Create splits for this configuration
        comparison.split_data_stratified_temporal()
        
        # Run nested CV with Bayesian optimization
        comparison.perform_nested_cv_for_strategy(strategy_to_test, n_calls=n_calls)
        
        # Train final model
        comparison.train_final_model_for_strategy(strategy_to_test)
        
        # Store results
        results = comparison.strategies[strategy_to_test]['results'].copy()
        nested_cv = comparison.strategies[strategy_to_test]['nested_cv_results'].copy()
        
        results['split_config'] = split_config
        results['cv_auc_mean'] = nested_cv['mean_auc']
        results['cv_auc_std'] = nested_cv['std_auc']
        
        all_results.append(results)
        
        print(f"\n📊 {split_config.upper()} RESULTS:")
        print(f"   CV AUC: {nested_cv['mean_auc']:.4f} ± {nested_cv['std_auc']:.4f}")
        print(f"   Test AUC: {results['auc']:.4f}")
        print(f"   Validation AUC: {results['val_auc']:.4f}")
    
    # Create comprehensive comparison
    print(f"\n" + "="*80)
    print(f"🏆 BAYESIAN SPLIT STRATEGY COMPARISON RESULTS")
    print(f"="*80)
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('auc', ascending=False)
    
    print(f"\n📊 RANKING BY TEST AUC:")
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        config_name = row['split_config'].upper()
        test_auc = row['auc']
        cv_auc = row['cv_auc_mean']
        val_auc = row['val_auc']
        
        print(f"   {i}. {config_name:>10}: Test={test_auc:.4f} | CV={cv_auc:.4f} | Val={val_auc:.4f}")
    
    # Statistical analysis
    best_config = results_df.iloc[0]
    print(f"\n🏆 RECOMMENDED SPLIT CONFIGURATION: {best_config['split_config'].upper()}")
    print(f"   📊 Test AUC: {best_config['auc']:.4f}")
    print(f"   🎯 CV AUC: {best_config['cv_auc_mean']:.4f} ± {best_config['cv_auc_std']:.4f}")
    print(f"   📈 Validation AUC: {best_config['val_auc']:.4f}")
    
    # Performance differences
    auc_diff_vs_standard = best_config['auc'] - results_df[results_df['split_config'] == 'standard']['auc'].iloc[0]
    print(f"   📈 Improvement over standard: {auc_diff_vs_standard:+.4f}")
    
    # Save results
    results_path = os.path.join(comparison.run_dir, 'bayesian_split_strategy_analysis.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved: {results_path}")
    
    return comparison, results_df

if __name__ == "__main__":
    comparison, summary = main() 