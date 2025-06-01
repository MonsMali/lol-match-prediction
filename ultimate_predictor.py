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
    print("⚠️ CatBoost not available. Install with: pip install catboost")
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from advanced_feature_engineering import AdvancedFeatureEngineering
warnings.filterwarnings('ignore')

class UltimateLoLPredictor:
    """
    Ultimate League of Legends match prediction system combining:
    - Advanced feature engineering
    - Multiple state-of-the-art ML algorithms
    - Rigorous train/validation/test split
    - Hyperparameter optimization
    - Ensemble methods
    """
    
    def __init__(self, data_path="Dataset collection/target_leagues_dataset.csv"):
        self.data_path = data_path
        self.feature_engineering = AdvancedFeatureEngineering(data_path)
        
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
        
    def prepare_advanced_features(self):
        """Prepare the complete advanced feature set."""
        print("🚀 ULTIMATE LOL MATCH PREDICTION SYSTEM")
        print("=" * 80)
        
        # Load and engineer features
        df = self.feature_engineering.load_and_analyze_data()
        advanced_features = self.feature_engineering.create_advanced_features()
        final_features = self.feature_engineering.apply_advanced_encoding()
        
        # Get target variable
        self.X = final_features
        self.y = df['result']
        
        print(f"\n📊 FINAL FEATURE SUMMARY:")
        print(f"   Total features: {self.X.shape[1]}")
        print(f"   Total samples: {self.X.shape[0]}")
        print(f"   Class distribution: {self.y.value_counts().to_dict()}")
        
        return self.X, self.y
    
    def split_data_temporally(self, train_size=0.6, val_size=0.2, test_size=0.2):
        """Split data temporally for realistic evaluation."""
        print(f"\n📊 TEMPORAL DATA SPLITTING")
        
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
        
        print(f"   📈 Training: {self.X_train.shape} ({train_size:.1%})")
        print(f"   📊 Validation: {self.X_val.shape} ({val_size:.1%})")
        print(f"   📉 Test: {self.X_test.shape} ({test_size:.1%})")
        
        # Print date ranges
        print(f"   📅 Train: {df_sorted.loc[train_indices, 'date'].min()} to {df_sorted.loc[train_indices, 'date'].max()}")
        print(f"   📅 Val: {df_sorted.loc[val_indices, 'date'].min()} to {df_sorted.loc[val_indices, 'date'].max()}")
        print(f"   📅 Test: {df_sorted.loc[test_indices, 'date'].min()} to {df_sorted.loc[test_indices, 'date'].max()}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Convert back to DataFrames
        self.X_train_scaled = pd.DataFrame(self.X_train_scaled, columns=self.X.columns, index=self.X_train.index)
        self.X_val_scaled = pd.DataFrame(self.X_val_scaled, columns=self.X.columns, index=self.X_val.index)
        self.X_test_scaled = pd.DataFrame(self.X_test_scaled, columns=self.X.columns, index=self.X_test.index)
        
        # Print class distributions
        print(f"\n📋 CLASS DISTRIBUTIONS:")
        print(f"   Train: {self.y_train.value_counts(normalize=True).round(3).to_dict()}")
        print(f"   Val: {self.y_val.value_counts(normalize=True).round(3).to_dict()}")
        print(f"   Test: {self.y_test.value_counts(normalize=True).round(3).to_dict()}")
    
    def split_data_stratified_temporal(self, train_size=0.6, val_size=0.2, test_size=0.2):
        """Split data with stratified temporal approach - handling meta evolution."""
        print(f"\n🎮 STRATIFIED TEMPORAL DATA SPLITTING (META-AWARE)")
        print("   Strategy: Balance training across all years to handle meta evolution")
        
        # Ensure sizes sum to 1.0
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split sizes must sum to 1.0"
        
        # Get the original dataframe with year information
        df = self.feature_engineering.df.copy()
        df_sorted = df.sort_values('date')
        
        # Get unique years
        years = sorted(df_sorted['year'].unique())
        print(f"   📅 Years available: {years}")
        
        # Initialize split indices
        train_indices = []
        val_indices = []
        test_indices = []
        
        # Split each year proportionally
        for year in years:
            year_data = df_sorted[df_sorted['year'] == year].sort_values('date')
            year_size = len(year_data)
            
            if year_size < 10:  # Skip years with too few matches
                print(f"   ⚠️ Skipping {year}: only {year_size} matches")
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
        
        print(f"\n   📊 FINAL STRATIFIED SPLIT:")
        print(f"   📈 Training: {self.X_train.shape}")
        print(f"   📊 Validation: {self.X_val.shape}")
        print(f"   📉 Test: {self.X_test.shape}")
        
        # Print year distributions
        train_years = df_sorted.loc[train_indices, 'year'].value_counts().sort_index()
        val_years = df_sorted.loc[val_indices, 'year'].value_counts().sort_index()
        test_years = df_sorted.loc[test_indices, 'year'].value_counts().sort_index()
        
        print(f"\n   📈 Training year distribution: {train_years.to_dict()}")
        print(f"   📊 Validation year distribution: {val_years.to_dict()}")
        print(f"   📉 Test year distribution: {test_years.to_dict()}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Convert back to DataFrames
        self.X_train_scaled = pd.DataFrame(self.X_train_scaled, columns=self.X.columns, index=self.X_train.index)
        self.X_val_scaled = pd.DataFrame(self.X_val_scaled, columns=self.X.columns, index=self.X_val.index)
        self.X_test_scaled = pd.DataFrame(self.X_test_scaled, columns=self.X.columns, index=self.X_test.index)
        
        # Print class distributions
        print(f"\n📋 CLASS DISTRIBUTIONS:")
        print(f"   Train: {self.y_train.value_counts(normalize=True).round(3).to_dict()}")
        print(f"   Val: {self.y_val.value_counts(normalize=True).round(3).to_dict()}")
        print(f"   Test: {self.y_test.value_counts(normalize=True).round(3).to_dict()}")
    
    def train_advanced_models(self, quick_mode=False):
        """Train multiple advanced ML models."""
        print(f"\n🤖 TRAINING ADVANCED MODEL SUITE")
        if quick_mode:
            print("⚡ QUICK MODE: Reduced hyperparameter search for faster training")
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
                        random_state=42, n_jobs=-1, eval_metric='logloss'
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
                        random_state=42, n_jobs=-1, class_weight='balanced', verbose=-1
                    ),
                    'params': {
                        'n_estimators': [200],
                        'max_depth': [8],
                        'learning_rate': [0.1]
                    },
                    'use_scaled': False
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
                        random_state=42, n_jobs=-1, eval_metric='logloss'
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
                        random_state=42, n_jobs=-1, class_weight='balanced', verbose=-1
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
                    random_state=42, verbose=False, auto_class_weights='Balanced'
                ),
                'params': {
                    'iterations': [200, 300],
                    'depth': [6, 8],
                    'learning_rate': [0.1, 0.15],
                    'l2_leaf_reg': [3, 5]
                },
                'use_scaled': False
            }
        
        # Train each model with hyperparameter optimization
        for name, config in models_config.items():
            print(f"\n🔄 Training {name}...")
            
            model = config['model']
            params = config['params']
            use_scaled = config['use_scaled']
            
            # Select appropriate data
            X_train_data = self.X_train_scaled if use_scaled else self.X_train
            X_val_data = self.X_val_scaled if use_scaled else self.X_val
            
            # Hyperparameter optimization
            grid_search = GridSearchCV(
                model, params, cv=3, scoring='f1', n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_train_data, self.y_train)
            best_model = grid_search.best_estimator_
            
            print(f"   Best params: {grid_search.best_params_}")
            
            # Evaluate on validation set
            y_pred = best_model.predict(X_val_data)
            y_pred_proba = best_model.predict_proba(X_val_data)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_val, y_pred)
            precision = precision_score(self.y_val, y_pred)
            recall = recall_score(self.y_val, y_pred)
            f1_performance = f1_score(self.y_val, y_pred)
            auc = roc_auc_score(self.y_val, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(best_model, X_train_data, self.y_train, cv=5, scoring='f1')
            
            # Store results
            self.models[name] = best_model
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1_performance,
                'auc': auc,
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'best_params': grid_search.best_params_,
                'use_scaled': use_scaled
            }
            
            print(f"   ✅ Validation F1: {f1_performance:.4f} | Accuracy: {accuracy:.4f} | AUC: {auc:.4f}")
            print(f"   📊 CV F1: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    def create_ultimate_ensemble(self):
        """Create sophisticated ensemble of best models."""
        print(f"\n🎯 CREATING ULTIMATE ENSEMBLE")
        
        # Get models that support probability prediction
        prob_models = [(name, model) for name, model in self.models.items() 
                      if hasattr(model, 'predict_proba')]
        
        if len(prob_models) < 2:
            print("   ⚠️ Not enough models for ensemble")
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
        
        print("   📊 Ensemble composition:")
        for name, weight in zip(model_names, weights):
            print(f"      {name}: {weight:.3f}")
        
        # Create sophisticated voting ensemble
        voting_clf = VotingClassifier(
            estimators=prob_models,
            voting='soft',
            weights=weights
        )
        
        # Train ensemble
        voting_clf.fit(self.X_train, self.y_train)
        
        # Evaluate ensemble
        y_pred_ensemble = voting_clf.predict(self.X_val)
        y_pred_proba_ensemble = voting_clf.predict_proba(self.X_val)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_val, y_pred_ensemble)
        f1_performance = f1_score(self.y_val, y_pred_ensemble)
        auc = roc_auc_score(self.y_val, y_pred_proba_ensemble)
        
        # Store ensemble results
        self.models['Ultimate Ensemble'] = voting_clf
        self.results['Ultimate Ensemble'] = {
            'accuracy': accuracy,
            'f1': f1_performance,
            'auc': auc,
            'use_scaled': False  # Ensemble handles scaling internally
        }
        
        print(f"   ✅ Ensemble F1: {f1_performance:.4f} | Accuracy: {accuracy:.4f} | AUC: {auc:.4f}")
    
    def evaluate_models(self):
        """Comprehensive model evaluation on validation set."""
        print(f"\n📊 MODEL EVALUATION SUMMARY")
        print("=" * 60)
        
        # Create results summary
        results_summary = []
        for name, results in self.results.items():
            results_summary.append({
                'Model': name,
                'Accuracy': results['accuracy'],
                'F1 Score': results['f1'],
                'AUC': results['auc'],
                'CV F1 Mean': results.get('cv_f1_mean', 0),
                'CV F1 Std': results.get('cv_f1_std', 0)
            })
        
        results_df = pd.DataFrame(results_summary).sort_values('F1 Score', ascending=False)
        
        print("\n🏆 VALIDATION PERFORMANCE RANKING:")
        print(results_df.round(4).to_string(index=False))
        
        # Best model
        best_model_name = results_df.iloc[0]['Model']
        
        print(f"\n🥇 BEST MODEL: {best_model_name}")
        print(f"   📈 F1 Score: {results_df.iloc[0]['F1 Score']:.4f}")
        print(f"   📊 Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
        print(f"   🎯 AUC: {results_df.iloc[0]['AUC']:.4f}")
        
        return best_model_name, results_df
    
    def final_test_evaluation(self, best_model_name):
        """Final evaluation on completely unseen test set."""
        print(f"\n🎯 FINAL TEST SET EVALUATION")
        print("=" * 60)
        print(f"⚠️  EVALUATING ON COMPLETELY UNSEEN TEST DATA")
        print(f"🏆 Best Model: {best_model_name}")
        
        best_model = self.models[best_model_name]
        use_scaled = self.results[best_model_name]['use_scaled']
        
        # Select appropriate test data
        X_test_data = self.X_test_scaled if use_scaled else self.X_test
        
        # Make predictions
        y_pred_test = best_model.predict(X_test_data)
        y_pred_proba_test = best_model.predict_proba(X_test_data)[:, 1]
        
        # Calculate final metrics
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        test_precision = precision_score(self.y_test, y_pred_test)
        test_recall = recall_score(self.y_test, y_pred_test)
        test_f1_performance = f1_score(self.y_test, y_pred_test)
        test_auc = roc_auc_score(self.y_test, y_pred_proba_test)
        
        print(f"\n📊 FINAL TEST RESULTS:")
        print(f"   🎯 Accuracy: {test_accuracy:.4f}")
        print(f"   📈 Precision: {test_precision:.4f}")
        print(f"   📉 Recall: {test_recall:.4f}")
        print(f"   📊 F1 Score: {test_f1_performance:.4f}")
        print(f"   📊 AUC: {test_auc:.4f}")
        
        # Compare with validation performance
        val_results = self.results[best_model_name]
        print(f"\n📋 VALIDATION vs TEST COMPARISON:")
        print(f"   F1: {val_results['f1']:.4f} → {test_f1_performance:.4f} (Δ: {val_results['f1'] - test_f1_performance:+.4f})")
        print(f"   Accuracy: {val_results['accuracy']:.4f} → {test_accuracy:.4f} (Δ: {val_results['accuracy'] - test_accuracy:+.4f})")
        
        # Classification report
        print(f"\n🔍 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(self.y_test, y_pred_test))
        
        # Confusion matrix visualization
        cm = confusion_matrix(self.y_test, y_pred_test)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Final Test Confusion Matrix - {best_model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('ultimate_test_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance analysis
        self.analyze_feature_importance(best_model_name)
        
        return {
            'model': best_model_name,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1_performance,
            'test_auc': test_auc,
            'generalization_gap': val_results['f1'] - test_f1_performance
        }
    
    def analyze_feature_importance(self, model_name):
        """Analyze feature importance for interpretability."""
        print(f"\n🎯 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 40)
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n📊 TOP 20 FEATURES - {model_name}:")
            for i, (_, row) in enumerate(importance_df.head(20).iterrows(), 1):
                print(f"   {i:2d}. {row['feature']}: {row['importance']:.4f}")
            
            # Save feature importance plot
            plt.figure(figsize=(12, 10))
            top_features = importance_df.head(20)
            sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
            plt.title(f'Top 20 Feature Importance - {model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(f'ultimate_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save feature importance data
            importance_df.to_csv('ultimate_feature_importance.csv', index=False)
            
        else:
            print(f"   ⚠️ {model_name} does not provide feature importances")
    
    def save_ultimate_model(self, best_model_name):
        """Save the ultimate model and all preprocessing components."""
        print(f"\n💾 SAVING ULTIMATE MODEL SYSTEM")
        
        best_model = self.models[best_model_name]
        
        # Save model components
        joblib.dump(best_model, 'ultimate_best_model.joblib')
        joblib.dump(self.scaler, 'ultimate_scaler.joblib')
        joblib.dump(self.feature_engineering, 'ultimate_feature_engineering.joblib')
        
        # Save feature names and metadata
        metadata = {
            'best_model_name': best_model_name,
            'feature_names': list(self.X.columns),
            'num_features': len(self.X.columns),
            'model_performance': self.results[best_model_name],
            'feature_engineering_config': {
                'champion_characteristics_count': len(self.feature_engineering.champion_characteristics),
                'meta_strength_combinations': len(self.feature_engineering.champion_meta_strength),
                'team_compositions': len(self.feature_engineering.team_compositions)
            }
        }
        
        joblib.dump(metadata, 'ultimate_model_metadata.joblib')
        
        print(f"   ✅ Saved: ultimate_best_model.joblib")
        print(f"   ✅ Saved: ultimate_scaler.joblib")
        print(f"   ✅ Saved: ultimate_feature_engineering.joblib")
        print(f"   ✅ Saved: ultimate_model_metadata.joblib")

def main(quick_mode=False, use_stratified_temporal=False):
    """Main execution function."""
    print("🎯 ULTIMATE LEAGUE OF LEGENDS MATCH PREDICTION SYSTEM")
    if quick_mode:
        print("⚡ RUNNING IN QUICK MODE - Reduced training time")
    if use_stratified_temporal:
        print("🎮 USING STRATIFIED TEMPORAL SPLIT - Meta-aware methodology")
    else:
        print("📊 USING PURE TEMPORAL SPLIT - Academic rigor methodology")
    print("=" * 80)
    
    # Initialize ultimate predictor
    predictor = UltimateLoLPredictor()
    
    # Prepare advanced features
    X, y = predictor.prepare_advanced_features()
    
    # Split data using chosen methodology
    if use_stratified_temporal:
        predictor.split_data_stratified_temporal()
    else:
        predictor.split_data_temporally()
    
    # Train advanced model suite
    predictor.train_advanced_models(quick_mode=quick_mode)
    
    # Create ultimate ensemble
    predictor.create_ultimate_ensemble()
    
    # Evaluate models
    best_model, results = predictor.evaluate_models()
    
    # Final test evaluation
    final_results = predictor.final_test_evaluation(best_model)
    
    # Save ultimate model
    predictor.save_ultimate_model(best_model)
    
    print(f"\n🎉 ULTIMATE SYSTEM TRAINING COMPLETE!")
    print(f"🏆 Best Model: {best_model}")
    print(f"📊 Final Test F1: {final_results['test_f1']:.4f}")
    print(f"🎯 Final Test Accuracy: {final_results['test_accuracy']:.4f}")
    print(f"📈 Generalization Gap: {final_results['generalization_gap']:.4f}")
    
    return predictor, final_results

if __name__ == "__main__":
    # Methodological comparison options:
    # Option 1: Pure temporal (academic rigor)
    # predictor, results = main(quick_mode=False, use_stratified_temporal=False)
    
    # Option 2: Stratified temporal (meta-aware) - YOUR SUGGESTION
    # predictor, results = main(quick_mode=False, use_stratified_temporal=True)
    
    # Current: Quick test with your stratified approach
    predictor, results = main(quick_mode=False, use_stratified_temporal=True) 