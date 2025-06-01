#!/usr/bin/env python3
"""
🎨 INDIVIDUAL THESIS PLOTS GENERATOR
===================================

This script creates individual, high-quality visualizations perfect for thesis inclusion.
Each plot is saved as a separate file with optimal sizing and formatting.

Usage:
    python create_individual_thesis_plots.py

Features:
    📊 12 individual high-resolution plots
    🎓 Thesis-optimized sizing and formatting
    📝 Perfect for academic document inclusion
    🖼️ Publication-quality PNG files at 300 DPI
    📁 Organized output in separate folders

Author: Your Breakthrough Research Team
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_individual_plots(best_model, X_train, X_val, X_test, y_train, y_val, y_test, feature_names, best_model_name="Logistic Regression"):
    """🎨 Create individual thesis-quality plots - one file per visualization."""
    print(f"\n🎨 CREATING INDIVIDUAL THESIS PLOTS FOR: {best_model_name}")
    print("=" * 80)
    print("📊 Individual plots optimized for thesis inclusion")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
                                   confusion_matrix, roc_auc_score,
                                   accuracy_score, f1_score, precision_score, recall_score)
        
        # Try to import learning_curve from correct module
        try:
            from sklearn.model_selection import learning_curve
            LEARNING_CURVE_AVAILABLE = True
        except ImportError:
            LEARNING_CURVE_AVAILABLE = False
            print("⚠️ Learning curve not available - will skip learning curve plot")
        
        # Try to import calibration_curve, with fallback
        try:
            from sklearn.calibration import calibration_curve
            CALIBRATION_AVAILABLE = True
        except ImportError:
            try:
                from sklearn.metrics import calibration_curve
                CALIBRATION_AVAILABLE = True
            except ImportError:
                CALIBRATION_AVAILABLE = False
                print("⚠️ Calibration curve not available - will skip calibration plot")
        
        # PCA import with fallback
        try:
            from sklearn.decomposition import PCA
            PCA_AVAILABLE = True
        except ImportError:
            PCA_AVAILABLE = False
            print("⚠️ PCA not available - will skip PCA plots")
        
        from datetime import datetime
        import pandas as pd
        
        # Set high-quality style for thesis
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            try:
                plt.style.use('seaborn-whitegrid')
            except OSError:
                plt.style.use('default')
                print("⚠️ Using default matplotlib style")
        
        try:
            sns.set_palette("husl")
        except:
            print("⚠️ Seaborn palette not available, using default")
        
        # Create output directory
        output_dir = "Individual_Thesis_Plots"
        os.makedirs(output_dir, exist_ok=True)
        
        # Make predictions for all sets
        y_train_pred = best_model.predict(X_train)
        y_train_proba = best_model.predict_proba(X_train)[:, 1]
        y_val_pred = best_model.predict(X_val)
        y_val_proba = best_model.predict_proba(X_val)[:, 1]
        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
        
        datasets = [
            (y_train, y_train_proba, 'Training', 'blue'),
            (y_val, y_val_proba, 'Validation', 'green'),
            (y_test, y_test_proba, 'Test', 'red')
        ]
        
        created_plots = []
        
        # 1. ROC Curves for All Sets
        print("📊 Creating ROC Curves plot...")
        plt.figure(figsize=(10, 8))
        for y_true, y_prob, label, color in datasets:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=3, 
                    label=f'{label} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.7)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'ROC Curves: {best_model_name}', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, "01_roc_curves.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        created_plots.append(filename)
        print(f"✅ Saved: {filename}")
        
        # 2. Precision-Recall Curves
        print("📊 Creating Precision-Recall Curves plot...")
        plt.figure(figsize=(10, 8))
        for y_true, y_prob, label, color in datasets:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, color=color, lw=3,
                    label=f'{label} (AUC = {pr_auc:.3f})')
        
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title(f'Precision-Recall Curves: {best_model_name}', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, "02_precision_recall_curves.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        created_plots.append(filename)
        print(f"✅ Saved: {filename}")
        
        # 3. Confusion Matrix
        print("📊 Creating Confusion Matrix plot...")
        plt.figure(figsize=(8, 6))
        cm_test = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', 
                   cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
        plt.title(f'Confusion Matrix: {best_model_name} (Test Set)', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('Actual', fontsize=14)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, "03_confusion_matrix.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        created_plots.append(filename)
        print(f"✅ Saved: {filename}")
        
        # 4. Probability Calibration
        if CALIBRATION_AVAILABLE:
            print("📊 Creating Probability Calibration plot...")
            try:
                plt.figure(figsize=(8, 8))
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_test, y_test_proba, n_bins=10)
                plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                        label=f'{best_model_name}', color='red', markersize=10, lw=3)
                plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", lw=2)
                plt.xlabel('Mean Predicted Probability', fontsize=14)
                plt.ylabel('Fraction of Positives', fontsize=14)
                plt.title(f'Probability Calibration: {best_model_name}', fontsize=16, fontweight='bold')
                plt.legend(fontsize=12)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                
                filename = os.path.join(output_dir, "04_probability_calibration.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                created_plots.append(filename)
                print(f"✅ Saved: {filename}")
            except Exception as cal_error:
                print(f"⚠️ Calibration plot failed: {str(cal_error)}")
        else:
            print("⚠️ Skipping calibration plot (not available)")
        
        # 5. Feature Importance
        if hasattr(best_model, 'coef_'):
            print("📊 Creating Feature Importance plot...")
            plt.figure(figsize=(12, 10))
            feature_importance = np.abs(best_model.coef_[0])
            
            # Get top 20 features for better visibility
            top_indices = np.argsort(feature_importance)[-20:]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = feature_importance[top_indices]
            
            bars = plt.barh(range(len(top_features)), top_importance, 
                           color='skyblue', alpha=0.8, edgecolor='navy', linewidth=0.5)
            plt.yticks(range(len(top_features)), 
                      [f[:25] + '...' if len(f) > 25 else f for f in top_features])
            plt.xlabel('Absolute Coefficient Value', fontsize=14)
            plt.title(f'Top 20 Feature Importance: {best_model_name}', fontsize=16, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            
            # Add values on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}', ha='left', va='center', fontsize=10)
            
            plt.tight_layout()
            
            filename = os.path.join(output_dir, "05_feature_importance.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            created_plots.append(filename)
            print(f"✅ Saved: {filename}")
        
        # 6. Learning Curves
        if LEARNING_CURVE_AVAILABLE:
            print("📊 Creating Learning Curves plot...")
            try:
                plt.figure(figsize=(10, 8))
                train_sizes, train_scores, val_scores = learning_curve(
                    best_model, X_train, y_train, cv=5, 
                    train_sizes=np.linspace(0.1, 1.0, 10), scoring='roc_auc',
                    n_jobs=-1, random_state=42)
                
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                val_mean = np.mean(val_scores, axis=1)
                val_std = np.std(val_scores, axis=1)
                
                plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training', lw=3, markersize=8)
                plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                               alpha=0.2, color='blue')
                plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation', lw=3, markersize=8)
                plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                               alpha=0.2, color='red')
                
                plt.xlabel('Training Set Size', fontsize=14)
                plt.ylabel('AUC Score', fontsize=14)
                plt.title(f'Learning Curves: {best_model_name}', fontsize=16, fontweight='bold')
                plt.legend(fontsize=12)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                
                filename = os.path.join(output_dir, "06_learning_curves.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                created_plots.append(filename)
                print(f"✅ Saved: {filename}")
            except Exception as lc_error:
                print(f"⚠️ Learning curves failed: {str(lc_error)}")
        else:
            print("⚠️ Skipping learning curves (not available)")
        
        # 7. Prediction Distribution
        print("📊 Creating Prediction Distribution plot...")
        plt.figure(figsize=(10, 8))
        plt.hist(y_test_proba[y_test == 0], bins=30, alpha=0.7, 
                label='Loss (0)', color='red', density=True, edgecolor='darkred')
        plt.hist(y_test_proba[y_test == 1], bins=30, alpha=0.7, 
                label='Win (1)', color='blue', density=True, edgecolor='darkblue')
        plt.xlabel('Predicted Probability', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title(f'Prediction Distribution: {best_model_name}', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, "07_prediction_distribution.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        created_plots.append(filename)
        print(f"✅ Saved: {filename}")
        
        # 8. Performance Metrics Summary
        print("📊 Creating Performance Metrics Summary...")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        metrics_text = f"""
MODEL: {best_model_name}
{'='*50}

TEST SET PERFORMANCE:
• AUC-ROC: {roc_auc_score(y_test, y_test_proba):.4f}
• Accuracy: {accuracy_score(y_test, y_test_pred):.4f}
• F1 Score: {f1_score(y_test, y_test_pred):.4f}
• Precision: {precision_score(y_test, y_test_pred):.4f}
• Recall: {recall_score(y_test, y_test_pred):.4f}

CALIBRATION METRICS:
• Brier Score: {np.mean((y_test_proba - y_test)**2):.4f}
• Log Loss: {-np.mean(y_test * np.log(y_test_proba + 1e-15) + (1 - y_test) * np.log(1 - y_test_proba + 1e-15)):.4f}

DATASET INFORMATION:
• Training samples: {len(y_train):,}
• Validation samples: {len(y_val):,}  
• Test samples: {len(y_test):,}
• Total features: {len(feature_names)}
        """
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                fontsize=14, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=1", facecolor="lightblue", alpha=0.8))
        
        ax.set_title('Performance Metrics Summary', fontsize=18, fontweight='bold', pad=20)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, "08_performance_metrics.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        created_plots.append(filename)
        print(f"✅ Saved: {filename}")
        
        # 9. Cross-Validation Performance
        try:
            print("📊 Creating Cross-Validation Performance plot...")
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
            
            plt.figure(figsize=(10, 8))
            plt.hist(cv_scores, bins=5, alpha=0.7, color='green', edgecolor='darkgreen', linewidth=2)
            plt.axvline(np.mean(cv_scores), color='red', linestyle='--', linewidth=3,
                       label=f'Mean: {np.mean(cv_scores):.3f}')
            plt.xlabel('CV AUC Score', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.title(f'Cross-Validation Scores: {best_model_name}', fontsize=16, fontweight='bold')
            plt.legend(fontsize=12)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            filename = os.path.join(output_dir, "09_cross_validation.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            created_plots.append(filename)
            print(f"✅ Saved: {filename}")
        except Exception as cv_error:
            print(f"⚠️ Cross-validation plot failed: {str(cv_error)}")
        
        # 10. Residuals Analysis
        print("📊 Creating Residuals Analysis plot...")
        plt.figure(figsize=(10, 8))
        residuals = y_test_proba - y_test
        plt.scatter(y_test_proba, residuals, alpha=0.6, s=30, color='purple', edgecolors='darkmagenta')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Probability', fontsize=14)
        plt.ylabel('Residuals (Pred - Actual)', fontsize=14)
        plt.title(f'Residuals Analysis: {best_model_name}', fontsize=16, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, "10_residuals_analysis.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        created_plots.append(filename)
        print(f"✅ Saved: {filename}")
        
        # 11. Decision Boundary Visualization (PCA)
        if PCA_AVAILABLE:
            print("📊 Creating Decision Boundary plot...")
            try:
                plt.figure(figsize=(10, 8))
                # Use PCA to reduce to 2D for visualization
                pca = PCA(n_components=2, random_state=42)
                X_test_pca = pca.fit_transform(X_test)
                
                # Create decision boundary
                h = 0.02
                x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
                y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                   np.arange(y_min, y_max, h))
                
                # Create a dummy model for PCA space
                from sklearn.linear_model import LogisticRegression as LR
                pca_model = LR(random_state=42)
                pca_model.fit(X_test_pca, y_test)
                
                Z = pca_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
                Z = Z.reshape(xx.shape)
                
                plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
                scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], 
                                    c=y_test, cmap='RdYlBu', s=30, alpha=0.8, edgecolors='black')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance explained)', fontsize=14)
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance explained)', fontsize=14)
                plt.title(f'Decision Boundary (PCA): {best_model_name}', fontsize=16, fontweight='bold')
                plt.colorbar(scatter, label='Match Result')
                plt.tight_layout()
                
                filename = os.path.join(output_dir, "11_decision_boundary.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                created_plots.append(filename)
                print(f"✅ Saved: {filename}")
            except Exception as pca_error:
                print(f"⚠️ Decision boundary plot failed: {str(pca_error)}")
        else:
            print("⚠️ Skipping decision boundary plot (PCA not available)")
        
        # 12. Feature Correlation with Target
        if hasattr(best_model, 'coef_'):
            print("📊 Creating Feature-Target Correlation plot...")
            plt.figure(figsize=(12, 10))
            # Calculate correlation with target for top features
            top_feature_indices = np.argsort(np.abs(best_model.coef_[0]))[-15:]
            correlations = []
            feature_names_short = []
            
            for idx in top_feature_indices:
                corr = np.corrcoef(X_test.iloc[:, idx] if hasattr(X_test, 'iloc') else X_test[:, idx], y_test)[0, 1]
                correlations.append(corr)
                feature_name = feature_names[idx]
                feature_names_short.append(feature_name[:20] + '...' if len(feature_name) > 20 else feature_name)
            
            bars = plt.barh(range(len(correlations)), correlations, 
                           color=['red' if c < 0 else 'blue' for c in correlations],
                           alpha=0.7, edgecolor='black', linewidth=0.5)
            plt.yticks(range(len(correlations)), feature_names_short)
            plt.xlabel('Correlation with Target', fontsize=14)
            plt.title(f'Feature-Target Correlation: {best_model_name}', fontsize=16, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            plt.tight_layout()
            
            filename = os.path.join(output_dir, "12_feature_correlation.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            created_plots.append(filename)
            print(f"✅ Saved: {filename}")
        
        print(f"\n✅ SUCCESS! Created {len(created_plots)} individual thesis plots")
        print(f"📁 All plots saved in: {os.path.abspath(output_dir)}/")
        print(f"🎓 Ready for thesis inclusion!")
        
        return created_plots
        
    except Exception as e:
        print(f"⚠️ Visualization error: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def create_individual_thesis_visualizations():
    """Create individual visualizations for thesis inclusion."""
    print("🎨 INDIVIDUAL THESIS PLOTS GENERATOR")
    print("=" * 60)
    print("📊 Creating individual thesis-quality visualizations")
    
    try:
        # Import the advanced feature engineering
        from advanced_feature_engineering import AdvancedFeatureEngineering
        
        # Initialize and prepare data
        print("\n🔄 Initializing feature engineering and loading data...")
        
        # Get the directory of this script and build the path to the dataset
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(os.path.dirname(script_dir), "Dataset collection", "target_leagues_dataset.csv")
        
        feature_engineering = AdvancedFeatureEngineering(data_path)
        
        # Load and engineer features
        df = feature_engineering.load_and_analyze_data()
        print(f"✅ Loaded dataset: {df.shape}")
        
        # Create advanced features
        print("⚡ Creating advanced features (vectorized)...")
        advanced_features = feature_engineering.create_advanced_features_vectorized()
        final_features = feature_engineering.apply_advanced_encoding_optimized()
        
        # Get target variable
        X = final_features
        y = df['result']
        feature_names = list(X.columns)
        
        print(f"📊 Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Apply stratified random temporal split
        print("🚀 Applying Stratified Random Temporal split...")
        
        # Get unique years and create splits
        years = sorted(df['year'].unique())
        print(f"📅 Years available: {years}")
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        # Split each year with random sampling
        for year in years:
            year_data = df[df['year'] == year]
            year_size = len(year_data)
            
            if year_size < 10:  # Skip years with too few matches
                continue
            
            # Random shuffle within year
            year_data_shuffled = year_data.sample(frac=1.0, random_state=42)
            
            # Calculate split points
            train_end = int(year_size * 0.6)
            val_end = int(year_size * 0.8)
            
            # Random split (not chronological!)
            year_train = year_data_shuffled.iloc[:train_end]
            year_val = year_data_shuffled.iloc[train_end:val_end]
            year_test = year_data_shuffled.iloc[val_end:]
            
            # Collect indices
            train_indices.extend(year_train.index.tolist())
            val_indices.extend(year_val.index.tolist())
            test_indices.extend(year_test.index.tolist())
        
        # Split data
        X_train = X.loc[train_indices]
        X_val = X.loc[val_indices]
        X_test = X.loc[test_indices]
        y_train = y.loc[train_indices]
        y_val = y.loc[val_indices]
        y_test = y.loc[test_indices]
        
        print(f"📊 Data splits: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        
        # Scale features for Logistic Regression
        print("⚡ Scaling features for Logistic Regression...")
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrames for easier handling
        import pandas as pd
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X.columns, index=X_val.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
        
        # Quick Logistic Regression training for visualization
        print("\n⚡ Training Logistic Regression for visualization...")
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        
        lr_model = LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced')
        lr_model.fit(X_train_scaled, y_train)
        
        # Quick evaluation
        y_val_proba = lr_model.predict_proba(X_val_scaled)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_proba)
        
        print(f"⚡ Logistic Regression Validation AUC: {val_auc:.4f}")
        
        # Create individual visualizations
        print(f"\n🎨 Creating individual thesis plots...")
        created_plots = create_individual_plots(
            lr_model, X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train, y_val, y_test, feature_names, "Logistic Regression"
        )
        
        if created_plots:
            print(f"\n🎓 THESIS PLOTS READY!")
            print(f"📁 Location: Individual_Thesis_Plots/")
            print(f"📊 Created {len(created_plots)} individual plots")
            print(f"🖼️ All plots at 300 DPI for publication quality")
            print(f"📝 Perfect for thesis inclusion!")
            
            # Print file list for easy reference
            print(f"\n📋 INDIVIDUAL PLOT FILES:")
            for i, plot_file in enumerate(created_plots, 1):
                print(f"   {i:2d}. {os.path.basename(plot_file)}")
                
        else:
            print("❌ Plot creation failed")
            
    except Exception as e:
        print(f"❌ Error in visualization creation: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function for standalone execution."""
    print(__doc__)
    create_individual_thesis_visualizations()

if __name__ == "__main__":
    main() 