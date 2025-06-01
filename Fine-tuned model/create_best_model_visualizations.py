#!/usr/bin/env python3
"""
🎨 STANDALONE BEST MODEL VISUALIZATION GENERATOR
================================================

This script creates comprehensive, thesis-quality visualizations for the best performing model.
Perfect for academic presentations, thesis defense, and publication figures.

Usage:
    python create_best_model_visualizations.py

Features:
    📊 12 comprehensive visualization plots
    🏆 Focused on Logistic Regression (consistent winner)
    📈 ROC curves, PR curves, calibration, feature importance
    📋 Learning curves, decision boundaries, residuals analysis
    🎯 High-resolution, publication-quality output

Author: Your Breakthrough Research Team
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_comprehensive_visualizations(best_model, X_train, X_val, X_test, y_train, y_val, y_test, feature_names, best_model_name="Logistic Regression"):
    """🎨 Create comprehensive visualization suite - completely self-contained."""
    print(f"\n🎨 CREATING COMPREHENSIVE VISUALIZATION SUITE FOR: {best_model_name}")
    print("=" * 80)
    print("📊 Thesis-quality visualizations for academic presentation")
    
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
        
        # Set high-quality style
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
        
        # Make predictions for all sets
        y_train_pred = best_model.predict(X_train)
        y_train_proba = best_model.predict_proba(X_train)[:, 1]
        y_val_pred = best_model.predict(X_val)
        y_val_proba = best_model.predict_proba(X_val)[:, 1]
        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Create comprehensive figure with subplots
        fig = plt.figure(figsize=(20, 24))
        fig.suptitle(f'🏆 COMPREHENSIVE ANALYSIS: {best_model_name} (THESIS QUALITY)', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # 1. ROC Curves for All Sets
        ax1 = plt.subplot(4, 3, 1)
        datasets = [
            (y_train, y_train_proba, 'Training', 'blue'),
            (y_val, y_val_proba, 'Validation', 'green'),
            (y_test, y_test_proba, 'Test', 'red')
        ]
        
        for y_true, y_prob, label, color in datasets:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            ax1.plot(fpr, tpr, color=color, lw=2, 
                    label=f'{label} (AUC = {roc_auc:.3f})')
        
        ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('🎯 ROC Curves: Train/Val/Test')
        ax1.legend(loc="lower right")
        ax1.grid(alpha=0.3)
        
        # 2. Precision-Recall Curves
        ax2 = plt.subplot(4, 3, 2)
        for y_true, y_prob, label, color in datasets:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall, precision)
            ax2.plot(recall, precision, color=color, lw=2,
                    label=f'{label} (AUC = {pr_auc:.3f})')
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('📈 Precision-Recall Curves')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Confusion Matrix
        ax3 = plt.subplot(4, 3, 3)
        cm_test = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax3,
                   cbar_kws={'label': 'Count'})
        ax3.set_title('🎯 Confusion Matrix (Test Set)')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # 4. Probability Calibration (conditional)
        ax4 = plt.subplot(4, 3, 4)
        if CALIBRATION_AVAILABLE:
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_test, y_test_proba, n_bins=10)
                ax4.plot(mean_predicted_value, fraction_of_positives, "s-", 
                        label=f'{best_model_name}', color='red', markersize=8)
                ax4.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                ax4.set_xlabel('Mean Predicted Probability')
                ax4.set_ylabel('Fraction of Positives')
                ax4.set_title('📊 Probability Calibration')
                ax4.legend()
                ax4.grid(alpha=0.3)
            except Exception as cal_error:
                ax4.text(0.5, 0.5, f'Calibration Plot\nUnavailable\n({str(cal_error)[:30]}...)', 
                        transform=ax4.transAxes, ha='center', va='center',
                        fontsize=12, style='italic')
                ax4.set_title('📊 Probability Calibration')
        else:
            ax4.text(0.5, 0.5, 'Calibration Plot\nNot Available\n(sklearn version)', 
                    transform=ax4.transAxes, ha='center', va='center',
                    fontsize=12, style='italic')
            ax4.set_title('📊 Probability Calibration')
        
        # 5. Feature Importance (for Logistic Regression)
        ax5 = plt.subplot(4, 3, 5)
        if hasattr(best_model, 'coef_'):
            feature_importance = np.abs(best_model.coef_[0])
            
            # Get top 15 features
            top_indices = np.argsort(feature_importance)[-15:]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = feature_importance[top_indices]
            
            bars = ax5.barh(range(len(top_features)), top_importance, 
                           color='skyblue', alpha=0.8)
            ax5.set_yticks(range(len(top_features)))
            ax5.set_yticklabels([f[:20] + '...' if len(f) > 20 else f 
                               for f in top_features])
            ax5.set_xlabel('Absolute Coefficient Value')
            ax5.set_title('🔍 Top 15 Feature Importance')
            ax5.grid(axis='x', alpha=0.3)
            
            # Add values on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax5.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        # 6. Learning Curves
        ax6 = plt.subplot(4, 3, 6)
        if LEARNING_CURVE_AVAILABLE:
            try:
                train_sizes, train_scores, val_scores = learning_curve(
                    best_model, X_train, y_train, cv=5, 
                    train_sizes=np.linspace(0.1, 1.0, 10), scoring='roc_auc',
                    n_jobs=-1, random_state=42)
                
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                val_mean = np.mean(val_scores, axis=1)
                val_std = np.std(val_scores, axis=1)
                
                ax6.plot(train_sizes, train_mean, 'o-', color='blue', label='Training')
                ax6.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                               alpha=0.2, color='blue')
                ax6.plot(train_sizes, val_mean, 'o-', color='red', label='Validation')
                ax6.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                               alpha=0.2, color='red')
                
                ax6.set_xlabel('Training Set Size')
                ax6.set_ylabel('AUC Score')
                ax6.set_title('📈 Learning Curves')
                ax6.legend()
                ax6.grid(alpha=0.3)
            except Exception as lc_error:
                ax6.text(0.5, 0.5, f'Learning Curves\nUnavailable\n({str(lc_error)[:30]}...)', 
                        transform=ax6.transAxes, ha='center', va='center',
                        fontsize=12, style='italic')
                ax6.set_title('📈 Learning Curves')
        else:
            ax6.text(0.5, 0.5, 'Learning Curves\nNot Available\n(sklearn version)', 
                    transform=ax6.transAxes, ha='center', va='center',
                    fontsize=12, style='italic')
            ax6.set_title('📈 Learning Curves')
        
        # 7. Prediction Distribution
        ax7 = plt.subplot(4, 3, 7)
        ax7.hist(y_test_proba[y_test == 0], bins=30, alpha=0.7, 
                label='Loss (0)', color='red', density=True)
        ax7.hist(y_test_proba[y_test == 1], bins=30, alpha=0.7, 
                label='Win (1)', color='blue', density=True)
        ax7.set_xlabel('Predicted Probability')
        ax7.set_ylabel('Density')
        ax7.set_title('📊 Prediction Distribution')
        ax7.legend()
        ax7.grid(alpha=0.3)
        
        # 8. Performance Metrics Summary
        ax8 = plt.subplot(4, 3, 8)
        ax8.axis('off')
        
        metrics_text = f"""
🏆 MODEL: {best_model_name}
{'='*40}

📊 TEST SET PERFORMANCE:
• AUC-ROC: {roc_auc_score(y_test, y_test_proba):.4f}
• Accuracy: {accuracy_score(y_test, y_test_pred):.4f}
• F1 Score: {f1_score(y_test, y_test_pred):.4f}
• Precision: {precision_score(y_test, y_test_pred):.4f}
• Recall: {recall_score(y_test, y_test_pred):.4f}

🔬 CALIBRATION:
• Brier Score: {np.mean((y_test_proba - y_test)**2):.4f}
• Log Loss: {-np.mean(y_test * np.log(y_test_proba + 1e-15) + 
                     (1 - y_test) * np.log(1 - y_test_proba + 1e-15)):.4f}

🎯 DATASET INFO:
• Training: {len(y_train)} samples
• Validation: {len(y_val)} samples  
• Test: {len(y_test)} samples
• Features: {len(feature_names)} engineered features
        """
        
        ax8.text(0.05, 0.95, metrics_text, transform=ax8.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # 9. Cross-Validation Performance
        ax9 = plt.subplot(4, 3, 9)
        try:
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
            ax9.hist(cv_scores, bins=5, alpha=0.7, color='green', edgecolor='black')
            ax9.axvline(np.mean(cv_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(cv_scores):.3f}')
            ax9.set_xlabel('CV AUC Score')
            ax9.set_ylabel('Frequency')
            ax9.set_title('📊 Cross-Validation Scores')
            ax9.legend()
            ax9.grid(alpha=0.3)
        except ImportError:
            ax9.text(0.5, 0.5, 'Cross-Validation\nScores\nNot Available', 
                    transform=ax9.transAxes, ha='center', va='center',
                    fontsize=12, style='italic')
            ax9.set_title('📊 Cross-Validation Scores')
        except Exception as cv_error:
            ax9.text(0.5, 0.5, f'Cross-Validation\nError\n({str(cv_error)[:20]}...)', 
                    transform=ax9.transAxes, ha='center', va='center',
                    fontsize=12, style='italic')
            ax9.set_title('📊 Cross-Validation Scores')
        
        # 10. Residuals Analysis
        ax10 = plt.subplot(4, 3, 10)
        residuals = y_test_proba - y_test
        ax10.scatter(y_test_proba, residuals, alpha=0.6, s=20, color='purple')
        ax10.axhline(y=0, color='red', linestyle='--')
        ax10.set_xlabel('Predicted Probability')
        ax10.set_ylabel('Residuals (Pred - Actual)')
        ax10.set_title('🔍 Residuals Analysis')
        ax10.grid(alpha=0.3)
        
        # 11. Decision Boundary Visualization (PCA)
        ax11 = plt.subplot(4, 3, 11)
        if PCA_AVAILABLE:
            try:
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
                
                ax11.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
                scatter = ax11.scatter(X_test_pca[:, 0], X_test_pca[:, 1], 
                                     c=y_test, cmap='RdYlBu', s=20, alpha=0.8)
                ax11.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
                ax11.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
                ax11.set_title('🎯 Decision Boundary (PCA)')
                plt.colorbar(scatter, ax=ax11)
                
            except Exception:
                ax11.text(0.5, 0.5, 'Decision Boundary\nVisualization\nUnavailable', 
                          transform=ax11.transAxes, ha='center', va='center',
                          fontsize=12, style='italic')
                ax11.set_title('🎯 Decision Boundary')
        else:
            ax11.text(0.5, 0.5, 'Decision Boundary\nVisualization\nNot Available\n(PCA unavailable)', 
                      transform=ax11.transAxes, ha='center', va='center',
                      fontsize=12, style='italic')
            ax11.set_title('🎯 Decision Boundary')
        
        # 12. Feature Correlation with Target (Top Features)
        ax12 = plt.subplot(4, 3, 12)
        if hasattr(best_model, 'coef_'):
            # Calculate correlation with target for top features
            top_feature_indices = np.argsort(np.abs(best_model.coef_[0]))[-10:]
            correlations = []
            feature_names_short = []
            
            for idx in top_feature_indices:
                corr = np.corrcoef(X_test.iloc[:, idx] if hasattr(X_test, 'iloc') else X_test[:, idx], y_test)[0, 1]
                correlations.append(corr)
                feature_name = feature_names[idx]
                feature_names_short.append(feature_name[:15] + '...' if len(feature_name) > 15 else feature_name)
            
            bars = ax12.barh(range(len(correlations)), correlations, 
                           color=['red' if c < 0 else 'blue' for c in correlations],
                           alpha=0.7)
            ax12.set_yticks(range(len(correlations)))
            ax12.set_yticklabels(feature_names_short)
            ax12.set_xlabel('Correlation with Target')
            ax12.set_title('🔗 Feature-Target Correlation')
            ax12.grid(axis='x', alpha=0.3)
            ax12.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        # 📁 CREATE VISUALIZATIONS FOLDER & SAVE
        viz_folder = "Best_Model_Visualizations"
        os.makedirs(viz_folder, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'comprehensive_{best_model_name.replace(" ", "_").lower()}_{timestamp}.png'
        filepath = os.path.join(viz_folder, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ Comprehensive visualization saved: {filepath}")
        print(f"📁 Location: {os.path.abspath(filepath)}")
        print(f"📊 12 advanced plots created for thesis presentation")
        
        return filepath
        
    except Exception as e:
        print(f"⚠️ Visualization error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_visualizations_for_best_model():
    """Create comprehensive visualizations for the best model."""
    print("🎨 STANDALONE BEST MODEL VISUALIZATION GENERATOR")
    print("=" * 60)
    print("📊 Creating thesis-quality visualizations for Logistic Regression")
    
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
        
        # Create comprehensive visualizations
        print(f"\n🎨 Creating comprehensive visualizations...")
        filepath = create_comprehensive_visualizations(
            lr_model, X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train, y_val, y_test, feature_names, "Logistic Regression"
        )
        
        if filepath:
            print(f"\n✅ SUCCESS! Comprehensive visualizations created")
            print(f"📁 File: {filepath}")
            print(f"📊 12 advanced plots generated for thesis presentation")
            print(f"🎯 Ready for academic use!")
        else:
            print("❌ Visualization creation failed")
            
    except Exception as e:
        print(f"❌ Error in visualization creation: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function for standalone execution."""
    print(__doc__)
    create_visualizations_for_best_model()

if __name__ == "__main__":
    main() 