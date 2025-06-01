import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_focused_dataset():
    """Analyze the focused dataset to help decide between ML and DL."""
    print("🔍 ANALYZING FOCUSED DATASET FOR MODEL SELECTION")
    print("=" * 60)
    
    # Load the focused features dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    input_path = os.path.join(project_root, "data", "focused_features_match_level.csv")

    # Try alternative paths if not found
    if not os.path.exists(input_path):
        alternative_paths = [
            'Dataset collection/focused_features_match_level.csv',
            '../Dataset collection/focused_features_match_level.csv'
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                input_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Could not find focused_features_match_level.csv")

    print(f"📂 Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"📊 DATASET OVERVIEW:")
    print(f"   Shape: {df.shape}")
    print(f"   Total matches: {len(df):,}")
    print(f"   Features: {len(df.columns)}")
    
    print(f"\n📋 COLUMNS:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    # Analyze by league
    print(f"\n🏆 TOP LEAGUES ANALYSIS:")
    top_leagues = ['LCK', 'LPL', 'LEC', 'LCS']
    for league in top_leagues:
        count = len(df[df['league'] == league])
        print(f"   • {league}: {count:,} matches")
    
    # Show sample data
    print(f"\n📋 SAMPLE DATA:")
    sample_cols = ['league', 'team', 'side', 'result', 'top_champion', 'mid_champion', 'bot_champion']
    available_cols = [col for col in sample_cols if col in df.columns]
    print(df[available_cols].head(5).to_string(index=False))
    
    # Analyze data characteristics for ML vs DL decision
    print(f"\n🤖 MODEL SELECTION ANALYSIS:")
    print("=" * 60)
    
    # Data size
    total_samples = len(df)
    print(f"📈 Data Size: {total_samples:,} samples")
    
    # Feature types
    categorical_features = []
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'result':
            categorical_features.append(col)
    
    print(f"🏷️  Categorical Features: {len(categorical_features)}")
    for feat in categorical_features[:10]:  # Show first 10
        unique_count = df[feat].nunique()
        print(f"   • {feat}: {unique_count} unique values")
    
    # Champion features
    champion_cols = [col for col in df.columns if 'champion' in col]
    print(f"🎮 Champion Features: {len(champion_cols)}")
    
    # Missing data
    missing_data = df.isnull().sum()
    high_missing = missing_data[missing_data > 0]
    print(f"❓ Features with Missing Data: {len(high_missing)}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 60)
    
    if total_samples < 50000:
        print("📊 TRADITIONAL MACHINE LEARNING RECOMMENDED")
        print("   Reasons:")
        print("   • Dataset size < 50K samples")
        print("   • Mostly categorical features")
        print("   • Interpretability important for esports")
        print("   • Faster training and inference")
        
        print(f"\n🎯 SUGGESTED ML ALGORITHMS:")
        print("   1. Random Forest (handles categorical well)")
        print("   2. Gradient Boosting (XGBoost/LightGBM)")
        print("   3. Logistic Regression (interpretable)")
        print("   4. Support Vector Machine")
        
    else:
        print("🧠 DEEP LEARNING COULD BE CONSIDERED")
        print("   Reasons:")
        print("   • Large dataset (>50K samples)")
        print("   • Complex feature interactions possible")
        
        print(f"\n🎯 SUGGESTED DL APPROACHES:")
        print("   1. Neural Networks with embedding layers")
        print("   2. Transformer-based models")
        print("   3. Ensemble of ML + DL")
    
    print(f"\n🔧 FEATURE ENGINEERING NEEDS:")
    print("   • One-hot encode categorical features")
    print("   • Create champion synergy features")
    print("   • Team historical performance")
    print("   • Patch meta features")
    
    return df

if __name__ == "__main__":
    df = analyze_focused_dataset() 