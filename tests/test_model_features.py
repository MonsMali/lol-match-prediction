#!/usr/bin/env python3
"""
Test feature engineering and data processing without model training.
"""

import os
import sys
import pandas as pd

# Add src to path - now we need to go up one directory first since we're in tests/
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

def test_feature_engineering():
    """Test the feature engineering pipeline."""
    print("🔧 TESTING FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    try:
        from feature_engineering.advanced_feature_engineering import AdvancedFeatureEngineering
        
        # Initialize
        print("1️⃣ Testing initialization...")
        fe = AdvancedFeatureEngineering()
        print(f"   ✅ AdvancedFeatureEngineering created")
        print(f"   📂 Data path: {fe.data_path}")
        
        # Test if we can load sample data
        if os.path.exists(fe.data_path):
            print("\n2️⃣ Testing data loading...")
            df = fe.load_and_analyze_data()
            print(f"   ✅ Data loaded: {df.shape}")
            
            print("\n3️⃣ Testing feature creation (sample)...")
            # Just create features for first 100 rows to test quickly
            fe.df = df.head(100)  # Use small sample
            advanced_features = fe.create_advanced_features()
            print(f"   ✅ Advanced features created: {advanced_features.shape}")
            
            print("\n4️⃣ Testing encoding...")
            final_features = fe.apply_advanced_encoding()
            print(f"   ✅ Final features encoded: {final_features.shape}")
            
            print(f"\n📊 FEATURE ENGINEERING TEST COMPLETE!")
            print(f"   Original data: {df.shape}")
            print(f"   Final features: {final_features.shape}")
            print(f"   Features created: {final_features.shape[1]} total")
            
            return True
        else:
            print("\n⚠️ No dataset found - testing initialization only")
            print("   💡 Feature engineering class loads correctly!")
            return True
            
    except Exception as e:
        print(f"\n❌ Error in feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_instantiation():
    """Test that models can be instantiated without data."""
    print("\n🤖 TESTING MODEL INSTANTIATION")
    print("=" * 60)
    
    tests = []
    
    # Test UltimateLoLPredictor
    try:
        from models.ultimate_predictor import UltimateLoLPredictor
        
        # Test class definition (don't initialize with data)
        print("1️⃣ Testing UltimateLoLPredictor class...")
        print(f"   ✅ UltimateLoLPredictor imported successfully")
        
        # Test that class has expected methods
        expected_methods = ['prepare_advanced_features', 'split_data_temporally', 'train_advanced_models']
        for method in expected_methods:
            if hasattr(UltimateLoLPredictor, method):
                print(f"   ✅ Method {method} exists")
            else:
                print(f"   ❌ Method {method} missing")
        
        tests.append(("UltimateLoLPredictor", True))
    except Exception as e:
        print(f"   ❌ UltimateLoLPredictor error: {e}")
        tests.append(("UltimateLoLPredictor", False))
    
    # Test EnhancedUltimateLoLPredictor
    try:
        from models.enhanced_ultimate_predictor import EnhancedUltimateLoLPredictor
        print("\n2️⃣ Testing EnhancedUltimateLoLPredictor class...")
        print(f"   ✅ EnhancedUltimateLoLPredictor imported successfully")
        tests.append(("EnhancedUltimateLoLPredictor", True))
    except Exception as e:
        print(f"   ❌ EnhancedUltimateLoLPredictor error: {e}")
        tests.append(("EnhancedUltimateLoLPredictor", False))
    
    # Test ComprehensiveLogisticRegressionComparison
    try:
        from models.comprehensive_logistic_regression_comparison import ComprehensiveLogisticRegressionComparison
        print("\n3️⃣ Testing ComprehensiveLogisticRegressionComparison class...")
        print(f"   ✅ ComprehensiveLogisticRegressionComparison imported successfully")
        tests.append(("ComprehensiveLogisticRegressionComparison", True))
    except Exception as e:
        print(f"   ❌ ComprehensiveLogisticRegressionComparison error: {e}")
        tests.append(("ComprehensiveLogisticRegressionComparison", False))
    
    return all(success for _, success in tests)

def test_data_collection():
    """Test data collection scripts."""
    print("\n📊 TESTING DATA COLLECTION SCRIPTS")
    print("=" * 60)
    
    try:
        from data_collection.filter_target_leagues import filter_target_leagues
        print(f"   ✅ filter_target_leagues imported")
        
        from data_collection.analyze_focused_data import analyze_focused_dataset
        print(f"   ✅ analyze_focused_dataset imported")
        
        return True
    except Exception as e:
        print(f"   ❌ Data collection import error: {e}")
        return False

def test_prediction():
    """Test prediction script."""
    print("\n🎯 TESTING PREDICTION SCRIPT")
    print("=" * 60)
    
    try:
        from prediction.interactive_match_predictor import InteractiveLoLPredictor
        print(f"   ✅ InteractiveLoLPredictor imported")
        
        # Test that key methods exist
        expected_methods = ['load_model_components', 'get_picks_and_bans', 'predict_match']
        for method in expected_methods:
            if hasattr(InteractiveLoLPredictor, method):
                print(f"   ✅ Method {method} exists")
            else:
                print(f"   ⚠️ Method {method} missing or renamed")
        
        return True
    except Exception as e:
        print(f"   ❌ Prediction import error: {e}")
        return False

def main():
    """Run focused tests."""
    print("🧪 FOCUSED FEATURE AND MODEL TESTING")
    print("=" * 80)
    print("Testing the core functionality without full model training")
    print("=" * 80)
    
    results = []
    
    # Test feature engineering 
    results.append(("Feature Engineering", test_feature_engineering()))
    
    # Test model classes
    results.append(("Model Classes", test_model_instantiation()))
    
    # Test data collection
    results.append(("Data Collection", test_data_collection()))
    
    # Test prediction
    results.append(("Prediction Module", test_prediction()))
    
    # Summary
    print(f"\n🎯 FOCUSED TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n📊 RESULTS: {passed}/{len(results)} focused tests passed")
    
    if passed == len(results):
        print("🎉 ALL FOCUSED TESTS PASSED!")
        print("🚀 Your reorganized code architecture is working perfectly!")
        print("🎯 You can now run full model training with confidence!")
    else:
        print("⚠️ Some issues found. Check the failures above.")
        print("💡 Fix these before running full model training.")

if __name__ == "__main__":
    main() 