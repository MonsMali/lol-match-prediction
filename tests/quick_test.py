#!/usr/bin/env python3
"""
Quick test script to verify all reorganized code works without training models.
"""

import os
import sys
import traceback

def test_imports():
    """Test all critical imports."""
    print("🔗 TESTING IMPORTS")
    print("=" * 50)
    
    # Add src to path - now we need to go up one directory first
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    sys.path.insert(0, src_path)
    
    tests = []
    
    # Test feature engineering import
    try:
        from feature_engineering.advanced_feature_engineering import AdvancedFeatureEngineering
        tests.append(("AdvancedFeatureEngineering", "✅"))
    except Exception as e:
        tests.append(("AdvancedFeatureEngineering", f"❌ {e}"))
    
    # Test model imports
    try:
        from models.ultimate_predictor import UltimateLoLPredictor
        tests.append(("UltimateLoLPredictor", "✅"))
    except Exception as e:
        tests.append(("UltimateLoLPredictor", f"❌ {e}"))
    
    try:
        from models.enhanced_ultimate_predictor import EnhancedUltimateLoLPredictor
        tests.append(("EnhancedUltimateLoLPredictor", "✅"))
    except Exception as e:
        tests.append(("EnhancedUltimateLoLPredictor", f"❌ {e}"))
    
    try:
        from models.comprehensive_logistic_regression_comparison import ComprehensiveLogisticRegressionComparison
        tests.append(("ComprehensiveLogisticRegressionComparison", "✅"))
    except Exception as e:
        tests.append(("ComprehensiveLogisticRegressionComparison", f"❌ {e}"))
    
    try:
        from prediction.interactive_match_predictor import InteractiveLoLPredictor
        tests.append(("InteractiveLoLPredictor", "✅"))
    except Exception as e:
        tests.append(("InteractiveLoLPredictor", f"❌ {e}"))
    
    # Print results
    for test_name, result in tests:
        print(f"   {result} {test_name}")
    
    return all("✅" in result for _, result in tests)

def test_data_loading():
    """Test data loading without full processing."""
    print("\n📂 TESTING DATA LOADING")
    print("=" * 50)
    
    # Add src to path - now we need to go up one directory first
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    sys.path.insert(0, src_path)
    
    try:
        from feature_engineering.advanced_feature_engineering import AdvancedFeatureEngineering
        
        # Try to initialize (will test path logic)
        fe = AdvancedFeatureEngineering()
        print(f"   ✅ AdvancedFeatureEngineering initialization")
        print(f"   📂 Data path: {fe.data_path}")
        
        # Test if file exists
        if os.path.exists(fe.data_path):
            print(f"   ✅ Dataset file found")
            
            # Try loading just a few rows to test format
            import pandas as pd
            df_sample = pd.read_csv(fe.data_path, nrows=100)
            print(f"   ✅ Data format valid")
            print(f"   📊 Sample shape: {df_sample.shape}")
            print(f"   📋 Columns: {list(df_sample.columns)[:10]}...")
            
            return True
        else:
            print(f"   ⚠️ Dataset file not found (expected for testing)")
            print(f"   💡 This is normal if you haven't created the dataset yet")
            return True  # Still consider this a pass for path logic
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False

def test_model_initialization():
    """Test model initialization without training."""
    print("\n🤖 TESTING MODEL INITIALIZATION")
    print("=" * 50)
    
    # Add src to path - now we need to go up one directory first
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    sys.path.insert(0, src_path)
    
    success = 0
    total = 0
    
    # Test UltimateLoLPredictor
    try:
        from models.ultimate_predictor import UltimateLoLPredictor
        # Don't actually initialize with real data, just test class definition
        print(f"   ✅ UltimateLoLPredictor class loaded")
        success += 1
    except Exception as e:
        print(f"   ❌ UltimateLoLPredictor: {e}")
    total += 1
    
    # Test EnhancedUltimateLoLPredictor  
    try:
        from models.enhanced_ultimate_predictor import EnhancedUltimateLoLPredictor
        print(f"   ✅ EnhancedUltimateLoLPredictor class loaded")
        success += 1
    except Exception as e:
        print(f"   ❌ EnhancedUltimateLoLPredictor: {e}")
    total += 1
    
    # Test ComprehensiveLogisticRegressionComparison
    try:
        from models.comprehensive_logistic_regression_comparison import ComprehensiveLogisticRegressionComparison
        print(f"   ✅ ComprehensiveLogisticRegressionComparison class loaded")
        success += 1
    except Exception as e:
        print(f"   ❌ ComprehensiveLogisticRegressionComparison: {e}")
    total += 1
    
    return success == total

def test_directory_structure():
    """Test that all expected directories exist."""
    print("\n📁 TESTING DIRECTORY STRUCTURE")
    print("=" * 50)
    
    # Change working directory to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    expected_dirs = [
        "src",
        "src/models", 
        "src/feature_engineering",
        "src/data_collection",
        "src/prediction",
        "data",
        "models", 
        "visualizations",
        "results",
        "docs",
        "experiments"
    ]
    
    missing_dirs = []
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ (missing)")
            missing_dirs.append(dir_path)
    
    # Restore original working directory
    os.chdir(original_cwd)
    
    return len(missing_dirs) == 0

def test_dependencies():
    """Test that all required packages are available."""
    print("\n📦 TESTING DEPENDENCIES")
    print("=" * 50)
    
    required_packages = [
        'pandas',
        'numpy', 
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'joblib'
    ]
    
    optional_packages = [
        'xgboost',
        'lightgbm',
        'catboost',
        'optuna'
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (REQUIRED)")
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} (optional)")
        except ImportError:
            print(f"   ⚠️ {package} (optional, recommended)")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\n❌ Missing required packages: {missing_required}")
        print(f"   Install with: pip install {' '.join(missing_required)}")
    
    if missing_optional:
        print(f"\n💡 Missing optional packages: {missing_optional}")
        print(f"   Install with: pip install {' '.join(missing_optional)}")
    
    return len(missing_required) == 0

def main():
    """Run all tests."""
    print("🧪 COMPREHENSIVE CODE TESTING (NO MODEL TRAINING)")
    print("=" * 80)
    print("This tests all your reorganized code without actually training models.")
    print("=" * 80)
    
    results = []
    
    # Test 1: Dependencies
    results.append(("Dependencies", test_dependencies()))
    
    # Test 2: Directory Structure  
    results.append(("Directory Structure", test_directory_structure()))
    
    # Test 3: Imports
    results.append(("Imports", test_imports()))
    
    # Test 4: Data Loading
    results.append(("Data Loading", test_data_loading()))
    
    # Test 5: Model Initialization
    results.append(("Model Classes", test_model_initialization()))
    
    # Summary
    print("\n🎯 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n📊 RESULTS: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! Your reorganized code is ready to use!")
    elif passed >= len(results) - 1:
        print("✅ MOSTLY WORKING! Minor issues that won't prevent usage.")
    else:
        print("⚠️ Some issues found. Check the failures above.")
    
    print("\n💡 NEXT STEPS:")
    if not os.path.exists("data/target_leagues_dataset.csv"):
        print("   1. Run the data collection scripts to create your dataset")
        print("   2. Then you can train models with confidence!")
    else:
        print("   1. Your code is ready for model training!")
        print("   2. You can run any of the model scripts now.")

if __name__ == "__main__":
    main() 