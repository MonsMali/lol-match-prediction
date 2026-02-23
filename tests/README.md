# Testing Utilities

This directory contains testing scripts to verify the thesis code works correctly without running full model training.

## Test Scripts

### 🧪 `quick_test.py`
**Comprehensive system test**
- Tests all imports and dependencies
- Verifies directory structure
- Tests data loading
- Validates model class definitions

```bash
python tests/quick_test.py
```

### 🔧 `test_model_features.py`
**Focused feature engineering test**
- Tests feature engineering pipeline with sample data
- Validates model instantiation
- Tests data collection scripts
- Checks prediction module

```bash
python tests/test_model_features.py
```

## When to Use

- **Before model training**: Verify everything works
- **After code changes**: Ensure nothing broke
- **Debugging**: Isolate issues without waiting for training
- **Demonstration**: Show advisors the code works

## Expected Results

Both scripts should show:
- ✅ All tests PASSED
- 🎉 Ready for model training

If tests fail, check the specific error messages and fix before proceeding with model training. 