# To-Do List: Make Project Runnable

The project has structural issues that prevent it from running. Below is a checklist of what needs to be done.

---

## Structure Fixes (Can be done locally)

- [ ] Create `__init__.py` files in all `src/` subdirectories:
  - `src/__init__.py`
  - `src/data_collection/__init__.py`
  - `src/data_processing/__init__.py`
  - `src/feature_engineering/__init__.py`
  - `src/prediction/__init__.py`
- [ ] Standardize data paths (fix `Data/` vs `data/` inconsistency across files)
- [ ] Fix or remove broken test imports that reference non-existent modules

---

## Files to Download from Google Cloud

- [ ] **Dataset:** `complete_target_leagues_dataset.csv` → place in `Data/` directory
- [ ] **Trained model:** `bayesian_best_model_Logistic_Regression.joblib` (or similar) → place in `models/bayesian_optimized_models/`
- [ ] **Scaler file** (if separate) → place alongside model

---

## Directories to Create (if not included in download)

- [ ] `Data/` (for datasets)
- [ ] `models/` (for trained models)
- [ ] `models/bayesian_optimized_models/`

---

## Missing Code (may need to recover from Google Cloud or recreate)

These files are referenced by tests but don't exist in the repository:

- [ ] `src/models/ultimate_predictor.py`
- [ ] `src/models/enhanced_ultimate_predictor.py`
- [ ] `src/models/comprehensive_logistic_regression_comparison.py`

---

## Verification Steps (after completing above)

1. Run `pip install -r requirements.txt`
2. Run tests: `python -m pytest tests/`
3. Run interactive predictor: `python src/prediction/interactive_match_predictor.py`
