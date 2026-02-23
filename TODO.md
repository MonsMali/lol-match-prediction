# Next Steps Todo List

## GPU Training Support

GPU training is possible for some models in your stack:

| Model | GPU Support | How to Enable |
|-------|-------------|---------------|
| **XGBoost** | Yes | `tree_method='gpu_hist', device='cuda'` |
| **LightGBM** | Yes | `device='gpu'` |
| **CatBoost** | Yes (excellent) | `task_type='GPU'` |
| **Random Forest** | No | CPU only (scikit-learn) |
| **Logistic Regression** | No | CPU only (scikit-learn) |
| **Neural Networks** | Yes | Requires PyTorch/TensorFlow |

To enable GPU, modify the model configurations in `src/models/trainer.py`.

---

## Prioritized Tasks

### High Priority

- [ ] **Test the new system end-to-end**
  - Run `python src/models/trainer.py` to verify multi-metric evaluation works
  - Check that composite scoring ranks models correctly

- [ ] **Test data pipeline**
  - Run `python src/data/pipeline.py --status` to check pipeline status
  - Try incremental download: `python src/data/downloader.py --incremental`

- [ ] **Enable GPU training** (if you want faster training)
  - Modify XGBoost/LightGBM/CatBoost configs in `src/models/trainer.py`
  - Install CUDA-enabled versions of the libraries

### Medium Priority

- [ ] **Run full training with new metrics**
  - Execute complete training run to get baseline composite scores
  - Save the best model to production

- [ ] **Set up continuous learning baseline**
  - Run `python src/training/trainer.py --train` to create first versioned model
  - This establishes the baseline for drift detection

### Lower Priority

- [ ] **Web app development** (mentioned in CLAUDE.md as future work)
  - Design API endpoints for predictions
  - Create frontend for draft simulation

---

## Quick Reference Commands

```bash
# Check everything works
cd /mnt/d/Tese
pip install -r requirements.txt

# Test training with new composite metrics
python src/models/trainer.py

# Check data pipeline status
python src/data/pipeline.py --status

# Download latest data
python src/data/downloader.py --incremental

# Run continuous training
python src/training/trainer.py --train --quick

# Check drift detection
python src/training/trainer.py --status
```

---

## New Features Implemented (2024)

### Multi-Metric Evaluation (`src/evaluation/`)
- Composite scoring: AUC (30%), Log Loss (25%), Brier (20%), ECE (15%), F1 (10%)
- Models now ranked by composite score instead of F1 only

### Data Pipeline (`src/data/`)
- `downloader.py` - Automated Oracle's Elixir S3 downloads
- `schema.py` - Schema validation and adaptation
- `pipeline.py` - Full/incremental data pipeline

### Continuous Learning (`src/training/`)
- `versioning.py` - Model version management with promotion/rollback
- `drift.py` - Performance and feature drift detection
- `scheduler.py` - Scheduled and trigger-based retraining
- `trainer.py` - Continuous training with auto-promotion
