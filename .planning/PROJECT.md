# LoL Draft Predictor Web App

## What This Is

A public-facing web application that lets users simulate professional League of Legends champion draft phases (picks and bans) and get real-time win probability predictions powered by the trained ML model from the thesis. The app features a LoL-themed UI with champion portraits from Riot's Data Dragon CDN, a visual draft board resembling the actual pro draft screen, and both step-by-step (live viewing) and bulk entry (quick lookup) modes.

## Core Value

Users can simulate a professional draft as it happens live and instantly see the predicted win probability for each team — turning the thesis ML model into an interactive, publicly accessible tool.

## Requirements

### Validated

- ML prediction pipeline (AdvancedFeatureEngineering + Logistic Regression model) — existing
- Champion validation with fuzzy matching — existing
- Professional draft order logic (ban/pick sequence) — existing
- Team database for major leagues (LCK, LEC, LCS, LPL) — existing
- Best-of-series prediction support — existing

### Active

- [ ] Web-based draft simulator with LoL-themed visual draft board
- [ ] Step-by-step draft mode (follow live broadcast, one champion at a time)
- [ ] Bulk entry draft mode (enter all picks/bans at once for quick predictions)
- [ ] Champion search/selection with portraits from Riot Data Dragon CDN
- [ ] Team selection interface for major professional leagues
- [ ] Real-time win probability display after draft completes
- [ ] Role assignment UI for mapping picked champions to positions
- [ ] Python API backend exposing the prediction pipeline as REST endpoints
- [ ] Model file upload/swap capability for loading newly trained models
- [ ] Deployment on free cloud tier (Render or Railway)

### Out of Scope

- In-game live stats or real-time game data feeds — prediction is pre-match only
- User accounts or authentication — public tool, no login needed
- Historical prediction tracking or accuracy dashboards — keep it focused on predictions
- Mobile-native app — web-responsive is sufficient
- Automated data ingestion from Oracle's Elixir — manual model updates via upload
- Browser-only model inference (ONNX conversion) — using Python backend instead

## Context

This is the web deployment phase of a Master's thesis project at Aarhus University titled "Novel Temporal Validation for Evolving Competitive Environments - A League of Legends Machine Learning Framework." The core ML system already exists with:

- **Best model**: Logistic Regression achieving 82.97% AUC-ROC
- **Dataset**: 37,502 professional matches (2014-2024) from LPL, LCK, LCS, LEC, Worlds, MSI
- **Features**: 33+ advanced engineered features using only pre-match information
- **Existing predictor**: `src/prediction/interactive_match_predictor.py` — a fully working CLI predictor with draft simulation, champion validation, and prediction logic that the web app will wrap

The prediction pipeline requires Python (scikit-learn, pandas, numpy, joblib) to run, so a Python backend is necessary. The existing `InteractiveLoLPredictor` class handles model loading, feature engineering setup, champion validation, and prediction — the web app essentially provides a UI layer and API on top of this.

Champion assets (portraits, splash art) will be pulled from Riot's Data Dragon CDN, which is free and always current with the latest patches.

The user is currently retraining models on Google Colab (T4 GPU) with updated data and wants the ability to swap in new model files after training completes.

## Constraints

- **Hosting**: Free cloud tier (Render or Railway) — must stay within free tier limits
- **Backend**: Python required — model depends on scikit-learn, pandas, numpy, joblib
- **Assets**: Champion images from Riot Data Dragon CDN — no bundled image assets
- **Data privacy**: No user data collection, no authentication, no cookies beyond essential
- **Model size**: Joblib model files must be small enough for free tier memory limits
- **Cold start**: Free tier services spin down when idle — acceptable ~30s cold start

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| All-on-free-cloud (not GitHub Pages + API split) | Simpler single deployment, Python backend needed anyway | -- Pending |
| Riot Data Dragon for champion assets | Free, always current, no manual asset management | -- Pending |
| LoL-themed UI with visual draft board | Public-facing tool should feel authentic to the game | -- Pending |
| Both step-by-step and bulk draft modes | Step-by-step for live viewing, bulk for quick lookups | -- Pending |
| Prediction-only scope (no history/insights) | Keep v1 focused and shippable | -- Pending |

---
*Last updated: 2026-02-24 after initialization*
