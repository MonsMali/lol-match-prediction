# Requirements: LoL Draft Predictor Web App

**Defined:** 2026-02-24
**Core Value:** Users can simulate a professional draft and instantly see predicted win probability for each team

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Inference Refactor

- [ ] **INFER-01**: Prediction pipeline runs without loading the full 37K-row CSV dataset
- [ ] **INFER-02**: Pre-serialized artifacts (champion meta, team stats, synergies) load directly from joblib files
- [ ] **INFER-03**: Clean adapter interface wraps InteractiveLoLPredictor for API consumption (accepts draft dict, returns probabilities)
- [ ] **INFER-04**: Inference runs within 512MB RAM budget (Render free tier constraint)

### Backend API

- [ ] **API-01**: FastAPI REST endpoint accepts a complete draft (teams, picks, bans, roles) and returns win probability for each team
- [ ] **API-02**: ML model loads once at startup via FastAPI lifespan context manager, stored in app.state
- [ ] **API-03**: Champion list endpoint returns all valid champions with metadata (name, image URL from Data Dragon)
- [ ] **API-04**: Team list endpoint returns professional teams grouped by league (LCK, LEC, LCS, LPL)
- [ ] **API-05**: CORS configured to allow frontend requests
- [ ] **API-06**: Health/readiness endpoint returns model status and version info
- [ ] **API-07**: Model file upload endpoint allows swapping in newly trained model files (security-gated with secret token)
- [ ] **API-08**: Data Dragon champion name mapping handles mismatches (Wukong=MonkeyKing, Nunu, etc.)

### Frontend Draft Board

- [x] **DRAFT-01**: Champion portrait grid displays all champions using Riot Data Dragon CDN images
- [x] **DRAFT-02**: Champion search/filter allows finding champions by name with fuzzy matching
- [x] **DRAFT-03**: Two-sided draft board shows blue team and red team with 5 pick slots and 5 ban slots each
- [x] **DRAFT-04**: Team selection interface for major professional leagues (LCK, LEC, LCS, LPL)
- [x] **DRAFT-05**: Role assignment UI maps picked champions to positions (Top, Jungle, Mid, Bot, Support)
- [x] **DRAFT-06**: Win probability display shows percentage for each team after draft completes
- [x] **DRAFT-07**: LoL-themed visual design with champion portraits, team colors, and draft board resembling pro broadcast

### Live Draft Mode

- [x] **LIVE-01**: Step-by-step draft mode follows the real professional draft order (ban-ban-ban-pick-pick-pick sequence)
- [x] **LIVE-02**: User enters one champion at a time as it happens on broadcast
- [x] **LIVE-03**: Draft board updates visually with each pick/ban selection

### Bulk Entry Mode

- [x] **BULK-01**: Bulk entry mode allows entering all 10 bans and 10 picks at once
- [x] **BULK-02**: Quick prediction without following step-by-step draft sequence

### Best-of-Series

- [ ] **BOS-01**: Best-of-series toggle supports BO3 and BO5 series formats
- [ ] **BOS-02**: Series score tracking across multiple games
- [ ] **BOS-03**: User can input a new draft for each game in the series

### Deployment

- [ ] **DEPLOY-01**: Application deploys on Render free tier as single-process (uvicorn)
- [ ] **DEPLOY-02**: Frontend SPA built and served by FastAPI as static files (single deployment)
- [ ] **DEPLOY-03**: Production requirements.txt excludes training-only dependencies (XGBoost, LightGBM, CatBoost, Optuna)
- [ ] **DEPLOY-04**: Cold-start UX shows loading screen while model initializes (~30s on free tier)
- [ ] **DEPLOY-05**: Keep-alive strategy reduces cold start frequency for active periods

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Enhanced Predictions

- **EPRED-01**: Partial-draft inference shows probability updates as picks/bans are entered (before draft completes)
- **EPRED-02**: Feature importance breakdown shows which factors drove the prediction
- **EPRED-03**: Confidence indicator shows model certainty level

### History and Analytics

- **HIST-01**: Shareable prediction URLs allow sharing draft predictions with others
- **HIST-02**: Past predictions log for reviewing previous sessions
- **HIST-03**: Model accuracy tracking dashboard

## Out of Scope

| Feature | Reason |
|---------|--------|
| User accounts / authentication | Public tool, no login needed |
| In-game live stats or real-time game data | Prediction is pre-match only (thesis design) |
| Pick/ban recommendations | Model is a classifier, not a counterfactual ranker |
| Mobile-native app | Web-responsive is sufficient |
| Automated Oracle's Elixir data ingestion | Manual model updates via upload |
| Browser-only ONNX inference | Python backend required for feature engineering |
| Multiple concurrent model versions | Single model at a time is sufficient for v1 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFER-01 | Phase 1 - ML Adapter | Pending |
| INFER-02 | Phase 1 - ML Adapter | Pending |
| INFER-03 | Phase 1 - ML Adapter | Pending |
| INFER-04 | Phase 1 - ML Adapter | Pending |
| API-01 | Phase 2 - FastAPI Backend | Pending |
| API-02 | Phase 2 - FastAPI Backend | Pending |
| API-03 | Phase 2 - FastAPI Backend | Pending |
| API-04 | Phase 2 - FastAPI Backend | Pending |
| API-05 | Phase 2 - FastAPI Backend | Pending |
| API-06 | Phase 2 - FastAPI Backend | Pending |
| API-07 | Phase 2 - FastAPI Backend | Pending |
| API-08 | Phase 2 - FastAPI Backend | Pending |
| DRAFT-01 | Phase 3 - React Draft Board | Complete |
| DRAFT-02 | Phase 3 - React Draft Board | Complete |
| DRAFT-03 | Phase 3 - React Draft Board | Complete |
| DRAFT-04 | Phase 3 - React Draft Board | Complete |
| DRAFT-05 | Phase 3 - React Draft Board | Complete |
| DRAFT-06 | Phase 3 - React Draft Board | Complete |
| DRAFT-07 | Phase 3 - React Draft Board | Complete |
| LIVE-01 | Phase 3 - React Draft Board | Complete |
| LIVE-02 | Phase 3 - React Draft Board | Complete |
| LIVE-03 | Phase 3 - React Draft Board | Complete |
| BULK-01 | Phase 3 - React Draft Board | Complete |
| BULK-02 | Phase 3 - React Draft Board | Complete |
| BOS-01 | Phase 3 - React Draft Board | Pending |
| BOS-02 | Phase 3 - React Draft Board | Pending |
| BOS-03 | Phase 3 - React Draft Board | Pending |
| DEPLOY-01 | Phase 4 - Integration and Deployment | Pending |
| DEPLOY-02 | Phase 4 - Integration and Deployment | Pending |
| DEPLOY-03 | Phase 4 - Integration and Deployment | Pending |
| DEPLOY-04 | Phase 4 - Integration and Deployment | Pending |
| DEPLOY-05 | Phase 4 - Integration and Deployment | Pending |

**Coverage:**
- v1 requirements: 31 total
- Mapped to phases: 31
- Unmapped: 0

---
*Requirements defined: 2026-02-24*
*Last updated: 2026-02-24 after roadmap creation*
