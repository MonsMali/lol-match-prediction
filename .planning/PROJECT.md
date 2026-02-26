# LoL Draft Predictor Web App

## What This Is

A public-facing web application that lets users simulate professional League of Legends champion draft phases (picks and bans) and get real-time win probability predictions powered by a trained ML model (VotingClassifier, 82.97% AUC-ROC). The app features a LoL-themed UI with champion portraits from Riot's Data Dragon CDN, a visual draft board with both step-by-step (live viewing) and bulk entry (quick lookup) modes, best-of-series tracking, and deploys as a single service on Render's free tier.

## Core Value

Users can simulate a professional draft as it happens live and instantly see the predicted win probability for each team -- turning the thesis ML model into an interactive, publicly accessible tool.

## Current State

Shipped v1.0 on 2026-02-26 with 3,451 LOC (1,728 Python + 1,723 TypeScript) across 44 files.

Tech stack: Python (FastAPI, scikit-learn 1.5.0, joblib, numpy) + TypeScript (React, Vite, Tailwind v4, Zustand, TanStack Query).

All 31 v1 requirements satisfied. Deployed as single Render free-tier service with cold-start warm-up UX.

## Requirements

### Validated

- Lightweight ML adapter runs predictions from joblib artifacts (167 MB RSS, no CSV loading) -- v1.0
- FastAPI REST API with predict, champions, teams, health endpoints -- v1.0
- Champion list with DDragon image URLs and name mismatch handling (12 special cases) -- v1.0
- Team list grouped by league (LCK, LEC, LCS, LPL) -- v1.0
- CORS and model lifespan management -- v1.0
- Admin model upload with bearer token security -- v1.0
- React draft board with LoL-themed dark UI, gold accents, team colors -- v1.0
- Champion search/filter with DDragon CDN portraits -- v1.0
- Step-by-step live draft mode following 20-step professional ban/pick order -- v1.0
- Bulk entry mode for quick predictions -- v1.0
- Role assignment UI (Top, Jungle, Mid, Bot, Support) -- v1.0
- Win probability split bar display -- v1.0
- Best-of-series tracker (BO3/BO5) with score tracking -- v1.0
- Single Render free-tier deployment (FastAPI serves SPA + API) -- v1.0
- Production requirements excluding training-only dependencies -- v1.0
- Cold-start warm-up screen polling /health until model ready -- v1.0
- Keep-alive strategy via external cron -- v1.0

### Active

(None -- start next milestone to define new requirements)

### Out of Scope

- In-game live stats or real-time game data feeds -- prediction is pre-match only (thesis design)
- User accounts or authentication -- public tool, no login needed
- Historical prediction tracking or accuracy dashboards -- keep it focused
- Mobile-native app -- web-responsive is sufficient
- Automated data ingestion from Oracle's Elixir -- manual model updates via upload
- Browser-only model inference (ONNX conversion) -- Python backend required for feature engineering
- Partial-draft inference (mid-draft probability updates) -- deferred to v2
- Feature importance breakdown -- deferred to v2
- Shareable prediction URLs -- deferred to v2

## Context

This is the web deployment phase of a Master's thesis at Aarhus University titled "Novel Temporal Validation for Evolving Competitive Environments - A League of Legends Machine Learning Framework."

The core ML system exists with:
- **Production model**: VotingClassifier (RF + GBM + LR + SVM) with 27 features, achieving 82.97% AUC-ROC
- **Dataset**: 37,502 professional matches (2014-2024) from LPL, LCK, LCS, LEC, Worlds, MSI
- **Features**: 27 advanced engineered features using only pre-match information (no in-game stats)

Known technical constraints:
- scikit-learn pinned to 1.5.0 (model incompatible with >= 1.6)
- League inference hardcoded to LCK (no team-to-league mapping in artifacts)
- DDragon version pinned to 14.24.1

## Constraints

- **Hosting**: Render free tier -- single-process uvicorn, ~30s cold start acceptable
- **Backend**: Python required -- model depends on scikit-learn 1.5.0, joblib, numpy
- **Assets**: Champion images from Riot Data Dragon CDN -- no bundled image assets
- **Data privacy**: No user data collection, no authentication, no cookies beyond essential
- **Memory**: 512 MB limit -- adapter uses 167 MB, leaving 344 MB headroom for FastAPI

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Single Render deployment (FastAPI serves SPA + API) | Simpler than split deployment, one cold start | Good -- works well, single process handles both |
| Riot Data Dragon CDN for champion assets | Free, always current, no manual asset management | Good -- 167 champions load correctly |
| LoL-themed UI with visual draft board | Public-facing tool should feel authentic | Good -- dark theme with team colors |
| Both step-by-step and bulk draft modes | Step-by-step for live viewing, bulk for quick lookups | Good -- both modes fully functional |
| Prediction-only scope (no history/insights) | Keep v1 focused and shippable | Good -- shipped in 3 days |
| VotingClassifier instead of LogisticRegression | Production artifact was VotingClassifier, not pure LR | Good -- adapter targets actual production model |
| scikit-learn pinned to 1.5.0 | Model fails to load on newer versions | Revisit -- tech debt for future model retraining |
| useChampionLookup hook for image URLs | Slot components need API image_url, not constructed URLs | Good -- handles all 12 special-name champions |
| Raw fetch for WarmUpScreen health polling | Works independently of QueryClient, avoids race conditions | Good -- reliable cold-start UX |
| autoDeploy: false in render.yaml | Prevents accidental deployments | Good -- manual deploy control |

---
*Last updated: 2026-02-26 after v1.0 milestone*
