# Roadmap: LoL Draft Predictor Web App

## Overview

The project wraps an existing, trained ML model (82.97% AUC-ROC Logistic Regression) in a public-facing web application. Work proceeds in four phases that follow a strict dependency chain: the ML inference layer must be refactored first, then the API backend is built against it, then the React draft board is built against the API contract, and finally the full stack is integrated and deployed to Render's free tier. No phase delivers value to a public user until Phase 4 completes — earlier phases build internal capability that Phase 4 ships.

## Phases

- [ ] **Phase 1: ML Adapter** - Refactor the inference path to eliminate the full CSV load and expose a clean `predict_from_draft` interface
- [ ] **Phase 2: FastAPI Backend** - Build the REST API layer with all prediction, champion, team, and model-management endpoints
- [ ] **Phase 3: React Draft Board** - Build the complete frontend draft simulator with both draft modes, role assignment, and win probability display
- [ ] **Phase 4: Integration and Deployment** - Connect frontend to live backend, deploy to Render, and implement cold-start UX

## Phase Details

### Phase 1: ML Adapter
**Goal**: The prediction pipeline runs as a lightweight, artifact-only process callable from Python without loading the 37K-row CSV
**Depends on**: Nothing (first phase)
**Requirements**: INFER-01, INFER-02, INFER-03, INFER-04

**Success Criteria** (what must be TRUE):
  1. A Python call to `predict_from_draft(draft_dict)` returns win probabilities for blue and red team without loading any CSV file
  2. Inference uses only pre-serialized `.joblib` artifacts (champion meta strength, team history, champion synergies) loaded at import time
  3. The adapter runs within 512 MB RAM on a cold Python process (measured with `memory_profiler`)
  4. The adapter accepts a structured draft dict and returns a typed result object with blue/red win probability

**Plans**: 3 plans

Plans:
- [x] 01-01-PLAN.md -- Audit artifacts, resolve open questions, create schemas and normalization
- [x] 01-02-PLAN.md -- Build validation, feature computation, and LoLDraftAdapter core
- [ ] 01-03-PLAN.md -- Validate memory footprint and end-to-end prediction correctness

---

### Phase 2: FastAPI Backend
**Goal**: A running FastAPI service exposes the ML adapter as REST endpoints with correct schemas, CORS, model lifecycle management, and champion/team data endpoints
**Depends on**: Phase 1
**Requirements**: API-01, API-02, API-03, API-04, API-05, API-06, API-07, API-08

**Success Criteria** (what must be TRUE):
  1. `POST /api/predict` accepts a complete draft payload and returns win probabilities for both teams
  2. `GET /api/champions` returns all valid champions with name, DDragon image URL, and internal key (handles Wukong/MonkeyKing and Nunu mismatches)
  3. `GET /api/teams` returns professional teams grouped by league (LCK, LEC, LCS, LPL)
  4. `GET /health` returns model status and version info immediately, even before the model finishes loading
  5. The ML model loads exactly once at startup via FastAPI lifespan and is accessible from `app.state` for all subsequent requests

**Plans**: TBD

Plans:
- [ ] 02-01: Scaffold FastAPI app with lifespan model loading, Pydantic schemas, and CORS middleware
- [ ] 02-02: Implement `/api/predict` endpoint calling adapter via `asyncio.to_thread`
- [ ] 02-03: Implement `/api/champions`, `/api/teams`, `/health`, and model upload endpoints
- [ ] 02-04: Pin scikit-learn version, build production requirements.txt, and verify startup in isolation

---

### Phase 3: React Draft Board
**Goal**: Users can enter a complete professional draft (both bulk and step-by-step), assign roles, select teams, and see the win probability display — all in a LoL-themed UI
**Depends on**: Phase 2 (API contract established; frontend can develop against schema definitions)
**Requirements**: DRAFT-01, DRAFT-02, DRAFT-03, DRAFT-04, DRAFT-05, DRAFT-06, DRAFT-07, LIVE-01, LIVE-02, LIVE-03, BULK-01, BULK-02, BOS-01, BOS-02, BOS-03

**Success Criteria** (what must be TRUE):
  1. User can search for any champion by name and see their portrait from Riot Data Dragon CDN; duplicate picks/bans are visually prevented
  2. User can select professional teams from LCK, LEC, LCS, and LPL leagues and the draft board displays both teams with 5 pick slots and 5 ban slots each
  3. In step-by-step mode, the draft board enforces the real professional draft order (ban-pick sequence), advancing one slot at a time as the user enters each champion
  4. In bulk entry mode, the user can fill all 10 bans and 10 picks at once and submit immediately for a prediction
  5. After completing a draft, the win probability for blue team and red team is displayed as a split bar with percentage values
  6. User can assign a role (Top, Jungle, Mid, Bot, Support) to each picked champion before submitting
  7. User can toggle BO3 or BO5 series mode, track series score, and enter a fresh draft for each game

**Plans**: TBD

Plans:
- [ ] 03-01: Scaffold React + Vite + TypeScript + Tailwind project; set up Zustand draft store and TanStack Query
- [ ] 03-02: Build ChampionGrid component with DDragon portrait loading, search/filter, and duplicate prevention
- [ ] 03-03: Build two-sided draft board layout (blue/red split, pick/ban slots, team selector, role assignment UI)
- [ ] 03-04: Implement step-by-step live draft mode with enforced professional draft order
- [ ] 03-05: Implement bulk entry mode and win probability display component
- [ ] 03-06: Add best-of-series toggle, series score tracking, and per-game draft entry

---

### Phase 4: Integration and Deployment
**Goal**: The full stack runs as a single Render deployment where a browser user can complete a real draft against the live ML model and receive a win probability prediction
**Depends on**: Phase 3
**Requirements**: DEPLOY-01, DEPLOY-02, DEPLOY-03, DEPLOY-04, DEPLOY-05

**Success Criteria** (what must be TRUE):
  1. Visiting the Render URL in a browser loads the React app and a user can complete a full draft and receive a prediction from the real ML model without any local setup
  2. After the service spins down from idle, the first page load shows a "warming up" loading indicator and prediction requests succeed once the model finishes loading (no 504 errors to the user)
  3. A keep-alive cron job runs at 10-minute intervals and reduces cold starts during active usage periods
  4. The production deployment uses a trimmed requirements file that excludes XGBoost, LightGBM, CatBoost, and Optuna

**Plans**: TBD

Plans:
- [ ] 04-01: Build React SPA with `npm run build` and configure FastAPI StaticFiles to serve Vite dist output
- [ ] 04-02: Write Render deployment config (`render.yaml`), trim production requirements, and validate first deploy
- [ ] 04-03: Implement cold-start UX (frontend warm-up ping, loading screen) and configure keep-alive cron

---

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. ML Adapter | 2/3 | In progress | - |
| 2. FastAPI Backend | 0/4 | Not started | - |
| 3. React Draft Board | 0/6 | Not started | - |
| 4. Integration and Deployment | 0/3 | Not started | - |
