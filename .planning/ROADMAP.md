# Roadmap: LoL Draft Predictor Web App

## Overview

The project wraps an existing, trained ML model (82.97% AUC-ROC Logistic Regression) in a public-facing web application. Work proceeds in four phases that follow a strict dependency chain: the ML inference layer must be refactored first, then the API backend is built against it, then the React draft board is built against the API contract, and finally the full stack is integrated and deployed to Render's free tier. No phase delivers value to a public user until Phase 4 completes — earlier phases build internal capability that Phase 4 ships.

## Phases

- [x] **Phase 1: ML Adapter** - Refactor the inference path to eliminate the full CSV load and expose a clean `predict_from_draft` interface
- [x] **Phase 2: FastAPI Backend** - Build the REST API layer with all prediction, champion, team, and model-management endpoints
- [x] **Phase 3: React Draft Board** - Build the complete frontend draft simulator with both draft modes, role assignment, and win probability display
- [ ] **Phase 4: Integration and Deployment** - Connect frontend to live backend, deploy to Render, and implement cold-start UX
- [ ] **Phase 5: DDragon Image URL Fix** - Fix broken champion portrait URLs in slot components for 12 special-name champions

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
- [x] 01-03-PLAN.md -- Validate memory footprint and end-to-end prediction correctness

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

**Plans**: 3 plans

Plans:
- [x] 02-01-PLAN.md -- Scaffold FastAPI app with lifespan, Pydantic schemas, CORS, champion mapping, team data, and health endpoint
- [x] 02-02-PLAN.md -- Implement predict, champions, and teams routers with integration smoke test
- [x] 02-03-PLAN.md -- Admin model upload endpoint with hot-swap and production requirements file

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

**Plans**: 5 plans

Plans:
- [x] 03-01-PLAN.md -- Scaffold Vite + React + TypeScript + Tailwind v4 project with LoL theme, Zustand draft store, and TanStack Query hooks
- [x] 03-02-PLAN.md -- Build ChampionGrid with DDragon portraits and search, draft board layout with team panels, slots, selectors, and mode toggle
- [x] 03-03-PLAN.md -- Wire step-by-step live draft mode with 20-step professional draft order and visual turn indicators
- [x] 03-04-PLAN.md -- Implement bulk entry mode, role assignment UI, prediction submission, and win probability display
- [x] 03-05-PLAN.md -- Add best-of-series tracker (BO3/BO5) with score tracking and visual verification checkpoint

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

**Plans**: 2 plans

Plans:
- [ ] 04-01-PLAN.md -- Build React SPA, configure FastAPI SPA catch-all, validate production requirements
- [ ] 04-02-PLAN.md -- Create WarmUpScreen cold-start UX component and render.yaml deployment config

---

### Phase 5: DDragon Image URL Fix
**Goal**: All champion portraits render correctly in slot components, including the 12 champions whose DDragon ID differs from display name
**Depends on**: Phase 3
**Requirements**: DRAFT-07, API-08 (visual correctness fix)
**Gap Closure:** Closes integration gap from v1.0 audit

**Success Criteria** (what must be TRUE):
  1. `PickSlot.tsx`, `BanRow.tsx`, and `RoleAssignment.tsx` use the API-provided `image_url` instead of constructing URLs from display names
  2. All 12 affected champions (Wukong, Kai'Sa, Nunu & Willump, Kha'Zix, Cho'Gath, Vel'Koz, Rek'Sai, Kog'Maw, Bel'Veth, K'Sante, LeBlanc, Renata Glasc) render correct portraits

**Plans**: 1 plan

Plans:
- [ ] 05-01: Store image_url in Zustand state and update slot components to use it

---

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. ML Adapter | 3/3 | Complete    | 2026-02-24 |
| 2. FastAPI Backend | 3/3 | Complete    | 2026-02-24 |
| 3. React Draft Board | 5/5 | Complete | 2026-02-25 |
| 4. Integration and Deployment | 0/2 | Not started | - |
| 5. DDragon Image URL Fix | 0/1 | Not started | - |
