# Roadmap: LoL Draft Predictor Web App

## Milestones

- Shipped **v1.0 LoL Draft Predictor Web App** -- Phases 1-5 (shipped 2026-02-26)

## Phases

<details>
<summary>Shipped v1.0 (Phases 1-5) -- 2026-02-26</summary>

- [x] Phase 1: ML Adapter (3/3 plans) -- completed 2026-02-24
- [x] Phase 2: FastAPI Backend (3/3 plans) -- completed 2026-02-24
- [x] Phase 3: React Draft Board (5/5 plans) -- completed 2026-02-25
- [x] Phase 4: Integration and Deployment (2/2 plans) -- completed 2026-02-26
- [x] Phase 5: DDragon Image URL Fix (1/1 plan) -- completed 2026-02-26

See: `.planning/milestones/v1.0-ROADMAP.md` for full details.

</details>

## Future Ideas

### Auto-Refresh Colab Notebook
Create a single Colab notebook that, in one "Run All" click:
1. Downloads the latest Oracle's Elixir raw data for all years
2. Regenerates the processed dataset (with `game_length`)
3. Retrains all models with GPU acceleration
4. Saves the new `.joblib` files back to Google Drive

This keeps the model up to date with new patches and meta shifts without any infrastructure. Could optionally be scheduled with Colab Pro's scheduler to run monthly.

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. ML Adapter | v1.0 | 3/3 | Complete | 2026-02-24 |
| 2. FastAPI Backend | v1.0 | 3/3 | Complete | 2026-02-24 |
| 3. React Draft Board | v1.0 | 5/5 | Complete | 2026-02-25 |
| 4. Integration and Deployment | v1.0 | 2/2 | Complete | 2026-02-26 |
| 5. DDragon Image URL Fix | v1.0 | 1/1 | Complete | 2026-02-26 |
