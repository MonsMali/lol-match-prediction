# Milestones

## v1.0 LoL Draft Predictor Web App (Shipped: 2026-02-26)

**Phases completed:** 5 phases, 14 plans, 0 tasks

**Key accomplishments:**
- Built lightweight ML adapter (167 MB RSS) that runs predictions from joblib artifacts without loading the 37K-row CSV
- Created FastAPI REST API with predict, champions, teams, health, and admin upload endpoints
- Built React draft board with LoL-themed UI and champion portraits from Riot Data Dragon CDN
- Implemented both step-by-step (20-step pro order) and bulk entry draft modes with role assignment
- Added best-of-series tracker (BO3/BO5) with score tracking across games
- Deployed as single Render service with cold-start warm-up screen and keep-alive strategy

**Stats:** 3,451 LOC (1,728 Python + 1,723 TypeScript), 44 files, 3 days

**Tech debt accepted:**
- render.yaml env var MODEL_UPLOAD_TOKEN does not match code's ADMIN_TOKEN (admin upload locked on Render)
- Dead isPredicting state in DraftBoard.tsx (WinProbability spinner never fires)
- 13 items require human browser/deployment verification

---

