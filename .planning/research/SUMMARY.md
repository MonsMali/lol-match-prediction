# Project Research Summary

**Project:** LoL Draft Predictor Web App
**Domain:** Python ML pipeline wrapped in a React SPA + FastAPI REST API, deployed on a free cloud hosting tier
**Researched:** 2026-02-24
**Confidence:** HIGH

## Executive Summary

This project wraps an existing, trained Logistic Regression model (82.97% AUC-ROC, 37,502 professional matches) in a public-facing web application. The recommended pattern is a single FastAPI service that serves both the prediction REST API and the compiled React SPA as static files — one deployment, no CORS, no split cold starts. The ML model loads once at startup via FastAPI's lifespan context manager and is called in a thread pool for each prediction request. The frontend interacts directly with Riot's Data Dragon CDN for champion assets and sends a complete draft payload to the backend for prediction. This is a well-understood architecture (Python ML API + SPA) with no novel technical risk at the architecture level.

The critical constraint is Render's free tier 512 MB RAM ceiling. The existing `AdvancedFeatureEngineering` class loads the full 37,502-row CSV into memory at startup, which was designed for training runs, not a constrained server. This must be refactored before deployment: pre-serialized lookup artifacts (champion meta strength, team history, champion synergies) already exist as `.joblib` files and should be used directly by the inference path, eliminating the need to load the raw dataset on the server. This refactor is the single highest-priority prerequisite for the project.

The MVP feature set is well-scoped: a bulk-entry draft board with champion portraits, professional team selection, role assignment, and a win-probability split bar. Step-by-step live draft mode, shareable URLs, and confidence indicators are the next layer of differentiators but should be deferred until the core flow is validated. The model upload feature carries a genuine security risk (joblib/pickle RCE) and must be protected with an authentication token or replaced with a Git-based model update workflow.

## Key Findings

### Recommended Stack

The web layer adds FastAPI (>=0.115.0) as the backend API framework with Uvicorn as the single-process ASGI server, React 19 + TypeScript + Vite as the frontend, Tailwind CSS v4 for styling, and Zustand for client-side draft state. The existing ML stack (scikit-learn, pandas, numpy, joblib) is unchanged. Deployment targets Render's permanent free tier (512 MB RAM, 0.1 CPU, 15-minute idle spin-down). Riot's Data Dragon CDN is used directly from the browser for champion portraits — no backend proxying.

FastAPI is the clear choice over Flask or Django for this use case: async ASGI, built-in Pydantic v2 validation, and automatic OpenAPI docs. React over HTMX because the draft board requires complex interactive client state (pick/ban sequencing, champion grid with search, slot management) that is poorly suited to the server-round-trip model. Zustand over Redux because the draft state is global but simple; Zustand adds ~1 KB with no boilerplate. TanStack Query v5 handles loading/error states for prediction API calls without `useEffect` boilerplate.

**Core technologies:**
- FastAPI + Uvicorn (single process): Python REST API + static file serving — async, validated, free-tier memory-safe
- React 19 + TypeScript + Vite: SPA frontend — component model maps directly to draft slots; Vite dist output served by FastAPI
- Tailwind CSS v4: Utility CSS — fast iteration for dark LoL-themed UI; Vite plugin eliminates config file
- Zustand 5: Client draft state — lightweight, selective re-renders, no provider boilerplate
- TanStack Query v5: API call lifecycle — loading states and error handling for prediction requests
- Render (free tier): Deployment — permanent free hosting; Railway is trial-only as of 2025
- Riot Data Dragon CDN: Champion assets — no API key, always-current, browser-direct

### Expected Features

**Must have (table stakes):**
- Champion grid with portraits and search — pool is 160+; text-only is unusable; Riot DDragon CDN covers this
- Correct professional draft order enforced (blue picks 1, red picks 2-3, etc.)
- 5 picks and 5 bans per team — hardcoded constraint matching pro format
- Visual two-sided draft board (blue/red split layout)
- Professional team selection from LCK, LEC, LCS, LPL
- Duplicate champion prevention (client + server-side)
- Win probability output with split bar visualization (single core value)
- Blue/red side labeling throughout
- Cold-start loading indicator ("warming up" message for ~20-30 second spin-up)
- Mobile-readable responsive layout (not pixel-perfect, but not broken)
- Human-readable error messaging on API failure

**Should have (competitive differentiators):**
- Thesis-backed 82.97% AUC claim prominently displayed — only tool in the landscape with peer-validated accuracy
- "Pre-match only, no in-game stats" framing — differentiates from Riot/AWS broadcast tools
- Step-by-step live draft mode (intermediate predictions after each pick)
- Shareable draft URL via query parameter encoding (no server storage required)
- Confidence indicator alongside probability (existing `confidence.py` module)
- Best-of-series toggle (BO1/BO3/BO5)
- Role assignment UI per pick slot

**Defer to v2+:**
- Model upload/swap endpoint (security risk; use Git-based workflow initially)
- Historical accuracy dashboard (requires storing predictions and ground truth)
- Per-champion pick recommendations (changes product scope; out of model design)
- Animated draft reveal sequence (high CSS effort, low functional value)
- User accounts and history (auth + database complexity not justified for v1)
- Fearless draft mode (cross-game state tracking out of scope)

### Architecture Approach

A single FastAPI process serves the prediction REST API (`/api/*`) and the compiled React SPA as static files mounted on `/`. The ML model (InteractiveLoLPredictor + AdvancedFeatureEngineering) is loaded once at startup via the lifespan context manager and stored in `app.state`. Per-request prediction calls run in `asyncio.to_thread()` to avoid blocking the async event loop with CPU-bound sklearn/pandas work. All draft state lives in the browser (Zustand store); the backend receives a complete draft payload in a single POST and returns a probability. Champion assets are fetched browser-to-CDN, never proxied through the backend.

**Major components:**
1. FastAPI Application Shell — HTTP routing, Pydantic v2 request/response validation, static file serving, lifespan model loading
2. ML Adapter Layer — wraps `InteractiveLoLPredictor` and `AdvancedFeatureEngineering` into a clean `predict_from_draft(DraftPayload) -> PredictionResult` interface callable from the API layer
3. Pydantic Schemas — `DraftPayload`, `PredictionResponse`, `ChampionListResponse`, `TeamListResponse`; these are the API contract shared by backend routes and frontend TypeScript types
4. React SPA — Draft board component, ChampionGrid with DDragon integration, TeamSelector, WinProbabilityDisplay; all state in Zustand store
5. Data Dragon Cache — Browser-side sessionStorage cache of champion list JSON; champion portraits fetched directly as `<img>` tags from CDN

### Critical Pitfalls

1. **OOM kill on startup from dataset load** — `AdvancedFeatureEngineering.load_and_analyze_data()` loads the full 37,502-row CSV at startup. On Render's 512 MB free tier this will cause the process to be OOM-killed before serving any requests. Prevention: create a lean inference path that loads only pre-serialized `.joblib` artifacts (champion meta strength, team history, synergies — already exist in `models/`); do NOT bundle or load the raw CSV on the web server.

2. **Joblib model upload is a remote code execution vector** — Loading user-uploaded `.joblib` files executes arbitrary Python (pickle). A public upload endpoint without authentication gives any visitor full shell access to the container. Prevention: remove the upload endpoint from v1 and use Git-based model updates; if kept, protect with a secret token header and never load from untrusted sources.

3. **Cold start timeouts cause first-user 504 errors** — The 15-minute idle spin-down causes ~30-90 second cold starts. Render's default request timeout is 30 seconds; the first request after idle will 504 before the server is ready. Prevention: add a `/health` endpoint that responds immediately (before model load), implement a frontend warm-up ping on page load, and use a free keep-alive cron (cron-job.org at 10-minute intervals) to prevent spin-down.

4. **scikit-learn version mismatch breaks model loading after Colab retraining** — Colab auto-installs latest scikit-learn; the server runs whatever is pinned in requirements.txt. Versions that cross a major release boundary (e.g., 1.2.x to 1.3.x) can break `.joblib` loading or produce silent incorrect results. Prevention: pin the exact scikit-learn version used in Colab; add a startup log check comparing loaded version to the expected version stored in model metadata.

5. **Data Dragon champion key mismatches break portrait URLs** — `Wukong` is keyed as `MonkeyKing`, `Nunu & Willump` as `Nunu`, etc. Using display names directly in portrait URLs produces 404s for specific champions. Prevention: on startup fetch `champion.json`, build an internal `id` map, use the `id` field (matches image filename exactly) for portrait URL construction; add image fallback for any 404.

## Implications for Roadmap

Based on research, the dependency chain is clear: the ML adapter interface must be defined first (everything else depends on it), then schemas, then backend routes and frontend can proceed in parallel, then integration, then deployment.

### Phase 1: ML Adapter and Inference Refactor

**Rationale:** This is the blocking dependency for everything else. The existing `InteractiveLoLPredictor` must be wrapped in a clean interface, and the dataset-loading code path must be removed from the inference path. Without this, the server will OOM-kill on startup and no deployment will succeed.

**Delivers:** A standalone `predict_from_draft(draft_input) -> PredictionResult` function that loads only pre-serialized artifacts (no CSV), is callable synchronously from Python, and returns win probabilities and confidence. Validated by calling it from a Python script against a known draft.

**Addresses:** Table-stakes prediction output; role assignment input; team selection input.

**Avoids:** Pitfall 1 (OOM on dataset load), Pitfall 6 (dataset-dependent feature engineering on server), Pitfall 5 (scikit-learn version pinning must be confirmed here).

**Research flag:** Does NOT need additional research. This is internal refactoring of known code with well-understood patterns.

### Phase 2: FastAPI Backend Shell

**Rationale:** Once the ML adapter interface is stable, the API layer can be built against it. Schemas must be defined before frontend work begins to establish the API contract.

**Delivers:** A running FastAPI application with `/api/predict`, `/api/champions`, `/api/teams`, and `/health` endpoints. Prediction endpoint calls the ML Adapter via `asyncio.to_thread`. Pydantic schemas defined for all inputs/outputs. CORS middleware configured. Model loads once via lifespan.

**Addresses:** Win probability output; team selection data; champion list data; loading/error states.

**Avoids:** Pitfall 3 (cold start — `/health` endpoint must be fast), Pitfall 4 (single Uvicorn worker only), Pitfall 9 (CORS configured from day one).

**Research flag:** Standard FastAPI patterns; no additional research needed.

### Phase 3: React Frontend — Draft Board

**Rationale:** Frontend can build against the Pydantic schema definitions (TypeScript types) in parallel with the backend, but integration requires the backend to be running. Frontend delivers the complete draft UX before connecting to the real API.

**Delivers:** ChampionGrid with DDragon portrait integration and search, two-sided draft board (blue/red), pick/ban slot components with correct pro draft order enforcement, team selector for LCK/LEC/LCS/LPL, role assignment per pick slot, win probability split bar display, loading indicator, error messaging, responsive layout.

**Addresses:** All table-stakes features. Mobile-readable layout.

**Avoids:** Pitfall 7 (Data Dragon key mismatches — build `id` map at startup; use `id` field for URLs), Pitfall 11 (fuzzy match ambiguity — always confirm champion portrait before committing to draft slot).

**Research flag:** Standard React patterns; no additional research needed. Data Dragon URL mapping is well-documented in PITFALLS.md.

### Phase 4: Integration and Deployment

**Rationale:** Connect frontend to live backend, build the Vite dist, mount on FastAPI, deploy to Render. The cold-start UX and keep-alive strategy must be implemented here.

**Delivers:** End-to-end draft prediction working in a browser against the real ML model. Deployed to Render. Frontend warm-up ping on page load. UptimeRobot/cron-job.org keep-alive configured. `requirements-prod.txt` trimmed of training-only dependencies (XGBoost, LightGBM, CatBoost, Optuna).

**Avoids:** Pitfall 3 (cold start UX), Pitfall 10 (build timeout from heavy dependencies).

**Research flag:** No additional research needed. Render deploy config and build commands are fully specified in STACK.md.

### Phase 5: Differentiators and Polish

**Rationale:** Once the core flow is validated end-to-end, add the features that set this tool apart from the competitive landscape.

**Delivers:** Step-by-step live draft mode (intermediate predictions per pick), shareable draft URL (query parameter encoding), confidence indicator, best-of-series toggle, thesis accuracy claim and methodology link prominently displayed, "pre-match only" framing.

**Addresses:** All "should have" differentiator features.

**Avoids:** Pitfall 8 (new champion handling — return neutral response for unknown champions; display clear message).

**Research flag:** Step-by-step mode needs verification that `InteractiveLoLPredictor` handles UNKNOWN tokens correctly for partial drafts (MEDIUM confidence per FEATURES.md). Validate before building the backend endpoint for this mode.

### Phase Ordering Rationale

- Phase 1 (ML Adapter) must come first — it is the dependency for Phase 2, and without it no deployment is possible.
- Phases 2 and 3 (Backend + Frontend) can proceed in parallel once Phase 1 defines the schema/interface contract.
- Phase 4 (Integration/Deployment) is a gate — nothing is public until this completes. Cold-start UX must be built here, not retrofitted.
- Phase 5 (Differentiators) is intentionally last — the step-by-step mode and confidence indicator require backend work that depends on validating the core prediction flow first.
- The model upload feature is explicitly excluded from this roadmap due to the joblib RCE risk. It can be added in a later phase with authentication if needed.

### Research Flags

Phases needing deeper research during planning:
- **Phase 5 (Step-by-step mode):** Verify that partial-draft inference with UNKNOWN champion tokens works correctly in `InteractiveLoLPredictor`. This is MEDIUM confidence — referenced in code inspection but not confirmed end-to-end. A quick test call with a partial draft before planning this phase is recommended.

Phases with standard, well-documented patterns (skip additional research):
- **Phase 1 (ML Adapter):** Internal Python refactoring; no external dependencies.
- **Phase 2 (FastAPI Backend):** FastAPI lifespan, thread pool, Pydantic schemas are fully documented in official FastAPI docs.
- **Phase 3 (React Frontend):** Standard React + Vite + Tailwind patterns; Data Dragon URL mapping is fully resolved in PITFALLS.md.
- **Phase 4 (Integration/Deployment):** Render config fully specified in STACK.md.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All technology choices backed by official documentation and production usage evidence as of 2025. Railway free tier removal is verified via official Railway pricing docs. |
| Features | HIGH | Table stakes derived from surveying 8+ competing tools (LoLDraftAI, DraftGap, Diff15, drafting.gg, draftlol, etc.). MVP scope is consistent with project constraints. |
| Architecture | HIGH | Single-service pattern with FastAPI StaticFiles and lifespan model loading is well-documented and matches Render free tier constraints. One LOW confidence source (SvelteKit architecture) is not in the recommendation. |
| Pitfalls | HIGH | OOM pitfall, joblib RCE, and multi-worker RAM multiplication all sourced from official documentation (Render community, joblib docs, FastAPI GitHub). scikit-learn version mismatch sourced from official scikit-learn issue tracker. |

**Overall confidence:** HIGH

### Gaps to Address

- **Partial draft inference:** Whether `InteractiveLoLPredictor` handles UNKNOWN tokens for partial drafts (needed for step-by-step mode) is MEDIUM confidence. Validate with a test call before planning Phase 5.

- **Memory profiling:** The exact RAM footprint of loading only the pre-serialized `.joblib` artifacts (without the full CSV) on the web server has not been measured. Profile with `memory_profiler` locally before first Render deployment. If loading the three artifact files (champion meta, team history, synergies) still exceeds ~400 MB, the upgrade to Render Starter ($7/month) or lazy loading on first request are the mitigation paths.

- **scikit-learn version:** The exact version used during the last training run is not confirmed in research. This must be checked against the deployed `requirements.txt` before deploying the web server.

- **Frontend framework divergence:** ARCHITECTURE.md mentions Svelte as an alternative to React in one section, while STACK.md recommends React throughout. The recommendation is React 19 — Svelte is not adopted.

## Sources

### Primary (HIGH confidence)
- [FastAPI official docs — lifespan, StaticFiles, server workers](https://fastapi.tiangolo.com/advanced/events/)
- [joblib official docs — pickle security warning](https://joblib.readthedocs.io/en/latest/persistence.html)
- [scikit-learn official docs — model persistence and version compatibility](https://scikit-learn.org/stable/model_persistence.html)
- [Render community — OOM kill reports on free tier](https://community.render.com/t/deployment-error-ran-out-of-memory-used-over-512mb/14215)
- [Railway official pricing docs — no permanent free tier](https://docs.railway.com/reference/pricing/plans)
- [Tailwind CSS v4 announcement (Jan 2025)](https://tailwindcss.com/blog/tailwindcss-v4)
- [Riot Data Dragon CDN documentation](https://riot-api-libraries.readthedocs.io/en/latest/ddragon.html)
- [AWS Security Blog — joblib pickle RCE](https://aws.amazon.com/blogs/security/enhancing-cloud-security-in-ai-ml-the-little-pickle-story/)
- [FastAPI GitHub — multi-worker ML memory issue #2425](https://github.com/fastapi/fastapi/issues/2425)
- [scikit-learn 1.3.0 backward compatibility break — GitHub Issue #26798](https://github.com/scikit-learn/scikit-learn/issues/26798)

### Secondary (MEDIUM confidence)
- [TestDriven.io — FastAPI + React integration guide](https://testdriven.io/blog/fastapi-react/)
- [FastAPI production patterns 2025](https://orchestrator.dev/blog/2025-1-30-fastapi-production-patterns/)
- [Medium — FastAPI cold start mitigation on Render free tier](https://medium.com/@saveriomazza/how-to-keep-your-fastapi-server-active-on-renders-free-tier-93767b70365c)
- [RCVolus lol-pick-ban-ui reference implementation (GitHub)](https://github.com/RCVolus/lol-pick-ban-ui)
- [DEV Community — React state management 2025 (Zustand)](https://dev.to/cristiansifuentes/react-state-management-in-2025-context-api-vs-zustand-385m)
- [Diff15 — closest competitor tool surveyed](https://diff15.com/)

### Tertiary (LOW confidence)
- [SvelteKit + FastAPI architecture article](https://johal.in/next-gen-frontend-architectures-sveltekit-with-python-fastapi-backend-for-interactive-uIs-2025/) — mentioned in ARCHITECTURE.md but not adopted; React is the recommendation

---
*Research completed: 2026-02-24*
*Ready for roadmap: yes*
