# Architecture Patterns

**Domain:** Python ML pipeline wrapped in web API + themed frontend
**Researched:** 2026-02-24
**Question answered:** How should a web app that wraps an existing Python ML pipeline be structured?

---

## Recommended Architecture

A single-process Python backend serving both the REST API and the compiled frontend as static files. The ML pipeline lives entirely in-process — no sidecar, no microservice, no model server. The frontend is a compiled SPA (static HTML/JS/CSS) mounted onto FastAPI via `StaticFiles`.

```
Browser
  |
  | HTTP (same origin — no CORS)
  v
FastAPI (single Render/Railway service)
  |
  |-- /api/*         REST endpoints (prediction, champions, teams, model upload)
  |-- /              Serves compiled frontend SPA (index.html fallback for SPA routing)
  |
  +-- ML Layer (in-process, loaded once at startup via lifespan)
        |-- InteractiveLoLPredictor (loads .joblib artifacts)
        |-- AdvancedFeatureEngineering (pre-computed champion meta, team history)
        |-- EdgeCaseHandler (confidence penalties)
        |-- ConfidenceEstimator (calibrated win probability)
```

**Why single process, not split deployment:**
- Free tier on Render gives 512 MB RAM and 0.1 CPU. Two services means two cold starts and two 512 MB budgets fighting each other.
- Serving the frontend from FastAPI eliminates all CORS configuration.
- The ML model must stay in Python anyway (scikit-learn, pandas, joblib). A second service would duplicate the cold start cost with no benefit.
- FastAPI's `StaticFiles` mount is zero-overhead — it delegates directly to Starlette's file serving.

**Single-deployment trade-off:** If the frontend ever needs server-side rendering (SSR) or edge deployment, this pattern would need to split. For a public prediction tool with no auth and no SSR requirements, it is the right call.

---

## Component Boundaries

| Component | Responsibility | Technology | Communicates With |
|-----------|---------------|------------|-------------------|
| FastAPI application | HTTP routing, request validation, response serialization | FastAPI + Uvicorn | ML Core (in-process), Frontend (file serving) |
| ML Core (Predictor) | Load model artifacts, run feature engineering, return win probability | `InteractiveLoLPredictor`, `AdvancedFeatureEngineering` | FastAPI (in-process function call) |
| Model Registry | Hold current in-memory model; accept file upload to swap artifacts | `app.state.model` singleton via lifespan | FastAPI upload endpoint |
| Pydantic Schemas | Validate and document all API request/response shapes | Pydantic v2 | FastAPI endpoints |
| Frontend SPA | Draft board UI, champion search, pick/ban sequence, result display | Svelte or plain HTML/JS/CSS | FastAPI `/api/*` via fetch() |
| Champion Assets | Champion portraits, champion list | Riot Data Dragon CDN (external) | Frontend (direct CDN fetch, never proxied through backend) |
| Data Dragon Cache | Cache champion list JSON in frontend sessionStorage | Browser storage | Frontend only |

### ML Core boundary detail

The ML Core is not a separate process. It is an object instantiated once at FastAPI startup and stored in `app.state`. Every API request calls methods on that object synchronously in a thread pool (FastAPI's `run_in_threadpool` or `asyncio.to_thread`) because `joblib`/`sklearn` inference is CPU-bound synchronous code.

```python
# Startup: load once
@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor = InteractiveLoLPredictor()  # loads .joblib files
    predictor.setup_feature_engineering()
    app.state.predictor = predictor
    yield
    # shutdown: nothing to release for joblib

app = FastAPI(lifespan=lifespan)

# Per-request: call in thread pool to avoid blocking event loop
@app.post("/api/predict")
async def predict(payload: DraftPayload):
    result = await asyncio.to_thread(
        app.state.predictor.predict_from_draft, payload
    )
    return result
```

**Why this matters for build order:** The API layer cannot be built until the predictor's public interface is stable. Define the interface (what a DraftPayload looks like, what the response contains) before building the API routes.

---

## Data Flow

### Prediction Request (primary flow)

```
User fills draft board in browser
  |
  | POST /api/predict
  | Body: { team1: {...}, team2: {...}, picks: [...], bans: [...] }
  v
FastAPI route
  |-- Pydantic validates the payload structure
  |-- Calls predictor.predict_from_draft(payload) in thread pool
        |
        +-- AdvancedFeatureEngineering.create_features_for_match(draft)
        |     Computes champion meta strength, team history, interaction features
        |
        +-- EdgeCaseHandler.check(draft)
        |     Returns confidence penalty if unknown champions/teams/patches
        |
        +-- scaler.transform(features)
        +-- model.predict_proba(features)
        +-- ConfidenceEstimator.calibrate(raw_probability, penalty)
        |
        Returns: { team1_win_prob, team2_win_prob, confidence, top_features }
  |
  v
FastAPI serializes response JSON
  |
  v
Browser receives { team1_win_prob: 0.63, team2_win_prob: 0.37, confidence: "HIGH" }
  |
  v
Draft board UI updates win probability display
```

### Champion Data Flow (frontend-only, no backend involved)

```
Browser startup
  |
  | GET https://ddragon.leagueoflegends.com/api/versions.json
  | GET https://ddragon.leagueoflegends.com/cdn/{latest}/data/en_US/champion.json
  v
Frontend caches champion list in sessionStorage
  |
  v
Champion search box queries in-memory champion list
  |
  v
Portrait image: <img src="https://ddragon.leagueoflegends.com/cdn/{ver}/img/champion/{name}.png">
```

**Key decision:** Champion assets come from Riot's CDN directly to the browser. The backend never proxies champion images. This avoids bandwidth costs on the free tier and keeps the backend stateless with respect to game assets.

### Model Upload Flow

```
User selects new .joblib files (best_model.joblib, scaler.joblib, encoders.joblib)
  |
  | POST /api/admin/upload-model (multipart form)
  v
FastAPI saves uploaded files to models/production/ on the server's filesystem
  |
  v
FastAPI calls predictor.reload_model(new_paths)
  |
  v
app.state.predictor is updated in-place (or replaced)
  |
  v
Response: { success: true, model_metrics: {...} }
```

**Limitation:** On Render/Railway free tier, the filesystem is ephemeral. Uploaded model files will be lost on the next deploy or restart. For the thesis use case (manual model swap after a Colab training run), this is acceptable — the user uploads after each Colab session. If persistence is needed later, model files would go to an S3 bucket or be committed to the repo.

---

## Patterns to Follow

### Pattern 1: Lifespan Context Manager for Model Loading

Load all `.joblib` artifacts once at startup. Never load models inside request handlers.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor = InteractiveLoLPredictor()
    predictor.setup_feature_engineering()
    app.state.predictor = predictor
    yield

app = FastAPI(lifespan=lifespan)
```

**Why:** On a free-tier service with 512 MB RAM, loading a joblib model on every request would exhaust memory within seconds under any real load. Loading once at startup and reusing the singleton is the only viable pattern.

### Pattern 2: Thread Pool for Synchronous ML Inference

Wrap all synchronous sklearn/pandas calls in `asyncio.to_thread`.

```python
result = await asyncio.to_thread(app.state.predictor.predict, payload)
```

**Why:** FastAPI's event loop is async. Blocking it with CPU-bound sklearn inference will freeze all concurrent requests. `asyncio.to_thread` moves the work to a thread pool, keeping the event loop responsive.

### Pattern 3: Pydantic Request/Response Schemas as Contract

Define all input/output shapes as Pydantic v2 models before implementing routes.

```python
class ChampionPick(BaseModel):
    champion: str
    role: Literal["top", "jungle", "mid", "bot", "support"]

class DraftPayload(BaseModel):
    team1_name: str
    team2_name: str
    team1_picks: list[ChampionPick]
    team2_picks: list[ChampionPick]
    team1_bans: list[str]
    team2_bans: list[str]
    patch: str | None = None
    is_playoffs: bool = False

class PredictionResponse(BaseModel):
    team1_win_probability: float
    team2_win_probability: float
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    top_features: list[dict]
```

**Why:** The Pydantic schema IS the API contract. FastAPI auto-generates OpenAPI docs from it. The frontend developer knows exactly what to send and what to expect. This decouples frontend build from backend implementation.

### Pattern 4: Mount Compiled Frontend on Root

Build the frontend SPA to static files, copy the dist/ output into the FastAPI project, and mount on `/`.

```python
from fastapi.staticfiles import StaticFiles

# API routes first (more specific)
app.include_router(api_router, prefix="/api")

# Frontend catch-all last (less specific)
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
```

**Why:** Single deployment on free tier. The `html=True` parameter makes FastAPI serve `index.html` for unmatched routes, enabling SPA client-side routing.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Loading the Model Inside a Request Handler

**What:** Calling `joblib.load("best_model.joblib")` inside a route function.

**Why bad:** Each request re-reads and deserializes the model from disk. On a free tier with slow I/O and 512 MB RAM, this will cause timeouts and OOM errors under any load. Warm-up time is ~2-4 seconds per request.

**Instead:** Load once in the lifespan context manager, store in `app.state`.

### Anti-Pattern 2: Proxying Champion Images Through the Backend

**What:** Fetching champion portraits from Data Dragon in Python and re-serving them to the browser.

**Why bad:** Image proxying would consume the free tier's bandwidth cap and memory. Riot's CDN already handles caching and global distribution.

**Instead:** Champion `<img>` tags point directly to `ddragon.leagueoflegends.com`. The backend only serves prediction API responses.

### Anti-Pattern 3: Storing Session State in the Backend

**What:** Keeping partial draft state (picks and bans as they happen) in a server-side session or database.

**Why bad:** The free tier is stateless and ephemeral. Multiple requests may hit after a cold start with a fresh process. Session state would be lost. Also requires a session store (Redis) which is not in scope.

**Instead:** All draft state lives in the frontend (browser memory / React/Svelte component state). The backend receives the complete draft in a single POST request and returns the prediction.

### Anti-Pattern 4: Blocking the Event Loop with Pandas Operations

**What:** Running `AdvancedFeatureEngineering.create_advanced_features()` (which uses pandas) directly in an async route without `asyncio.to_thread`.

**Why bad:** Pandas operations are synchronous and CPU-bound. Running them in the event loop blocks all other requests for the duration.

**Instead:** Always wrap with `await asyncio.to_thread(...)`.

### Anti-Pattern 5: Two Separate Deployments on Free Tier

**What:** Deploying the FastAPI backend on one Render service and the frontend on Vercel/Netlify as a separate service.

**Why bad:** Two cold starts. CORS configuration required and fragile. The Render service spins down after 15 minutes of inactivity — the first user after a sleep period waits 30 seconds for the backend, then another request cycle for the frontend to resolve the CORS preflight. Doubles operational complexity.

**Instead:** Serve compiled frontend from FastAPI. One deployment, one domain, no CORS.

---

## Suggested Build Order

Dependencies determine build order. Each component can only be built after its dependencies are stable.

```
1. ML Adapter Layer (highest dependency — everything else depends on it)
   Define: predict_from_draft(DraftPayload) -> PredictionResult interface
   Wrap: InteractiveLoLPredictor methods into clean, testable functions
   Validate: Can call predictor synchronously with a known draft, get a probability

2. Pydantic Schemas (no dependencies — pure data shapes)
   Define: DraftPayload, PredictionResponse, ChampionListResponse, TeamListResponse
   These schemas are the API contract used by routes AND frontend

3. FastAPI Application Shell (depends on: ML Adapter, Schemas)
   Wire: lifespan model loading
   Wire: /api/predict route using ML Adapter and Schemas
   Wire: /api/champions and /api/teams helper endpoints
   Wire: /api/admin/upload-model for model swap
   Validate: curl /api/predict returns a valid probability

4. Frontend SPA (depends on: Schemas — can build in parallel with FastAPI)
   Build: Draft board component (pick/ban slots, sequence logic)
   Build: Champion search with Data Dragon CDN integration
   Build: Team selector for LCK/LEC/LCS/LPL teams
   Build: Win probability display
   Compile to static files

5. Integration — Mount Frontend on FastAPI (depends on: steps 3 and 4)
   Copy frontend/dist into FastAPI project
   Mount StaticFiles on /
   Test full round-trip in local environment

6. Deployment (depends on: step 5)
   Configure Render or Railway service
   Set environment variables (MODEL_PATH, etc.)
   Validate cold start behavior (~30s acceptable per PROJECT.md)
```

**Parallelism opportunity:** Steps 2, 3, and 4 can proceed in parallel once step 1 (the ML adapter interface) is defined. The frontend only needs the Pydantic schema definitions to know what JSON to send and expect — not a running backend.

---

## Scalability Considerations

| Concern | Free Tier (now) | Paid Tier (if needed) |
|---------|----------------|----------------------|
| Memory | 512 MB — model must stay under ~100 MB serialized | Increase RAM, keep architecture same |
| Cold start | ~30s spin-down after 15 min idle; use UptimeRobot ping to keep alive | Always-on service eliminates cold start |
| Concurrency | Uvicorn handles concurrent requests; ML inference in thread pool; adequate for thesis demo traffic | Add Gunicorn with multiple Uvicorn workers |
| Model updates | Manual upload via /api/admin/upload-model; ephemeral filesystem means re-upload after restarts | Mount persistent disk or use S3 for artifact storage |
| Frontend CDN | No CDN on free tier; FastAPI serves static files directly | Add Cloudflare in front of the service (free) |

---

## Key Architecture Decisions

| Decision | Chosen | Alternative Considered | Why |
|----------|--------|----------------------|-----|
| Deployment topology | Single FastAPI service (API + static files) | Separate API + CDN frontend | Free tier constraints; simpler ops |
| ML inference model | In-process singleton via lifespan | External model server (TorchServe, Seldon) | No infrastructure budget; scikit-learn is lightweight |
| Frontend framework | Svelte (compiled, zero-runtime) or plain HTML/JS | React, Vue | Smaller bundle, no Node runtime needed in production; SvelteKit also works |
| Champion assets | Riot Data Dragon CDN direct from browser | Bundle assets or proxy through backend | Free, always current, no backend bandwidth cost |
| Draft state location | Frontend (client-side) | Server-side session | Stateless backend; free tier is ephemeral |
| Model swap mechanism | File upload to /api/admin/upload-model | Git commit + redeploy | Faster iteration post-Colab training |

---

## Sources

- FastAPI lifespan documentation: [https://fastapi.tiangolo.com/advanced/events/](https://fastapi.tiangolo.com/advanced/events/) — HIGH confidence
- FastAPI StaticFiles for SPA: [https://github.com/fastapi/fastapi/discussions/5443](https://github.com/fastapi/fastapi/discussions/5443) — HIGH confidence
- FastAPI production patterns 2025: [https://orchestrator.dev/blog/2025-1-30-fastapi-production-patterns/](https://orchestrator.dev/blog/2025-1-30-fastapi-production-patterns/) — MEDIUM confidence
- Render free tier limits (512 MB RAM, 15 min spin-down): [https://www.freetiers.com/directory/render](https://www.freetiers.com/directory/render) — MEDIUM confidence
- Data Dragon CDN URL structure: [https://riot-api-libraries.readthedocs.io/en/latest/ddragon.html](https://riot-api-libraries.readthedocs.io/en/latest/ddragon.html) — MEDIUM confidence (verified by multiple community sources)
- SvelteKit + FastAPI architecture 2025: [https://johal.in/next-gen-frontend-architectures-sveltekit-with-python-fastapi-backend-for-interactive-uIs-2025/](https://johal.in/next-gen-frontend-architectures-sveltekit-with-python-fastapi-backend-for-interactive-uIs-2025/) — LOW confidence (verify SvelteKit adapter docs before committing)
- FastAPI ML model loading patterns: [https://apxml.com/courses/fastapi-ml-deployment/chapter-3-integrating-ml-models/loading-models-fastapi](https://apxml.com/courses/fastapi-ml-deployment/chapter-3-integrating-ml-models/loading-models-fastapi) — MEDIUM confidence

---

*Architecture research: 2026-02-24*
