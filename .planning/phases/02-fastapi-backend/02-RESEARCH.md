# Phase 2: FastAPI Backend - Research

**Researched:** 2026-02-24
**Domain:** FastAPI REST API wrapping a Python ML adapter
**Confidence:** HIGH

## Summary

Phase 2 wraps the Phase 1 `LoLDraftAdapter` (a synchronous, CPU-bound Python class that loads joblib artifacts and returns win probabilities) in a FastAPI service with four endpoint groups: prediction, champions, teams, and health. The adapter is already built and tested (167 MB RSS, 27-feature VotingClassifier, singleton pattern). The API layer adds Pydantic request/response schemas, CORS, a lifespan-managed model lifecycle, a secured model-upload endpoint, and Data Dragon champion image URLs.

FastAPI 0.133.0 (latest as of 2026-02-24) on Python >= 3.10 is the locked choice. The adapter's `predict_from_draft` is blocking (scikit-learn inference + numpy), so the endpoint must either use a plain `def` (FastAPI auto-threads it) or wrap it in `asyncio.to_thread`. The DDragon version is pinned to `14.24.1` with a hardcoded mismatch dictionary.

**Primary recommendation:** Use FastAPI's lifespan context manager to load the adapter once at startup, store it on `app.state`, and expose it to route handlers via a thin `Depends()` function. Keep the API surface minimal -- four routers (`predict`, `champions`, `teams`, `health`) plus one admin router (`upload`).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Response contains only `blue_win_probability` and `red_win_probability` -- no metadata, no confidence scores, no feature breakdowns (those are v2: EPRED-02, EPRED-03)
- Do not echo the submitted draft back in the response
- Teams are required in the predict payload (not optional)
- Unknown team names accepted with fallback to average/neutral team stats (no error, silent fallback matching adapter behavior)
- Pin a specific DDragon version (hardcoded, e.g., 14.24.1) -- no runtime fetch of latest version
- Champion name mismatch mapping (Wukong=MonkeyKing, Nunu, etc.) lives as a hardcoded Python dictionary in the backend
- /api/champions returns only champions the model knows about (from training data), not the full DDragon roster
- Each champion entry includes the full DDragon CDN image URL (backend constructs it, frontend just uses it)
- Model upload endpoint secured with a bearer token read from ADMIN_TOKEN environment variable
- Hot-swap model in memory on upload -- requests during reload get the old model, zero downtime
- Upload accepts a full artifact bundle (model + scaler + encoders) to ensure consistency
- Validate the new model before swapping by running a test prediction with a dummy draft; only swap if it succeeds
- Strict draft validation: require exactly 10 picks (5 per team), 10 bans (5 per team), 2 teams, all roles assigned
- Use FastAPI's standard error format: `{"detail": "message"}` with appropriate HTTP status codes
- Validate champion names at the API layer against the known list before calling the adapter (fail fast with 422)
- Return 503 Service Unavailable if model is still loading when a predict request arrives; frontend retries based on /health status

### Claude's Discretion
- API route prefix structure and versioning
- Pydantic schema field naming conventions
- CORS allowed origins configuration
- Internal code organization (routers, dependencies, utils)
- asyncio.to_thread usage for blocking ML inference

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| API-01 | FastAPI REST endpoint accepts a complete draft and returns win probability for each team | Pydantic request schema maps to adapter's `DraftInput`; response is two floats. Blocking inference handled via `def` endpoint or `asyncio.to_thread`. |
| API-02 | ML model loads once at startup via FastAPI lifespan context manager, stored in app.state | FastAPI lifespan pattern verified in official docs -- `@asynccontextmanager` with `yield`, adapter stored on `app.state.adapter`. |
| API-03 | Champion list endpoint returns all valid champions with metadata (name, image URL from DDragon) | Adapter exposes `valid_champions` set. DDragon CDN URL pattern: `https://ddragon.leagueoflegends.com/cdn/14.24.1/img/champion/{id}.png`. Mismatch dict maps display names to DDragon IDs. |
| API-04 | Team list endpoint returns professional teams grouped by league | Existing `teams_db` dict in `interactive_match_predictor.py` provides league groupings for LCK/LEC/LCS/LPL. Adapter's `valid_teams` set provides canonical names. |
| API-05 | CORS configured to allow frontend requests | `CORSMiddleware` from FastAPI/Starlette. Dev: `localhost:5173` (Vite default). Prod: same origin (SPA served by FastAPI). |
| API-06 | Health/readiness endpoint returns model status and version info | Adapter's `get_status()` returns `AdapterStatus` dataclass with all needed fields. Must respond even during loading (return `{"status": "loading"}` before adapter init completes). |
| API-07 | Model file upload endpoint allows swapping in newly trained model files (security-gated) | Bearer token from `ADMIN_TOKEN` env var. `UploadFile` for multipart upload. Validate new artifacts with test prediction before swapping. |
| API-08 | DDragon champion name mapping handles mismatches | Hardcoded dict mapping canonical names to DDragon IDs. Key mismatches: Wukong->MonkeyKing, Nunu & Willump->Nunu, Bel'Veth->Belveth, Kai'Sa->KaiSa, etc. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| fastapi | 0.133.0 | Web framework | Official recommendation, async-first, Pydantic-native, OpenAPI auto-docs |
| uvicorn | 0.34.x | ASGI server | FastAPI's recommended server; included in `fastapi[standard]` |
| pydantic | 2.x | Request/response validation | Bundled with FastAPI; v2 is the current standard |
| python-multipart | 0.0.x | File upload parsing | Required by FastAPI for `UploadFile` support |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| starlette | (via fastapi) | CORS middleware, Request objects | Always -- FastAPI wraps Starlette |
| joblib | >=1.1.0 | Model artifact loading | Already in requirements.txt |
| scikit-learn | ==1.5.0 | ML model runtime | MUST pin to 1.5.0 -- model fails on newer versions |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| FastAPI | Flask | Flask lacks async, Pydantic integration, auto-OpenAPI. No reason to switch. |
| uvicorn | gunicorn+uvicorn | Render free tier is single-process; gunicorn adds complexity with no benefit. |

**Installation:**
```bash
pip install "fastapi[standard]" python-multipart
```

Note: `fastapi[standard]` installs uvicorn, python-multipart, and other common dependencies. scikit-learn==1.5.0 is already required by Phase 1.

## Architecture Patterns

### Recommended Project Structure
```
api/
    __init__.py
    main.py              # FastAPI app creation, lifespan, CORS, router inclusion
    dependencies.py      # get_adapter(), require_admin_token() dependencies
    schemas.py           # Pydantic request/response models (NOT adapter dataclasses)
    routers/
        __init__.py
        predict.py       # POST /api/predict
        champions.py     # GET /api/champions
        teams.py         # GET /api/teams
        health.py        # GET /health
        admin.py         # POST /api/admin/upload-model
    champion_mapping.py  # DDragon ID mapping dict and URL builder
    team_data.py         # Team-to-league grouping dict
```

This lives alongside the existing `src/` directory. The `api/` package imports from `src.adapter`.

### Pattern 1: Lifespan Model Loading
**What:** Load the adapter singleton once at startup, store on `app.state`.
**When to use:** Always -- this is the only correct pattern for heavy ML models.
**Example:**
```python
# Source: https://fastapi.tiangolo.com/advanced/events/
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load adapter (blocking, ~2s)
    app.state.adapter = LoLDraftAdapter()
    app.state.model_ready = True
    yield
    # Shutdown: nothing to clean up (GC handles it)

app = FastAPI(lifespan=lifespan)
```

### Pattern 2: Dependency Injection for Adapter Access
**What:** Thin `Depends()` function that reads `request.app.state.adapter`.
**When to use:** Every route that needs the adapter.
**Example:**
```python
from fastapi import Depends, Request, HTTPException

def get_adapter(request: Request) -> LoLDraftAdapter:
    if not getattr(request.app.state, "model_ready", False):
        raise HTTPException(status_code=503, detail="Model is loading")
    return request.app.state.adapter
```

### Pattern 3: Blocking Inference in def Endpoints
**What:** Declare the predict endpoint as `def` (not `async def`) so FastAPI auto-runs it in a thread pool.
**When to use:** When the endpoint body is entirely CPU-bound/blocking (scikit-learn inference).
**Example:**
```python
@router.post("/predict")
def predict(payload: DraftRequest, adapter = Depends(get_adapter)):
    # FastAPI runs this in a threadpool automatically
    result = adapter.predict_from_draft(draft_dict)
    return {"blue_win_probability": result.blue_win_prob, ...}
```
Alternative: use `async def` with `await asyncio.to_thread(adapter.predict_from_draft, draft_dict)`. Both are valid; plain `def` is simpler and the FastAPI-recommended approach for blocking code.

### Pattern 4: Bearer Token Admin Security
**What:** Dependency that checks `Authorization: Bearer <token>` against env var.
**When to use:** Admin-only endpoints (model upload).
**Example:**
```python
import os
from fastapi import Depends, HTTPException, Header

def require_admin_token(authorization: str = Header(...)):
    expected = os.environ.get("ADMIN_TOKEN")
    if not expected or authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid admin token")
```

### Pattern 5: Hot-Swap Model Upload
**What:** Accept uploaded artifacts, validate with test prediction, swap atomically.
**When to use:** POST /api/admin/upload-model.
**Example:**
```python
@router.post("/upload-model")
async def upload_model(
    model_file: UploadFile,
    scaler_file: UploadFile,
    encoders_file: UploadFile,
    _: None = Depends(require_admin_token),
):
    # 1. Save to temp directory
    # 2. Create new LoLDraftAdapter(artifacts_dir=temp_dir)
    # 3. Run test prediction with dummy draft
    # 4. If success: overwrite production files, reset singleton
    # 5. If failure: return 422 with error details
```

### Anti-Patterns to Avoid
- **Loading model per request:** Never call `LoLDraftAdapter()` inside a route handler -- it is a singleton but the first call is expensive (~2s). Always load in lifespan.
- **Using `async def` for blocking code:** Calling `adapter.predict_from_draft()` inside `async def` without `asyncio.to_thread` blocks the event loop. Use plain `def` or wrap explicitly.
- **Storing model in module-level global:** Fragile, hard to test, no startup guarantee. Use `app.state`.
- **Returning adapter dataclasses directly:** FastAPI cannot serialize stdlib `@dataclass` automatically with Pydantic v2. Convert to Pydantic models or dicts.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Request validation | Manual dict checking | Pydantic `BaseModel` | Automatic 422 errors with field details |
| CORS headers | Manual header injection | `CORSMiddleware` | Handles preflight, credential rules, origin matching |
| OpenAPI docs | Manual Swagger setup | FastAPI auto-generation | Free `/docs` and `/redoc` from type annotations |
| File upload parsing | Manual multipart parsing | `UploadFile` + `python-multipart` | Handles spooling, large files, async read |
| Thread pool for sync code | Manual `concurrent.futures` | FastAPI's auto-threading of `def` endpoints | Built-in, zero config |
| Error responses | Custom error format | `HTTPException(status_code=N, detail="msg")` | Standard `{"detail": "msg"}` format |

**Key insight:** FastAPI gives you validation, serialization, documentation, and async handling for free if you declare types correctly. The entire API layer should be mostly type declarations and thin glue code calling the adapter.

## Common Pitfalls

### Pitfall 1: scikit-learn Version Mismatch
**What goes wrong:** Model fails to load with `ModuleNotFoundError` on sklearn internals.
**Why it happens:** The production model was trained with scikit-learn 1.5.0. Versions >= 1.6 removed Cython modules the VotingClassifier depends on.
**How to avoid:** Pin `scikit-learn==1.5.0` in requirements.txt. Add a startup check that verifies the version.
**Warning signs:** `ModuleNotFoundError: No module named 'sklearn.tree._classes'` at startup.

### Pitfall 2: Adapter Singleton Reset on Hot-Swap
**What goes wrong:** After uploading new model files, the singleton still serves the old model.
**Why it happens:** `LoLDraftAdapter` uses `__new__` singleton pattern with `_initialized` guard. Overwriting files on disk does not invalidate the in-memory singleton.
**How to avoid:** After saving new artifacts, explicitly reset the singleton: `LoLDraftAdapter._instance = None; LoLDraftAdapter._instance = LoLDraftAdapter()`. Then update `app.state.adapter`.
**Warning signs:** Upload succeeds but predictions don't change.

### Pitfall 3: DDragon ID Mismatch Breaks Champion Images
**What goes wrong:** Frontend shows broken images for champions like Wukong, Kai'Sa.
**Why it happens:** DDragon uses internal IDs (MonkeyKing, KaiSa) that differ from display names.
**How to avoid:** Maintain a hardcoded mapping dict. The backend constructs the full URL: `https://ddragon.leagueoflegends.com/cdn/14.24.1/img/champion/{ddragon_id}.png`.
**Warning signs:** 404 errors on champion image URLs in browser network tab.

### Pitfall 4: 503 vs Startup Race Condition
**What goes wrong:** First requests after cold start get Python tracebacks instead of clean 503.
**Why it happens:** If the adapter loads in the lifespan `yield` block, no request can arrive before it completes (lifespan blocks startup). But if loading is deferred or fails silently, `app.state.adapter` may be `None`.
**How to avoid:** Use a `model_ready` flag on `app.state`. Set it to `False` before lifespan, `True` after adapter loads successfully. The `get_adapter` dependency checks this flag and returns 503 if false. The `/health` endpoint returns status regardless of the flag.
**Warning signs:** Unhandled `AttributeError: 'State' object has no attribute 'adapter'`.

### Pitfall 5: Team Names Not in Training Data
**What goes wrong:** User sends a team name the model was not trained on. Adapter raises `ValueError`.
**Why it happens:** The adapter's `validate_draft` calls `normalize_team_name` which raises on unknown teams.
**How to avoid:** Per CONTEXT.md decision, unknown teams should be accepted with silent fallback. The API layer must catch `ValueError` from team normalization and either retry with a default team or handle at the adapter level. This may require a small adapter modification or a try/except in the predict route.
**Warning signs:** 500 errors when users type team names not in the training dataset.

### Pitfall 6: CORS Credentials + Wildcard Origins
**What goes wrong:** Browser blocks requests with `credentials: 'include'`.
**Why it happens:** When `allow_credentials=True`, CORS spec forbids wildcard origins.
**How to avoid:** For dev, explicitly list `http://localhost:5173`. For prod, the SPA is served from the same origin so CORS is not needed. Use `allow_credentials=False` with `allow_origins=["*"]` as the simple default, or list specific origins.
**Warning signs:** Browser console shows CORS preflight failures.

## Code Examples

### Complete Lifespan with Model Loading
```python
# Source: https://fastapi.tiangolo.com/advanced/events/
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.adapter import LoLDraftAdapter

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_ready = False
    try:
        app.state.adapter = LoLDraftAdapter()
        app.state.model_ready = True
    except Exception as e:
        app.state.adapter = None
        app.state.model_load_error = str(e)
    yield

app = FastAPI(title="LoL Draft Predictor", lifespan=lifespan)
```

### Pydantic Request Schema for Prediction
```python
from pydantic import BaseModel, Field

class TeamDraft(BaseModel):
    top: str
    jungle: str
    mid: str
    bot: str
    support: str

class PredictRequest(BaseModel):
    blue_team: str
    red_team: str
    blue_picks: TeamDraft
    red_picks: TeamDraft
    blue_bans: list[str] = Field(min_length=5, max_length=5)
    red_bans: list[str] = Field(min_length=5, max_length=5)
    patch: str | None = None

class PredictResponse(BaseModel):
    blue_win_probability: float
    red_win_probability: float
```

### Champion Endpoint with DDragon URLs
```python
DDRAGON_VERSION = "14.24.1"
DDRAGON_BASE = f"https://ddragon.leagueoflegends.com/cdn/{DDRAGON_VERSION}/img/champion"

# Maps canonical display name -> DDragon asset ID
DDRAGON_ID_MAP = {
    "Wukong": "MonkeyKing",
    "Nunu & Willump": "Nunu",
    "Kai'Sa": "KaiSa",
    "Kha'Zix": "Khazix",
    "Cho'Gath": "Chogath",
    "Vel'Koz": "Velkoz",
    "Rek'Sai": "RekSai",
    "Kog'Maw": "KogMaw",
    "Bel'Veth": "Belveth",
    "K'Sante": "KSante",
    "LeBlanc": "Leblanc",
    "Renata Glasc": "Renata",
    # Most champions: remove spaces, keep casing
}

def get_ddragon_url(champion_name: str) -> str:
    ddragon_id = DDRAGON_ID_MAP.get(champion_name, champion_name.replace(" ", "").replace("'", "").replace(".", ""))
    return f"{DDRAGON_BASE}/{ddragon_id}.png"
```

### Health Endpoint (Always Responds)
```python
@router.get("/health")
async def health(request: Request):
    if getattr(request.app.state, "model_ready", False):
        status = request.app.state.adapter.get_status()
        return {
            "status": "ready",
            "model_name": status.model_name,
            "model_version": status.model_version,
            "champion_count": status.champion_count,
            "team_count": status.team_count,
            "memory_usage_mb": round(status.memory_usage_mb, 1),
        }
    return {"status": "loading"}
```

### CORS Configuration
```python
# Source: https://fastapi.tiangolo.com/tutorial/cors/
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # Alt dev server
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `@app.on_event("startup")` | `lifespan` async context manager | FastAPI 0.93+ (2023) | Old events deprecated; lifespan is the only recommended path |
| Pydantic v1 `BaseModel` | Pydantic v2 `BaseModel` | Pydantic 2.0 (2023) | v2 is 5-50x faster; FastAPI 0.100+ requires v2 |
| `Optional[str]` | `str \| None` | Python 3.10+ | FastAPI requires Python 3.10+ |

**Deprecated/outdated:**
- `@app.on_event("startup")` / `@app.on_event("shutdown")`: Replaced by lifespan. Still works but officially deprecated.
- Pydantic v1 `schema()` / `Config` class: Replaced by `model_json_schema()` / `model_config` in v2.

## Open Questions

1. **Team-to-League Mapping Completeness**
   - What we know: `interactive_match_predictor.py` has a hardcoded dict with ~10 teams per league (40 total). The adapter's `valid_teams` set (from `team_historical_performance.joblib`) likely has many more teams (historical rosters, Worlds/MSI teams, etc.).
   - What's unclear: How many teams in the training data lack a league assignment? Are there teams from leagues outside LCK/LEC/LCS/LPL (e.g., CBLOL, PCS)?
   - Recommendation: For the `/api/teams` endpoint, use the hardcoded mapping from the predictor for the 4 target leagues, and put remaining teams under an "Other" group. This can be refined later.

2. **Unknown Team Fallback in Adapter**
   - What we know: The CONTEXT.md decision says "unknown team names accepted with fallback to average/neutral team stats (no error, silent fallback)." But the adapter currently raises `ValueError` on unknown teams via `normalize_team_name`.
   - What's unclear: Whether the fallback should be implemented at the API layer (catch error, substitute a default) or the adapter layer (modify normalization).
   - Recommendation: Handle at the API layer with a try/except around team normalization. If team validation fails, catch the error and substitute a neutral/average team name. This avoids modifying the tested Phase 1 adapter.

3. **DDragon ID Mapping Completeness**
   - What we know: The main mismatches are documented (Wukong, Nunu, apostrophe names). Most champions have DDragon IDs that equal their name with spaces/punctuation removed.
   - What's unclear: The full list of all 160+ champions that exist in the training data and whether the generic rule (remove spaces/apostrophes) covers all of them.
   - Recommendation: Build the mapping by fetching `https://ddragon.leagueoflegends.com/cdn/14.24.1/data/en_US/champion.json` once during development (not at runtime). Cross-reference DDragon `name` fields against `adapter.valid_champions` to build the complete map. Hardcode the result.

## Sources

### Primary (HIGH confidence)
- [FastAPI Official Docs - Lifespan Events](https://fastapi.tiangolo.com/advanced/events/) - Verified lifespan pattern, `yield`-based context manager
- [FastAPI Official Docs - CORS](https://fastapi.tiangolo.com/tutorial/cors/) - CORSMiddleware configuration, credential constraints
- [FastAPI Official Docs - File Uploads](https://fastapi.tiangolo.com/tutorial/request-files/) - UploadFile pattern, python-multipart requirement
- [FastAPI Official Docs - Bigger Applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/) - APIRouter pattern, prefix/tags, project structure
- [FastAPI Official Docs - Dependencies](https://fastapi.tiangolo.com/tutorial/dependencies/) - Depends(), Request access, app.state pattern
- [FastAPI Official Docs - Async](https://fastapi.tiangolo.com/async/) - def vs async def behavior, auto-threading
- [PyPI FastAPI](https://pypi.org/project/fastapi/) - Version 0.133.0, Python >= 3.10

### Secondary (MEDIUM confidence)
- [DDragon CDN](https://ddragon.leagueoflegends.com/cdn/14.24.1/data/en_US/champion.json) - Champion data structure, ID vs name mismatches verified for Wukong/MonkeyKing, Bel'Veth/Belveth
- Phase 1 adapter source code (`src/adapter/`) - Interface verified by reading all module files

### Tertiary (LOW confidence)
- DDragon ID mapping completeness: Only spot-checked a few champions. Full mapping needs to be built during implementation by cross-referencing DDragon JSON against the training data champion set.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - FastAPI is well-documented, version verified on PyPI, all patterns verified in official docs
- Architecture: HIGH - Router/dependency patterns are FastAPI's documented best practices; adapter interface is fully known from Phase 1 code
- Pitfalls: HIGH - scikit-learn version issue is proven (Phase 1 verification); singleton reset is a known pattern; DDragon mismatches verified against CDN

**Research date:** 2026-02-24
**Valid until:** 2026-03-24 (FastAPI is stable; DDragon version is pinned)
