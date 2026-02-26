# Phase 4: Integration and Deployment - Research

**Researched:** 2026-02-25
**Domain:** Render deployment, FastAPI static file serving, cold-start UX
**Confidence:** HIGH

## Summary

Phase 4 integrates the React SPA (Phase 3) with the FastAPI backend (Phase 2) into a single Render free-tier deployment. The frontend is pre-built via `npm run build` and served by FastAPI's `StaticFiles` mount. Render's Python native runtime includes Node.js and npm in the build environment, so a single `buildCommand` can install Python deps and build the frontend without Docker.

The main technical challenges are: (1) SPA catch-all routing so direct URL access serves `index.html` instead of 404, (2) trimming production dependencies to fit the free-tier memory budget, and (3) graceful cold-start UX since Render spins down free services after 15 minutes of inactivity.

**Primary recommendation:** Use `StaticFiles(directory="frontend/dist", html=True)` mounted at `/` after all API routes, with a catch-all `/{path:path}` fallback returning `index.html` for SPA client-side routing. Build the frontend as part of Render's `buildCommand`. Use an external cron service (cron-job.org or UptimeRobot) to ping `/health` every 10 minutes.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DEPLOY-01 | Application deploys on Render free tier as single-process (uvicorn) | render.yaml config with `plan: free`, `startCommand: uvicorn`, port $PORT |
| DEPLOY-02 | Frontend SPA built and served by FastAPI as static files (single deployment) | StaticFiles mount + SPA catch-all pattern + combined buildCommand |
| DEPLOY-03 | Production requirements.txt excludes training-only dependencies | requirements-prod.txt already exists, verified correct |
| DEPLOY-04 | Cold-start UX shows loading screen while model initializes | Frontend warm-up ping to /health, loading overlay pattern |
| DEPLOY-05 | Keep-alive strategy reduces cold start frequency | External cron service pinging /health every 10 minutes |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| FastAPI | >=0.133.0 | Web framework + static file serving | Already in use, includes Starlette StaticFiles |
| uvicorn | (bundled with fastapi[standard]) | ASGI server | Single-process, Render default |
| Starlette StaticFiles | (bundled with FastAPI) | Serve compiled React SPA | Built-in, no extra dependency |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Vite | ^7.3.1 | Frontend build tool | Build step only (not deployed) |
| cron-job.org | N/A (external) | Keep-alive pings | Free external cron, no code needed |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| External cron | Render Cron Job service | Render cron jobs are a separate paid service; external is free |
| StaticFiles html=True | nginx reverse proxy | Would require Docker, overcomplicates free-tier deploy |
| Pre-built dist in git | Build on Render | Building on Render keeps repo clean but adds ~30s to deploy; pre-committing dist is simpler and faster |

**Installation (production):**
```bash
pip install -r requirements-prod.txt
```

## Architecture Patterns

### Recommended Project Structure (deployment-relevant)
```
/
├── api/
│   ├── main.py              # FastAPI app + StaticFiles mount
│   └── routers/             # API endpoints (/api/*, /health)
├── frontend/
│   ├── dist/                # Built SPA (served by FastAPI)
│   │   ├── index.html
│   │   └── assets/
│   ├── src/                 # Source (not deployed)
│   └── package.json
├── requirements-prod.txt    # Production deps only
├── render.yaml              # Render IaC config
└── Procfile                 # (optional, render.yaml preferred)
```

### Pattern 1: FastAPI StaticFiles SPA Serving
**What:** Mount the Vite `dist/` directory as static files, with a catch-all fallback to `index.html` for client-side routing.
**When to use:** Always in production when serving SPA from the same process.
**Example:**
```python
# In api/main.py, AFTER all router includes:
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

# Catch-all for SPA client-side routing (must be LAST route)
@app.get("/{path:path}")
async def spa_fallback(path: str):
    """Serve static files or fall back to index.html for SPA routing."""
    file_path = DIST_DIR / path
    if file_path.is_file():
        return FileResponse(file_path)
    return FileResponse(DIST_DIR / "index.html")

# Mount static assets (CSS, JS, images) - serves /assets/* directly
app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="spa")
```

**Important ordering:** The `/{path:path}` catch-all route and `StaticFiles` mount must come AFTER all API routers (`/api/*`, `/health`). FastAPI matches routes in declaration order, so API routes must be registered first.

### Pattern 2: render.yaml Infrastructure-as-Code
**What:** Declarative deployment configuration checked into the repository.
**When to use:** Always for Render deployments.
**Example:**
```yaml
services:
  - type: web
    name: lol-draft-predictor
    runtime: python
    plan: free
    buildCommand: |
      pip install -r requirements-prod.txt &&
      cd frontend && npm ci && npm run build
    startCommand: uvicorn api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: "3.12"
    autoDeploy: false
```

**Key details:**
- Render's Python runtime includes Node.js and npm in the build environment (verified via Render native runtimes docs)
- Default port is 10000, accessed via `$PORT` environment variable
- `npm ci` preferred over `npm install` for reproducible builds
- `autoDeploy: false` prevents accidental deploys on every push

### Pattern 3: Cold-Start Warm-Up Ping
**What:** Frontend pings `/health` on initial load; shows loading screen until model reports "ready".
**When to use:** Always on Render free tier (15-minute spin-down).
**Example:**
```typescript
// Frontend: useWarmUp hook
function useWarmUp() {
  const [ready, setReady] = useState(false)

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch('/health')
        const data = await res.json()
        if (data.status === 'ready') {
          setReady(true)
          return
        }
      } catch { /* server still waking */ }
      setTimeout(poll, 2000)  // retry every 2 seconds
    }
    poll()
  }, [])

  return ready
}
```

### Anti-Patterns to Avoid
- **Mounting StaticFiles before API routers:** API routes will never match; all requests go to static files
- **Using `app.mount("/static", ...)` for SPA:** Breaks client-side routing since paths like `/draft` won't resolve
- **Committing `node_modules/` to git:** Bloats repo; use `npm ci` in build step
- **Setting CORS to `allow_origins=["*"]` in production:** Not needed when SPA and API share the same origin
- **Gunicorn with multiple workers on free tier:** 512MB RAM budget; single uvicorn process is correct

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Static file serving | Custom file-reading middleware | `StaticFiles(html=True)` | Handles content types, caching headers, 304 responses |
| Keep-alive pinging | Custom background thread in FastAPI | External cron service (cron-job.org) | Zero code, free, reliable, doesn't consume app memory |
| SPA routing fallback | Middleware that intercepts 404s | Catch-all route + `FileResponse` | Simpler, debuggable, explicit |
| Production dep trimming | Manual pip uninstall script | Separate `requirements-prod.txt` | Already exists, clean separation |

**Key insight:** Render free tier is constrained (512MB RAM, 15-min spin-down, limited build time). Every custom solution adds complexity and risk. Use platform-native patterns.

## Common Pitfalls

### Pitfall 1: Route Order Kills API Endpoints
**What goes wrong:** `StaticFiles` mount at `/` intercepts `/api/predict` and returns 404 or `index.html` instead of the API response.
**Why it happens:** FastAPI processes mounts and routes in registration order. A greedy `app.mount("/", ...)` registered before API routers captures everything.
**How to avoid:** Register all API routers FIRST (`health`, `predict`, `champions`, `teams`, `admin`), then add the SPA catch-all LAST.
**Warning signs:** API calls return HTML instead of JSON; health endpoint stops responding.

### Pitfall 2: Vite Asset Paths Break When Served From Subdirectory
**What goes wrong:** CSS/JS assets fail to load with 404 errors because Vite builds with absolute paths (`/assets/index-abc.js`).
**Why it happens:** Vite defaults to `base: '/'` which is correct for root serving. If `base` is changed to `'./'`, asset paths break differently.
**How to avoid:** Keep Vite `base` as default (`'/'`). Ensure StaticFiles serves from the dist root at `/`.
**Warning signs:** Blank white page in browser, 404 errors in network tab for `.js`/`.css` files.

### Pitfall 3: CORS Errors Disappear But Then Return
**What goes wrong:** CORS works in development (Vite proxy), breaks after deployment.
**Why it happens:** In production, SPA and API share the same origin (same Render URL), so CORS is not needed. But the existing CORS middleware only allows `localhost:5173` and `localhost:3000`.
**How to avoid:** CORS middleware can remain for local dev. In production, same-origin requests bypass CORS entirely. No changes needed unless the API is also accessed from a different domain.
**Warning signs:** None in production (same origin). Only an issue if API is called cross-origin.

### Pitfall 4: Cold Start Returns 503 on First Prediction
**What goes wrong:** User submits a draft while the model is still loading; `get_adapter` dependency returns 503.
**Why it happens:** The `/health` endpoint works during loading (no adapter dependency), but `/api/predict` uses `Depends(get_adapter)` which checks `model_ready`.
**How to avoid:** Frontend must poll `/health` and only enable the "Predict" button when `status === "ready"`. Show a loading overlay until warm-up completes.
**Warning signs:** 503 responses from `/api/predict` in the first 30 seconds after cold start.

### Pitfall 5: Build Fails on Render Due to Memory/Time Limits
**What goes wrong:** `npm run build` or `pip install` exceeds Render free-tier build limits.
**Why it happens:** Free tier has limited build resources. Installing `scikit-learn` from source takes time.
**How to avoid:** Use `requirements-prod.txt` (excludes heavy training libs). Consider committing `frontend/dist/` to git to skip the Node build step entirely on Render.
**Warning signs:** Build timeout errors in Render dashboard.

### Pitfall 6: $PORT Hardcoded Instead of Environment Variable
**What goes wrong:** Service starts on port 8000 but Render expects port 10000.
**Why it happens:** Local dev uses port 8000, Render assigns `$PORT` (default 10000).
**How to avoid:** Always use `--port $PORT` in the start command within render.yaml.
**Warning signs:** Service deploys but health checks fail; Render marks service as unhealthy.

## Code Examples

### Complete render.yaml
```yaml
# render.yaml - Render Infrastructure-as-Code
services:
  - type: web
    name: lol-draft-predictor
    runtime: python
    plan: free
    buildCommand: pip install -r requirements-prod.txt && cd frontend && npm ci && npm run build
    startCommand: uvicorn api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: "3.12"
      - key: MODEL_UPLOAD_TOKEN
        generateValue: true
    autoDeploy: false
```

### FastAPI Static File Integration (api/main.py additions)
```python
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ... existing app setup, middleware, router includes ...

# ---------------------------------------------------------------------------
# SPA static file serving (MUST be AFTER all API routers)
# ---------------------------------------------------------------------------
DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if DIST_DIR.is_dir():
    @app.get("/{path:path}")
    async def serve_spa(path: str):
        file_path = DIST_DIR / path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(DIST_DIR / "index.html")
```

### Frontend Warm-Up Component
```typescript
// WarmUpScreen.tsx
import { useEffect, useState } from 'react'

export function WarmUpScreen({ children }: { children: React.ReactNode }) {
  const [ready, setReady] = useState(false)
  const [status, setStatus] = useState('Connecting to server...')

  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      try {
        const res = await fetch('/health')
        const data = await res.json()
        if (data.status === 'ready' && !cancelled) {
          setReady(true)
          return
        }
        setStatus('Loading ML model...')
      } catch {
        setStatus('Warming up server...')
      }
      if (!cancelled) setTimeout(poll, 2000)
    }
    poll()
    return () => { cancelled = true }
  }, [])

  if (ready) return <>{children}</>

  return (
    <div className="warm-up-screen">
      <h2>LoL Draft Predictor</h2>
      <p>{status}</p>
      <div className="spinner" />
    </div>
  )
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Separate frontend/backend deploys | Single-process SPA serving | Standard since FastAPI 0.100+ | One cold start, one URL, simpler |
| Gunicorn + uvicorn workers | Single uvicorn process | Render free tier constraint | Fits 512MB RAM budget |
| Docker multi-stage builds | Render native runtime | Render Python runtime includes npm | Simpler config, faster builds |
| Self-hosted cron for keep-alive | External free cron services | Always available | Zero maintenance, free |

**Deprecated/outdated:**
- Render's free tier used to have 750 free instance hours/month -- now free instances spin down after 15 minutes of inactivity regardless. External pinging keeps them alive within the monthly hour budget.

## Open Questions

1. **Should `frontend/dist/` be committed to git or built on Render?**
   - What we know: Building on Render works (npm available in Python runtime) but adds ~60s to deploy. Committing dist is simpler and faster.
   - What's unclear: Whether the repo owner prefers clean git history (no build artifacts) vs. faster deploys.
   - Recommendation: Build on Render via buildCommand for cleaner repo. If build times are problematic, switch to committing dist.

2. **MODEL_UPLOAD_TOKEN secret management**
   - What we know: API-07 requires a secret token for model uploads. render.yaml supports `generateValue: true` for auto-generated secrets.
   - What's unclear: Whether the user needs to know the generated value (to use the upload endpoint).
   - Recommendation: Use Render dashboard environment variables for the token so the user can see/set it.

## Sources

### Primary (HIGH confidence)
- [Render FastAPI deploy docs](https://render.com/docs/deploy-fastapi) - render.yaml structure, start command, port config
- [Render native runtimes](https://render.com/docs/native-runtimes) - Node.js/npm available in Python runtime builds
- [FastAPI Static Files docs](https://fastapi.tiangolo.com/tutorial/static-files/) - StaticFiles mount, html=True parameter
- [Render examples FastAPI](https://github.com/render-examples/fastapi/blob/main/render.yaml) - Official render.yaml template

### Secondary (MEDIUM confidence)
- [Render community: free tier spin-down](https://community.render.com/t/do-web-services-on-a-free-tier-go-to-sleep-after-some-time-inactive/3303) - 15-minute inactivity confirmed
- [FastAPI SPA serving discussion](https://github.com/fastapi/fastapi/discussions/5134) - Catch-all routing patterns
- [Render web services docs](https://render.com/docs/web-services) - Default port 10000, $PORT variable

### Tertiary (LOW confidence)
- [Keep-alive strategies](https://sergeiliski.medium.com/how-to-run-a-full-time-app-on-renders-free-tier-without-it-sleeping-bec26776d0b9) - 750 hours/month budget (may have changed)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - FastAPI StaticFiles and uvicorn are well-documented, already in use
- Architecture: HIGH - SPA-from-FastAPI is a well-established pattern with official examples
- Pitfalls: HIGH - Route ordering and CORS issues are widely documented in FastAPI community
- Render config: MEDIUM - Native runtime tool availability confirmed but not runtime-specific

**Research date:** 2026-02-25
**Valid until:** 2026-03-25 (Render pricing/features may change; core patterns stable)
