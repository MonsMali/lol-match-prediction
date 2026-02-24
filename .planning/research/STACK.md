# Technology Stack — Web Layer

**Project:** LoL Draft Predictor Web App
**Researched:** 2026-02-24
**Scope:** Web framework, frontend, deployment. ML stack (scikit-learn, pandas, numpy, joblib) is pre-existing and not reconsidered here.

---

## Recommended Stack

### Backend API

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| FastAPI | >=0.115.0 (latest stable ~0.128.x) | Python REST API framework, serves prediction endpoints and static files | HIGH |
| Pydantic | v2 (bundled with FastAPI) | Request/response validation and serialization | HIGH |
| Uvicorn | >=0.30.0 | ASGI server — single process on free tier | HIGH |
| python-multipart | >=0.0.9 | Required by FastAPI for file upload endpoints (model swap) | HIGH |

**Why FastAPI over Flask:**
FastAPI is the 2025 standard for Python ML APIs. It provides async request handling (ASGI vs Flask's WSGI), automatic OpenAPI docs at `/docs`, and Pydantic v2 validation at no extra configuration cost. The existing `InteractiveLoLPredictor` class is CPU-bound during prediction (pandas + sklearn inference), so async gives no throughput gain on the hot path — but FastAPI still wins on DX, validation, and ecosystem momentum. Flask is viable but would require manual schema validation and produces worse developer experience. FastAPI is used in production by Hugging Face, Uber, and Microsoft for ML serving as of 2025.

**Why single Uvicorn process (not Gunicorn + Uvicorn workers):**
On Render's free tier (512MB RAM), loading the full ML pipeline (scikit-learn model + AdvancedFeatureEngineering data structures) consumes roughly 100-200MB at startup. Multiple Gunicorn workers would multiply this footprint and OOM-kill the service. Single-worker Uvicorn is the correct choice for memory-constrained free tier deployment.

### Frontend

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| React | 19.x (latest stable) | Component-based UI, draft board state management | HIGH |
| TypeScript | >=5.4 | Type safety for draft state, champion data structures | HIGH |
| Vite | >=6.0 | Build tool, dev server with proxy to FastAPI | HIGH |
| Tailwind CSS | 4.x (stable since Jan 2025) | Utility CSS, dark theme, responsive layout | HIGH |
| Zustand | >=5.0 | Lightweight client state (draft picks/bans, team selections) | MEDIUM |

**Why React over HTMX:**
The draft board requires complex interactive client state: which slots are filled, which champion is hovered, pick/ban sequencing, champion portrait grid with search/filter, and real-time probability display after API response. This is interactive UI with significant client-side state — not a form-submit pattern. HTMX's server-centric model is poorly suited; React's component model maps directly to draft slots, champion grids, and team panels.

**Why React over Vue:**
React 19 is production-stable as of December 2024. It has the largest ecosystem for LoL/esports fan projects (reference implementations exist for champion select UIs on GitHub using React + Vite). TypeScript integration is mature. React's market share is 44.7% vs Vue's ~16% — hiring and community resources are not a concern for a thesis project, but ecosystem breadth matters for finding reference implementations of drag-and-drop champion grids and draft UI patterns.

**Why Vite over Create React App:**
CRA is deprecated and unmaintained. Vite is the 2025 standard bundler for React projects. It has a first-party Tailwind CSS v4 plugin, sub-second HMR, and a simple proxy configuration for local FastAPI development. The built `/dist` output is served directly by FastAPI as static files — no separate frontend server needed in production.

**Why Tailwind CSS v4:**
Released stable January 22, 2025. CSS-first configuration, Vite plugin integration, and 5x faster builds vs v3. For a LoL-themed dark UI (dark backgrounds, gold accents, panel layouts), Tailwind's utility classes are faster to iterate than writing custom CSS. v4 removes the need for a separate `tailwind.config.js` when using the Vite plugin.

**Why Zustand:**
The draft board state (selected champions, ban slots, team assignments, current draft step) is global client state shared across multiple React components. Zustand provides this with ~1KB bundle overhead and no boilerplate — no reducers, no providers. React Context is viable but causes unnecessary re-renders across the entire tree when draft state changes. For this app's scale, Zustand is the right weight. Redux is overkill.

### Asset Integration

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| Riot Data Dragon CDN | No install — external CDN | Champion portrait images, served directly from Riot's CDN | HIGH |

**URL patterns (HIGH confidence — verified against current DDragon docs):**
- Champion square icon: `https://ddragon.leagueoflegends.com/cdn/{version}/img/champion/{ChampionName}.png`
- Champion JSON data: `https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json`
- Latest version endpoint: `https://ddragon.leagueoflegends.com/api/versions.json`

The backend should fetch the latest DDragon version on startup and cache the champion list. No Riot API key is required for Data Dragon — it is a public CDN.

### Deployment

| Platform | Tier | RAM | Cold Start | Verdict |
|----------|------|-----|------------|---------|
| Render | Free | 512MB | ~30-50s after 15min idle | Recommended |
| Railway | Free trial only (30 days, then $5/mo min) | 512MB | Similar | Not viable for permanent free hosting |

**Recommendation: Render.**

Railway removed its permanent free tier — it now provides only a 30-day trial before requiring the $5/month Hobby plan. Render maintains a genuine permanent free tier with 512MB RAM and 750 instance-hours/month. The 15-minute idle spin-down and ~30-second cold start are acceptable for a thesis demo tool.

**Memory budget on Render free tier (512MB):**

| Component | Estimated RAM |
|-----------|--------------|
| Python runtime + FastAPI + Uvicorn | ~50MB |
| pandas + numpy import | ~80MB |
| scikit-learn model (Logistic Regression, 33 features) | ~5-10MB |
| AdvancedFeatureEngineering data structures (champion meta, team history) | ~50-100MB |
| pandas DataFrame in memory during prediction | ~20MB peak |
| **Total estimated** | **~200-260MB** |

Headroom: ~250MB remaining. The LR model is small (joblib file is likely under 5MB). Risk is the champion_meta_strength and team_historical_performance artifacts — verify these are not enormous before deployment. If memory pressure occurs, lazy-load heavy artifacts on first prediction rather than at startup.

### Supporting Libraries (new additions for web layer)

| Library | Version | Purpose | Confidence |
|---------|---------|---------|------------|
| `fastapi[standard]` | >=0.115.0 | FastAPI with all standard extras (uvicorn, pydantic, etc.) | HIGH |
| `python-multipart` | >=0.0.9 | File upload support for model swap endpoint | HIGH |
| `httpx` | >=0.27.0 | Async HTTP client for DDragon version/champion fetch from backend | MEDIUM |
| `tanstack/react-query` (npm) | v5 | Server state for API calls (prediction requests, champion list fetch) | MEDIUM |

**Why TanStack Query (React Query) for frontend API calls:**
Prediction requests and champion data fetches need loading states, error handling, and caching. TanStack Query v5 provides this with minimal code. Without it, managing fetch lifecycle in `useEffect` + `useState` creates brittle boilerplate. For a public-facing tool, proper loading/error UX matters.

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Backend framework | FastAPI | Flask | WSGI (sync-only), no built-in validation, inferior DX in 2025 |
| Backend framework | FastAPI | Django REST Framework | Heavyweight — Django's ORM, admin, and auth add complexity for a stateless ML API |
| Frontend framework | React + Vite | HTMX + Jinja2 templates | Insufficient for interactive draft board; server-round-trip per click is too slow for responsive champion selection |
| Frontend framework | React + Vite | Vue 3 + Vite | Viable alternative; ecosystem narrower for LoL fan app reference implementations |
| Frontend framework | React + Vite | Next.js | SSR complexity unnecessary; this is a stateless client-side app with a Python API — no Node.js server needed |
| CSS | Tailwind v4 | CSS Modules | More verbose for rapid UI iteration; Tailwind utility classes map well to dark-themed panel layouts |
| State management | Zustand | Redux Toolkit | Overkill; Redux adds boilerplate for no benefit at this scale |
| State management | Zustand | React Context | Causes full tree re-renders on draft state changes; Zustand is selective |
| Deployment | Render | Railway | Railway has no permanent free tier as of 2025 — trial only |
| Deployment | Render | Fly.io | Free tier more restricted; cold starts less predictable |
| Deployment | Render | Vercel + separate API | Vercel does not support Python long-running servers; splits deployment into two services unnecessarily |
| ASGI server | Uvicorn (single) | Gunicorn + Uvicorn workers | Multiple workers would OOM on 512MB Render free tier given ML artifact memory |

---

## Installation

```bash
# Backend additions (add to existing requirements.txt)
fastapi[standard]>=0.115.0
python-multipart>=0.0.9
httpx>=0.27.0

# Frontend (in a /frontend directory)
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install -D tailwindcss @tailwindcss/vite
npm install zustand @tanstack/react-query
```

**Monorepo structure (recommended):**

```
/tese (repo root)
├── src/                    # Existing Python ML pipeline (unchanged)
├── models/                 # Existing model artifacts (unchanged)
├── api/                    # New: FastAPI app
│   ├── main.py             # FastAPI app, routes, static file serving
│   ├── predict.py          # Wraps InteractiveLoLPredictor
│   └── ddragon.py          # DDragon version/champion cache
├── frontend/               # New: React + Vite app
│   ├── src/
│   │   ├── components/     # DraftBoard, ChampionGrid, TeamPanel, etc.
│   │   ├── store/          # Zustand draft state
│   │   └── api/            # TanStack Query hooks
│   ├── dist/               # Built output — served by FastAPI
│   └── vite.config.ts
├── requirements.txt        # Updated with FastAPI dependencies
└── render.yaml             # Render deployment config
```

**Build and serve pattern:**

```bash
# Build frontend (run before deploying or in Render build command)
cd frontend && npm run build

# FastAPI serves built frontend from /frontend/dist
# FastAPI mounts /frontend/dist as static files
# All non-API routes serve /frontend/dist/index.html (SPA routing)
```

**Render deployment config (`render.yaml`):**

```yaml
services:
  - type: web
    name: lol-draft-predictor
    runtime: python
    buildCommand: "pip install -r requirements.txt && cd frontend && npm install && npm run build"
    startCommand: "uvicorn api.main:app --host 0.0.0.0 --port $PORT"
    envVars:
      - key: PYTHON_VERSION
        value: 3.12.3
```

---

## Sources

- FastAPI vs Flask for ML serving (2025): [FastAPI vs Flask 2025 — Strapi](https://strapi.io/blog/fastapi-vs-flask-python-framework-comparison), [Imarticus Blog](https://imarticus.org/blog/flask-vs-fastapi-which-is-better-for-deploying-ml-models/)
- FastAPI production deployment: [Render FastAPI best practices](https://render.com/articles/fastapi-production-deployment-best-practices), [FastAPI server workers docs](https://fastapi.tiangolo.com/deployment/server-workers/)
- Pydantic v2 with FastAPI: [Pydantic docs](https://docs.pydantic.dev/latest/), [Medium — FastAPI + Pydantic V2](https://medium.com/@connect.hashblock/fastapi-pydantic-v2-my-favorite-upgrade-this-year-af6e150ea0e9)
- React 19 production stability: [ifourtechnolab — React 18 vs 19](https://www.ifourtechnolab.com/blog/react-18-vs-react-19-key-differences-to-know-for-2024)
- Tailwind CSS v4 stable release: [Tailwind v4.0 announcement](https://tailwindcss.com/blog/tailwindcss-v4)
- Zustand 2025 adoption: [Zustand docs](https://zustand.docs.pmnd.rs/), [DEV Community — State Management 2025](https://dev.to/cristiansifuentes/react-state-management-in-2025-context-api-vs-zustand-385m)
- Vite + React + FastAPI integration: [TestDriven.io — FastAPI React](https://testdriven.io/blog/fastapi-react/), [DEV Community — Modern Full-Stack Setup](https://dev.to/stamigos/modern-full-stack-setup-fastapi-reactjs-vite-mui-with-typescript-2mef)
- Render free tier specs: [Render free tier docs](https://render.com/docs/free), [freetiers.com — Render](https://www.freetiers.com/directory/render)
- Railway pricing (no permanent free tier): [Railway pricing docs](https://docs.railway.com/reference/pricing/plans), [SaaS Price Pulse — Railway 2026](https://www.saaspricepulse.com/tools/railway)
- Riot Data Dragon CDN: [DDragon documentation](https://riot-api-libraries.readthedocs.io/en/latest/ddragon.html), [Hextechdocs](https://hextechdocs.dev/)
- LoL draft board reference implementations: [RCVolus lol-pick-ban-ui](https://github.com/RCVolus/lol-pick-ban-ui), [raymondchiu10 — LoL Champion Viewer with Vite + React](https://github.com/raymondchiu10/lol-champion-viewer)
- FastAPI file upload: [FastAPI request files docs](https://fastapi.tiangolo.com/tutorial/request-files/)
