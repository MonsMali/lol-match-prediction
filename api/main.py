"""FastAPI application entry point.

Creates the app with lifespan-managed model loading, CORS middleware,
and router inclusion. The LoLDraftAdapter singleton is loaded once at
startup and stored in app.state for dependency injection.

In production the compiled React SPA is served from ``frontend/dist/``
via a catch-all route registered *after* all API routers so that
``/health``, ``/api/*``, and ``/admin/*`` still respond with JSON.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.adapter import LoLDraftAdapter

from api.routers import admin, champions, health, predict, teams

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model on startup, clean up on shutdown.

    Sets app.state.model_ready to False initially. On successful
    adapter instantiation, sets model_ready to True and stores the
    adapter. On failure, stores the error message for diagnostics.
    """
    app.state.model_ready = False
    app.state.adapter = None
    app.state.model_load_error = None

    try:
        adapter = LoLDraftAdapter()
        app.state.adapter = adapter
        app.state.model_ready = True
        logger.info("Model loaded successfully: %s", adapter.model_name)
    except Exception as exc:
        app.state.model_load_error = str(exc)
        logger.error("Failed to load model: %s", exc)

    yield

    # Shutdown: no cleanup needed (singleton handles its own state)


app = FastAPI(
    title="LoL Draft Predictor",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS middleware -- allow Vite dev server and common local ports
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(champions.router)
app.include_router(teams.router)
app.include_router(admin.router)

# ---------------------------------------------------------------------------
# SPA static file serving (MUST be AFTER all API routers)
# ---------------------------------------------------------------------------
DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if DIST_DIR.is_dir():

    @app.get("/{path:path}")
    async def serve_spa(path: str):
        """Serve static files or fall back to index.html for SPA routing."""
        file_path = DIST_DIR / path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(DIST_DIR / "index.html")
