"""FastAPI application entry point.

Creates the app with lifespan-managed model loading, CORS middleware,
and router inclusion. The LoLDraftAdapter singleton is loaded once at
startup and stored in app.state for dependency injection.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.adapter import LoLDraftAdapter

from api.routers import health

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

# TODO: predict router (Plan 02)
# TODO: champions router (Plan 03)
# TODO: teams router (Plan 03)
# TODO: admin router (Plan 03)
