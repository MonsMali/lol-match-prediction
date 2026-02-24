"""FastAPI dependency injection functions.

Provides the loaded LoLDraftAdapter singleton to route handlers
and enforces admin authentication for protected endpoints.
"""

import os

from fastapi import Header, HTTPException, Request

from src.adapter import LoLDraftAdapter


def get_adapter(request: Request) -> LoLDraftAdapter:
    """Return the loaded adapter or 503 if still loading.

    Used as a FastAPI dependency in endpoints that require the
    model to be ready (predict, champions, teams). The /health
    endpoint deliberately does NOT use this dependency so it can
    always respond.

    Raises:
        HTTPException: 503 Service Unavailable if the model has
            not finished loading yet.
    """
    if not getattr(request.app.state, "model_ready", False):
        raise HTTPException(status_code=503, detail="Model is loading")
    return request.app.state.adapter


def require_admin_token(authorization: str = Header(...)) -> None:
    """Validate Bearer token against ADMIN_TOKEN environment variable.

    Used as a FastAPI dependency for admin-only endpoints such as
    model upload.

    Raises:
        HTTPException: 401 Unauthorized if the token is missing,
            the environment variable is not set, or the token does
            not match.
    """
    expected = os.environ.get("ADMIN_TOKEN")
    if not expected or authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid admin token")
