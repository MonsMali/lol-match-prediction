"""Health check endpoint.

Always responds regardless of model loading state, making it
suitable for uptime monitoring and load balancer health probes.
Does NOT use the get_adapter dependency (which returns 503 during
loading).
"""

from fastapi import APIRouter, Request

from api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Return current service health and model status.

    When the model is loaded, returns full diagnostic information
    including model name, version, champion/team counts, and memory
    usage. When still loading, returns status="loading" with all
    optional fields as None.
    """
    if getattr(request.app.state, "model_ready", False):
        adapter = request.app.state.adapter
        status = adapter.get_status()
        return HealthResponse(
            status="ready",
            model_name=status.model_name,
            model_version=status.model_version,
            champion_count=status.champion_count,
            team_count=status.team_count,
            memory_usage_mb=round(status.memory_usage_mb, 1),
        )
    return HealthResponse(status="loading")
