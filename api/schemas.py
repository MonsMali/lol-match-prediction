"""Pydantic v2 request/response models for all API endpoints.

Defines the data shapes for prediction requests, champion lists,
team lists, and health checks. Used by FastAPI for automatic
validation, serialization, and OpenAPI documentation.
"""

from pydantic import BaseModel, Field


class TeamDraft(BaseModel):
    """Role-to-champion mapping for one side of a draft."""

    top: str
    jungle: str
    mid: str
    bot: str
    support: str


class PredictRequest(BaseModel):
    """Complete draft submission for a match prediction.

    Both teams, all 10 picks (5 per team by role), and all 10 bans
    (5 per team) are required. Patch is optional and defaults to
    the latest available in the training data.
    """

    blue_team: str
    red_team: str
    blue_picks: TeamDraft
    red_picks: TeamDraft
    blue_bans: list[str] = Field(min_length=5, max_length=5)
    red_bans: list[str] = Field(min_length=5, max_length=5)
    patch: str | None = None


class PredictResponse(BaseModel):
    """Prediction result with win probabilities only.

    No metadata, no confidence scores, no echoed draft.
    Probabilities are floats in [0, 1] and sum to 1.0.
    """

    blue_win_probability: float
    red_win_probability: float


class ChampionInfo(BaseModel):
    """Single champion entry with DDragon image URL."""

    name: str
    key: str
    image_url: str


class ChampionListResponse(BaseModel):
    """List of all champions known to the model."""

    champions: list[ChampionInfo]


class TeamListResponse(BaseModel):
    """Teams grouped by league."""

    teams: dict[str, list[str]]


class HealthResponse(BaseModel):
    """Health check response, always returned even during model loading."""

    status: str
    model_name: str | None = None
    model_version: str | None = None
    champion_count: int | None = None
    team_count: int | None = None
    memory_usage_mb: float | None = None
