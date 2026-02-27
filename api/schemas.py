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
    top_player: str | None = None
    jungle_player: str | None = None
    mid_player: str | None = None
    bot_player: str | None = None
    support_player: str | None = None


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


class InsightFactor(BaseModel):
    """A single impact factor explaining part of the prediction."""

    label: str
    impact_pct: float
    description: str


class ChampionSuggestion(BaseModel):
    """A suggested champion swap that would improve win probability."""

    role: str
    champion: str
    delta_pct: float
    current_champion: str


class PickImpact(BaseModel):
    """Marginal impact of a single champion pick on win probability."""

    role: str
    champion: str
    impact_pct: float


class ModelMeta(BaseModel):
    """Staleness and provenance metadata for the model."""

    training_patch: str
    training_year: int
    domain: str = "professional"


class TeamContextResponse(BaseModel):
    """Team-level context surfaced alongside the prediction."""

    historical_winrate: float
    recent_winrate: float
    form_trend: float
    meta_adaptation: float


class PredictResponse(BaseModel):
    """Fast path: win probabilities + SHAP insights + pick impacts + model metadata."""

    blue_win_probability: float
    red_win_probability: float
    blue_insights: list[InsightFactor] = []
    red_insights: list[InsightFactor] = []
    blue_pick_impacts: list[PickImpact] = []
    red_pick_impacts: list[PickImpact] = []
    blue_team_context: TeamContextResponse | None = None
    red_team_context: TeamContextResponse | None = None
    model: ModelMeta


class SuggestionsResponse(BaseModel):
    """Async path: champion swap suggestions for both sides."""

    blue_suggestions: list[ChampionSuggestion] = []
    red_suggestions: list[ChampionSuggestion] = []


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


class RosterResponse(BaseModel):
    """Team rosters: team name -> role -> player name."""

    rosters: dict[str, dict[str, str]]


class HealthResponse(BaseModel):
    """Health check response, always returned even during model loading."""

    status: str
    model_name: str | None = None
    model_version: str | None = None
    champion_count: int | None = None
    team_count: int | None = None
    memory_usage_mb: float | None = None
