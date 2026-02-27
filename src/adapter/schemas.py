"""Typed schemas for the LoL Draft Adapter.

Defines the data structures for draft input, prediction results,
and adapter health status. All schemas use stdlib dataclasses
to avoid external dependencies.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DraftInput:
    """Complete draft information for a single match prediction.

    All 10 picks (by role) and 10 bans must be provided.
    Champion names use display format (e.g. "Kai'Sa", "Lee Sin").
    The adapter normalizes names internally before feature computation.

    Attributes:
        blue_team: Blue side team name (must match training data).
        red_team: Red side team name (must match training data).
        blue_picks: Role-to-champion mapping for blue side.
            Keys must be: top, jungle, mid, bot, support.
        red_picks: Role-to-champion mapping for red side.
            Keys must be: top, jungle, mid, bot, support.
        blue_bans: List of 5 champion display names banned by blue side.
        red_bans: List of 5 champion display names banned by red side.
        patch: Game patch version string (e.g. "14.18"). Defaults to
            the latest patch available in the training data when None.
    """

    blue_team: str
    red_team: str
    blue_picks: dict[str, str]
    red_picks: dict[str, str]
    blue_bans: list[str]
    red_bans: list[str]
    blue_players: Optional[dict[str, str]] = None
    red_players: Optional[dict[str, str]] = None
    patch: Optional[str] = None


VALID_ROLES = frozenset({"top", "jungle", "mid", "bot", "support"})


@dataclass
class PredictionResult:
    """Prediction output returned by the adapter.

    Probabilities are floats in [0, 1] and always sum to 1.0.

    Attributes:
        blue_win_prob: Probability that blue side wins.
        red_win_prob: Probability that red side wins.
        model_name: Human-readable model identifier
            (e.g. "VotingClassifier").
        model_version: Version string for traceability
            (e.g. "enhanced-v1").
        blue_insights: Per-prediction impact factors for blue side.
        red_insights: Per-prediction impact factors for red side.
        training_patch: The latest patch the model was trained on.
        training_year: The latest year in training data.
    """

    blue_win_prob: float
    red_win_prob: float
    model_name: str
    model_version: str
    blue_insights: list[dict[str, Any]] = field(default_factory=list)
    red_insights: list[dict[str, Any]] = field(default_factory=list)
    blue_pick_impacts: list[PickImpact] = field(default_factory=list)
    red_pick_impacts: list[PickImpact] = field(default_factory=list)
    blue_team_context: "TeamContext | None" = None
    red_team_context: "TeamContext | None" = None
    training_patch: str = "14.18"
    training_year: int = 2024


@dataclass
class PickImpact:
    """Marginal impact of a single champion pick on win probability.

    Computed via leave-one-out: replace the champion with a neutral
    baseline and measure the probability delta.

    Attributes:
        role: Position (top, jungle, mid, bot, support).
        champion: Champion display name.
        impact_pct: Percentage-point change in win probability
            attributable to this pick (positive = helps, negative = hurts).
    """

    role: str
    champion: str
    impact_pct: float


@dataclass
class TeamContext:
    """Team-level context surfaced in the prediction response.

    These values are derived from the lookup dicts and represent
    the team's historical strength independent of the current draft.

    Attributes:
        historical_winrate: Team's overall win rate (0-1).
        recent_winrate: Win rate over last 10 matches (0-1).
        form_trend: Recent minus overall winrate (positive = hot streak).
        meta_adaptation: Team's average meta strength minus 0.5
            (positive = above-average meta picks).
    """

    historical_winrate: float
    recent_winrate: float
    form_trend: float
    meta_adaptation: float


@dataclass
class SuggestionResult:
    """Champion swap suggestions for both sides.

    Computed asynchronously from the primary prediction.
    """

    blue_suggestions: list[dict[str, Any]] = field(default_factory=list)
    red_suggestions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AdapterStatus:
    """Health and diagnostic information for the loaded adapter.

    Returned by ``LoLDraftAdapter.get_status()`` to support the
    ``/health`` API endpoint in Phase 2.

    Attributes:
        loaded_artifacts: List of artifact file names successfully loaded.
        model_name: Class name of the loaded model.
        model_version: Version string matching PredictionResult.
        memory_usage_mb: Resident set size of the current process in MB.
        champion_count: Number of unique champions in lookup dicts.
        team_count: Number of unique teams in lookup dicts.
    """

    loaded_artifacts: list[str]
    model_name: str
    model_version: str
    memory_usage_mb: float
    champion_count: int
    team_count: int
