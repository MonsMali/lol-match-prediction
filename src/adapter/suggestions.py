"""Proactive champion suggestion engine.

For each role, computes the marginal win probability delta of
swapping the current pick for top meta alternatives. Returns one
suggestion per role, with each champion appearing at most once.

Exposes a two-phase API: build_suggestion_features (pure
computation, no model calls) and resolve_suggestions (takes
pre-computed probabilities). This allows the adapter to batch
all suggestion model calls from both sides into a single
predict_proba invocation.
"""

from __future__ import annotations

import logging
from typing import Any

from src.adapter.features import compute_features_for_side

logger = logging.getLogger(__name__)

_ROLES = ("top", "jungle", "mid", "bot", "support")

# Candidates per role (top N by meta strength).
_MAX_CANDIDATES = 5

_MAX_SUGGESTIONS = 5
_MIN_DELTA = 0.5  # percentage points


def build_suggestion_features(
    current_picks: dict[str, str],
    bans: list[str],
    team: str,
    side: str,
    patch: str,
    year: int,
    league: str,
    playoffs: int,
    split: str,
    champion_characteristics: dict,
    champion_meta_strength: dict,
    champion_popularity: dict,
    team_historical_performance: dict,
    ban_priority: dict,
    lane_advantages: dict,
    champion_archetypes: dict,
    archetype_advantages: dict,
    team_advantages: dict,
    target_encoders: dict,
    all_champions: set[str],
    opponent_picks: dict[str, str],
    matchups: dict | None = None,
) -> tuple[list[tuple[str, str]], list[list[float]]]:
    """Build feature vectors for all candidate swaps.

    Returns:
        keys: List of (role, candidate_champion) tuples.
        features: Corresponding feature vectors (not yet scaled).
    """
    committed = (
        set(current_picks.values())
        | set(opponent_picks.values())
        | set(bans)
    )

    meta_on_patch = {
        champ: strength
        for (p, champ), strength in champion_meta_strength.items()
        if p == patch and champ not in committed
    }

    if not meta_on_patch:
        return [], []

    ranked = sorted(meta_on_patch.items(), key=lambda x: x[1], reverse=True)
    candidate_pool = [name for name, _ in ranked[:_MAX_CANDIDATES]]

    common_ctx = dict(
        bans=bans,
        team=team,
        side=side,
        patch=patch,
        year=year,
        league=league,
        playoffs=playoffs,
        split=split,
        champion_characteristics=champion_characteristics,
        champion_meta_strength=champion_meta_strength,
        champion_popularity=champion_popularity,
        team_historical_performance=team_historical_performance,
        ban_priority=ban_priority,
        lane_advantages=lane_advantages,
        champion_archetypes=champion_archetypes,
        archetype_advantages=archetype_advantages,
        team_advantages=team_advantages,
        target_encoders=target_encoders,
    )

    keys: list[tuple[str, str]] = []
    features: list[list[float]] = []

    for role in _ROLES:
        current_champ = current_picks.get(role)
        if not current_champ:
            continue

        for candidate in candidate_pool:
            if candidate == current_champ:
                continue
            alt_picks = dict(current_picks)
            alt_picks[role] = candidate
            feats = compute_features_for_side(picks=alt_picks, **common_ctx)
            keys.append((role, candidate))
            features.append(feats)

    return keys, features


def resolve_suggestions(
    keys: list[tuple[str, str]],
    probas: list[float],
    current_picks: dict[str, str],
    current_win_prob: float,
) -> list[dict[str, Any]]:
    """Convert raw probabilities into ranked, deduplicated suggestions."""
    raw: list[dict[str, Any]] = []
    for (role, candidate), prob in zip(keys, probas):
        delta = float(prob) - current_win_prob
        delta_pct = round(delta * 100, 1)
        if delta_pct >= _MIN_DELTA:
            raw.append({
                "role": role,
                "champion": candidate,
                "delta_pct": delta_pct,
                "current_champion": current_picks[role],
            })

    raw.sort(key=lambda s: s["delta_pct"], reverse=True)

    # Greedy dedup: each champion once, each role once
    used_champs: set[str] = set()
    used_roles: set[str] = set()
    result: list[dict[str, Any]] = []

    for s in raw:
        if s["champion"] in used_champs or s["role"] in used_roles:
            continue
        result.append(s)
        used_champs.add(s["champion"])
        used_roles.add(s["role"])
        if len(result) >= _MAX_SUGGESTIONS:
            break

    return result
