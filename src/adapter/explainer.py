"""Exact SHAP-based per-prediction feature impact explanation.

Uses LinearExplainer for the Logistic Regression model. This is
exact and fast (~1ms), producing deterministic SHAP values in
log-odds space that are converted to probability-space impacts.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src.adapter.features import EXPECTED_FEATURES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Human-readable labels for each feature
# ---------------------------------------------------------------------------
_FEATURE_LABELS: dict[str, dict[str, str]] = {
    "team_avg_winrate": {
        "label": "Champion pool strength",
        "high": "strong champion pool",
        "low": "weak champion pool",
    },
    "team_meta_strength": {
        "label": "Meta alignment",
        "high": "strong meta picks",
        "low": "off-meta picks",
    },
    "team_meta_consistency": {
        "label": "Draft consistency",
        "high": "consistently strong picks",
        "low": "uneven pick strength",
    },
    "meta_advantage": {
        "label": "Meta advantage",
        "high": "above-average meta draft",
        "low": "below-average meta draft",
    },
    "team_overall_winrate": {
        "label": "Team strength",
        "high": "strong historical record",
        "low": "weak historical record",
    },
    "team_recent_winrate": {
        "label": "Recent form",
        "high": "winning recently",
        "low": "losing recently",
    },
    "team_form_trend": {
        "label": "Form trend",
        "high": "trending upward",
        "low": "trending downward",
    },
    "team_experience": {
        "label": "Team experience",
        "high": "experienced roster",
        "low": "limited experience",
    },
    "team_lane_advantage": {
        "label": "Lane matchups",
        "high": "favorable lane matchups",
        "low": "unfavorable lane matchups",
    },
    "lane_advantage_consistency": {
        "label": "Lane consistency",
        "high": "consistent lane strength",
        "low": "uneven lane matchups",
    },
    "strongest_lane_advantage": {
        "label": "Best lane matchup",
        "high": "a dominant lane matchup",
        "low": "no standout lane",
    },
    "weakest_lane_advantage": {
        "label": "Weakest lane",
        "high": "no lane liability",
        "low": "a vulnerable lane",
    },
    "team_archetype_advantage": {
        "label": "Archetype advantage",
        "high": "favorable playstyle matchup",
        "low": "unfavorable playstyle matchup",
    },
    "team_historical_advantage": {
        "label": "Head-to-head record",
        "high": "strong against opponent",
        "low": "weak against opponent",
    },
    "team_matchup_consistency": {
        "label": "Matchup consistency",
        "high": "consistent vs opponents",
        "low": "inconsistent vs opponents",
    },
    "lane_meta_synergy": {
        "label": "Lane-meta synergy",
        "high": "meta picks in favorable lanes",
        "low": "meta picks in weak lanes",
    },
    "meta_form_interaction": {
        "label": "Meta x Form",
        "high": "strong team on strong meta picks",
        "low": "form or meta is weak",
    },
    "scaling_experience_interaction": {
        "label": "Scaling x Experience",
        "high": "experienced team with scaling comp",
        "low": "scaling comp without experience",
    },
    "side_blue": {
        "label": "Side advantage",
        "high": "blue side advantage",
        "low": "red side advantage",
    },
    "high_priority_bans": {
        "label": "Ban pressure",
        "high": "targeted high-priority bans",
        "low": "low-impact bans",
    },
    "composition_balance": {
        "label": "Composition balance",
        "high": "diverse scaling profile",
        "low": "one-dimensional scaling",
    },
    "team_popularity": {
        "label": "Pick popularity",
        "high": "popular, proven picks",
        "low": "niche picks",
    },
    "team_flexibility": {
        "label": "Champion flexibility",
        "high": "flexible champion picks",
        "low": "narrow champion picks",
    },
    "team_scaling": {
        "label": "Team scaling",
        "high": "strong late-game scaling",
        "low": "weak late-game scaling",
    },
    "team_early_strength": {
        "label": "Early game power",
        "high": "strong early game",
        "low": "weak early game",
    },
    "team_late_strength": {
        "label": "Late game power",
        "high": "strong late game",
        "low": "weak late game",
    },
}

# Features to suppress from insight display (constant or non-informative)
_SUPPRESS_FEATURES = {
    "champion_count",
    "composition_historical_winrate",
    "ban_count",
    "ban_diversity",
    "playoffs",
    "year",
    "lanes_with_advantage",
    "favorable_matchups",
    "unfavorable_matchups",
    "league_target_encoded",
    "team_target_encoded",
    "patch_target_encoded",
    "split_target_encoded",
    "top_champion_target_encoded",
    "jng_champion_target_encoded",
    "mid_champion_target_encoded",
    "bot_champion_target_encoded",
    "sup_champion_target_encoded",
}

_MAX_INSIGHTS = 5
_MIN_IMPACT = 0.005  # 0.5 percentage points


def create_explainer(model: Any, scaler: Any) -> Any | None:
    """Create a SHAP LinearExplainer for the Logistic Regression model.

    Uses the scaler mean as background data. LinearExplainer produces
    exact SHAP values for linear models.
    """
    if not SHAP_AVAILABLE:
        logger.warning("shap not installed -- SHAP insights disabled")
        return None

    try:
        background = np.zeros((1, len(EXPECTED_FEATURES)))
        explainer = shap.LinearExplainer(
            model,
            background,
            feature_perturbation="interventional",
        )
        logger.info("LinearExplainer ready for LogisticRegression")
        return explainer
    except Exception:
        logger.warning("Failed to initialize LinearExplainer", exc_info=True)
        return None


def compute_impact_insights(
    explainer: Any | None,
    scaled_features: np.ndarray,
    side_label: str,
) -> list[dict[str, Any]]:
    """Compute human-readable impact insights for a single prediction.

    Deterministic: the same input always produces the same output.
    Returns top factors by absolute impact.
    """
    if explainer is None:
        return []

    try:
        shap_values = explainer.shap_values(scaled_features)
    except Exception:
        logger.warning("SHAP value computation failed", exc_info=True)
        return []

    # LinearExplainer may return list [class0, class1] or just class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    values = shap_values.flatten()
    if len(values) != len(EXPECTED_FEATURES):
        logger.warning(
            "SHAP output length %d != expected %d",
            len(values), len(EXPECTED_FEATURES),
        )
        return []

    # Convert log-odds SHAP values to approximate probability impact
    # For LR: sigmoid'(0) = 0.25, so prob_impact ~ shap_value * 0.25
    # This is a first-order approximation near p=0.5
    prob_scale = 0.25

    candidates = []
    for i, feat_name in enumerate(EXPECTED_FEATURES):
        if feat_name in _SUPPRESS_FEATURES:
            continue
        raw_impact = float(values[i])
        impact = raw_impact * prob_scale
        if abs(impact) < _MIN_IMPACT:
            continue

        info = _FEATURE_LABELS.get(feat_name)
        if info is None:
            continue

        impact_pct = round(impact * 100, 1)
        if impact > 0:
            description = f"{info['label']}: {info['high']} (+{abs(impact_pct)}%)"
        else:
            description = f"{info['label']}: {info['low']} (-{abs(impact_pct)}%)"

        candidates.append({
            "feature": feat_name,
            "label": info["label"],
            "impact": impact,
            "impact_pct": impact_pct,
            "description": description,
        })

    candidates.sort(key=lambda c: abs(c["impact"]), reverse=True)
    return candidates[:_MAX_INSIGHTS]
