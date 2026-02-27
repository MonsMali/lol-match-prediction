"""Exact SHAP-based per-prediction feature impact explanation.

Uses TreeExplainer in interventional mode with probability output
for RandomForest and GradientBoosting. Produces deterministic,
exact SHAP values in probability space (~2ms total).

SVM and LogisticRegression are excluded: SVM has no exact
decomposition (RBF kernel), and LinearExplainer outputs in a
different scale (raw coefficient space) that cannot be naively
combined with tree probability-space values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src.adapter.features import EXPECTED_FEATURES

logger = logging.getLogger(__name__)

_FEATURE_LABELS: dict[str, dict[str, str]] = {
    "team_avg_meta_strength": {
        "label": "Champion meta strength",
        "high": "strong meta picks",
        "low": "weak meta picks",
    },
    "team_avg_synergy": {
        "label": "Team synergy",
        "high": "high champion synergy",
        "low": "low champion synergy",
    },
    "team_composition_strength": {
        "label": "Composition strength",
        "high": "strong overall composition",
        "low": "weak overall composition",
    },
    "team_historical_winrate": {
        "label": "Team historical form",
        "high": "strong historical record",
        "low": "weak historical record",
    },
    "team_max_meta_strength": {
        "label": "Strongest pick in meta",
        "high": "a dominant meta pick",
        "low": "no standout meta pick",
    },
    "team_min_meta_strength": {
        "label": "Weakest pick in meta",
        "high": "no liability picks",
        "low": "a liability in the draft",
    },
    "team_meta_variance": {
        "label": "Draft consistency",
        "high": "uneven draft strength",
        "low": "consistent draft strength",
    },
    "team_max_synergy": {
        "label": "Best champion pairing",
        "high": "a strong champion pairing",
        "low": "no standout synergy pair",
    },
    "team_min_synergy": {
        "label": "Weakest champion pairing",
        "high": "no weak pairings",
        "low": "a weak champion pairing",
    },
    "team_synergy_variance": {
        "label": "Synergy consistency",
        "high": "uneven synergy across pairs",
        "low": "consistent synergy",
    },
    "meta_synergy_product": {
        "label": "Meta x Synergy",
        "high": "meta picks that also synergize",
        "low": "meta picks that clash",
    },
    "meta_synergy_ratio": {
        "label": "Meta vs Synergy balance",
        "high": "individually strong picks over team play",
        "low": "team play over individual strength",
    },
    "historical_meta_product": {
        "label": "Team form x Meta",
        "high": "a strong team on strong picks",
        "low": "team or picks are weak",
    },
    "composition_strength_gap": {
        "label": "Draft vs Team form gap",
        "high": "draft exceeds team baseline",
        "low": "draft below team baseline",
    },
    "ban_pressure_ratio": {
        "label": "Ban pressure",
        "high": "bans targeted the opponent",
        "low": "bans did not pressure picks",
    },
    "side_blue": {
        "label": "Side advantage",
        "high": "blue side advantage",
        "low": "red side advantage",
    },
}

_SUPPRESS_FEATURES = {
    "champion_diversity",
    "ban_count",
    "ban_diversity",
    "team_champions_banned",
    "playoffs",
    "year",
    "league_encoded",
    "team_encoded",
    "side_encoded",
    "patch_encoded",
    "split_encoded",
}

_MAX_INSIGHTS = 5
_MIN_IMPACT = 0.005  # 0.5 percentage points


@dataclass
class EnsembleExplainer:
    """Weight-averaged exact SHAP explainers for tree estimators.

    All explainers output probability-space values, so they can be
    directly averaged.
    """

    explainers: list[tuple[str, Any]]  # (estimator_name, TreeExplainer)
    weights: list[float]               # normalized weights

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Compute weight-averaged SHAP values in probability space.

        Returns (n_samples, n_features) for the positive class.
        """
        weighted_sum = np.zeros((X.shape[0], X.shape[1]))

        for (name, exp), w in zip(self.explainers, self.weights):
            vals = exp.shap_values(X)

            # TreeExplainer output varies by model:
            # - RF: (n_samples, n_features, 2) -- take class 1
            # - GBM: (n_samples, n_features) -- already positive class
            if isinstance(vals, list):
                vals = vals[1]
            elif isinstance(vals, np.ndarray) and vals.ndim == 3:
                vals = vals[:, :, 1]

            weighted_sum += w * vals

        return weighted_sum


def create_explainer(model: Any, scaler: Any) -> EnsembleExplainer | None:
    """Create exact SHAP explainers for the TreeEnsemble.

    The model is a _TreeEnsemble with .rf, .gbm, .rf_w, .gbm_w
    attributes. Both estimators get interventional TreeExplainer
    with model_output='probability'.
    """
    if not SHAP_AVAILABLE:
        logger.warning("shap not installed -- SHAP insights disabled")
        return None
    try:
        background = np.zeros((1, len(EXPECTED_FEATURES)))

        explainers: list[tuple[str, Any]] = []
        used_weights: list[float] = []

        for name, est, w in [
            ("RF", model.rf, model.rf_w),
            ("GBM", model.gbm, model.gbm_w),
        ]:
            exp = shap.TreeExplainer(
                est,
                data=background,
                feature_perturbation="interventional",
                model_output="probability",
            )
            explainers.append((name, exp))
            used_weights.append(w)
            logger.info("TreeExplainer (probability) ready for %s", name)

        return EnsembleExplainer(explainers=explainers, weights=used_weights)

    except Exception:
        logger.warning("Failed to initialize explainers", exc_info=True)
        return None


def compute_impact_insights(
    explainer: EnsembleExplainer | None,
    scaled_features: np.ndarray,
    side_label: str,
) -> list[dict[str, Any]]:
    """Compute human-readable impact insights for a single prediction.

    Deterministic: the same input always produces the same output.
    Returns top factors by absolute impact in probability space.
    """
    if explainer is None:
        return []

    try:
        shap_values = explainer.shap_values(scaled_features)
    except Exception:
        logger.warning("SHAP value computation failed", exc_info=True)
        return []

    values = shap_values.flatten()
    if len(values) != len(EXPECTED_FEATURES):
        logger.warning(
            "SHAP output length %d != expected %d",
            len(values), len(EXPECTED_FEATURES),
        )
        return []

    candidates = []
    for i, feat_name in enumerate(EXPECTED_FEATURES):
        if feat_name in _SUPPRESS_FEATURES:
            continue
        impact = float(values[i])
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
