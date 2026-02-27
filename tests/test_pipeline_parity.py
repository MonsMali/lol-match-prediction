"""Verify adapter single-row feature computation matches training batch pipeline.

Loads a subset of the dataset, runs the batch pipeline, then computes
features for individual rows via the adapter and checks parity.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.formulas import CANONICAL_48_FEATURES


def _load_fe_and_data():
    """Load feature engineering object and a small data slice."""
    from src.features.engineering import AdvancedFeatureEngineering

    fe = AdvancedFeatureEngineering()
    fe.load_and_analyze_data()
    features_df = fe.create_advanced_features_vectorized()
    features_df = fe.apply_advanced_encoding()
    return fe, features_df


def _adapter_features_for_row(fe, row):
    """Compute 48 features for a single row using the adapter function."""
    from src.adapter.features import compute_features_for_side

    picks = {
        "top": str(row.get("top_champion", "Unknown")),
        "jungle": str(row.get("jng_champion", "Unknown")),
        "mid": str(row.get("mid_champion", "Unknown")),
        "bot": str(row.get("bot_champion", "Unknown")),
        "support": str(row.get("sup_champion", "Unknown")),
    }
    bans = [str(row.get(f"ban{i}", "NoBan")) for i in range(1, 6)]
    team = str(row.get("team", "Unknown"))
    side = "Blue" if row.get("side") == "Blue" else "Red"
    patch = str(row.get("patch", "Unknown"))
    year = int(row.get("year", 2023))
    league = str(row.get("league", "Unknown"))
    playoffs = int(row.get("playoffs", 0))
    split = str(row.get("split", "Unknown"))

    return compute_features_for_side(
        picks=picks, bans=bans, team=team, side=side,
        patch=patch, year=year, league=league, playoffs=playoffs, split=split,
        champion_characteristics=fe.champion_characteristics,
        champion_meta_strength=fe.champion_meta_strength,
        champion_popularity=fe.champion_popularity,
        team_historical_performance=fe.team_historical_performance,
        ban_priority=getattr(fe, "ban_priority", {}),
        lane_advantages=getattr(fe, "lane_advantages", {}),
        champion_archetypes=getattr(fe, "champion_archetypes", {}),
        archetype_advantages=getattr(fe, "archetype_advantages", {}),
        team_advantages=getattr(fe, "team_advantages", {}),
        target_encoders=fe.target_encoders,
    )


@pytest.mark.slow
def test_feature_parity():
    """Check adapter matches batch pipeline for 5 random rows."""
    fe, features_df = _load_fe_and_data()

    # Only check features present in both
    common_features = [f for f in CANONICAL_48_FEATURES if f in features_df.columns]
    assert len(common_features) >= 40, (
        f"Expected at least 40 common features, got {len(common_features)}"
    )

    # Sample 5 rows
    sample_indices = features_df.sample(n=min(5, len(features_df)), random_state=42).index

    for idx in sample_indices:
        row = fe.df.loc[idx]
        adapter_feats = _adapter_features_for_row(fe, row)

        for i, feat_name in enumerate(CANONICAL_48_FEATURES):
            if feat_name not in features_df.columns:
                continue
            batch_val = float(features_df.loc[idx, feat_name])
            adapter_val = float(adapter_feats[i])

            # Target encodings may differ (batch uses full data, adapter uses transform)
            if "target_encoded" in feat_name:
                continue

            # Per-row team perf may differ from final snapshot
            if feat_name in ("team_overall_winrate", "team_recent_winrate",
                             "team_form_trend", "team_experience"):
                continue

            np.testing.assert_allclose(
                adapter_val, batch_val, atol=0.05,
                err_msg=f"Feature {feat_name} mismatch at row {idx}: "
                        f"adapter={adapter_val}, batch={batch_val}",
            )


if __name__ == "__main__":
    test_feature_parity()
    print("Parity test passed.")
