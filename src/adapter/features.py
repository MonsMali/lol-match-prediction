"""Single-match feature computation for the LoL Draft Adapter.

Computes the exact 27-feature vector expected by the production
VotingClassifier model (``models/production/best_model.joblib``).

The feature order is hard-coded to match the scaler's
``feature_names_in_`` attribute.  Every value is computed from
pre-serialized lookup dicts -- no CSV or DataFrame loading occurs.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np


# -----------------------------------------------------------------------
# Canonical feature order -- MUST match scaler.feature_names_in_ exactly
# -----------------------------------------------------------------------
EXPECTED_FEATURES: list[str] = [
    "team_avg_meta_strength",
    "team_min_meta_strength",
    "team_max_meta_strength",
    "team_meta_variance",
    "team_avg_synergy",
    "team_min_synergy",
    "team_max_synergy",
    "team_synergy_variance",
    "champion_diversity",
    "team_composition_strength",
    "team_historical_winrate",
    "ban_count",
    "ban_diversity",
    "team_champions_banned",
    "playoffs",
    "side_blue",
    "year",
    "league_encoded",
    "team_encoded",
    "side_encoded",
    "patch_encoded",
    "split_encoded",
    "meta_synergy_product",
    "meta_synergy_ratio",
    "historical_meta_product",
    "composition_strength_gap",
    "ban_pressure_ratio",
]

# Default meta strength when a (patch, champion) pair is unknown
_DEFAULT_META = 0.5

# Default synergy when a (champion, champion) pair is unknown
_DEFAULT_SYNERGY = 0.5

# Default team winrate for unknown teams
_DEFAULT_WINRATE = 0.5


def compute_features_for_side(
    picks: dict[str, str],
    bans: list[str],
    team: str,
    side: str,
    patch_float: float,
    year: int,
    league: str,
    playoffs: int,
    split: str,
    meta_strength: dict[tuple[float, str], float],
    synergies: dict[tuple[str, str], float],
    team_perf: dict[str, float],
    encoders: dict[str, Any],
) -> list[float]:
    """Compute the 27-feature vector for one side of a match.

    All champion and team names must already be normalized to their
    canonical forms (see ``validation.validate_draft``).

    Args:
        picks: Mapping of role -> champion name (5 entries).
        bans: List of 5 banned champion names.
        team: Canonical team name.
        side: ``"Blue"`` or ``"Red"``.
        patch_float: Patch as a float (e.g. ``14.18``).  Lookup dicts
            use float keys; encoders use string keys -- this function
            handles the conversion internally.
        year: Calendar year of the match.
        league: League identifier (e.g. ``"LCK"``).
        playoffs: ``1`` if playoffs, else ``0``.
        split: Split identifier (e.g. ``"Summer"``).
        meta_strength: ``{(patch_float, champion): float}`` lookup.
        synergies: ``{(champion_a, champion_b): float}`` lookup.
        team_perf: ``{team_name: float}`` overall winrate lookup.
        encoders: ``{column_name: LabelEncoder}`` for categoricals.

    Returns:
        A list of 27 floats in the exact order of
        :data:`EXPECTED_FEATURES`.
    """
    champ_names = list(picks.values())  # 5 champions

    # ------------------------------------------------------------------
    # 1. Meta strength statistics (4 features)
    # ------------------------------------------------------------------
    meta_vals = [
        meta_strength.get((patch_float, c), _DEFAULT_META)
        for c in champ_names
    ]
    avg_meta = float(np.mean(meta_vals))
    min_meta = float(np.min(meta_vals))
    max_meta = float(np.max(meta_vals))
    var_meta = float(np.var(meta_vals))

    # ------------------------------------------------------------------
    # 2. Synergy statistics (4 features)
    # ------------------------------------------------------------------
    syn_vals: list[float] = []
    for c1, c2 in combinations(champ_names, 2):
        # Synergy dict may store (a,b) or (b,a) -- check both
        val = synergies.get((c1, c2))
        if val is None:
            val = synergies.get((c2, c1), _DEFAULT_SYNERGY)
        syn_vals.append(val)

    if syn_vals:
        avg_syn = float(np.mean(syn_vals))
        min_syn = float(np.min(syn_vals))
        max_syn = float(np.max(syn_vals))
        var_syn = float(np.var(syn_vals))
    else:
        avg_syn = _DEFAULT_SYNERGY
        min_syn = _DEFAULT_SYNERGY
        max_syn = _DEFAULT_SYNERGY
        var_syn = 0.0

    # ------------------------------------------------------------------
    # 3. Champion diversity (1 feature)
    # ------------------------------------------------------------------
    champion_diversity = float(len(set(champ_names)))  # always 5

    # ------------------------------------------------------------------
    # 4. Team composition strength (1 feature)
    # ------------------------------------------------------------------
    team_composition_strength = avg_meta * avg_syn

    # ------------------------------------------------------------------
    # 5. Team historical winrate (1 feature)
    # ------------------------------------------------------------------
    team_winrate = team_perf.get(team, _DEFAULT_WINRATE)

    # ------------------------------------------------------------------
    # 6. Ban features (3 features)
    # ------------------------------------------------------------------
    valid_bans = [b for b in bans if b and b != "NoBan"]
    ban_count = float(len(valid_bans))
    ban_diversity = float(len(set(valid_bans)))

    # team_champions_banned: count of bans that are among the team's
    # picked champions (very rare -- near zero in training data).
    team_champions_banned = float(
        sum(1 for b in valid_bans if b in champ_names)
    )

    # ------------------------------------------------------------------
    # 7. Match context (3 features)
    # ------------------------------------------------------------------
    f_playoffs = float(playoffs)
    f_side_blue = 1.0 if side == "Blue" else 0.0
    f_year = float(year)

    # ------------------------------------------------------------------
    # 8. Label-encoded categoricals (5 features)
    # ------------------------------------------------------------------
    league_enc = _label_encode(encoders, "league", league)
    team_enc = _label_encode(encoders, "team", team)
    side_enc = _label_encode(encoders, "side", side)

    # Encoders use string patch format (e.g. "14.18"), while lookup
    # dicts use float.  Convert float -> string for encoder lookup.
    patch_str = _patch_float_to_str(patch_float)
    patch_enc = _label_encode(encoders, "patch", patch_str)
    split_enc = _label_encode(encoders, "split", split)

    # ------------------------------------------------------------------
    # 9. Interaction features (5 features)
    # ------------------------------------------------------------------
    meta_synergy_product = avg_meta * avg_syn  # same as composition_strength
    eps = 1e-10
    meta_synergy_ratio = avg_meta / (avg_syn + eps)
    historical_meta_product = team_winrate * avg_meta
    composition_strength_gap = team_composition_strength - team_winrate
    ban_pressure_ratio = (
        team_champions_banned / (ban_count + eps) if ban_count > 0 else 0.0
    )

    # ------------------------------------------------------------------
    # Assemble in canonical order
    # ------------------------------------------------------------------
    return [
        avg_meta,                    # team_avg_meta_strength
        min_meta,                    # team_min_meta_strength
        max_meta,                    # team_max_meta_strength
        var_meta,                    # team_meta_variance
        avg_syn,                     # team_avg_synergy
        min_syn,                     # team_min_synergy
        max_syn,                     # team_max_synergy
        var_syn,                     # team_synergy_variance
        champion_diversity,          # champion_diversity
        team_composition_strength,   # team_composition_strength
        team_winrate,                # team_historical_winrate
        ban_count,                   # ban_count
        ban_diversity,               # ban_diversity
        team_champions_banned,       # team_champions_banned
        f_playoffs,                  # playoffs
        f_side_blue,                 # side_blue
        f_year,                      # year
        league_enc,                  # league_encoded
        team_enc,                    # team_encoded
        side_enc,                    # side_encoded
        patch_enc,                   # patch_encoded
        split_enc,                   # split_encoded
        meta_synergy_product,        # meta_synergy_product
        meta_synergy_ratio,          # meta_synergy_ratio
        historical_meta_product,     # historical_meta_product
        composition_strength_gap,    # composition_strength_gap
        ban_pressure_ratio,          # ban_pressure_ratio
    ]


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _label_encode(
    encoders: dict[str, Any],
    column: str,
    value: str,
) -> float:
    """Apply a LabelEncoder, returning a fallback for unseen labels.

    Unseen labels are mapped to ``-1`` (a conventional sentinel).
    The scaler will still normalize this value, so predictions degrade
    gracefully rather than crashing.
    """
    enc = encoders.get(column)
    if enc is None:
        return -1.0
    try:
        return float(enc.transform([value])[0])
    except ValueError:
        # Label not seen during training
        return -1.0


def _patch_float_to_str(patch_float: float) -> str:
    """Convert a float patch (14.18) to the string format used by encoders.

    The encoder classes include strings like ``"14.18"``, ``"12.1"``,
    ``"4.1"`` etc.  A naive ``str(14.1)`` produces ``"14.1"`` which
    matches.  However ``str(14.10)`` also produces ``"14.1"`` in
    Python -- and the encoder stores ``"14.1"`` for patch 14.10, so
    this conversion is safe.
    """
    # Handle the special "Unknown" sentinel
    if patch_float is None:
        return "Unknown"

    # Remove trailing zeros: 14.10 -> "14.1", 14.18 -> "14.18"
    formatted = f"{patch_float:.2f}"
    # Strip trailing '0' then trailing '.' if needed
    formatted = formatted.rstrip("0").rstrip(".")
    return formatted
