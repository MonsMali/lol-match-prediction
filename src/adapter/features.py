"""Single-match 48-feature computation for the thesis LR model.

Computes the exact 48-feature vector expected by the production
Logistic Regression model and its StandardScaler. Every value is
computed from pre-serialized lookup dicts -- no CSV or DataFrame
loading occurs.

The feature order is defined in ``src.features.formulas.CANONICAL_48_FEATURES``
and matches ``scaler.feature_names_in_`` exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.features.formulas import (
    CANONICAL_48_FEATURES,
    CANONICAL_55_FEATURES,
    PLAYER_FEATURES,
    DEFAULT_CHAMPION_CHAR,
    DEFAULT_LANE_ADVANTAGE,
    DEFAULT_LANE_CONFIDENCE,
    DEFAULT_ARCHETYPE_ADVANTAGE,
    DEFAULT_META,
    DEFAULT_PLAYER_PERF,
    DEFAULT_PLAYER_MASTERY,
    DEFAULT_TEAM_PERF,
    DEFAULT_WINRATE,
)


EXPECTED_FEATURES = CANONICAL_48_FEATURES

# Role-to-column mapping used by training pipeline
_ROLE_TO_COL = {
    "top": "top_champion",
    "jungle": "jng_champion",
    "mid": "mid_champion",
    "bot": "bot_champion",
    "support": "sup_champion",
}

_ROLES_ORDERED = ["top", "jungle", "mid", "bot", "support"]


def compute_features_for_side(
    picks: dict[str, str],
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
    players: dict[str, str] | None = None,
    player_performance: dict | None = None,
    player_champion_mastery: dict | None = None,
) -> list[float]:
    """Compute the feature vector for one side of a match.

    All champion and team names must already be normalized to their
    canonical forms. Feature order matches CANONICAL_48_FEATURES
    (= the scaler's ``feature_names_in_``), with optional player
    features appended when player data is available.

    Args:
        picks: Mapping of role -> champion name (5 entries).
        bans: List of 5 banned champion names.
        team: Canonical team name.
        side: ``"Blue"`` or ``"Red"``.
        patch: Patch as a string (e.g. ``"14.18"``).
        year: Calendar year of the match.
        league: League identifier (e.g. ``"LCK"``).
        playoffs: ``1`` if playoffs, else ``0``.
        split: Split identifier (e.g. ``"Summer"``).
        champion_characteristics: Champion stats dict.
        champion_meta_strength: ``{(patch, champion): float}`` lookup.
        champion_popularity: ``{(patch, champion): dict}`` lookup.
        team_historical_performance: ``{team: dict}`` performance lookup.
        ban_priority: ``{champion: dict}`` ban stats.
        lane_advantages: ``{role: {champ: {opponent: dict}}}`` lookup.
        champion_archetypes: ``{champion: str}`` archetype mapping.
        archetype_advantages: ``{role: {matchup_key: float}}`` lookup.
        team_advantages: ``{team: {opponent: float}}`` h2h lookup.
        target_encoders: ``{column: TargetEncoder}`` fitted encoders.
        players: Optional mapping of role -> player name (5 entries).
        player_performance: Optional ``{player: dict}`` performance lookup.
        player_champion_mastery: Optional ``{player: {champion: dict}}``.

    Returns:
        A list of floats. 48 features if no player data, 55 if player
        features are included (matches the scaler's feature count).
    """
    champ_names = [picks.get(r, "") for r in _ROLES_ORDERED]
    valid_champions = [c for c in champ_names if c and c != "Unknown"]

    # ------------------------------------------------------------------
    # 1. Champion characteristics (6 features)
    # ------------------------------------------------------------------
    char_metrics = []
    for champ in valid_champions:
        char = champion_characteristics.get(champ, DEFAULT_CHAMPION_CHAR)
        char_metrics.append([
            char.get("win_rate", 0.5),
            char.get("early_game_strength", 0.5),
            char.get("late_game_strength", 0.5),
            char.get("scaling_factor", 0),
            char.get("flexibility", 1),
        ])

    if char_metrics:
        arr = np.array(char_metrics)
        team_avg_winrate = float(np.mean(arr[:, 0]))
        team_early_strength = float(np.mean(arr[:, 1]))
        team_late_strength = float(np.mean(arr[:, 2]))
        team_scaling = float(np.mean(arr[:, 3]))
        team_flexibility = float(np.mean(arr[:, 4]))
        composition_balance = float(np.std(arr[:, 3]))
    else:
        team_avg_winrate = 0.5
        team_early_strength = 0.5
        team_late_strength = 0.5
        team_scaling = 0.0
        team_flexibility = 1.0
        composition_balance = 0.0

    # ------------------------------------------------------------------
    # 2. Meta strength (4 features)
    # ------------------------------------------------------------------
    meta_vals = []
    pop_vals = []
    for champ in valid_champions:
        meta_vals.append(
            champion_meta_strength.get((patch, champ), DEFAULT_META)
        )
        pop_data = champion_popularity.get((patch, champ), {"popularity": 0})
        pop_vals.append(pop_data.get("popularity", 0) if isinstance(pop_data, dict) else 0)

    if meta_vals:
        team_meta_strength = float(np.mean(meta_vals))
        team_meta_consistency = 1.0 - float(np.std(meta_vals))
        team_popularity = float(np.mean(pop_vals))
        meta_advantage = team_meta_strength - 0.5
    else:
        team_meta_strength = DEFAULT_META
        team_meta_consistency = 1.0
        team_popularity = 0.0
        meta_advantage = 0.0

    # ------------------------------------------------------------------
    # 3. Ban analysis (3 features)
    # ------------------------------------------------------------------
    valid_bans = [b for b in bans if b and b != "NoBan"]
    ban_count = float(len(valid_bans))
    ban_diversity = float(len(set(valid_bans)))

    high_priority_bans = 0.0
    for ban in valid_bans:
        ban_data = ban_priority.get(ban, {})
        total = ban_data.get("total_bans", 0)
        early = ban_data.get("early_bans", 0)
        if total > 0 and early / total > 0.5:
            high_priority_bans += 1.0

    # ------------------------------------------------------------------
    # 4. Team performance (4 features)
    # ------------------------------------------------------------------
    perf = team_historical_performance.get(team, DEFAULT_TEAM_PERF)
    if isinstance(perf, (int, float)):
        # Legacy format: flat winrate float
        team_overall_winrate = float(perf)
        team_recent_winrate = float(perf)
        team_form_trend = 0.0
        team_experience = 0.5
    else:
        team_overall_winrate = perf.get("overall_winrate", 0.5)
        team_recent_winrate = perf.get("recent_winrate", 0.5)
        team_form_trend = perf.get("form_trend", 0.0)
        team_experience = min(perf.get("games_played", 0) / 100.0, 1.0)

    # ------------------------------------------------------------------
    # 5. Composition synergy (1 feature, default)
    # ------------------------------------------------------------------
    composition_historical_winrate = 0.5

    # ------------------------------------------------------------------
    # 6. Context (4 features)
    # ------------------------------------------------------------------
    f_playoffs = float(playoffs)
    f_side_blue = 1.0 if side == "Blue" else 0.0
    f_year = float(year)
    champion_count = float(len(valid_champions))

    # ------------------------------------------------------------------
    # 7. Interaction features (2 features)
    # ------------------------------------------------------------------
    meta_form_interaction = team_meta_strength * team_form_trend
    scaling_experience_interaction = team_scaling * team_experience

    # ------------------------------------------------------------------
    # 8. Lane matchup advantages (7 features)
    # ------------------------------------------------------------------
    lane_advs = []
    lane_confs = []
    archetype_advs = []

    for role in _ROLES_ORDERED:
        champ = picks.get(role, "")
        if not champ or champ == "Unknown":
            lane_advs.append(DEFAULT_LANE_ADVANTAGE)
            lane_confs.append(DEFAULT_LANE_CONFIDENCE)
            archetype_advs.append(DEFAULT_ARCHETYPE_ADVANTAGE)
            continue

        # Lane advantage: average across all known opponents
        role_la = lane_advantages.get(role, {}).get(champ, {})
        if role_la:
            avg_adv = float(np.mean([d["advantage"] for d in role_la.values()]))
            avg_conf = float(np.mean([d["confidence"] for d in role_la.values()]))
            lane_advs.append(avg_adv)
            lane_confs.append(avg_conf)
        else:
            lane_advs.append(DEFAULT_LANE_ADVANTAGE)
            lane_confs.append(DEFAULT_LANE_CONFIDENCE)

        # Archetype advantage
        champ_archetype = champion_archetypes.get(champ, "Balanced")
        role_arch_adv = archetype_advantages.get(role, {})
        arch_perf = [
            v for k, v in role_arch_adv.items()
            if k.startswith(f"{champ_archetype}_vs_")
        ]
        archetype_advs.append(float(np.mean(arch_perf)) if arch_perf else DEFAULT_ARCHETYPE_ADVANTAGE)

    team_lane_advantage = float(np.mean(lane_advs))
    lane_advantage_consistency = 1.0 - float(np.std(lane_advs))
    lane_matchup_confidence = float(np.mean(lane_confs))
    strongest_lane_advantage = float(max(lane_advs))
    weakest_lane_advantage = float(min(lane_advs))
    lanes_with_advantage = float(sum(1 for a in lane_advs if a > 0.55))
    team_archetype_advantage = float(np.mean(archetype_advs))

    # ------------------------------------------------------------------
    # 9. Team head-to-head (4 features)
    # ------------------------------------------------------------------
    team_adv_list = list((team_advantages.get(team, {})).values())
    if team_adv_list:
        team_historical_advantage = float(np.mean(team_adv_list))
        team_matchup_consistency = 1.0 - float(np.std(team_adv_list))
        favorable_matchups = float(sum(1 for a in team_adv_list if a > 0.6))
        unfavorable_matchups = float(sum(1 for a in team_adv_list if a < 0.4))
    else:
        team_historical_advantage = 0.5
        team_matchup_consistency = 1.0
        favorable_matchups = 0.0
        unfavorable_matchups = 0.0

    # ------------------------------------------------------------------
    # 10. Advanced interaction features (4 features)
    # ------------------------------------------------------------------
    lane_meta_synergy = team_lane_advantage * team_meta_strength
    experience_matchup_confidence = team_experience * lane_matchup_confidence
    form_matchup_interaction = team_form_trend * team_historical_advantage
    scaling_lane_advantage = team_scaling * strongest_lane_advantage

    # ------------------------------------------------------------------
    # 11. Target encodings (9 features)
    # ------------------------------------------------------------------
    league_te = _target_encode(target_encoders, "league", league)
    team_te = _target_encode(target_encoders, "team", team)
    patch_te = _target_encode(target_encoders, "patch", patch)
    split_te = _target_encode(target_encoders, "split", split)

    top_te = _target_encode(target_encoders, "top_champion", picks.get("top", "Unknown"))
    jng_te = _target_encode(target_encoders, "jng_champion", picks.get("jungle", "Unknown"))
    mid_te = _target_encode(target_encoders, "mid_champion", picks.get("mid", "Unknown"))
    bot_te = _target_encode(target_encoders, "bot_champion", picks.get("bot", "Unknown"))
    sup_te = _target_encode(target_encoders, "sup_champion", picks.get("support", "Unknown"))

    # ------------------------------------------------------------------
    # 12. Player performance (7 features, optional)
    # ------------------------------------------------------------------
    player_features = _compute_player_features(
        picks, players, player_performance, player_champion_mastery
    )

    # ------------------------------------------------------------------
    # Assemble in CANONICAL order
    # ------------------------------------------------------------------
    base_features = [
        team_avg_winrate,                # 0
        team_early_strength,             # 1
        team_late_strength,              # 2
        team_scaling,                    # 3
        team_flexibility,                # 4
        composition_balance,             # 5
        team_meta_strength,              # 6
        team_meta_consistency,           # 7
        team_popularity,                 # 8
        meta_advantage,                  # 9
        ban_count,                       # 10
        ban_diversity,                   # 11
        high_priority_bans,              # 12
        team_overall_winrate,            # 13
        team_recent_winrate,             # 14
        team_form_trend,                 # 15
        team_experience,                 # 16
        composition_historical_winrate,  # 17
        f_playoffs,                      # 18
        f_side_blue,                     # 19
        f_year,                          # 20
        champion_count,                  # 21
        meta_form_interaction,           # 22
        scaling_experience_interaction,  # 23
        team_lane_advantage,             # 24
        lane_advantage_consistency,      # 25
        lane_matchup_confidence,         # 26
        strongest_lane_advantage,        # 27
        weakest_lane_advantage,          # 28
        lanes_with_advantage,            # 29
        team_archetype_advantage,        # 30
        team_historical_advantage,       # 31
        team_matchup_consistency,        # 32
        favorable_matchups,              # 33
        unfavorable_matchups,            # 34
        lane_meta_synergy,              # 35
        experience_matchup_confidence,   # 36
        form_matchup_interaction,        # 37
        scaling_lane_advantage,          # 38
        league_te,                       # 39
        team_te,                         # 40
        patch_te,                        # 41
        split_te,                        # 42
        top_te,                          # 43
        jng_te,                          # 44
        mid_te,                          # 45
        bot_te,                          # 46
        sup_te,                          # 47
    ]

    # Append player features only if the model expects them
    if player_features is not None:
        base_features.extend(player_features)

    return base_features


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _compute_player_features(
    picks: dict[str, str],
    players: dict[str, str] | None,
    player_performance: dict | None,
    player_champion_mastery: dict | None,
) -> list[float] | None:
    """Compute 7 player-level features for one side.

    Returns None if player data is not available (model does not
    expect player features). Returns a list of 7 floats otherwise.
    """
    if not player_performance or not players:
        return None

    winrates = []
    recents = []
    experiences = []
    masteries = []
    forms = []

    for role in _ROLES_ORDERED:
        player_name = players.get(role, "")
        champion = picks.get(role, "")

        perf = player_performance.get(player_name, DEFAULT_PLAYER_PERF)
        wr = perf.get("winrate", 0.5)
        rr = perf.get("recent_winrate", 0.5)
        gp = min(perf.get("games", 0) / 100.0, 1.0)

        # Champion mastery
        cm = 0.5
        if player_champion_mastery and player_name in player_champion_mastery:
            cm_data = player_champion_mastery[player_name].get(champion, DEFAULT_PLAYER_MASTERY)
            cm = cm_data.get("mastery", 0.5)

        winrates.append(wr)
        recents.append(rr)
        experiences.append(gp)
        masteries.append(cm)
        forms.append(rr - wr)

    avg_wr = float(np.mean(winrates))
    avg_recent = float(np.mean(recents))
    avg_exp = float(np.mean(experiences))
    avg_mastery = float(np.mean(masteries))
    min_mastery = float(min(masteries))
    avg_form = float(np.mean(forms))
    mastery_exp_interaction = avg_mastery * avg_exp

    return [
        avg_wr,                    # team_avg_player_winrate
        avg_recent,                # team_avg_player_recent
        avg_exp,                   # team_avg_player_experience
        avg_mastery,               # team_avg_champion_mastery
        min_mastery,               # team_min_champion_mastery
        avg_form,                  # team_player_form
        mastery_exp_interaction,   # player_mastery_experience_interaction
    ]


def _target_encode(
    encoders: dict[str, Any],
    column: str,
    value: str,
) -> float:
    """Apply a fitted TargetEncoder, returning 0.5 for unseen values.

    TargetEncoder.transform expects a 2D array and returns a 2D array.
    Unseen categories are mapped to the global target mean by sklearn,
    which is close to 0.5 for balanced binary classification.
    """
    enc = encoders.get(column)
    if enc is None:
        return 0.5

    try:
        import numpy as _np
        result = enc.transform(_np.array([[value]]))
        return float(result[0, 0])
    except Exception:
        return 0.5
