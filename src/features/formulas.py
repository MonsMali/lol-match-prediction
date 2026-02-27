"""Shared constants and pure formula functions for feature engineering.

Imported by both the training pipeline (engineering.py) and the
production adapter (adapter/features.py) to guarantee identical
feature computation.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Meta strength formula weights
# ---------------------------------------------------------------------------
META_STRENGTH_WIN_WEIGHT = 0.7
META_STRENGTH_POP_WEIGHT = 0.3
META_STRENGTH_POP_CAP = 0.5

# ---------------------------------------------------------------------------
# Default fallback values for missing lookups
# ---------------------------------------------------------------------------
DEFAULT_META = 0.5
DEFAULT_WINRATE = 0.5
DEFAULT_SYNERGY = 0.5
DEFAULT_POPULARITY = 0.0
DEFAULT_LANE_ADVANTAGE = 0.5
DEFAULT_LANE_CONFIDENCE = 0.1
DEFAULT_ARCHETYPE_ADVANTAGE = 0.5

# ---------------------------------------------------------------------------
# Default player performance for unknown players
# ---------------------------------------------------------------------------
DEFAULT_PLAYER_PERF = {
    "winrate": 0.5,
    "recent_winrate": 0.5,
    "games": 0,
}

DEFAULT_PLAYER_MASTERY = {
    "mastery": 0.5,
    "games": 0,
}

# ---------------------------------------------------------------------------
# Default champion characteristics for unknown champions
# ---------------------------------------------------------------------------
DEFAULT_CHAMPION_CHAR = {
    "win_rate": 0.5,
    "early_game_strength": 0.5,
    "late_game_strength": 0.5,
    "scaling_factor": 0,
    "flexibility": 1,
}

# ---------------------------------------------------------------------------
# Default team performance for unknown teams
# ---------------------------------------------------------------------------
DEFAULT_TEAM_PERF = {
    "overall_winrate": 0.5,
    "recent_winrate": 0.5,
    "form_trend": 0.0,
    "games_played": 0,
}

# ---------------------------------------------------------------------------
# Canonical 48-feature list -- exact order expected by the scaler
# ---------------------------------------------------------------------------
# Legacy 48-feature list for backward compatibility with pre-player models
CANONICAL_48_FEATURES: list[str] = [
    "team_avg_winrate",                # 0  champion char avg
    "team_early_strength",             # 1  champion char avg
    "team_late_strength",              # 2  champion char avg
    "team_scaling",                    # 3  champion char avg
    "team_flexibility",                # 4  champion char avg
    "composition_balance",             # 5  std of scaling
    "team_meta_strength",              # 6  meta avg
    "team_meta_consistency",           # 7  1 - std(meta)
    "team_popularity",                 # 8  popularity avg
    "meta_advantage",                  # 9  meta - 0.5
    "ban_count",                       # 10 ban analysis
    "ban_diversity",                   # 11 ban analysis
    "high_priority_bans",              # 12 ban analysis
    "team_overall_winrate",            # 13 team perf
    "team_recent_winrate",             # 14 team perf
    "team_form_trend",                 # 15 team perf
    "team_experience",                 # 16 team perf
    "composition_historical_winrate",  # 17 comp synergy
    "playoffs",                        # 18 context
    "side_blue",                       # 19 context
    "year",                            # 20 context
    "champion_count",                  # 21 context
    "meta_form_interaction",           # 22 interaction
    "scaling_experience_interaction",  # 23 interaction
    "team_lane_advantage",             # 24 lane matchup
    "lane_advantage_consistency",      # 25 lane matchup
    "lane_matchup_confidence",         # 26 lane matchup
    "strongest_lane_advantage",        # 27 lane matchup
    "weakest_lane_advantage",          # 28 lane matchup
    "lanes_with_advantage",            # 29 lane matchup
    "team_archetype_advantage",        # 30 lane matchup
    "team_historical_advantage",       # 31 h2h
    "team_matchup_consistency",        # 32 h2h
    "favorable_matchups",              # 33 h2h
    "unfavorable_matchups",            # 34 h2h
    "lane_meta_synergy",              # 35 advanced interaction
    "experience_matchup_confidence",   # 36 advanced interaction
    "form_matchup_interaction",        # 37 advanced interaction
    "scaling_lane_advantage",          # 38 advanced interaction
    "league_target_encoded",           # 39 target encoding
    "team_target_encoded",             # 40 target encoding
    "patch_target_encoded",            # 41 target encoding
    "split_target_encoded",            # 42 target encoding
    "top_champion_target_encoded",     # 43 target encoding
    "jng_champion_target_encoded",     # 44 target encoding
    "mid_champion_target_encoded",     # 45 target encoding
    "bot_champion_target_encoded",     # 46 target encoding
    "sup_champion_target_encoded",     # 47 target encoding
]

# 7 new player-level features (available after retraining with player data)
PLAYER_FEATURES: list[str] = [
    "team_avg_player_winrate",              # 48 player perf
    "team_avg_player_recent",               # 49 player perf
    "team_avg_player_experience",           # 50 player perf
    "team_avg_champion_mastery",            # 51 player mastery
    "team_min_champion_mastery",            # 52 player mastery
    "team_player_form",                     # 53 player form
    "player_mastery_experience_interaction", # 54 player interaction
]

# Full feature list including player features (55 features total)
CANONICAL_55_FEATURES: list[str] = CANONICAL_48_FEATURES + PLAYER_FEATURES


def compute_meta_strength(win_rate: float, popularity: float) -> float:
    """Compute champion meta strength from win rate and popularity.

    Mirrors the formula in ``_calculate_meta_indicators``:
        meta_strength = win_rate * 0.7 + min(popularity, 0.5) * 0.3
    """
    capped_pop = min(popularity, META_STRENGTH_POP_CAP)
    return win_rate * META_STRENGTH_WIN_WEIGHT + capped_pop * META_STRENGTH_POP_WEIGHT
