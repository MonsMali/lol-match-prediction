"""Team-to-league grouping for the teams endpoint.

Provides a hardcoded mapping of major professional leagues to their
team rosters using the canonical team names from the training data.
At runtime, the /api/teams endpoint may augment this with the full
adapter.valid_teams set.

Team names must match those in the adapter's valid_teams set (derived
from team_historical_performance.joblib). The predictor.py file uses
abbreviations (T1, GenG, etc.) but the training data uses full names
(Gen.G, Hanwha Life Esports, etc.).
"""

# Canonical team names matching training data
TEAMS_BY_LEAGUE: dict[str, list[str]] = {
    "LCK": [
        "T1",
        "Gen.G",
        "KT Rolster",
        "DRX",
        "Dplus KIA",
        "Hanwha Life Esports",
        "Nongshim RedForce",
        "FEARX",
        "BRION",
        "OKSavingsBank BRION",
    ],
    "LEC": [
        "G2 Esports",
        "Fnatic",
        "Team Vitality",
        "SK Gaming",
        "Team Heretics",
        "KOI",
        "GiantX",
        "Karmine Corp",
        "Team BDS",
        "Natus Vincere",
    ],
    "LCS": [
        "Team Liquid",
        "Cloud9",
        "FlyQuest",
        "Dignitas",
        "Shopify Rebellion",
        "Sentinels",
        "LYON",
        "Disguised",
    ],
    "LPL": [
        "JD Gaming",
        "Bilibili Gaming",
        "Weibo Gaming",
        "Top Esports",
        "EDward Gaming",
        "Invictus Gaming",
        "Team WE",
        "Oh My God",
        "LNG Esports",
        "Ultra Prime",
        "Anyone's Legend",
        "LGD Gaming",
        "Ninjas in Pyjamas",
        "ThunderTalk Gaming",
    ],
}
