"""Build a role-specific champion matchup matrix from the match dataset.

Pairs rows by gameid to extract direct lane-vs-lane outcomes.
Produces models/champion_matchups.joblib with format:
    {(role, champion_a, champion_b): {"win_rate": float, "games": int}}

win_rate is the fraction of games champion_a won against champion_b
in the given role. Only matchups with >= MIN_GAMES are retained.
"""

import os
import sys
from collections import defaultdict

import joblib
import pandas as pd

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATASET_PATH, MODELS_DIR

MIN_GAMES = 3

ROLE_COLUMNS = {
    "top": "top_champion",
    "jungle": "jng_champion",
    "mid": "mid_champion",
    "bot": "bot_champion",
    "support": "sup_champion",
}


def build_matchup_matrix(dataset_path: str = None) -> dict:
    """Compute direct role-specific matchup win rates.

    Each match in the dataset has two rows (one per team) sharing
    the same gameid.  For each role we extract the two champions
    that faced each other and record the outcome from each
    perspective.
    """
    path = dataset_path or str(DATASET_PATH)
    print(f"Loading dataset from {path}")
    df = pd.read_csv(path)
    print(f"  {len(df)} rows, {df['gameid'].nunique()} unique games")

    # Split into blue/red sides and merge on gameid
    blue = df[df["side"] == "Blue"].set_index("gameid")
    red = df[df["side"] == "Red"].set_index("gameid")

    # Keep only games that have both sides
    common_ids = blue.index.intersection(red.index)
    blue = blue.loc[common_ids]
    red = red.loc[common_ids]

    print(f"  {len(common_ids)} games with both sides present")

    # Accumulate (role, champA, champB) -> {wins, games}
    counts: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"wins": 0, "games": 0}
    )

    for role, col in ROLE_COLUMNS.items():
        blue_champs = blue[col].values
        red_champs = red[col].values
        blue_results = blue["result"].values

        for b_champ, r_champ, b_result in zip(
            blue_champs, red_champs, blue_results
        ):
            if pd.isna(b_champ) or pd.isna(r_champ):
                continue

            b_champ = str(b_champ).strip()
            r_champ = str(r_champ).strip()

            if b_champ == r_champ:
                continue  # mirror matchup, skip

            # Blue champion's perspective
            counts[(role, b_champ, r_champ)]["games"] += 1
            counts[(role, b_champ, r_champ)]["wins"] += int(b_result)

            # Red champion's perspective
            counts[(role, r_champ, b_champ)]["games"] += 1
            counts[(role, r_champ, b_champ)]["wins"] += 1 - int(b_result)

    # Filter by minimum games and compute win rates
    matchups: dict[tuple[str, str, str], dict] = {}
    for key, data in counts.items():
        if data["games"] >= MIN_GAMES:
            matchups[key] = {
                "win_rate": data["wins"] / data["games"],
                "games": data["games"],
            }

    # Summary statistics
    total_entries = len(matchups)
    roles_summary = defaultdict(int)
    for (role, _, _) in matchups:
        roles_summary[role] += 1

    print(f"\nMatchup matrix built:")
    print(f"  {total_entries} total entries (>= {MIN_GAMES} games)")
    for role in ROLE_COLUMNS:
        print(f"  {role}: {roles_summary[role]} matchup pairs")

    return matchups


def main():
    matchups = build_matchup_matrix()
    out_path = MODELS_DIR / "champion_matchups.joblib"
    joblib.dump(matchups, out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
