"""Refresh lookup tables from fresh Oracle's Elixir data without retraining.

The frozen LR model + frozen scaler + fresh lookup tables = accurate
predictions on the current meta.

Usage:
    python scripts/refresh_lookups.py --data-path data/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv
    python scripts/refresh_lookups.py --download
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
from src.config import MODELS_DIR, RAW_DATA_DIR, TARGET_LEAGUES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh lookup tables from fresh Oracle's Elixir data."
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to Oracle's Elixir CSV file.",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Auto-download latest data from Oracle's Elixir S3.",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # 1. Resolve data path
    # ---------------------------------------------------------------
    if args.download:
        from src.data.downloader import download_latest
        data_path = download_latest(RAW_DATA_DIR)
        print(f"Downloaded data to: {data_path}")
    elif args.data_path:
        data_path = Path(args.data_path)
        if not data_path.exists():
            print(f"ERROR: File not found: {data_path}")
            sys.exit(1)
    else:
        # Try to find latest raw file
        raw_files = sorted(RAW_DATA_DIR.glob("*.csv"))
        if not raw_files:
            print("ERROR: No CSV files found in data/raw/. Use --data-path or --download.")
            sys.exit(1)
        data_path = raw_files[-1]
        print(f"Using latest raw file: {data_path}")

    # ---------------------------------------------------------------
    # 2. Load and filter data
    # ---------------------------------------------------------------
    import pandas as pd

    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"  Raw rows: {len(df)}")

    # Filter to target leagues
    if "league" in df.columns:
        df = df[df["league"].isin(TARGET_LEAGUES)]
        print(f"  After league filter: {len(df)}")

    # Filter to team-level rows (one row per team per game)
    if "position" in df.columns:
        df = df[df["position"] == "team"]
        print(f"  After team filter: {len(df)}")

    if len(df) == 0:
        print("ERROR: No data after filtering. Check league names and data format.")
        sys.exit(1)

    # ---------------------------------------------------------------
    # 3. Run feature engineering analysis
    # ---------------------------------------------------------------
    print("\nRunning feature engineering analysis...")

    # We need to save the filtered data temporarily for the FE class
    temp_path = PROJECT_ROOT / "data" / "processed" / "complete_target_leagues_dataset.csv"
    df.to_csv(temp_path, index=False)

    from src.features.engineering import AdvancedFeatureEngineering

    fe = AdvancedFeatureEngineering(data_path=str(temp_path))
    fe.load_and_analyze_data()

    # ---------------------------------------------------------------
    # 4. Save individual lookup artifacts
    # ---------------------------------------------------------------
    print("\nSaving lookup artifacts...")

    artifacts = {
        "champion_meta_strength": fe.champion_meta_strength,
        "champion_popularity": fe.champion_popularity,
        "champion_characteristics": fe.champion_characteristics,
        "team_historical_performance": fe.team_historical_performance,
        "champion_matchups": getattr(fe, "lane_matchups", {}),
        "lane_advantages": getattr(fe, "lane_advantages", {}),
        "champion_archetypes": getattr(fe, "champion_archetypes", {}),
        "archetype_advantages": getattr(fe, "archetype_advantages", {}),
        "team_advantages": getattr(fe, "team_advantages", {}),
        "ban_priority": getattr(fe, "ban_priority", {}),
    }

    for name, data in artifacts.items():
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(data, path)
        print(f"  Saved {name}: {len(data)} entries -> {path}")

    # Also save target encoders if available
    if fe.target_encoders:
        joblib.dump(fe.target_encoders, MODELS_DIR / "target_encoders.joblib")
        print(f"  Saved target_encoders: {len(fe.target_encoders)} encoders")

    # ---------------------------------------------------------------
    # 5. Write metadata
    # ---------------------------------------------------------------
    patches = set()
    for key in fe.champion_meta_strength.keys():
        if isinstance(key, tuple) and len(key) == 2:
            patches.add(str(key[0]))

    teams = set(fe.team_historical_performance.keys())
    champions = set(fe.champion_characteristics.keys())

    metadata = {
        "refreshed_at": datetime.now().isoformat(),
        "source_file": str(data_path.name),
        "total_rows": len(df),
        "latest_patch": max(patches) if patches else "unknown",
        "patch_count": len(patches),
        "team_count": len(teams),
        "champion_count": len(champions),
    }

    metadata_path = MODELS_DIR / "lookup_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"\n  Metadata: {metadata_path}")
    for k, v in metadata.items():
        print(f"    {k}: {v}")

    # ---------------------------------------------------------------
    # 6. Validate
    # ---------------------------------------------------------------
    print("\nValidating saved artifacts...")
    for name in artifacts:
        loaded = joblib.load(MODELS_DIR / f"{name}.joblib")
        assert isinstance(loaded, dict), f"{name} is not a dict"
        print(f"  {name}: OK ({len(loaded)} entries)")

    print("\nRefresh complete. Lookup tables updated successfully.")
    print("The frozen LR model + scaler remain unchanged.")


if __name__ == "__main__":
    main()
