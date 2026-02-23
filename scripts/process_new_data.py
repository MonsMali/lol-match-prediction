"""
Process New Oracle's Elixir Data and Merge with Existing Dataset.

This script:
1. Processes raw Oracle's Elixir CSV files from data/raw/
2. Cleans and transforms to team-match format
3. Merges with existing processed dataset
4. Reports on new matches added

Usage:
    1. Download raw files from Google Drive to data/raw/
       Files should be named: YYYY_LoL_esports_match_data_from_OraclesElixir.csv
    2. Run: python scripts/process_new_data.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# Configuration
RAW_DATA_DIR = project_root / "data" / "raw"
PROCESSED_DATA_DIR = project_root / "data" / "processed"
EXISTING_DATASET = PROCESSED_DATA_DIR / "complete_target_leagues_dataset.csv"
OUTPUT_DATASET = PROCESSED_DATA_DIR / "complete_target_leagues_dataset.csv"
BACKUP_DIR = PROCESSED_DATA_DIR / "backups"

# Target leagues
TARGET_LEAGUES = ['LPL', 'LCK', 'LEC', 'LCS', 'EU LCS', 'NA LCS', 'WLDs', 'MSI', 'Worlds']

# League name standardization
LEAGUE_MAPPING = {
    'NA LCS': 'LCS',
    'EU LCS': 'LEC',
    'Worlds': 'WLDs',
}


def get_raw_files():
    """Get all raw Oracle's Elixir files."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    pattern = "*_LoL_esports_match_data_from_OraclesElixir*.csv"
    files = list(RAW_DATA_DIR.glob(pattern))

    # Also check for simpler patterns
    if not files:
        files = list(RAW_DATA_DIR.glob("*OraclesElixir*.csv"))

    if not files:
        files = list(RAW_DATA_DIR.glob("*.csv"))

    # Sort by year in filename
    def extract_year(f):
        try:
            return int(f.name.split('_')[0])
        except:
            return 0

    files.sort(key=extract_year)
    return files


def load_existing_dataset():
    """Load the existing processed dataset."""
    if EXISTING_DATASET.exists():
        df = pd.read_csv(EXISTING_DATASET)
        print(f"Loaded existing dataset: {len(df):,} matches")
        return df
    else:
        print("No existing dataset found - will create new one")
        return None


def process_raw_file(file_path):
    """Process a single raw Oracle's Elixir file.

    Transforms player-level rows to team-match format.
    """
    year = extract_year_from_filename(file_path.name)
    print(f"\nProcessing {year} data from {file_path.name}...")

    try:
        df = pd.read_csv(file_path, low_memory=False)
        original_size = len(df)
        print(f"  Loaded: {original_size:,} rows")

        # Filter to target leagues
        if 'league' not in df.columns:
            print(f"  ERROR: No 'league' column found")
            return None

        df = df[df['league'].isin(TARGET_LEAGUES)].copy()
        print(f"  After league filter: {len(df):,} rows")

        if len(df) == 0:
            print(f"  No target league data found")
            return None

        # Standardize league names
        df['league'] = df['league'].replace(LEAGUE_MAPPING)

        # Add year if missing
        if 'year' not in df.columns:
            df['year'] = year

        # Transform to team-match format
        df_transformed = transform_to_team_matches(df)

        if df_transformed is None or len(df_transformed) == 0:
            print(f"  No valid team matches created")
            return None

        print(f"  Transformed: {len(df_transformed):,} team matches")
        return df_transformed

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def extract_year_from_filename(filename):
    """Extract year from filename."""
    try:
        return int(filename.split('_')[0])
    except:
        return 0


def transform_to_team_matches(df):
    """Transform player-level data to team-match format."""

    # Determine team column
    team_col = 'teamname' if 'teamname' in df.columns else 'team'

    if team_col not in df.columns:
        print("  ERROR: No team column found")
        return None

    # Group by game and team
    grouped = df.groupby(['gameid', team_col])

    team_matches = []

    for (gameid, team), group in grouped:
        # Skip non-target leagues
        league = group.iloc[0].get('league', '')
        if league not in ['LCS', 'LEC', 'LCK', 'LPL', 'WLDs', 'MSI']:
            continue

        # Skip incomplete groups (need at least 4 players)
        if len(group) < 4:
            continue

        first_row = group.iloc[0]

        # Create team match record
        team_match = {
            'gameid': gameid,
            'league': league,
            'date': first_row.get('date'),
            'patch': first_row.get('patch', 'Unknown'),
            'split': first_row.get('split', 'Unknown'),
            'playoffs': first_row.get('playoffs', 0),
            'year': first_row.get('year', 2024),
            'game': first_row.get('game', 1),
            'team': team,
            'side': first_row.get('side', 'Blue'),
            'result': first_row.get('result', 0)
        }

        # Add bans
        for i in range(1, 6):
            ban_col = f'ban{i}'
            team_match[ban_col] = first_row.get(ban_col, 'NoBan')

        # Pivot champions to position columns
        position_mapping = {
            'top': 'top_champion',
            'jng': 'jng_champion',
            'jungle': 'jng_champion',
            'jgl': 'jng_champion',
            'mid': 'mid_champion',
            'middle': 'mid_champion',
            'bot': 'bot_champion',
            'adc': 'bot_champion',
            'carry': 'bot_champion',
            'sup': 'sup_champion',
            'support': 'sup_champion',
            'supp': 'sup_champion'
        }

        position_champion_map = {}
        for _, player in group.iterrows():
            position = player.get('position', '')
            champion = player.get('champion', '')

            if pd.notna(position) and pd.notna(champion) and position != 'team':
                mapped = position_mapping.get(str(position).lower())
                if mapped:
                    position_champion_map[mapped] = champion

        # Add champion columns
        for pos in ['top_champion', 'jng_champion', 'mid_champion', 'bot_champion', 'sup_champion']:
            team_match[pos] = position_champion_map.get(pos, 'Unknown')

        # Allow up to 2 unknown champions
        unknown_count = sum(1 for pos in ['top_champion', 'jng_champion', 'mid_champion', 'bot_champion', 'sup_champion']
                          if team_match[pos] == 'Unknown')
        if unknown_count <= 2:
            team_matches.append(team_match)

    if not team_matches:
        return None

    return pd.DataFrame(team_matches)


def clean_dataset(df):
    """Clean and validate the dataset."""
    original_size = len(df)

    # Ensure result is binary
    if 'result' in df.columns:
        df['result'] = pd.to_numeric(df['result'], errors='coerce')
        df = df[df['result'].isin([0, 1])].copy()

    # Parse dates
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df[df['date'].notna()].copy()

    # Remove duplicates
    df = df.drop_duplicates(subset=['gameid', 'team'], keep='first')

    # Filter teams with minimum games
    team_counts = df['team'].value_counts()
    teams_to_keep = team_counts[team_counts >= 5].index
    df = df[df['team'].isin(teams_to_keep)].copy()

    print(f"  Cleaning: {original_size:,} -> {len(df):,} matches")
    return df


def merge_datasets(existing_df, new_df):
    """Merge existing and new datasets, avoiding duplicates."""
    if existing_df is None:
        return new_df

    if new_df is None or len(new_df) == 0:
        return existing_df

    # Ensure date columns are consistent before merging
    if 'date' in existing_df.columns:
        existing_df['date'] = pd.to_datetime(existing_df['date'], errors='coerce')
    if 'date' in new_df.columns:
        new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')

    # Combine
    combined = pd.concat([existing_df, new_df], ignore_index=True)

    # Remove duplicates (keep existing data priority)
    before = len(combined)
    combined = combined.drop_duplicates(subset=['gameid', 'team'], keep='first')
    after = len(combined)

    # Sort by date (ensure all dates are datetime)
    if 'date' in combined.columns:
        combined['date'] = pd.to_datetime(combined['date'], errors='coerce')
        combined = combined.sort_values('date').reset_index(drop=True)

    print(f"Merged: {len(existing_df):,} existing + {len(new_df):,} new = {after:,} total ({before - after} duplicates removed)")

    return combined


def backup_existing():
    """Create backup of existing dataset."""
    if EXISTING_DATASET.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"dataset_backup_{timestamp}.csv"

        import shutil
        shutil.copy(EXISTING_DATASET, backup_path)
        print(f"Backup created: {backup_path}")


def print_summary(df):
    """Print dataset summary."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Total matches: {len(df):,}")
    print(f"Unique teams: {df['team'].nunique():,}")
    print(f"Unique games: {df['gameid'].nunique():,}")

    if 'date' in df.columns:
        dates = pd.to_datetime(df['date'])
        print(f"Date range: {dates.min().date()} to {dates.max().date()}")

    print("\nMatches by year:")
    year_counts = df['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count:,}")

    print("\nMatches by league:")
    league_counts = df['league'].value_counts()
    for league, count in league_counts.items():
        print(f"  {league}: {count:,}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("PROCESS NEW ORACLE'S ELIXIR DATA")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Check for raw files
    raw_files = get_raw_files()

    if not raw_files:
        print(f"\nNo raw data files found in {RAW_DATA_DIR}")
        print("\nPlease download Oracle's Elixir files from Google Drive:")
        print("  https://drive.google.com/drive/u/1/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH")
        print(f"\nPlace them in: {RAW_DATA_DIR}")
        print("Expected format: YYYY_LoL_esports_match_data_from_OraclesElixir.csv")
        return None

    print(f"\nFound {len(raw_files)} raw data files:")
    for f in raw_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")

    # Load existing dataset
    existing_df = load_existing_dataset()
    existing_gameids = set()
    if existing_df is not None:
        existing_gameids = set(existing_df['gameid'].unique())
        print(f"Existing unique games: {len(existing_gameids):,}")

    # Process each raw file
    new_dfs = []
    for raw_file in raw_files:
        df_processed = process_raw_file(raw_file)
        if df_processed is not None:
            # Filter to only truly new games
            new_games = ~df_processed['gameid'].isin(existing_gameids)
            df_new = df_processed[new_games].copy()

            if len(df_new) > 0:
                new_dfs.append(df_new)
                print(f"  New matches: {len(df_new):,}")
            else:
                print(f"  No new matches (all already in dataset)")

    if not new_dfs:
        print("\nNo new data to add.")
        if existing_df is not None:
            print_summary(existing_df)
        return existing_df

    # Combine new data
    print("\nCombining new data...")
    new_df = pd.concat(new_dfs, ignore_index=True)
    new_df = clean_dataset(new_df)

    print(f"Total new matches to add: {len(new_df):,}")

    # Backup existing
    backup_existing()

    # Merge with existing
    final_df = merge_datasets(existing_df, new_df)

    # Save
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_DATASET, index=False)
    print(f"\nSaved to: {OUTPUT_DATASET}")

    # Summary
    print_summary(final_df)

    # Show what was added
    if existing_df is not None:
        new_matches = len(final_df) - len(existing_df)
        print(f"\nNEW MATCHES ADDED: {new_matches:,}")

        if 'year' in new_df.columns:
            print("New matches by year:")
            for year, count in new_df['year'].value_counts().sort_index().items():
                print(f"  {year}: {count:,}")

    return final_df


if __name__ == "__main__":
    main()
