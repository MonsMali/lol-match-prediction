"""
Data Refresh Script for LoL Match Prediction System.

Downloads new data from Oracle's Elixir and runs quality checks.
Prepares data for model retraining.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.pipeline import DataPipeline, PipelineResult
from src.data.downloader import OraclesElixirDownloader
from src.config import DATASET_PATH, RAW_DATA_DIR, PROCESSED_DATA_DIR

# Import quality module
try:
    from src.data.quality import (
        DataValidator,
        create_quality_report
    )
    QUALITY_AVAILABLE = True
except ImportError:
    QUALITY_AVAILABLE = False


def check_current_status():
    """Check current data status before refresh."""
    print("=" * 60)
    print("CURRENT DATA STATUS")
    print("=" * 60)

    pipeline = DataPipeline()
    status = pipeline.get_pipeline_status()

    print(f"\nCurrent version: {status['current_version']}")
    print(f"Dataset exists: {status['dataset_exists']}")

    if status['dataset_exists']:
        print(f"Dataset size: {status['dataset_size']:.1f} MB")
        print(f"Row count: {status.get('row_count', 'N/A'):,}")
        print(f"Last update: {status['last_update']}")

    dl_status = status['download_status']
    print(f"\nRemote Data:")
    print(f"  Available files: {dl_status['available_remote']}")
    print(f"  Already downloaded: {dl_status['downloaded']}")
    print(f"  Pending download: {dl_status['pending']}")
    print(f"  Years available: {dl_status['years_available']}")

    return status


def download_new_data(full_refresh: bool = False, start_year: int = 2014):
    """Download new data from Oracle's Elixir.

    Args:
        full_refresh: If True, re-download all data
        start_year: Starting year for full refresh
    """
    print("\n" + "=" * 60)
    print("DATA DOWNLOAD")
    print("=" * 60)

    pipeline = DataPipeline()

    if full_refresh:
        print(f"\nRunning full pipeline from {start_year}...")
        result = pipeline.run_full_pipeline(start_year=start_year)
    else:
        print("\nRunning incremental update...")
        result = pipeline.run_incremental_update()

    return result


def run_quality_checks():
    """Run data quality checks on the downloaded data."""
    if not QUALITY_AVAILABLE:
        print("\nData quality module not available. Skipping quality checks.")
        return None

    if not DATASET_PATH.exists():
        print("\nNo dataset found. Skipping quality checks.")
        return None

    print("\n" + "=" * 60)
    print("DATA QUALITY CHECKS")
    print("=" * 60)

    import pandas as pd
    df = pd.read_csv(DATASET_PATH)

    # Run quality report
    report = create_quality_report(df)
    print(report)

    return report


def summarize_for_training():
    """Summarize data readiness for training."""
    if not DATASET_PATH.exists():
        print("\nNo dataset found. Please download data first.")
        return

    import pandas as pd
    df = pd.read_csv(DATASET_PATH)

    print("\n" + "=" * 60)
    print("DATA READY FOR TRAINING")
    print("=" * 60)

    print(f"\nTotal matches: {len(df):,}")

    if 'year' in df.columns:
        print(f"\nMatches by year:")
        year_counts = df['year'].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"  {year}: {count:,}")

    if 'league' in df.columns:
        print(f"\nMatches by league:")
        league_counts = df['league'].value_counts()
        for league, count in league_counts.head(10).items():
            print(f"  {league}: {count:,}")

    if 'date' in df.columns:
        dates = pd.to_datetime(df['date'])
        print(f"\nDate range: {dates.min().date()} to {dates.max().date()}")

    print("\nTo train models, run:")
    print("  python src/models/trainer.py")
    print("\nOr with options:")
    print("  python -c \"from src.models.trainer import main; main(use_enhanced_v2=True)\"")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Refresh LoL match data")
    parser.add_argument('--status', action='store_true', help='Show current status only')
    parser.add_argument('--full', action='store_true', help='Run full data refresh')
    parser.add_argument('--start-year', type=int, default=2014, help='Start year for full refresh')
    parser.add_argument('--skip-quality', action='store_true', help='Skip quality checks')

    args = parser.parse_args()

    print("LOL MATCH PREDICTION - DATA REFRESH")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Check current status
    status = check_current_status()

    if args.status:
        return

    # Download data
    result = download_new_data(
        full_refresh=args.full,
        start_year=args.start_year
    )

    # Run quality checks
    if not args.skip_quality:
        run_quality_checks()

    # Summarize for training
    summarize_for_training()


if __name__ == "__main__":
    main()
