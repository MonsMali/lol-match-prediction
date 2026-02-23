"""
Data Pipeline for LoL Match Prediction System.

Orchestrates data download, validation, processing, and versioning
for the complete data workflow.
"""

import os
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_PATH, ensure_dirs
)
from src.data.downloader import OraclesElixirDownloader
from src.data.schema import SchemaValidator, ValidationResult
from src.data.filter import TARGET_LEAGUES


@dataclass
class PipelineResult:
    """Result of a pipeline run."""
    success: bool
    new_matches: int = 0
    total_matches: int = 0
    data_version: str = ""
    output_path: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0


class DataPipeline:
    """
    Complete data pipeline for LoL match prediction.

    Handles:
    - Data download from Oracle's Elixir
    - Schema validation and adaptation
    - League filtering
    - Dataset merging and deduplication
    - Data versioning
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize pipeline.

        Args:
            output_dir: Directory for processed data output
        """
        ensure_dirs()
        self.output_dir = output_dir or PROCESSED_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.downloader = OraclesElixirDownloader()
        self.validator = SchemaValidator()

        # Version tracking
        self.versions_dir = self.output_dir / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def run_full_pipeline(self, start_year: int = 2014,
                          end_year: Optional[int] = None) -> PipelineResult:
        """
        Run complete data pipeline from scratch.

        Args:
            start_year: First year to include
            end_year: Last year to include (defaults to current year)

        Returns:
            PipelineResult with pipeline outcome
        """
        start_time = datetime.now()

        if end_year is None:
            end_year = datetime.now().year

        print(f"\n{'='*60}")
        print("FULL DATA PIPELINE")
        print(f"{'='*60}")
        print(f"Processing years: {start_year} - {end_year}")

        result = PipelineResult(success=False)

        try:
            # Step 1: Download data
            print(f"\n[1/5] Downloading data...")
            downloaded_files = self.downloader.download_all(start_year, end_year)
            if not downloaded_files:
                result.errors.append("No files downloaded")
                return result

            print(f"  Downloaded {len(downloaded_files)} files")

            # Step 2: Load and validate data
            print(f"\n[2/5] Loading and validating data...")
            all_data = []
            for file_path in downloaded_files:
                df = self._load_and_validate_file(file_path)
                if df is not None:
                    all_data.append(df)
                else:
                    result.warnings.append(f"Skipped invalid file: {file_path.name}")

            if not all_data:
                result.errors.append("No valid data files found")
                return result

            # Step 3: Merge datasets
            print(f"\n[3/5] Merging datasets...")
            merged_df = pd.concat(all_data, ignore_index=True)
            print(f"  Merged: {len(merged_df)} rows")

            # Step 4: Filter and deduplicate
            print(f"\n[4/5] Filtering leagues and deduplicating...")
            processed_df = self._filter_and_deduplicate(merged_df)
            print(f"  After filtering: {len(processed_df)} matches")

            # Step 5: Save and version
            print(f"\n[5/5] Saving dataset...")
            version_tag = self._create_data_version(processed_df)
            output_path = self._save_dataset(processed_df, version_tag)

            # Calculate results
            result.success = True
            result.total_matches = len(processed_df)
            result.new_matches = len(processed_df)  # Full pipeline = all new
            result.data_version = version_tag
            result.output_path = str(output_path)

        except Exception as e:
            result.errors.append(f"Pipeline error: {str(e)}")

        result.processing_time_seconds = (datetime.now() - start_time).total_seconds()

        self._print_result(result)
        return result

    def run_incremental_update(self) -> PipelineResult:
        """
        Run incremental data update.

        Downloads new/updated files and merges with existing dataset.

        Returns:
            PipelineResult with update outcome
        """
        start_time = datetime.now()

        print(f"\n{'='*60}")
        print("INCREMENTAL DATA UPDATE")
        print(f"{'='*60}")

        result = PipelineResult(success=False)

        try:
            # Load existing dataset
            existing_df = None
            existing_count = 0
            if DATASET_PATH.exists():
                print(f"Loading existing dataset...")
                existing_df = pd.read_csv(DATASET_PATH)
                existing_count = len(existing_df)
                print(f"  Existing matches: {existing_count}")

            # Download incremental updates
            print(f"\nChecking for updates...")
            new_files = self.downloader.download_incremental()

            if not new_files:
                print("  No new files to process")
                result.success = True
                result.total_matches = existing_count
                result.new_matches = 0
                result.data_version = self._get_current_version()
                result.output_path = str(DATASET_PATH)
                return result

            print(f"  Found {len(new_files)} new/updated files")

            # Process new files
            new_data = []
            for file_path in new_files:
                df = self._load_and_validate_file(file_path)
                if df is not None:
                    new_data.append(df)

            if not new_data:
                result.warnings.append("No valid data in new files")
                result.success = True
                result.total_matches = existing_count
                return result

            # Merge with existing
            new_df = pd.concat(new_data, ignore_index=True)
            new_df = self._filter_and_deduplicate(new_df)

            if existing_df is not None:
                merged_df = self.merge_datasets(existing_df, new_df)
            else:
                merged_df = new_df

            # Save updated dataset
            version_tag = self._create_data_version(merged_df)
            output_path = self._save_dataset(merged_df, version_tag)

            result.success = True
            result.total_matches = len(merged_df)
            result.new_matches = len(merged_df) - existing_count
            result.data_version = version_tag
            result.output_path = str(output_path)

        except Exception as e:
            result.errors.append(f"Update error: {str(e)}")

        result.processing_time_seconds = (datetime.now() - start_time).total_seconds()

        self._print_result(result)
        return result

    def merge_datasets(self, existing: pd.DataFrame,
                       new: pd.DataFrame) -> pd.DataFrame:
        """
        Merge existing and new datasets with deduplication.

        Args:
            existing: Existing dataset
            new: New data to merge

        Returns:
            Merged and deduplicated DataFrame
        """
        # Combine datasets
        merged = pd.concat([existing, new], ignore_index=True)

        # Deduplicate by gameid and team
        if 'gameid' in merged.columns and 'teamname' in merged.columns:
            # Keep the most recent entry (last occurrence)
            merged = merged.drop_duplicates(
                subset=['gameid', 'teamname'],
                keep='last'
            )

        # Sort by date
        if 'date' in merged.columns:
            merged = merged.sort_values('date').reset_index(drop=True)

        return merged

    def _load_and_validate_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load and validate a single data file."""
        try:
            df = pd.read_csv(file_path, low_memory=False)

            # Validate schema
            validation = self.validator.validate_schema(df)

            if not validation.is_valid:
                print(f"  Warning: {file_path.name} has schema issues")
                # Try to adapt schema
                df = self.validator.adapt_schema(df)
                validation = self.validator.validate_schema(df)

                if not validation.is_valid:
                    print(f"  Error: Could not adapt schema for {file_path.name}")
                    return None

            return df

        except Exception as e:
            print(f"  Error loading {file_path.name}: {e}")
            return None

    def _filter_and_deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to target leagues and deduplicate."""
        # Filter to target leagues
        if 'league' in df.columns:
            original_count = len(df)
            df = df[df['league'].isin(TARGET_LEAGUES)]
            filtered_count = len(df)
            print(f"  Filtered from {original_count} to {filtered_count} rows")

        # Deduplicate
        if 'gameid' in df.columns and 'teamname' in df.columns:
            before_dedup = len(df)
            df = df.drop_duplicates(subset=['gameid', 'teamname'])
            after_dedup = len(df)
            if before_dedup != after_dedup:
                print(f"  Removed {before_dedup - after_dedup} duplicate rows")

        # Sort by date
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)

        return df

    def _create_data_version(self, df: pd.DataFrame) -> str:
        """
        Create version tag for dataset.

        Version format: v{YYYY}.{MM}.{DD}_{hash}
        """
        # Create hash from data characteristics
        hash_input = f"{len(df)}_{df['gameid'].nunique() if 'gameid' in df.columns else 0}"
        if 'date' in df.columns:
            hash_input += f"_{df['date'].max()}"
        data_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]

        # Create version tag
        now = datetime.now()
        version = f"v{now.year}.{now.month:02d}.{now.day:02d}_{data_hash}"

        return version

    def _save_dataset(self, df: pd.DataFrame, version_tag: str) -> Path:
        """Save dataset with versioning."""
        # Save main dataset
        main_path = DATASET_PATH
        df.to_csv(main_path, index=False)
        print(f"  Saved: {main_path}")

        # Save versioned copy
        version_path = self.versions_dir / f"dataset_{version_tag}.csv"
        df.to_csv(version_path, index=False)
        print(f"  Versioned: {version_path}")

        # Update version info
        self._update_version_info(version_tag, len(df))

        return main_path

    def _update_version_info(self, version_tag: str, row_count: int) -> None:
        """Update version tracking file."""
        info_path = self.versions_dir / "version_history.txt"

        entry = f"{datetime.now().isoformat()}|{version_tag}|{row_count}\n"

        with open(info_path, 'a') as f:
            f.write(entry)

    def _get_current_version(self) -> str:
        """Get current dataset version."""
        info_path = self.versions_dir / "version_history.txt"

        if info_path.exists():
            with open(info_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    parts = last_line.split('|')
                    if len(parts) >= 2:
                        return parts[1]

        return "unknown"

    def _print_result(self, result: PipelineResult) -> None:
        """Print pipeline result summary."""
        print(f"\n{'='*60}")
        print("PIPELINE RESULT")
        print(f"{'='*60}")
        print(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Total matches: {result.total_matches:,}")
        print(f"New matches: {result.new_matches:,}")
        print(f"Data version: {result.data_version}")
        print(f"Output: {result.output_path}")
        print(f"Processing time: {result.processing_time_seconds:.1f}s")

        if result.errors:
            print(f"\nErrors:")
            for error in result.errors:
                print(f"  - {error}")

        if result.warnings:
            print(f"\nWarnings:")
            for warning in result.warnings[:5]:  # Limit warnings shown
                print(f"  - {warning}")
            if len(result.warnings) > 5:
                print(f"  ... and {len(result.warnings) - 5} more")

    def get_pipeline_status(self) -> Dict:
        """
        Get current pipeline status.

        Returns:
            Dictionary with status information
        """
        status = {
            'current_version': self._get_current_version(),
            'dataset_exists': DATASET_PATH.exists(),
            'dataset_size': 0,
            'last_update': None,
            'download_status': self.downloader.get_download_status()
        }

        if DATASET_PATH.exists():
            status['dataset_size'] = DATASET_PATH.stat().st_size / 1024 / 1024  # MB
            df = pd.read_csv(DATASET_PATH, nrows=0)  # Just get columns
            status['column_count'] = len(df.columns)

            # Get row count efficiently
            with open(DATASET_PATH, 'r') as f:
                status['row_count'] = sum(1 for _ in f) - 1  # Subtract header

            # Last modification time
            status['last_update'] = datetime.fromtimestamp(
                DATASET_PATH.stat().st_mtime
            ).isoformat()

        return status

    def list_versions(self) -> List[Dict]:
        """List all dataset versions."""
        versions = []
        info_path = self.versions_dir / "version_history.txt"

        if info_path.exists():
            with open(info_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 3:
                        versions.append({
                            'timestamp': parts[0],
                            'version': parts[1],
                            'row_count': int(parts[2])
                        })

        return versions

    def restore_version(self, version_tag: str) -> bool:
        """
        Restore a previous dataset version.

        Args:
            version_tag: Version to restore

        Returns:
            True if successful
        """
        version_path = self.versions_dir / f"dataset_{version_tag}.csv"

        if not version_path.exists():
            print(f"Version {version_tag} not found")
            return False

        # Backup current
        if DATASET_PATH.exists():
            backup_path = self.versions_dir / f"dataset_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            shutil.copy(DATASET_PATH, backup_path)
            print(f"Current dataset backed up to: {backup_path}")

        # Restore version
        shutil.copy(version_path, DATASET_PATH)
        print(f"Restored version: {version_tag}")

        return True


def main():
    """CLI interface for data pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="LoL Match Data Pipeline")
    parser.add_argument('--full', action='store_true', help='Run full pipeline')
    parser.add_argument('--update', action='store_true', help='Run incremental update')
    parser.add_argument('--status', action='store_true', help='Show pipeline status')
    parser.add_argument('--versions', action='store_true', help='List dataset versions')
    parser.add_argument('--restore', type=str, help='Restore specific version')
    parser.add_argument('--start-year', type=int, default=2014, help='Start year for full pipeline')

    args = parser.parse_args()

    pipeline = DataPipeline()

    if args.full:
        pipeline.run_full_pipeline(start_year=args.start_year)
    elif args.update:
        pipeline.run_incremental_update()
    elif args.status:
        status = pipeline.get_pipeline_status()
        print("\nPipeline Status:")
        print(f"  Current version: {status['current_version']}")
        print(f"  Dataset exists: {status['dataset_exists']}")
        if status['dataset_exists']:
            print(f"  Dataset size: {status['dataset_size']:.1f} MB")
            print(f"  Row count: {status.get('row_count', 'N/A'):,}")
            print(f"  Last update: {status['last_update']}")
        print(f"\nDownload Status:")
        dl_status = status['download_status']
        print(f"  Available files: {dl_status['available_remote']}")
        print(f"  Downloaded: {dl_status['downloaded']}")
        print(f"  Pending: {dl_status['pending']}")
    elif args.versions:
        versions = pipeline.list_versions()
        print("\nDataset Versions:")
        for v in versions[-10:]:  # Show last 10
            print(f"  {v['version']} - {v['row_count']:,} rows ({v['timestamp'][:10]})")
    elif args.restore:
        pipeline.restore_version(args.restore)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
