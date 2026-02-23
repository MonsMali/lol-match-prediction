"""
Oracle's Elixir Data Downloader.

Automated download of professional LoL match data from Oracle's Elixir S3 bucket.
Supports incremental updates and file discovery.
"""

import os
import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import requests
from xml.etree import ElementTree

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import RAW_DATA_DIR, ensure_dirs


@dataclass
class FileInfo:
    """Information about a downloadable file."""
    key: str
    url: str
    year: int
    size: int
    last_modified: datetime
    etag: str


class OraclesElixirDownloader:
    """
    Download manager for Oracle's Elixir match data.

    Oracle's Elixir provides professional LoL match data via S3 bucket.
    This class handles discovery and download of data files.
    """

    S3_BASE = "https://oracleselixir-downloadable-match-data.s3-us-west-2.amazonaws.com"
    S3_LIST_URL = f"{S3_BASE}?list-type=2"

    # File patterns for match data
    MATCH_DATA_PATTERN = re.compile(r'(\d{4}).*match.*data.*\.csv', re.IGNORECASE)

    def __init__(self, download_dir: Optional[Path] = None):
        """
        Initialize the downloader.

        Args:
            download_dir: Directory to save downloaded files.
                         Defaults to data/raw/
        """
        self.download_dir = download_dir or RAW_DATA_DIR
        ensure_dirs()
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Track downloaded files
        self.manifest_path = self.download_dir / "download_manifest.txt"
        self.downloaded_files = self._load_manifest()

    def _load_manifest(self) -> Dict[str, str]:
        """Load manifest of previously downloaded files with their ETags."""
        manifest = {}
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '|' in line:
                        parts = line.split('|')
                        if len(parts) >= 2:
                            manifest[parts[0]] = parts[1]
        return manifest

    def _save_manifest(self) -> None:
        """Save manifest of downloaded files."""
        with open(self.manifest_path, 'w') as f:
            for key, etag in self.downloaded_files.items():
                f.write(f"{key}|{etag}\n")

    def discover_available_files(self, year: Optional[int] = None) -> List[FileInfo]:
        """
        Discover available data files from S3 bucket.

        Args:
            year: Optional year to filter files. If None, returns all years.

        Returns:
            List of FileInfo objects for available files
        """
        files = []
        continuation_token = None

        print(f"Discovering files from Oracle's Elixir S3...")

        while True:
            # Build request URL
            url = self.S3_LIST_URL
            if continuation_token:
                url += f"&continuation-token={continuation_token}"

            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"Error accessing S3 bucket: {e}")
                return files

            # Parse XML response
            root = ElementTree.fromstring(response.content)
            namespace = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}

            # Extract file information
            for content in root.findall('.//s3:Contents', namespace):
                key = content.find('s3:Key', namespace)
                size = content.find('s3:Size', namespace)
                last_modified = content.find('s3:LastModified', namespace)
                etag = content.find('s3:ETag', namespace)

                if key is None or size is None:
                    continue

                key_text = key.text
                if not key_text.endswith('.csv'):
                    continue

                # Check if it matches match data pattern
                match = self.MATCH_DATA_PATTERN.search(key_text)
                if match:
                    file_year = int(match.group(1))

                    # Filter by year if specified
                    if year is not None and file_year != year:
                        continue

                    file_info = FileInfo(
                        key=key_text,
                        url=f"{self.S3_BASE}/{key_text}",
                        year=file_year,
                        size=int(size.text) if size.text else 0,
                        last_modified=datetime.fromisoformat(
                            last_modified.text.replace('Z', '+00:00')
                        ) if last_modified is not None and last_modified.text else datetime.now(),
                        etag=etag.text.strip('"') if etag is not None and etag.text else ""
                    )
                    files.append(file_info)

            # Check for pagination
            is_truncated = root.find('.//s3:IsTruncated', namespace)
            if is_truncated is not None and is_truncated.text == 'true':
                next_token = root.find('.//s3:NextContinuationToken', namespace)
                if next_token is not None:
                    continuation_token = next_token.text
                else:
                    break
            else:
                break

        # Sort by year and then by last modified
        files.sort(key=lambda f: (f.year, f.last_modified))

        print(f"Found {len(files)} data files")
        return files

    def download_file(self, file_info: FileInfo, force: bool = False) -> Optional[Path]:
        """
        Download a single file from S3.

        Args:
            file_info: FileInfo object describing the file to download
            force: If True, download even if file already exists with same ETag

        Returns:
            Path to downloaded file, or None if download failed
        """
        filename = os.path.basename(file_info.key)
        local_path = self.download_dir / filename

        # Check if already downloaded with same ETag
        if not force and file_info.key in self.downloaded_files:
            if self.downloaded_files[file_info.key] == file_info.etag:
                if local_path.exists():
                    print(f"  Skipping {filename} (already downloaded, unchanged)")
                    return local_path

        print(f"  Downloading {filename} ({file_info.size / 1024 / 1024:.1f} MB)...")

        try:
            response = requests.get(file_info.url, stream=True, timeout=300)
            response.raise_for_status()

            # Download with progress
            with open(local_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            # Update manifest
            self.downloaded_files[file_info.key] = file_info.etag
            self._save_manifest()

            print(f"  Downloaded: {filename}")
            return local_path

        except requests.RequestException as e:
            print(f"  Error downloading {filename}: {e}")
            return None

    def download_year(self, year: int, force: bool = False) -> List[Path]:
        """
        Download all match data files for a specific year.

        Args:
            year: Year to download
            force: If True, re-download even if files exist

        Returns:
            List of paths to downloaded files
        """
        print(f"\nDownloading data for {year}...")
        files = self.discover_available_files(year)

        if not files:
            print(f"No files found for {year}")
            return []

        downloaded = []
        for file_info in files:
            path = self.download_file(file_info, force)
            if path:
                downloaded.append(path)

        return downloaded

    def download_incremental(self, since_date: Optional[datetime] = None) -> List[Path]:
        """
        Download files modified since a given date.

        Args:
            since_date: Only download files modified after this date.
                       If None, downloads files not in manifest.

        Returns:
            List of paths to downloaded files
        """
        print(f"\nIncremental download...")
        files = self.discover_available_files()

        if since_date:
            files = [f for f in files if f.last_modified > since_date]
            print(f"Found {len(files)} files modified since {since_date}")
        else:
            # Download only new or updated files
            files = [f for f in files
                     if f.key not in self.downloaded_files
                     or self.downloaded_files.get(f.key) != f.etag]
            print(f"Found {len(files)} new or updated files")

        downloaded = []
        for file_info in files:
            path = self.download_file(file_info)
            if path:
                downloaded.append(path)

        return downloaded

    def download_all(self, start_year: int = 2014, end_year: Optional[int] = None,
                     force: bool = False) -> List[Path]:
        """
        Download all match data files in year range.

        Args:
            start_year: First year to download
            end_year: Last year to download (defaults to current year)
            force: If True, re-download all files

        Returns:
            List of paths to downloaded files
        """
        if end_year is None:
            end_year = datetime.now().year

        print(f"\nDownloading all data from {start_year} to {end_year}...")

        downloaded = []
        for year in range(start_year, end_year + 1):
            year_files = self.download_year(year, force)
            downloaded.extend(year_files)

        print(f"\nTotal: {len(downloaded)} files downloaded")
        return downloaded

    def get_download_status(self) -> Dict[str, any]:
        """
        Get status of downloaded files.

        Returns:
            Dictionary with download statistics
        """
        local_files = list(self.download_dir.glob("*.csv"))
        available_files = self.discover_available_files()

        available_keys = {f.key for f in available_files}
        downloaded_keys = set(self.downloaded_files.keys())

        return {
            'local_files': len(local_files),
            'available_remote': len(available_files),
            'downloaded': len(downloaded_keys),
            'pending': len(available_keys - downloaded_keys),
            'total_size_mb': sum(f.size for f in available_files) / 1024 / 1024,
            'years_available': sorted(set(f.year for f in available_files))
        }

    def list_local_files(self) -> List[Tuple[str, int, datetime]]:
        """
        List locally downloaded files.

        Returns:
            List of tuples (filename, size_bytes, modified_time)
        """
        files = []
        for path in self.download_dir.glob("*.csv"):
            stat = path.stat()
            files.append((
                path.name,
                stat.st_size,
                datetime.fromtimestamp(stat.st_mtime)
            ))
        return sorted(files, key=lambda x: x[2], reverse=True)


def main():
    """CLI interface for data downloader."""
    import argparse

    parser = argparse.ArgumentParser(description="Download Oracle's Elixir match data")
    parser.add_argument('--year', type=int, help='Download specific year')
    parser.add_argument('--all', action='store_true', help='Download all available data')
    parser.add_argument('--incremental', action='store_true', help='Incremental update')
    parser.add_argument('--status', action='store_true', help='Show download status')
    parser.add_argument('--force', action='store_true', help='Force re-download')

    args = parser.parse_args()

    downloader = OraclesElixirDownloader()

    if args.status:
        status = downloader.get_download_status()
        print("\nDownload Status:")
        print(f"  Local files: {status['local_files']}")
        print(f"  Available remote: {status['available_remote']}")
        print(f"  Downloaded: {status['downloaded']}")
        print(f"  Pending: {status['pending']}")
        print(f"  Total size: {status['total_size_mb']:.1f} MB")
        print(f"  Years available: {status['years_available']}")
    elif args.year:
        downloader.download_year(args.year, force=args.force)
    elif args.all:
        downloader.download_all(force=args.force)
    elif args.incremental:
        downloader.download_incremental()
    else:
        # Default: show status and list available files
        files = downloader.discover_available_files()
        print(f"\nAvailable files ({len(files)}):")
        for f in files:
            print(f"  {f.key} ({f.year}) - {f.size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
