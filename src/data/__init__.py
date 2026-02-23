# Data module
"""
Data collection, processing, filtering, and pipeline utilities.

Modules:
- filter: Filter matches by target leagues
- extractor: Extract relevant features from raw data
- analyzer: Analyze Oracle's Elixir data structure
- processor: Create and process the complete dataset
- downloader: Download data from Oracle's Elixir S3
- schema: Schema validation and adaptation
- pipeline: Complete data pipeline orchestration
- quality: Data quality utilities (temporal weighting, augmentation, outlier detection)
"""

from .filter import TARGET_LEAGUES, filter_target_leagues
from .downloader import OraclesElixirDownloader
from .schema import SchemaValidator, validate_and_report
from .pipeline import DataPipeline, PipelineResult
from .quality import (
    TemporalWeighter,
    DataAugmenter,
    OutlierDetector,
    DataValidator,
    DataQualityReport,
    AugmentationResult,
    create_quality_report
)

__all__ = [
    # Filter
    'TARGET_LEAGUES',
    'filter_target_leagues',
    # Downloader
    'OraclesElixirDownloader',
    # Schema
    'SchemaValidator',
    'validate_and_report',
    # Pipeline
    'DataPipeline',
    'PipelineResult',
    # Quality
    'TemporalWeighter',
    'DataAugmenter',
    'OutlierDetector',
    'DataValidator',
    'DataQualityReport',
    'AugmentationResult',
    'create_quality_report',
]
