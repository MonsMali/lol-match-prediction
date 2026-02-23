"""
Data Quality Module for LoL Match Prediction.

Provides data quality improvements including:
- Temporal weighting for training samples
- Data augmentation through match mirroring
- Outlier detection and analysis
- Data validation and integrity checks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


@dataclass
class DataQualityReport:
    """Report on data quality analysis."""
    total_samples: int
    missing_value_summary: Dict[str, int]
    outlier_count: int
    outlier_indices: List[int]
    duplicate_count: int
    temporal_coverage: Dict[str, int]
    quality_score: float


@dataclass
class AugmentationResult:
    """Result from data augmentation."""
    original_size: int
    augmented_size: int
    augmentation_ratio: float
    augmented_df: pd.DataFrame


class TemporalWeighter:
    """
    Calculate temporal weights for training samples.

    More recent matches are weighted higher to account for meta evolution
    and changing competitive landscape.
    """

    def __init__(self, decay_type: str = 'exponential', half_life_days: int = 180):
        """
        Initialize the temporal weighter.

        Args:
            decay_type: Type of decay ('exponential', 'linear', 'step')
            half_life_days: Days until weight is halved (for exponential decay)
        """
        self.decay_type = decay_type
        self.half_life_days = half_life_days
        self.decay_rate = np.log(2) / half_life_days

    def calculate_weights(self, dates: pd.Series,
                          reference_date: Optional[datetime] = None) -> np.ndarray:
        """
        Calculate sample weights based on temporal distance.

        Args:
            dates: Series of match dates
            reference_date: Reference date (default: most recent date in data)

        Returns:
            Array of sample weights (0 to 1)
        """
        dates = pd.to_datetime(dates)

        if reference_date is None:
            reference_date = dates.max()

        days_since = (reference_date - dates).dt.days.values

        if self.decay_type == 'exponential':
            weights = np.exp(-self.decay_rate * days_since)
        elif self.decay_type == 'linear':
            max_days = days_since.max()
            weights = 1 - (days_since / (max_days + 1))
        elif self.decay_type == 'step':
            # Recent (< 6 months): 1.0
            # Medium (6-12 months): 0.7
            # Old (> 12 months): 0.4
            weights = np.where(days_since < 180, 1.0,
                              np.where(days_since < 365, 0.7, 0.4))
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")

        # Normalize to [0.1, 1.0] to avoid zero weights
        weights = np.clip(weights, 0.1, 1.0)

        return weights

    def calculate_weights_by_patch(self, df: pd.DataFrame,
                                   current_patch: Optional[str] = None) -> np.ndarray:
        """
        Calculate weights based on patch distance.

        Matches on the same patch as the target get higher weight.

        Args:
            df: DataFrame with 'patch' column
            current_patch: Current patch (default: most recent patch)

        Returns:
            Array of sample weights
        """
        if current_patch is None:
            # Get most recent patch
            current_patch = df.sort_values('date').iloc[-1]['patch']

        # Extract patch version numbers
        def parse_patch(patch_str):
            try:
                parts = str(patch_str).split('.')
                major = int(parts[0])
                minor = int(parts[1]) if len(parts) > 1 else 0
                return major * 100 + minor
            except (ValueError, IndexError):
                return 0

        current_version = parse_patch(current_patch)
        patch_versions = df['patch'].apply(parse_patch)

        # Calculate patch distance
        patch_distance = np.abs(current_version - patch_versions)

        # Exponential decay based on patch distance
        # Each patch difference reduces weight by ~20%
        weights = 0.8 ** patch_distance

        # Normalize to [0.1, 1.0]
        weights = np.clip(weights, 0.1, 1.0)

        return weights.values

    def get_combined_weights(self, df: pd.DataFrame,
                            temporal_weight: float = 0.6,
                            patch_weight: float = 0.4) -> np.ndarray:
        """
        Get combined temporal and patch-based weights.

        Args:
            df: DataFrame with 'date' and 'patch' columns
            temporal_weight: Weight for temporal component
            patch_weight: Weight for patch component

        Returns:
            Combined sample weights
        """
        time_weights = self.calculate_weights(df['date'])
        patch_weights = self.calculate_weights_by_patch(df)

        combined = temporal_weight * time_weights + patch_weight * patch_weights

        # Normalize to sum to len(df) for sklearn compatibility
        combined = combined / combined.sum() * len(df)

        return combined


class DataAugmenter:
    """
    Data augmentation for match prediction.

    Provides methods to increase effective dataset size while
    maintaining data integrity.
    """

    def __init__(self):
        """Initialize the data augmenter."""
        pass

    def mirror_matches(self, df: pd.DataFrame) -> AugmentationResult:
        """
        Create mirrored versions of matches by swapping team perspectives.

        For each match, creates a complementary sample with inverted result
        (if team A wins, mirrored shows team B losing).

        Note: This should be used carefully and only on training data
        to avoid leakage.

        Args:
            df: DataFrame with match data

        Returns:
            AugmentationResult with augmented DataFrame
        """
        original_size = len(df)

        # Create mirrored dataframe
        mirrored_df = df.copy()

        # Invert result
        mirrored_df['result'] = 1 - mirrored_df['result']

        # Swap blue/red side if present
        if 'side' in mirrored_df.columns:
            mirrored_df['side'] = mirrored_df['side'].apply(
                lambda x: 'Red' if x == 'Blue' else 'Blue'
            )

        # Add augmentation flag
        df_original = df.copy()
        df_original['is_augmented'] = 0
        mirrored_df['is_augmented'] = 1

        # Combine
        augmented_df = pd.concat([df_original, mirrored_df], ignore_index=True)

        return AugmentationResult(
            original_size=original_size,
            augmented_size=len(augmented_df),
            augmentation_ratio=len(augmented_df) / original_size,
            augmented_df=augmented_df
        )

    def create_temporal_samples(self, df: pd.DataFrame,
                               window_size: int = 10,
                               step_size: int = 5) -> pd.DataFrame:
        """
        Create additional samples using sliding window aggregation.

        For each team, creates aggregated features over a sliding window
        of recent matches.

        Args:
            df: DataFrame with match data
            window_size: Number of matches in window
            step_size: Step between windows

        Returns:
            DataFrame with additional temporal samples
        """
        # Sort by team and date
        df_sorted = df.sort_values(['team', 'date']).copy()

        # Group by team
        team_groups = df_sorted.groupby('team')

        temporal_samples = []

        for team, group in team_groups:
            if len(group) < window_size:
                continue

            # Create sliding windows
            for start_idx in range(0, len(group) - window_size + 1, step_size):
                window = group.iloc[start_idx:start_idx + window_size]

                # Aggregate features (mean for numeric, mode for categorical)
                sample = {
                    'team': team,
                    'window_winrate': window['result'].mean(),
                    'window_games': len(window),
                    'window_start': window['date'].min(),
                    'window_end': window['date'].max(),
                }

                temporal_samples.append(sample)

        return pd.DataFrame(temporal_samples)


class OutlierDetector:
    """
    Detect outliers in match data.

    Identifies unusual matches that may represent data errors
    or exceptional circumstances.
    """

    def __init__(self, method: str = 'iqr', threshold: float = 1.5):
        """
        Initialize the outlier detector.

        Args:
            method: Detection method ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for outlier detection
        """
        self.method = method
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame,
                       feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect outliers in the dataset.

        Args:
            df: DataFrame with features
            feature_columns: Columns to check for outliers

        Returns:
            Dictionary with outlier information
        """
        if feature_columns is None:
            # Use numeric columns
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outlier_mask = pd.Series(False, index=df.index)

        for col in feature_columns:
            if col not in df.columns:
                continue

            values = df[col].dropna()

            if self.method == 'iqr':
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.threshold * IQR
                upper_bound = Q3 + self.threshold * IQR
                col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

            elif self.method == 'zscore':
                mean = values.mean()
                std = values.std()
                if std > 0:
                    z_scores = np.abs((df[col] - mean) / std)
                    col_outliers = z_scores > self.threshold
                else:
                    col_outliers = pd.Series(False, index=df.index)

            elif self.method == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                predictions = iso_forest.fit_predict(df[[col]].fillna(0))
                col_outliers = pd.Series(predictions == -1, index=df.index)

            else:
                raise ValueError(f"Unknown method: {self.method}")

            outlier_mask = outlier_mask | col_outliers

        return {
            'outlier_mask': outlier_mask,
            'outlier_count': outlier_mask.sum(),
            'outlier_percentage': outlier_mask.mean() * 100,
            'outlier_indices': df[outlier_mask].index.tolist()
        }

    def analyze_upsets(self, df: pd.DataFrame,
                      prediction_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze upset matches (unexpected results).

        Args:
            df: DataFrame with match data
            prediction_proba: Optional prediction probabilities

        Returns:
            Dictionary with upset analysis
        """
        upsets = []

        if prediction_proba is not None:
            # High confidence wrong predictions are upsets
            predicted = (prediction_proba > 0.5).astype(int)
            actual = df['result'].values

            # Confident predictions that were wrong
            high_conf_wrong = (
                (np.abs(prediction_proba - 0.5) > 0.3) &
                (predicted != actual)
            )

            upsets = df[high_conf_wrong].index.tolist()

        return {
            'upset_count': len(upsets),
            'upset_indices': upsets,
            'upset_percentage': len(upsets) / len(df) * 100 if len(df) > 0 else 0
        }


class DataValidator:
    """
    Validate data integrity and quality.
    """

    def __init__(self):
        """Initialize the data validator."""
        self.required_columns = [
            'date', 'team', 'result', 'patch', 'league',
            'top_champion', 'jng_champion', 'mid_champion',
            'bot_champion', 'sup_champion'
        ]

    def validate_dataset(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Perform comprehensive data validation.

        Args:
            df: DataFrame to validate

        Returns:
            DataQualityReport with validation results
        """
        # Check missing values
        missing_summary = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_summary[col] = missing_count

        # Check for duplicates
        duplicate_count = df.duplicated().sum()

        # Check temporal coverage
        if 'year' in df.columns:
            temporal_coverage = df['year'].value_counts().to_dict()
        elif 'date' in df.columns:
            df_temp = df.copy()
            df_temp['year'] = pd.to_datetime(df_temp['date']).dt.year
            temporal_coverage = df_temp['year'].value_counts().to_dict()
        else:
            temporal_coverage = {}

        # Detect outliers
        outlier_detector = OutlierDetector()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            outlier_result = outlier_detector.detect_outliers(df, numeric_cols)
            outlier_count = outlier_result['outlier_count']
            outlier_indices = outlier_result['outlier_indices']
        else:
            outlier_count = 0
            outlier_indices = []

        # Calculate quality score (0-100)
        quality_score = 100.0

        # Penalize for missing required columns
        for col in self.required_columns:
            if col not in df.columns:
                quality_score -= 5

        # Penalize for missing values (max -20)
        missing_ratio = sum(missing_summary.values()) / (len(df) * len(df.columns))
        quality_score -= min(missing_ratio * 100, 20)

        # Penalize for duplicates (max -10)
        duplicate_ratio = duplicate_count / len(df)
        quality_score -= min(duplicate_ratio * 100, 10)

        # Penalize for outliers (max -10)
        outlier_ratio = outlier_count / len(df)
        quality_score -= min(outlier_ratio * 50, 10)

        quality_score = max(quality_score, 0)

        return DataQualityReport(
            total_samples=len(df),
            missing_value_summary=missing_summary,
            outlier_count=outlier_count,
            outlier_indices=outlier_indices,
            duplicate_count=duplicate_count,
            temporal_coverage=temporal_coverage,
            quality_score=quality_score
        )

    def check_temporal_integrity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check temporal integrity of the dataset.

        Ensures no future data leakage in temporal features.

        Args:
            df: DataFrame with match data

        Returns:
            Dictionary with temporal integrity results
        """
        issues = []

        if 'date' not in df.columns:
            return {'status': 'unknown', 'issues': ['No date column found']}

        df_sorted = df.sort_values('date')

        # Check for date parsing issues
        try:
            dates = pd.to_datetime(df_sorted['date'])
        except Exception as e:
            issues.append(f"Date parsing error: {str(e)}")

        # Check for future dates
        today = datetime.now()
        future_dates = dates[dates > today]
        if len(future_dates) > 0:
            issues.append(f"Found {len(future_dates)} matches with future dates")

        # Check for suspicious date gaps
        date_diffs = dates.diff().dt.days.dropna()
        if date_diffs.max() > 365:
            issues.append("Found gaps > 1 year in match dates")

        return {
            'status': 'ok' if len(issues) == 0 else 'issues_found',
            'issues': issues,
            'date_range': (dates.min(), dates.max()),
            'total_days': (dates.max() - dates.min()).days
        }


def create_quality_report(df: pd.DataFrame) -> str:
    """
    Create a formatted quality report for the dataset.

    Args:
        df: DataFrame to analyze

    Returns:
        Formatted string report
    """
    validator = DataValidator()
    report = validator.validate_dataset(df)
    temporal = validator.check_temporal_integrity(df)

    output = []
    output.append("=" * 60)
    output.append("DATA QUALITY REPORT")
    output.append("=" * 60)
    output.append(f"\nTotal Samples: {report.total_samples}")
    output.append(f"Quality Score: {report.quality_score:.1f}/100")
    output.append(f"\nDuplicates: {report.duplicate_count}")
    output.append(f"Outliers: {report.outlier_count}")

    output.append(f"\nMissing Values:")
    if report.missing_value_summary:
        for col, count in sorted(report.missing_value_summary.items(),
                                 key=lambda x: x[1], reverse=True)[:10]:
            output.append(f"  - {col}: {count}")
    else:
        output.append("  None")

    output.append(f"\nTemporal Coverage:")
    for year, count in sorted(report.temporal_coverage.items()):
        output.append(f"  - {year}: {count} matches")

    output.append(f"\nTemporal Integrity: {temporal['status']}")
    if temporal['issues']:
        for issue in temporal['issues']:
            output.append(f"  - {issue}")

    return "\n".join(output)
