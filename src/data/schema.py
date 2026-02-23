"""
Schema Validation and Adaptation for LoL Match Data.

Handles schema validation, detection, and adaptation to ensure
consistent data format across different data sources and versions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SchemaInfo:
    """Information about a DataFrame schema."""
    columns: List[str]
    dtypes: Dict[str, str]
    null_counts: Dict[str, int]
    sample_values: Dict[str, any]
    row_count: int
    detected_version: str = "unknown"


@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool
    missing_columns: List[str] = field(default_factory=list)
    extra_columns: List[str] = field(default_factory=list)
    type_mismatches: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class SchemaValidator:
    """
    Schema validator for Oracle's Elixir match data.

    Validates that incoming data has required columns and proper types
    for the LoL match prediction system.
    """

    # Required columns for match prediction (pre-match features only)
    REQUIRED_COLUMNS = [
        'gameid',
        'league',
        'date',
        'teamname',
        'side',
        'result',
        'patch',
        'playoffs',
        'ban1', 'ban2', 'ban3', 'ban4', 'ban5',
    ]

    # Champion pick columns (one per position)
    PICK_COLUMNS = ['pick1', 'pick2', 'pick3', 'pick4', 'pick5']

    # Alternative column names for picks (position-based)
    POSITION_PICK_COLUMNS = {
        'top': 'pick1',
        'jng': 'pick2',
        'mid': 'pick3',
        'bot': 'pick4',
        'sup': 'pick5'
    }

    # Expected data types
    EXPECTED_DTYPES = {
        'gameid': 'object',
        'league': 'object',
        'teamname': 'object',
        'side': 'object',
        'result': ['int64', 'float64', 'int32', 'float32'],
        'playoffs': ['int64', 'float64', 'int32', 'float32', 'bool'],
    }

    # Columns that should be present for optimal feature engineering
    RECOMMENDED_COLUMNS = [
        'year',
        'split',
        'position',
        'champion',
        'gamelength',
    ]

    def __init__(self, strict: bool = False):
        """
        Initialize validator.

        Args:
            strict: If True, validation fails on warnings too
        """
        self.strict = strict

    def detect_schema(self, df: pd.DataFrame) -> SchemaInfo:
        """
        Detect the schema of a DataFrame.

        Args:
            df: DataFrame to analyze

        Returns:
            SchemaInfo object with schema details
        """
        # Get dtype as string
        dtypes = {col: str(df[col].dtype) for col in df.columns}

        # Get null counts
        null_counts = df.isnull().sum().to_dict()

        # Get sample values (first non-null value for each column)
        sample_values = {}
        for col in df.columns:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                sample_values[col] = non_null.iloc[0]
            else:
                sample_values[col] = None

        # Detect schema version based on columns present
        version = self._detect_version(df)

        return SchemaInfo(
            columns=list(df.columns),
            dtypes=dtypes,
            null_counts=null_counts,
            sample_values=sample_values,
            row_count=len(df),
            detected_version=version
        )

    def _detect_version(self, df: pd.DataFrame) -> str:
        """Detect data schema version based on available columns."""
        columns = set(df.columns)

        # Check for different Oracle's Elixir data formats
        if 'position' in columns and 'champion' in columns:
            return "player_level"  # Player-level data (one row per player)
        elif all(f'pick{i}' in columns for i in range(1, 6)):
            return "team_level_picks"  # Team-level with pick columns
        elif all(col in columns for col in ['top', 'jng', 'mid', 'bot', 'sup']):
            return "team_level_positions"  # Team-level with position columns
        elif 'teamname' in columns and 'result' in columns:
            return "team_level_basic"  # Basic team-level
        else:
            return "unknown"

    def validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate that DataFrame has required schema.

        Args:
            df: DataFrame to validate

        Returns:
            ValidationResult with validation details
        """
        columns = set(df.columns)
        result = ValidationResult(is_valid=True)

        # Check required columns
        required = set(self.REQUIRED_COLUMNS)
        missing = required - columns
        if missing:
            # Check if pick columns are present in alternative form
            pick_missing = set(self.PICK_COLUMNS) & missing
            if pick_missing:
                # Check for position-based columns
                has_position_cols = all(
                    col in columns for col in self.POSITION_PICK_COLUMNS.keys()
                )
                if has_position_cols:
                    missing = missing - pick_missing
                    result.warnings.append(
                        "Using position-based pick columns instead of pick1-5"
                    )

            if missing:
                result.missing_columns = list(missing)
                result.is_valid = False

        # Check for extra columns (informational)
        expected = set(self.REQUIRED_COLUMNS + self.PICK_COLUMNS + self.RECOMMENDED_COLUMNS)
        extra = columns - expected
        if extra:
            # Filter out known in-game stats (not a problem)
            known_extras = {'kills', 'deaths', 'assists', 'gold', 'cs',
                           'dragons', 'barons', 'towers', 'inhibitors'}
            extra = extra - known_extras
            if extra:
                result.extra_columns = list(extra)[:10]  # Limit to 10

        # Check data types
        for col, expected_types in self.EXPECTED_DTYPES.items():
            if col in columns:
                actual_type = str(df[col].dtype)
                if isinstance(expected_types, list):
                    if actual_type not in expected_types:
                        result.type_mismatches[col] = (actual_type, str(expected_types))
                else:
                    if actual_type != expected_types:
                        result.type_mismatches[col] = (actual_type, expected_types)

        if result.type_mismatches:
            result.warnings.append(f"Type mismatches found: {result.type_mismatches}")

        # Check recommended columns
        recommended = set(self.RECOMMENDED_COLUMNS)
        missing_recommended = recommended - columns
        if missing_recommended:
            result.warnings.append(
                f"Missing recommended columns: {missing_recommended}"
            )

        # Check for data quality issues
        if 'result' in columns:
            unique_results = df['result'].dropna().unique()
            if not all(r in [0, 1, 0.0, 1.0] for r in unique_results):
                result.warnings.append(
                    f"Unexpected result values: {unique_results}"
                )

        if self.strict and result.warnings:
            result.is_valid = False

        return result

    def adapt_schema(self, df: pd.DataFrame,
                     target_schema: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Adapt DataFrame to target schema.

        Handles column renaming, type conversion, and missing column creation.

        Args:
            df: DataFrame to adapt
            target_schema: Target column list. If None, uses REQUIRED_COLUMNS.

        Returns:
            Adapted DataFrame
        """
        df = df.copy()

        # Detect current schema
        schema_info = self.detect_schema(df)

        # Handle position-based pick columns
        if schema_info.detected_version == "team_level_positions":
            for pos_col, pick_col in self.POSITION_PICK_COLUMNS.items():
                if pos_col in df.columns and pick_col not in df.columns:
                    df[pick_col] = df[pos_col]

        # Handle player-level data (needs aggregation to team level)
        if schema_info.detected_version == "player_level":
            df = self._aggregate_to_team_level(df)

        # Ensure required columns exist
        if target_schema is None:
            target_schema = self.REQUIRED_COLUMNS + self.PICK_COLUMNS

        for col in target_schema:
            if col not in df.columns:
                # Create with appropriate default
                if col.startswith('ban') or col.startswith('pick'):
                    df[col] = None
                elif col == 'playoffs':
                    df[col] = 0
                elif col == 'year':
                    if 'date' in df.columns:
                        df['year'] = pd.to_datetime(df['date']).dt.year
                    else:
                        df['year'] = None
                else:
                    df[col] = None

        # Convert data types
        if 'result' in df.columns:
            df['result'] = pd.to_numeric(df['result'], errors='coerce').fillna(0).astype(int)

        if 'playoffs' in df.columns:
            df['playoffs'] = pd.to_numeric(df['playoffs'], errors='coerce').fillna(0).astype(int)

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        return df

    def _aggregate_to_team_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate player-level data to team level.

        Args:
            df: Player-level DataFrame (one row per player per game)

        Returns:
            Team-level DataFrame (one row per team per game)
        """
        # Group by game and team
        if 'position' not in df.columns:
            raise ValueError("Cannot aggregate: 'position' column required")

        # Position to pick mapping
        position_map = {
            'top': 'pick1',
            'jng': 'pick2',
            'jungle': 'pick2',
            'mid': 'pick3',
            'bot': 'pick4',
            'adc': 'pick4',
            'sup': 'pick5',
            'support': 'pick5'
        }

        # Get team-level columns (same for all players on team)
        team_cols = ['gameid', 'league', 'date', 'teamname', 'side', 'result',
                     'patch', 'playoffs', 'ban1', 'ban2', 'ban3', 'ban4', 'ban5']
        team_cols = [c for c in team_cols if c in df.columns]

        # Aggregate
        team_data = []

        for (gameid, teamname), group in df.groupby(['gameid', 'teamname']):
            row = {}

            # Copy team-level columns from first row
            for col in team_cols:
                row[col] = group[col].iloc[0]

            # Extract champion picks by position
            for _, player_row in group.iterrows():
                pos = str(player_row.get('position', '')).lower()
                if pos in position_map:
                    pick_col = position_map[pos]
                    row[pick_col] = player_row.get('champion')

            team_data.append(row)

        return pd.DataFrame(team_data)

    def compare_schemas(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
        """
        Compare schemas of two DataFrames.

        Args:
            df1: First DataFrame
            df2: Second DataFrame

        Returns:
            Dictionary with comparison results
        """
        schema1 = self.detect_schema(df1)
        schema2 = self.detect_schema(df2)

        cols1 = set(schema1.columns)
        cols2 = set(schema2.columns)

        return {
            'only_in_first': list(cols1 - cols2),
            'only_in_second': list(cols2 - cols1),
            'common': list(cols1 & cols2),
            'type_differences': {
                col: (schema1.dtypes.get(col), schema2.dtypes.get(col))
                for col in cols1 & cols2
                if schema1.dtypes.get(col) != schema2.dtypes.get(col)
            },
            'version_first': schema1.detected_version,
            'version_second': schema2.detected_version
        }


def validate_and_report(df: pd.DataFrame) -> bool:
    """
    Convenience function to validate data and print report.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid, False otherwise
    """
    validator = SchemaValidator()
    schema = validator.detect_schema(df)
    result = validator.validate_schema(df)

    print(f"\nSchema Validation Report")
    print(f"{'='*50}")
    print(f"Detected schema version: {schema.detected_version}")
    print(f"Row count: {schema.row_count}")
    print(f"Column count: {len(schema.columns)}")

    if result.is_valid:
        print(f"\nStatus: VALID")
    else:
        print(f"\nStatus: INVALID")

    if result.missing_columns:
        print(f"\nMissing required columns:")
        for col in result.missing_columns:
            print(f"  - {col}")

    if result.warnings:
        print(f"\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    return result.is_valid
