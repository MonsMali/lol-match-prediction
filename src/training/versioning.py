"""
Model Version Management for LoL Match Prediction System.

Handles model lifecycle including versioning, comparison, promotion,
and rollback capabilities.
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
import joblib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import MODELS_DIR, PRODUCTION_MODELS_DIR, EXPERIMENTS_DIR, ensure_dirs


@dataclass
class ModelMetadata:
    """Metadata for a model version."""
    version: str
    created_at: str
    model_type: str
    algorithm: str
    metrics: Dict[str, float]
    training_config: Dict[str, Any] = field(default_factory=dict)
    data_version: str = ""
    data_hash: str = ""
    feature_count: int = 0
    training_samples: int = 0
    validation_samples: int = 0
    test_samples: int = 0
    is_production: bool = False
    promoted_at: Optional[str] = None
    notes: str = ""


class ModelVersionManager:
    """
    Manages model versions with storage, comparison, and promotion.

    Features:
    - Version-controlled model storage
    - Metadata tracking
    - Production model management
    - Version comparison
    - Rollback capability
    """

    VERSION_PREFIX = "v"
    METADATA_FILENAME = "metadata.json"
    MODEL_FILENAME = "model.joblib"
    SCALER_FILENAME = "scaler.joblib"
    ENCODERS_FILENAME = "encoders.joblib"

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize version manager.

        Args:
            base_dir: Base directory for model storage
        """
        ensure_dirs()
        self.base_dir = base_dir or MODELS_DIR
        self.versions_dir = self.base_dir / "versions"
        self.production_dir = PRODUCTION_MODELS_DIR

        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.production_dir.mkdir(parents=True, exist_ok=True)

        # History tracking
        self.history_file = self.base_dir / "version_history.json"

    def _generate_version(self) -> str:
        """Generate a new version string."""
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        return f"{self.VERSION_PREFIX}{timestamp}"

    def _get_version_dir(self, version: str) -> Path:
        """Get directory path for a version."""
        return self.versions_dir / version

    def save_model_version(self, model: Any, metadata: Dict,
                           scaler: Any = None, encoders: Any = None,
                           version: Optional[str] = None) -> str:
        """
        Save a new model version.

        Args:
            model: Trained model object
            metadata: Dictionary with model metadata
            scaler: Optional scaler object
            encoders: Optional encoders object
            version: Optional specific version string

        Returns:
            Version string of saved model
        """
        if version is None:
            version = self._generate_version()

        version_dir = self._get_version_dir(version)
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = version_dir / self.MODEL_FILENAME
        joblib.dump(model, model_path)

        # Save scaler if provided
        if scaler is not None:
            scaler_path = version_dir / self.SCALER_FILENAME
            joblib.dump(scaler, scaler_path)

        # Save encoders if provided
        if encoders is not None:
            encoders_path = version_dir / self.ENCODERS_FILENAME
            joblib.dump(encoders, encoders_path)

        # Create and save metadata
        model_metadata = ModelMetadata(
            version=version,
            created_at=datetime.now().isoformat(),
            model_type=type(model).__name__,
            algorithm=metadata.get('algorithm', 'Unknown'),
            metrics=metadata.get('metrics', {}),
            training_config=metadata.get('training_config', {}),
            data_version=metadata.get('data_version', ''),
            data_hash=metadata.get('data_hash', ''),
            feature_count=metadata.get('feature_count', 0),
            training_samples=metadata.get('training_samples', 0),
            validation_samples=metadata.get('validation_samples', 0),
            test_samples=metadata.get('test_samples', 0),
            notes=metadata.get('notes', '')
        )

        metadata_path = version_dir / self.METADATA_FILENAME
        with open(metadata_path, 'w') as f:
            json.dump(asdict(model_metadata), f, indent=2)

        # Update history
        self._add_to_history(version, model_metadata)

        print(f"Saved model version: {version}")
        return version

    def load_model_version(self, version: str) -> Tuple[Any, ModelMetadata]:
        """
        Load a specific model version.

        Args:
            version: Version string to load

        Returns:
            Tuple of (model, metadata)
        """
        version_dir = self._get_version_dir(version)

        if not version_dir.exists():
            raise ValueError(f"Version {version} not found")

        # Load model
        model_path = version_dir / self.MODEL_FILENAME
        model = joblib.load(model_path)

        # Load metadata
        metadata_path = version_dir / self.METADATA_FILENAME
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        metadata = ModelMetadata(**metadata_dict)

        return model, metadata

    def load_version_components(self, version: str) -> Dict[str, Any]:
        """
        Load all components of a model version.

        Args:
            version: Version string

        Returns:
            Dictionary with model, scaler, encoders, metadata
        """
        version_dir = self._get_version_dir(version)

        if not version_dir.exists():
            raise ValueError(f"Version {version} not found")

        components = {}

        # Load model
        model_path = version_dir / self.MODEL_FILENAME
        components['model'] = joblib.load(model_path)

        # Load scaler if exists
        scaler_path = version_dir / self.SCALER_FILENAME
        if scaler_path.exists():
            components['scaler'] = joblib.load(scaler_path)

        # Load encoders if exists
        encoders_path = version_dir / self.ENCODERS_FILENAME
        if encoders_path.exists():
            components['encoders'] = joblib.load(encoders_path)

        # Load metadata
        metadata_path = version_dir / self.METADATA_FILENAME
        with open(metadata_path, 'r') as f:
            components['metadata'] = ModelMetadata(**json.load(f))

        return components

    def compare_versions(self, v1: str, v2: str) -> Dict:
        """
        Compare two model versions.

        Args:
            v1: First version
            v2: Second version

        Returns:
            Comparison dictionary
        """
        _, meta1 = self.load_model_version(v1)
        _, meta2 = self.load_model_version(v2)

        comparison = {
            'versions': (v1, v2),
            'metric_comparison': {},
            'config_differences': [],
            'summary': {}
        }

        # Compare metrics
        all_metrics = set(meta1.metrics.keys()) | set(meta2.metrics.keys())
        for metric in all_metrics:
            val1 = meta1.metrics.get(metric)
            val2 = meta2.metrics.get(metric)

            if val1 is not None and val2 is not None:
                diff = val2 - val1
                pct_change = (diff / val1 * 100) if val1 != 0 else 0

                comparison['metric_comparison'][metric] = {
                    v1: val1,
                    v2: val2,
                    'difference': diff,
                    'pct_change': pct_change
                }

        # Identify winner based on key metrics
        auc_comp = comparison['metric_comparison'].get('auc', {})
        f1_comp = comparison['metric_comparison'].get('f1', {})

        if auc_comp and f1_comp:
            auc_winner = v2 if auc_comp.get('difference', 0) > 0 else v1
            f1_winner = v2 if f1_comp.get('difference', 0) > 0 else v1

            comparison['summary'] = {
                'auc_winner': auc_winner,
                'f1_winner': f1_winner,
                'recommended': auc_winner if auc_winner == f1_winner else 'unclear'
            }

        return comparison

    def promote_to_production(self, version: str) -> bool:
        """
        Promote a version to production.

        Args:
            version: Version to promote

        Returns:
            True if successful
        """
        version_dir = self._get_version_dir(version)

        if not version_dir.exists():
            raise ValueError(f"Version {version} not found")

        # Backup current production if exists
        current_production = self.get_production_version()
        if current_production:
            backup_dir = self.production_dir.parent / "production_backup"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"production_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if self.production_dir.exists():
                shutil.copytree(self.production_dir, backup_path)
                print(f"Backed up current production to: {backup_path}")

        # Copy version to production
        for filename in [self.MODEL_FILENAME, self.SCALER_FILENAME,
                        self.ENCODERS_FILENAME, self.METADATA_FILENAME]:
            src = version_dir / filename
            dst = self.production_dir / filename.replace('model', 'best_model')

            if src.exists():
                shutil.copy(src, dst)

        # Update metadata to mark as production
        metadata_path = version_dir / self.METADATA_FILENAME
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        metadata['is_production'] = True
        metadata['promoted_at'] = datetime.now().isoformat()

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Also save to production dir
        with open(self.production_dir / self.METADATA_FILENAME, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Promoted {version} to production")
        return True

    def rollback_production(self, to_version: str) -> bool:
        """
        Rollback production to a previous version.

        Args:
            to_version: Version to rollback to

        Returns:
            True if successful
        """
        return self.promote_to_production(to_version)

    def get_production_version(self) -> Optional[str]:
        """Get the current production version."""
        metadata_path = self.production_dir / self.METADATA_FILENAME

        if not metadata_path.exists():
            # Check for legacy production model
            legacy_path = self.production_dir / "best_model.joblib"
            if legacy_path.exists():
                return "legacy"
            return None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return metadata.get('version')

    def list_versions(self, include_metrics: bool = True) -> List[Dict]:
        """
        List all available versions.

        Args:
            include_metrics: Include metrics in listing

        Returns:
            List of version info dictionaries
        """
        versions = []

        for version_dir in sorted(self.versions_dir.iterdir()):
            if not version_dir.is_dir():
                continue

            metadata_path = version_dir / self.METADATA_FILENAME
            if not metadata_path.exists():
                continue

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            info = {
                'version': metadata.get('version'),
                'created_at': metadata.get('created_at'),
                'algorithm': metadata.get('algorithm'),
                'is_production': metadata.get('is_production', False)
            }

            if include_metrics:
                info['metrics'] = metadata.get('metrics', {})

            versions.append(info)

        return versions

    def cleanup_old_versions(self, keep_count: int = 10) -> List[str]:
        """
        Remove old versions keeping only the most recent.

        Args:
            keep_count: Number of versions to keep

        Returns:
            List of removed version strings
        """
        versions = self.list_versions(include_metrics=False)

        # Sort by created_at, oldest first
        versions.sort(key=lambda v: v.get('created_at', ''))

        # Don't delete production or keep_count most recent
        to_delete = []

        for v in versions[:-keep_count]:
            if not v.get('is_production', False):
                to_delete.append(v['version'])

        # Delete
        removed = []
        for version in to_delete:
            version_dir = self._get_version_dir(version)
            if version_dir.exists():
                shutil.rmtree(version_dir)
                removed.append(version)
                print(f"Removed version: {version}")

        return removed

    def _add_to_history(self, version: str, metadata: ModelMetadata) -> None:
        """Add version to history file."""
        history = []

        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                history = json.load(f)

        history.append({
            'version': version,
            'created_at': metadata.created_at,
            'algorithm': metadata.algorithm,
            'metrics': metadata.metrics
        })

        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def get_version_history(self) -> List[Dict]:
        """Get complete version history."""
        if not self.history_file.exists():
            return []

        with open(self.history_file, 'r') as f:
            return json.load(f)


def print_version_comparison(comparison: Dict) -> None:
    """Print formatted version comparison."""
    v1, v2 = comparison['versions']

    print(f"\n{'='*60}")
    print(f"VERSION COMPARISON: {v1} vs {v2}")
    print(f"{'='*60}")

    print(f"\nMetric Comparison:")
    for metric, values in comparison['metric_comparison'].items():
        diff = values['difference']
        pct = values['pct_change']
        direction = "+" if diff > 0 else ""

        print(f"  {metric}:")
        print(f"    {v1}: {values[v1]:.4f}")
        print(f"    {v2}: {values[v2]:.4f}")
        print(f"    Change: {direction}{diff:.4f} ({direction}{pct:.1f}%)")

    if comparison.get('summary'):
        print(f"\nSummary:")
        print(f"  AUC Winner: {comparison['summary'].get('auc_winner')}")
        print(f"  F1 Winner: {comparison['summary'].get('f1_winner')}")
        print(f"  Recommended: {comparison['summary'].get('recommended')}")
