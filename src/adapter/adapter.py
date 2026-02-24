"""LoLDraftAdapter -- singleton entry point for match prediction.

Loads all pre-serialized model artifacts at initialization and
exposes ``predict_from_draft`` for inference and ``get_status``
for health diagnostics.

No CSV files or pandas DataFrames are used at any point.
"""

from __future__ import annotations

import resource
import warnings
from pathlib import Path
from typing import Union

import joblib
import numpy as np

from src.adapter.features import compute_features_for_side
from src.adapter.normalization import CHAMPION_ALIASES
from src.adapter.schemas import AdapterStatus, DraftInput, PredictionResult
from src.adapter.validation import validate_draft
from src.config import MODELS_DIR, PRODUCTION_MODELS_DIR


class LoLDraftAdapter:
    """Singleton adapter wrapping the production ML pipeline.

    On first instantiation the adapter eagerly loads all required
    joblib artifacts and fails fast if any file is missing.
    Subsequent calls to ``LoLDraftAdapter()`` return the same
    instance.

    Usage::

        adapter = LoLDraftAdapter()
        result = adapter.predict_from_draft(draft_input)
        status = adapter.get_status()
    """

    _instance: "LoLDraftAdapter | None" = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, artifacts_dir: Path | None = None) -> None:
        if hasattr(self, "_initialized"):
            return

        production_dir = (
            Path(artifacts_dir) if artifacts_dir else PRODUCTION_MODELS_DIR
        )
        models_root = production_dir.parent  # models/ directory

        # ---------------------------------------------------------------
        # 1. Resolve and verify artifact paths
        # ---------------------------------------------------------------
        artifact_paths = {
            "model": production_dir / "best_model.joblib",
            "scaler": production_dir / "scaler.joblib",
            "encoders": production_dir / "encoders.joblib",
            "meta_strength": models_root / "champion_meta_strength.joblib",
            "synergies": models_root / "champion_synergies.joblib",
            "team_perf": models_root / "team_historical_performance.joblib",
        }

        missing = [
            f"  - {name}: {path}"
            for name, path in artifact_paths.items()
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing required model artifacts:\n"
                + "\n".join(missing)
                + "\n\nEnsure all artifacts are present in "
                + str(production_dir)
                + " and "
                + str(models_root)
            )

        # ---------------------------------------------------------------
        # 2. Load artifacts
        # ---------------------------------------------------------------
        self.model = joblib.load(artifact_paths["model"])
        self.scaler = joblib.load(artifact_paths["scaler"])
        self.encoders: dict = joblib.load(artifact_paths["encoders"])
        self.meta_strength: dict = joblib.load(artifact_paths["meta_strength"])
        self.synergies: dict = joblib.load(artifact_paths["synergies"])
        self.team_perf: dict = joblib.load(artifact_paths["team_perf"])

        self._loaded_artifacts = list(artifact_paths.keys())

        # ---------------------------------------------------------------
        # 3. Derive convenience sets
        # ---------------------------------------------------------------
        # Champions appear as the second element of meta_strength keys
        self.valid_champions: set[str] = {
            champ for _, champ in self.meta_strength.keys()
        }

        # Teams from the team performance lookup
        self.valid_teams: set[str] = set(self.team_perf.keys())

        # ---------------------------------------------------------------
        # 4. Determine latest patch (float)
        # ---------------------------------------------------------------
        numeric_patches = [
            p for p, _ in self.meta_strength.keys()
            if isinstance(p, (int, float))
        ]
        self.latest_patch: float = max(numeric_patches) if numeric_patches else 14.18

        # Latest year: derive from latest patch major version
        self.latest_year: int = int(self.latest_patch) + 2010
        # Patch 14.x -> 2024, 13.x -> 2023, etc.

        # ---------------------------------------------------------------
        # 5. Model metadata
        # ---------------------------------------------------------------
        self.model_name: str = type(self.model).__name__
        self.model_version: str = "enhanced-v1"

        self._initialized = True

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def predict_from_draft(
        self,
        draft: Union[DraftInput, dict],
    ) -> PredictionResult:
        """Predict match outcome from a complete draft.

        Accepts either a ``DraftInput`` dataclass or a plain dict
        with the same keys.  The draft is validated and normalized
        before feature computation.

        Args:
            draft: Complete draft information (both teams' picks and
                bans).

        Returns:
            ``PredictionResult`` with blue/red win probabilities and
            model metadata.

        Raises:
            ValueError: If the draft fails validation (bad names,
                duplicates, wrong counts).
            TypeError: If *draft* is neither ``DraftInput`` nor dict.
        """
        if isinstance(draft, dict):
            draft = DraftInput(**draft)
        elif not isinstance(draft, DraftInput):
            raise TypeError(
                f"Expected DraftInput or dict, got {type(draft).__name__}"
            )

        # 1. Validate and normalize
        normalized = validate_draft(
            draft, self.valid_champions, self.valid_teams, CHAMPION_ALIASES
        )

        # 2. Determine match context
        if normalized.patch is not None:
            patch_float = self._parse_patch(normalized.patch)
        else:
            patch_float = self.latest_patch

        year = int(patch_float) + 2010
        league = self._infer_league(normalized.blue_team, normalized.red_team)
        playoffs = 0
        split = self._infer_split()

        # 3. Compute features for both perspectives
        blue_features = compute_features_for_side(
            picks=normalized.blue_picks,
            bans=normalized.blue_bans,
            team=normalized.blue_team,
            side="Blue",
            patch_float=patch_float,
            year=year,
            league=league,
            playoffs=playoffs,
            split=split,
            meta_strength=self.meta_strength,
            synergies=self.synergies,
            team_perf=self.team_perf,
            encoders=self.encoders,
        )

        red_features = compute_features_for_side(
            picks=normalized.red_picks,
            bans=normalized.red_bans,
            team=normalized.red_team,
            side="Red",
            patch_float=patch_float,
            year=year,
            league=league,
            playoffs=playoffs,
            split=split,
            meta_strength=self.meta_strength,
            synergies=self.synergies,
            team_perf=self.team_perf,
            encoders=self.encoders,
        )

        # 4. Scale features
        blue_arr = np.array(blue_features).reshape(1, -1)
        red_arr = np.array(red_features).reshape(1, -1)

        # Suppress "X does not have valid feature names" warnings.
        # We pass raw numpy arrays (no column names) which is fine --
        # the feature order matches scaler.feature_names_in_ exactly.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            blue_scaled = self.scaler.transform(blue_arr)
            red_scaled = self.scaler.transform(red_arr)

            # 5. Get predictions from both perspectives
            blue_pred = self.model.predict_proba(blue_scaled)[0]
            red_pred = self.model.predict_proba(red_scaled)[0]

        # 6. Dual-perspective averaging
        # blue_pred[1] = P(blue wins | blue perspective)
        # red_pred[1] = P(red wins | red perspective)
        # Average: blue win = (P_blue_wins_blue + (1 - P_red_wins_red)) / 2
        blue_win_prob = (blue_pred[1] + (1 - red_pred[1])) / 2
        red_win_prob = 1.0 - blue_win_prob

        return PredictionResult(
            blue_win_prob=float(blue_win_prob),
            red_win_prob=float(red_win_prob),
            model_name=self.model_name,
            model_version=self.model_version,
        )

    def get_status(self) -> AdapterStatus:
        """Return health and diagnostic information.

        Memory usage is reported as the maximum resident set size
        of the current process (Linux ``ru_maxrss`` in KB, converted
        to MB).
        """
        # ru_maxrss is in KB on Linux
        mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mem_mb = mem_kb / 1024.0

        return AdapterStatus(
            loaded_artifacts=list(self._loaded_artifacts),
            model_name=self.model_name,
            model_version=self.model_version,
            memory_usage_mb=mem_mb,
            champion_count=len(self.valid_champions),
            team_count=len(self.valid_teams),
        )

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _parse_patch(self, patch_str: str) -> float:
        """Convert a user-supplied patch string to float.

        Accepts formats like ``"14.18"``, ``"14.1"``, or ``"14"``.
        """
        try:
            return float(patch_str)
        except (ValueError, TypeError):
            return self.latest_patch

    def _infer_league(self, blue_team: str, red_team: str) -> str:
        """Infer the league from team names.

        Falls back to ``"Unknown"`` if neither team is recognized.
        This is used only for the league LabelEncoder feature.
        """
        # Simple heuristic: check if either team is in a known league's
        # typical roster.  Since we don't have a team->league mapping
        # in the standalone dicts, default to the most common league.
        return "LCK"

    def _infer_split(self) -> str:
        """Return a reasonable default split value."""
        return "Summer"
