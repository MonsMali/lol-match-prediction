"""LoLDraftAdapter -- singleton entry point for match prediction.

Loads all pre-serialized model artifacts at initialization and
exposes ``predict_from_draft`` for the primary prediction path
and ``compute_suggestions`` for the asynchronous suggestion path.

The prediction model is a tree-only sub-ensemble (RandomForest +
GradientBoosting) extracted from the production VotingClassifier.
This guarantees that SHAP explanations are mathematically exact
and sum to the predicted probability.
"""

from __future__ import annotations

import resource
import warnings
from datetime import date
from pathlib import Path
from typing import Union

import joblib
import numpy as np

from src.adapter.explainer import compute_impact_insights, create_explainer
from src.adapter.features import compute_features_for_side
from src.adapter.normalization import CHAMPION_ALIASES
from src.adapter.schemas import (
    AdapterStatus,
    DraftInput,
    PredictionResult,
    SuggestionResult,
)
from src.adapter.suggestions import build_suggestion_features, resolve_suggestions
from src.adapter.validation import validate_draft
from src.config import MODELS_DIR, PRODUCTION_MODELS_DIR


# Team-to-league mapping for league inference.
_TEAM_TO_LEAGUE: dict[str, str] = {}
_TEAMS_BY_LEAGUE: dict[str, list[str]] = {
    "LCK": [
        "T1", "Gen.G", "KT Rolster", "DRX", "Dplus KIA",
        "Hanwha Life Esports", "Liiv SANDBOX", "Nongshim RedForce",
        "BNK FEARX", "OKSavingsBank BRION",
    ],
    "LEC": [
        "G2 Esports", "Fnatic", "MAD Lions KOI", "Team Vitality",
        "Team BDS", "SK Gaming", "Team Heretics", "KOI", "GiantX",
        "Karmine Corp",
    ],
    "LCS": [
        "Team Liquid", "Cloud9", "FlyQuest", "100 Thieves", "TSM",
        "Dignitas", "Immortals", "Evil Geniuses", "Counter Logic Gaming",
        "Golden Guardians",
    ],
    "LPL": [
        "JD Gaming", "Bilibili Gaming", "Weibo Gaming", "Top Esports",
        "EDward Gaming", "Invictus Gaming", "Team WE", "Oh My God",
        "LNG Esports", "Ultra Prime",
    ],
}
for _league, _teams in _TEAMS_BY_LEAGUE.items():
    for _team in _teams:
        _TEAM_TO_LEAGUE[_team] = _league


class _TreeEnsemble:
    """Lightweight tree-only sub-ensemble for explainable prediction.

    Wraps RandomForest + GradientBoosting with their original
    VotingClassifier weights, normalized to sum to 1.
    """

    def __init__(self, rf, gbm, rf_weight: float, gbm_weight: float):
        self.rf = rf
        self.gbm = gbm
        total = rf_weight + gbm_weight
        self.rf_w = rf_weight / total
        self.gbm_w = gbm_weight / total

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of RF and GBM probabilities."""
        rf_p = self.rf.predict_proba(X)
        gbm_p = self.gbm.predict_proba(X)
        return self.rf_w * rf_p + self.gbm_w * gbm_p


class LoLDraftAdapter:
    """Singleton adapter wrapping the production ML pipeline.

    On first instantiation the adapter eagerly loads all required
    joblib artifacts and fails fast if any file is missing.
    Subsequent calls to ``LoLDraftAdapter()`` return the same
    instance.
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
        models_root = production_dir.parent

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
            "matchups": models_root / "champion_matchups.joblib",
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
        full_model = joblib.load(artifact_paths["model"])
        self.scaler = joblib.load(artifact_paths["scaler"])
        self.encoders: dict = joblib.load(artifact_paths["encoders"])
        self.meta_strength: dict = joblib.load(artifact_paths["meta_strength"])
        self.synergies: dict = joblib.load(artifact_paths["synergies"])
        self.team_perf: dict = joblib.load(artifact_paths["team_perf"])
        self.matchups: dict = joblib.load(artifact_paths["matchups"])

        self._loaded_artifacts = list(artifact_paths.keys())

        # ---------------------------------------------------------------
        # 3. Extract tree-only sub-ensemble
        # ---------------------------------------------------------------
        estimators = full_model.named_estimators_
        weights = full_model.weights if full_model.weights is not None else [1.0] * len(estimators)
        weight_map = dict(zip(estimators.keys(), weights))

        self.model = _TreeEnsemble(
            rf=estimators["Random Forest"],
            gbm=estimators["Gradient Boosting"],
            rf_weight=weight_map["Random Forest"],
            gbm_weight=weight_map["Gradient Boosting"],
        )

        # ---------------------------------------------------------------
        # 4. Derive convenience sets
        # ---------------------------------------------------------------
        self.valid_champions: set[str] = {
            champ for _, champ in self.meta_strength.keys()
        }
        self.valid_teams: set[str] = set(self.team_perf.keys())

        # ---------------------------------------------------------------
        # 5. Determine latest patch
        # ---------------------------------------------------------------
        numeric_patches = [
            p for p, _ in self.meta_strength.keys()
            if isinstance(p, (int, float))
        ]
        self.latest_patch: float = max(numeric_patches) if numeric_patches else 14.18
        self.latest_year: int = int(self.latest_patch) + 2010

        # ---------------------------------------------------------------
        # 6. Model metadata
        # ---------------------------------------------------------------
        self.model_name: str = "TreeEnsemble(RF+GBM)"
        self.model_version: str = "enhanced-v2"

        self.training_patch_str: str = (
            f"{self.latest_patch:.2f}".rstrip("0").rstrip(".")
        )
        self.training_year: int = self.latest_year

        # ---------------------------------------------------------------
        # 7. SHAP explainer (exact, probability-space)
        # ---------------------------------------------------------------
        self.explainer = create_explainer(self.model, self.scaler)

        self._initialized = True

    # -------------------------------------------------------------------
    # Public API: Primary prediction (fast path)
    # -------------------------------------------------------------------

    def predict_from_draft(
        self,
        draft: Union[DraftInput, dict],
    ) -> PredictionResult:
        """Predict match outcome with SHAP explanations.

        This is the fast path (~15ms). Returns win probabilities and
        per-feature impact insights. Does NOT compute suggestions.
        """
        normalized, ctx = self._validate_and_contextualize(draft)

        # Compute features for both perspectives
        blue_features = compute_features_for_side(
            picks=normalized.blue_picks,
            bans=normalized.blue_bans,
            team=normalized.blue_team,
            side="Blue",
            **ctx,
        )
        red_features = compute_features_for_side(
            picks=normalized.red_picks,
            bans=normalized.red_bans,
            team=normalized.red_team,
            side="Red",
            **ctx,
        )

        blue_arr = np.array(blue_features).reshape(1, -1)
        red_arr = np.array(red_features).reshape(1, -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            blue_scaled = self.scaler.transform(blue_arr)
            red_scaled = self.scaler.transform(red_arr)

            blue_pred = self.model.predict_proba(blue_scaled)[0]
            red_pred = self.model.predict_proba(red_scaled)[0]

        # Dual-perspective averaging
        blue_win_prob = (blue_pred[1] + (1 - red_pred[1])) / 2
        red_win_prob = 1.0 - blue_win_prob

        # SHAP insights (exact, deterministic, ~2ms per side)
        blue_insights = compute_impact_insights(
            self.explainer, blue_scaled, "Blue"
        )
        red_insights = compute_impact_insights(
            self.explainer, red_scaled, "Red"
        )

        return PredictionResult(
            blue_win_prob=float(blue_win_prob),
            red_win_prob=float(red_win_prob),
            model_name=self.model_name,
            model_version=self.model_version,
            blue_insights=blue_insights,
            red_insights=red_insights,
            training_patch=self.training_patch_str,
            training_year=self.training_year,
        )

    # -------------------------------------------------------------------
    # Public API: Suggestions (async path)
    # -------------------------------------------------------------------

    def compute_draft_suggestions(
        self,
        draft: Union[DraftInput, dict],
    ) -> SuggestionResult:
        """Compute champion swap suggestions for both sides.

        This is the slower path (~100-150ms) and should be called
        asynchronously after predict_from_draft.
        """
        normalized, ctx = self._validate_and_contextualize(draft)

        # Get current win probabilities for delta computation
        blue_features = compute_features_for_side(
            picks=normalized.blue_picks,
            bans=normalized.blue_bans,
            team=normalized.blue_team,
            side="Blue",
            **ctx,
        )
        red_features = compute_features_for_side(
            picks=normalized.red_picks,
            bans=normalized.red_bans,
            team=normalized.red_team,
            side="Red",
            **ctx,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            blue_scaled = self.scaler.transform(
                np.array(blue_features).reshape(1, -1)
            )
            red_scaled = self.scaler.transform(
                np.array(red_features).reshape(1, -1)
            )
            blue_win_prob = float(
                self.model.predict_proba(blue_scaled)[0][1]
            )
            red_win_prob = float(
                self.model.predict_proba(red_scaled)[0][1]
            )

        # Build suggestion features for both sides
        common_sugg = dict(
            all_champions=self.valid_champions,
            matchups=self.matchups,
            **ctx,
        )
        blue_keys, blue_feats = build_suggestion_features(
            current_picks=normalized.blue_picks,
            bans=normalized.blue_bans,
            team=normalized.blue_team,
            side="Blue",
            opponent_picks=normalized.red_picks,
            **common_sugg,
        )
        red_keys, red_feats = build_suggestion_features(
            current_picks=normalized.red_picks,
            bans=normalized.red_bans,
            team=normalized.red_team,
            side="Red",
            opponent_picks=normalized.blue_picks,
            **common_sugg,
        )

        # Single batched model call for all candidates
        all_feats = blue_feats + red_feats
        blue_suggestions = []
        red_suggestions = []

        if all_feats:
            sugg_arr = np.array(all_feats)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                sugg_scaled = self.scaler.transform(sugg_arr)
                sugg_probas = self.model.predict_proba(sugg_scaled)[:, 1]

            n_blue = len(blue_feats)
            blue_suggestions = resolve_suggestions(
                blue_keys,
                sugg_probas[:n_blue].tolist(),
                normalized.blue_picks,
                blue_win_prob,
            )
            red_suggestions = resolve_suggestions(
                red_keys,
                sugg_probas[n_blue:].tolist(),
                normalized.red_picks,
                red_win_prob,
            )

        return SuggestionResult(
            blue_suggestions=blue_suggestions,
            red_suggestions=red_suggestions,
        )

    def get_status(self) -> AdapterStatus:
        """Return health and diagnostic information."""
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

    def _validate_and_contextualize(
        self,
        draft: Union[DraftInput, dict],
    ) -> tuple[DraftInput, dict]:
        """Validate draft and build match context dict.

        Shared by predict_from_draft and compute_draft_suggestions.
        """
        if isinstance(draft, dict):
            draft = DraftInput(**draft)
        elif not isinstance(draft, DraftInput):
            raise TypeError(
                f"Expected DraftInput or dict, got {type(draft).__name__}"
            )

        normalized = validate_draft(
            draft, self.valid_champions, self.valid_teams, CHAMPION_ALIASES
        )

        if normalized.patch is not None:
            patch_float = self._parse_patch(normalized.patch)
        else:
            patch_float = self.latest_patch

        ctx = dict(
            patch_float=patch_float,
            year=int(patch_float) + 2010,
            league=self._infer_league(normalized.blue_team, normalized.red_team),
            playoffs=0,
            split=self._infer_split(),
            meta_strength=self.meta_strength,
            synergies=self.synergies,
            team_perf=self.team_perf,
            encoders=self.encoders,
        )

        return normalized, ctx

    def _parse_patch(self, patch_str: str) -> float:
        try:
            return float(patch_str)
        except (ValueError, TypeError):
            return self.latest_patch

    def _infer_league(self, blue_team: str, red_team: str) -> str:
        blue_league = _TEAM_TO_LEAGUE.get(blue_team)
        red_league = _TEAM_TO_LEAGUE.get(red_team)
        if blue_league:
            return blue_league
        if red_league:
            return red_league
        return "Unknown"

    @staticmethod
    def _infer_split() -> str:
        return "Spring" if date.today().month <= 6 else "Summer"
