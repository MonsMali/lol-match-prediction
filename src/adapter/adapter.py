"""LoLDraftAdapter -- singleton entry point for match prediction.

Loads the thesis Logistic Regression model (48 features) and all
pre-serialized lookup dicts at initialization. Exposes
``predict_from_draft`` for the primary prediction path and
``compute_suggestions`` for the asynchronous suggestion path.

The model is a standalone LogisticRegression trained on 48 features
with TargetEncoder categoricals. SHAP explanations use
LinearExplainer (exact for linear models).
"""

from __future__ import annotations

import json
import logging
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
    PickImpact,
    PredictionResult,
    SuggestionResult,
    TeamContext,
)
from src.adapter.suggestions import build_suggestion_features, resolve_suggestions
from src.adapter.validation import validate_draft
from src.config import MODELS_DIR, PRODUCTION_MODELS_DIR

logger = logging.getLogger(__name__)

# Team-to-league mapping for league inference.
_TEAM_TO_LEAGUE: dict[str, str] = {}
_TEAMS_BY_LEAGUE: dict[str, list[str]] = {
    "LCK": [
        "T1", "Gen.G", "KT Rolster", "DRX", "Dplus KIA",
        "Hanwha Life Esports", "Nongshim RedForce",
        "BNK FEARX", "OKSavingsBank BRION", "kwangdong freecs",
    ],
    "LEC": [
        "G2 Esports", "Fnatic", "Team Vitality",
        "Team BDS", "SK Gaming", "Team Heretics", "KOI", "GiantX",
        "Karmine Corp", "Natus Vincere",
    ],
    "LCS": [
        "Team Liquid", "Cloud9", "FlyQuest", "Dignitas",
        "Shopify Rebellion", "Sentinels", "LYON", "Disguised",
    ],
    "LPL": [
        "JD Gaming", "Bilibili Gaming", "Weibo Gaming", "Top Esports",
        "EDward Gaming", "Invictus Gaming", "Team WE", "Oh My God",
        "LNG Esports", "Ultra Prime", "Anyone's Legend", "LGD Gaming",
        "Ninjas in Pyjamas", "TT",
    ],
}
for _league, _teams in _TEAMS_BY_LEAGUE.items():
    for _team in _teams:
        _TEAM_TO_LEAGUE[_team] = _league


class LoLDraftAdapter:
    """Singleton adapter wrapping the thesis LR prediction pipeline.

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
        # 1. Resolve artifact paths
        # ---------------------------------------------------------------
        # Primary: standalone LR model. Fallback: best_model.joblib
        lr_model_path = production_dir / "lr_model.joblib"
        legacy_model_path = production_dir / "best_model.joblib"

        artifact_paths = {
            "scaler": models_root / "ultimate_scaler.joblib",
            "meta_strength": models_root / "champion_meta_strength.joblib",
            "team_perf": models_root / "team_historical_performance.joblib",
            "matchups": models_root / "champion_matchups.joblib",
        }

        # Determine model path
        if lr_model_path.exists():
            artifact_paths["model"] = lr_model_path
        elif legacy_model_path.exists():
            artifact_paths["model"] = legacy_model_path
        else:
            artifact_paths["model"] = lr_model_path  # Will trigger missing error

        # Check for production scaler fallback
        if not artifact_paths["scaler"].exists():
            prod_scaler = production_dir / "scaler.joblib"
            if prod_scaler.exists():
                artifact_paths["scaler"] = prod_scaler

        missing = [
            f"  - {name}: {path}"
            for name, path in artifact_paths.items()
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing required model artifacts:\n"
                + "\n".join(missing)
                + "\n\nFor the LR model, either:\n"
                + "  a) Extract from Colab and save as models/production/lr_model.joblib\n"
                + "  b) Ensure models/production/best_model.joblib exists"
            )

        # ---------------------------------------------------------------
        # 2. Load artifacts
        # ---------------------------------------------------------------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_model = joblib.load(artifact_paths["model"])
            self.scaler = joblib.load(artifact_paths["scaler"])
            self.meta_strength: dict = joblib.load(artifact_paths["meta_strength"])
            self.team_perf: dict = joblib.load(artifact_paths["team_perf"])
            self.matchups: dict = joblib.load(artifact_paths["matchups"])

        self._loaded_artifacts = list(artifact_paths.keys())

        # ---------------------------------------------------------------
        # 3. Extract model
        # ---------------------------------------------------------------
        if isinstance(raw_model, dict):
            # Training result dict -- extract LogisticRegression
            if "Logistic Regression" in raw_model:
                self.model = raw_model["Logistic Regression"]
            else:
                # Take first model
                self.model = next(iter(raw_model.values()))
        elif hasattr(raw_model, "named_estimators_"):
            # VotingClassifier -- extract LR if present
            estimators = raw_model.named_estimators_
            if "Logistic Regression" in estimators:
                self.model = estimators["Logistic Regression"]
            else:
                # Fallback: use the full VotingClassifier
                self.model = raw_model
        else:
            # Standalone model (expected path for lr_model.joblib)
            self.model = raw_model

        # ---------------------------------------------------------------
        # 4. Load optional lookup artifacts
        # ---------------------------------------------------------------
        self.champion_characteristics: dict = self._load_optional(
            models_root / "champion_characteristics.joblib", {}
        )
        self.champion_popularity: dict = self._load_optional(
            models_root / "champion_popularity.joblib", {}
        )
        self.ban_priority: dict = self._load_optional(
            models_root / "ban_priority.joblib", {}
        )
        self.lane_advantages: dict = self._load_optional(
            models_root / "lane_advantages.joblib", {}
        )
        self.champion_archetypes: dict = self._load_optional(
            models_root / "champion_archetypes.joblib", {}
        )
        self.archetype_advantages: dict = self._load_optional(
            models_root / "archetype_advantages.joblib", {}
        )
        self.team_advantages: dict = self._load_optional(
            models_root / "team_advantages.joblib", {}
        )
        self.target_encoders: dict = self._load_optional(
            models_root / "target_encoders.joblib", {}
        )

        # Fallback: try loading from production encoders.joblib
        if not self.target_encoders:
            prod_encoders_path = production_dir / "encoders.joblib"
            if prod_encoders_path.exists():
                self.target_encoders = self._load_optional(prod_encoders_path, {})

        # Player performance artifacts (optional -- available after retraining)
        self.player_performance: dict = self._load_optional(
            models_root / "player_performance.joblib", {}
        )
        self.player_champion_mastery: dict = self._load_optional(
            models_root / "player_champion_mastery.joblib", {}
        )

        # ---------------------------------------------------------------
        # 5. Derive convenience sets
        # ---------------------------------------------------------------
        self.valid_champions: set[str] = set()
        # Champions from meta_strength keys
        for key in self.meta_strength.keys():
            if isinstance(key, tuple) and len(key) == 2:
                self.valid_champions.add(str(key[1]))
        # Champions from champion_characteristics
        self.valid_champions.update(self.champion_characteristics.keys())

        self.valid_teams: set[str] = set()
        if self.team_perf:
            self.valid_teams = set(self.team_perf.keys())

        # ---------------------------------------------------------------
        # 6. Determine latest patch
        # ---------------------------------------------------------------
        patches_str = set()
        for key in self.meta_strength.keys():
            if isinstance(key, tuple) and len(key) == 2:
                patches_str.add(str(key[0]))

        # Parse as float for comparison, keep string for display
        patch_floats = {}
        for p in patches_str:
            try:
                patch_floats[p] = float(p)
            except (ValueError, TypeError):
                pass

        if patch_floats:
            self.latest_patch_str = max(patch_floats, key=patch_floats.get)
            self.latest_patch_float = patch_floats[self.latest_patch_str]
        else:
            self.latest_patch_str = "14.18"
            self.latest_patch_float = 14.18

        self.latest_year = int(self.latest_patch_float) + 2010

        # ---------------------------------------------------------------
        # 7. Model metadata
        # ---------------------------------------------------------------
        model_class = type(self.model).__name__
        self.model_name: str = model_class
        self.model_version: str = "thesis-lr-v1"

        self.training_patch_str: str = self.latest_patch_str
        self.training_year: int = self.latest_year

        # Load refresh metadata if available
        self.lookup_metadata: dict = {}
        metadata_path = models_root / "lookup_metadata.json"
        if metadata_path.exists():
            try:
                self.lookup_metadata = json.loads(metadata_path.read_text())
            except Exception:
                pass

        # ---------------------------------------------------------------
        # 8. SHAP explainer (exact for linear models)
        # ---------------------------------------------------------------
        self.explainer = create_explainer(self.model, self.scaler)

        self._initialized = True
        logger.info(
            "LoLDraftAdapter initialized: model=%s, champions=%d, teams=%d, patch=%s",
            self.model_name, len(self.valid_champions),
            len(self.valid_teams), self.training_patch_str,
        )

    @staticmethod
    def _load_optional(path: Path, default: Any) -> Any:
        """Load a joblib artifact, returning default on any failure."""
        if not path.exists():
            return default
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return joblib.load(path)
        except Exception as e:
            logger.warning("Failed to load optional artifact %s: %s", path, e)
            return default

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
            players=normalized.blue_players,
            **ctx,
        )
        red_features = compute_features_for_side(
            picks=normalized.red_picks,
            bans=normalized.red_bans,
            team=normalized.red_team,
            side="Red",
            players=normalized.red_players,
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

        # SHAP insights (exact, deterministic)
        blue_insights = compute_impact_insights(
            self.explainer, blue_scaled, "Blue"
        )
        red_insights = compute_impact_insights(
            self.explainer, red_scaled, "Red"
        )

        # Per-pick impact (leave-one-out)
        blue_pick_impacts = self._compute_pick_impacts(
            normalized.blue_picks, normalized.blue_bans,
            normalized.blue_team, "Blue", ctx, blue_win_prob,
        )
        red_pick_impacts = self._compute_pick_impacts(
            normalized.red_picks, normalized.red_bans,
            normalized.red_team, "Red", ctx, red_win_prob,
        )

        # Build team context
        blue_team_ctx = self._build_team_context(
            normalized.blue_team, blue_features
        )
        red_team_ctx = self._build_team_context(
            normalized.red_team, red_features
        )

        return PredictionResult(
            blue_win_prob=float(blue_win_prob),
            red_win_prob=float(red_win_prob),
            model_name=self.model_name,
            model_version=self.model_version,
            blue_insights=blue_insights,
            red_insights=red_insights,
            blue_pick_impacts=blue_pick_impacts,
            red_pick_impacts=red_pick_impacts,
            blue_team_context=blue_team_ctx,
            red_team_context=red_team_ctx,
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
        """Compute champion swap suggestions for both sides."""
        normalized, ctx = self._validate_and_contextualize(draft)

        # Get current win probabilities for delta computation
        blue_features = compute_features_for_side(
            picks=normalized.blue_picks,
            bans=normalized.blue_bans,
            team=normalized.blue_team,
            side="Blue",
            players=normalized.blue_players,
            **ctx,
        )
        red_features = compute_features_for_side(
            picks=normalized.red_picks,
            bans=normalized.red_bans,
            team=normalized.red_team,
            side="Red",
            players=normalized.red_players,
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
    # Per-pick impact (leave-one-out)
    # -------------------------------------------------------------------

    def _compute_pick_impacts(
        self,
        picks: dict[str, str],
        bans: list[str],
        team: str,
        side: str,
        ctx: dict,
        baseline_win_prob: float,
    ) -> list[PickImpact]:
        """Compute each champion's marginal impact via leave-one-out."""
        _NEUTRAL = "__neutral__"
        impacts: list[PickImpact] = []

        for role, champion in picks.items():
            modified_picks = dict(picks)
            modified_picks[role] = _NEUTRAL

            modified_features = compute_features_for_side(
                picks=modified_picks,
                bans=bans,
                team=team,
                side=side,
                **ctx,
            )

            modified_arr = np.array(modified_features).reshape(1, -1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                modified_scaled = self.scaler.transform(modified_arr)
                modified_prob = float(
                    self.model.predict_proba(modified_scaled)[0][1]
                )

            delta = baseline_win_prob - modified_prob
            impacts.append(PickImpact(
                role=role,
                champion=champion,
                impact_pct=round(delta * 100, 1),
            ))

        impacts.sort(key=lambda p: p.impact_pct)
        return impacts

    # -------------------------------------------------------------------
    # Team context builder
    # -------------------------------------------------------------------

    def _build_team_context(
        self, team: str, features: list[float],
    ) -> TeamContext:
        """Build team context from lookup dicts and computed features.

        Feature indices reference CANONICAL_48_FEATURES order.
        """
        from src.features.formulas import DEFAULT_TEAM_PERF

        perf = self.team_perf.get(team, DEFAULT_TEAM_PERF)
        if isinstance(perf, (int, float)):
            hist_wr = float(perf)
            recent_wr = float(perf)
            form = 0.0
        else:
            hist_wr = perf.get("overall_winrate", 0.5)
            recent_wr = perf.get("recent_winrate", 0.5)
            form = perf.get("form_trend", 0.0)

        # meta_advantage is feature index 9
        meta_adaptation = features[9] if len(features) > 9 else 0.0

        return TeamContext(
            historical_winrate=hist_wr,
            recent_winrate=recent_wr,
            form_trend=form,
            meta_adaptation=meta_adaptation,
        )

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _validate_and_contextualize(
        self,
        draft: Union[DraftInput, dict],
    ) -> tuple[DraftInput, dict]:
        """Validate draft and build match context dict."""
        if isinstance(draft, dict):
            draft = DraftInput(**draft)
        elif not isinstance(draft, DraftInput):
            raise TypeError(
                f"Expected DraftInput or dict, got {type(draft).__name__}"
            )

        normalized = validate_draft(
            draft, self.valid_champions, self.valid_teams, CHAMPION_ALIASES
        )

        # Determine patch string
        if normalized.patch is not None:
            patch_str = normalized.patch
        else:
            patch_str = self.latest_patch_str

        try:
            patch_float = float(patch_str)
        except (ValueError, TypeError):
            patch_float = self.latest_patch_float

        ctx = dict(
            patch=patch_str,
            year=int(patch_float) + 2010,
            league=self._infer_league(normalized.blue_team, normalized.red_team),
            playoffs=0,
            split=self._infer_split(),
            champion_characteristics=self.champion_characteristics,
            champion_meta_strength=self.meta_strength,
            champion_popularity=self.champion_popularity,
            team_historical_performance=self.team_perf,
            ban_priority=self.ban_priority,
            lane_advantages=self.lane_advantages,
            champion_archetypes=self.champion_archetypes,
            archetype_advantages=self.archetype_advantages,
            team_advantages=self.team_advantages,
            target_encoders=self.target_encoders,
            player_performance=self.player_performance if self.player_performance else None,
            player_champion_mastery=self.player_champion_mastery if self.player_champion_mastery else None,
        )

        return normalized, ctx

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
