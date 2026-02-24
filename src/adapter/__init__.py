"""LoL Draft Adapter -- typed interface to the ML prediction pipeline.

Public API
----------
Core:
    LoLDraftAdapter       Singleton adapter with predict_from_draft / get_status.
    predict_from_draft    Convenience function (instantiates singleton internally).

Schemas:
    DraftInput            Complete draft input for a match prediction.
    PredictionResult      Typed prediction output (probabilities + metadata).
    AdapterStatus         Health / diagnostic information.

Normalization:
    normalize_champion_name   Resolve display names to canonical form.
    normalize_team_name       Resolve team names to canonical form.
"""

from src.adapter.adapter import LoLDraftAdapter
from src.adapter.schemas import AdapterStatus, DraftInput, PredictionResult
from src.adapter.normalization import normalize_champion_name, normalize_team_name


def predict_from_draft(draft) -> PredictionResult:
    """Predict match outcome from a complete draft.

    Convenience wrapper that obtains the ``LoLDraftAdapter`` singleton
    and delegates to its ``predict_from_draft`` method.

    Args:
        draft: A ``DraftInput`` instance or a plain dict with the
            same keys (blue_team, red_team, blue_picks, red_picks,
            blue_bans, red_bans, and optionally patch).

    Returns:
        ``PredictionResult`` with blue/red win probabilities.
    """
    adapter = LoLDraftAdapter()
    return adapter.predict_from_draft(draft)


__all__ = [
    "LoLDraftAdapter",
    "predict_from_draft",
    "DraftInput",
    "PredictionResult",
    "AdapterStatus",
    "normalize_champion_name",
    "normalize_team_name",
]
