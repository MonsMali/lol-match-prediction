"""LoL Draft Adapter -- typed interface to the ML prediction pipeline.

Public API
----------
Schemas:
    DraftInput          Complete draft input for a match prediction.
    PredictionResult    Typed prediction output (probabilities + metadata).
    AdapterStatus       Health / diagnostic information.

Normalization:
    normalize_champion_name   Resolve display names to canonical form.
    normalize_team_name       Resolve team names to canonical form.
"""

from src.adapter.schemas import AdapterStatus, DraftInput, PredictionResult
from src.adapter.normalization import normalize_champion_name, normalize_team_name

__all__ = [
    "DraftInput",
    "PredictionResult",
    "AdapterStatus",
    "normalize_champion_name",
    "normalize_team_name",
]
