"""POST /api/predict endpoint.

Accepts a complete draft payload (10 picks by role + 10 bans + team
names) and returns blue/red win probabilities via the LoLDraftAdapter.

Champion names are validated at the API layer before calling the
adapter. Unknown team names are accepted with a silent fallback.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_adapter
from api.schemas import PredictRequest, PredictResponse
from src.adapter import LoLDraftAdapter
from src.adapter.normalization import normalize_champion_name, CHAMPION_ALIASES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["prediction"])


@router.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    adapter: LoLDraftAdapter = Depends(get_adapter),
) -> PredictResponse:
    """Predict match outcome from a complete draft.

    Validates all 20 champion names (10 picks + 10 bans) against the
    model's known champion set before calling the adapter. Returns 422
    if any champion is unrecognized after normalization.

    Unknown team names are silently replaced with a fallback team so
    the prediction can still proceed (per CONTEXT.md decision).
    """
    # ------------------------------------------------------------------
    # 1. Validate all champion names up front (fail fast)
    # ------------------------------------------------------------------
    all_champion_names: list[str] = [
        payload.blue_picks.top,
        payload.blue_picks.jungle,
        payload.blue_picks.mid,
        payload.blue_picks.bot,
        payload.blue_picks.support,
        payload.red_picks.top,
        payload.red_picks.jungle,
        payload.red_picks.mid,
        payload.red_picks.bot,
        payload.red_picks.support,
        *payload.blue_bans,
        *payload.red_bans,
    ]

    invalid_champions: list[str] = []
    for name in all_champion_names:
        try:
            normalize_champion_name(name, adapter.valid_champions, CHAMPION_ALIASES)
        except ValueError:
            invalid_champions.append(name)

    if invalid_champions:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid champion names: {', '.join(invalid_champions)}",
        )

    # ------------------------------------------------------------------
    # 2. Build draft dict for the adapter
    # ------------------------------------------------------------------
    draft_dict = {
        "blue_team": payload.blue_team,
        "red_team": payload.red_team,
        "blue_picks": payload.blue_picks.model_dump(),
        "red_picks": payload.red_picks.model_dump(),
        "blue_bans": payload.blue_bans,
        "red_bans": payload.red_bans,
        "patch": payload.patch,
    }

    # ------------------------------------------------------------------
    # 3. Call adapter with silent team fallback
    # ------------------------------------------------------------------
    try:
        result = adapter.predict_from_draft(draft_dict)
    except ValueError as exc:
        error_msg = str(exc).lower()
        # If the error is about an unknown team, retry with fallback
        if "unknown team" in error_msg:
            fallback_team = sorted(adapter.valid_teams)[0]
            logger.warning(
                "Unknown team in prediction request, falling back to '%s': %s",
                fallback_team,
                exc,
            )
            if "blue" in error_msg or draft_dict["blue_team"] not in adapter.valid_teams:
                draft_dict["blue_team"] = fallback_team
            if "red" in error_msg or draft_dict["red_team"] not in adapter.valid_teams:
                draft_dict["red_team"] = fallback_team
            try:
                result = adapter.predict_from_draft(draft_dict)
            except ValueError as retry_exc:
                raise HTTPException(status_code=422, detail=str(retry_exc))
        else:
            raise HTTPException(status_code=422, detail=str(exc))

    return PredictResponse(
        blue_win_probability=result.blue_win_prob,
        red_win_probability=result.red_win_prob,
    )
