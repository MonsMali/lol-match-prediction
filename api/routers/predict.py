"""POST /api/predict and POST /api/suggestions endpoints.

/api/predict is the fast path (~15ms): returns win probabilities
and SHAP-based impact insights.

/api/suggestions is the async path (~100ms): returns champion swap
recommendations. Called separately so the frontend can render the
probability bar instantly.
"""

import asyncio
import logging
from functools import partial

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_adapter
from api.schemas import (
    ChampionSuggestion,
    InsightFactor,
    ModelMeta,
    PickImpact,
    PredictRequest,
    PredictResponse,
    SuggestionsResponse,
    TeamContextResponse,
)
from src.adapter import LoLDraftAdapter
from src.adapter.normalization import normalize_champion_name, CHAMPION_ALIASES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["prediction"])


def _validate_champions(
    payload: PredictRequest, adapter: LoLDraftAdapter
) -> None:
    """Validate all 20 champion names. Raises HTTPException on failure."""
    all_names: list[str] = [
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
    invalid = []
    for name in all_names:
        try:
            normalize_champion_name(name, adapter.valid_champions, CHAMPION_ALIASES)
        except ValueError:
            invalid.append(name)
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid champion names: {', '.join(invalid)}",
        )


def _build_draft_dict(payload: PredictRequest) -> dict:
    bp = payload.blue_picks
    rp = payload.red_picks
    result: dict = {
        "blue_team": payload.blue_team,
        "red_team": payload.red_team,
        "blue_picks": {"top": bp.top, "jungle": bp.jungle, "mid": bp.mid, "bot": bp.bot, "support": bp.support},
        "red_picks": {"top": rp.top, "jungle": rp.jungle, "mid": rp.mid, "bot": rp.bot, "support": rp.support},
        "blue_bans": payload.blue_bans,
        "red_bans": payload.red_bans,
        "patch": payload.patch,
    }
    # Pass player names if provided
    blue_players = {r: getattr(bp, f"{r}_player") for r in ("top", "jungle", "mid", "bot", "support") if getattr(bp, f"{r}_player", None)}
    red_players = {r: getattr(rp, f"{r}_player") for r in ("top", "jungle", "mid", "bot", "support") if getattr(rp, f"{r}_player", None)}
    if blue_players:
        result["blue_players"] = blue_players
    if red_players:
        result["red_players"] = red_players
    return result


async def _call_with_team_fallback(adapter, method, draft_dict):
    """Call an adapter method with silent team fallback on unknown teams."""
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            None, partial(method, draft_dict)
        )
    except ValueError as exc:
        error_msg = str(exc).lower()
        if "unknown team" not in error_msg:
            raise HTTPException(status_code=422, detail=str(exc))

        fallback_team = sorted(adapter.valid_teams)[0]
        logger.warning(
            "Unknown team, falling back to '%s': %s", fallback_team, exc
        )
        if "blue" in error_msg or draft_dict["blue_team"] not in adapter.valid_teams:
            draft_dict["blue_team"] = fallback_team
        if "red" in error_msg or draft_dict["red_team"] not in adapter.valid_teams:
            draft_dict["red_team"] = fallback_team
        try:
            return await loop.run_in_executor(
                None, partial(method, draft_dict)
            )
        except ValueError as retry_exc:
            raise HTTPException(status_code=422, detail=str(retry_exc))


@router.post("/predict", response_model=PredictResponse)
async def predict(
    payload: PredictRequest,
    adapter: LoLDraftAdapter = Depends(get_adapter),
) -> PredictResponse:
    """Fast path: win probabilities + SHAP insights (~15ms)."""
    _validate_champions(payload, adapter)
    draft_dict = _build_draft_dict(payload)

    result = await _call_with_team_fallback(
        adapter, adapter.predict_from_draft, draft_dict
    )

    blue_ctx = None
    if result.blue_team_context:
        tc = result.blue_team_context
        blue_ctx = TeamContextResponse(
            historical_winrate=tc.historical_winrate,
            recent_winrate=tc.recent_winrate,
            form_trend=tc.form_trend,
            meta_adaptation=tc.meta_adaptation,
        )
    red_ctx = None
    if result.red_team_context:
        tc = result.red_team_context
        red_ctx = TeamContextResponse(
            historical_winrate=tc.historical_winrate,
            recent_winrate=tc.recent_winrate,
            form_trend=tc.form_trend,
            meta_adaptation=tc.meta_adaptation,
        )

    return PredictResponse(
        blue_win_probability=result.blue_win_prob,
        red_win_probability=result.red_win_prob,
        blue_insights=[
            InsightFactor(
                label=ins["label"],
                impact_pct=ins["impact_pct"],
                description=ins["description"],
            )
            for ins in result.blue_insights
        ],
        red_insights=[
            InsightFactor(
                label=ins["label"],
                impact_pct=ins["impact_pct"],
                description=ins["description"],
            )
            for ins in result.red_insights
        ],
        blue_pick_impacts=[
            PickImpact(
                role=p.role,
                champion=p.champion,
                impact_pct=p.impact_pct,
            )
            for p in result.blue_pick_impacts
        ],
        red_pick_impacts=[
            PickImpact(
                role=p.role,
                champion=p.champion,
                impact_pct=p.impact_pct,
            )
            for p in result.red_pick_impacts
        ],
        blue_team_context=blue_ctx,
        red_team_context=red_ctx,
        model=ModelMeta(
            training_patch=result.training_patch,
            training_year=result.training_year,
        ),
    )


@router.post("/suggestions", response_model=SuggestionsResponse)
async def suggestions(
    payload: PredictRequest,
    adapter: LoLDraftAdapter = Depends(get_adapter),
) -> SuggestionsResponse:
    """Async path: champion swap suggestions (~100ms)."""
    _validate_champions(payload, adapter)
    draft_dict = _build_draft_dict(payload)

    result = await _call_with_team_fallback(
        adapter, adapter.compute_draft_suggestions, draft_dict
    )

    return SuggestionsResponse(
        blue_suggestions=[
            ChampionSuggestion(
                role=s["role"],
                champion=s["champion"],
                delta_pct=s["delta_pct"],
                current_champion=s["current_champion"],
            )
            for s in result.blue_suggestions
        ],
        red_suggestions=[
            ChampionSuggestion(
                role=s["role"],
                champion=s["champion"],
                delta_pct=s["delta_pct"],
                current_champion=s["current_champion"],
            )
            for s in result.red_suggestions
        ],
    )
