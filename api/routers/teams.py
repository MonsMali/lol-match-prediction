"""GET /api/teams endpoint.

Returns professional teams grouped by league. Only teams that the
model actually knows about are included. Teams not assigned to any
major league appear under the "Other" key.
"""

from fastapi import APIRouter, Depends

from api.dependencies import get_adapter
from api.schemas import TeamListResponse
from api.team_data import TEAMS_BY_LEAGUE
from src.adapter import LoLDraftAdapter

router = APIRouter(prefix="/api", tags=["teams"])


@router.get("/teams", response_model=TeamListResponse)
async def list_teams(
    adapter: LoLDraftAdapter = Depends(get_adapter),
) -> TeamListResponse:
    """Return teams grouped by league, filtered to model-known teams.

    Iterates over the hardcoded TEAMS_BY_LEAGUE mapping and keeps
    only teams present in adapter.valid_teams. Any valid team not
    assigned to a major league is collected under the "Other" key.
    """
    valid = adapter.valid_teams
    assigned: set[str] = set()
    result: dict[str, list[str]] = {}

    for league, roster in TEAMS_BY_LEAGUE.items():
        known = [t for t in roster if t in valid]
        if known:
            result[league] = sorted(known)
            assigned.update(known)

    # Collect unassigned valid teams under "Other"
    other = sorted(valid - assigned)
    if other:
        result["Other"] = other

    return TeamListResponse(teams=result)
