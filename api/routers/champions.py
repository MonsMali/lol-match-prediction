"""GET /api/champions endpoint.

Returns the list of champions that the model knows about, each with
a DDragon CDN image URL for the frontend to display portraits.
"""

from fastapi import APIRouter, Depends

from api.champion_mapping import get_ddragon_url
from api.dependencies import get_adapter
from api.schemas import ChampionInfo, ChampionListResponse
from src.adapter import LoLDraftAdapter

router = APIRouter(prefix="/api", tags=["champions"])


@router.get("/champions", response_model=ChampionListResponse)
async def list_champions(
    adapter: LoLDraftAdapter = Depends(get_adapter),
) -> ChampionListResponse:
    """Return all champions known to the loaded model.

    Each entry includes the canonical name, key (same as name), and
    a DDragon CDN URL for the champion's square portrait image.

    The response is stable for the lifetime of the server process --
    the frontend can safely cache this response.
    """
    champions = [
        ChampionInfo(
            name=name,
            key=name,
            image_url=get_ddragon_url(name),
        )
        for name in sorted(adapter.valid_champions)
    ]
    return ChampionListResponse(champions=champions)
