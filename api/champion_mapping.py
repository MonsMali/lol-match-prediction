"""Champion name to DDragon asset ID mapping.

Maps display names (as they appear in the training data and the
adapter's valid_champions set) to the DDragon image file identifier.
Most champions use their display name directly, but some require
special handling due to differences between Riot's internal IDs
and public display names.
"""

import re

DDRAGON_VERSION = "14.24.1"

DDRAGON_BASE = (
    f"https://ddragon.leagueoflegends.com/cdn/{DDRAGON_VERSION}/img/champion"
)

# Display name -> DDragon asset ID for champions where they differ.
# Most champions use their display name with spaces/punctuation removed,
# but these require explicit mapping.
DDRAGON_ID_MAP: dict[str, str] = {
    "Wukong": "MonkeyKing",
    "Nunu & Willump": "Nunu",
    "Kai'Sa": "Kaisa",
    "Kha'Zix": "Khazix",
    "Cho'Gath": "Chogath",
    "Vel'Koz": "Velkoz",
    "Rek'Sai": "RekSai",
    "Kog'Maw": "KogMaw",
    "Bel'Veth": "Belveth",
    "K'Sante": "KSante",
    "LeBlanc": "Leblanc",
    "Renata Glasc": "Renata",
}


def get_ddragon_url(champion_name: str) -> str:
    """Build the full DDragon CDN URL for a champion's square portrait.

    Looks up the champion in DDRAGON_ID_MAP first. If not found,
    falls back to stripping spaces, apostrophes, periods, and
    ampersands from the display name.

    Args:
        champion_name: Champion display name as stored in the adapter.

    Returns:
        Full URL to the champion's square portrait PNG on DDragon CDN.
    """
    if champion_name in DDRAGON_ID_MAP:
        ddragon_id = DDRAGON_ID_MAP[champion_name]
    else:
        # Strip spaces, apostrophes, periods, ampersands
        ddragon_id = re.sub(r"[\s'.&]", "", champion_name)
    return f"{DDRAGON_BASE}/{ddragon_id}.png"
