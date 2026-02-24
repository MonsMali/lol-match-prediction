"""Champion and team name normalization for the LoL Draft Adapter.

The training data uses specific canonical forms for champion and team
names.  This module maps common aliases, display-name variants, and
API identifiers to those canonical forms, and provides fuzzy-match
error messages when a name is not recognized.
"""

from difflib import get_close_matches


# ---------------------------------------------------------------------------
# Champion alias map
# ---------------------------------------------------------------------------
# Keys are LOWERCASE alternate names; values are the canonical form found
# in the serialized lookup dictionaries (champion_meta_strength, etc.).
#
# The canonical names come from Oracle's Elixir data and use the Riot
# display-name format: "Kai'Sa", "Kha'Zix", "Lee Sin", "Jarvan IV", etc.
#
# This map covers:
#   - Riot API identifiers (MonkeyKing, FiddleSticks, etc.)
#   - Common shorthand / abbreviations (J4, TF, MF, etc.)
#   - Frequent misspellings and missing-apostrophe variants
#   - Old vs new names (Nunu -> Nunu & Willump)
# ---------------------------------------------------------------------------

CHAMPION_ALIASES: dict[str, str] = {
    # Riot API id -> display name
    "monkeyking": "Wukong",
    "fiddlesticks": "Fiddlesticks",  # Riot API used "FiddleSticks" pre-rework

    # Nunu rename
    "nunu": "Nunu & Willump",
    "nunu and willump": "Nunu & Willump",
    "nunu&willump": "Nunu & Willump",

    # Apostrophe-less variants
    "kaisa": "Kai'Sa",
    "kai sa": "Kai'Sa",
    "khazix": "Kha'Zix",
    "kha zix": "Kha'Zix",
    "chogath": "Cho'Gath",
    "cho gath": "Cho'Gath",
    "reksai": "Rek'Sai",
    "rek sai": "Rek'Sai",
    "velkoz": "Vel'Koz",
    "vel koz": "Vel'Koz",
    "kogmaw": "Kog'Maw",
    "kog maw": "Kog'Maw",
    "belveth": "Bel'Veth",
    "bel veth": "Bel'Veth",
    "ksante": "K'Sante",

    # Multi-word name variants (no space)
    "leesin": "Lee Sin",
    "twistedfate": "Twisted Fate",
    "missfortune": "Miss Fortune",
    "xinzhao": "Xin Zhao",
    "jarvaniv": "Jarvan IV",
    "drmundo": "Dr. Mundo",
    "dr mundo": "Dr. Mundo",
    "renataglasc": "Renata Glasc",
    "aurelionsol": "Aurelion Sol",
    "tahmkench": "Tahm Kench",
    "masteryi": "Master Yi",

    # Common abbreviations
    "j4": "Jarvan IV",
    "jarvan": "Jarvan IV",
    "tf": "Twisted Fate",
    "mf": "Miss Fortune",
    "asol": "Aurelion Sol",
    "lb": "LeBlanc",
    "leblanc": "LeBlanc",
    "le blanc": "LeBlanc",

    # Common misspellings
    "talyah": "Taliyah",
    "taliya": "Taliyah",
    "talia": "Taliyah",
    "xin": "Xin Zhao",
    "hiem": "Heimerdinger",
    "heimer": "Heimerdinger",
    "fiddle": "Fiddlesticks",
    "ww": "Warwick",
    "mundo": "Dr. Mundo",
    "yi": "Master Yi",
}


def normalize_champion_name(
    name: str,
    valid_champions: set[str],
    aliases: dict[str, str] | None = None,
) -> str:
    """Resolve a champion name to its canonical form.

    Resolution order:
      1. Alias map lookup (case-insensitive).
      2. Case-insensitive match against the valid champion set.
      3. Raise ``ValueError`` with fuzzy suggestions.

    Args:
        name: Raw champion name from caller input.
        valid_champions: Set of canonical champion names from training data.
        aliases: Override alias map. Defaults to ``CHAMPION_ALIASES``.

    Returns:
        The canonical champion name string.

    Raises:
        ValueError: If the name cannot be resolved. The message includes
            up to 3 fuzzy-match suggestions when available.
    """
    if aliases is None:
        aliases = CHAMPION_ALIASES

    # 1. Check alias map (case-insensitive key lookup)
    alias_result = aliases.get(name.lower().strip())
    if alias_result is not None:
        # Verify the alias target is actually in the valid set
        for valid in valid_champions:
            if valid.lower() == alias_result.lower():
                return valid
        # Alias points to a name not in valid set -- fall through to fuzzy

    # 2. Case-insensitive exact match against valid champions
    name_stripped = name.strip()
    for valid in valid_champions:
        if valid.lower() == name_stripped.lower():
            return valid

    # 3. No match -- build a helpful error with fuzzy suggestions
    suggestions = get_close_matches(
        name_stripped, list(valid_champions), n=3, cutoff=0.6
    )
    msg = f"Unknown champion: '{name_stripped}'"
    if suggestions:
        msg += f". Did you mean: {', '.join(suggestions)}?"
    raise ValueError(msg)


def normalize_team_name(
    name: str,
    valid_teams: set[str],
) -> str:
    """Resolve a team name to its canonical form.

    Performs case-insensitive matching against the set of teams
    present in the training data.

    Args:
        name: Raw team name from caller input.
        valid_teams: Set of canonical team names from training data.

    Returns:
        The canonical team name string.

    Raises:
        ValueError: If the team is not found. The message lists all
            valid teams to help the caller correct the input.
    """
    name_stripped = name.strip()
    for valid in valid_teams:
        if valid.lower() == name_stripped.lower():
            return valid

    # Try fuzzy match for a helpful suggestion
    suggestions = get_close_matches(
        name_stripped, list(valid_teams), n=5, cutoff=0.5
    )
    msg = f"Unknown team: '{name_stripped}'"
    if suggestions:
        msg += f". Did you mean: {', '.join(suggestions)}?"
    else:
        msg += f". Valid teams ({len(valid_teams)}): {', '.join(sorted(valid_teams))}"
    raise ValueError(msg)
