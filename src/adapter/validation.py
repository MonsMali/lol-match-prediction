"""Draft validation for the LoL Draft Adapter.

Validates team names, champion names, role keys, ban counts, and
duplicate champions before feature computation.  All validation
errors are ``ValueError`` with descriptive messages.
"""

from src.adapter.normalization import normalize_champion_name, normalize_team_name
from src.adapter.schemas import DraftInput, VALID_ROLES


def validate_draft(
    draft: DraftInput,
    valid_champions: set[str],
    valid_teams: set[str],
    aliases: dict[str, str],
) -> DraftInput:
    """Validate and normalize a draft input.

    Resolution order mirrors the user's mental model: teams first,
    then picks, then bans, then structural checks (roles, counts,
    duplicates).  The first failing check raises immediately so the
    caller gets the most actionable error.

    Args:
        draft: Raw draft input from the caller.
        valid_champions: Canonical champion names from training data.
        valid_teams: Canonical team names from training data.
        aliases: Champion alias map for normalization.

    Returns:
        A **new** ``DraftInput`` with all names resolved to their
        canonical forms.

    Raises:
        ValueError: On any validation failure.
    """
    # 1. Validate and normalize team names
    blue_team = normalize_team_name(draft.blue_team, valid_teams)
    red_team = normalize_team_name(draft.red_team, valid_teams)

    # 2. Validate role keys
    blue_roles = set(draft.blue_picks.keys())
    red_roles = set(draft.red_picks.keys())

    if blue_roles != VALID_ROLES:
        missing = VALID_ROLES - blue_roles
        extra = blue_roles - VALID_ROLES
        parts: list[str] = []
        if missing:
            parts.append(f"missing roles: {', '.join(sorted(missing))}")
        if extra:
            parts.append(f"extra roles: {', '.join(sorted(extra))}")
        raise ValueError(
            f"Blue picks have invalid roles ({'; '.join(parts)}). "
            f"Required: {', '.join(sorted(VALID_ROLES))}"
        )

    if red_roles != VALID_ROLES:
        missing = VALID_ROLES - red_roles
        extra = red_roles - VALID_ROLES
        parts = []
        if missing:
            parts.append(f"missing roles: {', '.join(sorted(missing))}")
        if extra:
            parts.append(f"extra roles: {', '.join(sorted(extra))}")
        raise ValueError(
            f"Red picks have invalid roles ({'; '.join(parts)}). "
            f"Required: {', '.join(sorted(VALID_ROLES))}"
        )

    # 3. Validate ban counts
    if len(draft.blue_bans) != 5:
        raise ValueError(
            f"Blue bans must be exactly 5, got {len(draft.blue_bans)}"
        )
    if len(draft.red_bans) != 5:
        raise ValueError(
            f"Red bans must be exactly 5, got {len(draft.red_bans)}"
        )

    # 4. Normalize and validate all champion names (picks)
    blue_picks: dict[str, str] = {}
    for role in sorted(VALID_ROLES):
        blue_picks[role] = normalize_champion_name(
            draft.blue_picks[role], valid_champions, aliases
        )

    red_picks: dict[str, str] = {}
    for role in sorted(VALID_ROLES):
        red_picks[role] = normalize_champion_name(
            draft.red_picks[role], valid_champions, aliases
        )

    # 5. Normalize and validate all champion names (bans)
    blue_bans = [
        normalize_champion_name(b, valid_champions, aliases)
        for b in draft.blue_bans
    ]
    red_bans = [
        normalize_champion_name(b, valid_champions, aliases)
        for b in draft.red_bans
    ]

    # 6. Check for duplicate champions across ALL picks and bans
    all_champions: list[str] = (
        list(blue_picks.values())
        + list(red_picks.values())
        + blue_bans
        + red_bans
    )
    seen: set[str] = set()
    duplicates: list[str] = []
    for champ in all_champions:
        if champ in seen:
            duplicates.append(champ)
        seen.add(champ)

    if duplicates:
        unique_dupes = sorted(set(duplicates))
        raise ValueError(
            f"Duplicate champions in draft: {', '.join(unique_dupes)}. "
            "Each champion may appear at most once across all picks and bans."
        )

    # 7. Return normalized draft
    return DraftInput(
        blue_team=blue_team,
        red_team=red_team,
        blue_picks=blue_picks,
        red_picks=red_picks,
        blue_bans=blue_bans,
        red_bans=red_bans,
        patch=draft.patch,
    )
