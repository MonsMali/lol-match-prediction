"""End-to-end tests for the LoL Draft Adapter.

Validates prediction correctness, memory footprint, validation error paths,
and the no-CSV guarantee required for Render free-tier deployment.

All tests share a module-scoped adapter fixture to avoid redundant artifact
loading (the adapter is a singleton anyway, but the fixture makes the intent
explicit and provides pre-computed test data).
"""

import inspect
import time

import pytest

from src.adapter.adapter import LoLDraftAdapter
from src.adapter.schemas import AdapterStatus, DraftInput, PredictionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def adapter():
    """Return the singleton adapter instance."""
    return LoLDraftAdapter()


@pytest.fixture(scope="module")
def teams(adapter):
    """Return two valid team names from the training data."""
    return sorted(adapter.valid_teams)[:2]


@pytest.fixture(scope="module")
def champions(adapter):
    """Return 20 valid champion names for constructing drafts."""
    return sorted(adapter.valid_champions)[:20]


def _make_draft(teams, champions, offset=0):
    """Build a DraftInput from sorted team/champion lists.

    Args:
        teams: List of at least 2 team names.
        champions: List of at least ``offset + 20`` champion names.
        offset: Starting index into the champion list.
    """
    c = champions
    o = offset
    return DraftInput(
        blue_team=teams[0],
        red_team=teams[1],
        blue_picks={
            "top": c[o], "jungle": c[o + 1], "mid": c[o + 2],
            "bot": c[o + 3], "support": c[o + 4],
        },
        red_picks={
            "top": c[o + 5], "jungle": c[o + 6], "mid": c[o + 7],
            "bot": c[o + 8], "support": c[o + 9],
        },
        blue_bans=[c[o + 10], c[o + 11], c[o + 12], c[o + 13], c[o + 14]],
        red_bans=[c[o + 15], c[o + 16], c[o + 17], c[o + 18], c[o + 19]],
    )


# ---------------------------------------------------------------------------
# Tests: Singleton
# ---------------------------------------------------------------------------

def test_adapter_singleton():
    """LoLDraftAdapter returns the same instance on repeated calls."""
    a1 = LoLDraftAdapter()
    a2 = LoLDraftAdapter()
    assert a1 is a2


# ---------------------------------------------------------------------------
# Tests: Status / Health
# ---------------------------------------------------------------------------

def test_get_status(adapter):
    """get_status returns an AdapterStatus with plausible diagnostics."""
    status = adapter.get_status()
    assert isinstance(status, AdapterStatus)
    assert status.champion_count > 100
    assert status.team_count > 10
    assert status.memory_usage_mb > 0
    assert isinstance(status.model_name, str) and len(status.model_name) > 0
    assert isinstance(status.model_version, str) and len(status.model_version) > 0


# ---------------------------------------------------------------------------
# Tests: Prediction correctness
# ---------------------------------------------------------------------------

def test_predict_from_draft_basic(adapter, teams, champions):
    """Prediction returns valid probabilities that sum to 1.0."""
    draft = _make_draft(teams, champions)
    result = adapter.predict_from_draft(draft)

    assert isinstance(result, PredictionResult)
    assert 0.0 < result.blue_win_prob < 1.0
    assert 0.0 < result.red_win_prob < 1.0
    assert abs(result.blue_win_prob + result.red_win_prob - 1.0) < 1e-6
    assert isinstance(result.model_name, str) and len(result.model_name) > 0
    assert isinstance(result.model_version, str) and len(result.model_version) > 0


def test_predict_swapped_teams_changes_result(adapter):
    """Swapping blue and red sides produces a different prediction."""
    teams = sorted(adapter.valid_teams)[:2]
    champs = sorted(adapter.valid_champions)[:20]

    draft_a = DraftInput(
        blue_team=teams[0], red_team=teams[1],
        blue_picks={
            "top": champs[0], "jungle": champs[1], "mid": champs[2],
            "bot": champs[3], "support": champs[4],
        },
        red_picks={
            "top": champs[5], "jungle": champs[6], "mid": champs[7],
            "bot": champs[8], "support": champs[9],
        },
        blue_bans=[champs[10], champs[11], champs[12], champs[13], champs[14]],
        red_bans=[champs[15], champs[16], champs[17], champs[18], champs[19]],
    )

    # Swap: team B goes blue, team A goes red; picks follow their teams
    draft_b = DraftInput(
        blue_team=teams[1], red_team=teams[0],
        blue_picks={
            "top": champs[5], "jungle": champs[6], "mid": champs[7],
            "bot": champs[8], "support": champs[9],
        },
        red_picks={
            "top": champs[0], "jungle": champs[1], "mid": champs[2],
            "bot": champs[3], "support": champs[4],
        },
        blue_bans=[champs[15], champs[16], champs[17], champs[18], champs[19]],
        red_bans=[champs[10], champs[11], champs[12], champs[13], champs[14]],
    )

    result_a = adapter.predict_from_draft(draft_a)
    result_b = adapter.predict_from_draft(draft_b)

    assert result_a.blue_win_prob != result_b.blue_win_prob, (
        "Swapping blue/red should change the prediction"
    )


def test_predict_not_degenerate(adapter):
    """Different drafts produce different, non-trivial predictions."""
    all_teams = sorted(adapter.valid_teams)
    all_champs = sorted(adapter.valid_champions)

    # Three drafts with different teams and champions
    drafts = []
    for i in range(3):
        t = all_teams[i * 2: i * 2 + 2]
        c = all_champs[i * 20: i * 20 + 20]
        drafts.append(_make_draft(t, c))

    results = [adapter.predict_from_draft(d) for d in drafts]
    probs = [r.blue_win_prob for r in results]

    # Not all identical
    assert len(set(probs)) > 1, (
        f"All predictions are identical ({probs[0]:.4f}), features may be broken"
    )

    # None is exactly 0.5 (would indicate all features defaulting)
    for p in probs:
        assert p != 0.5, "Prediction is exactly 0.5, features may all be defaulting"


def test_predict_completes_under_one_second(adapter, teams, champions):
    """A single prediction completes within 1 second."""
    draft = _make_draft(teams, champions)

    start = time.monotonic()
    adapter.predict_from_draft(draft)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, f"Prediction took {elapsed:.2f}s, expected < 1s"


# ---------------------------------------------------------------------------
# Tests: Validation error paths
# ---------------------------------------------------------------------------

def test_validation_unknown_champion(adapter, teams, champions):
    """An unknown champion raises ValueError with fuzzy suggestion."""
    draft = DraftInput(
        blue_team=teams[0], red_team=teams[1],
        blue_picks={
            "top": "Aatro",  # close to Aatrox -- triggers "Did you mean"
            "jungle": champions[1], "mid": champions[2],
            "bot": champions[3], "support": champions[4],
        },
        red_picks={
            "top": champions[5], "jungle": champions[6], "mid": champions[7],
            "bot": champions[8], "support": champions[9],
        },
        blue_bans=champions[10:15],
        red_bans=champions[15:20],
    )

    with pytest.raises(ValueError, match="Unknown champion"):
        adapter.predict_from_draft(draft)

    # Verify the suggestion mechanism works
    try:
        adapter.predict_from_draft(draft)
    except ValueError as exc:
        assert "Did you mean" in str(exc), (
            f"Expected fuzzy suggestion, got: {exc}"
        )


def test_validation_unknown_team(adapter, champions):
    """An unknown team raises ValueError mentioning the invalid name."""
    draft = DraftInput(
        blue_team="FakeTeam123",
        red_team="AnotherFakeTeam",
        blue_picks={
            "top": champions[0], "jungle": champions[1], "mid": champions[2],
            "bot": champions[3], "support": champions[4],
        },
        red_picks={
            "top": champions[5], "jungle": champions[6], "mid": champions[7],
            "bot": champions[8], "support": champions[9],
        },
        blue_bans=champions[10:15],
        red_bans=champions[15:20],
    )

    with pytest.raises(ValueError, match="FakeTeam123"):
        adapter.predict_from_draft(draft)


def test_validation_duplicate_champion(adapter, teams, champions):
    """A champion appearing in both blue and red picks raises ValueError."""
    draft = DraftInput(
        blue_team=teams[0], red_team=teams[1],
        blue_picks={
            "top": champions[0], "jungle": champions[1], "mid": champions[2],
            "bot": champions[3], "support": champions[4],
        },
        red_picks={
            "top": champions[0],  # duplicate: same as blue top
            "jungle": champions[6], "mid": champions[7],
            "bot": champions[8], "support": champions[9],
        },
        blue_bans=champions[10:15],
        red_bans=champions[15:20],
    )

    with pytest.raises(ValueError, match="(?i)duplicate"):
        adapter.predict_from_draft(draft)


def test_validation_missing_role(adapter, teams, champions):
    """Missing a required role key raises ValueError."""
    draft = DraftInput(
        blue_team=teams[0], red_team=teams[1],
        blue_picks={
            "top": champions[0], "jungle": champions[1],
            "mid": champions[2], "bot": champions[3],
            # "support" deliberately omitted
        },
        red_picks={
            "top": champions[5], "jungle": champions[6], "mid": champions[7],
            "bot": champions[8], "support": champions[9],
        },
        blue_bans=champions[10:15],
        red_bans=champions[15:20],
    )

    with pytest.raises(ValueError, match="(?i)role"):
        adapter.predict_from_draft(draft)


def test_validation_wrong_ban_count(adapter, teams, champions):
    """Fewer than 5 bans raises ValueError."""
    draft = DraftInput(
        blue_team=teams[0], red_team=teams[1],
        blue_picks={
            "top": champions[0], "jungle": champions[1], "mid": champions[2],
            "bot": champions[3], "support": champions[4],
        },
        red_picks={
            "top": champions[5], "jungle": champions[6], "mid": champions[7],
            "bot": champions[8], "support": champions[9],
        },
        blue_bans=[champions[10], champions[11], champions[12]],  # only 3
        red_bans=champions[15:20],
    )

    with pytest.raises(ValueError, match="5"):
        adapter.predict_from_draft(draft)


# ---------------------------------------------------------------------------
# Tests: Static guarantees
# ---------------------------------------------------------------------------

def test_no_csv_loading():
    """The adapter module does not contain any CSV loading code."""
    import src.adapter.adapter as adapter_mod
    import src.adapter.features as features_mod
    import src.adapter.validation as validation_mod
    import src.adapter.normalization as normalization_mod

    for mod in [adapter_mod, features_mod, validation_mod, normalization_mod]:
        source = inspect.getsource(mod)
        assert "read_csv" not in source, (
            f"{mod.__name__} contains 'read_csv' -- no CSV loading allowed"
        )


def test_memory_under_512mb(adapter):
    """Process RSS stays under the 512 MB Render free-tier budget."""
    status = adapter.get_status()
    assert status.memory_usage_mb < 512, (
        f"Memory usage {status.memory_usage_mb:.1f} MB exceeds 512 MB limit"
    )
