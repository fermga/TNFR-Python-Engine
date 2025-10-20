"""Tests for _state_from_thresholds."""

import pytest

from tnfr.constants import (
    STATE_DISSONANT,
    STATE_STABLE,
    STATE_TRANSITION,
    normalise_state_token,
)
from tnfr.metrics.diagnosis import _state_from_thresholds


def test_state_from_thresholds_checks_all_conditions():
    cfg = {
        "stable": {"Rloc_hi": 0.8, "dnfr_lo": 0.2},
        "dissonance": {"Rloc_lo": 0.4, "dnfr_hi": 0.5},
    }
    assert _state_from_thresholds(0.9, 0.1, cfg) == STATE_STABLE
    assert _state_from_thresholds(0.3, 0.6, cfg) == STATE_DISSONANT
    assert _state_from_thresholds(0.5, 0.3, cfg) == STATE_TRANSITION


def test_normalise_state_token_accepts_canonical_tokens_without_warning():
    for token in (STATE_STABLE, STATE_DISSONANT, STATE_TRANSITION):
        assert normalise_state_token(token) == token


@pytest.mark.parametrize(
    "legacy_token",
    ["estable", "disonante", "transicion", "transición"],
)
def test_normalise_state_token_rejects_spanish_tokens(legacy_token: str):
    with pytest.raises(ValueError, match="state token must be one of"):
        normalise_state_token(legacy_token)


def test_normalise_state_token_rejects_unknown_tokens():
    with pytest.raises(ValueError, match="state token must be one of"):
        normalise_state_token("mystery")
