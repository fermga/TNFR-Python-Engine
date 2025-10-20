"""Tests for _state_from_thresholds."""

import pytest

from tnfr.constants import (
    STATE_DISSONANT,
    STATE_STABLE,
    STATE_TRANSITION,
    disable_spanish_state_tokens,
    enable_spanish_state_tokens,
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


def test_normalise_state_token_ignores_spanish_tokens_without_opt_in():
    disable_spanish_state_tokens()
    for legacy_token in ("estable", "disonante", "transicion", "transición"):
        assert normalise_state_token(legacy_token) == legacy_token


def test_enable_spanish_state_tokens_emits_futurewarning():
    disable_spanish_state_tokens()
    with pytest.warns(FutureWarning, match="Spanish state tokens require explicit opt-in"):
        enable_spanish_state_tokens()


def test_normalise_state_token_maps_spanish_values_when_enabled():
    disable_spanish_state_tokens()
    enable_spanish_state_tokens(warn=False)
    try:
        for legacy_token, canonical in (
            ("estable", STATE_STABLE),
            ("disonante", STATE_DISSONANT),
            ("transicion", STATE_TRANSITION),
            ("transición", STATE_TRANSITION),
        ):
            with pytest.warns(FutureWarning, match="Spanish state token"):
                assert normalise_state_token(legacy_token) == canonical
    finally:
        disable_spanish_state_tokens()
