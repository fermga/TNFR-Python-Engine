"""Tests for _state_from_thresholds."""

from tnfr.metrics.diagnosis import _state_from_thresholds
from tnfr.config.operator_names import TRANSITION


def test_state_from_thresholds_checks_all_conditions():
    cfg = {
        "stable": {"Rloc_hi": 0.8, "dnfr_lo": 0.2},
        "dissonance": {"Rloc_lo": 0.4, "dnfr_hi": 0.5},
    }
    assert _state_from_thresholds(0.9, 0.1, cfg) == "estable"
    assert _state_from_thresholds(0.3, 0.6, cfg) == "disonante"
    assert _state_from_thresholds(0.5, 0.3, cfg) == TRANSITION
