"""Tests for normalize_weights helper."""
import logging
from tnfr.helpers import normalize_weights


def test_normalize_weights_warns_on_negative_value(caplog):
    weights = {"a": -1.0, "b": 2.0}
    with caplog.at_level("WARNING"):
        normalize_weights(weights, ("a", "b"))
    assert any("Pesos negativos" in m for m in caplog.messages)


def test_normalize_weights_warns_on_negative_default(caplog):
    with caplog.at_level("WARNING"):
        normalize_weights({}, ("a", "b"), default=-0.5)
    assert any("Pesos negativos" in m for m in caplog.messages)
