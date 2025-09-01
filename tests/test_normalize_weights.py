"""Tests for normalize_weights helper."""
import logging
import math
import pytest
from tnfr.helpers import normalize_weights


def test_normalize_weights_warns_on_negative_value(caplog):
    weights = {"a": -1.0, "b": 2.0}
    with caplog.at_level("WARNING"):
        normalize_weights(weights, ("a", "b"))
    assert any("Pesos negativos" in m for m in caplog.messages)


def test_normalize_weights_raises_on_negative_value():
    weights = {"a": -1.0, "b": 2.0}
    with pytest.raises(ValueError):
        normalize_weights(weights, ("a", "b"), error_on_negative=True)


def test_normalize_weights_warns_on_negative_default(caplog):
    with caplog.at_level("WARNING"):
        normalize_weights({}, ("a", "b"), default=-0.5)
    assert any("Pesos negativos" in m for m in caplog.messages)


def test_normalize_weights_raises_on_negative_default():
    with pytest.raises(ValueError):
        normalize_weights({}, ("a", "b"), default=-0.5, error_on_negative=True)


def test_normalize_weights_high_precision():
    weights = {str(i): 0.1 for i in range(10)}
    norm = normalize_weights(weights, weights.keys())
    assert all(v == 0.1 for v in norm.values())
    assert math.isclose(math.fsum(norm.values()), 1.0)
