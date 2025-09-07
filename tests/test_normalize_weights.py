"""Tests for normalize_weights helper."""

import math
import pytest
from tnfr.collections_utils import normalize_weights


def test_normalize_weights_warns_on_negative_value(caplog):
    weights = {"a": -1.0, "b": 2.0}
    with caplog.at_level("WARNING"):
        norm = normalize_weights(weights, ("a", "b"))
    assert any("Negative weights" in m for m in caplog.messages)
    assert norm["a"] == 0.0
    assert math.isclose(norm["b"], 1.0)


def test_normalize_weights_raises_on_negative_value():
    weights = {"a": -1.0, "b": 2.0}
    with pytest.raises(ValueError):
        normalize_weights(weights, ("a", "b"), error_on_negative=True)


def test_normalize_weights_warns_on_negative_default(caplog):
    with caplog.at_level("WARNING"):
        normalize_weights({}, ("a", "b"), default=-0.5)
    assert any("Negative weights" in m for m in caplog.messages)


def test_normalize_weights_raises_on_negative_default():
    with pytest.raises(ValueError):
        normalize_weights({}, ("a", "b"), default=-0.5, error_on_negative=True)


def test_normalize_weights_warns_on_non_numeric_value(caplog):
    weights = {"a": "not-a-number", "b": 2.0}
    with caplog.at_level("WARNING"):
        norm = normalize_weights(weights, ("a", "b"), default=1.0)
    assert any("Could not convert" in m for m in caplog.messages)
    assert math.isclose(math.fsum(norm.values()), 1.0)
    assert norm == pytest.approx({"a": 1 / 3, "b": 2 / 3})


def test_normalize_weights_raises_on_non_numeric_value():
    weights = {"a": "not-a-number", "b": 2.0}
    with pytest.raises(ValueError):
        normalize_weights(weights, ("a", "b"), error_on_negative=True)


def test_normalize_weights_high_precision():
    weights = {str(i): 0.1 for i in range(10)}
    norm = normalize_weights(weights, weights.keys())
    assert all(v == 0.1 for v in norm.values())
    assert math.isclose(math.fsum(norm.values()), 1.0)
