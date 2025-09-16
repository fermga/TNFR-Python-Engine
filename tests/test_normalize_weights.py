"""Tests for normalize_weights helper."""

import logging
import math
from types import MappingProxyType

import pytest
import tnfr.collections_utils as cu
from tnfr.collections_utils import normalize_weights
from tnfr.logging_utils import warn_once as create_warn_once


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


def test_normalize_weights_warn_once(caplog):
    weights = {"x": -1.0}
    logger = logging.getLogger("tnfr.collections_utils")
    warn_once = create_warn_once(logger, cu.NEGATIVE_WEIGHTS_MSG)

    with caplog.at_level("WARNING"):
        normalize_weights(weights, ("x",), warn_once=warn_once)
    assert any("Negative weights" in m for m in caplog.messages)
    caplog.clear()

    # second call with same key should not warn
    with caplog.at_level("WARNING"):
        normalize_weights(weights, ("x",), warn_once=warn_once)
    assert not any("Negative weights" in m for m in caplog.messages)

    # creating a new warn-once callable should allow warning again
    caplog.clear()
    fresh_warn_once = create_warn_once(logger, cu.NEGATIVE_WEIGHTS_MSG)
    with caplog.at_level("WARNING"):
        normalize_weights(weights, ("x",), warn_once=fresh_warn_once)
    assert any("Negative weights" in m for m in caplog.messages)


def test_normalize_weights_raises_on_non_numeric_value():
    weights = {"a": "not-a-number", "b": 2.0}
    with pytest.raises(ValueError):
        normalize_weights(weights, ("a", "b"), error_on_conversion=True)


def test_normalize_weights_error_on_negative_does_not_raise_conversion(caplog):
    """error_on_negative should not affect conversion errors."""
    weights = {"a": "not-a-number", "b": 2.0}
    with caplog.at_level("WARNING"):
        norm = normalize_weights(weights, ("a", "b"), error_on_negative=True, default=1.0)
    assert any("Could not convert" in m for m in caplog.messages)
    assert math.isclose(math.fsum(norm.values()), 1.0)


def test_normalize_weights_high_precision():
    weights = {str(i): 0.1 for i in range(10)}
    norm = normalize_weights(weights, weights.keys())
    assert all(v == 0.1 for v in norm.values())
    assert math.isclose(math.fsum(norm.values()), 1.0)


def test_normalize_weights_deduplicates_keys():
    weights = {"a": -1.0, "b": -1.0}
    dup_keys = ["a", "b", "a"]
    unique_keys = ["a", "b"]
    norm_dup = normalize_weights(weights, dup_keys)
    norm_unique = normalize_weights(weights, unique_keys)
    expected = {"a": 0.5, "b": 0.5}
    assert norm_dup == norm_unique
    assert norm_dup == pytest.approx(expected)


def test_normalize_weights_dedup_and_defaults():
    weights = {"a": "1", "c": "bad"}
    norm = normalize_weights(weights, ["a", "b", "c", "a"], default=2.0)
    assert math.isclose(math.fsum(norm.values()), 1.0)
    assert norm == pytest.approx({"a": 1 / 5, "b": 2 / 5, "c": 2 / 5})


def test_normalize_weights_accepts_mapping_proxy():
    weights = MappingProxyType({"a": 1.0, "b": 2.0})
    norm = normalize_weights(weights, weights.keys())
    assert norm == pytest.approx({"a": 1 / 3, "b": 2 / 3})
