"""Tests for normalize_weights helper."""

import math
from types import MappingProxyType

import pytest

from tnfr.utils import negative_weights_warn_once, normalize_weights


def test_normalize_weights_warns_on_negative_value(caplog):
    weights = {"warn-negative-value-a": -1.0, "warn-negative-value-b": 2.0}
    with caplog.at_level("WARNING"):
        norm = normalize_weights(
            weights,
            ("warn-negative-value-a", "warn-negative-value-b"),
        )
    assert any("Negative weights" in m for m in caplog.messages)
    assert norm["warn-negative-value-a"] == 0.0
    assert math.isclose(norm["warn-negative-value-b"], 1.0)


def test_normalize_weights_raises_on_negative_value():
    weights = {"a": -1.0, "b": 2.0}
    with pytest.raises(ValueError):
        normalize_weights(weights, ("a", "b"), error_on_negative=True)


def test_normalize_weights_warns_on_negative_default(caplog):
    with caplog.at_level("WARNING"):
        normalize_weights(
            {}, ("warn-negative-default-a", "warn-negative-default-b"), default=-0.5
        )
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
    first_key = "warn-once-key-1"
    weights = {first_key: -1.0}
    warn_once = negative_weights_warn_once()
    with caplog.at_level("WARNING"):
        normalize_weights(weights, (first_key,), warn_once=warn_once)
    assert any("Negative weights" in m for m in caplog.messages)
    caplog.clear()

    # second call with same key should not warn
    with caplog.at_level("WARNING"):
        normalize_weights(weights, (first_key,), warn_once=warn_once)
    assert not any("Negative weights" in m for m in caplog.messages)

    # new keys should still trigger warnings
    caplog.clear()
    second_key = "warn-once-key-2"
    with caplog.at_level("WARNING"):
        normalize_weights(
            {second_key: -1.0},
            (second_key,),
            warn_once=warn_once,
        )
    assert any("Negative weights" in m for m in caplog.messages)


def test_normalize_weights_warns_each_time_when_dedup_disabled(caplog):
    key = "warn-disabled-key"
    weights = {key: -1.0}

    with caplog.at_level("WARNING"):
        normalize_weights(weights, (key,), warn_once=False)
    first_messages = [m for m in caplog.messages if "Negative weights" in m]
    assert first_messages, "First invocation should warn about negative weights"

    caplog.clear()

    with caplog.at_level("WARNING"):
        normalize_weights(weights, (key,), warn_once=False)
    second_messages = [m for m in caplog.messages if "Negative weights" in m]
    assert second_messages, "Second invocation should warn when deduplication disabled"


def test_normalize_weights_raises_on_non_numeric_value():
    weights = {"a": "not-a-number", "b": 2.0}
    with pytest.raises(ValueError):
        normalize_weights(weights, ("a", "b"), error_on_conversion=True)


def test_normalize_weights_error_on_negative_does_not_raise_conversion(caplog):
    """error_on_negative should not affect conversion errors."""
    weights = {"a": "not-a-number", "b": 2.0}
    with caplog.at_level("WARNING"):
        norm = normalize_weights(
            weights, ("a", "b"), error_on_negative=True, default=1.0
        )
    assert any("Could not convert" in m for m in caplog.messages)
    assert math.isclose(math.fsum(norm.values()), 1.0)


def test_normalize_weights_high_precision():
    weights = {str(i): 0.1 for i in range(10)}
    norm = normalize_weights(weights, weights.keys())
    assert all(v == 0.1 for v in norm.values())
    assert math.isclose(math.fsum(norm.values()), 1.0)


def test_normalize_weights_deduplicates_keys():
    weights = {"dedup-a": -1.0, "dedup-b": -1.0}
    dup_keys = ["dedup-a", "dedup-b", "dedup-a"]
    unique_keys = ["dedup-a", "dedup-b"]
    norm_dup = normalize_weights(weights, dup_keys)
    norm_unique = normalize_weights(weights, unique_keys)
    expected = {"dedup-a": 0.5, "dedup-b": 0.5}
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
