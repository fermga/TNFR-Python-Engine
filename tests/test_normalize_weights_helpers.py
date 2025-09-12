import math
import pytest
import tnfr.collections_utils as cu


def test_convert_weights_dedup_and_defaults():
    weights, keys = cu._convert_weights({"a": "1", "c": "bad"}, ["a", "b", "c", "a"], 2.0, error_on_conversion=False)
    assert keys == ["a", "b", "c"]
    assert weights == {"a": 1.0, "b": 2.0, "c": 2.0}


def test_convert_weights_error_on_conversion():
    with pytest.raises(ValueError):
        cu._convert_weights({"a": "bad"}, ["a"], 0.0, error_on_conversion=True)


def test_handle_negative_weights_clamps(caplog):
    cu.clear_warned_negative_keys()
    weights = {"a": -1.0, "b": 2.0}
    with caplog.at_level("WARNING"):
        total = cu._handle_negative_weights(weights, error_on_negative=False, warn_once=False)
    assert total == 2.0
    assert weights["a"] == 0.0
    assert any("Negative weights" in m for m in caplog.messages)


def test_handle_negative_weights_raises():
    with pytest.raises(ValueError):
        cu._handle_negative_weights({"a": -1.0}, error_on_negative=True, warn_once=True)


def test_normalize_non_negative_uniform():
    weights = {"a": 0.0, "b": 0.0}
    keys = ["a", "b"]
    norm = cu._normalize_non_negative_weights(weights, keys, 0.0)
    assert norm == {"a": 0.5, "b": 0.5}


def test_normalize_non_negative_divides():
    weights = {"a": 1.0, "b": 3.0}
    keys = ["a", "b"]
    norm = cu._normalize_non_negative_weights(weights, keys, 4.0)
    assert math.isclose(norm["a"], 0.25)
    assert math.isclose(norm["b"], 0.75)
