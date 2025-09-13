import math
import pytest
import tnfr.collections_utils as cu


def test_prepare_weights_dedup_and_defaults():
    weights, keys, total = cu._prepare_weights(
        {"a": "1", "c": "bad"}, ["a", "b", "c", "a"], 2.0,
        error_on_conversion=False,
        error_on_negative=False,
        warn_once=False,
    )
    assert keys == ["a", "b", "c"]
    assert weights == {"a": 1.0, "b": 2.0, "c": 2.0}
    assert math.isclose(total, 5.0)


def test_prepare_weights_error_on_conversion():
    with pytest.raises(ValueError):
        cu._prepare_weights(
            {"a": "bad"}, ["a"], 0.0,
            error_on_conversion=True,
            error_on_negative=False,
            warn_once=False,
        )


def test_prepare_weights_clamps_negatives(caplog):
    cu.clear_warned_negative_keys()
    weights, keys, total = cu._prepare_weights(
        {"a": -1.0, "b": 2.0}, ["a", "b"], 0.0,
        error_on_conversion=False,
        error_on_negative=False,
        warn_once=False,
    )
    assert any("Negative weights" in m for m in caplog.messages)
    assert weights == {"a": 0.0, "b": 2.0}
    assert keys == ["a", "b"]
    assert total == 2.0


def test_prepare_weights_raises_on_negative():
    with pytest.raises(ValueError):
        cu._prepare_weights(
            {"a": -1.0}, ["a"], 0.0,
            error_on_conversion=False,
            error_on_negative=True,
            warn_once=True,
        )
