import pytest

from tnfr.collections_utils import prepare_weights


def test_prepare_weights_clamps_and_dedups():
    weights, keys, total = prepare_weights(
        {"a": 1, "b": -2},
        ["a", "b", "c", "a"],
        0.5,
        error_on_conversion=False,
        error_on_negative=False,
        warn_once=True,
    )
    assert keys == ["a", "b", "c"]
    assert weights == {"a": 1.0, "b": 0.0, "c": 0.5}
    assert total == pytest.approx(1.5)
