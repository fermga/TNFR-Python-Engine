import pytest

from tnfr.metrics.trig import accumulate_cos_sin


def test_accumulate_cos_sin_accumulates_pairs_and_ignores_missing():
    pairs = [(1.0, 0.0), None, (0.0, 1.0), (None, 0.5), (1.0, None)]
    sum_cos, sum_sin, processed = accumulate_cos_sin(pairs)
    assert processed is True
    assert sum_cos == pytest.approx(1.0)
    assert sum_sin == pytest.approx(1.0)


def test_accumulate_cos_sin_handles_empty_iterable():
    sum_cos, sum_sin, processed = accumulate_cos_sin([None, (None, None)])
    assert processed is False
    assert sum_cos == pytest.approx(0.0)
    assert sum_sin == pytest.approx(0.0)


def test_accumulate_cos_sin_maintains_precision_with_large_values():
    large = 1e16
    pairs = iter([(large, 0.0), (-large, 0.0), (1.0, 0.0)])
    sum_cos, sum_sin, processed = accumulate_cos_sin(pairs)
    assert processed is True
    assert sum_cos == pytest.approx(1.0)
    assert sum_sin == pytest.approx(0.0)
