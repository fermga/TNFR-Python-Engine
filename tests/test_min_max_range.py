import pytest
from tnfr.metrics_utils import min_max_range


def test_min_max_range_generator():
    vals = (x for x in [1.0, 2.0, -1.0])
    assert min_max_range(vals) == pytest.approx((-1.0, 2.0))


def test_min_max_range_empty_generator_returns_default():
    vals = (x for x in [])
    assert min_max_range(vals, default=(-2.0, 1.0)) == pytest.approx((-2.0, 1.0))
