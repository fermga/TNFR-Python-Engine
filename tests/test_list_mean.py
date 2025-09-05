import pytest
from tnfr.helpers import list_mean


def test_list_mean_non_empty():
    assert list_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_list_mean_empty_returns_default():
    assert list_mean([], default=5.0) == 5.0


def test_list_mean_generator():
    assert list_mean(x for x in (1.0, 2.0, 3.0)) == pytest.approx(2.0)


def test_list_mean_empty_generator_returns_default():
    assert list_mean((x for x in ()), default=3.3) == pytest.approx(3.3)
