import math
import pytest

from tnfr.metrics_utils import _mean_phase


def test_mean_phase_without_numpy():
    neigh = [1, 2]
    cos_th = {1: math.cos(0.0), 2: math.cos(math.pi / 2)}
    sin_th = {1: math.sin(0.0), 2: math.sin(math.pi / 2)}
    th_i = 0.0
    th_bar = _mean_phase(neigh, cos_th, sin_th, None, th_i)
    assert th_bar == pytest.approx(math.pi / 4)


def test_mean_phase_with_numpy_stub():
    neigh = [1, 2]
    cos_th = {1: math.cos(0.0), 2: math.cos(math.pi / 2)}
    sin_th = {1: math.sin(0.0), 2: math.sin(math.pi / 2)}
    th_i = 0.0

    class DummyNP:
        def fromiter(self, iterable, dtype, count):
            vals = list(iterable)
            class Arr(list):
                def mean(self):
                    return sum(self) / len(self)
            return Arr(vals)

        def arctan2(self, y, x):
            return math.atan2(y, x)

    th_bar = _mean_phase(neigh, cos_th, sin_th, DummyNP(), th_i)
    assert th_bar == pytest.approx(math.pi / 4)
