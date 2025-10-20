import math
import pytest

import tnfr.metrics.trig as trig
from tnfr.metrics.trig import (
    neighbor_phase_mean_list,
    _neighbor_phase_mean_core,
)


def test_neighbor_phase_mean_core_missing_trig():
    neigh = [1, 2, 3]
    cos_th = {1: 1.0, 2: 0.0}
    sin_th = {1: 0.0, 2: 1.0}

    angle = _neighbor_phase_mean_core(neigh, cos_th, sin_th, np=None, fallback=0.5)
    assert angle == pytest.approx(math.pi / 4)

    assert _neighbor_phase_mean_core([3], cos_th, sin_th, np=None, fallback=0.5) == pytest.approx(
        0.5
    )


def test_neighbor_phase_mean_list_delegates_generic(monkeypatch):
    neigh = [1]
    cos_th = {1: 1.0}
    sin_th = {1: 0.0}
    captured = {}

    def fake_generic(neigh_arg, cos_map=None, sin_map=None, np=None, fallback=0.0):
        captured["args"] = (neigh_arg, cos_map, sin_map, np, fallback)
        return 1.23

    monkeypatch.setattr(
        "tnfr.metrics.trig._neighbor_phase_mean_generic", fake_generic
    )
    result = neighbor_phase_mean_list(neigh, cos_th, sin_th, np=None, fallback=0.0)
    assert result == pytest.approx(1.23)
    assert captured["args"] == (neigh, cos_th, sin_th, None, 0.0)


def test_neighbor_phase_mean_generic_uses_cached_numpy(monkeypatch):
    calls = 0

    class FakeArray(list):
        @property
        def size(self):
            return len(self)

    class FakeNumpy:
        def fromiter(self, iterable, dtype=float):
            return FakeArray(list(iterable))

        def mean(self, arr):
            if not arr:
                return 0.0
            return sum(arr) / len(arr)

        def arctan2(self, y, x):
            return math.atan2(y, x)

    fake_np = FakeNumpy()

    def fake_get_numpy():
        nonlocal calls
        calls += 1
        return fake_np

    monkeypatch.setattr(trig, "get_numpy", fake_get_numpy)

    original_core = trig._neighbor_phase_mean_core
    captured_np = []

    def wrapped_core(neigh, cos_map, sin_map, np_arg, fallback):
        captured_np.append(np_arg)
        return original_core(neigh, cos_map, sin_map, np_arg, fallback)

    monkeypatch.setattr(trig, "_neighbor_phase_mean_core", wrapped_core)

    result = trig._neighbor_phase_mean_generic(
        [1, 2],
        cos_map={1: 1.0, 2: 0.0},
        sin_map={1: 0.0, 2: 1.0},
        fallback=0.5,
    )

    assert calls == 1
    assert captured_np == [fake_np]
    assert result == pytest.approx(math.pi / 4)
