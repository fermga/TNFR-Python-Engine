import math
import pytest

from tnfr.helpers.numeric import (
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
        "tnfr.helpers.numeric._neighbor_phase_mean_generic", fake_generic
    )
    result = neighbor_phase_mean_list(neigh, cos_th, sin_th, np=None, fallback=0.0)
    assert result == pytest.approx(1.23)
    assert captured["args"] == (neigh, cos_th, sin_th, None, 0.0)
