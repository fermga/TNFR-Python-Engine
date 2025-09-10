import math
import pytest

from tnfr.helpers.numeric import neighbor_phase_mean_list


def test_neighbor_phase_mean_list_missing_trig():
    neigh = [1, 2, 3]
    cos_th = {1: 1.0, 2: 0.0}
    sin_th = {1: 0.0, 2: 1.0}

    angle = neighbor_phase_mean_list(neigh, cos_th, sin_th, fallback=0.5)
    assert angle == pytest.approx(math.pi / 4)

    assert neighbor_phase_mean_list([3], cos_th, sin_th, fallback=0.5) == pytest.approx(0.5)
