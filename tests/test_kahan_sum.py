import math

import pytest

from tnfr.helpers.numeric import kahan_sum, kahan_sum2d


def test_kahan_sum_compensates_cancellation():
    xs = [1e16, 1.0, -1e16]
    assert sum(xs) == 0.0
    result = kahan_sum(xs)
    assert result == pytest.approx(math.fsum(xs))


def test_kahan_sum2d_compensates_cancellation():
    pairs = [(1e16, 1e16), (1.0, 1.0), (-1e16, -1e16)]
    res_x, res_y = kahan_sum2d(pairs)
    exp_x = math.fsum(p[0] for p in pairs)
    exp_y = math.fsum(p[1] for p in pairs)
    assert res_x == pytest.approx(exp_x)
    assert res_y == pytest.approx(exp_y)
