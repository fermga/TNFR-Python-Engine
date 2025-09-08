import math

import pytest

from tnfr.helpers.numeric import kahan_sum


def test_kahan_sum_compensates_cancellation():
    xs = [1e16, 1.0, -1e16]
    assert sum(xs) == 0.0
    result = kahan_sum(xs)
    assert result == pytest.approx(math.fsum(xs))
