import pytest
import numpy as np

from tnfr.metrics.trig_cache import _compute_trig_python, compute_theta_trig


def test_compute_theta_trig_numpy_matches_python():
    pairs = [
        ("a", {"theta": 0.1}),
        ("b", {"theta": -2.3}),
        ("c", 1.7),
    ]

    trig_py = _compute_trig_python(pairs)
    trig_np = compute_theta_trig(pairs, np=np)

    assert trig_py.theta.keys() == trig_np.theta.keys()
    for n in trig_py.theta:
        assert trig_py.theta[n] == pytest.approx(trig_np.theta[n])
        assert trig_py.cos[n] == pytest.approx(trig_np.cos[n])
        assert trig_py.sin[n] == pytest.approx(trig_np.sin[n])

