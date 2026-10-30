"""Keep auxiliary graph-wave diagnostics accurate at separated root scales."""

from decimal import Decimal, localcontext

import networkx as nx
import numpy as np
import pytest

from tnfr.physics.structural_diffusion import (
    _damped_wave_roots,
    damped_wave_rates,
    verify_overdamped_projection,
)


def test_strong_damping_preserves_the_nonzero_slow_root():
    graph = nx.path_graph(2)
    eigenvalues, slow, fast = damped_wave_rates(graph, gamma=1e10)
    with localcontext() as context:
        context.prec = 80
        damping = Decimal("1e10")
        stiffness = Decimal(2)
        reference = (-damping + (damping**2 - 4 * stiffness).sqrt()) / 2

    assert slow[1] < 0.0
    assert slow[1] == pytest.approx(float(reference), rel=2e-15, abs=0)
    np.testing.assert_allclose(slow * fast, eigenvalues, rtol=2e-15, atol=0)


@pytest.mark.parametrize("gamma", (0.0, 2.0, -2.0))
def test_zero_critical_and_complex_roots_keep_the_polynomial(gamma):
    stiffness = np.array([0.0, 1.0, 4.0])
    plus, minus = _damped_wave_roots(stiffness, gamma)
    np.testing.assert_allclose(plus + minus, -gamma, rtol=0, atol=2e-15)
    np.testing.assert_allclose(plus * minus, stiffness, rtol=2e-15, atol=0)
    np.testing.assert_allclose(
        plus**2 + gamma * plus + stiffness, 0.0, rtol=0, atol=2e-15
    )
    assert plus[-1].imag > 0
    assert minus[-1] == pytest.approx(plus[-1].conjugate())
    scalar_plus, scalar_minus = _damped_wave_roots(0.0, gamma)
    assert scalar_plus == plus[0]
    assert scalar_minus == minus[0]


def test_projection_observer_uses_the_resolved_slow_root():
    graph = nx.path_graph(2)
    nx.set_node_attributes(graph, {0: 0.75, 1: 0.25}, "EPI")

    result = verify_overdamped_projection(graph, gamma=1e10, n_time_samples=4)

    assert result.slowest_slow_rate == pytest.approx(2e-10, rel=2e-15, abs=0)
    assert result.max_rate_rel_error < 2e-15
    assert result.trajectory_max_rel_error < 2e-14
    assert result.projects_to_diffusion
