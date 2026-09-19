"""Scope of existing phase updates on repeated strict-chart prism triples.

Ideal algebra, finite coordinator invocations and the separately configured
free-advance proposal have distinct claims. These controls neither select a
primitive phase law nor certify a repeated full-runtime trajectory.
"""

import math

import pytest

from tests.physics._internal_mode_fixture import NODES, _graph
from tnfr.alias import get_theta_attr
from tnfr.dynamics.coordination import coordinate_global_local_phase
from tnfr.dynamics.phase_evolution import propose_u3_gated_phase_step


def _repeated_triple():
    """A nonzero phasor-mean correction inside a chart centered on pi."""
    graph = _graph()
    eta = (math.pi / 12, math.pi / 12, -math.pi / 6)
    for node in NODES:
        graph.nodes[node]["theta"] = math.pi + eta[node[1]]
        # Every fine node sees exactly one representative of each phase.
        assert sorted(other[1] for other in graph.neighbors(node)) == [0, 1, 2]
    # Independent closed radical expression for Arg(2 exp(i*pi/12)+exp(-i*pi/6)).
    delta = math.atan(
        (math.sqrt(6) - math.sqrt(2) - 1) / (math.sqrt(6) + math.sqrt(2) + math.sqrt(3))
    )
    assert 0 < delta < math.pi / 12
    return graph, eta, delta


def test_repeated_mean_relaxation_has_only_a_scalar_contrast_map():
    s = pytest.importorskip("sympy")
    a, b, beta, delta = s.symbols("a b beta delta", real=True)
    eta = s.Matrix([a + b, -a + b, -2 * b])
    ones = s.ones(3, 1)
    theta = beta * ones + eta

    # The repeated triple gives identical local/global phasor targets beta+delta.
    # delta=pi*c may be nonzero: keeping beta fixed would lose the mean update.
    for gain in (s.Rational(1, 2), s.Integer(1), s.Rational(3, 2)):
        proposal = theta + gain * ((beta + delta) * ones - theta)
        mean = s.simplify(sum(proposal) / 3)
        contrast = s.simplify(proposal - mean * ones)
        assert mean == beta + gain * delta
        assert contrast == (1 - gain) * eta
        assert s.simplify(eta.cross(contrast)) == s.zeros(3, 1)

    # Changing scalar gains through history does not add a transverse direction,
    # conditional on this same repeated regular chart at each invocation.
    gain_1, gain_2 = s.symbols("gain_1 gain_2", real=True)
    twice = (1 - gain_2) * ((1 - gain_1) * eta)
    assert s.simplify(twice - (1 - gain_1) * (1 - gain_2) * eta) == s.zeros(3, 1)
    assert s.simplify(eta.cross(twice)) == s.zeros(3, 1)


@pytest.mark.parametrize(
    ("global_gain", "local_gain"),
    ((0.5, 0.0), (0.0, 0.5), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75)),
)
def test_actual_coordinator_retains_the_mean_and_scalar_contrast_with_rounding(
    global_gain, local_gain
):
    graph, eta, delta = _repeated_triple()
    held = tuple(
        (graph.nodes[node]["EPI"], graph.nodes[node]["nu_f"]) for node in NODES
    )
    gain = global_gain + local_gain
    assert (
        coordinate_global_local_phase(
            graph, global_force=global_gain, local_force=local_gain, n_jobs=1
        )
        is None
    )
    actual = tuple(get_theta_attr(graph.nodes[node]) for node in NODES)
    expected_mean = math.pi + gain * delta
    expected = tuple(expected_mean + (1 - gain) * eta[node[1]] for node in NODES)
    # Local/global trig reductions and phase writes are separately rounded.
    assert actual == pytest.approx(expected, abs=4e-15, rel=0)
    for fiber in (actual[:3], actual[3:]):
        mean = math.fsum(fiber) / 3
        assert mean == pytest.approx(expected_mean, abs=4e-15, rel=0)
        assert mean > math.pi
        assert tuple(value - mean for value in fiber) == pytest.approx(
            tuple((1 - gain) * value for value in eta), abs=4e-15, rel=0
        )
    assert (
        tuple((graph.nodes[node]["EPI"], graph.nodes[node]["nu_f"]) for node in NODES)
        == held
    )


def test_uniform_free_advance_moves_the_mean_while_sine_coupling_dissipates_contrast():
    s = pytest.importorskip("sympy")
    eta_exact = (s.pi / 12, s.pi / 12, -s.pi / 6)
    strength = s.Rational(1, 4)
    rate = tuple(
        s.simplify(strength * sum(s.sin(other - value) for other in eta_exact) / 3)
        for value in eta_exact
    )
    assert sum(rate) == 0
    variance_rate = s.simplify(2 * sum(x * y for x, y in zip(eta_exact, rate)))
    pair_budget = s.simplify(
        -2
        * strength
        / 3
        * sum(
            (eta_exact[i] - eta_exact[j]) * s.sin(eta_exact[i] - eta_exact[j])
            for i in range(3)
            for j in range(i + 1, 3)
        )
    )
    assert variance_rate == pair_budget == -s.sqrt(2) * s.pi / 24
    # For a regular triple every pair term is nonnegative. This is the
    # continuous sine-model comparison, not a claim for arbitrary Euler dt.

    graph, _, _ = _repeated_triple()
    phases = tuple(get_theta_attr(graph.nodes[node]) for node in NODES)
    dt = 0.125
    without_free = propose_u3_gated_phase_step(
        graph, NODES, phases, (0.0,) * 6, dt=dt, coupling_strength=float(strength)
    )
    with_free = propose_u3_gated_phase_step(
        graph, NODES, phases, (1.0,) * 6, dt=dt, coupling_strength=float(strength)
    )
    expected_without = tuple(
        phases[index] + dt * float(rate[node[1]]) for index, node in enumerate(NODES)
    )
    assert tuple(without_free) == pytest.approx(expected_without, abs=2e-15, rel=0)
    assert tuple(with_free - without_free) == pytest.approx((dt,) * 6, abs=2e-15, rel=0)
    for proposed in (without_free, with_free):
        mean = math.fsum(proposed[:3]) / 3
        contrast = tuple(value - mean for value in proposed[:3])
        expected_contrast = tuple(
            float(x) + dt * float(y) for x, y in zip(eta_exact, rate)
        )
        assert contrast == pytest.approx(expected_contrast, abs=2e-15, rel=0)
    # The proposal reads declared frequency vectors; it does not mutate a graph
    # or derive this angular-frequency identification from the EPI equation.
    assert tuple(get_theta_attr(graph.nodes[node]) for node in NODES) == phases
