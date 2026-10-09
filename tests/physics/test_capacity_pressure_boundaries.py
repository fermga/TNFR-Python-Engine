"""Production boundaries of a held-capacity EPI pressure cancellation.

These finite controls exercise the canonical pressure reader and named
all-target kernels. They do not identify a fixed capacity profile as an
autonomous localized entity or certify future full multichannel execution.
The prospective support-averaging capacity law below is an explicit comparison
in a fixed nondimensional clock, not an installed runtime or a unique closure.
"""

import math
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import (
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_SI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.integrators import update_epi_via_nodal_equation
from tnfr.metrics.common import merge_and_normalize_weights
from tnfr.operators.definitions import Coupling, Silence
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors
from tnfr.operators.network_stage import (
    TWO_PHASE_JACOBI,
    execute_coupling_stage,
    execute_pointwise_stage,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.phase_response import (
    derive_joint_nodal_response,
    derive_phase_response,
)
from tnfr.physics.winding_certificates import certify_phase_winding


def _weights(graph):
    return merge_and_normalize_weights(
        graph, "DNFR_WEIGHTS", ("phase", "epi", "vf", "topo"), default=0.0
    )


def _field(graph, aliases):
    return np.array([get_attr(graph.nodes[node], aliases, 0.0) for node in graph])


def _initialize(graph, capacities, epi, *, winding=0):
    count = len(graph)
    for index, node in enumerate(graph):
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: float(epi[index]),
                ALIAS_VF[0]: float(capacities[index]),
                ALIAS_THETA[0]: math.tau * winding * index / count,
                ALIAS_DNFR[0]: 0.0,
                ALIAS_SI[0]: 1.0,
                "glyph_history": ["AL"],
            }
        )
    return graph


def _pinned_cycle(*, winding=1):
    graph = nx.cycle_graph(8)
    capacities = (1, 1, 1, 2, 1, 1, 1, 1)
    weights = _weights(graph)
    ratio = Fraction(weights["vf"]) / Fraction(weights["epi"])
    epi = [float(1 - ratio * capacity) for capacity in capacities]
    return _initialize(graph, capacities, epi, winding=winding)


def _cycle_laplacian(values):
    """Exact unit-cycle L_rw from the two adjacent scalar coordinates."""
    values = tuple(
        value if isinstance(value, Fraction) else Fraction(float(value))
        for value in values
    )
    count = len(values)
    return tuple(
        values[index] - (values[index - 1] + values[(index + 1) % count]) / 2
        for index in range(count)
    )


def _named_stage(graph, operator):
    # A one-step named-kernel observation, with the retained AL history and
    # the production live grammar authoritative; no whole-word claim follows.
    executor = (
        execute_coupling_stage
        if isinstance(operator, Coupling)
        else execute_pointwise_stage
    )
    result = executor(
        graph, operator, tuple(graph), compute_delta_nfr=default_compute_delta_nfr
    )
    assert result.glyph == operator.glyph.value
    assert result.schedule == TWO_PHASE_JACOBI
    assert all(
        graph.nodes[node]["glyph_history"][-1] == operator.glyph.value for node in graph
    )


@pytest.mark.parametrize("winding", [0, 1])
def test_default_full_pressure_cancels_on_unit_cycle_with_uniform_twist(winding):
    graph = _pinned_cycle(winding=winding)
    before_epi = _field(graph, ALIAS_EPI).copy()
    before_capacity = _field(graph, ALIAS_VF).copy()

    default_compute_delta_nfr(graph)

    assert graph.graph["_dnfr_weights"] == _weights(graph)
    assert graph.graph["_dnfr_weights"]["phase"] > 0.0
    np.testing.assert_allclose(_field(graph, ALIAS_DNFR), 0.0, atol=5e-16)
    np.testing.assert_array_equal(_field(graph, ALIAS_EPI), before_epi)
    np.testing.assert_array_equal(_field(graph, ALIAS_VF), before_capacity)
    assert np.ptp(before_epi) > 0.1
    certificate = certify_phase_winding(graph, range(8))
    assert certificate.winding == winding
    assert certificate.u3_admissible


def test_regular_weighted_degree_does_not_identify_epi_and_capacity_walks():
    graph = nx.cycle_graph(4)
    for node in graph:
        graph.edges[node, (node + 1) % 4]["weight"] = 1.0 if node % 2 == 0 else 3.0
    _initialize(graph, (1, 2, 3, 4), (4, 3, 2, 1))
    graph.graph["DNFR_WEIGHTS"] = {
        "phase": 0.0,
        "epi": 1.0,
        "vf": 1.0,
        "topo": 0.0,
    }

    default_compute_delta_nfr(graph)

    assert set(dict(graph.degree(weight="weight")).values()) == {4.0}
    assert _weights(graph)["epi"] == _weights(graph)["vf"] == 0.5
    np.testing.assert_array_equal(_field(graph, ALIAS_DNFR), (-0.25, -0.25, 0.25, 0.25))


def test_zero_weight_edge_remains_in_the_capacity_neighborhood():
    graph = nx.cycle_graph(4)
    graph.edges[0, 1]["weight"] = 0.0
    _initialize(graph, (1, 2, 3, 4), (4, 3, 2, 1))
    graph.graph["DNFR_WEIGHTS"] = {
        "phase": 0.0,
        "epi": 1.0,
        "vf": 1.0,
        "topo": 0.0,
    }

    default_compute_delta_nfr(graph)

    assert graph.has_edge(0, 1)
    np.testing.assert_array_equal(_field(graph, ALIAS_DNFR), (-0.5, -0.5, 0.0, 0.0))


@pytest.mark.parametrize("epi", [(0, 0, 0), (0.25, 0.5, 0.75), (1, -1, 2)])
def test_weighted_path_capacity_forcing_has_nonzero_conserved_total_drift(epi):
    graph = nx.path_graph(3)
    graph.edges[0, 1]["weight"] = 1.0
    graph.edges[1, 2]["weight"] = 3.0
    _initialize(graph, (1, 2, 4), epi)
    graph.graph["DNFR_WEIGHTS"] = {
        "phase": 0.0,
        "epi": 1.0,
        "vf": 1.0,
        "topo": 0.0,
    }

    default_compute_delta_nfr(graph)

    metric = np.array([1.0, 2.0, 0.75])
    rate = _field(graph, ALIAS_VF) * _field(graph, ALIAS_DNFR)
    # h^T x' = -w_vf d_W^T L_support nu = -3/2, independently of x.
    assert metric @ rate == pytest.approx(-1.5, abs=2e-15)


def test_default_um_capacity_sync_breaks_the_pin_after_full_pressure_refresh():
    graph = _pinned_cycle()
    graph.graph.update(UM_BIDIRECTIONAL=False, UM_FUNCTIONAL_LINKS=False)
    default_compute_delta_nfr(graph)
    before_epi = _field(graph, ALIAS_EPI).copy()
    before_capacity = _field(graph, ALIAS_VF).copy()
    frequency_weight = Fraction(_weights(graph)["vf"])
    sync = Fraction(
        resolve_runtime_operator_factors(None, "UM", graph.graph)["UM_vf_sync"]
    )
    laplacian = _cycle_laplacian(before_capacity)
    expected_capacity = [
        float(Fraction(value) - sync * lap)
        for value, lap in zip(before_capacity, laplacian)
    ]
    expected_pressure = [
        float(frequency_weight * sync * lap) for lap in _cycle_laplacian(laplacian)
    ]

    _named_stage(graph, Coupling())

    np.testing.assert_array_equal(_field(graph, ALIAS_EPI), before_epi)
    np.testing.assert_allclose(
        _field(graph, ALIAS_VF), expected_capacity, rtol=0.0, atol=3e-16
    )
    np.testing.assert_allclose(
        _field(graph, ALIAS_DNFR), expected_pressure, rtol=0.0, atol=5e-16
    )
    assert np.max(np.abs(_field(graph, ALIAS_DNFR))) > 0.001
    assert certify_phase_winding(graph, range(8)).winding == 1


def test_default_global_silence_breaks_the_pin_without_changing_epi():
    graph = _pinned_cycle()
    default_compute_delta_nfr(graph)
    before_epi = _field(graph, ALIAS_EPI).copy()
    before_capacity = _field(graph, ALIAS_VF).copy()
    frequency_weight = Fraction(_weights(graph)["vf"])
    factor = Fraction(
        resolve_runtime_operator_factors(None, "SHA", graph.graph)["SHA_vf_factor"]
    )
    expected_pressure = [
        float(frequency_weight * (1 - factor) * lap)
        for lap in _cycle_laplacian(before_capacity)
    ]

    _named_stage(graph, Silence())

    np.testing.assert_array_equal(_field(graph, ALIAS_EPI), before_epi)
    np.testing.assert_allclose(
        _field(graph, ALIAS_VF),
        float(factor) * before_capacity,
        rtol=0.0,
        atol=3e-16,
    )
    np.testing.assert_allclose(
        _field(graph, ALIAS_DNFR), expected_pressure, rtol=0.0, atol=5e-16
    )
    assert np.max(np.abs(_field(graph, ALIAS_DNFR))) > 0.001


def test_zero_capacity_freezes_epi_with_nonzero_refreshed_pressure():
    graph = _pinned_cycle(winding=0)
    graph.graph.update(
        GLYPH_FACTORS={"SHA_vf_factor": 0.0},
        GAMMA={"type": "none"},
        DT_MIN=0.0,
    )
    before_epi = _field(graph, ALIAS_EPI).copy()

    _named_stage(graph, Silence())
    assert np.max(np.abs(_field(graph, ALIAS_DNFR))) > 0.01
    np.testing.assert_array_equal(_field(graph, ALIAS_VF), np.zeros(8))
    update_epi_via_nodal_equation(graph, dt=0.125, t=0.0, method="euler")

    np.testing.assert_array_equal(_field(graph, ALIAS_EPI), before_epi)


def test_prospective_support_averaging_reduces_to_checkerboard_and_mean():
    """Derive the invariant sector from the actual unit-cycle adjacency."""
    sp = pytest.importorskip("sympy")
    count = 8
    laplacian = sp.Matrix.hstack(
        *(
            sp.Matrix(_cycle_laplacian(int(row == col) for row in range(count)))
            for col in range(count)
        )
    )
    ones = sp.ones(count, 1)
    mode = sp.Matrix([(-1) ** index for index in range(count)])
    eigenvalue = 2
    assert laplacian * ones == sp.zeros(count, 1)
    assert laplacian * mode == eigenvalue * mode
    assert sum(mode) == 0

    a, e, f = sp.symbols("a e f", positive=True)
    epsilon = sp.symbols("epsilon", nonnegative=True)
    b, c, d = sp.symbols("b c d", real=True)
    capacity, epi = a * ones + c * mode, b * ones + d * mode
    # Assume a > abs(c); zero primitive phase and fixed unit support remove
    # phase/topology pressure. Gamma is absent. The capacity row is supplied.
    pressure = -e * laplacian * epi - f * laplacian * capacity
    epi_rate = sp.diag(*capacity) * pressure
    capacity_rate = -epsilon * laplacian * capacity
    mean_rate = sp.simplify(sum(epi_rate) / count)
    mode_rate = sp.simplify((mode.T * epi_rate)[0] / count)
    assert sp.simplify(mean_rate + eigenvalue * c * (e * d + f * c)) == 0
    assert sp.simplify(mode_rate + a * eigenvalue * (e * d + f * c)) == 0
    assert sp.simplify(mean_rate - c * mode_rate / a) == 0
    assert sp.simplify(epi_rate - mean_rate * ones - mode_rate * mode) == sp.zeros(
        count, 1
    )
    assert sp.simplify(capacity_rate + epsilon * eigenvalue * c * mode) == sp.zeros(
        count, 1
    )
    assert sum(capacity_rate) == 0
    assert capacity_rate.subs(epsilon, 0) == sp.zeros(count, 1)


def test_prospective_support_averaging_solution_resonance_and_limit_order():
    """Check the IVP for arbitrary positive decay rates, including equality."""
    sp = pytest.importorskip("sympy")
    t = sp.symbols("t", nonnegative=True)
    alpha, beta, k = sp.symbols("alpha beta k", positive=True)
    c0 = sp.symbols("c0", real=True, nonzero=True)
    d0 = sp.symbols("d0", real=True)
    c = c0 * sp.exp(-beta * t)
    d = d0 * sp.exp(-alpha * t) - k * c0 * (sp.exp(-beta * t) - sp.exp(-alpha * t)) / (
        alpha - beta
    )
    assert c.subs(t, 0) == c0
    assert d.subs(t, 0) == d0
    assert sp.diff(c, t) == -beta * c
    assert sp.simplify(sp.diff(d, t) + alpha * d + k * c) == 0
    resonant = (d0 - k * c0 * t) * sp.exp(-alpha * t)
    assert sp.simplify(sp.limit(d, beta, alpha) - resonant) == 0
    assert (
        sp.simplify(sp.diff(resonant, t) + alpha * resonant + k * c.subs(beta, alpha))
        == 0
    )
    assert sp.limit(resonant, t, sp.oo) == sp.limit(d, t, sp.oo) == 0
    assert sp.limit(c, t, sp.oo) == 0

    # alpha = e*a*lambda, beta = epsilon*lambda, k = a*lambda*f.
    # An initially pressure-balanced profile is held when epsilon = 0.
    pinned = d.subs(d0, -k * c0 / alpha)
    held = sp.simplify(pinned.subs(beta, 0))
    assert held == -k * c0 / alpha
    assert held.is_zero is False
    assert sp.simplify(sp.diff(pinned, t).subs(t, 0)) == 0
    assert sp.simplify(sp.diff(pinned, t, 2).subs(t, 0)) == beta * c0 * k
    assert sp.limit(sp.limit(pinned, t, sp.oo), beta, 0, dir="+") == 0
    assert sp.limit(sp.limit(pinned, beta, 0, dir="+"), t, sp.oo) == held

    a, e, f, epsilon, eigenvalue = sp.symbols("a e f epsilon lambda", positive=True)
    mean_shift = (
        k
        * c0**2
        * beta
        / (a * (alpha - beta))
        * (
            (1 - sp.exp(-2 * beta * t)) / (2 * beta)
            - (1 - sp.exp(-(alpha + beta) * t)) / (alpha + beta)
        )
    )
    assert mean_shift.subs(t, 0) == 0
    assert sp.simplify(sp.diff(mean_shift, t) - c * sp.diff(pinned, t) / a) == 0
    final_shift = sp.simplify(sp.limit(mean_shift, t, sp.oo))
    assert final_shift == k * c0**2 / (2 * a * (alpha + beta))
    constitutive_parameters = {
        alpha: e * a * eigenvalue,
        beta: epsilon * eigenvalue,
        k: a * eigenvalue * f,
    }
    assert (
        sp.simplify(
            final_shift.subs(constitutive_parameters)
            - f * c0**2 / (2 * (e * a + epsilon))
        )
        == 0
    )
    resonant_shift = sp.limit(mean_shift, beta, alpha)
    assert (
        sp.simplify(
            sp.diff(resonant_shift, t)
            - c.subs(beta, alpha) * sp.diff(resonant.subs(d0, held), t) / a
        )
        == 0
    )
    assert sp.limit(resonant_shift, t, sp.oo) == final_shift.subs(beta, alpha)

    # Predetermined e=f=1/2, a=1, c0=1/4, d0=-1/4, epsilon=1/4.
    # The mean also moves: omitting it would fail the full nodal equation.
    declared = {alpha: 1, beta: sp.Rational(1, 2), k: 1, c0: sp.Rational(1, 4)}
    concrete_c, concrete_d = c.subs(declared), pinned.subs(declared)
    concrete_b = (
        sp.Rational(1, 2) + (1 - sp.exp(-t)) / 16 - (1 - sp.exp(-3 * t / 2)) / 24
    )
    assert concrete_b.subs(t, 0) == sp.Rational(1, 2)
    assert (
        sp.simplify(sp.diff(concrete_b, t) - concrete_c * sp.diff(concrete_d, t)) == 0
    )
    assert sp.simplify(concrete_d + concrete_c).subs(t, 0) == 0


def test_prospective_analytic_sample_agrees_with_production_pressure_and_rate():
    """Compare the declared curve at t=2*log(2); no coupled solver is installed."""
    mode = tuple((-1) ** index for index in range(8))
    initial = _initialize(
        nx.cycle_graph(8),
        tuple(Fraction(1) + Fraction(value, 4) for value in mode),
        tuple(Fraction(1, 2) - Fraction(value, 4) for value in mode),
    )
    initial.graph["DNFR_WEIGHTS"] = {
        "phase": 0.0,
        "epi": 1.0,
        "vf": 1.0,
        "topo": 0.0,
    }
    default_compute_delta_nfr(initial)
    captured = capture_non_epi_forcing(initial)
    source = captured.snapshot
    assert (
        captured.kernel_pressure_defect == captured.stored_pressure_residual == (0,) * 8
    )
    assert source.stored_pressure == source.rate == (0,) * 8
    reference = derive_phase_response(
        cosine_gram=((1,) * 8,) * 8,
        mean_neighbors=source.support_neighbors,
        receiver_sources=tuple((index,) for index in range(8)),
        phase_factor=0,
    )
    response = derive_joint_nodal_response(
        source,
        reference,
        epi_weight=Fraction(1, 2),
        phase_weight=0,
        capacity_weight=Fraction(1, 2),
        phase_rate_over_pi=(0,) * 8,
        capacity_rate=tuple(-value / 4 for value in _cycle_laplacian(source.capacity)),
    )
    assert response.pressure_rate == tuple(Fraction(value, 8) for value in mode)
    assert response.epi_acceleration == tuple(
        Fraction(value, 8) + Fraction(1, 32) for value in mode
    )

    # r = exp(-t/2) is exactly 1/2 at this predetermined mathematical time.
    # These expressions are the analytic solution checked independently above.
    r = Fraction(1, 2)
    c = r / 4
    d = r**2 / 4 - r / 2
    b = Fraction(1, 2) + (1 - r**2) / 16 - (1 - r**3) / 24
    d_rate = (r - r**2) / 4
    b_rate = (r**2 - r**3) / 16
    capacities = tuple(1 + c * value for value in mode)
    epi = tuple(b + d * value for value in mode)
    expected_rate = np.array([float(b_rate + d_rate * value) for value in mode])
    graph = _initialize(nx.cycle_graph(8), capacities, epi, winding=0)
    graph.graph.update(
        DNFR_WEIGHTS=dict(initial.graph["DNFR_WEIGHTS"]),
        GAMMA={"type": "none"},
        DT_MIN=0.0,
    )

    default_compute_delta_nfr(graph)

    assert _weights(graph)["epi"] == _weights(graph)["vf"] == 0.5
    expected_pressure = np.array([float(-(d + c) * value) for value in mode])
    # b=49/96 is not binary64-exact. Bound materialization/arithmetic error;
    # this comparison makes no exact-real execution or asymptotic claim.
    np.testing.assert_allclose(
        _field(graph, ALIAS_DNFR), expected_pressure, rtol=0.0, atol=2e-16
    )
    before_epi = _field(graph, ALIAS_EPI).copy()
    dt = 0.125
    update_epi_via_nodal_equation(graph, dt=dt, t=2 * math.log(2), method="euler")

    # The shared step holds sampled pressure/capacity. It verifies the nodal
    # rate, not the exact finite-time flow of the prospective coupled model.
    np.testing.assert_allclose(
        _field(graph, ALIAS_DEPI), expected_rate, rtol=0.0, atol=2e-16
    )
    np.testing.assert_allclose(
        _field(graph, ALIAS_EPI), before_epi + dt * expected_rate, rtol=0.0, atol=2e-16
    )
    np.testing.assert_array_equal(
        _field(graph, ALIAS_VF), np.array(capacities, dtype=float)
    )
    np.testing.assert_array_equal(_field(graph, ALIAS_THETA), np.zeros(8))
