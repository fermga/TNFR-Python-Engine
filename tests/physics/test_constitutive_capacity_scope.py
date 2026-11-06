"""Capacity-law independence, constitutive admission and live-policy controls.

The analytic completions and local capacity/form relations below are declared
comparisons, not uniquely derived TNFR laws or complete-runtime certificates.
The final admission controls execute actual pressure, Si and capacity events.
"""

from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_SI, ALIAS_VF
from tnfr.dynamics.adaptation import adapt_vf_after_structural_stability
from tnfr.dynamics.canonical import compute_extended_nodal_system
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.metrics.sense_index import compute_Si
from tnfr.physics._cycle_algebra import dot
from tnfr.physics.fields import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length_with_provenance,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.phase_response import (
    derive_joint_nodal_response,
    derive_phase_response,
)
from tnfr.physics.structural_diffusion import (
    compute_diffusion_energy,
    structural_diffusion_operator,
)
from tnfr.physics.support_transport import _from_data


def _initial_graph():
    graph = nx.path_graph(2)
    for node, epi in enumerate((0.625, 0.375)):
        graph.nodes[node].update(
            EPI=epi,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            glyph_history=[],
            epi_history=[epi],
            stable_count=0,
        )
    # Initialize stored pressure from its actual owner. No EPI step is taken.
    pressure = capture_non_epi_forcing(graph).full_kernel_pressure
    for node, value in zip(graph, pressure, strict=True):
        graph.nodes[node]["delta_nfr"] = float(value)
    return graph


def _tetrad(graph):
    return (
        compute_structural_potential(graph),
        compute_phase_gradient(graph),
        compute_phase_curvature(graph),
        estimate_coherence_length_with_provenance(graph),
    )


def _laplacian(graph):
    nodes, matrix = structural_diffusion_operator(graph)
    assert nodes == list(graph)
    return tuple(tuple(Q(float(value)) for value in row) for row in matrix)


def test_initial_complete_state_and_tetrad_do_not_contain_a_capacity_law():
    first = _initial_graph()
    second = deepcopy(first)
    assert dict(first.nodes(data=True)) == dict(second.nodes(data=True))
    assert list(first.edges(data=True)) == list(second.edges(data=True))
    assert first.graph == second.graph
    assert _tetrad(first) == _tetrad(second)
    # Retain the correlation estimator's provenance, not an assumed identity.
    correlation = _tetrad(first)[3]
    assert correlation.method == "spectral_gap"
    assert correlation.value > 0

    source = capture_non_epi_forcing(first)
    assert source.epi_weight > 0
    assert source.phase_gradient == source.forcing == (0, 0)
    assert (
        source.snapshot.capacity_gradient == source.snapshot.topology_gradient == (0, 0)
    )
    assert source.kernel_pressure_defect == source.stored_pressure_residual == (0, 0)
    assert source.snapshot.stored_pressure == tuple(
        source.epi_weight * value for value in source.snapshot.epi_gradient
    )
    assert any(source.snapshot.stored_pressure)

    balance = compute_diffusion_energy(first)
    assert Q(balance.energy) == source.snapshot.dirichlet_energy == Q(1, 32)
    assert source.snapshot.energy_rate == source.epi_weight * Q(balance.energy_rate) < 0


def test_same_initial_pressure_and_rate_allow_distinct_exact_accelerations():
    graph = _initial_graph()
    source = capture_non_epi_forcing(graph)
    laplacian = _laplacian(graph)
    rate = source.snapshot.rate
    pressure = source.snapshot.stored_pressure
    assert rate == pressure  # Both initial capacities are one.
    assert tuple(dot(row, source.snapshot.epi) for row in laplacian) == (
        Q(1, 4),
        Q(-1, 4),
    )

    # nu'_A=(0,0), nu'_B=(1,1). Both have zero capacity-source derivative.
    capacity_rates = ((Q(0), Q(0)), (Q(1), Q(1)))
    assert all(
        tuple(dot(row, rate_nu) for row in laplacian) == (0, 0)
        for rate_nu in capacity_rates
    )
    reference = derive_phase_response(
        cosine_gram=((1, 1), (1, 1)),
        mean_neighbors=((1,), (0,)),
        receiver_sources=((0,), (1,)),
        phase_factor=Q(0),
    )
    weights = dict(source.normalized_weights)
    responses = tuple(
        derive_joint_nodal_response(
            source.snapshot,
            reference,
            epi_weight=weights["epi"],
            phase_weight=weights["phase"],
            capacity_weight=weights["vf"],
            phase_rate_over_pi=(0, 0),
            capacity_rate=rate_nu,
        )
        for rate_nu in capacity_rates
    )
    pressure_rate = tuple(-source.epi_weight * dot(row, rate) for row in laplacian)
    assert all(response.capacity_pressure_rate == (0, 0) for response in responses)
    assert all(response.pressure_rate == pressure_rate for response in responses)
    accelerations = tuple(response.epi_acceleration for response in responses)
    assert tuple(b - a for a, b in zip(*accelerations, strict=True)) == pressure
    assert accelerations[0] != accelerations[1]
    # Initial positivity, dissipation and source compatibility hold for both.
    assert all(value > 0 for value in source.snapshot.epi + source.snapshot.capacity)
    assert source.snapshot.energy_rate < 0


def test_two_analytic_completions_satisfy_the_same_nodal_law_exactly():
    symbolic = pytest.importorskip("sympy")
    graph = _initial_graph()
    source = capture_non_epi_forcing(graph)

    def rational(value):
        return symbolic.Rational(value.numerator, value.denominator)

    laplacian = symbolic.Matrix(
        [[rational(value) for value in row] for row in _laplacian(graph)]
    )
    e = rational(source.epi_weight)
    t = symbolic.Symbol("t", positive=True)
    ones, mode = symbolic.ones(2, 1), symbolic.Matrix([1, -1])
    assert laplacian * mode == 2 * mode
    c, d = symbolic.Rational(1, 2), symbolic.Rational(1, 8)
    fields, energies = [], []
    for capacity, clock in ((1, t), (1 + t, t + t**2 / 2)):
        amplitude = symbolic.exp(-2 * e * clock)
        epi = c * ones + d * amplitude * mode
        pressure = -e * laplacian * epi
        assert symbolic.simplify(epi.diff(t) - capacity * pressure) == symbolic.zeros(
            2, 1
        )
        assert epi.subs(t, 0) == symbolic.Matrix(
            [rational(v) for v in source.snapshot.epi]
        )
        assert pressure.subs(t, 0) == symbolic.Matrix(
            [rational(v) for v in source.snapshot.stored_pressure]
        )
        assert laplacian * (capacity * ones) == symbolic.zeros(2, 1)
        energy = rational(source.snapshot.dirichlet_energy) * amplitude**2
        assert symbolic.simplify(energy.diff(t) + 4 * e * capacity * energy) == 0
        assert (-energy.diff(t)).is_positive
        fields.append(epi)
        energies.append(energy)

    assert fields[0].diff(t).subs(t, 0) == fields[1].diff(t).subs(t, 0)
    assert (
        symbolic.simplify((fields[1] - fields[0]).diff(t, 2).subs(t, 0))
        == -2 * e * d * mode
    )
    assert symbolic.simplify(energies[1] / energies[0]) == symbolic.exp(-2 * e * t**2)
    # On [0,1], capacities lie in [1,2], phases stay equal, and
    # 0 < amplitude <= 1 keeps both EPI coordinates between c-d and c+d.
    assert c - d > 0


def test_optional_zero_flux_does_not_remove_the_prescribed_phase_response():
    result = compute_extended_nodal_system(1.0, 0.5, 0.0, 0.0, 0.0)
    assert result.classical_derivative == 0.5
    assert result.phase_derivative == 0.575
    assert result.dnfr_derivative == 0.0
    # Absolute phase is not an input to this particular response formula.
    shifted = compute_extended_nodal_system(1.0, 0.5, 1.0, 0.0, 0.0)
    assert shifted.phase_derivative == result.phase_derivative


@pytest.mark.parametrize("divergence", [-1.0, 0.0, 1.0])
def test_optional_pressure_response_has_no_pressure_restoring_term(divergence):
    rates = tuple(
        compute_extended_nodal_system(
            1.0, pressure, 0.0, 0.0, divergence
        ).dnfr_derivative
        for pressure in (-0.5, 0.0, 0.5)
    )
    assert rates == (-(1.0 + 0.135) * divergence,) * 3


@pytest.fixture(scope="module")
def inverse_capacity_controls():
    """Exact declared states, not two equal binary64 execution endpoints."""
    graph = _initial_graph()
    capture = capture_non_epi_forcing(graph)
    weights = dict(capture.normalized_weights)
    e, b = weights["epi"], weights["vf"]
    assert e > 0 and b > 0
    assert capture.phase_gradient == capture.snapshot.topology_gradient == (0, 0)
    source = capture.snapshot
    epi = tuple(reversed(source.epi))
    delta = epi[1] - epi[0]
    f = e * delta
    u = f / (4 * b)
    states = []
    for factor in (1, 3):
        capacity = (2 * factor * u, factor * u)
        initial = _from_data(
            source.nodes,
            source.conductance,
            source.support_neighbors,
            epi,
            capacity,
            (0, 0),
        )
        # Actual implemented channel definitions, interpreted exactly on the
        # declared rational state. No rate is used to construct pressure.
        pressure = tuple(
            e * epi_gradient + b * capacity_gradient
            for epi_gradient, capacity_gradient in zip(
                initial.epi_gradient, initial.capacity_gradient, strict=True
            )
        )
        states.append(
            _from_data(
                initial.nodes,
                initial.conductance,
                initial.support_neighbors,
                initial.epi,
                initial.capacity,
                pressure,
            )
        )
    reference = derive_phase_response(
        cosine_gram=((1, 1), (1, 1)),
        mean_neighbors=((1,), (0,)),
        receiver_sources=((0,), (1,)),
        phase_factor=Q(0),
    )
    return {
        "states": tuple(states),
        "weights": weights,
        "reference": reference,
        "laplacian": _laplacian(graph),
        "f": f,
        "u": u,
    }


def test_fixed_full_pressure_law_need_not_identify_capacity_from_one_nodal_rate(
    inverse_capacity_controls,
):
    control = inverse_capacity_controls
    first, second = control["states"]
    f, u, b = control["f"], control["u"], control["weights"]["vf"]
    assert first.epi == second.epi
    assert first.conductance == second.conductance
    assert first.support_neighbors == second.support_neighbors
    assert second.capacity == tuple(3 * value for value in first.capacity)
    assert all(value > 0 for value in first.capacity + second.capacity)
    assert first.stored_pressure == (3 * f / 4, -3 * f / 4)
    assert second.stored_pressure == (f / 4, -f / 4)
    q = 3 * f**2 / (16 * b)
    assert first.rate == second.rate == (2 * q, -q)
    assert any(first.rate)
    assert first.capacity == (2 * u, u)
    # This is a noninjectivity of the specified constitutive map, not the
    # freedom to replace pressure by a posterior reconstruction x_dot/nu.
    assert first.capacity_gradient != second.capacity_gradient


def test_held_capacity_acceleration_separates_the_same_rate_states(
    inverse_capacity_controls,
):
    control = inverse_capacity_controls
    weights = control["weights"]
    first, second = control["states"]
    responses = tuple(
        derive_joint_nodal_response(
            state,
            control["reference"],
            epi_weight=weights["epi"],
            phase_weight=weights["phase"],
            capacity_weight=weights["vf"],
            phase_rate_over_pi=(0, 0),
            capacity_rate=(0, 0),
        )
        for state in (first, second)
    )
    expected = tuple(
        -weights["epi"] * dot(row, first.rate) for row in control["laplacian"]
    )
    assert any(expected)
    assert all(response.pressure_rate == expected for response in responses)
    assert all(response.capacity_pressure_rate == (0, 0) for response in responses)
    assert responses[1].epi_acceleration == tuple(
        3 * value for value in responses[0].epi_acceleration
    )
    assert responses[0].epi_acceleration != responses[1].epi_acceleration
    # The supplied zero capacity/phase velocities define this comparison.
    # Equal instantaneous EPI rates do not imply equal later trajectories.


def test_self_loop_normalization_hides_capacity_from_pure_transport_only(
    inverse_capacity_controls,
):
    graphs = [_initial_graph(), _initial_graph()]
    for node, capacity in enumerate((0.5, 0.25)):
        graphs[0].nodes[node]["nu_f"] = capacity
    graphs[1].add_edge(0, 0, weight=1.0)
    graphs[1].add_edge(1, 1, weight=3.0)
    captures, energies = [], []
    for graph in graphs:
        graph.graph["DNFR_WEIGHTS"] = dict(phase=0.0, epi=1.0, vf=0.0, topo=0.0)
        default_compute_delta_nfr(graph)
        captures.append(capture_non_epi_forcing(graph))
        energies.append(compute_diffusion_energy(graph))
    first, second = (capture.snapshot for capture in captures)
    assert first.epi == second.epi == (Q(5, 8), Q(3, 8))
    assert first.capacity == (Q(1, 2), Q(1, 4))
    assert second.capacity == (1, 1)
    assert first.stored_pressure == (Q(-1, 4), Q(1, 4))
    assert second.stored_pressure == (Q(-1, 8), Q(1, 16))
    assert first.rate == second.rate == (Q(-1, 8), Q(1, 16))
    assert first.dirichlet_gradient == second.dirichlet_gradient
    assert first.dirichlet_energy == second.dirichlet_energy == Q(1, 32)
    assert first.energy_rate == second.energy_rate == Q(-3, 64)
    for energy in energies:
        assert tuple(map(Q, energy.mobility)) == (Q(1, 2), Q(1, 4))
        assert tuple(map(Q, energy.epi_rate)) == first.rate
        assert Q(energy.energy) == first.dirichlet_energy
        assert Q(energy.energy_rate) == first.energy_rate
    assert all(capture.stored_pressure_residual == (0, 0) for capture in captures)
    # The capacity channel breaks this equivalence in the full declared mix.
    # These are exact modeled rates using retained default coefficients, not
    # a claim that pressure sums are represented without rounding.
    e = inverse_capacity_controls["weights"]["epi"]
    b = inverse_capacity_controls["weights"]["vf"]
    full_rates = tuple(
        tuple(
            nu * (e * epi_gradient + b * capacity_gradient)
            for nu, epi_gradient, capacity_gradient in zip(
                state.capacity, state.epi_gradient, state.capacity_gradient, strict=True
            )
        )
        for state in (first, second)
    )
    assert tuple(a - z for a, z in zip(*full_rates, strict=True)) == (
        -b / 8,
        b / 16,
    )


def test_local_capacity_form_relation_has_conditional_charge_and_energy_balance():
    symbolic = pytest.importorskip("sympy")
    graph = nx.cycle_graph(4)
    laplacian = symbolic.Matrix(_laplacian(graph))
    degree = 2 * symbolic.eye(4)
    stiffness = degree * laplacian
    x = symbolic.Matrix(symbolic.symbols("x0:4", real=True))
    e, f = symbolic.symbols("e f", positive=True)
    z, origin = symbolic.symbols("z origin", real=True)
    g = symbolic.Function("g")
    capacity = x.applyfunc(g)
    h = e * x + f * capacity
    rate = -symbolic.diag(*capacity) * laplacian * h

    # These primitives require a supplied common positive g on the chart.
    # They are not inferred from a trajectory or from the nodal product.
    charge = sum(2 * symbolic.Integral(1 / g(z), (z, origin, value)) for value in x)
    energy = sum(
        2 * symbolic.Integral((e * z + f * g(z)) / g(z), (z, origin, value))
        for value in x
    )
    charge_gradient = symbolic.Matrix([charge.diff(value) for value in x])
    energy_gradient = symbolic.Matrix([energy.diff(value) for value in x])
    assert charge_gradient == symbolic.Matrix([2 / value for value in capacity])
    assert symbolic.simplify(charge_gradient.dot(rate)) == 0
    dissipation = (h.T * stiffness * h)[0]
    assert symbolic.simplify(energy_gradient.dot(rate) + dissipation) == 0
    assert (
        symbolic.expand(dissipation - sum((h[i] - h[j]) ** 2 for i, j in graph.edges))
        == 0
    )


def test_local_affine_capacity_full_jacobian_exposes_the_slope_boundary():
    symbolic = pytest.importorskip("sympy")
    laplacian = symbolic.Matrix(_laplacian(nx.cycle_graph(4)))
    x = symbolic.Matrix(symbolic.symbols("x0:4", real=True))
    a, b = symbolic.symbols("a b", real=True)
    e, f, capacity_at_a = symbolic.symbols("e f capacity_at_a", positive=True)
    capacity = x.applyfunc(lambda value: capacity_at_a + b * (value - a))
    h = e * x + f * capacity
    rate = -symbolic.diag(*capacity) * laplacian * h
    jacobian = rate.jacobian(x)
    full_expected = (
        -b * symbolic.diag(*(laplacian * h))
        - (e + f * b) * symbolic.diag(*capacity) * laplacian
    )
    assert symbolic.simplify(jacobian - full_expected) == symbolic.zeros(4)
    uniform_jacobian = jacobian.subs(dict.fromkeys(x, a))
    assert symbolic.simplify(
        uniform_jacobian + capacity_at_a * (e + f * b) * laplacian
    ) == symbolic.zeros(4)

    alternating = symbolic.Matrix([1, -1, 1, -1])
    assert laplacian * alternating == 2 * alternating
    coefficient = -2 * capacity_at_a * (e + f * b)
    assert symbolic.simplify(
        uniform_jacobian * alternating - coefficient * alternating
    ) == symbolic.zeros(4, 1)
    assert coefficient.subs(b, 0).is_negative
    assert coefficient.subs(b, -e / f) == 0
    assert coefficient.subs(b, -2 * e / f).is_positive
    # At cancellation the entire supplied affine law has zero pressure,
    # not merely a zero eigenvalue of the uniform-state linearization.
    assert symbolic.simplify(rate.subs(b, -e / f)) == symbolic.zeros(4, 1)


def test_positive_local_slopes_give_the_conditional_equilibrium_metric_identity():
    symbolic = pytest.importorskip("sympy")
    laplacian = symbolic.Matrix(_laplacian(nx.cycle_graph(4)))
    degree = 2 * symbolic.eye(4)
    capacities = symbolic.symbols("g0:4", positive=True)
    slopes = symbolic.symbols("s0:4", positive=True)
    # This is the linearization at a supplied equilibrium with h constant
    # and s_i=h'(x_i)>0; it does not establish existence of such an equilibrium.
    slope_matrix = symbolic.diag(*slopes)
    jacobian = -symbolic.diag(*capacities) * laplacian * slope_matrix
    metric = symbolic.diag(
        *(2 * slope / capacity for slope, capacity in zip(slopes, capacities))
    )
    assert symbolic.simplify(
        metric * jacobian + slope_matrix * degree * laplacian * slope_matrix
    ) == symbolic.zeros(4)
    null_direction = symbolic.Matrix([1 / slope for slope in slopes])
    assert jacobian * null_direction == symbolic.zeros(4, 1)
    charge_gradient = symbolic.Matrix([2 / capacity for capacity in capacities])
    # The sole connected-cycle null direction is not tangent to fixed Q.
    assert laplacian.rank() == 3
    assert charge_gradient.dot(null_direction).is_positive


@pytest.mark.parametrize("slope", [0, -1, -2])
def test_declared_local_capacity_slopes_distinguish_fresh_nodal_responses(slope):
    graph = nx.cycle_graph(4)
    graph.graph["DNFR_WEIGHTS"] = dict(phase=0.0, epi=0.5, vf=0.5, topo=0.0)
    epi = tuple(Q(1, 2) + Q((-1) ** node, 8) for node in graph)
    capacities = tuple(1 + slope * (value - Q(1, 2)) for value in epi)
    for node, value, capacity in zip(graph, epi, capacities, strict=True):
        graph.nodes[node].update(
            EPI=float(value), nu_f=float(capacity), theta=0.0, delta_nfr=0.0
        )
    # One predeclared state, fixed unit C4, held equal phases and no Gamma.
    # nu'=slope*x' is an additional comparison premise, not engine selection.
    default_compute_delta_nfr(graph)
    capture = capture_non_epi_forcing(graph)
    source = capture.snapshot
    laplacian = _laplacian(graph)
    e = f = Q(1, 2)
    expected_pressure = tuple(-(e + f * slope) * dot(row, epi) for row in laplacian)
    expected_rate = tuple(
        capacity * pressure
        for capacity, pressure in zip(capacities, expected_pressure, strict=True)
    )
    assert all(capacity > 0 for capacity in capacities)
    assert source.epi == epi and source.capacity == capacities
    assert (
        capture.kernel_pressure_defect == capture.stored_pressure_residual == (0,) * 4
    )
    assert capture.phase_gradient == source.topology_gradient == (0,) * 4
    assert source.stored_pressure == expected_pressure
    assert source.rate == expected_rate
    capacity_rate = tuple(slope * value for value in expected_rate)
    reference = derive_phase_response(
        cosine_gram=((1,) * 4,) * 4,
        mean_neighbors=source.support_neighbors,
        receiver_sources=tuple((node,) for node in graph),
        phase_factor=Q(0),
    )
    response = derive_joint_nodal_response(
        source,
        reference,
        epi_weight=e,
        phase_weight=Q(0),
        capacity_weight=f,
        phase_rate_over_pi=(0,) * 4,
        capacity_rate=capacity_rate,
    )
    expected_pressure_rate = tuple(
        -(e + f * slope) * dot(row, expected_rate) for row in laplacian
    )
    assert response.capacity_rate == capacity_rate
    assert response.capacity_pressure_rate == tuple(
        -f * dot(row, capacity_rate) for row in laplacian
    )
    assert response.pressure_rate == expected_pressure_rate
    assert response.epi_acceleration == tuple(
        slope * rate * pressure + capacity * pressure_rate
        for rate, pressure, capacity, pressure_rate in zip(
            expected_rate,
            expected_pressure,
            capacities,
            expected_pressure_rate,
            strict=True,
        )
    )
    # Exact initial change of the alternating contrast: damping, cancellation,
    # or growth. This finite state does not certify a complete trajectory.
    contrast_rate = (
        sum((-1) ** node * value for node, value in enumerate(source.rate)) / 4
    )
    assert contrast_rate == -(1 + slope) / Q(8)


def _capacity_admission_graph(epi, capacity):
    graph = _initial_graph()
    inject_defaults(graph)
    graph.graph.update(
        DNFR_WEIGHTS=dict(phase=0.0, epi=0.5, vf=0.5, topo=0.0),
        VF_ADAPT_TAU=2,
        VF_ADAPT_MU=0.25,
        EPS_DNFR_STABLE=0.5,
        VF_MIN=0.0,
        SELECTOR_THRESHOLDS={"si_hi": 0.0},
        _t=0.0,
    )
    for node, form, rate in zip(graph, epi, capacity, strict=True):
        graph.nodes[node].update(EPI=float(form), nu_f=float(rate))
    default_compute_delta_nfr(graph)
    compute_Si(graph, inplace=True)
    return graph


def _capacity_channel(graph, aliases):
    return tuple(Q(get_attr(graph.nodes[node], aliases, strict=True)) for node in graph)


def _execute_two_qualifying_capacity_calls(graph):
    """Execute the declared policy using current pressure and actual Si.

    The low Si cutoff and wide pressure cutoff are explicit comparison policy,
    not an emergent law or a claim that the default policy admits these states.
    """
    initial_capacity = _capacity_channel(graph, ALIAS_VF)
    for call in (1, 2):
        default_compute_delta_nfr(graph)
        compute_Si(graph, inplace=True)
        assert all(value >= 0 for value in _capacity_channel(graph, ALIAS_SI))
        assert all(
            abs(value) <= Q(1, 2) for value in _capacity_channel(graph, ALIAS_DNFR)
        )
        adapt_vf_after_structural_stability(graph, n_jobs=1)
        assert tuple(graph.nodes[node]["stable_count"] for node in graph) == (call,) * 2
        assert graph.graph["_t"] == 0.0  # Counts are invocations, not elapsed time.
        if call == 1:
            assert _capacity_channel(graph, ALIAS_VF) == initial_capacity


def test_capacity_policy_jump_leaves_local_form_law_and_changes_fresh_pressure():
    # Supplied graph law g(x)=x on x>0, with the existing pressure channels.
    graph = _capacity_admission_graph((Q(5, 8), Q(3, 8)), (Q(5, 8), Q(3, 8)))
    before = capture_non_epi_forcing(graph)
    assert before.snapshot.epi == before.snapshot.capacity == (Q(5, 8), Q(3, 8))
    assert before.snapshot.stored_pressure == (Q(-1, 4), Q(1, 4))
    assert before.snapshot.rate == (Q(-5, 32), Q(3, 32))
    assert before.stored_pressure_residual == before.kernel_pressure_defect == (0, 0)
    # A continuous graph-law lift would require nu_dot=g'(x)*x_dot=x_dot.
    # Holding capacity instead has this nonzero tangency defect.
    assert tuple(-rate for rate in before.snapshot.rate) == (Q(5, 32), Q(-3, 32))

    _execute_two_qualifying_capacity_calls(graph)
    after = capture_non_epi_forcing(graph)
    assert after.snapshot.epi == before.snapshot.epi
    assert after.snapshot.capacity == (Q(9, 16), Q(7, 16))
    assert tuple(
        capacity - form
        for capacity, form in zip(
            after.snapshot.capacity, after.snapshot.epi, strict=True
        )
    ) == (Q(-1, 16), Q(1, 16))
    assert after.snapshot.stored_pressure == before.snapshot.stored_pressure
    assert after.snapshot.rate == (Q(-9, 64), Q(7, 64))
    assert after.full_kernel_pressure == (Q(-3, 16), Q(3, 16))
    assert after.stored_pressure_residual == (Q(-1, 16), Q(1, 16))
    assert after.kernel_pressure_defect == (0, 0)
    assert after.snapshot.epi_gradient == before.snapshot.epi_gradient
    assert before.snapshot.capacity_gradient == (Q(-1, 4), Q(1, 4))
    assert after.snapshot.capacity_gradient == (Q(-1, 8), Q(1, 8))
    # Execute the actual refresh; the detached predicted pressure is not used
    # as an externally supplied replacement for the pressure law.
    default_compute_delta_nfr(graph)
    refreshed = capture_non_epi_forcing(graph)
    assert refreshed.snapshot.stored_pressure == after.full_kernel_pressure
    assert refreshed.snapshot.rate == (Q(-27, 256), Q(21, 256))
    assert refreshed.stored_pressure_residual == (0, 0)


def test_capacity_event_can_reactivate_zero_outside_unforced_local_law_tangency():
    # This is a separately admitted nonnegative boundary of g(x)=x; the
    # positive-g charge/energy theorem above is not applied at zero capacity.
    graph = _capacity_admission_graph((0, Q(1, 2)), (0, Q(1, 2)))
    before = capture_non_epi_forcing(graph)
    assert before.snapshot.stored_pressure == (Q(1, 2), Q(-1, 2))
    assert before.snapshot.rate == (0, Q(-1, 4))
    assert before.snapshot.capacity[0] == before.snapshot.epi[0] == 0
    # A finite C1 local g' times this zero unforced form row requires nu_dot=0
    # at node 0. A capacity-only hybrid jump has a different admission test.
    _execute_two_qualifying_capacity_calls(graph)
    after = capture_non_epi_forcing(graph)
    assert after.snapshot.epi == before.snapshot.epi
    assert after.snapshot.capacity == (Q(1, 8), Q(3, 8))
    assert after.snapshot.rate == (Q(1, 16), Q(-3, 16))
    assert after.full_kernel_pressure == (Q(3, 8), Q(-3, 8))
    default_compute_delta_nfr(graph)
    refreshed = capture_non_epi_forcing(graph)
    assert refreshed.snapshot.rate == (Q(3, 64), Q(-9, 64))
    assert refreshed.stored_pressure_residual == (0, 0)


def test_uniform_capacity_policy_fixed_point_preserves_a_constant_graph_law():
    graph = _capacity_admission_graph((Q(5, 8), Q(3, 8)), (Q(1, 2), Q(1, 2)))
    before = capture_non_epi_forcing(graph)
    assert before.snapshot.rate == (Q(-1, 16), Q(1, 16))
    assert before.snapshot.capacity_gradient == (0, 0)
    _execute_two_qualifying_capacity_calls(graph)
    after = capture_non_epi_forcing(graph)
    assert after.snapshot.epi == before.snapshot.epi
    assert after.snapshot.capacity == before.snapshot.capacity == (Q(1, 2),) * 2
    assert after.snapshot.rate == before.snapshot.rate
    assert after.stored_pressure_residual == (0, 0)
    # g=1/2 has g'=0: held capacity is tangent even while EPI moves, and this
    # particular native event preserves the relation. Not every writer fails.
