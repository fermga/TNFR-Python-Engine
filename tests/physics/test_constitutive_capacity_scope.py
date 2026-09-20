"""Detached capacity-law independence and optional constitutive-law controls.

The two analytic completions are counterexamples to a claimed implication,
not proposed TNFR laws, executed trajectories or runtime certificates.
"""

from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.canonical import compute_extended_nodal_system
from tnfr.dynamics.dnfr import default_compute_delta_nfr
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
