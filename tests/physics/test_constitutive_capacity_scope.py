"""Detached capacity-law independence and optional constitutive-law controls.

The two analytic completions are counterexamples to a claimed implication,
not proposed TNFR laws, executed trajectories or runtime certificates.
"""

from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.canonical import compute_extended_nodal_system
from tnfr.physics._cycle_algebra import dot
from tnfr.physics.fields import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length_with_provenance,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.phase_response import derive_joint_nodal_response, derive_phase_response
from tnfr.physics.structural_diffusion import (
    compute_diffusion_energy,
    structural_diffusion_operator,
)


def _initial_graph():
    graph = nx.path_graph(2)
    for node, epi in enumerate((0.625, 0.375)):
        graph.nodes[node].update(
            EPI=epi, nu_f=1.0, theta=0.0, delta_nfr=0.0,
            glyph_history=[], epi_history=[epi], stable_count=0,
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
    assert source.snapshot.capacity_gradient == source.snapshot.topology_gradient == (0, 0)
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
    assert tuple(dot(row, source.snapshot.epi) for row in laplacian) == (Q(1, 4), Q(-1, 4))

    # nu'_A=(0,0), nu'_B=(1,1). Both have zero capacity-source derivative.
    capacity_rates = ((Q(0), Q(0)), (Q(1), Q(1)))
    assert all(tuple(dot(row, rate_nu) for row in laplacian) == (0, 0)
               for rate_nu in capacity_rates)
    reference = derive_phase_response(
        cosine_gram=((1, 1), (1, 1)), mean_neighbors=((1,), (0,)),
        receiver_sources=((0,), (1,)), phase_factor=Q(0),
    )
    weights = dict(source.normalized_weights)
    responses = tuple(
        derive_joint_nodal_response(
            source.snapshot, reference, epi_weight=weights["epi"],
            phase_weight=weights["phase"], capacity_weight=weights["vf"],
            phase_rate_over_pi=(0, 0), capacity_rate=rate_nu,
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

    laplacian = symbolic.Matrix([
        [rational(value) for value in row] for row in _laplacian(graph)
    ])
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
        assert symbolic.simplify(epi.diff(t) - capacity * pressure) == symbolic.zeros(2, 1)
        assert epi.subs(t, 0) == symbolic.Matrix([rational(v) for v in source.snapshot.epi])
        assert pressure.subs(t, 0) == symbolic.Matrix([
            rational(v) for v in source.snapshot.stored_pressure
        ])
        assert laplacian * (capacity * ones) == symbolic.zeros(2, 1)
        energy = rational(source.snapshot.dirichlet_energy) * amplitude**2
        assert symbolic.simplify(energy.diff(t) + 4 * e * capacity * energy) == 0
        assert (-energy.diff(t)).is_positive
        fields.append(epi)
        energies.append(energy)

    assert fields[0].diff(t).subs(t, 0) == fields[1].diff(t).subs(t, 0)
    assert symbolic.simplify(
        (fields[1] - fields[0]).diff(t, 2).subs(t, 0)
    ) == -2 * e * d * mode
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
        compute_extended_nodal_system(1.0, pressure, 0.0, 0.0, divergence).dnfr_derivative
        for pressure in (-0.5, 0.0, 0.5)
    )
    assert rates == (-(1.0 + 0.135) * divergence,) * 3
