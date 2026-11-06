"""Static sufficient-link controls for the supplied relational law.

Exact Krylov products below concern materialized generator coefficients.
Independent ideal consensus formulas are compared numerically; they are not
transcendental certificates. No trajectory, event or physical bond is inferred.
See RELATIONAL_PATTERN_COMPOSITION.md#effective-link-admission for the proofs.
"""

import math
from fractions import Fraction as Q

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
    evaluate_relational_uniform_tangent,
)
from tnfr.mathematics._exact_linear_algebra import exact_matrix_product as product
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.mathematics.linear_observation import derive_coordinate_memory

MODEL = RelationalExchangeModel(1.25, epi_weight=3.0, phase_weight=2.0)
RING_MODEL = RelationalExchangeModel(1.0)
ZERO_BLOCK = ((Q(0), Q(0)), (Q(0), Q(0)))


def _reference(graph, capacities):
    for node in graph:
        graph.nodes[node].update(EPI=-0.25, theta=0.0, nu_f=capacities[node])
    return graph


def _consensus_edge_matrix():
    """Ideal common two-coordinate block, independent of graph assembly."""
    return np.array(
        [
            [MODEL.epi_weight, MODEL.phase_weight / math.pi],
            [-MODEL.phase_weight / (MODEL.storage_scale * math.pi), 0.0],
        ]
    )


def _moments(report, donor, receiver, maximum_power):
    """Two-column exact Krylov recurrence, avoiding full matrix powers/ranks."""
    size = len(report.field.nodes)
    donor_index = report.field.nodes.index(donor)
    receiver_index = report.field.nodes.index(receiver)
    seed = tuple(
        (Q(i == donor_index), Q(i == size + donor_index)) for i in range(2 * size)
    )
    generator = tuple(tuple(map(Q, row)) for row in report.generator)
    result = []
    for power in range(maximum_power + 1):
        result.append((seed[receiver_index], seed[size + receiver_index]))
        if power < maximum_power:
            seed = product(generator, seed)
    return tuple(result)


def _determinant(block):
    return block[0][0] * block[1][1] - block[0][1] * block[1][0]


def _mediated_rings(*, a=0, delta=0, eta=0):
    """Declared ring offset and independent mediator form/phase preparation."""
    graph = nx.Graph()
    graph.add_nodes_from(range(11))
    graph.add_edges_from(
        (offset + i, offset + (i + 1) % 5) for offset in (0, 5) for i in range(5)
    )
    graph.add_edges_from(((0, 10), (10, 5)))
    for node in graph:
        phase = math.tau * (node % 5) / 5 if node < 10 else 0.0
        offset = 0 if node < 5 else delta if node < 10 else delta / 2 + eta
        graph.nodes[node].update(
            EPI=float(a) if node == 10 else 0.0,
            nu_f=1.0,
            theta=phase + float(offset),
        )
    return graph


@pytest.mark.parametrize("distance", (3, 4))
def test_unique_path_first_nonzero_block_uses_receiving_capacities(distance):
    path = ("source", "inner-b", "inner-a", "far", "receiver")[: distance + 1]
    graph = nx.Graph()
    graph.add_nodes_from(reversed(path))
    graph.add_edges_from(zip(path, path[1:]))
    capacities = dict(zip(path, (1.5, 0.5, 2.0, 1.25, 0.75)))
    _reference(graph, capacities)
    report = evaluate_relational_uniform_tangent(graph, model=MODEL)
    blocks = _moments(report, path[0], path[-1], distance)
    assert blocks[:distance] == (ZERO_BLOCK,) * distance
    assert _determinant(blocks[distance]) != 0
    coefficient = math.prod(capacities[node] / graph.degree[node] for node in path[1:])
    expected = coefficient * np.linalg.matrix_power(_consensus_edge_matrix(), distance)
    np.testing.assert_allclose(
        np.array(blocks[distance], float), expected, rtol=3e-15, atol=1e-17
    )

    # The full independent Kronecker formula also checks the node-major
    # coordinate order and the diagonal blocks, not just the selected path.
    nodes = report.field.nodes
    adjacency = nx.to_numpy_array(graph, nodelist=nodes)
    degrees = adjacency.sum(axis=1)
    laplacian = np.diag(degrees) - adjacency
    mobility = np.diag([capacities[node] / graph.degree[node] for node in nodes])
    expected_full = np.kron(-mobility @ laplacian, _consensus_edge_matrix())
    interleaved = tuple(
        index for i in range(len(nodes)) for index in (i, len(nodes) + i)
    )
    native = np.array(report.generator)[np.ix_(interleaved, interleaved)]
    np.testing.assert_allclose(native, expected_full, rtol=3e-15, atol=1e-17)


def test_diamond_shortest_paths_add_with_positive_scalar_weights_at_consensus():
    graph = nx.Graph()
    graph.add_nodes_from(("receiver", "upper", "source", "lower"))
    graph.add_edges_from(
        (
            ("source", "upper"),
            ("upper", "receiver"),
            ("source", "lower"),
            ("lower", "receiver"),
        )
    )
    capacities = {"source": 0.75, "upper": 0.5, "lower": 1.5, "receiver": 2.0}
    report = evaluate_relational_uniform_tangent(
        _reference(graph, capacities), model=MODEL
    )
    blocks = _moments(report, "source", "receiver", 2)
    assert blocks[:2] == (ZERO_BLOCK,) * 2
    upper_weight = capacities["receiver"] * capacities["upper"] / 4
    lower_weight = capacities["receiver"] * capacities["lower"] / 4
    expected = (upper_weight + lower_weight) * np.linalg.matrix_power(
        _consensus_edge_matrix(), 2
    )
    assert upper_weight > 0 and lower_weight > 0 and _determinant(blocks[2]) != 0
    np.testing.assert_allclose(
        np.array(blocks[2], float), expected, rtol=3e-15, atol=1e-17
    )


def test_eliminated_middle_node_retains_the_exact_second_moment_in_memory():
    graph = _reference(nx.path_graph(3), {0: 1.0, 1: 0.5, 2: 2.0})
    report = evaluate_relational_uniform_tangent(graph, model=MODEL)
    memory = derive_coordinate_memory(report.generator, (0, 2, 3, 5))
    receiver, donor = (1, 3), (0, 2)
    direct = tuple(
        tuple(memory.visible_generator[i][j] for j in donor) for i in receiver
    )
    kernel = tuple(tuple(memory.kernel_at_zero[i][j] for j in donor) for i in receiver)
    moments = _moments(report, 0, 2, 2)
    assert direct == moments[1] == ZERO_BLOCK
    assert kernel == moments[2] and _determinant(kernel) != 0
    assert (
        kernel[1][1] < 0
    )  # Joint phase/form memory is not a positive diffusion kernel.
    assert memory.hidden_indices == (1, 4)
    expected = 0.5 * np.linalg.matrix_power(_consensus_edge_matrix(), 2)
    np.testing.assert_allclose(
        np.array(kernel, float), expected, rtol=3e-15, atol=1e-17
    )


def test_zero_capacity_separator_is_an_exact_static_null_until_a_bypass_exists():
    graph = _reference(nx.path_graph(5), {0: 1.0, 1: 1.0, 2: 0.0, 3: 1.0, 4: 1.0})
    report = evaluate_relational_uniform_tangent(graph, model=MODEL)
    dimension = len(report.generator)
    blocks = _moments(report, 0, 4, dimension - 1)
    # Cayley-Hamilton extends these zeros to all orders for this supplied
    # finite rational generator. No sampled-time observation is used.
    assert blocks == (ZERO_BLOCK,) * dimension
    graph.add_edge(1, 3)
    bypass = evaluate_relational_uniform_tangent(graph, model=MODEL)
    bypass_blocks = _moments(bypass, 0, 4, 3)
    assert bypass_blocks[:3] == (ZERO_BLOCK,) * 3
    assert _determinant(bypass_blocks[3]) != 0
    assert bypass.generator[2] == bypass.generator[7] == (0.0,) * dimension


def test_frozen_donor_can_influence_a_receiver_through_its_initial_state():
    graph = _reference(nx.path_graph(2), {0: 0.0, 1: 2.0})
    report = evaluate_relational_uniform_tangent(graph, model=MODEL)
    assert report.generator[0] == report.generator[2] == (0.0,) * 4
    assert _determinant(_moments(report, 0, 1, 1)[1]) != 0
    baseline = evaluate_relational_exchange(graph, model=MODEL)
    amplitude = 1 / 16
    graph.nodes[0]["EPI"] += amplitude
    changed = evaluate_relational_exchange(graph, model=MODEL)
    assert changed.form_rate[0] == changed.phase_rate[0] == 0.0
    response = (
        changed.form_rate[1] - baseline.form_rate[1],
        changed.phase_rate[1] - baseline.phase_rate[1],
    )
    expected = amplitude * 2 * _consensus_edge_matrix()[:, 0]
    assert response == pytest.approx(expected, rel=3e-15, abs=1e-17)


def test_nonlinear_receiver_rows_depend_only_on_unchanged_local_state_across_a_cut():
    graph = _reference(nx.path_graph(5), {0: 1.0, 1: 1.0, 2: 0.0, 3: 1.0, 4: 1.0})
    for node, form, phase in zip(
        graph, (0.125, -0.2, 0.4, -0.1, 0.3), (-0.15, -0.05, 0.1, 0.2, 0.3)
    ):
        graph.nodes[node].update(EPI=form, theta=phase)
    first = evaluate_relational_exchange(graph, model=MODEL)
    other = graph.copy()
    other.nodes[0].update(EPI=0.25, theta=-0.2)
    other.nodes[1].update(EPI=0.125, theta=-0.15)
    second = evaluate_relational_exchange(other, model=MODEL)
    assert first.form_rate[1] != second.form_rate[1]
    assert first.form_rate[2] == first.phase_rate[2] == 0.0
    assert second.form_rate[2] == second.phase_rate[2] == 0.0
    assert first.form_rate[3:] == second.form_rate[3:]
    assert first.phase_rate[3:] == second.phase_rate[3:]
    # These two snapshots check local dependence only. Future isolation needs
    # the separate held-support/capacity and regular-law uniqueness premises.


def test_shifted_two_ring_witness_is_admitted_inside_the_declared_recovery_budget():
    graph = _mediated_rings()
    baseline = evaluate_relational_exchange(graph, model=RING_MODEL)
    delta = Q(1, 128)
    offsets = (Q(0),) * 5 + (delta,) * 5 + (delta / 2,)
    for node, offset in enumerate(offsets):
        graph.nodes[node]["theta"] += float(offset)
    witness = evaluate_relational_exchange(graph, model=RING_MODEL)
    assert witness.pressure[0] > 0 > witness.pressure[5]
    assert witness.phase_rate == (0.0,) * 11
    assert float(witness.storage - baseline.storage) == pytest.approx(
        4 * math.sin(float(delta) / 4) ** 2, rel=0, abs=3e-15
    )

    # These are declared lift offsets, not differences inferred by wrapping
    # rounded phases. Bounds check this witness; the theorem owns convergence.
    mean_offset = sum(offsets, Q(0)) / len(offsets)
    norm_squared = sum(((value - mean_offset) ** 2 for value in offsets), Q(0))
    assert norm_squared == 5 * delta**2 / 2 == Q(5, 32768)
    radius_squared = pi_interval() ** 2 / 800
    radius_floor = Q(9, 800)
    assert radius_squared.lo > radius_floor > norm_squared
    curvature_floor = Q(1, 10)
    assert sin(pi_interval() / 20).lo > curvature_floor
    spectral_floor = Q(4, len(graph) * nx.diameter(graph))
    assert spectral_floor == Q(2, 33)
    storage_floor = curvature_floor * spectral_floor * radius_floor / 2
    assert storage_floor == Q(9, 264000) > delta**2 / 4
    ideal_excess = 4 * sin(I(delta) / 4) ** 2
    assert ideal_excess.hi < delta**2 / 4 == Q(1, 65536)


def test_environmental_family_has_independent_native_storage_and_centered_norm():
    a, delta, eta = Q(1, 256), -Q(1, 128), -Q(1, 1024)
    baseline = evaluate_relational_exchange(_mediated_rings(), model=RING_MODEL)
    field = evaluate_relational_exchange(
        _mediated_rings(a=a, delta=delta, eta=eta), model=RING_MODEL
    )
    expected_excess = float(a**2) + 2 * (
        1 - math.cos(float(delta) / 2) * math.cos(float(eta))
    )
    assert float(field.storage - baseline.storage) == pytest.approx(
        expected_excess, rel=0, abs=4e-15
    )
    assert field.form_storage == a**2
    offsets = (Q(0),) * 5 + (delta,) * 5 + (delta / 2 + eta,)
    forms = tuple(map(Q, field.epi))
    norm_squared = sum(
        sum(((value - sum(values, Q(0)) / 11) ** 2 for value in values), Q(0))
        for values in (forms, offsets)
    )
    assert norm_squared == Q(10, 11) * (a**2 + eta**2) + Q(5, 2) * delta**2
    budget = a**2 + eta**2 + delta**2 / 4
    assert norm_squared <= 10 * budget
    assert any(field.form_rate) and any(field.phase_rate)


def test_nonzero_environmental_preparation_satisfies_both_certified_basin_conditions():
    a, delta, eta = Q(1, 512), Q(1, 256), Q(1, 512)
    field = evaluate_relational_exchange(
        _mediated_rings(a=a, delta=delta, eta=eta), model=RING_MODEL
    )
    budget = a**2 + eta**2 + delta**2 / 4
    assert budget == Q(3, 262144)
    radius_squared = pi_interval() ** 2 / 800
    kappa = radius_squared * sin(pi_interval() / 20) / 33
    assert budget < Q(9, 264000) < kappa.lo
    ideal_excess = I(a) ** 2 + 2 * (1 - cos(I(delta) / 2) * cos(I(eta)))
    assert ideal_excess.hi < budget
    norm_squared = Q(10, 11) * (a**2 + eta**2) + Q(5, 2) * delta**2
    assert norm_squared <= 10 * budget < radius_squared.lo
    assert (10 * kappa).hi < radius_squared.lo
    assert field.form_rate[5] > 0 > field.phase_rate[5]
    # Static admission/budget checks, not a numerical capture trajectory.


def test_identical_visible_preparations_and_mediator_pressure_can_give_distinct_port_rates():
    delta, eta = Q(1, 256), Q(1, 512)
    e, w, beta = (
        RING_MODEL.epi_weight,
        RING_MODEL.phase_weight,
        RING_MODEL.storage_scale,
    )
    a = -w * float(eta) / (e * math.pi)
    baseline = evaluate_relational_exchange(
        _mediated_rings(delta=delta), model=RING_MODEL
    )
    compensated = evaluate_relational_exchange(
        _mediated_rings(a=a, delta=delta, eta=eta), model=RING_MODEL
    )
    assert baseline.epi[:10] == compensated.epi[:10]
    assert baseline.phase[:10] == compensated.phase[:10]
    assert baseline.capacity == compensated.capacity
    assert abs(baseline.pressure[10]) < 3e-17
    assert abs(compensated.pressure[10]) < 3e-17
    assert baseline.phase_rate[5] == 0.0
    r = 1 + 2 * math.cos(math.tau / 5)
    assert compensated.phase_rate[5] == pytest.approx(
        -w * a / (beta * math.pi * r), rel=3e-15, abs=1e-17
    )
    assert compensated.phase_rate[5] > 0
    assert compensated.form_rate[5] != baseline.form_rate[5]

    # Ideal compensated a uses mathematical pi; the native state above is
    # its binary64 witness, whose pressure residual is not forced to zero.
    ideal_a = -Q(w) * I(eta) / (Q(e) * pi_interval())
    compensated_budget = ideal_a**2 + I(eta) ** 2 + I(delta) ** 2 / 4
    kappa = pi_interval() ** 2 * sin(pi_interval() / 20) / (800 * 33)
    assert delta**2 / 4 < compensated_budget.lo < compensated_budget.hi < kappa.lo
