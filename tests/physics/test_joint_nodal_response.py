"""Exact fixed-support joint response controls, not constitutive or runtime laws.

Runtime pressure is captured where it can be matched exactly. Rational phase
geometry remains detached: matching support does not authenticate a live chart.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
import math

import networkx as nx
import pytest

from tnfr.mathematics.krylov import exact_rank
from tnfr.physics._cycle_algebra import dot
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.phase_response import (
    derive_joint_nodal_response,
    derive_phase_response,
)
from tnfr.physics.support_transport import observe_support_transport


def _capture(graph, epi, capacity, phases=None, *, exact_pressure=True):
    phases = phases or (0.0,) * len(graph)
    graph.graph["DNFR_WEIGHTS"] = {
        "epi": 0.5, "phase": 0.25, "vf": 0.25, "topo": 0.0,
    }
    for node, x, nu, phase in zip(graph, epi, capacity, phases, strict=True):
        graph.nodes[node].update(EPI=x, nu_f=nu, theta=phase, delta_nfr=0.0)
    initial = capture_non_epi_forcing(graph)
    for node, pressure in zip(graph, initial.full_kernel_pressure, strict=True):
        graph.nodes[node]["delta_nfr"] = float(pressure)
    captured = capture_non_epi_forcing(graph)
    assert captured.stored_pressure_residual == (0,) * len(graph)
    if exact_pressure:
        assert captured.kernel_pressure_defect == (0,) * len(graph)
    return captured.snapshot


def _reference(snapshot, vectors=None, *, neighbors=None):
    vectors = vectors or ((Q(1), Q(0)),) * len(snapshot.nodes)
    gram = tuple(tuple(dot(left, right) for right in vectors) for left in vectors)
    return derive_phase_response(
        cosine_gram=gram,
        mean_neighbors=neighbors or snapshot.support_neighbors,
        receiver_sources=tuple((i,) for i in range(len(vectors))),
        # A disabled operator stage is still distinct from the source response.
        phase_factor=Q(0),
    )


def _joint(snapshot, reference=None, **overrides):
    options = dict(
        epi_weight=Q(1, 2), phase_weight=Q(1, 4), capacity_weight=Q(1, 4),
        phase_rate_over_pi=(0,) * len(snapshot.nodes),
        capacity_rate=(0,) * len(snapshot.nodes),
    )
    options.update(overrides)
    return derive_joint_nodal_response(
        snapshot, reference or _reference(snapshot), **options,
    )


def _compensation_source():
    return _capture(nx.path_graph(2), (0.25, 0.0), (1.0, 1.5))


def test_common_rotation_has_no_phase_pressure_response_and_does_not_mutate_graph():
    graph = nx.path_graph(3)
    source = _capture(graph, (0.125, 0.25, 0.375), (1.0,) * 3)
    before = deepcopy(graph)
    fixed = _joint(source)
    rotating = _joint(source, phase_rate_over_pi=(1, 1, 1))

    assert any(source.stored_pressure)
    assert rotating.phase_pressure_rate == (0, 0, 0)
    assert rotating.pressure_rate == fixed.pressure_rate
    assert rotating.epi_acceleration == fixed.epi_acceleration
    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))
    assert dict(graph.edges) == dict(before.edges)
    assert graph.graph == before.graph


def test_consensus_phase_and_capacity_directions_cancel_without_a_zero_source_jacobian():
    source = _compensation_source()
    reference = _reference(source)
    result = _joint(source, reference, phase_rate_over_pi=(1, -1), capacity_rate=(-1, 1))

    assert source.stored_pressure == source.rate == (0, 0)
    assert reference.jacobian == ((1, 0), (0, 1))
    assert result.phase_geometry.scaled_source_jacobian == ((-1, 1), (1, -1))
    assert result.phase_geometry.rank == 1
    assert result.phase_geometry.tangent_dimension == 1
    assert result.epi_pressure_rate == (0, 0)
    assert result.phase_pressure_rate == (Q(-1, 2), Q(1, 2))
    assert result.capacity_pressure_rate == (Q(1, 2), Q(-1, 2))
    assert result.pressure_rate == result.epi_acceleration == (0, 0)
    assert result.capacity_acceleration == result.pressure_acceleration == (0, 0)


def test_zero_pressure_and_zero_epi_rate_do_not_certify_joint_equilibrium():
    source = _compensation_source()
    result = _joint(source, phase_rate_over_pi=(1, -1), capacity_rate=(0, 0))
    assert source.stored_pressure == source.rate == (0, 0)
    assert result.pressure_rate == (Q(-1, 2), Q(1, 2))
    assert result.epi_acceleration == (Q(-1, 2), Q(3, 4))
    # A smooth phase completion can leave the zero-pressure set immediately.
    # Its existence is not a proposed autonomous law or an executed trajectory.


def test_strict_u3_star_has_a_phase_response_no_capacity_direction_can_cancel():
    graph = nx.star_graph(3)
    source = _capture(
        graph, (0.25,) * 4, (1.0,) * 4,
        (0.0, 0.0, 0.0, math.atan2(4, 3)), exact_pressure=False,
    )
    vectors = ((Q(1), Q(0)),) * 3 + ((Q(3, 5), Q(4, 5)),)
    reference = _reference(source, vectors)
    degree = (3, 1, 1, 1)
    # The exact rational Gram has strictly positive edge cosines. Rounded
    # runtime angles above are not asserted to realize that exact Gram.
    assert all(reference.cosine_gram[i][j] > 0 for i, j in graph.edges)
    assert reference.mean_response[0] == (0, Q(13, 37), Q(13, 37), Q(11, 37))
    result = _joint(
        source, reference, phase_rate_over_pi=(0, 0, 0, 1), capacity_rate=(2, -3, 5, 7),
    )
    jacobian = result.phase_geometry.scaled_source_jacobian
    assert result.phase_geometry.rank == 3
    assert result.phase_geometry.tangent_dimension == 1
    assert all(dot(row, (1, 1, 1, 1)) == 0 for row in jacobian)
    assert tuple(dot(degree, column) for column in zip(*jacobian)) == (
        0, Q(2, 37), Q(2, 37), Q(-4, 37),
    )
    assert dot(degree, result.phase_pressure_rate) == Q(-1, 37)
    assert dot(degree, result.capacity_pressure_rate) == 0
    # Check the whole capacity-response range, not just the example rate:
    # d^T B_U = 0 prevents cancelling this nonzero weighted phase sum.
    for direction in ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)):
        response = _joint(source, reference, capacity_rate=direction)
        assert dot(degree, response.capacity_pressure_rate) == 0


def test_weighted_epi_and_unweighted_support_keep_zero_edges_and_single_loops_distinct():
    graph = nx.Graph()
    graph.add_nodes_from(range(3))
    graph.add_weighted_edges_from(((0, 1, 2.0), (1, 2, 0.0), (0, 2, 1.0), (0, 0, 3.0)))
    source = _capture(graph, (0.0, 1.0, 1.0), (1.0,) * 3)
    assert source.support_neighbors == ((0, 1, 2), (0, 2), (0, 1))
    assert source.stored_pressure == (Q(1, 4), Q(-1, 2), Q(-1, 2))
    result = _joint(source, phase_rate_over_pi=(0, 1, 0), capacity_rate=(0, 1, 0))

    assert result.epi_pressure_rate == (Q(-3, 16), Q(3, 8), Q(3, 8))
    assert result.phase_pressure_rate == (Q(1, 12), Q(-1, 4), Q(1, 8))
    assert result.capacity_pressure_rate == result.phase_pressure_rate
    assert result.pressure_rate == (Q(-1, 48), Q(-1, 8), Q(5, 8))
    assert result.capacity_acceleration == (0, Q(-1, 2), 0)
    assert result.epi_acceleration == (Q(-1, 48), Q(-5, 8), Q(5, 8))

    reordered = _reference(
        source, neighbors=tuple(tuple(reversed(row)) for row in source.support_neighbors),
    )
    reordered_result = _joint(
        source, reordered, phase_rate_over_pi=(0, 1, 0), capacity_rate=(0, 1, 0),
    )
    assert reordered_result.pressure_rate == result.pressure_rate


def test_symmetric_conductance_does_not_imply_undirected_unique_support():
    graph = nx.DiGraph()
    graph.add_nodes_from(range(3))
    graph.add_weighted_edges_from((
        (0, 1, 1.0), (1, 0, 1.0), (1, 2, 1.0), (2, 1, 1.0), (0, 2, 0.0),
    ))
    source = _capture(graph, (0.25, 0.5, 0.75), (1.0,) * 3)
    weights = {(i, j): value for i, j, value in source.conductance}
    assert all(weights[j, i] == value for (i, j), value in weights.items())
    assert (0, 2) not in weights
    assert source.support_neighbors == ((1, 2), (0, 2), (1,))

    # This support contains a one-way zero-conductance arc. Its explicit
    # unweighted B_U=U-I does not have the support-degree left null vector.
    support_degree = (2, 2, 1)
    support_difference = ((-1, Q(1, 2), Q(1, 2)), (Q(1, 2), -1, Q(1, 2)), (0, 1, -1))
    assert tuple(dot(support_degree, column) for column in zip(*support_difference)) == (-1, 0, 1)

    # The joint observer still admits this effective-symmetric model and
    # computes its declared derivative. The later undirected-support range
    # theorem must impose its stronger premise independently.
    result = _joint(source, capacity_rate=(1, 0, 0))
    assert source.stored_pressure == (Q(1, 8), 0, Q(-1, 8))
    assert result.epi_pressure_rate == (Q(-1, 16), 0, Q(1, 16))
    assert result.phase_pressure_rate == (0, 0, 0)
    assert result.capacity_pressure_rate == (Q(-1, 4), Q(1, 8), 0)
    assert dot(support_degree, result.capacity_pressure_rate) == Q(-1, 4)
    assert result.pressure_rate == (Q(-5, 16), Q(1, 8), Q(1, 16))
    assert result.capacity_acceleration == (Q(1, 8), 0, 0)
    assert result.epi_acceleration == (Q(-3, 16), Q(1, 8), Q(1, 16))


@pytest.mark.parametrize("star", (False, True))
def test_joint_source_rank_is_not_the_phase_only_rank(star):
    if star:
        source = _capture(nx.star_graph(3), (0.25,) * 4, (1.0,) * 4)
        vectors = ((Q(1), Q(0)),) * 3 + ((Q(3, 5), Q(4, 5)),)
        reference = _reference(source, vectors)
        # This exact geometry is deliberately separate from the captured
        # consensus phase. Only the declared source differential is tested.
        expected_rank, expected_kernel = 4, 4
    else:
        source = _compensation_source()
        reference = _reference(source)
        expected_rank, expected_kernel = 1, 3
    size = len(source.nodes)
    columns = []
    for channel in ("phase_rate_over_pi", "capacity_rate"):
        for index in range(size):
            direction = tuple(int(j == index) for j in range(size))
            response = _joint(source, reference, **{channel: direction})
            columns.append(tuple(a + b for a, b in zip(
                response.phase_pressure_rate, response.capacity_pressure_rate, strict=True,
            )))
    matrix = tuple(zip(*columns))
    assert exact_rank(matrix) == expected_rank
    assert 2 * size - exact_rank(matrix) == expected_kernel


def test_channel_coefficients_are_used_as_declared_without_normalization():
    source = _compensation_source()
    result = _joint(source, phase_weight=3, capacity_weight=4,
                    phase_rate_over_pi=(1, -1), capacity_rate=(-1, 1))
    assert result.phase_weight == 3
    assert result.capacity_weight == 4
    assert result.phase_pressure_rate == (-6, 6)
    assert result.capacity_pressure_rate == (8, -8)
    assert result.pressure_rate == (2, -2)
    # These intentionally different weights are not a live pressure certificate.
    assert "phase_gram_state_and_pressure_compatibility_not_certified" in result.scope


def test_stale_snapshot_caches_are_recomputed_from_primitive_state():
    source = _compensation_source()
    corrupted = replace(source, rate=(17, 19), epi_gradient=(23, 29),
                        capacity_gradient=(31, 37), energy_rate=41)
    result = _joint(corrupted, capacity_rate=(1, 1))
    expected = _joint(source, capacity_rate=(1, 1))
    assert result == expected
    assert result.source == source


def test_tampered_phase_response_cache_is_rejected():
    source = _compensation_source()
    reference = replace(_reference(source), mean_response=((1, 0), (0, 1)))
    with pytest.raises(ValueError, match="rebuilt coefficients"):
        _joint(source, reference)


@pytest.mark.parametrize("name", ("epi_weight", "phase_weight", "capacity_weight"))
@pytest.mark.parametrize("value,error", (
    (-1, ValueError), (float("nan"), ValueError), (float("inf"), ValueError), (True, TypeError),
))
def test_invalid_channel_weights_are_rejected(name, value, error):
    with pytest.raises(error):
        _joint(_compensation_source(), **{name: value})


@pytest.mark.parametrize("name", ("phase_rate_over_pi", "capacity_rate"))
@pytest.mark.parametrize("value,error", (
    ((0,), ValueError), ((0, 0, 0), ValueError), ((0, float("nan")), ValueError),
    ((0, True), TypeError), ({0, 1}, TypeError), ({0: 0, 1: 0}, TypeError),
))
def test_invalid_or_unordered_rate_vectors_are_rejected(name, value, error):
    with pytest.raises(error):
        _joint(_compensation_source(), **{name: value})


def test_phase_reference_requires_the_same_support_even_when_phase_weight_is_zero():
    source = _compensation_source()
    different = _reference(source, neighbors=((0,), (1,)))
    with pytest.raises(ValueError, match="match the snapshot support"):
        _joint(source, different, phase_weight=0)


@pytest.mark.parametrize("count", (0, 1))
def test_empty_graphs_and_isolates_are_explicitly_outside_the_joint_domain(count):
    graph = nx.empty_graph(count)
    for node in graph:
        graph.nodes[node].update(EPI=0.0, nu_f=1.0, delta_nfr=0.0)
    source = observe_support_transport(graph)
    reference = _reference(_compensation_source())
    with pytest.raises(ValueError, match="nonempty support"):
        _joint(source, reference, epi_weight=0, phase_weight=0, capacity_weight=0)


def test_inputs_must_be_typed_and_the_result_is_immutable():
    source = _compensation_source()
    reference = _reference(source)
    kwargs = dict(epi_weight=0, phase_weight=0, capacity_weight=0,
                  phase_rate_over_pi=(0, 0), capacity_rate=(0, 0))
    with pytest.raises(TypeError):
        derive_joint_nodal_response({}, reference, **kwargs)
    with pytest.raises(TypeError):
        derive_joint_nodal_response(source, {}, **kwargs)
    result = derive_joint_nodal_response(source, reference, **kwargs)
    with pytest.raises(FrozenInstanceError):
        result.pressure_rate = (1, 1)
    assert result.pressure_rate == result.epi_acceleration == (0, 0)


def test_p2_finite_joint_compensation_family_is_an_exact_detached_completion():
    symbolic = pytest.importorskip("sympy")
    t = symbolic.Symbol("t", real=True)
    half, quarter = symbolic.Rational(1, 2), symbolic.Rational(1, 4)
    laplacian = symbolic.Matrix([[1, -1], [-1, 1]])
    epi = symbolic.Matrix([quarter, 0])
    capacity = symbolic.Matrix([1 - t, 3 * half + t])
    theta_over_pi = symbolic.Matrix([t, -t])
    # For |t|<1/4 the singleton means have these regular lifted displacements;
    # capacities are positive and the edge lies strictly inside the U3 gate.
    phase_source = -laplacian * theta_over_pi
    pressure = -half * laplacian * epi + quarter * phase_source - quarter * laplacian * capacity
    assert symbolic.simplify(pressure) == symbolic.zeros(2, 1)
    assert epi.diff(t) == symbolic.diag(*capacity) * pressure
    source = _compensation_source()
    assert tuple(epi) == source.epi
    assert tuple(capacity.subs(t, 0)) == source.capacity
    result = _joint(
        source, phase_rate_over_pi=tuple(map(int, theta_over_pi.diff(t))),
        capacity_rate=tuple(map(int, capacity.diff(t))),
    )
    assert result.pressure_rate == result.epi_acceleration == (0, 0)
    # This is a logical completion of the nodal identity, not a selected law
    # for phase/capacity or evidence of a graph executor following this path.
