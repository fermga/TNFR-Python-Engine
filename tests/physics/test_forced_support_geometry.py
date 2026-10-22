"""Independent field-preserving realization controls on held graph inputs.

The exact reference uses materialized edge lengths before path accumulation,
represented phase coefficients and a declared fixed nodal law. Canonical field
readings are separate numerical observations, not a trajectory or proof that
phase, capacity, support or the complete tetrad evolve autonomously.
"""

from copy import deepcopy
from fractions import Fraction as F

import networkx as nx
import pytest

from tnfr.mathematics.krylov import exact_rank
from tnfr.physics.canonical import compute_structural_potential
from tnfr.physics.geometry_realization import observe_forced_support_geometry

BLOCKS = ((0, 2), (1,))


def _apply(matrix, vector):
    return tuple(
        sum((a * b for a, b in zip(row, vector, strict=True)), F(0)) for row in matrix
    )


def _product(left, right):
    return tuple(
        tuple(
            sum((a * b for a, b in zip(row, column, strict=True)), F(0))
            for column in zip(*right, strict=True)
        )
        for row in left
    )


def _sum_vectors(*vectors):
    return tuple(sum(values, F(0)) for values in zip(*vectors, strict=True))


def _negative(matrix):
    return tuple(tuple(-value for value in row) for row in matrix)


def _graph(*, second_length=2, capacity=(1, 1, 1), mixed=False, stored=(7, 7, 7)):
    graph = nx.path_graph(3)
    graph.edges[0, 1].update(weight=1, length=1)
    graph.edges[1, 2].update(weight=1, length=second_length)
    phase = (0.37, 1.19, 0.37) if mixed else (0, 0, 0)
    for node, x, nu, theta, pressure in zip(
        graph, (F(3, 4), F(1, 4), F(1, 2)), capacity, phase, stored, strict=True
    ):
        graph.nodes[node].update(EPI=x, nu_f=nu, theta=theta, delta_nfr=pressure)
    graph.graph["DNFR_WEIGHTS"] = {
        "phase": 1 if mixed else 0,
        "epi": 1,
        "vf": 1 if mixed else 0,
        "topo": 1 if mixed else 0,
    }
    return graph


def _p3_kernel(second_length):
    length = F(second_length)
    far = 1 + length
    short_entry = 1 / length**2 if length else F(0)
    return (
        (F(0), F(1), 1 / far**2),
        (F(1), F(0), short_entry),
        (1 / far**2, short_entry, F(0)),
    )


def _assert_output_reconstruction(result):
    r = result.closure.projection
    x = result.fine_capture.snapshot.epi
    assert result.model_output == _sum_vectors(
        _apply(result.potential_rows, x), result.potential_offset
    )
    assert result.model_output == _apply(r, result.model_potential)
    assert result.realization.output_state == (*_apply(r, x), *result.model_output)
    rates = tuple(
        nu * pressure
        for nu, pressure in zip(
            result.fine_capture.snapshot.capacity, result.model_pressure, strict=True
        )
    )
    # The held affine field offset has zero derivative; capacity enters only
    # through xdot when differentiating its EPI-dependent observation rows.
    assert result.realization.output_rate == (
        *_apply(r, rates),
        *_apply(result.potential_rows, rates),
    )
    assert (
        _sum_vectors(
            result.model_output,
            result.fresh_pressure_potential_defect,
            result.stored_pressure_potential_defect,
            result.field_arithmetic_residual,
        )
        == result.observed_potential
    )


@pytest.mark.parametrize("length,dimension", ((1, 2), (2, 3)))
def test_known_p3_field_obligations_are_carried_by_the_public_realization(
    length, dimension
):
    graph = _graph(second_length=length)
    result = observe_forced_support_geometry(graph, BLOCKS)
    assert result.closure.all_state_affine_closed
    assert result.geometry.nodes == (0, 1, 2)
    assert result.geometry.kernel == _p3_kernel(length)
    assert result.potential_offset == (0, 0)
    assert exact_rank((*result.closure.projection, *result.potential_rows)) == dimension
    assert result.realization.dimension == dimension
    _assert_output_reconstruction(result)


def test_nonclosed_epi_partition_is_completed_by_the_shared_realization():
    result = observe_forced_support_geometry(_graph(), ((0, 1), (2,)))
    assert not result.closure.all_state_affine_closed
    assert result.closure.witness is not None
    assert result.realization.dimension == 3
    assert result.realization.output_rate[:2] == result.closure.projected_nodal_rate
    _assert_output_reconstruction(result)


def test_heterogeneous_capacity_does_not_turn_pressure_observation_into_rate_observation():
    graph = _graph(capacity=(2, 3, 2))
    # The source is declared before capture. It is not reconstructed from
    # stored pressure, which deliberately remains unrelated to this law.
    graph.graph["DNFR_WEIGHTS"] = {"phase": 0, "epi": 1, "vf": 1, "topo": 0}
    result = observe_forced_support_geometry(graph, BLOCKS)
    capture, closure = result.fine_capture, result.closure
    assert dict(capture.normalized_weights) == {
        "phase": 0,
        "epi": F(1, 2),
        "vf": F(1, 2),
        "topo": 0,
    }
    assert capture.forcing == (F(1, 2), F(-1, 2), F(1, 2))
    assert closure.projection == ((F(1, 2), F(0), F(1, 2)), (F(0), F(1), F(0)))
    # Independently: p=(1/2)*(neighbor EPI contrast + capacity contrast).
    assert result.model_pressure == (F(1, 4), F(-5, 16), F(3, 8))
    assert result.model_potential == (F(-13, 48), F(11, 32), F(-29, 576))
    assert result.model_output == (F(-185, 1152), F(11, 32))
    assert result.potential_offset == (F(-37, 144), F(5, 8))
    assert result.potential_rows == (
        (F(37, 288), F(-37, 144), F(37, 288)),
        (F(-1, 2), F(5, 8), F(-1, 8)),
    )
    rate = tuple(
        nu * p
        for nu, p in zip(capture.snapshot.capacity, result.model_pressure, strict=True)
    )
    assert rate == (F(1, 2), F(-15, 16), F(3, 4))
    assert result.model_potential != _apply(result.geometry.kernel, rate)
    assert result.potential_rows != _negative(
        _product(
            _product(closure.projection, result.geometry.kernel),
            closure.micro_generator,
        )
    )
    _assert_output_reconstruction(result)

    # The affine offset and linear rows are properties of the held model,
    # while a fresh second capture supplies a different current EPI state.
    changed = deepcopy(graph)
    for node, value in zip(changed, (F(1, 8), F(3, 4), F(7, 8)), strict=True):
        changed.nodes[node]["EPI"] = value
    second = observe_forced_support_geometry(changed, BLOCKS)
    assert second.potential_offset == result.potential_offset
    assert second.potential_rows == result.potential_rows
    assert second.model_output != result.model_output
    _assert_output_reconstruction(second)


def test_field_residuals_use_fine_pressure_vectors_without_capacity_factors():
    graph = _graph(capacity=(2, 3, 2), mixed=True)
    result = observe_forced_support_geometry(graph, BLOCKS)
    capture = result.fine_capture
    projected_kernel = _product(result.closure.projection, result.geometry.kernel)
    assert any(capture.kernel_pressure_defect)
    assert any(capture.stored_pressure_residual)
    assert result.fresh_pressure_potential_defect == _apply(
        projected_kernel, capture.kernel_pressure_defect
    )
    assert result.stored_pressure_potential_defect == _apply(
        projected_kernel, capture.stored_pressure_residual
    )
    stored_exact = _apply(projected_kernel, capture.snapshot.stored_pressure)
    observed = compute_structural_potential(graph)
    represented_field = tuple(
        F(float(observed[node])) for node in result.geometry.nodes
    )
    assert result.observed_potential == _apply(
        result.closure.projection, represented_field
    )
    assert result.field_arithmetic_residual == tuple(
        a - b for a, b in zip(result.observed_potential, stored_exact, strict=True)
    )
    wrong_weighting = tuple(
        nu * p
        for nu, p in zip(
            capture.snapshot.capacity, capture.stored_pressure_residual, strict=True
        )
    )
    assert result.stored_pressure_potential_defect != _apply(
        projected_kernel, wrong_weighting
    )
    _assert_output_reconstruction(result)


def test_exact_edge_path_sum_is_not_a_rationalized_runtime_distance():
    graph = _graph(second_length=2.0**-54, stored=(0, 0, 1))
    for node in graph:
        graph.nodes[node]["EPI"] = 0
    result = observe_forced_support_geometry(graph, BLOCKS)
    exact_distance = 1 + F(1, 2**54)
    assert result.geometry.distances[0][2] == exact_distance
    assert result.geometry.kernel[0][2] == 1 / exact_distance**2
    assert result.geometry.kernel[0][2] != 1
    assert 1.0 + 2.0**-54 == 1.0
    # The endpoint contribution of stored p_2 is rounded using distance 1
    # by the field reader; the exact represented-edge model retains 1+2^-54.
    assert result.model_output == (0, 0)
    assert result.observed_potential == (F(1, 2), F(2**108))
    assert result.field_arithmetic_residual == (
        F(1, 2) - result.geometry.kernel[0][2] / 2,
        F(0),
    )
    assert result.field_arithmetic_residual[0] > 0
    _assert_output_reconstruction(result)


def test_parallel_minimum_legacy_length_and_node_alignment_are_retained():
    graph = nx.MultiGraph()
    graph.add_nodes_from((2, 0, 1))
    graph.add_edge(0, 1, length=F(3, 2), weight=3)
    graph.add_edge(0, 1, length=F(1, 2), weight=2)
    graph.add_edge(1, 2, weight=2)  # Legacy path length 2; conductance remains 2.
    for node in graph:
        graph.nodes[node].update(EPI=F(node, 8), nu_f=1, theta=0, delta_nfr=7)
    graph.graph["DNFR_WEIGHTS"] = {"epi": 1}
    result = observe_forced_support_geometry(graph, BLOCKS)
    assert result.geometry.nodes == result.fine_capture.snapshot.nodes == (2, 0, 1)
    assert result.geometry.kernel == (
        (F(0), F(4, 25), F(1, 4)),
        (F(4, 25), F(0), F(4)),
        (F(1, 4), F(4), F(0)),
    )
    assert {(i, j): value for i, j, value in result.geometry.edge_lengths} == {
        (0, 2): F(2),
        (2, 0): F(2),
        (1, 2): F(1, 2),
        (2, 1): F(1, 2),
    }
    assert {
        (i, j): value for i, j, value in result.fine_capture.snapshot.conductance
    } == {(0, 2): F(2), (2, 0): F(2), (1, 2): F(5), (2, 1): F(5)}
    assert result.geometry.length_policy
    _assert_output_reconstruction(result)


def test_unit_length_fallback_and_zero_distance_exclusion_match_declared_semantics():
    graph = _graph(second_length=1)
    for left, right in graph.edges:
        graph.edges[left, right].clear()
    unit = observe_forced_support_geometry(graph, BLOCKS)
    assert unit.geometry.kernel == _p3_kernel(1)
    graph.edges[1, 2]["length"] = 0
    zero = observe_forced_support_geometry(graph, BLOCKS)
    assert zero.geometry.distances[1][2] == 0
    assert zero.geometry.kernel == _p3_kernel(0)
    assert (
        zero.fine_capture.snapshot.conductance == unit.fine_capture.snapshot.conductance
    )
    assert zero.closure.micro_generator == unit.closure.micro_generator
    _assert_output_reconstruction(zero)


def test_stored_pressure_changes_only_residuals_and_observer_does_not_mutate_graph():
    graph = _graph(mixed=True, capacity=(2, 3, 2))
    before = deepcopy((dict(graph.nodes(data=True)), dict(graph.edges), graph.graph))
    result = observe_forced_support_geometry(graph, BLOCKS)
    assert (dict(graph.nodes(data=True)), dict(graph.edges), graph.graph) == before
    other = deepcopy(graph)
    other.nodes[0]["delta_nfr"] = F(-3, 8)
    changed = observe_forced_support_geometry(other, BLOCKS)
    assert changed.model_pressure == result.model_pressure
    assert changed.potential_rows == result.potential_rows
    assert changed.potential_offset == result.potential_offset
    assert (
        changed.fresh_pressure_potential_defect
        == result.fresh_pressure_potential_defect
    )
    assert (
        changed.stored_pressure_potential_defect
        != result.stored_pressure_potential_defect
    )
    assert changed.observed_potential != result.observed_potential
    _assert_output_reconstruction(changed)


@pytest.mark.parametrize("length", (True, -1, float("inf"), float("nan")))
def test_invalid_authoritative_length_is_rejected(length):
    with pytest.raises((TypeError, ValueError)):
        observe_forced_support_geometry(_graph(second_length=length), BLOCKS)


def test_exact_realization_budget_exhaustion_returns_no_partial_result():
    graph = _graph()
    before = deepcopy((dict(graph.nodes(data=True)), dict(graph.edges), graph.graph))
    with pytest.raises(ValueError, match="budget"):
        observe_forced_support_geometry(graph, BLOCKS, max_rank_calls=1)
    assert (dict(graph.nodes(data=True)), dict(graph.edges), graph.graph) == before
