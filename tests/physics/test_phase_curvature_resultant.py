"""Circular read-out admission and fixed-component arithmetic, without dynamics."""

import math
from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import set_attr
from tnfr.config import get_precision_mode, set_precision_mode
from tnfr.constants.aliases import ALIAS_THETA
from tnfr.physics import canonical
from tnfr.physics._helpers import neighborhood_arrays
from tnfr.physics.fields import (
    PhaseCurvatureObservation,
    UndefinedPhaseCurvatureError,
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_telemetry,
    observe_phase_curvature,
)
from tnfr.physics.vectorized_ops import compute_phase_gradient_and_curvature_vectorized
from tnfr.utils.cache import reset_global_cache


@pytest.fixture(autouse=True)
def isolated_caches_and_mode():
    old_mode = get_precision_mode()
    reset_global_cache()
    yield
    set_precision_mode(old_mode)
    reset_global_cache()


def _star(phases, *, center=0.3, order=None):
    graph = nx.DiGraph()
    graph.add_node("center")
    graph.add_nodes_from(range(len(phases)))
    for neighbor in range(len(phases)) if order is None else order:
        graph.add_edge("center", neighbor, weight=0.0)
    for node, phase in zip(graph, (center, *phases)):
        set_attr(graph.nodes[node], ALIAS_THETA, phase)
    return graph


def _row(graph):
    return observe_phase_curvature(graph).rows[0]


def _manual(phases):
    array = np.asarray(phases, dtype=np.float64)
    components = tuple(zip(map(float, np.cos(array)), map(float, np.sin(array))))
    real = sum((Fraction.from_float(c) for c, _ in components), Fraction())
    imag = sum((Fraction.from_float(s) for _, s in components), Fraction())
    return components, real, imag


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def test_nonzero_small_resultant_uses_direction_without_angle_mean():
    graph = _star((0.0, math.pi + 1e-10))
    observation = observe_phase_curvature(graph)
    row = observation.rows[0]
    _, real, imag = _manual((0.0, math.pi + 1e-10))
    scale = max(abs(real), abs(imag))
    expected_angle = math.atan2(float(imag / scale), float(real / scale))
    assert 0 < scale < Fraction(1, 10**9)
    assert (row.resultant.real_sum, row.resultant.imag_sum) == (real, imag)
    assert row.resultant.angle == expected_angle
    assert row.curvature == _wrap(0.3 - expected_angle)
    old_arithmetic_fallback = _wrap(0.3 - (math.pi + 1e-10) / 2)
    assert abs(_wrap(row.curvature - old_arithmetic_fallback)) > 3.0
    assert compute_phase_curvature(graph)["center"] == row.curvature
    assert compute_structural_telemetry(graph)["curv_phi"]["center"] == row.curvature


def test_actual_represented_cancellation_is_unavailable_but_gradient_remains_defined():
    # The analytic sum at represented pi is not exactly zero: this fixture
    # establishes cancellation only after binary64 sine/cosine materialization.
    graph = _star((0.0, 0.0, math.pi, -math.pi))
    observation = observe_phase_curvature(graph)
    row = observation.rows[0]
    assert row.status == "undefined_represented_resultant"
    assert row.resultant.joint_zero
    assert row.resultant.real_sum == row.resultant.imag_sum == 0
    assert row.resultant.angle is None and row.curvature is None
    assert (
        "represented_zero_does_not_certify_exact_real_trigonometric_zero"
        in observation.scope
    )
    expected_gradient = sum(abs(_wrap(0.3 - p)) for p in (0, 0, math.pi, -math.pi)) / 4
    assert compute_phase_gradient(graph)["center"] == expected_gradient
    for compute in (compute_phase_curvature, compute_structural_telemetry):
        with pytest.raises(UndefinedPhaseCurvatureError, match="represented") as error:
            compute(graph)
        assert error.value.nodes == ("center",)
    assert compute_phase_gradient(graph)["center"] == expected_gradient


def test_isolates_and_empty_graph_have_no_nonempty_resultant():
    empty = observe_phase_curvature(nx.Graph())
    assert empty.nodes == empty.rows == empty.components == ()
    assert compute_phase_curvature(nx.Graph()) == {}
    graph = _star((), center=2.5)
    row = _row(graph)
    assert row.status == "isolated_zero_convention"
    assert row.gradient == row.curvature == 0.0 and row.resultant is None


@pytest.mark.parametrize("order", [(0, 1, 2, 3), (3, 2, 1, 0), (1, 3, 0, 2)])
def test_fixed_materialized_components_are_permutation_invariant(order):
    phases = (0.0, math.pi, math.pi / 2, -math.pi / 2 + 1e-12)
    graph = _star(phases, order=order)
    observation = observe_phase_curvature(graph)
    row = observation.rows[0]
    components, real, imag = _manual(phases)
    assert row.neighbors == order
    assert observation.neighbor_order[0] == order
    assert row.resultant.components == tuple(components[j] for j in order)
    assert (row.resultant.real_sum, row.resultant.imag_sum) == (real, imag)
    expected_angle = math.atan2(
        float(imag / max(abs(real), abs(imag))),
        float(real / max(abs(real), abs(imag))),
    )
    assert row.resultant.angle == expected_angle
    assert row.curvature == _wrap(0.3 - expected_angle)


@pytest.mark.parametrize("rotation", [0.125, 1.5, -2.25])
def test_well_conditioned_rotation_covariance_with_float_tolerance(rotation):
    phases, center = (0.2, 0.4, 0.8), 0.6
    graph = _star(phases, center=center)
    rotated = _star(
        tuple(_wrap(p + rotation) for p in phases), center=_wrap(center + rotation)
    )
    assert _row(rotated).curvature == pytest.approx(_row(graph).curvature, abs=2e-15)
    assert compute_phase_gradient(rotated) == pytest.approx(
        compute_phase_gradient(graph), abs=2e-15
    )


@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
@pytest.mark.parametrize("mode", ["standard", "research"])
def test_graph_array_telemetry_share_materialization_and_domain(
    graph_type, mode, monkeypatch
):
    set_precision_mode(mode)
    graph = graph_type()
    graph.add_nodes_from(["a", "b", "c", "isolated"])
    graph.add_edges_from([("a", "a"), ("a", "b"), ("a", "b"), ("b", "c")])
    for node, phase in zip(graph, (0.1, 2 * math.pi - 0.2, 0.8, 1.5)):
        set_attr(graph.nodes[node], ALIAS_THETA, phase)
    before = deepcopy(
        (dict(graph.nodes(data=True)), dict(graph.graph), list(graph.edges(data=True)))
    )
    observation = observe_phase_curvature(graph)
    assert isinstance(observation, PhaseCurvatureObservation)
    assert observation.requested_precision_mode == mode
    assert "numpy_binary64_trigonometric_components" in observation.scope
    dtype = canonical._get_precision_dtype()
    source, target, counts = neighborhood_arrays(graph, list(graph), dtype=dtype)
    gradient, curvature = compute_phase_gradient_and_curvature_vectorized(
        np.asarray(observation.primitive_phases, dtype=np.longdouble),
        source,
        target,
        counts,
        dtype=dtype,
    )
    assert gradient.tolist() == [row.gradient for row in observation.rows]
    assert curvature.tolist() == [row.curvature for row in observation.rows]
    monkeypatch.setattr(canonical, "_VECTORIZATION_AVAILABLE", False)
    assert compute_phase_curvature(graph) == dict(zip(graph, curvature.tolist()))
    assert all(abs(v) <= math.pi for v in curvature)
    telemetry = compute_structural_telemetry(graph)
    assert telemetry["curv_phi"] == compute_phase_curvature(graph)
    assert telemetry["grad_phi"] == compute_phase_gradient(graph)
    # Spectral telemetry may add a graph-owned cache; the curvature observation
    # itself is detached and never writes channels or topology.
    assert dict(graph.nodes(data=True)) == before[0]
    assert list(graph.edges(data=True)) == before[2]
    with pytest.raises(FrozenInstanceError):
        observation.rows = ()


def test_array_path_rejects_represented_cancellation():
    graph = _star((0.0, 0.0, math.pi, -math.pi))
    source, target, counts = neighborhood_arrays(graph, list(graph))
    phases = np.asarray(observe_phase_curvature(graph).primitive_phases)
    with pytest.raises(UndefinedPhaseCurvatureError):
        compute_phase_gradient_and_curvature_vectorized(phases, source, target, counts)


@pytest.mark.parametrize(
    "bad", [True, "0", 1j, float("nan"), float("inf"), Fraction(1, 2**2000)]
)
@pytest.mark.parametrize(
    "compute",
    [
        observe_phase_curvature,
        compute_phase_gradient,
        compute_phase_curvature,
        compute_structural_telemetry,
    ],
)
def test_invalid_authoritative_phase_cannot_reuse_valid_cached_result(bad, compute):
    graph = _star((0.1, 0.2))
    compute(graph)
    graph.nodes["center"][ALIAS_THETA[0]] = bad
    graph.nodes["center"][ALIAS_THETA[1]] = 0.3
    with pytest.raises((TypeError, ValueError)):
        compute(graph)


def test_observation_cache_retains_actual_neighbor_order_and_precision():
    graph = _star((0.1, 0.2, 0.3))
    first = observe_phase_curvature(graph)
    first_map = compute_phase_curvature(graph)
    assert observe_phase_curvature(graph) is first
    assert compute_phase_curvature(graph) == first_map
    graph.remove_edge("center", 0)
    graph.add_edge("center", 0, weight=0.0)
    reordered = observe_phase_curvature(graph)
    assert reordered is not first
    assert reordered.rows[0].neighbors == (1, 2, 0)
    assert reordered.rows[0].resultant.real_sum == first.rows[0].resultant.real_sum
    set_precision_mode("research")
    assert observe_phase_curvature(graph) is not reordered
    assert observe_phase_curvature(graph).requested_precision_mode == "research"


@pytest.mark.parametrize(
    "compute,key",
    [
        (compute_phase_gradient, None),
        (compute_phase_curvature, None),
        (compute_structural_telemetry, "curv_phi"),
        (compute_structural_telemetry, "grad_phi"),
    ],
)
def test_caller_mutation_cannot_poison_cached_phase_readout(compute, key):
    graph = _star((0.1, 0.2))
    result = compute(graph)
    target = result if key is None else result[key]
    original = dict(target)
    target["center"] = None
    target["unrelated"] = 10.0
    fresh = compute(graph)
    assert (fresh if key is None else fresh[key]) == original


@pytest.mark.parametrize(
    "source,target,counts",
    [
        ([1, 1], [0, 0], [2, 0]),  # repeated neighbor is not a multigraph degree
        ([1], [0], [2, 0]),  # denominator cannot disagree with incidence
        ([2], [0], [1, 0]),
        ([True], [0], [1, 0]),
    ],
)
def test_array_adapter_validates_neighborhood_contract(source, target, counts):
    with pytest.raises(ValueError):
        compute_phase_gradient_and_curvature_vectorized(
            [0.0, 0.2], source, target, counts
        )


def test_nonrepresentable_phase_difference_is_explicitly_rejected():
    graph = _star(
        (-float.fromhex("0x1.fffffffffffffp+1023"),),
        center=float.fromhex("0x1.fffffffffffffp+1023"),
    )
    with pytest.raises(ValueError, match="differences must be finite"):
        observe_phase_curvature(graph)
