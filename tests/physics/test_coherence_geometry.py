"""Exact level-set geometry of the canonical local coherence kernel."""

import math

import networkx as nx
import pytest

from tnfr.constants import DNFR_PRIMARY, dEPI_PRIMARY
from tnfr.metrics.common import compute_coherence, structural_coherence
from tnfr.physics import (
    coherence_level_set_geometry,
    fixed_capacity_coherence_level_set_geometry,
    network_coherence_level_set_geometry,
)


def test_half_coherence_is_the_unit_l1_diamond():
    result = coherence_level_set_geometry(0.5)

    assert result.l1_radius == pytest.approx(1.0)
    assert result.euclidean_radius_minimum == pytest.approx(1.0 / math.sqrt(2.0))
    assert result.euclidean_radius_maximum == pytest.approx(1.0)
    assert result.intrinsic_dimension == 1
    assert len(result.singular_points) == 4
    assert result.smooth_away_from_singular_points
    assert not result.is_globally_smooth_embedded_manifold


@pytest.mark.parametrize("pressure,velocity", [(1.0, 0.0), (0.0, -1.0), (0.4, 0.6)])
def test_every_unit_diamond_probe_has_half_coherence(pressure, velocity):
    assert structural_coherence(pressure, velocity) == pytest.approx(0.5)


def test_full_coherence_collapses_to_the_equilibrium_point():
    result = coherence_level_set_geometry(1.0)

    assert result.is_equilibrium_point
    assert result.intrinsic_dimension == 0
    assert result.singular_points == ((0.0, 0.0),)
    assert result.l1_radius == 0.0
    assert result.is_globally_smooth_embedded_manifold
    assert not result.regular_gradient_available
    assert result.regular_gradient_norm == 0.0


def test_regular_gradient_norm_follows_the_exact_kernel_derivative():
    coherence = 0.25
    result = coherence_level_set_geometry(coherence)
    assert result.regular_gradient_available
    assert result.regular_gradient_norm == pytest.approx(math.sqrt(2.0) * coherence**2)


@pytest.mark.parametrize(
    "value", [0.0, -0.1, 1.1, float("nan"), float("inf"), True]
)
def test_nonfinite_state_levels_are_rejected(value):
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        coherence_level_set_geometry(value)


def test_unrepresentable_finite_level_radius_is_rejected():
    with pytest.raises(ValueError, match="radius exceeds"):
        coherence_level_set_geometry(1e-320)


def test_unrepresentable_nonzero_regular_gradient_is_not_reported_as_zero():
    with pytest.raises(ValueError, match="regular gradient norm"):
        coherence_level_set_geometry(1e-200)


def test_two_node_network_level_is_a_four_dimensional_cross_polytope():
    result = network_coherence_level_set_geometry(0.5, 2)
    strata = result.stratification

    assert result.ambient_dimension == 4
    assert result.mean_l1_radius == pytest.approx(1.0)
    assert result.total_l1_radius == pytest.approx(2.0)
    assert result.euclidean_radius_minimum == pytest.approx(1.0)
    assert result.euclidean_radius_maximum == pytest.approx(2.0)
    assert result.regular_gradient_norm == pytest.approx(0.25)
    assert strata.intrinsic_dimension == 3
    assert strata.vertex_count == 8
    assert strata.singular_face_dimension_range == (0, 2)
    assert [strata.face_count(k) for k in range(4)] == [8, 24, 32, 16]
    assert strata.regular_facet_count() == 16
    assert not strata.level_set_is_globally_smooth_embedded_manifold
    assert result.superlevel_is_closed_convex
    assert not result.temporal_attraction_certified


def test_network_certificate_matches_the_runtime_mean_aggregation():
    graph = nx.Graph()
    graph.add_nodes_from(("a", "b"))
    graph.nodes["a"].update({DNFR_PRIMARY: 1.0, dEPI_PRIMARY: 0.25})
    graph.nodes["b"].update({DNFR_PRIMARY: 0.0, dEPI_PRIMARY: -0.75})

    result = network_coherence_level_set_geometry(
        compute_coherence(graph), graph.number_of_nodes()
    )

    observed_total_radius = sum(
        abs(graph.nodes[node][key])
        for node in graph
        for key in (DNFR_PRIMARY, dEPI_PRIMARY)
    )
    assert compute_coherence(graph) == pytest.approx(0.5)
    assert result.total_l1_radius == pytest.approx(observed_total_radius)


def test_replicating_a_local_observation_preserves_mean_coherence_geometry():
    local = coherence_level_set_geometry(0.4)
    network = network_coherence_level_set_geometry(0.4, 3)

    assert network.mean_l1_radius == pytest.approx(local.l1_radius)
    assert network.total_l1_radius == pytest.approx(3.0 * local.l1_radius)
    assert network.regular_gradient_norm == pytest.approx(
        local.regular_gradient_norm / math.sqrt(3.0)
    )


def test_fixed_capacity_slice_enforces_the_nodal_equation_geometry():
    result = fixed_capacity_coherence_level_set_geometry(0.5, (0.0, 1.0))

    assert result.nodal_equation_enforced
    assert result.ambient_dimension == 4
    assert result.nodal_slice_dimension == 2
    assert result.weighted_pressure_radius == pytest.approx(2.0)
    assert result.pressure_axis_vertex_magnitudes == pytest.approx((2.0, 1.0))
    assert result.embedded_axis_vertex_radii == pytest.approx(
        (2.0, math.sqrt(2.0))
    )
    assert result.euclidean_radius_minimum == pytest.approx(2.0 / math.sqrt(3.0))
    assert result.euclidean_radius_maximum == pytest.approx(2.0)
    assert result.regular_gradient_norm == pytest.approx(math.sqrt(3.0) / 8.0)
    assert result.stratification.intrinsic_dimension == 1
    assert result.stratification.vertex_count == 4
    assert result.stratification.face_count(0) == 4
    assert result.stratification.face_count(1) == 4
    assert not result.temporal_attraction_certified


def test_fixed_capacity_slice_points_reproduce_network_coherence():
    capacities = (0.5, 2.0)
    pressures = (0.4, -0.4666666666666667)
    rates = tuple(
        capacity * pressure
        for capacity, pressure in zip(capacities, pressures, strict=True)
    )
    graph = nx.Graph()
    graph.add_nodes_from(range(2))
    for node, pressure, rate in zip(
        graph, pressures, rates, strict=True
    ):
        graph.nodes[node].update(
            {DNFR_PRIMARY: pressure, dEPI_PRIMARY: rate}
        )

    certificate = fixed_capacity_coherence_level_set_geometry(
        compute_coherence(graph), capacities
    )

    assert rates == pytest.approx((0.2, -0.9333333333333333))
    assert compute_coherence(graph) == pytest.approx(0.5)
    assert certificate.weighted_pressure_radius == pytest.approx(
        sum(
            (1.0 + capacity) * abs(pressure)
            for capacity, pressure in zip(
                capacities, pressures, strict=True
            )
        )
    )


def test_extreme_capacity_uses_bounded_embedding_ratio():
    result = fixed_capacity_coherence_level_set_geometry(
        0.5, (float.fromhex("0x1.fffffffffffffp+1023"),)
    )

    assert result.pressure_axis_vertex_magnitudes[0] > 0.0
    assert result.embedded_axis_vertex_radii == pytest.approx((1.0,))
    assert result.euclidean_radius_minimum == pytest.approx(1.0)
    assert result.euclidean_radius_maximum == pytest.approx(1.0)


def test_one_node_nodal_slice_is_two_smooth_points_away_from_equilibrium():
    result = fixed_capacity_coherence_level_set_geometry(0.5, (1.0,))
    strata = result.stratification

    assert strata.intrinsic_dimension == 0
    assert strata.vertex_count == 2
    assert strata.singular_face_dimension_range is None
    assert strata.level_set_is_globally_smooth_embedded_manifold
    assert strata.kernel_is_differentiable_everywhere_on_level
    assert result.pressure_axis_vertex_magnitudes == pytest.approx((0.5,))
    assert result.euclidean_radius_minimum == pytest.approx(1.0 / math.sqrt(2.0))
    assert result.euclidean_radius_maximum == pytest.approx(1.0 / math.sqrt(2.0))


def test_equilibrium_network_and_nodal_slice_are_degenerate_points():
    ambient = network_coherence_level_set_geometry(1.0, 3)
    nodal = fixed_capacity_coherence_level_set_geometry(1.0, (0.0, 1.0, 2.0))

    for result in (ambient, nodal):
        assert result.stratification.is_degenerate_point
        assert result.stratification.intrinsic_dimension == 0
        assert result.stratification.vertex_count == 1
        assert result.stratification.face_count(0) == 1
        assert result.stratification.regular_facet_count() == 0
        assert not result.regular_gradient_available
        assert result.regular_gradient_norm == 0.0


@pytest.mark.parametrize("node_count", [0, -1, 1.0, True])
def test_network_geometry_rejects_invalid_node_count(node_count):
    with pytest.raises(ValueError, match="positive integer"):
        network_coherence_level_set_geometry(0.5, node_count)


@pytest.mark.parametrize(
    "capacities",
    [(), [], "1.0", (True,), (-0.1,), (float("nan"),), (float("inf"),)],
)
def test_nodal_slice_rejects_invalid_capacity_domains(capacities):
    with pytest.raises(ValueError, match="capacities"):
        fixed_capacity_coherence_level_set_geometry(0.5, capacities)


@pytest.mark.parametrize(
    "capacities",
    [
        {"first": 1.0, "second": 2.0},
        {1.0, 2.0},
        (capacity for capacity in (1.0, 2.0)),
    ],
    ids=("mapping", "set", "generator"),
)
def test_nodal_slice_rejects_nonsequence_capacity_collections(capacities):
    with pytest.raises(ValueError, match="reusable sequence"):
        fixed_capacity_coherence_level_set_geometry(0.5, capacities)


@pytest.mark.parametrize("capacities", [(0.0, 1.0), [0.0, 1.0]])
def test_nodal_slice_accepts_reusable_tuple_and_list(capacities):
    result = fixed_capacity_coherence_level_set_geometry(0.5, capacities)

    assert result.capacities == (0.0, 1.0)
    assert result.node_count == 2


@pytest.mark.parametrize("face_dimension", [-1, 4, 1.0, True])
def test_cross_polytope_face_queries_are_domain_checked(face_dimension):
    strata = network_coherence_level_set_geometry(0.5, 2).stratification
    with pytest.raises(ValueError, match="face_dimension"):
        strata.face_count(face_dimension)
