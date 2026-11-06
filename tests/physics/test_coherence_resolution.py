"""Exact fixed controls for supplied-width coherence observations.

Scientific fixtures are frozen in structural-observation-design-20261022.
Independent rational pair sums test the readout, not a fitted physical model.
"""

import math
from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction
from itertools import combinations

import networkx as nx
import pytest

from tnfr.physics.coherence_resolution import observe_coherence_resolution

Q = Fraction
RADII = (0, 1, 1, 2, 2, 3, 3)
COHERENCE = tuple(Q(1, 2 ** (radius + 1)) for radius in RADII)
ETA = Q(1, 2**20)


def _star(*, perturbed=False, directed=False):
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(range(7))
    for node, coherence in enumerate(COHERENCE):
        graph.nodes[node]["delta_nfr"] = float(1 / coherence - 1)
        if node:
            radius = Q(RADII[node]) + (ETA * 2 ** (node - 1) if perturbed else 0)
            graph.add_edge(0, node, length=float(radius), weight=1.0)
    return graph


def _exact_point(radius, width, *, perturbed=False, directed=False):
    lengths = tuple(
        Q(value) + (ETA * 2 ** (node - 1) if perturbed and node else 0)
        for node, value in enumerate(RADII)
    )
    pairs = (
        tuple((0, node) for node in range(1, 7))
        if directed
        else tuple(combinations(range(7), 2))
    )
    weighted = tuple(
        (left, right, weight)
        for left, right in pairs
        if (
            weight := max(
                Q(0), 1 - abs(lengths[left] + lengths[right] - radius) / width
            )
        )
        > 0
    )
    mass = sum((weight for _, _, weight in weighted), Q(0))

    def average(function):
        return (
            sum(
                (weight * function(left, right) for left, right, weight in weighted),
                Q(0),
            )
            / mass
        )

    global_mean = sum(COHERENCE) / 7
    product = average(lambda left, right: COHERENCE[left] * COHERENCE[right])
    endpoint = average(lambda left, right: (COHERENCE[left] + COHERENCE[right]) / 2)
    source = average(lambda left, right: COHERENCE[left]) if directed else endpoint
    target = average(lambda left, right: COHERENCE[right]) if directed else endpoint
    second = average(
        lambda left, right: (COHERENCE[left] ** 2 + COHERENCE[right] ** 2) / 2
    )
    return {
        "weight_mass": mass,
        "normalized_mass": mass / len(pairs),
        "contributing_pair_count": len(weighted),
        "product_numerator": mass * product,
        "product_mean": product,
        "endpoint_mean": endpoint,
        "endpoint_second_moment": second,
        "source_mean": source,
        "target_mean": target,
        "source_second_moment": (
            average(lambda left, right: COHERENCE[left] ** 2) if directed else second
        ),
        "target_second_moment": (
            average(lambda left, right: COHERENCE[right] ** 2) if directed else second
        ),
        "centered_product": average(
            lambda left, right: (COHERENCE[left] - global_mean)
            * (COHERENCE[right] - global_mean)
        ),
        "endpoint_covariance": average(
            lambda left, right: (COHERENCE[left] - source) * (COHERENCE[right] - target)
        ),
        "semivariogram": average(
            lambda left, right: (COHERENCE[left] - COHERENCE[right]) ** 2 / 2
        ),
    }


def _assert_oracle(point, oracle):
    assert point.available and point.status == "observed"
    assert point.product_numerator_status == "represented"
    for field, value in oracle.items():
        if field == "contributing_pair_count":
            assert getattr(point, field) == value
        else:
            assert getattr(point, field) == pytest.approx(
                float(value), rel=2e-13, abs=2e-14
            )


@pytest.mark.parametrize("perturbed", (False, True))
def test_exact_pair_weights_preserve_products_and_retain_endpoint_changes(perturbed):
    report = observe_coherence_resolution(
        _star(perturbed=perturbed), radii=range(1, 7), bandwidth=Q(1, 4)
    )
    assert report.profile.positive_pair_count == 21
    for radius, point in enumerate(report.points, 1):
        oracle = _exact_point(Q(radius), Q(1, 4), perturbed=perturbed)
        _assert_oracle(point, oracle)
        assert oracle["product_mean"] == Q(1, 2 ** (radius + 2))
    if perturbed:
        assert len(report.profile.bins) == 21
        assert all(row.pair_count == 1 for row in report.profile.bins)
        assert report.profile.fit.status == "insufficient_bins"
        mean = _exact_point(Q(2), Q(1, 4), perturbed=True)["endpoint_mean"]
        assert mean == (Q(7, 8) - 18 * ETA) / (3 - 60 * ETA)
        assert mean - Q(7, 24) == -ETA / (2 * (3 - 60 * ETA))
    else:
        assert not report.profile.bins[-1].fit_included
        assert report.points[-1].contributing_pair_count == 1
        assert report.points[-1].weight_mass == 1
    # Observation availability is not the old regression's repeated-bin rule.
    assert all(point.available for point in report.points)


def test_width_changes_pair_weighting_and_covariance_keeps_between_bin_term():
    report = observe_coherence_resolution(_star(), radii=[1], bandwidth=Q(3, 2))
    point = report.points[0]
    oracle = _exact_point(Q(1), Q(3, 2))
    assert oracle["weight_mass"] == 3
    assert oracle["product_numerator"] == Q(5, 16)
    assert oracle["product_mean"] == Q(5, 48)
    assert oracle["endpoint_mean"] == Q(25, 72)
    assert oracle["endpoint_covariance"] == Q(5, 48) - Q(25, 72) ** 2
    _assert_oracle(point, oracle)
    covariance_without_between_bins = (
        2 * report.profile.bins[0].endpoint_covariance
        + report.profile.bins[1].endpoint_covariance
    ) / 3
    assert not math.isclose(
        point.endpoint_covariance, covariance_without_between_bins, abs_tol=1e-8
    )
    assert point.semivariogram == pytest.approx(
        point.endpoint_second_moment - point.product_mean, abs=2e-14
    )


@pytest.mark.parametrize("perturbed", (False, True))
def test_metric_unit_change_scales_radii_and_width_without_changing_observation(
    perturbed,
):
    graph = _star(perturbed=perturbed)
    before = observe_coherence_resolution(graph, radii=range(1, 7), bandwidth=Q(1, 4))
    for _, _, data in graph.edges(data=True):
        data["length"] *= 3
    after = observe_coherence_resolution(
        graph, radii=range(3, 19, 3), bandwidth=Q(3, 4)
    )
    for radius, (left, right) in enumerate(
        zip(before.points, after.points, strict=True), 1
    ):
        assert right.radius == 3 * left.radius
        _assert_oracle(right, _exact_point(Q(radius), Q(1, 4), perturbed=perturbed))


def test_directed_source_roles_remain_distinct_and_covariance_is_not_global_centering():
    report = observe_coherence_resolution(
        _star(directed=True), radii=[1], bandwidth=Q(3, 2)
    )
    point = report.points[0]
    _assert_oracle(point, _exact_point(Q(1), Q(3, 2), directed=True))
    assert report.profile.endpoint_convention == "ordered outgoing roles"
    assert point.source_mean == 0.5
    assert point.target_mean != point.source_mean
    assert point.endpoint_covariance == 0
    assert point.centered_product != 0


def test_empty_query_is_unavailable_with_explicit_zero_mass():
    report = observe_coherence_resolution(_star(), radii=[20], bandwidth=Q(1, 4))
    point = report.points[0]
    assert not point.available and point.status == "no_positive_kernel_mass"
    assert point.weight_mass == point.normalized_mass == point.product_numerator == 0
    assert point.product_numerator_status == "empty_support"
    assert point.contributing_pair_count == 0
    assert point.product_mean is None
    assert point.semivariogram is None
    assert point.endpoint_covariance is None
    assert point.centered_product is None


@pytest.mark.parametrize("size", (0, 1))
def test_empty_pair_population_has_no_normalized_support_fraction(size):
    report = observe_coherence_resolution(nx.empty_graph(size), radii=[0], bandwidth=1)
    assert report.profile.positive_pair_count == 0
    point = report.points[0]
    assert not point.available
    assert point.normalized_mass is None
    assert point.weight_mass == 0


def test_zero_distance_and_disconnected_omissions_remain_visible():
    graph = nx.Graph()
    graph.add_nodes_from(range(4))
    graph.add_edge(0, 1, length=0)
    graph.add_edge(1, 2, length=1)
    report = observe_coherence_resolution(graph, radii=[1], bandwidth=Q(1, 4))
    assert report.profile.positive_pair_count == 2
    assert report.profile.zero_distance_pair_count == 1
    assert report.profile.unreachable_pair_count == 3
    assert report.points[0].weight_mass == 2
    assert report.points[0].normalized_mass == report.points[0].product_mean == 1


def test_parallel_paths_reuse_live_metric_owner_and_reports_are_detached():
    graph = _star()
    saved = deepcopy(graph)
    original = observe_coherence_resolution(graph, radii=[1, 2], bandwidth=Q(1, 4))
    assert nx.utils.graphs_equal(graph, saved)
    parallel = nx.MultiGraph(graph)
    for left, right, data in graph.edges(data=True):
        parallel.add_edge(left, right, length=data["length"] + 1, weight=False)
    assert (
        observe_coherence_resolution(parallel, radii=[1, 2], bandwidth=Q(1, 4))
        == original
    )
    with pytest.raises(FrozenInstanceError):
        original.bandwidth = 2
    with pytest.raises(FrozenInstanceError):
        original.points[0].product_mean = 0
    graph.nodes[1]["delta_nfr"] = 0
    assert (
        observe_coherence_resolution(graph, radii=[1, 2], bandwidth=Q(1, 4)).points
        != original.points
    )


@pytest.mark.parametrize(
    "invalid", (0, -1, True, "1", math.inf, math.nan, Q(1, 10**400))
)
def test_invalid_width_is_rejected_before_observing_a_graph(invalid):
    with pytest.raises(ValueError, match="bandwidth"):
        observe_coherence_resolution(None, radii=[1], bandwidth=invalid)


@pytest.mark.parametrize(
    "invalid", ([], [True], ["1"], [-1], [math.inf], [math.nan], [Q(1, 10**400)], [[1]])
)
def test_invalid_queries_are_rejected_before_observing_a_graph(invalid):
    with pytest.raises(ValueError, match="radii"):
        observe_coherence_resolution(None, radii=invalid, bandwidth=1)


def test_finite_extreme_queries_do_not_overflow_distance_over_width():
    report = observe_coherence_resolution(
        _star(),
        radii=[1, float.fromhex("0x1.fffffffffffffp+1023")],
        bandwidth=float.fromhex("0x0.0000000000001p-1022"),
    )
    assert report.points[0].product_mean == Q(1, 8)
    assert report.points[0].weight_mass == 2
    assert not report.points[1].available


def test_tiny_kernel_mass_does_not_erase_a_representable_normalized_product():
    graph = nx.path_graph(2)
    graph.edges[0, 1]["length"] = 1.0
    graph.nodes[0]["delta_nfr"] = 0.0
    graph.nodes[1]["delta_nfr"] = 1e308
    report = observe_coherence_resolution(
        graph, radii=[0], bandwidth=math.nextafter(1.0, 2.0)
    )
    point = report.points[0]
    represented_product = report.profile.bins[0].product_mean
    assert represented_product > 0.0
    assert point.available and point.weight_mass > 0.0
    # One observed pair has normalized weight one, independently of the
    # representability of its raw weighted numerator. No near-zero tolerance
    # may allow this positive normalized result to be replaced by zero.
    assert point.product_mean == represented_product
    exact_numerator = Q(point.weight_mass) * Q(represented_product)
    assert exact_numerator > 0 and float(exact_numerator) == 0.0
    assert point.product_numerator is None
    assert point.product_numerator_status == "positive_underflow"
    assert point.semivariogram == report.profile.bins[0].semivariogram
    assert point.centered_product == report.profile.bins[0].centered_product


@pytest.mark.parametrize("field", ("pressure", "metric"))
def test_live_authoritative_state_is_readmitted(field):
    graph = _star()
    observe_coherence_resolution(graph, radii=[1], bandwidth=1)
    if field == "pressure":
        graph.nodes[0]["delta_nfr"] = True
        graph.nodes[0]["dnfr"] = 0
    else:
        graph.edges[0, 1]["length"] = True
    with pytest.raises((ValueError, TypeError)):
        observe_coherence_resolution(graph, radii=[1], bandwidth=1)
