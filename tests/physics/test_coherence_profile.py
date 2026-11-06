"""Fixed analytical controls for the shared static coherence-profile observer.

Scientific fixtures are frozen in the xi-observation-research-20261022
protocol. They specify stored fields, not trajectories or inferred laws.
"""

import math
from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.config import get_precision_mode, set_precision_mode
from tnfr.physics import canonical
from tnfr.physics._coherence_fit import fit_coherence_length
from tnfr.physics.fields import observe_coherence_profile
from tnfr.physics.vectorized_ops import compute_coherence_length_vectorized


def _directed_star(*, oscillatory=False, amplitude=1.0):
    graph = nx.DiGraph()
    graph.add_node(0, delta_nfr=1.0 / amplitude - 1.0)
    for radius in range(1, 5):
        value = (0.5 if radius % 2 else 1 / 16) if oscillatory else 2.0**-radius
        for _ in range(3):
            node = len(graph)
            graph.add_node(node, delta_nfr=1.0 / (amplitude * value) - 1.0)
            graph.add_edge(0, node, length=float(radius), weight=1.0)
    return graph


def _radial_star(*, fragile=False):
    graph = nx.Graph()
    graph.add_node(0, delta_nfr=1.0)
    for index, base_radius in enumerate((1, 1, 2, 2, 3, 3)):
        radius = base_radius + (2.0 ** (index - 20) if fragile else 0.0)
        graph.add_node(index + 1, delta_nfr=2.0 ** (radius + 1) - 1.0)
        graph.add_edge(0, index + 1, length=radius, weight=1.0)
    return graph


def _pressure(graph):
    return {node: graph.nodes[node].get("delta_nfr", 0.0) for node in graph}


@pytest.mark.parametrize("vectorized", (False, True))
def test_exact_exponential_fit_and_endpoint_composition(vectorized, monkeypatch):
    graph = _directed_star()
    monkeypatch.setattr(canonical, "_VECTORIZATION_AVAILABLE", vectorized)
    profile = observe_coherence_profile(graph)
    fit = profile.fit
    assert fit.available and fit.status == "fitted"
    assert fit.value == pytest.approx(1 / math.log(2), rel=2e-14)
    assert fit.value == canonical.estimate_coherence_length(graph)
    assert fit.value == compute_coherence_length_vectorized(
        graph, list(graph), _pressure(graph)
    )
    assert fit.slope == pytest.approx(-math.log(2), rel=2e-14)
    assert fit.intercept == pytest.approx(0.0, abs=2e-14)
    assert fit.r_squared == pytest.approx(1.0, abs=2e-14)
    assert fit.residual_sum_squares < 1e-26
    assert fit.residual_rms < 1e-13
    assert fit.design_rank == 2
    assert len(fit.singular_values) == 2
    assert fit.rcond > 0
    assert fit.distance_scale == 4
    assert fit.selected_distances == (1, 2, 3, 4)
    assert profile.global_mean == pytest.approx(float(Fraction(61, 208)))
    assert profile.candidate_pair_count == 156
    assert profile.positive_pair_count == 12
    assert profile.unreachable_pair_count == 144
    assert profile.zero_distance_pair_count == 0
    assert profile.endpoint_convention == "ordered outgoing roles"
    for row in profile.bins:
        target = 2.0**-row.distance
        assert row.pair_count == 3
        assert row.source_mean == row.source_second_moment == 1
        assert row.target_mean == row.product_mean == target
        assert row.target_second_moment == target**2
        assert row.endpoint_covariance == 0
        assert row.semivariogram == 0.5 * (1 - target) ** 2
        assert row.fit_included and row.fit_exclusion_reason is None


def test_bad_exponential_shape_is_disclosed_without_changing_scalar_policy():
    graph = _directed_star(oscillatory=True)
    profile = observe_coherence_profile(graph)
    assert profile.fit.value == pytest.approx(5 / (3 * math.log(2)), rel=2e-14)
    assert profile.fit.value == canonical.estimate_coherence_length(graph)
    assert profile.fit.slope == pytest.approx(-3 * math.log(2) / 5, rel=2e-14)
    assert profile.fit.intercept == pytest.approx(-math.log(2), rel=2e-14)
    assert profile.fit.r_squared == pytest.approx(1 / 5, abs=2e-14)
    assert profile.fit.residual_sum_squares == pytest.approx(36 * math.log(2) ** 2 / 5)
    assert tuple(row.product_mean for row in profile.bins) == (0.5, 1 / 16, 0.5, 1 / 16)


@pytest.mark.parametrize("mode", ("standard", "high", "research"))
def test_precision_mode_keeps_fixed_exact_control_and_records_its_origin(mode):
    previous = get_precision_mode()
    try:
        set_precision_mode(mode)
        graph = _radial_star()
        profile = observe_coherence_profile(graph)
        assert profile.fit.value == pytest.approx(1 / math.log(2), rel=2e-14)
        assert profile.fit.value == canonical.estimate_coherence_length(graph)
        assert profile.precision_mode == mode
        assert "binary64" in profile.arithmetic_dtype
        assert "no goodness-of-fit acceptance test" in profile.fit_policy
    finally:
        set_precision_mode(previous)


def test_same_fit_length_discards_amplitude_retained_by_profile():
    first = observe_coherence_profile(_directed_star())
    half = observe_coherence_profile(_directed_star(amplitude=0.5))
    assert half.fit.value == pytest.approx(first.fit.value, rel=2e-14)
    assert half.fit.r_squared == pytest.approx(first.fit.r_squared)
    assert half.fit.intercept - first.fit.intercept == pytest.approx(-2 * math.log(2))
    assert half.global_mean == first.global_mean / 2
    assert half.global_variance == first.global_variance / 4
    for left, right in zip(first.bins, half.bins):
        assert right.product_mean == left.product_mean / 4
        assert right.endpoint_mean == left.endpoint_mean / 2
        assert right.endpoint_second_moment == left.endpoint_second_moment / 4
        assert right.centered_product == left.centered_product / 4
        assert right.semivariogram == left.semivariogram / 4


def test_radial_star_retains_all_bins_and_exact_moment_identities():
    profile = observe_coherence_profile(_radial_star())
    assert profile.global_mean == pytest.approx(float(Fraction(11, 56)))
    assert profile.positive_pair_count == profile.candidate_pair_count == 21
    assert tuple(row.pair_count for row in profile.bins) == (2, 3, 6, 5, 4, 1)
    assert profile.fit.selected_distances == (1, 2, 3, 4, 5)
    assert profile.fit.value == pytest.approx(1 / math.log(2), rel=2e-14)
    assert profile.fit.intercept == pytest.approx(-2 * math.log(2), rel=2e-14)
    assert profile.bins[-1].fit_exclusion_reason == "insufficient_bin_pairs"
    assert not profile.bins[-1].fit_included
    for row in profile.bins:
        assert row.product_mean == 2.0 ** (-row.distance - 2)
        assert row.source_mean == row.target_mean == row.endpoint_mean
        assert (
            row.source_second_moment
            == row.target_second_moment
            == row.endpoint_second_moment
        )
        assert row.centered_product == pytest.approx(
            row.product_mean
            - 2 * profile.global_mean * row.endpoint_mean
            + profile.global_mean**2,
            abs=1e-16,
        )
        assert row.endpoint_covariance == pytest.approx(
            row.product_mean - row.endpoint_mean**2, abs=1e-16
        )
        assert row.semivariogram == pytest.approx(
            row.endpoint_second_moment - row.product_mean, abs=1e-16
        )
    assert any(row.centered_product < 0 for row in profile.bins)


@pytest.mark.parametrize("factory", (_radial_star, _directed_star))
def test_distance_unit_changes_fit_length_only(factory):
    graph = factory()
    before = observe_coherence_profile(graph)
    for _, _, edge in graph.edges(data=True):
        edge["length"] *= 3
    after = observe_coherence_profile(graph)
    assert after.fit.value == pytest.approx(3 * before.fit.value, rel=2e-14)
    assert after.fit.slope == pytest.approx(before.fit.slope / 3, rel=2e-14)
    assert after.fit.intercept == before.fit.intercept
    assert after.fit.r_squared == before.fit.r_squared
    assert after.distance_weighting == before.distance_weighting
    for left, right in zip(before.bins, after.bins):
        assert right.distance == 3 * left.distance
        assert right.product_mean == left.product_mean
        assert right.centered_product == left.centered_product
        assert right.semivariogram == left.semivariogram


def test_tiny_metric_perturbation_removes_every_repeated_bin():
    profile = observe_coherence_profile(_radial_star(fragile=True))
    assert profile.positive_pair_count == 21
    assert len(profile.bins) == 21
    assert all(row.pair_count == 1 for row in profile.bins)
    assert all(
        row.fit_exclusion_reason == "insufficient_bin_pairs" for row in profile.bins
    )
    assert profile.fit.value is None and not profile.fit.available
    assert profile.fit.status == "insufficient_bins"
    assert profile.fit.selected_distances == ()
    for row in profile.bins:
        assert row.product_mean == pytest.approx(2.0 ** (-row.distance - 2), rel=2e-14)


def test_constant_profile_keeps_zero_structure_without_spectral_substitution():
    graph = nx.path_graph(8)
    nx.set_node_attributes(graph, 1.0, "delta_nfr")
    profile = observe_coherence_profile(graph)
    assert profile.global_mean == 0.5 and profile.global_variance == 0
    assert profile.fit.status == "constant_products"
    assert profile.fit.value is None
    assert (
        canonical.estimate_coherence_length_with_provenance(graph).method
        == "spectral_gap"
    )
    for row in profile.bins:
        assert row.product_mean == 0.25
        assert row.centered_product == row.semivariogram == row.endpoint_covariance == 0


def test_pressure_sign_change_is_invisible_to_entire_coherence_profile():
    graph = _radial_star()
    before = observe_coherence_profile(graph)
    graph.nodes[1]["delta_nfr"] *= -1
    assert observe_coherence_profile(graph) == before


def test_profiles_are_detached_immutable_and_refresh_from_live_inputs():
    graph = _radial_star()
    saved = deepcopy(graph)
    profile = observe_coherence_profile(graph)
    assert nx.utils.graphs_equal(graph, saved)
    with pytest.raises(FrozenInstanceError):
        profile.global_mean = 0
    with pytest.raises(FrozenInstanceError):
        profile.bins[0].product_mean = 0
    with pytest.raises(FrozenInstanceError):
        profile.fit.value = 0
    graph.nodes[1]["delta_nfr"] = 0
    assert observe_coherence_profile(graph).bins != profile.bins


def test_undirected_endpoint_moments_ignore_traversal_orientation():
    graph = _radial_star()
    reordered = nx.Graph()
    reordered.add_nodes_from(reversed(tuple(graph.nodes(data=True))))
    reordered.add_edges_from(reversed(tuple(graph.edges(data=True))))
    first, reverse = observe_coherence_profile(graph), observe_coherence_profile(
        reordered
    )
    assert reverse.nodes == tuple(reversed(first.nodes))
    assert reverse.bins == first.bins
    assert reverse.fit == first.fit


@pytest.mark.parametrize("size", (0, 1))
def test_empty_pair_domain_is_explicit(size):
    profile = observe_coherence_profile(nx.empty_graph(size))
    assert profile.bins == ()
    assert profile.candidate_pair_count == profile.positive_pair_count == 0
    assert profile.fit.status == "insufficient_pairs"
    assert profile.global_mean == (1.0 if size else None)
    assert profile.global_variance == (0.0 if size else None)


def test_zero_lengths_and_disconnection_partition_the_sample():
    graph = nx.Graph()
    graph.add_nodes_from(range(4))
    graph.add_edge(0, 1, length=0.0)
    graph.add_edge(1, 2, length=1.0)
    profile = observe_coherence_profile(graph)
    assert profile.candidate_pair_count == 6
    assert profile.positive_pair_count == 2
    assert profile.zero_distance_pair_count == 1
    assert profile.unreachable_pair_count == 3


@pytest.mark.parametrize("directed", (False, True))
def test_partial_source_counts_match_streamed_and_declared_matrix_paths(directed):
    graph = _directed_star() if directed else _radial_star()
    nodes = tuple(graph)
    sources = (nodes[-1], nodes[0])
    matrix = nx.floyd_warshall_numpy(graph, nodelist=list(nodes), weight="length")
    options = ({}, {"materialize": True}, {"distance_matrix": matrix})
    profiles = [
        fit_coherence_length(
            graph,
            nodes,
            _pressure(graph),
            sources=sources,
            return_profile=True,
            **option,
        )
        for option in options
    ]
    expected = len(sources) * (len(nodes) - 1) if directed else 2 * (len(nodes) - 2) + 1
    for profile in profiles:
        assert profile.sources == sources
        assert profile.candidate_pair_count == expected
        assert (
            profile.candidate_pair_count
            == profile.positive_pair_count
            + profile.zero_distance_pair_count
            + profile.unreachable_pair_count
        )
        assert profile.bins == profiles[0].bins
        assert profile.fit == profiles[0].fit
    assert profiles[-1].distance_origin == "caller-declared matrix"


def test_parallel_minimum_and_unused_conductance_preserve_profile():
    simple = _radial_star()
    multi = nx.MultiGraph(simple)
    for left, right, data in simple.edges(data=True):
        multi.add_edge(left, right, length=data["length"] + 10, weight=False)
    assert observe_coherence_profile(multi) == observe_coherence_profile(simple)


@pytest.mark.parametrize(
    "invalid", (True, "1", math.nan, math.inf, Fraction(1, 10**400))
)
def test_invalid_authoritative_pressure_cannot_hide_behind_other_alias(invalid):
    graph = _radial_star()
    graph.nodes[0]["delta_nfr"] = invalid
    graph.nodes[0]["dnfr"] = 0.0
    with pytest.raises((ValueError, TypeError), match="pressure"):
        observe_coherence_profile(graph)


@pytest.mark.parametrize("invalid", (True, -1, math.inf))
def test_invalid_geometry_is_not_a_reported_fit_failure(invalid):
    graph = _radial_star()
    graph.edges[0, 1]["length"] = invalid
    with pytest.raises(ValueError, match="edge length"):
        observe_coherence_profile(graph)


@pytest.mark.parametrize("dtype", (int, bool, complex, object))
def test_dtype_cannot_silently_reinterpret_admitted_pressure(dtype):
    graph = _radial_star()
    with pytest.raises(ValueError, match="floating dtype"):
        fit_coherence_length(
            graph, tuple(graph), _pressure(graph), sources=tuple(graph), dtype=dtype
        )


@pytest.mark.parametrize("value", (1e-100, 1e100))
def test_narrow_dtype_cannot_erase_or_overflow_a_finite_pressure(value):
    graph = nx.path_graph(2)
    with pytest.raises(ValueError, match="unrepresentable"):
        fit_coherence_length(
            graph, tuple(graph), {0: value}, sources=tuple(graph), dtype=np.float32
        )


def test_extra_observations_are_optional_for_existing_scalar_calls(monkeypatch):
    from tnfr.physics import _coherence_fit

    def forbid_diagnostics(*args, **kwargs):
        raise AssertionError("scalar estimator must not compute endpoint diagnostics")

    monkeypatch.setattr(_coherence_fit, "_observe_bin", forbid_diagnostics)
    graph = _directed_star()
    result = compute_coherence_length_vectorized(graph, list(graph), _pressure(graph))
    assert result == pytest.approx(1 / math.log(2), rel=2e-14)
