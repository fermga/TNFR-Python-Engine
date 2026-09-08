"""Canonical coherence and cache contracts for sparse TNFR graphs."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tnfr.errors import TNFRValueError
from tnfr.metrics.common import finite_mean_absolute, structural_coherence
from tnfr.sparse import CompactAttributeStore, SparseCache, SparseTNFRGraph


def _phase_stressed_pair(*, frequencies: tuple[float, float] = (2.0, 3.0)):
    graph = SparseTNFRGraph(2)
    graph.add_edge(0, 1)
    graph.node_attributes.set_theta(1, np.pi / 2.0)
    for node, frequency in enumerate(frequencies):
        graph.node_attributes.set_vf(node, frequency)
    return graph


def test_evolve_sparse_uses_last_step_nodal_rate_in_canonical_coherence():
    graph = _phase_stressed_pair()
    nodes = [0, 1]
    pressure = graph.compute_dnfr_sparse(nodes)
    rate = graph.node_attributes.get_vfs(nodes) * pressure
    mean_pressure = float(np.mean(np.abs(pressure), dtype=np.float64))
    mean_rate = float(np.mean(np.abs(rate), dtype=np.float64))

    result = graph.evolve_sparse(dt=0.25, steps=1)

    assert result["final_coherence"] == pytest.approx(
        structural_coherence(mean_pressure, mean_rate)
    )
    assert result["final_coherence"] != pytest.approx(
        structural_coherence(mean_pressure)
    )
    assert result["mean_abs_dnfr"] == pytest.approx(mean_pressure)
    assert result["mean_abs_depi"] == pytest.approx(mean_rate)
    assert result["coherence_depi_source"] == "last_step_nodal_rate"
    assert type(result["final_coherence"]) is float


def test_zero_step_sparse_coherence_is_explicitly_static_and_forgets_prior_rate():
    graph = _phase_stressed_pair()
    dynamic = graph.evolve_sparse(dt=0.1, steps=1)
    epi_before = graph.node_attributes.get_epis([0, 1]).copy()
    pressure = graph.compute_dnfr_sparse([0, 1])
    mean_pressure = float(np.mean(np.abs(pressure), dtype=np.float64))

    static = graph.evolve_sparse(dt=0.0, steps=0)

    np.testing.assert_array_equal(graph.node_attributes.get_epis([0, 1]), epi_before)
    assert static["final_coherence"] == pytest.approx(
        structural_coherence(mean_pressure, 0.0)
    )
    assert static["mean_abs_dnfr"] == pytest.approx(mean_pressure)
    assert static["mean_abs_depi"] == 0.0
    assert static["coherence_depi_source"] == "static_default_zero"
    assert static["final_coherence"] > dynamic["final_coherence"]


def test_sparse_coherence_aggregates_channel_magnitudes_through_shared_kernel():
    graph = SparseTNFRGraph(2)

    result = graph._compute_coherence(
        dnfr_values=[-2.0, 4.0],
        depi_values=[-1.0, 3.0],
        return_means=True,
    )

    assert result == (structural_coherence(3.0, 2.0), 3.0, 2.0)
    scalar = graph._compute_coherence(
        dnfr_values=[-2.0, 4.0], depi_values=[-1.0, 3.0]
    )
    assert scalar == structural_coherence(3.0, 2.0)
    assert type(scalar) is float


def test_shared_mean_magnitude_is_finite_safe_and_validates_scalars():
    largest = sys.float_info.max

    assert finite_mean_absolute([largest, largest], name="test") == largest
    with pytest.raises(TypeError, match="not bool"):
        finite_mean_absolute([True], name="test")
    with pytest.raises(ValueError, match="finite"):
        finite_mean_absolute([np.inf], name="test")


@pytest.mark.parametrize(
    ("channel", "values", "message"),
    [
        ("dnfr_values", [0.0], "one value per node"),
        ("depi_values", [[0.0, 0.0]], "one value per node"),
        ("dnfr_values", [0.0, np.nan], "only finite"),
        ("depi_values", [True, False], "finite real"),
    ],
)
def test_sparse_coherence_rejects_invalid_channels(channel, values, message):
    graph = SparseTNFRGraph(2)
    arguments = {
        "dnfr_values": [0.0, 0.0],
        "depi_values": [0.0, 0.0],
    }
    arguments[channel] = values

    with pytest.raises(TNFRValueError, match=message):
        graph._compute_coherence(**arguments)


@pytest.mark.parametrize("steps", [True, -1, 1.5])
def test_evolve_sparse_rejects_invalid_step_counts(steps):
    with pytest.raises(TNFRValueError, match="nonnegative integer"):
        SparseTNFRGraph(1).evolve_sparse(steps=steps)


@pytest.mark.parametrize("dt", [True, np.nan, np.inf, 0.0, -0.1])
def test_evolve_sparse_rejects_invalid_dynamic_timesteps(dt):
    with pytest.raises(TNFRValueError, match="dt must"):
        SparseTNFRGraph(1).evolve_sparse(dt=dt, steps=1)


def test_sparse_pressure_cache_is_invalidated_by_owned_topology_and_phase_changes():
    graph = SparseTNFRGraph(2)
    graph.node_attributes.set_theta(1, np.pi / 2.0)
    np.testing.assert_array_equal(graph.compute_dnfr_sparse(), [0.0, 0.0])

    graph.add_edge(0, 1)
    assert np.all(np.abs(graph.compute_dnfr_sparse()) > 0.99)

    graph.node_attributes.set_theta(1, 0.0)
    np.testing.assert_array_equal(graph.compute_dnfr_sparse(), [0.0, 0.0])

@pytest.mark.parametrize("node_count", [True, np.bool_(False), 1.0, 0, -1])
def test_sparse_graph_rejects_invalid_node_counts(node_count):
    with pytest.raises(TNFRValueError, match="node_count must"):
        SparseTNFRGraph(node_count)


def test_sparse_graph_normalizes_numpy_integer_domain_and_seed():
    graph = SparseTNFRGraph(
        np.int64(3),
        expected_density=np.float32(0.0),
        seed=np.int64(7),
    )

    assert graph.node_count == 3
    assert type(graph.node_count) is int
    assert graph.expected_density == 0.0
    assert graph.seed == 7
    assert type(graph.seed) is int


@pytest.mark.parametrize(
    "density",
    [True, np.bool_(False), np.nan, np.inf, -0.1, 1.1, "0.5"],
)
def test_sparse_graph_rejects_invalid_edge_probabilities(density):
    with pytest.raises(TNFRValueError, match="expected_density must"):
        SparseTNFRGraph(2, expected_density=density)


@pytest.mark.parametrize("seed", [True, 1.5, -1, 2**32])
def test_sparse_graph_rejects_invalid_randomstate_seeds(seed):
    with pytest.raises(TNFRValueError, match="seed must"):
        SparseTNFRGraph(2, seed=seed)


@pytest.mark.parametrize(
    ("setter", "getter", "initial"),
    [
        ("set_vf", "get_vf", 2.0),
        ("set_theta", "get_theta", 1.0),
        ("set_si", "get_si", 0.5),
        ("set_epi", "get_epi", -2.0),
        ("set_dnfr", "get_dnfr", -3.0),
    ],
)
@pytest.mark.parametrize("invalid", [True, np.nan, np.inf, -np.inf])
def test_compact_setters_reject_invalid_scalars_without_erasing_state(
    setter,
    getter,
    initial,
    invalid,
):
    store = CompactAttributeStore(1)
    getattr(store, setter)(0, initial)

    with pytest.raises(TNFRValueError):
        getattr(store, setter)(0, invalid)

    assert getattr(store, getter)(0) == pytest.approx(initial)


def test_compact_store_preserves_representable_near_default_values():
    store = CompactAttributeStore(1)

    store.set_epi(0, 1e-11)

    assert store.get_epi(0) == pytest.approx(float(np.float32(1e-11)))
    assert store.get_epi(0) != 0.0


@pytest.mark.parametrize("value", [1e39, 1e-50])
def test_compact_store_rejects_values_outside_float32_storage_domain(value):
    store = CompactAttributeStore(1)

    with pytest.raises(TNFRValueError, match="representable"):
        store.set_epi(0, value)


@pytest.mark.parametrize(
    ("setter", "value", "message"),
    [("set_vf", -0.1, "vf must be nonnegative"),
     ("set_si", -0.1, "si must be nonnegative")],
)
def test_compact_store_rejects_negative_nonnegative_channels(
    setter,
    value,
    message,
):
    store = CompactAttributeStore(1)

    with pytest.raises(TNFRValueError, match=message):
        getattr(store, setter)(0, value)


def test_compact_phase_is_canonicalized_and_notifies_only_on_stored_change():
    notifications = []
    store = CompactAttributeStore(
        1,
        on_theta_change=lambda: notifications.append("changed"),
    )

    store.set_theta(0, 2.0 * np.pi)
    assert store.get_theta(0) == 0.0
    assert notifications == []

    store.set_theta(0, -np.pi / 2.0)
    assert store.get_theta(0) == pytest.approx(3.0 * np.pi / 2.0)
    assert notifications == ["changed"]


@pytest.mark.parametrize("node_id", [True, -1, 2, 0.5])
def test_compact_store_rejects_nodes_outside_fixed_integer_domain(node_id):
    store = CompactAttributeStore(2)

    with pytest.raises(TNFRValueError, match="node_id must"):
        store.set_epi(node_id, 1.0)
    with pytest.raises(TNFRValueError, match="node_id must"):
        store.get_epi(node_id)


@pytest.mark.parametrize(
    "weight",
    [True, np.nan, np.inf, -1.0, 0.0, 1e39, 1e-50],
)
def test_add_edge_rejects_invalid_weights_without_mutation(weight):
    graph = SparseTNFRGraph(2)
    graph.compute_dnfr_sparse()
    cached_before = dict(graph._dnfr_cache._cache)

    with pytest.raises(TNFRValueError, match="weight must"):
        graph.add_edge(0, 1, weight=weight)

    assert graph.number_of_edges() == 0
    assert graph.adjacency.nnz == 0
    assert graph._dnfr_cache._cache == cached_before


@pytest.mark.parametrize("endpoint", [True, -1, 2, 0.5])
def test_add_edge_rejects_invalid_endpoints_before_mutation(endpoint):
    graph = SparseTNFRGraph(2)

    with pytest.raises(TNFRValueError, match="node_id must"):
        graph.add_edge(endpoint, 1)

    assert graph.number_of_edges() == 0


def test_self_loops_and_ordinary_edges_have_consistent_undirected_counts():
    graph = SparseTNFRGraph(2)

    graph.add_edge(0, 0, weight=0.5)
    assert graph.number_of_edges() == 1
    graph.add_edge(0, 1)
    assert graph.number_of_edges() == 2
    graph.add_edge(1, 0, weight=0.75)
    assert graph.number_of_edges() == 2


@pytest.mark.parametrize("node_ids", [[0, True], [0, -1], [0, 2], [0, 0.5]])
def test_sparse_pressure_validates_complete_node_request_before_cache_access(
    node_ids,
):
    graph = SparseTNFRGraph(2)
    graph.compute_dnfr_sparse([0])
    cached_before = dict(graph._dnfr_cache._cache)

    with pytest.raises(TNFRValueError, match="node_id must"):
        graph.compute_dnfr_sparse(node_ids)

    assert graph._dnfr_cache._cache == cached_before


def test_sparse_pressure_rejects_invalid_direct_adjacency_data():
    graph = SparseTNFRGraph(2)
    graph.adjacency[0, 1] = np.nan

    with pytest.raises(TNFRValueError, match="adjacency weights"):
        graph.compute_dnfr_sparse()


def test_sparse_cache_is_capacity_bounded_and_uses_write_age():
    cache = SparseCache(capacity=2, ttl_steps=2)

    cache.update({0: 0.0, 1: 1.0, 2: 2.0})
    assert list(cache._cache) == [1, 2]
    cache.step()
    assert cache.get(1) == 1.0
    cache.step()
    assert cache.get(1) is None
    assert cache.get(2) is None


def test_sparse_cache_batch_validation_is_atomic():
    cache = SparseCache(capacity=2)
    cache.update({0: 1.0})
    state_before = dict(cache._cache)

    with pytest.raises(TNFRValueError, match="cache value"):
        cache.update({1: 2.0, 2: np.nan})

    assert cache._cache == state_before


@pytest.mark.parametrize(
    ("capacity", "ttl_steps"),
    [(True, 1), (-1, 1), (1, True), (1, 0)],
)
def test_sparse_cache_rejects_invalid_configuration(capacity, ttl_steps):
    with pytest.raises(TNFRValueError):
        SparseCache(capacity=capacity, ttl_steps=ttl_steps)


def test_sparse_nodal_step_rejects_unrepresentable_epi_before_node_writes():
    graph = SparseTNFRGraph(2)
    largest = float(np.finfo(np.float32).max)
    graph.add_edge(0, 1, weight=largest)
    graph.node_attributes.set_theta(1, np.pi / 2.0)
    graph.node_attributes.set_vf(0, largest)
    graph.node_attributes.set_vf(1, largest)
    epi_before = graph.node_attributes.get_epis([0, 1]).copy()

    with pytest.raises(TNFRValueError, match="updated epi must be representable"):
        graph.evolve_sparse(dt=1.0, steps=1)

    np.testing.assert_array_equal(
        graph.node_attributes.get_epis([0, 1]),
        epi_before,
    )
    assert graph.node_attributes._dnfr_sparse == {}
