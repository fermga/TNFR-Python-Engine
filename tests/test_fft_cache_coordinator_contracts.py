"""Ownership and key-identity contracts for FFT cache coordination."""

from __future__ import annotations

import networkx as nx
import numpy as np

from tnfr.dynamics.fft_cache_coordinator import FFTCacheCoordinator


def test_kernel_parameters_are_type_safe_and_cached_values_are_detached() -> None:
    graph = nx.path_graph(3)
    cache = FFTCacheCoordinator()

    first = cache.get_kernel(
        graph,
        "window",
        lambda: np.array([1.0]),
        {"a": "1|b=2"},
    )
    first[0] = 99.0
    distinct = cache.get_kernel(
        graph,
        "window",
        lambda: np.array([2.0]),
        {"a": "1", "b": 2},
    )
    repeated = cache.get_kernel(
        graph,
        "window",
        lambda: np.array([3.0]),
        {"a": "1|b=2"},
    )

    np.testing.assert_allclose(distinct, [2.0])
    np.testing.assert_allclose(repeated, [1.0])
    assert cache.get_stats()["kernel_misses"] == 2
    assert cache.get_stats()["kernel_hits"] == 1


def test_public_spectral_basis_cannot_modify_local_cached_basis() -> None:
    graph = nx.path_graph(4)
    cache = FFTCacheCoordinator()
    first = cache.get_spectral_basis(graph)
    expected_values = first.eigenvalues.copy()
    expected_vectors = first.eigenvectors.copy()

    first.eigenvalues.setflags(write=True)
    first.eigenvectors.setflags(write=True)
    first.eigenvalues[0] = 99.0
    first.eigenvectors[0, 0] = 99.0
    repeated = cache.get_spectral_basis(graph)

    assert repeated is not first
    np.testing.assert_allclose(repeated.eigenvalues, expected_values)
    np.testing.assert_allclose(repeated.eigenvectors, expected_vectors)
    assert not repeated.eigenvalues.flags.writeable
    assert not repeated.eigenvectors.flags.writeable


def test_kernel_key_tracks_live_nodal_state_for_arbitrary_builders() -> None:
    graph = nx.path_graph(2)
    graph.nodes[0]["EPI"] = 0.1
    cache = FFTCacheCoordinator()

    first = cache.get_kernel(
        graph, "stateful", lambda: np.array([graph.nodes[0]["EPI"]])
    )
    graph.nodes[0]["EPI"] = 0.7
    second = cache.get_kernel(
        graph, "stateful", lambda: np.array([graph.nodes[0]["EPI"]])
    )

    np.testing.assert_allclose(first, [0.1])
    np.testing.assert_allclose(second, [0.7])
