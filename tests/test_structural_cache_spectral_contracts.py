"""Spectral authentication and time semantics for the structural cache."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics.fft_cache_coordinator import FFTCacheCoordinator
from tnfr.dynamics.structural_cache import StructuralCoherenceCache


def _graph(node_order: tuple[object, ...] = (0, 1)) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(node_order)
    graph.add_edge(0, 1, weight=1.0)
    for node, epi in ((0, 0.0), (1, 1.0)):
        graph.nodes[node].update(
            EPI=epi,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            dEPI=0.0,
        )
    return graph


def test_supplied_spectral_basis_must_match_graph_and_node_order() -> None:
    source = _graph((0, 1))
    reordered = _graph((1, 0))
    basis = FFTCacheCoordinator().get_spectral_basis(source)
    cache = StructuralCoherenceCache()

    with pytest.raises(ValueError, match="signature.*live graph and node order"):
        cache.get_structural_fields(reordered, spectral_basis=basis)

    assert cache.get_cache_stats()["structural_entries"] == 0


@pytest.mark.parametrize(
    ("channel", "replacement", "message"),
    [
        ("eigenvalues", np.array([0.0]), "shape"),
        ("eigenvectors", np.eye(1), "shape"),
        ("eigenvalues", np.array([0.0, np.nan]), "finite"),
        ("eigenvectors", np.array([[1.0, 0.0], [0.0, np.nan]]), "finite"),
        ("eigenvectors", np.eye(2), "diagonalize"),
    ],
)
def test_supplied_spectral_basis_must_satisfy_full_basis_contract(
    channel: str, replacement: np.ndarray, message: str
) -> None:
    graph = _graph()
    basis = FFTCacheCoordinator().get_spectral_basis(graph)
    values = {
        "signature": basis.signature,
        "eigenvalues": basis.eigenvalues,
        "eigenvectors": basis.eigenvectors,
    }
    values[channel] = replacement
    malformed = SimpleNamespace(**values)
    cache = StructuralCoherenceCache()

    with pytest.raises(ValueError, match=message):
        cache.get_structural_fields(graph, spectral_basis=malformed)

    assert cache.get_cache_stats()["structural_entries"] == 0


def test_structural_spectral_snapshot_cannot_poison_cached_basis() -> None:
    graph = _graph()
    basis = FFTCacheCoordinator().get_spectral_basis(graph)
    cache = StructuralCoherenceCache()
    first = cache.get_structural_fields(graph, spectral_basis=basis)
    expected = np.array(first.eigenvalues, copy=True)

    assert first.eigenvalues is not None
    first.eigenvalues.setflags(write=True)
    first.eigenvalues[:] = 99.0
    second = cache.get_structural_fields(graph, spectral_basis=basis)

    np.testing.assert_array_equal(second.eigenvalues, expected)
    assert not np.shares_memory(first.eigenvalues, second.eigenvalues)


def test_structural_cache_separates_creation_time_from_live_graph_time() -> None:
    graph = _graph()
    cache = StructuralCoherenceCache()
    first = cache.get_structural_fields(graph)

    assert first.state_time is None
    assert first.timestamp is None
    assert first.created_at > 0.0

    graph.graph["_t"] = 7.5
    second = cache.get_structural_fields(graph)

    assert second.state_time == pytest.approx(7.5)
    assert second.timestamp == pytest.approx(7.5)
    assert second.created_at == first.created_at
    assert cache.hits == 1


@pytest.mark.parametrize(
    "bad_time",
    [True, np.bool_(True), "later", float("nan"), float("inf")],
)
def test_structural_cache_rejects_invalid_graph_state_time_on_hit(
    bad_time: object,
) -> None:
    graph = _graph()
    cache = StructuralCoherenceCache()
    cache.get_structural_fields(graph)
    graph.graph["_t"] = bad_time

    with pytest.raises(ValueError, match="state time"):
        cache.get_structural_fields(graph)
