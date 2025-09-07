"""Pruebas de operators."""

from tnfr.node import NodoNX
from tnfr.operators import (
    random_jitter,
    clear_rng_cache,
    apply_glyph,
    _select_dominant_glyph,
)
from types import SimpleNamespace
from tnfr.constants import inject_defaults
import networkx as nx
import pytest


def test_random_jitter_deterministic_with_and_without_cache(graph_canon):
    clear_rng_cache()
    G = graph_canon()
    G.add_node(0)
    n0 = NodoNX(G, 0)

    # Without explicit cache: uses global LRU cache and advances sequence
    j1 = random_jitter(n0, 0.5)
    j2 = random_jitter(n0, 0.5)
    assert j1 != j2

    # Clearing the LRU cache reproduces the deterministic sequence
    clear_rng_cache()
    j3 = random_jitter(n0, 0.5)
    j4 = random_jitter(n0, 0.5)
    assert [j3, j4] == [j1, j2]

    # With explicit cache: reproduces deterministic sequence independently
    cache = {}
    j5 = random_jitter(n0, 0.5, cache)
    j6 = random_jitter(n0, 0.5, cache)
    assert j5 == j1
    assert j6 == j2

    # Replaying with a fresh cache reproduces the sequence
    cache2 = {}
    seq = [random_jitter(n0, 0.5, cache2) for _ in range(2)]
    assert seq == [j5, j6]


def test_random_jitter_zero_amplitude(graph_canon):
    G = graph_canon()
    G.add_node(0)
    n0 = NodoNX(G, 0)
    assert random_jitter(n0, 0.0) == 0.0


def test_random_jitter_negative_amplitude(graph_canon):
    G = graph_canon()
    G.add_node(0)
    n0 = NodoNX(G, 0)
    with pytest.raises(ValueError):
        random_jitter(n0, -0.1)


def test_rng_cache_disabled_with_size_zero(graph_canon):
    clear_rng_cache()
    G = graph_canon()
    G.graph["JITTER_CACHE_SIZE"] = 0
    G.add_node(0)
    n0 = NodoNX(G, 0)
    j1 = random_jitter(n0, 0.5)
    j2 = random_jitter(n0, 0.5)
    assert j1 == j2


def test_um_candidate_subset_proximity():
    G = nx.Graph()
    inject_defaults(G)
    for i, th in enumerate([0.0, 0.1, 0.2, 1.0]):
        G.add_node(i, **{"θ": th, "EPI": 0.5, "Si": 0.5})

    G.graph["UM_FUNCTIONAL_LINKS"] = True
    G.graph["UM_COMPAT_THRESHOLD"] = -1.0
    G.graph["UM_CANDIDATE_COUNT"] = 2
    G.graph["UM_CANDIDATE_MODE"] = "proximity"

    apply_glyph(G, 0, "UM")

    assert G.has_edge(0, 1)
    assert G.has_edge(0, 2)
    assert not G.has_edge(0, 3)


def test_select_dominant_glyph_prefers_higher_epi():
    node = SimpleNamespace(EPI=1.0, epi_kind="self")
    neigh = [
        SimpleNamespace(EPI=-3.0, epi_kind="n1"),
        SimpleNamespace(EPI=2.0, epi_kind="n2"),
    ]
    assert _select_dominant_glyph(node, neigh) == "n1"


def test_select_dominant_glyph_returns_node_kind_on_tie():
    node = SimpleNamespace(EPI=1.0, epi_kind="self")
    neigh = [SimpleNamespace(EPI=-1.0, epi_kind="n1")]
    assert _select_dominant_glyph(node, neigh) == "self"


def test_select_dominant_glyph_no_neighbors():
    node = SimpleNamespace(EPI=1.0, epi_kind="self")
    assert _select_dominant_glyph(node, []) == "self"
