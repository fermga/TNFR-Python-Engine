"""Pruebas de operators."""

from tnfr.node import NodoNX
from tnfr.operators import (
    random_jitter,
    clear_rng_cache,
    apply_glyph,
    _mix_epi_with_neighbors,
    _get_jitter_cache,
    _JITTER_GRAPHS,
)
import tnfr.operators as operators
from types import SimpleNamespace
from tnfr.constants import inject_defaults
import pytest
from weakref import WeakKeyDictionary


def test_random_jitter_deterministic(graph_canon):
    clear_rng_cache()
    G = graph_canon()
    G.add_node(0)
    n0 = NodoNX(G, 0)

    j1 = random_jitter(n0, 0.5)
    j2 = random_jitter(n0, 0.5)
    assert j1 != j2

    clear_rng_cache()
    j3 = random_jitter(n0, 0.5)
    j4 = random_jitter(n0, 0.5)
    assert [j3, j4] == [j1, j2]


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


def test_get_jitter_cache_node_attribute():
    clear_rng_cache()
    node = SimpleNamespace(graph={})
    cache1 = _get_jitter_cache(node)
    cache1["a"] = 1
    cache2 = _get_jitter_cache(node)
    assert cache1 is cache2
    assert hasattr(node, "_jitter_seed_hash")
    assert node._jitter_seed_hash is cache1
    assert "_jitter_seed_hash" not in node.graph
    assert node.graph not in _JITTER_GRAPHS


def test_get_jitter_cache_graph_fallback():
    class G(dict):
        __hash__ = object.__hash__

    class SlotNode:
        __slots__ = ("graph", "__weakref__")

        def __init__(self):
            self.graph = G()

    clear_rng_cache()
    node = SlotNode()
    cache1 = _get_jitter_cache(node)
    cache1["b"] = 2
    cache2 = _get_jitter_cache(node)
    assert cache1 is cache2
    assert not hasattr(node, "_jitter_seed_hash")
    graph_cache = node.graph["_jitter_seed_hash"]
    assert isinstance(graph_cache, WeakKeyDictionary)
    assert graph_cache[node] is cache1
    assert node.graph in _JITTER_GRAPHS


def test_rng_cache_disabled_with_size_zero(graph_canon):
    from tnfr.rng import set_cache_maxsize
    from tnfr.constants import DEFAULTS

    clear_rng_cache()
    set_cache_maxsize(0)
    G = graph_canon()
    G.add_node(0)
    n0 = NodoNX(G, 0)
    j1 = random_jitter(n0, 0.5)
    j2 = random_jitter(n0, 0.5)
    assert j1 == j2
    set_cache_maxsize(DEFAULTS["JITTER_CACHE_SIZE"])


def test_jitter_seq_purges_old_entries(monkeypatch):
    clear_rng_cache()
    monkeypatch.setattr(operators, "_JITTER_MAX_ENTRIES", 4)
    graph = SimpleNamespace(graph={})
    nodes = [SimpleNamespace(G=graph) for _ in range(5)]
    first_key = (0, id(nodes[0]))
    for n in nodes:
        random_jitter(n, 0.1)
    assert len(operators._JITTER_SEQ) == 4
    assert first_key not in operators._JITTER_SEQ


def test_um_candidate_subset_proximity(graph_canon):
    G = graph_canon()
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


def test_mix_epi_with_neighbors_prefers_higher_epi():
    neigh = [
        SimpleNamespace(EPI=-3.0, epi_kind="n1"),
        SimpleNamespace(EPI=2.0, epi_kind="n2"),
    ]
    node = SimpleNamespace(EPI=1.0, epi_kind="self", neighbors=lambda: neigh)
    epi_bar, dominant = _mix_epi_with_neighbors(node, 0.25, "EN")
    assert epi_bar == pytest.approx(-0.5)
    assert node.EPI == pytest.approx(0.625)
    assert dominant == "n1"
    assert node.epi_kind == "n1"


def test_mix_epi_with_neighbors_returns_node_kind_on_tie():
    neigh = [SimpleNamespace(EPI=1.0, epi_kind="n1")]
    node = SimpleNamespace(EPI=1.0, epi_kind="self", neighbors=lambda: neigh)
    epi_bar, dominant = _mix_epi_with_neighbors(node, 0.25, "EN")
    assert epi_bar == pytest.approx(1.0)
    assert node.EPI == pytest.approx(1.0)
    assert dominant == "self"
    assert node.epi_kind == "self"


def test_mix_epi_with_neighbors_no_neighbors():
    node = SimpleNamespace(EPI=1.0, epi_kind="self", neighbors=lambda: [])
    epi_bar, dominant = _mix_epi_with_neighbors(node, 0.25, "EN")
    assert epi_bar == pytest.approx(1.0)
    assert node.EPI == pytest.approx(1.0)
    assert dominant == "EN"
    assert node.epi_kind == "EN"
