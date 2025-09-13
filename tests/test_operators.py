"""Pruebas de operators."""

from tnfr.node import NodoNX
from tnfr.operators import (
    JITTER_MANAGER,
    random_jitter,
    apply_glyph,
    _mix_epi_with_neighbors,
    _get_jitter_cache,
)
import tnfr.operators as operators
from types import SimpleNamespace
from tnfr.constants import inject_defaults
import pytest
from weakref import WeakKeyDictionary
from tnfr.types import Glyph


def test_glyph_operations_complete():
    assert set(operators.GLYPH_OPERATIONS) == set(Glyph)


def test_random_jitter_deterministic(graph_canon):
    JITTER_MANAGER.clear()
    G = graph_canon()
    G.add_node(0)
    n0 = NodoNX(G, 0)

    j1 = random_jitter(n0, 0.5)
    j2 = random_jitter(n0, 0.5)
    assert j1 != j2

    JITTER_MANAGER.clear()
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
    JITTER_MANAGER.clear()
    node = SimpleNamespace(graph={})
    cache1 = _get_jitter_cache(node)
    cache1["a"] = 1
    cache2 = _get_jitter_cache(node)
    assert cache1 is cache2
    assert hasattr(node, "_jitter_seed_hash")
    assert node._jitter_seed_hash is cache1
    assert "_jitter_seed_hash" not in node.graph
    assert node.graph not in JITTER_MANAGER.graphs


def test_get_jitter_cache_graph_fallback():
    class G(dict):
        __hash__ = object.__hash__

    class SlotNode:
        __slots__ = ("graph", "__weakref__")

        def __init__(self):
            self.graph = G()

    JITTER_MANAGER.clear()
    node = SlotNode()
    cache1 = _get_jitter_cache(node)
    cache1["b"] = 2
    cache2 = _get_jitter_cache(node)
    assert cache1 is cache2
    assert not hasattr(node, "_jitter_seed_hash")
    graph_cache = node.graph["_jitter_seed_hash"]
    assert isinstance(graph_cache, WeakKeyDictionary)
    assert graph_cache[node] is cache1
    assert node.graph in JITTER_MANAGER.graphs


def test_rng_cache_disabled_with_size_zero(graph_canon):
    from tnfr.rng import set_cache_maxsize
    from tnfr.constants import DEFAULTS

    JITTER_MANAGER.clear()
    set_cache_maxsize(0)
    G = graph_canon()
    G.add_node(0)
    n0 = NodoNX(G, 0)
    j1 = random_jitter(n0, 0.5)
    j2 = random_jitter(n0, 0.5)
    assert j1 == j2
    set_cache_maxsize(DEFAULTS["JITTER_CACHE_SIZE"])


def test_jitter_seq_purges_old_entries():
    JITTER_MANAGER.clear()
    operators.JITTER_MANAGER.setup(force=True, max_entries=4)
    graph = SimpleNamespace(graph={})
    nodes = [SimpleNamespace(G=graph) for _ in range(5)]
    first_key = (0, id(nodes[0]))
    for n in nodes:
        random_jitter(n, 0.1)
    assert len(operators.JITTER_MANAGER.seq) == 4
    assert first_key not in operators.JITTER_MANAGER.seq


def test_jitter_manager_respects_custom_max_entries():
    JITTER_MANAGER.clear()
    operators.JITTER_MANAGER.max_entries = 8
    operators.JITTER_MANAGER.setup(force=True)
    assert operators.JITTER_MANAGER.settings["max_entries"] == 8
    operators.JITTER_MANAGER.setup()
    assert operators.JITTER_MANAGER.settings["max_entries"] == 8


def test_jitter_manager_setup_override_size():
    JITTER_MANAGER.clear()
    operators.JITTER_MANAGER.setup(force=True, max_entries=5)
    assert operators.JITTER_MANAGER.settings["max_entries"] == 5
    operators.JITTER_MANAGER.setup(max_entries=7)
    assert operators.JITTER_MANAGER.settings["max_entries"] == 7


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
