"""Pruebas de operators."""

from tnfr.node import NodeNX
from tnfr.operators import (
    get_jitter_manager,
    reset_jitter_manager,
    random_jitter,
    apply_glyph,
    get_neighbor_epi,
    _mix_epi_with_neighbors,
)
import tnfr.operators as operators
from types import SimpleNamespace
from tnfr.constants import inject_defaults
import pytest
from tnfr.types import Glyph


def test_glyph_operations_complete():
    assert set(operators.GLYPH_OPERATIONS) == set(Glyph)


def test_random_jitter_deterministic(graph_canon):
    reset_jitter_manager()
    G = graph_canon()
    G.add_node(0)
    n0 = NodeNX(G, 0)

    j1 = random_jitter(n0, 0.5)
    j2 = random_jitter(n0, 0.5)
    assert j1 != j2

    manager = get_jitter_manager()
    manager.clear()
    j3 = random_jitter(n0, 0.5)
    j4 = random_jitter(n0, 0.5)
    assert [j3, j4] == [j1, j2]


def test_jitter_cache_metrics(graph_canon):
    reset_jitter_manager()
    G = graph_canon()
    G.add_node(0)
    node = NodeNX(G, 0)

    manager = get_jitter_manager()
    telemetry = manager.cache.manager
    before = telemetry.get_metrics("scoped_counter:jitter")

    random_jitter(node, 0.5)
    random_jitter(node, 0.5)

    after = telemetry.get_metrics("scoped_counter:jitter")
    assert after.misses - before.misses == 1
    assert after.hits - before.hits == 1


def test_random_jitter_zero_amplitude(graph_canon):
    G = graph_canon()
    G.add_node(0)
    n0 = NodeNX(G, 0)
    assert random_jitter(n0, 0.0) == 0.0


def test_random_jitter_negative_amplitude(graph_canon):
    G = graph_canon()
    G.add_node(0)
    n0 = NodeNX(G, 0)
    with pytest.raises(ValueError):
        random_jitter(n0, -0.1)


def test_rng_cache_disabled_with_size_zero(graph_canon):
    from tnfr.rng import set_cache_maxsize
    from tnfr.constants import DEFAULTS

    reset_jitter_manager()
    set_cache_maxsize(0)
    G = graph_canon()
    G.add_node(0)
    n0 = NodeNX(G, 0)
    j1 = random_jitter(n0, 0.5)
    j2 = random_jitter(n0, 0.5)
    assert j1 == j2
    set_cache_maxsize(DEFAULTS["JITTER_CACHE_SIZE"])


def test_jitter_seq_purges_old_entries():
    reset_jitter_manager()
    manager = operators.get_jitter_manager()
    manager.setup(force=True, max_entries=4)
    graph = SimpleNamespace(graph={})
    nodes = [SimpleNamespace(G=graph) for _ in range(5)]
    first_key = (0, id(nodes[0]))
    for n in nodes:
        random_jitter(n, 0.1)
    assert len(manager.seq) == 4
    assert first_key not in manager.seq


def test_jitter_manager_respects_custom_max_entries():
    reset_jitter_manager()
    manager = operators.get_jitter_manager()
    manager.max_entries = 8
    assert manager.settings["max_entries"] == 8
    manager.setup(force=True)
    assert manager.settings["max_entries"] == 8


def test_jitter_manager_setup_override_size():
    reset_jitter_manager()
    manager = operators.get_jitter_manager()
    manager.setup(force=True, max_entries=5)
    assert manager.settings["max_entries"] == 5
    manager.setup(max_entries=7)
    assert manager.settings["max_entries"] == 7


def test_jitter_manager_clear_resets_state():
    reset_jitter_manager()
    manager = operators.get_jitter_manager()
    graph = SimpleNamespace(graph={})
    nodes = [SimpleNamespace(G=graph) for _ in range(3)]
    for node in nodes:
        random_jitter(node, 0.1)
    assert len(manager.seq) == 3
    manager.clear()
    assert len(manager.seq) == 0


def test_get_neighbor_epi_without_graph_preserves_state():
    neigh = [
        SimpleNamespace(EPI=2.0),
        SimpleNamespace(EPI=4.0),
    ]
    node = SimpleNamespace(EPI=1.0, neighbors=lambda: neigh)

    result, epi_bar = get_neighbor_epi(node)

    assert result == neigh
    assert epi_bar == pytest.approx(3.0)
    assert node.EPI == pytest.approx(1.0)


def test_get_neighbor_epi_with_graph_returns_wrapped_nodes(graph_canon):
    G = graph_canon()
    G.add_node(0, EPI=1.0)
    G.add_node(1, EPI=2.0)
    G.add_node(2, EPI=4.0)
    G.add_edge(0, 1)
    G.add_edge(0, 2)

    node = NodeNX(G, 0)
    neighbors, epi_bar = get_neighbor_epi(node)

    assert {n.n for n in neighbors} == {1, 2}
    assert all(hasattr(n, "EPI") for n in neighbors)
    assert epi_bar == pytest.approx(3.0)
    assert node.EPI == pytest.approx(1.0)


def test_get_neighbor_epi_no_neighbors_returns_defaults(graph_canon):
    G = graph_canon()
    G.add_node(0, EPI=1.5)

    node = NodeNX(G, 0)
    neighbors, epi_bar = get_neighbor_epi(node)

    assert neighbors == []
    assert epi_bar == pytest.approx(1.5)
    assert node.EPI == pytest.approx(1.5)


def test_get_neighbor_epi_without_epi_alias_returns_empty(graph_canon):
    G = graph_canon()
    G.add_node(0, EPI=2.0)
    G.add_node(1)
    G.add_edge(0, 1)

    node = NodeNX(G, 0)
    neighbors, epi_bar = get_neighbor_epi(node)

    assert neighbors == []
    assert epi_bar == pytest.approx(2.0)
    assert node.EPI == pytest.approx(2.0)


def test_um_candidate_subset_proximity(graph_canon):
    G = graph_canon()
    inject_defaults(G)
    for i, th in enumerate([0.0, 0.1, 0.2, 1.0]):
        G.add_node(i, **{"theta": th, "EPI": 0.5, "Si": 0.5})

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
