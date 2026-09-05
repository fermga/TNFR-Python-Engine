"""Explicit offset batching preserves nodal replay and has bounded lifetime."""

import asyncio
from contextlib import nullcontext
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import pytest

from tnfr.node import NodeNX
from tnfr.operators.jitter import random_jitter
from tnfr.utils import cache
from tnfr.utils.cache import stable_node_offsets


def _graph(graph_type=nx.Graph, sort=False):
    graph = graph_type()
    graph.add_nodes_from([12, "b", 2, (1, "a")])
    graph.graph.update(RANDOM_SEED=7, SORT_NODES=sort)
    return graph


@pytest.mark.parametrize("graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
@pytest.mark.parametrize("sort", [False, True])
def test_scoped_draws_and_progress_match_ordinary_offsets_exactly(graph_type, sort):
    ordinary, batched = _graph(graph_type, sort), _graph(graph_type, sort)
    for _ in range(3):
        expected = [random_jitter(NodeNX.from_graph(ordinary, node), 0.1) for node in ordinary]
        with stable_node_offsets(batched) as nodes:
            assert nodes == tuple(batched)
            actual = [random_jitter(NodeNX.from_graph(batched, node), 0.1) for node in nodes]
        assert actual == expected
        assert [d["_rng_jitter_progress"] for _, d in batched.nodes(data=True)] == [
            d["_rng_jitter_progress"] for _, d in ordinary.nodes(data=True)
        ]


def test_warm_scope_amortizes_snapshot_validation(monkeypatch):
    graph = nx.empty_graph(120)
    graph.graph["RANDOM_SEED"] = 7
    adapters = [NodeNX.from_graph(graph, node) for node in graph]
    cache.ensure_node_offset_map(graph)
    comparisons = []
    original = cache._same_node_snapshot

    def counted(previous, current):
        comparisons.append(len(current))
        return original(previous, current)

    monkeypatch.setattr(cache, "_same_node_snapshot", counted)
    for node in adapters:
        random_jitter(node, 0.1)
    assert sum(comparisons) == 120 * 120
    comparisons.clear()
    with stable_node_offsets(graph):
        for node in adapters:
            random_jitter(node, 0.1)
    assert sum(comparisons) == 2 * 120


def test_public_offset_still_detects_same_size_reordering_inside_scope():
    graph = nx.empty_graph(3)
    node = NodeNX.from_graph(graph, 2)
    with pytest.raises(RuntimeError, match="not rolled back"):
        with stable_node_offsets(graph):
            graph.remove_node(0)
            graph.add_node(0)
            assert node.offset() == 1
    assert node.offset() == 1


@pytest.mark.parametrize("replacement", [True, 1.0, (True,)])
def test_final_boundary_rejects_equal_valued_cross_type_replacement(replacement):
    original = (1,) if isinstance(replacement, tuple) else 1
    graph = nx.Graph()
    graph.add_node(original)
    with pytest.raises(RuntimeError, match="node order"):
        with stable_node_offsets(graph):
            graph.remove_node(original)
            graph.add_node(replacement)
    assert cache.cached_node_list(graph)[0] is replacement


@pytest.mark.parametrize("mutation", ["size", "sort", "storage", "target"])
def test_detectable_mid_scope_changes_reject_draw_before_progress(mutation):
    graph = nx.empty_graph(3)
    graph.graph["RANDOM_SEED"] = 7
    node = NodeNX.from_graph(graph, 2)
    with pytest.raises(RuntimeError, match="stable_node_offsets"):
        with stable_node_offsets(graph):
            if mutation == "size":
                graph.add_node(9)
            elif mutation == "sort":
                graph.graph["SORT_NODES"] = True
            elif mutation == "storage":
                graph._node = dict(graph._node)
            else:
                graph.remove_node(2)
                graph.add_node(2)
            random_jitter(node, 0.1)
    assert "_rng_jitter_progress" not in graph.nodes[2]


def test_nested_scopes_restore_outer_and_copies_use_their_own_lookup(monkeypatch):
    graph = _graph()
    clone = graph.copy()
    node, copied = NodeNX.from_graph(graph, 2), NodeNX.from_graph(clone, 2)
    lookups = []
    original = NodeNX.offset

    def counted(self):
        lookups.append(self.G)
        return original(self)

    monkeypatch.setattr(NodeNX, "offset", counted)
    with stable_node_offsets(graph):
        random_jitter(node, 0.1)
        random_jitter(copied, 0.1)
        with stable_node_offsets(graph):
            random_jitter(node, 0.1)
        with stable_node_offsets(clone):
            random_jitter(copied, 0.1)
            random_jitter(node, 0.1)
        random_jitter(node, 0.1)
    assert lookups == [clone]
    assert graph.nodes[2]["_rng_jitter_progress"]["draws"] == 4
    assert clone.nodes[2]["_rng_jitter_progress"]["draws"] == 2


def test_nested_scope_cannot_rebase_an_already_invalid_outer_scope():
    graph = nx.empty_graph(3)
    entered = False
    with pytest.raises(RuntimeError, match="stable_node_offsets"):
        with stable_node_offsets(graph):
            graph.remove_node(0)
            graph.add_node(0)
            with stable_node_offsets(graph):
                entered = True
    assert not entered


def test_exception_cleanup_preserves_original_error_and_expires_copied_context():
    graph = _graph()
    node = NodeNX.from_graph(graph, 2)
    with pytest.raises(LookupError, match="original") as caught:
        with stable_node_offsets(graph):
            inherited = copy_context()
            graph.graph["SORT_NODES"] = True
            raise LookupError("original")
    if hasattr(caught.value, "__notes__"):
        assert "stable_node_offsets" in caught.value.__notes__[0]
    assert inherited.run(cache._scoped_node_offset, graph, 2) is None
    random_jitter(node, 0.1)
    assert graph.nodes[2]["_rng_jitter_progress"]["offset"] == node.offset()


def test_inner_exception_leaves_valid_outer_scope_active():
    graph = _graph()
    with stable_node_offsets(graph):
        with pytest.raises(ValueError, match="inner"):
            with stable_node_offsets(graph):
                raise ValueError("inner")
        assert cache._scoped_node_offset(graph, 2) == 2
    assert cache._scoped_node_offset(graph, 2) is None


def test_copied_execution_context_does_not_grant_other_thread_fast_access():
    graph = _graph()
    with stable_node_offsets(graph):
        inherited = copy_context()
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(inherited.run, cache._scoped_node_offset, graph, 2).result() is None


def test_child_async_task_does_not_inherit_parent_fast_access():
    graph = _graph()

    async def child():
        return cache._scoped_node_offset(graph, 2)

    async def parent():
        with stable_node_offsets(graph):
            # Deliberately exercise context inheritance, without graph writes.
            task = asyncio.create_task(child())
            assert await task is None
            assert cache._scoped_node_offset(graph, 2) == 2

    asyncio.run(parent())


@pytest.mark.parametrize("kind", ["copy_view", "filtered", "subclass"])
def test_dynamic_views_and_graph_subclasses_keep_full_lookup_contract(kind):
    graph = nx.path_graph(4)
    if kind == "copy_view":
        unsupported = graph.copy(as_view=True)
    elif kind == "filtered":
        unsupported = nx.subgraph_view(graph, filter_node=lambda node: node != 1)
    else:
        class CustomGraph(nx.Graph):
            pass
        unsupported = CustomGraph(graph)
    with pytest.raises(TypeError, match="view/subclass"):
        with stable_node_offsets(unsupported):
            pass
    with stable_node_offsets(graph):
        assert cache._scoped_node_offset(unsupported, 2) is None
        assert NodeNX.from_graph(unsupported, 2).offset() == list(unsupported).index(2)


def test_custom_node_offset_semantics_are_not_overridden():
    class CustomNode(NodeNX):
        def offset(self):
            return 23

    graph = _graph()
    node = CustomNode(graph, 2)
    with stable_node_offsets(graph):
        random_jitter(node, 0.1)
    assert graph.nodes[2]["_rng_jitter_progress"]["offset"] == 23


def _canonical_records(scoped):
    from tnfr.alias import get_attr
    from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_THETA, ALIAS_DNFR
    from tnfr.initialization import init_node_attrs
    from tnfr.metrics.common import compute_coherence
    from tnfr.metrics.sense_index import compute_Si
    from tnfr.sdk.simple import _run_network_sequence

    graph = nx.cycle_graph(12)
    graph.graph.update(RANDOM_SEED=7, INIT_RANDOM_PHASE=False, INIT_EPI_VALUE=0.0,
                       INIT_VF_MODE="uniform", INIT_VF_MIN=0.4, INIT_VF_MAX=0.7,
                       OZ_NOISE_MODE=True, OZ_SIGMA=0.1, GLYPH_HYSTERESIS_WINDOW=20)
    init_node_attrs(graph)
    records = []

    def record(operator):
        sense = compute_Si(graph, inplace=False)
        records.append((operator, compute_coherence(graph), [
            (node, *(get_attr(data, alias, 0.0)
                     for alias in (ALIAS_EPI, ALIAS_VF, ALIAS_THETA, ALIAS_DNFR)),
             sense[node], tuple(data.get("glyph_history", [])))
            for node, data in graph.nodes(data=True)
        ]))

    record("initial")
    with stable_node_offsets(graph) if scoped else nullcontext():
        _run_network_sequence(
            graph, ["emission", "coherence", "dissonance", "coherence", "silence"],
            cycles=3, validate=True, on_step=record,
        )
    return records


def test_validated_canonical_word_has_exact_same_nodal_telemetry_and_history():
    records = _canonical_records(False)
    assert len(records) == 16
    assert _canonical_records(True) == records


def test_callback_membership_change_is_rejected_at_scope_exit_without_rollback():
    from tnfr.initialization import init_node_attrs
    from tnfr.sdk.simple import _run_network_sequence

    graph = nx.cycle_graph(4)
    graph.graph.update(RANDOM_SEED=7, INIT_RANDOM_PHASE=False)
    init_node_attrs(graph)

    def callback(operator):
        if operator == "silence":
            graph.add_node(99)

    with pytest.raises(RuntimeError, match="not rolled back"):
        with stable_node_offsets(graph):
            _run_network_sequence(graph, ["emission", "coherence", "silence"], on_step=callback)
    assert 99 in graph
    assert list(graph.nodes[0]["glyph_history"]) == ["AL", "IL", "SHA"]
