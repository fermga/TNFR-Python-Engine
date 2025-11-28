import math
from typing import Any

import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants import get_aliases
import tnfr.metrics.sense_index as sense_index_mod
from tnfr.metrics.sense_index import compute_Si
from tnfr.metrics.trig_cache import get_trig_cache
from tnfr.utils import increment_edge_version

ALIAS_THETA = get_aliases("THETA")
ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")


def test_compute_Si_vectorized_uses_bulk_helper(monkeypatch, graph_canon):
    np = pytest.importorskip("numpy")

    calls: list[dict[str, Any]] = []

    original_helper = compute_Si.__globals__["neighbor_phase_mean_bulk"]

    def capture_helper(edge_src, edge_dst, **kwargs):
        calls.append(
            {
                "edge_src": np.asarray(edge_src, dtype=np.intp).copy(),
                "edge_dst": np.asarray(edge_dst, dtype=np.intp).copy(),
                "node_count": kwargs.get("node_count"),
            }
        )
        return original_helper(edge_src, edge_dst, **kwargs)

    monkeypatch.setattr("tnfr.metrics.sense_index.neighbor_phase_mean_bulk", capture_helper)
    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: np)

    G = graph_canon()
    G.add_edge(1, 2)
    for n in G.nodes:
        set_attr(G.nodes[n], ALIAS_THETA, 0.0)
        set_attr(G.nodes[n], ALIAS_VF, 0.0)
        set_attr(G.nodes[n], ALIAS_DNFR, 0.0)

    compute_Si(G, inplace=False)

    assert len(calls) == 1
    recorded = calls[0]
    assert recorded["node_count"] == G.number_of_nodes()
    assert recorded["edge_src"].shape == recorded["edge_dst"].shape
    assert recorded["edge_dst"].size > 0


def _configure_graph(graph):
    graph.add_nodes_from(range(4))
    graph.add_edges_from(((0, 1), (1, 2), (2, 3)))
    phases = [0.0, math.pi / 5, math.pi / 3, math.pi / 2]
    vf_values = [0.2, 0.5, 0.8, 1.1]
    dnfr_values = [0.1, 0.3, 0.4, 0.6]
    for node in graph.nodes:
        set_attr(graph.nodes[node], ALIAS_THETA, phases[node])
        set_attr(graph.nodes[node], ALIAS_VF, vf_values[node])
        set_attr(graph.nodes[node], ALIAS_DNFR, dnfr_values[node])


def test_compute_Si_vectorized_avoids_abs_max_recompute(monkeypatch, graph_canon):
    np = pytest.importorskip("numpy")

    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: np)

    def forbidden(*args, **kwargs):
        raise AssertionError("vector path should not recompute cached maxima")

    monkeypatch.setattr("tnfr.metrics.common.multi_recompute_abs_max", forbidden)

    G = graph_canon()
    _configure_graph(G)

    expected_vfmax = max(abs(get_attr(G.nodes[node], ALIAS_VF, 0.0)) for node in G.nodes)
    expected_dnfrmax = max(abs(get_attr(G.nodes[node], ALIAS_DNFR, 0.0)) for node in G.nodes)

    vectorized = compute_Si(G, inplace=False)

    assert G.graph["_vfmax"] == pytest.approx(expected_vfmax)
    assert G.graph["_dnfrmax"] == pytest.approx(expected_dnfrmax)

    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: None)

    fallback = compute_Si(G, inplace=False)

    assert set(vectorized) == set(fallback)
    for node in vectorized:
        assert fallback[node] == pytest.approx(vectorized[node])


def test_compute_Si_vectorized_buffer_reuse_matches_python(monkeypatch, graph_canon):
    np = pytest.importorskip("numpy")

    G_python = graph_canon()
    _configure_graph(G_python)

    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: None)
    baseline = compute_Si(G_python, inplace=False)

    G_vector = graph_canon()
    _configure_graph(G_vector)

    captured_buffers: list[dict[str, Any]] = []
    original_bulk = sense_index_mod.neighbor_phase_mean_bulk

    def tracking_bulk(*args, **kwargs):
        captured_buffers.append(
            {
                name: kwargs.get(name)
                for name in (
                    "neighbor_cos_sum",
                    "neighbor_sin_sum",
                    "neighbor_counts",
                    "mean_cos",
                    "mean_sin",
                )
            }
        )
        return original_bulk(*args, **kwargs)

    monkeypatch.setattr(
        "tnfr.metrics.sense_index.neighbor_phase_mean_bulk",
        tracking_bulk,
    )
    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: np)

    vector_first = compute_Si(G_vector, inplace=False)
    vector_second = compute_Si(G_vector, inplace=False)

    assert captured_buffers
    first_buffers = captured_buffers[0]
    last_buffers = captured_buffers[-1]
    for name in first_buffers:
        buf_first = first_buffers[name]
        buf_last = last_buffers[name]
        assert isinstance(buf_first, np.ndarray)
        assert buf_first.shape == (G_vector.number_of_nodes(),)
        assert np.issubdtype(buf_first.dtype, np.floating)
        assert buf_first is buf_last

    assert set(vector_first) == set(baseline)
    for node in baseline:
        assert vector_first[node] == pytest.approx(baseline[node])
        assert vector_second[node] == pytest.approx(baseline[node])


def test_compute_Si_vectorized_matches_python(monkeypatch, graph_canon):
    np = pytest.importorskip("numpy")

    G_vec = graph_canon()
    G_py = graph_canon()
    _configure_graph(G_vec)
    _configure_graph(G_py)

    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: np)
    trig_cache = get_trig_cache(G_vec, np=np)

    original_nodes = list(G_vec.nodes)
    reversed_nodes = list(reversed(original_nodes))
    edges = list(G_vec.edges())
    vec_data = {node: dict(G_vec.nodes[node]) for node in original_nodes}
    py_data = {node: dict(G_py.nodes[node]) for node in original_nodes}

    for graph, data_map in ((G_vec, vec_data), (G_py, py_data)):
        graph.remove_nodes_from(list(graph.nodes))
        for node in reversed_nodes:
            graph.add_node(node)
            graph.nodes[node].update(data_map[node])
        graph.add_edges_from(edges)

    trig_cache.order = ()
    vectorized = compute_Si(G_vec, inplace=False)

    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: None)
    python = compute_Si(G_py, inplace=False)

    assert list(G_vec.nodes) == reversed_nodes
    assert set(vectorized) == set(python)
    for node in vectorized:
        assert python[node] == vectorized[node]


def test_compute_Si_python_parallel_matches(monkeypatch, graph_canon):
    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: None)

    G = graph_canon()
    _configure_graph(G)

    sequential = compute_Si(G, inplace=False, n_jobs=1)
    parallel = compute_Si(G, inplace=False, n_jobs=2)

    assert set(sequential) == set(parallel)
    for node in sequential:
        assert parallel[node] == pytest.approx(sequential[node])


def test_compute_Si_reads_jobs_from_graph(monkeypatch, graph_canon):
    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: None)

    captured = []

    class DummyExecutor:
        def __init__(self, max_workers):
            captured.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            result = fn(*args, **kwargs)

            class DummyFuture:
                def result(self_inner):
                    return result

            return DummyFuture()

    monkeypatch.setattr("tnfr.metrics.sense_index.ProcessPoolExecutor", DummyExecutor)

    G = graph_canon()
    _configure_graph(G)
    G.graph["SI_N_JOBS"] = "3"

    compute_Si(G, inplace=False)

    assert captured == [3]


def test_compute_Si_vectorized_respects_chunk_size(monkeypatch, graph_canon):
    np = pytest.importorskip("numpy")

    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: np)

    G = graph_canon()
    _configure_graph(G)

    isolated_node = "isolated"
    G.add_node(isolated_node)
    set_attr(G.nodes[isolated_node], ALIAS_THETA, 0.0)
    set_attr(G.nodes[isolated_node], ALIAS_VF, 0.0)
    set_attr(G.nodes[isolated_node], ALIAS_DNFR, 0.0)

    baseline = compute_Si(G, inplace=False)

    captured: list[tuple[int | None, int]] = []

    def fake_resolve(chunk, total, **kwargs):
        captured.append((chunk, total))
        return 1

    monkeypatch.setattr("tnfr.metrics.sense_index.resolve_chunk_size", fake_resolve)

    connected_nodes = sum(1 for node in G.nodes if G.degree(node) > 0)
    chunked = compute_Si(G, inplace=False, chunk_size=5)

    assert captured and captured[0][0] == 5
    assert captured[0][1] == connected_nodes
    assert set(chunked) == set(baseline)
    for node in baseline:
        assert chunked[node] == pytest.approx(baseline[node])


def test_compute_Si_vectorized_chunked_results_match(monkeypatch, graph_canon):
    np = pytest.importorskip("numpy")

    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: np)

    node_count = 12

    def populate(graph):
        graph.remove_nodes_from(list(graph.nodes))
        graph.add_nodes_from(range(node_count))
        graph.add_edges_from((i, i + 1) for i in range(node_count - 1))
        for node in graph.nodes:
            set_attr(graph.nodes[node], ALIAS_THETA, (node + 1) * math.pi / (node_count + 2))
            set_attr(graph.nodes[node], ALIAS_VF, 0.15 + 0.07 * node)
            set_attr(graph.nodes[node], ALIAS_DNFR, 0.05 + 0.04 * node)

    G_baseline = graph_canon()
    G_chunked = graph_canon()
    populate(G_baseline)
    populate(G_chunked)

    baseline = compute_Si(G_baseline, inplace=False)

    original_resolve = sense_index_mod.resolve_chunk_size
    captured: list[tuple[int | None, int, int]] = []
    workspace_calls: list[int] = []

    def tracking_resolve(chunk_pref, total, **kwargs):
        result = original_resolve(chunk_pref, total, **kwargs)
        captured.append((chunk_pref, total, result))
        return result

    monkeypatch.setattr("tnfr.metrics.sense_index.resolve_chunk_size", tracking_resolve)

    original_workspace = sense_index_mod._ensure_chunk_workspace

    def tracking_workspace(G, *, mask_count, np):
        workspace_calls.append(mask_count)
        return original_workspace(G, mask_count=mask_count, np=np)

    monkeypatch.setattr(
        "tnfr.metrics.sense_index._ensure_chunk_workspace",
        tracking_workspace,
    )

    chunked = compute_Si(G_chunked, inplace=False, chunk_size=5)
    neighbor_count = sum(1 for node in G_chunked.nodes if G_chunked.degree(node) > 0)

    assert captured and captured[-1][0] == 5
    resolved_chunk = captured[-1][2]
    assert resolved_chunk <= 5
    assert resolved_chunk < node_count
    assert captured[-1][1] == node_count
    assert workspace_calls == [neighbor_count]
    assert set(chunked) == set(baseline)
    for node in baseline:
        assert chunked[node] == pytest.approx(baseline[node])


def test_compute_Si_vectorized_skips_isolated_nodes(monkeypatch, graph_canon):
    np = pytest.importorskip("numpy")

    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: np)

    captured_masks: list[Any] = []
    original_bulk = sense_index_mod.neighbor_phase_mean_bulk

    def tracking_bulk(*args, **kwargs):
        mean_theta, mask = original_bulk(*args, **kwargs)
        captured_masks.append(mask.copy())
        return mean_theta, mask

    monkeypatch.setattr("tnfr.metrics.sense_index.neighbor_phase_mean_bulk", tracking_bulk)

    remainder_inputs: list[Any] = []
    original_remainder = np.remainder

    def tracking_remainder(x1, x2, out=None, where=True, **kwargs):
        remainder_inputs.append(np.asarray(x1).copy())
        return original_remainder(x1, x2, out=out, where=where, **kwargs)

    monkeypatch.setattr(np, "remainder", tracking_remainder)

    G = graph_canon()
    G.add_nodes_from(range(4))
    G.add_edge(0, 1)

    for node in G.nodes:
        set_attr(G.nodes[node], ALIAS_THETA, float(node) * math.pi / 8)
        set_attr(G.nodes[node], ALIAS_VF, 0.0)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.0)

    monkeypatch.setattr(
        "tnfr.metrics.sense_index.resolve_chunk_size",
        lambda chunk_pref, total, **kwargs: total,
    )

    compute_Si(G, inplace=False)

    assert captured_masks
    mask = captured_masks[0]
    assert mask.dtype == np.bool_
    connected = int(mask.sum())
    assert connected == 2
    assert not mask[2]
    assert not mask[3]
    assert remainder_inputs
    assert len(remainder_inputs) == 1
    assert remainder_inputs[0].shape == (connected,)

    remainder_inputs.clear()
    monkeypatch.setattr(
        "tnfr.metrics.sense_index.resolve_chunk_size",
        lambda chunk_pref, total, **kwargs: 1,
    )

    compute_Si(G, inplace=False)

    assert remainder_inputs
    assert len(remainder_inputs) == connected
    assert sum(arr.size for arr in remainder_inputs) == connected
    assert all(arr.size > 0 for arr in remainder_inputs)


def test_compute_Si_python_parallel_chunk_size(monkeypatch, graph_canon):
    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: None)

    payload_lengths: list[int] = []

    class DummyExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, chunk, **kwargs):
            payload_lengths.append(len(chunk))
            result = fn(chunk, **kwargs)

            class DummyFuture:
                def __init__(self_inner, value):
                    self_inner.value = value

                def result(self_inner):
                    return self_inner.value

            return DummyFuture(result)

    monkeypatch.setattr("tnfr.metrics.sense_index.ProcessPoolExecutor", DummyExecutor)

    G = graph_canon()
    _configure_graph(G)

    compute_Si(G, inplace=False, n_jobs=3, chunk_size=2)

    assert payload_lengths
    assert all(1 <= length <= 2 for length in payload_lengths)

    payload_lengths.clear()
    G.graph["SI_CHUNK_SIZE"] = 3

    compute_Si(G, inplace=False, n_jobs=3)

    assert payload_lengths
    assert all(1 <= length <= 3 for length in payload_lengths)


def test_compute_Si_edge_indices_cache_invalidation(monkeypatch, graph_canon):
    np = pytest.importorskip("numpy")

    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: np)

    from collections import Counter

    builder_counts: Counter[Any] = Counter()
    original_edge_cache = compute_Si.__globals__["edge_version_cache"]

    def tracking_edge_cache(G, key, builder, **kwargs):
        def tracked_builder():
            builder_counts[key] += 1
            return builder()

        return original_edge_cache(G, key, tracked_builder, **kwargs)

    monkeypatch.setattr("tnfr.metrics.sense_index.edge_version_cache", tracking_edge_cache)

    G = graph_canon()
    _configure_graph(G)

    _ = compute_Si(G, inplace=False)
    trig_cache = get_trig_cache(G, np=np)
    assert trig_cache.edge_src is not None
    assert trig_cache.edge_dst is not None
    edge_keys = [
        key for key in builder_counts if isinstance(key, tuple) and key and key[0] == "_si_edges"
    ]
    assert edge_keys and sum(builder_counts[key] for key in edge_keys) == 1

    compute_Si(G, inplace=False)
    trig_again = get_trig_cache(G, np=np)
    assert sum(builder_counts[key] for key in edge_keys) == 1
    assert trig_again.edge_src is trig_cache.edge_src
    assert trig_again.edge_dst is trig_cache.edge_dst

    G.add_edge(0, 3)
    increment_edge_version(G)

    vector_after = compute_Si(G, inplace=False)
    trig_after = get_trig_cache(G, np=np)
    assert sum(builder_counts[key] for key in edge_keys) == 2
    assert trig_after.edge_src is not None
    assert trig_after.edge_dst is not None
    assert (
        trig_after.edge_src is not trig_again.edge_src
        or trig_after.edge_dst is not trig_again.edge_dst
    )

    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: None)
    python_after = compute_Si(G, inplace=False)

    assert set(vector_after) == set(python_after)
    for node in vector_after:
        assert python_after[node] == pytest.approx(vector_after[node])


def test_compute_Si_reuses_structural_arrays(monkeypatch, graph_canon):
    np = pytest.importorskip("numpy")

    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: np)

    rebuilds = 0

    original_rebuild = sense_index_mod._SiStructuralCache.rebuild

    def tracking_rebuild(self, node_ids, data_by_node, *, np):
        nonlocal rebuilds
        rebuilds += 1
        return original_rebuild(self, node_ids, data_by_node, np=np)

    monkeypatch.setattr(sense_index_mod._SiStructuralCache, "rebuild", tracking_rebuild)

    G = graph_canon()
    _configure_graph(G)

    compute_Si(G, inplace=False)
    assert rebuilds == 1

    compute_Si(G, inplace=False)
    assert rebuilds == 1


def test_compute_Si_structural_cache_invalidated_on_attribute_change(monkeypatch, graph_canon):
    np = pytest.importorskip("numpy")

    monkeypatch.setattr("tnfr.metrics.sense_index.get_numpy", lambda: np)

    rebuilds = 0

    original_rebuild = sense_index_mod._SiStructuralCache.rebuild

    def tracking_rebuild(self, node_ids, data_by_node, *, np):
        nonlocal rebuilds
        rebuilds += 1
        return original_rebuild(self, node_ids, data_by_node, np=np)

    monkeypatch.setattr(sense_index_mod._SiStructuralCache, "rebuild", tracking_rebuild)

    G = graph_canon()
    _configure_graph(G)

    compute_Si(G, inplace=False)
    assert rebuilds == 1

    set_attr(G.nodes[0], ALIAS_VF, 0.123)

    compute_Si(G, inplace=False)
    assert rebuilds == 2

    set_attr(G.nodes[1], ALIAS_DNFR, 0.789)

    compute_Si(G, inplace=False)
    assert rebuilds == 3
