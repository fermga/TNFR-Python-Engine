"""Unit tests for vectorized dynamics evolution and performance boundaries."""

import logging
import math
import time
from contextlib import contextmanager, nullcontext

import networkx as nx
import pytest

from tnfr.alias import collect_attr, get_attr, set_attr
from tnfr.constants import get_aliases
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.dynamics.dnfr import (
    _accumulate_neighbors_numpy,
    _build_edge_index_arrays,
    _build_neighbor_sums_common,
    _compute_dnfr,
    _init_neighbor_sums,
    _prefer_sparse_accumulation,
    _prepare_dnfr_data,
    _resolve_numpy_degree_array,
)
from tnfr.helpers.numeric import angle_diff
from tnfr.utils import mark_dnfr_prep_dirty
from tnfr.utils.cache import DNFR_PREP_STATE_KEY, DnfrPrepState, _graph_cache_manager

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")


@contextmanager
def numpy_disabled(monkeypatch):
    import tnfr.dynamics.dnfr as dnfr_module

    with monkeypatch.context() as ctx:
        ctx.setattr(dnfr_module, "get_numpy", lambda: None)
        yield


def _setup_graph():
    G = nx.path_graph(5)
    for n in G.nodes:
        set_attr(G.nodes[n], ALIAS_THETA, 0.1 * (n + 1))
        set_attr(G.nodes[n], ALIAS_EPI, 0.2 * (n + 1))
        set_attr(G.nodes[n], ALIAS_VF, 0.3 * (n + 1))
    G.graph["DNFR_WEIGHTS"] = {
        "phase": 0.4,
        "epi": 0.3,
        "vf": 0.2,
        "topo": 0.1,
    }
    return G


def _get_prep_state(G):
    manager = _graph_cache_manager(G.graph)
    state = manager.get(DNFR_PREP_STATE_KEY)
    assert isinstance(state, DnfrPrepState)
    return manager, state


def _build_invalid_state_graph():
    G = nx.path_graph(6)
    for n in G.nodes:
        set_attr(G.nodes[n], ALIAS_THETA, 0.0)
        set_attr(G.nodes[n], ALIAS_EPI, 0.0)
        set_attr(G.nodes[n], ALIAS_VF, 0.0)
    G.graph["DNFR_WEIGHTS"] = {
        "phase": 1.0,
        "epi": 1.0,
        "vf": 1.0,
        "topo": 0.0,
    }

    noisy = G.nodes[1]
    for alias in ALIAS_THETA:
        noisy[alias] = "bad-phase"
    for alias in ALIAS_EPI:
        noisy[alias] = "bad-epi"
    for alias in ALIAS_VF:
        noisy[alias] = "bad-vf"

    nan_value = float("nan")
    unstable = G.nodes[3]
    for alias in ALIAS_THETA:
        unstable[alias] = nan_value
    for alias in ALIAS_EPI:
        unstable[alias] = nan_value
    for alias in ALIAS_VF:
        unstable[alias] = nan_value

    return G, (1, 3)


@pytest.mark.parametrize("disable_numpy", [False, True])
def test_default_compute_delta_nfr_paths(disable_numpy, monkeypatch):
    if not disable_numpy:
        pytest.importorskip("numpy")
    G = _setup_graph()
    if disable_numpy:
        with numpy_disabled(monkeypatch):
            default_compute_delta_nfr(G)
    else:
        default_compute_delta_nfr(G)
    dnfr = collect_attr(G, G.nodes, ALIAS_DNFR, 0.0)
    assert len(dnfr) == 5


@pytest.mark.parametrize("numpy_mode", ["vectorized", "python"])
def test_default_compute_delta_nfr_clamps_invalid_states(numpy_mode, caplog, monkeypatch):
    if numpy_mode == "vectorized":
        pytest.importorskip("numpy")
        context = nullcontext()
    else:
        context = numpy_disabled(monkeypatch)

    G, bad_nodes = _build_invalid_state_graph()

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="tnfr.utils.data"):
        with context:
            default_compute_delta_nfr(G)

    dnfr = collect_attr(G, G.nodes, ALIAS_DNFR, 0.0)
    for node in bad_nodes:
        assert math.isfinite(dnfr[node])
        assert dnfr[node] == pytest.approx(0.0)

    messages = [record.getMessage() for record in caplog.records if record.name == "tnfr.utils.data"]
    assert any("Could not convert value" in msg for msg in messages)
    assert any("Non-finite value" in msg for msg in messages)


def test_default_vectorization_auto_enabled_when_numpy_available():
    np = pytest.importorskip("numpy")
    G = _setup_graph()
    default_compute_delta_nfr(G)
    _, state = _get_prep_state(G)
    cache = state.cache
    assert cache is not None
    assert cache.theta_np is not None
    assert cache.edge_src is not None and isinstance(cache.edge_src, np.ndarray)
    assert isinstance(cache.grad_total_np, np.ndarray)
    assert isinstance(cache.grad_phase_np, np.ndarray)


def test_vectorization_falls_back_without_numpy(monkeypatch):
    G = _setup_graph()
    with numpy_disabled(monkeypatch):
        default_compute_delta_nfr(G)
    _, state = _get_prep_state(G)
    cache = state.cache
    assert cache is not None
    assert cache.theta_np is None
    assert cache.edge_src is None
    assert cache.grad_total_np is None
    assert cache.grad_phase_np is None


def test_numpy_available_when_vectorized_disabled(monkeypatch):
    np = pytest.importorskip("numpy")

    baseline = _build_weighted_graph(nx.path_graph, 5, 0.2)
    baseline.graph["vectorized_dnfr"] = False
    accelerated = _build_weighted_graph(nx.path_graph, 5, 0.2)
    accelerated.graph["vectorized_dnfr"] = False

    with numpy_disabled(monkeypatch):
        default_compute_delta_nfr(baseline)

    default_compute_delta_nfr(accelerated)

    dnfr_baseline = collect_attr(baseline, baseline.nodes, ALIAS_DNFR, 0.0)
    dnfr_accelerated = collect_attr(accelerated, accelerated.nodes, ALIAS_DNFR, 0.0)

    assert dnfr_accelerated == pytest.approx(dnfr_baseline)

    _, state = _get_prep_state(accelerated)
    cache = state.cache
    assert cache is not None
    assert isinstance(cache.edge_src, np.ndarray)
    assert isinstance(cache.neighbor_x_np, np.ndarray)


def test_vectorized_gradients_cached_and_reused():
    np = pytest.importorskip("numpy")
    G = _build_weighted_graph(nx.path_graph, 4, 0.3)
    default_compute_delta_nfr(G)
    manager, state = _get_prep_state(G)
    cache = state.cache
    assert isinstance(cache.grad_total_np, np.ndarray)
    assert isinstance(cache.grad_phase_np, np.ndarray)
    before = cache.grad_total_np.copy()

    # Run again to confirm the buffers are reused
    default_compute_delta_nfr(G)
    _, state_after = _get_prep_state(G)
    cache2 = state_after.cache
    assert cache2 is cache
    assert cache2.grad_total_np is cache.grad_total_np
    assert cache2.grad_phase_np is cache.grad_phase_np
    after = cache2.grad_total_np
    assert after.shape == before.shape
    stats = manager.get_metrics(DNFR_PREP_STATE_KEY)
    assert stats.hits >= 1


def test_compute_dnfr_auto_vectorizes_when_numpy_present(monkeypatch):
    np = pytest.importorskip("numpy")
    del np

    graph = _build_weighted_graph(nx.path_graph, 6, 0.25)
    data = _prepare_dnfr_data(graph)

    import tnfr.dynamics.dnfr as dnfr_module

    calls = []

    original = dnfr_module._build_neighbor_sums_common

    def _spy(G_inner, data_inner, *, use_numpy):
        calls.append(use_numpy)
        return original(G_inner, data_inner, use_numpy=use_numpy)

    monkeypatch.setattr(dnfr_module, "_build_neighbor_sums_common", _spy)

    _compute_dnfr(graph, data)
    assert calls and calls[-1] is True

    calls.clear()
    with numpy_disabled(monkeypatch):
        fallback_graph = _build_weighted_graph(nx.path_graph, 6, 0.25)
        fallback_data = _prepare_dnfr_data(fallback_graph)
        _compute_dnfr(fallback_graph, fallback_data)
    assert calls and calls[-1] is False


def _build_weighted_graph(factory, n_nodes: int, topo_weight: float):
    G = factory(n_nodes)
    for idx, node in enumerate(G.nodes):
        set_attr(G.nodes[node], ALIAS_THETA, 0.15 * (idx + 1))
        set_attr(G.nodes[node], ALIAS_EPI, 0.05 * (idx + 2))
        set_attr(G.nodes[node], ALIAS_VF, 0.12 * (idx + 3))
    G.graph["DNFR_WEIGHTS"] = {
        "phase": 0.35,
        "epi": 0.25,
        "vf": 0.3,
        "topo": topo_weight,
    }
    return G


@pytest.mark.parametrize("factory", [nx.path_graph, nx.complete_graph])
@pytest.mark.parametrize("topo_weight", [0.0, 0.4])
def test_vectorized_matches_reference(factory, topo_weight, monkeypatch):
    np = pytest.importorskip("numpy")
    del np  # only needed to guarantee NumPy availability

    G_list = _build_weighted_graph(factory, 6, topo_weight)
    G_vec = _build_weighted_graph(factory, 6, topo_weight)

    with numpy_disabled(monkeypatch):
        default_compute_delta_nfr(G_list)

    default_compute_delta_nfr(G_vec)

    dnfr_list = collect_attr(G_list, G_list.nodes, ALIAS_DNFR, 0.0)
    dnfr_vec = collect_attr(G_vec, G_vec.nodes, ALIAS_DNFR, 0.0)
    assert dnfr_vec == pytest.approx(dnfr_list)
    assert G_vec.graph.get("_DNFR_META") == G_list.graph.get("_DNFR_META")
    assert G_vec.graph.get("_dnfr_hook_name") == G_list.graph.get("_dnfr_hook_name")


def _build_dense_graph_regression():
    G = nx.complete_graph(5)
    G.remove_edge(1, 2)
    angles = [
        0.0,
        math.pi / 4.0,
        -math.pi / 4.0,
        3.0 * math.pi / 4.0,
        -3.0 * math.pi / 4.0,
    ]
    epi_values = [0.25 * (idx + 1) for idx in range(len(angles))]
    vf_values = [0.1 * (idx + 2) for idx in range(len(angles))]
    for idx, node in enumerate(G.nodes):
        set_attr(G.nodes[node], ALIAS_THETA, angles[idx])
        set_attr(G.nodes[node], ALIAS_EPI, epi_values[idx])
        set_attr(G.nodes[node], ALIAS_VF, vf_values[idx])
    G.graph["DNFR_WEIGHTS"] = {
        "phase": 0.4,
        "epi": 0.3,
        "vf": 0.2,
        "topo": 0.1,
    }
    return G


def _manual_dense_dnfr_expected(G):
    weights = G.graph["DNFR_WEIGHTS"]
    degs = {node: float(deg) for node, deg in G.degree()}
    expected = []
    nodes = list(G.nodes)
    for node in nodes:
        nd = G.nodes[node]
        th_i = float(get_attr(nd, ALIAS_THETA, 0.0))
        epi_i = float(get_attr(nd, ALIAS_EPI, 0.0))
        vf_i = float(get_attr(nd, ALIAS_VF, 0.0))
        neighbors = list(G.neighbors(node))
        count = len(neighbors)
        if count:
            cos_sum = 0.0
            sin_sum = 0.0
            epi_sum = 0.0
            vf_sum = 0.0
            deg_sum = 0.0
            for v in neighbors:
                th_v = float(get_attr(G.nodes[v], ALIAS_THETA, th_i))
                cos_sum += math.cos(th_v)
                sin_sum += math.sin(th_v)
                epi_sum += float(get_attr(G.nodes[v], ALIAS_EPI, epi_i))
                vf_sum += float(get_attr(G.nodes[v], ALIAS_VF, vf_i))
                deg_sum += degs[v]
            inv = 1.0 / count
            cos_avg = cos_sum * inv
            sin_avg = sin_sum * inv
            if math.hypot(cos_avg, sin_avg) <= 1e-12:
                th_bar = th_i
            else:
                th_bar = math.atan2(sin_avg, cos_avg)
            epi_bar = epi_sum * inv
            vf_bar = vf_sum * inv
            deg_bar = deg_sum * inv if weights.get("topo", 0.0) else 0.0
        else:
            th_bar = th_i
            epi_bar = epi_i
            vf_bar = vf_i
            deg_bar = degs[node]
        g_phase = -angle_diff(th_i, th_bar) / math.pi
        g_epi = epi_bar - epi_i
        g_vf = vf_bar - vf_i
        if weights.get("topo", 0.0) and count:
            g_topo = deg_bar - degs[node]
        else:
            g_topo = 0.0
        dnfr_value = (
            weights.get("phase", 0.0) * g_phase
            + weights.get("epi", 0.0) * g_epi
            + weights.get("vf", 0.0) * g_vf
            + weights.get("topo", 0.0) * g_topo
        )
        expected.append(dnfr_value)
    return expected


def _legacy_numpy_stack_accumulation(
    G,
    data,
    *,
    x,
    y,
    epi_sum,
    vf_sum,
    count,
    deg_sum,
    np,
):
    nodes = data["nodes"]
    if not nodes:
        return x, y, epi_sum, vf_sum, count, deg_sum, None

    cache = data.get("cache")
    epi = data.get("epi_np")
    if epi is None:
        epi = np.array(data["epi"], dtype=float)
        data["epi_np"] = epi
        if cache is not None:
            cache.epi_np = epi
    cos_th = data.get("cos_theta_np")
    if cos_th is None:
        cos_th = np.array(data["cos_theta"], dtype=float)
        data["cos_theta_np"] = cos_th
        if cache is not None:
            cache.cos_theta_np = cos_th
    sin_th = data.get("sin_theta_np")
    if sin_th is None:
        sin_th = np.array(data["sin_theta"], dtype=float)
        data["sin_theta_np"] = sin_th
        if cache is not None:
            cache.sin_theta_np = sin_th
    vf = data.get("vf_np")
    if vf is None:
        vf = np.array(data["vf"], dtype=float)
        data["vf_np"] = vf
        if cache is not None:
            cache.vf_np = vf

    edge_src = data.get("edge_src")
    edge_dst = data.get("edge_dst")
    if edge_src is None or edge_dst is None:
        edge_src, edge_dst = _build_edge_index_arrays(G, nodes, data["idx"], np)
        data["edge_src"] = edge_src
        data["edge_dst"] = edge_dst
        if cache is not None:
            cache.edge_src = edge_src
            cache.edge_dst = edge_dst

    count.fill(0.0)
    if edge_src.size:
        np.add.at(count, edge_src, 1.0)

    component_sources = [cos_th, sin_th, epi, vf]
    deg_column = None
    deg_array = None
    if deg_sum is not None:
        deg_sum.fill(0.0)
        deg_array = _resolve_numpy_degree_array(data, count, cache=cache, np=np)
        if deg_array is not None:
            deg_column = len(component_sources)
            component_sources.append(deg_array)

    stacked = np.empty((len(nodes), len(component_sources)), dtype=float)
    for col, src_vec in enumerate(component_sources):
        np.copyto(stacked[:, col], src_vec, casting="unsafe")

    accum = np.zeros_like(stacked)
    if edge_src.size:
        edge_values = np.empty((edge_src.size, len(component_sources)), dtype=float)
        np.copyto(edge_values, stacked[edge_dst], casting="unsafe")
        np.add.at(accum, edge_src, edge_values)

    np.copyto(x, accum[:, 0], casting="unsafe")
    np.copyto(y, accum[:, 1], casting="unsafe")
    np.copyto(epi_sum, accum[:, 2], casting="unsafe")
    np.copyto(vf_sum, accum[:, 3], casting="unsafe")
    degs = None
    if deg_column is not None and deg_sum is not None:
        np.copyto(deg_sum, accum[:, deg_column], casting="unsafe")
        degs = deg_array

    return x, y, epi_sum, vf_sum, count, deg_sum, degs


def test_sparse_graph_prefers_edge_accumulation_and_matches_dnfr(monkeypatch):
    np = pytest.importorskip("numpy")
    del np  # only needed to guarantee NumPy availability for the heuristic

    template = _build_weighted_graph(nx.path_graph, 96, 0.35)

    G_vector = template.copy()
    data = _prepare_dnfr_data(G_vector)
    assert data["prefer_sparse"] is True
    assert data["A"] is None
    assert data["edge_count"] == G_vector.number_of_edges()
    assert _prefer_sparse_accumulation(len(data["nodes"]), data["edge_count"])

    # Vector ΔNFR must match the fallback computation while using cached edges.
    _compute_dnfr(G_vector, data)
    dnfr_vector = collect_attr(G_vector, G_vector.nodes, ALIAS_DNFR, 0.0)

    with numpy_disabled(monkeypatch):
        G_fallback = template.copy()
        default_compute_delta_nfr(G_fallback)
    dnfr_fallback = collect_attr(G_fallback, G_fallback.nodes, ALIAS_DNFR, 0.0)
    assert dnfr_vector == pytest.approx(dnfr_fallback)

    cache = data["cache"]
    assert cache is not None
    assert cache.edge_src is not None and cache.edge_dst is not None

    loops = 6
    sparse_data = data.copy()
    sparse_data["prefer_sparse"] = True
    sparse_data["A"] = None
    edge_buffer_ids = set()
    accum_ids = set()
    for _ in range(loops):
        _build_neighbor_sums_common(G_vector, sparse_data, use_numpy=True)
        edge_buffer = cache.neighbor_edge_values_np
        if edge_buffer is not None:
            edge_buffer_ids.add(id(edge_buffer))
        accum = cache.neighbor_accum_np
        if accum is not None:
            accum_ids.add(id(accum))
    assert len(edge_buffer_ids) <= 1
    assert len(accum_ids) <= 1
    assert cache.deg_array is not None

    loop_data = data.copy()
    x_id = id(cache.neighbor_x)
    y_id = id(cache.neighbor_y)
    epi_id = id(cache.neighbor_epi_sum)
    vf_id = id(cache.neighbor_vf_sum)
    count_id = id(cache.neighbor_count)
    deg_buffer_id = (
        id(cache.neighbor_deg_sum) if cache.neighbor_deg_sum is not None else None
    )
    with numpy_disabled(monkeypatch):
        for _ in range(loops):
            _build_neighbor_sums_common(G_vector, loop_data, use_numpy=False)
        assert id(cache.neighbor_x) == x_id
        assert id(cache.neighbor_y) == y_id
        assert id(cache.neighbor_epi_sum) == epi_id
        assert id(cache.neighbor_vf_sum) == vf_id
        assert id(cache.neighbor_count) == count_id
        if deg_buffer_id is not None:
            assert id(cache.neighbor_deg_sum) == deg_buffer_id


@pytest.mark.parametrize("factory", [nx.path_graph, nx.complete_graph])
@pytest.mark.parametrize("topo_weight", [0.0, 0.35])
def test_edge_accumulation_neighbor_sums_match_loop(factory, topo_weight, monkeypatch):
    np = pytest.importorskip("numpy")

    base = _build_weighted_graph(factory, 12, topo_weight)
    G_vec = base.copy()
    G_loop = base.copy()

    data_vec = _prepare_dnfr_data(G_vec)
    vec = _build_neighbor_sums_common(G_vec, data_vec, use_numpy=True)

    with numpy_disabled(monkeypatch):
        data_loop = _prepare_dnfr_data(G_loop)
        loop = _build_neighbor_sums_common(G_loop, data_loop, use_numpy=False)

    assert vec is not None and loop is not None
    for a, b in zip(vec[:-1], loop[:-1]):
        if a is None or b is None:
            assert a is b is None
        else:
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)

    # ``degs`` output can be numpy array or list depending on branch.
    vec_degs = vec[-1]
    loop_degs = loop[-1]
    if vec_degs is None or loop_degs is None:
        assert vec_degs is loop_degs is None
    else:
        np.testing.assert_allclose(vec_degs, loop_degs, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("factory", [nx.path_graph, nx.complete_graph])
@pytest.mark.parametrize("topo_weight", [0.0, 0.45])
def test_edge_accumulation_matches_legacy_stack(factory, topo_weight):
    np = pytest.importorskip("numpy")

    base = _build_weighted_graph(factory, 20, topo_weight)

    G_current = base.copy()
    data_current = _prepare_dnfr_data(G_current)
    current_buffers = _init_neighbor_sums(data_current, np=np)
    vec = _accumulate_neighbors_numpy(
        G_current,
        data_current,
        x=current_buffers[0],
        y=current_buffers[1],
        epi_sum=current_buffers[2],
        vf_sum=current_buffers[3],
        count=current_buffers[4],
        deg_sum=current_buffers[5],
        np=np,
    )

    G_legacy = base.copy()
    data_legacy = _prepare_dnfr_data(G_legacy)
    legacy_buffers = _init_neighbor_sums(data_legacy, np=np)
    legacy = _legacy_numpy_stack_accumulation(
        G_legacy,
        data_legacy,
        x=legacy_buffers[0],
        y=legacy_buffers[1],
        epi_sum=legacy_buffers[2],
        vf_sum=legacy_buffers[3],
        count=legacy_buffers[4],
        deg_sum=legacy_buffers[5],
        np=np,
    )

    for new_arr, old_arr in zip(vec[:-1], legacy[:-1]):
        if new_arr is None or old_arr is None:
            assert new_arr is old_arr is None
        else:
            np.testing.assert_allclose(new_arr, old_arr, rtol=1e-9, atol=1e-9)

    vec_degs = vec[-1]
    legacy_degs = legacy[-1]
    if vec_degs is None or legacy_degs is None:
        assert vec_degs is legacy_degs is None
    else:
        np.testing.assert_allclose(vec_degs, legacy_degs, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("topo_weight", [0.0, 0.4])
def test_edge_accumulation_buffers_cached_and_stable(topo_weight, monkeypatch):
    np = pytest.importorskip("numpy")

    base = _build_weighted_graph(nx.path_graph, 24, topo_weight)

    G_vec = base.copy()
    data_vec = _prepare_dnfr_data(G_vec)
    buffers = _init_neighbor_sums(data_vec, np=np)
    result = _accumulate_neighbors_numpy(
        G_vec,
        data_vec,
        x=buffers[0],
        y=buffers[1],
        epi_sum=buffers[2],
        vf_sum=buffers[3],
        count=buffers[4],
        deg_sum=buffers[5],
        np=np,
    )

    cache = data_vec.get("cache")
    assert cache is not None
    edge_values = cache.neighbor_edge_values_np
    accumulator = cache.neighbor_accum_np
    assert edge_values is not None
    assert accumulator is not None
    chunk_size = data_vec.get("neighbor_chunk_size") or data_vec["edge_count"]
    assert edge_values.shape == (chunk_size, accumulator.shape[0])

    with numpy_disabled(monkeypatch):
        loop_graph = base.copy()
        loop_data = _prepare_dnfr_data(loop_graph)
        expected_loop = _build_neighbor_sums_common(
            loop_graph,
            loop_data,
            use_numpy=False,
        )

    vector_outputs = result[:-1]
    loop_outputs = expected_loop[:-1]
    for vec_arr, loop_arr in zip(vector_outputs, loop_outputs):
        if vec_arr is None or loop_arr is None:
            assert vec_arr is loop_arr is None
        else:
            np.testing.assert_allclose(
                vec_arr,
                np.asarray(loop_arr, dtype=float),
                rtol=1e-9,
                atol=1e-9,
            )

    vec_degs = result[-1]
    loop_degs = expected_loop[-1]
    if vec_degs is None or loop_degs is None:
        assert vec_degs is loop_degs is None
    else:
        np.testing.assert_allclose(
            vec_degs,
            np.asarray(loop_degs, dtype=float),
            rtol=1e-9,
            atol=1e-9,
        )

    snapshots = [arr.copy() if arr is not None else None for arr in vector_outputs]
    deg_snapshot = result[5].copy() if result[5] is not None else None

    for arr in buffers:
        if arr is not None:
            arr.fill(-1.0)

    result_second = _accumulate_neighbors_numpy(
        G_vec,
        data_vec,
        x=buffers[0],
        y=buffers[1],
        epi_sum=buffers[2],
        vf_sum=buffers[3],
        count=buffers[4],
        deg_sum=buffers[5],
        np=np,
    )

    assert cache.neighbor_edge_values_np is edge_values
    assert cache.neighbor_accum_np is accumulator

    for arr, snapshot in zip(result_second[:-1], snapshots):
        if arr is None or snapshot is None:
            assert arr is snapshot is None
        else:
            np.testing.assert_allclose(arr, snapshot, rtol=1e-9, atol=1e-9)

    second_deg = result_second[5]
    if deg_snapshot is None or second_deg is None:
        assert deg_snapshot is second_deg is None
    else:
        np.testing.assert_allclose(
            second_deg,
            deg_snapshot,
            rtol=1e-9,
            atol=1e-9,
        )

    first_degs = result[-1]
    second_degs = result_second[-1]
    if first_degs is None or second_degs is None:
        assert first_degs is second_degs is None
    else:
        np.testing.assert_allclose(
            second_degs,
            first_degs,
            rtol=1e-9,
            atol=1e-9,
        )


def test_dense_graph_uses_dense_accumulation_by_default(monkeypatch):
    np = pytest.importorskip("numpy")
    del np

    G_dense = _build_weighted_graph(nx.complete_graph, 32, 0.2)
    data = _prepare_dnfr_data(G_dense)
    assert data["prefer_sparse"] is False
    assert data["A"] is not None
    assert not _prefer_sparse_accumulation(len(data["nodes"]), data["edge_count"])

    # Dense computation still matches the fallback path.
    _compute_dnfr(G_dense, data)
    dnfr_dense = collect_attr(G_dense, G_dense.nodes, ALIAS_DNFR, 0.0)

    with numpy_disabled(monkeypatch):
        G_fallback = G_dense.copy()
        default_compute_delta_nfr(G_fallback)
    dnfr_fallback = collect_attr(G_fallback, G_fallback.nodes, ALIAS_DNFR, 0.0)
    assert dnfr_dense == pytest.approx(dnfr_fallback)


def test_dense_graph_dnfr_modes_stable(monkeypatch):
    template = _build_dense_graph_regression()
    expected = _manual_dense_dnfr_expected(template)

    with numpy_disabled(monkeypatch):
        G_fallback = template.copy()
        default_compute_delta_nfr(G_fallback)
    fallback_dnfr = collect_attr(G_fallback, G_fallback.nodes, ALIAS_DNFR, 0.0)
    assert fallback_dnfr == pytest.approx(expected)

    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        return

    G_vectorized = template.copy()
    default_compute_delta_nfr(G_vectorized)
    vector_dnfr = collect_attr(G_vectorized, G_vectorized.nodes, ALIAS_DNFR, 0.0)
    assert vector_dnfr == pytest.approx(expected)
    assert vector_dnfr == pytest.approx(fallback_dnfr)


def test_sparse_graph_can_force_dense_mode(monkeypatch):
    np = pytest.importorskip("numpy")
    del np

    G_sparse = _build_weighted_graph(nx.path_graph, 16, 0.25)
    G_sparse.graph["dnfr_force_dense"] = True

    data = _prepare_dnfr_data(G_sparse)
    assert data["dense_override"] is True
    assert data["prefer_sparse"] is False
    assert data["A"] is not None

    _compute_dnfr(G_sparse, data)

    with numpy_disabled(monkeypatch):
        fallback = G_sparse.copy()
        default_compute_delta_nfr(fallback)

    dnfr_dense = collect_attr(G_sparse, G_sparse.nodes, ALIAS_DNFR, 0.0)
    dnfr_fallback = collect_attr(fallback, fallback.nodes, ALIAS_DNFR, 0.0)
    assert dnfr_dense == pytest.approx(dnfr_fallback)


@pytest.mark.parametrize("topo_weight", [0.0, 0.45])
def test_dense_adjacency_accumulation_matches_loop(topo_weight, monkeypatch):
    np = pytest.importorskip("numpy")

    base = _build_weighted_graph(nx.complete_graph, 10, topo_weight)
    data_dense = _prepare_dnfr_data(base)
    nodes = data_dense["nodes"]
    data_dense["A"] = nx.to_numpy_array(base, nodelist=nodes, dtype=float)
    data_dense["prefer_sparse"] = False

    dense_neighbor_sums = _build_neighbor_sums_common(base, data_dense, use_numpy=True)
    assert dense_neighbor_sums is not None

    cache = data_dense["cache"]
    assert cache is not None
    degree_vector = data_dense.get("dense_degree_np")
    assert degree_vector is not None
    assert cache.dense_degree_np is degree_vector

    dense_dnfr_graph = base.copy()
    dense_data = data_dense.copy()
    dense_data["cache"] = cache
    _compute_dnfr(dense_dnfr_graph, dense_data)
    dnfr_dense = collect_attr(dense_dnfr_graph, dense_dnfr_graph.nodes, ALIAS_DNFR, 0.0)

    with numpy_disabled(monkeypatch):
        fallback_graph = base.copy()
        default_compute_delta_nfr(fallback_graph)
    dnfr_fallback = collect_attr(fallback_graph, fallback_graph.nodes, ALIAS_DNFR, 0.0)

    np.testing.assert_allclose(dnfr_dense, dnfr_fallback, rtol=1e-9, atol=1e-9)

    # Ensure repeated dense accumulation reuses cached degree buffers.
    repeated = _build_neighbor_sums_common(base, data_dense, use_numpy=True)
    assert repeated is not None
    assert data_dense.get("dense_degree_np") is degree_vector
    assert cache.dense_degree_np is degree_vector


def test_broadcast_accumulation_dense_graph_equivalence():
    np = pytest.importorskip("numpy")

    dense_graph = _build_weighted_graph(nx.complete_graph, 36, 0.4)
    data_vec = _prepare_dnfr_data(dense_graph)
    data_vec["prefer_sparse"] = True
    data_vec["A"] = None
    _compute_dnfr(dense_graph, data_vec)

    vector_dnfr = collect_attr(dense_graph, dense_graph.nodes, ALIAS_DNFR, 0.0)

    loop_graph = dense_graph.copy()
    loop_graph.graph["vectorized_dnfr"] = False
    loop_data = _prepare_dnfr_data(loop_graph)
    _compute_dnfr(loop_graph, loop_data)
    loop_dnfr = collect_attr(loop_graph, loop_graph.nodes, ALIAS_DNFR, 0.0)

    np.testing.assert_allclose(vector_dnfr, loop_dnfr, rtol=1e-9, atol=1e-9)

    cache = data_vec.get("cache")
    assert cache is not None
    edge_buffer = cache.neighbor_edge_values_np
    signature = cache.neighbor_accum_signature
    assert isinstance(edge_buffer, np.ndarray)
    accumulator = cache.neighbor_accum_np
    assert isinstance(accumulator, np.ndarray)
    expected_rows = 4 + 1  # cos, sin, epi, vf, count
    if data_vec["w_topo"] != 0.0:
        expected_rows += 1
    chunk_size = data_vec.get("neighbor_chunk_size") or data_vec["edge_count"]
    assert edge_buffer.shape == (chunk_size, expected_rows)
    assert accumulator.shape == (expected_rows, len(data_vec["nodes"]))

    for idx, node in enumerate(dense_graph.nodes):
        set_attr(dense_graph.nodes[node], ALIAS_EPI, 0.17 * (idx + 5))
        set_attr(dense_graph.nodes[node], ALIAS_VF, 0.11 * (idx + 7))

    data_vec = _prepare_dnfr_data(dense_graph)
    data_vec["prefer_sparse"] = True
    data_vec["A"] = None
    _compute_dnfr(dense_graph, data_vec)

    assert id(cache.neighbor_edge_values_np) == id(edge_buffer)
    assert cache.neighbor_accum_signature == signature
    assert id(cache.neighbor_accum_np) == id(accumulator)

    loop_after = dense_graph.copy()
    loop_after.graph["vectorized_dnfr"] = False
    loop_after_data = _prepare_dnfr_data(loop_after)
    _compute_dnfr(loop_after, loop_after_data)
    updated_vector = collect_attr(dense_graph, dense_graph.nodes, ALIAS_DNFR, 0.0)
    updated_loop = collect_attr(loop_after, loop_after.nodes, ALIAS_DNFR, 0.0)

    np.testing.assert_allclose(updated_vector, updated_loop, rtol=1e-9, atol=1e-9)


def test_sparse_bincount_accumulation_matches_manual():
    np = pytest.importorskip("numpy")

    base = _build_weighted_graph(nx.path_graph, 14, 0.25)
    data_vec = _prepare_dnfr_data(base)
    data_vec["prefer_sparse"] = True
    data_vec["A"] = None

    sums = _build_neighbor_sums_common(base, data_vec, use_numpy=True)
    x_vec, y_vec, epi_vec, vf_vec, count_vec, deg_vec, degs = sums

    nodes = data_vec["nodes"]
    idx = data_vec["idx"]
    cos = data_vec["cos_theta"]
    sin = data_vec["sin_theta"]
    epi = data_vec["epi"]
    vf = data_vec["vf"]
    deg_list = data_vec.get("deg_list")

    size = len(nodes)
    expected_x = np.zeros(size, dtype=float)
    expected_y = np.zeros(size, dtype=float)
    expected_epi = np.zeros(size, dtype=float)
    expected_vf = np.zeros(size, dtype=float)
    expected_count = np.zeros(size, dtype=float)
    expected_deg = np.zeros(size, dtype=float)

    for i, node in enumerate(nodes):
        deg_i = degs[i] if degs is not None else 0.0
        for neighbor in base.neighbors(node):
            j = idx[neighbor]
            expected_x[i] += cos[j]
            expected_y[i] += sin[j]
            expected_epi[i] += epi[j]
            expected_vf[i] += vf[j]
            if count_vec is not None:
                expected_count[i] += 1.0
            if deg_vec is not None:
                expected_deg[i] += deg_list[j] if deg_list is not None else deg_i

    np.testing.assert_allclose(np.asarray(x_vec, dtype=float), expected_x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(y_vec, dtype=float), expected_y, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(epi_vec, dtype=float), expected_epi, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(vf_vec, dtype=float), expected_vf, rtol=1e-12, atol=1e-12
    )

    if count_vec is not None:
        np.testing.assert_allclose(
            np.asarray(count_vec, dtype=float), expected_count, rtol=1e-12, atol=1e-12
        )
    if deg_vec is not None:
        np.testing.assert_allclose(
            np.asarray(deg_vec, dtype=float), expected_deg, rtol=1e-12, atol=1e-12
        )


def test_broadcast_accumulation_invalidation_on_edge_change():
    np = pytest.importorskip("numpy")

    base = _build_weighted_graph(nx.path_graph, 24, 0.25)
    data_vec = _prepare_dnfr_data(base)
    _compute_dnfr(base, data_vec)

    cache = data_vec.get("cache")
    assert cache is not None
    old_signature = cache.neighbor_accum_signature
    old_edge_shape = cache.neighbor_edge_values_np.shape
    old_accum = cache.neighbor_accum_np
    assert isinstance(old_accum, np.ndarray)

    base.add_edge(0, len(base) - 1)
    mark_dnfr_prep_dirty(base)

    refreshed = _prepare_dnfr_data(base)
    refreshed["prefer_sparse"] = True
    refreshed["A"] = None
    _compute_dnfr(base, refreshed)

    new_signature = cache.neighbor_accum_signature
    assert new_signature != old_signature
    assert cache.neighbor_edge_values_np.shape != old_edge_shape
    assert cache.neighbor_accum_np is not old_accum

    loop_graph = base.copy()
    loop_graph.graph["vectorized_dnfr"] = False
    loop_data = _prepare_dnfr_data(loop_graph)
    _compute_dnfr(loop_graph, loop_data)

    vector_dnfr = collect_attr(base, base.nodes, ALIAS_DNFR, 0.0)
    loop_dnfr = collect_attr(loop_graph, loop_graph.nodes, ALIAS_DNFR, 0.0)

    np.testing.assert_allclose(vector_dnfr, loop_dnfr, rtol=1e-9, atol=1e-9)


def _build_large_random_graph(np_module, *, nodes=480, edges=9600, seed=20240517):
    """Create a dense-ish random graph with seeded TNFR attributes."""

    G = nx.gnm_random_graph(nodes, edges, seed=seed)
    rng = np_module.random.default_rng(seed + 1337)
    for node in G.nodes:
        angle = float(rng.uniform(-math.pi, math.pi))
        epi = float(rng.normal(0.0, 1.0))
        vf = float(rng.uniform(-0.5, 0.5))
        set_attr(G.nodes[node], ALIAS_THETA, angle)
        set_attr(G.nodes[node], ALIAS_EPI, epi)
        set_attr(G.nodes[node], ALIAS_VF, vf)
    G.graph["DNFR_WEIGHTS"] = {
        "phase": 0.3,
        "epi": 0.25,
        "vf": 0.25,
        "topo": 0.2,
    }
    return G


def test_vectorized_matches_python_and_is_faster_large_graph(monkeypatch):
    np = pytest.importorskip("numpy")

    def _timed_run(graph, *, disable_numpy: bool) -> float:
        if disable_numpy:
            with numpy_disabled(monkeypatch):
                start = time.perf_counter()
                default_compute_delta_nfr(graph)
                return time.perf_counter() - start
        start = time.perf_counter()
        default_compute_delta_nfr(graph)
        return time.perf_counter() - start

    base_vector = _build_large_random_graph(np, seed=20240517)
    base_python = base_vector.copy()

    vector_time = _timed_run(base_vector, disable_numpy=False)
    python_time = _timed_run(base_python, disable_numpy=True)

    vec_dnfr = np.array(collect_attr(base_vector, base_vector.nodes, ALIAS_DNFR, 0.0))
    py_dnfr = np.array(collect_attr(base_python, base_python.nodes, ALIAS_DNFR, 0.0))
    np.testing.assert_allclose(vec_dnfr, py_dnfr, rtol=1e-10, atol=1e-10)

    timings_vector = [vector_time]
    timings_python = [python_time]

    for seed in (20240518, 20240519):
        sample_vector = _build_large_random_graph(np, seed=seed)
        sample_python = sample_vector.copy()
        timings_vector.append(_timed_run(sample_vector, disable_numpy=False))
        timings_python.append(_timed_run(sample_python, disable_numpy=True))

    assert min(timings_python) > 0.0
    assert min(timings_vector) < min(timings_python)
