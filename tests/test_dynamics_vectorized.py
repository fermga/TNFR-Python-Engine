"""Pruebas de dynamics vectorized."""

import math
import time

import pytest
import networkx as nx

from tnfr.dynamics.dnfr import (
    _accumulate_neighbors_numpy,
    _build_edge_index_arrays,
    _build_neighbor_sums_common,
    _compute_dnfr,
    _init_neighbor_sums,
    _prepare_dnfr_data,
    _prefer_sparse_accumulation,
    _resolve_numpy_degree_array,
)

from tnfr.dynamics import default_compute_delta_nfr
from tnfr.constants import get_aliases
from tnfr.alias import collect_attr, get_attr, set_attr
from tnfr.helpers.numeric import angle_diff

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")


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


@pytest.mark.parametrize("vectorized", [False, True])
def test_default_compute_delta_nfr_paths(vectorized):
    if vectorized:
        pytest.importorskip("numpy")
    G = _setup_graph()
    G.graph["vectorized_dnfr"] = vectorized
    default_compute_delta_nfr(G)
    dnfr = collect_attr(G, G.nodes, ALIAS_DNFR, 0.0)
    assert len(dnfr) == 5


def test_default_vectorization_auto_enabled_when_numpy_available():
    np = pytest.importorskip("numpy")
    G = _setup_graph()
    default_compute_delta_nfr(G)
    cache = G.graph.get("_dnfr_prep_cache")
    assert cache is not None
    assert cache.theta_np is not None
    assert cache.edge_src is not None and isinstance(cache.edge_src, np.ndarray)
    assert isinstance(cache.grad_total_np, np.ndarray)
    assert isinstance(cache.grad_phase_np, np.ndarray)


def test_vectorization_can_be_disabled_explicitamente():
    pytest.importorskip("numpy")
    G = _setup_graph()
    # Para desactivar la ruta vectorizada basta con fijar este marcador a ``False``.
    G.graph["vectorized_dnfr"] = False
    default_compute_delta_nfr(G)
    cache = G.graph.get("_dnfr_prep_cache")
    assert cache is not None
    assert cache.theta_np is None
    assert cache.edge_src is None
    assert cache.grad_total_np is None
    assert cache.grad_phase_np is None


def test_vectorized_gradients_cached_and_reused():
    np = pytest.importorskip("numpy")
    G = _build_weighted_graph(nx.path_graph, 4, 0.3)
    default_compute_delta_nfr(G)
    cache = G.graph.get("_dnfr_prep_cache")
    assert isinstance(cache.grad_total_np, np.ndarray)
    assert isinstance(cache.grad_phase_np, np.ndarray)
    before = cache.grad_total_np.copy()

    # Ejecutar de nuevo para comprobar que los buffers se reutilizan
    default_compute_delta_nfr(G)
    cache2 = G.graph.get("_dnfr_prep_cache")
    assert cache2 is cache
    assert cache2.grad_total_np is cache.grad_total_np
    assert cache2.grad_phase_np is cache.grad_phase_np
    after = cache2.grad_total_np
    assert after.shape == before.shape


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
def test_vectorized_matches_reference(factory, topo_weight):
    np = pytest.importorskip("numpy")
    del np  # only needed to guarantee NumPy availability

    G_list = _build_weighted_graph(factory, 6, topo_weight)
    G_list.graph["vectorized_dnfr"] = False
    G_vec = _build_weighted_graph(factory, 6, topo_weight)

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
        edge_src, edge_dst = _build_edge_index_arrays(
            G, nodes, data["idx"], np
        )
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
        deg_array = _resolve_numpy_degree_array(
            data, count, cache=cache, np=np
        )
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


def test_sparse_graph_prefers_edge_accumulation_and_matches_dnfr():
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
    _compute_dnfr(G_vector, data, use_numpy=True)
    dnfr_vector = collect_attr(G_vector, G_vector.nodes, ALIAS_DNFR, 0.0)

    G_fallback = template.copy()
    G_fallback.graph["vectorized_dnfr"] = False
    default_compute_delta_nfr(G_fallback)
    dnfr_fallback = collect_attr(G_fallback, G_fallback.nodes, ALIAS_DNFR, 0.0)
    assert dnfr_vector == pytest.approx(dnfr_fallback)

    cache = data["cache"]
    assert cache is not None
    assert cache.edge_src is not None and cache.edge_dst is not None

    # Compare runtimes between vectorised edge accumulation and the fallback loop.
    loops = 10
    sparse_data = data.copy()
    sparse_data["prefer_sparse"] = True
    sparse_data["A"] = None
    start = time.perf_counter()
    for _ in range(loops):
        _build_neighbor_sums_common(G_vector, sparse_data, use_numpy=True)
    sparse_time = time.perf_counter() - start

    loop_data = data.copy()
    start = time.perf_counter()
    for _ in range(loops):
        _build_neighbor_sums_common(G_vector, loop_data, use_numpy=False)
    loop_time = time.perf_counter() - start

    assert sparse_time < loop_time


@pytest.mark.parametrize("factory", [nx.path_graph, nx.complete_graph])
@pytest.mark.parametrize("topo_weight", [0.0, 0.35])
def test_edge_accumulation_neighbor_sums_match_loop(factory, topo_weight):
    np = pytest.importorskip("numpy")

    base = _build_weighted_graph(factory, 12, topo_weight)
    G_vec = base.copy()
    G_loop = base.copy()

    data_vec = _prepare_dnfr_data(G_vec)
    vec = _build_neighbor_sums_common(G_vec, data_vec, use_numpy=True)

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
def test_edge_accumulation_workspace_cached_and_stable(topo_weight):
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
    workspace = cache.neighbor_workspace_np
    contrib = cache.neighbor_contrib_np
    assert workspace is not None and contrib is not None

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

    snapshots = [
        arr.copy() if arr is not None else None for arr in vector_outputs
    ]
    deg_snapshot = (
        result[5].copy() if result[5] is not None else None
    )

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

    assert cache.neighbor_workspace_np is workspace
    assert cache.neighbor_contrib_np is contrib

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

def test_dense_graph_uses_dense_accumulation_by_default():
    np = pytest.importorskip("numpy")
    del np

    G_dense = _build_weighted_graph(nx.complete_graph, 32, 0.2)
    data = _prepare_dnfr_data(G_dense)
    assert data["prefer_sparse"] is False
    assert data["A"] is not None
    assert not _prefer_sparse_accumulation(len(data["nodes"]), data["edge_count"])

    # Dense computation still matches the fallback path.
    _compute_dnfr(G_dense, data, use_numpy=True)
    dnfr_dense = collect_attr(G_dense, G_dense.nodes, ALIAS_DNFR, 0.0)

    G_fallback = G_dense.copy()
    G_fallback.graph["vectorized_dnfr"] = False
    default_compute_delta_nfr(G_fallback)
    dnfr_fallback = collect_attr(G_fallback, G_fallback.nodes, ALIAS_DNFR, 0.0)
    assert dnfr_dense == pytest.approx(dnfr_fallback)


def test_dense_graph_dnfr_modes_stable():
    template = _build_dense_graph_regression()
    expected = _manual_dense_dnfr_expected(template)

    G_fallback = template.copy()
    G_fallback.graph["vectorized_dnfr"] = False
    default_compute_delta_nfr(G_fallback)
    fallback_dnfr = collect_attr(G_fallback, G_fallback.nodes, ALIAS_DNFR, 0.0)
    assert fallback_dnfr == pytest.approx(expected)

    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        return

    G_vectorized = template.copy()
    default_compute_delta_nfr(G_vectorized)
    vector_dnfr = collect_attr(
        G_vectorized, G_vectorized.nodes, ALIAS_DNFR, 0.0
    )
    assert vector_dnfr == pytest.approx(expected)
    assert vector_dnfr == pytest.approx(fallback_dnfr)


def test_sparse_graph_can_force_dense_mode():
    np = pytest.importorskip("numpy")
    del np

    G_sparse = _build_weighted_graph(nx.path_graph, 16, 0.25)
    G_sparse.graph["dnfr_force_dense"] = True

    data = _prepare_dnfr_data(G_sparse)
    assert data["dense_override"] is True
    assert data["prefer_sparse"] is False
    assert data["A"] is not None

    _compute_dnfr(G_sparse, data, use_numpy=True)

    fallback = G_sparse.copy()
    fallback.graph["vectorized_dnfr"] = False
    default_compute_delta_nfr(fallback)

    dnfr_dense = collect_attr(G_sparse, G_sparse.nodes, ALIAS_DNFR, 0.0)
    dnfr_fallback = collect_attr(fallback, fallback.nodes, ALIAS_DNFR, 0.0)
    assert dnfr_dense == pytest.approx(dnfr_fallback)


@pytest.mark.parametrize("topo_weight", [0.0, 0.45])
def test_dense_adjacency_accumulation_matches_loop(topo_weight):
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
    _compute_dnfr(dense_dnfr_graph, dense_data, use_numpy=True)
    dnfr_dense = collect_attr(dense_dnfr_graph, dense_dnfr_graph.nodes, ALIAS_DNFR, 0.0)

    fallback_graph = base.copy()
    fallback_graph.graph["vectorized_dnfr"] = False
    default_compute_delta_nfr(fallback_graph)
    dnfr_fallback = collect_attr(
        fallback_graph, fallback_graph.nodes, ALIAS_DNFR, 0.0
    )

    np.testing.assert_allclose(dnfr_dense, dnfr_fallback, rtol=1e-9, atol=1e-9)

    # Ensure repeated dense accumulation reuses cached degree buffers.
    repeated = _build_neighbor_sums_common(base, data_dense, use_numpy=True)
    assert repeated is not None
    assert data_dense.get("dense_degree_np") is degree_vector
    assert cache.dense_degree_np is degree_vector
