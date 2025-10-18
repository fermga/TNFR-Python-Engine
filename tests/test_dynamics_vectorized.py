"""Pruebas de dynamics vectorized."""

import math
import time

import pytest
import networkx as nx

from tnfr.dynamics.dnfr import (
    _build_neighbor_sums_common,
    _compute_dnfr,
    _prepare_dnfr_data,
    _prefer_sparse_accumulation,
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

    assert sparse_time * 2 < loop_time


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


def test_dense_graph_prefers_edge_accumulation():
    np = pytest.importorskip("numpy")
    del np

    G_dense = _build_weighted_graph(nx.complete_graph, 32, 0.2)
    data = _prepare_dnfr_data(G_dense)
    assert data["prefer_sparse"] is True
    assert data["A"] is None
    assert _prefer_sparse_accumulation(len(data["nodes"]), data["edge_count"])

    # Edge-based computation still matches the fallback path.
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
