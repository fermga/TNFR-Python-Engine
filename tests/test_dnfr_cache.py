"""Pruebas de dnfr cache."""

import math

import pytest
import networkx as nx

import tnfr.utils.init as utils_init

from tnfr.dynamics import default_compute_delta_nfr
from tnfr.constants import (
    THETA_PRIMARY,
    EPI_PRIMARY,
    VF_PRIMARY,
    DNFR_PRIMARY,
)
from tnfr.utils import (
    cached_node_list,
    cached_nodes_and_A,
    increment_edge_version,
)


def _counting_trig(monkeypatch):
    import math

    cos_calls = {"n": 0}
    sin_calls = {"n": 0}
    orig_cos = math.cos
    orig_sin = math.sin

    def cos_wrapper(x):
        cos_calls["n"] += 1
        return orig_cos(x)

    def sin_wrapper(x):
        sin_calls["n"] += 1
        return orig_sin(x)

    monkeypatch.setattr(math, "cos", cos_wrapper)
    monkeypatch.setattr(math, "sin", sin_wrapper)
    return cos_calls, sin_calls


def _setup_graph():
    G = nx.path_graph(3)
    for n in G.nodes:
        G.nodes[n][THETA_PRIMARY] = 0.1 * (n + 1)
        G.nodes[n][EPI_PRIMARY] = 0.2 * (n + 1)
        G.nodes[n][VF_PRIMARY] = 0.3 * (n + 1)
    G.graph["DNFR_WEIGHTS"] = {
        "phase": 1.0,
        "epi": 0.0,
        "vf": 0.0,
        "topo": 0.0,
    }
    return G


def _collect_dnfr(G):
    return [float(G.nodes[n].get(DNFR_PRIMARY, 0.0)) for n in G.nodes]


@pytest.mark.parametrize("vectorized", [False, True])
def test_cache_invalidated_on_graph_change(vectorized):
    if vectorized:
        pytest.importorskip("numpy")

    G = _setup_graph()
    G.graph["vectorized_dnfr"] = vectorized
    default_compute_delta_nfr(G, cache_size=2)
    nodes1, _ = cached_nodes_and_A(G, cache_size=2)

    G.add_edge(2, 3)  # Cambia número de nodos y aristas
    for attr, scale in ((THETA_PRIMARY, 0.1), (EPI_PRIMARY, 0.2), (VF_PRIMARY, 0.3)):
        G.nodes[3][attr] = scale * 4
    increment_edge_version(G)
    G.graph["vectorized_dnfr"] = vectorized
    default_compute_delta_nfr(G, cache_size=2)
    nodes2, _ = cached_nodes_and_A(G, cache_size=2)

    assert len(nodes2) == 4
    assert nodes1 is not nodes2

    G.add_edge(3, 4)
    increment_edge_version(G)
    G.graph["vectorized_dnfr"] = vectorized
    default_compute_delta_nfr(G, cache_size=2)
    nodes3, _ = cached_nodes_and_A(G, cache_size=2)
    assert nodes3 is not nodes2


def test_cache_is_per_graph():
    G1 = _setup_graph()
    G2 = _setup_graph()
    default_compute_delta_nfr(G1)
    default_compute_delta_nfr(G2)
    nodes1, _ = cached_nodes_and_A(G1)
    nodes2, _ = cached_nodes_and_A(G2)
    assert nodes1 is not nodes2


@pytest.mark.parametrize("vectorized", [False, True])
def test_neighbor_sum_buffers_reused_and_results_stable(vectorized):
    if vectorized:
        pytest.importorskip("numpy")

    G = nx.path_graph(5)
    for idx, node in enumerate(G.nodes):
        G.nodes[node][THETA_PRIMARY] = 0.15 * (idx + 1)
        G.nodes[node][EPI_PRIMARY] = 0.05 * (idx + 2)
        G.nodes[node][VF_PRIMARY] = 0.08 * (idx + 3)
    G.graph["DNFR_WEIGHTS"] = {
        "phase": 0.4,
        "epi": 0.3,
        "vf": 0.2,
        "topo": 0.1,
    }
    G.graph["vectorized_dnfr"] = vectorized

    default_compute_delta_nfr(G)
    first = _collect_dnfr(G)
    cache = G.graph.get("_dnfr_prep_cache")
    assert cache is not None

    list_buffers = (
        cache.neighbor_x,
        cache.neighbor_y,
        cache.neighbor_epi_sum,
        cache.neighbor_vf_sum,
        cache.neighbor_count,
        cache.neighbor_deg_sum,
    )
    bar_list_buffers = (
        cache.th_bar,
        cache.epi_bar,
        cache.vf_bar,
        cache.deg_bar,
    )
    for buf in list_buffers[:-1]:
        assert isinstance(buf, list)
        assert len(buf) == len(G)
    if list_buffers[-1] is not None:
        assert len(list_buffers[-1]) == len(G)
    if vectorized:
        assert all(buf is None for buf in bar_list_buffers)
    else:
        try:
            import numpy as np
        except ModuleNotFoundError:  # pragma: no cover - optional dependency
            np = None

        np_bar_buffers = (
            cache.th_bar_np,
            cache.epi_bar_np,
            cache.vf_bar_np,
            cache.deg_bar_np,
        )

        for idx, buf in enumerate(bar_list_buffers[:-1]):
            if buf is not None:
                assert isinstance(buf, list)
                assert len(buf) == len(G)
            elif np is not None:
                arr = np_bar_buffers[idx]
                assert arr is not None
                assert arr.shape == (len(G),)
            else:
                pytest.fail("Expected list buffers when NumPy is unavailable")

        last_buf = bar_list_buffers[-1]
        if last_buf is not None:
            assert len(last_buf) == len(G)
        elif np is not None and np_bar_buffers[-1] is not None:
            assert np_bar_buffers[-1].shape == (len(G),)

    if vectorized:
        arr_buffers = (
            cache.neighbor_x_np,
            cache.neighbor_y_np,
            cache.neighbor_epi_sum_np,
            cache.neighbor_vf_sum_np,
            cache.neighbor_count_np,
            cache.neighbor_deg_sum_np,
        )
        bar_arr_buffers = (
            cache.th_bar_np,
            cache.epi_bar_np,
            cache.vf_bar_np,
            cache.deg_bar_np,
        )
        for arr in arr_buffers[:-1]:
            assert arr is not None
            assert arr.shape == (len(G),)
        if arr_buffers[-1] is not None:
            assert arr_buffers[-1].shape == (len(G),)
        for arr in bar_arr_buffers[:-1]:
            assert arr is not None
            assert arr.shape == (len(G),)
        if bar_arr_buffers[-1] is not None:
            assert bar_arr_buffers[-1].shape == (len(G),)
    else:
        assert cache.neighbor_x_np is None
        assert cache.neighbor_y_np is None
        bar_arr_buffers = (None, None, None, None)

    # Corrupt buffers to ensure they are cleaned instead of recreated.
    for buf in list_buffers:
        if buf is not None and buf:
            buf[0] = 999.0
    for buf in bar_list_buffers:
        if buf is not None and buf:
            buf[0] = 555.0
    if vectorized:
        for arr in arr_buffers:
            if arr is not None and arr.size:
                arr.fill(777.0)
        for arr in bar_arr_buffers:
            if arr is not None and arr.size:
                arr.fill(333.0)

    default_compute_delta_nfr(G)
    second = _collect_dnfr(G)

    assert second == pytest.approx(first)
    assert cache.neighbor_x is list_buffers[0]
    assert cache.neighbor_y is list_buffers[1]
    assert cache.neighbor_epi_sum is list_buffers[2]
    assert cache.neighbor_vf_sum is list_buffers[3]
    assert cache.neighbor_count is list_buffers[4]
    if list_buffers[5] is not None:
        assert cache.neighbor_deg_sum is list_buffers[5]
    if cache.th_bar is not None:
        assert cache.th_bar is bar_list_buffers[0]
        assert cache.epi_bar is bar_list_buffers[1]
        assert cache.vf_bar is bar_list_buffers[2]
        if bar_list_buffers[3] is not None:
            assert cache.deg_bar is bar_list_buffers[3]
    if vectorized:
        assert cache.neighbor_x_np is arr_buffers[0]
        assert cache.neighbor_y_np is arr_buffers[1]
        assert cache.neighbor_epi_sum_np is arr_buffers[2]
        assert cache.neighbor_vf_sum_np is arr_buffers[3]
        assert cache.neighbor_count_np is arr_buffers[4]
        if arr_buffers[5] is not None:
            assert cache.neighbor_deg_sum_np is arr_buffers[5]
        assert cache.th_bar_np is bar_arr_buffers[0]
        assert cache.epi_bar_np is bar_arr_buffers[1]
        assert cache.vf_bar_np is bar_arr_buffers[2]
        if bar_arr_buffers[3] is not None:
            assert cache.deg_bar_np is bar_arr_buffers[3]

    # Los resultados deben permanecer invariantes tras recomputar.
    for before, after in zip(first, second):
        assert math.isfinite(after)
        assert before == pytest.approx(after)


def test_cache_invalidated_on_node_rename():
    G = _setup_graph()
    nodes1 = cached_node_list(G)

    nx.relabel_nodes(G, {2: 9}, copy=False)

    nodes2 = cached_node_list(G)

    assert nodes2 is not nodes1
    assert set(nodes2) == {0, 1, 9}


def test_prepare_dnfr_data_refreshes_cached_vectors(monkeypatch):
    original_cached_import = utils_init.cached_import

    def fake_cached_import(module, attr=None, **kwargs):
        if module == "numpy":
            return None
        return original_cached_import(module, attr=attr, **kwargs)

    monkeypatch.setattr(utils_init, "cached_import", fake_cached_import)
    cos_calls, sin_calls = _counting_trig(monkeypatch)
    G = _setup_graph()
    default_compute_delta_nfr(G)

    cos_first = cos_calls["n"]
    sin_first = sin_calls["n"]

    # Subsequent call without modifications should refresh cached trig values
    default_compute_delta_nfr(G)
    assert cos_calls["n"] == cos_first + len(G)
    assert sin_calls["n"] == sin_first + len(G)


@pytest.mark.parametrize("vectorized", [False, True])
def test_default_compute_delta_nfr_updates_on_state_change(vectorized):
    if vectorized:
        pytest.importorskip("numpy")

    G = _setup_graph()
    G.graph["DNFR_WEIGHTS"] = {
        "phase": 1.0,
        "epi": 1.0,
        "vf": 1.0,
        "topo": 0.0,
    }
    G.graph["vectorized_dnfr"] = vectorized

    default_compute_delta_nfr(G, cache_size=2)
    before = {n: G.nodes[n].get(DNFR_PRIMARY, 0.0) for n in G.nodes}

    # Modify only the central node without touching topology
    target = 1
    G.nodes[target][THETA_PRIMARY] += 0.5
    G.nodes[target][EPI_PRIMARY] += 1.2
    G.nodes[target][VF_PRIMARY] -= 0.8

    default_compute_delta_nfr(G, cache_size=2)
    after = {n: G.nodes[n].get(DNFR_PRIMARY, 0.0) for n in G.nodes}

    assert not math.isclose(before[target], after[target])
    assert any(not math.isclose(before[n], after[n]) for n in G.nodes)


def test_cached_nodes_and_A_reuses_until_edge_change():
    pytest.importorskip("numpy")

    G = _setup_graph()

    nodes1, A1 = cached_nodes_and_A(G, cache_size=2)
    nodes2, A2 = cached_nodes_and_A(G, cache_size=2)

    assert nodes1 is nodes2
    assert A1 is A2

    G.add_edge(2, 3)
    for attr, scale in ((THETA_PRIMARY, 0.1), (EPI_PRIMARY, 0.2), (VF_PRIMARY, 0.3)):
        G.nodes[3][attr] = scale * 4
    increment_edge_version(G)

    nodes3, A3 = cached_nodes_and_A(G, cache_size=2)

    assert nodes3 is not nodes2
    assert A3 is not A2


def test_cached_node_list_reuses_tuple():
    G = _setup_graph()

    nodes1 = cached_node_list(G)
    nodes2 = cached_node_list(G)

    assert nodes1 is nodes2


def test_cached_node_list_invalidate_on_node_addition():
    G = _setup_graph()

    nodes1 = cached_node_list(G)
    G.add_node(99)

    nodes2 = cached_node_list(G)

    assert nodes2 is not nodes1
    assert set(nodes2) == {0, 1, 2, 99}


def test_cached_node_list_invalidate_on_node_rename():
    G = _setup_graph()

    nodes1 = cached_node_list(G)
    nx.relabel_nodes(G, {2: 9}, copy=False)

    nodes2 = cached_node_list(G)

    assert nodes2 is not nodes1
    assert set(nodes2) == {0, 1, 9}


def test_cached_nodes_and_A_returns_none_without_numpy(monkeypatch, graph_canon):
    monkeypatch.setattr(utils_init, "cached_import", lambda *a, **k: None)
    G = graph_canon()
    G.add_edge(0, 1)
    nodes, A = cached_nodes_and_A(G)
    assert A is None
    assert isinstance(nodes, tuple)
    assert nodes == (0, 1)


def test_cached_nodes_and_A_requires_numpy(monkeypatch, graph_canon):
    monkeypatch.setattr(utils_init, "cached_import", lambda *a, **k: None)
    G = graph_canon()
    G.add_edge(0, 1)
    with pytest.raises(RuntimeError):
        cached_nodes_and_A(G, require_numpy=True)
