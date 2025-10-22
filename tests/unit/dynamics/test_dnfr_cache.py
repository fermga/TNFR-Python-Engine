"""Unit tests for DNFR cache management and numerical consistency."""



import math

import pytest
from contextlib import contextmanager, nullcontext

import networkx as nx

import tnfr.utils.init as utils_init

from tnfr.dynamics import default_compute_delta_nfr
from tnfr.dynamics.dnfr import _accumulate_neighbors_numpy, _prepare_dnfr_data
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
from tnfr.utils.cache import DNFR_PREP_STATE_KEY, DnfrPrepState, _graph_cache_manager


@contextmanager
def numpy_disabled(monkeypatch):
    import tnfr.dynamics.dnfr as dnfr_module

    with monkeypatch.context() as ctx:
        ctx.setattr(dnfr_module, "get_numpy", lambda: None)
        yield


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


def _get_prep_state(G):
    manager = _graph_cache_manager(G.graph)
    state = manager.get(DNFR_PREP_STATE_KEY)
    assert isinstance(state, DnfrPrepState)
    return manager, state


def test_prepare_dnfr_data_populates_degree_cache_without_topology_weight():
    np = pytest.importorskip("numpy")

    G = _setup_graph()

    data = _prepare_dnfr_data(G)

    deg_list = data["deg_list"]
    assert isinstance(deg_list, list)
    assert deg_list == pytest.approx([1.0, 2.0, 1.0])

    deg_array = data["deg_array"]
    assert deg_array is not None
    assert getattr(deg_array, "shape", None) == (len(deg_list),)
    np.testing.assert_allclose(deg_array, np.array([1.0, 2.0, 1.0]))

    manager, state = _get_prep_state(G)
    cache = data["cache"]
    assert cache is not None
    assert cache.deg_list is deg_list
    assert cache.deg_array is deg_array
    stats = manager.get_metrics(DNFR_PREP_STATE_KEY)
    assert stats.misses == 1
    assert stats.hits == 0


def test_accumulate_neighbors_numpy_prefers_degree_cache():
    np = pytest.importorskip("numpy")

    G = _setup_graph()

    data = _prepare_dnfr_data(G)
    manager, state = _get_prep_state(G)
    cache = data["cache"]
    assert state.cache is cache

    fake_deg = np.array([10.0, 20.0, 30.0])
    data["deg_array"] = fake_deg
    if cache is not None:
        cache.deg_array = fake_deg

    n = len(data["nodes"])
    x = np.zeros(n, dtype=float)
    y = np.zeros(n, dtype=float)
    epi_sum = np.zeros(n, dtype=float)
    vf_sum = np.zeros(n, dtype=float)
    count = np.zeros(n, dtype=float)

    _accumulate_neighbors_numpy(
        G,
        data,
        x=x,
        y=y,
        epi_sum=epi_sum,
        vf_sum=vf_sum,
        count=count,
        deg_sum=None,
        np=np,
    )
    np.testing.assert_allclose(count, fake_deg)

    data["deg_array"] = None
    if cache is not None:
        cache.deg_array = None

    x2 = np.zeros(n, dtype=float)
    y2 = np.zeros(n, dtype=float)
    epi_sum2 = np.zeros(n, dtype=float)
    vf_sum2 = np.zeros(n, dtype=float)
    count2 = np.zeros(n, dtype=float)

    _accumulate_neighbors_numpy(
        G,
        data,
        x=x2,
        y=y2,
        epi_sum=epi_sum2,
        vf_sum=vf_sum2,
        count=count2,
        deg_sum=None,
        np=np,
    )
    np.testing.assert_allclose(count2, np.array([1.0, 2.0, 1.0]))
    stats = manager.get_metrics(DNFR_PREP_STATE_KEY)
    assert stats.misses == 1


def test_degree_cache_refreshes_after_graph_mutation():
    np = pytest.importorskip("numpy")

    G = _setup_graph()

    data_before = _prepare_dnfr_data(G)
    manager, state_before = _get_prep_state(G)
    cache_before = data_before["cache"]
    assert cache_before is not None
    assert state_before.cache is cache_before
    stats = manager.get_metrics(DNFR_PREP_STATE_KEY)
    assert stats.misses == 1
    assert stats.hits == 0

    deg_array_before = np.array(data_before["deg_array"], copy=True)
    np.testing.assert_allclose(deg_array_before, np.array([1.0, 2.0, 1.0]))

    n = len(data_before["nodes"])
    x = np.zeros(n, dtype=float)
    y = np.zeros(n, dtype=float)
    epi_sum = np.zeros(n, dtype=float)
    vf_sum = np.zeros(n, dtype=float)
    count = np.zeros(n, dtype=float)

    _accumulate_neighbors_numpy(
        G,
        data_before,
        x=x,
        y=y,
        epi_sum=epi_sum,
        vf_sum=vf_sum,
        count=count,
        deg_sum=None,
        np=np,
    )
    np.testing.assert_allclose(count, deg_array_before)

    G.add_edge(0, 2)
    increment_edge_version(G)

    data_after = _prepare_dnfr_data(G)
    _, state_after = _get_prep_state(G)
    cache_after = data_after["cache"]
    assert cache_after is not None
    assert cache_after is not cache_before
    assert state_after.cache is cache_after

    expected_deg = np.array([2.0, 2.0, 2.0])
    np.testing.assert_allclose(data_after["deg_array"], expected_deg)
    assert data_after["deg_list"] == pytest.approx(expected_deg.tolist())
    assert cache_after.deg_array is data_after["deg_array"]
    assert cache_after.deg_list is data_after["deg_list"]

    x_new = np.zeros(len(data_after["nodes"]), dtype=float)
    y_new = np.zeros(len(data_after["nodes"]), dtype=float)
    epi_sum_new = np.zeros(len(data_after["nodes"]), dtype=float)
    vf_sum_new = np.zeros(len(data_after["nodes"]), dtype=float)
    count_new = np.zeros(len(data_after["nodes"]), dtype=float)

    _accumulate_neighbors_numpy(
        G,
        data_after,
        x=x_new,
        y=y_new,
        epi_sum=epi_sum_new,
        vf_sum=vf_sum_new,
        count=count_new,
        deg_sum=None,
        np=np,
    )
    np.testing.assert_allclose(count_new, expected_deg)
    np.testing.assert_allclose(deg_array_before, np.array([1.0, 2.0, 1.0]))
    stats_after = manager.get_metrics(DNFR_PREP_STATE_KEY)
    assert stats_after.misses == 2
    assert stats_after.hits == 0

    data_third = _prepare_dnfr_data(G)
    cache_third = data_third["cache"]
    assert cache_third is cache_after
    stats_final = manager.get_metrics(DNFR_PREP_STATE_KEY)
    assert stats_final.misses == 2
    assert stats_final.hits == 1


@pytest.mark.parametrize("vectorized", [False, True])
def test_cache_invalidated_on_graph_change(vectorized, monkeypatch):
    if vectorized:
        pytest.importorskip("numpy")
        context_factory = nullcontext
    else:
        context_factory = lambda: numpy_disabled(monkeypatch)

    G = _setup_graph()
    with context_factory():
        default_compute_delta_nfr(G, cache_size=2)
        nodes1, _ = cached_nodes_and_A(G, cache_size=2)

        G.add_edge(2, 3)  # Changes the number of nodes and edges
        for attr, scale in ((THETA_PRIMARY, 0.1), (EPI_PRIMARY, 0.2), (VF_PRIMARY, 0.3)):
            G.nodes[3][attr] = scale * 4
        increment_edge_version(G)
        default_compute_delta_nfr(G, cache_size=2)
        nodes2, _ = cached_nodes_and_A(G, cache_size=2)

        assert len(nodes2) == 4
        assert nodes1 is not nodes2

        G.add_edge(3, 4)
        increment_edge_version(G)
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
def test_neighbor_sum_buffers_reused_and_results_stable(vectorized, monkeypatch):
    if vectorized:
        pytest.importorskip("numpy")
        context_factory = nullcontext
    else:
        context_factory = lambda: numpy_disabled(monkeypatch)

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
    with context_factory():
        default_compute_delta_nfr(G)
    first = _collect_dnfr(G)
    manager, state = _get_prep_state(G)
    cache = state.cache
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

    with context_factory():
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

    stats = manager.get_metrics(DNFR_PREP_STATE_KEY)
    assert stats.misses >= 1
    assert stats.hits >= 1


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
def test_default_compute_delta_nfr_updates_on_state_change(vectorized, monkeypatch):
    if vectorized:
        pytest.importorskip("numpy")
        context = nullcontext()
    else:
        context = numpy_disabled(monkeypatch)

    G = _setup_graph()
    G.graph["DNFR_WEIGHTS"] = {
        "phase": 1.0,
        "epi": 1.0,
        "vf": 1.0,
        "topo": 0.0,
    }
    with context:
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
