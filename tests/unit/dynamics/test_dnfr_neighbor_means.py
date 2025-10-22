from contextlib import contextmanager

import networkx as nx
import pytest

from tnfr.alias import collect_attr, set_attr
from tnfr.constants import get_aliases
from tnfr.dynamics import default_compute_delta_nfr
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


def _build_graph():
    G = nx.path_graph(4)
    isolated = 99
    G.add_node(isolated)
    for idx, node in enumerate(G.nodes):
        set_attr(G.nodes[node], ALIAS_THETA, 0.25 * (idx + 1))
        set_attr(G.nodes[node], ALIAS_EPI, 0.125 * (idx + 2))
        set_attr(G.nodes[node], ALIAS_VF, 0.075 * (idx + 3))
    G.graph["DNFR_WEIGHTS"] = {
        "phase": 0.4,
        "epi": 0.3,
        "vf": 0.2,
        "topo": 0.1,
    }
    return G


def _collect_dnfr(G):
    return collect_attr(G, G.nodes, ALIAS_DNFR, 0.0)


def _get_prep_state(G):
    manager = _graph_cache_manager(G.graph)
    state = manager.get(DNFR_PREP_STATE_KEY)
    assert isinstance(state, DnfrPrepState)
    return manager, state


def test_neighbor_means_vectorized_matches_python(monkeypatch):
    pytest.importorskip("numpy")

    G_vec = _build_graph()
    G_py = _build_graph()

    with numpy_disabled(monkeypatch):
        default_compute_delta_nfr(G_py)

    default_compute_delta_nfr(G_vec)

    assert _collect_dnfr(G_vec) == pytest.approx(_collect_dnfr(G_py))


def test_neighbor_mean_workspaces_reused(monkeypatch):
    np = pytest.importorskip("numpy")

    G = _build_graph()
    default_compute_delta_nfr(G)
    manager, state = _get_prep_state(G)
    cache = state.cache
    assert cache is not None

    inv_first = cache.neighbor_inv_count_np
    cos_first = cache.neighbor_cos_avg_np
    sin_first = cache.neighbor_sin_avg_np
    tmp_first = cache.neighbor_mean_tmp_np
    length_first = cache.neighbor_mean_length_np
    assert all(
        isinstance(arr, np.ndarray)
        for arr in (inv_first, cos_first, sin_first, tmp_first, length_first)
    )

    with numpy_disabled(monkeypatch):
        default_compute_delta_nfr(G)
    default_compute_delta_nfr(G)

    _, state_after = _get_prep_state(G)
    cache_after = state_after.cache
    assert cache_after is cache
    assert cache_after.neighbor_inv_count_np is inv_first
    assert cache_after.neighbor_cos_avg_np is cos_first
    assert cache_after.neighbor_sin_avg_np is sin_first
    assert cache_after.neighbor_mean_tmp_np is tmp_first
    assert cache_after.neighbor_mean_length_np is length_first
    stats = manager.get_metrics(DNFR_PREP_STATE_KEY)
    assert stats.hits >= 1


def test_pure_python_parallel_matches_serial(monkeypatch):
    G_serial = _build_graph()
    G_parallel = _build_graph()

    with numpy_disabled(monkeypatch):
        default_compute_delta_nfr(G_serial, n_jobs=1)
        default_compute_delta_nfr(G_parallel, n_jobs=3)

    assert _collect_dnfr(G_parallel) == pytest.approx(_collect_dnfr(G_serial))


def test_vectorized_n_jobs_argument():
    pytest.importorskip("numpy")

    G_reference = _build_graph()
    G_vectorized = _build_graph()

    default_compute_delta_nfr(G_reference)
    default_compute_delta_nfr(G_vectorized, n_jobs=4)

    assert _collect_dnfr(G_vectorized) == pytest.approx(_collect_dnfr(G_reference))
