import math

import pytest

import tnfr.utils.init as utils_init
from tnfr.alias import set_attr
from tnfr.constants import get_aliases
from tnfr.metrics.sense_index import compute_Si
from tnfr.metrics.trig import neighbor_phase_mean, neighbor_phase_mean_bulk
from tnfr.metrics.trig_cache import get_trig_cache

ALIAS_THETA = get_aliases("THETA")
ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")

TRIG_SENTINEL_KEYS = ("_cos_th", "_sin_th", "_thetas", "_trig_cache")


def test_trig_cache_reuse_between_modules(monkeypatch, graph_canon):
    cos_calls = 0
    sin_calls = 0
    orig_cos = math.cos
    orig_sin = math.sin

    def cos_wrapper(x):
        nonlocal cos_calls
        cos_calls += 1
        return orig_cos(x)

    def sin_wrapper(x):
        nonlocal sin_calls
        sin_calls += 1
        return orig_sin(x)

    monkeypatch.setattr(math, "cos", cos_wrapper)
    monkeypatch.setattr(math, "sin", sin_wrapper)
    original_cached_import = utils_init.cached_import

    def fake_cached_import(module, attr=None, **kwargs):
        if module == "numpy":
            return None
        return original_cached_import(module, attr=attr, **kwargs)

    monkeypatch.setattr(utils_init, "cached_import", fake_cached_import)

    G = graph_canon()
    sentinel = object()
    for key in TRIG_SENTINEL_KEYS:
        G.graph[key] = sentinel
    G.add_edge(1, 2)
    set_attr(G.nodes[1], ALIAS_THETA, 0.0)
    set_attr(G.nodes[2], ALIAS_THETA, math.pi / 2)
    set_attr(G.nodes[1], ALIAS_VF, 0.0)
    set_attr(G.nodes[2], ALIAS_VF, 0.0)
    set_attr(G.nodes[1], ALIAS_DNFR, 0.0)
    set_attr(G.nodes[2], ALIAS_DNFR, 0.0)

    trig1 = get_trig_cache(G)
    version = G.graph.get("_trig_version", 0)
    assert cos_calls == 2
    assert sin_calls == 2

    assert neighbor_phase_mean(G, 1) == pytest.approx(math.pi / 2)
    assert cos_calls == 2
    assert sin_calls == 2

    compute_Si(G, inplace=False)
    assert cos_calls == 2
    assert sin_calls == 2

    trig2 = get_trig_cache(G)
    assert trig1 is trig2
    assert G.graph.get("_trig_version") == version
    for key in TRIG_SENTINEL_KEYS:
        assert G.graph[key] is sentinel


def test_trig_cache_rebuilds_after_direct_theta_mutation(monkeypatch, graph_canon):
    cos_calls = 0
    sin_calls = 0
    orig_cos = math.cos
    orig_sin = math.sin

    def cos_wrapper(x):
        nonlocal cos_calls
        cos_calls += 1
        return orig_cos(x)

    def sin_wrapper(x):
        nonlocal sin_calls
        sin_calls += 1
        return orig_sin(x)

    monkeypatch.setattr(math, "cos", cos_wrapper)
    monkeypatch.setattr(math, "sin", sin_wrapper)

    original_cached_import = utils_init.cached_import

    def fake_cached_import(module, attr=None, **kwargs):
        if module == "numpy":
            return None
        return original_cached_import(module, attr=attr, **kwargs)

    monkeypatch.setattr(utils_init, "cached_import", fake_cached_import)

    G = graph_canon()
    G.add_edge(1, 2)
    set_attr(G.nodes[1], ALIAS_THETA, 0.0)
    set_attr(G.nodes[2], ALIAS_THETA, math.pi / 2)
    set_attr(G.nodes[1], ALIAS_VF, 0.0)
    set_attr(G.nodes[2], ALIAS_VF, 0.0)
    set_attr(G.nodes[1], ALIAS_DNFR, 0.0)
    set_attr(G.nodes[2], ALIAS_DNFR, 0.0)

    trig1 = get_trig_cache(G)
    base_version = G.graph.get("_trig_version", 0)
    assert trig1.theta[2] == pytest.approx(math.pi / 2)
    assert cos_calls == 2
    assert sin_calls == 2

    assert neighbor_phase_mean(G, 1) == pytest.approx(math.pi / 2)
    assert cos_calls == 2
    assert sin_calls == 2

    G.nodes[2][ALIAS_THETA[0]] = math.pi

    updated_mean = neighbor_phase_mean(G, 1)
    assert updated_mean == pytest.approx(math.pi)
    assert cos_calls == 4
    assert sin_calls == 4
    assert G.graph.get("_trig_version") == base_version + 1

    trig2 = get_trig_cache(G)
    assert trig2 is not trig1
    assert trig2.theta[2] == pytest.approx(math.pi)
    assert cos_calls == 4
    assert sin_calls == 4


def test_trig_cache_rebuilds_after_direct_theta_mutation_numpy(monkeypatch, graph_canon):
    np = pytest.importorskip("numpy")

    cos_calls = 0
    sin_calls = 0
    orig_cos = np.cos
    orig_sin = np.sin

    def cos_wrapper(values, *args, **kwargs):
        nonlocal cos_calls
        cos_calls += 1
        return orig_cos(values, *args, **kwargs)

    def sin_wrapper(values, *args, **kwargs):
        nonlocal sin_calls
        sin_calls += 1
        return orig_sin(values, *args, **kwargs)

    monkeypatch.setattr(np, "cos", cos_wrapper)
    monkeypatch.setattr(np, "sin", sin_wrapper)

    G = graph_canon()
    G.add_edge(1, 2)
    set_attr(G.nodes[1], ALIAS_THETA, 0.0)
    set_attr(G.nodes[2], ALIAS_THETA, math.pi / 2)
    set_attr(G.nodes[1], ALIAS_VF, 0.0)
    set_attr(G.nodes[2], ALIAS_VF, 0.0)
    set_attr(G.nodes[1], ALIAS_DNFR, 0.0)
    set_attr(G.nodes[2], ALIAS_DNFR, 0.0)

    trig1 = get_trig_cache(G, np=np)
    base_version = G.graph.get("_trig_version", 0)
    assert trig1.theta[2] == pytest.approx(math.pi / 2)
    assert cos_calls == 1
    assert sin_calls == 1

    G.nodes[2][ALIAS_THETA[0]] = math.pi

    trig2 = get_trig_cache(G, np=np)
    assert trig2 is not trig1
    assert trig2.theta[2] == pytest.approx(math.pi)
    assert cos_calls == 2
    assert sin_calls == 2
    assert G.graph.get("_trig_version") == base_version + 1


def test_neighbor_phase_mean_bulk_isolated_nodes():
    np = pytest.importorskip("numpy")

    theta = np.array([0.0, math.pi / 2, -math.pi / 2], dtype=float)
    cos_values = np.cos(theta)
    sin_values = np.sin(theta)

    edge_src = np.array([0, 1], dtype=np.intp)
    edge_dst = np.array([1, 0], dtype=np.intp)

    mean_theta, has_neighbors = neighbor_phase_mean_bulk(
        edge_src,
        edge_dst,
        cos_values=cos_values,
        sin_values=sin_values,
        theta_values=theta,
        node_count=theta.size,
        np=np,
    )

    assert has_neighbors.tolist() == [True, True, False]
    assert mean_theta.dtype == theta.dtype
    assert mean_theta[0] == pytest.approx(math.pi / 2)
    assert mean_theta[1] == pytest.approx(0.0)
    assert mean_theta[2] == pytest.approx(theta[2])


def test_neighbor_phase_mean_bulk_reuses_buffers():
    np = pytest.importorskip("numpy")

    theta = np.array([0.0, math.pi / 3, -math.pi / 4, math.pi], dtype=float)
    cos_values = np.cos(theta)
    sin_values = np.sin(theta)

    edge_src = np.array([0, 1, 2], dtype=np.intp)
    edge_dst = np.array([1, 2, 1], dtype=np.intp)

    cos_sum = np.empty(theta.size, dtype=float)
    sin_sum = np.empty(theta.size, dtype=float)
    counts = np.empty(theta.size, dtype=float)
    mean_cos = np.empty(theta.size, dtype=float)
    mean_sin = np.empty(theta.size, dtype=float)

    cos_sum.fill(np.nan)
    sin_sum.fill(np.nan)
    counts.fill(np.nan)
    mean_cos.fill(np.nan)
    mean_sin.fill(np.nan)

    mean_theta, mask = neighbor_phase_mean_bulk(
        edge_src,
        edge_dst,
        cos_values=cos_values,
        sin_values=sin_values,
        theta_values=theta,
        node_count=theta.size,
        np=np,
        neighbor_cos_sum=cos_sum,
        neighbor_sin_sum=sin_sum,
        neighbor_counts=counts,
        mean_cos=mean_cos,
        mean_sin=mean_sin,
    )

    fresh_theta, fresh_mask = neighbor_phase_mean_bulk(
        edge_src,
        edge_dst,
        cos_values=cos_values,
        sin_values=sin_values,
        theta_values=theta,
        node_count=theta.size,
        np=np,
    )

    assert mask.tolist() == [False, True, True, False]
    assert fresh_mask.tolist() == mask.tolist()
    assert cos_sum is not None and sin_sum is not None
    assert counts is not None and mean_cos is not None and mean_sin is not None
    assert np.all(np.isfinite(cos_sum))
    assert np.all(np.isfinite(sin_sum))
    assert np.all(np.isfinite(counts))
    assert np.all(np.isfinite(mean_cos))
    assert np.all(np.isfinite(mean_sin))
    assert counts[0] == 0.0
    assert counts[3] == 0.0
    assert mean_cos[0] == 0.0
    assert mean_sin[3] == 0.0
    np.testing.assert_allclose(mean_theta, fresh_theta, rtol=1e-12, atol=1e-12)

    cos_sum.fill(np.nan)
    sin_sum.fill(np.nan)
    counts.fill(np.nan)
    mean_cos.fill(np.nan)
    mean_sin.fill(np.nan)

    empty_theta, empty_mask = neighbor_phase_mean_bulk(
        np.array([], dtype=np.intp),
        np.array([], dtype=np.intp),
        cos_values=cos_values,
        sin_values=sin_values,
        theta_values=theta,
        node_count=theta.size,
        np=np,
        neighbor_cos_sum=cos_sum,
        neighbor_sin_sum=sin_sum,
        neighbor_counts=counts,
        mean_cos=mean_cos,
        mean_sin=mean_sin,
    )

    assert empty_theta.shape == theta.shape
    assert np.allclose(empty_theta, theta)
    assert not empty_mask.any()
    assert np.all(cos_sum == 0.0)
    assert np.all(sin_sum == 0.0)
    assert np.all(counts == 0.0)
    assert np.all(mean_cos == 0.0)
    assert np.all(mean_sin == 0.0)
