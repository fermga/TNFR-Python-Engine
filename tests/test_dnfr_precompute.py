"""Pruebas de dnfr precompute."""

import pytest
from contextlib import contextmanager

import networkx as nx

from tnfr.dynamics import (
    _prepare_dnfr_data,
    _compute_dnfr,
)
from tnfr.constants import get_aliases
from tnfr.alias import set_attr, collect_attr, get_attr

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


def _setup_graph(size: int = 5, factory=nx.path_graph):
    G = factory(size)
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


def test_strategies_share_precomputed_data(monkeypatch):
    np = pytest.importorskip("numpy")
    del np

    G = _setup_graph()
    data = _prepare_dnfr_data(G)
    with numpy_disabled(monkeypatch):
        _compute_dnfr(G, data)
    dnfr_loop = collect_attr(G, G.nodes, ALIAS_DNFR, 0.0)
    for n in G.nodes:
        set_attr(G.nodes[n], ALIAS_DNFR, 0.0)
    _compute_dnfr(G, data)
    dnfr_vec = collect_attr(G, G.nodes, ALIAS_DNFR, 0.0)
    assert dnfr_loop == pytest.approx(dnfr_vec)


def test_prepare_dnfr_numpy_vectors_match_aliases():
    pytest.importorskip("numpy")
    G = _setup_graph()
    data = _prepare_dnfr_data(G)
    A = data.get("A")
    if A is not None:
        assert A.shape == (len(data["nodes"]), len(data["nodes"]))
    assert data["theta_np"] is not None
    assert data["epi_np"] is not None
    assert data["vf_np"] is not None
    assert data["cos_theta_np"] is not None
    assert data["sin_theta_np"] is not None
    assert data["deg_array"] is not None

    theta_expected = [get_attr(G.nodes[n], ALIAS_THETA, 0.0) for n in G.nodes]
    epi_expected = [get_attr(G.nodes[n], ALIAS_EPI, 0.0) for n in G.nodes]
    vf_expected = [get_attr(G.nodes[n], ALIAS_VF, 0.0) for n in G.nodes]
    deg_expected = [float(G.degree(n)) for n in G.nodes]

    assert data["theta_np"].tolist() == pytest.approx(theta_expected)
    assert data["epi_np"].tolist() == pytest.approx(epi_expected)
    assert data["vf_np"].tolist() == pytest.approx(vf_expected)
    assert data["deg_array"].tolist() == pytest.approx(deg_expected)

    # Ensure we can reuse the prepared data for the vectorised computation
    _compute_dnfr(G, data)
    dnfr_vec = collect_attr(G, G.nodes, ALIAS_DNFR, 0.0)
    assert all(isinstance(val, float) for val in dnfr_vec)


@pytest.mark.parametrize(
    "factory,size",
    [
        (nx.path_graph, 7),
        (nx.complete_graph, 6),
    ],
)
def test_numpy_broadcast_fallback_matches_python(factory, size, monkeypatch):
    np = pytest.importorskip("numpy")
    del np

    template = _setup_graph(size=size, factory=factory)
    broadcast = template.copy()
    data = _prepare_dnfr_data(broadcast)
    _compute_dnfr(broadcast, data)
    dnfr_broadcast = collect_attr(broadcast, broadcast.nodes, ALIAS_DNFR, 0.0)

    with numpy_disabled(monkeypatch):
        python_only = template.copy()
        data_loop = _prepare_dnfr_data(python_only)
        _compute_dnfr(python_only, data_loop)
    dnfr_loop = collect_attr(python_only, python_only.nodes, ALIAS_DNFR, 0.0)

    assert dnfr_broadcast == pytest.approx(dnfr_loop, rel=1e-9, abs=1e-9)
