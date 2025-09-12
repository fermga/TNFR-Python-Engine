"""Pruebas de dnfr precompute."""

import pytest
import networkx as nx

from tnfr.dynamics import (
    _prepare_dnfr_data,
    _compute_dnfr,
)
from tnfr.constants import get_aliases
from tnfr.alias import get_attr, set_attr

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


@pytest.mark.xfail(reason="vectorized strategy differs under alias mapping")
def test_strategies_share_precomputed_data():
    pytest.importorskip("numpy")
    G = _setup_graph()
    data = _prepare_dnfr_data(G)
    _compute_dnfr(G, data, use_numpy=False)
    dnfr_loop = [get_attr(G.nodes[n], ALIAS_DNFR, 0.0) for n in G.nodes]
    for n in G.nodes:
        set_attr(G.nodes[n], ALIAS_DNFR, 0.0)
    data = _prepare_dnfr_data(G)
    _compute_dnfr(G, data, use_numpy=True)
    dnfr_vec = [get_attr(G.nodes[n], ALIAS_DNFR, 0.0) for n in G.nodes]
    assert dnfr_loop == pytest.approx(dnfr_vec)
