"""Pruebas de initialization."""

import networkx as nx

from tnfr.initialization import init_node_attrs
from tnfr.constants import (
    attach_defaults,
    VF_KEY,
    THETA_KEY,
    ALIAS_VF,
    ALIAS_THETA,
)
from tnfr.alias import get_attr


def test_init_node_attrs_reproducible():
    seed = 123
    G1 = nx.path_graph(5)
    attach_defaults(G1)
    G1.graph["RANDOM_SEED"] = seed
    init_node_attrs(G1)
    attrs1 = {
        n: (d["EPI"], d[THETA_KEY], d[VF_KEY], d["Si"])
        for n, d in G1.nodes(data=True)
    }

    G2 = nx.path_graph(5)
    attach_defaults(G2)
    G2.graph["RANDOM_SEED"] = seed
    init_node_attrs(G2)
    attrs2 = {
        n: (d["EPI"], d[THETA_KEY], d[VF_KEY], d["Si"])
        for n, d in G2.nodes(data=True)
    }

    assert attrs1 == attrs2


def test_init_node_attrs_reversed_uniform_bounds():
    seed = 2024
    G1 = nx.path_graph(3)
    attach_defaults(G1)
    G1.graph.update(
        {
            "RANDOM_SEED": seed,
            "INIT_VF_MODE": "uniform",
            "INIT_VF_MIN": 0.8,
            "INIT_VF_MAX": 0.2,
        }
    )
    init_node_attrs(G1)
    vfs1 = [d[VF_KEY] for _, d in G1.nodes(data=True)]

    G2 = nx.path_graph(3)
    attach_defaults(G2)
    G2.graph.update(
        {
            "RANDOM_SEED": seed,
            "INIT_VF_MODE": "uniform",
            "INIT_VF_MIN": 0.2,
            "INIT_VF_MAX": 0.8,
        }
    )
    init_node_attrs(G2)
    vfs2 = [d[VF_KEY] for _, d in G2.nodes(data=True)]

    assert vfs1 == vfs2


def test_init_node_attrs_alias_access():
    G = nx.path_graph(2)
    attach_defaults(G)
    init_node_attrs(G)
    for _, d in G.nodes(data=True):
        d_ascii = {"nu_f": d[VF_KEY], "theta": d[THETA_KEY]}
        assert get_attr(d, ALIAS_VF, 0.0) == d[VF_KEY]
        assert get_attr(d, ALIAS_THETA, 0.0) == d[THETA_KEY]
        assert get_attr(d_ascii, ALIAS_VF, 0.0) == d[VF_KEY]
        assert get_attr(d_ascii, ALIAS_THETA, 0.0) == d[THETA_KEY]
