import networkx as nx

from tnfr.initialization import init_node_attrs
from tnfr.constants import attach_defaults


def test_init_node_attrs_reproducible():
    seed = 123
    G1 = nx.path_graph(5)
    attach_defaults(G1)
    G1.graph["RANDOM_SEED"] = seed
    init_node_attrs(G1)
    attrs1 = {n: (d["EPI"], d["θ"], d["νf"], d["Si"]) for n, d in G1.nodes(data=True)}

    G2 = nx.path_graph(5)
    attach_defaults(G2)
    G2.graph["RANDOM_SEED"] = seed
    init_node_attrs(G2)
    attrs2 = {n: (d["EPI"], d["θ"], d["νf"], d["Si"]) for n, d in G2.nodes(data=True)}

    assert attrs1 == attrs2
