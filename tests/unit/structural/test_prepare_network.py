import networkx as nx

from tnfr.ontosim import prepare_network


def test_prepare_network_init_attrs_por_defecto():
    G = nx.path_graph(3)
    prepare_network(G)
    assert all("θ" in d for _, d in G.nodes(data=True))


def test_prepare_network_sin_init_attrs():
    G = nx.path_graph(3)
    prepare_network(G, init_attrs=False)
    assert all("θ" not in d for _, d in G.nodes(data=True))
