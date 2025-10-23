import networkx as nx

from tnfr.ontosim import prepare_network


def test_prepare_network_initializes_attrs_by_default():
    G = nx.path_graph(3)
    prepare_network(G)
    assert all("theta" in d for _, d in G.nodes(data=True))


def test_prepare_network_allows_disabling_init_attrs():
    G = nx.path_graph(3)
    prepare_network(G, init_attrs=False)
    assert all("theta" not in d for _, d in G.nodes(data=True))
