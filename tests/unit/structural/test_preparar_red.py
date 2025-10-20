import networkx as nx

from tnfr.ontosim import prepare_network, preparar_red


def test_prepare_network_init_attrs_por_defecto():
    G = nx.path_graph(3)
    prepare_network(G)
    assert all("θ" in d for _, d in G.nodes(data=True))


def test_prepare_network_sin_init_attrs():
    G = nx.path_graph(3)
    prepare_network(G, init_attrs=False)
    assert all("θ" not in d for _, d in G.nodes(data=True))


def test_preparar_red_alias_sigue_activo():
    G = nx.path_graph(2)
    preparar_red(G)
    assert all("θ" in d for _, d in G.nodes(data=True))
    assert preparar_red is prepare_network
