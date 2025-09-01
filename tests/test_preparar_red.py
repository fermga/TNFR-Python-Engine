import networkx as nx

from tnfr.ontosim import preparar_red


def test_preparar_red_init_attrs_por_defecto():
    G = nx.path_graph(3)
    preparar_red(G)
    assert all('θ' in d for _, d in G.nodes(data=True))


def test_preparar_red_sin_init_attrs():
    G = nx.path_graph(3)
    preparar_red(G, init_attrs=False)
    assert all('θ' not in d for _, d in G.nodes(data=True))

