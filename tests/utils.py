import networkx as nx
from tnfr.constants import inject_defaults


def build_graph(n, graph_canon=None):
    G = graph_canon() if graph_canon is not None else nx.Graph()
    inject_defaults(G)
    for i in range(n):
        G.add_node(i, **{"θ": 0.0, "EPI": 0.0})
    return G
