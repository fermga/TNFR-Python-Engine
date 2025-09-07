import networkx as nx
from tnfr.constants import attach_defaults


def build_graph(n):
    G = nx.Graph()
    attach_defaults(G)
    for i in range(n):
        G.add_node(i, **{"θ": 0.0, "EPI": 0.0})
    return G
