"""Pruebas de remesh community."""
import networkx as nx
from tnfr.constants import attach_defaults
from tnfr.operators import apply_topological_remesh


def test_remesh_community_reduces_nodes_and_preserves_connectivity(graph_canon):
    G = graph_canon()
    G.add_edges_from([(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(2,3)])
    attach_defaults(G)
    for n in G.nodes():
        G.nodes[n]["EPI"] = float(n)

    n_before = G.number_of_nodes()
    apply_topological_remesh(G, mode="community")
    assert nx.is_connected(G)
    assert G.number_of_nodes() < n_before
    ev = G.graph.get("history", {}).get("remesh_events", [])
    assert ev and ev[-1].get("mode") == "community"
