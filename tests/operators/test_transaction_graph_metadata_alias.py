"""Graph-owned references must still read live metadata after restoration."""

import networkx as nx
import pytest

from tnfr.node import NodeNX
from tnfr.operators.network_stage import GraphTransactionSnapshot
from tnfr.operators.retained_organization import capture_retained_organization


@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
@pytest.mark.parametrize("operation", ["noop", "observation", "rollback"])
def test_cached_node_keeps_live_graph_metadata_alias(graph_type, operation):
    graph = graph_type(policy_value=3)
    graph.add_node("parent", EPI=0.5, nu_f=1.0, theta=0.25, delta_nfr=0.125)
    adapter = NodeNX.from_graph(graph, "parent")
    metadata = graph.graph
    assert adapter.G is graph and adapter.graph is metadata

    if operation == "observation":
        capture_retained_organization(graph, ("parent",))
    else:
        transaction = GraphTransactionSnapshot(graph)
        if operation == "rollback":
            graph.graph["policy_value"] = 99
            graph.add_node("temporary")
            adapter.graph = {"policy_value": -1}
        transaction.restore(graph)

    assert tuple(graph) == ("parent",)
    assert graph.graph is metadata and graph.graph["policy_value"] == 3
    assert NodeNX.from_graph(graph, "parent") is adapter
    assert adapter.G is graph and adapter.graph is graph.graph
    graph.graph["policy_value"] = 7
    assert adapter.graph["policy_value"] == 7


def test_nested_graph_metadata_reference_restores_original_owner_and_values():
    graph = nx.Graph(policy_value=3)
    graph.graph["nested"] = {"owner": graph.graph}
    metadata = graph.graph
    nested = graph.graph["nested"]
    transaction = GraphTransactionSnapshot(graph)
    nested["owner"] = {"policy_value": -1}
    graph.graph["policy_value"] = 99
    transaction.restore(graph)
    assert graph.graph is metadata
    assert graph.graph["nested"] is nested
    assert nested["owner"] is metadata
    assert nested["owner"]["policy_value"] == 3
