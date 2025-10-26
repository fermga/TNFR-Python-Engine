import networkx as nx
import pytest

from tnfr.utils.graph import get_graph, get_graph_mapping, supports_add_edge


class Sentinel:
    """Simple sentinel lacking a ``graph`` attribute."""


def test_get_graph_raises_type_error_for_unsupported_object():
    with pytest.raises(TypeError, match="Unsupported graph object: metadata mapping not accessible"):
        get_graph(Sentinel())


def test_get_graph_mapping_warns_and_returns_none_for_non_mapping_value():
    graph = {"tnfr": ["not", "a", "mapping"]}
    warn_msg = "tnfr metadata must be a mapping"

    with pytest.warns(UserWarning, match=warn_msg):
        result = get_graph_mapping(graph, "tnfr", warn_msg)

    assert result is None


def test_supports_add_edge_returns_true_for_networkx_graph():
    graph = nx.Graph()

    assert supports_add_edge(graph) is True


def test_supports_add_edge_returns_false_for_sentinel_without_add_edge():
    sentinel = Sentinel()

    assert supports_add_edge(sentinel) is False
