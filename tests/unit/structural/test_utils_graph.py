import pytest

from tnfr.utils.graph import get_graph, get_graph_mapping


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
