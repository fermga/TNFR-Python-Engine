import math

import networkx as nx
import pytest

from tnfr.alias import get_theta_attr, set_theta, set_theta_attr


def test_get_theta_attr_warns_and_normalizes_fase() -> None:
    data = {"fase": math.pi / 3}

    with pytest.warns(DeprecationWarning):
        value = get_theta_attr(data, 0.0)

    assert value == pytest.approx(math.pi / 3)
    assert "fase" not in data
    assert data["theta"] == pytest.approx(math.pi / 3)
    assert data["phase"] == pytest.approx(math.pi / 3)


def test_set_theta_rewrites_legacy_key_to_english_aliases() -> None:
    graph = nx.Graph()
    graph.add_node(0, fase=0.0)

    # Ensure the compatibility shim migrates legacy storage.
    with pytest.warns(DeprecationWarning):
        assert get_theta_attr(graph.nodes[0], 0.0) == pytest.approx(0.0)

    # Updating the phase should keep only English aliases in the payload.
    set_theta(graph, 0, math.pi / 4)
    set_theta_attr(graph.nodes[0], math.pi / 2)

    node_data = graph.nodes[0]
    assert "fase" not in node_data
    assert node_data["theta"] == pytest.approx(math.pi / 2)
    assert node_data["phase"] == pytest.approx(node_data["theta"])
