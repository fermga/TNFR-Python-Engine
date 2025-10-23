import math

import networkx as nx
import pytest

from tnfr.alias import get_theta_attr, set_theta, set_theta_attr

from tests.legacy_tokens import LEGACY_PHASE_ALIAS


def test_get_theta_attr_ignores_legacy_key() -> None:
    data = {LEGACY_PHASE_ALIAS: math.pi / 3}

    value = get_theta_attr(data, 0.0)

    assert value == 0.0
    assert "theta" not in data
    assert "phase" not in data


def test_set_theta_keeps_only_english_aliases() -> None:
    graph = nx.Graph()
    graph.add_node(0)

    set_theta(graph, 0, math.pi / 4)
    set_theta_attr(graph.nodes[0], math.pi / 2)

    node_data = graph.nodes[0]
    assert LEGACY_PHASE_ALIAS not in node_data
    assert node_data["theta"] == pytest.approx(math.pi / 2)
    assert node_data["phase"] == pytest.approx(node_data["theta"])
