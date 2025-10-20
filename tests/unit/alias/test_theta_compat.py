import math

import networkx as nx
import pytest

from tnfr.alias import get_theta_attr, set_theta, set_theta_attr
from tnfr.utils import migrate_legacy_phase_attributes


def test_get_theta_attr_rejects_legacy_key() -> None:
    data = {"fase": math.pi / 3}

    with pytest.raises(ValueError, match="Legacy node attribute \"fase\""):
        get_theta_attr(data, 0.0)


def test_migration_restores_theta_access() -> None:
    data = {"fase": math.pi / 3}

    assert migrate_legacy_phase_attributes({0: data}) == 1
    value = get_theta_attr(data, 0.0)

    assert value == pytest.approx(math.pi / 3)
    assert data["theta"] == pytest.approx(math.pi / 3)
    assert data["phase"] == pytest.approx(math.pi / 3)


def test_migration_converts_symbol_key() -> None:
    data = {"θ": math.pi / 6}

    assert migrate_legacy_phase_attributes({0: data}) == 1
    assert data["theta"] == pytest.approx(math.pi / 6)
    assert data["phase"] == pytest.approx(math.pi / 6)


def test_set_theta_keeps_only_english_aliases() -> None:
    graph = nx.Graph()
    graph.add_node(0)

    set_theta(graph, 0, math.pi / 4)
    set_theta_attr(graph.nodes[0], math.pi / 2)

    node_data = graph.nodes[0]
    assert "fase" not in node_data
    assert node_data["theta"] == pytest.approx(math.pi / 2)
    assert node_data["phase"] == pytest.approx(node_data["theta"])
