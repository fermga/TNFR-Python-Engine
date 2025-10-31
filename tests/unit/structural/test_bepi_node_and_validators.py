"""Integration tests for BEPI storage across nodes and validators."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF
from tnfr.mathematics import BEPIElement
from tnfr.node import NodeNX
from tnfr.validation.graph import run_validators
from tnfr.validation.runtime import GraphCanonicalValidator


def _make_bepi() -> BEPIElement:
    return BEPIElement(
        (1.0 + 0.0j, 0.2 + 0.3j, -0.5 + 0.1j),
        (0.1 + 0.2j, -0.3 + 0.4j, 0.5 + 0.0j),
        (0.0, 0.5, 1.0),
    )


def _configure_bounds(graph: nx.Graph, *, epi_max: float = 1.0) -> None:
    graph.graph.update({
        "EPI_MIN": 0.0,
        "EPI_MAX": epi_max,
        "VF_MIN": 0.0,
        "VF_MAX": 3.0,
    })


def test_nodenx_epi_roundtrip_serializes_bepi() -> None:
    graph = nx.Graph()
    graph.add_node("n0")
    node = NodeNX(graph, "n0")

    element = _make_bepi()
    node.EPI = element

    stored = graph.nodes["n0"][ALIAS_EPI[0]]
    assert set(stored) == {"continuous", "discrete", "grid"}
    np.testing.assert_allclose(np.array(stored["continuous"]), element.f_continuous)
    np.testing.assert_allclose(np.array(stored["discrete"]), element.a_discrete)
    np.testing.assert_allclose(np.array(stored["grid"], dtype=float), element.x_grid)

    roundtrip = node.EPI
    np.testing.assert_allclose(roundtrip.f_continuous, element.f_continuous)
    np.testing.assert_allclose(roundtrip.a_discrete, element.a_discrete)
    np.testing.assert_allclose(roundtrip.x_grid, element.x_grid)


def test_graph_validators_accept_bepi_payload() -> None:
    graph = nx.Graph()
    _configure_bounds(graph, epi_max=2.0)
    graph.add_node(0)
    node = NodeNX(graph, 0)
    node.EPI = _make_bepi()
    node.vf = 1.5

    run_validators(graph)


def test_graph_validators_reject_malformed_bepi() -> None:
    graph = nx.Graph()
    _configure_bounds(graph, epi_max=2.0)
    graph.add_node(0)
    set_attr(graph.nodes[0], ALIAS_VF, 1.0)
    malformed = {
        "continuous": (1.0 + 0.0j, 0.5 + 0.1j),
        "discrete": (0.1 + 0.2j, 0.2 + 0.3j),
        "grid": (0.0, 0.25, 0.5),
    }
    graph.nodes[0][ALIAS_EPI[0]] = malformed

    with pytest.raises(ValueError):
        run_validators(graph)


def test_runtime_validator_clamps_bepi_components() -> None:
    graph = nx.Graph()
    _configure_bounds(graph, epi_max=0.4)
    graph.add_node(0)
    node = NodeNX(graph, 0)

    exaggerated = BEPIElement(
        (1.2 + 0.0j, -0.9 + 0.4j, 0.3 + 0.9j),
        (0.9 + 0.6j, -1.2 + 0.0j, 0.5 + 0.2j),
        (0.0, 0.5, 1.0),
    )
    node.EPI = exaggerated
    node.vf = 2.5

    validator = GraphCanonicalValidator()
    outcome = validator.validate(graph)

    assert outcome.passed
    clamped = node.EPI
    assert np.max(np.abs(clamped.f_continuous)) <= 0.4 + 1e-12
    assert np.max(np.abs(clamped.a_discrete)) <= 0.4 + 1e-12

