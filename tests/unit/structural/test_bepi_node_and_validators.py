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

def test_bepi_pickle_serialization() -> None:
    """Test BEPIElement pickle serialization preserves structural integrity."""
    import pickle

    element = _make_bepi()

    # Serialize with pickle
    pickled = pickle.dumps(element)
    restored = pickle.loads(pickled)

    # Verify structural integrity (TNFR invariant #1: EPI as coherent form)
    np.testing.assert_allclose(restored.f_continuous, element.f_continuous)
    np.testing.assert_allclose(restored.a_discrete, element.a_discrete)
    np.testing.assert_allclose(restored.x_grid, element.x_grid)

    # Verify it's still a valid BEPIElement
    assert isinstance(restored, BEPIElement)

def test_bepi_json_serialization() -> None:
    """Test BEPIElement JSON serialization via serialize_bepi_json helper."""
    import json
    from tnfr.types import serialize_bepi_json, deserialize_bepi_json

    element = _make_bepi()

    # Serialize to JSON-compatible format
    json_data = serialize_bepi_json(element)
    assert set(json_data.keys()) == {"continuous", "discrete", "grid"}

    # Verify JSON compatibility
    json_str = json.dumps(json_data)
    assert isinstance(json_str, str)

    # Deserialize and verify structural integrity
    loaded_data = json.loads(json_str)
    restored = deserialize_bepi_json(loaded_data)

    np.testing.assert_allclose(restored.f_continuous, element.f_continuous)
    np.testing.assert_allclose(restored.a_discrete, element.a_discrete)
    np.testing.assert_allclose(restored.x_grid, element.x_grid)

def test_bepi_yaml_serialization() -> None:
    """Test BEPIElement YAML serialization via serialize_bepi_json helper."""
    yaml = pytest.importorskip("yaml")
    from tnfr.types import serialize_bepi_json, deserialize_bepi_json

    element = _make_bepi()

    # Serialize to YAML-compatible format
    json_data = serialize_bepi_json(element)

    # Verify YAML safe_dump works
    yaml_str = yaml.safe_dump(json_data)
    assert isinstance(yaml_str, str)

    # Deserialize and verify structural integrity
    loaded_data = yaml.safe_load(yaml_str)
    restored = deserialize_bepi_json(loaded_data)

    np.testing.assert_allclose(restored.f_continuous, element.f_continuous)
    np.testing.assert_allclose(restored.a_discrete, element.a_discrete)
    np.testing.assert_allclose(restored.x_grid, element.x_grid)

def test_bepi_nested_serialization_preserves_fractality() -> None:
    """Test nested BEPI structures preserve operational fractality (invariant #7)."""
    import pickle

    # Create nested structure: a list of BEPIElements (simulating fractal nesting)
    elements = [
        _make_bepi(),
        BEPIElement(
            (0.5 + 0.1j, 0.3 + 0.2j, 0.1 + 0.4j),
            (0.2 + 0.1j, 0.4 + 0.3j, 0.6 + 0.0j),
            (0.0, 0.5, 1.0),
        ),
    ]

    # Pickle the nested structure
    pickled = pickle.dumps(elements)
    restored = pickle.loads(pickled)

    # Verify all elements preserved
    assert len(restored) == len(elements)
    for orig, rest in zip(elements, restored):
        np.testing.assert_allclose(rest.f_continuous, orig.f_continuous)
        np.testing.assert_allclose(rest.a_discrete, orig.a_discrete)
        np.testing.assert_allclose(rest.x_grid, orig.x_grid)
