"""Canonical signed scalar projection for graph-carried EPI."""

from __future__ import annotations

from types import MappingProxyType

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_EPI
from tnfr.mathematics import BEPIElement
from tnfr.node import NodeNX
from tnfr.physics.structural_diffusion import structural_field
from tnfr.types import (
    deserialize_bepi_json,
    ensure_bepi,
    real_scalar_epi,
    scalarize_epi,
    serialize_bepi,
    serialize_bepi_json,
)


@pytest.mark.parametrize("value", [-0.8, 0.0, 0.8])
def test_scalar_embedding_keeps_sign_across_runtime_and_storage(value: float) -> None:
    element = ensure_bepi(value)
    storage = serialize_bepi(element)
    json_storage = serialize_bepi_json(element)

    assert element.real_scalar_embedding() == value
    assert real_scalar_epi(storage) == value
    assert float(element) == value
    assert scalarize_epi(storage) == value
    assert scalarize_epi(json_storage) == value
    assert float(deserialize_bepi_json(json_storage)) == value

    graph = nx.Graph()
    graph.add_node(0, EPI=value)
    node = NodeNX.from_graph(graph, 0)
    assert float(node.EPI) == value

    node.EPI = value
    assert get_attr(graph.nodes[0], ALIAS_EPI) == value
    assert graph.nodes[0]["EPI"] == storage


def test_richer_bepi_retains_maximum_magnitude_projection() -> None:
    nonuniform = BEPIElement((-2.0, -2.0), (-2.0, -1.0), (0.0, 1.0))
    complex_uniform = BEPIElement((3.0 + 4.0j,) * 2, (3.0 + 4.0j,) * 2, (0.0, 1.0))

    assert nonuniform.real_scalar_embedding() is None
    assert complex_uniform.real_scalar_embedding() is None
    assert float(nonuniform) == 2.0
    assert float(complex_uniform) == 5.0
    assert scalarize_epi(serialize_bepi(nonuniform)) == 2.0
    assert scalarize_epi(serialize_bepi(complex_uniform)) == 5.0


def test_numeric_comparison_uses_scalar_projection_but_abs_remains_a_norm() -> None:
    negative = ensure_bepi(-0.8)
    nonuniform = BEPIElement((-2.0, -2.0), (-2.0, -1.0), (0.0, 1.0))

    assert negative == -0.8
    assert negative != 0.8
    assert abs(negative) == 0.8
    assert nonuniform == 2.0
    assert nonuniform != -2.0
    assert abs(nonuniform) == 2.0


def test_missing_bepi_component_does_not_claim_a_scalar_embedding() -> None:
    no_discrete_component = BEPIElement((-2.0, -2.0), (), (0.0, 1.0))

    assert no_discrete_component.real_scalar_embedding() is None
    assert float(no_discrete_component) == 2.0
    assert abs(no_discrete_component) == 2.0


@pytest.mark.parametrize(
    "payload_factory",
    [
        lambda value: value,
        ensure_bepi,
        serialize_bepi,
        serialize_bepi_json,
        lambda value: MappingProxyType(serialize_bepi(value)),
    ],
)
def test_structural_field_accepts_every_signed_scalar_representation(
    payload_factory,
) -> None:
    graph = nx.Graph()
    graph.add_node(0, EPI=payload_factory(-0.8))

    assert structural_field(graph).tolist() == [-0.8]


@pytest.mark.parametrize(
    "payload",
    [
        BEPIElement((-0.8, -0.7), (-0.8, -0.8), (0.0, 1.0)),
        BEPIElement((1.0j, 1.0j), (1.0j, 1.0j), (0.0, 1.0)),
    ],
)
def test_structural_field_rejects_non_scalar_bepi(payload: BEPIElement) -> None:
    graph = nx.Graph()
    graph.add_node(0, EPI=payload)

    with pytest.raises(ValueError, match="uniform real BEPI embedding"):
        structural_field(graph)
