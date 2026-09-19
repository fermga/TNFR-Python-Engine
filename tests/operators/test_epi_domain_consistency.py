"""Affine operators preserve the exact signed scalar EPI domain."""

from __future__ import annotations

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.errors import TNFRValueError
from tnfr.mathematics import BEPIElement
from tnfr.operators import apply_glyph
from tnfr.operators.definitions import Emission, Reception
from tnfr.types import Glyph, ensure_bepi, real_scalar_epi, serialize_bepi


def _graph(target=0.2, neighbor=0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(
        0,
        EPI=target,
        nu_f=1.0,
        delta_nfr=0.2,
        phase=0.0,
        Si=0.8,
        epi_kind="seed",
    )
    graph.add_node(
        1,
        EPI=neighbor,
        nu_f=1.0,
        delta_nfr=0.1,
        phase=0.1,
        Si=0.8,
        epi_kind="seed",
    )
    graph.add_edge(0, 1)
    return graph


def _rich_bepi() -> BEPIElement:
    return BEPIElement((-0.8, -0.7), (-0.8, -0.8), (0.0, 1.0))


def _plain_state(graph: nx.Graph) -> tuple[object, object]:
    return deepcopy(dict(graph.nodes(data=True))), deepcopy(graph.graph)


@pytest.mark.parametrize("factory", [lambda value: value, ensure_bepi, serialize_bepi])
def test_emission_accepts_all_exact_scalar_representations(factory) -> None:
    graph = _graph(target=factory(-0.2))

    apply_glyph(graph, 0, Glyph.AL)

    assert real_scalar_epi(graph.nodes[0]["EPI"]) > -0.2


@pytest.mark.parametrize("glyph", [Glyph.AL, Glyph.VAL, Glyph.NUL])
@pytest.mark.parametrize("stored", [False, True])
def test_unary_affine_glyphs_reject_rich_epi_before_runtime_metadata(
    glyph: Glyph, stored: bool
) -> None:
    rich = _rich_bepi()
    graph = _graph(target=serialize_bepi(rich) if stored else rich)
    before = _plain_state(graph)

    with pytest.raises(TNFRValueError, match="uniform-real BEPI"):
        apply_glyph(graph, 0, glyph)

    assert _plain_state(graph) == before
    assert "_node_cache" not in graph.graph


@pytest.mark.parametrize(
    ("glyph", "operator"),
    [(Glyph.EN, Reception()), (Glyph.RA, None)],
)
def test_neighbor_affine_glyphs_reject_only_consumed_rich_epi_atomically(
    glyph: Glyph, operator
) -> None:
    graph = _graph(neighbor=serialize_bepi(_rich_bepi()))
    before = _plain_state(graph)

    with pytest.raises(TNFRValueError, match="uniform-real BEPI"):
        if operator is None:
            apply_glyph(graph, 0, glyph)
        else:
            operator(graph, 0)

    assert _plain_state(graph) == before
    assert "_node_cache" not in graph.graph
    assert "_reception_sources" not in graph.nodes[0]


def test_public_emission_rejects_rich_epi_before_lineage_or_history() -> None:
    graph = _graph(target=serialize_bepi(_rich_bepi()))
    before = _plain_state(graph)

    with pytest.raises(TNFRValueError, match="uniform-real BEPI"):
        Emission()(graph, 0)

    assert _plain_state(graph) == before
    assert "_emission_activated" not in graph.nodes[0]
    assert "glyph_history" not in graph.nodes[0]
