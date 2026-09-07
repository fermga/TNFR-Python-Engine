"""REMESH resolves alpha through the shared glyph-factor contract."""

from __future__ import annotations

from collections import deque

import networkx as nx
import pytest

from tnfr.operators.factor_contracts import GlyphFactorValidationError
from tnfr.operators.remesh import _remesh_alpha_info, apply_network_remesh


def test_remesh_alpha_precedence_is_explicit_and_validated():
    graph = nx.Graph(REMESH_ALPHA=0.25)
    assert _remesh_alpha_info(graph) == (0.25, "REMESH_ALPHA")

    graph.graph["GLYPH_FACTORS"] = {"REMESH_alpha": 0.75}
    assert _remesh_alpha_info(graph) == (
        0.75,
        "GLYPH_FACTORS.REMESH_alpha",
    )

    graph.graph["REMESH_ALPHA_HARD"] = True
    assert _remesh_alpha_info(graph) == (0.25, "REMESH_ALPHA")


def test_remesh_context_does_not_reject_an_unrelated_pending_override():
    graph = nx.Graph(GLYPH_FACTORS={"REMESH_alpha": 0.5, "RA_epi_diff": 2.0})

    assert _remesh_alpha_info(graph)[0] == 0.5


@pytest.mark.parametrize(
    "attributes",
    [
        {"GLYPH_FACTORS": {"REMESH_alpha": -0.1}},
        {"GLYPH_FACTORS": {"REMESH_alpha": 1.1}},
        {"REMESH_ALPHA": float("nan")},
        {"REMESH_ALPHA_HARD": True, "REMESH_ALPHA": float("inf")},
    ],
)
def test_remesh_rejects_invalid_alpha_before_mutating_epi(attributes):
    graph = nx.path_graph(2)
    graph.graph.update(attributes)
    graph.graph.update(REMESH_TAU_GLOBAL=1, REMESH_TAU_LOCAL=1)
    graph.graph["_epi_hist"] = deque(
        [{0: 0.0, 1: 0.0}, {0: 0.5, 1: -0.5}]
    )
    graph.nodes[0]["EPI"] = 0.25
    graph.nodes[1]["EPI"] = -0.25
    before = dict(nx.get_node_attributes(graph, "EPI"))

    with pytest.raises(GlyphFactorValidationError, match="REMESH_alpha"):
        apply_network_remesh(graph)

    assert nx.get_node_attributes(graph, "EPI") == before
