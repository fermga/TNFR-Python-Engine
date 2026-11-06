"""Supplied UM access participates in live grammar without waiving other rules."""

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.operators.grammar_application import enforce_canonical_grammar
from tnfr.operators.grammar_dynamics import (
    enforce_grammar_on_glyph,
    filter_candidates,
    validate_candidate,
)
from tnfr.operators.grammar_types import StructuralGrammarError


def _graph():
    graph = nx.Graph(UM_FUNCTIONAL_LINKS=True, GRAMMAR_REJECTION_MODE="raise")
    graph.add_node("target", EPI=0.5, nu_f=1.0, theta=0.0, glyph_history=["AL", "IL"])
    graph.add_node("old", EPI=0.5, nu_f=1.0, theta=math.pi)
    graph.add_node("partner", EPI=0.5, nu_f=1.0, theta=0.2)
    graph.add_edge("target", "old", weight=0.37)
    return graph


def _state(graph):
    return deepcopy(
        (graph.graph, dict(graph.nodes(data=True)), list(graph.edges(data=True)))
    )


def test_explicit_partner_changes_the_live_u3_read_without_changing_proactive_selection():
    graph = _graph()
    before = _state(graph)
    assert not validate_candidate(graph, "target", "UM").allowed
    assert filter_candidates(graph, "target", ["UM", "IL"]) == ["IL"]
    admitted = validate_candidate(graph, "target", "UM", potential_partner="partner")
    assert admitted.allowed and admitted.violations == []
    assert (
        enforce_canonical_grammar(graph, "target", "UM", potential_partner="partner")
        == "UM"
    )
    assert _state(graph) == before


@pytest.mark.parametrize("invalid", ["phase", "support", "unknown", "self"])
def test_requested_access_is_re_admitted_not_treated_as_phase_permission(invalid):
    graph = _graph()
    graph.nodes["old"]["theta"] = 0.0
    partner = "partner"
    if invalid == "phase":
        graph.nodes[partner]["theta"] = math.pi
    elif invalid == "support":
        graph.add_edge(partner, "old")
    elif invalid == "unknown":
        partner = "missing"
    else:
        partner = "target"
    before = _state(graph)
    assert validate_candidate(graph, "target", "UM").allowed
    candidate = validate_candidate(graph, "target", "UM", potential_partner=partner)
    assert not candidate.allowed
    assert any(violation.rule == "U3" for violation in candidate.violations)
    with pytest.raises(StructuralGrammarError) as error:
        enforce_canonical_grammar(graph, "target", "UM", potential_partner=partner)
    assert error.value.rule == "U3"
    assert _state(graph) == before


def test_supplied_partner_cannot_discharge_u2_debt():
    graph = _graph()
    graph.nodes["target"].update(glyph_history=["OZ"] * 3, _grammar_u2_debt=3)
    before = _state(graph)
    result = validate_candidate(graph, "target", "UM", potential_partner="partner")
    assert not result.allowed
    assert any(violation.rule == "U2" for violation in result.violations)
    with pytest.raises(StructuralGrammarError) as error:
        enforce_grammar_on_glyph(graph, "target", "UM", potential_partner="partner")
    assert error.value.rule == "U2"
    assert _state(graph) == before


@pytest.mark.parametrize("glyph", ["RA", "EN", "AL", "IL", "OZ", "THOL", "invalid"])
def test_potential_request_is_rejected_for_other_grammar_candidates(glyph):
    graph = _graph()
    before = _state(graph)
    with pytest.raises(ValueError, match="only for UM"):
        validate_candidate(graph, "target", glyph, potential_partner="partner")
    with pytest.raises(ValueError, match="only for UM"):
        enforce_canonical_grammar(graph, "target", glyph, potential_partner="partner")
    assert _state(graph) == before


def test_fallback_does_not_convert_rejected_request_into_um_permission():
    graph = _graph()
    graph.graph["GRAMMAR_REJECTION_MODE"] = "fallback"
    graph.nodes["partner"]["theta"] = math.pi
    before = _state(graph)
    assert (
        enforce_canonical_grammar(graph, "target", "UM", potential_partner="partner")
        != "UM"
    )
    assert _state(graph) == before
