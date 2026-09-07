"""Read-only grammar observation reports preserve semantic distinctions."""

from __future__ import annotations

from copy import deepcopy

import networkx as nx

from tnfr.operators.grammar_observations import observe_grammar
from tnfr.operators import Coupling, Silence
from tnfr.types import serialize_bepi


def test_observe_grammar_does_not_mutate_history_and_separates_phase_request():
    graph = nx.Graph()
    graph.add_node(0, EPI=1.0, glyph_history=["AL", "IL"])
    before = list(graph.nodes[0]["glyph_history"])
    report = observe_grammar(graph, 0, ["UM"])
    assert list(graph.nodes[0]["glyph_history"]) == before
    assert report.history_length == 2
    assert report.prior_coherence is True
    assert report.phase_gate_requested is True
    assert report.phase_preconditions_checked is True
    assert report.phase_gate_allowed is True
    assert report.u2_debt == 0


def test_observe_grammar_reports_live_debt_without_reclassifying_u6():
    graph = nx.Graph()
    graph.add_node(0, EPI=1.0, glyph_history=["VAL", "VAL"])
    report = observe_grammar(graph, 0, ["IL"])
    assert report.u2_debt == 2
    assert report.incremental_allowed == (True,)
    assert report.u6_checked is False


def test_observe_grammar_reports_u3_rejection_without_mutation():
    graph = nx.Graph()
    graph.add_edge(0, 1)
    graph.nodes[0].update(EPI=1.0, theta=0.0, glyph_history=["AL", "IL"])
    graph.nodes[1].update(EPI=1.0, theta=3.0, glyph_history=[])
    before = dict(graph.nodes[0])
    report = observe_grammar(graph, 0, [Coupling()])
    assert report.phase_gate_requested
    assert report.phase_preconditions_checked
    assert report.phase_gate_allowed is False
    assert report.incremental_allowed == (False,)
    assert dict(graph.nodes[0]) == before


def test_observe_grammar_reports_declared_recursivity_depth():
    from tnfr.operators import Recursivity

    graph = nx.Graph()
    graph.add_node(0, EPI=1.0, glyph_history=["AL", "IL"])
    report = observe_grammar(graph, 0, [Recursivity(depth=3)])
    assert report.max_recursivity_depth == 3


def test_observe_grammar_keeps_truncated_history_debt_without_prepayment():
    graph = nx.Graph()
    graph.add_node(0, EPI=1.0, glyph_history=["VAL", "VAL"])
    report = observe_grammar(graph, 0, ["EN", "VAL"])
    assert report.u2_debt == 2
    assert report.incremental_allowed == (True, False)


def test_observe_grammar_separates_history_contracts_and_telemetry_presence():
    graph = nx.Graph()
    graph.add_node(0, EPI=1.0, glyph_history=["AL", "OZ", "IL"])
    graph.graph["operator_metrics"] = [{"operator": "coherence"}]
    report = observe_grammar(graph, 0, ["SHA"], u6_checked=True)
    assert report.accepted_history == ("emission", "dissonance", "coherence")
    assert report.recent_destabilizer == "dissonance"
    assert report.recent_destabilizer_distance == 2
    assert report.declared_contracts == ("Silence",)
    assert report.contract_postconditions_checked is False
    assert report.contract_satisfied is None
    assert report.trajectory_telemetry_present is True
    assert report.u6_checked is True


def test_observe_grammar_accepts_explicit_postcondition_result_only():
    graph = nx.Graph()
    graph.add_node(0, EPI=1.0, glyph_history=["AL"])
    report = observe_grammar(graph, 0, ["SHA"], contract_satisfied=True)
    assert report.contract_postconditions_checked is True
    assert report.contract_satisfied is True


def test_observe_grammar_reads_serialized_signed_scalar_epi_without_mutation():
    graph = nx.Graph()
    graph.add_node(0, EPI=serialize_bepi(-0.5), glyph_history=["AL"])
    before = deepcopy(dict(graph.nodes[0]))

    report = observe_grammar(graph, 0, [Silence()])

    assert not report.sequence_message.startswith("grammar validation unavailable")
    assert dict(graph.nodes[0]) == before
