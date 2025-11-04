"""Structural tests covering the consolidated runtime validation facade."""

from __future__ import annotations

import networkx as nx

from tnfr.constants import THETA_KEY, VF_KEY, inject_defaults
from tnfr.initialization import init_node_attrs
from tnfr.validation import ValidationOutcome, Validator
from tnfr.validation.runtime import GraphCanonicalValidator, validate_canon

def _build_unstable_graph() -> nx.Graph:
    graph = nx.path_graph(4)
    inject_defaults(graph)
    init_node_attrs(graph, override=True)
    for node in graph.nodes:
        data = graph.nodes[node]
        data["EPI"] = 3.0 if node % 2 == 0 else -3.0
        data[VF_KEY] = 4.0
        data[THETA_KEY] = 5.0
    return graph

def test_graph_validator_emits_validation_outcome() -> None:
    graph = _build_unstable_graph()
    validator = GraphCanonicalValidator()

    assert isinstance(validator, Validator)

    outcome = validator.validate(graph)
    assert isinstance(outcome, ValidationOutcome)
    assert outcome.subject is graph
    assert outcome.passed is True
    assert outcome.summary["clamped"]
    assert outcome.artifacts is not None
    assert "clamped_nodes" in outcome.artifacts

    report = validator.report(outcome)
    assert "validation" in report.lower()

def test_validate_canon_is_consistent_with_graph_validator() -> None:
    graph = _build_unstable_graph()
    outcome = validate_canon(graph)

    assert isinstance(outcome, ValidationOutcome)
    assert outcome.subject is graph
    assert outcome.passed is True
    assert outcome.summary["clamped"]

    validator = GraphCanonicalValidator()
    follow_up = validator.validate(graph)
    assert follow_up.summary["clamped"] == ()
    assert validator.report(follow_up) == "Graph canonical validation passed without adjustments."
