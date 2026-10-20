"""Authoritative grammar history is admitted before decisions or state writes."""

from collections import deque
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.glyph_history import (
    current_operator_step,
    push_glyph,
    recent_glyph,
    record_glyph_occurrence,
)
from tnfr.operators import apply_glyph
from tnfr.operators.definitions import Emission
from tnfr.operators.grammar_application import (
    apply_glyph_with_grammar,
    on_applied_glyph,
)
from tnfr.operators.grammar_debt import (
    OPERATOR_STEP_KEY,
    PRIOR_COHERENCE_KEY,
    U2_DEBT_KEY,
    node_debt,
    node_has_prior_coherence,
    require_grammar_state,
)
from tnfr.operators.grammar_dynamics import (
    validate_candidate,
    validate_sequence_incremental,
)
from tnfr.operators.network_stage import execute_pointwise_stage
from tnfr.types import Glyph


def _graph():
    graph = nx.path_graph(2)
    graph.graph["GRAMMAR_REJECTION_MODE"] = "raise"
    for node in graph:
        graph.nodes[node].update(
            EPI=0.25, nu_f=0.5, theta=0.0, delta_nfr=0.0, glyph_history=["AL", "IL"]
        )
    return graph


@pytest.mark.parametrize(
    "history",
    [
        None,
        "bad",
        b"AL",
        2,
        {"AL": 1},
        {"AL"},
        ["unknown"],
        ["IL", "unknown"],
        [True],
        [None],
    ],
)
@pytest.mark.parametrize(
    "reader", ["candidate", "shadow", "record", "callback", "recent"]
)
def test_malformed_history_neither_grants_context_nor_gets_discarded(history, reader):
    graph = _graph()
    node = graph.nodes[0]
    node.update(EPI=0.0, glyph_history=history)
    before = deepcopy(dict(node))
    with pytest.raises((TypeError, ValueError), match="glyph_history"):
        if reader == "candidate":
            validate_candidate(graph, 0, "IL")
        elif reader == "shadow":
            validate_sequence_incremental(graph, 0, ["IL"])
        elif reader == "record":
            push_glyph(node, "EN", 7)
        elif reader == "callback":
            on_applied_glyph(graph, 0, "EN")
        else:
            recent_glyph(node, "IL", 3)
    assert dict(node) == before
    assert node["glyph_history"] is history


@pytest.mark.parametrize("value", [None, True, -1, 1.5, "3", Fraction(1), np.bool_(1)])
def test_invalid_authoritative_debt_cannot_fall_back_to_a_shorter_trace(value):
    node = {"glyph_history": ["AL"], U2_DEBT_KEY: value}
    before = deepcopy(node)
    for reader in (node_debt, require_grammar_state):
        with pytest.raises((TypeError, ValueError), match=U2_DEBT_KEY):
            reader(node)
    with pytest.raises((TypeError, ValueError), match=U2_DEBT_KEY):
        push_glyph(node, "EN", 7)
    assert node == before


@pytest.mark.parametrize("value", [None, 0, 1, "false", [], np.bool_(True)])
def test_invalid_authoritative_coherence_flag_cannot_be_recovered_from_labels(value):
    node = {"glyph_history": ["IL"], PRIOR_COHERENCE_KEY: value}
    before = deepcopy(node)
    for reader in (node_has_prior_coherence, require_grammar_state):
        with pytest.raises(TypeError, match=PRIOR_COHERENCE_KEY):
            reader(node)
    with pytest.raises(TypeError, match=PRIOR_COHERENCE_KEY):
        push_glyph(node, "EN", 7)
    assert node == before


@pytest.mark.parametrize("container", [list, tuple, deque])
def test_missing_counters_reconstruct_ordered_canonical_aliases(container):
    history = container([Glyph.VAL, "Expansion", "Glyph.IL", "reception"])
    node = {"glyph_history": history}
    assert node_debt(node) == 1
    assert node_has_prior_coherence(node) is True
    assert recent_glyph(node, "coherence", 2)
    assert not recent_glyph(node, "VAL", 2)
    assert set(node) == {"glyph_history"}
    push_glyph(node, "EN", 2)
    assert tuple(node["glyph_history"]) == ("EN", "EN")
    assert node_debt(node) == 1
    assert node_has_prior_coherence(node) is True


@pytest.mark.parametrize("container", [list, tuple, deque])
def test_recording_normalizes_retained_and_new_identifier_aliases(container):
    history = container([Glyph.AL, "Expansion", "Glyph.IL", "reception"])
    node = {"glyph_history": history}
    push_glyph(node, "Dissonance", 8)
    assert tuple(node["glyph_history"]) == ("AL", "VAL", "IL", "EN", "OZ")
    assert all(type(item) is str for item in node["glyph_history"])
    assert current_operator_step(node) == 5
    assert node_debt(node) == 1
    assert node_has_prior_coherence(node) is True


@pytest.mark.parametrize("counter", [0, np.int32(2), np.int64(2), np.uint64(2)])
def test_valid_integral_debt_remains_authoritative_after_trace_eviction(counter):
    node = {"glyph_history": [], U2_DEBT_KEY: counter, PRIOR_COHERENCE_KEY: False}
    assert node_debt(node) == int(counter)
    push_glyph(node, "IL", 0)
    assert node_debt(node) == max(0, int(counter) - 1)
    assert node_has_prior_coherence(node) is True


@pytest.mark.parametrize("glyph", ["unknown", True, None, ["IL"]])
def test_recording_rejects_an_invalid_new_identifier_before_any_write(glyph):
    node = {"glyph_history": deque(["AL"], maxlen=1)}
    history = node["glyph_history"]
    before = deepcopy(node)
    with pytest.raises(ValueError, match="canonical operator identifiers"):
        push_glyph(node, glyph, 7)
    assert node == before
    assert node["glyph_history"] is history


@pytest.mark.parametrize("reader", ["preflight", "record", "occurrence", "recent"])
def test_one_shot_history_rejects_without_consumption_or_replacement(reader):
    history = (glyph for glyph in ["AL", "IL"])
    node = {"glyph_history": history}
    with pytest.raises(ValueError, match="replayable"):
        if reader == "preflight":
            require_grammar_state(node)
        elif reader == "record":
            push_glyph(node, "AL", 7)
        elif reader == "occurrence":
            record_glyph_occurrence(node, "AL")
        else:
            recent_glyph(node, "IL", 3)
    assert node == {"glyph_history": history}
    assert list(history) == ["AL", "IL"]


@pytest.mark.parametrize(
    "path", ["primitive", "public", "multiple", "stage", "callback"]
)
@pytest.mark.parametrize(
    "field,value",
    [
        ("glyph_history", ["unknown"]),
        (U2_DEBT_KEY, -1),
        (PRIOR_COHERENCE_KEY, 1),
        (OPERATOR_STEP_KEY, -1),
    ],
)
def test_execution_preflight_rejects_invalid_state_before_form_or_metadata(
    path, field, value
):
    graph = _graph()
    target = 1 if path in {"multiple", "stage"} else 0
    graph.nodes[target][field] = value
    before = deepcopy((dict(graph.nodes(data=True)), dict(graph.graph)))
    with pytest.raises((TypeError, ValueError)):
        if path == "primitive":
            apply_glyph(graph, 0, "AL")
        elif path == "public":
            Emission()(graph, 0)
        elif path == "multiple":
            apply_glyph_with_grammar(graph, (0, 1), "AL")
        elif path == "callback":
            on_applied_glyph(graph, 0, "AL")
        else:
            execute_pointwise_stage(
                graph, Emission(), (0, 1), timestamp="2030-01-02T03:04:05+00:00"
            )
    assert (dict(graph.nodes(data=True)), dict(graph.graph)) == before
    assert "_node_cache" not in graph.graph


def test_absent_history_remains_a_valid_uninitialized_state():
    graph = _graph()
    graph.nodes[0].pop("glyph_history")
    graph.nodes[0]["EPI"] = 0.0
    require_grammar_state(graph.nodes[0])
    assert node_debt(graph.nodes[0]) == 0
    assert node_has_prior_coherence(graph.nodes[0]) is False
    assert not validate_candidate(graph, 0, "IL").allowed
    assert validate_candidate(graph, 0, "AL").allowed
    assert "glyph_history" not in graph.nodes[0]


@pytest.mark.parametrize("value", [None, True, -1, 1.5, "2", Fraction(1), np.bool_(1)])
def test_invalid_authoritative_operator_index_cannot_restart_from_trace_length(value):
    node = {"glyph_history": ["AL"], OPERATOR_STEP_KEY: value}
    before = deepcopy(node)
    for reader in (current_operator_step, require_grammar_state):
        with pytest.raises((TypeError, ValueError), match=OPERATOR_STEP_KEY):
            reader(node)
    with pytest.raises((TypeError, ValueError), match=OPERATOR_STEP_KEY):
        push_glyph(node, "EN", 7)
    assert node == before


@pytest.mark.parametrize("counter", [np.int32(0), np.int64(7), np.uint64(7)])
def test_operator_index_admits_integral_values_and_preserves_retained_length(counter):
    node = {"glyph_history": ["AL", "IL"], OPERATOR_STEP_KEY: counter}
    expected = max(int(counter), 2)
    assert current_operator_step(node) == expected
    push_glyph(node, "reception", 1)
    assert node[OPERATOR_STEP_KEY] == expected + 1
    assert tuple(node["glyph_history"]) == ("EN",)


@pytest.mark.parametrize(
    "kind", ["missing", "list", "tuple", "deque", "bounded", "zero"]
)
def test_explicit_occurrence_recording_preserves_retention_and_does_not_execute(kind):
    graph = _graph()
    node = graph.nodes[0]
    node.pop("glyph_history")
    if kind == "list":
        node["glyph_history"] = [Glyph.AL]
    elif kind == "tuple":
        node["glyph_history"] = (Glyph.AL,)
    elif kind == "deque":
        node["glyph_history"] = deque([Glyph.AL])
    elif kind == "bounded":
        node["glyph_history"] = deque([Glyph.AL], maxlen=2)
    elif kind == "zero":
        node["glyph_history"] = deque(maxlen=0)
    initial_count = len(node.get("glyph_history", ()))
    physical_state = {key: node[key] for key in ("EPI", "nu_f", "theta", "delta_nfr")}
    for alias in ("Coherence", "Expansion", "reception"):
        on_applied_glyph(graph, 0, alias)
    expected = (["AL"] if initial_count else []) + ["IL", "VAL", "EN"]
    if kind == "bounded":
        expected = expected[-2:]
        assert node["glyph_history"].maxlen == 2
    elif kind == "zero":
        expected = []
        assert node["glyph_history"].maxlen == 0
    elif kind == "deque":
        assert node["glyph_history"].maxlen is None
    else:
        assert isinstance(node["glyph_history"], list)
    assert list(node["glyph_history"]) == expected
    assert all(type(item) is str for item in node["glyph_history"])
    assert current_operator_step(node) == initial_count + 3
    assert node_debt(node) == 1
    assert node_has_prior_coherence(node) is True
    assert {key: node[key] for key in physical_state} == physical_state


def test_explicit_occurrence_rejects_unknown_identifier_without_partial_record():
    graph = _graph()
    before = deepcopy(dict(graph.nodes[0]))
    with pytest.raises(ValueError, match="canonical operator identifiers"):
        on_applied_glyph(graph, 0, "unknown")
    assert dict(graph.nodes[0]) == before
