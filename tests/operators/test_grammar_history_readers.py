"""Canonical history views are fresh, read-only and shared by their consumers."""

from collections import deque
from copy import deepcopy
from dataclasses import FrozenInstanceError

import networkx as nx
import pytest

from tnfr.errors import TNFRValueError
from tnfr.glyph_history import count_glyphs, push_glyph, recent_glyph
from tnfr.glyph_runtime import last_glyph
from tnfr.operators import grammar_debt
from tnfr.operators.grammar_debt import (
    OPERATOR_STEP_KEY,
    PRIOR_COHERENCE_KEY,
    U2_DEBT_KEY,
    read_grammar_state,
)
from tnfr.operators.preconditions.dissonance import _validate_oz_no_overload
from tnfr.operators.preconditions.mutation import (
    diagnose_mutation_readiness,
    record_destabilizer_context,
    validate_grammar_u4b,
)
from tnfr.selector import _apply_selector_hysteresis
from tnfr.types import Glyph
from tnfr.validation.graph import run_validators


def _graph(history):
    graph = nx.Graph()
    graph.add_node(0, EPI=0.25, nu_f=0.5, delta_nfr=0.1, glyph_history=history)
    return graph


def _hysteresis(node):
    thresholds = {
        "si_hi": 0.8,
        "si_lo": 0.2,
        "dnfr_hi": 0.8,
        "dnfr_lo": 0.2,
        "accel_hi": 0.8,
        "accel_lo": 0.2,
    }
    return _apply_selector_hysteresis(node, 0.8, 0.5, 0.5, thresholds, 0.1)


@pytest.mark.parametrize("container", [list, tuple, deque])
@pytest.mark.parametrize(
    "alias", ["IL", "coherence", "Coherence", "Glyph.IL", Glyph.IL]
)
def test_history_aliases_agree_in_observation_and_hysteresis(container, alias):
    history = container(["emission", alias])
    graph = _graph(history)
    before = deepcopy(dict(graph.nodes[0]))
    assert last_glyph(graph.nodes[0]) == "IL"
    assert recent_glyph(graph.nodes[0], "Coherence", 1)
    assert count_glyphs(graph) == {"AL": 1, "IL": 1}
    assert count_glyphs(graph, window=1) == {"IL": 1}
    assert count_glyphs(graph, last_only=True) == {"IL": 1}
    assert _hysteresis(graph.nodes[0]) == "IL"
    assert dict(graph.nodes[0]) == before
    assert graph.nodes[0]["glyph_history"] is history


@pytest.mark.parametrize("key", ["source_glyph", "last_glyph"])
def test_provenance_fallback_normalizes_an_identifier_without_inventing_history(key):
    node = {key: "Coherence", "epi_kind": "OZ"}
    assert last_glyph(node) == "IL"
    assert "glyph_history" not in node
    assert not read_grammar_state(node).prior_coherence


def test_malformed_authoritative_provenance_cannot_fall_through_to_an_alias():
    with pytest.raises(ValueError, match="canonical operator"):
        last_glyph({"source_glyph": "unknown", "last_glyph": "IL"})


def test_state_view_is_immutable_and_each_read_rechecks_changed_primitives():
    node = {"glyph_history": ["Coherence", "Expansion"]}
    state = read_grammar_state(node)
    assert state.history == ("IL", "VAL")
    assert state.u2_debt == 1
    assert state.prior_coherence
    assert state.operator_step == 2
    with pytest.raises(FrozenInstanceError):
        state.u2_debt = 0
    node["glyph_history"].append("reception")
    assert state.history == ("IL", "VAL")
    assert read_grammar_state(node).history == ("IL", "VAL", "EN")
    node[U2_DEBT_KEY] = True
    with pytest.raises(TypeError, match=U2_DEBT_KEY):
        read_grammar_state(node)


@pytest.mark.parametrize("persisted", [False, True])
def test_recording_admits_each_retained_identifier_once_without_skipping_old_tokens(
    monkeypatch, persisted
):
    history = ["Coherence", "Expansion", "reception"] * 64
    node = {"glyph_history": history}
    if persisted:
        node.update({U2_DEBT_KEY: 4, PRIOR_COHERENCE_KEY: True, OPERATOR_STEP_KEY: 500})
    parsed = []
    original = grammar_debt.glyph_function_name

    def track(value):
        if value != "AL":
            parsed.append(value)
        return original(value)

    monkeypatch.setattr(grammar_debt, "glyph_function_name", track)
    push_glyph(node, "AL", 3)
    assert parsed == history
    assert tuple(node["glyph_history"]) == ("VAL", "EN", "AL")
    assert node[U2_DEBT_KEY] == (4 if persisted else 1)
    assert node[OPERATOR_STEP_KEY] == (501 if persisted else 193)


_READERS = ("last", "count", "hysteresis", "u4b", "context", "readiness", "overload")


def _read(graph, reader):
    node = graph.nodes[0]
    if reader == "last":
        return last_glyph(node)
    if reader == "count":
        return count_glyphs(graph)
    if reader == "hysteresis":
        return _hysteresis(node)
    if reader == "u4b":
        return validate_grammar_u4b(graph, 0)
    if reader == "context":
        return record_destabilizer_context(graph, 0)
    if reader == "overload":
        return _validate_oz_no_overload(graph, 0)
    return diagnose_mutation_readiness(graph, 0)


@pytest.mark.parametrize("reader", _READERS)
@pytest.mark.parametrize("history", [None, "IL", {"IL": 1}, ["IL", "unknown"]])
def test_direct_readers_reject_malformed_history_before_context_writes(reader, history):
    graph = _graph(history)
    graph.nodes[0]["source_glyph"] = "IL"
    before = deepcopy(dict(graph.nodes[0]))
    with pytest.raises((TypeError, ValueError), match="glyph_history"):
        _read(graph, reader)
    assert dict(graph.nodes[0]) == before
    assert graph.nodes[0]["glyph_history"] is history
    assert "_mutation_context" not in graph.nodes[0]


@pytest.mark.parametrize("reader", _READERS)
def test_direct_readers_never_consume_one_shot_history(reader):
    history = (glyph for glyph in ["IL", "OZ"])
    graph = _graph(history)
    with pytest.raises(ValueError, match="replayable"):
        _read(graph, reader)
    assert graph.nodes[0]["glyph_history"] is history
    assert list(history) == ["IL", "OZ"]
    assert "_mutation_context" not in graph.nodes[0]


def test_valid_missing_history_remains_unavailable_context_with_existing_convention():
    graph = _graph([])
    graph.nodes[0].pop("glyph_history")
    result = validate_grammar_u4b(graph, 0)
    assert result == {
        "destabilizer_operator": None,
        "destabilizer_distance": None,
        "recent_history": [],
    }
    assert "_mutation_context" not in graph.nodes[0]
    assert record_destabilizer_context(graph, 0) == result
    assert graph.nodes[0]["_mutation_context"] == result
    assert _validate_oz_no_overload(graph, 0) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("glyph_history", None),
        ("glyph_history", ["unknown"]),
        ("source_glyph", "unknown"),
        (U2_DEBT_KEY, True),
    ],
)
def test_graph_validation_preserves_structured_error_for_invalid_history(field, value):
    graph = _graph([])
    graph.nodes[0][field] = value
    with pytest.raises(TNFRValueError, match="Invalid glyph") as caught:
        run_validators(graph)
    assert caught.value.context["node"] == 0
    assert isinstance(caught.value.__cause__, (TypeError, ValueError))


def test_graph_validation_accepts_normalized_history_without_rewriting_it():
    history = ["Coherence"]
    graph = _graph(history)
    run_validators(graph)
    assert graph.nodes[0]["glyph_history"] is history
    assert history == ["Coherence"]
