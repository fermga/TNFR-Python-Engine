"""The native trace observes the actual grammar decision without replay."""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import networkx as nx
import pytest

from tnfr.operators import grammar_dynamics as owner
from tnfr.operators.grammar_application import enforce_canonical_grammar as adapter
from tnfr.operators.grammar_types import StructuralGrammarError
from tnfr.types import Glyph
from tnfr.validation import enforce_canonical_grammar as public_adapter


def _source(mode="raise"):
    graph = nx.Graph(GRAMMAR_REJECTION_MODE=mode)
    graph.add_node(
        "core",
        EPI=0.5,
        nu_f=1.0,
        theta=0.0,
        delta_nfr=0.0,
        glyph_history=["AL"],
    )
    return graph


@pytest.mark.parametrize(
    "consumer", [owner.enforce_grammar_on_glyph, adapter, public_adapter]
)
def test_accepted_observation_uses_one_actual_read_without_state_changes(
    monkeypatch, consumer
):
    graph = _source()
    before = deepcopy((graph.graph, dict(graph.nodes)))
    calls = []
    original = owner._check_violations

    def count(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "_check_violations", count)
    trace = []
    result = consumer(graph, "core", "IL", _decision_trace=trace)
    assert result == "IL" and calls == ["IL"]
    assert len(trace) == 1
    observed = trace[0]
    assert observed.allowed and observed.selected == observed.candidate == "IL"
    assert observed.outcome == "accepted" and observed.rejection_mode == "raise"
    assert observed.violations == ()
    assert (graph.graph, dict(graph.nodes)) == before
    with pytest.raises(FrozenInstanceError):
        observed.selected = "ZHIR"


@pytest.mark.parametrize(
    "consumer", [owner.enforce_grammar_on_glyph, adapter, public_adapter]
)
def test_strict_rejection_retains_actual_violations_and_no_selected_action(consumer):
    graph = _source()
    trace = []
    with pytest.raises(StructuralGrammarError):
        consumer(graph, "core", "ZHIR", _decision_trace=trace)
    assert len(trace) == 1
    observed = trace[0]
    assert not observed.allowed and observed.selected is None
    assert observed.candidate == "ZHIR" and observed.outcome == "rejected"
    assert any(
        rule == "U4b" and severity == "error"
        for rule, _, severity in observed.violations
    )
    assert graph.nodes["core"]["glyph_history"] == ["AL"]


def test_fallback_records_blocked_candidate_and_actual_alternative():
    graph = _source("fallback")
    trace = []
    returned = public_adapter(graph, "core", "ZHIR", _decision_trace=trace)
    assert returned == "IL"
    assert trace[0].candidate == "ZHIR" and not trace[0].allowed
    assert trace[0].selected == returned and trace[0].outcome == "fallback"
    assert trace[0].rejection_mode == "fallback"


def test_same_code_fallback_does_not_become_false_approval(monkeypatch):
    # Exercise the shared fallback's exhausted-alternative branch. This is a
    # synthetic grammar-owner control, not a realizability claim about a source.
    graph = _source("fallback")
    violations = [owner.GrammarViolation("test-rule", "blocked", "error")]
    monkeypatch.setattr(owner, "_check_violations", lambda *a, **k: (False, violations))
    trace = []
    assert (
        owner.enforce_grammar_on_glyph(graph, "core", "IL", _decision_trace=trace)
        == "IL"
    )
    assert trace[0].candidate == trace[0].selected == "IL"
    assert not trace[0].allowed and trace[0].outcome == "fallback"
    violations[0].message = "subsequently changed"
    assert trace[0].violations == (("test-rule", "blocked", "error"),)


def test_alternative_failure_preserves_original_exception_and_known_rejection(
    monkeypatch,
):
    graph = _source("fallback")
    failure = RuntimeError("alternative unavailable")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(owner, "suggest_alternative", fail)
    trace = []
    with pytest.raises(RuntimeError) as caught:
        owner.enforce_grammar_on_glyph(graph, "core", "ZHIR", _decision_trace=trace)
    assert caught.value is failure
    assert not trace[0].allowed and trace[0].selected is None
    assert trace[0].outcome == "alternative_failed"


@pytest.mark.parametrize("bad", [(), {}, True])
def test_invalid_sink_rejects_before_grammar_read(monkeypatch, bad):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid observer reached the decision")

    monkeypatch.setattr(owner, "_check_violations", unexpected)
    with pytest.raises(TypeError, match="exact list"):
        public_adapter(_source(), "core", "IL", _decision_trace=bad)


def test_list_subclass_is_not_an_executable_observer():
    class Sink(list):
        def append(self, value):
            pytest.fail("custom append must never run")

    with pytest.raises(TypeError, match="exact list"):
        adapter(_source(), "core", "IL", _decision_trace=Sink())


def test_no_decision_is_fabricated_when_policy_is_invalid():
    graph = _source(True)
    trace = []
    with pytest.raises(TypeError, match="GRAMMAR_REJECTION_MODE"):
        public_adapter(graph, "core", "IL", _decision_trace=trace)
    assert trace == []


def test_public_glyph_output_and_existing_trace_entries_are_preserved():
    marker = object()
    trace = [marker]
    assert (
        public_adapter(_source(), "core", Glyph.IL, _decision_trace=trace) is Glyph.IL
    )
    assert trace[0] is marker and trace[1].selected == "IL"
    assert public_adapter(_source(), "core", Glyph.IL) is Glyph.IL


def test_disabled_capture_preserves_legacy_forwarding_signature(monkeypatch):
    import tnfr.validation as validation

    calls = []

    def existing(graph, node, candidate, context):
        calls.append((node, candidate, context))
        return "IL"

    monkeypatch.setattr(validation, "_ENFORCE_CANONICAL_GRAMMAR", existing)
    assert public_adapter(_source(), "core", "IL") == "IL"
    assert calls == [("core", "IL", None)]
