"""Declared finite selector/dispatch controls; no flow or historical producer.

Stored Si/pressure here are supplied unit inputs. The independent continuation
module owns the fresh relational Si source. Own timestamp history and grammar
context are supplied preparation, not an inferred trajectory.
"""

import pickle
from copy import deepcopy
from dataclasses import FrozenInstanceError

import networkx as nx
import pytest

from tnfr.constants import inject_defaults
from tnfr.dynamics import selectors
from tnfr.dynamics.selectors import (
    DefaultGlyphSelector,
    NativeOperatorDecision,
    ParametricGlyphSelector,
    apply_native_operator_batch,
)
from tnfr.operators import grammar_dynamics
from tnfr.operators.grammar_debt import reset_grammar_state_from_history
from tnfr.operators.grammar_types import StructuralGrammarError


def _source(*, count=3, selector=None):
    graph = nx.path_graph(count)
    inject_defaults(graph)
    graph.graph.update(
        _t=2.0,
        RANDOM_SEED=17,
        GRAMMAR_REJECTION_MODE="raise",
        GLYPH_SELECTOR_N_JOBS=1,
        GLYPH_SELECTOR_MARGIN=0.0,
        GLYPH_HYSTERESIS_WINDOW=10,
        history={
            "since_AL": dict.fromkeys(graph, 0),
            "since_EN": dict.fromkeys(graph, 0),
        },
    )
    if selector is not None:
        graph.graph["glyph_selector"] = selector
    for node in graph:
        data = graph.nodes[node]
        data.update(
            EPI=0.5,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            d2EPI=0.0,
            Si=0.1 if node == 0 else 0.9,
            glyph_history=["AL", "IL", "OZ"],
            epi_time_history=[(0.0, 0.25), (1.0, 0.375), (2.0, 0.5)],
        )
        reset_grammar_state_from_history(data)
    return graph


def _run(graph, trace):
    return apply_native_operator_batch(
        graph, graph.graph["history"], decision_trace=trace
    )


def _state(graph):
    return (
        deepcopy(dict(graph.nodes(data=True))),
        tuple(graph.edges(data="weight")),
        deepcopy(graph.graph["history"]),
        graph.graph["_t"],
    )


def test_default_actual_gate_grammar_and_dispatch_are_observed():
    graph = _source()
    trace = []
    assert _run(graph, trace) is None
    assert [item.node for item in trace] == list(graph)
    assert [item.index for item in trace] == [0, 1, 2]
    first = trace[0]
    assert first.selector_kind == first.selection.kind == "default"
    assert first.selector_evaluated is True
    assert first.selection.base_candidate == first.requested_glyph == "ZHIR"
    assert first.selection.normalized_metrics == (0.1, 0.0, 0.0)
    assert first.selection.score is None
    assert first.mutation_certificate.threshold_gate_satisfied
    assert first.mutation_certificate.source == "epi_time_history"
    assert first.mutation_abstention_reason is None
    assert first.post_gate_glyph == first.applied_glyph == "ZHIR"
    assert first.grammar_decision.outcome == "accepted"
    assert first.status == "applied"
    assert graph.nodes[0]["theta"] == 0.25
    assert graph.nodes[0]["glyph_history"][-1] == "ZHIR"
    assert graph.graph["_t"] == 2.0
    with pytest.raises(FrozenInstanceError):
        first.status = "invented"


@pytest.mark.parametrize("selector", [DefaultGlyphSelector, ParametricGlyphSelector])
def test_optional_observation_preserves_the_actual_batch(selector):
    quiet = _source(selector=selector())
    observed = _source(selector=selector())
    _run(quiet, None)
    trace = []
    _run(observed, trace)
    assert _state(quiet) == _state(observed)
    assert len(trace) == len(observed)
    assert observed.graph["glyph_selector"]._preselection is None


@pytest.mark.parametrize("lag,expected", [("AL", "AL"), ("EN", "EN")])
def test_forced_lag_has_no_evaluated_counterfactual(lag, expected):
    graph = _source()
    graph.graph["history"][f"since_{lag}"][0] = graph.graph[f"{lag}_MAX_LAG"]
    if lag == "AL":
        graph.graph["history"]["since_EN"][0] = graph.graph["EN_MAX_LAG"]
    trace = []
    _run(graph, trace)
    first = trace[0]
    assert first.lag_reason == f"{lag.lower()}_lag"
    assert first.selector_evaluated is False and first.selection is None
    assert first.requested_glyph == first.applied_glyph == expected
    assert first.mutation_certificate is None
    assert graph.graph["history"][f"since_{lag}"][0] == 0
    assert trace[1].since_al == trace[1].since_en == 1


def test_parametric_score_override_records_consumed_branch():
    graph = _source(selector=ParametricGlyphSelector())
    # Declared unit policy: zero normalized pressure yields score exactly one.
    graph.graph["SELECTOR_WEIGHTS"] = {"w_si": 0.0, "w_dnfr": 1.0, "w_accel": 0.0}
    trace = []
    _run(graph, trace)
    selection = trace[0].selection
    assert selection.base_candidate == "ZHIR"
    assert selection.hysteresis_candidate is None
    assert selection.score > 0.66
    assert selection.override_candidate == "IL"
    assert selection.soft_filter_candidate == trace[0].requested_glyph == "IL"
    assert trace[0].mutation_certificate is None


def test_hysteresis_short_circuits_score_without_inventing_it():
    graph = _source(selector=ParametricGlyphSelector())
    graph.nodes[0]["Si"] = 0.25
    graph.graph["GLYPH_SELECTOR_MARGIN"] = 0.05
    graph.graph["OZ_NOISE_MODE"] = False
    trace = []
    _run(graph, trace)
    selection = trace[0].selection
    assert selection.base_candidate == "ZHIR"
    assert selection.hysteresis_candidate == "OZ"
    assert selection.score is selection.override_candidate is None
    assert selection.soft_filter_candidate is None
    assert trace[0].requested_glyph == "OZ"


@pytest.mark.parametrize(
    "case,reason",
    [
        ("missing", "missing_history"),
        ("stale", "stale_physical_endpoint"),
        ("threshold", "observed_threshold_not_crossed"),
        ("capacity", "inactive_structural_capacity"),
    ],
)
def test_actual_mutation_abstention_remains_distinct_from_selection(case, reason):
    graph = _source()
    data = graph.nodes[0]
    if case == "missing":
        del data["epi_time_history"]
    elif case == "stale":
        data["epi_time_history"][-1] = (2.0, 0.45)
    elif case == "threshold":
        data["epi_time_history"][-2] = (1.0, 0.45)
    else:
        data["nu_f"] = 0.0
    trace = []
    _run(graph, trace)
    first = trace[0]
    assert first.selection.base_candidate == first.requested_glyph == "ZHIR"
    assert first.mutation_abstention_reason == reason
    assert first.post_gate_glyph == first.applied_glyph == "IL"
    assert first.grammar_decision.candidate == "IL"
    assert first.status == "applied"
    assert graph.nodes[0]["theta"] == 0.0


def test_strict_grammar_failure_retains_completed_prefix_and_unexecuted_proposals():
    graph = _source()
    graph.nodes[1]["Si"] = 0.1
    graph.nodes[1]["glyph_history"] = ["AL", "IL", "EN"]
    reset_grammar_state_from_history(graph.nodes[1])
    trace = []
    with pytest.raises(StructuralGrammarError, match="recent destabilizer"):
        _run(graph, trace)
    assert [item.status for item in trace] == ["applied", "failed", "not_reached"]
    assert trace[0].applied_glyph == "ZHIR"
    assert trace[1].mutation_certificate.threshold_gate_satisfied
    assert trace[1].grammar_decision.outcome == "rejected"
    assert trace[1].failure_stage == "grammar"
    assert trace[1].applied_glyph is None
    assert trace[2].selector_evaluated is True
    assert trace[2].selection.base_candidate == "IL"
    assert trace[2].grammar_decision is trace[2].applied_glyph is None
    assert graph.nodes[0]["theta"] == 0.25
    assert graph.nodes[1]["theta"] == graph.nodes[2]["theta"] == 0.0


def test_same_code_fallback_is_not_reported_as_grammar_acceptance(monkeypatch):
    graph = _source(count=1)
    graph.nodes[0]["Si"] = 0.9
    graph.graph["GRAMMAR_REJECTION_MODE"] = "fallback"
    violation = grammar_dynamics.GrammarViolation("U_test", "unit blocked", "error")
    monkeypatch.setattr(
        grammar_dynamics,
        "_check_violations",
        lambda *args, **kwargs: (False, [violation]),
    )
    trace = []
    _run(graph, trace)
    first = trace[0]
    assert first.requested_glyph == first.applied_glyph == "IL"
    assert first.grammar_decision.outcome == "fallback"
    assert first.grammar_decision.allowed is False
    assert first.grammar_decision.violations == (("U_test", "unit blocked", "error"),)


def test_zero_history_window_cannot_hide_actual_dispatch():
    graph = _source()
    graph.graph["GLYPH_HYSTERESIS_WINDOW"] = 0
    trace = []
    _run(graph, trace)
    assert trace[0].applied_glyph == "ZHIR"
    assert not graph.nodes[0]["glyph_history"]
    assert graph.nodes[0]["theta"] == 0.25


@pytest.mark.parametrize("invalid", [(), {}, True, 1])
def test_invalid_trace_rejects_before_selector_prepare_or_memory(invalid):
    graph = _source()
    before = _state(graph)
    with pytest.raises(TypeError, match="exact list"):
        _run(graph, invalid)
    assert _state(graph) == before
    assert "_sel_norms" not in graph.graph


def test_list_subclasses_cannot_intercept_observation_writes():
    class HookedList(list):
        def extend(self, value):
            pytest.fail("must reject before invoking user list hooks")

    with pytest.raises(TypeError, match="exact list"):
        _run(_source(), HookedList())


def test_invalid_preflight_has_no_invented_target_decision():
    graph = _source()
    graph.graph["AL_MAX_LAG"] = True
    trace = []
    with pytest.raises(TypeError):
        _run(graph, trace)
    assert trace == []


def test_opaque_selector_is_called_once_and_none_is_explicitly_skipped():
    calls = []

    def selector(graph, node):
        calls.append(node)
        return None if node == 1 else "IL"

    graph = _source(selector=selector)
    trace = []
    _run(graph, trace)
    assert calls == list(graph)
    assert all(item.selector_kind == item.selection.kind == "opaque" for item in trace)
    assert all(item.selection.base_candidate is None for item in trace)
    assert trace[1].status == "skipped"
    assert trace[1].applied_glyph is None


def test_replaced_builtin_select_remains_opaque_and_is_not_bypassed(monkeypatch):
    calls = []

    def replacement(self, graph, node):
        calls.append(node)
        return "IL"

    monkeypatch.setattr(DefaultGlyphSelector, "select", replacement)
    graph = _source(selector=DefaultGlyphSelector())
    trace = []
    _run(graph, trace)
    assert calls == list(graph)
    assert all(item.selector_kind == "opaque" for item in trace)


def test_selection_failure_has_no_operator_effect_and_preserves_known_proposals():
    error = RuntimeError("declared selector failure")

    def selector(graph, node):
        if node == 1:
            raise error
        return "IL"

    graph = _source(selector=selector)
    before = deepcopy(dict(graph.nodes(data=True)))
    trace = []
    with pytest.raises(RuntimeError) as caught:
        _run(graph, trace)
    assert caught.value is error
    assert trace[0].selection.selected == "IL"
    assert trace[0].status == trace[2].status == "not_reached"
    assert trace[1].status == "failed" and trace[1].failure_stage == "selection"
    assert dict(graph.nodes(data=True)) == before
    assert graph.graph["history"]["since_AL"] == dict.fromkeys(graph, 1)


def test_direct_dispatch_failure_is_not_claimed_as_no_effect(monkeypatch):
    graph = _source()
    error = RuntimeError("declared partial primitive failure")

    def failure(graph, node, glyph, **kwargs):
        graph.nodes[node]["unit_partial_write"] = True
        raise error

    monkeypatch.setattr(selectors, "apply_glyph", failure)
    trace = []
    with pytest.raises(RuntimeError) as caught:
        _run(graph, trace)
    assert caught.value is error
    assert graph.nodes[0]["unit_partial_write"]
    assert trace[0].status == "failed" and trace[0].failure_stage == "dispatch"
    assert trace[0].applied_glyph is None


def test_worker_transport_preserves_actual_builtin_decisions(monkeypatch):
    class PickleExecutor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, arguments):
            for argument in arguments:
                restored = pickle.loads(pickle.dumps(argument))
                yield pickle.loads(pickle.dumps(function(restored)))

    import tnfr.dynamics as dynamics

    monkeypatch.setattr(dynamics, "ProcessPoolExecutor", PickleExecutor)
    graph = _source(selector=ParametricGlyphSelector())
    graph.graph["GLYPH_SELECTOR_N_JOBS"] = 2
    graph.graph["SELECTOR_WEIGHTS"] = {"w_si": 0.0, "w_dnfr": 1.0, "w_accel": 0.0}
    trace = []
    _run(graph, trace)
    assert [item.node for item in trace] == list(graph)
    assert trace[0].selection.base_candidate == "ZHIR"
    assert trace[0].selection.override_candidate == "IL"
    assert all(item.status == "applied" for item in trace)


def test_existing_trace_prefix_is_retained():
    marker = NativeOperatorDecision("previous", 0, "opaque")
    trace = [marker]
    _run(_source(), trace)
    assert trace[0] is marker and len(trace) == 4


def test_invalid_private_trace_preserves_prepared_builtin_state():
    graph = _source()
    selector = DefaultGlyphSelector()
    selector.prepare(graph, tuple(graph))
    prepared = selector._preselection
    with pytest.raises(TypeError, match="exact list"):
        selectors._apply_glyphs(
            graph, selector, graph.graph["history"], _decision_trace=()
        )
    assert selector._preselection is prepared
    assert graph.graph["history"]["since_AL"] == dict.fromkeys(graph, 0)


def test_unprintable_exception_is_preserved_by_passive_observation():
    class UnprintableError(RuntimeError):
        def __str__(self):
            raise ValueError("message unavailable")

    error = UnprintableError()

    def selector(graph, node):
        raise error

    graph = _source(selector=selector)
    trace = []
    with pytest.raises(UnprintableError) as caught:
        _run(graph, trace)
    assert caught.value is error
    assert trace[0].failure_message == "<exception message unavailable>"
    assert trace[0].failure_stage == "selection"


def test_live_mutation_gate_observes_changes_after_prepared_selection(monkeypatch):
    graph = _source()
    graph.nodes[0]["Si"] = 0.9
    graph.nodes[1]["Si"] = 0.1
    original = selectors.apply_glyph

    def actual_then_invalidate(subject, node, glyph, **kwargs):
        original(subject, node, glyph, **kwargs)
        if node == 0:
            del subject.nodes[1]["epi_time_history"]

    monkeypatch.setattr(selectors, "apply_glyph", actual_then_invalidate)
    trace = []
    _run(graph, trace)
    assert trace[1].selection.base_candidate == "ZHIR"
    assert trace[1].mutation_abstention_reason == "missing_history"
    assert trace[1].applied_glyph == "IL"
    assert trace[1].mutation_certificate.source is None


@pytest.mark.parametrize(
    "selector_type", [DefaultGlyphSelector, ParametricGlyphSelector]
)
@pytest.mark.parametrize("shadow", ["select", "prepare"])
def test_instance_shadowed_builtin_hook_is_opaque_without_changing_call_contract(
    selector_type, shadow
):
    observed_selector = selector_type()
    legacy_selector = selector_type()
    observed_calls, legacy_calls = [], []

    def install(selector, calls):
        if shadow == "select":

            def custom_select(graph, node):
                calls.append(node)
                return "IL"

            selector.select = custom_select
        else:
            canonical_prepare = selector.prepare

            def custom_prepare(graph, nodes):
                calls.append(tuple(nodes))
                canonical_prepare(graph, nodes)

            selector.prepare = custom_prepare

    install(observed_selector, observed_calls)
    install(legacy_selector, legacy_calls)
    observed = _source(selector=observed_selector)
    legacy = _source(selector=legacy_selector)
    _run(legacy, None)
    trace = []
    _run(observed, trace)
    assert observed_calls == legacy_calls
    assert _state(observed) == _state(legacy)
    assert all(item.selector_kind == item.selection.kind == "opaque" for item in trace)
    assert all(item.selection.base_candidate is None for item in trace)
