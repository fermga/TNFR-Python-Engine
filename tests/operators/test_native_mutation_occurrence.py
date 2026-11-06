"""Finite native occurrence contracts; no flow or historical fixture imports.

Scalar metrics, own timestamped samples and grammar context are supplied unit
inputs. The independent participation file owns fresh relational diagnostics.
"""

import math
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants import inject_defaults
from tnfr.dynamics import selectors
from tnfr.dynamics._mutation_occurrence import (
    CONSUME_POLICY,
    CONSUMED_INTERVALS_KEY,
    POLICY_KEY,
)
from tnfr.operators import Mutation, apply_glyph
from tnfr.operators.grammar_debt import reset_grammar_state_from_history
from tnfr.operators.grammar_types import StructuralGrammarError
from tnfr.operators.network_stage import execute_pointwise_stage
from tnfr.operators.preconditions import OperatorPreconditionError

KEY = ("epi_time_history", 1.0, 2.0)


def _source(count=1, *, policy=CONSUME_POLICY):
    graph = nx.path_graph(count)
    inject_defaults(graph)
    graph.graph.update(
        _t=2.0,
        RANDOM_SEED=17,
        GRAMMAR_REJECTION_MODE="raise",
        GLYPH_SELECTOR_N_JOBS=1,
        GLYPH_HYSTERESIS_WINDOW=10,
        AL_MAX_LAG=100,
        EN_MAX_LAG=100,
        history={
            "since_AL": dict.fromkeys(graph, 0),
            "since_EN": dict.fromkeys(graph, 0),
        },
    )
    if policy is not None:
        graph.graph[POLICY_KEY] = policy
    for node in graph:
        data = graph.nodes[node]
        data.update(
            EPI=0.5,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            d2EPI=0.0,
            Si=0.1,
            glyph_history=["IL", "OZ", "IL"],
            epi_time_history=[(0.0, 0.0), (1.0, 0.25), (2.0, 0.5)],
        )
        reset_grammar_state_from_history(data)
    return graph


def _run(graph):
    trace = []
    selectors.apply_native_operator_batch(
        graph, graph.graph["history"], decision_trace=trace
    )
    return trace


def _nodes(graph):
    return deepcopy(dict(graph.nodes(data=True)))


@pytest.mark.parametrize("policy", [None, "level", CONSUME_POLICY])
def test_two_successive_requests_distinguish_existing_level_from_consumption(policy):
    graph = _source(policy=policy)
    first, second = _run(graph)[0], _run(graph)[0]
    assert first.applied_glyph == "ZHIR"
    assert second.requested_glyph == "ZHIR"
    if policy == CONSUME_POLICY:
        assert first.mutation_interval == second.mutation_interval == KEY
        assert first.mutation_interval_previously_consumed is False
        assert first.mutation_interval_committed
        assert second.mutation_interval_previously_consumed is True
        assert not second.mutation_interval_committed
        assert second.mutation_abstention_reason == "mutation_interval_already_consumed"
        assert second.post_gate_glyph == second.applied_glyph == "IL"
        assert graph.nodes[0]["theta"] == 0.25
        assert graph.nodes[0][CONSUMED_INTERVALS_KEY] == (KEY,)
    else:
        assert (
            first.mutation_occurrence_policy
            == second.mutation_occurrence_policy
            == "level"
        )
        assert first.mutation_interval is second.mutation_interval is None
        assert not first.mutation_interval_committed
        assert second.applied_glyph == "ZHIR"
        assert graph.nodes[0]["theta"] == 0.5
        assert CONSUMED_INTERVALS_KEY not in graph.nodes[0]
    assert graph.graph["_t"] == 2.0


@pytest.mark.parametrize("bad", [True, False, None, 0, (), "once", "LEVEL"])
def test_invalid_policy_rejects_before_selector_constructor_or_lags(bad):
    graph = _source()
    graph.graph[POLICY_KEY] = bad

    class Forbidden(selectors.DefaultGlyphSelector):
        def __init__(self):
            pytest.fail("invalid policy reached selector constructor")

    graph.graph["glyph_selector"] = Forbidden
    before, history = _nodes(graph), deepcopy(graph.graph["history"])
    with pytest.raises(ValueError, match=POLICY_KEY):
        _run(graph)
    assert _nodes(graph) == before and graph.graph["history"] == history


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        {},
        ([],),
        (("epi_time_history", True, 2.0),),
        (("epi_time_history", 0.0, math.nan),),
        (("epi_time_history", 0.0, math.inf),),
        (("epi_time_history", 2.0, 1.0),),
        (("epi_time_history", 1.0, 1.0),),
        (("epi_history", 1.0, 2.0),),
        (("epi_time_history", 10**400, 2.0),),
        (("epi_time_history", Fraction(1, 10**400), 2.0),),
        (KEY, ("epi_time_history", 1, 2)),
    ],
)
def test_all_consumed_memory_is_admitted_before_prepare_even_for_other_target(bad):
    graph = _source(count=2)
    graph.nodes[1][CONSUMED_INTERVALS_KEY] = bad
    graph.nodes[1]["Si"] = 0.9

    class Forbidden(selectors.DefaultGlyphSelector):
        def prepare(self, *args):
            pytest.fail("invalid ledger reached preparation")

    graph.graph["glyph_selector"] = Forbidden()
    history = deepcopy(graph.graph["history"])
    with pytest.raises((TypeError, ValueError)):
        _run(graph)
    assert graph.graph["history"] == history
    assert graph.nodes[0]["theta"] == graph.nodes[1]["theta"] == 0.0


def test_level_policy_does_not_read_or_rewrite_an_unused_ledger():
    graph = _source(policy="level")
    opaque = object()
    graph.nodes[0][CONSUMED_INTERVALS_KEY] = opaque
    assert _run(graph)[0].applied_glyph == "ZHIR"
    assert graph.nodes[0][CONSUMED_INTERVALS_KEY] is opaque


@pytest.mark.parametrize("source", ["epi_history", "_epi_history"])
@pytest.mark.parametrize("policy", ["level", CONSUME_POLICY])
def test_legacy_unit_steps_are_not_invented_timestamp_identities(source, policy):
    graph = _source(policy=policy)
    del graph.nodes[0]["epi_time_history"]
    graph.nodes[0][source] = [0.25, 0.5]
    decision = _run(graph)[0]
    assert decision.mutation_certificate.threshold_gate_satisfied
    if policy == CONSUME_POLICY:
        assert decision.applied_glyph == "IL"
        assert decision.mutation_abstention_reason == "unidentified_mutation_interval"
    else:
        assert decision.applied_glyph == "ZHIR"
    assert decision.mutation_interval is None
    assert CONSUMED_INTERVALS_KEY not in graph.nodes[0]


def test_values_threshold_clock_and_phase_changes_do_not_renew_interval():
    graph = _source()
    _run(graph)
    graph.nodes[0]["epi_time_history"][-2] = (1.0, 0.125)
    graph.graph["ZHIR_THRESHOLD_XI"] = 0.2
    graph.graph["_t"] = 99.0
    graph.nodes[0]["theta"] = 1.0
    result = _run(graph)[0]
    assert result.mutation_certificate.observed_depi_dt == 0.375
    assert result.mutation_interval == KEY
    assert result.applied_glyph == "IL" and not result.mutation_interval_committed


def test_new_supplied_times_are_new_key_but_a_b_a_replay_is_not():
    graph = _source()
    _run(graph)
    graph.nodes[0]["epi_time_history"] = [(2.0, 0.25), (3.0, 0.5)]
    second = _run(graph)[0]
    assert second.applied_glyph == "ZHIR" and second.mutation_interval_committed
    second_key = ("epi_time_history", 2.0, 3.0)
    assert graph.nodes[0][CONSUMED_INTERVALS_KEY] == (KEY, second_key)
    graph.nodes[0]["epi_time_history"] = [(1.0, 0.25), (2.0, 0.5)]
    replay = _run(graph)[0]
    assert replay.applied_glyph == "IL" and replay.mutation_interval == KEY
    assert graph.nodes[0][CONSUMED_INTERVALS_KEY] == (KEY, second_key)


def test_receipts_are_own_node_state_not_global_interval_claims():
    graph = _source(count=2)
    trace = _run(graph)
    assert [row.applied_glyph for row in trace] == ["ZHIR", "ZHIR"]
    assert all(row.mutation_interval_committed for row in trace)
    assert (
        graph.nodes[0][CONSUMED_INTERVALS_KEY]
        == graph.nodes[1][CONSUMED_INTERVALS_KEY]
        == (KEY,)
    )


def test_mode_switch_and_bounded_history_do_not_erase_consumption():
    graph = _source()
    _run(graph)
    graph.graph[POLICY_KEY] = "level"
    assert _run(graph)[0].applied_glyph == "ZHIR"
    graph.graph[POLICY_KEY] = CONSUME_POLICY
    graph.graph["GLYPH_HYSTERESIS_WINDOW"] = 0
    final = _run(graph)[0]
    assert final.applied_glyph == "IL"
    assert not graph.nodes[0]["glyph_history"]
    assert graph.nodes[0][CONSUMED_INTERVALS_KEY] == (KEY,)


@pytest.mark.parametrize("failure", ["missing", "threshold", "inactive", "grammar"])
def test_rejected_or_substituted_request_does_not_consume(failure):
    graph = _source()
    if failure == "missing":
        del graph.nodes[0]["epi_time_history"]
    elif failure == "threshold":
        graph.nodes[0]["epi_time_history"][-2] = (1.0, 0.5)
    elif failure == "inactive":
        graph.nodes[0]["nu_f"] = 0.0
    else:
        graph.nodes[0]["glyph_history"] = ["IL"]
        reset_grammar_state_from_history(graph.nodes[0])
    if failure == "grammar":
        with pytest.raises(StructuralGrammarError):
            _run(graph)
    else:
        assert _run(graph)[0].applied_glyph == "IL"
    assert CONSUMED_INTERVALS_KEY not in graph.nodes[0]


def test_dispatch_failure_does_not_consume_and_admitted_retry_can_succeed(monkeypatch):
    graph = _source()
    original = selectors.apply_glyph
    error = RuntimeError("declared dispatch failure")

    def fail(subject, node, glyph, **kwargs):
        subject.nodes[node]["partial_unit_effect"] = True
        raise error

    monkeypatch.setattr(selectors, "apply_glyph", fail)
    with pytest.raises(RuntimeError) as caught:
        _run(graph)
    assert caught.value is error and graph.nodes[0]["partial_unit_effect"]
    assert CONSUMED_INTERVALS_KEY not in graph.nodes[0]
    monkeypatch.setattr(selectors, "apply_glyph", original)
    assert _run(graph)[0].mutation_interval_committed


def test_later_grammar_failure_keeps_successful_prefix_consumed():
    graph = _source(count=2)
    graph.nodes[1]["glyph_history"] = ["IL"]
    reset_grammar_state_from_history(graph.nodes[1])
    trace = []
    with pytest.raises(StructuralGrammarError):
        selectors.apply_native_operator_batch(
            graph, graph.graph["history"], decision_trace=trace
        )
    assert trace[0].applied_glyph == "ZHIR" and trace[0].mutation_interval_committed
    assert graph.nodes[0][CONSUMED_INTERVALS_KEY] == (KEY,)
    assert CONSUMED_INTERVALS_KEY not in graph.nodes[1]
    graph.nodes[1]["glyph_history"] = ["IL", "OZ", "IL"]
    reset_grammar_state_from_history(graph.nodes[1])
    retry = _run(graph)
    assert [row.applied_glyph for row in retry] == ["IL", "ZHIR"]


@pytest.mark.parametrize("hook", ["prepare", "select"])
def test_hook_modified_consumption_memory_is_readmitted_before_any_dispatch(hook):
    graph = _source()

    class Hook(selectors.DefaultGlyphSelector):
        def prepare(self, subject, nodes):
            super().prepare(subject, nodes)
            if hook == "prepare":
                subject.nodes[0][CONSUMED_INTERVALS_KEY] = None

        def select(self, subject, node):
            if hook == "select":
                subject.nodes[0][CONSUMED_INTERVALS_KEY] = None
            return super().select(subject, node)

    graph.graph["glyph_selector"] = Hook()
    with pytest.raises(TypeError, match=CONSUMED_INTERVALS_KEY):
        _run(graph)
    assert graph.nodes[0]["theta"] == 0.0
    assert graph.graph["history"]["since_AL"][0] == (0 if hook == "prepare" else 1)


@pytest.mark.parametrize("requested", ["ZHIR", "IL"])
def test_final_actual_zhir_consumes_even_when_grammar_supplies_it(
    monkeypatch, requested
):
    graph = _source()
    if requested == "IL":
        graph.nodes[0]["Si"] = 0.9
    # Synthetic adapter control of final-glyph admission; no new grammar law.
    monkeypatch.setattr(selectors, "enforce_canonical_grammar", lambda *a, **k: "ZHIR")
    first = _run(graph)[0]
    assert first.requested_glyph == requested and first.applied_glyph == "ZHIR"
    assert first.mutation_interval == KEY and first.mutation_interval_committed
    with pytest.raises(OperatorPreconditionError, match="already_consumed"):
        _run(graph)
    assert graph.nodes[0]["theta"] == 0.25


def test_grammar_changed_evidence_is_readmitted_before_actual_zhir(monkeypatch):
    graph = _source()
    graph.nodes[0][CONSUMED_INTERVALS_KEY] = (("epi_time_history", 2.0, 3.0),)
    calls = []

    def grammar(subject, node, glyph, **kwargs):
        calls.append(glyph)
        subject.nodes[node]["epi_time_history"] = [(2.0, 0.25), (3.0, 0.5)]
        return "ZHIR"

    monkeypatch.setattr(selectors, "enforce_canonical_grammar", grammar)
    with pytest.raises(OperatorPreconditionError, match="already_consumed"):
        _run(graph)
    assert calls == ["ZHIR"] and graph.nodes[0]["theta"] == 0.0


@pytest.mark.parametrize("path", ["direct", "public", "stage"])
def test_native_policy_does_not_change_explicit_intervention_paths(path):
    graph = _source()
    if path == "direct":
        apply_glyph(graph, 0, "ZHIR")
    elif path == "public":
        Mutation()(graph, 0)
    else:
        execute_pointwise_stage(graph, Mutation(), (0,), require_exact_operator=True)
    assert graph.nodes[0]["theta"] == 0.25
    assert CONSUMED_INTERVALS_KEY not in graph.nodes[0]


def test_external_memory_mutation_during_dispatch_rejects_receipt_after_effect(
    monkeypatch,
):
    graph = _source()
    original = selectors.apply_glyph
    other_key = ("epi_time_history", -1.0, 0.0)

    def external_mutation(subject, node, glyph, **kwargs):
        original(subject, node, glyph, **kwargs)
        subject.nodes[node][CONSUMED_INTERVALS_KEY] = (other_key,)

    monkeypatch.setattr(selectors, "apply_glyph", external_mutation)
    trace = []
    with pytest.raises(RuntimeError, match="memory changed"):
        selectors.apply_native_operator_batch(
            graph, graph.graph["history"], decision_trace=trace
        )
    assert graph.nodes[0]["theta"] == 0.25
    assert graph.nodes[0][CONSUMED_INTERVALS_KEY] == (other_key,)
    assert trace[0].applied_glyph == "ZHIR"
    assert not trace[0].mutation_interval_committed


def test_consumed_container_subclasses_are_rejected_without_running_hooks():
    class BadTuple(tuple):
        def __iter__(self):
            pytest.fail("ledger hook must not run")

    graph = _source()
    graph.nodes[0][CONSUMED_INTERVALS_KEY] = BadTuple((KEY,))
    with pytest.raises(TypeError, match="exact tuple"):
        _run(graph)


@pytest.mark.parametrize("later_rejection", [False, True])
def test_opt_in_trace_is_passive_across_two_calls_and_a_committed_prefix(
    later_rejection,
):
    plain, observed = _source(count=2), _source(count=2)
    if later_rejection:
        for graph in (plain, observed):
            graph.nodes[1]["glyph_history"] = ["IL"]
            reset_grammar_state_from_history(graph.nodes[1])
    for call in range(2):
        trace = []
        failures = []
        for graph, sink in ((plain, None), (observed, trace)):
            try:
                selectors.apply_native_operator_batch(
                    graph, graph.graph["history"], decision_trace=sink
                )
            except StructuralGrammarError as exc:
                failures.append((type(exc), str(exc)))
        assert len(failures) == (2 if later_rejection else 0)
        if failures:
            assert failures[0] == failures[1]
        assert _nodes(plain) == _nodes(observed)
        assert plain.graph["history"] == observed.graph["history"]
        assert tuple(plain.edges) == tuple(observed.edges)
        assert plain.graph["_t"] == observed.graph["_t"] == 2.0
        assert observed.nodes[0][CONSUMED_INTERVALS_KEY] == (KEY,)
        assert trace[0].mutation_interval_committed is (call == 0)
        assert trace[0].applied_glyph == ("ZHIR" if call == 0 else "IL")
