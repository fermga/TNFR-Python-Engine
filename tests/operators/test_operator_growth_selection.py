"""Independent boundaries for state-conditioned hybrid Mutation participation.

Successful evidence always comes from actual bounded local operator schedules.
Fault injection tests rejection and restoration, not an alternative source law.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants import inject_defaults
from tnfr.operators import event_runtime
from tnfr.operators import operator_growth_mutation as consumer
from tnfr.operators.event_runtime import execute_operator_event_schedule
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.operators.factor_contracts import GlyphFactorValidationError
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import STAGE_SCHEDULE_KEY
from tnfr.operators.operator_growth_mutation import (
    execute_eligible_operator_growth_mutation,
    execute_operator_growth_mutation,
)
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.utils._structural_signature import structural_proof_signature

CORE, PEER = "core", "peer"
NODES = (CORE, PEER)
WORD = ("emission", "coherence", "dissonance", "coherence", "silence")
RECEIPT = "_operator_growth_mutation_end"
ERRORS = (TypeError, ValueError, RuntimeError, OperatorPreconditionError)


def _graph(*, xi=0.0625):
    graph = nx.Graph()
    graph.add_edge(CORE, PEER, weight=1.0)
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        INTEGRATOR_METHOD="euler",
        GAMMA={"type": "none"},
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_REJECTION_MODE="raise",
        RANDOM_SEED=31,
        ZHIR_THRESHOLD_XI=xi,
    )
    graph.graph["GLYPH_FACTORS"].update(
        AL_boost=0.125,
        EN_mix=0.5,
        IL_dnfr_factor=0.0,
        SHA_vf_factor=0.5,
        ZHIR_theta_shift=0.25,
    )
    for node, epi in zip(NODES, (0.375, 0.125), strict=True):
        graph.nodes[node].update(
            EPI=epi,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            epi_kind="form",
            glyph_history=[],
        )
    return graph


def _schedule(*, word=WORD, start=0.0, duration=1.0):
    return build_operator_event_schedule(
        word,
        start_time=start,
        flow_durations=(duration / 2, duration / 2, *((0.0,) * (len(word) - 1))),
    )


def _execute(graph, schedule=None, candidates=None, **kwargs):
    return execute_eligible_operator_growth_mutation(
        graph,
        _schedule() if schedule is None else schedule,
        candidates,
        context={"initial_epi_nonzero": True},
        method="euler",
        n_jobs=1,
        **kwargs,
    )


def _state(graph):
    return structural_proof_signature(
        (
            graph.graph,
            tuple(graph.nodes(data=True)),
            tuple(graph.edges(data=True)),
            getattr(graph, "_last_operator_applied", None),
        ),
        opaque_references=(graph,),
    )


def _assert_source_committed_without_mutation(graph, result):
    assert not result.executed
    assert result.selected_targets == () and result.stage_result is None
    assert graph.graph["_t"] == 1.0
    assert tuple(graph.nodes[node]["EPI"] for node in NODES) == (0.5, 0.25)
    assert len(graph.graph["hybrid_event_log"]) == len(WORD)
    assert graph.graph[STAGE_SCHEDULE_KEY]["glyph"] == "SHA"
    assert graph.graph[STAGE_SCHEDULE_KEY]["nodes_processed"] == 2
    for node in NODES:
        assert graph.nodes[node]["theta"] == 0.0
        assert graph.nodes[node]["nu_f"] == 0.5
        assert read_grammar_state(graph.nodes[node]).history == (
            "AL",
            "IL",
            "OZ",
            "IL",
            "SHA",
        )
        assert RECEIPT not in graph.nodes[node]


@pytest.mark.parametrize("candidates", [None, (PEER, CORE)])
def test_default_and_explicit_candidate_order_bind_actual_selected_stage(candidates):
    graph = _graph()
    result = _execute(graph, candidates=candidates)
    expected = NODES if candidates is None else candidates
    assert result.candidates == result.selected_targets == expected
    assert result.source_result.target_nodes == result.growth.nodes == NODES
    assert tuple(item.node for item in result.decisions) == expected
    assert all(item.selected and item.reasons == () for item in result.decisions)
    assert (
        tuple(item.node for item in result.stage_result.mutation_decision_observations)
        == expected
    )
    assert all(graph.nodes[node][RECEIPT] == Fraction(1) for node in NODES)


def test_explicit_candidate_subset_leaves_positive_peer_unrequested():
    graph = _graph()
    result = _execute(graph, candidates=(CORE,))
    assert result.candidates == result.selected_targets == (CORE,)
    assert result.growth.exact_net_growth_rate == (Fraction(1, 8),) * 2
    assert graph.nodes[CORE]["theta"] == 0.25
    assert graph.nodes[PEER]["theta"] == 0.0
    assert RECEIPT not in graph.nodes[PEER]
    assert read_grammar_state(graph.nodes[PEER]).history[-1] == "SHA"


@pytest.mark.parametrize("candidates", [(), [CORE], (CORE, CORE), ("absent",)])
def test_invalid_candidate_population_cannot_execute_a_source(candidates):
    graph = _graph()
    before = _state(graph)
    with pytest.raises(ERRORS):
        _execute(graph, candidates=candidates)
    assert _state(graph) == before


@pytest.mark.parametrize(
    "options", [{"_operator_growth_context": object()}, {"sequence_context": object()}]
)
def test_execution_options_cannot_inject_source_or_grammar_permission(options):
    graph = _graph()
    before = _state(graph)
    with pytest.raises(ERRORS):
        _execute(graph, mutation_kwargs=options)
    assert _state(graph) == before


def test_option_materialization_cannot_silently_reprepare_actual_source():
    graph = _graph()

    class MutatingOptions(Mapping):
        def __iter__(self):
            graph.nodes[CORE]["EPI"] = 0.75
            return iter(("collect_metrics",))

        def __len__(self):
            return 1

        def __getitem__(self, key):
            if key != "collect_metrics":
                raise KeyError(key)
            return False

    result = _execute(graph, mutation_kwargs=MutatingOptions())
    assert result.growth.exact_epi_before == (Fraction(3, 8), Fraction(1, 8))
    assert result.selected_targets == NODES


def test_all_noncrossing_preserves_source_without_calling_an_empty_mutation_stage(
    monkeypatch,
):
    graph = _graph(xi=0.125)

    def forbidden_stage(*args, **kwargs):
        raise AssertionError("empty selection must not invoke a Mutation stage")

    monkeypatch.setattr(consumer, "execute_pointwise_stage", forbidden_stage)
    result = _execute(graph)
    _assert_source_committed_without_mutation(graph, result)
    assert all(
        item.reasons == ("observed_threshold_not_crossed",) for item in result.decisions
    )


def test_grammar_abstention_does_not_search_for_or_execute_an_alternative(monkeypatch):
    from tnfr.operators import grammar_dynamics

    graph = _graph()

    def forbidden_alternative(*args, **kwargs):
        raise AssertionError("eligibility must not choose a replacement glyph")

    monkeypatch.setattr(grammar_dynamics, "suggest_alternative", forbidden_alternative)
    result = _execute(graph, _schedule(word=("emission", "silence")))
    assert result.growth.exact_net_growth_rate == (Fraction(1, 8),) * 2
    assert not result.executed
    for item in result.decisions:
        assert "grammar_rejected" in item.reasons
        assert "U4b" in item.grammar_violations
        assert read_grammar_state(graph.nodes[item.node]).history == ("AL", "SHA")
        assert RECEIPT not in graph.nodes[item.node]


@pytest.mark.parametrize("xi", [True, float("nan"), -0.25])
def test_invalid_threshold_is_an_error_not_an_abstention(xi):
    graph = _graph(xi=xi)
    before = _state(graph)
    with pytest.raises(ERRORS):
        _execute(graph)
    assert _state(graph) == before


@pytest.mark.parametrize("minimum", [True, float("nan"), -0.25])
def test_invalid_capacity_configuration_rejects_even_all_noncrossing(minimum):
    graph = _graph(xi=0.25)
    graph.graph["ZHIR_MIN_VF"] = minimum
    before = _state(graph)
    with pytest.raises(ERRORS):
        _execute(graph)
    assert _state(graph) == before


@pytest.mark.parametrize("history", [None, ["not_a_glyph"]])
def test_corrupt_grammar_state_is_not_reported_as_legitimate_nonparticipation(history):
    graph = _graph(xi=0.25)
    graph.nodes[PEER]["glyph_history"] = history
    before = _state(graph)
    with pytest.raises(ERRORS):
        _execute(graph)
    assert _state(graph) == before


@pytest.mark.parametrize("receipt", [None, True, 0.0, "0"])
def test_malformed_receipt_rejects_even_all_noncrossing(receipt):
    graph = _graph(xi=0.25)
    graph.nodes[PEER][RECEIPT] = receipt
    before = _state(graph)
    with pytest.raises(TypeError, match="Fraction"):
        _execute(graph)
    assert _state(graph) == before


def test_valid_capacity_floor_excludes_after_actual_source_without_erasing_it():
    graph = _graph()
    graph.graph["ZHIR_MIN_VF"] = 0.75
    result = _execute(graph)
    _assert_source_committed_without_mutation(graph, result)
    assert all(
        item.reasons == ("below_configured_capacity_floor",)
        and item.nu_f == 0.5
        and item.minimum_nu_f == 0.75
        for item in result.decisions
    )


@pytest.mark.parametrize(
    "duration,xi,selected",
    [(3.0, float(Fraction(1, 24)), NODES), (1.0, 0.125, ()), (1.25, 0.1, ())],
)
def test_exact_comparison_decides_selection_when_displayed_rate_equals_xi(
    duration, xi, selected
):
    graph = _graph(xi=xi)
    result = _execute(graph, _schedule(duration=duration))
    assert result.selected_targets == selected
    exact_rate = Fraction(1, 8) / Fraction.from_float(duration)
    assert float(exact_rate) == xi
    for item in result.decisions:
        assert item.exact_rate == exact_rate
        assert item.exact_xi == Fraction.from_float(xi)
        assert item.selected is (exact_rate > item.exact_xi)


@pytest.mark.parametrize("start", [0.0, 0.5])
def test_consumed_window_excludes_growth_after_external_clock_and_history_reset(
    start,
):
    graph = _graph()
    first = _execute(graph)
    assert first.selected_targets == NODES
    # This is an explicit caller intervention, not an autonomous continuation.
    # Clear only continuous samples so the source recorder can use the reset
    # clock. Actual operator history and consumed-window memory stay intact.
    graph.graph["_t"] = start
    for node in NODES:
        graph.nodes[node]["epi_time_history"] = ()
    continuation = ("reception", *WORD[1:])
    result = _execute(graph, _schedule(word=continuation, start=start))
    assert result.growth.exact_net_growth_rate == (Fraction(-1, 8), Fraction(1, 8))
    assert not result.executed and result.stage_result is None
    decisions = {item.node: item for item in result.decisions}
    assert decisions[PEER].reasons == ("window_overlaps_consumed_time",)
    assert all(item.previous_consumed_end == 1 for item in result.decisions)
    assert tuple(graph.nodes[node]["EPI"] for node in NODES) == (0.375, 0.375)
    for node in NODES:
        assert graph.nodes[node][RECEIPT] == 1
        assert graph.nodes[node]["glyph_history"].count("ZHIR") == 1


def test_clock_reset_alone_remains_a_source_error_instead_of_abstention():
    graph = _graph()
    _execute(graph)
    graph.graph["_t"] = 0.0
    before = _state(graph)
    continuation = ("reception", *WORD[1:])
    with pytest.raises(ValueError, match="Runtime time precedes the latest EPI sample"):
        _execute(graph, _schedule(word=continuation))
    assert _state(graph) == before
    assert all(graph.nodes[node][RECEIPT] == 1 for node in NODES)


def test_modified_selection_report_cannot_waive_live_window_consumption(monkeypatch):
    graph = _graph()
    # Supplied valid ledger state tests an admission boundary, not the origin
    # of this receipt. The growth itself still comes from a real source run.
    graph.nodes[CORE][RECEIPT] = Fraction(1)
    before = _state(graph)
    original = consumer._select_growth_candidates

    def forced_selection(*args, **kwargs):
        decisions = original(*args, **kwargs)
        assert decisions[0].reasons == ("window_overlaps_consumed_time",)
        return tuple(replace(item, reasons=()) for item in decisions)

    monkeypatch.setattr(consumer, "_select_growth_candidates", forced_selection)
    with pytest.raises(ValueError, match="overlap"):
        _execute(graph, candidates=(CORE,))
    assert _state(graph) == before


def test_selected_stage_failure_restores_source_instead_of_becoming_abstention():
    graph = _graph()
    graph.graph["GLYPH_FACTORS"]["ZHIR_theta_shift"] = 0.0
    before = _state(graph)
    with pytest.raises(GlyphFactorValidationError, match="ZHIR_theta_shift"):
        _execute(graph)
    assert _state(graph) == before


class _CorruptingMonitor:
    def __init__(self, channel):
        self.channel = channel

    def before_operator(self, graph, node):
        pass

    def after_operator(self, graph, node, operator):
        if operator != "mutation" or node != CORE:
            return
        if self.channel == "peer":
            graph.nodes[PEER]["unrequested_change"] = True
        elif self.channel == "history":
            graph.nodes[CORE]["glyph_history"] = ["ZHIR"]
        elif self.channel == "receipt":
            graph.nodes[CORE][RECEIPT] = Fraction(99)
        else:
            raise RuntimeError("injected selected-stage failure")


@pytest.mark.parametrize(
    "channel,message",
    [
        ("peer", "protected state or ports"),
        ("history", "grammar transition"),
        ("receipt", "consumption memory"),
        ("raise", "injected selected-stage failure"),
    ],
)
def test_selected_stage_callback_failure_restores_whole_source_and_receipts(
    channel, message
):
    graph = _graph()
    graph.graph["integrity_monitor"] = _CorruptingMonitor(channel)
    before = _state(graph)
    with pytest.raises(RuntimeError, match=message):
        _execute(graph, candidates=(CORE,))
    assert _state(graph) == before


def test_strict_interface_keeps_error_and_rollback_for_same_noncrossing_source():
    graph = _graph(xi=0.125)
    before = _state(graph)
    with pytest.raises(OperatorPreconditionError, match="exact signed"):
        execute_operator_growth_mutation(
            graph,
            _schedule(),
            NODES,
            context={"initial_epi_nonzero": True},
            method="euler",
            n_jobs=1,
        )
    assert _state(graph) == before
    result = _execute(graph)
    _assert_source_committed_without_mutation(graph, result)


@pytest.mark.parametrize("fault", ["stale_graph", "tampered_result"])
def test_source_provenance_failure_cannot_be_reported_as_empty_selection(
    monkeypatch, fault
):
    graph = _graph(xi=0.125)
    before = _state(graph)
    original = consumer._operator_growth_source_context

    @contextmanager
    def corrupted_source(*args, **kwargs):
        with original(*args, **kwargs) as source:
            if fault == "stale_graph":
                args[0].nodes[CORE]["delta_nfr"] = 0.125
            else:
                object.__setattr__(source.result, "_proof_stamp", ())
            yield source

    monkeypatch.setattr(consumer, "_operator_growth_source_context", corrupted_source)
    with pytest.raises(ValueError, match="stale|intact|source"):
        _execute(graph)
    assert _state(graph) == before


def test_unavailable_capture_is_an_error_even_without_crossing(monkeypatch):
    graph = _graph(xi=0.125)
    before = _state(graph)
    monkeypatch.setattr(
        event_runtime, "_capture_interval_endpoint", lambda graph: (None, False)
    )
    with pytest.raises(ValueError, match="source unavailable"):
        _execute(graph)
    assert _state(graph) == before


def test_past_foreign_execution_result_cannot_replace_the_actual_source():
    source_graph = _graph()
    result = execute_operator_event_schedule(
        source_graph,
        _schedule(),
        context={"initial_epi_nonzero": True},
        method="euler",
        n_jobs=1,
        include_stage_certificates=True,
    )
    graph = _graph()
    before = _state(graph)
    with pytest.raises(TypeError, match="OperatorEventSchedule"):
        _execute(graph, result)
    assert _state(graph) == before


def test_zero_retention_window_does_not_erase_lifetime_admission_or_consumption():
    graph = _graph()
    result = _execute(graph, mutation_kwargs={"window": 0})
    assert result.selected_targets == NODES
    for node in NODES:
        grammar = read_grammar_state(graph.nodes[node])
        assert grammar.history == () and grammar.operator_step == 6
        assert grammar.prior_coherence and grammar.u2_debt == 1
        assert graph.nodes[node][RECEIPT] == 1
