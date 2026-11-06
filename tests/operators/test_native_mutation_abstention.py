"""Finite native abstention boundaries, without a historical source producer.

Metrics, grammar and own temporal samples are declared unit inputs. These
controls assess current request admission and actual dispatch, not acquisition
of the supplied evidence or an autonomous selection law.
"""

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_VF
from tnfr.dynamics import selectors
from tnfr.dynamics._mutation_occurrence import CONSUMED_INTERVALS_KEY
from tnfr.operators.grammar_debt import (
    read_grammar_state,
    reset_grammar_state_from_history,
)
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.validation import StructuralGrammarError

POLICY = "NATIVE_ZHIR_ABSTENTION_POLICY"
INTERVAL = ("epi_time_history", 0.0, 1.0)


def _source(count=1, *, policy="skip"):
    graph = nx.path_graph(count)
    nx.set_edge_attributes(graph, 1.0, "weight")
    inject_defaults(graph)
    graph.graph.update(
        _t=1.0,
        RANDOM_SEED=31,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        GLYPH_SELECTOR_N_JOBS=1,
        GLYPH_HYSTERESIS_WINDOW=10,
        AL_MAX_LAG=100,
        EN_MAX_LAG=100,
        ZHIR_THRESHOLD_XI=0.0625,
        ZHIR_MIN_VF=0.0,
        history={
            "since_AL": dict.fromkeys(graph, 0),
            "since_EN": dict.fromkeys(graph, 0),
        },
    )
    if policy is not None:
        graph.graph[POLICY] = policy
    graph.graph["GLYPH_FACTORS"]["ZHIR_theta_shift"] = 0.25
    for data in graph.nodes.values():
        data.update(
            EPI=0.5,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            d2EPI=0.0,
            Si=0.1,
            epi_kind="form",
            glyph_history=["IL", "OZ", "IL"],
            epi_time_history=[(0.0, 0.0), (1.0, 0.5)],
        )
        reset_grammar_state_from_history(data)
    return graph


def _block(graph, condition, node=0):
    data = graph.nodes[node]
    if condition == "missing":
        del data["epi_time_history"]
    elif condition == "stale":
        data["epi_time_history"][-1] = (1.0, 0.25)
    elif condition == "noncrossing":
        data["epi_time_history"][0] = (0.0, 0.5)
    elif condition == "inactive":
        data["nu_f"] = 0.0
    elif condition == "below_floor":
        graph.graph["ZHIR_MIN_VF"] = 1.5
    else:
        raise AssertionError(condition)


def _run(graph, trace=None):
    selectors.apply_native_operator_batch(
        graph, graph.graph["history"], decision_trace=trace
    )


def _state(graph):
    return (
        deepcopy(dict(graph.nodes(data=True))),
        deepcopy(graph.graph["history"]),
        tuple(graph.edges(data="weight")),
        graph.graph["_t"],
    )


def _assert_once_counted(graph):
    for key in ("since_AL", "since_EN"):
        assert graph.graph["history"][key] == dict.fromkeys(graph, 1)


@pytest.mark.parametrize(
    "condition,reason",
    [
        ("missing", "missing_history"),
        ("stale", "stale_physical_endpoint"),
        ("noncrossing", "observed_threshold_not_crossed"),
        ("inactive", "inactive_structural_capacity"),
        ("below_floor", "below_configured_capacity_floor"),
    ],
)
def test_skip_preserves_node_without_grammar_dispatch_or_history_cleanup(
    monkeypatch, condition, reason
):
    graph = _source()
    _block(graph, condition)
    before = deepcopy(dict(graph.nodes(data=True)))

    def forbidden(*args, **kwargs):
        pytest.fail("an abstained request reached grammar or physical execution")

    for owner in (
        "enforce_canonical_grammar",
        "apply_glyph",
        "capture_native_form_history",
        "finish_native_form_history",
    ):
        monkeypatch.setattr(selectors, owner, forbidden)
    trace = []
    _run(graph, trace)
    assert dict(graph.nodes(data=True)) == before
    assert graph.graph["_t"] == 1.0
    _assert_once_counted(graph)
    row = trace[0]
    assert row.status == "abstained"
    assert row.requested_glyph == row.selection.base_candidate == "ZHIR"
    assert row.mutation_abstention_policy == "skip"
    assert row.mutation_abstention_reason == reason
    assert row.post_gate_glyph is row.applied_glyph is None
    assert row.grammar_decision is row.physical_history_invalidated is None
    assert not row.mutation_interval_committed
    assert CONSUMED_INTERVALS_KEY not in graph.nodes[0]
    event = graph.graph["history"]["mutation_abstentions"][0]
    assert "policy" not in event
    assert event["requested_glyph"] == "ZHIR"
    assert event["applied_glyph"] is None and event["reason"] == reason


@pytest.mark.parametrize("policy", [None, "coherence"])
@pytest.mark.parametrize("condition", ["missing", "noncrossing", "inactive"])
def test_absent_or_coherence_policy_still_executes_real_coherence(policy, condition):
    graph = _source(policy=policy)
    _block(graph, condition)
    before = read_grammar_state(graph.nodes[0])
    trace = []
    _run(graph, trace)
    after = read_grammar_state(graph.nodes[0])
    assert trace[0].status == "applied"
    assert trace[0].mutation_abstention_policy == "coherence"
    assert trace[0].requested_glyph == "ZHIR"
    assert trace[0].post_gate_glyph == trace[0].applied_glyph == "IL"
    assert trace[0].grammar_decision.candidate == "IL"
    assert after.operator_step == before.operator_step + 1
    assert after.history == (*before.history, "IL")
    assert "policy" not in graph.graph["history"]["mutation_abstentions"][0]
    _assert_once_counted(graph)


def test_observation_is_passive_across_abstention_and_an_actual_successor():
    plain, observed = _source(2), _source(2)
    for graph in (plain, observed):
        _block(graph, "missing")
    _run(plain)
    trace = []
    _run(observed, trace)
    assert _state(plain) == _state(observed)
    assert [row.status for row in trace] == ["abstained", "applied"]
    assert trace[1].applied_glyph == "ZHIR"
    assert observed.nodes[0]["theta"] == 0.0
    assert observed.nodes[1]["theta"] == 0.25


def test_opaque_none_remains_skipped_not_a_mutation_abstention():
    graph = _source()
    graph.graph["glyph_selector"] = lambda *_: None
    before = deepcopy(dict(graph.nodes(data=True)))
    trace = []
    _run(graph, trace)
    assert trace[0].status == "skipped"
    assert trace[0].requested_glyph is trace[0].applied_glyph is None
    assert trace[0].mutation_abstention_reason is None
    assert "mutation_abstentions" not in graph.graph["history"]
    assert dict(graph.nodes(data=True)) == before
    _assert_once_counted(graph)


@pytest.mark.parametrize("bad", [None, True, 0, (), [], "SKIP", "none", ""])
def test_invalid_policy_rejects_before_constructor_preparation_or_lags(bad):
    graph = _source()
    graph.graph[POLICY] = bad

    class Forbidden(selectors.DefaultGlyphSelector):
        def __init__(self):
            pytest.fail("invalid abstention policy reached selector construction")

    graph.graph["glyph_selector"] = Forbidden
    before = _state(graph)
    trace = []
    with pytest.raises(ValueError, match=POLICY):
        _run(graph, trace)
    assert _state(graph) == before and trace == []


class _HookSelector(selectors.AbstractSelector):
    def __init__(self, boundary, value):
        self.boundary, self.value = boundary, value
        self.calls = 0

    def prepare(self, graph, nodes):
        if self.boundary == "prepare":
            graph.graph[POLICY] = self.value

    def select(self, graph, node):
        self.calls += 1
        if self.boundary == "select":
            graph.graph[POLICY] = self.value
        return "ZHIR"


@pytest.mark.parametrize("boundary", ["prepare", "select"])
def test_hook_modified_invalid_policy_is_readmitted_without_dispatch(boundary):
    graph = _source()
    selector = _HookSelector(boundary, "invalid")
    graph.graph["glyph_selector"] = selector
    before = deepcopy(dict(graph.nodes(data=True)))
    trace = []
    with pytest.raises(ValueError, match=POLICY):
        _run(graph, trace)
    assert graph.graph[POLICY] == "invalid"  # Hook effects are not transactional.
    assert dict(graph.nodes(data=True)) == before
    expected = 0 if boundary == "prepare" else 1
    assert graph.graph["history"]["since_AL"][0] == expected
    assert graph.graph["history"]["since_EN"][0] == expected
    assert selector.calls == expected
    assert trace[0].applied_glyph is None


@pytest.mark.parametrize("boundary", ["prepare", "select"])
@pytest.mark.parametrize("changed", ["coherence", "skip"])
def test_valid_hook_policy_change_controls_the_live_request(boundary, changed):
    graph = _source(policy="skip" if changed == "coherence" else "coherence")
    _block(graph, "missing")
    selector = _HookSelector(boundary, changed)
    graph.graph["glyph_selector"] = selector
    trace = []
    _run(graph, trace)
    assert selector.calls == 1
    assert trace[0].mutation_abstention_policy == changed
    assert trace[0].status == ("abstained" if changed == "skip" else "applied")
    assert trace[0].applied_glyph == (None if changed == "skip" else "IL")
    _assert_once_counted(graph)


@pytest.mark.parametrize("condition", ["inactive", "missing", "noncrossing"])
@pytest.mark.parametrize("bad", [True, math.nan, -0.125])
@pytest.mark.parametrize("policy", ["coherence", "skip"])
def test_invalid_capacity_floor_is_not_hidden_by_an_earlier_gate_blocker(
    condition, bad, policy
):
    graph = _source(policy=policy)
    _block(graph, condition)
    graph.graph["ZHIR_MIN_VF"] = bad
    before = deepcopy(dict(graph.nodes(data=True)))
    trace = []
    with pytest.raises(OperatorPreconditionError, match="ZHIR_MIN_VF"):
        _run(graph, trace)
    assert dict(graph.nodes(data=True)) == before
    assert "mutation_abstentions" not in graph.graph["history"]
    assert trace[0].status == "failed" and trace[0].applied_glyph is None
    _assert_once_counted(graph)


@pytest.mark.parametrize("case", ["missing", "boolean", "negative", "nonfinite"])
def test_invalid_requested_capacity_is_not_a_valid_skip(case):
    graph = _source()
    _block(graph, "missing")
    if case == "missing":
        for key in ALIAS_VF:
            graph.nodes[0].pop(key, None)
    else:
        graph.nodes[0]["nu_f"] = {
            "boolean": True,
            "negative": -1.0,
            "nonfinite": math.inf,
        }[case]
    before = deepcopy(dict(graph.nodes(data=True)))
    trace = []
    with pytest.raises(OperatorPreconditionError):
        _run(graph, trace)
    assert dict(graph.nodes(data=True)) == before
    assert "mutation_abstentions" not in graph.graph["history"]
    assert trace[0].status == "failed" and trace[0].applied_glyph is None


def test_unrequested_mutation_floor_does_not_block_actual_coherence():
    graph = _source()
    graph.graph["ZHIR_MIN_VF"] = True
    graph.nodes[0]["Si"] = 0.9
    trace = []
    _run(graph, trace)
    assert trace[0].requested_glyph == trace[0].applied_glyph == "IL"
    assert trace[0].mutation_certificate is None
    assert "mutation_abstentions" not in graph.graph["history"]


@pytest.mark.parametrize("bad", [True, math.nan])
def test_invalid_threshold_remains_an_error_when_temporal_evidence_is_missing(bad):
    graph = _source()
    _block(graph, "missing")
    graph.graph["ZHIR_THRESHOLD_XI"] = bad
    trace = []
    with pytest.raises(OperatorPreconditionError, match="ZHIR_THRESHOLD_XI"):
        _run(graph, trace)
    assert trace[0].status == "failed" and trace[0].applied_glyph is None
    assert "mutation_abstentions" not in graph.graph["history"]


def test_skipped_phase_parameters_are_not_certified_or_consumed():
    graph = _source()
    _block(graph, "missing")
    graph.graph["GLYPH_FACTORS"]["ZHIR_theta_shift"] = math.nan
    trace = []
    _run(graph, trace)
    assert trace[0].status == "abstained"
    assert graph.nodes[0]["theta"] == 0.0


@pytest.mark.parametrize("sink", [(), {}, "invalid"])
def test_invalid_abstention_sink_rejects_without_a_glyph_or_record(sink):
    graph = _source()
    _block(graph, "missing")
    graph.graph["history"]["mutation_abstentions"] = sink
    before = deepcopy(dict(graph.nodes(data=True)))
    trace = []
    with pytest.raises(TypeError, match="mutation_abstentions"):
        _run(graph, trace)
    assert graph.graph["history"]["mutation_abstentions"] is sink
    assert dict(graph.nodes(data=True)) == before
    assert trace[0].status == "failed"
    assert trace[0].applied_glyph is trace[0].grammar_decision is None
    _assert_once_counted(graph)


@pytest.mark.parametrize("policy", ["coherence", "skip"])
def test_explicit_none_sink_records_the_actual_abstention_outcome(policy):
    graph = _source(policy=policy)
    _block(graph, "missing")
    graph.graph["history"]["mutation_abstentions"] = None
    trace = []
    _run(graph, trace)
    sink = graph.graph["history"]["mutation_abstentions"]
    assert type(sink) is list and len(sink) == 1
    assert sink[0]["requested_glyph"] == "ZHIR"
    assert sink[0]["reason"] == "missing_history"
    assert sink[0]["applied_glyph"] == trace[0].applied_glyph
    assert sink[0]["applied_glyph"] == (None if policy == "skip" else "IL")
    assert "policy" not in sink[0]
    _assert_once_counted(graph)


def test_sink_failure_keeps_the_real_prefix_and_its_consumed_receipt():
    graph = _source(3)
    graph.graph["NATIVE_ZHIR_OCCURRENCE_POLICY"] = "consume_timestamped_interval"
    _block(graph, "missing", 1)
    graph.graph["history"]["mutation_abstentions"] = ()
    untouched = {node: deepcopy(graph.nodes[node]) for node in (1, 2)}
    trace = []
    with pytest.raises(TypeError, match="mutation_abstentions"):
        _run(graph, trace)
    assert [row.status for row in trace] == ["applied", "failed", "not_reached"]
    assert trace[0].applied_glyph == "ZHIR"
    assert trace[0].mutation_interval_committed
    assert graph.nodes[0][CONSUMED_INTERVALS_KEY] == (INTERVAL,)
    assert graph.nodes[0]["theta"] == 0.25
    assert all(graph.nodes[node] == untouched[node] for node in untouched)
    _assert_once_counted(graph)


def test_later_invalid_floor_keeps_an_unrelated_real_coherence_prefix():
    graph = _source(3)
    graph.nodes[0]["Si"] = 0.9
    graph.graph["ZHIR_MIN_VF"] = True
    _block(graph, "noncrossing", 1)
    steps = tuple(read_grammar_state(graph.nodes[node]).operator_step for node in graph)
    trace = []
    with pytest.raises(OperatorPreconditionError, match="ZHIR_MIN_VF"):
        _run(graph, trace)
    assert [row.status for row in trace] == ["applied", "failed", "not_reached"]
    assert trace[0].applied_glyph == "IL"
    assert tuple(
        read_grammar_state(graph.nodes[node]).operator_step for node in graph
    ) == (
        steps[0] + 1,
        steps[1],
        steps[2],
    )
    assert "mutation_abstentions" not in graph.graph["history"]
    _assert_once_counted(graph)


def test_skip_does_not_hide_grammar_rejection_of_an_eligible_request():
    graph = _source()
    graph.nodes[0]["glyph_history"] = ["IL"]
    reset_grammar_state_from_history(graph.nodes[0])
    trace = []
    with pytest.raises(StructuralGrammarError, match="recent destabilizer"):
        _run(graph, trace)
    assert trace[0].mutation_certificate.threshold_gate_satisfied
    assert trace[0].status == "failed" and trace[0].applied_glyph is None
    assert trace[0].mutation_abstention_reason is None
    assert "mutation_abstentions" not in graph.graph["history"]


def test_eligible_dispatch_failure_remains_a_failure_with_its_partial_effects(
    monkeypatch,
):
    graph = _source()
    graph.graph["NATIVE_ZHIR_OCCURRENCE_POLICY"] = "consume_timestamped_interval"
    apply = selectors.apply_glyph
    failure = RuntimeError("declared failure after the actual primitive")

    def fail_after_actual_action(subject, node, glyph, **kwargs):
        apply(subject, node, glyph, **kwargs)
        raise failure

    monkeypatch.setattr(selectors, "apply_glyph", fail_after_actual_action)
    trace = []
    with pytest.raises(RuntimeError) as caught:
        _run(graph, trace)
    assert caught.value is failure
    assert trace[0].status == "failed" and trace[0].failure_stage == "dispatch"
    assert trace[0].applied_glyph is None
    assert graph.nodes[0]["theta"] == 0.25
    assert read_grammar_state(graph.nodes[0]).history[-1] == "ZHIR"
    assert CONSUMED_INTERVALS_KEY not in graph.nodes[0]
    assert "mutation_abstentions" not in graph.graph["history"]


def test_actual_consumed_request_abstains_without_renewing_grammar_or_receipt():
    graph = _source()
    graph.graph["NATIVE_ZHIR_OCCURRENCE_POLICY"] = "consume_timestamped_interval"
    first = []
    _run(graph, first)
    assert first[0].applied_glyph == "ZHIR" and first[0].mutation_interval_committed
    before = deepcopy(dict(graph.nodes(data=True)))
    second = []
    _run(graph, second)
    assert dict(graph.nodes(data=True)) == before
    assert second[0].status == "abstained"
    assert second[0].mutation_abstention_reason == "mutation_interval_already_consumed"
    assert second[0].mutation_interval == INTERVAL
    assert second[0].mutation_interval_previously_consumed is True
    assert not second[0].mutation_interval_committed
    assert graph.graph["history"]["since_AL"][0] == 2
    assert graph.graph["history"]["since_EN"][0] == 2


def test_empty_retention_window_does_not_clear_existing_context_on_skip():
    graph = _source()
    _block(graph, "missing")
    graph.graph["GLYPH_HYSTERESIS_WINDOW"] = 0
    before = deepcopy(graph.nodes[0])
    _run(graph)
    assert graph.nodes[0] == before
    assert len(graph.graph["history"]["mutation_abstentions"]) == 1
