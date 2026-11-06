"""Independent admission and failure boundaries for the live native window.

Unit grammar and waiting memory are explicitly supplied, not acquired structure.
The actual source runs its own flow and native calls. Fault injections below are
engineering boundary controls, never a new canonical law or source evidence.
"""

import math
from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_EPI
from tnfr.dynamics import selectors
from tnfr.operators import native_positive_window as consumer
from tnfr.operators import operator_growth_mutation as selection_owner
from tnfr.operators._operator_growth_trigger import (
    _native_positive_window_source_context,
    _operator_growth_source_context,
)
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.operators.grammar_debt import reset_grammar_state_from_history
from tnfr.operators.native_positive_window import execute_native_positive_window
from tnfr.operators.network_stage import _detached_stage_graph
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.types import require_finite_real_scalar_epi

CORE, PEER = "core", "peer"
NODES = (CORE, PEER)
RECEIPT = "_operator_growth_mutation_end"


def _graph():
    graph = nx.Graph()
    graph.add_edge(CORE, PEER, weight=1.0)
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=41,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        GLYPH_SELECTOR_N_JOBS=1,
        GAMMA={"type": "none"},
        use_extended_dynamics=False,
        ZHIR_THRESHOLD_XI=0.0625,
        IL_PRECONDITIONS={"warn_zero_dnfr": False},
        history={
            "since_AL": {CORE: graph.graph["AL_MAX_LAG"], PEER: 0},
            "since_EN": {CORE: 0, PEER: graph.graph["EN_MAX_LAG"]},
        },
    )
    graph.graph["GLYPH_FACTORS"].update(
        AL_boost=0.125, EN_mix=0.5, IL_dnfr_factor=0.0, ZHIR_theta_shift=0.25
    )
    for node, form in ((CORE, 0.375), (PEER, 0.25)):
        data = graph.nodes[node]
        data.update(
            EPI=form,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            Si=1.0,
            d2EPI=0.0,
            epi_kind="form",
            glyph_history=["IL", "OZ", "IL"],
        )
        reset_grammar_state_from_history(data)
    return graph


def _form(graph, node):
    return require_finite_real_scalar_epi(
        get_attr(graph.nodes[node], ALIAS_EPI, strict=True, conv=lambda value: value)
    )


def _state(graph):
    return (
        deepcopy(dict(graph.nodes(data=True))),
        deepcopy(graph.graph.get("history")),
        graph.graph["_t"],
        tuple(graph.edges(data="weight")),
        deepcopy(graph.graph.get("hybrid_event_log")),
    )


def _execute(graph, **kwargs):
    return execute_native_positive_window(graph, duration=1.0, **kwargs)


def _forbid_source(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid request reached the actual source")

    monkeypatch.setattr(consumer, "_produce_native_positive_window", forbidden)


@pytest.mark.parametrize("duration", [True, 0.0, -1.0, math.nan, math.inf, "1"])
def test_invalid_duration_fails_before_source_or_missing_history_creation(
    monkeypatch, duration
):
    graph = _graph()
    del graph.graph["history"]
    before = _state(graph)
    _forbid_source(monkeypatch)
    with pytest.raises((TypeError, ValueError)):
        execute_native_positive_window(graph, duration=duration)
    assert "history" not in graph.graph
    assert _state(graph) == before


@pytest.mark.parametrize("candidates", [[], (), (CORE, CORE), ("foreign",)])
def test_invalid_candidates_do_not_produce_a_source(monkeypatch, candidates):
    graph = _graph()
    before = _state(graph)
    _forbid_source(monkeypatch)
    with pytest.raises((TypeError, ValueError)):
        _execute(graph, candidates=candidates)
    assert _state(graph) == before


@pytest.mark.parametrize(
    "options",
    [
        {"unknown": True},
        {"window": True},
        {"validate_preconditions": "invalid-boolean"},
        {"_operator_growth_context": object()},
    ],
)
def test_malformed_response_request_cannot_run_a_source(monkeypatch, options):
    graph = _graph()
    before = _state(graph)
    _forbid_source(monkeypatch)
    expected = (
        OperatorPreconditionError
        if "validate_preconditions" in options
        else (TypeError, ValueError)
    )
    with pytest.raises(expected):
        _execute(graph, mutation_kwargs=options)
    assert _state(graph) == before


@pytest.mark.parametrize("jobs", [True, 1.5, "1"])
def test_invalid_parallel_request_precedes_source_and_history_installation(
    monkeypatch, jobs
):
    graph = _graph()
    del graph.graph["history"]
    before = _state(graph)
    _forbid_source(monkeypatch)
    with pytest.raises((TypeError, ValueError)):
        _execute(graph, n_jobs=jobs)
    assert "history" not in graph.graph
    assert _state(graph) == before


@pytest.mark.parametrize("field", ["xi", "capacity", "minimum", "receipt"])
def test_invalid_consumed_scalar_or_receipt_is_not_inactive_abstention(
    monkeypatch, field
):
    graph = _graph()
    graph.nodes[CORE]["nu_f"] = 0.0
    if field == "xi":
        graph.graph["ZHIR_THRESHOLD_XI"] = True
    elif field == "capacity":
        graph.nodes[CORE]["nu_f"] = True
    elif field == "minimum":
        graph.graph["ZHIR_MIN_VF"] = True
    else:
        graph.nodes[CORE][RECEIPT] = True
    before = _state(graph)
    _forbid_source(monkeypatch)
    expected = (
        OperatorPreconditionError
        if field in ("capacity", "minimum")
        else (TypeError, ValueError)
    )
    with pytest.raises(expected):
        _execute(graph)
    assert _state(graph) == before


@pytest.mark.parametrize(
    "domain", ["pressure", "gamma", "extended", "selector", "history"]
)
def test_unsupported_source_domain_is_rejected_before_actual_flow(monkeypatch, domain):
    graph = _graph()
    if domain == "pressure":
        graph.nodes[CORE]["delta_nfr"] = 0.125
    elif domain == "gamma":
        graph.graph["GAMMA"] = {"type": "kuramoto", "K": 0.0}
    elif domain == "extended":
        graph.graph["use_extended_dynamics"] = True
    elif domain == "selector":
        graph.graph["glyph_selector"] = lambda _graph, _node: "AL"
    else:
        graph.graph["history"] = []
    before = _state(graph)
    _forbid_source(monkeypatch)
    with pytest.raises((TypeError, ValueError)):
        _execute(graph)
    assert _state(graph) == before


def test_candidate_subset_retains_complete_native_source_and_only_own_response():
    graph = _graph()
    result = _execute(graph, candidates=(PEER,))
    assert result.candidates == result.selected_targets == (PEER,)
    assert tuple(row.node for row in result.source_result.native_decisions) == NODES
    assert tuple(
        row.applied_glyph for row in result.source_result.native_decisions
    ) == ("AL", "EN")
    assert (_form(graph, CORE), _form(graph, PEER)) == (0.5, 0.375)
    assert graph.nodes[CORE]["theta"] == 0.0
    assert graph.nodes[PEER]["theta"] == 0.25
    assert RECEIPT not in graph.nodes[CORE]
    assert graph.nodes[PEER][RECEIPT] == Q(1)
    assert result.growth.exact_net_growth_rate == (Q(1, 8), Q(1, 8))


@pytest.mark.parametrize(
    "binding", ["dispatcher", "selector_subclass", "integrator_factory"]
)
def test_runtime_substitution_cannot_run_before_canonical_binding_admission(
    monkeypatch, binding
):
    graph = _graph()
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("unadmitted callable executed")

    if binding == "dispatcher":
        monkeypatch.setattr(selectors, "apply_glyph", forbidden)
    elif binding == "selector_subclass":

        class CustomSelector(selectors.DefaultGlyphSelector):
            def __init__(self):
                forbidden()

        graph.graph["glyph_selector"] = CustomSelector
    else:
        graph.graph["integrator"] = forbidden
    before = _state(graph)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _execute(graph)
    assert not calls
    assert _state(graph) == before


def test_missing_history_is_created_by_valid_execution_without_forcing_an_action():
    graph = _graph()
    del graph.graph["history"]
    result = _execute(graph)
    assert type(graph.graph["history"]) is dict
    assert result.candidates == NODES and result.selected_targets == ()
    assert not result.executed and result.stage_result is None
    assert tuple(
        row.applied_glyph for row in result.source_result.native_decisions
    ) == ("IL", "IL")
    assert graph.graph["_t"] == 1.0
    assert all(RECEIPT not in graph.nodes[node] for node in graph)
    assert all(graph.graph["history"]["since_AL"][node] == 1 for node in graph)


def test_all_noncrossing_commits_source_without_mutation_or_empty_stage():
    graph = _graph()
    graph.graph["ZHIR_THRESHOLD_XI"] = 1.0
    result = _execute(graph)
    assert result.selected_targets == () and result.stage_result is None
    assert all(
        "observed_threshold_not_crossed" in item.reasons for item in result.decisions
    )
    assert (_form(graph, CORE), _form(graph, PEER)) == (0.5, 0.375)
    assert graph.graph["_t"] == 1.0
    assert all(graph.nodes[node]["theta"] == 0.0 for node in graph)
    assert all(RECEIPT not in graph.nodes[node] for node in graph)
    assert all(graph.nodes[node]["glyph_history"][-1] != "ZHIR" for node in graph)


def test_late_native_failure_preserves_elapsed_flow_and_actual_first_action():
    graph = _graph()
    graph.graph["history"]["since_EN"][PEER] = 0
    graph.graph["GLYPH_FACTORS"]["IL_dnfr_factor"] = math.nan
    with pytest.raises(ValueError, match="IL_dnfr_factor"):
        _execute(graph)
    assert graph.graph["_t"] == 1.0
    assert _form(graph, CORE) == 0.5 and _form(graph, PEER) == 0.25
    assert graph.nodes[CORE]["glyph_history"][-1] == "AL"
    assert graph.nodes[PEER]["glyph_history"] == ["IL", "OZ", "IL"]
    assert graph.nodes[CORE]["epi_time_history"] == ()
    assert tuple(graph.nodes[PEER]["epi_time_history"])[-2:] == (
        (0.0, 0.25),
        (1.0, 0.25),
    )
    assert graph.graph["history"]["since_AL"][CORE] == 0
    assert all(RECEIPT not in graph.nodes[node] for node in graph)
    assert all(graph.nodes[node]["theta"] == 0.0 for node in graph)


def test_returned_records_are_frozen_observations_not_reusable_activation_inputs():
    graph = _graph()
    result = _execute(graph)
    with pytest.raises(FrozenInstanceError):
        result.selected_targets = ()
    with pytest.raises(FrozenInstanceError):
        result.growth.exact_duration = Q(9)
    before = _state(graph)
    with pytest.raises(TypeError):
        execute_native_positive_window(
            graph, duration=1.0, source_result=result.source_result
        )
    with pytest.raises(TypeError):
        execute_native_positive_window(graph, duration=1.0, growth=result.growth)
    assert _state(graph) == before
    assert result.growth.exact_epi_after == (Q(1, 2), Q(3, 8))


def test_overlap_abstention_keeps_actual_source_and_existing_receipt():
    graph = _graph()
    graph.nodes[CORE][RECEIPT] = Q(2)
    result = _execute(graph, candidates=(CORE,))
    assert result.selected_targets == () and result.stage_result is None
    assert result.decisions[0].reasons == ("window_overlaps_consumed_time",)
    assert result.decisions[0].previous_consumed_end == Q(2)
    assert graph.nodes[CORE][RECEIPT] == Q(2)
    assert (_form(graph, CORE), _form(graph, PEER)) == (0.5, 0.375)
    assert graph.graph["_t"] == 1.0
    assert graph.nodes[CORE]["theta"] == 0.0


class _FailingResponseMonitor:
    def before_operator(self, graph, node):
        pass

    def after_operator(self, graph, node, operator):
        if operator == "mutation":
            graph.nodes[PEER]["injected_response_write"] = True
            raise RuntimeError("declared response callback failure")


@pytest.mark.parametrize("fault", ["phase_factor", "postcommit_monitor"])
def test_failed_response_restores_completed_source_instead_of_pre_flow_state(
    monkeypatch, fault
):
    graph = _graph()
    if fault == "phase_factor":
        graph.graph["GLYPH_FACTORS"]["ZHIR_theta_shift"] = 0.0
    else:
        graph.graph["integrity_monitor"] = _FailingResponseMonitor()
    checkpoints = []
    actual_stage = consumer._execute_growth_stage

    def observed_stage(subject, *args, **kwargs):
        checkpoints.append(_state(subject))
        return actual_stage(subject, *args, **kwargs)

    monkeypatch.setattr(consumer, "_execute_growth_stage", observed_stage)
    with pytest.raises(
        (ValueError, RuntimeError), match="ZHIR_theta_shift|response callback failure"
    ):
        _execute(graph)
    assert len(checkpoints) == 1 and _state(graph) == checkpoints[0]
    assert graph.graph["_t"] == 1.0
    assert (_form(graph, CORE), _form(graph, PEER)) == (0.5, 0.375)
    assert graph.graph["history"]["since_AL"][CORE] == 0
    assert graph.graph["history"]["since_EN"][PEER] == 0
    assert all(graph.nodes[node]["theta"] == 0.0 for node in graph)
    assert all(RECEIPT not in graph.nodes[node] for node in graph)
    assert graph.nodes[CORE]["glyph_history"][-1] == "AL"
    assert graph.nodes[PEER]["glyph_history"][-1] == "EN"


def test_cancelling_outside_dispatch_changes_do_not_supply_operator_permission(
    monkeypatch,
):
    graph = _graph()
    before_dispatch = selectors._NativeEPIBatchCapture.before_dispatch
    finish = selectors._NativeEPIBatchCapture.finish
    calls, observed = [], []

    def external_gap(capture):
        calls.append(True)
        # Declared observer fault: two nonoperator changes cancel in net EPI.
        capture.graph.nodes[PEER]["EPI"] += 0.125 if len(calls) == 1 else -0.125
        return before_dispatch(capture)

    def retain_capture(capture, *args, **kwargs):
        report = finish(capture, *args, **kwargs)
        observed.append(report)
        return report

    monkeypatch.setattr(
        selectors._NativeEPIBatchCapture, "before_dispatch", external_gap
    )
    monkeypatch.setattr(selectors._NativeEPIBatchCapture, "finish", retain_capture)
    with pytest.raises(ValueError, match="outside-dispatch"):
        _execute(graph)
    assert len(calls) == 2 and len(observed) == 1
    assert observed[0].exact_outside_dispatch_increment == (Q(0), Q(0))
    assert observed[0].exact_outside_dispatch_variation == (Q(0), Q(1, 4))
    assert graph.graph["_t"] == 1.0
    assert all(graph.nodes[node]["theta"] == 0.0 for node in graph)
    assert all(RECEIPT not in graph.nodes[node] for node in graph)


@pytest.mark.parametrize("change", ["storage", "population", "clock"])
def test_source_identity_or_clock_change_never_reaches_supplementary_response(
    monkeypatch, change
):
    graph = _graph()
    before_dispatch = selectors._NativeEPIBatchCapture.before_dispatch
    changed = False

    def corrupt_boundary(capture):
        nonlocal changed
        endpoint = before_dispatch(capture)
        if not changed:
            changed = True
            if change == "storage":
                capture.graph._node[CORE] = dict(capture.graph._node[CORE])
            elif change == "population":
                capture.graph.add_node("foreign", **deepcopy(capture.graph.nodes[PEER]))
            else:
                capture.graph.graph["_t"] = 2.0
        return endpoint

    monkeypatch.setattr(
        selectors._NativeEPIBatchCapture, "before_dispatch", corrupt_boundary
    )
    with pytest.raises(
        ValueError, match="population|storage|clock|coverage|disconnected"
    ):
        _execute(graph)
    assert changed
    assert graph.graph["_t"] == (2.0 if change == "clock" else 1.0)
    assert all(graph.nodes[node]["theta"] == 0.0 for node in NODES)
    assert all(RECEIPT not in graph.nodes[node] for node in NODES)


def test_validation_and_candidate_probing_never_replay_the_actual_source(monkeypatch):
    graph = _graph()
    produce = consumer._produce_native_positive_window
    calls = []

    def actual_source(*args, **kwargs):
        calls.append(True)
        return produce(*args, **kwargs)

    monkeypatch.setattr(consumer, "_produce_native_positive_window", actual_source)
    result = _execute(graph)
    assert calls == [True]
    assert result.selected_targets == NODES
    assert (_form(graph, CORE), _form(graph, PEER)) == (0.5, 0.375)
    assert graph.graph["_t"] == 1.0


def _actual_context(graph):
    schedule = build_operator_event_schedule(
        (), start_time=graph.graph["_t"], flow_durations=(1.0,)
    )
    return _native_positive_window_source_context(
        graph, schedule, NODES, graph.graph["history"], method="euler", n_jobs=1
    )


@pytest.mark.parametrize("domain", ["frozen_capacity_pressure", "disabled_grammar"])
def test_private_source_lifetime_readmits_domain_without_relying_on_public_preflight(
    domain,
):
    graph = _graph()
    if domain == "frozen_capacity_pressure":
        # Zero observed flow is insufficient to prove the required pressure law.
        graph.nodes[CORE]["nu_f"] = 0.0
        graph.nodes[CORE]["delta_nfr"] = 0.125
    else:
        graph.graph["GRAMMAR_CANON"] = {"enabled": False}
    before = _state(graph)
    with pytest.raises(ValueError, match="zero held pressure|enabled grammar"):
        with _actual_context(graph):
            pytest.fail("inadmissible source minted a live response context")
    assert _state(graph) == before


def test_actual_live_context_requires_own_stage_snapshot_and_retires_after_use():
    graph = _graph()
    with _actual_context(graph) as source:
        snapshot = _detached_stage_graph(graph)
        with pytest.raises(ValueError, match="authorized stage snapshot"):
            source.gate_for(snapshot, CORE)
        with pytest.raises(ValueError, match="foreign"):
            source.validate_live_before_commit(graph.copy())
        source.bind_snapshot(graph, snapshot)
        gate = source.gate_for(snapshot, CORE)
        assert gate.threshold.certificate.source == "native_positive_window_net_growth"
        assert gate.threshold.certificate.exact_observed_rate == Q(1, 8)
        with pytest.raises(ValueError, match="already bound"):
            source.bind_snapshot(graph, _detached_stage_graph(graph))
    with pytest.raises(ValueError, match="inactive"):
        source.validate_live_before_commit(graph)
    assert graph.graph["_t"] == 1.0
    assert all(graph.nodes[node]["theta"] == 0.0 for node in graph)
    assert all(RECEIPT not in graph.nodes[node] for node in graph)


@pytest.mark.parametrize("fault", ["live_endpoint", "derived_growth", "native_record"])
def test_native_source_change_is_not_authorized_by_earlier_capture(fault):
    graph = _graph()
    with _actual_context(graph) as source:
        if fault == "live_endpoint":
            graph.nodes[PEER]["theta"] = 0.125
        elif fault == "derived_growth":
            object.__setattr__(
                source.observation, "exact_net_growth_rate", (Q(9), Q(9))
            )
        else:
            object.__setattr__(source.result.native_batch, "completed", False)
        with pytest.raises(ValueError, match="stale|changed|complete"):
            source.validate_live_before_commit(graph)
    assert all(RECEIPT not in graph.nodes[node] for node in graph)


def test_equal_arithmetic_from_another_actual_flow_cannot_replace_source_ownership():
    graph, other_graph = _graph(), _graph()
    with _actual_context(graph) as source, _actual_context(other_graph) as other:
        assert source.result.flow_result is not other.result.flow_result
        assert source.observation.exact_epi_before == other.observation.exact_epi_before
        assert source.observation.exact_epi_after == other.observation.exact_epi_after
        assert (
            source.observation.exact_net_growth_rate
            == other.observation.exact_net_growth_rate
        )
        # Both flow results are real intact runs; arithmetic equality cannot
        # replace which live production supplied this source lifetime.
        object.__setattr__(source.result, "flow_result", other.result.flow_result)
        with pytest.raises(ValueError, match="source changed"):
            source.validate_live_before_commit(graph)
    assert graph.graph["_t"] == other_graph.graph["_t"] == 1.0
    assert all(RECEIPT not in graph.nodes[node] for node in graph)


def test_native_mutation_precedence_is_a_defensive_policy_seam_not_acquired_evidence():
    graph = _graph()
    with _actual_context(graph) as source:
        assert tuple(item.glyph for item in source.result.native_batch.dispatches) == (
            "AL",
            "EN",
        )
        # Inject only the explicit predicate's precedence input. No real native
        # ZHIR or positive physical secant is claimed or inserted into source.
        decisions = selection_owner._select_growth_candidates(
            source, NODES, (None, None), native_mutated=(CORE,)
        )
        assert decisions[0].reasons == ("native_mutation_precedence",)
        assert not decisions[0].selected and decisions[1].selected
    assert all(graph.nodes[node]["theta"] == 0.0 for node in graph)
    assert all(RECEIPT not in graph.nodes[node] for node in graph)


def _actual_schedule_context(graph):
    schedule = build_operator_event_schedule(
        (), start_time=graph.graph["_t"], flow_durations=(1.0,)
    )
    return _operator_growth_source_context(
        graph, schedule, NODES, method="euler", n_jobs=1
    )


def test_existing_actual_schedule_source_readmits_without_a_native_only_stamp():
    graph = _graph()
    with _actual_schedule_context(graph) as source:
        observed = source.readmit_source()
        assert observed.available
        assert observed.source == "operator_schedule_net_growth"
        assert observed.exact_duration == Q(1)
        assert observed.exact_net_growth_rate == (Q(0), Q(0))
        assert source.validate_live_before_commit(graph).exact_epi_after == (
            Q(3, 8),
            Q(1, 4),
        )
        source.bind_snapshot(graph, _detached_stage_graph(graph))
    assert graph.graph["_t"] == 1.0
    assert all(RECEIPT not in graph.nodes[node] for node in graph)


@pytest.mark.parametrize("original_family", ["schedule", "native"])
def test_actual_source_family_substitution_rejects_in_both_directions(original_family):
    schedule_graph, native_graph = _graph(), _graph()
    native_graph.graph["history"] = {}
    with (
        _actual_schedule_context(schedule_graph) as schedule_source,
        _actual_context(native_graph) as native_source,
    ):
        assert (
            schedule_source.observation.exact_epi_before
            == native_source.observation.exact_epi_before
        )
        assert (
            schedule_source.observation.exact_epi_after
            == native_source.observation.exact_epi_after
        )
        assert (
            schedule_source.observation.exact_net_growth_rate
            == native_source.observation.exact_net_growth_rate
            == (Q(0), Q(0))
        )
        if original_family == "schedule":
            source, replacement, graph = schedule_source, native_source, schedule_graph
        else:
            source, replacement, graph = native_source, schedule_source, native_graph
        # Two real executions with equal arithmetic still have distinct source
        # families. The optional native stamp cannot erase that distinction.
        object.__setattr__(source, "_source_result", replacement.result)
        with pytest.raises(ValueError, match="native positive-window source changed"):
            source.validate_live_before_commit(graph)
    for graph in (schedule_graph, native_graph):
        assert graph.graph["_t"] == 1.0
        assert all(graph.nodes[node]["theta"] == 0.0 for node in graph)
        assert all(RECEIPT not in graph.nodes[node] for node in graph)
