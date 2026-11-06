"""Independent admission, exact-threshold and rollback controls for hybrid ZHIR.

Every successful trigger below is produced by the actual local event executor.
Scoped fault injection tests rejection boundaries; it never creates a successful
source or replaces a numerical expectation with the implementation under test.
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
from tnfr.operators._operator_growth_trigger import (
    OperatorGrowthTriggerCertificate,
    _operator_growth_source_context,
    _OperatorGrowthContext,
    _validate_operator_growth_trigger_certificate,
)
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.operators.mutation import Mutation
from tnfr.operators.network_stage import _detached_stage_graph
from tnfr.operators.operator_growth_mutation import execute_operator_growth_mutation
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.utils._structural_signature import structural_proof_signature

CORE, PEER = "core", "peer"
NODES = (CORE, PEER)
WATERMARK = "_operator_growth_mutation_end"
WORD = ("emission", "coherence", "dissonance", "coherence", "silence")
ERRORS = (TypeError, ValueError, RuntimeError, OperatorPreconditionError)


def _graph(*, start=0.0, xi=0.0625):
    graph = nx.Graph()
    graph.add_edge(CORE, PEER, weight=1.0)
    inject_defaults(graph)
    graph.graph.update(
        _t=float(start),
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


def _schedule(*, start=0.0, duration=1.0, word=WORD):
    return build_operator_event_schedule(
        word,
        start_time=start,
        flow_durations=(duration / 2, duration / 2, 0.0, 0.0, 0.0, 0.0),
    )


def _execute(graph, schedule=None, targets=NODES, **kwargs):
    return execute_operator_growth_mutation(
        graph,
        _schedule() if schedule is None else schedule,
        targets,
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


@pytest.mark.parametrize("targets", [(), [CORE], (CORE, CORE), ("absent",)])
def test_invalid_target_selection_rejects_without_source_execution(targets):
    graph = _graph()
    before = _state(graph)
    with pytest.raises(ERRORS):
        _execute(graph, targets=targets)
    assert _state(graph) == before


@pytest.mark.parametrize(
    "options",
    [
        [],
        {"compute_delta_nfr": lambda graph: None},
        {"sequence_context": object()},
        {"_operator_growth_context": object()},
        {"unknown": True},
    ],
)
def test_unadmitted_mutation_options_cannot_inject_a_source_or_callback(options):
    graph = _graph()
    before = _state(graph)
    with pytest.raises(ERRORS):
        _execute(graph, mutation_kwargs=options)
    assert _state(graph) == before


def test_option_materialization_cannot_silently_reprepare_the_source_graph():
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
    assert result.growth.exact_epi_after == (Fraction(1, 2), Fraction(1, 4))


@pytest.mark.parametrize("watermark", [None, True, 0.0, "0", (0, 1)])
def test_malformed_present_watermark_is_not_treated_as_absent(watermark):
    graph = _graph()
    graph.nodes[CORE][WATERMARK] = watermark
    before = _state(graph)
    with pytest.raises(TypeError, match="Fraction"):
        _execute(graph)
    assert _state(graph) == before


@pytest.mark.parametrize("start", [0.0, 0.5])
def test_identical_or_overlapping_window_rejects_even_after_clock_reset(start):
    graph = _graph()
    _execute(graph, targets=(CORE,))
    assert graph.nodes[CORE][WATERMARK] == Fraction(1)
    graph.graph["_t"] = start  # Deliberate external reset cannot erase the receipt.
    before = _state(graph)
    with pytest.raises(ValueError, match="overlaps"):
        _execute(graph, _schedule(start=start), targets=(CORE,))
    assert _state(graph) == before
    assert WATERMARK not in graph.nodes[PEER]


def test_shared_window_boundary_allows_actual_disjoint_continuation():
    graph = _graph()
    first = _execute(graph)
    continuation = ("reception", *WORD[1:])
    second = _execute(graph, _schedule(start=1.0, word=continuation), targets=(PEER,))
    assert first.growth.exact_end_time == second.growth.exact_start_time == 1
    assert second.growth.exact_net_growth_rate == (Fraction(-1, 8), Fraction(1, 8))
    assert graph.nodes[CORE][WATERMARK] == 1
    assert graph.nodes[PEER][WATERMARK] == 2
    assert graph.nodes[CORE]["glyph_history"].count("ZHIR") == 1
    assert graph.nodes[PEER]["glyph_history"].count("ZHIR") == 2


def test_signed_clock_origin_retains_a_negative_consumed_end():
    graph = _graph(start=-2.0)
    first = _execute(graph, _schedule(start=-2.0))
    assert first.growth.exact_start_time == -2
    assert first.growth.exact_end_time == -1
    assert all(graph.nodes[node][WATERMARK] == -1 for node in NODES)
    continuation = ("reception", *WORD[1:])
    second = _execute(graph, _schedule(start=-1.0, word=continuation), targets=(PEER,))
    assert second.growth.exact_start_time == -1
    assert graph.nodes[CORE][WATERMARK] == -1
    assert graph.nodes[PEER][WATERMARK] == 0


def test_zero_duration_and_mismatched_live_clock_leave_no_source_effects():
    for start, duration in ((0.0, 0.0), (1.0, 1.0)):
        graph = _graph()
        before = _state(graph)
        with pytest.raises(ERRORS):
            _execute(graph, _schedule(start=start, duration=duration))
        assert _state(graph) == before


@pytest.mark.parametrize(
    "xi", [True, float("nan"), float("inf"), -0.1, Fraction(1, 10**400)]
)
def test_invalid_threshold_rolls_back_the_successfully_executed_source(xi):
    graph = _graph(xi=xi)
    before = _state(graph)
    with pytest.raises(ERRORS):
        _execute(graph)
    assert _state(graph) == before
    assert all(WATERMARK not in graph.nodes[node] for node in NODES)


def test_live_capacity_floor_rejects_after_source_silence_and_restores_everything():
    graph = _graph()
    graph.graph["ZHIR_MIN_VF"] = 0.75
    before = _state(graph)
    with pytest.raises(OperatorPreconditionError, match="active structural frequency"):
        _execute(graph)
    assert _state(graph) == before
    assert all(WATERMARK not in graph.nodes[node] for node in NODES)


def test_zero_history_window_preserves_lifetime_context_and_actual_mutation():
    graph = _graph()
    result = _execute(graph, mutation_kwargs={"window": 0})
    assert len(result.stage_result.mutation_decision_observations) == 2
    for node in NODES:
        assert not graph.nodes[node]["glyph_history"]
        assert graph.nodes[node]["_operator_step"] == 6
        assert graph.nodes[node]["_grammar_prior_coherence"] is True
        assert graph.nodes[node]["_grammar_u2_debt"] == 1
        assert graph.nodes[node]["theta"] == 0.25
        assert graph.nodes[node][WATERMARK] == 1


def test_exact_growth_crosses_when_its_projection_equals_the_threshold():
    graph = _graph(xi=float(Fraction(1, 24)))
    result = _execute(
        graph, _schedule(duration=3.0), mutation_kwargs={"collect_metrics": True}
    )
    for decision in result.stage_result.mutation_decision_observations:
        certificate = decision.trigger_certificate
        assert type(certificate) is OperatorGrowthTriggerCertificate
        assert certificate.exact_observed_rate == Fraction(1, 24)
        assert certificate.exact_xi == Fraction.from_float(float(Fraction(1, 24)))
        assert certificate.exact_observed_rate > certificate.exact_xi
        assert certificate.observed_depi_dt == certificate.xi
        assert certificate.observed_crossed is True
        assert certificate.threshold_gate_satisfied is True
        assert decision._proof_fields_are_intact()
    metrics = [row for row in graph.graph["operator_metrics"] if row["glyph"] == "ZHIR"]
    assert len(metrics) == 2
    assert all(row["threshold_met"] is True for row in metrics)


def test_public_exact_certificate_rejects_fixed_width_fraction_components():
    np = pytest.importorskip("numpy")
    result = _execute(_graph())
    certificate = result.stage_result.mutation_decision_observations[
        0
    ].trigger_certificate
    malformed = Fraction(1)
    object.__setattr__(malformed, "_numerator", np.int64(1))
    object.__setattr__(malformed, "_denominator", np.int64(1))
    with pytest.raises(TypeError):
        _validate_operator_growth_trigger_certificate(
            replace(certificate, exact_duration=malformed)
        )


@pytest.mark.parametrize("duration,xi", [(1.0, 0.125), (1.25, 0.1)])
def test_exact_equality_or_rate_below_rounded_threshold_cannot_trigger(duration, xi):
    rate = Fraction(1, 8) / Fraction.from_float(duration)
    assert float(rate) == xi
    assert rate <= Fraction.from_float(xi)
    graph = _graph(xi=xi)
    before = _state(graph)
    with pytest.raises(OperatorPreconditionError, match="exact signed"):
        _execute(graph, _schedule(duration=duration))
    assert _state(graph) == before


class _CorruptingMonitor:
    def __init__(self, channel):
        self.channel = channel

    def before_operator(self, graph, node):
        pass

    def after_operator(self, graph, node, operator):
        if operator != "mutation" or node != CORE:
            return
        if self.channel == "form":
            graph.nodes[CORE]["EPI"] += 0.125
        elif self.channel == "peer":
            graph.nodes[PEER]["external_corruption"] = True
        elif self.channel == "support":
            graph.remove_edge(CORE, PEER)
        elif self.channel == "clock":
            graph.graph["_t"] += 1.0
        elif self.channel == "watermark":
            graph.nodes[CORE][WATERMARK] = Fraction(99)
        elif self.channel == "phase":
            graph.nodes[CORE]["theta"] = 0.75
        elif self.channel == "grammar":
            graph.nodes[CORE]["glyph_history"].pop()
        elif self.channel == "truncated_history":
            graph.nodes[CORE]["glyph_history"] = ["ZHIR"]
        elif self.channel == "step":
            graph.nodes[CORE]["_operator_step"] += 1
        elif self.channel == "debt":
            graph.nodes[CORE]["_grammar_u2_debt"] = 0
        elif self.channel == "trigger":
            graph.nodes[CORE]["_zhir_gate_history_key"] = "epi_time_history"
        elif self.channel == "temporal":
            graph.nodes[CORE]["epi_time_history"] = ((0.0, 0.0), (1.0, 0.5))
        elif self.channel == "raise":
            raise RuntimeError("injected postcommit failure")


@pytest.mark.parametrize(
    "channel,message",
    [
        ("form", "protected state or ports"),
        ("peer", "protected state or ports"),
        ("support", "protected state or ports"),
        ("clock", "source clock"),
        ("watermark", "consumption memory changed during stage"),
        ("phase", "phase decision"),
        ("grammar", "grammar"),
        ("truncated_history", "grammar"),
        ("step", "grammar"),
        ("debt", "grammar"),
        ("trigger", "trigger"),
        ("temporal", "temporal evidence"),
        ("raise", "injected postcommit failure"),
    ],
)
def test_postcommit_corruption_rolls_back_the_whole_source_and_mutation(
    channel, message
):
    graph = _graph()
    graph.graph["integrity_monitor"] = _CorruptingMonitor(channel)
    before = _state(graph)
    with pytest.raises(RuntimeError, match=message):
        _execute(graph, targets=(CORE,))
    assert _state(graph) == before


@contextmanager
def _actual_source(graph):
    with _operator_growth_source_context(
        graph,
        _schedule(),
        (CORE,),
        context={"initial_epi_nonzero": True},
        method="euler",
        n_jobs=1,
    ) as source:
        yield source


def test_private_context_cannot_be_constructed_from_an_old_result():
    graph = _graph()
    with _actual_source(graph) as source:
        result = source.result
    before = _state(graph)
    with pytest.raises(TypeError, match="live source production"):
        _OperatorGrowthContext(graph, (CORE,), result, _authority=object())
    with pytest.raises(TypeError, match="OperatorEventSchedule"):
        _execute(graph, result)
    with pytest.raises(TypeError, match="private live context"):
        Mutation()(graph, CORE, _operator_growth_context=source.observation)
    assert _state(graph) == before


def test_foreign_owner_with_equal_endpoints_cannot_bind_a_snapshot():
    graph = _graph()
    with _actual_source(graph) as source:
        foreign = _detached_stage_graph(graph)
        with pytest.raises(ValueError, match="foreign"):
            source.bind_snapshot(foreign, _detached_stage_graph(foreign))


@pytest.mark.parametrize("channel", ["form", "history", "edge"])
def test_stale_live_state_rejects_despite_unchanged_other_coordinates(channel):
    graph = _graph()
    with _actual_source(graph) as source:
        if channel == "form":
            graph.nodes[CORE]["EPI"] += 0.125
        elif channel == "history":
            graph.nodes[CORE]["glyph_history"].append("SHA")
        else:
            graph.edges[CORE, PEER]["weight"] = 2.0
        with pytest.raises(ValueError, match="stale"):
            source.bind_snapshot(graph, _detached_stage_graph(graph))


def test_only_authorized_snapshot_and_requested_target_can_supply_the_gate():
    graph = _graph()
    with _actual_source(graph) as source:
        snapshot = _detached_stage_graph(graph)
        source.bind_snapshot(graph, snapshot)
        with pytest.raises(ValueError, match="authorized stage snapshot"):
            source.gate_for(_detached_stage_graph(graph), CORE)
        with pytest.raises(ValueError, match="not requested"):
            source.gate_for(snapshot, PEER)
        with pytest.raises(ValueError, match="already bound"):
            source.bind_snapshot(graph, snapshot)
        snapshot.nodes[CORE]["EPI"] += 0.125
        with pytest.raises(ValueError, match="snapshot changed"):
            source.gate_for(snapshot, CORE)


def test_context_closes_on_successful_exit_and_on_exception():
    for fail in (False, True):
        graph = _graph()
        try:
            with _actual_source(graph) as source:
                if fail:
                    raise RuntimeError("injected source lifetime failure")
        except RuntimeError as exc:
            assert fail and str(exc) == "injected source lifetime failure"
        with pytest.raises(ValueError, match="inactive"):
            source.bind_snapshot(graph, _detached_stage_graph(graph))


def test_missing_capture_cannot_authorize_growth_and_restores_source(monkeypatch):
    graph = _graph()
    before = _state(graph)
    monkeypatch.setattr(
        event_runtime, "_capture_interval_endpoint", lambda graph: (None, False)
    )
    with pytest.raises(ValueError, match="source unavailable"):
        _execute(graph)
    assert _state(graph) == before


def test_context_is_retired_after_wrapper_stage_failure(monkeypatch):
    graph = _graph()
    before = _state(graph)
    captured = []

    def fail_stage(*args, **kwargs):
        captured.append(kwargs["_operator_growth_context"])
        raise RuntimeError("injected stage failure")

    monkeypatch.setattr(consumer, "execute_pointwise_stage", fail_stage)
    with pytest.raises(RuntimeError, match="injected stage failure"):
        _execute(graph)
    assert _state(graph) == before
    assert len(captured) == 1
    with pytest.raises(ValueError, match="inactive"):
        captured[0].bind_snapshot(graph, _detached_stage_graph(graph))
