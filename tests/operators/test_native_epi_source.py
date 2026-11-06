"""Native EPI capture boundaries on finite, locally prepared unit graphs.

These are observations of actual native dispatch boundaries, not a continuous
source, an authenticated canonical map or permission to activate Mutation.
Custom hooks below are explicitly declared instrumentation controls.
"""

import math
from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_EPI
from tnfr.dynamics import selectors
from tnfr.operators.grammar_debt import reset_grammar_state_from_history


def _proposal(graph, node):
    return "AL" if node == 0 else "IL"


def _source(count=2):
    graph = nx.path_graph(count)
    nx.set_edge_attributes(graph, 1.0, "weight")
    inject_defaults(graph)
    graph.graph.update(
        _t=1.0,
        RANDOM_SEED=31,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        GLYPH_SELECTOR_N_JOBS=1,
        AL_MAX_LAG=100,
        EN_MAX_LAG=100,
        glyph_selector=_proposal,
        history={
            "since_AL": dict.fromkeys(graph, 0),
            "since_EN": dict.fromkeys(graph, 0),
        },
    )
    graph.graph["GLYPH_FACTORS"].update(AL_boost=0.125, IL_dnfr_factor=0.5)
    for node, data in graph.nodes(data=True):
        data.update(
            EPI=(node + 1) / 4,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.125,
            d2EPI=0.0,
            Si=0.9,
            epi_kind="form",
            glyph_history=["AL", "IL"],
        )
        reset_grammar_state_from_history(data)
    return graph


def _run(graph, epi_trace=None, decision_trace=None):
    return selectors.apply_native_operator_batch(
        graph,
        graph.graph["history"],
        epi_trace=epi_trace,
        decision_trace=decision_trace,
    )


def _state(graph):
    return (
        deepcopy(dict(graph.nodes(data=True))),
        deepcopy(graph.graph["history"]),
        tuple(graph.edges(data="weight")),
        graph.graph["_t"],
    )


def _assert_unavailable_totals(report):
    assert not report.available
    for name in (
        "exact_net_increment",
        "exact_dispatch_increment",
        "exact_dispatch_variation",
        "exact_outside_dispatch_increment",
        "exact_outside_dispatch_variation",
    ):
        assert getattr(report, name) is None


def test_actual_capture_has_signed_endpoints_without_manufacturing_time_or_history():
    graph = _source()
    epi, decisions = [], []
    assert _run(graph, epi, decisions) is None
    assert len(epi) == 1
    report = epi[0]
    assert report.completed and report.available and report.clock_preserved
    assert report.failure_type is None and report.unavailability_reasons == ()
    assert report.initial.nodes == report.final.nodes == (0, 1)
    assert report.initial.exact_time == report.final.exact_time == Fraction(1)
    assert report.initial.exact_epi == (Fraction(1, 4), Fraction(1, 2))
    assert report.final.exact_epi == (Fraction(3, 8), Fraction(1, 2))
    assert report.exact_net_increment == (Fraction(1, 8), Fraction(0))
    assert report.exact_dispatch_increment == report.exact_net_increment
    assert report.exact_dispatch_variation == report.exact_net_increment
    assert report.exact_outside_dispatch_increment == (Fraction(0),) * 2
    assert report.exact_outside_dispatch_variation == (Fraction(0),) * 2
    assert tuple(item.node for item in report.dispatches) == (0, 1)
    assert tuple(item.glyph for item in report.dispatches) == ("AL", "IL")
    assert all(
        item.returned and item.failure_type is None for item in report.dispatches
    )
    assert report.dispatches[0].exact_increment == (Fraction(1, 8), Fraction(0))
    assert report.dispatches[1].exact_increment == (Fraction(0),) * 2
    assert [row.applied_glyph for row in decisions] == ["AL", "IL"]
    assert all("epi_time_history" not in data for data in graph.nodes.values())
    assert not hasattr(report, "exact_net_growth_rate")
    assert graph.graph["_t"] == 1.0
    with pytest.raises(FrozenInstanceError):
        report.completed = False
    graph.nodes[0]["EPI"] = 0.875
    assert report.final.exact_epi == (Fraction(3, 8), Fraction(1, 2))


def test_optional_capture_and_decision_trace_never_replay_the_selector_or_dispatch(
    monkeypatch,
):
    calls, dispatched, states = [], [], []
    apply = selectors.apply_glyph

    def proposal(graph, node):
        calls.append(node)
        return _proposal(graph, node)

    def actual(graph, node, glyph, **kwargs):
        dispatched.append((node, glyph))
        return apply(graph, node, glyph, **kwargs)

    monkeypatch.setattr(selectors, "apply_glyph", actual)
    for capture, observe in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ):
        graph = _source()
        graph.graph["glyph_selector"] = proposal
        epi, decisions = ([] if capture else None), ([] if observe else None)
        assert _run(graph, epi, decisions) is None
        states.append(_state(graph))
        if capture:
            assert len(epi) == 1 and epi[0].completed
    assert calls == [0, 1] * 4
    assert dispatched == [(0, "AL"), (1, "IL")] * 4
    assert all(state == states[0] for state in states)


def test_existing_capture_prefix_is_retained():
    marker = object()
    epi = [marker]
    _run(_source(), epi)
    assert len(epi) == 2 and epi[0] is marker and epi[1].completed


@pytest.mark.parametrize("sink", [(), {}, 0, "invalid"])
def test_invalid_capture_sink_rejects_before_selector_construction(sink):
    graph = _source()

    class Forbidden(selectors.DefaultGlyphSelector):
        def __init__(self):
            pytest.fail("invalid capture sink reached selector construction")

    graph.graph["glyph_selector"] = Forbidden
    before = _state(graph)
    with pytest.raises(TypeError, match="exact list"):
        _run(graph, sink)
    assert _state(graph) == before


def test_capture_list_subclass_cannot_intercept_report_writes():
    class Sink(list):
        def append(self, value):
            pytest.fail("capture must reject subclassed sinks")

    graph = _source()
    before = _state(graph)
    with pytest.raises(TypeError, match="exact list"):
        _run(graph, Sink())
    assert _state(graph) == before


def test_capture_and_decision_sinks_cannot_alias():
    graph, sink = _source(), []
    before = _state(graph)
    with pytest.raises(ValueError):
        _run(graph, sink, sink)
    assert sink == [] and _state(graph) == before


def test_original_preflight_failure_still_emits_one_incomplete_batch_report():
    graph = _source()
    graph.graph["AL_MAX_LAG"] = True
    epi, decisions = [], []
    with pytest.raises(TypeError):
        _run(graph, epi, decisions)
    assert len(epi) == 1 and decisions == []
    report = epi[0]
    assert not report.completed and report.failure_type == "TypeError"
    assert report.available and report.dispatches == ()
    assert report.exact_net_increment == (Fraction(0),) * 2
    assert graph.graph["history"]["since_AL"] == {0: 0, 1: 0}


def test_constructor_failure_retains_its_outside_dispatch_change_and_original_error():
    graph = _source()
    failure = RuntimeError("declared selector constructor failure")

    class Constructor(selectors.DefaultGlyphSelector):
        def __init__(self):
            graph.nodes[0]["EPI"] = 0.5
            raise failure

    graph.graph["glyph_selector"] = Constructor
    epi = []
    with pytest.raises(RuntimeError) as caught:
        _run(graph, epi)
    assert caught.value is failure
    report = epi[0]
    assert not report.completed and report.available
    assert report.dispatches == ()
    assert report.exact_net_increment == (Fraction(1, 4), Fraction(0))
    assert report.exact_dispatch_increment == (Fraction(0),) * 2
    assert report.exact_outside_dispatch_increment == report.exact_net_increment


@pytest.mark.parametrize("case", ["missing", "boolean", "nonfinite", "complex", "rich"])
def test_unavailable_form_observation_does_not_tighten_a_nonform_action(case):
    graph = _source(1)
    graph.graph["GRAMMAR_CANON"] = {"enabled": False}
    graph.graph["glyph_selector"] = lambda *_: "SHA"
    if case == "missing":
        for alias in ALIAS_EPI:
            graph.nodes[0].pop(alias, None)
    else:
        graph.nodes[0]["EPI"] = {
            "boolean": True,
            "nonfinite": math.nan,
            "complex": 1j,
            "rich": {
                "continuous": (0.25, 0.5),
                "discrete": (0.25, 0.5),
                "grid": (0.0, 1.0),
            },
        }[case]
        # An invalid authoritative value must not fall through to this alias.
        graph.nodes[0][ALIAS_EPI[1]] = 0.25
    raw = graph.nodes[0].get("EPI")
    epi, decisions = [], []
    _run(graph, epi, decisions)
    report = epi[0]
    assert report.completed and decisions[0].applied_glyph == "SHA"
    assert graph.nodes[0]["nu_f"] < 1.0
    assert graph.nodes[0].get("EPI") is raw
    _assert_unavailable_totals(report)
    assert "endpoint_unavailable" in report.unavailability_reasons
    assert not report.initial.available
    assert report.dispatches[0].exact_increment is None
    assert "epi_time_history" not in graph.nodes[0]


def test_unsupported_numeric_subclass_is_not_executed_by_optional_observation():
    calls = []

    class OpaqueFloat(float):
        def __float__(self):
            calls.append("float")
            raise RuntimeError("observation invoked a custom numeric hook")

    graph = _source(1)
    graph.graph["GRAMMAR_CANON"] = {"enabled": False}
    graph.graph["glyph_selector"] = lambda *_: "SHA"
    raw = OpaqueFloat(0.25)
    graph.nodes[0]["EPI"] = raw
    epi = []
    _run(graph, epi)
    assert calls == [] and graph.nodes[0]["EPI"] is raw
    assert graph.nodes[0]["nu_f"] < 1.0
    assert epi[0].completed
    _assert_unavailable_totals(epi[0])


@pytest.mark.parametrize("kind", ["raw", "rational", "serialized"])
def test_supported_signed_form_and_clock_retain_their_represented_values(kind):
    graph = _source(1)
    graph.graph["glyph_selector"] = lambda *_: "SHA"
    graph.graph["_t"] = 0.1
    graph.nodes[0]["EPI"] = {
        "raw": -0.25,
        "rational": Fraction(-1, 4),
        "serialized": {
            "continuous": (-0.25, -0.25),
            "discrete": (-0.25, -0.25),
            "grid": (0.0, 1.0),
        },
    }[kind]
    epi = []
    _run(graph, epi)
    report = epi[0]
    assert report.available and report.completed
    assert report.initial.exact_epi == report.final.exact_epi == (Fraction(-1, 4),)
    assert (
        report.initial.exact_time == report.final.exact_time == Fraction.from_float(0.1)
    )
    assert report.initial.exact_time != Fraction(1, 10)
    assert report.exact_dispatch_increment == (Fraction(0),)


def test_selector_change_is_reported_outside_the_actual_operator_increment():
    graph, calls = _source(), []

    def proposal(subject, node):
        calls.append(node)
        if node == 0:
            subject.nodes[0]["EPI"] = 0.5
        return _proposal(subject, node)

    graph.graph["glyph_selector"] = proposal
    epi = []
    _run(graph, epi)
    report = epi[0]
    assert calls == [0, 1] and report.available
    assert report.exact_net_increment == (Fraction(3, 8), Fraction(0))
    assert report.exact_dispatch_increment == (Fraction(1, 8), Fraction(0))
    assert report.exact_outside_dispatch_increment == (Fraction(1, 4), Fraction(0))
    assert (
        report.exact_outside_dispatch_variation
        == report.exact_outside_dispatch_increment
    )


def test_grammar_side_effect_is_not_relabelled_as_an_operator_effect(monkeypatch):
    graph = _source()
    grammar = selectors.enforce_canonical_grammar

    def external(subject, node, glyph, **kwargs):
        accepted = grammar(subject, node, glyph, **kwargs)
        if node == 0:
            subject.nodes[1]["EPI"] = 0.75
        return accepted

    monkeypatch.setattr(selectors, "enforce_canonical_grammar", external)
    epi = []
    _run(graph, epi)
    report = epi[0]
    assert report.available
    assert report.exact_net_increment == (Fraction(1, 8), Fraction(1, 4))
    assert report.exact_dispatch_increment == (Fraction(1, 8), Fraction(0))
    assert report.exact_outside_dispatch_increment == (Fraction(0), Fraction(1, 4))
    assert report.dispatches[0].before.exact_epi[1] == Fraction(3, 4)


def test_full_population_capture_includes_nonlocal_effects_of_an_opaque_dispatch(
    monkeypatch,
):
    graph = _source()
    apply = selectors.apply_glyph

    def external(subject, node, glyph, **kwargs):
        apply(subject, node, glyph, **kwargs)
        if node == 0:
            subject.nodes[1]["EPI"] = 0.75

    monkeypatch.setattr(selectors, "apply_glyph", external)
    epi = []
    _run(graph, epi)
    report = epi[0]
    assert report.available
    assert report.dispatches[0].exact_increment == (Fraction(1, 8), Fraction(1, 4))
    assert report.exact_dispatch_increment == report.exact_net_increment
    assert report.exact_outside_dispatch_increment == (Fraction(0),) * 2
    # This captures effects of that callable; its glyph label is not map authentication.


def test_zero_outside_net_does_not_hide_opposite_changes_between_dispatches(
    monkeypatch,
):
    graph = _source()
    grammar = selectors.enforce_canonical_grammar

    def external(subject, node, glyph, **kwargs):
        accepted = grammar(subject, node, glyph, **kwargs)
        subject.nodes[1]["EPI"] = 0.75 if node == 0 else 0.5
        return accepted

    monkeypatch.setattr(selectors, "enforce_canonical_grammar", external)
    epi = []
    _run(graph, epi)
    report = epi[0]
    assert report.available and report.completed
    assert report.exact_net_increment == (Fraction(1, 8), Fraction(0))
    assert report.exact_dispatch_increment == report.exact_net_increment
    assert report.exact_outside_dispatch_increment == (Fraction(0),) * 2
    assert report.exact_outside_dispatch_variation == (Fraction(0), Fraction(1, 2))
    nonzero_gaps = tuple(
        gap for gap in report.exact_outside_dispatch_increments if any(gap)
    )
    assert nonzero_gaps == (
        (Fraction(0), Fraction(1, 4)),
        (Fraction(0), Fraction(-1, 4)),
    )


def test_partial_dispatch_failure_preserves_exact_prefix_and_original_exception(
    monkeypatch,
):
    graph = _source(3)
    apply = selectors.apply_glyph
    failure = RuntimeError("declared partial second dispatch")

    def partial(subject, node, glyph, **kwargs):
        if node == 1:
            subject.nodes[node]["EPI"] = 0.625
            raise failure
        return apply(subject, node, glyph, **kwargs)

    monkeypatch.setattr(selectors, "apply_glyph", partial)
    epi, decisions = [], []
    with pytest.raises(RuntimeError) as caught:
        _run(graph, epi, decisions)
    assert caught.value is failure
    report = epi[0]
    assert not report.completed and report.available
    assert report.failure_type == "RuntimeError"
    assert len(report.dispatches) == 2
    assert report.dispatches[0].returned and not report.dispatches[1].returned
    assert report.dispatches[1].failure_type == "RuntimeError"
    assert report.exact_net_increment == (Fraction(1, 8), Fraction(1, 8), Fraction(0))
    assert report.exact_dispatch_increment == report.exact_net_increment
    assert report.exact_outside_dispatch_increment == (Fraction(0),) * 3
    assert [row.status for row in decisions] == ["applied", "failed", "not_reached"]


def test_unreadable_partial_endpoint_does_not_replace_the_original_failure(monkeypatch):
    graph = _source()
    failure = RuntimeError("declared unreadable partial endpoint")
    value = object()

    def partial(subject, node, glyph, **kwargs):
        subject.nodes[node]["EPI"] = value
        raise failure

    monkeypatch.setattr(selectors, "apply_glyph", partial)
    epi = []
    with pytest.raises(RuntimeError) as caught:
        _run(graph, epi)
    assert caught.value is failure and graph.nodes[0]["EPI"] is value
    report = epi[0]
    assert not report.completed and report.failure_type == "RuntimeError"
    assert len(report.dispatches) == 1 and not report.dispatches[0].returned
    _assert_unavailable_totals(report)
    assert report.dispatches[0].exact_increment is None


def test_failed_dispatch_and_failed_capture_keep_the_original_error_without_false_accounting(
    monkeypatch,
):
    graph = _source()
    graph.nodes[0]["epi_time_history"] = [(0.0, 0.125), (1.0, 0.25)]
    failure = RuntimeError("declared actual partial dispatch failure")

    def partial(subject, node, glyph, **kwargs):
        subject.nodes[node]["EPI"] = 0.5
        raise failure

    def capture_failure(*args, **kwargs):
        raise TypeError("declared secondary capture failure")

    monkeypatch.setattr(selectors, "apply_glyph", partial)
    monkeypatch.setattr(
        selectors._NativeEPIBatchCapture, "after_dispatch", capture_failure
    )
    epi = []
    with pytest.raises(RuntimeError) as caught:
        _run(graph, epi)
    assert caught.value is failure
    assert graph.nodes[0]["EPI"] == 0.5
    assert graph.nodes[0]["epi_time_history"] == ()
    report = epi[0]
    assert not report.completed and report.failure_type == "RuntimeError"
    assert "dispatch_capture_failed" in report.unavailability_reasons
    _assert_unavailable_totals(report)
    assert report.exact_outside_dispatch_increments is None


@pytest.mark.parametrize("glyph", ["AL", "ZHIR"])
def test_returned_dispatch_with_failed_capture_keeps_its_mark_and_finishes_lifecycle(
    monkeypatch,
    glyph,
):
    graph = _source()
    graph.nodes[0]["epi_time_history"] = [(0.0, 0.125), (1.0, 0.25)]
    if glyph == "ZHIR":
        graph.graph.update(
            ZHIR_THRESHOLD_XI=0.0625,
            NATIVE_ZHIR_OCCURRENCE_POLICY="consume_timestamped_interval",
            glyph_selector=lambda _graph, node: "ZHIR" if node == 0 else "IL",
        )
        graph.graph["GLYPH_FACTORS"]["ZHIR_theta_shift"] = 0.25
        # Declared grammar admission and own positive samples are independent
        # inputs to this recorder-failure control, not acquired by observation.
        graph.nodes[0]["glyph_history"] = ["AL", "IL", "OZ", "IL"]
        reset_grammar_state_from_history(graph.nodes[0])

    class ObservationFailure(RuntimeError):
        pass

    failure = ObservationFailure(f"declared capture failure after actual {glyph}")

    def capture_failure(*args, **kwargs):
        raise failure

    monkeypatch.setattr(
        selectors._NativeEPIBatchCapture, "after_dispatch", capture_failure
    )
    epi, decisions = [], []
    with pytest.raises(ObservationFailure) as caught:
        _run(graph, epi, decisions)
    assert caught.value is failure
    assert decisions[0].status == "failed"
    assert decisions[0].applied_glyph == glyph
    assert decisions[1].status == "not_reached"
    assert decisions[0].failure_stage == "epi_capture"
    if glyph == "AL":
        assert graph.nodes[0]["epi_time_history"] == ()
        assert decisions[0].physical_history_invalidated is True
        assert graph.graph["history"]["since_AL"][0] == 0
    else:
        assert graph.nodes[0]["epi_time_history"] == [(0.0, 0.125), (1.0, 0.25)]
        assert decisions[0].physical_history_invalidated is False
        assert graph.nodes[0]["theta"] == 0.25
        assert decisions[0].mutation_interval_committed is True
        assert graph.nodes[0]["_native_zhir_consumed_intervals"] == (
            ("epi_time_history", 0.0, 1.0),
        )
    report = epi[0]
    assert not report.completed and report.failure_type == "ObservationFailure"
    assert report.final.exact_epi == (
        Fraction(3, 8) if glyph == "AL" else Fraction(1, 4),
        Fraction(1, 2),
    )
    assert "dispatch_capture_failed" in report.unavailability_reasons
    _assert_unavailable_totals(report)


def test_cleanup_failure_does_not_turn_a_returned_dispatch_into_a_failed_one(
    monkeypatch,
):
    graph = _source()
    failure = RuntimeError("declared cleanup failure")

    def fail(*args):
        raise failure

    monkeypatch.setattr(selectors, "finish_native_form_history", fail)
    epi = []
    with pytest.raises(RuntimeError) as caught:
        _run(graph, epi)
    assert caught.value is failure
    report = epi[0]
    assert report.available and not report.completed
    assert len(report.dispatches) == 1 and report.dispatches[0].returned
    assert report.dispatches[0].failure_type is None
    assert report.exact_net_increment == (Fraction(1, 8), Fraction(0))


def test_selection_failure_can_have_only_outside_dispatch_changes():
    graph = _source()

    class Unprintable(RuntimeError):
        def __str__(self):
            raise ValueError("message unavailable")

    failure = Unprintable()

    def fail(subject, node):
        subject.nodes[0]["EPI"] = 0.5
        raise failure

    graph.graph["glyph_selector"] = fail
    epi, decisions = [], []
    with pytest.raises(Unprintable) as caught:
        _run(graph, epi, decisions)
    assert caught.value is failure
    report = epi[0]
    assert report.available and not report.completed
    assert report.dispatches == () and report.failure_type == "Unprintable"
    assert report.exact_dispatch_increment == (Fraction(0),) * 2
    assert report.exact_outside_dispatch_increment == (Fraction(1, 4), Fraction(0))
    assert report.exact_net_increment == report.exact_outside_dispatch_increment


@pytest.mark.parametrize("change", ["population", "order", "storage"])
def test_changed_node_identity_or_order_makes_fixed_population_totals_unavailable(
    change,
):
    graph = _source()
    changed = False

    def proposal(subject, node):
        nonlocal changed
        if not changed:
            changed = True
            if change == "population":
                subject.add_node(2, **deepcopy(subject.nodes[1]))
            elif change == "order":
                subject._node[0] = subject._node.pop(0)
            else:
                subject._node[0] = dict(subject.nodes[0])
        return _proposal(subject, node)

    graph.graph["glyph_selector"] = proposal
    epi = []
    if change == "storage":
        _run(graph, epi)
        assert epi[0].completed
    else:
        with pytest.raises(ValueError, match="ordered batch targets"):
            _run(graph, epi)
        assert not epi[0].completed
    _assert_unavailable_totals(epi[0])
    assert "node_population_or_storage_changed" in epi[0].unavailability_reasons


def test_selector_clock_change_is_not_invented_flow_duration():
    graph = _source()

    def proposal(subject, node):
        subject.graph["_t"] = 2.0
        return _proposal(subject, node)

    graph.graph["glyph_selector"] = proposal
    epi = []
    _run(graph, epi)
    report = epi[0]
    assert report.completed and not report.clock_preserved
    assert report.initial.exact_time == 1 and report.final.exact_time == 2
    _assert_unavailable_totals(report)
    assert "clock_changed" in report.unavailability_reasons
    assert not hasattr(report, "exact_net_growth_rate")


def test_skipped_and_abstained_requests_are_not_recorded_as_dispatches():
    graph = _source(3)
    graph.graph["NATIVE_ZHIR_ABSTENTION_POLICY"] = "skip"
    graph.graph["glyph_selector"] = lambda _graph, node: (None, "ZHIR", "IL")[node]
    epi, decisions = [], []
    _run(graph, epi, decisions)
    assert [row.status for row in decisions] == ["skipped", "abstained", "applied"]
    report = epi[0]
    assert report.available and report.completed
    assert tuple((item.node, item.glyph) for item in report.dispatches) == ((2, "IL"),)
    assert (
        report.exact_net_increment
        == report.exact_dispatch_increment
        == (Fraction(0),) * 3
    )


def test_exact_jump_accounting_does_not_overflow_finite_represented_endpoints(
    monkeypatch,
):
    graph = _source(1)
    graph.graph["GRAMMAR_CANON"] = {"enabled": False}
    graph.graph["glyph_selector"] = lambda *_: "SHA"
    graph.nodes[0]["EPI"] = -1e308

    def external(subject, node, glyph, **kwargs):
        subject.nodes[node]["EPI"] = 1e308

    monkeypatch.setattr(selectors, "apply_glyph", external)
    epi = []
    _run(graph, epi)
    report = epi[0]
    expected = 2 * Fraction.from_float(1e308)
    assert report.available and report.completed
    assert report.exact_net_increment == (expected,)
    assert (
        report.exact_dispatch_increment
        == report.exact_dispatch_variation
        == (expected,)
    )
