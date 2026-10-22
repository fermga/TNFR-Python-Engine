"""The standard observer requires fresh interval evidence, not stale counts."""

from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.alias import set_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.dynamics.runtime import step
from tnfr.glyph_history import ensure_history
from tnfr.metrics.coherence import _track_stability
from tnfr.metrics.core import register_metrics_callbacks
from tnfr.observers import (
    _after_step_reorg,
    _before_step_reorg,
    attach_standard_observer,
)
from tnfr.telemetry import ensure_nu_f_telemetry
from tnfr.utils import CallbackEvent
from tnfr.utils.callbacks import CallbackManager


def _graph():
    graph = nx.empty_graph(4)
    inject_defaults(graph)
    graph.graph.update(_t=0.0, PHASE_ADAPT={"enabled": False}, CALLBACKS_STRICT=True)
    for node in graph:
        graph.nodes[node].update(
            EPI=0.1, nu_f=1.0, theta=0.0, delta_nfr=0.0, dEPI_dt=0.0, Si=0.5
        )
    return graph


def _sample(graph, unstable=2):
    for node in graph:
        graph.nodes[node]["delta_nfr"] = 1.0 if node < unstable else 0.0
    _track_stability(graph, ensure_history(graph), 99.0, 1e-3, 1e-3)


def _snapshot(graph):
    return ensure_nu_f_telemetry(graph, history_limit=None).snapshot()


@pytest.mark.parametrize("stale", [None, 0.0, 0.5, 1.0])
def test_missing_current_stability_is_not_a_reorganisation_count(stale):
    graph = _graph()
    if stale is not None:
        graph.graph["history"] = {"stable_frac": [stale]}
    _before_step_reorg(graph, {"step": 0, "dt": 0.5})
    graph.graph["_t"] = 0.5
    # Runtime context can itself carry the previous history value.
    _after_step_reorg(graph, {"step": 0, "stable_frac": stale})
    assert _snapshot(graph).windows == ()
    assert graph.graph["_std_observer_reorg"]["status"] == "unavailable"
    assert graph.graph["_std_observer_reorg"]["last_reorganisations"] is None


@pytest.mark.parametrize("unstable", [0, 2, 4])
def test_fresh_proxy_uses_current_history_and_actual_elapsed_time_once(unstable):
    graph = _graph()
    _before_step_reorg(graph, {"step": 3, "dt": 100.0})
    graph.graph["_t"] = 0.5
    _sample(graph, unstable)
    context = {"step": 3, "stable_frac": 0.25}
    _after_step_reorg(graph, context)
    _after_step_reorg(graph, context)
    snapshot = _snapshot(graph)
    assert len(snapshot.windows) == 1
    assert snapshot.total_reorganisations == unstable
    assert snapshot.total_duration == 0.5
    assert snapshot.rate_hz_str == 2 * unstable
    assert snapshot.windows[0].start == 0.0
    assert snapshot.windows[0].end == 0.5


def test_mismatched_or_unpaired_after_does_not_reuse_a_window():
    graph = _graph()
    _after_step_reorg(graph, {"step": 0, "stable_frac": 0.0})
    assert _snapshot(graph).windows == ()
    _before_step_reorg(graph, {"step": 1})
    graph.graph["_t"] = 0.5
    _sample(graph)
    _after_step_reorg(graph, {"step": 0})
    assert _snapshot(graph).windows == ()
    _after_step_reorg(graph, {"step": 1})
    assert len(_snapshot(graph).windows) == 1


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        True,
        "0.5",
        float("nan"),
        float("inf"),
        -0.1,
        1.1,
        Fraction(1) + Fraction(1, 2**100),
    ],
)
def test_invalid_fresh_fraction_is_unavailable_not_clamped(invalid):
    graph = _graph()
    _before_step_reorg(graph, {"step": 0})
    graph.graph["_t"] = 0.5
    _sample(graph)
    graph.graph["history"]["stable_frac"][-1] = invalid
    _after_step_reorg(graph, {"step": 0, "stable_frac": 1.0})
    assert _snapshot(graph).windows == ()
    assert graph.graph["_std_observer_reorg"]["reason"] == "invalid_current_stability"


@pytest.mark.parametrize("boundary", ["before", "after"])
@pytest.mark.parametrize("clock", [True, "0.5", Fraction(1, 2**2000), float("nan")])
def test_primitive_clock_is_admitted_before_conversion(boundary, clock):
    graph = _graph()
    if boundary == "after":
        _before_step_reorg(graph, {"step": 0})
        graph.graph["_t"] = 0.5
        _sample(graph)
    graph.graph["_t"] = clock
    callback = _before_step_reorg if boundary == "before" else _after_step_reorg
    with pytest.raises((TypeError, ValueError)):
        callback(graph, {"step": 0})
    assert _snapshot(graph).windows == ()


@pytest.mark.parametrize("duration", [0.0, -0.5])
def test_nonpositive_actual_interval_does_not_use_requested_dt(duration):
    graph = _graph()
    _before_step_reorg(graph, {"step": 0, "dt": 1.0})
    _sample(graph)
    graph.graph["_t"] = duration
    _after_step_reorg(graph, {"step": 0})
    assert _snapshot(graph).windows == ()
    assert graph.graph["_std_observer_reorg"]["reason"] == "nonpositive_interval"


@pytest.mark.parametrize("observer_first", [False, True])
def test_real_runtime_observation_is_fresh_in_both_registration_orders(observer_first):
    graph = _graph()
    graph.graph["HISTORY_MAXLEN"] = 1
    graph.graph["METRICS"]["verbosity"] = "basic"
    register = (
        (attach_standard_observer, register_metrics_callbacks)
        if observer_first
        else (register_metrics_callbacks, attach_standard_observer)
    )
    for callback in register:
        callback(graph)
    # Re-registration does not duplicate the producer or consumer.
    register_metrics_callbacks(graph)
    attach_standard_observer(graph)
    for _ in range(2):
        step(graph, dt=0.125, use_Si=False, apply_glyphs=False)
    snapshot = _snapshot(graph)
    assert len(snapshot.windows) == 2
    assert snapshot.total_reorganisations == 0
    assert snapshot.total_duration == 0.25
    assert graph.graph["_stability_observation_revision"] == 2


def test_disabled_metrics_cannot_reuse_previous_runtime_evidence():
    graph = _graph()
    register_metrics_callbacks(graph)
    attach_standard_observer(graph)
    step(graph, dt=0.125, use_Si=False, apply_glyphs=False)
    assert len(_snapshot(graph).windows) == 1
    previous_payload = deepcopy(graph.graph["telemetry"]["nu_f_snapshot"])
    previous_rates = tuple(graph.graph["history"]["nu_f_rate_hz_str"])
    graph.graph["METRICS"]["enabled"] = False
    step(graph, dt=0.125, use_Si=False, apply_glyphs=False)
    assert len(_snapshot(graph).windows) == 1
    assert graph.graph["_std_observer_reorg"]["reason"] == "missing_current_stability"
    assert graph.graph["telemetry"]["nu_f_snapshot"] == previous_payload
    assert tuple(graph.graph["history"]["nu_f_rate_hz_str"]) == previous_rates


@pytest.mark.parametrize("history_length", [1, 8])
def test_runtime_publishes_completed_window_once_and_preserves_limits(history_length):
    graph = _graph()
    graph.graph["HISTORY_MAXLEN"] = history_length
    graph.graph["METRICS"]["verbosity"] = "detailed"
    accumulator = ensure_nu_f_telemetry(
        graph, confidence_level=0.8, history_limit=2, window_limit=3
    )
    attach_standard_observer(graph)
    register_metrics_callbacks(graph)

    for unstable, duration, total_count, total_duration, rate in (
        (1, 0.25, 1, 0.25, 4.0),
        (3, 0.5, 4, 0.75, 16.0 / 3.0),
    ):

        def pressure(live_graph):
            for node in live_graph:
                set_attr(live_graph.nodes[node], ALIAS_DNFR, float(node < unstable))

        graph.graph["compute_delta_nfr"] = pressure
        step(graph, dt=duration, use_Si=False, apply_glyphs=False)
        summary = graph.graph["telemetry"]["nu_f"]
        assert summary["total_reorganisations"] == total_count
        assert summary["total_duration"] == total_duration
        assert summary["rate_hz_str"] == pytest.approx(rate)
        assert graph.graph["history"]["nu_f_rate_hz_str"][-1] == pytest.approx(rate)
        assert accumulator.confidence_level == 0.8
        assert accumulator.history_limit == 2
        assert accumulator.window_limit == 3

    assert len(graph.graph["history"]["nu_f_rate_hz_str"]) == min(history_length, 2)
    assert len(accumulator.snapshot().windows) == 2
    assert len(graph.graph["telemetry"]["nu_f_history"]) == 2


def test_observer_retains_existing_accumulator_configuration_and_windows():
    graph = _graph()
    accumulator = ensure_nu_f_telemetry(
        graph, confidence_level=0.8, history_limit=2, window_limit=3
    )
    accumulator.record_counts(1, 0.125)
    attach_standard_observer(graph)
    _before_step_reorg(graph, {"step": 0})
    graph.graph["_t"] = 0.5
    _sample(graph)
    _after_step_reorg(graph, {"step": 0})
    assert graph.graph["_tnfr_nu_f_accumulator"] is accumulator
    assert accumulator.confidence_level == 0.8
    assert accumulator.history_limit == 2
    assert accumulator.window_limit == 3
    assert len(accumulator.snapshot().windows) == 2


def test_callback_dependency_reorders_only_the_named_consumer_and_is_idempotent():
    graph = nx.Graph()
    manager = CallbackManager()
    seen = []
    for name in ("consumer", "first", "producer", "last"):
        manager.register_callback(
            graph,
            CallbackEvent.AFTER_STEP,
            lambda _graph, _ctx, name=name: seen.append(name),
            name=name,
        )
    for _ in range(2):
        manager._order_callback_after(
            graph, CallbackEvent.AFTER_STEP, "consumer", "producer"
        )
    manager.invoke_callbacks(graph, CallbackEvent.AFTER_STEP)
    assert seen == ["first", "producer", "consumer", "last"]
    copied = deepcopy(graph)
    assert tuple(copied.graph["callbacks"]["after_step"]) == tuple(
        graph.graph["callbacks"]["after_step"]
    )
    before = dict(graph.graph["callbacks"]["after_step"])
    manager._order_callback_after(
        graph, CallbackEvent.AFTER_STEP, "missing", "producer"
    )
    manager._order_callback_after(
        graph, CallbackEvent.AFTER_STEP, "consumer", "missing"
    )
    assert graph.graph["callbacks"]["after_step"] == before
