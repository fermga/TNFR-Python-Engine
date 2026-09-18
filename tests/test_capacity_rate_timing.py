"""Capacity diagnostics use observed time, with explicit missing evidence."""

from __future__ import annotations

from copy import deepcopy
import math
import sys

import networkx as nx
import pytest

from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_D2VF, ALIAS_DVF, ALIAS_VF
from tnfr.dynamics.runtime import step
from tnfr.metrics import coherence
from tnfr.metrics.capacity_rates import observe_capacity_rates
from tnfr.metrics.core import _metrics_step, register_metrics_callbacks


def _graph(*, capacity: float = 1.0, time: float | None = 0.0) -> nx.Graph:
    graph = nx.empty_graph(1)
    graph.nodes[0].update(EPI=0.2, nu_f=capacity, phase=0.0)
    if time is not None:
        graph.graph["_t"] = time
    return graph


def _track(graph: nx.Graph, history: dict) -> None:
    coherence._track_stability(graph, history, 99.0, 1e-3, 1e-3, n_jobs=2)


def test_irregular_quadratic_samples_give_midpoint_second_difference() -> None:
    first = observe_capacity_rates(1.0, 0.0)
    second = observe_capacity_rates(2.0, 1.0, first.samples)
    third = observe_capacity_rates(10.0, 3.0, second.samples)

    assert first.rate is first.second_difference is None
    assert second.rate == 1.0
    assert second.second_difference is None
    assert third.rate == 4.0
    assert third.second_difference == 2.0
    assert third.as_payload()["rate_interval"] == [1.0, 3.0]
    assert third.as_payload()["second_difference_times"] == [0.0, 1.0, 3.0]


def test_capacity_derivatives_transform_with_capacity_and_time_units() -> None:
    samples = ()
    for time, capacity in ((0.0, 1.0), (1.0, 2.0), (3.0, 10.0)):
        result = observe_capacity_rates(capacity / 4.0, 4.0 * time, samples)
        samples = result.samples

    assert result.rate == 4.0 / 4.0**2
    assert result.second_difference == 2.0 / 4.0**3


def test_exact_time_differences_avoid_rounded_midpoint_collapse() -> None:
    origin = 2.0**53
    first = observe_capacity_rates(1.0, origin)
    second = observe_capacity_rates(5.0, origin + 2.0, first.samples)
    third = observe_capacity_rates(17.0, origin + 4.0, second.samples)

    assert third.rate == 6.0
    assert third.second_difference == 2.0


def test_unrepresentable_secant_is_not_reported_as_zero_or_infinity() -> None:
    first = observe_capacity_rates(0.0, 0.0)
    overflow = observe_capacity_rates(sys.float_info.max, math.ulp(0.0), first.samples)
    underflow = observe_capacity_rates(math.ulp(0.0), sys.float_info.max, first.samples)

    assert overflow.rate is underflow.rate is None
    assert overflow.rate_status == underflow.rate_status == "unrepresentable"


def test_second_difference_availability_does_not_require_float_secants() -> None:
    first = observe_capacity_rates(0.0, 0.0)
    second = observe_capacity_rates(sys.float_info.max / 2.0, math.ulp(0.0), first.samples)
    third = observe_capacity_rates(sys.float_info.max, 2 * math.ulp(0.0), second.samples)

    assert second.rate is third.rate is None
    assert third.rate_status == "unrepresentable"
    assert third.second_difference == 0.0
    assert third.second_difference_status == "available"


def test_identical_duplicate_retains_history_but_adds_no_interval() -> None:
    first = observe_capacity_rates(1.0, 0.0)
    second = observe_capacity_rates(2.0, 1.0, first.samples)
    duplicate = observe_capacity_rates(2.0, 1.0, second.samples)
    following = observe_capacity_rates(10.0, 3.0, duplicate.samples)

    assert duplicate.status == "duplicate_time"
    assert duplicate.samples == second.samples
    assert duplicate.rate is duplicate.second_difference is None
    assert following.second_difference == 2.0


def test_same_time_jump_restarts_derivative_evidence() -> None:
    graph = _graph()
    history: dict = {}
    _track(graph, history)
    graph.graph["_t"] = 1.0
    graph.nodes[0]["nu_f"] = 2.0
    _track(graph, history)
    graph.nodes[0]["nu_f"] = 7.0
    _track(graph, history)

    data = graph.nodes[0]
    assert data["_capacity_rate_samples"] == ((1.0, 7.0),)
    assert data["capacity_rate_diagnostic"]["status"] == "same_time_jump"
    assert data[ALIAS_DVF[0]] is data[ALIAS_D2VF[0]] is None
    graph.graph["_t"] = 3.0
    graph.nodes[0]["nu_f"] = 9.0
    _track(graph, history)
    assert data[ALIAS_DVF[0]] == 1.0
    assert data[ALIAS_D2VF[0]] is None


@pytest.mark.parametrize("use_numpy", [True, False])
def test_tracker_uses_actual_times_and_reports_coverage(
    monkeypatch: pytest.MonkeyPatch, use_numpy: bool
) -> None:
    if not use_numpy:
        monkeypatch.setattr(coherence, "np", None)
    graph = _graph()
    history: dict = {}
    for time, capacity in ((0.0, 1.0), (1.0, 2.0), (3.0, 10.0)):
        graph.graph["_t"] = time
        graph.nodes[0]["nu_f"] = capacity
        _track(graph, history)

    assert history["B"] == [None, None, 2.0]
    assert history["capacity_rate_coverage"][-1] == {
        "node_count": 1, "first_available": 1, "second_available": 1,
        "status": "complete",
    }
    assert history["stable_frac"] == [1.0, 1.0, 1.0]
    assert history["delta_Si"] == [0.0, 0.0, 0.0]


def test_missing_time_or_capacity_is_explicit_and_clears_continuity() -> None:
    graph = _graph(time=None)
    history: dict = {}
    _track(graph, history)
    assert graph.nodes[0]["capacity_rate_diagnostic"]["status"] == "missing_time"
    assert graph.nodes[0]["_capacity_rate_samples"] == ()
    graph.graph["_t"] = 2.0
    _track(graph, history)
    assert graph.nodes[0]["capacity_rate_diagnostic"]["status"] == "initial_sample"
    del graph.nodes[0]["nu_f"]
    graph.graph["_t"] = 3.0
    _track(graph, history)
    assert graph.nodes[0]["capacity_rate_diagnostic"]["status"] == "missing_capacity"
    assert graph.nodes[0]["_capacity_rate_samples"] == ()
    assert history["B"] == [None, None, None]


def test_legacy_aliases_work_but_untimestamped_derivatives_are_not_evidence() -> None:
    graph = _graph()
    data = graph.nodes[0]
    del data["nu_f"]
    data.update({ALIAS_VF[-1]: 1.0, "_prev_vf": -900.0, "_prev_dvf": 700.0,
                 ALIAS_DVF[-1]: 11.0, ALIAS_D2VF[-1]: 12.0})
    history: dict = {}
    _track(graph, history)

    assert data[ALIAS_DVF[-1]] is data[ALIAS_D2VF[-1]] is None
    assert "_prev_vf" not in data and "_prev_dvf" not in data
    graph.graph["_t"] = 2.0
    data[ALIAS_VF[-1]] = 5.0
    _track(graph, history)
    assert data[ALIAS_DVF[-1]] == 2.0
    assert data[ALIAS_D2VF[-1]] is None


@pytest.mark.parametrize("failure", ["backward", "invalid_last_node", "bad_samples"])
def test_tracker_validates_all_nodes_before_writes(failure: str) -> None:
    graph = _graph(time=2.0)
    graph.add_node(1, nu_f=1.0)
    history: dict = {}
    _track(graph, history)
    graph.graph["_t"] = 3.0
    graph.nodes[0]["nu_f"] = 2.0
    if failure == "backward":
        graph.graph["_t"] = 1.0
    elif failure == "invalid_last_node":
        graph.nodes[1][ALIAS_VF[0]] = float("inf")
        graph.nodes[1][ALIAS_VF[-1]] = 1.0
    else:
        graph.nodes[1]["_capacity_rate_samples"] = ((2.0, 1.0), (1.0, 1.0))
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_history = deepcopy(history)

    with pytest.raises((TypeError, ValueError)):
        _track(graph, history)
    assert dict(graph.nodes(data=True)) == before_nodes
    assert history == before_history


def test_callback_chronology_failure_precedes_history_and_sense_writes() -> None:
    graph = _graph(time=2.0)
    graph.graph["METRICS"] = {"enabled": True, "verbosity": "basic"}
    _metrics_step(graph)
    graph.graph["_t"] = 1.0
    graph.nodes[0]["Si"] = 0.9
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_history = deepcopy(graph.graph["history"])

    with pytest.raises(ValueError, match="precedes"):
        _metrics_step(graph)

    assert dict(graph.nodes(data=True)) == before_nodes
    assert graph.graph["history"] == before_history


@pytest.mark.parametrize("invalid_time", [float("nan"), float("inf"), True, None])
def test_invalid_provided_time_is_not_missing_or_a_default_origin(
    invalid_time: object,
) -> None:
    graph = _graph()
    graph.graph.update(_t=invalid_time, METRICS={"enabled": True, "verbosity": "basic"})
    before_nodes = deepcopy(dict(graph.nodes(data=True)))

    with pytest.raises((TypeError, ValueError)):
        _metrics_step(graph)

    assert "history" not in graph.graph
    assert dict(graph.nodes(data=True)) == before_nodes


def test_aggregate_never_silently_averages_only_available_nodes() -> None:
    graph = _graph()
    history: dict = {}
    for time in (0.0, 1.0):
        graph.graph["_t"] = time
        graph.nodes[0]["nu_f"] = 1.0 + time**2
        _track(graph, history)
    graph.add_node(1, nu_f=1.0)
    graph.graph["_t"] = 3.0
    graph.nodes[0]["nu_f"] = 10.0
    _track(graph, history)

    assert graph.nodes[0][ALIAS_D2VF[0]] == 2.0
    assert history["B"][-1] is None
    assert history["capacity_rate_coverage"][-1]["second_available"] == 1
    assert history["capacity_rate_coverage"][-1]["node_count"] == 2


def test_empty_graph_second_difference_is_unavailable() -> None:
    history: dict = {}
    _track(nx.Graph(), history)
    assert history["B"] == [None]
    assert history["capacity_rate_coverage"][-1]["status"] == "empty"


def test_registering_metrics_never_invents_an_absent_time_origin() -> None:
    graph = _graph(time=None)
    graph.graph["METRICS"] = {"enabled": True, "verbosity": "basic"}
    register_metrics_callbacks(graph)
    assert "_capacity_rate_samples" not in graph.nodes[0]
    _metrics_step(graph)
    assert graph.nodes[0]["capacity_rate_diagnostic"]["status"] == "missing_time"


def test_runtime_metrics_honor_step_time_overrides_with_known_initial_time() -> None:
    graph = _graph()
    inject_defaults(graph)
    graph.graph.update(
        DT=0.5, VF_MAX=32.0,
        METRICS={"enabled": True, "verbosity": "basic"},
    )
    register_metrics_callbacks(graph)
    assert graph.nodes[0]["_capacity_rate_samples"] == ((0.0, 1.0),)

    # These are supplied endpoint controls for the observer, not a proposed
    # capacity evolution law. EPI still follows the shared runtime integrator.
    graph.nodes[0]["nu_f"] = 2.0
    step(graph, dt=1.0, use_Si=False, apply_glyphs=False)
    graph.nodes[0]["nu_f"] = 10.0
    step(graph, dt=2.0, use_Si=False, apply_glyphs=False)

    assert graph.graph["_t"] == 3.0
    assert graph.nodes[0][ALIAS_DVF[0]] == 4.0
    assert graph.nodes[0][ALIAS_D2VF[0]] == 2.0
    assert graph.graph["history"]["B"] == [None, 2.0]
