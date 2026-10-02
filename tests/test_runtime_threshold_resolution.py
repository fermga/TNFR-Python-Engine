"""Policy defaults and invalid overrides agree across actual runtime consumers."""

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.alias import get_theta_attr
from tnfr.config.defaults_core import SELECTOR_THRESHOLD_DEFAULTS
from tnfr.constants import DEFAULTS, inject_defaults
from tnfr.dynamics.adaptation import adapt_vf_after_structural_stability
from tnfr.dynamics.coordination import coordinate_global_local_phase
from tnfr.dynamics.selectors import _default_selector_logic
from tnfr.metrics.coherence import _aggregate_si
from tnfr.selector import _selector_thresholds


def _pair():
    graph = nx.path_graph(2)
    inject_defaults(graph)
    graph.graph["VF_ADAPT_TAU"] = 1
    graph.graph["VF_ADAPT_MU"] = 0.5
    for node in graph:
        graph.nodes[node].update(
            {"Si": 0.6, "ΔNFR": 0.0, "νf": 1.0 + node, "theta": float(node)}
        )
    return graph


def _history():
    return {key: [] for key in ("Si_mean", "Si_hi_frac", "Si_lo_frac")}


@pytest.mark.parametrize("patch", [{}, {"dnfr_lo": 0.1}])
def test_partial_selector_policy_is_shared_by_selector_metrics_and_capacity(patch):
    graph = _pair()
    graph.graph["SELECTOR_THRESHOLDS"] = patch
    graph.graph["GLYPH_THRESHOLDS"] = {"hi": 0.9, "lo": 0.8}
    history = _history()

    assert _default_selector_logic(graph, 0) == "IL"
    _aggregate_si(graph, history)
    adapt_vf_after_structural_stability(graph)

    assert history["Si_hi_frac"] == [1.0]
    assert history["Si_lo_frac"] == [0.0]
    assert [graph.nodes[n]["νf"] for n in graph] == [1.5, 1.5]
    assert [graph.nodes[n]["stable_count"] for n in graph] == [1, 1]


def test_explicit_selector_sense_override_reaches_all_three_consumers():
    graph = _pair()
    graph.graph["SELECTOR_THRESHOLDS"] = {"si_hi": 0.8}
    history = _history()

    assert _default_selector_logic(graph, 0) == "RA"
    _aggregate_si(graph, history)
    adapt_vf_after_structural_stability(graph)

    assert history["Si_hi_frac"] == [0.0]
    assert [graph.nodes[n]["νf"] for n in graph] == [1.0, 2.0]
    assert [graph.nodes[n]["stable_count"] for n in graph] == [0, 0]


@pytest.mark.parametrize("pressure_sign", [-1, 1])
@pytest.mark.parametrize("outside", ["neither", "sense", "pressure"])
def test_capacity_gate_keeps_inclusive_default_threshold_boundaries(
    pressure_sign, outside
):
    graph = _pair()
    high = graph.graph["SELECTOR_THRESHOLDS"]["si_hi"]
    eps = graph.graph["EPS_DNFR_STABLE"]
    sense = math.nextafter(high, -math.inf) if outside == "sense" else high
    pressure = math.nextafter(eps, math.inf) if outside == "pressure" else eps
    for node in graph:
        graph.nodes[node].update({"Si": sense, "ΔNFR": pressure_sign * pressure})
    history = _history()

    _aggregate_si(graph, history)
    adapt_vf_after_structural_stability(graph)

    admitted = outside == "neither"
    assert [graph.nodes[n]["stable_count"] for n in graph] == [int(admitted)] * 2
    assert [graph.nodes[n]["νf"] for n in graph] == (
        [1.5, 1.5] if admitted else [1.0, 2.0]
    )
    assert history["Si_hi_frac"] == [float(outside != "sense")]


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.1, 1.1, "0.5"])
@pytest.mark.parametrize("consumer", ["selector", "capacity", "aggregate"])
def test_invalid_selector_policy_rejects_consistently_before_writes(value, consumer):
    graph = _pair()
    graph.graph["SELECTOR_THRESHOLDS"] = {"si_hi": value}
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    history = _history()

    with pytest.raises(ValueError, match="si_hi"):
        if consumer == "selector":
            _default_selector_logic(graph, 0)
        elif consumer == "capacity":
            adapt_vf_after_structural_stability(graph)
        else:
            _aggregate_si(graph, history)

    assert dict(graph.nodes(data=True)) == before_nodes
    assert history == _history()


def test_resolved_selector_policy_is_detached_and_observes_in_place_overrides():
    graph = _pair()
    configured = graph.graph["SELECTOR_THRESHOLDS"]
    old = _selector_thresholds(graph)
    old["si_hi"] = 0.9
    assert _selector_thresholds(graph)["si_hi"] == SELECTOR_THRESHOLD_DEFAULTS["si_hi"]
    configured["si_hi"] = 0.7
    assert _selector_thresholds(graph)["si_hi"] == 0.7


@pytest.mark.parametrize("mode", ["legacy", "exact_components_v1"])
@pytest.mark.parametrize("patch", [{}, {"enabled": True}, {"up": 0.25}])
def test_partial_phase_policy_matches_explicit_default_overlay(mode, patch):
    partial, full = _pair(), _pair()
    partial.graph["PHASE_ADAPT"] = patch
    full.graph["PHASE_ADAPT"] = {**DEFAULTS["PHASE_ADAPT"], **patch}
    for graph in (partial, full):
        coordinate_global_local_phase(graph, global_reduction=mode)

    assert partial.graph["history"]["phase_state"][-1] == "stable"
    assert partial.graph["PHASE_K_GLOBAL"] == full.graph["PHASE_K_GLOBAL"]
    assert partial.graph["PHASE_K_LOCAL"] == full.graph["PHASE_K_LOCAL"]
    assert partial.graph["history"] == full.graph["history"]
    assert [get_theta_attr(partial.nodes[n]) for n in partial] == [
        get_theta_attr(full.nodes[n]) for n in full
    ]
    assert partial.graph["PHASE_ADAPT"] == patch


@pytest.mark.parametrize("mode", ["legacy", "exact_components_v1"])
@pytest.mark.parametrize(
    "patch",
    [
        {"enabled": "false"},
        {"up": float("nan")},
        {"down": True},
        {"R_hi": 1.1},
        {"R_lo": 0.9},
        {"disr_lo": 0.8},
        {"kL_min": 1.0},
        {"kG_min": -0.1},
        {"up": 1.1},
    ],
)
def test_invalid_phase_policy_rejects_before_history_or_state_changes(mode, patch):
    graph = _pair()
    graph.graph["PHASE_ADAPT"] = patch
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_keys = set(graph.graph)
    before_gains = graph.graph["PHASE_K_GLOBAL"], graph.graph["PHASE_K_LOCAL"]

    with pytest.raises((TypeError, ValueError)):
        coordinate_global_local_phase(graph, global_reduction=mode)

    assert dict(graph.nodes(data=True)) == before_nodes
    assert set(graph.graph) == before_keys
    assert (graph.graph["PHASE_K_GLOBAL"], graph.graph["PHASE_K_LOCAL"]) == before_gains
