"""OZ monitor policy is bound to its local before/after interval.

Finite synthetic operator events only; these controls do not establish a
stochastic destabilization law or a continuous TNFR trajectory.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators.definitions import Coherence, Dissonance, Emission, Silence
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.network_stage import execute_dissonance_stage
from tnfr.physics.integrity import MonitorMode, enable_integrity_monitor


def _graph(*, noise: bool = True) -> nx.DiGraph:
    graph = nx.DiGraph(
        GLYPH_FACTORS={"OZ_dnfr_factor": 2.0},
        RANDOM_SEED=19,
        OZ_NOISE_MODE=noise,
        OZ_SIGMA=0.25,
        OZ_MIN_PROPAGATION=0.0,
    )
    for node, pressure in enumerate((0.2, 0.1)):
        graph.add_node(
            node,
            **{
                ALIAS_EPI[0]: 0.5,
                ALIAS_VF[0]: 1.0,
                ALIAS_DNFR[0]: pressure,
                ALIAS_THETA[0]: 0.0,
                "glyph_history": ["AL"],
                "source_glyph": "AL",
            },
        )
    graph.add_edge(0, 1, weight=1.0)
    return graph


def _context():
    return ValidatedSequence([Emission(), Dissonance(), Coherence(), Silence()]).step(1)


@pytest.mark.parametrize("staged", [False, True])
def test_real_noise_event_can_reduce_pressure_without_local_contract_failure(staged):
    graph = _graph()
    monitor = enable_integrity_monitor(graph, mode=MonitorMode.OBSERVE)

    if staged:
        execute_dissonance_stage(graph, Dissonance(), (0,), sequence_context=_context())
    else:
        Dissonance()(graph, 0, sequence_context=_context())

    # Existing seed 19, offset 0, first draw: -0.03957071586097255.
    # This fixed fixture neither searches seeds nor fits a statistical claim.
    assert graph.nodes[0][ALIAS_DNFR[0]] == pytest.approx(0.16042928413902746)
    assert graph.nodes[0]["_rng_jitter_progress"]["draws"] == 1
    assert graph.nodes[1][ALIAS_DNFR[0]] > 0.1
    report = monitor.latest_report
    assert report.postcondition_evaluated is True
    assert report.postcondition_ok is True


@pytest.mark.parametrize("staged", [False, True])
def test_real_amplification_still_satisfies_local_contract(staged):
    graph = _graph(noise=False)
    monitor = enable_integrity_monitor(graph, mode=MonitorMode.OBSERVE)
    if staged:
        execute_dissonance_stage(graph, Dissonance(), (0,), sequence_context=_context())
    else:
        Dissonance()(graph, 0, sequence_context=_context())
    assert graph.nodes[0][ALIAS_DNFR[0]] == pytest.approx(0.4)
    assert monitor.latest_report.postcondition_ok is True


def test_stage_monitor_checks_local_amplification_before_incoming_cancellation():
    graph = _graph(noise=False)
    graph.nodes[0][ALIAS_DNFR[0]] = 0.3
    graph.nodes[1][ALIAS_DNFR[0]] = -0.2
    monitor = enable_integrity_monitor(graph, mode=MonitorMode.OBSERVE)

    execute_dissonance_stage(graph, Dissonance(), (0, 1), sequence_context=_context())

    # Target 1's local -0.4 passes before the incoming +0.3 is reduced.
    assert graph.nodes[1][ALIAS_DNFR[0]] == pytest.approx(-0.1)
    assert len(monitor.summary.reports) == 2
    assert all(report.postcondition_evaluated for report in monitor.summary.reports)
    assert all(report.postcondition_ok for report in monitor.summary.reports)


@pytest.mark.parametrize("noise_before", [False, True])
def test_mode_is_bound_before_live_policy_and_stale_telemetry_change(noise_before):
    graph = _graph(noise=noise_before)
    monitor = enable_integrity_monitor(graph, mode=MonitorMode.OBSERVE)
    monitor.before_operator(graph, 0)
    graph.graph["OZ_NOISE_MODE"] = not noise_before
    graph.graph["operator_metrics"] = [
        {"operator": "dissonance", "node": 0, "noise_mode": not noise_before}
    ]
    graph.nodes[0]["_oz_precondition_context"] = {"noise_mode": not noise_before}
    graph.nodes[0][ALIAS_DNFR[0]] = 0.1

    report = monitor.after_operator(graph, 0, "dissonance")

    assert report.postcondition_evaluated is True
    assert report.postcondition_ok is noise_before
    if not noise_before:
        assert "decreased" in report.postcondition_detail


@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("before_invalid", [False, True])
def test_noise_never_licenses_nonfinite_pressure(nonfinite, before_invalid):
    graph = _graph()
    monitor = enable_integrity_monitor(graph, mode=MonitorMode.OBSERVE)
    if before_invalid:
        graph.nodes[0][ALIAS_DNFR[0]] = nonfinite
    monitor.before_operator(graph, 0)
    graph.nodes[0][ALIAS_DNFR[0]] = 0.1 if before_invalid else nonfinite

    report = monitor.after_operator(graph, 0, "dissonance")

    assert report.postcondition_evaluated is True
    assert report.postcondition_ok is False
    assert "finite" in report.postcondition_detail


@pytest.mark.parametrize("mode", [True, "true"])
def test_capture_reuses_runtime_noise_flag_admission(mode):
    graph = _graph()
    graph.graph["OZ_NOISE_MODE"] = mode
    monitor = enable_integrity_monitor(graph, mode=MonitorMode.OBSERVE)
    monitor.before_operator(graph, 0)
    graph.nodes[0][ALIAS_DNFR[0]] = 0.1
    assert monitor.after_operator(graph, 0, "dissonance").postcondition_ok is True


def test_invalid_captured_policy_is_not_repaired_by_later_valid_policy():
    graph = _graph()
    graph.graph["OZ_NOISE_MODE"] = "not-a-runtime-flag"
    monitor = enable_integrity_monitor(graph, mode=MonitorMode.OBSERVE)
    monitor.before_operator(graph, 0)
    graph.graph["OZ_NOISE_MODE"] = True
    graph.nodes[0][ALIAS_DNFR[0]] = 0.1
    report = monitor.after_operator(graph, 0, "dissonance")
    assert report.postcondition_ok is False
    assert "policy" in report.postcondition_detail


def test_invalid_oz_policy_does_not_block_unrelated_postcondition():
    graph = _graph()
    graph.graph["OZ_NOISE_MODE"] = "not-a-runtime-flag"
    monitor = enable_integrity_monitor(graph, mode=MonitorMode.OBSERVE)
    monitor.before_operator(graph, 0)
    report = monitor.after_operator(graph, 0, "coherence")
    assert report.postcondition_evaluated is True
    assert report.postcondition_ok is True


def test_noise_capture_is_single_use_and_next_amplification_is_not_exempt():
    graph = _graph()
    monitor = enable_integrity_monitor(graph, mode=MonitorMode.OBSERVE)
    monitor.before_operator(graph, 0)
    graph.nodes[0][ALIAS_DNFR[0]] = 0.1
    first = monitor.after_operator(graph, 0, "dissonance")
    graph.graph["OZ_NOISE_MODE"] = False
    monitor.before_operator(graph, 0)
    graph.nodes[0][ALIAS_DNFR[0]] = 0.05
    second = monitor.after_operator(graph, 0, "dissonance")
    unsampled = monitor.after_operator(graph, 0, "dissonance")
    assert first.postcondition_ok is True
    assert second.postcondition_ok is False
    assert unsampled.postcondition_evaluated is False
