"""Real THOL/NAV events distinguish invariants from invented effect thresholds."""

import math

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators.definitions import SelfOrganization, Transition
from tnfr.operators.network_stage import (
    execute_pointwise_stage,
    execute_self_organization_stage,
)
from tnfr.physics.integrity import enable_integrity_monitor


@pytest.mark.parametrize("staged", [False, True])
def test_signed_thol_pressure_can_reduce_coherence_while_preserving_parent(staged):
    graph = nx.Graph(GLYPH_FACTORS={"THOL_accel": 0.25}, THOL_MAX_BIFURCATION_DEPTH=0)
    graph.add_node(
        0,
        EPI=9.0,
        nu_f=1.0,
        theta=0.0,
        delta_nfr=0.5,
        epi_time_history=[(0.0, 0.0), (1.0, 1.0), (3.0, 9.0)],
        glyph_history=["IL", "OZ"],
    )
    monitor = enable_integrity_monitor(graph)

    if staged:
        execute_self_organization_stage(graph, SelfOrganization(), (0,))
    else:
        SelfOrganization()(graph, 0)

    assert graph.nodes[0]["delta_nfr"] == 1.0
    assert graph.nodes[0]["EPI"] == 9.0
    assert graph.nodes[0]["nu_f"] == 1.0
    assert graph.nodes[0]["theta"] == 0.0
    assert tuple(graph) == (0,)
    assert monitor.latest_report.postcondition_evaluated is True
    assert monitor.latest_report.postcondition_ok is True


@pytest.mark.parametrize("staged", [False, True])
def test_real_transition_effect_below_old_cutoff_is_not_a_noop(staged):
    graph = nx.Graph(
        NAV_RANDOM=False, NAV_MIN_VF=0.0, GLYPH_FACTORS={"NAV_jitter": 1e-20}
    )
    graph.add_node(
        0, EPI=0.5, nu_f=0.0, delta_nfr=0.0, theta=0.0, glyph_history=["AL", "IL"]
    )
    monitor = enable_integrity_monitor(graph)

    if staged:
        execute_pointwise_stage(graph, Transition(), (0,), phase_shift=1e-20)
    else:
        Transition()(graph, 0, phase_shift=1e-20)

    assert 0.0 < graph.nodes[0]["delta_nfr"] < 1e-9
    assert graph.nodes[0]["theta"] == 1e-20
    assert graph.nodes[0]["nu_f"] == 0.0
    assert monitor.latest_report.postcondition_evaluated is True
    assert monitor.latest_report.postcondition_ok is True


def _interval_graph():
    graph = nx.Graph()
    graph.add_node(0, EPI=0.0, nu_f=0.0, theta=0.0, delta_nfr=0.0)
    return graph, enable_integrity_monitor(graph)


@pytest.mark.parametrize("aliases", [ALIAS_EPI, ALIAS_VF, ALIAS_THETA])
def test_thol_checks_exact_parent_invariance(aliases):
    graph, monitor = _interval_graph()
    monitor.before_operator(graph, 0)
    graph.nodes[0][aliases[0]] = 1e-20
    report = monitor.after_operator(graph, 0, "self_organization")
    assert report.postcondition_ok is False
    assert "changed" in report.postcondition_detail


@pytest.mark.parametrize("before_invalid", [False, True])
@pytest.mark.parametrize("aliases", [ALIAS_EPI, ALIAS_VF, ALIAS_DNFR, ALIAS_THETA])
def test_thol_invalid_authoritative_endpoint_cannot_fall_back(aliases, before_invalid):
    graph, monitor = _interval_graph()
    if before_invalid:
        graph.nodes[0][aliases[0]] = math.nan
    monitor.before_operator(graph, 0)
    graph.nodes[0][aliases[0]] = 0.0 if before_invalid else math.nan
    report = monitor.after_operator(graph, 0, "self_organization")
    assert report.postcondition_ok is False
    assert "finite and admitted" in report.postcondition_detail


@pytest.mark.parametrize("aliases", [ALIAS_VF, ALIAS_DNFR, ALIAS_THETA])
def test_nav_any_represented_effect_counts(aliases):
    graph, monitor = _interval_graph()
    monitor.before_operator(graph, 0)
    graph.nodes[0][aliases[0]] = 1e-20
    assert monitor.after_operator(graph, 0, "transition").postcondition_ok is True


@pytest.mark.parametrize("phase", [0.0, math.tau, -math.tau])
def test_nav_noop_or_pure_phase_chart_change_is_not_an_effect(phase):
    graph, monitor = _interval_graph()
    monitor.before_operator(graph, 0)
    graph.nodes[0][ALIAS_THETA[0]] = phase
    report = monitor.after_operator(graph, 0, "transition")
    assert report.postcondition_ok is False
    assert "No state change" in report.postcondition_detail


@pytest.mark.parametrize("aliases", [ALIAS_VF, ALIAS_DNFR, ALIAS_THETA])
def test_nav_nonfinite_value_cannot_supply_an_effect(aliases):
    graph, monitor = _interval_graph()
    monitor.before_operator(graph, 0)
    graph.nodes[0][aliases[0]] = math.inf
    report = monitor.after_operator(graph, 0, "transition")
    assert report.postcondition_ok is False
    assert "finite and admitted" in report.postcondition_detail
