"""Real integrity monitors participate in stage transactions without enum hooks."""

from __future__ import annotations

import copyreg
from copy import deepcopy
from enum import Enum

import networkx as nx
import pytest

from tnfr.errors import TNFRValueError
from tnfr.operators.definitions import Coherence, Dissonance, Emission, Silence
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.network_stage import (
    GraphTransactionSnapshot,
    execute_dissonance_stage,
)
from tnfr.physics.integrity import MonitorMode, enable_integrity_monitor


def _source(mode=MonitorMode.OBSERVE):
    graph = nx.Graph(
        GLYPH_FACTORS={"OZ_dnfr_factor": 2.0},
        OZ_NOISE_MODE=False,
        RANDOM_SEED=19,
    )
    for node in (0, 1):
        graph.add_node(
            node, EPI=0.5, nu_f=1.0, delta_nfr=0.2, theta=0.0, glyph_history=["AL"]
        )
    graph.add_edge(0, 1, weight=1.0)
    monitor = enable_integrity_monitor(
        graph,
        mode=mode,
        conservation_threshold=0.0,
        lyapunov_tolerance=1e6,
        charge_drift_threshold=1e6,
    )
    return graph, monitor


def _context():
    return ValidatedSequence([Emission(), Dissonance(), Coherence(), Silence()]).step(1)


@pytest.mark.parametrize("mode", tuple(MonitorMode))
def test_real_monitor_modes_work_in_atomic_dissonance_stage(mode):
    graph, monitor = _source(mode)
    execute_dissonance_stage(graph, Dissonance(), (0,), sequence_context=_context())
    assert graph.graph["integrity_monitor"] is monitor
    assert monitor.mode is mode
    assert graph.nodes[0]["delta_nfr"] == pytest.approx(0.4)
    if mode is MonitorMode.OFF:
        assert monitor.latest_report is None
    else:
        assert monitor.latest_report.postcondition_evaluated
        assert monitor.latest_report.postcondition_ok


def test_late_failure_restores_real_monitor_reports_and_graph():
    graph, monitor = _source()
    monitor.before_operator(graph, 0)
    monitor.after_operator(graph, 0, "coherence")
    report = monitor.latest_report
    summary = monitor.summary
    reports = summary.reports
    before_nodes = deepcopy(dict(graph.nodes))
    before_graph = deepcopy(
        {key: value for key, value in graph.graph.items() if key != "integrity_monitor"}
    )

    def fail_after_committed_event(subject):
        assert monitor.summary.total_operators == 2
        assert monitor.latest_report is not report
        monitor.mode = MonitorMode.OFF
        subject.nodes[0]["late_marker"] = True
        raise RuntimeError("synthetic late refresh rejection")

    with pytest.raises(RuntimeError, match="late refresh rejection"):
        execute_dissonance_stage(
            graph,
            Dissonance(),
            (0,),
            sequence_context=_context(),
            compute_delta_nfr=fail_after_committed_event,
        )

    assert graph.graph["integrity_monitor"] is monitor
    assert monitor.mode is MonitorMode.OBSERVE
    assert monitor.summary is summary
    assert monitor.summary.reports is reports
    assert monitor.latest_report is report
    assert monitor.summary.total_operators == 1
    assert len(reports) == 1
    assert monitor._snapshot_before is None
    assert monitor._node_state_before == {}
    assert dict(graph.nodes) == before_nodes
    assert {
        key: value for key, value in graph.graph.items() if key != "integrity_monitor"
    } == before_graph


def test_core_enum_member_namespace_is_captured_not_assumed_immutable(monkeypatch):
    graph, monitor = _source()
    payload = ["before"]
    monkeypatch.setattr(MonitorMode.OBSERVE, "audit_payload", payload, raising=False)
    snapshot = GraphTransactionSnapshot(graph)
    payload.append("after")
    monitor.mode = MonitorMode.ENFORCE
    snapshot.restore(graph)
    assert monitor.mode is MonitorMode.OBSERVE
    assert MonitorMode.OBSERVE.audit_payload is payload
    assert payload == ["before"]


@pytest.mark.parametrize("owner", (MonitorMode, Enum))
def test_modified_core_enum_copy_hooks_reject_without_execution(monkeypatch, owner):
    graph, _monitor = _source()
    calls = []

    def trap(self, memo):
        calls.append("called")
        return self

    monkeypatch.setattr(owner, "__deepcopy__", trap, raising=False)
    with pytest.raises(TNFRValueError, match="MonitorMode.*modified copy-protocol"):
        GraphTransactionSnapshot(graph)
    assert not calls


def test_core_enum_instance_copy_hook_is_not_trusted(monkeypatch):
    graph, _monitor = _source()
    calls = []

    def trap(memo):
        calls.append("called")
        return MonitorMode.OBSERVE

    # Restore absence, not the inherited Enum bound method, at test teardown.
    monkeypatch.setitem(vars(MonitorMode.OBSERVE), "__deepcopy__", trap)
    with pytest.raises(TNFRValueError, match="instance copy-protocol"):
        GraphTransactionSnapshot(graph)
    assert not calls


def test_core_enum_registered_copy_reducer_is_not_trusted(monkeypatch):
    graph, _monitor = _source()
    calls = []

    def trap(value):
        calls.append("called")
        return str, ("observe",)

    monkeypatch.setitem(copyreg.dispatch_table, MonitorMode, trap)
    with pytest.raises(TNFRValueError, match="registered copyreg"):
        GraphTransactionSnapshot(graph)
    assert not calls


def test_external_enum_does_not_inherit_core_admission():
    class ExternalMode(Enum):
        OBSERVE = "observe"

    graph, monitor = _source()
    monitor.mode = ExternalMode.OBSERVE
    with pytest.raises(TNFRValueError, match="copy-protocol|custom __deepcopy__"):
        GraphTransactionSnapshot(graph)


def test_fabricated_unregistered_core_enum_instance_rejects():
    graph, monitor = _source()
    monitor.mode = object.__new__(MonitorMode)
    with pytest.raises(TNFRValueError, match="registered singleton"):
        GraphTransactionSnapshot(graph)
