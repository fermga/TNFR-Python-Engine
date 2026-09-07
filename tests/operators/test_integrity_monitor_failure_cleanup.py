"""Failed operators cannot leave a stale integrity-monitor interval."""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, THETA_PRIMARY, VF_PRIMARY
from tnfr.operators.definitions_base import Operator
from tnfr.physics.integrity import StructuralIntegrityMonitor
from tnfr.types import Glyph


class _ProbeOperator(Operator):
    __register__ = False
    name = "Coherence"
    glyph = Glyph.IL


def _graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(
        0,
        **{
            EPI_PRIMARY: 0.4,
            VF_PRIMARY: 1.0,
            DNFR_PRIMARY: 0.2,
            THETA_PRIMARY: 0.0,
        },
    )
    return graph


def test_discard_pending_operator_preserves_completed_reports():
    graph = _graph()
    monitor = StructuralIntegrityMonitor()
    monitor.before_operator(graph, 0)
    monitor.after_operator(graph, 0, "Coherence")
    completed = monitor.latest_report

    monitor.before_operator(graph, 0)
    monitor.discard_pending_operator()

    assert monitor.latest_report is completed
    assert monitor._snapshot_before is None
    assert monitor._node_state_before == {}
    assert monitor._charge_before == 0.0


def test_operator_failure_discards_pending_monitor_snapshot(monkeypatch):
    graph = _graph()
    monitor = StructuralIntegrityMonitor()
    graph.graph["integrity_monitor"] = monitor

    def fail_after_snapshot(*_args, **_kwargs):
        raise RuntimeError("synthetic operator failure")

    monkeypatch.setattr(
        "tnfr.operators.grammar_application._apply_selected_glyph",
        fail_after_snapshot,
    )

    with pytest.raises(RuntimeError, match="synthetic operator failure"):
        _ProbeOperator()._execute(graph, 0)

    assert monitor.latest_report is None
    assert monitor._snapshot_before is None
    assert monitor._node_state_before == {}
    assert monitor._charge_before == 0.0
