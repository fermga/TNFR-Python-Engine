"""Experimental relaxation snapshots use canonical local coherence."""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.operators.metrics_u6 import measure_tau_relax_observed


def _snapshot(*, pressure: float, rate: float) -> dict[str, object]:
    graph = nx.Graph()
    graph.add_node(
        "n",
        nu_f=1.0,
        delta_nfr=pressure,
        dEPI_dt=rate,
    )
    return measure_tau_relax_observed(graph, "n")


def test_singleton_under_high_pressure_is_not_reported_as_perfectly_coherent() -> None:
    result = _snapshot(pressure=9.0, rate=0.0)

    assert result["coherence_initial"] == pytest.approx(0.1)
    assert result["coherence_kind"] == "radius_one_structural_coherence"


def test_relaxation_snapshot_coherence_includes_recorded_epi_rate() -> None:
    static = _snapshot(pressure=1.0, rate=0.0)
    moving = _snapshot(pressure=1.0, rate=2.0)

    assert static["coherence_initial"] == pytest.approx(0.5)
    assert moving["coherence_initial"] == pytest.approx(0.25)


def test_snapshot_does_not_claim_an_observed_relaxation_time() -> None:
    result = _snapshot(pressure=0.0, rate=0.0)

    assert result["metric_type"] == "u6_relaxation_snapshot"
    assert result["measurement_kind"] == "initial_snapshot_with_estimated_timescale"
    assert result["tau_relax_observed"] is None
    assert result["requires_monitoring_infrastructure"] is True