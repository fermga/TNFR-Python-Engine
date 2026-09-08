"""Lifecycle stabilization must be classified from structural coherence."""

from __future__ import annotations

import networkx as nx

from tnfr.operators.lifecycle import LifecycleState, get_lifecycle_state


def _state(*, depi: float) -> LifecycleState:
    graph = nx.Graph()
    graph.add_node(
        "n",
        EPI=0.9,
        nu_f=1.0,
        phase=0.0,
        delta_nfr=0.1,
        dEPI_dt=depi,
    )
    return get_lifecycle_state(graph, "n")


def test_lifecycle_does_not_treat_epi_magnitude_as_coherence() -> None:
    assert _state(depi=0.0) is LifecycleState.STABILIZATION
    assert _state(depi=1.0) is LifecycleState.ACTIVATION