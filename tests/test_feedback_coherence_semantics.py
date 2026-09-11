"""Feedback decisions must use the canonical local coherence channels."""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.config.operator_names import COHERENCE, DISSONANCE, SELF_ORGANIZATION
from tnfr.dynamics.feedback import StructuralFeedbackLoop


def _loop(*, dnfr: float, depi: float) -> StructuralFeedbackLoop:
    graph = nx.Graph()
    graph.add_node(
        "n",
        EPI=1.0,
        nu_f=1.0,
        phase=0.0,
        delta_nfr=dnfr,
        dEPI_dt=depi,
    )
    return StructuralFeedbackLoop(graph, "n")


def test_feedback_local_coherence_uses_constitutive_kernel() -> None:
    assert _loop(dnfr=1.0, depi=0.0)._compute_local_coherence() == pytest.approx(
        0.5
    )
    assert _loop(dnfr=0.0, depi=0.5)._compute_local_coherence() == pytest.approx(
        2.0 / 3.0
    )


def test_recorded_rate_changes_feedback_operator_selection() -> None:
    static = _loop(dnfr=0.0, depi=0.0)
    driven = _loop(dnfr=0.0, depi=1.0)
    for loop in (static, driven):
        loop.target_coherence = 0.8
        loop.COHERENCE_TOL_LOW = 0.1
        loop.COHERENCE_TOL_HIGH = 0.1
        loop.DNFR_THRESHOLD = 10.0
        loop.EPI_THRESHOLD = 0.0

    assert static.regulate() == DISSONANCE
    assert driven.regulate() == COHERENCE


@pytest.mark.parametrize("pressure", [-0.2, 0.2])
def test_feedback_pressure_decision_is_sign_symmetric(pressure: float) -> None:
    loop = _loop(dnfr=pressure, depi=0.0)
    loop.target_coherence = 0.7
    loop.COHERENCE_TOL_LOW = 0.5
    loop.COHERENCE_TOL_HIGH = 0.5
    loop.DNFR_THRESHOLD = 0.1

    assert loop.regulate() == SELF_ORGANIZATION