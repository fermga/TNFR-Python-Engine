"""Check the introductory object's identity, response and conditional scope."""

from __future__ import annotations

import math
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tests.example_protocol_helpers import load_example
from tnfr.sdk import Network

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "01_foundations"
    / "01_hello_world.py"
)


@pytest.fixture(scope="module")
def introduction():
    """Evaluate the tiny public workflow once, retaining its complete reports."""
    return load_example(EXAMPLE).hello_world()


def test_introductory_identity_agrees_with_independent_wrapped_edge_sum(
    introduction,
):
    for name in ("initial_capture", "final_capture"):
        certificate = introduction[name]
        phase = dict(zip(certificate.field.nodes, certificate.field.phase))
        cycle = certificate.cycle
        turn = math.fsum(
            math.remainder(phase[right] - phase[left], math.tau)
            for left, right in zip(cycle, cycle[1:] + cycle[:1])
        )

        assert turn == pytest.approx(math.tau, abs=1e-14)
        assert certificate.cycle_winding == round(turn / math.tau)
        assert certificate.admitted
        assert certificate.storage_margin_lower_bound > 0


def test_introduction_retains_actual_joint_response_and_exact_step_defects(
    introduction,
):
    step = introduction["step"]
    assert introduction["initial_capture"].field == step.before
    assert introduction["final_capture"].field == step.after
    assert step.t_after - step.t_before == Q(1, 1024)
    assert step.before.epi != step.after.epi
    assert step.before.phase != step.after.phase
    assert step.before.edges == step.after.edges
    assert step.before.capacity == step.after.capacity

    for i, node in enumerate(step.before.nodes):
        actual = Q(step.after.epi[i]) - Q(step.before.epi[i])
        predicted = Q(step.dt) * Q(step.before.form_rate[i])
        assert actual - predicted == step.epi_update_defect[i]
        assert introduction["network"].G.nodes[node]["EPI"] == step.after.epi[i]

    assert (
        step.energy_change - Q(step.dt) * step.before.storage_rate
        == step.energy_step_defect
    )
    assert step.energy_step_defect != 0


def test_initial_storage_loss_matches_independent_cycle_form_laplacian(
    introduction,
):
    field = introduction["initial_capture"].field
    graph = introduction["network"].G
    form = dict(zip(field.nodes, map(Q, field.epi)))
    expected = Q(0)
    for node in graph:
        gradient = sum((form[node] - form[neighbor] for neighbor in graph[node]), Q(0))
        expected += Q(1, 2) * gradient**2 / graph.degree[node]

    assert expected > 0
    assert field.continuous_loss == expected


def test_same_support_at_phase_consensus_does_not_certify_winding_one(
    introduction,
):
    graph = introduction["network"].G.copy()
    for node in graph:
        graph.nodes[node]["theta"] = 0
    control = Network(graph).relational_cycle_capture(
        introduction["initial_capture"].field.model, cycle=tuple(graph)
    )

    assert control.cycle_winding == 0
    assert not control.admitted
    assert control.target_sector is None
    assert "declared_cycle_period_not_certified" in control.unavailable_reasons
