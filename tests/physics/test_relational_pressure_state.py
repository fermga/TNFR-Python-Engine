"""Pressure is an observation; a P2 relative chart retains missing state.

The ideal closure argument belongs to
JOINT_PARAMETER_RESPONSE.md#pressure-state-closure. These static native checks
retain ordinary binary64 tolerances, not exact transcendental identities or
trajectory certificates. Pressure is evaluated before its derivative is read.
"""

import math

import networkx as nx
import pytest

from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
)

MODEL = RelationalExchangeModel(1.5, epi_weight=3.0, phase_weight=1.0)
CAPACITY = 0.75


def _pair(form_difference, phase_difference):
    graph = nx.path_graph(2)
    for node, sign in ((0, -1), (1, 1)):
        graph.nodes[node].update(
            EPI=sign * form_difference / 2,
            theta=sign * phase_difference / 2,
            nu_f=CAPACITY,
            delta_nfr=99.0,
        )
    return graph


def _pressure_rate(field):
    """Differentiate the independent regular P2 law p=e*d+w*delta/pi."""
    form_difference_rate = field.form_rate[1] - field.form_rate[0]
    phase_difference_rate = field.phase_rate[1] - field.phase_rate[0]
    return (
        MODEL.epi_weight * form_difference_rate
        + MODEL.phase_weight / math.pi * phase_difference_rate
    )


def _directional_pressure_probe(graph, field):
    """Two static pressure evaluations along F; neither is a time step."""
    increment = 2.0**-16
    samples = []
    for sign in (-1, 1):
        probe = graph.copy()
        for index, node in enumerate(field.nodes):
            probe.nodes[node]["EPI"] += sign * increment * field.form_rate[index]
            probe.nodes[node]["theta"] += sign * increment * field.phase_rate[index]
        samples.append(evaluate_relational_exchange(probe, model=MODEL).pressure[0])
    return (samples[1] - samples[0]) / (2 * increment)


def test_compensated_zero_pressure_is_not_full_equilibrium_or_a_closed_pressure_state():
    delta = math.pi / 8
    e, w, beta = MODEL.epi_weight, MODEL.phase_weight, MODEL.storage_scale
    difference = -w * delta / (math.pi * e)
    graph = _pair(difference, delta)
    compensated = evaluate_relational_exchange(graph, model=MODEL)
    consensus = evaluate_relational_exchange(_pair(0.0, 0.0), model=MODEL)
    assert compensated.pressure == pytest.approx(consensus.pressure, rel=0, abs=3e-17)
    assert max(map(abs, compensated.form_rate)) < 3e-17
    assert consensus.form_rate == consensus.phase_rate == (0.0, 0.0)
    assert compensated.phase_rate[0] > 0 > compensated.phase_rate[1]

    metric = math.pi * math.sin(delta) / delta
    expected = -2 * CAPACITY * w**3 * delta / (beta * math.pi**2 * e * metric)
    assert expected < 0
    assert _pressure_rate(consensus) == 0.0
    assert _pressure_rate(compensated) == pytest.approx(expected, rel=3e-14, abs=1e-17)
    assert _directional_pressure_probe(graph, compensated) == pytest.approx(
        expected, rel=0, abs=2e-12
    )
    # The theorem uses exactly equal ideal pressure; tiny native residuals
    # above are reported as rounding, not promoted to exact state equality.


@pytest.mark.parametrize(
    "difference,delta", ((3 / 8, -0.3), (-1 / 4, 0.4), (1 / 8, 0.0))
)
def test_pressure_and_form_contrast_reconstruct_the_regular_relative_field(
    difference, delta
):
    graph = _pair(difference, delta)
    field = evaluate_relational_exchange(graph, model=MODEL)
    e, w, beta = MODEL.epi_weight, MODEL.phase_weight, MODEL.storage_scale
    expected_pressure = e * difference + w * delta / math.pi
    assert field.pressure == pytest.approx(
        (expected_pressure, -expected_pressure), rel=3e-15, abs=2e-17
    )
    assert field.pressure != (99.0, 99.0)

    # Reconstruction consumes independently calculated p and observed d.
    # It is a chart change for this supplied law, not a pressure definition
    # obtained by dividing the evaluated form rate by capacity.
    pressure = field.pressure[0]
    recovered_delta = math.pi * (pressure - e * difference) / w
    assert recovered_delta == pytest.approx(delta, rel=0, abs=8e-16)
    metric = (
        math.pi
        if recovered_delta == 0
        else (math.pi * math.sin(recovered_delta) / recovered_delta)
    )
    expected_difference_rate = -2 * CAPACITY * pressure
    expected_phase_difference_rate = 2 * CAPACITY * w * difference / (beta * metric)
    expected_pressure_rate = (
        -2 * e * CAPACITY * pressure
        + 2 * CAPACITY * w**2 * difference / (beta * math.pi * metric)
    )
    assert field.form_rate[1] - field.form_rate[0] == pytest.approx(
        expected_difference_rate, rel=3e-15, abs=2e-17
    )
    assert field.phase_rate[1] - field.phase_rate[0] == pytest.approx(
        expected_phase_difference_rate, rel=3e-15, abs=2e-17
    )
    assert _pressure_rate(field) == pytest.approx(
        expected_pressure_rate, rel=3e-15, abs=2e-17
    )
    assert _directional_pressure_probe(graph, field) == pytest.approx(
        expected_pressure_rate, rel=0, abs=2e-12
    )
