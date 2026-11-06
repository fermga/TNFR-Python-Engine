"""Full-field controls for a finite moving-pattern neighborhood.

The derivative checks evaluate two instantaneous states with independent
graph-neighbor equations. No trajectory, numerical differentiator, source
search or formation claim is involved.
"""

from dataclasses import replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_partition import (
    assess_sine_moving_pattern_window,
    assess_sine_phase_offset_partition,
)


def _reference(*, moving=True):
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    graph.add_edges_from((i, (i + 1) % 5) for i in range(5))
    graph.add_edges_from((i, i + 5) for i in range(5))
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    source = bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )
    family = assess_sine_phase_offset_partition(
        source,
        blocks=(tuple(range(5)), tuple(range(5, 10))),
        phase_offset_turns=tuple(Q(i, 5) for i in range(5)) * 2,
    )
    return graph, family.evaluate(
        collective_form=(Q(1, 20), Q(-3, 20)) if moving else (0, 0),
        collective_phase_turns=(Q(-2, 5), Q(-2, 5)),
    )


def _window(reference, **overrides):
    arguments = dict(
        form_error_bounds=(Q(1, 1000),) * 10,
        phase_error_bounds=(Q(1, 1000),) * 10,
        scaled_horizon=5,
        receiver_phase_radius=Q(1, 10),
        contact_phase_radius=Q(1, 10),
        reference_contact_bound=Q(1, 4),
    )
    return assess_sine_moving_pattern_window(reference, **(arguments | overrides))


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _flow(graph, form, phase):
    return (
        tuple(
            mp.fsum(mp.sin(phase[j] - phase[i]) for j in graph[i]) / graph.degree[i]
            for i in graph
        ),
        tuple(
            mp.fsum(form[i] - form[j] for j in graph[i]) / graph.degree[i]
            for i in graph
        ),
    )


@pytest.mark.parametrize("direction", (-1, 1))
def test_relative_storage_identity_uses_actual_full_field_and_reference_pace(direction):
    graph, reference = _reference()
    report = _window(reference)
    assert report.whole_window_retention_certified
    assert report.initial_chart_certified
    assert report.reference_contact_envelope_certified
    assert report.phase_flat_acquisition_status == "not_excluded"
    assert report.retention_margin_lower_bound > 0

    with mp.workdps(90):
        xbar = tuple(_mp(value) for value in reference.fine_form)
        thetabar = tuple(2 * mp.pi * _mp(value) for value in reference.fine_phase_turns)
        xerror = tuple(_mp(Q(i - 4, 10000)) for i in graph)
        receiver_error = tuple(_mp(Q(i - 2, 20000)) for i in range(5))
        thetaerror = receiver_error + tuple(
            value + (_mp(Q(direction, 2000)) if i == 0 else 0)
            for i, value in enumerate(receiver_error)
        )
        x = tuple(a + b for a, b in zip(xbar, xerror))
        theta = tuple(a + b for a, b in zip(thetabar, thetaerror))
        dx, dtheta = _flow(graph, x, theta)
        dxbar, dthetabar = _flow(graph, xbar, thetabar)

        relative_storage = mp.mpf(0)
        direct_derivative = mp.mpf(0)
        closed_derivative = mp.mpf(0)
        actual_storage_derivative = mp.mpf(0)
        receiver_storage = mp.mpf(0)
        for i, j in graph.edges:
            eta = thetaerror[j] - thetaerror[i]
            delta = theta[j] - theta[i]
            delta_bar = thetabar[j] - thetabar[i]
            phase_velocity = dtheta[j] - dtheta[i]
            reference_velocity = dthetabar[j] - dthetabar[i]
            form_error = xerror[i] - xerror[j]
            relative_storage += (
                form_error**2 / 2
                + mp.cos(delta_bar)
                - mp.cos(delta)
                - mp.sin(delta_bar) * eta
            )
            direct_derivative += (
                form_error * (dx[i] - dx[j] - dxbar[i] + dxbar[j])
                + (mp.sin(delta) - mp.sin(delta_bar)) * phase_velocity
                - mp.cos(delta_bar) * reference_velocity * eta
            )
            closed_derivative += reference_velocity * (
                mp.sin(delta) - mp.sin(delta_bar) - mp.cos(delta_bar) * eta
            )
            actual_storage_derivative += (x[i] - x[j]) * (dx[i] - dx[j]) + mp.sin(
                delta
            ) * phase_velocity
            if i < 5 and j < 5:
                receiver_storage += (x[i] - x[j]) ** 2 / 2 + 1 - mp.cos(delta)
                assert abs(reference_velocity) < mp.mpf("1e-80")

        assert abs(actual_storage_derivative) < mp.mpf("1e-80")
        assert abs(direct_derivative - closed_derivative) < mp.mpf("1e-80")
        assert direction * closed_derivative > 0
        assert 0 < relative_storage <= _mp(report.initial_relative_storage_upper_bound)
        assert (
            abs(closed_derivative)
            <= _mp(report.growth_rate_upper_bound) * relative_storage
        )
        v5 = 5 * (1 - mp.cos(2 * mp.pi / 5))
        assert 0 < receiver_storage - v5 <= relative_storage
        assert _mp(report.propagated_relative_storage_upper_bound) < _mp(
            report.relative_storage_barrier
        )


def test_zero_pulse_trapping_does_not_prove_phase_flat_acquisition():
    _, reference = _reference(moving=False)
    report = _window(reference)
    assert report.whole_window_retention_certified
    assert report.reference_libration_energy_bounds.contains(0)
    assert report.growth_rate_upper_bound == 0
    assert (
        report.propagated_relative_storage_upper_bound
        == report.initial_relative_storage_upper_bound
    )
    assert report.initial_full_storage_bounds.hi < Q(7, 2)
    assert report.phase_flat_acquisition_status == "excluded"
    # This is an already prepared twist surrounded by a small invariant
    # neighborhood. The same conserved budget excludes a flat-phase origin.
    assert reference.fine_phase_turns[:5] == tuple(Q(i - 2, 5) for i in range(5))


def test_leaf_uncertainty_cannot_be_hidden_by_an_exact_receiver_state():
    _, reference = _reference()
    report = _window(
        reference,
        form_error_bounds=(Q(0),) * 10,
        phase_error_bounds=(Q(0),) * 5 + (Q(1, 5),) * 5,
    )
    assert not report.initial_chart_certified
    assert not report.whole_window_retention_certified
    assert report.receiver_storage_excess_upper_bound is None


def test_outside_chart_energy_lower_bound_cannot_assume_positive_relative_storage():
    _, reference = _reference(moving=False)
    report = _window(
        reference,
        form_error_bounds=(Q(0),) * 10,
        phase_error_bounds=(Q(3),) * 10,
    )
    assert not report.initial_chart_certified
    assert not report.whole_window_retention_certified
    # Every reference phase lies in [-4*pi/5,4*pi/5]. This error box
    # therefore includes the exact consensus state with zero total storage;
    # the positive Bregman remainder is unavailable outside the acute chart.
    assert report.initial_full_storage_bounds.contains(0)


def test_cached_reference_rows_storage_and_verdicts_cannot_certify_the_window():
    _, reference = _reference()
    expected = _window(reference)
    poisoned_partition = replace(
        reference.partition, invariance_certified=False, quotient_counts=None
    )
    poisoned = replace(
        reference,
        partition=poisoned_partition,
        fine_form=(Q(0),) * 10,
        fine_phase_turns=(Q(0),) * 10,
        block_phase_rates=(Q(0), Q(0)),
        full_phase_rates=(Q(0),) * 10,
        full_form_storage=Q(0),
        full_storage_bounds=I(0),
        full_row_equality_certified=False,
    )
    rebuilt = _window(poisoned)
    assert rebuilt.whole_window_retention_certified
    assert (
        rebuilt.reference_libration_energy_bounds
        == expected.reference_libration_energy_bounds
    )
    assert rebuilt.growth_rate_upper_bound == expected.growth_rate_upper_bound
    assert (
        rebuilt.propagated_relative_storage_upper_bound
        == expected.propagated_relative_storage_upper_bound
    )
    assert rebuilt.initial_full_storage_bounds == expected.initial_full_storage_bounds
    assert rebuilt.phase_flat_acquisition_status == "not_excluded"
