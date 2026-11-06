"""Independent collective-pulse derivatives from complete nodal equations.

Exact formal jets and instantaneous high-precision rows are algebraic controls;
no trajectory or retained research response is evaluated here.
"""

from dataclasses import replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, cos, pi_interval
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_partition import (
    assess_sine_moving_pattern_window,
    assess_sine_phase_offset_partition,
    observe_sine_collective_pulse,
)


def _source(*, pulse=Q(119, 100), seed=Q(1, 20), leaf_phase=(Q(0),) * 5):
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    graph.add_edges_from((i, (i + 1) % 5) for i in range(5))
    graph.add_edges_from((i, i + 5) for i in range(5))
    form_seed = (seed, -seed, Q(0), Q(0), Q(0))
    for i in range(5):
        graph.nodes[i].update(EPI=pulse / 4 + form_seed[i], theta=0, nu_f=1)
        graph.nodes[i + 5].update(EPI=-3 * pulse / 4, theta=leaf_phase[i], nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    source = bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )
    # Graph capture materializes binary64. These algebraic controls supply
    # their declared exact primitives explicitly; the observer must rebuild
    # consumed fields from them rather than trust the capture's cached rows.
    source = replace(
        source,
        epi=tuple(pulse / 4 + value for value in form_seed) + (-3 * pulse / 4,) * 5,
        phase=(Q(0),) * 5 + tuple(leaf_phase),
    )
    return graph, source


def _observe(source):
    return observe_sine_collective_pulse(
        source, cycle=tuple(range(5)), contact_turn_offsets=(0,) * 5
    )


def _product(left, right, order=4):
    return tuple(
        sum((left[j] * right[k - j] for j in range(k + 1)), Q(0))
        for k in range(order + 1)
    )


def _flat_nodal_jet(graph, form):
    """Solve the nodal coefficient recurrences, retaining every fine row."""
    x = [[Q(value)] + [Q(0)] * 4 for value in form]
    theta = [[Q(0)] * 5 for _ in graph]
    for order in range(4):
        for i in graph:
            sine_coefficient = Q(0)
            phase_coefficient = Q(0)
            for j in graph[i]:
                gap = tuple(theta[j][k] - theta[i][k] for k in range(5))
                cube = _product(_product(gap, gap), gap)
                sine_coefficient += gap[order] - cube[order] / 6
                phase_coefficient += x[i][order] - x[j][order]
            x[i][order + 1] = sine_coefficient / (graph.degree[i] * (order + 1))
            theta[i][order + 1] = phase_coefficient / (graph.degree[i] * (order + 1))
    return tuple(map(tuple, x)), tuple(map(tuple, theta))


def _collective_energy_jet(x, theta):
    u = tuple(
        sum((x[i][k] - x[i + 5][k] for i in range(5)), Q(0)) / 5 for k in range(5)
    )
    phase = tuple(
        sum((theta[i + 5][k] - theta[i][k] for i in range(5)), Q(0)) / 5
        for k in range(5)
    )
    form_square = _product(u, u)
    phase_square = _product(phase, phase)
    phase_fourth = _product(phase_square, phase_square)
    return tuple(
        form_square[k] / 2 + phase_square[k] / 2 - phase_fourth[k] / 24
        for k in range(5)
    )


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _full_rows(graph, source):
    form = tuple(_mp(value) for value in source.epi)
    phase = tuple(_mp(value) for value in source.phase)
    form_rate = tuple(
        mp.fsum(mp.sin(phase[j] - phase[i]) for j in graph[i]) / graph.degree[i]
        for i in graph
    )
    phase_rate = tuple(
        mp.fsum(form[i] - form[j] for j in graph[i]) / graph.degree[i] for i in graph
    )
    u = mp.fsum(form[i] - form[i + 5] for i in range(5)) / 5
    phi = mp.fsum(phase[i + 5] - phase[i] for i in range(5)) / 5
    du = mp.fsum(form_rate[i] - form_rate[i + 5] for i in range(5)) / 5
    dphi = mp.fsum(phase_rate[i + 5] - phase_rate[i] for i in range(5)) / 5
    return u, phi, du, dphi, u * du + mp.sin(phi) * dphi


def test_flat_seed_energy_jet_is_derived_from_every_nodal_equation():
    graph, source = _source()
    report = _observe(source)
    x, theta = _flat_nodal_jet(graph, source.epi)
    energy_jet = _collective_energy_jet(x, theta)
    assert report.flat_phase_jet_available
    assert (
        report.first_three_energy_derivatives
        == tuple(
            factor * energy_jet[order] for order, factor in ((1, 1), (2, 2), (3, 6))
        )
        == (0, 0, 0)
    )
    assert report.fourth_energy_derivative == 24 * energy_jet[4]
    u = sum((source.epi[i] - source.epi[i + 5] for i in range(5)), Q(0)) / 5
    assert report.fourth_energy_derivative == 4 * u**2 / 135 > 0
    rates = tuple(theta[i + 5][1] - theta[i][1] for i in range(5))
    mean_rate = sum(rates, Q(0)) / 5
    deviations = tuple(value - mean_rate for value in rates)
    assert report.contact_phase_rates == rates
    assert report.contact_rate_deviations == deviations
    assert (
        report.contact_rate_variance
        == sum((value**2 for value in deviations), Q(0)) / 5
        == Q(1, 180)
    )
    assert (
        report.contact_rate_third_central_moment
        == sum((value**3 for value in deviations), Q(0)) / 5
        == 0
    )
    energy = sum(
        ((source.epi[i] - source.epi[j]) ** 2 / 2 for i, j in graph.edges), Q(0)
    )
    assert energy == Q(14201, 4000)
    assert report.full_storage_bounds.contains(energy)
    assert sum((graph.degree[i] * source.epi[i] for i in graph), Q(0)) == 0
    # This positive fourth derivative means initial absorption into the
    # collective pulse, not immediate outward transfer into receiver shape.


def test_relative_phase_derivatives_match_full_fine_edge_rows_at_general_states():
    graph, base = _source()
    states = (
        (
            tuple(Q(i * i - 7 * i + 3, 19) for i in range(10)),
            tuple(Q(i * i - 3 * i, 17) for i in range(10)),
        ),
        (
            tuple(Q((-1) ** i * (i + 1), 13) for i in range(10)),
            tuple(Q((-1) ** i * (2 * i + 1), 23) for i in range(10)),
        ),
    )
    for forms, phases in states:
        source = replace(base, epi=forms, phase=phases)
        report = _observe(source)
        velocity = tuple(
            sum((forms[i] - forms[j] for j in graph[i]), Q(0)) / graph.degree[i]
            for i in graph
        )
        mean_velocity = sum(velocity[:5], Q(0)) / 5
        assert report.receiver_relative_phase_rates == tuple(
            value - mean_velocity for value in velocity[:5]
        )
        for i in range(5):
            internal = (
                sum((forms[i] - forms[j] for j in graph[i] if j < 5), Q(0))
                / graph.degree[i]
            )
            assert report.receiver_internal_phase_rates[i] == internal
            assert report.receiver_contact_phase_rates[i] == (
                report.receiver_relative_phase_rates[i] - internal
            )
        assert sum(report.receiver_relative_phase_rates, Q(0)) == 0
        with mp.workdps(90):
            form_rates = tuple(
                mp.fsum(mp.sin(_mp(phases[j] - phases[i])) for j in graph[i])
                / graph.degree[i]
                for i in graph
            )
            form_accelerations = tuple(
                mp.fsum(
                    mp.cos(_mp(phases[j] - phases[i])) * _mp(velocity[j] - velocity[i])
                    for j in graph[i]
                )
                / graph.degree[i]
                for i in graph
            )
            # Differentiate the full theta'=K L x row, then subtract the
            # receiver mean. The oracle does not use reduced L+I/L+4I rows.
            for full_form_derivatives, bounds in (
                (form_rates, report.receiver_relative_phase_acceleration_bounds),
                (form_accelerations, report.receiver_relative_phase_jerk_bounds),
            ):
                full_phase_derivatives = tuple(
                    mp.fsum(
                        full_form_derivatives[i] - full_form_derivatives[j]
                        for j in graph[i]
                    )
                    / graph.degree[i]
                    for i in graph
                )
                mean = mp.fsum(full_phase_derivatives[:5]) / 5
                for expected, interval in zip(full_phase_derivatives[:5], bounds):
                    expected -= mean
                    assert _mp(interval.lo) <= expected <= _mp(interval.hi)


def test_zero_relative_phase_rate_and_acceleration_do_not_remove_live_jerk():
    graph, base = _source()
    receiver = (Q(1), Q(-1), Q(0), Q(0), Q(0))
    internal_gradient = tuple(
        sum((receiver[i] - receiver[j] for j in graph[i] if j < 5), Q(0))
        for i in range(5)
    )
    leaves = tuple(x + gradient for x, gradient in zip(receiver, internal_gradient))
    source = replace(base, epi=receiver + leaves, phase=(Q(0),) * 10)
    report = _observe(source)
    _, phase_jet = _flat_nodal_jet(graph, source.epi)
    assert report.receiver_internal_phase_rates == tuple(
        value / 3 for value in internal_gradient
    )
    assert report.receiver_contact_phase_rates == tuple(
        -value / 3 for value in internal_gradient
    )
    assert report.receiver_relative_phase_rates == (0,) * 5
    assert report.receiver_relative_phase_acceleration_bounds == (I(0),) * 5
    exact_jerk = tuple(6 * row[3] for row in phase_jet[:5])
    mean = sum(exact_jerk, Q(0)) / 5
    assert exact_jerk == (Q(22, 9), Q(-22, 9), Q(1), Q(0), Q(-1))
    for value, interval in zip(exact_jerk, report.receiver_relative_phase_jerk_bounds):
        assert interval.contains(value - mean)
    assert report.receiver_relative_phase_jerk_bounds[0].lo > 0
    assert report.receiver_relative_phase_jerk_bounds[1].hi < 0


def test_unseeded_control_has_a_genuine_rotational_symmetry_obstruction():
    graph, source = _source(seed=0)
    report = _observe(source)
    rotation = tuple((i + 1) % 5 for i in range(5)) + tuple(
        5 + (i + 1) % 5 for i in range(5)
    )
    edges = {frozenset(edge) for edge in graph.edges}
    assert {frozenset((rotation[i], rotation[j])) for i, j in graph.edges} == edges
    assert tuple(source.epi[i] for i in rotation) == source.epi
    assert tuple(source.phase[i] for i in rotation) == source.phase
    assert report.contact_rate_variance == 0
    assert report.fourth_energy_derivative == 0
    x, theta = _flat_nodal_jet(graph, source.epi)
    assert _collective_energy_jet(x, theta)[1:] == (0, 0, 0, 0)
    # Graph and state symmetry are preserved by the unique full flow.
    # All five receiver phases therefore remain equal; the control cannot
    # acquire receiver winding even though the contact pulse can move.


def test_contact_phase_variance_changes_the_mean_energy_rate_at_equal_means():
    a, radius = Q(1, 4), Q(1, 5)
    graph, uniform = _source(pulse=Q(2, 5), seed=0, leaf_phase=(a,) * 5)
    _, varied = _source(
        pulse=Q(2, 5), seed=0, leaf_phase=(a + radius, a - radius, a, a, a)
    )
    reports = (_observe(uniform), _observe(varied))
    with mp.workdps(90):
        observations = (_full_rows(graph, uniform), _full_rows(graph, varied))
        assert all(
            abs(a - b) < mp.mpf("1e-80")
            for a, b in zip(observations[0][:2], observations[1][:2])
        )
        assert abs(observations[0][-1]) < mp.mpf("1e-80")
        assert observations[1][-1] < 0
        for report, values in zip(reports, observations):
            u, phi, du, dphi, rate = values
            assert abs(_mp(report.form_gap) - u) < mp.mpf("1e-80")
            assert (
                _mp(report.mean_contact_phase_bounds.lo)
                <= phi
                <= _mp(report.mean_contact_phase_bounds.hi)
            )
            assert (
                _mp(report.form_gap_rate_bounds.lo)
                <= du
                <= _mp(report.form_gap_rate_bounds.hi)
            )
            assert abs(_mp(report.mean_contact_phase_rate) - dphi) < mp.mpf("1e-80")
            assert (
                _mp(report.collective_energy_rate_bounds.lo)
                <= rate
                <= _mp(report.collective_energy_rate_bounds.hi)
            )
            assert (
                _mp(report.feedback_energy_rate_bounds.lo)
                <= rate
                <= _mp(report.feedback_energy_rate_bounds.hi)
            )
        assert reports[1].collective_energy_rate_bounds.hi < 0
    assert not reports[1].flat_phase_jet_available


def test_cycle_correlation_rejects_a_target_hidden_by_independent_edge_overlap():
    graph, source = _source()
    family = assess_sine_phase_offset_partition(
        source,
        blocks=(tuple(range(5)), tuple(range(5, 10))),
        phase_offset_turns=tuple(Q(i, 5) for i in range(5)) * 2,
    )
    target = family.evaluate(
        collective_form=(Q(1, 20), Q(-3, 20)),
        collective_phase_turns=(Q(-2, 5), Q(-2, 5)),
    )
    window = assess_sine_moving_pattern_window(
        target,
        form_error_bounds=(Q(1, 1000),) * 10,
        phase_error_bounds=(Q(1, 1000),) * 10,
        scaled_horizon=5,
        receiver_phase_radius=Q(1, 10),
        contact_phase_radius=Q(1, 10),
        reference_contact_bound=Q(1, 4),
    )
    energy = sum(
        ((source.epi[i] - source.epi[j]) ** 2 / 2 for i, j in graph.edges), Q(0)
    )
    independent_edges = I(0)
    for i, j in graph.edges:
        form_gap = target.fine_form[j] - target.fine_form[i]
        phase_gap = (
            2
            * pi_interval()
            * (target.fine_phase_turns[j] - target.fine_phase_turns[i])
        )
        independent_edges += (
            I(form_gap - Q(1, 500), form_gap + Q(1, 500)) ** 2 / 2
            + 1
            - cos(phase_gap + I(Q(-1, 500), Q(1, 500)))
        )
    assert independent_edges.contains(energy)
    # The five receiver gaps sum to a full turn. Jensen on the admitted
    # acute cycle gives V_R>=V5, retaining a constraint lost by the edge box.
    v5 = 5 * (I(1) - cos(2 * pi_interval() / 5))
    lower = v5.lo + Q(5, 2) * (Q(1, 5) - Q(1, 500)) ** 2
    assert lower > energy
    assert window.initial_full_storage_bounds.lo > energy
    assert window.whole_window_retention_certified
    assert window.phase_flat_acquisition_status == "not_excluded"
    # Neither maintenance nor the absence of the generic 7/2 obstruction
    # makes this particular lower-energy source compatible with the target.


def test_equal_negative_feedback_has_internal_onset_but_not_equal_winding_eligibility():
    mean_lag, amplitude = Q(1, 4), Q(1, 10)
    arrangements = ((1, -1, -1, 1, 0), (1, 1, -1, -1, 0))
    preparations = tuple(
        _source(
            pulse=Q(1),
            seed=0,
            leaf_phase=tuple(mean_lag + amplitude * sign for sign in arrangement),
        )
        for arrangement in arrangements
    )
    reports = tuple(_observe(source) for _, source in preparations)
    assert all(report.collective_energy_rate_bounds.hi < 0 for report in reports)
    assert reports[0].form_gap == reports[1].form_gap == 1
    assert reports[0].mean_contact_phase_bounds == reports[1].mean_contact_phase_bounds
    assert reports[0].collective_energy_bounds == reports[1].collective_energy_bounds
    assert (
        reports[0].contact_resultant_real_bounds
        == reports[1].contact_resultant_real_bounds
    )
    assert reports[0].contact_resultant_imag_bounds.contains(0)
    assert reports[1].contact_resultant_imag_bounds.contains(0)

    with mp.workdps(90):
        full_energies = []
        collective_rates = []
        for graph, source in preparations:
            form = tuple(_mp(value) for value in source.epi)
            phase = tuple(_mp(value) for value in source.phase)
            velocity = mp.matrix(
                [
                    mp.fsum(mp.sin(phase[j] - phase[i]) for j in graph[i])
                    / graph.degree[i]
                    for i in graph
                ]
                + [
                    mp.fsum(form[i] - form[j] for j in graph[i]) / graph.degree[i]
                    for i in graph
                ]
            )
            # Differentiate the actual twenty-component field on the full
            # graph. No reduced receiver formula generates these derivatives.
            jacobian = mp.matrix(20, 20)
            for i in graph:
                inverse_degree = mp.mpf(1) / graph.degree[i]
                for j in graph[i]:
                    coefficient = mp.cos(phase[j] - phase[i]) * inverse_degree
                    jacobian[i, 10 + j] += coefficient
                    jacobian[i, 10 + i] -= coefficient
                    jacobian[10 + i, j] -= inverse_degree
                    jacobian[10 + i, i] += inverse_degree
            acceleration = jacobian * velocity
            sine_lags = tuple(mp.sin(phase[i + 5] - phase[i]) for i in range(5))
            centered_lags = tuple(
                phase[i + 5] - phase[i] - _mp(mean_lag) for i in range(5)
            )
            assert abs(mp.fsum(mp.sin(value) for value in centered_lags)) < mp.mpf(
                "1e-80"
            )
            assert abs(
                mp.fsum(mp.cos(value) for value in centered_lags) / 5
                - (1 + 4 * mp.cos(_mp(amplitude))) / 5
            ) < mp.mpf("1e-80")
            for i in range(5):
                expected = (
                    6 * sine_lags[i] - sine_lags[(i - 1) % 5] - sine_lags[(i + 1) % 5]
                ) / 9
                assert abs(acceleration[10 + i] - expected) < mp.mpf("1e-80")

            form_second = mp.mpf(0)
            phase_fourth = mp.mpf(0)
            expected_form_second = mp.mpf(0)
            for i in range(5):
                j = (i + 1) % 5
                assert phase[i] == phase[j] and form[i] == form[j]
                assert abs(velocity[10 + i] - velocity[10 + j]) < mp.mpf("1e-80")
                form_second += (velocity[i] - velocity[j]) ** 2 + (
                    form[i] - form[j]
                ) * (acceleration[i] - acceleration[j])
                # Initial receiver gaps and their first derivatives vanish.
                # Four derivatives of 1-cos(gap) leave 3*(gap'')**2.
                phase_fourth += 3 * (acceleration[10 + i] - acceleration[10 + j]) ** 2
                expected_form_second += (sine_lags[i] - sine_lags[j]) ** 2 / 9
            assert form_second > 0 and phase_fourth > 0
            assert abs(form_second - expected_form_second) < mp.mpf("1e-80")
            full_energies.append(
                mp.fsum(
                    (form[i] - form[j]) ** 2 / 2 + 1 - mp.cos(phase[j] - phase[i])
                    for i, j in graph.edges
                )
            )
            collective_rates.append(_full_rows(graph, source)[-1])
        assert abs(full_energies[0] - full_energies[1]) < mp.mpf("1e-80")
        assert abs(collective_rates[0] - collective_rates[1]) < mp.mpf("1e-80")
        assert collective_rates[0] < 0

    graph, reflected = preparations[0]
    reflection = tuple((3 - i) % 5 for i in range(5)) + tuple(
        5 + (3 - i) % 5 for i in range(5)
    )
    edges = {frozenset(edge) for edge in graph.edges}
    assert {frozenset((reflection[i], reflection[j])) for i, j in graph.edges} == edges
    assert tuple(reflected.epi[i] for i in reflection) == reflected.epi
    assert tuple(reflected.phase[i] for i in reflection) == reflected.phase
    _, asymmetric = preparations[1]
    assert not any(
        all(
            asymmetric.phase[5 + i] == asymmetric.phase[5 + (k - i) % 5]
            for i in range(5)
        )
        for k in range(5)
    )
    # The first full trajectory preserves a receiver reflection, excluding
    # acute nonzero winding despite negative pulse feedback and positive
    # internal onsets. Breaking that reflection is not a formation proof.
