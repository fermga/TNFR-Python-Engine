"""Independent field and energy controls for finite directed saddle passage.

Only static states and exact linear algebra are evaluated. No numerical flow,
shooting, parameter sweep or previously frozen producer is executed.
"""

from dataclasses import replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._exact_linear_algebra import exact_matrix_inverse
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_corridor import (
    assess_sine_saddle_corridor,
    assess_sine_saddle_retention_band,
)


def _lift(row):
    a, b, c, d = row
    return (a, b, Q(0), -b, -a, c, d, Q(0), -d, -c)


def _source(*, forms=(Q(1, 6), Q(1, 6), Q(1, 4), Q(5, 24)), phases=None):
    if phases is None:
        phases = (Q(-9, 5), Q(-9, 10), Q(-9, 5), Q(-9, 10))
    graph = nx.cycle_graph(5)
    graph.add_edges_from((i, i + 5) for i in range(5))
    for i in graph:
        graph.nodes[i].update(EPI=0, theta=0, nu_f=1)
    captured = bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )
    return graph, replace(captured, epi=_lift(forms), phase=_lift(phases))


def _report(source):
    return assess_sine_saddle_corridor(
        source, cycle=range(5), lower_phase=-2, upper_phase=Q(-3, 2)
    )


def _mp(value):
    return mp.mpf(value.numerator) / value.denominator


def _product(matrix, row):
    return tuple(sum((a * b for a, b in zip(line, row)), Q(0)) for line in matrix)


def _dot(left, right):
    return sum((a * b for a, b in zip(left, right)), Q(0))


@pytest.fixture(scope="module")
def prepared():
    graph, source = _source()
    return graph, source, _report(source)


def test_directed_passage_has_compatible_retention_budget_but_no_acute_claim(prepared):
    graph, source, report = prepared
    with mp.workdps(95):
        x, phase = tuple(map(_mp, source.epi)), tuple(map(_mp, source.phase))
        energy = sum(
            (x[i] - x[j]) ** 2 / 2 + 1 - mp.cos(phase[j] - phase[i])
            for i, j in graph.edges
        )
        assert (
            _mp(report.full_storage_bounds.lo)
            <= energy
            <= _mp(report.full_storage_bounds.hi)
        )
        floor = 5 * (1 - mp.cos(2 * mp.pi / 5))
        ceiling = floor + 9 * mp.pi**2 / 1600
        assert mp.mpf("3.5") < energy < ceiling
        assert 5 - mp.cos(mp.mpf("3.6")) - 4 * mp.cos(mp.mpf(".9")) + mp.mpf(
            53
        ) / 576 == pytest.approx(energy, rel=mp.mpf("1e-90"))
    assert report.directed_exit_certified
    assert report.initial_winding == 1 and report.exit_winding == 0
    assert report.initial_reaction_velocity == Q(1, 12)
    assert report.initial_momentum == Q(53, 24)
    assert report.initial_cycle_gap_bounds[-1].abs_max > 3
    assert 0 < report.residence_time_upper_bound < 28
    assert report.residence_time_upper_bound * report.force_lower_bound == (
        report.momentum_abs_upper_bound - report.initial_momentum
    )
    assert report.original_time_upper_bound > 3 * report.residence_time_upper_bound


def test_momentum_geometry_and_sharp44_follow_from_full_form_storage(prepared):
    graph, _, report = prepared
    # Reconstruct the restricted storage metric from every original edge,
    # including the fixed nodes and private contacts, rather than a quoted G.
    basis = tuple(_lift(tuple(Q(i == j) for i in range(4))) for j in range(4))
    metric = tuple(
        tuple(
            sum(
                (basis[i][a] - basis[i][b]) * (basis[j][a] - basis[j][b])
                for a, b in graph.edges
            )
            for j in range(4)
        )
        for i in range(4)
    )
    inverse = exact_matrix_inverse(metric)
    momentum = tuple(map(Q, (6, 3, 2, 1)))
    rate = tuple(
        sum(basis[j][0] - basis[j][neighbor] for neighbor in graph[0]) / graph.degree[0]
        for j in range(4)
    )
    dual_momentum, dual_rate = _product(inverse, momentum), _product(inverse, rate)
    assert _dot(momentum, dual_momentum) == Q(53, 2)
    assert _dot(rate, dual_rate) == Q(2, 9)
    assert _dot(momentum, dual_rate) == 1
    assert _dot(momentum, report.form_coordinates) == report.initial_momentum
    assert _dot(rate, report.form_coordinates) == report.initial_reaction_velocity

    coefficient = _dot(momentum, dual_rate) / _dot(rate, dual_rate)
    constrained = tuple(a - coefficient * b for a, b in zip(dual_momentum, dual_rate))
    energy = _dot(constrained, _product(metric, constrained)) / 2
    assert _dot(rate, constrained) == 0
    assert _dot(momentum, constrained) ** 2 == 44 * energy
    # At a lower exit u'<=0. The residual dual vector has squared norm22;
    # the positive9/2 multiple of u' cannot increase a positive momentum.
    residual = tuple(a - coefficient * b for a, b in zip(momentum, rate))
    assert coefficient == Q(9, 2)
    assert _dot(residual, _product(inverse, residual)) == 22


def test_full_nonlinear_rows_give_the_momentum_force_with_live_contacts():
    graph, source = _source(
        forms=(Q(1, 5), Q(-1, 20), Q(1, 6), Q(1, 9)),
        phases=(Q(-9, 5), Q(-7, 8), Q(-17, 10), Q(-19, 20)),
    )
    report = _report(source)
    with mp.workdps(95):
        theta = tuple(map(_mp, source.phase))
        form_rates = tuple(
            sum(mp.sin(theta[j] - theta[i]) for j in graph[i]) / graph.degree[i]
            for i in graph
        )
        full_force = sum(
            weight * form_rates[i] for weight, i in zip((6, 3, 2, 1), (0, 1, 5, 6))
        )
        u, v = theta[:2]
        structural_force = -2 * (mp.sin(u / 2) * mp.cos(v - u / 2) + mp.sin(2 * u))
        assert abs(full_force - structural_force) < mp.mpf("1e-90")
        assert (
            _mp(report.initial_momentum_rate_bounds.lo)
            <= full_force
            <= _mp(report.initial_momentum_rate_bounds.hi)
        )
        # Environmental sine currents cancel only in this full momentum;
        # the receiver's individual form row still depends on its contact.
        contact = mp.sin(theta[5] - theta[0]) / 3
        assert abs(contact) > mp.mpf(".01")
        assert full_force >= _mp(report.force_lower_bound)


def test_reversing_forms_does_not_pass_a_squared_momentum_test(prepared):
    _, source, positive = prepared
    reversed_source = replace(source, epi=tuple(-value for value in source.epi))
    negative = _report(reversed_source)
    assert negative.full_storage_bounds == positive.full_storage_bounds
    assert negative.initial_momentum**2 == positive.initial_momentum**2
    assert negative.positive_force_certified
    assert not negative.left_exit_excluded
    assert not negative.directed_exit_certified
    assert negative.exit_winding is None
    assert negative.residence_time_upper_bound is None
    assert "lower_exit_momentum_exclusion_not_certified" in negative.reasons


def test_negative_force_bound_remains_an_actual_conditional_lower_bound():
    graph, source = _source(forms=(Q(1), Q(1), Q(3, 2), Q(5, 4)))
    report = _report(source)
    assert report.force_lower_bound < 0
    assert not report.directed_exit_certified
    # Compare the lower estimate against the analytical force at the declared
    # left face with cos(s) at its energetic bound. Negative force requires
    # the larger left prefactor; the smaller right one would be unsound.
    with mp.workdps(95):
        lower, energy = _mp(report.lower_phase), _mp(report.full_storage_bounds.hi)
        bracket = 6 + 8 * mp.cos(lower) + 6 * mp.cos(lower) ** 2 - energy
        envelope = -mp.tan(lower / 2) * bracket / 2
        assert _mp(report.force_lower_bound) <= envelope
    assert report.initial_momentum_rate_bounds != I(0)


def test_retention_band_is_a_conditional_theorem_not_a_captured_trajectory(prepared):
    _, source, _ = prepared
    asymmetric = replace(source, epi=(Q(100),) + source.epi[1:])
    report = assess_sine_saddle_retention_band(asymmetric, cycle=range(5))
    assert not report.reduction.source_membership_certified
    assert report.theorem_certified
    assert not report.captured_source_retention_certified
    assert not report.captured_source_formation_certified
    assert report.conditional_premises
    assert report.checkpoint_offset_from_deep_hit == Q(-1, 2)
    assert report.retained_scaled_duration == 1


def test_retention_band_margin_follows_from_paired_face_energy(prepared):
    report = assess_sine_saddle_retention_band(prepared[1], cycle=range(5))
    with mp.workdps(95):
        u = _mp(report.lower_phase)
        # Put one of the paired receiver gaps exactly on the acute face.
        # The other equals -u-pi/2, and the closing gap is2pi+2u.
        edge_gaps = (mp.pi / 2, -u - mp.pi / 2) * 2 + (2 * mp.pi + 2 * u,)
        face = sum(1 - mp.cos(gap) for gap in edge_gaps)
        surplus = face - _mp(report.full_storage_upper_bound)
        assert (
            _mp(report.paired_face_surplus_bounds.lo)
            <= surplus
            <= _mp(report.paired_face_surplus_bounds.hi)
        )
        assert surplus / 4 > _mp(report.cycle_gap_margin_lower_bound)
        assert face - mp.mpf("3.5") == pytest.approx(
            2 * (mp.sin(u) + mp.mpf(".5")) ** 2, rel=mp.mpf("1e-85")
        )


def test_dwell_time_and_independent_target_radius_preserve_the_whole_window(prepared):
    report = assess_sine_saddle_retention_band(prepared[1], cycle=range(5))
    nearest_face = min(
        report.deep_phase - report.lower_phase,
        report.upper_phase - report.deep_phase,
    )
    duration_each_side = nearest_face / report.reaction_speed_upper_bound
    assert duration_each_side > report.retained_scaled_duration / 2
    assert report.backward_band_residence_lower_bound == duration_each_side
    assert report.forward_band_residence_lower_bound == duration_each_side
    assert report.scaled_dwell_lower_bound == 2 * duration_each_side
    with mp.workdps(95):
        speed = mp.sqrt(
            4
            * (_mp(report.full_storage_upper_bound) - 5 * (1 - mp.cos(2 * mp.pi / 5)))
            / 9
        )
        assert speed <= _mp(report.computed_reaction_speed_upper_bound)
        assert speed < _mp(report.reaction_speed_upper_bound)
        gap_error = 2 * _mp(report.target_phase_error_bound) * mp.exp(2)
        assert gap_error < _mp(report.target_cycle_gap_error_upper_bound)
        assert gap_error < _mp(report.cycle_gap_margin_lower_bound)
    assert report.target_retention_margin_lower_bound > 0
