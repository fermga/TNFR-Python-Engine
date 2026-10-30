"""Independent algebraic controls for uncertain, positive held capacities.

All examples are constructed. No retained source, event, trajectory or nominal
certificate is loaded. Multiplying rate equations by pi below keeps their exact
arithmetic rational; it does not select another clock or integrate a response.
"""

from dataclasses import fields
from fractions import Fraction as Q
from itertools import product

import pytest

from tnfr.mathematics._rational_interval import pi_interval
from tnfr.research import sine_preparation_metric as metric
from tnfr.research.sine_capacity_metric import bound_capacity_metric_cells


def _dot(left, right):
    return sum((a * b for a, b in zip(left, right, strict=True)), Q(0))


def _diagonal(diagonal, vector):
    return tuple(a * b for a, b in zip(diagonal, vector, strict=True))


def _add(left, right):
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _subtract(left, right):
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _flux(size, edges, values):
    answer = [Q(0)] * size
    for (i, j), value in zip(edges, values, strict=True):
        answer[i] += value
        answer[j] -= value
    return tuple(answer)


def _laplacian(edges, weights, vector):
    return _flux(
        len(vector),
        edges,
        tuple(w * (vector[i] - vector[j]) for (i, j), w in zip(edges, weights)),
    )


@pytest.mark.parametrize("secants", [(Q(-2, 3), Q(1, 4), Q(5, 6)), (Q(1),) * 3])
def test_exact_metric_cancellation_retains_both_capacity_forcing_terms(secants):
    # This is the exact matrix identity for any admitted signed secants.
    # The coefficients are not offered as a particular sine-solution segment.
    edges = ((0, 1), (1, 2), (0, 2))
    weights = (Q(2), Q(3), Q(5))
    nominal_mobility = (Q(1), Q(2), Q(3))
    member_mobility = (Q(3, 2), Q(7, 4), Q(4))
    mismatch = _subtract(member_mobility, nominal_mobility)
    u, v = (Q(2), Q(-1), Q(3)), (Q(-2), Q(4), Q(1))
    nominal_x = (Q(-1), Q(2), Q(4))
    nominal_edge_sines = (Q(1, 3), Q(-1, 2), Q(2, 5))
    sine_nodes = _flux(3, edges, nominal_edge_sines)
    form_gradient = _laplacian(edges, weights, nominal_x)
    forcing_form = _diagonal(tuple(-d for d in mismatch), sine_nodes)
    forcing_phase = _diagonal(mismatch, form_gradient)

    cv = _laplacian(edges, secants, v)
    ku = _laplacian(edges, weights, u)
    lv = _laplacian(edges, (Q(1),) * 3, v)
    scaled_form_rate = _add(
        _diagonal(tuple(-d for d in member_mobility), cv), forcing_form
    )
    scaled_phase_rate = _add(_diagonal(member_mobility, ku), forcing_phase)
    half_scaled_energy_rate = _dot(ku, scaled_form_rate) + _dot(lv, scaled_phase_rate)
    cancellation = _dot(ku, _diagonal(member_mobility, _subtract(lv, cv)))
    form_work, phase_work = _dot(ku, forcing_form), _dot(lv, forcing_phase)

    assert half_scaled_energy_rate == cancellation + form_work + phase_work
    assert form_work != 0 and phase_work != 0
    assert half_scaled_energy_rate != cancellation + form_work
    assert half_scaled_energy_rate != cancellation + phase_work
    if secants == (Q(1),) * 3:
        assert cancellation == 0
        assert half_scaled_energy_rate == form_work + phase_work != 0


def _same_state_rows():
    # x=(0,1), theta=(0,pi/2), one edge of conductance 2, bare degrees 1.
    # Thus B sin(B.T theta)=(-1,1) and L_weighted x=(-2,2), exactly.
    nominal_mobility, member_mobility = (Q(1), Q(1, 2)), (Q(2), Q(1, 2))
    sine_nodes, form_gradient = (Q(-1), Q(1)), (Q(-2), Q(2))

    def rows(mobility):
        return (
            _diagonal(tuple(-d for d in mobility), sine_nodes),
            _diagonal(mobility, form_gradient),
        )

    return rows(nominal_mobility), rows(member_mobility)


def test_equal_initial_state_has_nonzero_error_derivatives_in_both_rows():
    nominal, member = _same_state_rows()
    error_rates = tuple(_subtract(a, b) for a, b in zip(member, nominal, strict=True))
    assert error_rates == ((Q(1), Q(0)), (Q(-2), Q(0)))
    # Smoothness implies departure from zero error, without a numerical solve.
    # A homogeneous nonnegative radius comparison initialized at zero would
    # remain zero and therefore cannot bound this two-capacity family.
    assert (
        2 * (error_rates[0][0] - error_rates[0][1]) ** 2
        + (error_rates[1][0] - error_rates[1][1]) ** 2
        == 6
    )


def test_each_members_weights_conserve_means_but_nominal_weights_drift():
    nominal, member = _same_state_rows()
    nominal_alpha, member_alpha = (Q(1, 3), Q(2, 3)), (Q(1, 5), Q(4, 5))
    assert tuple(_dot(nominal_alpha, row) for row in nominal) == (0, 0)
    assert tuple(_dot(member_alpha, row) for row in member) == (0, 0)
    assert tuple(_dot(nominal_alpha, row) for row in member) == (Q(1, 3), Q(-2, 3))
    errors = tuple(_subtract(a, b) for a, b in zip(member, nominal, strict=True))
    assert tuple(_dot(nominal_alpha, row) for row in errors) == (Q(1, 3), Q(-2, 3))


def test_constant_child_requires_both_drifting_offsets():
    nominal, member = _same_state_rows()
    error_rates = tuple(_subtract(a, b) for a, b in zip(member, nominal, strict=True))
    alpha = (Q(1, 3), Q(2, 3))
    mean_rates = tuple(_dot(alpha, row) for row in error_rates)
    offset_rates = tuple(-value for value in mean_rates)
    assert offset_rates == (Q(-1, 3), Q(2, 3))
    for row, mean_rate, offset_rate in zip(
        error_rates, mean_rates, offset_rates, strict=True
    ):
        centered_parent_rate = row[0] - mean_rate
        actual_portrait_rate = -row[0]  # The isolated child's error is constant.
        assert actual_portrait_rate == offset_rate - centered_parent_rate
        assert actual_portrait_rate != -centered_parent_rate


def test_uniform_growth_allowance_encloses_signed_secant_and_capacity_vertices():
    edges, weights = ((0, 1), (1, 2), (0, 2)), (Q(2), Q(3), Q(5))
    capacities = (Q(2), Q(4), Q(6))
    capacity_box = ((Q(1), Q(3)), (Q(3), Q(5)), (Q(4), Q(8)))
    secant_box = ((Q(-3, 4), Q(-1, 4)), (Q(1, 4), Q(3, 4)), (Q(-1), Q(1, 2)))
    context = metric.prepare_metric_context(
        capacities=capacities, edges=edges, conductances=weights, degrees=(2, 2, 2)
    )
    coefficient = metric.bound_secant_metric(context, secant_intervals=secant_box)
    eta = max(
        max(nominal - lower, upper - nominal) / nominal
        for nominal, (lower, upper) in zip(capacities, capacity_box, strict=True)
    )
    assert eta == Q(1, 2)
    norm_upper = coefficient.norm_upper + eta * coefficient.degree_upper
    vectors = ((Q(1), Q(1), Q(1)), (Q(2), Q(-1), Q(3)), (Q(-2), Q(4), Q(1)))
    for member_capacities, secants, u, v in product(
        product(*capacity_box), product(*secant_box), vectors, vectors
    ):
        mobility = tuple(capacity / 2 for capacity in member_capacities)
        ku = _laplacian(edges, weights, u)
        lv = _laplacian(edges, (Q(1),) * 3, v)
        cv = _laplacian(edges, secants, v)
        cross = _dot(ku, _diagonal(mobility, _subtract(lv, cv)))
        form_energy, phase_energy = _dot(u, ku), _dot(v, lv)
        assert cross**2 <= norm_upper**2 * form_energy * phase_energy

    # A single nonacute edge gives an independent witness that omitting
    # capacity uncertainty from the coefficient is not generally sound.
    edge = ((0, 1),)
    context = metric.prepare_metric_context(
        capacities=(1, Q(1, 2)), edges=edge, conductances=(4,), degrees=(1, 1)
    )
    coefficient = metric.bound_secant_metric(context, secant_intervals=((-1, -1),))
    u = v = (Q(1), Q(-1))
    ku = _laplacian(edge, (Q(4),), u)
    lv = _laplacian(edge, (Q(1),), v)
    cv = _laplacian(edge, (Q(-1),), v)
    cross = _dot(ku, _diagonal((Q(2), Q(1)), _subtract(lv, cv)))
    energy_product = _dot(u, ku) * _dot(v, lv)
    assert cross == 96 and energy_product == 64
    assert cross**2 > coefficient.norm_upper**2 * energy_product
    assert (
        cross**2
        <= (coefficient.norm_upper + coefficient.degree_upper) ** 2 * energy_product
    )


def _two_node_family(*, cut=False, errors=Q(1, 1024)):
    context = metric.prepare_metric_context(
        capacities=(1, Q(1, 2)),
        edges=((0, 1),),
        conductances=(2,),
        degrees=(1, 1),
    )
    reading = (
        {
            "child_form_error": (errors, 2 * errors),
            "child_phase_error": (3 * errors, errors),
        }
        if cut
        else {"child": 1}
    )
    return metric.prepare_metric_family(
        context,
        form_errors=((errors, 2 * errors), (3 * errors, errors)),
        phase_errors=((2 * errors, errors), (errors, 4 * errors)),
        parent=0,
        reference_weight=2,
        **reading,
    )


@pytest.mark.parametrize("cut", [False, True])
def test_zero_capacity_uncertainty_recovers_old_cells_exactly(cut):
    family = _two_node_family(cut=cut)
    # Uniform nominal x=4 and theta=-1 are an exact constant full-sine
    # solution for every positive capacity. No trajectory was synthesized.
    phase_cells = (((-1, -1), (-1, -1)),) * 2
    form_cells = (((4, 4), (4, 4)),) * 2
    durations, allowances = (Q(1, 64), Q(1, 32)), (Q(1, 8),)
    old = metric.bound_metric_cells(
        family, phase_cells=phase_cells, edge_allowances=allowances, durations=durations
    )
    new = bound_capacity_metric_cells(
        family,
        capacity_intervals=((1, 1), (Q(1, 2), Q(1, 2))),
        phase_cells=phase_cells,
        form_cells=form_cells,
        edge_allowances=allowances,
        durations=durations,
    )
    assert old.status == new.status == "completed"
    assert old.reason is new.reason is None
    assert new.relative_capacity_error == 0
    assert old.initial_edge_margins == new.initial_edge_margins
    assert old.final_normalized_radius == new.final_normalized_radius
    assert old.final_radius_upper == new.final_radius_upper
    assert old.final_portrait_error_upper == new.final_portrait_error_upper
    assert old.attempted_cells == new.attempted_cells == 2
    assert old.completed_cells == new.completed_cells == 2
    assert new.final_form_mean_error == family.form_mean_error
    assert new.final_phase_mean_error == family.phase_mean_error
    assert new.final_form_offset == family.form_offset
    assert new.final_phase_offset == family.phase_offset
    for old_step, new_step in zip(old.steps, new.steps, strict=True):
        # Resource accounting may include the additional primitive batch;
        # every scientific value shared with the old cell is exact equality.
        for name in (
            "index",
            "time",
            "duration",
            "inflated_edges",
            "normalized_radius_before",
            "normalized_radius_after",
            "radius_upper",
            "edge_margins",
            "portrait_error_upper",
            "domain_passed",
        ):
            assert getattr(old_step, name) == getattr(new_step, name)
        for field in fields(old_step.coefficient):
            if field.name != "maximum_bits":
                assert getattr(old_step.coefficient, field.name) == getattr(
                    new_step.coefficient, field.name
                )
        assert new_step.gamma_upper == old_step.coefficient.gamma_upper
        assert new_step.forcing_upper == new_step.forcing_norm_squared == 0
        assert new_step.form_mean_drift_upper == new_step.phase_mean_drift_upper == 0


@pytest.mark.parametrize("cut", [False, True])
def test_uncertain_capacity_encloses_exact_uniform_equilibrium(cut):
    family = _two_node_family(cut=cut, errors=Q(0))
    result = bound_capacity_metric_cells(
        family,
        capacity_intervals=((Q(1, 4), 2), (Q(1, 8), Q(3, 4))),
        phase_cells=(((0, 0), (0, 0)),) * 2,
        form_cells=(((4, 4), (4, 4)),) * 2,
        edge_allowances=(Q(1, 8),),
        durations=(Q(1, 64), Q(1, 32)),
    )
    assert result.status == "completed"
    assert result.relative_capacity_error == 1
    # The shared sine kernel may return a tiny outward interval at zero.
    # Inclusion of this exact solution does not demand a sharp zero bound.
    assert 0 <= result.final_radius_upper < Q(1, 2**100)
    assert 0 <= result.final_portrait_error_upper < Q(1, 2**100)
    assert 0 <= result.final_form_mean_drift_upper < Q(1, 2**100)
    assert 0 <= result.final_phase_mean_drift_upper < Q(1, 2**100)
    for step in result.steps:
        assert (
            step.nominal_form_gradient == step.forcing_phase_intervals == ((0, 0),) * 2
        )
        for lower, upper in step.nominal_sine_nodes + step.forcing_form_intervals:
            assert lower <= 0 <= upper
        for lower, upper in (
            step.form_mean_error,
            step.phase_mean_error,
            step.form_offset,
            step.phase_offset,
        ):
            assert lower <= 0 <= upper


def test_forcing_and_both_cut_drift_enclose_an_exact_instantaneous_witness():
    family = _two_node_family(cut=True, errors=Q(0))
    result = bound_capacity_metric_cells(
        family,
        capacity_intervals=((1, 2), (Q(1, 2), Q(1, 2))),
        phase_cells=(((0, 0), (1, 2)),),
        form_cells=(((0, 0), (1, 1)),),
        edge_allowances=(Q(1, 8),),
        durations=(Q(1, 1024),),
    )
    # This raw box contains the exact state theta=(0,pi/2), x=(0,1).
    # It is not asserted to enclose a nominal solution cell: this checks
    # arithmetic inclusion of its instantaneous forcing and mean rates only.
    step = result.steps[0]
    nominal, member = _same_state_rows()
    for actual_row, intervals in zip(
        (_subtract(a, b) for a, b in zip(member, nominal, strict=True)),
        (step.forcing_form_intervals, step.forcing_phase_intervals),
        strict=True,
    ):
        for actual, (lower, upper) in zip(actual_row, intervals, strict=True):
            assert lower <= actual <= upper
    assert step.forcing_norm_squared >= 6
    assert step.forcing_norm_squared > 4  # Omitting the form row gives 4.
    assert step.forcing_norm_squared > 2  # Omitting the phase row gives 2.
    assert step.forcing_upper > 0
    # The exact witness has absolute mean rates 1/(3*pi), 2/(3*pi).
    # Multiplication by a rigorous lower pi bound makes these rational
    # sufficient inequalities, independently of the implementation's kappa.
    assert step.form_mean_rate_upper * pi_interval().lo >= Q(1, 3)
    assert step.phase_mean_rate_upper * pi_interval().lo >= Q(2, 3)
    assert step.form_mean_drift_upper > 0 and step.phase_mean_drift_upper > 0
    assert step.form_offset == (-step.form_mean_drift_upper, step.form_mean_drift_upper)
    assert step.phase_offset == (
        -step.phase_mean_drift_upper,
        step.phase_mean_drift_upper,
    )
    assert step.portrait_error_upper > family.observation_root_upper * step.radius_upper
