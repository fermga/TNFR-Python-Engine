"""Constructed reset/metric interfaces, without a successor trajectory."""

from fractions import Fraction as Q
from itertools import product

import pytest

from tnfr.research import sine_preparation_metric as metric
from tnfr.research.nul_reset import bound_nul_reset


@pytest.fixture(autouse=True)
def no_continuation_or_coefficient_evaluation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("reset controls must not evaluate a continuation")

    for name in ("bound_metric_cells", "bound_secant_metric", "_coefficient_bound"):
        monkeypatch.setattr(metric, name, forbidden)


def _errors(triple):
    lower, nominal, upper = triple
    return nominal - lower, upper - nominal


def _mean(weights, values):
    return sum((a * value for a, value in zip(weights, values, strict=True)), Q(0))


def _energy(form, phase, edges, weights):
    return sum(
        (
            weight * (form[i] - form[j]) ** 2 + (phase[i] - phase[j]) ** 2
            for (i, j), weight in zip(edges, weights, strict=True)
        ),
        Q(0),
    )


def _coordinates(reset):
    """Read actual proposal outputs independently of the image error fields."""
    return tuple(Q(proposal.epi_after) for proposal in reset.proposals)


def test_represented_capacity_reset_rebuilds_mean_weights_and_metric_initial_box():
    reset = bound_nul_reset(
        factor=0.3,
        capacity=0.7,
        form_bounds=(0.5, 0.625, 0.875),
        pressure_bounds=(-0.25, 0.125, 0.5),
    )
    # The successor law consumes the represented product, not the ideal product.
    represented_capacity = Q(0.7 * 0.3)
    assert reset.capacity_after == represented_capacity
    assert represented_capacity != Q(0.7) * Q(0.3)
    capacities = (represented_capacity, Q(1), Q(3, 2))
    degrees = (1, 2, 1)
    edges, weights = ((0, 1), (1, 2)), (Q(2), Q(3))
    context = metric.prepare_metric_context(
        capacities=capacities, edges=edges, conductances=weights, degrees=degrees
    )
    rho = tuple(Q(d) / nu for d, nu in zip(degrees, capacities, strict=True))
    alpha = tuple(value / sum(rho) for value in rho)
    old_rho = (Q(1) / Q(0.7), Q(2), Q(2, 3))
    old_alpha = tuple(value / sum(old_rho) for value in old_rho)
    assert context.rho == rho
    assert context.alpha == alpha != old_alpha
    assert context.beta == (represented_capacity, Q(1, 2), Q(3, 2))

    forms = (
        _coordinates(reset),
        (Q(1, 4), Q(3, 8), Q(1, 2)),
        (Q(-1, 2), Q(-3, 8), Q(-1, 4)),
    )
    # The primary NUL branch preserves these primitive phase coordinates.
    phases = (
        (Q(1, 8), Q(1, 4), Q(3, 8)),
        (Q(1, 4), Q(5, 16), Q(7, 16)),
        (Q(-1, 8), Q(0), Q(1, 16)),
    )
    form_errors = tuple(_errors(values) for values in forms)
    phase_errors = tuple(_errors(values) for values in phases)
    assert form_errors[0] == (
        reset.image_form.lower_error,
        reset.image_form.upper_error,
    )
    family = metric.prepare_metric_family(
        context,
        form_errors=form_errors,
        phase_errors=phase_errors,
        parent=0,
        child=2,
        reference_weight=2,
    )
    expected_square = sum(
        weight * (max(form_errors[i]) + max(form_errors[j])) ** 2
        + (max(phase_errors[i]) + max(phase_errors[j])) ** 2
        for (i, j), weight in zip(edges, weights, strict=True)
    )
    assert family.initial_norm_squared == expected_square > 0
    assert family.initial_radius_upper**2 >= expected_square
    for errors, actual in (
        (form_errors, family.form_mean_error),
        (phase_errors, family.phase_mean_error),
    ):
        expected = (
            -_mean(alpha, tuple(pair[0] for pair in errors)),
            _mean(alpha, tuple(pair[1] for pair in errors)),
        )
        assert actual == expected
        assert actual[0] < 0 < actual[1]

    means = ([], [])
    for values in product(*(forms + phases)):
        u = tuple(values[i] - forms[i][1] for i in range(3))
        v = tuple(values[i + 3] - phases[i][1] for i in range(3))
        means[0].append(_mean(alpha, u))
        means[1].append(_mean(alpha, v))
        assert _energy(u, v, edges, weights) <= family.initial_norm_squared
        # Connected observations remove a common mean, but not constituent error.
        portrait_square = (v[2] - v[0]) ** 2 + 2 * (u[2] - u[0]) ** 2
        assert portrait_square <= family.observation_factor * expected_square
    assert (min(means[0]), max(means[0])) == family.form_mean_error
    assert (min(means[1]), max(means[1])) == family.phase_mean_error


def test_reset_to_cut_host_keeps_changed_means_and_both_child_offsets():
    reset = bound_nul_reset(
        factor=0.5,
        capacity=1.0,
        form_bounds=(0.5, 0.625, 0.875),
        pressure_bounds=(-0.25, 0.0, 0.5),
    )
    context = metric.prepare_metric_context(
        capacities=(reset.capacity_after, Q(2)),
        edges=((0, 1),),
        conductances=(2,),
        degrees=(1, 1),
    )
    assert context.alpha == (Q(4, 5), Q(1, 5))
    forms = (_coordinates(reset), (Q(1, 4), Q(3, 8), Q(5, 8)))
    phases = ((Q(1, 8), Q(1, 4), Q(1, 2)), (Q(5, 16), Q(3, 8), Q(1, 2)))
    child_form = (Q(1, 4), Q(3, 8), Q(7, 16))
    child_phase = (Q(-1, 4), Q(-1, 8), Q(1, 8))
    family = metric.prepare_metric_family(
        context,
        form_errors=tuple(_errors(values) for values in forms),
        phase_errors=tuple(_errors(values) for values in phases),
        parent=0,
        reference_weight=2,
        child_form_error=_errors(child_form),
        child_phase_error=_errors(child_phase),
    )
    assert family.initial_norm_squared == Q(27, 64)
    assert family.form_mean_error == (Q(-3, 40), Q(3, 20))
    assert family.phase_mean_error == (Q(-9, 80), Q(9, 40))
    assert family.form_offset == (Q(-11, 40), Q(11, 80))
    assert family.phase_offset == (Q(-7, 20), Q(29, 80))
    offsets = ([], [])
    for values in product(*(forms + phases + (child_form, child_phase))):
        u = tuple(values[i] - forms[i][1] for i in range(2))
        v = tuple(values[i + 2] - phases[i][1] for i in range(2))
        uc, vc = values[4] - child_form[1], values[5] - child_phase[1]
        # Exact weights are determined by the successor capacities.
        mean_u, mean_v = (4 * u[0] + u[1]) / 5, (4 * v[0] + v[1]) / 5
        bx, bt = uc - mean_u, vc - mean_v
        offsets[0].append(bx)
        offsets[1].append(bt)
        assert _energy(u, v, ((0, 1),), (Q(2),)) <= Q(27, 64)
        assert bt**2 + 2 * bx**2 <= family.offset_norm_upper**2
        portrait_square = (vc - v[0]) ** 2 + 2 * (uc - u[0]) ** 2
        portrait_upper = (
            family.observation_root_upper * family.initial_radius_upper
            + family.offset_norm_upper
        )
        assert portrait_square <= portrait_upper**2
    assert (min(offsets[0]), max(offsets[0])) == family.form_offset
    assert (min(offsets[1]), max(offsets[1])) == family.phase_offset
    # Dropping either changed mean fails at an attainable corner of the box.
    assert min(offsets[0]) < -_errors(child_form)[0]
    assert max(offsets[0]) > _errors(child_form)[1]
    assert min(offsets[1]) < -_errors(child_phase)[0]
    assert max(offsets[1]) > _errors(child_phase)[1]


def test_zero_reset_state_error_cannot_compare_different_capacity_laws():
    reset = bound_nul_reset(
        factor=0.5,
        capacity=1.0,
        form_bounds=(0.25,) * 3,
        pressure_bounds=(0.5,) * 3,
    )
    context = metric.prepare_metric_context(
        capacities=(reset.capacity_after, Q(2)),
        edges=((0, 1),),
        conductances=(1,),
        degrees=(1, 1),
    )
    family = metric.prepare_metric_family(
        context,
        form_errors=((0, 0),) * 2,
        phase_errors=((0, 0),) * 2,
        parent=0,
        child=1,
        reference_weight=1,
    )
    assert family.initial_norm_squared == family.initial_radius_upper == 0
    # Compare laws at the SAME successor x and equal phases. S = 0, while
    # pi * theta_dot = diag(nu / degree) L_W x differs exactly.
    x = (reset.image_form.nominal, Q(3, 4))
    q = (x[0] - x[1], x[1] - x[0])
    old_scaled_phase_row = (q[0], 2 * q[1])
    new_scaled_phase_row = (context.beta[0] * q[0], context.beta[1] * q[1])
    assert old_scaled_phase_row == (Q(-5, 8), Q(5, 4))
    assert new_scaled_phase_row == (Q(-5, 16), Q(5, 4))
    assert new_scaled_phase_row[0] - old_scaled_phase_row[0] == Q(5, 16)
    # Hence a same-law, zero-error estimate cannot reuse a pre-event nominal law.
