"""Report-free exact controls for nonlinear modified-energy bounds."""

from fractions import Fraction as Q

import numpy as np
import pytest

from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_product,
    exact_symmetric_semidefinite,
)
from tnfr.physics._sine_lyapunov import (
    _sine_common_quadratic_coefficients,
    _sine_common_quadratic_initial_upper,
    _sine_lyapunov_coefficients,
    _sine_lyapunov_initial_upper,
    _sine_lyapunov_return_squared,
)

GAP, RATE = Q(1, 2), Q(2)
ETA_LOW, ETA_HIGH, COSINE = Q(1, 16), Q(1, 8), Q(1, 3)


def _coefficients(**changes):
    values = dict(
        eta_lower=ETA_LOW,
        eta_upper=ETA_HIGH,
        cosine_lower=COSINE,
        gap_lower=GAP,
        rate_upper=RATE,
    )
    return _sine_lyapunov_coefficients(**(values | changes))


@pytest.mark.parametrize("damping", [GAP, Q(1), RATE])
@pytest.mark.parametrize("hessian_upper", [Q(1), Q(3, 2)])
@pytest.mark.parametrize("stiffness_position", [Q(0), Q(1, 2), Q(1)])
@pytest.mark.parametrize(
    "position,velocity",
    [(Q(0), Q(1)), (Q(1), Q(0)), (Q(1), Q(2)), (Q(-2), Q(1)), (Q(3, 7), Q(-5, 11))],
)
def test_exact_damped_gradient_rows_prove_decay_and_energy_sandwich(
    damping, stiffness_position, position, velocity, hessian_upper
):
    mu, maximum, lower, upper, decay = _coefficients(phase_hessian_upper=hessian_upper)
    epsilon = GAP / 4
    stiffness = mu + stiffness_position * (maximum - mu)
    potential = stiffness * position**2 / 2
    modified = (
        velocity**2 / 2
        + potential
        + epsilon * position * velocity
        + epsilon * damping * position**2 / 2
    )
    # Differentiate the full two-row quadratic gradient system independently;
    # no helper supplies the gradient or the exact mixed-term cancellation.
    position_rate, velocity_rate = velocity, -damping * velocity - stiffness * position
    position_gradient = (
        stiffness * position + epsilon * velocity + epsilon * damping * position
    )
    velocity_gradient = velocity + epsilon * position
    derivative = position_gradient * position_rate + velocity_gradient * velocity_rate
    assert (
        derivative
        == -(damping - epsilon) * velocity**2 - epsilon * stiffness * position**2
    )
    assert derivative <= -decay * modified
    assert (
        velocity**2 / 4 + lower * position**2
        <= modified
        <= 3 * velocity**2 / 4 + upper * position**2
    )


def test_own_hessian_upper_changes_nonlinear_stiffness_and_decay():
    sine, enlarged = _coefficients(), _coefficients(phase_hessian_upper=Q(3, 2))
    assert enlarged[0] == sine[0] and enlarged[2] == sine[2]
    assert enlarged[1] == Q(3, 2) * sine[1]
    assert enlarged[3] - sine[3] == ETA_HIGH * RATE**2 / 4
    assert enlarged[4] < sine[4]


@pytest.mark.parametrize("upper", [True, 1.0, Q(0), Q(-1), Q(1, 4)])
def test_invalid_or_contradictory_hessian_upper_rejects(upper):
    with pytest.raises(ValueError, match="Hessian|hessian"):
        _coefficients(phase_hessian_upper=upper)


@pytest.mark.parametrize(
    "form,phase", [(Q(1, 10), Q(1, 20)), (Q(-3, 10), Q(1, 5)), (Q(0), Q(-1, 2))]
)
def test_initial_bound_encloses_exact_physical_coordinate_energy(form, phase):
    _, _, _, upper, _ = _coefficients()
    # An exact member of the admitted class: A=1, gamma=1/4 and quadratic
    # phase curvature1, giving transformed stiffness eta/2=1/32.
    eta, gamma, epsilon = Q(1, 16), Q(1, 4), GAP / 4
    physical_potential = phase**2 / 2
    actual = (
        eta * (form**2 + physical_potential) / 2
        + epsilon * gamma * phase * form
        + epsilon * phase**2 / 2
    )
    bound = _sine_lyapunov_initial_upper(
        eta_upper=ETA_HIGH,
        rate_upper=RATE,
        gap_lower=GAP,
        position_upper=upper,
        form_norm_upper=abs(form),
        phase_norm_upper=abs(phase),
    )
    assert type(bound) is Q and actual <= bound


@pytest.mark.parametrize(
    "changes", [{"cosine_lower": None}, {"cosine_lower": Q(0)}, {"eta_lower": Q(0)}]
)
def test_missing_or_zero_coercivity_never_supplies_a_decay_rate(changes):
    mu, maximum, lower, upper, decay = _coefficients(**changes)
    assert decay is None
    assert maximum == ETA_HIGH * RATE**2
    assert upper > 0
    if changes.get("cosine_lower", 1) is None:
        assert mu is lower is None
    else:
        assert mu == 0 and lower == GAP**2 / 16


def test_exact_decay_below_interval_grid_preserves_nonzero_return_bounds():
    energy, form_square, phase_square = _sine_lyapunov_return_squared(
        initial_upper=Q(1, 3),
        decay_upper=Q(1, 2**512),
        eta_lower=Q(1, 10),
        gap_lower=Q(1, 5),
        rate_upper=Q(2),
        position_lower=Q(1, 400),
    )
    assert energy == Q(1, 3 * 2**512)
    assert form_square == Q(200, 3 * 2**512)
    assert phase_square == Q(800, 3 * 2**512)
    assert all(
        type(value) is Q and 0 < value < Q(1, 2**128)
        for value in (energy, form_square, phase_square)
    )
    # Recovering the underlying energy from either norm checks both metric
    # denominators independently, including the form factor four.
    assert form_square * Q(1, 10) * Q(1, 5) / 4 == energy
    assert phase_square * Q(1, 400) / 2 == energy


@pytest.mark.parametrize(
    "initial,decay", [(Q(0), Q(1)), (Q(3, 7), Q(0)), (Q(3, 7), Q(1))]
)
def test_zero_and_no_decay_keep_exact_endpoint_semantics(initial, decay):
    returned, form_square, phase_square = _sine_lyapunov_return_squared(
        initial_upper=initial,
        decay_upper=decay,
        eta_lower=Q(1, 10),
        gap_lower=Q(1, 5),
        rate_upper=Q(2),
        position_lower=Q(1, 400),
    )
    assert returned == initial * decay
    assert form_square == 200 * returned
    assert phase_square == 800 * returned


def _common_coefficients(**changes):
    bounds = dict(
        gamma_lower=Q(1, 8),
        gamma_upper=Q(1, 2),
        cosine_lower=Q(1, 2),
        gap_lower=Q(1),
        capacity_lower=Q(1),
        capacity_upper=Q(3),
    )
    return _sine_common_quadratic_coefficients(**(bounds | changes))


def _matrix(rows):
    return tuple(tuple(Q(value) for value in row) for row in rows)


def _transpose(matrix):
    return tuple(zip(*matrix))


def _scale(matrix, value):
    return tuple(tuple(value * entry for entry in row) for row in matrix)


def _sum(left, right):
    return tuple(
        tuple(a + b for a, b in zip(row_a, row_b)) for row_a, row_b in zip(left, right)
    )


def _blocks(upper_left, upper_right, lower_left, lower_right):
    return tuple(a + b for a, b in zip(upper_left, upper_right)) + tuple(
        a + b for a, b in zip(lower_left, lower_right)
    )


@pytest.mark.parametrize("gamma", (Q(1, 8), Q(1, 4), Q(1, 2)))
@pytest.mark.parametrize("stiffness", ("lower", "noncommuting", "upper"))
def test_common_metric_controls_noncommuting_full_rows_exactly(gamma, stiffness):
    # H, K and slot-dependent V need not share an eigenbasis. Exact PSD
    # checks below establish the entire quadratic inequality, not samples.
    hessian = _matrix(((3, 1), (1, 2)))
    stiffness_matrix = {
        "lower": _scale(hessian, Q(1, 2)),
        "noncommuting": _matrix(((Q(7, 4), Q(1, 2)), (Q(1, 2), Q(3, 2)))),
        "upper": hessian,
    }[stiffness]
    capacity = _matrix(((1, 0), (0, 3)))
    inverse_capacity = _matrix(((1, 0), (0, Q(1, 3))))
    identity = _matrix(((1, 0), (0, 1)))
    zero = _scale(identity, Q(0))
    assert exact_symmetric_semidefinite(_sum(hessian, _scale(identity, Q(-1))))
    assert exact_symmetric_semidefinite(
        _sum(stiffness_matrix, _scale(hessian, Q(-1, 2)))
    )
    assert exact_symmetric_semidefinite(_sum(hessian, _scale(stiffness_matrix, Q(-1))))
    multiply = exact_matrix_product
    assert multiply(capacity, hessian) != multiply(hessian, capacity)
    if stiffness == "noncommuting":
        assert multiply(hessian, stiffness_matrix) != multiply(
            stiffness_matrix, hessian
        )
        assert multiply(capacity, stiffness_matrix) != multiply(
            stiffness_matrix, capacity
        )
    vh = multiply(capacity, hessian)
    vk = multiply(capacity, stiffness_matrix)
    generator = _scale(
        _blocks(_scale(vh, Q(-1)), _scale(vk, -gamma), _scale(vh, gamma), zero),
        Q(1, 4),
    )
    metric = _blocks(
        inverse_capacity,
        _scale(inverse_capacity, gamma),
        _scale(inverse_capacity, gamma),
        inverse_capacity,
    )
    derivative = _sum(
        multiply(_transpose(generator), metric), multiply(metric, generator)
    )
    expected_derivative = _blocks(
        _scale(hessian, -(1 - gamma**2) / 2),
        _scale(stiffness_matrix, -gamma / 4),
        _scale(stiffness_matrix, -gamma / 4),
        _scale(stiffness_matrix, -(gamma**2) / 2),
    )
    assert derivative == expected_derivative
    lower, upper, decay = _common_coefficients()
    full_identity = _blocks(identity, zero, zero, identity)
    assert exact_symmetric_semidefinite(_sum(metric, _scale(full_identity, -lower)))
    assert exact_symmetric_semidefinite(
        _sum(_scale(full_identity, upper), _scale(metric, Q(-1)))
    )
    assert exact_symmetric_semidefinite(
        _scale(_sum(derivative, _scale(metric, decay)), Q(-1))
    )


@pytest.mark.parametrize("cross_sign", (-1, 0, 1))
def test_common_initial_bound_keeps_the_sign_of_weighted_cross_terms(cross_sign):
    form = (Q(2, 3), Q(-3, 5))
    phase = tuple(cross_sign * value for value in form)
    capacities = (Q(1, 4), Q(7, 3))
    square = sum((a * a + b * b) / nu for a, b, nu in zip(form, phase, capacities))
    cross = sum(a * b / nu for a, b, nu in zip(form, phase, capacities))
    lower, upper = Q(1, 8), Q(1, 2)
    bound = _sine_common_quadratic_initial_upper(
        gamma_lower=lower,
        gamma_upper=upper,
        weighted_square=square,
        weighted_cross=cross,
    )
    endpoints = (square + 2 * lower * cross, square + 2 * upper * cross)
    assert bound == max(endpoints)
    assert bound >= square + 2 * Q(1, 4) * cross
    if cross_sign < 0:
        assert bound > square + 2 * upper * cross


def test_common_metric_normalizes_all_fraction_payloads_before_products():
    gap = Q(np.int64(2**62), np.int64(3))
    capacity = Q(np.int64(2**61), np.int64(5))
    with np.errstate(over="raise"):
        lower, upper, rate = _common_coefficients(
            gap_lower=gap, capacity_lower=capacity, capacity_upper=capacity
        )
        initial = _sine_common_quadratic_initial_upper(
            gamma_lower=Q(1, 8),
            gamma_upper=Q(1, 2),
            weighted_square=Q(np.int64(2**62), np.int64(3)),
            weighted_cross=Q(np.int64(-(2**61)), np.int64(3)),
        )
    assert rate == Q(2**123, 15) * Q(1, 64) * Q(1, 2) / 6
    assert initial == Q(2**62, 3) - Q(2**59, 3)
    assert all(
        type(value.numerator) is type(value.denominator) is int
        for value in (lower, upper, rate, initial)
    )


@pytest.mark.parametrize("changes", ({"gamma_lower": Q(0)}, {"cosine_lower": Q(0)}))
def test_common_zero_lower_coercivity_gives_no_positive_decay(changes):
    lower, upper, rate = _common_coefficients(**changes)
    assert lower > 0 and upper >= lower and rate == 0


def test_common_metric_preserves_zero_and_subgrid_positive_initial_bounds():
    for square, cross in ((Q(0), Q(0)), (Q(1, 2**1200), Q(-1, 2**1201))):
        bound = _sine_common_quadratic_initial_upper(
            gamma_lower=Q(1, 8),
            gamma_upper=Q(1, 2),
            weighted_square=square,
            weighted_cross=cross,
        )
        assert bound == square + cross / 4
        assert bool(bound) == bool(square)


@pytest.mark.parametrize(
    "changes, message",
    (
        ({"gamma_lower": Q(-1)}, "gamma"),
        ({"gamma_lower": Q(3, 4)}, "gamma"),
        ({"gamma_upper": Q(1, 2) + Q(1, 2**1200)}, "gamma"),
        ({"gamma_upper": Q(0)}, "gamma"),
        ({"cosine_lower": Q(-1)}, "cosine"),
        ({"cosine_lower": Q(1) + Q(1, 2**1200)}, "cosine"),
        ({"gap_lower": Q(0)}, "gap"),
        ({"gap_lower": Q(-1)}, "gap"),
        ({"capacity_lower": Q(0)}, "capacity"),
        ({"capacity_lower": Q(4)}, "capacity"),
        ({"capacity_upper": Q(-1)}, "capacity"),
    ),
)
def test_common_coefficient_boundaries_reject_contradictory_hypotheses(
    changes, message
):
    with pytest.raises(ValueError, match=message):
        _common_coefficients(**changes)


@pytest.mark.parametrize(
    "name",
    (
        "gamma_lower",
        "gamma_upper",
        "cosine_lower",
        "gap_lower",
        "capacity_lower",
        "capacity_upper",
    ),
)
@pytest.mark.parametrize("bad", (True, 1, 0.5, float("nan"), "1/2"))
def test_common_coefficients_require_exact_rational_bounds(name, bad):
    with pytest.raises(TypeError, match="Fraction"):
        _common_coefficients(**{name: bad})


@pytest.mark.parametrize(
    "changes, error",
    (
        ({"gamma_lower": True}, TypeError),
        ({"gamma_upper": 0.5}, TypeError),
        ({"weighted_square": 1}, TypeError),
        ({"weighted_cross": float("nan")}, TypeError),
        ({"weighted_square": Q(-1)}, ValueError),
        ({"weighted_cross": Q(1)}, ValueError),
        ({"weighted_cross": Q(-1)}, ValueError),
        ({"gamma_upper": Q(3, 4)}, ValueError),
    ),
)
def test_common_initial_bound_admits_its_own_rationals_and_joint_inequality(
    changes, error
):
    values = dict(
        gamma_lower=Q(1, 8),
        gamma_upper=Q(1, 2),
        weighted_square=Q(1),
        weighted_cross=Q(0),
    )
    with pytest.raises(error):
        _sine_common_quadratic_initial_upper(**(values | changes))
