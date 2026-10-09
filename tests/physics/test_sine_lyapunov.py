"""Report-free exact controls for nonlinear modified-energy bounds."""

from fractions import Fraction as Q

import pytest

from tnfr.physics._sine_lyapunov import (
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
@pytest.mark.parametrize("stiffness_position", [Q(0), Q(1, 2), Q(1)])
@pytest.mark.parametrize(
    "position,velocity",
    [(Q(0), Q(1)), (Q(1), Q(0)), (Q(1), Q(2)), (Q(-2), Q(1)), (Q(3, 7), Q(-5, 11))],
)
def test_exact_damped_gradient_rows_prove_decay_and_energy_sandwich(
    damping, stiffness_position, position, velocity
):
    mu, maximum, lower, upper, decay = _coefficients()
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
