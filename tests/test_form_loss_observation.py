"""Full-response loss obstructions and the observation premises they consume."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import mpmath as mp
import pytest

from tnfr.mathematics._exact_linear_algebra import exact_matrix_product as product
from tnfr.mathematics.linear_observation import bound_form_loss_observation_error

# Conservative P3 in the exact edge energy chart (u_left,u_right,v_left,v_right).
P3 = (
    (0, 0, -Q(3, 2), -Q(1, 2)),
    (0, 0, -Q(1, 2), -Q(3, 2)),
    (Q(3, 2), Q(1, 2), 0, 0),
    (Q(1, 2), Q(3, 2), 0, 0),
)
LEFT = ((1, 0, 0, 0), (0, 0, 1, 0))
ROTATION = ((0, -1), (1, 0))


def _transpose(matrix):
    return tuple(zip(*matrix))


def _mp_matrix(matrix):
    return mp.matrix(
        [
            [mp.mpf(Q(value).numerator) / Q(value).denominator for value in row]
            for row in matrix
        ]
    )


def _mp(value):
    return mp.mpf(value.numerator) / value.denominator


def _bound(**kwargs):
    arguments = dict(damping=1, exchange=1, horizon=Q(1, 4))
    arguments.update(kwargs)
    return bound_form_loss_observation_error(P3, LEFT, **arguments)


@pytest.mark.parametrize("damping", (Q(1), Q(2), Q(4)))
@pytest.mark.parametrize("horizon", (Q(1, 100), Q(2)))
def test_bound_is_below_independent_full_response_and_commutator(damping, horizon):
    result = _bound(damping=damping, horizon=horizon)
    with mp.workdps(80):
        time = _mp(result.witness_time)
        observation = _mp_matrix(LEFT)
        actual = observation * mp.expm(time * _mp_matrix(P3)) * observation.T
        target = mp.expm(time * mp.matrix([[-_mp(damping), -1], [1, 0]]))
        rotation = _mp_matrix(ROTATION)
        # An independent closed P3 propagator retains both exchange channels.
        exact_p3 = mp.cos(time / 2) * mp.expm(3 * time * rotation / 2)
        assert mp.norm(actual - exact_p3) < mp.mpf("1e-75")
        defect = target * rotation - rotation * target
        singular_values = mp.svd(actual - target, compute_uv=False)
        commutator_values = mp.svd(defect, compute_uv=False)
        lower = _mp(result.uniform_error_lower_bound)
        assert 0 < lower <= commutator_values[0] / 2 <= singular_values[0]
        assert 0 < result.witness_time <= horizon
    # Damping 1, 2 and 4 cover underdamped, critical and overdamped targets.
    # A lower bound at a witness time is not the actual supremum error.


def test_frozen_p3_budget_is_excluded_without_fitting_an_effective_coefficient():
    result = _bound()
    assert result.target_generator == ((-1, -1), (1, 0))
    assert result.witness_time == Q(1, 4)
    assert result.uniform_error_lower_bound == Q(1, 16)
    declared_budget = Q(1, 32)
    assert result.uniform_error_lower_bound > declared_budget
    assert result.preparation == _transpose(LEFT)
    with pytest.raises(FrozenInstanceError):
        result.horizon = Q(10)


def test_zero_loss_bound_does_not_assert_a_zero_approximation_error():
    result = _bound(damping=0, exchange=Q(3, 2))
    assert result.uniform_error_lower_bound == 0
    with mp.workdps(60):
        time = _mp(result.horizon)
        observation = _mp_matrix(LEFT)
        actual = observation * mp.expm(time * _mp_matrix(P3)) * observation.T
        target = mp.expm(time * _mp_matrix(result.target_generator))
        # Reversible hidden exchange still attenuates this finite-time signal.
        assert mp.norm(actual - target) > mp.mpf("0.001")
    zero_horizon = _bound(horizon=0)
    assert zero_horizon.witness_time == zero_horizon.uniform_error_lower_bound == 0


def test_tiny_exact_inputs_and_large_horizon_keep_a_positive_rational_bound():
    tiny = Q(1, 10**400)
    result = _bound(damping=tiny, exchange=tiny, horizon=1)
    assert result.damping == result.exchange == tiny
    assert result.witness_time == 1
    assert 0 < result.uniform_error_lower_bound < tiny
    assert float(result.uniform_error_lower_bound) == 0
    long_horizon = _bound(damping=tiny, exchange=tiny, horizon=1 / tiny)
    assert long_horizon.witness_time == 1 / (4 * tiny)
    assert long_horizon.uniform_error_lower_bound == Q(1, 16)


def test_one_shot_inputs_are_detached_and_real_coefficients_keep_their_representation():
    generator = [list(row) for row in P3]
    output = [list(row) for row in LEFT]
    result = bound_form_loss_observation_error(
        (iter(row) for row in generator),
        (iter(row) for row in output),
        damping=0.1,
        exchange=0.2,
        horizon=0.3,
    )
    assert result.damping == Q(0.1) != Q(1, 10)
    assert result.exchange == Q(0.2)
    assert result.horizon == Q(0.3)
    generator[0][0] = 17
    output[0][0] = 2
    assert result.generator == P3
    assert result.output_rows == LEFT


@pytest.mark.parametrize("field", ("damping", "exchange", "horizon"))
@pytest.mark.parametrize("value", (True, float("nan"), float("inf"), "1", 1j, -1))
def test_invalid_comparison_parameters_are_rejected_without_a_bound(field, value):
    with pytest.raises((TypeError, ValueError)):
        _bound(**{field: value})


def test_zero_exchange_is_not_admitted_as_the_declared_target():
    with pytest.raises(ValueError):
        _bound(exchange=0)


@pytest.mark.parametrize("field", ("generator", "output", "damping", "horizon"))
def test_nonzero_real_underflow_is_not_replaced_by_an_admissible_zero(field):
    class UnderflowingReal(float):
        def __float__(self):
            return 0.0

    generator = [list(row) for row in P3]
    output = [list(row) for row in LEFT]
    arguments = dict(damping=1, exchange=1, horizon=1)
    if field == "generator":
        generator[0][0] = UnderflowingReal(1.0)
    elif field == "output":
        output[0][1] = UnderflowingReal(1.0)
    else:
        arguments[field] = UnderflowingReal(1.0)
    with pytest.raises(ValueError, match="underflows"):
        bound_form_loss_observation_error(generator, output, **arguments)


@pytest.mark.parametrize(
    "generator, output",
    (
        ((), LEFT),
        (((0, 1),), ((1, 0), (0, 1))),
        (((0, 0, 0),) * 3, ((1, 0, 0), (0, 1, 0))),
        (P3, (LEFT[0],)),
        (P3, ((1, 0), (0, 1))),
        (P3, ((2, 0, 0, 0), (0, 0, 2, 0))),
        (P3, (LEFT[0], LEFT[0])),
        (P3, ((True, 0, 0, 0), LEFT[1])),
        (P3, ((float("nan"), 0, 0, 0), LEFT[1])),
        (((0, -1), (2, 0)), ((1, 0), (0, 1))),
        # Real skew alone does not imply commutation with the chosen rotation.
        (((0, -1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), LEFT),
    ),
)
def test_invalid_dimensions_isometry_or_generator_symmetry_are_rejected(
    generator, output
):
    with pytest.raises((TypeError, ValueError)):
        bound_form_loss_observation_error(
            generator, output, damping=1, exchange=1, horizon=1
        )


def test_asymmetric_orthogonal_observation_breaks_rotation_without_deriving_damping():
    output = ((1, 0, 0, 0), (0, 0, Q(3, 5), Q(4, 5)))
    preparation = _transpose(output)
    assert product(output, preparation) == ((1, 0), (0, 1))
    first = product(product(output, P3), preparation)
    second = product(product(product(output, P3), P3), preparation)
    assert first == ((0, -Q(13, 10)), (Q(13, 10), 0))
    assert second == ((-Q(5, 2), 0), (0, -Q(197, 50)))
    assert product(second, ROTATION) != product(ROTATION, second)
    with pytest.raises(ValueError):
        bound_form_loss_observation_error(P3, output, damping=1, exchange=1, horizon=1)
    # Orthogonal preparation still makes the initial response skew. Removing
    # rotation covariance does not by itself produce the desired loss row.


def test_hidden_preparation_can_match_loss_derivative_but_not_the_full_target():
    # y=(u_left,v_left), with deliberately supplied (u_right,v_right)=(0,u_left).
    correlated_lift = ((1, 0), (0, 0), (0, 1), (1, 0))
    assert product(LEFT, correlated_lift) == ((1, 0), (0, 1))
    first = product(product(LEFT, P3), correlated_lift)
    assert first == ((-Q(1, 2), -Q(3, 2)), (Q(3, 2), 0))
    second = product(product(product(LEFT, P3), P3), correlated_lift)
    assert second == ((-Q(5, 2), 0), (-Q(3, 2), -Q(5, 2)))
    assert second != product(first, first)
    admitted = _bound(damping=Q(1, 2), exchange=Q(3, 2))
    assert admitted.preparation != correlated_lift
    assert product(_transpose(correlated_lift), correlated_lift) == ((2, 0), (0, 1))
    # This asymmetric supplied lift adds hidden storage depending on form.
    # It is not the orthogonal zero-hidden preparation used by the certificate.
