"""Initial-response loss obstruction without channel-rotation covariance."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import mpmath as mp
import pytest

from tnfr.mathematics._exact_linear_algebra import exact_matrix_product as product
from tnfr.mathematics.linear_observation import (
    bound_form_loss_observation_error,
    bound_orthogonal_form_loss_observation_error,
)

# Algebraic energy-chart control: D=((2,1),(0,1)), J=((0,-D),(D^T,0)).
# This fixture declares a matrix, not a TNFR graph or an ideal phase Hessian.
TRIANGULAR = (
    (0, 0, -2, -1),
    (0, 0, 0, -1),
    (2, 0, 0, 0),
    (1, 1, 0, 0),
)
LEFT = ((1, 0, 0, 0), (0, 0, 1, 0))
ASYMMETRIC = ((Q(3, 5), Q(4, 5), 0, 0), (0, 0, 1, 0))


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _matrix(rows):
    return mp.matrix([[_mp(value) for value in row] for row in rows])


def _bound(**overrides):
    parameters = dict(damping=1, exchange=2, horizon=1)
    parameters.update(overrides)
    return bound_orthogonal_form_loss_observation_error(TRIANGULAR, LEFT, **parameters)


@pytest.mark.parametrize("output", (LEFT, ASYMMETRIC))
@pytest.mark.parametrize("horizon", (Q(1, 100), Q(1)))
def test_non_covariant_bound_is_below_independent_full_response(output, horizon):
    result = bound_orthogonal_form_loss_observation_error(
        TRIANGULAR, output, damping=1, exchange=2, horizon=horizon
    )
    with pytest.raises(ValueError, match="commute"):
        bound_form_loss_observation_error(
            TRIANGULAR, output, damping=1, exchange=2, horizon=horizon
        )
    with mp.workdps(70):
        time = _mp(result.witness_time)
        observation = _matrix(output)
        actual = observation * mp.expm(time * _matrix(TRIANGULAR)) * observation.T
        target = mp.expm(time * mp.matrix([[-1, -2], [2, 0]]))
        error = mp.svd(actual - target, compute_uv=False)[0]
        assert 0 < _mp(result.uniform_error_lower_bound) <= error
        assert mp.svd(_matrix(TRIANGULAR), compute_uv=False)[0] <= 3
    assert 0 < result.witness_time <= horizon


def test_geometry_splits_second_response_but_keeps_skew_initial_response():
    preparation = tuple(zip(*LEFT))
    initial = product(product(LEFT, TRIANGULAR), preparation)
    second = product(product(product(LEFT, TRIANGULAR), TRIANGULAR), preparation)
    assert initial == ((0, -2), (2, 0))
    assert second == ((-5, 0), (0, -4))
    result = _bound()
    assert result.target_generator != initial
    assert result.method == "orthogonal_initial_response"
    assert result.generator_norm_upper_bound == 3
    # The independently calculated row-sum bound gives N=3, M=3, Q=18.
    assert result.witness_time == Q(1, 18)
    assert result.uniform_error_lower_bound == Q(1, 36)
    assert result.generator == TRIANGULAR
    assert result.output_rows == LEFT
    assert result.preparation == preparation
    assert result.target_generator == ((-1, -2), (2, 0))
    assert (result.damping, result.exchange, result.horizon) == (1, 2, 1)
    with pytest.raises(FrozenInstanceError):
        result.generator_norm_upper_bound = Q(0)


def test_zero_loss_or_horizon_bound_does_not_certify_agreement():
    zero_loss = _bound(damping=0)
    assert zero_loss.witness_time == zero_loss.uniform_error_lower_bound == 0
    with mp.workdps(60):
        observation = _matrix(LEFT)
        actual = observation * mp.expm(_matrix(TRIANGULAR)) * observation.T
        target = mp.expm(mp.matrix([[0, -2], [2, 0]]))
        assert mp.norm(actual - target) > mp.mpf("0.1")
    zero_horizon = _bound(horizon=0)
    assert zero_horizon.witness_time == zero_horizon.uniform_error_lower_bound == 0


def test_new_reader_retains_shared_admission_and_exact_small_positive_bound():
    # Shared admission has exhaustive coverage in test_form_loss_observation.
    # These witnesses check that the less restrictive API still invokes it.
    with pytest.raises(ValueError, match="skew"):
        bound_orthogonal_form_loss_observation_error(
            ((0, -1), (2, 0)),
            ((1, 0), (0, 1)),
            damping=1,
            exchange=2,
            horizon=1,
        )
    with pytest.raises(ValueError, match="orthonormal"):
        bound_orthogonal_form_loss_observation_error(
            TRIANGULAR, (LEFT[0], LEFT[0]), damping=1, exchange=2, horizon=1
        )
    with pytest.raises((TypeError, ValueError)):
        _bound(damping=True)
    tiny = Q(1, 10**400)
    result = _bound(damping=tiny, exchange=tiny)
    assert result.damping == result.exchange == tiny
    assert 0 < result.witness_time < tiny
    assert 0 < result.uniform_error_lower_bound < tiny**2
    assert float(result.uniform_error_lower_bound) == 0
