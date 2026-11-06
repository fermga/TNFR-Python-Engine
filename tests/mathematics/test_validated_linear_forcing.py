"""Independent exact solutions and domains of prescribed linear enclosures."""

from fractions import Fraction as Q

import pytest

from tnfr._exact_time import exp_unit_bounds
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I
from tnfr.mathematics._validated_linear_forcing import validated_linear_forcing_step


def _zero(time):
    return (time * 0,)


def _assert_contains(step, expected):
    for center, value in zip(step.endpoint_center, expected):
        assert I(center - step.endpoint_radius, center + step.endpoint_radius).contains(
            value
        )


def test_integrated_polynomial_at_nonzero_prescribed_time():
    calls = []

    def forcing(time):
        calls.append(time)
        return (3 * time**2,)

    step = validated_linear_forcing_step(
        (1,), 0, ((0,),), Q(1, 4), forcing, order=4, time=2
    )
    _assert_contains(step, (1 + Q(9, 4) ** 3 - 2**3,))
    assert step.endpoint_radius < Q(1, 10**35)
    assert step.remainder == (I(0),)
    assert len(calls) == 2
    assert calls[1].coeffs[0] == I(2, Q(9, 4))
    assert step.forcing_bound == 3 * Q(9, 4) ** 2
    assert step.state_bound == 1 + Q(1, 4) * step.forcing_bound


def test_coupled_metzler_system_has_independent_polynomial_solution():
    # q=(t^2+2t+3, -t^3+t+1), f=q'-Aq. A is nonsymmetric.
    def solution(time):
        return (time**2 + 2 * time + 3, -(time**3) + time + 1)

    def forcing(time):
        first, second = solution(time)
        return (
            2 * time + 2 + 2 * first - 2 * second,
            -3 * time**2 + 1 - first + second,
        )

    step = validated_linear_forcing_step(
        solution(Q(1, 2)),
        0,
        ((-2, 2), (1, -1)),
        Q(1, 16),
        forcing,
        order=12,
        time=Q(1, 2),
    )
    _assert_contains(step, solution(Q(9, 16)))
    assert step.endpoint_radius < Q(1, 10**13)


def test_first_order_step_retains_the_whole_time_derivative_remainder():
    step = validated_linear_forcing_step(
        (0,), 0, ((0,),), Q(1, 2), lambda time: (time,), order=1, time=1
    )
    # Integral from 1 to 3/2 of t dt is 5/8; the linear Taylor part is 1/2.
    assert step.endpoint_center == (Q(1, 2),)
    assert step.remainder == (I(Q(1, 8)),)
    assert step.local_error == Q(1, 8)
    _assert_contains(step, (Q(5, 8),))


def test_heat_common_direction_and_prior_radius_remain_admitted():
    step = validated_linear_forcing_step(
        (2, 2),
        Q(1, 7),
        ((-1, 1), (1, -1)),
        Q(1, 4),
        lambda time: (time * 0, time * 0),
        order=12,
    )
    _assert_contains(step, (2 + Q(1, 7),) * 2)
    _assert_contains(step, (2 - Q(1, 7),) * 2)
    assert step.endpoint_center == (2, 2)
    assert step.endpoint_radius >= Q(1, 7)
    assert step.endpoint_radius == I(Q(1, 7) + step.local_error).hi


def test_homogeneous_decay_contains_rational_exponential_enclosure():
    step = validated_linear_forcing_step((3,), 0, ((-2,),), Q(1, 4), _zero, order=16)
    lower, upper = exp_unit_bounds(Q(1, 2))
    _assert_contains(step, (I(3 / upper, 3 / lower),))
    assert step.local_error < Q(1, 10**18)


def test_multiple_steps_propagate_rounding_and_uncertain_initialization():
    center, radius = (Q(1, 3),), Q(1, 100)
    for index in range(4):
        step = validated_linear_forcing_step(
            center,
            radius,
            ((0,),),
            Q(1, 4),
            lambda time: (time,),
            order=3,
            time=Q(index, 4),
        )
        center, radius = step.endpoint_center, step.endpoint_radius
    _assert_contains(step, (Q(1, 3) + Q(1, 2) + Q(1, 100),))
    _assert_contains(step, (Q(1, 3) + Q(1, 2) - Q(1, 100),))
    assert radius - Q(1, 100) < Q(1, 10**35)


@pytest.mark.parametrize("matrix", [((1,),), ((-1, -1), (1, -1)), ((-1, 2), (1, -1))])
def test_noncontractive_generators_reject(matrix):
    with pytest.raises(ValueError, match="Metzler|row sums"):
        validated_linear_forcing_step((0,) * len(matrix), 0, matrix, 1, _zero, order=4)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["center", "generator", "radius", "duration", "time"])
def test_invalid_scalar_admission_precedes_forcing(bad, field):
    arguments = dict(center=(0,), radius=0, generator=((0,),), duration=1, time=0)
    arguments[field] = (
        (bad,) if field == "center" else ((bad,),) if field == "generator" else bad
    )

    def forbidden(_time):
        pytest.fail("invalid primitive reached the forcing callback")

    with pytest.raises((TypeError, ValueError)):
        validated_linear_forcing_step(**arguments, forcing=forbidden, order=4)


@pytest.mark.parametrize("order", [True, 1.0, 0, 17])
def test_taylor_order_admission(order):
    with pytest.raises((TypeError, ValueError), match="order"):
        validated_linear_forcing_step((0,), 0, ((0,),), 1, _zero, order=order)


@pytest.mark.parametrize("field,value", [("radius", -1), ("time", -1), ("duration", 0)])
def test_unsigned_step_primitives(field, value):
    arguments = dict(center=(0,), radius=0, generator=((0,),), duration=1, time=0)
    arguments[field] = value
    with pytest.raises(ValueError, match="nonnegative"):
        validated_linear_forcing_step(**arguments, forcing=_zero, order=4)


@pytest.mark.parametrize(
    "matrix,center", [(((0, 0),), (0,)), (((0,),), (0, 0)), ((), ())]
)
def test_dimension_admission(matrix, center):
    with pytest.raises(ValueError):
        validated_linear_forcing_step(center, 0, matrix, 1, _zero, order=4)


@pytest.mark.parametrize(
    "forcing,error",
    [
        (lambda time: (), ValueError),
        (lambda time: (I(0),), TypeError),
        (lambda time: (Jet.constant(0, 3),), ValueError),
    ],
)
def test_forcing_jet_contract(forcing, error):
    with pytest.raises(error, match="forcing"):
        validated_linear_forcing_step((0,), 0, ((0,),), 1, forcing, order=4)
