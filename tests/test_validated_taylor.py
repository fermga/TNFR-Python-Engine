"""Independent exact-solution controls for the shared enclosure step."""

from fractions import Fraction as Q
from math import factorial

import pytest

from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I
from tnfr.mathematics._validated_taylor import (
    comparison_matrix,
    flow_jets,
    validated_taylor_step,
)


def _everywhere(_):
    return (Q(1),)


def _zero(state):
    return tuple(0 * value for value in state)


def _exponential_reference(value):
    """Independent Taylor/Lagrange bound, with exp(abs(value)) < 2."""
    assert abs(value) <= Q(1, 2)
    partial = sum((value**index / factorial(index) for index in range(81)), Q(0))
    remainder = 2 * abs(value) ** 81 / factorial(81)
    return partial - remainder, partial + remainder


def test_linear_decay_encloses_exact_solution_and_retains_contracting_uncertainty():
    initial = I(Q(15, 16), Q(17, 16))
    duration = Q(1, 8)
    step, failed, reason = validated_taylor_step(
        (initial,),
        duration,
        lambda state: (-state[0],),
        _everywhere,
        order=8,
        time=Q(3, 2),
    )
    assert step is not None and failed is reason is None
    lower, upper = _exponential_reference(-duration)
    assert step.endpoint[0].lo <= initial.lo * lower
    assert initial.hi * upper <= step.endpoint[0].hi
    radius = step.propagated_initial_radii[0]
    assert initial.radius * upper <= radius < initial.radius
    assert step.time == Q(3, 2) and step.duration == duration
    assert step.picard_interior_margin > 0
    assert initial.subset_of(step.tube[0])
    assert step.endpoint[0].subset_of(step.tube[0])


def test_eight_dimensional_chain_retains_every_initial_interval_coordinate():
    # x_i'=x_(i+1), x_7'=0 has an exact finite polynomial solution. This
    # exercises cross-coordinate uncertainty without any research fixture.
    initial = tuple(I(index + 1 - Q(1, 32), index + 1 + Q(1, 32)) for index in range(8))
    duration = Q(1, 16)

    def flow(state):
        return state[1:] + (0 * state[-1],)

    step, failed, reason = validated_taylor_step(
        initial, duration, flow, _everywhere, order=8
    )
    assert step is not None and failed is reason is None
    for index, actual in enumerate(step.endpoint):
        coefficients = tuple(
            duration**power / factorial(power) for power in range(8 - index)
        )
        lower = sum(
            (factor * value.lo for factor, value in zip(coefficients, initial[index:])),
            Q(0),
        )
        upper = sum(
            (factor * value.hi for factor, value in zip(coefficients, initial[index:])),
            Q(0),
        )
        assert actual.lo <= lower <= upper <= actual.hi
        expected_radius = sum(
            (
                factor * value.radius
                for factor, value in zip(coefficients, initial[index:])
            ),
            Q(0),
        )
        assert step.propagated_initial_radii[index] >= expected_radius > 0
        assert step.local_remainder_bounds[index] == I(0)
        assert actual.subset_of(step.tube[index])
    assert len(step.endpoint) == len(step.propagated_initial_radii) == 8


def test_twenty_three_coordinates_retain_independent_exact_linear_solutions():
    rates = tuple(Q(index + 1, 24) for index in range(23))
    initial = tuple(
        I(Q(index + 1, 8) - Q(1, 64), Q(index + 1, 8) + Q(1, 64)) for index in range(23)
    )
    duration = Q(1, 8)

    def flow(state):
        return tuple(-rate * value for rate, value in zip(rates, state))

    step, failed, reason = validated_taylor_step(
        initial, duration, flow, _everywhere, order=8
    )
    assert step is not None and failed is reason is None
    assert len(step.endpoint) == 23
    assert step.picard_interior_margin > 0
    for source, rate, endpoint, radius in zip(
        initial, rates, step.endpoint, step.propagated_initial_radii
    ):
        lower, upper = _exponential_reference(-rate * duration)
        assert endpoint.lo <= source.lo * lower
        assert source.hi * upper <= endpoint.hi
        assert source.radius * upper <= radius < source.radius
    assert all(value.subset_of(tube) for value, tube in zip(step.endpoint, step.tube))


@pytest.mark.parametrize("initial", (I(1), I(1, Q(17, 16))))
def test_nonlinear_square_flow_encloses_rational_solution_on_the_whole_initial_box(
    initial,
):
    duration = Q(1, 64)
    step, failed, reason = validated_taylor_step(
        (initial,),
        duration,
        lambda state: (state[0] ** 2,),
        lambda tube: (tube[0].lo - Q(1, 2), 2 - tube[0].hi),
        order=8,
    )
    assert step is not None and failed is reason is None
    # x(t)=x0/(1-t*x0), increasing in x0 throughout this admitted box.
    for start in (initial.lo, initial.midpoint, initial.hi):
        exact = start / (1 - duration * start)
        assert step.endpoint[0].contains(exact)
    assert step.local_remainder_bounds[0].lo > 0
    assert step.picard_interior_margin > 0 and min(step.domain_lower_bounds) > 0
    if initial.radius:
        assert step.propagated_initial_radii[0] > initial.radius
    else:
        assert step.propagated_initial_radii == (Q(0),)


def test_inflation_without_strict_picard_inclusion_cannot_certify_a_step():
    # For h=1 and x'=x+1, the image upper endpoint is tube.hi+1. No
    # amount of this whole-box inflation can establish strict inclusion.
    step, failed, reason = validated_taylor_step(
        (I(0),),
        Q(1),
        lambda state: (state[0] + 1,),
        _everywhere,
        order=4,
    )
    assert step is None and failed is not None
    assert reason == "strict_Picard_inclusion_not_resolved"


def test_whole_tube_domain_failure_is_preserved_without_substituting_a_shorter_step():
    step, failed, reason = validated_taylor_step(
        (I(0),),
        Q(1),
        lambda state: (state[0] * 0 + 1,),
        lambda tube: (1 - tube[0].hi,),
        order=4,
        domain_failure="strict_upper_boundary_reached",
    )
    assert step is None and failed is not None
    assert failed[0].hi >= 1
    assert reason == "strict_upper_boundary_reached"


@pytest.mark.parametrize("margin", (float("nan"), float("inf"), True, 1.0))
def test_invalid_domain_evidence_cannot_produce_a_certificate(margin):
    with pytest.raises(TypeError):
        validated_taylor_step((I(1),), Q(1, 8), _zero, lambda tube: (margin,), order=2)


@pytest.mark.parametrize("state", ((), {I(1)}, {0: I(1)}, (I(1),) * 25))
def test_state_order_and_dimension_are_admitted_before_calculation(state):
    with pytest.raises((TypeError, ValueError)):
        validated_taylor_step(state, Q(1, 8), _zero, _everywhere, order=2)


@pytest.mark.parametrize("duration", (True, 0.125, Q(0), Q(-1)))
def test_duration_requires_a_strictly_positive_exact_rational(duration):
    with pytest.raises((TypeError, ValueError)):
        validated_taylor_step((I(1),), duration, _zero, _everywhere, order=2)


def test_missing_field_coordinate_retains_explicit_unavailable_result():
    step, failed, reason = validated_taylor_step(
        (I(1), I(2)),
        Q(1, 8),
        lambda state: (state[0],),
        _everywhere,
        order=2,
    )
    assert step is None and failed == (I(1), I(2))
    assert "field dimension differs from state" in reason


@pytest.mark.parametrize("operation", ("flow_jets", "comparison_matrix"))
@pytest.mark.parametrize("invalid", ("mapping", "scalar_row", "wrong_order"))
def test_derivative_callback_cannot_replace_ordered_same_order_jets(operation, invalid):
    def bad_field(state):
        if invalid == "mapping":
            return {0: state[0]}
        if invalid == "scalar_row":
            return (I(0),)
        return (Jet.constant(0, state[0].order + 1),)

    error = ValueError if invalid == "wrong_order" else TypeError
    with pytest.raises(error):
        if operation == "flow_jets":
            flow_jets((I(1),), 2, bad_field)
        else:
            comparison_matrix((I(1),), bad_field)
