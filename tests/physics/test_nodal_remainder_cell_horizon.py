"""Exact cell boundaries versus real carried-kernel continuation."""

from dataclasses import replace
from fractions import Fraction
import math

import pytest

from tnfr.dynamics._euler_kernel import (
    NodalRemainderState, advance_nodal_remainder, initialize_nodal_remainder,
)
from tnfr.physics.nodal_remainder import derive_nodal_remainder_cell_horizon


def horizon(state, pressure, **kwargs):
    return derive_nodal_remainder_cell_horizon(
        state=state, pressure=pressure,
        timestep=kwargs.pop("timestep", 1.0),
        capacity=kwargs.pop("capacity", (1.0,) * len(state.epi)), **kwargs,
    )


@pytest.mark.parametrize("x,p,steps", [
    (.5, 2.0**-54, 1), (.5, -2.0**-55, 1),
    (.5, 2.0**-56, 4), (.5, -2.0**-57, 4),
    (math.nextafter(.5, 1.0), 2.0**-54, 0),
    (math.nextafter(.5, 1.0), -2.0**-54, 0),
])
def test_even_and_odd_source_ties_match_actual_kernel(x, p, steps):
    state = initialize_nodal_remainder((x,))
    bound = horizon(state, (p,))
    assert bound.max_unchanged_steps == steps
    assert bound.first_exit_step == steps + 1
    current = state
    for _ in range(steps):
        current = advance_nodal_remainder(current, timestep=1.0, capacity=(1.0,), pressure=(p,)).after
        assert current.epi == state.epi
    assert current.exact_epi == bound.unchanged_endpoint
    current = advance_nodal_remainder(current, timestep=1.0, capacity=(1.0,), pressure=(p,)).after
    assert current.epi != state.epi
    assert current.exact_epi == bound.first_exit_exact
    assert bound.first_exit_leaves_cell == (True,)
    assert bound.first_exit_leaves_band == (False,)


def test_retained_remainder_shortens_source_prefix_without_reset():
    state = NodalRemainderState((.5,), (Fraction(1, 2**55),), .05, 1.0)
    bound = horizon(state, (2.0**-56,))
    assert bound.max_unchanged_steps == 2
    assert bound.unchanged_endpoint == (Fraction(1, 2) + Fraction(1, 2**54),)
    assert horizon(initialize_nodal_remainder((.5,)), (2.0**-56,)).max_unchanged_steps == 4


@pytest.mark.parametrize("value,p", [(1.0, 2.0**-56), (.05, -2.0**-64)])
def test_band_rejects_before_display_changes(value, p):
    state = initialize_nodal_remainder((value,))
    bound = horizon(state, (p,))
    assert bound.max_unchanged_steps == 0
    assert bound.first_exit_leaves_cell == (False,)
    assert bound.first_exit_leaves_band == (True,)
    with pytest.raises(ValueError, match="band"):
        advance_nodal_remainder(state, timestep=1.0, capacity=(1.0,), pressure=(p,))


def test_joint_prefix_and_mean_use_capacity_not_pressure_alone():
    state = initialize_nodal_remainder((.5, .5, .5))
    bound = horizon(state, (2.0**-56, -2.0**-56, 1.0), capacity=(1.0, 2.0, 0.0))
    assert bound.coordinate_step_limits == (4, 1, None)
    assert bound.max_unchanged_steps == 1
    assert bound.first_exit_leaves_cell == (False, True, False)
    assert bound.mean_increment == -Fraction(1, 3 * 2**56)


@pytest.mark.parametrize("h,nu,p", [(0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)])
def test_zero_nodal_increment_has_unbounded_constant_prefix(h, nu, p):
    bound = horizon(initialize_nodal_remainder((.5,)), (p,), timestep=h, capacity=(nu,))
    assert bound.coordinate_step_limits == (None,)
    assert bound.max_unchanged_steps is None
    assert bound.first_exit_step is None
    assert bound.unchanged_endpoint is None and bound.first_exit_exact is None
    assert bound.mean_increment == 0


def test_subnormal_source_uses_analytic_integer_horizon():
    # Iterating this prefix is infeasible. Exact area distinguishes it from
    # ordinary multiplication underflow, which would report a zero update.
    tiny = math.ulp(0.0)
    bound = horizon(initialize_nodal_remainder((.5,)), (tiny,), timestep=tiny, capacity=(tiny,))
    assert bound.exact_increment == (Fraction(1, 2**3222),)
    assert bound.max_unchanged_steps == 2**3168
    assert bound.first_exit_step == 2**3168 + 1
    assert bound.first_exit_leaves_cell == (True,)
    assert bound.first_exit_leaves_band == (False,)


@pytest.mark.parametrize("kwargs,error", [
    ({"timestep": -1.0}, ValueError), ({"timestep": math.nan}, ValueError),
    ({"timestep": True}, TypeError), ({"capacity": (-1.0,)}, ValueError),
    ({"capacity": (math.inf,)}, ValueError), ({"capacity": [1.0]}, TypeError),
    ({"capacity": (1.0, 1.0)}, ValueError),
])
def test_invalid_nodal_inputs_rejected(kwargs, error):
    with pytest.raises(error):
        horizon(initialize_nodal_remainder((.5,)), (1.0,), **kwargs)


@pytest.mark.parametrize("pressure,error", [
    ((math.nan,), ValueError), ((1,), TypeError), ([1.0], TypeError),
    ((), ValueError), ((1.0, 1.0), ValueError),
])
def test_invalid_pressure_inputs_rejected(pressure, error):
    with pytest.raises(error):
        horizon(initialize_nodal_remainder((.5,)), pressure)


def test_public_state_cannot_override_encoding_validation():
    class ForgedState(NodalRemainderState):
        @property
        def exact_epi(self):
            return (Fraction(1, 2),)

    state = ForgedState((.5,), (Fraction(1, 3),), .05, 1.0)
    with pytest.raises(ValueError, match="dyadic"):
        horizon(state, (0.0,))
    with pytest.raises(ValueError, match="nearest-even"):
        horizon(replace(initialize_nodal_remainder((.5,)), remainder=(Fraction(1, 4),)), (0.0,))
