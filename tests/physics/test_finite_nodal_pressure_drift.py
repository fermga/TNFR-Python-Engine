"""Carry-aware finite-class escape and false inference controls."""

from fractions import Fraction
import math

import pytest

from tnfr.dynamics._euler_kernel import NodalRemainderState, advance_nodal_remainder
from tnfr.physics.nodal_remainder_pressure import observe_finite_nodal_pressure_drift


def observe(**changes):
    arguments = dict(epi_states=((.5,),), pressure_vectors=((2.0**-56,),),
                     functional=(Fraction(1),), timestep=1.0)
    arguments.update(changes)
    return observe_finite_nodal_pressure_drift(**arguments)


@pytest.mark.parametrize("sign", [1, -1])
def test_class_bound_covers_all_carry_not_only_zero(sign):
    result = observe(pressure_vectors=((sign * 2.0**-56,),), functional=(Fraction(sign),))
    assert result.width == Fraction(3, 2**55)
    assert result.max_confined_steps == 6
    assert result.escape_step_bound == 7
    assert result.conditional_class_escape_certified
    assert not result.pressure_provenance_certified
    assert not result.positive_band_exit_certified
    # Start at the far source-cell midpoint tie: both endpoints round to
    # this even significand. The bound is attained by the actual kernel.
    remainder = Fraction(-1, 2**55) if sign > 0 else Fraction(1, 2**54)
    state = NodalRemainderState((.5,), (remainder,), .05, 1.0)
    for _ in range(6):
        state = advance_nodal_remainder(
            state, timestep=1.0, capacity=(1.0,), pressure=(sign * 2.0**-56,),
        ).after
        assert state.epi == (.5,)
    state = advance_nodal_remainder(
        state, timestep=1.0, capacity=(1.0,), pressure=(sign * 2.0**-56,),
    ).after
    assert state.epi != (.5,)
    assert .05 < state.exact_epi[0] < 1


def test_zero_mean_pressure_can_have_strict_structural_drift():
    result = observe(epi_states=((.5, .5),), pressure_vectors=((.01, -.01),),
                     functional=(Fraction(1), Fraction(-1)))
    assert sum(result.pressure_vectors[0]) == 0
    assert result.minimum_pressure_projection == 2 * Fraction(.01)
    assert result.conditional_class_escape_certified


def test_nonzero_pressures_that_cancel_convexly_are_inconclusive():
    result = observe(epi_states=((.5,), (.6,)), pressure_vectors=((.1,), (-.1,)))
    assert result.minimum_pressure_projection < 0
    assert result.max_confined_steps is None and result.escape_step_bound is None
    assert not result.conditional_class_escape_certified


def test_zero_gap_is_not_strict_separation():
    result = observe(epi_states=((.5,), (.6,)), pressure_vectors=((.1,), (0.0,)))
    assert result.minimum_pressure_projection == 0
    assert not result.conditional_class_escape_certified


def test_cell_envelopes_respect_global_band_and_functional_signs():
    result = observe(epi_states=((.05, 1.0),), pressure_vectors=((1.0, -1.0),),
                     functional=(Fraction(2), Fraction(-3)))
    assert result.class_lower == 2 * Fraction(.05) - 3
    assert result.width > 0
    assert result.max_confined_steps == 0
    assert result.escape_step_bound == 1


def test_odd_ties_use_conservative_outer_enclosure():
    x = math.nextafter(.5, 1.0)
    result = observe(epi_states=((x,),))
    assert result.class_lower == Fraction(x) - Fraction(1, 2**54)
    assert result.class_upper == Fraction(x) + Fraction(1, 2**54)
    assert result.max_confined_steps == 8


def test_tiny_pressure_has_exact_large_escape_bound():
    result = observe(pressure_vectors=((math.ulp(0.0),),), timestep=math.ulp(0.0))
    assert result.max_confined_steps == 3 * 2**2093
    assert result.escape_step_bound == result.max_confined_steps + 1


@pytest.mark.parametrize("changes,error", [
    ({"epi_states": []}, TypeError), ({"epi_states": ()}, ValueError),
    ({"epi_states": ((.5,), (.5,)), "pressure_vectors": ((.1,), (.2,))}, ValueError),
    ({"epi_states": ((.5, .6),)}, ValueError),
    ({"epi_states": ((math.nan,),)}, ValueError),
    ({"epi_states": ((0.0,),)}, ValueError),
    ({"pressure_vectors": []}, TypeError), ({"pressure_vectors": ()}, ValueError),
    ({"pressure_vectors": ((1,),)}, TypeError),
    ({"pressure_vectors": ((math.inf,),)}, ValueError),
    ({"functional": (1,)}, TypeError), ({"functional": ()}, TypeError),
    ({"functional": (Fraction(0),)}, ValueError),
    ({"timestep": 0.0}, ValueError), ({"timestep": -1.0}, ValueError),
    ({"timestep": True}, TypeError), ({"timestep": math.inf}, ValueError),
    ({"epi_lower": .7}, ValueError), ({"epi_upper": .4}, ValueError),
    ({"epi_lower": 0.0}, ValueError), ({"epi_upper": 2.0}, ValueError),
])
def test_invalid_primitive_classes_rejected(changes, error):
    with pytest.raises(error):
        observe(**changes)
