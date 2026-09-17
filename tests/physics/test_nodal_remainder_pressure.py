"""Independent signed EPI-readout balances for carried scalar coordinates."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F
import math

import pytest

from tnfr.dynamics._euler_kernel import NodalRemainderState, advance_nodal_remainder, initialize_nodal_remainder
from tnfr.physics.nodal_remainder_pressure import observe_nodal_remainder_pressure_readout


P2 = ((F(0), F(1)), (F(1), F(0)))
P3 = ((F(0), F(1), F(0)), (F(1), F(0), F(1)), (F(0), F(1), F(0)))


def _state(epi=(.75, .75), pressure=(2.**-56, 0.)):
    initial = initialize_nodal_remainder(epi)
    return advance_nodal_remainder(
        initial, timestep=1., capacity=(1.,) * len(epi), pressure=pressure,
    ).after


def _observe(**changes):
    arguments = {
        "state": _state(), "conductance": P2, "capacity": (1., 1.),
        "stored_pressure": (0., 0.), "epi_weight": .5,
    }
    arguments.update(changes)
    return observe_nodal_remainder_pressure_readout(**arguments)


def test_nonzero_carry_changes_exact_pressure_without_changing_visible_epi():
    result = _observe()
    a = F(1, 2**56)
    assert result.state.epi == (.75, .75)
    assert result.state.remainder == (a, 0)
    assert result.epi_pressure_visible == (0, 0)
    assert result.epi_pressure_reconstructed == (-a / 2, a / 2)
    assert result.readout_shift == (a / 2, -a / 2)
    assert result.stored_minus_visible_reference == (0, 0)
    assert result.stored_minus_reconstructed_reference == result.readout_shift
    assert result.pressure_identity_residual == (0, 0)


def test_regular_c6_common_capacity_cancels_both_arithmetic_readout_means():
    a = F(1, 2**56)
    pattern = (1, 0, -1, 2, 0, -2)
    matrix = tuple(tuple(F(int((j - i) % 6 in (1, 5))) for j in range(6)) for i in range(6))
    state = _state((.75,) * 6, tuple(float(a * value) for value in pattern))
    result = _observe(state=state, conductance=matrix, capacity=(2.,) * 6,
                      stored_pressure=(.125, -.125, .25, 0., -.25, .5))
    expected = tuple(a * (F(value) - F(pattern[i - 1] + pattern[(i + 1) % 6], 2)) / 2
                     for i, value in enumerate(pattern))
    assert result.readout_shift == expected
    assert any(result.readout_shift)
    assert result.mean_pressure_readout_shift == result.mean_nodal_readout_shift == 0
    assert result.strengths == (2,) * 6
    assert result.reversible_weights == (1,) * 6
    assert result.reversible_mean_nodal_readout_shift == 0
    assert result.degree_weighted_pressure_readout_shift == 0


def test_irregular_p3_has_nonzero_arithmetic_but_zero_degree_weighted_shift():
    a = F(1, 2**56)
    result = _observe(state=_state((.75,) * 3, (float(a), 0., 0.)), conductance=P3,
                      capacity=(1.,) * 3, stored_pressure=(0.,) * 3)
    assert result.readout_shift == (a / 2, -a / 4, 0)
    assert result.mean_pressure_readout_shift == result.mean_nodal_readout_shift == a / 12
    assert result.reversible_weights == (1, 2, 1)
    assert result.degree_weighted_pressure_readout_shift == result.reversible_mean_nodal_readout_shift == 0


def test_heterogeneous_capacity_changes_reversible_weights_and_arithmetic_mean():
    a = F(1, 2**56)
    result = _observe(capacity=(2., 1.))
    assert result.mean_pressure_readout_shift == 0
    assert result.nodal_readout_shift == (a, -a / 2)
    assert result.mean_nodal_readout_shift == a / 4
    assert result.reversible_weights == (F(1, 2), F(1))
    assert result.reversible_mean_nodal_readout_shift == 0
    # Zero supplied mean pressure also does not balance heterogeneous nodal area.
    supplied = _observe(capacity=(2., 1.), stored_pressure=(.125, -.125))
    assert sum(supplied.stored_pressure) == 0
    assert sum(nu * p for nu, p in zip(supplied.capacity, supplied.stored_pressure)) / 2 == F(1, 16)


def test_nonuniform_visible_field_keeps_other_pressure_channels_in_residual():
    a = F(1, 2**58)
    result = _observe(state=_state((.25, .5, .75), (float(a), 0., 0.)),
                      conductance=P3, capacity=(2., 1., 3.),
                      stored_pressure=(.25, .125, -.25))
    assert result.epi_pressure_visible == (F(1, 8), 0, -F(1, 8))
    assert result.epi_pressure_reconstructed == (F(1, 8) - a / 2, a / 4, -F(1, 8))
    assert result.stored_minus_visible_reference == (F(1, 8), F(1, 8), -F(1, 8))
    assert result.stored_minus_reconstructed_reference == (F(1, 8) + a / 2, F(1, 8) - a / 4, -F(1, 8))
    assert result.pressure_identity_residual == (0, 0, 0)
    assert result.mean_nodal_readout_shift == a / 4
    assert result.reversible_weights == (F(1, 2), F(2), F(1, 3))
    assert result.reversible_mean_nodal_readout_shift == 0


def test_exact_nonbinary_weight_is_not_rounded_to_binary64():
    result = _observe(epi_weight=F(1, 3))
    a = F(1, 2**56)
    assert result.epi_weight == F(1, 3)
    assert result.readout_shift == (a / 3, -a / 3)


@pytest.mark.parametrize("capacity", ((0., 1.), (1., 0.), (0., 0.)))
def test_zero_capacity_leaves_reversible_mean_undefined(capacity):
    result = _observe(capacity=capacity)
    assert result.reversible_weights is None
    assert result.reversible_mean_nodal_readout_shift is None
    assert result.degree_weighted_pressure_readout_shift == 0
    for nu, shift in zip(capacity, result.nodal_readout_shift):
        if nu == 0:
            assert shift == 0


def test_zero_epi_weight_has_zero_shift_without_discarding_stored_pressure():
    result = _observe(epi_weight=0., stored_pressure=(.125, -.5))
    assert result.readout_shift == result.nodal_readout_shift == (0, 0)
    assert result.epi_pressure_visible == result.epi_pressure_reconstructed == (0, 0)
    assert result.stored_minus_visible_reference == result.stored_minus_reconstructed_reference == (F(1, 8), -F(1, 2))


def test_disconnected_positive_rows_and_self_loops_use_shared_adjacency_convention():
    matrix = ((F(3), F(1)), (F(1), F(3)))
    a = F(1, 2**56)
    result = _observe(conductance=matrix)
    assert result.strengths == (4, 4)
    assert result.readout_shift == (a / 8, -a / 8)
    diagonal = ((F(2), F(0)), (F(0), F(7)))
    result = _observe(conductance=diagonal)
    assert result.readout_shift == (0, 0)
    assert result.reversible_mean_nodal_readout_shift == 0


@pytest.mark.parametrize("changes,error", (
    ({"state": object()}, TypeError),
    ({"conductance": []}, TypeError),
    ({"conductance": ()}, ValueError),
    ({"conductance": ((F(1),),)}, ValueError),
    ({"conductance": ((F(0),), (F(1),))}, ValueError),
    ({"conductance": ([F(0), F(1)], [F(1), F(0)])}, TypeError),
    ({"conductance": ((0., 1.), (1., 0.))}, TypeError),
    ({"conductance": ((F(0), F(1)), (F(2), F(0)))}, ValueError),
    ({"conductance": ((F(0), -F(1)), (-F(1), F(0)))}, ValueError),
    ({"conductance": ((F(0), F(0)), (F(0), F(1)))}, ValueError),
    ({"capacity": [1., 1.]}, TypeError),
    ({"capacity": (1.,)}, ValueError),
    ({"capacity": (True, 1.)}, TypeError),
    ({"capacity": (F(1), 1.)}, TypeError),
    ({"capacity": (-1., 1.)}, ValueError),
    ({"capacity": (math.inf, 1.)}, ValueError),
    ({"stored_pressure": ()}, ValueError),
    ({"stored_pressure": (0.,)}, ValueError),
    ({"stored_pressure": (1, 0.)}, TypeError),
    ({"stored_pressure": (math.nan, 0.)}, ValueError),
    ({"epi_weight": True}, TypeError),
    ({"epi_weight": 1}, TypeError),
    ({"epi_weight": -.5}, ValueError),
    ({"epi_weight": -F(1, 3)}, ValueError),
    ({"epi_weight": math.inf}, ValueError),
))
def test_invalid_primitive_inputs_are_rejected(changes, error):
    with pytest.raises(error):
        _observe(**changes)


@pytest.mark.parametrize("remainder", ((F(1, 3), F(0)), (F(1, 2**3223), F(0)),
                                       (F(1, 2**52), F(0))))
def test_forged_public_encoding_is_revalidated(remainder):
    forged = replace(_state(), remainder=remainder)
    with pytest.raises(ValueError):
        _observe(state=forged)


def test_observer_is_read_only_and_result_is_frozen():
    state = _state()
    before = (state.epi, state.remainder, state.exact_epi)
    result = _observe(state=state)
    assert (state.epi, state.remainder, state.exact_epi) == before
    assert result.state is state
    with pytest.raises(FrozenInstanceError):
        result.mean_nodal_readout_shift = F(1)


def test_overridden_reconstruction_property_cannot_replace_shared_validation():
    class WrongReconstruction(NodalRemainderState):
        @property
        def exact_epi(self):
            return (F(0), F(0))

    original = _state()
    state = WrongReconstruction(original.epi, original.remainder,
                                original.epi_lower, original.epi_upper)
    assert _observe(state=state).epi_pressure_reconstructed == _observe().epi_pressure_reconstructed
    forged = replace(state, remainder=(F(1, 3), F(0)))
    with pytest.raises(ValueError):
        _observe(state=forged)
