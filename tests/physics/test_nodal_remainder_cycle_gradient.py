"""Full nodal budgets must survive before taking a cycle Laplacian."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F

import pytest

from tnfr.dynamics._euler_kernel import (
    advance_nodal_remainder,
    initialize_nodal_remainder,
)
from tnfr.physics.nodal_remainder_pressure import observe_nodal_remainder_cycle_gradient

QUANTUM = F(1, 2**54)


def _transition(size=6, *, uniform=False, visible=False):
    initial = initialize_nodal_remainder(
        (0.5,) * size, epi_lower=0.375, epi_upper=0.625
    )
    pressure = (
        (2.0**-55,) * size
        if uniform
        else (2.0 ** (-53 if visible else -57), -(2.0 ** (-54 if visible else -57)))
        + (0.0,) * (size - 2)
    )
    step = advance_nodal_remainder(
        initial, timestep=1.0, capacity=(1.0,) * size, pressure=pressure
    )
    return initial, step


def _observe(initial, step, **overrides):
    arguments = dict(
        initial=initial,
        endpoint=step.after,
        nodal_area=step.exact_increment,
        epi_quantum=QUANTUM,
    )
    arguments.update(overrides)
    return observe_nodal_remainder_cycle_gradient(**arguments)


@pytest.mark.parametrize("size", (3, 6, 7))
def test_invisible_nodal_area_is_exactly_cancelled_by_carry_in_the_gradient(size):
    initial, step = _transition(size)
    result = _observe(initial, step)
    assert initial.epi == step.after.epi
    assert (
        result.gradient_indices_before == result.gradient_indices_after == (0,) * size
    )
    assert result.nodal_area == result.remainder_change == step.exact_increment
    assert any(result.nodal_gradient_term)
    assert result.nodal_gradient_term == tuple(
        -value for value in result.remainder_gradient_term
    )
    assert result.identity_residual == (0,) * size
    assert (
        not result.pressure_provenance_certified
        and not result.runtime_provenance_certified
    )


@pytest.mark.parametrize("size", (3, 6, 7))
def test_visible_index_change_uses_both_neighbors_with_the_correct_sign(size):
    initial, step = _transition(size, visible=True)
    result = _observe(initial, step)
    offsets = tuple((F(value) - F(0.5)) / QUANTUM for value in step.after.epi)
    expected = tuple(
        offsets[i - 1] + offsets[(i + 1) % size] - 2 * offsets[i] for i in range(size)
    )
    assert result.gradient_indices_after == result.gradient_index_change == expected
    assert sum(expected) == 0 and any(expected)
    assert result.nodal_area == step.exact_increment
    assert result.identity_residual == (0,) * size


@pytest.mark.parametrize("shift", (F(1), F(-1, 2**100), F(1, 2**3222)))
def test_uniform_corruption_is_rejected_even_though_the_laplacian_cannot_see_it(shift):
    initial, step = _transition()
    false_area = tuple(value + shift for value in step.exact_increment)
    with pytest.raises(ValueError, match="full reconstructed endpoint change"):
        _observe(initial, step, nodal_area=false_area)


def test_true_uniform_change_is_retained_even_with_zero_gradient_change():
    initial, step = _transition(uniform=True)
    result = _observe(initial, step)
    assert result.nodal_area == (F(1, 2**55),) * 6
    assert result.nodal_gradient_term == result.remainder_gradient_term == (0,) * 6
    with pytest.raises(ValueError, match="full reconstructed endpoint change"):
        _observe(initial, step, nodal_area=(F(0),) * 6)


@pytest.mark.parametrize("quantum", (F(0), F(-1), 2.0**-54, True))
def test_quantum_must_be_an_exact_positive_fraction(quantum):
    initial, step = _transition()
    with pytest.raises(ValueError, match="positive Fraction"):
        _observe(initial, step, epi_quantum=quantum)


def test_nonintegral_gradient_lattice_is_rejected():
    initial, step = _transition(visible=True)
    with pytest.raises(ValueError, match="integer lattice"):
        _observe(initial, step, epi_quantum=3 * QUANTUM)


@pytest.mark.parametrize("area", ([F(0)] * 6, (0,) * 6, (F(0),) * 5))
def test_area_encoding_and_dimension_are_explicit(area):
    initial, step = _transition()
    with pytest.raises(TypeError, match="matching tuple"):
        _observe(initial, step, nodal_area=area)


def test_two_nodes_do_not_define_the_declared_simple_cycle():
    initial, step = _transition(2)
    with pytest.raises(ValueError, match="at least three"):
        _observe(initial, step)


def test_mismatched_endpoint_dimensions_are_rejected():
    initial, step = _transition(3)
    _, other = _transition(6)
    with pytest.raises(ValueError, match="same size"):
        _observe(initial, step, endpoint=other.after)


def test_cycle_budget_is_immutable():
    initial, step = _transition()
    result = _observe(initial, step)
    with pytest.raises(FrozenInstanceError):
        result.nodal_area = (F(0),) * 6
