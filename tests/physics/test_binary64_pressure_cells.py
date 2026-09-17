"""Exact inverse pressure cells distinguish a trace from its endpoint or mean."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F
import math

import pytest

import tnfr.physics.binary64_nodal_flow as owner

TINY = math.ulp(0.0)


def _box(epi=(.5,), pressure=(0.0,), **kwargs):
    return owner.derive_binary64_quarter_pressure_box(epi=epi, pressure=pressure, **kwargs)


def _trace(epi, pressure):
    """Independent primitive spelling, without invoking inverse-cell helpers."""
    values = epi
    rows = []
    for _ in range(4):
        values = tuple(x + .0625 * (1.0 * p + 0.0) for x, p in zip(values, pressure, strict=True))
        rows.append(values)
    return tuple(rows)


def test_half_stasis_recovers_the_exact_asymmetric_pressure_band():
    box = _box()
    cell = box.coordinates[0]
    assert cell.increment_lower == -F(1, 2**55)
    assert cell.increment_upper == F(1, 2**54)
    assert cell.increment_lower_closed and cell.increment_upper_closed
    assert cell.first_pressure == -2**-51
    assert cell.last_pressure == 2**-50
    assert box.contains((-2.0**-51,)) and box.contains((2.0**-50,))
    assert not box.contains((math.nextafter(-2**-51, -math.inf),))
    assert not box.contains((math.nextafter(2**-50, math.inf),))
    assert _trace((.5,), (cell.first_pressure,)) == ((.5,),) * 4
    assert _trace((.5,), (cell.last_pressure,)) == ((.5,),) * 4


@pytest.mark.parametrize("source,pressure,first,last,lower,upper,closed,expected", (
    (1, 0, -8, 8, -8, 8, True, (1, 1, 1, 1)),
    (1, 16, 9, 23, 8, 24, False, (2, 3, 4, 5)),
    (1, 32, 24, 40, 24, 40, True, (3, 5, 7, 9)),
    (9, -16, -23, -9, -24, -8, False, (8, 7, 6, 5)),
    (9, -32, -40, -24, -40, -24, True, (7, 5, 3, 1)),
))
def test_subnormal_scaling_preimage_uses_increment_parity(
    source, pressure, first, last, lower, upper, closed, expected,
):
    box = _box(epi=(source * TINY,), pressure=(pressure * TINY,), epi_lower=TINY)
    cell = box.coordinates[0]
    assert (cell.first_pressure, cell.last_pressure) == (first * TINY, last * TINY)
    assert (cell.pressure_lower, cell.pressure_upper) == (lower * F(TINY), upper * F(TINY))
    assert cell.pressure_lower_closed is cell.pressure_upper_closed is closed
    trace = tuple((value * TINY,) for value in expected)
    assert tuple(step.after for step in box.flow.substeps) == trace
    for candidate in (cell.first_pressure, cell.last_pressure):
        assert box.contains((candidate,))
        assert _trace(box.flow.epi, (candidate,)) == trace
    for candidate in (math.nextafter(cell.first_pressure, -math.inf), math.nextafter(cell.last_pressure, math.inf)):
        assert not box.contains((candidate,))
        assert _trace(box.flow.epi, (candidate,)) != trace


@pytest.mark.parametrize("value", (0.0, TINY, 2 * TINY, 2**-1022, .25, .5, .75, 1.0))
def test_signed_rounding_cells_reflect_bounds_and_tie_flags(value):
    positive = owner._rounding_cell(value, F(value))
    negative = owner._rounding_cell(-value, -F(value))
    assert negative.lower == -positive.upper
    assert negative.upper == -positive.lower
    assert positive.even_significand == negative.even_significand
    assert positive.contains_exact_input and negative.contains_exact_input
    if value == 0.0:
        assert positive.lower == -F(1, 2**1075)
        assert positive.upper == F(1, 2**1075)
        assert positive.even_significand


@pytest.mark.parametrize("epi,pressure", (
    ((.5,), (.125,)),
    ((.5,), (-.125,)),
    ((.75,), (0.0,)),
    ((.375,), (-.03125,)),
    ((.99,), (.001,)),
    ((math.nextafter(.5, math.inf),), (2.0**-50,)),
    ((.5, .5, .75), (2.0**-50, -2.0**-50, .01)),
))
def test_cartesian_extrema_preserve_every_step_and_adjacent_outsiders_do_not(epi, pressure):
    box = _box(epi, pressure)
    expected = tuple(step.after for step in box.flow.substeps)
    for values in (tuple(cell.first_pressure for cell in box.coordinates),
                   tuple(cell.last_pressure for cell in box.coordinates), pressure):
        assert box.contains(values)
        assert _trace(epi, values) == expected
    for i, cell in enumerate(box.coordinates):
        assert cell.recorded_pressure_margin >= 0
        for outside in (math.nextafter(cell.first_pressure, -math.inf),
                        math.nextafter(cell.last_pressure, math.inf)):
            candidate = list(pressure)
            candidate[i] = outside
            candidate = tuple(candidate)
            assert not box.contains(candidate)
            assert _trace(epi, candidate) != expected


def test_zero_signs_are_numeric_trace_equivalent():
    first = _box(pressure=(0.0,))
    second = _box(pressure=(-0.0,))
    assert first.coordinates == second.coordinates
    assert first.contains((-0.0,)) and second.contains((0.0,))
    assert _trace((.5,), (0.0,)) == _trace((.5,), (-0.0,))


@pytest.mark.parametrize("candidate,error", (
    ([0.0], TypeError), ((), ValueError), ((0,), TypeError), ((True,), TypeError),
    ((F(0),), TypeError), ((math.nan,), ValueError), ((math.inf,), ValueError),
    ((0.0, 0.0), ValueError),
))
def test_membership_validates_actual_finite_represented_tuples(candidate, error):
    with pytest.raises(error):
        _box().contains(candidate)


def test_scope_is_a_trace_box_not_a_graph_or_future_certificate():
    box = _box()
    assert box.flow.epi == (.5,) and len(box.flow.substeps) == 4
    assert "not for its final" in type(box).__doc__
    assert "No graph" in type(box).__doc__
    with pytest.raises(FrozenInstanceError):
        box.coordinates = ()
    with pytest.raises(FrozenInstanceError):
        box.coordinates[0].first_pressure = 0.0


def test_unsupported_rounding_precondition_fails_before_derivation(monkeypatch):
    monkeypatch.setattr(owner, "uses_ieee_binary64_rounding", lambda: False)
    with pytest.raises(RuntimeError, match="IEEE"):
        _box()


@pytest.mark.parametrize("integer", (0, 1, 10, 2**20))
@pytest.mark.parametrize("fractional,sign", ((F(1, 8), 0), (F(3, 8), -1), (F(5, 8), 1), (F(7, 8), 0)))
def test_opposite_pressures_at_half_follow_the_two_lattice_bias_law(integer, fractional, sign):
    spacing = F(1, 2**53)
    a = integer + fractional
    q = float(a * spacing)
    pressure = 16 * q
    n, m = round(a), round(2 * a)
    assert 4 * spacing * n < F(1, 2)
    assert 2 * spacing * m < F(1, 4)
    result = owner.observe_binary64_unit_quarter_flow(epi=(.5, .5), pressure=(pressure, -pressure))
    assert result.mean_pressure == 0
    for ordinal, step in enumerate(result.substeps, 1):
        expected = (F(1, 2) + ordinal * spacing * n, F(1, 2) - ordinal * spacing * m / 2)
        assert tuple(map(F, step.after)) == expected
        assert sum(expected) - 1 == ordinal * sign * spacing / 2
    assert result.mean_after - result.mean_before == sign * spacing
