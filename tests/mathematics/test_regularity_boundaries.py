"""Independent range and policy controls for sampled EPI regularity."""

import math
from fractions import Fraction

import numpy as np
import pytest

from tnfr.errors import TNFRValueError
from tnfr.mathematics.epi import (
    BEPIElement,
    evaluate_composite_epi_regularity_transform,
)
from tnfr.mathematics.spaces import BanachSpaceEPI


@pytest.mark.parametrize("amplitude", [1e-100, 3.0, 1e200, 1e308, 1e200 * (1 + 2j)])
@pytest.mark.parametrize("length", [0.5, 1.0, 2.0])
def test_linear_two_sample_quotient_matches_exact_quadrature(amplitude, length):
    # The secant is constant. The two endpoint trapezoid is |a|² L / 2,
    # while its derivative energy is |a|² / L.
    square = (
        Fraction(float(np.real(amplitude))) ** 2
        + Fraction(float(np.imag(amplitude))) ** 2
    )
    length_q = Fraction(length)
    expected = float((square / length_q) / (1 + square * length_q / 2))
    actual = BanachSpaceEPI().derivative_regularity([0, amplitude], [0, length])
    assert actual == pytest.approx(expected, rel=3e-15, abs=0.0)


def test_large_discrete_tail_has_finite_regularities_and_cannot_false_pass():
    element = BEPIElement([0, 0], [1e200], [0, 1])
    result = evaluate_composite_epi_regularity_transform(
        element, lambda value: value * 0.5, tolerance=0
    )
    assert result.regularity_before == 1e200
    assert result.regularity_after == 5e199
    assert not result.satisfied
    assert result.required == 1e200
    assert result.deficit == 5e199
    assert result.ratio == 0.5


@pytest.mark.parametrize("amplitude", [1e-200, 1e200])
def test_discrete_norm_uses_both_complex_channels_without_squared_scale(amplitude):
    value = BanachSpaceEPI().composite_epi_regularity(
        [0, 0], [amplitude * (3 + 4j)], x_grid=[0, 1]
    )
    assert value == pytest.approx(5 * amplitude, rel=2e-15, abs=0.0)


@pytest.mark.parametrize("name", ["alpha", "beta", "gamma"])
@pytest.mark.parametrize(
    "bad", [True, np.bool_(False), "1", 1 + 0j, np.inf, np.nan, Fraction(1, 10**400)]
)
def test_composite_weights_reject_original_invalid_scalars(name, bad):
    with pytest.raises(TNFRValueError, match="strictly positive"):
        BanachSpaceEPI().composite_epi_regularity(
            [0, 0], [1], x_grid=[0, 1], **{name: bad}
        )


def test_composite_weights_compute_with_admitted_normalized_values():
    value = BanachSpaceEPI().composite_epi_regularity(
        [0, 0], [3], x_grid=[0, 1], beta=Fraction(2, 3)
    )
    assert value == 2.0


def test_unrepresentable_energy_arithmetic_and_output_reject():
    space = BanachSpaceEPI()
    with pytest.raises(TNFRValueError, match="finite energy arithmetic"):
        space.derivative_regularity([0, 1], [0, 1e-300])
    with pytest.raises(TNFRValueError, match="regularity must be finite"):
        space.composite_epi_regularity([0, 0], [1e308], x_grid=[0, 1], beta=2)
    with pytest.raises(TNFRValueError, match="finite spacings"):
        BEPIElement([0, 0], [], [-1e308, 1e308])


@pytest.mark.parametrize("amplitude,length", [(1e-200, 1.0), (1.0, 1e200)])
def test_positive_derivative_quotient_lost_to_zero_rejects(amplitude, length):
    square = Fraction(amplitude) ** 2
    exact = (square / Fraction(length)) / (1 + square * Fraction(length) / 2)
    assert exact > 0 and float(exact) == 0
    with pytest.raises(TNFRValueError, match="Nonzero derivative regularity"):
        BanachSpaceEPI().derivative_regularity([0, amplitude], [0, length])


@pytest.mark.parametrize("value", [0, 1e-200, 1e200])
def test_exact_constant_field_retains_zero_derivative_regularity(value):
    assert BanachSpaceEPI().derivative_regularity([value, value], [0, 1]) == 0


class _ObservedRegularity(BanachSpaceEPI):
    def __init__(self, before, after):
        self.values = iter((before, after))

    def composite_epi_regularity(self, *args, **kwargs):
        return next(self.values)


def _evaluate(before, after, **kwargs):
    return evaluate_composite_epi_regularity_transform(
        BEPIElement([0, 0], [], [0, 1]),
        lambda value: value,
        space=_ObservedRegularity(before, after),
        **kwargs,
    )


def test_tolerance_rounding_does_not_turn_a_strict_failure_into_success():
    report = _evaluate(1.0, np.nextafter(1.0, 0.0), tolerance=2**-54)
    assert not report.satisfied
    assert report.deficit == 2**-53


def test_product_rounding_does_not_turn_a_strict_failure_into_success():
    report = _evaluate(1 + 2**-52, 1 + 2**-51, kappa=1 + 2**-52, tolerance=0)
    assert not report.satisfied
    assert report.deficit == 2**-104


@pytest.mark.parametrize("name", ["kappa", "tolerance"])
@pytest.mark.parametrize(
    "bad", [True, np.bool_(False), "0", 0j, -1, np.inf, np.nan, Fraction(1, 10**400)]
)
def test_transform_policy_is_admitted_before_callback(name, bad):
    called = []
    with pytest.raises(TNFRValueError, match=name):
        evaluate_composite_epi_regularity_transform(
            BEPIElement([0, 0], [], [0, 1]),
            lambda element: called.append(element),
            **{name: bad},
        )
    assert not called


@pytest.mark.parametrize("bad", [np.inf, np.nan, -1, True, "1"])
@pytest.mark.parametrize("position", [0, 1])
def test_custom_observation_cannot_supply_an_invalid_regularity(bad, position):
    values = [1.0, 1.0]
    values[position] = bad
    with pytest.raises(TNFRValueError, match="regularity_"):
        _evaluate(*values)


def test_finite_baseline_report_overflow_is_distinct_from_zero_baseline_ratio():
    with pytest.raises(TNFRValueError, match="required regularity"):
        _evaluate(1e308, 1e308, kappa=2, tolerance=1e308)
    with pytest.raises(TNFRValueError, match="regularity ratio"):
        _evaluate(1e-300, 1e300)
    positive = _evaluate(0, 1)
    assert positive.satisfied and math.isinf(positive.ratio)
    zero = _evaluate(0, 0)
    assert zero.satisfied and zero.ratio == 1.0
