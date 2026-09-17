"""Independent exact fixtures for the restricted held binary64 Euler owner."""

from dataclasses import FrozenInstanceError
from fractions import Fraction
import math

import pytest

import tnfr.physics.binary64_nodal_flow as owner
from tnfr.physics.binary64_nodal_flow import observe_binary64_unit_quarter_flow


def _flow(epi=(0.5,), pressure=(0.0,), **kwargs):
    return observe_binary64_unit_quarter_flow(epi=epi, pressure=pressure, **kwargs)


def test_exact_normal_inputs_preserve_the_signed_nodal_budget():
    result = _flow(epi=(0.25, 0.5), pressure=(1.0, -0.5))
    assert result.endpoint == (0.5, 0.375)
    assert result.exact_endpoint == (Fraction(1, 2), Fraction(3, 8))
    assert result.ideal_endpoint == result.exact_endpoint
    assert result.exact_increment == (Fraction(1, 16), Fraction(-1, 32))
    assert result.endpoint_defect == result.scaling_defect == result.addition_defect == (0, 0)
    assert result.error_identity_residual == (0, 0)
    assert result.mean_before == Fraction(3, 8)
    assert result.mean_after == Fraction(7, 16)
    assert result.mean_pressure == Fraction(1, 4)
    assert result.mean_identity_residual == 0
    assert tuple(step.ordinal for step in result.substeps) == (1, 2, 3, 4)
    for step in result.substeps:
        assert step.addition_error == (0, 0)
        assert all(cell.contains_exact_input for cell in step.result_cells)
        assert not any(cell.lower_tie or cell.upper_tie for cell in step.result_cells)


@pytest.mark.parametrize("pressure", [-2.0**-51, 2.0**-50])
def test_half_epi_stasis_includes_both_asymmetric_midpoint_ties(pressure):
    result = _flow(pressure=(pressure,))
    assert result.endpoint == (0.5,)
    assert result.stasis == result.half_epi_pressure_band_membership == (True,)
    assert result.half_epi_pressure_band == (Fraction(-1, 2**51), Fraction(1, 2**50))
    assert result.endpoint_defect == (-Fraction(pressure) / 4,)
    assert result.scaling_defect == (0,)
    for step in result.substeps:
        source = step.source_cells[0]
        assert source.predecessor == 0.5 - 2.0**-54
        assert source.successor == 0.5 + 2.0**-53
        assert source.lower - Fraction(1, 2) == Fraction(-1, 2**55)
        assert source.upper - Fraction(1, 2) == Fraction(1, 2**54)
        assert source.even_significand and source.contains_exact_input
        assert source.lower_tie == (pressure < 0)
        assert source.upper_tie == (pressure > 0)


@pytest.mark.parametrize("boundary,outward", [(-2.0**-51, -math.inf), (2.0**-50, math.inf)])
def test_one_represented_pressure_outside_half_cell_changes_the_first_update(boundary, outward):
    pressure = math.nextafter(boundary, outward)
    result = _flow(pressure=(pressure,))
    assert result.half_epi_pressure_band_membership == result.stasis == (False,)
    first = result.substeps[0]
    assert first.stasis == (False,)
    assert not first.source_cells[0].contains_exact_input
    assert first.result_cells[0].contains_exact_input


@pytest.mark.parametrize("boundary,inward", [(-2.0**-51, math.inf), (2.0**-50, -math.inf)])
def test_one_represented_pressure_inside_half_cell_has_strict_stasis(boundary, inward):
    result = _flow(pressure=(math.nextafter(boundary, inward),))
    assert result.stasis == (True,)
    source = result.substeps[0].source_cells[0]
    assert source.contains_exact_input
    assert not source.lower_tie and not source.upper_tie


def test_half_pressure_membership_does_not_certify_stasis_at_another_epi():
    result = _flow(epi=(0.25,), pressure=(2.0**-50,))
    assert result.half_epi_pressure_band_membership == (True,)
    assert result.stasis == (False,)
    assert result.endpoint == (0.25 + 2.0**-52,)
    assert result.endpoint_defect == (0,)


def test_upper_one_cell_attains_the_addition_part_of_the_uniform_bound():
    result = _flow(epi=(1.0,), pressure=(2.0**-49,))
    assert result.endpoint == (1.0,)
    assert result.endpoint_defect == result.addition_defect == (Fraction(-1, 2**51),)
    assert result.scaling_defect == (0,)
    assert result.addition_error_bound == Fraction(1, 2**53)
    assert result.scaling_error_bound == Fraction(1, 2**1075)
    assert result.local_endpoint_error_bound == Fraction(1, 2**51) + Fraction(1, 2**1073)
    for step in result.substeps:
        assert step.addition_error == (Fraction(-1, 2**53),)
        cell = step.result_cells[0]
        assert cell.upper - 1 == Fraction(1, 2**53)
        assert 1 - cell.lower == Fraction(1, 2**54)
        assert cell.even_significand and cell.upper_tie and cell.contains_exact_input


@pytest.mark.parametrize(
    "source,pressure,expected,which_tie",
    [
        (math.nextafter(0.5, math.inf), 2.0**-50, 0.5 + 2.0**-52, "upper_tie"),
        (math.nextafter(0.5, -math.inf), -2.0**-51, 0.5 - 2.0**-53, "lower_tie"),
    ],
)
def test_odd_significand_excludes_midpoint_and_moves_to_even_neighbor(
    source, pressure, expected, which_tie,
):
    result = _flow(epi=(source,), pressure=(pressure,))
    first = result.substeps[0]
    assert first.after == (expected,)
    assert not first.source_cells[0].even_significand
    assert getattr(first.source_cells[0], which_tie)
    assert not first.source_cells[0].contains_exact_input
    assert first.result_cells[0].even_significand
    assert first.result_cells[0].contains_exact_input
    assert first.stasis == (False,)
    assert all(step.stasis == (True,) for step in result.substeps[1:])


def test_zero_mean_held_pressure_can_create_signed_epi_mean_drift():
    """This is a numeric input fixture, without a pressure-generation claim."""
    pressure = (2.0**-50, -2.0**-50) * 3
    result = _flow(epi=(0.5,) * 6, pressure=pressure)
    assert result.endpoint == (0.5, 0.5 - 2.0**-52) * 3
    assert result.mean_pressure == 0
    assert result.mean_after - result.mean_before == Fraction(-1, 2**53)
    assert result.mean_endpoint_defect == result.mean_addition_defect == Fraction(-1, 2**53)
    assert result.mean_scaling_defect == result.mean_identity_residual == 0
    assert result.endpoint_defect == (Fraction(-1, 2**52), Fraction(0)) * 3
    assert result.stasis == (True, False) * 3
    for index, total in enumerate(result.endpoint_defect):
        additions = sum(step.addition_error[index] for step in result.substeps)
        assert total == 4 * result.scaling_error[index] + additions
        assert abs(total) <= result.local_endpoint_error_bound


@pytest.mark.parametrize("pressure_multiplier,increment_multiplier,endpoint_multiplier,error_multiplier", [
    (1, 0, 1, Fraction(-1, 4)),
    (8, 0, 1, Fraction(-2)),
    (24, 2, 9, Fraction(2)),
])
def test_subnormal_scaling_is_retained_as_an_exact_signed_defect(
    pressure_multiplier, increment_multiplier, endpoint_multiplier, error_multiplier,
):
    tiny = math.ulp(0.0)
    result = _flow(epi=(tiny,), pressure=(pressure_multiplier * tiny,), epi_lower=tiny)
    assert result.rounded_increment == (increment_multiplier * tiny,)
    assert result.endpoint == (endpoint_multiplier * tiny,)
    expected_error = error_multiplier * Fraction(tiny)
    assert result.endpoint_defect == result.scaling_defect == (expected_error,)
    assert result.addition_defect == (0,)
    assert result.error_identity_residual == (0,)
    source = result.substeps[0].source_cells[0]
    assert source.lower == Fraction(tiny) / 2
    assert source.upper == 3 * Fraction(tiny) / 2
    assert not source.even_significand
    assert abs(result.scaling_error[0]) <= result.scaling_error_bound


def test_negative_zero_pressure_has_zero_signed_nodal_budget():
    result = _flow(pressure=(-0.0,))
    assert math.copysign(1.0, result.pressure[0]) == -1
    assert math.copysign(1.0, result.rate[0]) == 1
    assert result.endpoint == (0.5,)
    assert result.endpoint_defect == result.scaling_defect == result.addition_defect == (0,)


def test_scalar_owner_endpoint_is_verified_against_independent_exact_cell(monkeypatch):
    monkeypatch.setattr(owner, "euler_update", lambda epi, dt, rate: math.nextafter(epi, math.inf))
    with pytest.raises(RuntimeError, match="nearest-even addition cell"):
        _flow()


def test_platform_rounding_predicate_is_an_explicit_precondition(monkeypatch):
    monkeypatch.setattr(owner, "uses_ieee_binary64_rounding", lambda: False)
    with pytest.raises(RuntimeError, match="IEEE binary64"):
        _flow()


@pytest.mark.parametrize("field", ["epi", "pressure"])
@pytest.mark.parametrize("value", [[0.5], {0.5}, {0: 0.5}])
def test_vectors_must_be_actual_ordered_tuples(field, value):
    with pytest.raises(TypeError, match="ordered tuple"):
        _flow(**{field: value})


@pytest.mark.parametrize("field", ["epi", "pressure"])
def test_empty_vectors_are_rejected(field):
    with pytest.raises(ValueError, match="nonempty"):
        _flow(**{field: ()})


@pytest.mark.parametrize("field", ["epi", "pressure"])
@pytest.mark.parametrize("value", [True, 1, Fraction(1, 2), "0.5"])
def test_scalar_values_are_not_implicitly_rationalized_or_coerced(field, value):
    with pytest.raises(TypeError, match="actual binary64 float"):
        _flow(**{field: (value,)})


@pytest.mark.parametrize("field", ["epi", "pressure"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_scalars_are_rejected(field, value):
    with pytest.raises(ValueError, match="finite"):
        _flow(**{field: (value,)})


def test_float_subclasses_are_rejected():
    class ClaimedFloat(float):
        pass

    with pytest.raises(TypeError, match="actual binary64 float"):
        _flow(epi=(ClaimedFloat(0.5),))


def test_dimensions_must_match():
    with pytest.raises(ValueError, match="matching dimensions"):
        _flow(epi=(0.5, 0.5))


@pytest.mark.parametrize("kwargs,error", [
    ({"epi_lower": 0}, TypeError),
    ({"epi_upper": Fraction(1)}, TypeError),
    ({"epi_lower": math.nan}, ValueError),
    ({"epi_upper": math.inf}, ValueError),
    ({"epi_lower": 0.0}, ValueError),
    ({"epi_lower": -0.1}, ValueError),
    ({"epi_upper": 1.1}, ValueError),
    ({"epi_lower": 0.6, "epi_upper": 0.4}, ValueError),
])
def test_declared_band_is_positive_finite_and_ordered(kwargs, error):
    with pytest.raises(error):
        _flow(**kwargs)


@pytest.mark.parametrize("epi", [0.0, 0.01, 1.1])
def test_initial_epi_must_lie_in_the_declared_band(epi):
    with pytest.raises(ValueError, match="initial EPI"):
        _flow(epi=(epi,))


@pytest.mark.parametrize("epi,pressure", [(0.9, 1.0), (0.1, -1.0)])
def test_every_unprojected_substep_must_remain_inside_the_band(epi, pressure):
    with pytest.raises(ValueError, match="unclipped Euler output"):
        _flow(epi=(epi,), pressure=(pressure,))


def test_equal_positive_band_endpoints_allow_exact_stasis():
    result = _flow(epi_lower=0.5, epi_upper=0.5)
    assert result.stasis == (True,)
    with pytest.raises(ValueError, match="unclipped Euler output"):
        _flow(pressure=(1.0,), epi_lower=0.5, epi_upper=0.5)


def test_result_and_nested_rounding_records_are_frozen():
    result = _flow()
    with pytest.raises(FrozenInstanceError):
        result.endpoint = (0.25,)
    with pytest.raises(FrozenInstanceError):
        result.substeps[0].ordinal = 99
    with pytest.raises(FrozenInstanceError):
        result.substeps[0].source_cells[0].contains_exact_input = False


def test_tiny_held_pressure_pair_leakage_can_cause_one_ulp_drift_every_substep():
    """Supplied numeric pressures need not arise from a canonical phase state."""
    positive = math.nextafter(2.0**-50, math.inf)
    negative = math.nextafter(-2.0**-50, math.inf)
    spacing = Fraction(1, 2**53)
    result = _flow(epi=(0.75, 0.75), pressure=(positive, negative))
    assert Fraction(positive) == Fraction(1, 2**50) + Fraction(1, 2**102)
    assert Fraction(negative) == Fraction(-1, 2**50) + Fraction(1, 2**103)
    assert Fraction(positive) + Fraction(negative) == Fraction(3, 2**103)
    for ordinal, step in enumerate(result.substeps, 1):
        assert Fraction(step.after[0]) == Fraction(3, 4) + ordinal * spacing
        assert step.after[1] == 0.75
        assert not any(cell.lower_tie or cell.upper_tie for cell in step.result_cells)
    assert result.mean_after - result.mean_before == Fraction(1, 2**52)
    assert result.mean_pressure / 4 == Fraction(3, 2**106)
    assert result.mean_endpoint_defect == Fraction(1, 2**52) - Fraction(3, 2**106)
    assert result.mean_identity_residual == 0
