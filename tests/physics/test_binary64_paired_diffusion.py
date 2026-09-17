"""Exact fixtures and scope controls for the paired pure-EPI C6 class."""

from dataclasses import FrozenInstanceError
from fractions import Fraction
import math

import pytest

from tnfr.constants.canonical import CHANNEL_WEIGHT_SECONDARY
import tnfr.physics.binary64_nodal_flow as owner
from tnfr.physics.binary64_nodal_flow import (
    observe_binary64_paired_c6_diffusion,
    observe_binary64_unit_quarter_flow,
)


SOURCE = (13 / 16, 3 / 4, 11 / 16, 11 / 16, 3 / 4, 13 / 16)
GRADIENT = (Fraction(-1, 32), 0, Fraction(1, 32), Fraction(1, 32), 0, Fraction(-1, 32))


def _observe(epi=SOURCE, epi_weight=CHANNEL_WEIGHT_SECONDARY):
    return observe_binary64_paired_c6_diffusion(epi=epi, epi_weight=epi_weight)


def test_unit_coefficient_has_independent_exact_quarter_endpoint():
    result = _observe(epi_weight=1.0)
    expected = (Fraction(103, 128), Fraction(3, 4), Fraction(89, 128),
                Fraction(89, 128), Fraction(3, 4), Fraction(103, 128))
    assert result.ideal_pressure == GRADIENT
    assert result.pressure == result.vector_pressure == tuple(map(float, GRADIENT))
    assert result.flow.exact_endpoint == expected
    assert result.flow.endpoint_defect == result.pressure_rounding_defect == (0,) * 6
    assert result.binade_lower == Fraction(1, 2)
    assert result.binade_upper == 1
    assert result.spacing == Fraction(1, 2**53)
    assert result.center == Fraction(3, 4)
    assert result.source_interval == (Fraction(11, 16), Fraction(13, 16))
    assert result.endpoint_interval == (Fraction(89, 128), Fraction(103, 128))
    assert result.source_pair_sums == result.endpoint_pair_sums == (Fraction(3, 2),) * 3
    assert all(row == result.source_pair_sums for row in result.substep_pair_sums)
    assert result.pressure_pair_sums == (0,) * 3
    assert result.mean_drift == 0
    assert result.mean_preserved and result.interval_preserved
    assert result.repeated_pure_channel_invariant
    assert not result.full_runtime_certified


def test_actual_default_coefficient_preserves_pairs_without_substitution():
    result = _observe()
    weight = Fraction(CHANNEL_WEIGHT_SECONDARY)
    expected_pressure = tuple(weight * value for value in GRADIENT)
    assert result.epi_weight == CHANNEL_WEIGHT_SECONDARY
    assert result.ideal_pressure == expected_pressure
    assert result.pressure == result.vector_pressure == tuple(map(float, expected_pressure))
    assert result.pressure_binding_residual == result.pressure_rounding_defect == (0,) * 6
    assert result.flow.mean_before == result.flow.mean_after == Fraction(3, 4)
    assert result.flow.mean_endpoint_defect == result.mean_drift == 0
    assert result.endpoint_interval[0] > result.source_interval[0]
    assert result.endpoint_interval[1] < result.source_interval[1]
    assert not result.nonuniform_plateau


def test_mixed_sign_rows_bind_both_pressure_reducers_to_the_exact_model():
    source = (13 / 16, 25 / 32, 21 / 32, 11 / 16, 23 / 32, 27 / 32)
    result = _observe(epi=source)
    exact = tuple(map(Fraction, source))
    independently_derived = tuple(
        Fraction(CHANNEL_WEIGHT_SECONDARY) * ((exact[i - 1] + exact[(i + 1) % 6]) / 2 - exact[i])
        for i in range(6)
    )
    assert result.ideal_pressure == independently_derived
    assert result.pressure == result.vector_pressure == tuple(map(float, independently_derived))
    assert result.pressure_pair_sums == (0,) * 3
    assert result.endpoint_pair_sums == result.source_pair_sums


@pytest.mark.parametrize("weight", [CHANNEL_WEIGHT_SECONDARY, 1.0])
def test_nonuniform_one_ulp_pattern_can_remain_a_numeric_plateau(weight):
    center, spacing = 0.75, 2.0**-53
    source = (center + spacing, center, center, center - spacing, center, center)
    result = _observe(epi=source, epi_weight=weight)
    assert any(result.ideal_pressure)
    assert any(result.pressure)
    assert result.flow.endpoint == source
    assert result.nonuniform_plateau
    assert all(result.flow.stasis)
    assert result.mean_drift == 0
    assert result.repeated_pure_channel_invariant


def test_zero_weight_control_is_admitted_without_a_convergence_claim():
    result = _observe(epi_weight=0.0)
    assert result.pressure == result.vector_pressure == (0.0,) * 6
    assert result.flow.endpoint == SOURCE
    assert result.nonuniform_plateau
    assert not result.full_runtime_certified


def test_subnormal_coefficient_rounding_remains_sign_symmetric():
    result = _observe(epi_weight=math.ulp(0.0))
    assert any(result.ideal_pressure)
    assert not any(result.pressure)
    assert result.pressure_rounding_defect == tuple(-value for value in result.ideal_pressure)
    assert result.flow.endpoint == SOURCE
    assert result.nonuniform_plateau
    assert result.mean_drift == 0


@pytest.mark.parametrize("binade", [0.0625, 0.125, 0.25, 0.5])
@pytest.mark.parametrize("weight", [CHANNEL_WEIGHT_SECONDARY, 1.0])
def test_extreme_interior_lattice_states_remain_inside_the_original_interval(binade, weight):
    spacing = math.ulp(binade)
    lower, upper, center = binade + spacing, 2 * binade - spacing, 1.5 * binade
    source = (upper, lower, center, lower, upper, center)
    result = _observe(epi=source, epi_weight=weight)
    assert result.binade_lower == Fraction(binade)
    for step in result.flow.substeps:
        assert all(lower <= value <= upper for value in step.after)
        assert tuple(Fraction(step.after[i]) + Fraction(step.after[i + 3])
                     for i in range(3)) == (2 * Fraction(center),) * 3
    assert result.interval_preserved and result.mean_preserved


def test_lower_declared_boundary_can_be_an_interior_binade_constant():
    result = _observe(epi=(0.05,) * 6)
    assert result.binade_lower == Fraction(1, 32)
    assert result.binade_upper == Fraction(1, 16)
    assert result.center == Fraction(0.05)
    assert result.flow.endpoint == (0.05,) * 6
    assert not result.nonuniform_plateau


def test_refreshed_second_pure_channel_call_reenters_the_same_exact_class():
    first = _observe()
    second = _observe(epi=first.flow.endpoint)
    assert second.center == first.center
    assert second.spacing == first.spacing
    assert second.source_interval == first.endpoint_interval
    assert second.endpoint_interval[0] >= first.endpoint_interval[0]
    assert second.endpoint_interval[1] <= first.endpoint_interval[1]
    assert second.mean_drift == 0
    assert second.source_pair_sums == first.endpoint_pair_sums


def test_common_half_lattice_center_is_rejected_and_has_a_real_tie_obstruction():
    spacing = 2.0**-53
    source = (0.75,) * 3 + (0.75 + spacing,) * 3
    with pytest.raises(ValueError, match="half-grid"):
        _observe(epi=source)
    held = observe_binary64_unit_quarter_flow(
        epi=source, pressure=(2.0**-50,) * 3 + (-2.0**-50,) * 3,
    )
    assert held.endpoint == (0.75,) * 6
    assert held.mean_pressure == 0
    assert held.mean_after - held.mean_before == Fraction(-1, 2**54)
    assert all(cell.upper_tie for cell in held.substeps[0].source_cells[:3])
    assert all(cell.lower_tie for cell in held.substeps[0].source_cells[3:])


@pytest.mark.parametrize("source", [
    (0.5,) * 6,
    (1.0,) * 6,
    (0.5 + 2.0**-53,) * 3 + (0.5 - 2.0**-53,) * 3,
])
def test_inherited_half_epi_or_cross_binade_states_are_outside_the_class(source):
    with pytest.raises(ValueError, match="common normal binade"):
        _observe(epi=source)


def test_mismatched_opposite_pair_sums_are_rejected():
    source = (math.nextafter(0.75, math.inf),) + (0.75,) * 5
    with pytest.raises(ValueError, match="common exact pair sum"):
        _observe(epi=source)


@pytest.mark.parametrize("source", [SOURCE[:5], SOURCE + (0.75,)])
def test_only_ordered_six_node_cycle_coordinates_are_admitted(source):
    with pytest.raises(ValueError, match="exactly six"):
        _observe(epi=source)


@pytest.mark.parametrize("weight", [1, True, Fraction(1, 2), "0.5"])
def test_coefficient_must_already_be_a_represented_float(weight):
    with pytest.raises(TypeError, match="actual binary64 float"):
        _observe(epi_weight=weight)


@pytest.mark.parametrize("weight", [math.nan, math.inf, -math.inf, -0.1, 1.1])
def test_coefficient_requires_finite_unit_interval(weight):
    with pytest.raises(ValueError):
        _observe(epi_weight=weight)


@pytest.mark.parametrize("source,error", [
    (list(SOURCE), TypeError),
    ((), ValueError),
    ((1,) * 6, TypeError),
    ((Fraction(3, 4),) * 6, TypeError),
    ((math.nan,) * 6, ValueError),
    ((math.inf,) * 6, ValueError),
    ((0.01,) * 6, ValueError),
    ((1.5,) * 6, ValueError),
])
def test_source_types_and_declared_epi_band_are_validated(source, error):
    with pytest.raises(error):
        _observe(epi=source)


def test_scalar_reducer_binding_rejects_altered_numeric_pressure(monkeypatch):
    original = owner.mean_neighbor_difference

    def altered(*args, **kwargs):
        return math.nextafter(original(*args, **kwargs), math.inf)

    monkeypatch.setattr(owner, "mean_neighbor_difference", altered)
    with pytest.raises(RuntimeError, match="exact rounded C6 pressure"):
        _observe()


def test_vector_reducer_binding_is_checked_separately(monkeypatch):
    original = owner.edge_mean_differences

    def altered(*args, **kwargs):
        values = original(*args, **kwargs)
        values[0] = math.nextafter(float(values[0]), math.inf)
        return values

    monkeypatch.setattr(owner, "edge_mean_differences", altered)
    with pytest.raises(RuntimeError, match="exact rounded C6 pressure"):
        _observe()


def test_shared_euler_endpoint_still_requires_its_exact_rounding_cell(monkeypatch):
    original = owner.euler_update

    def altered(*args):
        return math.nextafter(original(*args), math.inf)

    monkeypatch.setattr(owner, "euler_update", altered)
    with pytest.raises(RuntimeError, match="nearest-even addition cell"):
        _observe()


def test_ieee_precondition_is_not_inferred_from_membership(monkeypatch):
    monkeypatch.setattr(owner, "uses_ieee_binary64_rounding", lambda: False)
    with pytest.raises(RuntimeError, match="IEEE binary64"):
        _observe()


def test_both_reducer_binding_requires_the_numpy_backend(monkeypatch):
    monkeypatch.setattr(owner, "np", None)
    with pytest.raises(RuntimeError, match="NumPy backend"):
        _observe()


def test_public_result_is_frozen_and_has_no_caller_supplied_reference_cache():
    result = _observe()
    with pytest.raises(FrozenInstanceError):
        result.center = Fraction(0)
    with pytest.raises(FrozenInstanceError):
        result.pressure = (0.0,) * 6
    with pytest.raises(TypeError):
        observe_binary64_paired_c6_diffusion(epi=SOURCE, epi_weight=1.0, reference=result)
