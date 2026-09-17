"""Complete-cell obstruction, exact parity witnesses and scope controls."""

from dataclasses import replace
from fractions import Fraction as F

import pytest

from tnfr.constants.canonical import CHANNEL_WEIGHT_PRIMARY, CHANNEL_WEIGHT_SECONDARY
from tnfr.dynamics._euler_kernel import (
    NODAL_REMAINDER_DENOMINATOR_BITS, _validate_nodal_remainder_state,
    initialize_nodal_remainder,
)
from tnfr.physics.binary64_nodal_flow import _rounding_cell
from tnfr.physics.c6_carried_cell_escape import (
    derive_c6_carried_complete_cell_obstruction, observe_c6_carried_complete_cell_escape,
)
from tnfr.physics.c6_carried_closure import derive_c6_carried_closure
from tnfr.physics.c6_carried_profile import derive_c6_carried_profile
from tnfr.physics.c6_pressure_lattice import derive_c6_pressure_lattice


PHASE = tuple(map(float.fromhex, (
    "0x1.0b8fb3e3956cbp-55", "0x1.0c152382d7365p+0", "0x1.0c152382d7365p+1",
    "0x1.921fb54442d18p+1", "0x1.0c152382d7365p+2", "0x1.4f1a6c638d03fp+2",
)))
OFFSETS = (-3, 0, 2, -1, 8, -5)
BALANCE = (
    (-6, -1, 2, 2, 8, -7), (-5, 2, 4, -2, 6, -7), (-4, -2, 0, 2, 6, -4),
    (-4, -1, 2, -2, 8, -5), (-3, 0, 2, -2, 8, -7), (-2, -1, 4, -2, 6, -7),
    (-2, 2, 0, -2, 6, -6),
)


def _epi(offsets):
    return tuple(.5 + n * 2.**-54 for n in offsets)


def _closure(*, phase=PHASE, timestep=1 / 16, lower=.375, upper=.625, epi=None):
    profile = derive_c6_carried_profile(derive_c6_pressure_lattice(
        phase=phase, epi_weight=CHANNEL_WEIGHT_SECONDARY, phase_weight=CHANNEL_WEIGHT_PRIMARY,
    ))
    state = initialize_nodal_remainder(_epi(OFFSETS) if epi is None else epi, epi_lower=lower, epi_upper=upper)
    return derive_c6_carried_closure(profile, state=state, timestep=timestep)


def test_uniform_negative_step_bound_is_derived_from_pressure_interval_endpoints():
    result = derive_c6_carried_complete_cell_obstruction(_closure())
    lattice = result.closure.base_tube.contraction.profile.lattice
    assert result.grid_quantum == F(1, 2**NODAL_REMAINDER_DENOMINATOR_BITS)
    assert result.minimum_cell_grid_width == lattice.epi_quantum / 2 - 2 * result.grid_quantum
    assert 0 < result.maximum_negative_increment < result.minimum_cell_grid_width
    assert result.maximum_negative_increment / lattice.epi_quantum < F(4, 25)
    for lo, hi, mlo, mhi, source in zip(
        result.pressure_lower, result.pressure_upper, result.closure.gradient_index_lower,
        result.closure.gradient_index_upper, lattice.sources, strict=True,
    ):
        assert lo == F(source + float(lattice.source.epi_weight * lattice.gradient_quantum * mlo))
        assert hi == F(source + float(lattice.source.epi_weight * lattice.gradient_quantum * mhi))
        assert lo <= hi
    assert result.finite_complete_cell_union_invariance_excluded
    assert not result.correlated_carry_region_excluded
    assert not result.saved_trajectory_escape_certified and not result.future_runtime_certified


@pytest.mark.parametrize("offsets", (OFFSETS, *BALANCE))
def test_greatest_admissible_carries_produce_one_sided_canonical_escape(offsets):
    bound = derive_c6_carried_complete_cell_obstruction(_closure())
    result = observe_c6_carried_complete_cell_escape(bound, epi_states=(_epi(offsets),))
    exact = _validate_nodal_remainder_state(result.state)
    assert result.family_invariance_excluded and not result.saved_trajectory_escape_certified
    assert not result.band_failure and result.endpoint is not None
    assert result.endpoint.after.exact_epi == result.exact_candidate
    assert result.endpoint_visible_sum > result.source_visible_sum
    assert all(y >= x for x, y in zip(result.state.epi, result.endpoint.after.epi, strict=True))
    for value, visible, pressure, changed in zip(
        exact, result.state.epi, result.pressure, result.endpoint.visible_increment, strict=True,
    ):
        cell = _rounding_cell(visible, value)
        assert cell.contains_exact_input
        assert value == cell.upper - (0 if cell.even_significand else bound.grid_quantum)
        assert changed > 0 if pressure > 0 else changed == 0
    assert result.state.remainder != bound.closure.base_tube.state.remainder


def test_family_witness_selects_the_maximum_visible_sum_and_escapes_the_entire_union():
    family = tuple(_epi(offsets) for offsets in BALANCE) + (_epi(OFFSETS),)
    result = observe_c6_carried_complete_cell_escape(
        derive_c6_carried_complete_cell_obstruction(_closure()), epi_states=family,
    )
    assert result.source_visible_sum == max(sum(map(F, row), F(0)) for row in family)
    assert result.endpoint.after.epi not in family
    assert result.endpoint_visible_sum > max(sum(map(F, row), F(0)) for row in family)


def test_physical_upper_boundary_is_reported_as_band_failure_without_a_fake_endpoint():
    result = observe_c6_carried_complete_cell_escape(
        derive_c6_carried_complete_cell_obstruction(_closure()), epi_states=((.625,) * 6,),
    )
    assert result.band_failure and result.endpoint is None and result.endpoint_visible_sum is None
    assert any(value > F(.625) for value in result.exact_candidate)
    assert all(value >= F(.375) for value in result.exact_candidate)
    assert not result.saved_trajectory_escape_certified


def test_physical_lower_boundary_still_holds_negative_pressure_coordinates():
    result = observe_c6_carried_complete_cell_escape(
        derive_c6_carried_complete_cell_obstruction(_closure()), epi_states=((.375,) * 6,),
    )
    assert not result.band_failure and result.endpoint is not None
    assert all(value >= F(.375) for value in result.exact_candidate)


def test_public_closure_and_obstruction_caches_are_fully_rederived():
    closure = _closure()
    expected = derive_c6_carried_complete_cell_obstruction(closure)
    forged = replace(closure, gradient_index_lower=(0,) * 6, gradient_index_upper=(0,) * 6)
    assert derive_c6_carried_complete_cell_obstruction(forged) == expected
    forged_bound = replace(expected, grid_quantum=F(1), minimum_cell_grid_width=F(0),
                           pressure_lower=(F(0),) * 6, pressure_upper=(F(0),) * 6)
    assert observe_c6_carried_complete_cell_escape(forged_bound, epi_states=(_epi(OFFSETS),)) == (
        observe_c6_carried_complete_cell_escape(expected, epi_states=(_epi(OFFSETS),)))


@pytest.mark.parametrize("value", (None, (), 1))
def test_wrong_closure_types_are_rejected(value):
    with pytest.raises(TypeError, match="C6CarriedClosure"):
        derive_c6_carried_complete_cell_obstruction(value)


@pytest.mark.parametrize("family", ((), [], ((_epi(OFFSETS)),) * 2))
def test_empty_unordered_and_duplicate_cell_families_are_rejected(family):
    with pytest.raises(ValueError):
        observe_c6_carried_complete_cell_escape(
            derive_c6_carried_complete_cell_obstruction(_closure()), epi_states=family,
        )


def test_wrong_obstruction_type_is_rejected():
    with pytest.raises(TypeError, match="C6CarriedCompleteCellObstruction"):
        observe_c6_carried_complete_cell_escape(_closure(), epi_states=(_epi(OFFSETS),))


def test_visible_family_must_satisfy_the_proved_gradient_interval():
    with pytest.raises(ValueError, match="gradient bounds"):
        observe_c6_carried_complete_cell_escape(
            derive_c6_carried_complete_cell_obstruction(_closure()), epi_states=((.375, .625) * 3,),
        )


def test_a_source_without_positive_pressure_everywhere_cannot_use_the_obstruction():
    with pytest.raises(ValueError, match="all-nonpositive"):
        derive_c6_carried_complete_cell_obstruction(_closure(phase=(0.,) * 6))


def test_a_large_nodal_step_abstains_from_the_complete_cell_argument():
    with pytest.raises(ValueError, match="negative nodal increments"):
        derive_c6_carried_complete_cell_obstruction(_closure(timestep=1.))


def test_a_singleton_band_has_no_uniform_full_cell_width():
    with pytest.raises(ValueError, match="at least one"):
        derive_c6_carried_complete_cell_obstruction(_closure(lower=.5, upper=.5, epi=(.5,) * 6))


def test_narrower_band_membership_is_enforced_before_selecting_a_witness():
    bound = derive_c6_carried_complete_cell_obstruction(_closure(lower=.49, upper=.51))
    with pytest.raises(ValueError, match="closure's declared band"):
        observe_c6_carried_complete_cell_escape(bound, epi_states=((.625,) * 6,))
