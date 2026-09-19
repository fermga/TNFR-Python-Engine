"""A bounded centered coordinate converts strict pressure into finite passage."""

import math
from dataclasses import replace
from fractions import Fraction as F

import pytest

from tnfr.dynamics._euler_kernel import (
    advance_nodal_remainder,
    initialize_nodal_remainder,
)
from tnfr.physics.c6_carried_passage import derive_c6_carried_positive_pressure_passage
from tnfr.physics.c6_carried_profile import derive_c6_carried_profile
from tnfr.physics.c6_carried_tube import derive_c6_carried_tube
from tnfr.physics.c6_pressure_lattice import (
    derive_c6_pressure_lattice,
    observe_c6_pressure_lattice,
)


def _tube(
    *,
    epi=None,
    phase=None,
    weight=0.5,
    phase_weight=0.5,
    lower=0.375,
    upper=0.625,
    timestep=0.5,
):
    lattice = derive_c6_pressure_lattice(
        phase=(0.0,) * 6 if phase is None else phase,
        epi_weight=weight,
        phase_weight=phase_weight,
    )
    if epi is None:
        epi = (math.nextafter(0.5, -math.inf),) + (0.5,) * 5
    state = initialize_nodal_remainder(epi, epi_lower=lower, epi_upper=upper)
    return derive_c6_carried_tube(
        derive_c6_carried_profile(lattice), state=state, timestep=timestep
    )


def test_guaranteed_passage_is_observed_before_the_derived_deadline():
    result = derive_c6_carried_positive_pressure_passage(_tube(), node=0)
    assert result.sign_hit_certified and result.band_covers_contradiction
    assert result.initial_pressure > 0 and result.centered_increment_lower > 0
    assert result.latest_pressure_index == result.contradiction_steps - 1
    state = result.tube.state
    for index in range(result.latest_pressure_index + 1):
        reading = observe_c6_pressure_lattice(
            result.tube.contraction.profile.lattice, epi=state.epi
        )
        assert reading.mean_pressure <= result.mean_pressure_upper
        assert all(
            abs(value) <= limit
            for value, limit in zip(
                reading.epi_reduction_error, result.product_error_bounds, strict=True
            )
        )
        assert all(
            abs(value) <= limit
            for value, limit in zip(
                reading.assembly_error, result.assembly_error_bounds, strict=True
            )
        )
        if reading.pressure[0] <= 0:
            break
        state = advance_nodal_remainder(
            state,
            timestep=result.tube.contraction.timestep,
            capacity=(1.0,) * 6,
            pressure=reading.pressure,
        ).after
    else:
        pytest.fail("the canonical carried map failed its finite passage theorem")
    assert index > 0
    assert not result.graph_provenance_certified
    assert not result.future_runtime_certified
    assert not result.infinite_trapping_certified


def test_contradiction_horizon_uses_strict_endpoint_inequality_and_is_minimal():
    result = derive_c6_carried_positive_pressure_passage(_tube(), node=0)
    n = result.contradiction_steps
    y0, rate = result.tube.initial_error[0], result.centered_increment_lower
    squared = result.centered_coordinate_squared_bound
    assert y0 + n * rate > 0 and (y0 + n * rate) ** 2 > squared
    assert y0 + (n - 1) * rate <= 0 or (y0 + (n - 1) * rate) ** 2 <= squared
    assert (
        result.minimum_positive_index == result.positive_sector.bounding_gradient_index
    )
    assert (
        result.minimum_positive_pressure
        == result.positive_sector.signed_pressure_margin
    )


@pytest.mark.parametrize("node", (True, 1.0, -1, 6, "0"))
def test_node_must_be_a_literal_ordered_cycle_index(node):
    with pytest.raises((TypeError, ValueError), match="C6 index"):
        derive_c6_carried_positive_pressure_passage(_tube(), node=node)


def test_forged_tube_caches_cannot_create_a_stronger_deadline():
    original = _tube()
    baseline = derive_c6_carried_positive_pressure_passage(original, node=0)
    forged = replace(
        original,
        energy_bound=F(0),
        initial_error=(F(0),) * 6,
        carry_bound=F(0),
        mean_increment_bound=F(0),
    )
    result = derive_c6_carried_positive_pressure_passage(forged, node=0)
    assert result == baseline


@pytest.mark.parametrize("node", (0, 1, 4))
def test_initial_nonpositive_pressure_is_a_zero_step_observation(node):
    result = derive_c6_carried_positive_pressure_passage(
        _tube(epi=(0.5,) * 6), node=node
    )
    assert result.initial_pressure == 0
    assert (
        result.sign_hit_certified
        and result.latest_pressure_index == result.contradiction_steps == 0
    )
    assert result.abstention_reason is None


def test_narrow_band_cannot_promote_a_conditional_contradiction_to_a_sign_hit():
    upper = math.nextafter(0.5, math.inf)
    tube = _tube(epi=(upper,) + (0.5,) * 5, lower=0.5, upper=upper)
    result = derive_c6_carried_positive_pressure_passage(tube, node=1)
    assert result.initial_pressure > 0 and result.centered_increment_lower > 0
    assert result.contradiction_steps > 0
    assert not result.band_horizon.tube_initially_admitted
    assert not result.band_covers_contradiction and not result.sign_hit_certified
    assert (
        result.latest_pressure_index is None
        and "band horizon" in result.abstention_reason
    )


def test_large_source_profile_can_prevent_a_strict_centered_increment_bound():
    tube = _tube(
        phase=(0.0, 0.25, 0.0, 0.0, 0.0, 0.0),
        weight=1e-300,
        epi=(0.5, 0.5, 0.5, 0.625, 0.375, 0.5),
        timestep=1 / 16,
    )
    result = derive_c6_carried_positive_pressure_passage(tube, node=4)
    assert result.initial_pressure > 0 and result.centered_increment_lower < 0
    assert not result.sign_hit_certified and result.contradiction_steps is None
    assert "strictly positive" in result.abstention_reason


def test_empty_positive_sector_does_not_invent_an_extreme_gradient():
    tiny = math.ulp(0.0)
    result = derive_c6_carried_positive_pressure_passage(
        _tube(epi=(0.5,) * 6, weight=tiny), node=0
    )
    assert result.positive_sector.empty
    assert result.minimum_positive_index is result.minimum_positive_pressure is None
    assert result.centered_increment_lower is None
    assert result.sign_hit_certified and result.latest_pressure_index == 0


def test_winding_source_obtains_finite_passage_without_an_observed_error_fit():
    phase = (float.fromhex("0x1.0b8fb3e3956cbp-55"),) + tuple(
        i * math.pi / 3 for i in range(1, 6)
    )
    delta = F(1, 2**54)
    epi = tuple(float(F(1, 2) + value * delta) for value in (-2, 0, 2, -2, 4, -4))
    tube = _tube(
        epi=epi,
        phase=phase,
        weight=float(F(3299399280161697, 2**54)),
        phase_weight=float(F(854047988748571, 2**50)),
        timestep=1 / 16,
    )
    result = derive_c6_carried_positive_pressure_passage(tube, node=4)
    assert result.sign_hit_certified
    assert result.minimum_positive_index == -21
    assert result.minimum_positive_pressure == F(38338268585241, 2**106)
    assert result.mean_pressure_upper < F(1, 2**100)
    assert 10000 < result.contradiction_steps < 50000
    assert result.contradiction_steps < result.band_horizon.maximum_steps
