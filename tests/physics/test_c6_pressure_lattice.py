"""Independent exact C6 sign, source-binding and Cartesian-scope checks."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F
import math

import numpy as np
import pytest

from tnfr.constants.canonical import CHANNEL_WEIGHT_PRIMARY, CHANNEL_WEIGHT_SECONDARY
from tnfr.dynamics._euler_kernel import NodalRemainderState, advance_nodal_remainder
from tnfr.mathematics._neighbor_differences import edge_mean_differences, mean_neighbor_difference
from tnfr.physics import c6_pressure_lattice as owner


PHASE = tuple(map(float.fromhex, (
    "0x1.0b8fb3e3956cbp-55", "0x1.0c152382d7365p+0", "0x1.0c152382d7365p+1",
    "0x1.921fb54442d18p+1", "0x1.0c152382d7365p+2", "0x1.4f1a6c638d03fp+2",
)))
DELTA = F(1, 2**54)
Q = DELTA / 2


def _derive(**changes):
    inputs = dict(phase=PHASE, epi_weight=CHANNEL_WEIGHT_SECONDARY,
                  phase_weight=CHANNEL_WEIGHT_PRIMARY)
    inputs.update(changes)
    return owner.derive_c6_pressure_lattice(**inputs)


@pytest.fixture(scope="module")
def reference():
    return _derive()


def _cell(value):
    lower = (F(math.nextafter(value, -math.inf)) + F(value)) / 2
    upper = (F(math.nextafter(value, math.inf)) + F(value)) / 2
    index = F(value) / F(math.ulp(value))
    assert index.denominator == 1
    return lower, upper, index.numerator % 2 == 0


def test_closed_phase_has_strict_joint_sign_obstruction_with_scoped_conclusion(reference):
    assert reference.source.phase == PHASE
    assert (reference.source.epi_lower, reference.source.epi_upper) == (.375, .625)
    assert reference.epi_quantum == DELTA and reference.gradient_quantum == Q
    assert tuple(row.nonnegative_min_index for row in reference.rows) == (2, 0, -5, 11, -21, 15)
    assert tuple(row.nonpositive_max_index for row in reference.rows) == (1, -1, -6, 10, -22, 14)
    assert reference.nonnegative_min_sum == 2
    assert reference.nonpositive_max_sum == -4
    assert reference.every_pressure_has_positive and reference.every_pressure_has_negative
    assert reference.no_cartesian_trap
    assert reference.positive_band_exit_certified is False
    assert sum(map(F, reference.sources)) == -F(1, 2**109)
    with pytest.raises(FrozenInstanceError):
        reference.no_cartesian_trap = False


def test_thresholds_use_exact_cell_parity_and_actual_source_reducer(reference):
    parities = set()
    for row, source in zip(reference.rows, reference.sources, strict=True):
        lower, upper, closed = _cell(-source)
        parities.add(closed)
        scaled_lower = lower / F(CHANNEL_WEIGHT_SECONDARY) / Q
        scaled_upper = upper / F(CHANNEL_WEIGHT_SECONDARY) / Q
        minimum = math.ceil(scaled_lower) + int(not closed and scaled_lower.denominator == 1)
        maximum = math.floor(scaled_upper) - int(not closed and scaled_upper.denominator == 1)
        assert row.nonnegative_min_index == minimum
        assert row.nonpositive_max_index == maximum
        # Realize these local integer differences far from either binade edge.

        def pressure(index):
            center = .4375
            neighbor = float(F(center) + index * DELTA)
            linear = mean_neighbor_difference(
                center, (neighbor, center), coefficient=CHANNEL_WEIGHT_SECONDARY,
            )
            return float(F(source) + F(linear))

        assert pressure(minimum - 1) < 0 <= pressure(minimum)
        assert pressure(maximum) <= 0 < pressure(maximum + 1)
    assert parities == {False, True}


@pytest.mark.parametrize("epi", (
    (.5,) * 6,
    (.375, .625, .375, .625, .375, .625),
    (.625, .375, .625, .375, .625, .375),
    (.375, math.nextafter(.375, math.inf), .5, math.nextafter(.5, math.inf), .625, .4375),
    (.5, math.nextafter(.5, 0.), math.nextafter(.5, math.inf), .5, .4375, .5625),
))
def test_actual_scalar_vector_and_fused_pressure_use_the_exact_compatible_lattice(reference, epi):
    observed = owner.observe_c6_pressure_lattice(reference, epi=epi)
    indices = tuple(int((F(value) - F(.375)) / DELTA) for value in epi)
    integer_gradient = tuple(indices[i - 1] + indices[(i + 1) % 6] - 2 * indices[i] for i in range(6))
    gradient = tuple((F(epi[i - 1]) + F(epi[(i + 1) % 6])) / 2 - F(epi[i]) for i in range(6))
    assert observed.epi_indices == indices
    assert observed.gradient_indices == integer_gradient
    assert observed.gradient_index_sum == sum(integer_gradient) == 0
    assert observed.gradient == gradient == tuple(value * Q for value in integer_gradient)
    expected_epi = tuple(float(F(CHANNEL_WEIGHT_SECONDARY) * value) for value in gradient)
    expected_pressure = tuple(float(F(a) + F(g)) for a, g in zip(reference.sources, expected_epi, strict=True))
    assert observed.epi_contributions == expected_epi
    assert observed.pressure == expected_pressure
    assert any(value > 0 for value in observed.pressure) and any(value < 0 for value in observed.pressure)
    reduction = tuple(F(g) - F(CHANNEL_WEIGHT_SECONDARY) * q for g, q in zip(expected_epi, gradient, strict=True))
    assembly = tuple(F(p) - F(a) - F(g) for p, a, g in zip(expected_pressure, reference.sources, expected_epi, strict=True))
    assert observed.epi_reduction_error == reduction
    assert observed.assembly_error == assembly
    assert observed.mean_pressure == sum(map(F, expected_pressure)) / 6
    assert observed.mean_epi_reduction_error == sum(reduction) / 6
    assert observed.mean_assembly_error == sum(assembly) / 6
    assert observed.mean_identity_residual == 0
    assert observed.mean_pressure == (observed.mean_phase_contribution + observed.mean_epi_reduction_error
                                      + observed.mean_assembly_error)
    assert observed.scalar_reducer_agreement and observed.vector_reducer_agreement and observed.fused_pressure_agreement


@pytest.mark.parametrize("center,neighbors", (
    (.375, (.625, .625)), (.625, (.375, .375)),
    (.5, (math.nextafter(.5, 0.), math.nextafter(.5, math.inf))),
    (.5, (math.nextafter(.5, 0.), .5)),
    (.375, (.375, math.nextafter(.375, math.inf))),
))
def test_both_production_reducers_preserve_exact_half_lattice_across_binades(center, neighbors):
    gradient = sum(map(F, neighbors)) / 2 - F(center)
    assert (gradient / Q).denominator == 1
    scalar = mean_neighbor_difference(center, neighbors, coefficient=1.)
    vector = edge_mean_differences(np.asarray((center, *neighbors)), np.asarray((0, 0)), np.asarray((1, 2)))
    assert F(scalar) == F(float(vector[0])) == gradient


def test_slab_boundary_attains_largest_proven_integer_gradient(reference):
    observed = owner.observe_c6_pressure_lattice(reference, epi=(.375, .625) * 3)
    assert observed.gradient_indices == (2**53, -2**53) * 3
    assert observed.epi_indices == (0, 2**52) * 3


def test_actual_zero_mean_pressure_is_not_zero_pressure_or_a_trapping_certificate(reference):
    observed = owner.observe_c6_pressure_lattice(reference, epi=(math.nextafter(.5, 0.), .5, .5, .5, .5, .5))
    assert observed.mean_pressure == 0
    assert any(observed.pressure)
    assert observed.mean_epi_reduction_error + observed.mean_assembly_error == -observed.mean_phase_contribution
    assert observed.reference.no_cartesian_trap


@pytest.mark.parametrize("sign", (-1, 1))
def test_cartesian_corner_obstruction_uses_exact_carried_coordinates_not_visible_cell_exit(reference, sign):
    radius = F(1, 2**60)
    state = NodalRemainderState((.5,) * 6, (sign * radius,) * 6, .05, 1.)
    corner = state.exact_epi
    pressure = owner.observe_c6_pressure_lattice(reference, epi=state.epi).pressure
    step = advance_nodal_remainder(state, timestep=.0625, capacity=(1.,) * 6, pressure=pressure)
    assert any(sign * (after - before) > 0 for before, after in zip(corner, step.after.exact_epi, strict=True))
    # This witness leaves the chosen coordinate product despite staying in
    # the displayed rounding cell and the much larger positive EPI band.
    assert step.after.epi == state.epi
    assert all(F(.05) <= value <= 1 for value in step.after.exact_epi)


def test_zero_phase_control_permits_the_uniform_singleton_product():
    reference = _derive(phase=(0.,) * 6)
    assert reference.nonnegative_min_sum == reference.nonpositive_max_sum == 0
    assert not reference.every_pressure_has_positive and not reference.every_pressure_has_negative
    assert not reference.no_cartesian_trap
    observed = owner.observe_c6_pressure_lattice(reference, epi=(.5,) * 6)
    assert observed.pressure == (0.,) * 6
    state = NodalRemainderState((.5,) * 6, (F(0),) * 6, .05, 1.)
    assert advance_nodal_remainder(state, timestep=.0625, capacity=(1.,) * 6, pressure=observed.pressure).after == state


@pytest.mark.parametrize("lower,upper", ((.5, 1.), (.0625, .125), (.5, .5), (.4375, .5625)))
def test_other_proven_slabs_rederive_their_own_quantum(lower, upper):
    reference = _derive(epi_lower=lower, epi_upper=upper)
    assert reference.epi_quantum == F(math.ulp(lower))
    assert reference.gradient_quantum == F(math.ulp(lower)) / 2
    observed = owner.observe_c6_pressure_lattice(reference, epi=(lower, upper) * 3)
    assert observed.gradient_index_sum == 0
    assert max(map(abs, observed.gradient_indices)) <= 2**53


@pytest.mark.parametrize("changes,error", (
    ({"epi_upper": math.nextafter(.625, math.inf)}, ValueError),
    ({"epi_lower": .05, "epi_upper": 1.}, ValueError),
    ({"epi_lower": .04}, ValueError), ({"epi_upper": .25}, ValueError),
    ({"epi_lower": F(3, 8)}, TypeError), ({"epi_weight": 0.}, ValueError),
    ({"phase": (math.nan, *PHASE[1:])}, ValueError),
))
def test_invalid_slab_and_primitive_source_inputs_are_rejected(changes, error):
    with pytest.raises(error):
        _derive(**changes)


@pytest.mark.parametrize("epi,error", (
    ([.5] * 6, TypeError), ((.5,) * 5, ValueError),
    ((True, .5, .5, .5, .5, .5), TypeError),
    ((math.nan, .5, .5, .5, .5, .5), ValueError),
    ((math.nextafter(.375, 0.), .5, .5, .5, .5, .5), ValueError),
    ((math.nextafter(.625, math.inf), .5, .5, .5, .5, .5), ValueError),
))
def test_observed_epi_must_belong_to_the_six_coordinate_slab(reference, epi, error):
    with pytest.raises(error):
        owner.observe_c6_pressure_lattice(reference, epi=epi)


def test_observer_rebuilds_all_outer_and_nested_derived_caches(reference):
    corrupted_source = replace(reference.source, rows=(), gradient_quantum=F(1), no_zero_pressure=False)
    forged = replace(reference, source=corrupted_source, sources=(0.,) * 6, epi_quantum=F(1),
                     gradient_quantum=F(1), rows=(), nonnegative_min_sum=-1,
                     nonpositive_max_sum=1, every_pressure_has_positive=False,
                     every_pressure_has_negative=False, no_cartesian_trap=False)
    observed = owner.observe_c6_pressure_lattice(forged, epi=(.5,) * 6)
    assert observed.reference == reference
    assert observed.pressure == reference.sources


@pytest.mark.parametrize("changes,error", (
    ({"epi_weight": F(1, 3)}, ValueError), ({"epi_weight": .5}, TypeError),
    ({"phase_weight": F(0)}, ValueError), ({"epi_lower": .05, "epi_upper": 1.}, ValueError),
))
def test_observer_revalidates_primitive_fields_of_forged_sources(reference, changes, error):
    forged = replace(reference, source=replace(reference.source, **changes))
    with pytest.raises(error):
        owner.observe_c6_pressure_lattice(forged, epi=(.5,) * 6)


def test_shared_reducer_disagreement_cannot_be_published_as_a_binding(reference, monkeypatch):
    monkeypatch.setattr(owner, "mean_neighbor_difference", lambda *args, **kwargs: 1.)
    with pytest.raises(RuntimeError, match="reducer"):
        owner.observe_c6_pressure_lattice(reference, epi=(.5,) * 6)
