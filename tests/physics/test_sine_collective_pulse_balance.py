"""Actual collective feedback, declared lifts and shared source admission."""

from copy import copy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import pytest

from tests.physics.test_sine_phase_offset_partition import CYCLE_LEAF_EDGES, _source
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_partition import observe_sine_collective_pulse
from tnfr.sdk.relational_reports import relational_report_to_dict

RELATIVE_PHASE_FIELDS = (
    "receiver_internal_phase_rates",
    "receiver_contact_phase_rates",
    "receiver_relative_phase_rates",
    "receiver_relative_phase_acceleration_bounds",
    "receiver_relative_phase_jerk_bounds",
)


@pytest.fixture(scope="module")
def source():
    u, epsilon = Q(119, 100), Q(1, 20)
    forms = (u / 4 + epsilon, u / 4 - epsilon) + (u / 4,) * 3 + (-3 * u / 4,) * 5
    return replace(_source(CYCLE_LEAF_EDGES), epi=forms)


def _observe(source, **changes):
    arguments = dict(cycle=tuple(range(5)), contact_turn_offsets=(0,) * 5)
    return observe_sine_collective_pulse(source, **(arguments | changes))


def test_candidate_exact_flat_phase_jet_and_energy_partition(source):
    report = _observe(source)
    u = Q(119, 100)
    assert report.form_gap == u
    assert report.receiver_form_mean == u / 4
    assert report.environment_form_mean == -3 * u / 4
    assert report.contact_phase_bounds == (I(0),) * 5
    assert report.contact_resultant_real_bounds.contains(1)
    assert report.contact_resultant_real_bounds.width < Q(1, 10**30)
    assert report.contact_resultant_imag_bounds == I(0)
    assert report.mean_contact_phase_rate == -4 * u / 3
    assert report.contact_rate_deviations == (
        -Q(7, 60),
        Q(7, 60),
        -Q(1, 60),
        0,
        Q(1, 60),
    )
    assert report.contact_rate_variance == Q(1, 180)
    assert report.contact_rate_third_central_moment == 0
    assert report.flat_phase_jet_available
    assert report.first_three_energy_derivatives == (0, 0, 0)
    assert report.fourth_energy_derivative == 4 * u**2 / 135 > 0
    assert report.collective_energy_bounds.contains(u**2 / 2)
    assert report.collective_energy_rate_bounds == I(0)
    assert report.feedback_energy_rate_bounds == I(0)
    assert report.remainder_storage_rate_bounds == I(0)
    assert report.full_storage_bounds.contains(Q(14201, 4000))
    assert report.remainder_storage_bounds.contains(Q(1, 100))
    assert report.full_storage_bounds.width < Q(1, 10**30)
    assert report.clock == "tau=t/pi"
    with pytest.raises(FrozenInstanceError):
        report.form_gap = 0


def test_collective_energy_is_lift_dependent_and_remainder_need_not_be_positive(source):
    equilibrium = replace(source, epi=(Q(0),) * 10)
    canonical = _observe(equilibrium)
    different_lift = _observe(equilibrium, contact_turn_offsets=(-1, 0, 0, 0, 0))
    assert canonical.full_storage_bounds == different_lift.full_storage_bounds
    assert canonical.collective_energy_bounds == I(0)
    assert different_lift.collective_energy_bounds.lo > 0
    assert different_lift.remainder_storage_bounds.hi < 0
    assert different_lift.contact_phase_bounds[0].lo > 6
    assert different_lift.mean_contact_sine_bounds == I(0)
    assert not different_lift.flat_phase_jet_available
    assert different_lift.first_three_energy_derivatives is None
    assert different_lift.fourth_energy_derivative is None


def test_exact_flat_phase_availability_is_not_an_interval_zero_overlap(source):
    common = _observe(replace(source, phase=(Q(17),) * 10))
    original = _observe(source)
    assert common.fourth_energy_derivative == original.fourth_energy_derivative
    assert common.flat_phase_jet_available
    tiny = replace(source, phase=(Q(1, 10**100),) + (Q(0),) * 9)
    report = _observe(tiny)
    assert not report.flat_phase_jet_available
    assert report.fourth_energy_derivative is None
    shifted_contacts = _observe(source, contact_turn_offsets=(1,) * 5)
    assert (
        shifted_contacts.collective_energy_bounds == original.collective_energy_bounds
    )
    assert not shifted_contacts.flat_phase_jet_available


def test_derived_source_cache_is_ignored_and_node_order_is_preserved(source):
    ordinary = _observe(source)
    poisoned = replace(
        source,
        form_gradient=(Q(999),),
        phase_rates=(I(700),),
        form_rates=(),
        storage=I(-10),
        relative_resultant=(),
    )
    actual = _observe(poisoned)
    assert actual.full_storage_bounds == ordinary.full_storage_bounds
    assert actual.contact_phase_rates == ordinary.contact_phase_rates
    assert actual.fourth_energy_derivative == ordinary.fourth_energy_derivative
    for field in RELATIVE_PHASE_FIELDS:
        assert getattr(actual, field) == getattr(ordinary, field)
    order = (8, 0, 5, 3, 6, 9, 1, 7, 4, 2)
    reordered = _source(CYCLE_LEAF_EDGES, order=order, label=lambda i: f"n:{i}")
    reordered = replace(reordered, epi=tuple(source.epi[i] for i in order))
    result = observe_sine_collective_pulse(
        reordered,
        cycle=tuple(f"n:{i}" for i in range(5)),
        contact_turn_offsets=(0,) * 5,
    )
    assert result.cycle_indices == tuple(order.index(i) for i in range(5))
    assert result.contact_phase_rates == ordinary.contact_phase_rates
    assert result.full_storage_bounds == ordinary.full_storage_bounds
    for field in RELATIVE_PHASE_FIELDS:
        assert getattr(result, field) == getattr(ordinary, field)


def test_receiver_relative_phase_derivatives_preserve_origins_and_contact_lifts(source):
    source = replace(source, phase=tuple(Q(i * i - 3 * i, 29) for i in range(10)))
    ordinary = _observe(source)
    shifted = _observe(
        replace(
            source,
            epi=tuple(value + Q(17, 11) for value in source.epi),
            phase=tuple(value - Q(13, 7) for value in source.phase),
        )
    )
    different_lifts = _observe(source, contact_turn_offsets=(1, -2, 0, 3, -1))
    for field in RELATIVE_PHASE_FIELDS:
        assert getattr(shifted, field) == getattr(ordinary, field)
        assert getattr(different_lifts, field) == getattr(ordinary, field)
    assert different_lifts.collective_energy_bounds != ordinary.collective_energy_bounds


def test_sdk_exports_exact_relative_phase_rates_and_interval_derivatives(source):
    report = _observe(source)
    exported = relational_report_to_dict(report)
    assert exported["report_type"] == "SineCollectivePulseBalance"
    body = exported["report"]
    for field in RELATIVE_PHASE_FIELDS:
        assert len(body[field]) == 5
    for encoded, exact in zip(
        body["receiver_relative_phase_rates"], report.receiver_relative_phase_rates
    ):
        assert encoded == {
            "numerator": exact.numerator,
            "denominator": exact.denominator,
        }
    assert body["clock"] == "tau=t/pi"
    assert report.to_dict()["schema"] == "tnfr.sine-collective-pulse-balance.v1"


@pytest.mark.parametrize(
    "offsets",
    ((0,) * 4, (0,) * 6, (False,) + (0,) * 4, (Q(0),) * 5, (0.0,) * 5, {0, 1}),
)
def test_declared_contact_lifts_require_ordered_exact_integer_offsets(source, offsets):
    with pytest.raises((TypeError, ValueError)):
        _observe(source, contact_turn_offsets=offsets)


@pytest.mark.parametrize("field", ("epi", "phase", "capacity"))
def test_all_source_primitives_are_readmitted(source, field):
    invalid = replace(source, **{field: (False,) + getattr(source, field)[1:]})
    with pytest.raises((TypeError, ValueError)):
        _observe(invalid)


@pytest.mark.parametrize("field", ("epi_weight", "phase_weight", "storage_scale"))
def test_model_booleans_cannot_supply_conservative_unit_coefficients(source, field):
    model = copy(source.reference_model)
    object.__setattr__(model, field, False if field == "epi_weight" else True)
    with pytest.raises((TypeError, ValueError)):
        _observe(replace(source, reference_model=model))


@pytest.mark.parametrize(
    "cycle", ((0, 1, 2, 3), (0, 2, 1, 3, 4), (0, 1, 2, 3, 99), {0, 1, 2, 3, 4})
)
def test_cycle_traversal_and_full_support_are_explicit(source, cycle):
    with pytest.raises((TypeError, ValueError)):
        _observe(source, cycle=cycle)


def test_unsupported_source_law_or_capacity_cannot_use_the_observer(source):
    with pytest.raises(TypeError):
        _observe(object())
    with pytest.raises(ValueError):
        _observe(replace(source, law="native"))
    with pytest.raises(ValueError):
        _observe(replace(source, capacity=(Q(0),) + (Q(1),) * 9))
    model = copy(source.reference_model)
    object.__setattr__(model, "epi_weight", Q(1, 2))
    with pytest.raises(ValueError):
        _observe(replace(source, reference_model=model))
