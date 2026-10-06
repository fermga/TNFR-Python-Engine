"""Admission and integrity of full-state moving-pattern Bregman windows."""

from copy import copy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import pytest

from tests.physics.test_sine_phase_offset_partition import (
    BLOCKS,
    CYCLE_LEAF_EDGES,
    OFFSETS,
    _source,
)
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_partition import (
    assess_sine_moving_pattern_window,
    assess_sine_phase_offset_partition,
)


@pytest.fixture(scope="module")
def family():
    return assess_sine_phase_offset_partition(
        _source(CYCLE_LEAF_EDGES), blocks=BLOCKS, phase_offset_turns=OFFSETS
    )


@pytest.fixture(scope="module")
def reference(family):
    return family.evaluate((Q(1, 20), -Q(3, 20)), (-Q(2, 5),) * 2)


def _assess(reference, **changes):
    arguments = dict(
        form_error_bounds=(Q(1, 1000),) * 10,
        phase_error_bounds=(Q(1, 1000),) * 10,
        scaled_horizon=5,
        receiver_phase_radius=Q(1, 10),
        contact_phase_radius=Q(1, 10),
        reference_contact_bound=Q(1, 4),
    )
    return assess_sine_moving_pattern_window(reference, **(arguments | changes))


def test_finite_window_retains_full_relative_state_and_actual_energy(reference):
    report = _assess(reference)
    assert report.status == "certified"
    assert report.reasons == ()
    assert report.whole_window_retention_certified
    assert report.initial_chart_certified
    assert report.reference_contact_envelope_certified
    assert report.reference_libration_energy_bounds == I(Q(1, 50))
    assert report.reference_form_gap == Q(1, 5)
    assert report.reference_contact_turn == 0
    assert report.initial_relative_storage_upper_bound == Q(1, 25000)
    assert (
        0
        < report.propagated_relative_storage_upper_bound
        < report.relative_storage_barrier
    )
    assert report.retention_margin_lower_bound > 0
    assert report.receiver_phase_error_upper_bound < report.receiver_phase_radius
    assert report.contact_phase_error_upper_bound < report.contact_phase_radius
    assert (
        report.receiver_storage_excess_upper_bound
        == report.propagated_relative_storage_upper_bound
    )
    assert (
        report.form_mean_error_upper_bound
        == report.phase_mean_error_upper_bound
        == Q(1, 1000)
    )
    assert report.initial_full_storage_bounds.lo < reference.full_storage_bounds.lo
    assert report.initial_full_storage_bounds.hi > reference.full_storage_bounds.hi
    assert report.initial_full_storage_bounds.lo > Q(7, 2)
    assert report.phase_flat_acquisition_status == "not_excluded"
    assert report.zero_winding_acquisition_status == "not_excluded"
    with pytest.raises(FrozenInstanceError):
        report.status = "unavailable"


def test_stationary_reference_retention_and_phase_flat_origin_are_separate(family):
    reference = family.evaluate((0, 0), (-Q(2, 5),) * 2)
    report = _assess(reference)
    assert report.whole_window_retention_certified
    assert report.growth_rate_upper_bound == 0
    assert report.growth_factor_upper_bound == 1
    assert (
        report.propagated_relative_storage_upper_bound
        == report.initial_relative_storage_upper_bound
    )
    assert report.initial_full_storage_bounds.hi < Q(7, 2)
    assert report.phase_flat_acquisition_status == "excluded"
    assert report.zero_winding_acquisition_status == "excluded"
    assert report.to_dict()["report"]["zero_winding_acquisition_status"] == "excluded"


def test_zero_error_bypasses_exponential_overflow_but_keeps_reference_guard(reference):
    report = _assess(
        reference,
        form_error_bounds=(0,) * 10,
        phase_error_bounds=(0,) * 10,
        scaled_horizon=10**8,
    )
    assert report.whole_window_retention_certified
    assert report.propagated_relative_storage_upper_bound == 0
    assert report.growth_factor_upper_bound is None
    assert report.edge_form_error_upper_bound == 0
    assert report.receiver_phase_error_upper_bound == 0
    assert report.contact_phase_error_upper_bound == 0
    unsupported = reference.partition.evaluate((1, 0), (-Q(2, 5),) * 2)
    failed = _assess(
        unsupported, form_error_bounds=(0,) * 10, phase_error_bounds=(0,) * 10
    )
    assert failed.status == "unavailable"
    assert not failed.reference_contact_envelope_certified
    assert failed.edge_form_error_upper_bound is None


def test_exponential_overflow_is_explicit_unavailability(reference):
    report = _assess(reference, scaled_horizon=10**8)
    assert report.status == "unavailable"
    assert report.growth_factor_upper_bound is None
    assert report.propagated_relative_storage_upper_bound is None
    assert report.receiver_storage_excess_upper_bound is None
    assert "finite_relative_storage_growth_bound_unavailable" in report.reasons


@pytest.mark.parametrize(
    "changes",
    (
        {"receiver_phase_radius": Q(2, 5)},
        {"contact_phase_radius": 2},
        {"phase_error_bounds": (Q(3, 50),) * 10},
        {"form_error_bounds": (1,) * 10},
    ),
)
def test_failed_sufficient_chart_or_storage_budget_is_not_a_dynamical_exclusion(
    reference, changes
):
    report = _assess(reference, **changes)
    assert report.status == "unavailable"
    assert not report.whole_window_retention_certified
    assert report.edge_form_error_upper_bound is None
    assert report.receiver_storage_excess_upper_bound is None
    assert report.reasons


def test_initial_contact_lift_is_checked_in_addition_to_storage(family):
    outside = family.evaluate((0, 0), (0, Q(1, 16)))
    report = _assess(outside)
    assert report.reference_contact_initial_margin_lower_bound < 0
    assert not report.reference_contact_envelope_certified
    assert report.status == "unavailable"
    wrapped = family.evaluate((Q(1, 20), -Q(3, 20)), (1000, -1000))
    equivalent = family.evaluate((Q(1, 20), -Q(3, 20)), (0, 0))
    assert (
        _assess(wrapped).reference_libration_energy_bounds
        == _assess(equivalent).reference_libration_energy_bounds
    )
    assert (
        _assess(wrapped).initial_full_storage_bounds
        == _assess(equivalent).initial_full_storage_bounds
    )


def test_primitive_box_energy_remains_valid_when_the_acute_chart_fails(family):
    reference = family.evaluate((0, 0), (0, 0))
    report = _assess(
        reference, form_error_bounds=(0,) * 10, phase_error_bounds=(10,) * 10
    )
    assert not report.initial_chart_certified
    # This declared phase box contains full phase consensus, whose storage0
    # refutes using the reference minimum as an unconditional box lower bound.
    assert report.initial_full_storage_bounds.contains(0)
    assert report.phase_flat_acquisition_status == "not_excluded"


def test_reference_and_partition_cached_fields_do_not_supply_window_evidence(reference):
    original = _assess(reference)
    fake_partition = replace(
        reference.partition,
        status="excluded",
        invariance_certified=False,
        quotient_counts=None,
        node_blocks=(),
    )
    fake = replace(
        reference,
        partition=fake_partition,
        fine_form=(Q(9000),),
        fine_phase_turns=(False,),
        full_storage_bounds=I(-999),
        block_phase_rates=(Q(123),),
    )
    rebuilt = _assess(fake)
    assert rebuilt.reference.fine_form == reference.fine_form
    assert rebuilt.reference.full_storage_bounds == reference.full_storage_bounds
    assert (
        rebuilt.propagated_relative_storage_upper_bound
        == original.propagated_relative_storage_upper_bound
    )
    assert rebuilt.initial_full_storage_bounds == original.initial_full_storage_bounds
    assert rebuilt.whole_window_retention_certified


@pytest.mark.parametrize("field", ("epi_weight", "phase_weight", "storage_scale"))
def test_authoritative_model_boolean_is_rejected_before_cached_reference_use(
    reference, field
):
    model = copy(reference.partition.source.reference_model)
    object.__setattr__(model, field, False if field == "epi_weight" else True)
    source = replace(reference.partition.source, reference_model=model)
    fake = replace(reference, partition=replace(reference.partition, source=source))
    with pytest.raises((TypeError, ValueError)):
        _assess(fake)


@pytest.mark.parametrize(
    "name",
    (
        "scaled_horizon",
        "receiver_phase_radius",
        "contact_phase_radius",
        "reference_contact_bound",
    ),
)
@pytest.mark.parametrize("value", (False, 0, -1, float("inf")))
def test_strict_scalar_parameter_admission(reference, name, value):
    with pytest.raises((TypeError, ValueError)):
        _assess(reference, **{name: value})


@pytest.mark.parametrize("name", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize(
    "value", ((0,) * 9, (False,) + (0,) * 9, (-1,) + (0,) * 9, {0, 1})
)
def test_full_ordered_error_vectors_are_required(reference, name, value):
    with pytest.raises((TypeError, ValueError)):
        _assess(reference, **{name: value})


def test_ordered_cycle_and_exact_winding_domain_are_not_inferred(family):
    wrong_order = replace(family, blocks=((0, 2, 1, 3, 4), BLOCKS[1])).evaluate(
        (0, 0), (0, 0)
    )
    with pytest.raises(ValueError, match="ordered C5"):
        _assess(wrong_order)
    flat = replace(family, phase_offset_turns=(0,) * 10).evaluate((0, 0), (0, 0))
    with pytest.raises(ValueError, match="fifth turn"):
        _assess(flat)
    larger_degree_source = _source(
        tuple((i, j) for i in range(5) for j in range(i + 1, 5))
        + tuple((i, i + 5) for i in range(5))
    )
    wrong_support = assess_sine_phase_offset_partition(
        larger_degree_source, blocks=BLOCKS, phase_offset_turns=OFFSETS
    ).evaluate((0, 0), (0, 0))
    with pytest.raises(ValueError, match="degrees"):
        _assess(wrong_support)
    with pytest.raises(TypeError):
        _assess(object())


def test_reversing_winding_preserves_window_but_records_orientation(family):
    positive = _assess(family.evaluate((Q(1, 20), -Q(3, 20)), (0, 0)))
    reversed_family = replace(
        family, phase_offset_turns=tuple(-value for value in OFFSETS)
    )
    negative = _assess(reversed_family.evaluate((Q(1, 20), -Q(3, 20)), (0, 0)))
    assert positive.orientation == 1 and negative.orientation == -1
    assert (
        positive.propagated_relative_storage_upper_bound
        == negative.propagated_relative_storage_upper_bound
    )
    assert positive.initial_full_storage_bounds == negative.initial_full_storage_bounds
    assert negative.whole_window_retention_certified
