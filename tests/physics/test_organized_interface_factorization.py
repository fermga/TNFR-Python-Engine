"""Finite-rank semantics and source/support gating on synthetic test histories.

The reserved resource corner is evaluated only by its frozen instrument.
This file rebuilds the earlier ORIGINAL source budget for dependency controls.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import product

import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import I
from tnfr.physics import _organized_interface_factorization as owner
from tnfr.physics._organized_joint_resources import _assess_organized_joint_resources


@pytest.fixture(scope="module", autouse=True)
def no_scientific_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def original_source():
    return _assess_organized_joint_resources(
        form_error_bounds=(Q(1, 10**10),) * 6,
        phase_error_bounds=(Q(1, 10**10),) * 6,
        quarter_dwell=Q(2 * 10**13),
        recovered_radius=Q(1, 10**70),
        decay_power=512,
        sequential_readout_error=Q(1, 10**15),
        contact_readout_error=Q(1, 10**10),
    )


@pytest.fixture
def assess(original_source, monkeypatch):
    def run(source=original_source):
        monkeypatch.setattr(
            owner, "_assess_organized_joint_resources", lambda **kw: source
        )
        return owner._assess_organized_interface_factorization()

    return run


def test_interval_determinant_contains_all_independent_rectangle_corners():
    matrix = ((I(-2, -1), I(3, 4)), (I(5, 7), I(-3, -2)))
    bound = owner._determinant(matrix)
    for a, b, c, d in product(*((v.lo, v.hi) for row in matrix for v in row)):
        assert bound.lo <= a * d - b * c <= bound.hi
    assert owner._zero_exclusion_margin(bound) > 0
    assert owner._zero_exclusion_margin(I(-1, 1)) < 0
    assert owner._zero_exclusion_margin(I(0, 1)) == 0


def test_observed_rank_alone_does_not_reject_a_true_scalar_factor():
    true = ((I(1), I(2)), (I(2), I(4)))
    observed = ((I(1), I(2)), (I(2), I(Q(41, 10))))
    assert owner._determinant(true) == I(0)
    assert owner._zero_exclusion_margin(owner._determinant(observed)) > 0
    corrected = tuple(
        tuple(v + I(-Q(1, 10), Q(1, 10)) for v in row) for row in observed
    )
    bound = owner._determinant(corrected)
    assert bound.lo <= 0 <= bound.hi


def test_original_history_control_keeps_finite_error_layers_and_same_source(assess):
    report = assess()
    assert report.admitted and report.noise_robust_exclusion_certified
    assert all(report.support_admission_flags)
    eta = report.raw_readout_error_upper_bound
    for i, j in product(range(2), repeat=2):
        true = report.true_response_matrix[i][j]
        observed = report.observed_response_matrix[i][j]
        rectified = report.rectification_matrix[i][j]
        assert observed == true + I(-2 * eta, 2 * eta)
        assert rectified == true + I(-4 * eta, 4 * eta)
        for error, correction in product((-2 * eta, 2 * eta), repeat=2):
            # The theorem encloses exact errors, not extra rounding produced
            # by two successive interval additions in this test.
            low = true.lo + error + correction
            high = true.hi + error + correction
            assert rectified.lo <= low <= high <= rectified.hi
    assert report.true_determinant_bounds == owner._determinant(
        report.true_response_matrix
    )
    assert report.determinant_bounds == owner._determinant(report.rectification_matrix)
    assert report.source.form_error_bounds == (Q(1, 10**10),) * 6
    assert report.raw_readings == 8
    with pytest.raises(FrozenInstanceError):
        report.admitted = False


def test_old_response_verdict_is_not_a_source_or_rank_premise(assess, original_source):
    margins = tuple(
        (name, Q(-1) if name in owner._OLD_RESPONSE_MARGINS else value)
        for name, value in original_source.sufficient_margins
    )
    report = assess(
        replace(original_source, admitted=False, sufficient_margins=margins)
    )
    assert report.admitted
    assert not set(owner._OLD_RESPONSE_MARGINS).intersection(
        dict(report.prerequisite_margins)
    )


@pytest.mark.parametrize(
    "stage", ("recovery", "outer_one", "outer_zero", "work", "observation")
)
def test_each_fresh_history_flag_is_required(assess, original_source, stage):
    if stage == "recovery":
        source = replace(
            original_source,
            recoveries=(
                replace(original_source.recoveries[0], recovery_certified=False),
            )
            + original_source.recoveries[1:],
        )
    elif stage == "observation":
        source = replace(
            original_source,
            observation=replace(original_source.observation, admitted=False),
        )
    else:
        field = {
            "outer_one": "conditional_capture_certified",
            "outer_zero": "control_conditional_capture_certified",
            "work": "attachment_budget_certified",
        }[stage]
        source = replace(
            original_source,
            outer_resources=replace(original_source.outer_resources, **{field: False}),
        )
    report = assess(source)
    assert report.true_rank_margin > 0 and report.rectification_rank_margin > 0
    assert not report.source_history_certified and not report.admitted
    assert not report.scalar_factorization_excluded


@pytest.mark.parametrize(
    "name",
    (
        "outer_one_form",
        "reprojected_fine_phase",
        "outer_capture",
        "contact_attachment_work",
        "host_internal_acute",
        "outer_zero_joined_storage",
    ),
)
def test_every_numeric_premise_remains_strict_even_with_passing_flags(
    assess, original_source, name
):
    source = replace(
        original_source,
        sufficient_margins=tuple(
            (key, Q(0) if key == name else value)
            for key, value in original_source.sufficient_margins
        ),
    )
    report = assess(source)
    assert all(value for _, value in report.prerequisite_flags)
    assert name in report.unresolved_conditions and not report.admitted


@pytest.mark.parametrize(
    "field,value",
    (
        ("clock_factor", Q(1, 5)),
        ("held_capacities", (Q(2),) * 54),
        ("phase_domain", "acute"),
        ("contact_edges", ()),
        ("outer_cycles", ()),
    ),
)
def test_actual_geometry_contract_cannot_be_replaced_by_source_success(
    assess, monkeypatch, field, value
):
    derive = owner._derive_organized_slot_registration
    monkeypatch.setattr(
        owner,
        "_derive_organized_slot_registration",
        lambda **kw: replace(derive(**kw), **{field: value}),
    )
    report = assess()
    assert report.source_history_certified
    assert not any(report.support_admission_flags)
    assert not report.finite_contact_certified and not report.admitted


def test_rank_one_moments_do_not_get_promoted_by_history_success(assess, monkeypatch):
    monkeypatch.setattr(
        owner, "_bounded_slot_registration_moments", lambda **kw: (I(1), I(0))
    )
    report = assess()
    assert report.finite_contact_certified
    assert not report.scalar_factorization_excluded and not report.admitted
    assert set(report.unresolved_conditions) == {
        "true_finite_rank",
        "noise_rectified_finite_rank",
    }


@pytest.mark.parametrize("duplicate", (False, True))
def test_incomplete_or_duplicated_margin_inventory_cannot_pass(
    assess, original_source, duplicate
):
    margins = original_source.sufficient_margins
    changed = margins + (margins[0],) if duplicate else margins[1:]
    report = assess(replace(original_source, sufficient_margins=changed))
    assert not report.admitted
    assert "complete_resource_margin_inventory" in report.unresolved_conditions


@pytest.mark.parametrize(
    "parameter",
    (
        "source",
        "slot_offsets",
        "relative_phase_origin",
        "horizon",
        "readout_error",
        "trajectory",
    ),
)
def test_reserved_assessor_accepts_no_retuned_values_or_reports(parameter):
    with pytest.raises(TypeError):
        owner._assess_organized_interface_factorization(**{parameter: None})
