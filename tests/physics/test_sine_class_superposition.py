"""Conditional four-history bounds, event budgets and observation scopes."""

import json
import subprocess
from decimal import Decimal
from fractions import Fraction as Q
from inspect import Parameter, signature

import numpy as np
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_class_superposition as owner
from tnfr.sdk.relational_reports import relational_report_to_dict
from tnfr.utils.io import json_loads


def _arguments():
    return dict(
        mediator_class=2,
        first_probe_amplitude=Q(1, 2000),
        second_probe_amplitude=Q(1, 2000),
        delay=Q(1, 20000),
        total_duration=Q(1, 10000),
        endpoint_radius=Q(1, 10**32),
        readout_error_bound=Q(1, 10**30),
        radius=Q(1, 12),
        contact_work_allowance=Q(1, 10**12),
        first_probe_work_allowance=Q(1, 500000),
        second_probe_work_allowance=Q(1, 500000),
    )


@pytest.fixture(scope="module", autouse=True)
def no_scientific_acquisition():
    from tnfr.mathematics import _validated_taylor
    from tnfr.physics import (
        _sine_formed_contact,
        relational_sine_class_mediation,
        relational_sine_formed_classes,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("superposition bounds must not execute a source, flow or worker")

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (
                _sine_formed_contact,
                ("_unprobed_handoff", "assess_sine_formed_class_pair"),
            ),
            (relational_sine_formed_classes, ("assess_sine_formed_class_pair",)),
            (
                relational_sine_class_mediation,
                ("assess_sine_class_mediation", "_unprobed_handoff"),
            ),
            (
                _validated_taylor,
                ("flow_jets", "picard_tube", "validated_box_taylor_step"),
            ),
            (subprocess, ("run", "Popen")),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def report():
    return owner.bound_sine_class_superposition(**_arguments())


def test_eleven_mandatory_primitives_do_not_admit_a_response_or_reset(report):
    parameters = signature(owner.bound_sine_class_superposition).parameters
    assert set(parameters) == set(_arguments()) and len(parameters) == 11
    assert all(
        p.kind == Parameter.KEYWORD_ONLY and p.default == Parameter.empty
        for p in parameters.values()
    )
    for key in (
        "source_handoff",
        "formation_time",
        "response",
        "initial_state",
        "second_initial_state",
        "order",
    ):
        with pytest.raises(TypeError):
            owner.bound_sine_class_superposition(**_arguments(), **{key: report})
    assert not hasattr(report, "formation_certified")
    assert not hasattr(report, "positive_finite_mixed_response")


@pytest.mark.parametrize("name", tuple(_arguments()))
@pytest.mark.parametrize("invalid", (True, np.bool_(False), float("inf"), float("nan")))
def test_all_authoritative_scalars_reject_before_model(name, invalid, monkeypatch):
    monkeypatch.setattr(
        owner, "_parameters", lambda *_: pytest.fail("admission bypassed")
    )
    values = _arguments()
    values[name] = invalid
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_superposition(**values)


@pytest.mark.parametrize(
    "key,value",
    (
        ("mediator_class", Q(1)),
        ("mediator_class", np.int64(1)),
        ("mediator_class", 3),
        ("delay", Q(1)),
        ("delay", -1),
        ("total_duration", Q(1, 4) + Q(1, 2**400)),
        ("radius", 0),
        ("radius", Q(1, 12) + Q(1, 2**400)),
        ("endpoint_radius", -1),
        ("readout_error_bound", -1),
        ("contact_work_allowance", -1),
        ("first_probe_work_allowance", -1),
        ("second_probe_work_allowance", -1),
        ("first_probe_amplitude", Decimal("1e-400")),
    ),
)
def test_domains_and_representation_are_checked_exactly(key, value):
    values = _arguments()
    values[key] = value
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_superposition(**values)


def test_symmetric_design_has_overlap_and_retained_tiny_formal_sign(report):
    a, b, s, t = (
        getattr(report, name)
        for name in (
            "first_probe_amplitude",
            "second_probe_amplitude",
            "delay",
            "total_duration",
        )
    )
    u = t - s
    first = s**2 * u**2 / 2 + 2 * s * u**3 / 3 + u**4 / 4
    second = s * u**3 / 3 + u**4 / 4
    assert report.formal_first_joint_time_integral == first
    assert report.formal_second_joint_time_integral == second
    assert (
        report.formal_mixed_coefficient_rational_factor
        == (a**2 * b * first + a * b**2 * second) / 384
        > 0
    )
    assert report.formal_mixed_coefficient_sign == 1
    assert report.formal_mixed_coefficient_bounds.lo == 0
    assert not report.exact_mixed_zero
    assert report.scalar_cancellation_admitted and report.four_record_overlap_admitted
    assert report.all_identities_certified and report.all_work_within_allowances
    assert report.status == "four_record_overlap"


def test_formal_coefficient_is_joint_time_quartic_not_remaining_time_quartic(report):
    values = _arguments()
    values.update(delay=report.delay * 3, total_duration=report.total_duration * 3)
    scaled = owner.bound_sine_class_superposition(**values)
    assert (
        scaled.formal_mixed_coefficient_rational_factor
        == 3**4 * report.formal_mixed_coefficient_rational_factor
    )
    values.update(
        delay=report.delay,
        total_duration=report.delay + 3 * (report.total_duration - report.delay),
    )
    remaining_only = owner.bound_sine_class_superposition(**values)
    assert (
        remaining_only.formal_mixed_coefficient_rational_factor
        != 3**4 * report.formal_mixed_coefficient_rational_factor
    )


def test_common_cubic_bound_and_record_overlap_retain_each_history(report):
    g, t, eps = report.gamma_bounds.hi, report.total_duration, report.endpoint_radius
    d, ell = 1 - 2 * g**2 * t**2, 1 - 3 * t
    cubic = Q(8, 3) * g**4 * t**4 / (d**3 * ell) * (1 + Q(2, 3) * g**2 * t**2 / ell)
    assert report.central_cubic_error_coefficient == cubic
    assert report.history_order == tuple(history.label for history in report.histories)
    a, b = report.first_probe_amplitude, report.second_probe_amplitude
    assert tuple(
        (row.first_probe_amplitude, row.second_probe_amplitude)
        for row in report.histories
    ) == ((0, 0), (a, 0), (0, b), (a, b))
    for row in report.histories:
        amplitude = abs(row.first_probe_amplitude) + abs(row.second_probe_amplitude)
        assert row.cumulative_amplitude_budget == amplitude
        assert (
            row.tangent_endpoint_discrepancy_upper_bound
            == cubic * amplitude**3 + 2 * eps / ell
        )
    assert report.maximum_tangent_endpoint_discrepancy_upper_bound == max(
        row.tangent_endpoint_discrepancy_upper_bound for row in report.histories
    )
    assert (
        report.true_mixed_upper_bound
        == cubic * ((abs(a) + abs(b)) ** 3 + abs(a) ** 3 + abs(b) ** 3) + 4 * eps / ell
    )


def test_second_work_uses_carried_history_and_does_not_reset_to_source(report):
    g, s, eps = report.gamma_bounds.hi, report.delay, report.endpoint_radius
    a, b = report.first_probe_amplitude, report.second_probe_amplitude
    neither, first, second, both = report.histories
    qa = (abs(a) + eps + 2 * g * s * eps) / (1 - 2 * g**2 * s**2)
    q0 = (eps + 2 * g * s * eps) / (1 - 2 * g**2 * s**2)
    assert (
        both.pre_second_form_coordinate_upper_bound
        == first.pre_second_form_coordinate_upper_bound
        == qa
    )
    assert (
        second.pre_second_form_coordinate_upper_bound
        == neither.pre_second_form_coordinate_upper_bound
        == q0
    )
    assert qa > q0 > eps
    assert both.second_probe_work_upper_bound == Q(3, 2) * b**2 + 6 * abs(b) * qa
    assert second.second_probe_work_upper_bound == Q(3, 2) * b**2 + 6 * abs(b) * q0
    assert both.second_probe_work_upper_bound > second.second_probe_work_upper_bound
    assert both.second_probe_work_bounds.lo < 0  # signed pre-event cross term
    for row in report.histories:
        assert (
            row.after_second_excess_storage_upper_bound
            == 22 * eps**2
            + row.first_probe_work_upper_bound
            + row.second_probe_work_upper_bound
        )
        assert row.after_second_radius_squared_upper_bound == 27 * (
            row.maximum_form_coordinate_upper_bound**2
            + row.maximum_phase_deviation_upper_bound**2
        )


def test_simultaneous_signed_reversal_preserves_bounds_but_reverses_formal_and_mean_sign(
    report,
):
    values = _arguments()
    values.update(
        first_probe_amplitude=-report.first_probe_amplitude,
        second_probe_amplitude=-report.second_probe_amplitude,
    )
    reversed_report = owner.bound_sine_class_superposition(**values)
    assert (
        reversed_report.formal_mixed_coefficient_rational_factor
        == -report.formal_mixed_coefficient_rational_factor
    )
    assert reversed_report.formal_mixed_coefficient_sign == -1
    assert reversed_report.true_mixed_upper_bound == report.true_mixed_upper_bound
    assert reversed_report.instantaneous_curvature_orientation == -1
    assert (
        reversed_report.oriented_instantaneous_curvature_lower_bound
        == report.oriented_instantaneous_curvature_lower_bound
    )
    for original, changed in zip(report.histories, reversed_report.histories):
        assert changed.final_form_mean_shift == -original.final_form_mean_shift
        assert changed.final_form_mean_bounds == -original.final_form_mean_bounds
        assert changed.second_probe_work_bounds == original.second_probe_work_bounds


@pytest.mark.parametrize("case", ("first_zero", "second_zero", "second_at_endpoint"))
def test_exact_scalar_null_does_not_assert_individual_record_overlap(case):
    values = _arguments()
    values["readout_error_bound"] = 0
    if case == "first_zero":
        values["first_probe_amplitude"] = 0
    elif case == "second_zero":
        values["second_probe_amplitude"] = 0
    else:
        values["delay"] = values["total_duration"]
    result = owner.bound_sine_class_superposition(**values)
    assert result.exact_mixed_zero and result.true_mixed_upper_bound == 0
    assert result.recorded_mixed_bounds == I(0)
    assert (
        result.scalar_cancellation_admitted and not result.four_record_overlap_admitted
    )
    if case == "second_at_endpoint":
        # A right derivative after that event concerns later times, so it can
        # be positive while the value at this final event is exactly zero.
        assert result.instantaneous_curvature_certified


def test_closed_noise_boundaries_distinguish_scalar_and_four_record_results(report):
    values = _arguments()
    values["readout_error_bound"] = report.true_mixed_upper_bound / 4
    scalar = owner.bound_sine_class_superposition(**values)
    assert scalar.scalar_cancellation_admitted
    assert not scalar.four_record_overlap_admitted
    values["readout_error_bound"] = (
        report.maximum_tangent_endpoint_discrepancy_upper_bound
    )
    overlap = owner.bound_sine_class_superposition(**values)
    assert overlap.four_record_overlap_admitted
    values["readout_error_bound"] -= Q(1, 2**500)
    below = owner.bound_sine_class_superposition(**values)
    assert not below.four_record_overlap_admitted
    assert below.scalar_cancellation_admitted


def test_closed_work_allowance_boundaries_do_not_change_response_guarantees(report):
    values = _arguments()
    values.update(
        contact_work_allowance=report.contact_work_upper_bound,
        first_probe_work_allowance=max(
            row.first_probe_work_upper_bound for row in report.histories
        ),
        second_probe_work_allowance=max(
            row.second_probe_work_upper_bound for row in report.histories
        ),
    )
    exact = owner.bound_sine_class_superposition(**values)
    assert exact.all_work_within_allowances
    for key in (
        "contact_work_allowance",
        "first_probe_work_allowance",
        "second_probe_work_allowance",
    ):
        changed = values.copy()
        changed[key] -= Q(1, 2**500)
        failed = owner.bound_sine_class_superposition(**changed)
        assert not failed.all_work_within_allowances
        assert failed.all_identities_certified
        assert failed.recorded_mixed_bounds == report.recorded_mixed_bounds


def test_identity_failure_does_not_hide_valid_global_response_bounds():
    values = _arguments()
    values["radius"] = Q(1, 10**6)
    result = owner.bound_sine_class_superposition(**values)
    assert not result.all_identities_certified
    assert result.histories[-1].after_second_radius_margin < 0
    assert result.histories[-1].after_second_storage_margin < 0
    assert result.four_record_overlap_admitted and result.all_work_within_allowances


def test_actual_source_curvature_bound_has_its_own_scope_and_availability(report):
    g, lo, a, b, s, eps = (
        report.gamma_bounds.hi,
        report.gamma_bounds.lo,
        abs(report.first_probe_amplitude),
        abs(report.second_probe_amplitude),
        report.delay,
        report.endpoint_radius,
    )
    baseline = 2 * eps / (1 - 3 * s)
    remainder = (
        4 * g * (a / (1 - 2 * g**2 * s**2)) * (1 + 2 * g**2 * s) * s**2 + baseline
    )
    lower, upper = max(Q(0), lo * a * s / 4 - remainder), g * a * s / 4 + remainder
    assert upper <= 1 and lower**2 / 3 - baseline**2 / 2 > 0
    assert (
        report.oriented_instantaneous_curvature_lower_bound
        == lo**2 * b * (lower**2 / 3 - baseline**2 / 2) / 12
        > 0
    )
    assert (
        report.instantaneous_curvature_certified and report.four_record_overlap_admitted
    )
    values = _arguments()
    values["endpoint_radius"] = Q(1, 100)
    unavailable = owner.bound_sine_class_superposition(**values)
    assert unavailable.oriented_instantaneous_curvature_lower_bound is None
    assert not unavailable.instantaneous_curvature_certified
    values.update(delay=0, endpoint_radius=0)
    zero = owner.bound_sine_class_superposition(**values)
    assert zero.instantaneous_curvature_exact_zero
    assert zero.oriented_instantaneous_curvature_lower_bound == 0


def test_class_choice_and_exact_tiny_source_do_not_rationalize_or_replay_a_family(
    report,
):
    values = _arguments()
    values.update(mediator_class=1, endpoint_radius=Q(1, 10**400))
    result = owner.bound_sine_class_superposition(**values)
    assert result.classes == (1, 1, 1)
    assert result.gamma_bounds == report.gamma_bounds
    assert (
        result.formal_mixed_coefficient_rational_factor
        == report.formal_mixed_coefficient_rational_factor
    )
    assert result.endpoint_radius == Q(1, 10**400) > 0
    assert result.source_mixed_error_upper_bound > 0


def test_sdk_projection_keeps_exact_factor_and_null_curvature(report):
    direct = report.to_dict()
    assert direct["schema"] == "tnfr.sine-class-superposition.v1"
    projected = relational_report_to_dict(report)
    assert projected["report"] == direct["report"]
    assert json_loads(json.dumps(projected, allow_nan=False)) == projected
    factor = projected["report"]["formal_mixed_coefficient_rational_factor"]
    assert (
        Q(factor["numerator"], factor["denominator"])
        == report.formal_mixed_coefficient_rational_factor
    )
    values = _arguments()
    values["endpoint_radius"] = Q(1, 100)
    unavailable = owner.bound_sine_class_superposition(**values)
    assert (
        relational_report_to_dict(unavailable)["report"][
            "oriented_instantaneous_curvature_lower_bound"
        ]
        is None
    )
