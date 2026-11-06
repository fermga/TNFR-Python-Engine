"""Primitive-only class contrast, honest error budgets and shared event policies."""

import json
import subprocess
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction as Q
from inspect import Parameter, signature
from math import factorial

import numpy as np
import pytest

from tnfr.physics import relational_sine_class_nonlinear_organization as owner
from tnfr.physics import relational_sine_class_nonlinear_protocol as previous
from tnfr.sdk.relational_reports import relational_report_to_dict
from tnfr.utils.io import json_loads


def _arguments():
    return dict(
        first_probe_amplitude=Q(1, 2000),
        second_probe_amplitude=Q(1, 2000),
        delay=Q(1),
        total_duration=Q(2),
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
        _sine_flow,
        _sine_formed_contact,
        relational_sine_class_readout,
        relational_sine_formed_classes,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("analytic class contrast must not run acquisition or a flow")

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (
                _sine_formed_contact,
                ("_unprobed_handoff", "assess_sine_formed_class_pair"),
            ),
            (relational_sine_formed_classes, ("assess_sine_formed_class_pair",)),
            (_sine_flow, ("_full_sine_field",)),
            (
                _validated_taylor,
                ("validated_box_taylor_step", "flow_jets", "picard_tube"),
            ),
            (
                relational_sine_class_readout,
                ("bound_sine_class_four_history_readout", "_full_sine_field"),
            ),
            (subprocess, ("run", "Popen", "check_output")),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def report():
    return owner.bound_sine_class_nonlinear_organization(**_arguments())


@pytest.fixture
def same_coefficient(report, monkeypatch):
    """Reuse the unchanged exact polynomial for domain and decision controls."""

    def coefficient(a, b, s, t):
        assert (a, b, s, t) == (Q(1, 2000), Q(1, 2000), Q(1), Q(2))
        return report.heat_cubic_channel_coefficients

    monkeypatch.setattr(owner, "_heat_cubic_channels", coefficient)


def test_ten_primitives_fixed_class_pair_and_no_cached_inputs():
    parameters = signature(owner.bound_sine_class_nonlinear_organization).parameters
    assert set(parameters) == set(_arguments()) and len(parameters) == 10
    assert all(
        p.kind == Parameter.KEYWORD_ONLY and p.default == Parameter.empty
        for p in parameters.values()
    )
    for key in (
        "mediator_class",
        "source_report",
        "recorded_response",
        "order",
        "second_initial_state",
    ):
        with pytest.raises(TypeError):
            owner.bound_sine_class_nonlinear_organization(**_arguments(), **{key: 0})


@pytest.mark.parametrize("key", tuple(_arguments()))
@pytest.mark.parametrize("invalid", (True, np.bool_(False), float("nan"), float("inf")))
def test_invalid_primitives_reject_before_any_coefficient(key, invalid, monkeypatch):
    def forbidden(*args):
        pytest.fail("coefficient work preceded primitive admission")

    monkeypatch.setattr(owner, "_parameters", forbidden)
    monkeypatch.setattr(owner, "_heat_cubic_channels", forbidden)
    values = {**_arguments(), key: invalid}
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_nonlinear_organization(**values)


@pytest.mark.parametrize(
    "key,value",
    (
        ("delay", -1),
        ("delay", Q(2) + Q(1, 2**400)),
        ("total_duration", Q(2) + Q(1, 2**400)),
        ("endpoint_radius", -1),
        ("readout_error_bound", -1),
        ("radius", 0),
        ("radius", Q(1, 12) + Q(1, 2**400)),
        ("contact_work_allowance", -1),
        ("first_probe_work_allowance", -1),
        ("second_probe_work_allowance", -1),
        ("first_probe_amplitude", Decimal("1e-400")),
    ),
)
def test_exact_domain_and_materialization_boundaries(key, value, monkeypatch):
    monkeypatch.setattr(
        owner,
        "_heat_cubic_channels",
        lambda *_: pytest.fail("invalid input reached polynomial"),
    )
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_nonlinear_organization(**{**_arguments(), key: value})


def test_fixed_design_is_unresolved_despite_negative_heat_coefficient(report):
    lower, upper = report.gamma_fourth_scaled_heat_contrast_bounds
    assert Q(-702, 10**32) < lower <= upper < Q(-700, 10**32)
    assert report.ideal_contrast_remainder_upper_bound > Q(7, 10**29)
    assert report.ideal_contrast_remainder_upper_bound > 10 * max(
        abs(lower), abs(upper)
    )
    assert report.predicted_orientation == -1
    assert report.true_contrast_bounds[0] < 0 < report.true_contrast_bounds[1]
    assert report.status == "bounds_only"
    assert not report.exact_contrast_zero
    assert not report.true_contrast_sign_certified
    assert not report.recorded_contrast_sign_certified
    assert not report.zero_contrast_record_sets_disjoint
    assert report.strict_recorded_sign_noise_ceiling is None
    assert report.all_identities_certified and report.all_work_within_allowances


def test_independent_class_sources_and_two_full_remainders_are_retained(report):
    terms = report.per_history_ideal_remainder_upper_bounds
    assert terms[0] == 0 and terms[1] == terms[2] > 0
    assert report.per_class_mixed_remainder_upper_bound == sum(terms, Q(0))
    assert report.ideal_contrast_remainder_upper_bound == 2 * sum(terms, Q(0))
    assert report.source_contrast_error_upper_bound == 8 * report.endpoint_radius / (
        1 - 2 * report.gamma_bounds.hi * report.total_duration
    )
    error = (
        report.ideal_contrast_remainder_upper_bound
        + report.source_contrast_error_upper_bound
    )
    assert report.total_true_contrast_error_upper_bound == error
    lo, hi = report.gamma_fourth_scaled_heat_contrast_bounds
    assert report.true_contrast_bounds == (lo - error, hi + error)
    assert report.recorded_contrast_bounds == (
        lo - error - 8 * report.readout_error_bound,
        hi + error + 8 * report.readout_error_bound,
    )
    assert len(report.uniform_history_bounds) == 4
    assert all(
        h.tangent_endpoint_discrepancy_upper_bound is None
        for h in report.uniform_history_bounds
    )


def test_zero_sensor_and_source_error_do_not_remove_method_remainder(
    report, same_coefficient
):
    result = owner.bound_sine_class_nonlinear_organization(
        **{
            **_arguments(),
            "endpoint_radius": 0,
            "readout_error_bound": 0,
        }
    )
    assert (
        result.source_contrast_error_upper_bound
        == result.readout_contrast_error_upper_bound
        == 0
    )
    assert result.true_contrast_bounds == result.recorded_contrast_bounds
    assert (
        result.ideal_contrast_remainder_upper_bound
        == report.ideal_contrast_remainder_upper_bound
    )
    assert result.true_contrast_bounds[0] < 0 < result.true_contrast_bounds[1]
    assert result.status == "bounds_only"


def test_common_heat_channels_cancel_before_any_interval_enclosure(report, monkeypatch):
    outer, mediator, bridge = report.heat_cubic_channel_coefficients
    monkeypatch.setattr(
        owner,
        "_heat_cubic_channels",
        lambda *_: (outer + 10**100, mediator, bridge - 10**100),
    )
    changed = owner.bound_sine_class_nonlinear_organization(**_arguments())
    assert (
        changed.heat_cubic_channel_coefficients
        != report.heat_cubic_channel_coefficients
    )
    assert (
        changed.gamma_fourth_scaled_heat_contrast_bounds
        == report.gamma_fourth_scaled_heat_contrast_bounds
    )
    assert changed.true_contrast_bounds == report.true_contrast_bounds


def test_reversing_both_signed_events_reverses_contrast_and_means(report, monkeypatch):
    monkeypatch.setattr(
        owner,
        "_heat_cubic_channels",
        lambda *_: tuple(-x for x in report.heat_cubic_channel_coefficients),
    )
    result = owner.bound_sine_class_nonlinear_organization(
        **{
            **_arguments(),
            "first_probe_amplitude": -Q(1, 2000),
            "second_probe_amplitude": -Q(1, 2000),
        }
    )
    assert result.true_contrast_bounds == tuple(
        -x for x in reversed(report.true_contrast_bounds)
    )
    assert result.predicted_orientation == 1
    assert (
        result.ideal_contrast_remainder_upper_bound
        == report.ideal_contrast_remainder_upper_bound
    )
    for before, after in zip(
        report.uniform_history_bounds, result.uniform_history_bounds
    ):
        assert after.final_form_mean_shift == -before.final_form_mean_shift
        assert after.first_probe_work_upper_bound == before.first_probe_work_upper_bound
        assert (
            after.second_probe_work_upper_bound == before.second_probe_work_upper_bound
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"first_probe_amplitude": 0},
        {"second_probe_amplitude": 0},
        {"delay": 2},
        {"delay": 0, "total_duration": 0},
    ),
)
def test_exact_pairing_zero_does_not_invent_a_noiseless_record(overrides):
    result = owner.bound_sine_class_nonlinear_organization(
        **{**_arguments(), **overrides}
    )
    assert result.exact_contrast_zero and result.true_contrast_bounds == (0, 0)
    assert result.recorded_contrast_bounds == (
        -8 * result.readout_error_bound,
        8 * result.readout_error_bound,
    )
    assert result.status == "exact_contrast_zero"
    assert not result.true_contrast_sign_certified


@pytest.fixture
def decision_control(monkeypatch):
    """A synthetic coefficient tests decision wiring, not scientific separation."""
    monkeypatch.setattr(owner, "_heat_cubic_channels", lambda *_: (Q(0), Q(-1), Q(0)))
    values = {**_arguments(), "readout_error_bound": 0}
    reference = owner.bound_sine_class_nonlinear_organization(**values)
    assert reference.oriented_true_contrast_lower_bound > 0
    return values, reference


def test_recorded_sign_and_zero_contrast_null_have_distinct_strict_boundaries(
    decision_control,
):
    values, reference = decision_control
    lower = reference.oriented_true_contrast_lower_bound
    sign_boundary = owner.bound_sine_class_nonlinear_organization(
        **{**values, "readout_error_bound": lower / 8}
    )
    assert sign_boundary.true_contrast_sign_certified
    assert sign_boundary.recorded_sign_margin == 0
    assert not sign_boundary.recorded_contrast_sign_certified
    null_boundary = owner.bound_sine_class_nonlinear_organization(
        **{**values, "readout_error_bound": lower / 16}
    )
    assert null_boundary.recorded_contrast_sign_certified
    assert null_boundary.zero_contrast_comparator_separation_margin == 0
    assert not null_boundary.zero_contrast_record_sets_disjoint
    inside = owner.bound_sine_class_nonlinear_organization(
        **{**values, "readout_error_bound": lower / 32}
    )
    assert inside.zero_contrast_record_sets_disjoint


def test_true_sign_equality_is_not_certified(decision_control):
    values, reference = decision_control
    nominal_lower = -reference.gamma_fourth_scaled_heat_contrast_bounds[1]
    epsilon = (
        (nominal_lower - reference.ideal_contrast_remainder_upper_bound)
        * reference.source_comparison_margin
        / 8
    )
    result = owner.bound_sine_class_nonlinear_organization(
        **{**values, "endpoint_radius": epsilon}
    )
    assert result.oriented_true_contrast_lower_bound == 0
    assert not result.true_contrast_sign_certified and result.status == "bounds_only"


def test_closed_work_limits_and_identity_do_not_select_contrast(
    report, same_coefficient
):
    limits = dict(
        contact_work_allowance=report.contact_work_upper_bound,
        first_probe_work_allowance=max(
            h.first_probe_work_upper_bound for h in report.uniform_history_bounds
        ),
        second_probe_work_allowance=max(
            h.second_probe_work_upper_bound for h in report.uniform_history_bounds
        ),
    )
    equality = owner.bound_sine_class_nonlinear_organization(
        **{**_arguments(), **limits}
    )
    assert equality.all_work_within_allowances
    assert min(h.second_probe_work_margin for h in equality.uniform_history_bounds) == 0
    failed = owner.bound_sine_class_nonlinear_organization(
        **{**_arguments(), "second_probe_work_allowance": 0, "radius": Q(1, 1000)}
    )
    assert not failed.all_work_within_allowances and not failed.all_identities_certified
    assert (
        failed.true_contrast_bounds
        == equality.true_contrast_bounds
        == report.true_contrast_bounds
    )


def test_exact_tiny_amplitude_and_represented_real_are_preserved():
    tiny = Q(1, 2**2000)
    result = owner.bound_sine_class_nonlinear_organization(
        **{**_arguments(), "first_probe_amplitude": tiny, "second_probe_amplitude": 0}
    )
    assert result.first_probe_amplitude == tiny > 0
    assert result.exact_contrast_zero
    represented = owner.bound_sine_class_nonlinear_organization(
        **{**_arguments(), "first_probe_amplitude": -0.125, "second_probe_amplitude": 0}
    )
    assert represented.first_probe_amplitude == Q(-1, 8)


@pytest.mark.parametrize(
    "a,b,s,t",
    (
        (Q(1, 7), Q(-1, 9), Q(1, 8), Q(1, 2)),
        (Q(0), Q(1), Q(0), Q(2)),
        (Q(1, 2000), Q(1, 2000), Q(1), Q(2)),
    ),
)
def test_shared_tail_preserves_original_exact_operation(a, b, s, t):
    eta = (2 * t) ** 33 / factorial(33)
    expected = (
        4
        * (abs(a) ** 2 * abs(b) + abs(a) * abs(b) ** 2)
        * (t - s)
        * ((1 + eta) ** 4 - 1)
    )
    assert previous._heat_truncation_bounds(a, b, s, t) == (eta, expected)
    assert owner._heat_truncation_bounds is previous._heat_truncation_bounds


def test_exact_sdk_projection_and_null_availability(report):
    projected = relational_report_to_dict(report)
    assert projected["report"] == report.to_dict()["report"]
    assert projected["schema"] == "tnfr.relational-report.v1"
    assert projected["report_type"] == "SineClassNonlinearOrganization"
    assert report.to_dict()["schema"] == "tnfr.sine-class-nonlinear-organization.v1"
    detached = json_loads(json.dumps(projected, allow_nan=False))
    lo = report.true_contrast_bounds[0]
    assert detached["report"]["true_contrast_bounds"][0] == {
        "numerator": lo.numerator,
        "denominator": lo.denominator,
    }
    assert detached["report"]["strict_recorded_sign_noise_ceiling"] is None
    # A detached metadata control checks null projection without another bound.
    unavailable = replace(report, oriented_true_contrast_lower_bound=None)
    assert unavailable.to_dict()["report"]["oriented_true_contrast_lower_bound"] is None
