"""Conditional finite protocol admission, strict noise margins and event reuse."""

import json
import subprocess
from dataclasses import asdict
from decimal import Decimal
from fractions import Fraction as Q
from inspect import Parameter, signature

import numpy as np
import pytest

from tnfr.physics import relational_sine_class_nonlinear_protocol as owner
from tnfr.physics import relational_sine_class_superposition as previous
from tnfr.sdk.relational_reports import relational_report_to_dict
from tnfr.utils.io import json_loads


def _arguments():
    return dict(
        mediator_class=2,
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
        _sine_formed_contact,
        relational_sine_class_mediation,
        relational_sine_formed_classes,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("analytic protocol must not execute acquisition, flow or a worker")

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
    return owner.bound_sine_class_nonlinear_protocol(**_arguments())


@pytest.fixture
def reuse_coefficient(report, monkeypatch):
    """Controls reuse only the unchanged costly rational polynomial coefficient."""

    def same_coefficient(a, b, s, t):
        assert (a, b, s, t) == (
            report.first_probe_amplitude,
            report.second_probe_amplitude,
            report.delay,
            report.total_duration,
        )
        return report.heat_cubic_channel_coefficients

    monkeypatch.setattr(owner, "_heat_cubic_channels", same_coefficient)


def test_eleven_primitives_and_fixed_algebra_budget():
    parameters = signature(owner.bound_sine_class_nonlinear_protocol).parameters
    assert set(parameters) == set(_arguments()) and len(parameters) == 11
    assert all(
        p.kind == Parameter.KEYWORD_ONLY and p.default == Parameter.empty
        for p in parameters.values()
    )
    for key in ("response", "source_handoff", "order", "second_initial_state"):
        with pytest.raises(TypeError):
            owner.bound_sine_class_nonlinear_protocol(**_arguments(), **{key: 0})


@pytest.mark.parametrize("name", tuple(_arguments()))
@pytest.mark.parametrize("invalid", (True, np.bool_(False), float("inf"), float("nan")))
def test_authoritative_invalid_values_fail_before_coefficients(
    name, invalid, monkeypatch
):
    monkeypatch.setattr(
        owner, "_parameters", lambda *_: pytest.fail("admission bypassed")
    )
    monkeypatch.setattr(
        owner, "_heat_cubic_channels", lambda *_: pytest.fail("admission bypassed")
    )
    values = _arguments()
    values[name] = invalid
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_nonlinear_protocol(**values)


@pytest.mark.parametrize(
    "key,value",
    (
        ("mediator_class", Q(1)),
        ("mediator_class", np.int64(1)),
        ("mediator_class", 3),
        ("delay", -1),
        ("delay", Q(2) + Q(1, 2**400)),
        ("total_duration", Q(2) + Q(1, 2**400)),
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
def test_exact_domains_before_heat_work(key, value, monkeypatch):
    monkeypatch.setattr(
        owner, "_heat_cubic_channels", lambda *_: pytest.fail("admission bypassed")
    )
    values = _arguments()
    values[key] = value
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_nonlinear_protocol(**values)


def test_declared_heat_design_has_negative_finite_separation(report):
    assert report.predicted_orientation == -1
    assert report.true_mixed_bounds[1] < -8 * report.readout_error_bound
    assert report.true_mixed_sign_certified and report.recorded_mixed_sign_certified
    assert report.four_record_sets_disjoint and report.status == "record_sets_disjoint"
    assert report.all_identities_certified and report.all_work_within_allowances
    assert report.heat_polynomial_order == 32
    assert not hasattr(report, "formation_certified")
    assert report.oriented_true_mixed_lower_bound > Q(3, 10**27)
    # The analytic heat coefficient can change sign from the short-time term.
    a, b, s, u = (
        report.first_probe_amplitude,
        report.second_probe_amplitude,
        report.delay,
        Q(1),
    )
    leading = (
        a * a * b * (s * s * u * u / 2 + 2 * s * u**3 / 3 + u**4 / 4)
        + a * b * b * (s * u**3 / 3 + u**4 / 4)
    ) / 384
    assert leading > 0 and report.gamma_fourth_scaled_heat_bounds[1] < 0


@pytest.mark.parametrize("divisor", (4, 8))
def test_noise_boundaries_are_strict_and_have_distinct_meaning(
    report, reuse_coefficient, divisor
):
    values = _arguments()
    values["readout_error_bound"] = report.oriented_true_mixed_lower_bound / divisor
    boundary = owner.bound_sine_class_nonlinear_protocol(**values)
    assert boundary.true_mixed_sign_certified
    assert not boundary.four_record_sets_disjoint
    if divisor == 4:
        assert boundary.recorded_sign_margin == 0
        assert not boundary.recorded_mixed_sign_certified
    else:
        assert boundary.disjoint_record_margin == 0
        assert boundary.recorded_mixed_sign_certified
    values["readout_error_bound"] *= Q(999, 1000)
    inside = owner.bound_sine_class_nonlinear_protocol(**values)
    assert inside.recorded_mixed_sign_certified
    assert inside.four_record_sets_disjoint == (divisor == 8)


def test_work_boundary_is_closed_and_observation_does_not_consume_policy(
    report, reuse_coefficient
):
    values = _arguments()
    values.update(
        contact_work_allowance=report.contact_work_upper_bound,
        first_probe_work_allowance=max(
            h.first_probe_work_upper_bound for h in report.histories
        ),
        second_probe_work_allowance=max(
            h.second_probe_work_upper_bound for h in report.histories
        ),
    )
    boundary = owner.bound_sine_class_nonlinear_protocol(**values)
    assert boundary.all_work_within_allowances
    values["second_probe_work_allowance"] = 0
    values["radius"] = Q(1, 10**10)
    failed = owner.bound_sine_class_nonlinear_protocol(**values)
    assert not failed.all_work_within_allowances and not failed.all_identities_certified
    assert failed.true_mixed_bounds == report.true_mixed_bounds
    assert failed.four_record_sets_disjoint


def test_unresolved_source_bound_is_not_a_linearity_or_overlap_verdict(
    reuse_coefficient,
):
    values = _arguments()
    values["endpoint_radius"] = Q(1, 100)
    result = owner.bound_sine_class_nonlinear_protocol(**values)
    assert result.true_mixed_bounds[0] < 0 < result.true_mixed_bounds[1]
    assert result.predicted_orientation == -1  # the heat term alone retains a sign
    assert result.oriented_true_mixed_lower_bound < 0
    assert result.recorded_sign_margin < 0 and result.disjoint_record_margin < 0
    assert not result.true_mixed_sign_certified
    assert not result.four_record_sets_disjoint
    assert result.strict_disjoint_noise_ceiling is None
    assert result.status == "bounds_only"
    assert not hasattr(result, "four_record_overlap_admitted")


def test_signed_simultaneous_reversal_reverses_finite_orientation(report):
    values = _arguments()
    values["first_probe_amplitude"] *= -1
    values["second_probe_amplitude"] *= -1
    reversed_report = owner.bound_sine_class_nonlinear_protocol(**values)
    assert reversed_report.heat_cubic_channel_coefficients == tuple(
        -value for value in report.heat_cubic_channel_coefficients
    )
    assert reversed_report.true_mixed_bounds == (
        -report.true_mixed_bounds[1],
        -report.true_mixed_bounds[0],
    )
    assert reversed_report.predicted_orientation == 1
    assert reversed_report.four_record_sets_disjoint
    for original, reverse in zip(report.histories, reversed_report.histories):
        assert original.final_form_mean_shift == -reverse.final_form_mean_shift
        assert (
            original.second_probe_work_upper_bound
            == reverse.second_probe_work_upper_bound
        )


def test_zero_source_and_sensor_errors_are_admitted(reuse_coefficient):
    values = _arguments()
    values.update(endpoint_radius=0, readout_error_bound=0)
    result = owner.bound_sine_class_nonlinear_protocol(**values)
    assert result.source_mixed_error_upper_bound == 0
    assert result.true_mixed_bounds == result.recorded_mixed_bounds
    assert result.four_record_sets_disjoint
    assert result.contact_work_bounds.lo == result.contact_work_bounds.hi == 0


@pytest.mark.parametrize(
    "updates",
    (
        {"first_probe_amplitude": 0},
        {"second_probe_amplitude": 0},
        {"delay": 2},
        {"delay": 0, "total_duration": 0},
    ),
)
def test_exact_nulls_do_not_become_false_signed_protocols(updates):
    values = _arguments()
    values.update(updates)
    result = owner.bound_sine_class_nonlinear_protocol(**values)
    assert result.exact_mixed_zero and result.true_mixed_bounds == (0, 0)
    assert result.predicted_orientation == 0
    assert result.oriented_true_mixed_lower_bound is None
    assert result.recorded_sign_margin is None and result.disjoint_record_margin is None
    assert not result.true_mixed_sign_certified and not result.four_record_sets_disjoint
    assert result.status == "exact_mixed_zero"


def test_old_short_horizon_admission_and_shared_event_values_are_preserved(monkeypatch):
    with pytest.raises(ValueError):
        previous.bound_sine_class_superposition(**_arguments())
    values = _arguments()
    values.update(delay=Q(1, 20000), total_duration=Q(1, 10000))
    old = previous.bound_sine_class_superposition(**values)
    # This control checks the shared event ledger, not the coefficient theorem.
    monkeypatch.setattr(owner, "_heat_cubic_channels", lambda *_: (Q(0),) * 3)
    new = owner.bound_sine_class_nonlinear_protocol(**values)
    for before, after in zip(old.histories, new.histories):
        expected, actual = asdict(before), asdict(after)
        assert expected.pop("tangent_endpoint_discrepancy_upper_bound") is not None
        assert actual.pop("tangent_endpoint_discrepancy_upper_bound") is None
        assert actual == expected
    assert new.contact_work_upper_bound == old.contact_work_upper_bound
    assert new.identity_barrier_lower_bound == old.identity_barrier_lower_bound


def test_tiny_exact_values_and_endpoint_pairs_survive_projection():
    values = _arguments()
    tiny = Q(1, 2**1000)
    values.update(
        first_probe_amplitude=tiny,
        second_probe_amplitude=0,
        endpoint_radius=tiny,
        readout_error_bound=tiny,
    )
    result = owner.bound_sine_class_nonlinear_protocol(**values)
    assert result.first_probe_amplitude == result.endpoint_radius == tiny
    assert result.recorded_mixed_bounds == (-4 * tiny, 4 * tiny)
    assert result.recorded_mixed_interval.lo < result.recorded_mixed_bounds[0]


def test_exact_polynomial_algebra_uses_affine_substitution_not_point_samples():
    polynomial = (Q(2, 3), Q(-5, 7), Q(11, 13))
    shifted = owner._affine_compose(polynomial, Q(3, 5), Q(-2, 7))
    product = owner._multiply(polynomial, (Q(-3, 11), Q(7, 17)))
    for point in (Q(-9, 2), Q(0), Q(4, 3)):

        def evaluate(coeffs, x):
            return sum((value * x**n for n, value in enumerate(coeffs)), Q(0))

        assert evaluate(shifted, point) == evaluate(
            polynomial, Q(3, 5) - Q(2, 7) * point
        )
        assert evaluate(product, point) == evaluate(polynomial, point) * (
            Q(-3, 11) + Q(7, 17) * point
        )


def test_sdk_serializes_exact_bounds_and_unavailable_individual_tangent_comparison(
    report,
):
    projected = relational_report_to_dict(report)
    restored = json_loads(json.dumps(projected, allow_nan=False))
    assert restored["schema"] == "tnfr.relational-report.v1"
    assert report.to_dict()["schema"] == "tnfr.sine-class-nonlinear-protocol.v1"
    assert restored["report"] == report.to_dict()["report"]
    body = restored["report"]
    assert body["histories"][0]["tangent_endpoint_discrepancy_upper_bound"] is None
    first = body["true_mixed_bounds"][0]
    assert (
        Q(int(first["numerator"]), int(first["denominator"]))
        == report.true_mixed_bounds[0]
    )
    assert body["four_record_sets_disjoint"] is True
