"""Primitive recovery resources retain the acquired source and clock map."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _sine_recovery_resources as owner

ERROR = Q(1, 10**10)


@pytest.fixture(scope="module", autouse=True)
def no_scientific_execution():
    with forbid_sine_regeneration():
        yield


def _assess(**changes):
    # This is the previously admitted preparation and recovery budget.
    # A separate prospective declaration owns any new resource witness.
    return owner._assess_sine_recovery_resources(
        **(
            dict(
                form_error_bound=ERROR,
                phase_error_bound=ERROR,
                quarter_dwell=Q(2 * 10**13),
                endpoint_radius=Q(1, 10**70),
                decay_power=512,
            )
            | changes
        )
    )


@pytest.fixture(scope="module")
def recovery():
    return _assess()


def test_actual_source_and_both_clock_rows_are_retained(recovery):
    source = recovery.source
    assert recovery.status == "certified_sine_recovery_resources"
    assert recovery.source_certified and recovery.decay_certified
    assert recovery.recovery_certified and not recovery.unresolved_conditions
    assert source.deformation == 0
    assert source.scaled_time == 100 and source.radius == Q(1, 12)
    assert source.form_error_bound == source.phase_error_bound == ERROR
    assert source.preparation_dimension == 16
    assert recovery.quarter_formation_time == 200
    assert recovery.half_law_dwell == 10**13
    assert recovery.quarter_handoff_time == 200 + 2 * 10**13
    assert recovery.original_handoff_time * Q(1023, 1024) == (
        recovery.quarter_handoff_time
    )
    assert recovery.endpoint_radius != recovery.form_error_bound
    assert recovery.exact_decay_upper_bound == Q(1, 2**512)
    assert recovery.endpoint_form_norm_squared_upper_bound < Q(1, 10**140)
    assert recovery.endpoint_phase_norm_squared_upper_bound < Q(1, 10**140)
    assert not hasattr(recovery, "attachment_work_bounds")
    with pytest.raises(FrozenInstanceError):
        recovery.quarter_dwell = Q(0)


def test_class_one_curvature_matches_independent_high_precision_math(recovery):
    mp = mpmath.mp.clone()
    mp.dps = 180

    def real(value):
        return mp.mpf(value.numerator) / value.denominator

    eta = 1 / (1023 * mp.pi) ** 2
    gap, maximum, epsilon = mp.mpf(1) / 5, 2, mp.mpf(1) / 20
    angle = 2 * mp.pi / 9 + mp.sqrt(2) / 12
    curvature = mp.cos(angle)
    mu, stiffness = eta * curvature * gap**2, eta * maximum**2
    lower = mu / 2 + gap**2 / 16
    upper = stiffness / 2 + epsilon * maximum / 2 + epsilon**2
    rate = min(4 * (gap - epsilon) / 3, epsilon * mu / upper)
    angle_bounds = recovery.class_one_radius_angle_bounds
    assert real(angle_bounds.lo) <= angle <= real(angle_bounds.hi)
    assert 0 < curvature - real(recovery.class_one_phase_hessian_lower_bound)
    assert curvature - real(recovery.class_one_phase_hessian_lower_bound) < mp.mpf(
        "1e-35"
    )
    for actual, bound in (
        (mu, recovery.transformed_stiffness_lower_bound),
        (lower, recovery.lyapunov_position_lower_coefficient),
        (rate, recovery.lyapunov_decay_rate),
    ):
        assert 0 <= actual - real(bound) < mp.mpf("1e-30")
    for actual, bound in (
        (stiffness, recovery.transformed_stiffness_upper_bound),
        (upper, recovery.lyapunov_position_upper_coefficient),
    ):
        assert 0 <= real(bound) - actual < mp.mpf("1e-30")
    assert recovery.class_one_phase_hessian_lower_bound > (
        recovery.source.phase_hessian_lower_bound
    )
    assert recovery.class_one_phase_hessian_upper_bound == 1
    assert mp.exp(-real(recovery.decay_exponent)) < real(
        recovery.exact_decay_upper_bound
    )


def test_energy_return_uses_distinct_form_and_phase_metrics(recovery):
    energy = recovery.returned_lyapunov_upper_bound
    assert energy == recovery.lyapunov_initial_upper_bound / 2**512
    assert (
        recovery.endpoint_form_norm_squared_upper_bound
        * (recovery.source.eta_bounds.lo / 20)
        == energy
    )
    assert (
        recovery.endpoint_phase_norm_squared_upper_bound
        * (recovery.lyapunov_position_lower_coefficient / 2)
        == energy
    )
    assert recovery.decay_exponent == (
        recovery.lyapunov_decay_rate * recovery.quarter_dwell / 2
    )
    assert recovery.decay_exponent_margin == recovery.decay_exponent - 512


def test_each_call_rebuilds_only_required_source_class(recovery, monkeypatch):
    calls = []
    source = replace(
        recovery.source,
        formation_certified_by_class=(True, False),
        source_budget_certified_by_class=(True, False),
        status="unavailable",
    )

    def fresh(**kwargs):
        calls.append(kwargs)
        return source

    monkeypatch.setattr(owner, "_assess_constitutive_class_pair", fresh)
    for _ in range(2):
        result = _assess()
        assert result.source is source
        assert result.recovery_certified
    assert (
        calls
        == [
            dict(
                deformation=Q(0),
                scaled_time=Q(100),
                form_error_bound=ERROR,
                phase_error_bound=ERROR,
                radius=Q(1, 12),
            )
        ]
        * 2
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"initial_zero_winding_certified": False},
        {"formation_certified_by_class": (False, True)},
        {"source_budget_certified_by_class": (False, True)},
    ],
)
def test_failed_source_is_not_replaced_by_positive_endpoint_margins(
    recovery, monkeypatch, changes
):
    source = replace(recovery.source, **changes)
    monkeypatch.setattr(owner, "_assess_constitutive_class_pair", lambda **kw: source)
    result = _assess()
    assert result.decay_certified
    assert result.endpoint_form_margin > 0 and result.endpoint_phase_margin > 0
    assert not result.source_certified and not result.recovery_certified
    assert result.unresolved_conditions == ("original_class_one_source_not_certified",)


def test_missing_decay_is_unavailable_and_keeps_radii_absent(recovery, monkeypatch):
    monkeypatch.setattr(
        owner, "_assess_constitutive_class_pair", lambda **kw: recovery.source
    )
    result = _assess(quarter_dwell=0)
    assert result.source_certified and not result.decay_certified
    assert not result.recovery_certified
    assert result.decay_exponent == 0 and result.decay_exponent_margin == -512
    assert result.exact_decay_upper_bound is None
    assert result.returned_lyapunov_upper_bound is None
    assert result.endpoint_form_norm_squared_upper_bound is None
    assert result.endpoint_phase_norm_squared_upper_bound is None
    assert "declared_dwell_decay_not_certified" in result.unresolved_conditions


def test_exponent_equality_admitted_without_large_exponent_policy(
    recovery, monkeypatch
):
    monkeypatch.setattr(
        owner, "_assess_constitutive_class_pair", lambda **kw: recovery.source
    )
    result = _assess(quarter_dwell=2 * 512 / recovery.lyapunov_decay_rate)
    assert result.decay_exponent_margin == 0
    assert result.decay_certified and result.recovery_certified
    # Integer power construction does not evaluate an exponential at this
    # exponent, so the old exponential evaluator's 4096 cap is irrelevant.
    result = _assess(quarter_dwell=Q(10**30))
    assert result.decay_exponent > 4096
    assert result.exact_decay_upper_bound == Q(1, 2**512)
    assert result.recovery_certified


@pytest.mark.parametrize("failed_coordinate", ["form", "phase"])
def test_strict_endpoint_margin_is_required(recovery, monkeypatch, failed_coordinate):
    monkeypatch.setattr(
        owner, "_assess_constitutive_class_pair", lambda **kw: recovery.source
    )
    q2 = recovery.endpoint_radius**2

    def nonstrict_return(**kwargs):
        return (
            Q(0),
            (q2 if failed_coordinate == "form" else Q(0)),
            (q2 if failed_coordinate == "phase" else Q(0)),
        )

    monkeypatch.setattr(owner, "_sine_lyapunov_return_squared", nonstrict_return)
    result = _assess()
    assert result.source_certified and result.decay_certified
    assert not result.recovery_certified
    assert result.unresolved_conditions == (
        f"actual_{failed_coordinate}_handoff_radius_not_certified",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        (field, value)
        for field in (
            "form_error_bound",
            "phase_error_bound",
            "quarter_dwell",
            "endpoint_radius",
            "decay_power",
        )
        for value in (True, float("nan"), float("inf"), "1")
    ]
    + [
        ("form_error_bound", Q(-1)),
        ("phase_error_bound", Q(-1)),
        ("quarter_dwell", Q(-1)),
        ("endpoint_radius", Q(0)),
        ("endpoint_radius", Q(-1)),
        ("decay_power", Q(0)),
        ("decay_power", Q(-1)),
        ("decay_power", Q(3, 2)),
        ("decay_power", 4097),
    ],
)
def test_invalid_primitives_reject_before_source_admission(monkeypatch, field, value):
    def forbidden(**kwargs):
        pytest.fail("source construction preceded primitive admission")

    monkeypatch.setattr(owner, "_assess_constitutive_class_pair", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: value})


@pytest.mark.parametrize("extra", [{"source": None}, {"deformation": 1}])
def test_reports_and_different_laws_are_not_accepted(extra):
    with pytest.raises(TypeError):
        _assess(**extra)
