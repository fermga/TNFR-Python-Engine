"""Static contract controls for the all-input continuous-actuation obstruction."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import I
from tnfr.physics import _organized_parent_continuous_actuation as owner
from tnfr.physics import _organized_parent_restoring_response as intervention
from tnfr.physics import _repeated_organized_composition as composition


_ERROR = Q(1, 10**10)


@pytest.fixture(scope="module", autouse=True)
def no_scientific_regeneration():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 180
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, value):
    assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)


def _assess(**changes):
    return owner._assess_organized_parent_continuous_actuation(
        **(
            dict(form_error_bounds=(_ERROR,) * 4, phase_error_bounds=(_ERROR,) * 4)
            | changes
        )
    )


@pytest.fixture(scope="module")
def obstruction():
    return _assess()


def _laplacian(values, edges):
    result = [Q(0)] * 36
    for left, right, weight in edges:
        difference = weight * (values[left] - values[right])
        result[left] += difference
        result[right] -= difference
    return tuple(result)


def test_fixed_actual_endpoint_displacement_and_global_phase_constraint(obstruction):
    forms = tuple(Q(i - 17, 10000) for i in range(36))
    phases = tuple(Q((i % 9) ** 2, 10000) for i in range(36))
    after_x, after_theta = intervention._apply_parent_registration_intervention(
        form=forms, phase_offsets=phases
    )
    displacement = tuple(right - left for left, right in zip(phases, after_theta))
    assert after_x == forms
    assert displacement == obstruction.phase_displacement
    assert displacement == (-Q(1, 4),) * 9 + (Q(1, 4),) * 9 + (Q(0),) * 18
    assert sum(displacement) == obstruction.phase_displacement_sum == 0
    assert (
        sum(value**2 for value in displacement)
        == obstruction.phase_displacement_norm_squared
        == Q(9, 8)
    )
    assert obstruction.endpoint_form_change == (Q(0),) * 36
    assert obstruction.acquisition.relative_preparation_dimension == 64
    for child in obstruction.acquisition.child_sources:
        assert child.form_error_bound == child.phase_error_bound == _ERROR
        assert child.scaled_handoff_time == 100 + 10**13


def test_global_loss_bound_and_both_structural_clock_changes(obstruction, mp):
    gamma, e = 1 / (1023 * mp.pi), mp.mpf(1023) / 1024
    norm_squared = mp.mpf(9) / 8
    tau_duration = 2 / gamma**2
    original_duration = tau_duration / e
    _contains(mp, obstruction.elapsed_scaled_duration_bounds, tau_duration)
    _contains(mp, obstruction.elapsed_original_duration_bounds, original_duration)
    # In the original clock theta_t=e*gamma*Lx/4, and the loss rate
    # is e*||Lx||^2/4. Transforming only a duration would give a wrong bound.
    tau_loss = 4 * norm_squared / (gamma**2 * tau_duration)
    original_loss = 4 * norm_squared / (e * gamma**2 * original_duration)
    slow_loss = 2 * norm_squared
    assert abs(tau_loss - slow_loss) < mp.mpf("1e-160")
    assert abs(original_loss - slow_loss) < mp.mpf("1e-160")
    assert obstruction.necessary_continuous_loss_lower_bound == Q(9, 4)
    assert obstruction.slow_duration == 1 and obstruction.work_budget == 3


@pytest.mark.parametrize(
    "gain,common",
    ((Q(0), Q(0)), (Q(1, 8), Q(0)), (Q(1, 2), Q(0)), (-Q(1, 8), Q(0)), (Q(0), Q(1, 7))),
)
def test_complete_forced_rows_have_signed_conjugate_input_power(gain, common, mp):
    forms = tuple(
        Q(2 * g - 1, 30) + Q(2 * a - 1, 40) + Q(j - 4, 300)
        for g in range(2)
        for a in range(2)
        for j in range(9)
    )
    offsets = tuple(
        Q(2 * g - 1, 200) + Q(2 * a - 1, 100) + Q((g + 1) * (j * j), 20000)
        for g in range(2)
        for a in range(2)
        for j in range(9)
    )
    observed = composition._observe_repeated_composition(
        interface="matching", form=forms, phase_offsets=offsets
    )
    edges = observed.geometry.weighted_edges
    q = _laplacian(forms, edges)
    assert q == observed.form_gradient
    forcing = tuple(gain * value + common for value in q)
    power = sum((value * force for value, force in zip(q, forcing)), Q(0))
    loss = sum((value**2 for value in q), Q(0)) / 4
    assert power == gain * 4 * loss
    # A uniform nonzero form input has zero storage power on this support.
    # Negative input power is extraction, not a nonnegative forcing norm.
    if common:
        assert sum(value**2 for value in forcing) > 0 and power == 0
    if gain < 0:
        assert power < 0
    gamma = 1 / (1023 * mp.pi)
    theta = tuple(
        2 * mp.pi * (i % 9 - 4) / 9 + _mp(mp, value) for i, value in enumerate(offsets)
    )
    current = [mp.mpf(0)] * 36
    for left, right, weight in edges:
        flow = _mp(mp, weight) * mp.sin(theta[right] - theta[left])
        current[left] += flow
        current[right] -= flow
    x = tuple(_mp(mp, value) for value in forms)
    dx = tuple(
        (-_mp(mp, value) + gamma * flow) / 4 + _mp(mp, force)
        for value, flow, force in zip(q, current, forcing)
    )
    dt = tuple(gamma * _mp(mp, value) / 4 for value in q)
    for i in range(36):
        _contains(mp, observed.form_rate_bounds[i] + forcing[i], dx[i])
        _contains(mp, observed.phase_rate_bounds[i], dt[i])
    # Differentiate the complete storage independently at a single static
    # state. This complex directional evaluation is not an evolved trajectory.
    step = mp.mpf("1e-60")
    state_x = tuple(value + mp.j * step * rate for value, rate in zip(x, dx))
    state_theta = tuple(value + mp.j * step * rate for value, rate in zip(theta, dt))
    energy = sum(
        _mp(mp, weight)
        * (
            (state_x[right] - state_x[left]) ** 2 / 2
            + 1
            - mp.cos(state_theta[right] - state_theta[left])
        )
        for left, right, weight in edges
    )
    derivative = mp.im(energy) / step
    assert abs(derivative - _mp(mp, power - loss)) < mp.mpf("1e-110")
    # The source and BOTH rows acquire the same clock factor.
    e = mp.mpf(1023) / 1024
    original_power = sum(
        _mp(mp, value) * e * _mp(mp, force) for value, force in zip(q, forcing)
    )
    slow_power = sum(
        _mp(mp, value) * 2 * _mp(mp, force) / gamma**2
        for value, force in zip(q, forcing)
    )
    assert abs(original_power / e - _mp(mp, power)) < mp.mpf("1e-160")
    assert abs(slow_power * gamma**2 / 2 - _mp(mp, power)) < mp.mpf("1e-160")


def test_fresh_source_work_and_loss_exclude_all_admitted_inputs(obstruction):
    report = obstruction
    assert report.source_certified and report.conditional_obstruction_certified
    assert report.obstruction_certified and not report.unresolved_conditions
    assert (
        report.event_work_error_upper_bound
        == 18 * report.acquisition.full_phase_error_upper_bound
    )
    assert report.event_work_residual_bounds == (
        -report.event_work_error_upper_bound,
        report.event_work_error_upper_bound,
    )
    assert report.endpoint_storage_change_lower_bound > 1
    assert (
        report.input_work_lower_bound
        == report.endpoint_storage_change_lower_bound + Q(9, 4)
    )
    assert report.work_budget_deficit == report.input_work_lower_bound - 3 > Q(1, 4)
    with pytest.raises(FrozenInstanceError):
        report.obstruction_certified = False


def test_endpoint_forms_exclude_constant_velocity_equality(obstruction, mp):
    mode = tuple(
        (2 * g - 1) * (2 * a - 1) for g in range(2) for a in range(2) for _ in range(9)
    )
    assert _laplacian(mode, obstruction.acquisition.geometry.weighted_edges) == tuple(
        4 * value for value in mode
    )
    mixed_displacement = (
        sum(value * delta for value, delta in zip(mode, obstruction.phase_displacement))
        / 36
    )
    assert mixed_displacement == -Q(1, 8)
    gamma = 1 / (1023 * mp.pi)
    required_mixed_form = _mp(mp, mixed_displacement) / (gamma * (2 / gamma**2))
    assert abs(required_mixed_form + gamma / 16) < mp.mpf("1e-160")
    actual_mixed_form_allowance = (
        obstruction.acquisition.full_form_error_upper_bound / 6
    )
    assert _mp(mp, actual_mixed_form_allowance) < abs(required_mixed_form)


@pytest.mark.parametrize("field", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize("bad", (True, "0", float("nan"), float("inf"), -Q(1, 10**100)))
def test_primitive_rejection_precedes_acquired_source_construction(
    field, bad, monkeypatch
):
    def forbidden(**kwargs):
        pytest.fail("invalid primitive reached source reconstruction")

    monkeypatch.setattr(owner, "_assess_organized_parent_acquisition", forbidden)
    values = [_ERROR] * 4
    values[-1] = bad
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: tuple(values)})


@pytest.mark.parametrize("field", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize("bad", ((), (_ERROR,) * 3, (_ERROR,) * 5, "0000", {_ERROR}))
def test_source_shapes_reject_before_reconstruction(field, bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid source shape reached reconstruction")

    monkeypatch.setattr(owner, "_assess_organized_parent_acquisition", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


@pytest.mark.parametrize(
    "dependency", ("sources_certified", "entry_certified", "child_identities_retained")
)
def test_missing_actual_source_evidence_withholds_the_obstruction(
    dependency, obstruction, monkeypatch
):
    source = replace(obstruction.acquisition, **{dependency: False})
    monkeypatch.setattr(
        owner, "_assess_organized_parent_acquisition", lambda **kwargs: source
    )
    report = _assess()
    assert report.conditional_obstruction_certified
    assert not report.source_certified and not report.obstruction_certified
    assert report.status == "unavailable" and report.unresolved_conditions


def test_later_unforced_prediction_is_not_an_actuation_premise(
    obstruction, monkeypatch
):
    source = replace(
        obstruction.acquisition,
        prediction_certified=False,
        admitted=False,
        status="unavailable",
    )
    monkeypatch.setattr(
        owner, "_assess_organized_parent_acquisition", lambda **kwargs: source
    )
    report = _assess()
    assert report.source_certified and report.obstruction_certified


@pytest.mark.parametrize(
    "endpoint_lower,excluded",
    ((Q(3, 4) - Q(1, 2**100), False), (Q(3, 4), False), (Q(3, 4) + Q(1, 2**100), True)),
)
def test_budget_exclusion_is_strict_not_a_realizability_verdict(
    endpoint_lower, excluded, obstruction, monkeypatch
):
    # Exercise the arithmetic decision boundary; these substituted work
    # intervals are not claimed to describe the actual frozen source.
    monkeypatch.setattr(
        owner,
        "_assess_organized_parent_acquisition",
        lambda **kwargs: obstruction.acquisition,
    )
    monkeypatch.setattr(
        owner,
        "_parent_registration_event_work_bounds",
        lambda **kwargs: (I(endpoint_lower), Q(0), I(endpoint_lower)),
    )
    report = _assess()
    assert report.source_certified
    assert report.conditional_obstruction_certified is excluded
    assert report.obstruction_certified is excluded
    if not excluded:
        assert report.status == "unavailable"
        assert (
            "strict_input_work_budget_exclusion_not_certified"
            in report.unresolved_conditions
        )


@pytest.mark.parametrize(
    "extra",
    (
        "source_report",
        "acquisition",
        "Gamma",
        "forcing",
        "phase_input",
        "duration",
        "work_budget",
        "target",
        "trajectory",
    ),
)
def test_frozen_input_class_budget_and_endpoint_cannot_be_replaced(extra):
    with pytest.raises(TypeError):
        _assess(**{extra: 0})


@pytest.mark.parametrize(
    "field", ("inner_gap_bounds", "outer_gap_bounds", "phase_error_bound")
)
@pytest.mark.parametrize("bad", (True, "0", float("nan"), float("inf")))
def test_shared_event_work_arithmetic_rejects_malformed_inputs(field, bad):
    values = dict(
        inner_gap_bounds=I(Q(1, 100)),
        outer_gap_bounds=I(Q(1, 1000)),
        phase_error_bound=Q(1, 10**70),
    )
    values[field] = bad
    with pytest.raises((TypeError, ValueError)):
        intervention._parent_registration_event_work_bounds(**values)


def test_shared_event_work_keeps_signed_subgrid_error_and_rejects_negative_radius():
    tiny = Q(1, 10**1000)
    nominal, error, actual = intervention._parent_registration_event_work_bounds(
        inner_gap_bounds=I(Q(1, 100)),
        outer_gap_bounds=I(Q(1, 1000)),
        phase_error_bound=tiny,
    )
    assert error == 18 * tiny > 0
    assert actual.lo <= nominal.lo - error and actual.hi >= nominal.hi + error
    with pytest.raises(ValueError):
        intervention._parent_registration_event_work_bounds(
            inner_gap_bounds=I(0), outer_gap_bounds=I(0), phase_error_bound=-tiny
        )
