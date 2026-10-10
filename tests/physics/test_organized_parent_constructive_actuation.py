"""Independent static controls for the prescribed complete-state actuation."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _organized_parent_constructive_actuation as owner


_ERROR = Q(1, 10**10)
_ZERO = (Q(0),) * 36
_EDGES = tuple(
    sorted(
        {
            tuple(sorted((18 * g + 9 * a + j, 18 * g + 9 * a + (j + 1) % 9)))
            for g in range(2)
            for a in range(2)
            for j in range(9)
        }
        | {(18 * g + j, 18 * g + 9 + j) for g in range(2) for j in range(9)}
        | {(9 * a + j, 18 + 9 * a + j) for a in range(2) for j in range(9)}
    )
)


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


def _laplacian(values):
    result = [0] * 36
    for left, right in _EDGES:
        difference = values[left] - values[right]
        result[left] += difference
        result[right] -= difference
    return tuple(result)


def _assess(**changes):
    return owner._assess_organized_parent_constructive_actuation(
        **(
            dict(form_error_bounds=(_ERROR,) * 4, phase_error_bounds=(_ERROR,) * 4)
            | changes
        )
    )


@pytest.fixture(scope="module")
def realization():
    return _assess()


@pytest.fixture(scope="module")
def asymmetric_state():
    # Retain nonzero common modes, unequal child means and fine directions.
    forms = tuple(
        Q(1, 7)
        + Q(2 * g - 1, 10**6)
        + Q((a + 1) * (j - 4), 10**7)
        + Q((g + 1) * (3 * j * j - 68), 3 * 10**8)
        for g in range(2)
        for a in range(2)
        for j in range(9)
    )
    offsets = tuple(
        Q(2, 7) + Q(2 * g - 1, 100) + Q(2 * a - 1, 70) + Q((g + a + 1) * j * j, 100000)
        for g in range(2)
        for a in range(2)
        for j in range(9)
    )
    return forms, offsets


def _integrated_product(left, right):
    return sum(
        (a * b / Q(i + j + 1) for i, a in enumerate(left) for j, b in enumerate(right)),
        Q(0),
    )


def test_graph_inverse_and_polynomial_gram_integrals_are_exact():
    path = owner._evaluate_parent_actuation_path(
        form=_ZERO, phase_offsets=_ZERO, path_fraction=Q(0)
    )
    assert len(_EDGES) == 72
    assert {
        tuple(sorted((i, j)))
        for i, j, weight in path.geometry.weighted_edges
        if weight == 1
    } == set(_EDGES)
    assert _laplacian(path.laplacian_inverse_displacement) == path.phase_displacement
    assert sum(path.laplacian_inverse_displacement) == sum(path.phase_displacement) == 0
    p, q = (Q(0), Q(6), Q(-6)), (Q(1), Q(-6), Q(6))
    assert _integrated_product(p, p) == Q(6, 5)
    assert _integrated_product(p, q) == -Q(1, 5)
    assert _integrated_product(q, q) == Q(1, 5)
    assert _integrated_product(p, (Q(1),)) == 1
    assert _integrated_product(q, (Q(1),)) == 0


@pytest.mark.parametrize("fraction", (Q(0), Q(1, 6), Q(1, 2), Q(5, 6), Q(1)))
def test_asymmetric_complete_path_rows_inputs_clocks_and_storage(
    fraction, asymmetric_state, mp
):
    forms, offsets = asymmetric_state
    path = owner._evaluate_parent_actuation_path(
        form=forms, phase_offsets=offsets, path_fraction=fraction
    )
    assert path.form == forms and path.phase_offsets == offsets
    assert path.initial_form_gradient == _laplacian(forms)
    assert path.form_mean == sum(forms) / 36 != 0
    gamma, h, e = 1 / (1023 * mp.pi), mp.mpf(4), mp.mpf(1023) / 1024
    v = _mp(mp, fraction)
    x0, theta0 = tuple(_mp(mp, x) for x in forms), tuple(_mp(mp, x) for x in offsets)
    k = tuple(
        mp.mpf(2 * (i // 9 % 2) - 1) / 4 if i < 18 else mp.mpf(0) for i in range(36)
    )
    inverse = tuple(_mp(mp, x) for x in path.laplacian_inverse_displacement)
    mean = sum(x0) / 36
    b = tuple(h * value / (2 * gamma) for value in _laplacian(x0))

    def state_at(value):
        p, q = 3 * value**2 - 2 * value**3, value - 3 * value**2 + 2 * value**3
        dp, dq = 6 * value - 6 * value**2, 1 - 6 * value + 6 * value**2
        x = tuple(
            mean + 2 * gamma * inv * dp / h + (origin - mean) * dq
            for inv, origin in zip(inverse, x0)
        )
        theta = tuple(
            origin + delta * p + slope * q for origin, delta, slope in zip(theta0, k, b)
        )
        return x, theta

    x, offsets_now = state_at(v)
    # Complex differentiation checks rates without reusing implementation derivatives.
    step = mp.mpf("1e-70")
    complex_x, complex_theta = state_at(v + mp.j * step)
    dx_v = tuple(mp.im(value) / step for value in complex_x)
    dt_v = tuple(mp.im(value) / step for value in complex_theta)
    dx_tau = tuple(gamma**2 * value / (2 * h) for value in dx_v)
    dt_tau = tuple(gamma**2 * value / (2 * h) for value in dt_v)
    gradient = _laplacian(x)
    theta = tuple(
        2 * mp.pi * (i % 9 - 4) / 9 + value for i, value in enumerate(offsets_now)
    )
    current = [mp.mpf(0)] * 36
    for left, right in _EDGES:
        flow = mp.sin(theta[right] - theta[left])
        current[left] += flow
        current[right] -= flow
    forcing = tuple(
        rate + (lap - gamma * sine) / 4
        for rate, lap, sine in zip(dx_tau, gradient, current)
    )
    for i in range(36):
        expected = {
            "initial_velocity_coefficient_bounds": b[i],
            "form_bounds": x[i],
            "phase_offset_bounds": offsets_now[i],
            "path_form_gradient_bounds": gradient[i],
            "phase_current_bounds": current[i],
            "form_tau_rate_bounds": dx_tau[i],
            "phase_tau_rate_bounds": dt_tau[i],
            "form_slow_rate_bounds": dx_v[i] / h,
            "phase_slow_rate_bounds": dt_v[i] / h,
            "form_original_rate_bounds": e * dx_tau[i],
            "phase_original_rate_bounds": e * dt_tau[i],
            "form_input_tau_bounds": forcing[i],
            "form_input_slow_bounds": 2 * forcing[i] / gamma**2,
            "form_input_original_bounds": e * forcing[i],
        }
        for field, value in expected.items():
            _contains(mp, getattr(path, field)[i], value)
        assert abs(dt_tau[i] - gamma * gradient[i] / 4) < mp.mpf("1e-135")
        if fraction in (0, 1):
            assert abs(x[i] - x0[i]) < mp.mpf("1e-170")
            assert abs(offsets_now[i] - theta0[i] - k[i] * fraction) < mp.mpf("1e-170")
            assert abs(dt_tau[i] - gamma * _laplacian(x0)[i] / 4) < mp.mpf("1e-135")
    assert abs(sum(x) - sum(x0)) < mp.mpf("1e-170")
    assert abs(sum(offsets_now) - sum(theta0)) < mp.mpf("1e-170")
    assert abs(sum(forcing)) < mp.mpf("1e-135")
    # Differentiate the complete storage at this static state, not a trajectory.
    energy_complex = sum(
        (x[right] + mp.j * step * dx_tau[right] - x[left] - mp.j * step * dx_tau[left])
        ** 2
        / 2
        + 1
        - mp.cos(
            theta[right]
            + mp.j * step * dt_tau[right]
            - theta[left]
            - mp.j * step * dt_tau[left]
        )
        for left, right in _EDGES
    )
    input_power = sum(lap * force for lap, force in zip(gradient, forcing))
    loss_rate = sum(lap**2 for lap in gradient) / 4
    assert abs(mp.im(energy_complex) / step - (input_power - loss_rate)) < mp.mpf(
        "1e-130"
    )
    # Integrate the velocity polynomial with independently computed Gram coefficients.
    p, q = (Q(0), Q(6), Q(-6)), (Q(1), Q(-6), Q(6))
    loss = (
        2
        / h
        * sum(
            delta**2 * _mp(mp, _integrated_product(p, p))
            + 2 * delta * slope * _mp(mp, _integrated_product(p, q))
            + slope**2 * _mp(mp, _integrated_product(q, q))
            for delta, slope in zip(k, b)
        )
    )
    _contains(mp, path.continuous_loss_bounds, loss)


def test_whole_path_polynomial_extrema_and_full_source_bounds(realization, mp):
    for v in ((3 - mp.sqrt(3)) / 6, (3 + mp.sqrt(3)) / 6):
        q = v - 3 * v**2 + 2 * v**3
        assert abs(abs(q) - 1 / (6 * mp.sqrt(3))) < mp.mpf("1e-170")
        assert abs(q) < mp.mpf(1) / 10
    assert 1 - 6 * Q(1, 2) + 6 * Q(1, 2) ** 2 == -Q(1, 2)
    report, source = realization, realization.acquisition
    assert report.full_path_form_error_upper_bound == source.full_form_error_upper_bound
    assert report.full_path_phase_error_upper_bound == (
        source.full_phase_error_upper_bound
        + 8 * source.full_form_error_upper_bound / (5 * source.geometry.gamma_bounds.lo)
    )
    assert report.full_path_phase_error_upper_bound < Q(1, 10**55)
    assert report.child_positive_gap_margin > 0 and report.child_acute_gap_margin > 0
    assert report.initial_velocity_coefficient_norm_upper_bound < Q(1, 4)
    assert report.continuous_loss_upper_bound < Q(59, 80)
    assert report.continuous_loss_lower_bound == Q(9, 16)
    assert report.input_work_lower_bound > 0
    assert report.input_work_upper_bound < Q(5, 2) < report.work_budget
    assert report.work_budget_margin == 3 - report.input_work_upper_bound > Q(1, 2)
    assert (
        report.event_work_error_upper_bound == 18 * source.full_phase_error_upper_bound
    )
    assert source.relative_preparation_dimension == 64
    for child in source.child_sources:
        assert child.form_error_bound == child.phase_error_bound == _ERROR


def test_exact_release_times_and_separate_source_admission(realization, mp):
    report = realization
    assert report.source_certified and report.conditional_realization_certified
    assert report.child_identities_retained and report.work_certified
    assert report.realization_certified and report.release_certified and report.admitted
    assert not report.unresolved_conditions
    assert report.source_slow_time == 4 and report.slow_duration == 4
    assert report.release_slow_time == 8
    assert report.readout_slow_time - report.release_slow_time == 1
    assert report.recovery_slow_time - report.release_slow_time == 4
    assert report.endpoint_form_change == _ZERO
    gamma = 1 / (1023 * mp.pi)
    _contains(mp, report.elapsed_scaled_duration_bounds, 8 / gamma**2)
    _contains(mp, report.elapsed_original_duration_bounds, 8 / gamma**2 * 1024 / 1023)
    with pytest.raises(FrozenInstanceError):
        report.realization_certified = False


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
@pytest.mark.parametrize("bad", (True, "0", float("nan"), float("inf")))
def test_every_path_coordinate_is_admitted_before_geometry(field, bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("malformed primitive reached geometry")

    monkeypatch.setattr(owner, "_derive_repeated_composition", forbidden)
    values = dict(form=_ZERO, phase_offsets=_ZERO, path_fraction=Q(1, 2))
    vector = list(_ZERO)
    vector[-1] = bad
    values[field] = vector
    with pytest.raises((TypeError, ValueError)):
        owner._evaluate_parent_actuation_path(**values)


@pytest.mark.parametrize(
    "bad", (True, "0", float("nan"), float("inf"), -Q(1, 10**100), 1 + Q(1, 10**100))
)
def test_path_fraction_is_admitted_before_geometry(bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("malformed path clock reached geometry")

    monkeypatch.setattr(owner, "_derive_repeated_composition", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner._evaluate_parent_actuation_path(
            form=_ZERO, phase_offsets=_ZERO, path_fraction=bad
        )


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
@pytest.mark.parametrize("bad", ((), _ZERO[:-1], _ZERO + (Q(0),), "0" * 36, {Q(0)}))
def test_path_vector_shape_rejects_before_geometry(field, bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("malformed vector reached geometry")

    monkeypatch.setattr(owner, "_derive_repeated_composition", forbidden)
    values = dict(form=_ZERO, phase_offsets=_ZERO, path_fraction=Q(0))
    values[field] = bad
    with pytest.raises((TypeError, ValueError)):
        owner._evaluate_parent_actuation_path(**values)


@pytest.mark.parametrize("field", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize("bad", (True, "0", float("nan"), float("inf"), -Q(1, 10**100)))
def test_original_primitives_reject_before_source_reconstruction(
    field, bad, monkeypatch
):
    def forbidden(**kwargs):
        pytest.fail("malformed source reached reconstruction")

    monkeypatch.setattr(owner, "_assess_organized_parent_restoring_response", forbidden)
    values = [_ERROR] * 4
    values[-1] = bad
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: values})


@pytest.mark.parametrize(
    "dependency", ("sources_certified", "entry_certified", "child_identities_retained")
)
def test_unavailable_source_cannot_certify_realization(
    dependency, realization, monkeypatch
):
    source = replace(realization.acquisition, **{dependency: False})
    restoring = replace(realization.restoring_response, acquisition=source)
    monkeypatch.setattr(
        owner, "_assess_organized_parent_restoring_response", lambda **kwargs: restoring
    )
    report = _assess()
    assert report.conditional_realization_certified
    assert not report.source_certified and not report.realization_certified
    assert (
        not report.work_certified
        and not report.release_certified
        and not report.admitted
    )
    assert report.status == "unavailable" and report.unresolved_conditions


@pytest.mark.parametrize(
    "dependency",
    ("child_identities_retained", "recovery_certified", "prediction_certified"),
)
def test_later_response_flags_control_only_release(
    dependency, realization, monkeypatch
):
    restoring = replace(realization.restoring_response, **{dependency: False})
    monkeypatch.setattr(
        owner, "_assess_organized_parent_restoring_response", lambda **kwargs: restoring
    )
    report = _assess()
    assert report.source_certified and report.realization_certified
    assert report.work_certified and report.child_identities_retained
    assert not report.release_certified and not report.admitted


def test_unrelated_old_overall_and_event_verdicts_do_not_replace_premises(
    realization, monkeypatch
):
    source = replace(
        realization.acquisition,
        admitted=False,
        prediction_certified=False,
        status="unavailable",
    )
    restoring = replace(
        realization.restoring_response,
        acquisition=source,
        admitted=False,
        source_certified=False,
        contrast_certified=False,
        status="unavailable",
    )
    monkeypatch.setattr(
        owner, "_assess_organized_parent_restoring_response", lambda **kwargs: restoring
    )
    report = _assess()
    assert (
        report.source_certified
        and report.realization_certified
        and report.release_certified
    )


@pytest.mark.parametrize(
    "margin,allowed", ((-Q(1, 2**100), False), (Q(0), True), (Q(1, 2**100), True))
)
def test_net_work_budget_boundary_is_inclusive_not_an_obstruction(
    margin, allowed, realization, monkeypatch
):
    # Exercise the decision inequality only; the frozen protocol remains h=4,W=3.
    monkeypatch.setattr(
        owner,
        "_assess_organized_parent_restoring_response",
        lambda **kwargs: realization.restoring_response,
    )
    monkeypatch.setattr(
        owner, "_WORK_BUDGET", realization.input_work_upper_bound + margin
    )
    report = _assess()
    assert report.work_budget_margin == margin
    assert report.work_certified is allowed and report.realization_certified is allowed
    if not allowed:
        assert report.status == "unavailable"
        assert (
            "forced_path_input_work_budget_not_certified"
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
def test_frozen_protocol_does_not_accept_alternate_inputs(extra):
    with pytest.raises(TypeError):
        _assess(**{extra: 0})
