"""Independent static and synthetic controls without evaluating the frozen gate."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from types import SimpleNamespace

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.physics import _organized_alternative_outer_acquisition as owner
from tnfr.physics import _organized_outer_winding as sine_owner
from tnfr.physics import relational_sine_entry

_ERROR = Q(1, 10**10)
_SYNTHETIC_RADIUS = Q(1, 10**4)
_EDGES = tuple(
    (9 * c + j, 9 * c + (j + 1) % 9) for c in range(5) for j in range(9)
) + tuple((9 * c + j, 9 * ((c + 1) % 5) + j) for j in range(9) for c in range(5))


@pytest.fixture(scope="module", autouse=True)
def no_frozen_or_sine_assessment():
    def forbidden(*args, **kwargs):
        pytest.fail("own-law static control attempted a sine acquisition verdict")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner, "_assess_constitutive_recovery", forbidden)
        patch.setattr(sine_owner, "_assess_organized_outer_winding", forbidden)
        patch.setattr(sine_owner, "_derive_organized_outer_resource_bounds", forbidden)
        patch.setattr(sine_owner, "certify_sine_prepared_entry", forbidden)
        patch.setattr(relational_sine_entry, "certify_sine_prepared_entry", forbidden)
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


@pytest.fixture(scope="module")
def bounds():
    # This deliberately enlarged conditional error radius is not the actual
    # recovered source allowance. No isolated source or gate is evaluated.
    return owner._derive_organized_alternative_outer_bounds(
        recovered_radius=_SYNTHETIC_RADIUS
    )


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _potential(mp, angle):
    return 1 - mp.cos(angle) + (1 - mp.cos(9 * angle)) ** 2 / (6 * 9**4)


def _current(mp, angle):
    return mp.sin(angle) + mp.sin(9 * angle) * (1 - mp.cos(9 * angle)) / 2187


def _fields(mp, form, phase, edges=_EDGES):
    gradient, current = [mp.mpf(0)] * len(form), [mp.mpf(0)] * len(form)
    storage = mp.mpf(0)
    for a, b in edges:
        difference, angle = form[a] - form[b], phase[b] - phase[a]
        gradient[a] += difference
        gradient[b] -= difference
        edge_current = _current(mp, angle)
        current[a] += edge_current
        current[b] -= edge_current
        storage += difference**2 / 2 + _potential(mp, angle)
    return gradient, current, storage


def test_actual_potential_and_both_rows_retain_full_support_and_loss(mp):
    form = tuple(Q((7 * i) % 19 - 9, 64) for i in range(45))
    offsets = tuple(Q((11 * i) % 23 - 11, 256) for i in range(45))
    phase = tuple(
        2 * mp.pi * (i % 9 - 4) / 9 + _mp(mp, offset)
        for i, offset in enumerate(offsets)
    )
    gradient, current, storage = _fields(
        mp, tuple(_mp(mp, value) for value in form), phase
    )
    gamma = 1 / (1023 * mp.pi)
    dx = tuple((-a + gamma * b) / 4 for a, b in zip(gradient, current))
    dt = tuple(gamma * a / 4 for a in gradient)
    loss = sum(a**2 for a in gradient) / 4
    observed = owner._observe_organized_alternative_outer(
        form=form, phase_offsets=offsets
    )
    assert len(observed.geometry.weighted_edges) == 90
    assert observed.geometry.weighted_degrees == (Q(4),) * 45
    _contains(mp, observed.storage_bounds, storage)
    assert abs(_mp(mp, observed.storage_tau_rate) + loss) < mp.mpf("1e-80")
    assert observed.balance_residual_bounds.contains(0)
    assert abs(sum(dx)) < mp.mpf("1e-80") and abs(sum(dt)) < mp.mpf("1e-80")
    assert abs(sum(dx[:9])) > mp.mpf(".01")
    for index in range(45):
        _contains(mp, observed.phase_current_bounds[index], current[index])
        _contains(mp, observed.form_tau_rate_bounds[index], dx[index])
        _contains(mp, observed.phase_tau_rate_bounds[index], dt[index])
        _contains(
            mp, observed.form_original_rate_bounds[index], 1023 * dx[index] / 1024
        )
        _contains(
            mp, observed.phase_original_rate_bounds[index], 1023 * dt[index] / 1024
        )
    epsilon = mp.mpf("1e-35")
    advanced_form = tuple(_mp(mp, a) + mp.j * epsilon * b for a, b in zip(form, dx))
    advanced_phase = tuple(a + mp.j * epsilon * b for a, b in zip(phase, dt))
    derivative = mp.im(_fields(mp, advanced_form, advanced_phase)[2]) / epsilon
    assert abs(derivative + loss) < mp.mpf("1e-55")


def test_carrier_restricts_both_actual_law_rows_with_half_clock(mp):
    forms = tuple(Q(value, 7) for value in (-4, 7, 1, -9, 5))
    origins = tuple(Q(value, 11) for value in (3, -2, 1, 4, -6))
    x = tuple(_mp(mp, value) for value in forms)
    phase = tuple(_mp(mp, value) for value in origins)
    primitive = _fields(mp, x, phase, tuple((c, (c + 1) % 5) for c in range(5)))
    fine_x = tuple(value for value in x for _ in range(9))
    fine_phase = tuple(
        2 * mp.pi * (j - 4) / 9 + phase[c] for c in range(5) for j in range(9)
    )
    fine = _fields(mp, fine_x, fine_phase)
    gamma = 1 / (1023 * mp.pi)
    for i in range(45):
        assert abs(fine[0][i] - primitive[0][i // 9]) < mp.mpf("1e-80")
        assert abs(fine[1][i] - primitive[1][i // 9]) < mp.mpf("1e-80")
        assert abs(
            2 * (-fine[0][i] + gamma * fine[1][i]) / 4
            - (-primitive[0][i // 9] + gamma * primitive[1][i // 9]) / 2
        ) < mp.mpf("1e-80")
        assert abs(
            2 * gamma * fine[0][i] / 4 - gamma * primitive[0][i // 9] / 2
        ) < mp.mpf("1e-80")
    expected = 45 * _potential(mp, 2 * mp.pi / 9) + 9 * primitive[2]
    assert abs(fine[2] - expected) < mp.mpf("1e-80")


def test_target_is_stationary_but_outer_stiffness_is_not_sine_stiffness(mp, bounds):
    alpha, beta = 2 * mp.pi / 9, 2 * mp.pi / 5
    phase = tuple(alpha * (j - 4) + beta * (c - 2) for c in range(5) for j in range(9))
    _, current, storage = _fields(mp, (mp.mpf(0),) * 45, phase)
    assert max(map(abs, current)) < mp.mpf("1e-80")
    _contains(mp, bounds.target_storage_bounds, storage)
    actual_stiffness = mp.cos(beta) + (mp.cos(9 * beta) - mp.cos(18 * beta)) / 243
    _contains(mp, bounds.target_outer_stiffness_bounds, actual_stiffness)
    assert abs(actual_stiffness - mp.cos(beta)) > mp.mpf(".004")
    hessian = np.zeros((45, 45))
    for a, b in _EDGES:
        angle = phase[b] - phase[a]
        value = float(mp.cos(angle) + (mp.cos(9 * angle) - mp.cos(18 * angle)) / 243)
        hessian[a, a] += value
        hessian[b, b] += value
        hessian[a, b] -= value
        hessian[b, a] -= value
    eigenvalues = np.linalg.eigvalsh(hessian)
    assert abs(eigenvalues[0]) < 1e-12 and eigenvalues[1] > 0.3


def test_actual_jensen_faces_and_empty_outer_faces(mp, bounds):
    v9, v5 = 9 * _potential(mp, 2 * mp.pi / 9), 5 * _potential(mp, 2 * mp.pi / 5)
    positive_inner = _potential(mp, mp.pi / 2) + 8 * _potential(mp, 3 * mp.pi / 16)
    negative_inner = _potential(mp, mp.pi / 2) + 8 * _potential(mp, 5 * mp.pi / 16)
    positive_outer = _potential(mp, mp.pi / 2) + 4 * _potential(mp, 3 * mp.pi / 8)
    _contains(mp, bounds.internal_cycle_minimum_bounds, v9)
    _contains(mp, bounds.outer_cycle_minimum_bounds, v5)
    _contains(mp, bounds.internal_boundary_excess_bounds, positive_inner - v9)
    _contains(mp, bounds.outer_boundary_excess_bounds, positive_outer - v5)
    _contains(mp, bounds.internal_negative_face_storage_bounds, negative_inner)
    assert negative_inner > positive_inner and positive_outer > v5
    assert len(bounds.signed_face_excess_lower_bounds) == 180
    assert sum(bounds.signed_face_feasible) == 135
    assert bounds.signed_face_excess_lower_bounds[90::2] == (None,) * 45
    assert all(bounds.signed_face_feasible[:90])
    # A negative outer face would leave four acute gaps summing to 5*pi/2,
    # while their closed-acute maximum is only 2*pi.
    assert 4 * mp.pi / 2 < 5 * mp.pi / 2


def test_fixed_transit_bounds_use_own_force_and_direct_edge_cauchy(mp, bounds):
    gamma = 1 / (1023 * mp.pi)
    current, stiffness = 1 + mp.mpf(1) / 1458, 1 + mp.mpf(2) / 243
    assert bounds.current_absolute_upper_bound == Q(1459, 1458)
    assert bounds.current_lipschitz_upper_bound == Q(245, 243)
    rho = (1 + mp.mpf(50) / 16) ** -16
    assert mp.exp(-50) < rho
    a = rho * mp.sqrt(320) / mp.pi + 2 * gamma**2 * mp.sqrt(10) * current
    p = a + 100 * gamma**2 * mp.sqrt(10) * current
    assert a <= _mp(mp, bounds.primitive_scaled_form_radius_upper_bound)
    assert p <= _mp(mp, bounds.primitive_phase_radius_upper_bound)
    assert _mp(mp, bounds.primitive_edge_form_radius_upper_bound) >= a / gamma
    assert _mp(mp, bounds.primitive_node_form_radius_upper_bound) >= a / (
        gamma * mp.sqrt(2)
    )
    q = _mp(mp, _SYNTHETIC_RADIUS)
    assert _mp(mp, bounds.full_phase_error_upper_bound) >= 6 * q * (1 + gamma)
    assert _mp(
        mp, bounds.full_form_error_upper_bound
    ) >= 3 * q + 24 * stiffness * gamma * 6 * q * (1 + gamma)
    assert (
        bounds.attachment_residual_work_upper_bound
        == 10 * (1 + Q(245, 243)) * _SYNTHETIC_RADIUS**2
    )
    assert (
        bounds.control_initial_excess_upper_bound
        == 20 * (1 + Q(245, 243)) * _SYNTHETIC_RADIUS**2
    )
    assert not hasattr(bounds, "source_certified")


def test_potential_curvature_split_has_actual_endpoint_and_no_sine_transfer(mp, bounds):
    assert bounds.acute_convexity_split_margin > 0
    assert (
        _mp(mp, bounds.acute_convexity_split_margin)
        <= mp.cos(4 * mp.pi / 9) - mp.mpf(2) / 243
    )
    for angle in (mp.mpf(0), 4 * mp.pi / 9, 17 * mp.pi / 36, mp.pi / 2):
        curvature = mp.cos(angle) + (mp.cos(9 * angle) - mp.cos(18 * angle)) / 243
        assert curvature > 0
    assert abs(mp.cos(mp.pi / 2)) < mp.mpf("1e-80")
    endpoint_curvature = (mp.cos(9 * mp.pi / 2) - mp.cos(9 * mp.pi)) / 243
    assert abs(endpoint_curvature - mp.mpf(1) / 243) < mp.mpf("1e-80")


def _fake_child(*, form_error_bound, phase_error_bound, **kwargs):
    # This is deliberately not a source certificate: only wiring and
    # suppression behavior are tested, with actual child assessment blocked.
    return SimpleNamespace(
        deformation=Q(1),
        form_error_bound=form_error_bound,
        phase_error_bound=phase_error_bound,
        recovery_certified=False,
        endpoint_form_norm_squared_upper_bound=None,
        endpoint_phase_norm_squared_upper_bound=None,
        formation_time=Q(100),
        relaxation_duration=Q(10**13),
        source=SimpleNamespace(
            preparation_dimension=16, initial_storage_upper_bounds=(Q(1),)
        ),
    )


def test_all_five_actual_law_requests_are_fresh_and_missing_source_suppresses(
    monkeypatch, bounds
):
    calls = []
    forms = tuple(_ERROR * Q(c + 1, 5) for c in range(5))
    phases = tuple(_ERROR * Q(5 - c, 5) for c in range(5))

    def recorded(**kwargs):
        calls.append(kwargs)
        return _fake_child(**kwargs)

    monkeypatch.setattr(owner, "_assess_constitutive_recovery", recorded)
    monkeypatch.setattr(
        owner, "_derive_organized_alternative_outer_bounds", lambda **kwargs: bounds
    )
    report = owner._assess_organized_alternative_outer_acquisition(
        form_error_bounds=forms, phase_error_bounds=phases
    )
    assert calls == [
        dict(deformation=Q(1), form_error_bound=a, phase_error_bound=b)
        for a, b in zip(forms, phases)
    ]
    assert len({id(child) for child in report.child_sources}) == 5
    assert not report.source_certified and not report.admitted
    assert not report.outer_entry_certified and not report.capture_certified
    assert not report.child_winding_retained_during_transit
    assert (
        not report.control_capture_certified
        and not report.control_outer_zero_winding_retained
    )
    assert not report.converges_to_target
    assert report.status == "unavailable"
    with pytest.raises(FrozenInstanceError):
        report.admitted = True


@pytest.mark.parametrize("field", ("form_error_bounds", "phase_error_bounds"))
@pytest.mark.parametrize(
    "bad",
    (
        True,
        "0",
        None,
        float("nan"),
        float("inf"),
        complex(0),
        -Q(1, 10**1000),
        _ERROR + Q(1, 10**1000),
    ),
)
def test_bad_allowances_reject_before_any_source_or_geometry(field, bad, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached a source or geometry")

    monkeypatch.setattr(owner, "_assess_constitutive_recovery", forbidden)
    monkeypatch.setattr(owner, "_derive_organized_outer_winding", forbidden)
    arguments = dict(form_error_bounds=(_ERROR,) * 5, phase_error_bounds=(_ERROR,) * 5)
    arguments[field] = (_ERROR,) * 4 + (bad,)
    with pytest.raises((TypeError, ValueError)):
        owner._assess_organized_alternative_outer_acquisition(**arguments)


@pytest.mark.parametrize(
    "bad", ((), (0,) * 4, (0,) * 6, "00000", {0}, {i: 0 for i in range(5)})
)
def test_bad_shapes_reject_before_sources(bad, monkeypatch):
    monkeypatch.setattr(
        owner,
        "_assess_constitutive_recovery",
        lambda **kwargs: pytest.fail("shape reached source"),
    )
    with pytest.raises((TypeError, ValueError)):
        owner._assess_organized_alternative_outer_acquisition(
            form_error_bounds=bad, phase_error_bounds=(0,) * 5
        )


@pytest.mark.parametrize(
    "extra",
    (
        "source",
        "deformation",
        "horizon",
        "clock",
        "budget",
        "recovered_radius",
        "trajectory",
        "form_shifts",
    ),
)
def test_reports_and_retuned_protocol_are_not_admitted(extra):
    with pytest.raises(TypeError):
        owner._assess_organized_alternative_outer_acquisition(
            form_error_bounds=(0,) * 5, phase_error_bounds=(0,) * 5, **{extra: 0}
        )


@pytest.mark.parametrize(
    "bad", (True, "0", None, float("nan"), float("inf"), complex(0), -Q(1, 10**1000))
)
def test_invalid_resource_radius_rejects_before_geometry(bad, monkeypatch):
    monkeypatch.setattr(
        owner,
        "_derive_organized_outer_winding",
        lambda: pytest.fail("radius reached geometry"),
    )
    with pytest.raises((TypeError, ValueError)):
        owner._derive_organized_alternative_outer_bounds(recovered_radius=bad)


def test_tiny_exact_errors_survive_shared_heat_and_work_arithmetic(monkeypatch, bounds):
    # Reuse already derived conditional constants; no new physical/source
    # case is assessed. Exact arithmetic can represent subgrid effects.
    q = Q(1, 10**400)
    gamma, stiffness = (
        bounds.geometry.gamma_bounds.hi,
        bounds.current_lipschitz_upper_bound,
    )
    x, phase = owner._prepared_heat_error_radii(
        initial_form_error=3 * q,
        initial_phase_error=3 * q,
        gamma_upper=gamma,
        phase_growth_upper=Q(2),
        form_convolution_factor=24 * stiffness,
    )
    assert 0 < phase == 6 * q * (1 + gamma)
    assert x == 3 * q + 24 * stiffness * gamma * phase > 0
    assert 10 * (1 + stiffness) * q**2 > 0


def test_synthetic_failed_product_conditions_suppress_full_claims(monkeypatch, bounds):
    def child(**kwargs):
        item = _fake_child(**kwargs)
        item.recovery_certified = True
        item.endpoint_form_norm_squared_upper_bound = Q(0)
        item.endpoint_phase_norm_squared_upper_bound = Q(0)
        return item

    monkeypatch.setattr(owner, "_assess_constitutive_recovery", child)
    failed = replace(
        bounds,
        conditional_outer_entry_certified=False,
        conditional_capture_certified=False,
        control_conditional_capture_certified=False,
        fine_error_certified=False,
        barrier_certified=False,
    )
    monkeypatch.setattr(
        owner, "_derive_organized_alternative_outer_bounds", lambda **kwargs: failed
    )
    report = owner._assess_organized_alternative_outer_acquisition(
        form_error_bounds=(0,) * 5, phase_error_bounds=(0,) * 5
    )
    assert report.source_certified
    assert not report.capture_certified and not report.converges_to_target
    assert not report.outer_entry_certified
    assert not report.control_capture_certified
    assert not report.admitted and report.unresolved_conditions


@pytest.mark.parametrize(
    "field,value",
    (
        ("deformation", Q(0)),
        ("formation_time", Q(200)),
        ("relaxation_duration", Q(2 * 10**13)),
        ("endpoint_form_norm_squared_upper_bound", Q(1, 10**140)),
        ("endpoint_phase_norm_squared_upper_bound", Q(1, 10**140)),
        ("endpoint_form_norm_squared_upper_bound", None),
    ),
)
def test_one_synthetic_wrong_law_clock_or_nonstrict_source_suppresses(
    monkeypatch, bounds, field, value
):
    calls = []

    def child(**kwargs):
        item = _fake_child(**kwargs)
        item.recovery_certified = True
        item.endpoint_form_norm_squared_upper_bound = Q(0)
        item.endpoint_phase_norm_squared_upper_bound = Q(0)
        if len(calls) == 3:
            setattr(item, field, value)
        calls.append(kwargs)
        return item

    monkeypatch.setattr(owner, "_assess_constitutive_recovery", child)
    monkeypatch.setattr(
        owner, "_derive_organized_alternative_outer_bounds", lambda **kwargs: bounds
    )
    report = owner._assess_organized_alternative_outer_acquisition(
        form_error_bounds=(0,) * 5, phase_error_bounds=(0,) * 5
    )
    assert len(calls) == 5
    assert report.recovered_source_flags == (True, True, True, False, True)
    assert not report.source_certified
    assert not report.outer_entry_certified and not report.capture_certified
    assert not report.child_winding_retained and not report.converges_to_target
    assert not report.control_capture_certified
    assert not report.admitted
