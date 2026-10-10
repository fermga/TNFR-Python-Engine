"""Independent complete-law and finite-admission controls for two-stage C9s."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _repeated_organized_composition as owner

_ZERO = (Q(0),) * 36
_ERROR = Q(1, 10**10)


@pytest.fixture(scope="module", autouse=True)
def no_scientific_regeneration():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 120
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, expected):
    assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)
    assert _mp(mp, interval.width) < mp.mpf("1e-28")


def _edges(interface):
    cycle = tuple(
        (18 * p + 9 * a + j, 18 * p + 9 * a + (j + 1) % 9, Q(1))
        for p in range(2)
        for a in range(2)
        for j in range(9)
    )
    inner = tuple(
        (18 * p + j, 18 * p + 9 + j, Q(1)) for p in range(2) for j in range(9)
    )
    shifts = (0,) if interface == "matching" else (-1, 0, 1)
    outer = tuple(
        (9 * a + j, 18 + 9 * a + (j + d) % 9, Q(1, len(shifts)))
        for a in range(2)
        for j in range(9)
        for d in shifts
    )
    return cycle, inner, outer


@pytest.fixture(scope="module", params=("matching", "symmetric"))
def arbitrary_state(request):
    form = tuple(Q((7 * i) % 19 - 9, 32) for i in range(36))
    offsets = tuple(Q((11 * i) % 23 - 11, 256) for i in range(36))
    return owner._observe_repeated_composition(
        interface=request.param, form=form, phase_offsets=offsets
    )


def test_full_weighted_rows_both_groupings_storage_and_work(arbitrary_state, mp):
    report = arbitrary_state
    geometry = report.geometry
    cycle, inner, outer = _edges(geometry.interface)
    edges = cycle + inner + outer
    assert geometry.weighted_edges == edges
    assert geometry.weighted_degrees == (Q(4),) * 36
    support_degree = 4 if geometry.interface == "matching" else 6
    assert geometry.support_degrees == (support_degree,) * 36
    alpha, gamma = 2 * mp.pi / 9, 1 / (1023 * mp.pi)
    x = tuple(_mp(mp, value) for value in report.form)
    theta = tuple(
        alpha * (i % 9 - 4) + _mp(mp, value)
        for i, value in enumerate(report.phase_offsets)
    )
    gradient, current = [mp.mpf(0)] * 36, [mp.mpf(0)] * 36
    storage, form_storage = [], []
    for a, b, weight in edges:
        w, difference, gap = _mp(mp, weight), x[a] - x[b], theta[b] - theta[a]
        gradient[a] += w * difference
        gradient[b] -= w * difference
        current[a] += w * mp.sin(gap)
        current[b] -= w * mp.sin(gap)
        form_storage.append(w * difference**2 / 2)
        storage.append(form_storage[-1] + w * (1 - mp.cos(gap)))
    dx = tuple((-a + gamma * b) / 4 for a, b in zip(gradient, current))
    dy = tuple(gamma * a / 4 for a in gradient)
    for i in range(36):
        _contains(mp, report.phase_lift_bounds[i], theta[i])
        assert abs(_mp(mp, report.form_gradient[i]) - gradient[i]) < mp.mpf("1e-110")
        _contains(mp, report.phase_current_bounds[i], current[i])
        _contains(mp, report.form_rate_bounds[i], dx[i])
        _contains(mp, report.phase_rate_bounds[i], dy[i])
    for size, observed_x, observed_t in (
        (9, report.child_form_rate_bounds, report.child_phase_rate_bounds),
        (18, report.parent_form_rate_bounds, report.parent_phase_rate_bounds),
    ):
        for index, start in enumerate(range(0, 36, size)):
            _contains(mp, observed_x[index], sum(dx[start : start + size]) / size)
            _contains(mp, observed_t[index], sum(dy[start : start + size]) / size)
    loss = sum(value**2 for value in gradient) / 4
    assert abs(_mp(mp, report.dissipation) - loss) < mp.mpf("1e-110")
    assert report.storage_rate == -report.dissipation < 0
    direct = sum(a * b - c * d for a, b, c, d in zip(gradient, dx, current, dy))
    _contains(mp, report.storage_rate_from_rows_bounds, direct)
    assert abs(direct + loss) < mp.mpf("1e-110")
    assert report.balance_residual_bounds.contains(0)
    assert abs(_mp(mp, report.form_storage) - sum(form_storage)) < mp.mpf("1e-110")
    _contains(mp, report.storage_bounds, sum(storage))
    _contains(mp, report.internal_storage_bounds, sum(storage[:36]))
    _contains(mp, report.first_attachment_storage_bounds, sum(storage[36:54]))
    actual_work = sum(storage[54:])
    _contains(mp, report.outer_attachment_work_bounds, actual_work)
    kappa = 1 if geometry.interface == "matching" else (1 + 2 * mp.cos(alpha)) / 3
    u = _mp(mp, report.parent_form_means[1] - report.parent_form_means[0])
    delta = _mp(
        mp, report.parent_phase_offset_means[1] - report.parent_phase_offset_means[0]
    )
    nominal = 9 * u**2 + 18 * (1 - kappa * mp.cos(delta))
    _contains(mp, report.nominal_outer_work_bounds, nominal)
    low, high = report.outer_work_residual_bounds
    assert _mp(mp, low) <= actual_work - nominal <= _mp(mp, high)
    outer_current = (
        sum(_mp(mp, w) * mp.sin(theta[b] - theta[a]) for a, b, w in outer) / 18
    )
    _contains(mp, report.outer_mean_current_bounds, outer_current)
    _contains(mp, report.parent_form_rate_bounds[0], (u + gamma * outer_current) / 4)
    _contains(mp, report.parent_phase_rate_bounds[0], -gamma * u / 4)
    _contains(
        mp,
        report.projected_current_defect_bounds,
        outer_current - kappa * mp.sin(delta),
    )


@pytest.mark.parametrize("interface", ("matching", "symmetric"))
def test_first_stage_eight_state_law_keeps_distinct_child_coordinates(interface, mp):
    forms = (-Q(1, 10), Q(1, 5), Q(3, 10), -Q(1, 5))
    origins = (-Q(3, 100), Q(1, 100), Q(2, 25), Q(1, 20))
    report = owner._observe_repeated_composition(
        interface=interface,
        form=tuple(forms[i // 9] for i in range(36)),
        phase_offsets=tuple(origins[i // 9] for i in range(36)),
    )
    x, phi = tuple(_mp(mp, v) for v in forms), tuple(_mp(mp, v) for v in origins)
    gamma, alpha = 1 / (1023 * mp.pi), 2 * mp.pi / 9
    kappa = 1 if interface == "matching" else (1 + 2 * mp.cos(alpha)) / 3
    q = tuple(2 * forms[i] - forms[i ^ 1] - forms[i ^ 2] for i in range(4))
    s = tuple(
        mp.sin(phi[i ^ 1] - phi[i]) + kappa * mp.sin(phi[i ^ 2] - phi[i])
        for i in range(4)
    )
    assert report.child_form_means == forms
    assert report.child_phase_offset_means == origins
    for i in range(4):
        _contains(
            mp, report.child_form_rate_bounds[i], (-_mp(mp, q[i]) + gamma * s[i]) / 4
        )
        _contains(mp, report.child_phase_rate_bounds[i], gamma * _mp(mp, q[i]) / 4)
        for node in range(9 * i, 9 * (i + 1)):
            assert report.form_gradient[node] == q[i]
            _contains(mp, report.phase_current_bounds[node], s[i])
    inner, outer = ((0, 1), (2, 3)), ((0, 2), (1, 3))
    h8 = 36 * (1 - mp.cos(alpha))
    h8 += mp.mpf(9) / 2 * sum((x[b] - x[a]) ** 2 for a, b in inner + outer)
    h8 += 9 * sum(1 - mp.cos(phi[b] - phi[a]) for a, b in inner)
    h8 += 9 * sum(1 - kappa * mp.cos(phi[b] - phi[a]) for a, b in outer)
    _contains(mp, report.storage_bounds, h8)
    assert report.dissipation == Q(9, 4) * sum(value**2 for value in q)


@pytest.mark.parametrize("interface", ("matching", "symmetric"))
def test_exact_two_stage_rigid_projection_and_joint_chart(interface, mp):
    forms = (-Q(3, 10), Q(1, 5))
    origins = (-Q(1, 40), Q(1, 40))
    report = owner._observe_repeated_composition(
        interface=interface,
        form=tuple(forms[i // 18] for i in range(36)),
        phase_offsets=tuple(origins[i // 18] for i in range(36)),
    )
    gamma = 1 / (1023 * mp.pi)
    kappa = 1 if interface == "matching" else (1 + 2 * mp.cos(2 * mp.pi / 9)) / 3
    scale = mp.sqrt(kappa)
    geometry = report.geometry
    _contains(mp, geometry.kappa_bounds, kappa)
    _contains(mp, geometry.form_chart_scale_bounds, scale)
    _contains(mp, geometry.effective_gamma_bounds, gamma * scale)
    assert geometry.clock_factor == Q(1, 4)
    assert report.parent_form_means == forms
    assert report.parent_phase_offset_means == origins
    assert report.child_form_means == (forms[0], forms[0], forms[1], forms[1])
    assert report.parent_form_residual_sup == report.parent_phase_residual_sup == 0
    assert report.projected_current_defect_bounds.contains(0)
    u, delta = _mp(mp, forms[1] - forms[0]), mp.mpf(1) / 20
    chart_form_rate = (u / scale) + gamma * scale * mp.sin(delta)
    chart_phase_rate = -gamma * scale * (u / scale)
    _contains(
        mp,
        report.parent_form_rate_bounds[0] / (geometry.form_chart_scale_bounds / 4),
        chart_form_rate,
    )
    _contains(mp, report.parent_phase_rate_bounds[0] * 4, chart_phase_rate)
    _contains(
        mp,
        report.outer_attachment_work_bounds,
        9 * u**2 + 18 * (1 - kappa * mp.cos(delta)),
    )


@pytest.fixture(scope="module")
def admission():
    return owner._assess_repeated_composition(
        form_radius=Q(1, 100), phase_radius=Q(1, 100)
    )


def test_full_radius_bootstrap_and_nonvacuous_finite_response(admission, mp):
    report = admission
    gamma, radius, delta = 1 / (1023 * mp.pi), mp.mpf(1001) / 100000, mp.mpf(1) / 20
    assert gamma < mp.mpf(1) / 3200
    xupper = mp.mpf(1) / 100 + gamma * (2 * radius + radius**2 / 2)
    tupper = mp.mpf(1) / 100 + 2 * gamma * radius
    assert xupper <= _mp(mp, report.form_bootstrap_upper) < radius
    assert tupper <= _mp(mp, report.phase_bootstrap_upper) < radius
    assert _mp(mp, report.child_lower_gap_margin) <= 2 * mp.pi / 9 - 2 * radius
    assert _mp(mp, report.child_branch_margin) <= mp.pi - 2 * mp.pi / 9 - 2 * radius
    assert (
        0 < _mp(mp, report.child_acute_margin) <= mp.pi / 2 - 2 * mp.pi / 9 - 2 * radius
    )
    assert report.parent_identity_margin == Q(1, 4) - 2 * Q(1001, 100000)
    # Solve the independent scalar comparison e'=gamma*e+gamma*R^2/2.
    comparison = radius**2 * mp.expm1(gamma) / 2
    assert (
        comparison
        <= _mp(mp, report.nonlinear_reduction_error_bound)
        < mp.mpf(1) / 60000000
    )
    explicit_error = gamma * (gamma**2 / 8 + 2 * radius**2) / 4
    assert (
        explicit_error
        <= _mp(mp, report.explicit_response_error_bound)
        < mp.mpf(1) / 60000000
    )
    expected = tuple(
        gamma * kappa * mp.sin(delta) * (-mp.expm1(-mp.mpf(1) / 2)) / 2
        for kappa in (mp.mpf(1), (1 + 2 * mp.cos(2 * mp.pi / 9)) / 3)
    )
    for bounds, value in zip(report.nominal_response_bounds, expected):
        _contains(mp, bounds, value)
    _contains(mp, report.nominal_contrast_bounds, expected[0] - expected[1])
    assert report.contrast_margin > Q(4, 10**7)
    assert report.contrast_resolution_margin > 0
    assert report.resolution_margin > 0
    assert report.outer_attachment_work_residual_bounds == (-Q(36, 10000), Q(72, 10000))
    assert (
        report.bootstrap_certified
        and report.identity_certified
        and report.response_certified
    )
    assert report.status == "certified_repeated_organized_composition"
    assert not report.unresolved_conditions


@pytest.mark.parametrize("interface", ("matching", "symmetric"))
def test_hidden_child_registration_changes_response_at_identical_parent_means(
    interface, mp
):
    h, delta = Q(1, 200), Q(1, 20)

    def offsets(reverse):
        return tuple(
            (-delta / 2 if i < 18 else delta / 2)
            + (h if (i // 9) % 2 == 0 else -h) * (-1 if reverse and i >= 18 else 1)
            for i in range(36)
        )

    aligned, reversed_parent = tuple(
        owner._observe_repeated_composition(
            interface=interface, form=_ZERO, phase_offsets=offsets(reverse)
        )
        for reverse in (False, True)
    )
    assert aligned.parent_form_means == reversed_parent.parent_form_means == (0, 0)
    assert (
        aligned.parent_phase_offset_means
        == reversed_parent.parent_phase_offset_means
        == (-delta / 2, delta / 2)
    )
    assert (
        aligned.child_phase_offset_means[:2]
        == reversed_parent.child_phase_offset_means[:2]
    )
    assert sorted(aligned.child_phase_offset_means[2:]) == sorted(
        reversed_parent.child_phase_offset_means[2:]
    )
    assert (
        aligned.parent_phase_residual_sup
        == reversed_parent.parent_phase_residual_sup
        == h
    )
    assert (
        aligned.internal_storage_bounds.lo <= reversed_parent.internal_storage_bounds.hi
    )
    assert (
        reversed_parent.internal_storage_bounds.lo <= aligned.internal_storage_bounds.hi
    )
    assert (
        aligned.first_attachment_storage_bounds.lo
        <= reversed_parent.first_attachment_storage_bounds.hi
    )
    assert (
        reversed_parent.first_attachment_storage_bounds.lo
        <= aligned.first_attachment_storage_bounds.hi
    )
    for observed in (aligned, reversed_parent):
        _contains(
            mp, observed.internal_storage_bounds, 36 * (1 - mp.cos(2 * mp.pi / 9))
        )
        _contains(
            mp,
            observed.first_attachment_storage_bounds,
            18 * (1 - mp.cos(2 * _mp(mp, h))),
        )
        _contains(
            mp,
            observed.internal_storage_bounds + observed.first_attachment_storage_bounds,
            36 * (1 - mp.cos(2 * mp.pi / 9)) + 18 * (1 - mp.cos(2 * _mp(mp, h))),
        )
    kappa = 1 if interface == "matching" else (1 + 2 * mp.cos(2 * mp.pi / 9)) / 3
    first = kappa * mp.sin(_mp(mp, delta))
    second = first * mp.cos(2 * _mp(mp, h))
    _contains(mp, aligned.outer_mean_current_bounds, first)
    _contains(mp, reversed_parent.outer_mean_current_bounds, second)
    assert (
        aligned.outer_mean_current_bounds.lo
        > reversed_parent.outer_mean_current_bounds.hi
    )
    assert (
        aligned.parent_form_rate_bounds[0].lo
        > reversed_parent.parent_form_rate_bounds[0].hi
    )
    assert abs(second - first) <= 2 * _mp(mp, h) ** 2


@pytest.fixture(scope="module")
def acquired_sources():
    return (
        owner._assess_acquired_repeated_composition(
            form_error_bounds=(_ERROR, _ERROR / 2),
            phase_error_bounds=(_ERROR / 3, _ERROR),
        ),
        owner._assess_acquired_repeated_composition(
            form_error_bounds=(_ERROR, _ERROR), phase_error_bounds=(_ERROR, Q(1, 10))
        ),
    )


def test_actual_independent_parents_keep_source_work_and_new_registration(
    acquired_sources,
):
    admitted, unavailable = acquired_sources
    assert admitted.sources_certified and admitted.admitted
    assert admitted.status == "certified_acquired_repeated_composition"
    assert admitted.parent_sources[0] is not admitted.parent_sources[1]
    assert admitted.incoming_residual_sup_bound == 15 * Q(1, 10**70)
    assert admitted.admission.initial_parent_phase_origins == (-Q(1, 40), Q(1, 40))
    assert admitted.admission.form_radius == admitted.incoming_residual_sup_bound
    for index, parent in enumerate(admitted.parent_sources):
        assert parent.law.interface == "matching" and parent.law.deformation == 0
        assert parent.recovery.form_error_bound == admitted.form_error_bounds[index]
        assert parent.recovery.phase_error_bound == admitted.phase_error_bounds[index]
        assert parent.identity_certified
        assert parent.attachment_work_excess_bounds[1] > 0
        assert parent.law.degree == 3
    assert all(
        set(item.weighted_degrees) == {Q(4)} for item in admitted.admission.geometry
    )
    assert not unavailable.sources_certified and not unavailable.admitted
    assert unavailable.status == "unavailable"
    assert (
        unavailable.admission.response_certified
    )  # Conditional domain is not a source proof.


@pytest.mark.parametrize("radius", (Q(1, 99), Q(1), Q(2)))
def test_failed_bootstrap_withholds_identity_and_response(radius):
    report = owner._assess_repeated_composition(form_radius=radius, phase_radius=0)
    assert report.status == "unavailable"
    assert (
        not report.bootstrap_certified
        and not report.identity_certified
        and not report.response_certified
    )
    assert "transverse_bootstrap_not_certified" in report.unresolved_conditions


@pytest.mark.parametrize("field", ("form_radius", "phase_radius"))
def test_exact_outward_bootstrap_boundary_is_unavailable(field, admission):
    radius, gamma = admission.transverse_radius, admission.geometry[0].gamma_bounds.hi
    endpoint = (
        radius - gamma * (2 * radius + radius**2 / 2)
        if field == "form_radius"
        else radius - 2 * gamma * radius
    )
    report = owner._assess_repeated_composition(
        **(dict(form_radius=0, phase_radius=0) | {field: endpoint})
    )
    margin = (
        report.form_bootstrap_margin
        if field == "form_radius"
        else report.phase_bootstrap_margin
    )
    assert margin == 0
    assert report.status == "unavailable"
    assert (
        not report.bootstrap_certified
        and not report.identity_certified
        and not report.response_certified
    )


def test_exact_tiny_radii_are_retained_without_materializing_zero():
    tiny = Q(1, 10**100)
    report = owner._assess_repeated_composition(form_radius=tiny, phase_radius=2 * tiny)
    assert report.form_radius == tiny and report.phase_radius == 2 * tiny
    assert report.outer_attachment_work_residual_bounds == (
        -144 * tiny**2,
        180 * tiny**2,
    )
    assert report.response_certified
    with pytest.raises(FrozenInstanceError):
        report.form_radius = 0


@pytest.mark.parametrize("field", ("form_radius", "phase_radius"))
@pytest.mark.parametrize("bad", (True, "0", float("nan"), float("inf"), -Q(1, 10**100)))
def test_radius_admission_precedes_geometry(field, bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid primitive reached geometry")

    monkeypatch.setattr(owner, "_derive_repeated_composition", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner._assess_repeated_composition(
            **(dict(form_radius=0, phase_radius=0) | {field: bad})
        )


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
@pytest.mark.parametrize("bad", (True, "0", float("nan"), float("inf"), object()))
def test_state_admission_precedes_geometry(field, bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid primitive reached geometry")

    monkeypatch.setattr(owner, "_derive_repeated_composition", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner._observe_repeated_composition(
            **(
                dict(interface="matching", form=_ZERO, phase_offsets=_ZERO)
                | {field: (bad, *_ZERO[1:])}
            )
        )


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
@pytest.mark.parametrize("bad", ((), (0,) * 35, (0,) * 37, "0" * 36, {0}))
def test_state_shape_rejects(field, bad):
    with pytest.raises((TypeError, ValueError)):
        owner._observe_repeated_composition(
            **(
                dict(interface="matching", form=_ZERO, phase_offsets=_ZERO)
                | {field: bad}
            )
        )


@pytest.mark.parametrize("bad", (True, 0, "all_to_all", "Matching", None))
def test_interface_is_an_explicit_fixed_control(bad):
    with pytest.raises(ValueError):
        owner._derive_repeated_composition(interface=bad)


@pytest.mark.parametrize("bad", (True, -Q(1, 10**100), float("nan"), "0"))
def test_all_original_source_primitives_admit_before_recovery(bad, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid second source reached first source reconstruction")

    monkeypatch.setattr(owner, "_assess_acquired_constitutive_inheritance", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner._assess_acquired_repeated_composition(
            form_error_bounds=(_ERROR, bad), phase_error_bounds=(_ERROR, _ERROR)
        )


@pytest.mark.parametrize("extra", ("horizon", "delta", "source_report", "response"))
def test_reports_responses_and_changed_protocol_do_not_replace_primitives(extra):
    with pytest.raises(TypeError):
        owner._assess_repeated_composition(form_radius=0, phase_radius=0, **{extra: 0})
