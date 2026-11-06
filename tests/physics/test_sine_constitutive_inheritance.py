"""Own-law acquired composition, support work and admission controls."""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _sine_constitutive_inheritance as owner

ERROR = Q(1, 10**10)
EPS = Q(1, 10**70)


@pytest.fixture(scope="module", autouse=True)
def no_scientific_regeneration():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 210
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _assess(**changes):
    inputs = dict(
        interface="symmetric",
        deformation=Q(1),
        form_error_bound=ERROR,
        phase_error_bound=ERROR,
    )
    return owner._assess_acquired_constitutive_inheritance(**(inputs | changes))


@pytest.fixture(scope="module")
def reports():
    return tuple(
        _assess(interface=name, deformation=coefficient)
        for name in ("matching", "symmetric")
        for coefficient in (Q(0), Q(1, 2), Q(1))
    )


def test_new_supports_have_their_actual_degrees_and_cyclic_registration(reports):
    for report in reports:
        law = report.law
        edges = set(tuple(sorted(edge)) for edge in law.edges)
        assert len(edges) == len(law.edges) == 18 + 9 * law.cross_degree
        assert set(law.nodes) == set(range(18))
        assert all(
            sum(node in edge for edge in edges) == law.degree for node in law.nodes
        )
        assert law.collective_clock_factor == Q(law.cross_degree, law.degree)
        for j in range(9):
            actual = {b - 9 for a, b in law.cross_edges if a == j}
            expected = (
                {j} if law.interface == "matching" else {j, (j - 1) % 9, (j + 1) % 9}
            )
            assert actual == expected


def test_recovery_is_reused_without_consuming_the_old_three_component_contact(reports):
    for report in reports:
        assert report.status == "certified_acquired_constitutive_inheritance"
        assert report.identity_certified and report.finite_projection_certified
        assert not report.unresolved_conditions
        assert report.recovery.recovery_certified
        assert report.recovery.source.form_error_bound == ERROR
        assert report.recovery.source.phase_error_bound == ERROR
        assert report.recovery.endpoint_radius == EPS
        assert report.recovery.scaled_handoff_time == 100 + 10**13
        assert report.conserved_form_mean == report.conserved_lifted_phase_mean == 0
        assert report.original_finite_duration * Q(1023, 1024) == report.finite_duration


def test_full_transverse_gap_curvature_and_barrier_are_independently_positive(
    reports, mp
):
    alpha, radius = 2 * mp.pi / 9, mp.mpf(1) / 12
    cycle_gap = 2 - 2 * mp.cos(alpha)
    assert cycle_gap > mp.mpf(2) / 5
    for report in reports:
        lam = _mp(mp, report.law.deformation)
        c = mp.cos(alpha + mp.sqrt(2) * radius) - 2 * lam / 243
        assert c > mp.mpf(1) / 2
        # Matching is C9 x K2; symmetric adds positive-semidefinite edge
        # Laplacians to it. A unit arbitrary transverse mode is controlled.
        assert c * cycle_gap / 2 > mp.mpf(1) / 10
        assert report.storage_coercivity_lower_bound == Q(1, 10)
        assert report.storage_barrier == Q(1, 10) * Q(1, 12) ** 2
        assert report.storage_margin > 0
        assert report.radius_margin > 0
        assert 10 * report.joined_excess_storage_upper_bound < (15 * EPS) ** 2
        assert report.entire_future_norm_upper_bound == 15 * EPS


def test_actual_attachment_work_and_storage_include_asymmetric_transverse_errors(
    reports, mp
):
    alpha = 2 * mp.pi / 9
    eps = _mp(mp, EPS)
    # Independent zero-mean, non-rigid perturbations on both retained rings.
    x = [eps * mp.mpf((j % 9) - 4) / (20 if j < 9 else -30) for j in range(18)]
    y = [
        eps * mp.mpf(((j % 9) - 4) ** 2 - mp.mpf(20) / 3) / (80 if j < 9 else -90)
        for j in range(18)
    ]
    theta = [alpha * (j % 9 - 4) + y[j] for j in range(18)]
    for start in (0, 9):
        assert abs(sum(x[start : start + 9])) < mp.mpf("1e-200")
        assert abs(sum(y[start : start + 9])) < mp.mpf("1e-200")
        assert sum(value**2 for value in x[start : start + 9]) < eps**2
        assert sum(value**2 for value in y[start : start + 9]) < eps**2

    def potential(angle, lam):
        return 1 - mp.cos(angle) + lam * (1 - mp.cos(9 * angle)) ** 2 / (6 * 9**4)

    for report in reports:
        law, lam = report.law, _mp(mp, report.law.deformation)
        actual_work = sum(
            (x[i] - x[j]) ** 2 / 2 + potential(theta[j] - theta[i], lam)
            for i, j in law.cross_edges
        )
        nominal_work = 0 if law.interface == "matching" else 18 * (1 - mp.cos(alpha))
        lo, hi = (
            law.nominal_attachment_work_bounds.lo,
            law.nominal_attachment_work_bounds.hi,
        )
        assert _mp(mp, lo) <= nominal_work <= _mp(mp, hi)
        correction = actual_work - nominal_work
        assert 0 < correction < _mp(mp, report.attachment_work_excess_bounds[1])
        total = sum(
            (x[i] - x[j]) ** 2 / 2 + potential(theta[j] - theta[i], lam)
            for i, j in law.edges
        )
        target = 18 * (1 - mp.cos(alpha)) + nominal_work
        assert 0 < total - target < _mp(mp, report.joined_excess_storage_upper_bound)
        assert report.attachment_work_excess_bounds[1] < Q(1, 10**137)
        if law.interface == "symmetric":
            assert nominal_work > 4
            # The nonzero subgrid correction is separately represented;
            # interval rounding of the O(1) baseline cannot resolve it.
            assert (
                law.nominal_attachment_work_bounds.width
                > report.attachment_work_excess_bounds[1]
            )


def test_finite_actual_projection_has_exact_phase_row_and_bounded_hidden_defect(
    reports, mp
):
    mtime, eps = mp.mpf(1) / 10, _mp(mp, EPS)
    assert mp.exp(3 * mtime) * eps < _mp(mp, Q(10, 7) * EPS)
    for report in reports:
        law = report.law
        lam, gamma = _mp(mp, law.deformation), 1 / (1023 * mp.pi)
        u, k = 1 + 2 * lam / 243, _mp(mp, law.collective_clock_factor)
        expected = 4 * gamma * k * u * _mp(mp, Q(10, 7) * EPS)
        assert report.finite_full_state_error_upper_bound == Q(10, 7) * EPS
        assert report.projected_phase_row_defect == 0
        assert expected <= _mp(mp, report.projected_form_row_defect_upper_bound)
        assert _mp(
            mp, report.projected_form_row_defect_upper_bound
        ) - expected < mp.mpf("1e-100")
        r = 1 if law.interface == "matching" else (1 + 2 * mp.cos(2 * mp.pi / 9)) / 3
        assert _mp(mp, report.finite_chart_form_error_upper_bound) >= _mp(
            mp, Q(10, 7) * EPS
        ) / mp.sqrt(r)


def test_mapped_parameter_interval_does_not_transfer_fine_source_guarantees():
    matching = owner._derive_constitutive_inheritance(
        interface="matching", deformation=1
    )
    symmetric = owner._derive_constitutive_inheritance(
        interface="symmetric", deformation=1
    )
    assert matching.retains_fixed_exchange_parameter
    assert matching.effective_deformation_within_unit_interval
    assert not symmetric.retains_fixed_exchange_parameter
    assert symmetric.effective_deformation_bounds.lo > 1
    assert symmetric.effective_deformation_within_unit_interval is False
    algebraic = owner._derive_constitutive_inheritance(
        interface="symmetric", deformation=2
    )
    assert algebraic.effective_deformation_bounds.lo > 2
    actual = _assess(deformation=2)
    assert actual.status == "unavailable"
    assert not actual.identity_certified
    assert "new_support_curvature_or_gap_not_certified" in actual.unresolved_conditions


def test_failed_source_is_not_replaced_by_exact_rigid_family_closure():
    report = _assess(phase_error_bound=Q(1, 10))
    assert report.status == "unavailable"
    assert not report.identity_certified
    assert not report.finite_projection_certified
    assert "isolated_source_recovery_not_certified" in report.unresolved_conditions


@pytest.mark.parametrize(
    "field,bad",
    [
        ("interface", True),
        ("interface", "all_to_all"),
        ("deformation", True),
        ("deformation", -1),
        ("deformation", float("nan")),
        ("form_error_bound", "0"),
        ("form_error_bound", -Q(1, 2**500)),
        ("phase_error_bound", float("inf")),
    ],
)
def test_invalid_primitives_reject_before_recovery(monkeypatch, field, bad):
    def forbidden(**kwargs):
        pytest.fail("invalid primitive reached scientific recovery")

    monkeypatch.setattr(owner, "_assess_constitutive_recovery", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: bad})


@pytest.mark.parametrize(
    "change",
    [
        {"form": [0]},
        {"form": [0, True]},
        {"form": [0, float("inf")]},
        {"relative_phase_phasor": (1, 1)},
        {"relative_phase_phasor": (True, 0)},
    ],
)
def test_static_collective_coordinates_admit_before_coefficients(monkeypatch, change):
    def forbidden(**kwargs):
        pytest.fail("invalid coordinate reached geometry")

    monkeypatch.setattr(owner, "_derive_constitutive_inheritance", forbidden)
    inputs = dict(
        interface="matching", deformation=0, form=(0, 0), relative_phase_phasor=(1, 0)
    )
    with pytest.raises((TypeError, ValueError)):
        owner._observe_inherited_constitutive_pair(**(inputs | change))


def test_reports_and_tuned_horizons_are_not_admitted_as_new_source_evidence(reports):
    for change in (
        {"source": reports[0].recovery},
        {"horizon": 3},
        {"contact_budget": 0},
    ):
        with pytest.raises(TypeError):
            _assess(**change)
