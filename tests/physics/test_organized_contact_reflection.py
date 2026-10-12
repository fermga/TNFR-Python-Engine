"""Independent complete-state checks of the declared algebraic saddle."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import I
from tnfr.physics import _organized_contact_reflection as owner


_EDGES = (
    tuple((9 * c + j, 9 * c + (j + 1) % 9) for c in range(6) for j in range(9))
    + tuple((9 * c + j, 9 * ((c + 1) % 5) + j) for c in range(5) for j in range(9))
    + tuple((9 * c + j, 45 + j) for c in range(5) for j in range(9))
)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def report():
    return owner._derive_organized_contact_reflection()


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 180
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, expected):
    assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)


@pytest.fixture(scope="module")
def independent_state(mp, report):
    # The midpoint's cubic defect is below the dyadic arithmetic resolution;
    # no second root solver or trajectory is needed for this independent audit.
    z = _mp(mp, sum(report.root_bracket) / 2)
    s = mp.sqrt(1 - z * z)
    a = mp.atan2(-s, z)
    b = mp.atan2(s * (1 + 2 * z) / 2, -(2 + z) * (1 + 2 * z) / 2)
    origins = (mp.mpf(0), b, a, -a, -b, mp.mpf(0))
    return tuple(
        2 * mp.pi * (j - 4) / 9 + origins[c] for c in range(6) for j in range(9)
    )


def test_root_isolated_by_exact_signs_monotonicity_and_fixed_budget(report):
    lower, upper = report.root_bracket

    def polynomial(z):
        return 16 * z**3 + 36 * z**2 + 24 * z + 1

    assert Q(-1, 20) < lower < upper < 0
    assert upper - lower == Q(1, 20 * 2**128)
    assert report.root_endpoint_values == (polynomial(lower), polynomial(upper))
    assert polynomial(lower) < 0 < polynomial(upper)
    assert report.root_derivative_bounds.lo > 20
    assert report.root_bounds.contains(lower) and report.root_bounds.contains(upper)
    assert report.root_isolated and report.bisections == 128


def test_unit_circle_and_stationarity_identities_are_independent_of_residual_flags(
    report, mp
):
    z = _mp(mp, sum(report.root_bracket) / 2)
    polynomial = 16 * z**3 + 36 * z**2 + 24 * z + 1
    (ca, sa), (cb, sb) = report.circle_coordinate_bounds
    a = mp.atan2(-mp.sqrt(1 - z * z), z)
    b = mp.atan2(mp.sqrt(1 - z * z) * (1 + 2 * z) / 2, -(2 + z) * (1 + 2 * z) / 2)
    for box, value in (
        (ca, mp.cos(a)),
        (sa, mp.sin(a)),
        (cb, mp.cos(b)),
        (sb, mp.sin(b)),
    ):
        _contains(mp, box, value)
    assert abs(polynomial) < mp.mpf("1e-38")
    assert abs(mp.sin(a - b) - 2 * mp.sin(b)) < mp.mpf("1e-37")
    assert abs(mp.sin(a - b) + mp.sin(a) * (1 + 2 * mp.cos(a))) < mp.mpf("1e-37")
    assert report.stationary_identity_certified


def test_complete_support_fields_and_means_against_independent_state(
    report, mp, independent_state
):
    actual_edges = tuple((a, b) for a, b, _ in report.geometry.weighted_edges)
    assert len(actual_edges) == len(set(actual_edges)) == len(_EDGES)
    assert set(actual_edges) == set(_EDGES)
    assert all(weight == 1 for _, _, weight in report.geometry.weighted_edges)
    assert len(report.form) == len(report.phase_lift_bounds) == 54
    assert all(value == 0 for value in report.form)
    assert abs(sum(independent_state)) < mp.mpf("1e-175")
    current = [mp.mpf(0)] * 54
    hessian = dict(zip(actual_edges, report.edge_hessian_bounds))
    for left, right in _EDGES:
        delta = independent_state[right] - independent_state[left]
        current[left] += mp.sin(delta)
        current[right] -= mp.sin(delta)
        _contains(mp, hessian[left, right], mp.cos(delta))
    for phase, phase_box, value, current_box in zip(
        independent_state,
        report.phase_lift_bounds,
        current,
        report.phase_current_bounds,
    ):
        _contains(mp, phase_box, phase)
        _contains(mp, current_box, value)
        assert current_box.contains(0)
        assert abs(value) < mp.mpf("1e-37")
    assert report.stationary_residuals_consistent


def test_donor_reconstruction_recovers_the_known_stationary_saddle(
    report, mp, independent_state
):
    """Host-only balance recovers the donor without certifying stability."""
    # This tolerance retains the existing cubic bracket's finite width.
    tolerance = mp.mpf("1e-35")
    host = independent_state[:45]
    reconstructed = []
    for slot in range(9):
        angles = tuple(host[9 * child + slot] for child in range(5))
        torque = tuple(
            sum(
                mp.sin(host[neighbor] - angles[child])
                for neighbor in (
                    9 * child + (slot - 1) % 9,
                    9 * child + (slot + 1) % 9,
                    9 * ((child - 1) % 5) + slot,
                    9 * ((child + 1) % 5) + slot,
                )
            )
            for child in range(5)
        )
        sine = tuple(mp.sin(angle) for angle in angles)
        cosine = tuple(mp.cos(angle) for angle in angles)
        g00 = sum(value**2 for value in sine)
        g01 = -sum(s * c for s, c in zip(sine, cosine))
        g11 = sum(value**2 for value in cosine)
        determinant = g00 * g11 - g01**2
        assert determinant > 0
        right0 = sum(s * t for s, t in zip(sine, torque))
        right1 = -sum(c * t for c, t in zip(cosine, torque))
        real = (g11 * right0 - g01 * right1) / determinant
        imaginary = (g00 * right1 - g01 * right0) / determinant
        donor = mp.mpc(real, imaginary)
        assert abs(abs(donor) ** 2 - 1) < tolerance
        assert abs(donor - mp.exp(mp.j * independent_state[45 + slot])) < tolerance
        assert (
            max(
                abs(s * real - c * imaginary - t)
                for s, c, t in zip(sine, cosine, torque)
            )
            < tolerance
        )
        reconstructed.append(donor)

    # Rebuild every joined current from the reconstructed circular coordinates.
    phasors = tuple(mp.exp(mp.j * phase) for phase in host) + tuple(reconstructed)
    current = [mp.mpf(0)] * 54
    quadratic = mp.mpf(0)
    for left, right in _EDGES:
        relative = mp.conj(phasors[left]) * phasors[right]
        current[left] += relative.imag
        current[right] -= relative.imag
        quadratic += (
            relative.real
            * _mp(mp, report.centered_cut[right] - report.centered_cut[left]) ** 2
        )
    assert max(abs(value) for value in current) < tolerance
    donor_gaps = tuple(
        mp.arg(mp.conj(reconstructed[slot]) * reconstructed[(slot + 1) % 9])
        for slot in range(9)
    )
    assert all(abs(gap) < mp.pi for gap in donor_gaps)
    assert abs(sum(donor_gaps) - 2 * mp.pi) < tolerance
    assert quadratic < -mp.mpf(27) / 4
    assert report.stationary_instability_certified


def test_every_marked_winding_and_nonacute_outer_gap(report, mp, independent_state):
    cycles = tuple(tuple(9 * c + j for j in range(9)) for c in range(6)) + tuple(
        tuple(9 * c + j for c in range(5)) for j in range(9)
    )
    for index, (cycle, sum_box) in enumerate(zip(cycles, report.winding_sum_bounds)):
        gaps = tuple(
            mp.arg(
                mp.exp(
                    1j
                    * (
                        independent_state[cycle[(k + 1) % len(cycle)]]
                        - independent_state[node]
                    )
                )
            )
            for k, node in enumerate(cycle)
        )
        assert all(abs(gap) < mp.pi for gap in gaps)
        assert abs(sum(gaps) - 2 * mp.pi) < mp.mpf("1e-175")
        _contains(mp, sum_box, sum(gaps))
        if index < 6:
            for gap in gaps:
                _contains(mp, report.internal_gap_bounds, gap)
        else:
            for gap, gap_box in zip(gaps, report.outer_gap_bounds):
                _contains(mp, gap_box, gap)
                assert abs(gap) > mp.pi / 2
    assert report.marked_windings_certified


def test_negative_full_hessian_cut_and_mean_leaf(report, mp, independent_state):
    assert sum(report.centered_cut) == 0
    assert len(report.centered_cut) == 54
    quadratic = sum(
        mp.cos(independent_state[right] - independent_state[left])
        * _mp(mp, report.centered_cut[right] - report.centered_cut[left]) ** 2
        for left, right in _EDGES
    )
    _contains(mp, report.cut_quadratic_bounds, quadratic)
    _contains(mp, report.algebraic_cut_quadratic_bounds, quadratic)
    assert report.cut_quadratic_bounds.hi < Q(-27, 4)
    assert report.stationary_instability_certified
    assert all(
        report.centered_cut[a] == report.centered_cut[b]
        for a, b, _ in report.geometry.internal_edges
    )


def test_nonuniform_nonacute_profile_preserves_the_analytic_saddle(
    report, mp, independent_state
):
    """A separate static theorem check, not another evaluated source protocol."""
    profile = (mp.mpf(0),) + tuple((j + 5) * mp.pi / 7 for j in range(1, 9))
    mean = sum(profile) / 9
    phases = tuple(
        value - mean + independent_state[9 * c + 4]
        for c in range(6)
        for value in profile
    )
    current = [mp.mpf(0)] * 54
    for left, right in _EDGES:
        value = mp.sin(phases[right] - phases[left])
        current[left] += value
        current[right] -= value
    assert max(abs(value) for value in current) < mp.mpf("1e-37")
    gaps = tuple(
        mp.arg(mp.exp(1j * (profile[(j + 1) % 9] - profile[j]))) for j in range(9)
    )
    assert abs(sum(gaps) - 2 * mp.pi) < mp.mpf("1e-175")
    assert max(gaps) > mp.pi / 2
    defect = 6 * sum(1 - mp.cos(gap - 2 * mp.pi / 9) for gap in gaps)
    assert defect > 9 * 5 * (7 - 3 * mp.sqrt(5)) / 8
    quadratic = sum(
        mp.cos(phases[right] - phases[left])
        * _mp(mp, report.centered_cut[right] - report.centered_cut[left]) ** 2
        for left, right in _EDGES
    )
    _contains(mp, report.cut_quadratic_bounds, quadratic)


@pytest.mark.parametrize("fault", ("current", "hessian"))
def test_failed_arithmetic_audit_cannot_certify_the_combined_result(monkeypatch, fault):
    original = owner._bounded_sine_edge_fields

    def damaged(**kwargs):
        current, potential, edge_current = original(**kwargs)
        if fault == "current":
            current = (I(1),) + current[1:]
        else:
            potential = (I(0),) * len(potential)
        return current, potential, edge_current

    monkeypatch.setattr(owner, "_bounded_sine_edge_fields", damaged)
    result = owner._derive_organized_contact_reflection()
    assert result.root_isolated and result.stationary_identity_certified
    assert result.marked_windings_certified
    if fault == "current":
        assert not result.stationary_residuals_consistent
        assert result.cut_quadratic_bounds.hi < 0
    else:
        assert result.stationary_residuals_consistent
        assert result.cut_quadratic_bounds.lo > 0
        assert result.algebraic_cut_quadratic_bounds.hi < 0
    assert not result.stationary_instability_certified


def test_example_is_immutable_and_retains_missing_source_scope(report):
    assert report.protocol_revision == "9c7c3ef4"
    assert report.source_status.startswith("unavailable_")
    assert report.geometry.external_phase_origin_bounds == I(0)
    with pytest.raises(FrozenInstanceError):
        report.stationary_instability_certified = False


@pytest.mark.parametrize(
    "keyword", ("root", "bisections", "source", "form", "geometry")
)
def test_declared_derivation_accepts_no_protocol_or_report_overrides(keyword):
    with pytest.raises(TypeError):
        owner._derive_organized_contact_reflection(**{keyword: 0})
