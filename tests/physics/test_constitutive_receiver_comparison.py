"""Cheap independent residual comparisons, without a frozen response replay."""

from dataclasses import replace
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._exact_linear_algebra import exact_symmetric_semidefinite
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics import _composition_compatible_constitutive as law_owner
from tnfr.physics import _constitutive_receiver_comparison as owner
from tnfr.physics import _organized_receiver_enclosure_geometry as sine_geometry

_LIMIT = Q(1, 10000)
_DURATION = Q(1, 8)


def _compare(**changes):
    inputs = dict(
        initial_center=(Q(0),) * 13,
        reference_tube=(I(0),) * 13,
        reference_endpoint=(I(0),) * 13,
        endpoint_center=(Q(0),) * 13,
        incoming_error=Q(1, 1000),
        deformation_bound=_LIMIT,
        velocity_defect_bound=Q(1, 100000),
        duration=_DURATION,
    )
    return owner._compare_constitutive_receiver_step(**(inputs | changes))


@pytest.fixture(scope="module")
def zero_report():
    return _compare(deformation_bound=Q(0))


@pytest.fixture(scope="module")
def broad_report():
    return _compare(reference_tube=(I(-1, 1),) * 13)


@pytest.fixture(scope="module", autouse=True)
def no_frozen_response_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _perturbation(mp, angle):
    return mp.cos(angle) * (1 - mp.cos(9 * angle)) ** 2


@pytest.mark.parametrize("angle", (Q(-19, 7), Q(-1, 10), Q(0), Q(1, 7), Q(17, 9)))
def test_factored_deformation_current_matches_independent_potential_derivative(
    angle, mp
):
    bounds = law_owner._deformation_current_bounds(angle)
    exact = mp.diff(lambda value: _perturbation(mp, value), _mp(mp, angle))
    assert _mp(mp, bounds.lo) <= exact <= _mp(mp, bounds.hi)
    assert _mp(mp, bounds.width) < mp.mpf("1e-30")


@pytest.mark.parametrize("multiple", (-9, -4, -1, 0, 1, 4, 9))
def test_factored_current_retains_cubic_zero_at_registered_target_gaps(multiple, mp):
    phase = 2 * multiple * pi_interval() / 9
    bounds = law_owner._deformation_current_bounds(phase)
    exact = mp.diff(lambda value: _perturbation(mp, value), 2 * multiple * mp.pi / 9)
    assert _mp(mp, bounds.lo) <= exact <= _mp(mp, bounds.hi)
    assert max(abs(bounds.lo), abs(bounds.hi)) < Q(1, 10**30)


def test_factored_current_encloses_a_nonzero_interval_not_just_its_midpoint(mp):
    phase = I(Q(17, 100), Q(19, 100))
    bounds = law_owner._deformation_current_bounds(phase)
    values = []
    for index in range(9):
        angle = _mp(mp, Q(17, 100) + Q(index, 400))
        value = mp.diff(lambda coordinate: _perturbation(mp, coordinate), angle)
        values.append(value)
        assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)
    assert max(values) - min(values) > mp.mpf("0.1")


def test_zero_reference_scalar_comparison_matches_independent_exponential(
    zero_report, mp
):
    report = zero_report
    assert report.status == "certified_local_comparison"
    assert report.subdivisions == len(report.cones) == 8
    rate = _mp(mp, report.growth_rate_upper_bound)
    initial = _mp(mp, report.incoming_error)
    defect = _mp(mp, report.velocity_defect_bound)
    assert rate < 0
    for index, cone in enumerate(report.cones):
        left, right = Q(index, 64), Q(index + 1, 64)
        assert cone.time_bounds == (left, right)
        time = _mp(mp, right)
        expected = mp.exp(rate * time) * initial + mp.expm1(rate * time) * defect / rate
        assert expected <= _mp(mp, cone.endpoint_error) < expected + mp.mpf("1e-25")
        assert cone.defect_upper_bound == report.velocity_defect_bound
        assert cone.initial_error == (
            report.incoming_error
            if index == 0
            else report.cones[index - 1].endpoint_error
        )
        assert cone.whole_error == max(cone.initial_error, cone.endpoint_error)
        assert all(phase.contains(0) for phase in cone.phase_box)
        assert cone.windings == (0, 0, 0)
        assert min(cone.zero_branch_margins) > 3
    assert report.continuous_endpoint_error == report.cones[-1].endpoint_error
    assert report.strict_error_margin > 0
    assert report.whole_error == report.incoming_error
    assert report.endpoint_windings == (0, 0, 0)
    assert report.receiver_zero_throughout


def test_endpoint_recentering_is_added_once_after_all_cones(zero_report, mp):
    displacement = Q(1, 10**6)
    shifted = _compare(
        deformation_bound=Q(0),
        endpoint_center=(displacement,) + (Q(0),) * 12,
    )
    expected = mp.sqrt(2) * _mp(mp, displacement)
    assert expected <= _mp(mp, shifted.recenter_radius) < expected + mp.mpf("1e-30")
    assert shifted.continuous_endpoint_error == zero_report.continuous_endpoint_error
    assert tuple(cone.endpoint_error for cone in shifted.cones) == tuple(
        cone.endpoint_error for cone in zero_report.cones
    )
    assert (
        shifted.endpoint_error
        >= shifted.continuous_endpoint_error + shifted.recenter_radius
    )
    assert (
        shifted.endpoint_error
        - shifted.continuous_endpoint_error
        - shifted.recenter_radius
        < Q(1, 10**30)
    )
    assert (
        shifted.endpoint_error
        < shifted.continuous_endpoint_error + 2 * shifted.recenter_radius
    )


def test_forward_and_endpoint_cones_both_narrow_the_same_reference(broad_report):
    report = broad_report
    assert report.status == "certified_local_comparison"
    assert all(value.contains(0) for value in report.reference_velocity_bounds)
    # Every full nodal sine velocity has magnitude at most one. The start
    # and endpoint are both exactly zero, so either end's first cone is
    # bounded by h/8. Dropping either directional constraint breaks its test.
    for cone in (report.cones[0], report.cones[-1]):
        assert max(value.abs_max for value in cone.phase_box) <= _DURATION / 8
    assert max(value.abs_max for value in report.cones[3].phase_box) > _DURATION / 8
    for cone in report.cones:
        left, right = cone.time_bounds
        for phase, velocity in zip(cone.phase_box, report.reference_velocity_bounds):
            forward = I(left, right) * velocity
            backward = -I(_DURATION - right, _DURATION - left) * velocity
            assert phase.lo >= max(forward.lo, backward.lo, Q(-1))
            assert phase.hi <= min(forward.hi, backward.hi, Q(1))
            assert phase.contains(0)


def test_changed_law_residual_forcing_and_growth_are_retained(broad_report, mp):
    report = broad_report
    assert report.deformation_jacobian_norm_bound == 656
    assert report.global_growth_bound == 2 * (1 + 328 * _LIMIT)
    assert (
        report.growth_rate_upper_bound
        == report.osl.growth_rate_upper_bound + 656 * _LIMIT
    )
    assert max(cone.forcing_norm_upper_bound for cone in report.cones) > 1
    rate = _mp(mp, report.growth_rate_upper_bound)
    previous = _mp(mp, report.incoming_error)
    for cone in report.cones:
        forcing = report.velocity_defect_bound + _LIMIT * cone.forcing_norm_upper_bound
        assert cone.defect_upper_bound == forcing > report.velocity_defect_bound
        duration = _mp(mp, cone.time_bounds[1] - cone.time_bounds[0])
        phi = mp.expm1(rate * duration) / rate if rate else duration
        previous = mp.exp(rate * duration) * previous + phi * _mp(mp, forcing)
        assert previous <= _mp(mp, cone.endpoint_error) < previous + mp.mpf("1e-25")
    # Independently differentiate P on a nonsymmetric point in the middle
    # cone. Full nodal cancellation and all 27 components are retained.
    cone = report.cones[3]
    phase = [mp.mpf(0)] * 27
    phase[0], phase[1] = mp.mpf("0.01"), -mp.mpf("0.01")
    assert all(
        _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
        for interval, value in zip(cone.phase_box, phase)
    )
    field = [mp.mpf(0)] * 27
    edges = tuple(
        (9 * r + j, 9 * r + (j + 1) % 9, Q(1)) for r in range(3) for j in range(9)
    ) + tuple(
        (9 * a + j, 9 * b + (j + offset) % 9, Q(1, 2))
        for a, b in ((0, 1), (1, 2), (2, 0))
        for j in range(9)
        for offset in (0, 1)
    )
    for left, right, weight in edges:
        angle = phase[right] - phase[left]
        current = (
            _mp(mp, weight) * mp.diff(lambda value: _perturbation(mp, value), angle) / 4
        )
        field[left] += current
        field[right] -= current
    assert mp.sqrt(sum(value**2 for value in field)) < _mp(
        mp, cone.forcing_norm_upper_bound
    )
    assert mp.sqrt(sum(value**2 for value in field)) > mp.mpf("0.001")


def test_fresh_growth_covers_every_mean_zero_direction(broad_report):
    report = broad_report
    assert report.osl.phase_mean_leaf_dimension == 26
    assert report.osl.phase_radius == report.prospective_radius
    edges = owner._receiver_weighted_edges()
    jacobian = [[Q(0)] * 27 for _ in range(27)]
    for (a, b, weight), cosine in zip(edges, report.osl.edge_cosine_lower_bounds):
        coefficient = weight * cosine / 4
        jacobian[a][a] -= coefficient
        jacobian[b][b] -= coefficient
        jacobian[a][b] += coefficient
        jacobian[b][a] += coefficient
    # e_i-e_26 is a complete basis, independently of the 13-coordinate
    # reflection imposed only on the reference. The Gram matrix is I+11^T.
    rate = report.growth_rate_upper_bound - 656 * _LIMIT
    slack = tuple(
        tuple(
            rate * (1 + int(i == j))
            - (jacobian[i][j] - jacobian[i][26] - jacobian[26][j] + jacobian[26][26])
            for j in range(26)
        )
        for i in range(26)
    )
    assert exact_symmetric_semidefinite(slack)


@pytest.mark.parametrize(
    "field",
    ("incoming_error", "deformation_bound", "velocity_defect_bound", "duration"),
)
@pytest.mark.parametrize("invalid", (True, float("nan"), float("inf"), "0", Q(-1)))
def test_invalid_scalar_primitives_reject_before_geometry(monkeypatch, field, invalid):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid comparison primitive reached geometry or law calculation")

    monkeypatch.setattr(owner, "_receiver_weighted_edges", forbidden)
    monkeypatch.setattr(owner, "_derive_composition_compatible_law", forbidden)
    with pytest.raises((ValueError, TypeError)):
        _compare(**{field: invalid})


@pytest.mark.parametrize(
    "changes",
    (
        {"initial_center": (Q(0),) * 12},
        {"initial_center": (True,) + (Q(0),) * 12},
        {"reference_tube": (I(0),) * 12},
        {"reference_endpoint": (Q(0),) * 13},
        {"endpoint_center": (float("nan"),) + (Q(0),) * 12},
        {"deformation_bound": Q(1, 324) + Q(1, 10**50)},
        {"duration": Q(0)},
        {"duration": Q(1)},
    ),
)
def test_invalid_reference_and_law_boundaries_reject_before_geometry(
    monkeypatch, changes
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid reference reached geometry")

    monkeypatch.setattr(owner, "_receiver_weighted_edges", forbidden)
    with pytest.raises((ValueError, TypeError)):
        _compare(**changes)


def test_disjoint_reference_data_is_rejected_instead_of_zero_residual():
    with pytest.raises(ValueError, match="outside"):
        _compare(reference_tube=(I(1),) * 13)
    with pytest.raises(ValueError, match="empty derivative cone"):
        _compare(reference_endpoint=(I(1),) * 13)


def test_failed_prospective_domain_has_no_winding_verdict(monkeypatch, zero_report):
    def failed_osl(*args, **kwargs):
        return replace(zero_report.osl, certified=False)

    monkeypatch.setattr(owner, "_receiver_local_osl", failed_osl)
    report = _compare(deformation_bound=Q(0))
    assert report.status == "unavailable"
    assert report.reason == "prospective_error_domain_not_closed"
    assert report.endpoint_error is None
    assert report.endpoint_windings is None
    assert report.receiver_zero_throughout is None
    assert report.donor_endpoint_nonunit is None
    assert all(
        cone.windings is None and cone.zero_branch_margins is None
        for cone in report.cones
    )


def test_static_windings_do_not_import_source_or_capture_certificates(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("static comparison imported a source or capture certificate")

    monkeypatch.setattr(sine_geometry, "_receiver_capture_geometry", forbidden)
    report = _compare()
    assert report.endpoint_windings == (0, 0, 0)
    assert report.donor_endpoint_nonunit is True
    assert (
        "source_release_and_uniform_velocity_defect_require_separate_admission"
        in report.scope
    )
    assert (
        "endpoint_winding_classification_does_not_supply_initial_donor_identity"
        in report.scope
    )
    assert "no_capture_event_or_physical_identification_verdict" in report.scope


@pytest.mark.parametrize(
    "windings,expected_donors,expected_receiver",
    (
        ((0, 0, 0), True, True),
        ((0, None, 0), None, True),
        ((0, 1, 0), False, True),
        ((None, 1, None), False, None),
        ((None, None, None), None, None),
        ((0, 0, 1), True, False),
    ),
)
def test_partial_winding_availability_cannot_certify_both_donors(
    monkeypatch, windings, expected_donors, expected_receiver
):
    # Exercise only the observation aggregation contract. These supplied
    # classifications are not trajectory or winding evidence.
    monkeypatch.setattr(
        owner, "_winding_observation", lambda *args: (windings, (Q(0),) * 3)
    )
    report = _compare(deformation_bound=Q(0))
    assert report.donor_endpoint_nonunit is expected_donors
    assert report.receiver_zero_throughout is expected_receiver
