"""Clock/basis-invariant necessary screens and explicit uncertainty controls."""

from fractions import Fraction as Q
from itertools import permutations

import numpy as np
import pytest

from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from tnfr.physics.relational_sine_scale import assess_sine_replica_stiffness_trace_curve
from tnfr.sdk import relational_report_to_dict


def _pulse_invariants():
    c = cos(2 * pi_interval() / 5)
    l = 1 - c
    traces, determinants = [], []
    # Construct full symmetric stiffness entries, not the trace-curve formula.
    for delta in (0, Q(1, 64), Q(1, 32)):
        C, S = cos(I(delta)), sin(I(delta))
        a = c * l**2 * C**2
        b = sqrt(l) * (1 - c**2) * C * S
        d = c * (1 - (1 + c) * S**2)
        traces.append(a + d)
        determinants.append(a * d - b**2)
    return traces, determinants


def test_pulse_geometry_is_not_excluded_but_excludes_affine_one_control_families():
    traces, determinants = _pulse_invariants()
    for indices in permutations(range(3)):
        report = assess_sine_replica_stiffness_trace_curve(
            trace_bounds=[traces[i] for i in indices],
            determinant_bounds=[determinants[i] for i in indices],
        )
        assert report.trace_separation_certified
        assert report.template_curve_status == "not_excluded"
        assert report.affine_family_status == "excluded"
        assert report.affine_obstruction_bounds.lo > 0


def test_constant_clock_scaling_preserves_the_decision():
    traces, determinants = _pulse_invariants()
    for rate_squared in (Q(1, 9), Q(7)):
        report = assess_sine_replica_stiffness_trace_curve(
            trace_bounds=[rate_squared * t for t in traces],
            determinant_bounds=[rate_squared**2 * v for v in determinants],
        )
        assert report.template_curve_status == "not_excluded"
        assert report.affine_family_status == "excluded"


def test_affine_stiffness_counterexample_requires_mass_normalization():
    mass = np.diag([2.0, 3.0])
    stiffness0 = np.array([[4.0, 1.0], [1.0, 6.0]])
    stiffness1 = np.array([[2.0, 1.0], [1.0, -1.0]])
    basis = np.array([[1.0, 2.0], [0.0, 1.0]])
    traces, determinants = [], []
    for rho in (0, 1, 2):
        K = stiffness0 + rho * stiffness1
        J = np.linalg.solve(mass, K)
        transformed = np.linalg.solve(basis.T @ mass @ basis, basis.T @ K @ basis)
        np.testing.assert_allclose(np.trace(J), np.trace(transformed), atol=1e-13)
        np.testing.assert_allclose(
            np.linalg.det(J), np.linalg.det(transformed), atol=1e-13
        )
        # Independent exact two-by-two determinant with M=diag(2,3).
        traces.append(Q(4 + 2 * rho, 2) + Q(6 - rho, 3))
        determinants.append(Q((4 + 2 * rho) * (6 - rho) - (1 + rho) ** 2, 6))
    report = assess_sine_replica_stiffness_trace_curve(
        trace_bounds=traces,
        determinant_bounds=determinants,
    )
    assert report.template_curve_status == "excluded"
    assert report.affine_family_status == "not_excluded"
    assert report.affine_obstruction_bounds.hi < 0


def test_repeated_or_uncertain_traces_abstain_without_dividing():
    for traces in ((1, 1, 2), (I(0, 2), I(1, 3), I(2, 4))):
        report = assess_sine_replica_stiffness_trace_curve(
            trace_bounds=traces,
            determinant_bounds=(0, 1, 2),
        )
        assert not report.trace_separation_certified
        assert report.template_curve_status == "unresolved_trace_separation"
        assert report.affine_family_status == "unresolved_trace_separation"


def test_wide_determinant_error_cannot_be_promoted_to_a_pass():
    report = assess_sine_replica_stiffness_trace_curve(
        trace_bounds=(1, 2, 3),
        determinant_bounds=(I(-100, 100),) * 3,
    )
    assert report.template_curve_status == "not_excluded"
    assert report.affine_family_status == "not_excluded"
    assert (
        "not_excluded_is_not_exact_equality_sufficiency_or_physical_admission"
        in report.scope
    )


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), "1"])
def test_invalid_primitive_observations_reject(bad):
    with pytest.raises((TypeError, ValueError)):
        assess_sine_replica_stiffness_trace_curve(
            trace_bounds=(1, bad, 3),
            determinant_bounds=(1, 2, 3),
        )


def test_forged_interval_and_wrong_lengths_reject():
    bad = I(1)
    object.__setattr__(bad, "lo", True)
    with pytest.raises((TypeError, ValueError)):
        assess_sine_replica_stiffness_trace_curve(
            trace_bounds=(bad, 2, 3),
            determinant_bounds=(1, 2, 3),
        )
    with pytest.raises(ValueError, match="three"):
        assess_sine_replica_stiffness_trace_curve(
            trace_bounds=(1, 2),
            determinant_bounds=(1, 2, 3),
        )


def test_sdk_retains_observations_and_non_admission_scope():
    report = assess_sine_replica_stiffness_trace_curve(
        trace_bounds=(2, 4, 6),
        determinant_bounds=(1, 4, 9),
    )
    data = relational_report_to_dict(report)
    assert data["report_type"] == "SineReplicaStiffnessTraceCurve"
    assert data["report"] == report.to_dict()["report"]
    assert data["report"]["template_curve_status"] == "excluded"
    assert data["report"]["affine_family_status"] == "not_excluded"
