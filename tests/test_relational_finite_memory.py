"""Finite-memory API admission and limiting contact bounds, without trajectories."""

import math
from fractions import Fraction as Q

import pytest

from tnfr.physics.relational_cycle_memory import certify_relational_cycle_memory


@pytest.fixture(scope="module")
def certificate():
    return certify_relational_cycle_memory()


def test_default_amplitude_certifies_distinct_limiting_offsets_and_readouts(
    certificate,
):
    report = certificate
    assert report.amplitude == Q(1, 2**20)
    assert report.admitted and report.status == "admitted"
    assert report.phase_signs_separated and report.contact_signs_separated
    assert all(passed for _, passed in report.proof_checks)
    assert report.first_exit_margin_lower_bound > 0
    assert report.trajectory_norm_upper_bound < report.neighborhood_radius
    assert report.phase_remainder_bound == Q(1, 2**45)
    assert report.unavailable_reasons == ()
    positive, negative = report.cases
    assert (positive.form_sign, negative.form_sign) == (1, -1)
    assert positive.limiting_phase_shift_bounds[0] > 0
    assert negative.limiting_phase_shift_bounds[1] < 0
    assert positive.left_contact_form_rate_bounds[1] < 0
    assert negative.left_contact_form_rate_bounds[0] > 0
    for case in report.cases:
        assert case.asymptotic.remainder_bound is None
        assert case.asymptotic.basin_admitted
        assert case.limiting_phase_shift_bounds == (
            case.leading_phase_shift_bounds[0] - report.phase_remainder_bound,
            case.leading_phase_shift_bounds[1] + report.phase_remainder_bound,
        )
        assert case.contact_storage_bounds[0] > 0
        assert case.readout_acute_margin_lower_bound > 0
        assert case.left_contact_form_rate_bounds == tuple(
            -value for value in reversed(case.right_contact_form_rate_bounds)
        )
    # The two enclosing intervals are symmetric; no assertion equates the
    # unknown nonlinear limiting offsets to negatives of one another.


def test_nonlinear_contact_encloses_independent_endpoint_values(certificate):
    s = pytest.importorskip("sympy")
    cosine = (s.sqrt(5) - 1) / 4
    for case in certificate.cases:
        for bound in case.limiting_phase_shift_bounds:
            delta = s.Rational(bound.numerator, bound.denominator)
            rate = s.atan(s.sin(delta) / (2 * cosine + s.cos(delta))) / (2 * s.pi)
            cost = 1 - s.cos(delta)
            for value, interval in (
                (rate, case.right_contact_form_rate_bounds),
                (cost, case.contact_storage_bounds),
            ):
                independently_evaluated = Q(str(s.N(value, 90)))
                assert interval[0] <= independently_evaluated <= interval[1]
    # The phase readout is nonlinear; a quadratic coefficient multiplied by
    # epsilon squared alone cannot replace this interval plus its remainder.
    leading_readout = tuple(
        certificate.amplitude**2 * value
        for value in certificate.cases[
            0
        ].asymptotic.quadratic_left_port_form_rate_coefficient_bounds
    )
    interval = certificate.cases[0].left_contact_form_rate_bounds
    assert interval[0] < leading_readout[0] < leading_readout[1] < interval[1]


def test_valid_capture_does_not_force_a_finite_sign_claim_at_larger_amplitude():
    report = certify_relational_cycle_memory(amplitude=Q(1, 1024))
    assert not report.admitted and report.status == "unavailable"
    assert all(case.asymptotic.basin_admitted for case in report.cases)
    assert not report.phase_signs_separated and not report.contact_signs_separated
    assert report.unavailable_reasons == (
        "limiting_phase_sign_separation_not_certified",
        "nonlinear_contact_rate_sign_separation_not_certified",
    )
    for case in report.cases:
        assert (
            case.limiting_phase_shift_bounds[0]
            < 0
            < case.limiting_phase_shift_bounds[1]
        )
        assert case.contact_storage_bounds[0] == 0 < case.contact_storage_bounds[1]


def test_tiny_exact_phase_shift_survives_unresolved_readout_precision():
    epsilon = Q(1, 2**200)
    report = certify_relational_cycle_memory(amplitude=epsilon)
    assert report.amplitude == epsilon
    assert report.phase_signs_separated and not report.contact_signs_separated
    assert not report.admitted
    assert report.unavailable_reasons == (
        "nonlinear_contact_rate_sign_separation_not_certified",
    )
    assert report.cases[0].limiting_phase_shift_bounds[0] > 0
    for case in report.cases:
        interval = case.left_contact_form_rate_bounds
        assert interval[0] < 0 < interval[1]
        assert interval != (0, 0)
        assert case.contact_storage_bounds[0] == 0 < case.contact_storage_bounds[1]


def test_fixed_certificate_never_evaluates_a_live_field_or_runs_a_step(monkeypatch):
    from tnfr.dynamics import relational

    def forbidden(*args, **kwargs):
        raise AssertionError("static memory certificate attempted engine execution")

    monkeypatch.setattr(relational, "evaluate_relational_exchange", forbidden)
    monkeypatch.setattr(relational, "step_relational_exchange", forbidden)
    assert certify_relational_cycle_memory().admitted


@pytest.mark.parametrize(
    "amplitude", (0, -1, Q(1, 1024) + Q(1, 2**50), math.inf, math.nan)
)
def test_nonpositive_nonfinite_or_out_of_scope_amplitudes_reject(amplitude):
    with pytest.raises(ValueError, match="amplitude"):
        certify_relational_cycle_memory(amplitude=amplitude)


def test_boolean_amplitude_cannot_be_a_physical_preparation():
    with pytest.raises(TypeError, match="amplitude"):
        certify_relational_cycle_memory(amplitude=True)
