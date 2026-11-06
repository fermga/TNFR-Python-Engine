"""Finite-clock memory readout admission and nonlinear response enclosures."""

from fractions import Fraction as Q

import pytest

from tnfr.physics.relational_cycle_memory import certify_relational_cycle_memory_readout


@pytest.fixture(scope="module")
def certificate():
    return certify_relational_cycle_memory_readout()


def test_default_readout_retains_residual_background_and_separates_both_ports(
    certificate,
):
    report = certificate
    assert report.admitted and report.status == "admitted"
    assert report.memory.admitted
    assert report.memory.amplitude == Q(1, 2**20)
    assert report.decay_blocks == 40
    assert report.horizon == Q(165120)
    assert report.block_duration == Q(4128)
    assert report.state_norm_upper_bound == Q(1, 2**57)
    assert report.mean_phase_tail_upper_bound == Q(1, 2**102)
    assert report.phase_signs_separated
    assert report.contact_signs_separated and report.rate_change_signs_separated
    assert report.unavailable_reasons == ()
    for case, limiting in zip(report.cases, report.memory.cases):
        assert case.unavailable_reasons == ()
        assert case.internal_acute_margin_lower_bound > 0
        assert case.bridge_acute_margin_lower_bound > 0
        assert case.left_resultant_real_bounds[0] > 1
        assert case.right_resultant_real_bounds[0] > 1
        assert case.mean_phase_bounds[0] < limiting.limiting_phase_shift_bounds[0]
        assert case.mean_phase_bounds[1] > limiting.limiting_phase_shift_bounds[1]
        assert case.bridge_phase_difference_bounds[0] < case.mean_phase_bounds[0]
        assert case.bridge_phase_difference_bounds[1] > case.mean_phase_bounds[1]
        assert case.right_no_contact_form_rate_bounds == (0, 0)
        assert (
            case.left_no_contact_form_rate_bounds[0]
            < 0
            < case.left_no_contact_form_rate_bounds[1]
        )
        assert case.right_contact_form_rate_change_bounds == (
            case.right_contact_form_rate_bounds
        )
        assert case.left_contact_form_rate_change_bounds[0] < (
            case.left_contact_form_rate_bounds[0]
        )
        assert case.left_contact_form_rate_change_bounds[1] > (
            case.left_contact_form_rate_bounds[1]
        )
        # Finite residual form and phase deviations do not justify copying
        # the limiting identity left_rate == -right_rate into this report.
        assert case.left_contact_form_rate_bounds != tuple(
            -value for value in reversed(case.right_contact_form_rate_bounds)
        )
        assert 0 < case.contact_storage_bounds[0] < case.contact_storage_bounds[1]
        left = case.left_contact_form_rate_change_bounds
        right = case.right_contact_form_rate_change_bounds
        if case.form_sign == 1:
            assert left[1] < 0 < right[0]
        else:
            assert right[1] < 0 < left[0]


def test_zero_horizon_retains_unresolved_nonzero_enclosures():
    report = certify_relational_cycle_memory_readout(decay_blocks=0)
    assert report.horizon == 0
    assert report.memory.admitted
    assert not report.admitted and report.status == "unavailable"
    assert not report.phase_signs_separated
    assert not report.contact_signs_separated
    assert not report.rate_change_signs_separated
    for case in report.cases:
        assert case.left_resultant_real_bounds[0] > 1
        assert case.right_resultant_real_bounds[0] > 1
        for bounds in (
            case.mean_phase_bounds,
            case.bridge_phase_difference_bounds,
            case.left_contact_form_rate_bounds,
            case.right_contact_form_rate_bounds,
            case.left_contact_form_rate_change_bounds,
        ):
            assert bounds[0] < 0 < bounds[1]
        assert case.contact_storage_bounds[0] == 0 < case.contact_storage_bounds[1]
        assert case.unavailable_reasons


def test_nonlinear_pressure_and_event_cost_enclose_independent_box_states(certificate):
    s = pytest.importorskip("sympy")

    def exact(value):
        return s.Rational(value.numerator, value.denominator)

    def enclosed(value, bounds):
        evaluated = Q(str(s.N(value, 90)))
        assert bounds[0] <= evaluated <= bounds[1]

    radius = exact(certificate.state_norm_upper_bound)
    kappa = 2 * s.pi / 5
    twist_cosine = (s.sqrt(5) - 1) / 4
    for case in certificate.cases:
        for direction in (-1, 1):
            # These nonzero centered states have combined norm R/2, hence
            # lie strictly inside the certified ball. They are static input
            # controls, not claimed points of the unknown recovery trajectory.
            u0, u1, u4 = direction * radius / 4, -direction * radius / 4, s.S.Zero
            v0, v1, v4 = direction * radius / 4, s.S.Zero, s.S.Zero
            m = exact(case.mean_phase_bounds[0 if direction == -1 else 1])
            delta = m + v0
            real = s.cos(kappa + v1 - v0) + s.cos(-kappa + v4 - v0) + s.cos(delta)
            imaginary = s.sin(kappa + v1 - v0) + s.sin(-kappa + v4 - v0) - s.sin(delta)
            left = -(3 * u0 - u1 - u4) / 6 + s.atan2(imaginary, real) / (2 * s.pi)
            right = u0 / 6 + s.atan2(s.sin(delta), 2 * twist_cosine + s.cos(delta)) / (
                2 * s.pi
            )
            before = -(2 * u0 - u1 - u4) / 4 + (v1 + v4 - 2 * v0) / (4 * s.pi)
            cost = u0**2 / 2 + 1 - s.cos(delta)
            for value, bounds in (
                (real, case.left_resultant_real_bounds),
                (left, case.left_contact_form_rate_bounds),
                (right, case.right_contact_form_rate_bounds),
                (before, case.left_no_contact_form_rate_bounds),
                (left - before, case.left_contact_form_rate_change_bounds),
                (right, case.right_contact_form_rate_change_bounds),
                (cost, case.contact_storage_bounds),
            ):
                enclosed(value, bounds)
            assert s.N(left + right, 90) != 0


def test_readout_never_executes_native_flow_or_a_contact(monkeypatch):
    from tnfr.dynamics import relational
    from tnfr.physics import relational_observations

    def forbidden(*args, **kwargs):
        raise AssertionError("static readout attempted live field or contact execution")

    monkeypatch.setattr(relational, "evaluate_relational_exchange", forbidden)
    monkeypatch.setattr(relational, "step_relational_exchange", forbidden)
    monkeypatch.setattr(
        relational_observations, "observe_relational_attachment", forbidden
    )
    assert certify_relational_cycle_memory_readout().admitted


@pytest.mark.parametrize("blocks", (True, False, 1.0, 0.5, Q(1), "40", None))
def test_decay_clock_requires_integral_nonboolean_blocks(blocks):
    with pytest.raises(TypeError, match="decay_blocks"):
        certify_relational_cycle_memory_readout(decay_blocks=blocks)


def test_negative_decay_horizon_rejects():
    with pytest.raises(ValueError, match="decay_blocks"):
        certify_relational_cycle_memory_readout(decay_blocks=-1)


def test_numpy_integral_clock_retains_exact_report_units():
    np = pytest.importorskip("numpy")
    report = certify_relational_cycle_memory_readout(decay_blocks=np.int64(40))
    assert report.admitted
    assert type(report.decay_blocks) is int
    assert type(report.horizon) is Q
