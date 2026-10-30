"""A slow output family retains its source and prices every discarded term.

Static algebraic witnesses verify the signed kernel, residues and nonlinear
error rows. Their formal roots are not computed poles of an acquired source.
No exponential, eigenvalue solver, source replay or trajectory is evaluated.
"""

import json
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import repeat
from math import prod

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _mixed_slow_response as owner

ERROR = Q(1, 10**10)
NOISE = Q(1, 10**9)
THRESHOLD = Q(1, 10**8)
DELTA = Q(1, 1024)
SECTION = 3 * DELTA / 16
ETA = Q(2, 10**69)
WEIGHT_ERROR = Q(2, 10**60)
MATURE = Q(1, 2**28)
SECTION_ERROR = Q(3, 2) * (14 * ETA + 2 * 10**10 * WEIGHT_ERROR)
QLO, QHI = SECTION - SECTION_ERROR / 2, SECTION + SECTION_ERROR / 2
GL, G = Q(1, 3300), Q(1, 3000)
KLO = Q(4, 9)
LAYER = Q(64)
EARLY = Q(3375000)
LATE = Q(392040000)


def _mp(mp, value):
    if isinstance(value, (int, Q, np.integer)):
        value = Q(value)
        return mp.mpf(value.numerator) / value.denominator
    return mp.mpf(value)


def _assess(**changes):
    values = dict(
        form_error_bounds=(ERROR, ERROR),
        phase_error_bounds=(ERROR, ERROR),
        early_observation_error_bound=NOISE,
        late_observation_error_bound=NOISE,
    )
    values.update(changes)
    return owner._assess_mixed_slow_response(**values)


def _roots():
    # Generic algebraic witnesses within the proved coarse root regions.
    # They are not asserted to be a realizable coefficient triple.
    return Q(1, 10**7), Q(1, 3), Q(5, 2)


def _derivatives(roots):
    return tuple(
        prod(other - rate for index, other in enumerate(roots) if index != row)
        for row, rate in enumerate(roots)
    )


def _residues(roots, beta_z, b0):
    return tuple(
        (beta_z - rate * b0) / derivative
        for rate, derivative in zip(roots, _derivatives(roots), strict=True)
    )


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_retained_producer():
    def forbidden(*args, **kwargs):
        pytest.fail("slow-response controls cannot replay or advance a source")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (_coupling_stage_kernel, "propose_coupling_stage"),
            (_coupling_stage_kernel, "propose_coupling_target"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (network_stage, "_commit_coupling_structure"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 180

    def forbidden(*args, **kwargs):
        pytest.fail("static slow-mode proof cannot evaluate an exponential or root")

    with pytest.MonkeyPatch.context() as patch:
        for name in ("exp", "expm", "eig", "polyroots", "odefun", "invertlaplace"):
            patch.setattr(context, name, forbidden)
        yield context


@pytest.fixture(scope="module")
def report():
    return _assess()


def test_nonlinear_error_rows_keep_the_phase_sum_and_the_same_initial_state(mp):
    gamma = 1 / (1023 * mp.pi)
    k = mp.mpf(13) / 27 - _mp(mp, DELTA) / (18 * mp.pi)
    coefficient = 1 + 2 * mp.cos(mp.pi / 9)
    beta, chi = gamma * coefficient / 9, 4 * gamma * k
    qhi = _mp(mp, QHI)

    def nonlinear(u, b, z, s):
        return (
            -u + b,
            u - (2 * k + 1) * b + 2 * beta * mp.sin(z / 2) * mp.cos(s / 2),
            -chi * b,
        )

    def tangent(u, b, z):
        return (-u + b, u - (2 * k + 1) * b + beta * z, -chi * b)

    full = (gamma * qhi / 3, gamma * qhi / 2, 2 * qhi / 3)
    linear = (gamma * qhi / 4, gamma * qhi / 5, qhi / 3)
    difference = tuple(a - b for a, b in zip(full, linear, strict=True))
    for s, z in ((qhi, qhi), (qhi / 2, 2 * qhi / 3), (qhi, mp.mpf(0))):
        sinc = mp.sin(z / 2) / (z / 2) if z else mp.mpf(1)
        defect = beta * z * (sinc * mp.cos(s / 2) - 1)
        assert defect <= 0
        assert abs(defect) <= beta * z * (s * s / 8 + z * z / 24)
        assert abs(defect) <= beta * qhi**3 / 6
    s = qhi / 2
    defect = beta * (2 * mp.sin(full[2] / 2) * mp.cos(s / 2) - full[2])
    actual_error = tuple(
        a - b for a, b in zip(nonlinear(*full, s), tangent(*linear), strict=True)
    )
    expected = tangent(*difference)
    expected = (expected[0], expected[1] + defect, expected[2])
    assert all(
        abs(a - b) < mp.mpf("1e-160")
        for a, b in zip(actual_error, expected, strict=True)
    )
    assert nonlinear(*full, s)[1] != nonlinear(*full, mp.mpf(0))[1]
    initial = (mp.mpf(0), gamma * qhi / 4, qhi)
    assert tuple(a - b for a, b in zip(initial, initial, strict=True)) == (0, 0, 0)
    # Same initialization gives zero state error, not zero nonlinear forcing.
    assert nonlinear(*initial, s)[1] != tangent(*initial)[1]


def test_cubic_forcing_kernel_is_signed_with_zero_dc_but_nonzero_response(report):
    roots = _roots()
    derivatives = _derivatives(roots)
    weights = tuple(
        -rate / derivative for rate, derivative in zip(roots, derivatives, strict=True)
    )
    assert weights[0] < 0 < weights[1] and weights[2] < 0
    assert sum(weights) == 0
    assert -sum(rate * weight for rate, weight in zip(roots, weights, strict=True)) == 1
    assert sum(weight / rate for rate, weight in zip(roots, weights, strict=True)) == 0
    for p in (Q(1, 9), Q(1), Q(3)):
        assert sum(
            weight / (p + rate) for rate, weight in zip(roots, weights, strict=True)
        ) == p / prod(p + rate for rate in roots)
    absolute_mass = sum(1 / abs(value) for value in derivatives)
    assert absolute_mass == 2 / ((roots[1] - roots[0]) * (roots[2] - roots[1])) < 6
    assert Q(1, 4) - Q(1, 36) == Q(2, 9)
    assert Q(2) - Q(1, 2) == Q(3, 2)
    assert 2 / (Q(2, 9) * Q(3, 2)) == 6
    assert report.kernel_abs_mass_upper_bound == 6
    assert report.kernel_signed_integral == 0 and not report.kernel_is_nonnegative
    assert report.kernel_transfer_numerator == "p"
    assert report.signed_kernel_certified
    # The initial derivative is positive and the slow residue is negative:
    # a zero signed integral cannot justify dropping accumulated forcing.
    assert absolute_mass > 0


def test_fast_residues_cancel_the_initial_slow_term_and_have_a_paid_tail(report):
    roots = _roots()
    beta_z = G * QHI / 3
    bmax = Q(9, 16) * G * QHI
    b0 = bmax / 2
    residues = _residues(roots, beta_z, b0)
    assert sum(residues) == 0
    assert -sum(rate * value for rate, value in zip(roots, residues, strict=True)) == b0
    assert residues[0] > 0 and residues[1] + residues[2] == -residues[0]
    upper = 3 * (beta_z + bmax / 2) + (beta_z + 3 * bmax) / 2
    assert upper == Q(137, 48) * G * QHI < 3 * G * QHI
    assert abs(residues[1]) + abs(residues[2]) <= upper
    assert Q(1, 4) * LAYER == 16
    assert report.fast_residue_factor == Q(137, 48)
    assert report.fast_residue_upper_bound == 3 * G * QHI
    assert report.fast_output_tail_upper_bound == 27 * G * QHI / 2**16
    assert report.fast_tail_certified
    assert report.all_three_residues_retained_before_layer


def test_slow_amplitude_bounds_come_from_the_carried_correlations(report):
    zlo = Q(233, 300) * QLO
    beta_lo, beta_hi = 2 * GL / 9, G / 3
    numerator_lower = beta_lo * zlo * (1 - G**2 / KLO)
    lower = 9 * numerator_lower / Q(3, 2)
    upper_from_denominator = 9 * beta_hi * QHI / Q(15, 64)
    assert numerator_lower > 0
    assert lower == Q(4, 3) * GL * zlo * (1 - G**2 / KLO)
    assert upper_from_denominator == Q(64, 5) * G * QHI < 16 * G * QHI
    assert report.slow_amplitude_bounds == (lower, 16 * G * QHI)
    assert Q(1, 4) - Q(1, 8) > 0
    assert (Q(1, 4) - Q(1, 8)) * (Q(2) - Q(1, 8)) == Q(15, 64)
    assert G**2 < KLO
    assert report.amplitude_certified and report.correlated_initialization_retained
    assert report.slow_amplitude_expression == "9*(beta*z0-r_s*b0)/P_prime(-r_s)"
    assert not report.amplitude_rate_marginals_independently_attainable


def test_uniform_approximation_adds_three_distinct_errors_without_time_growth(report):
    initial = 8 * MATURE / (1 + Q(10000, 3)) + 6 * G * 8 * MATURE + 6 * WEIGHT_ERROR
    fine = 200 * MATURE
    forcing = 2 * G * fine / 9 + 5 * G * fine**2 / 81 + 14 * WEIGHT_ERROR / 2700
    source_error = 3 * initial + Q(81, 8) * forcing
    nonlinear = 3 * G * QHI**3
    fast = 27 * G * QHI / 2**16
    total = source_error + nonlinear + fast
    assert report.original_output_error_upper_bound == source_error
    assert report.nonlinear_forcing_upper_bound == G * QHI**3 / 18
    assert 9 * 6 * report.nonlinear_forcing_upper_bound == nonlinear
    assert report.nonlinear_output_error_upper_bound == nonlinear
    assert report.total_output_error_upper_bound == total < NOISE
    assert report.approximation_margin == NOISE - total > 0
    assert report.layer_age == LAYER
    assert report.uniform_approximation_certified and report.nonlinear_error_certified
    assert source_error < total and nonlinear > 0 and fast > 0
    assert not report.relative_rate_error_certified


def test_fixed_slow_window_certifies_true_and_recorded_bands(report, record_property):
    rho_lo, rho_hi = 4 * GL**2 / 9, 2 * G**2 / 3
    assert rho_lo == Q(1, 24502500)
    assert EARLY == 1 / (4 * rho_hi) and LATE == 16 / rho_lo
    assert 2 * rho_hi * EARLY == Q(1, 2)
    assert rho_lo * LATE == 16
    assert LAYER < EARLY < LATE < report.earlier_sufficient_deadline
    assert report.slow_scale_bounds == (rho_lo, rho_hi)
    assert report.early_age == EARLY and report.late_age == LATE
    alo, ahi = report.slow_amplitude_bounds
    error = report.total_output_error_upper_bound
    early = (alo / 2 - error, ahi + error)
    late_upper = ahi / 2**16 + error
    assert report.early_true_bounds == early
    assert report.late_true_bounds == (-late_upper, late_upper)
    assert late_upper < NOISE
    assert report.early_record_bounds == (early[0] - NOISE, early[1] + NOISE)
    assert report.late_record_bounds == (-late_upper - NOISE, late_upper + NOISE)
    assert report.early_threshold_margin == early[0] - NOISE - THRESHOLD > 0
    assert report.late_true_return_margin == NOISE - late_upper > 0
    assert report.late_threshold_margin == THRESHOLD - late_upper - NOISE > 0
    assert report.early_response_certified and report.early_record_certified
    assert (
        report.late_return_certified
        and report.late_record_certified
        and report.certified
    )
    # These exact receipts reuse this assessment. Reporting their values
    # must not trigger a new source, exponential or endpoint evaluation.
    for name in (
        "original_output_error_upper_bound",
        "nonlinear_output_error_upper_bound",
        "fast_output_tail_upper_bound",
        "total_output_error_upper_bound",
        "approximation_margin",
        "layer_age",
        "early_age",
        "late_age",
        "threshold",
        "early_threshold_margin",
        "late_true_return_margin",
        "late_threshold_margin",
        "correlated_true_drop_lower_bound",
        "correlated_record_drop_lower_bound",
    ):
        record_property(name, str(getattr(report, name)))
    for name in (
        "slow_amplitude_bounds",
        "slow_scale_bounds",
        "early_true_bounds",
        "late_true_bounds",
        "early_record_bounds",
        "late_record_bounds",
    ):
        record_property(
            name, json.dumps([str(value) for value in getattr(report, name)])
        )
    record_property(
        "nominal_tangent_slow_rate_scale_bounds",
        json.dumps(
            [
                str(report.slow_rate_scale_bounds.lo),
                str(report.slow_rate_scale_bounds.hi),
            ]
        ),
    )


def test_correlated_decrement_keeps_one_amplitude_and_two_independent_errors(report):
    alo, ahi = report.slow_amplitude_bounds
    error = report.total_output_error_upper_bound
    true_drop = alo * (Q(1, 2) - Q(1, 2**16)) - 2 * error
    record_drop = true_drop - 2 * NOISE
    naive = report.early_true_bounds[0] - report.late_true_bounds[1]
    assert true_drop - naive == (ahi - alo) / 2**16 > 0
    assert report.correlated_true_drop_lower_bound == true_drop > 0
    assert report.correlated_record_drop_lower_bound == record_drop > 0
    assert report.correlated_true_drop_certified and report.correlated_drop_certified
    zero = _assess(
        early_observation_error_bound=Q(0), late_observation_error_bound=Q(0)
    )
    assert zero.early_record_bounds == zero.early_true_bounds
    assert zero.late_record_bounds == zero.late_true_bounds
    assert zero.correlated_record_drop_lower_bound == true_drop
    for name in ("early_observation_error_bound", "late_observation_error_bound"):
        changed = _assess(**{name: NOISE + Q(1, 10**100)})
        assert not changed.observation_allowances_admitted and not changed.certified
        assert not changed.early_record_certified and not changed.late_record_certified
        assert not changed.correlated_drop_certified
        assert changed.early_response_certified and changed.late_return_certified
        assert changed.correlated_true_drop_certified


def test_parent_derived_bounds_and_implicit_roots_are_rebuilt(monkeypatch, report):
    source = report.source
    cases = (
        ("source_association_certified", False),
        ("projected_rows_certified", False),
        ("nominal_cone_certified", False),
        ("uniform_output_certified", False),
        ("uniform_output_error_upper_bound", Q(0)),
        ("forcing_error_upper_bound", Q(0)),
        ("kernel_mass_upper_bound", Q(0)),
        ("cut_form_error_upper_bound", Q(0)),
        ("full_state_comparison_upper_bound", Q(0)),
        ("beta_bounds", I(0)),
        ("chi_bounds", I(0)),
        ("stiffness_bounds", I(0)),
        ("slow_rate_scale_bounds", I(0)),
        ("characteristic_coefficients", (I(1), I(1), I(1), I(1))),
        ("decay_rate_brackets", ((I(0), I(1)),) * 3),
        ("tangent_root_sign_bounds", (I(0),) * 6),
        ("tangent_laplace_numerator", "beta*z0"),
        ("tangent_initialization", "u0 set again at64"),
        ("tangent_coefficients_correlated", False),
        ("nonlinear_sum_phase_retained", False),
    )
    for name, value in cases:
        changed = replace(source, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_collective_relaxation", lambda **kwargs: changed
        )
        assert not _assess().certified


def test_source_history_phase_leaf_storage_and_clocks_remain_prerequisites(
    monkeypatch, report
):
    source = report.source
    for name, value in (
        ("clock", "wrong collective clock"),
        ("activation_time", Q(0)),
        ("initial_common_form_difference", Q(1)),
        ("whole_form_mean", Q(1)),
        ("full_observation_error_bound", NOISE),
        ("state_reset_performed", True),
    ):
        changed = replace(source, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_collective_relaxation", lambda **kwargs: changed
        )
        assert not _assess().certified
    persistent = source.source
    for name, value in (
        ("clock", "wrong persistent clock"),
        ("broad_trap_certified", False),
        ("uniform_tube_certified", False),
        ("selected_mu", Q(2)),
        ("original_phase_gauges_and_branches_retained", False),
    ):
        changed = replace(source, source=replace(persistent, **{name: value}))
        monkeypatch.setattr(
            owner, "_assess_mixed_collective_relaxation", lambda **kwargs: changed
        )
        assert not _assess().certified
    storage = persistent.source
    for name, value in (
        ("storage_domain_certified", False),
        ("actual_dissipation_certified", False),
        ("origin_reference", "uncentered form origin"),
    ):
        changed = replace(
            source, source=replace(persistent, source=replace(storage, **{name: value}))
        )
        monkeypatch.setattr(
            owner, "_assess_mixed_collective_relaxation", lambda **kwargs: changed
        )
        assert not _assess().certified
    monkeypatch.setattr(
        owner, "_assess_mixed_collective_relaxation", lambda **kwargs: source
    )
    registered = owner._derive_registered_phase_probe_geometry()
    for name, value in (
        ("bare_gap_lower_bound", Q(3)),
        ("spectral_gap_certified", False),
    ):
        altered = replace(
            registered, successor=replace(registered.successor, **{name: value})
        )
        monkeypatch.setattr(
            owner, "_derive_registered_phase_probe_geometry", lambda: altered
        )
        assert not _assess().certified


def test_old_record_comparator_and_global_verdicts_are_not_slow_response_premises(
    monkeypatch, report
):
    source = replace(
        report.source,
        certified=False,
        full_persistence_certified=False,
        form_only_gap_certified=False,
        independent_record_sets_separated=False,
        fixed_threshold_certified=False,
        full_record_bounds=(Q(0), Q(0)),
        form_only_record_bounds=(Q(0), Q(0)),
        full_true_bounds=(Q(0), Q(0)),
        tangent_certified=False,
        nominal_slow_mode_excited_certified=False,
        status="unrelated_old_result_unused",
        unresolved_conditions=("unused",),
    )
    monkeypatch.setattr(
        owner, "_assess_mixed_collective_relaxation", lambda **kwargs: source
    )
    result = _assess()
    assert (
        result.certified
        and result.tangent_roots_certified
        and result.amplitude_certified
    )
    assert (
        result.total_output_error_upper_bound == report.total_output_error_upper_bound
    )
    assert (
        result.correlated_record_drop_lower_bound
        == report.correlated_record_drop_lower_bound
    )


def test_raw_error_pairs_and_readout_budgets_precede_all_source_work(monkeypatch):
    for name in ("form_error_bounds", "phase_error_bounds"):
        assert not _assess(**{name: (ERROR + Q(1, 10**100), ERROR)}).certified

    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached source or derived arithmetic")

    monkeypatch.setattr(owner, "_assess_mixed_collective_relaxation", forbidden)
    invalid = (
        True,
        np.bool_(False),
        None,
        "0",
        float("nan"),
        float("inf"),
        complex(1, 2),
        -Q(1, 10**100),
    )
    for name in ("form_error_bounds", "phase_error_bounds"):
        for bad in invalid:
            for side in (0, 1):
                value = tuple(bad if index == side else Q(0) for index in range(2))
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    _assess(**{name: value})
        for bad in ((0,), (0, 0, 0), {0, 1}, {0: 0, 1: 0}, repeat(0)):
            with pytest.raises((TypeError, ValueError)):
                _assess(**{name: bad})
    for name in ("early_observation_error_bound", "late_observation_error_bound"):
        for bad in invalid:
            with pytest.raises((TypeError, ValueError, OverflowError)):
                _assess(**{name: bad})


def test_fixed_width_one_shot_admission_never_fits_or_reinitializes_a_slow_mode(report):
    error = Q(np.int64(ERROR.numerator), np.int64(ERROR.denominator))
    noise = Q(np.int64(NOISE.numerator), np.int64(NOISE.denominator))
    assert (
        _assess(
            form_error_bounds=(error for _ in range(2)),
            phase_error_bounds=(error for _ in range(2)),
            early_observation_error_bound=noise,
            late_observation_error_bound=noise,
        )
        == report
    )
    assert report.initialization == "u0=0; b0 and z0 carried from the same original cut"
    assert report.same_actual_history_required and report.independent_record_errors
    assert not report.observation_error_cancellation_assumed
    assert report.nonlinear_phase_sum_retained
    assert (
        not report.fitted_amplitude_or_rate and not report.old_record_verdict_consumed
    )
    assert not report.exponentials_evaluated and not report.roots_numerically_evaluated
    assert not report.source_trajectory_evaluated and not report.state_reset_performed
    assert not report.exact_actual_modal_closure and not report.measured_rate_identified
    assert (
        not report.relative_rate_error_certified
        and not report.first_return_time_identified
    )
    assert not report.physical_timescale_identified and not report.empirical_validation
    assert (
        report.newly_observed_aggregates == 0 and report.required_endpoint_records == 2
    )
    assert report.numerical_response_allowance == 0
    assert report.earlier_sufficient_deadline == 627264000000000
    with pytest.raises(FrozenInstanceError):
        report.layer_age = Q(0)
    for name in (
        "source",
        "endpoint",
        "amplitude",
        "slow_rate",
        "root",
        "horizon",
        "layer_age",
        "initial_velocity",
        "initial_phase",
        "observed_record",
        "threshold",
        "port_strength",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: Q(0)})
