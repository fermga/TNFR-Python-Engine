"""Actual form dissipation supports one longer conditional validity window.

Static full-state witnesses test the damping mechanism, not acquired
endpoints. Comparison growth is a limit of the certificate, not evidence
of actual instability or a measured lifetime.
"""

import json
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import repeat

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I, cos, pi_interval
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _mixed_phase_sensitive_interface as phase_owner
from tnfr.physics import _mixed_sustained_interaction as owner
from tnfr.physics import _sine_flow as flow_owner

ERROR = Q(1, 10**10)
NOISE = Q(1, 10**9)
DELTA = Q(1, 1024)
SECTION = 3 * DELTA / 16
ETA = Q(2, 10**69)
WEIGHT_ERROR = Q(2, 10**60)
MATURE = Q(1, 2**28)
SECTION_ERROR = Q(3, 2) * (14 * ETA + 2 * 10**10 * WEIGHT_ERROR)
QLO, QHI = SECTION - SECTION_ERROR / 2, SECTION + SECTION_ERROR / 2
G = Q(1, 3000)
SWITCH_AGE = Q(3375000)
HORIZON = Q(4096)


def _mp(mp, value):
    if isinstance(value, (int, Q, np.integer)):
        value = Q(value)
        return mp.mpf(value.numerator) / value.denominator
    return mp.mpf(value)


def _dot(first, second):
    return sum(a * b for a, b in zip(first, second, strict=True))


def _center(values):
    mean = sum(values) / len(values)
    return tuple(value - mean for value in values)


def _turns():
    return tuple(Q((1 + j // 9) * (j % 9 - 4), 9) for j in range(18))


def _support():
    bins = tuple(int((9 * turn) % 9) for turn in _turns())
    old = tuple(
        sorted(
            tuple(sorted((9 * p + j, 9 * p + (j + 1) % 9)))
            for p in (0, 1)
            for j in range(9)
        )
    )
    edges = {
        (a, b)
        for a in range(18)
        for b in range(a + 1, 18)
        if min((bins[b] - bins[a]) % 9, (bins[a] - bins[b]) % 9) in (1, 2)
        or (a // 9 != b // 9 and bins[a] == bins[b])
    }
    return old, tuple(sorted(edges - set(old)))


def _nominal_weights(mp):
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + (-1 if j < 9 else 1) * _mp(mp, DELTA) / 2
        for j, turn in enumerate(_turns())
    )
    return tuple(
        1 - abs((theta[b] - theta[a] + mp.pi) % (2 * mp.pi) - mp.pi) / (2 * mp.pi)
        for a, b in _support()[1]
    )


def _weights(mp, side):
    values = tuple(
        Q(mp.nstr(value, 210))
        + (1 if value < _mp(mp, Q(7, 8)) else -1)
        * WEIGHT_ERROR
        * Q(1 + (j + side) % 3, 6)
        for j, value in enumerate(_nominal_weights(mp))
    )
    assert all(Q(3, 4) <= value <= 1 for value in values)
    return values


def _edges(weights):
    old, new = _support()
    return tuple((a, b, Q(1)) for a, b in old) + tuple(
        (a, b, weight) for (a, b), weight in zip(new, weights, strict=True)
    )


def _h_apply(mp, values, weights):
    out = [mp.mpf(0)] * 36
    for side in (0, 1):
        for a, b, weight in _edges(weights[side]):
            a, b = a + 18 * side, b + 18 * side
            current = _mp(mp, weight) * (values[a] - values[b])
            out[a] += current
            out[b] -= current
    return tuple(out)


def _sine(mp, phases):
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, phase)
        for turn, phase in zip(_turns() * 2, phases, strict=True)
    )
    out = [mp.mpf(0)] * 36
    for side in (0, 1):
        for a, b, _ in _edges((Q(1),) * 63):
            a, b = a + 18 * side, b + 18 * side
            current = mp.sin(theta[b] - theta[a])
            out[a] += current
            out[b] -= current
    return tuple(out)


def _exchange(values, slots=18):
    first = tuple(values[j] - values[18 + j] if j < slots else 0 for j in range(18))
    return first + tuple(-value for value in first)


def _field(mp, forms, phases, weights):
    x = tuple(_mp(mp, value) for value in forms)
    hx, sine, gx = _h_apply(mp, x, weights), _sine(mp, phases), _exchange(x)
    gamma = 1 / (1023 * mp.pi)
    cross = tuple(
        mp.sin(_mp(mp, phases[j + 18]) - _mp(mp, phases[j])) for j in range(18)
    )
    cross = cross + tuple(-value for value in cross)
    lx = tuple(a + 9 * b for a, b in zip(hx, gx, strict=True))
    splus = tuple(a + 9 * b for a, b in zip(sine, cross, strict=True))
    return tuple((-a + gamma * b) / 9 for a, b in zip(lx, splus, strict=True)), tuple(
        gamma * a / 9 for a in lx
    )


def _state(forms, phases):
    return tuple(I(value) for value in forms) + tuple(
        2 * pi_interval() * turn + I(value)
        for turn, value in zip(_turns() * 2, phases, strict=True)
    )


def _new_field(weights):
    return phase_owner._mixed_phase_sensitive_field(
        first_new_edge_conductances=weights[0],
        second_new_edge_conductances=weights[1],
    )


def _assess(**changes):
    values = dict(
        form_error_bounds=(ERROR, ERROR),
        phase_error_bounds=(ERROR, ERROR),
        registered_observation_error_bound=NOISE,
        aligned_observation_error_bound=NOISE,
    )
    values.update(changes)
    return owner._assess_mixed_sustained_interaction(**values)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_retained_producer():
    static_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("sustained controls cannot replay or advance a source")

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
        # Permit the existing row builder for static full-state comparisons;
        # every integrator, acquisition producer and operator remains blocked.
        with pytest.MonkeyPatch.context() as static:
            static.setattr(flow_owner, "_sine_rate_evaluator", static_factory)
            yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 220

    def forbidden(*args, **kwargs):
        pytest.fail("static sustained proof cannot evaluate an exponential or root")

    with pytest.MonkeyPatch.context() as patch:
        for name in ("exp", "expm", "eig", "polyroots", "odefun", "invertlaplace"):
            patch.setattr(context, name, forbidden)
        yield context


@pytest.fixture(scope="module")
def report():
    return _assess()


def _copy_center(values):
    return _center(values[:18]) + _center(values[18:])


def _potential(mp, phases):
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, phase)
        for turn, phase in zip(_turns() * 2, phases, strict=True)
    )
    return sum(
        1 - mp.cos(theta[18 * side + b] - theta[18 * side + a])
        for side in (0, 1)
        for a, b, _ in _edges((Q(1),) * 63)
    )


def test_actual_weighted_centered_gap_survives_unequal_matching_blocks(mp):
    old, new = _support()
    bins = tuple(int((9 * turn) % 9) for turn in _turns())
    support = set(old) | set(new)
    for first in range(18):
        for second in range(first + 1, 18):
            delta = (bins[second] - bins[first]) % 9
            same_layer = first // 9 == second // 9
            expected = delta in (1, 2, 7, 8) or (not same_layer and delta == 0)
            assert ((first, second) in support) == expected
    # In phase-bin order the common-layer symbols are explicit Fourier
    # values; antisymmetric-layer symbols equal10. No eigen solver is used.
    for mode in range(1, 9):
        symbol = (
            8
            - 4 * cos(2 * pi_interval() * Q(mode, 9))
            - 4 * cos(4 * pi_interval() * Q(mode, 9))
        )
        assert symbol.lo >= 4
    assert 10 >= 4
    weights = (_weights(mp, 0), _weights(mp, 1))
    vector = _copy_center(tuple(Q((j * 7) % 17 - 8, 10000) for j in range(36)))
    square = sum(value**2 for value in vector)
    bare = sum(
        (vector[18 * side + a] - vector[18 * side + b]) ** 2
        for side in (0, 1)
        for a, b in support
    )
    weighted = sum(
        weight * (vector[18 * side + a] - vector[18 * side + b]) ** 2
        for side in (0, 1)
        for a, b, weight in _edges(weights[side])
    )
    paired = sum((vector[j] - vector[j + 18]) ** 2 for j in range(18))
    assert weighted >= Q(3, 4) * bare >= 3 * square
    assert weighted + 9 * paired >= weighted
    assert weighted + 9 * paired <= 36 * square
    # The proof needs no commutation of actual unequal internal blocks with G.
    vector = _copy_center((Q(1), Q(-1)) + (Q(0),) * 34)
    x = tuple(_mp(mp, value) for value in vector)
    hg = _h_apply(mp, _exchange(x), weights)
    gh = _exchange(_h_apply(mp, x, weights))
    assert max(abs(a - b) for a, b in zip(hg, gh, strict=True)) > mp.mpf("1e-80")


def test_complete_centered_rows_obey_component_dissipative_bounds(mp):
    weights = (_weights(mp, 0), _weights(mp, 1))
    centered_x = _copy_center(tuple(Q((j * 3) % 11 - 5, 10**5) for j in range(36)))
    centered_y = _copy_center(tuple(Q((j * 5) % 7 - 3, 10**5) for j in range(36)))
    common = (-Q(1, 1000),) * 18 + (Q(1, 1000),) * 18
    forms = tuple(a + b for a, b in zip(centered_x, common, strict=True))
    xc = tuple(_mp(mp, v) for v in centered_x)
    yc = tuple(_mp(mp, v) for v in centered_y)
    xnorm, ynorm = mp.sqrt(_dot(xc, xc)), mp.sqrt(_dot(yc, yc))
    flow = _new_field(weights)
    for shift, angle in ((I(0), mp.mpf(0)), (pi_interval() / 2, mp.pi / 2)):
        phases = tuple(
            _mp(mp, v) + (angle if j >= 18 else 0) for j, v in enumerate(centered_y)
        )
        dx, dy = _field(mp, forms, phases, weights)
        state = _state(forms, centered_y)
        state = state[:54] + tuple(v + shift for v in state[54:])
        for interval, value in zip(flow(state), dx + dy, strict=True):
            assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
        cdx, cdy = _copy_center(dx), _copy_center(dy)
        assert _dot(xc, cdx) <= -(xnorm**2) / 3 + 4 * _mp(mp, G) * xnorm * ynorm
        assert _dot(yc, cdy) <= 4 * _mp(mp, G) * xnorm * ynorm
        assert _dot(cdy, cdy) <= 16 * _mp(mp, G) ** 2 * xnorm**2
        lx = tuple(
            a + 9 * b
            for a, b in zip(_h_apply(mp, xc, weights), _exchange(xc), strict=True)
        )
        assert _dot(xc, lx) >= 3 * xnorm**2
    # Removing common form is a projection of the actual moving means,
    # not a claim that the new phase port preserves each copy gauge.
    assert abs(sum(dy[:18])) > mp.mpf("1e-6")


def test_damped_form_filter_and_phase_feedback_share_one_budget(report):
    tube = report.centered_envelope
    x0, y0, h = tube.initial_form_norm, tube.initial_phase_norm, tube.duration
    ybar, xbar = tube.uniform_phase_norm_upper_bound, tube.uniform_form_norm_upper_bound
    integral = tube.form_norm_integral_upper_bound
    assert tube.feedback_denominator == 1 - 48 * G**2 * h > 0
    assert integral == 3 * x0 + 12 * G * h * ybar
    assert ybar == y0 + 4 * G * integral
    assert xbar == max(x0, 12 * G * ybar)
    assert tube.phase_displacement_upper_bound == 4 * G * integral == ybar - y0
    assert tube.scalar_decay_upper_bound == Q(3) / (3 + h)
    assert (
        tube.endpoint_form_norm_upper_bound
        == x0 * tube.scalar_decay_upper_bound + 12 * G * ybar
    )
    # Scalar Duhamel with constant upper forcing is a convex combination.
    # d is a formal decay factor, not an evaluated exponential.
    for d in (Q(0), Q(1, 3), Q(1)):
        assert d * x0 + (1 - d) * 12 * G * ybar <= xbar
    assert tube.endpoint_form_norm_upper_bound < x0
    zero = owner._damped_centered_envelope(
        initial_form_norm=0, initial_phase_norm=0, duration=4096
    )
    assert (
        zero.uniform_form_norm_upper_bound
        == zero.uniform_phase_norm_upper_bound
        == zero.phase_displacement_upper_bound
        == 0
    )
    pure = owner._damped_centered_envelope(
        initial_form_norm=1, initial_phase_norm=0, duration=1
    )
    assert pure.uniform_phase_norm_upper_bound > 12 * G
    assert pure.uniform_form_norm_upper_bound == 1
    assert pure.form_norm_integral_upper_bound > 3


def test_phase_moment_uses_phase_residual_not_common_form(mp, report):
    centered = _copy_center(tuple(Q(j % 5 - 2, 10000) for j in range(36)))
    yc = tuple(_mp(mp, v) for v in centered)
    phase_square = _dot(yc, yc)
    assert (
        report.phase_moment_remainder_upper_bound
        == report.centered_envelope.uniform_phase_norm_upper_bound**2
    )
    for psi in (mp.mpf(0), mp.pi / 2):
        delta = tuple(psi + yc[j + 18] - yc[j] for j in range(18))
        remainder = sum(mp.sin(v) for v in delta) - 18 * mp.sin(psi)
        assert abs(remainder) <= phase_square
    # Large common form changes neither this residual nor centered phase;
    # inserting its norm into R would discard useful task precision.
    common = (-mp.mpf("0.01"),) * 18 + (mp.mpf("0.01"),) * 18
    assert _dot(common, common) > phase_square
    assert max(abs(value) for value in _copy_center(common)) < mp.mpf("1e-200")
    zero = (mp.mpf(0),) * 36
    weights = (_weights(mp, 0), _weights(mp, 1))
    phases = tuple(v + (mp.pi / 2 if j >= 18 else 0) for j, v in enumerate(yc))
    first = _field(mp, zero, phases, weights)[1]
    second = _field(mp, common, phases, weights)[1]
    assert max(
        abs(a - b)
        for a, b in zip(_copy_center(first), _copy_center(second), strict=True)
    ) < mp.mpf("1e-200")


def test_new_marked_edge_drift_replaces_unavailable_joint_growth(report):
    tube = report.centered_envelope
    entry = 4 * QHI + 200 * MATURE
    assert (
        tube.initial_form_norm
        == tube.initial_phase_norm
        == report.centered_entry_norm_upper_bound
        == entry
    )
    # These are separate conservative bounds, not a simultaneously attained
    # source corner; the original joint source allowance remains N0.
    assert report.old_joint_growth_denominator == 1 - 4 * G * HORIZON < 0
    assert tube.feedback_denominator > 0
    assert (
        tube.uniform_form_norm_upper_bound + tube.uniform_phase_norm_upper_bound
        < Q(1, 100)
        < Q(1, 32)
    )
    drift = Q(3, 2) * tube.phase_displacement_upper_bound
    assert Q(3, 2) ** 2 > 2
    assert report.marked_gap_upper_bound == QHI + 300 * MATURE + drift
    assert report.registration_margin == DELTA / 4 - report.marked_gap_upper_bound > 0
    aggregate = (
        report.slow_amplitude_bounds[1]
        + report.inherited_output_error_upper_bound
        + 9 * G
    )
    phase = 6 * ETA + 2 * G * HORIZON * aggregate / 9
    assert report.aggregate_absolute_upper_bound == aggregate
    assert report.relative_phase_deviation_upper_bound == phase
    assert (
        report.boundary_branch_margin
        == Q(1, 4) - phase - 2 * tube.uniform_phase_norm_upper_bound
        > 0
    )
    assert report.weighted_gap_certified and report.damped_centered_bound_certified
    assert report.centered_identity_certified and report.phase_branch_certified


def test_collective_error_prices_accumulated_phase_feedback(report):
    a, b, r = (
        report.inherited_output_error_upper_bound,
        6 * ETA,
        report.phase_moment_remainder_upper_bound,
    )
    denominator = 1 - 2 * G**2 * HORIZON
    matrix = ((Q(1), -9 * G), (-2 * G * HORIZON / 9, Q(1)))
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    assert determinant == denominator == report.prediction_feedback_denominator > 0
    u, v = report.uniform_form_error_upper_bound, report.phase_error_upper_bound
    assert matrix[0][0] * u + matrix[0][1] * v == a + G * r / 2
    assert matrix[1][0] * u + matrix[1][1] * v == b
    assert report.prediction_decay_upper_bound == Q(1, 2**32)
    endpoint = a * report.prediction_decay_upper_bound + 9 * G * v + G * r / 2
    assert report.endpoint_form_error_upper_bound == endpoint < u < NOISE
    assert 0 < v < NOISE
    assert report.form_prediction_margin == NOISE - max(u, endpoint) > 0
    assert report.phase_prediction_margin == NOISE - v > 0
    assert report.form_prediction_certified and report.phase_prediction_certified
    # A constant error bound alone gives no uniform-in-time phase bound.
    # This admissible comparison pair is not a TNFR trajectory witness.
    constant_error = Q(1, 10**9)
    early_integral_bound = 2 * G * HORIZON * constant_error / 9
    doubled_integral_bound = 2 * G * (2 * HORIZON) * constant_error / 9
    assert doubled_integral_bound == 2 * early_integral_bound > early_integral_bound


def test_longer_window_retains_storage_work_and_exchange(mp, report):
    tube = report.centered_envelope
    xbar, ybar = tube.uniform_form_norm_upper_bound, tube.uniform_phase_norm_upper_bound
    aggregate, remainder = (
        report.aggregate_absolute_upper_bound,
        report.phase_moment_remainder_upper_bound,
    )
    assert report.storage_remainder_upper_bound == 18 * (xbar**2 + ybar**2)
    assert (
        report.unresolved_exchange_integral_upper_bound
        == 2 * G * HORIZON * aggregate * remainder
    )
    assert (
        report.storage_prediction_error_upper_bound
        == report.storage_remainder_upper_bound
        + 2 * aggregate * report.uniform_form_error_upper_bound
        + 162 * report.phase_error_upper_bound
    )
    entry = report.centered_entry_norm_upper_bound
    work = 972 * ETA + Q(9, 2) * entry**3
    assert report.phase_switching_work_allowance == work
    assert report.aligned_storage_jump_bounds == (Q(0), 9 * (entry + 18 * ETA) ** 2)
    assert report.registered_storage_jump_bounds == (
        162 - work,
        162 + work + 9 * entry**2,
    )
    weights = (_weights(mp, 0), _weights(mp, 1))
    forms = _center(tuple(Q((j * 7) % 17 - 8, 10**5) for j in range(36)))
    centered = _copy_center(tuple(Q((j * 3) % 11 - 5, 10**5) for j in range(36)))
    x = tuple(_mp(mp, v) for v in forms)
    xc = _copy_center(x)
    yc = tuple(_mp(mp, v) for v in centered)
    phases = tuple(v + (mp.pi / 2 if j >= 18 else 0) for j, v in enumerate(yc))
    q, psi = -sum(x[:18]), (sum(phases[18:]) - sum(phases[:18])) / 18
    lx = tuple(
        a + 9 * b for a, b in zip(_h_apply(mp, x, weights), _exchange(x), strict=True)
    )
    internal = _potential(mp, phases) - _potential(mp, (Q(0),) * 36)
    paired = sum(1 - mp.cos(phases[j + 18] - phases[j]) for j in range(18))
    full = _dot(x, lx) / 2 + internal + 9 * paired
    coarse = q**2 + 162 * (1 - mp.cos(psi))
    assert (
        abs(full - coarse)
        <= 18 * (_dot(xc, xc) + _dot(yc, yc))
        <= _mp(mp, report.storage_remainder_upper_bound)
    )
    assert report.switching_work_certified and report.effective_balance_certified


def test_comparison_positive_mode_does_not_prove_actual_instability(report):
    matrix = ((-Q(1, 3), 4 * G), (4 * G, Q(0)))
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    witness = (12 * G, Q(1))
    applied = tuple(
        sum(matrix[row][column] * witness[column] for column in (0, 1))
        for row in (0, 1)
    )
    rayleigh = _dot(witness, applied)
    assert determinant == -16 * G**2 < 0
    assert applied == (Q(0), 48 * G**2)
    assert rayleigh == 48 * G**2 > 0
    assert report.comparison_matrix == matrix
    assert report.comparison_determinant == determinant
    assert report.comparison_positive_witness == witness
    assert report.comparison_positive_quadratic_value == rayleigh
    assert report.comparison_limitation_certified
    # A symmetric comparison matrix with negative determinant has one
    # positive eigenvalue. No eigenvalue or actual unstable response is read.


def _replace_at(value, path, **changes):
    if not path:
        return replace(value, **changes)
    return replace(
        value, **{path[0]: _replace_at(getattr(value, path[0]), path[1:], **changes)}
    )


def test_source_gap_history_and_initial_associations_are_readmitted(
    monkeypatch, report
):
    source = report.source
    cases = (
        ((), "source_association_certified", False),
        ((), "registration_covariance_certified", False),
        ((), "inherited_output_error_upper_bound", Q(0)),
        ((), "initial_relative_phase_error_upper_bound", Q(0)),
        ((), "centered_entry_norm_upper_bound", Q(0)),
        ((), "initial_registrations_turns", (Q(0), Q(0))),
        ((), "source_correlated_initialization_retained", False),
        ((), "whole_history_covariance_retained", False),
        ((), "state_reset_performed", True),
        ((), "phase_clamped", True),
        ((), "pressure_selected_from_preparation_label", True),
        ((), "common_storage_reference", "changed reference"),
        ((), "post_duration", Q(4096)),
        ((), "switch_time", Q(5385016)),
        ((), "clock", "wrong effective clock"),
        ((), "registered_observation_error_bound", NOISE),
        ((), "physical_capacities", (Q(0),) + (Q(1),) * 35),
        (("source",), "complete_precontact_history_translated", False),
        (("source",), "registration_turns", Q(0)),
        (("source",) * 2, "initial_relative_phase_mean_assumed_zero", True),
        (("source",) * 2, "phase_mean_error_upper_bound", Q(0)),
        (("source",) * 3, "clock", "wrong renewed clock"),
        (("source",) * 4, "signed_kernel_certified", False),
        (("source",) * 4, "fast_output_tail_upper_bound", Q(0)),
        (("source",) * 5, "cut_form_error_upper_bound", Q(0)),
        (("source",) * 6, "uniform_tube_certified", False),
        (("source",) * 6, "bregman_denominator", Q(1)),
        (("source",) * 7, "actual_dissipation_certified", False),
        (("source",) * 8, "exact_zero_entry_total_from_prior_laws", False),
        (("source",) * 8, "prefix_cut_duration", Q(10001)),
    )
    for path, name, value in cases:
        changed = _replace_at(source, path, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_effective_interaction", lambda **kwargs: changed
        )
        rejected = _assess()
        assert not rejected.certified and not rejected.weighted_gap_certified
        assert not rejected.centered_identity_certified
    path = ("source",) * 8 + ("handoff", "prefix", "reference")
    reference = (
        source.source.source.source.source.source.source.source.source.handoff.prefix.reference
    )
    for name, value in (
        ("phase_reset_error_bound", 4 * ETA),
        ("new_weight_range", (Q(1, 2), Q(1))),
    ):
        original = replace(reference.sources[0], **{name: value})
        changed = _replace_at(
            source,
            path,
            sources=(original, reference.sources[1]),
            donor=replace(reference.donor, source=original),
        )
        monkeypatch.setattr(
            owner, "_assess_mixed_effective_interaction", lambda **kwargs: changed
        )
        assert not _assess().certified
    for name, value in (
        ("original_supplied_phase_origins", (Q(0), DELTA)),
        ("recipient_isolated_deadline", Q(10**10 - 1)),
        ("actual_weight_deviation_upper_bound", Q(0)),
        ("clock", "wrong original clock"),
    ):
        changed = _replace_at(source, path, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_effective_interaction", lambda **kwargs: changed
        )
        assert not _assess().certified
    monkeypatch.setattr(
        owner, "_assess_mixed_effective_interaction", lambda **kwargs: source
    )
    geometry = owner._derive_registered_phase_probe_geometry()
    for changes in (
        dict(bare_gap_lower_bound=Q(3)),
        dict(weighted_gap_lower_bound=Q(2)),
        dict(spectral_gap_certified=False),
        dict(degrees=(8,) + (9,) * 17),
    ):
        altered = replace(geometry, successor=replace(geometry.successor, **changes))
        monkeypatch.setattr(
            owner, "_derive_registered_phase_probe_geometry", lambda: altered
        )
        assert not _assess().weighted_gap_certified


def test_short_window_verdicts_do_not_certify_the_longer_task(monkeypatch, report):
    source = report.source
    stale_transfer = replace(
        source.direct_transfer,
        uniform_form_error_upper_bound=Q(0),
        phase_error_upper_bound=Q(0),
        endpoint_form_error_upper_bound=Q(0),
    )
    changed = replace(
        source,
        certified=False,
        centered_identity_certified=False,
        phase_branch_certified=False,
        phase_moment_remainder_certified=False,
        switching_work_certified=False,
        effective_balance_certified=False,
        direct_prediction_certified=False,
        first_segment_certified=False,
        continuation_certified=False,
        memberwise_semigroup_certified=False,
        observation_allowances_admitted=False,
        registered_record_prediction_certified=False,
        aligned_record_prediction_certified=False,
        exact_memberwise_reduced_semigroup=False,
        direct_transfer=stale_transfer,
        first_transfer=stale_transfer,
        continued_transfer=stale_transfer,
        centered_norm_upper_bound=Q(0),
        phase_moment_remainder_upper_bound=Q(0),
        storage_remainder_upper_bound=Q(0),
        registered_record_error_upper_bound=Q(0),
        aligned_record_error_upper_bound=Q(0),
        aligned_storage_jump_bounds=(Q(0), Q(0)),
        registered_storage_jump_bounds=(Q(0), Q(0)),
        status="short_window_verdict_unused",
        unresolved_conditions=("unused",),
    )
    monkeypatch.setattr(
        owner, "_assess_mixed_effective_interaction", lambda **kwargs: changed
    )
    rebuilt = _assess()
    assert rebuilt.certified and rebuilt.weighted_gap_certified
    assert rebuilt.centered_envelope == report.centered_envelope
    assert (
        rebuilt.uniform_form_error_upper_bound == report.uniform_form_error_upper_bound
    )
    assert (
        rebuilt.registered_storage_jump_bounds == report.registered_storage_jump_bounds
    )
    assert source == report.source


def test_raw_helper_and_reader_values_are_admitted_before_arithmetic(monkeypatch):
    for name in ("form_error_bounds", "phase_error_bounds"):
        assert not _assess(**{name: (ERROR + Q(1, 10**100), ERROR)}).certified

    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached source work")

    monkeypatch.setattr(owner, "_assess_mixed_effective_interaction", forbidden)
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
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    _assess(
                        **{name: tuple(bad if j == side else Q(0) for j in range(2))}
                    )
        for bad in ((0,), (0, 0, 0), {0, 1}, {0: 0, 1: 0}, repeat(0)):
            with pytest.raises((TypeError, ValueError)):
                _assess(**{name: bad})
    for name in (
        "registered_observation_error_bound",
        "aligned_observation_error_bound",
    ):
        for bad in invalid:
            with pytest.raises((TypeError, ValueError, OverflowError)):
                _assess(**{name: bad})
    valid = dict(
        initial_form_norm=Q(1, 100), initial_phase_norm=Q(1, 100), duration=Q(4096)
    )
    for name in valid:
        for bad in invalid:
            with pytest.raises((TypeError, ValueError, OverflowError)):
                owner._damped_centered_envelope(**(valid | {name: bad}))
    cutoff = Q(1) / (48 * G**2)
    for duration in (cutoff, cutoff + 1):
        with pytest.raises(ValueError):
            owner._damped_centered_envelope(**(valid | {"duration": duration}))
    zero_time = owner._damped_centered_envelope(**(valid | {"duration": Q(0)}))
    assert zero_time.duration == 0 and zero_time.feedback_denominator == 1
    assert zero_time.scalar_decay_upper_bound == 1
    calls = []
    original = owner._exact

    def traced(value, label):
        calls.append(label)
        return original(value, label)

    monkeypatch.setattr(owner, "_exact", traced)
    with pytest.raises(TypeError):
        owner._damped_centered_envelope(
            initial_form_norm=-1, initial_phase_norm=0, duration=True
        )
    assert calls == ["initial_form_norm", "initial_phase_norm", "duration"]


def test_fixed_width_observations_and_immutable_scope_preserve_limits(
    report, record_property
):
    def fixed(value):
        return Q(np.int64(value.numerator), np.int64(value.denominator))

    assert (
        _assess(
            form_error_bounds=(fixed(ERROR) for _ in range(2)),
            phase_error_bounds=(fixed(ERROR) for _ in range(2)),
            registered_observation_error_bound=fixed(NOISE),
            aligned_observation_error_bound=fixed(NOISE),
        )
        == report
    )
    values = dict(
        initial_form_norm=Q(1, 1000), initial_phase_norm=Q(1, 1000), duration=HORIZON
    )
    assert owner._damped_centered_envelope(**values) == owner._damped_centered_envelope(
        **{k: fixed(v) for k, v in values.items()}
    )
    endpoint = report.endpoint_form_error_upper_bound
    assert (
        report.registered_record_error_upper_bound
        == report.aligned_record_error_upper_bound
        == endpoint + NOISE
    )
    assert (
        report.registered_record_prediction_certified
        and report.aligned_record_prediction_certified
    )
    assert report.certified
    for name in (
        "registered_observation_error_bound",
        "aligned_observation_error_bound",
    ):
        outside = _assess(**{name: NOISE + Q(1, 10**100)})
        assert not outside.observation_allowances_admitted and not outside.certified
        assert outside.form_prediction_certified and outside.phase_prediction_certified
        assert outside.centered_identity_certified
    exact = _assess(
        registered_observation_error_bound=0, aligned_observation_error_bound=0
    )
    assert (
        exact.registered_record_error_upper_bound
        == exact.aligned_record_error_upper_bound
        == endpoint
    )
    assert exact.certified
    assert report.initial_registrations_turns == (Q(0), Q(1, 4))
    assert report.post_duration == HORIZON and report.previous_certified_duration == 16
    assert report.horizon_extension_factor == 256
    assert report.switch_age == SWITCH_AGE and report.switch_time == 5385000
    assert report.total_duration == 5389096
    assert report.original_clock_horizon * Q(1023, 1024) == report.total_duration
    assert (
        report.bare_gap_lower_bound * report.minimum_internal_conductance
        == report.weighted_gap_lower_bound
        == 3
    )
    assert report.form_damping_lower_bound == Q(1, 3)
    assert report.centered_coupling_upper_bound == 4 * G
    assert report.newly_observed_aggregates == report.newly_observed_phases == 0
    assert (
        report.required_endpoint_records == 2
        and report.numerical_response_allowance == 0
    )
    for name in (
        "same_complete_and_effective_laws",
        "whole_history_covariance_retained",
        "source_correlated_initialization_retained",
        "actual_weighted_gap_used",
        "centered_phase_displacement_priced",
        "phase_only_moment_bound_used",
        "independent_record_errors",
        "original_attachment_work_retained_once",
    ):
        assert getattr(report, name) is True
    for name in (
        "unequal_internal_blocks_assumed_commuting",
        "boundary_convexity_assumed",
        "intrinsic_phase_accuracy_is_phase_reading",
        "observation_error_cancellation_assumed",
        "reference_shift_used_to_erase_switch_work",
        "state_reset_performed",
        "phase_clamped",
        "independent_marginal_state_reselection",
        "original_source_reacquired",
        "original_source_replayed",
        "original_source_trajectory_evaluated",
        "reduced_trajectory_evaluated",
        "exact_actual_two_coordinate_closure",
        "all_future_identity_claimed",
        "actual_asymptotic_attraction_certified",
        "actual_instability_claimed",
        "optimal_validity_horizon_identified",
        "comparison_cutoff_is_actual_lifetime",
        "endpoint_envelope_is_actual_norm_reduction",
        "unresolved_storage_assumed_nonnegative",
        "earlier_short_window_results_invalidated",
        "minimal_intrinsic_macrostate_claimed",
        "autonomous_pair_formation_claimed",
        "runtime_event_executed",
        "operator_occurrence_derived",
        "exponentials_evaluated",
        "spectral_roots_evaluated",
        "physical_property_identified",
        "empirical_validation",
    ):
        assert getattr(report, name) is False
    with pytest.raises(FrozenInstanceError):
        report.post_duration = Q(0)
    with pytest.raises(FrozenInstanceError):
        report.centered_envelope.initial_phase_norm = Q(0)
    for name in (
        "source",
        "horizon",
        "initial_phase",
        "phase_observation",
        "initial_reading",
        "threshold",
        "baseline",
        "port_strength",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: Q(0)})
    for name in (
        "switch_age",
        "switch_time",
        "post_duration",
        "total_duration",
        "horizon_extension_factor",
        "inherited_output_error_upper_bound",
        "initial_relative_phase_error_upper_bound",
        "centered_entry_norm_upper_bound",
        "old_joint_growth_denominator",
        "marked_gap_upper_bound",
        "registration_margin",
        "aggregate_absolute_upper_bound",
        "relative_phase_deviation_upper_bound",
        "boundary_branch_margin",
        "phase_moment_remainder_upper_bound",
        "prediction_feedback_denominator",
        "prediction_decay_upper_bound",
        "uniform_form_error_upper_bound",
        "phase_error_upper_bound",
        "endpoint_form_error_upper_bound",
        "form_prediction_margin",
        "phase_prediction_margin",
        "phase_switching_work_allowance",
        "storage_remainder_upper_bound",
        "unresolved_exchange_integral_upper_bound",
        "storage_prediction_error_upper_bound",
        "registered_record_error_upper_bound",
        "aligned_record_error_upper_bound",
        "comparison_determinant",
        "comparison_positive_quadratic_value",
    ):
        record_property(name, str(getattr(report, name)))
    for name in ("aligned_storage_jump_bounds", "registered_storage_jump_bounds"):
        record_property(name, json.dumps([str(v) for v in getattr(report, name)]))
    for name in (
        "initial_form_norm",
        "initial_phase_norm",
        "duration",
        "feedback_denominator",
        "uniform_form_norm_upper_bound",
        "uniform_phase_norm_upper_bound",
        "form_norm_integral_upper_bound",
        "phase_displacement_upper_bound",
        "endpoint_form_norm_upper_bound",
        "scalar_decay_upper_bound",
    ):
        record_property(
            "centered_envelope." + name, str(getattr(report.centered_envelope, name))
        )
