"""A reciprocal phase boundary has a controlled, possibly unresolved effect.

Static full-state oracles preserve source registration, unequal weights and
both new rows. Nonclosure witnesses belong to a larger inference class, not
to evaluated acquired endpoints. No trajectory or exponential is computed.
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
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _mixed_phase_sensitive_interface as owner
from tnfr.physics import _mixed_two_channel_circulation as field_owner
from tnfr.physics import _sine_flow as flow_owner

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
G = Q(1, 3000)
SWITCH_AGE = Q(3375000)
HORIZON = Q(8)
DECAY = Q(1, 2**16)


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


def _relative_common(values):
    whole = sum(values) / 36
    return tuple(
        sum(values[18 * side : 18 * (side + 1)]) / 18 - whole
        for side in (0, 1)
        for _ in range(18)
    )


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
    return owner._mixed_phase_sensitive_field(
        first_new_edge_conductances=weights[0],
        second_new_edge_conductances=weights[1],
    )


def _assess(**changes):
    values = dict(
        form_error_bounds=(ERROR, ERROR),
        phase_error_bounds=(ERROR, ERROR),
        phase_sensitive_observation_error_bound=NOISE,
        form_only_observation_error_bound=NOISE,
    )
    values.update(changes)
    return owner._assess_mixed_phase_sensitive_interface(**values)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_retained_producer():
    static_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("phase-interface controls cannot replay or advance a source")

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
        pytest.fail("static interface proof cannot evaluate an exponential or root")

    with pytest.MonkeyPatch.context() as patch:
        for name in ("exp", "expm", "eig", "polyroots", "odefun", "invertlaplace"):
            patch.setattr(context, name, forbidden)
        yield context


@pytest.fixture(scope="module")
def report():
    return _assess()


def test_phase_sensitive_field_matches_all_actual_rows_and_mean_balances(mp):
    forms = _center(tuple(Q((j * 7) % 17 - 8, 10**5) for j in range(36)))
    phases = _center(tuple(Q((j * 5) % 11 - 5, 10**5) for j in range(36)))
    weights = (_weights(mp, 0), _weights(mp, 1))
    assert weights[0] != weights[1]
    dx, dy = _field(mp, forms, phases, weights)
    state = _state(forms, phases)
    frozen = tuple(state)
    rates = _new_field(weights)(state)
    assert len(rates) == 72 and tuple(state) == frozen
    for interval, value in zip(rates, dx + dy, strict=True):
        assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
    gamma = 1 / (1023 * mp.pi)
    q = _mp(mp, -sum(forms[:18]))
    delta = tuple(_mp(mp, phases[j + 18] - phases[j]) for j in range(18))
    moment = sum(mp.sin(value) for value in delta)
    assert abs(-sum(dx[:18]) + 2 * q + gamma * moment) < mp.mpf("1e-200")
    psi_rate = (sum(dy[18:]) - sum(dy[:18])) / 18
    assert abs(psi_rate - 2 * gamma * q / 9) < mp.mpf("1e-200")
    assert abs(sum(dx)) < mp.mpf("1e-200")
    assert abs(sum(dy)) < mp.mpf("1e-200")
    x = tuple(_mp(mp, value) for value in forms)
    hx = _h_apply(mp, x, weights)
    delta_rate = tuple(
        gamma * (hx[j + 18] - hx[j]) / 9 + 2 * gamma * (x[j + 18] - x[j])
        for j in range(18)
    )
    direct = sum(mp.cos(d) * (dy[j + 18] - dy[j]) for j, d in enumerate(delta))
    exact = sum(mp.cos(d) * v for d, v in zip(delta, delta_rate, strict=True))
    residual = sum((mp.cos(d) - 1) * v for d, v in zip(delta, delta_rate, strict=True))
    assert abs(direct - exact) < mp.mpf("1e-200")
    assert abs(direct - 4 * gamma * q - residual) < mp.mpf("1e-200")


def test_relative_phase_moves_and_only_the_whole_gauge_remains_a_symmetry(mp):
    forms = (-Q(1, 10**5),) * 18 + (Q(1, 10**5),) * 18
    phases = tuple(Q((j % 3) - 1, 10**5) for j in range(36))
    weights = (_weights(mp, 0), _weights(mp, 1))
    dx, dy = _field(mp, forms, phases, weights)
    assert abs(sum(dy[:18])) > mp.mpf("1e-9")
    assert abs(sum(dy)) < mp.mpf("1e-200")
    common_shift = tuple(value + Q(2, 7) for value in phases)
    shifted = _field(mp, forms, common_shift, weights)
    assert max(
        abs(a - b)
        for row, other in zip((dx, dy), shifted, strict=True)
        for a, b in zip(row, other, strict=True)
    ) < mp.mpf("1e-200")
    relative_shift = tuple(
        value + (Q(1, 10**5) if j >= 18 else 0) for j, value in enumerate(phases)
    )
    changed = _field(mp, forms, relative_shift, weights)
    assert abs(sum(changed[0][:18]) - sum(dx[:18])) > mp.mpf("1e-9")
    assert changed[1] == dy
    # The old full-form port has independently conserved copy phase means;
    # the new reciprocal channel is an actual change to those phase rows.
    old = field_owner._mixed_two_channel_field(
        first_new_edge_conductances=weights[0],
        second_new_edge_conductances=weights[1],
        law="full",
    )(_state(forms, phases))
    for side in (0, 1):
        total = sum(old[36 + 18 * side : 54 + 18 * side], I(0))
        assert total.lo <= 0 <= total.hi
    assert _mp(mp, sum(old[36:54], I(0)).hi) < abs(sum(dy[:18]))


def test_phase_moment_has_cubic_error_and_no_exact_two_coordinate_closure(mp):
    a = Q(1, 10000)
    deltas = (a, a, -2 * a) + (Q(0),) * 15
    weights = (_weights(mp, 0), _weights(mp, 1))
    moments, rates = [], []
    for sign in (1, -1):
        phases = tuple(-sign * d / 2 for d in deltas) + tuple(
            sign * d / 2 for d in deltas
        )
        forms = (Q(0),) * 36
        dx, _ = _field(mp, forms, phases, weights)
        psi = (sum(phases[18:]) - sum(phases[:18])) / 18
        assert psi == 0 and -sum(forms[:18]) == 0
        moment = sum(mp.sin(sign * _mp(mp, d)) for d in deltas)
        expected = sign * 2 * mp.sin(_mp(mp, a)) * (1 - mp.cos(_mp(mp, a)))
        assert abs(moment - expected) < mp.mpf("1e-200")
        norm = mp.sqrt(sum(_mp(mp, value) ** 2 for value in phases))
        assert 0 < abs(moment) <= norm**3 / 2
        moments.append(moment)
        rates.append(-sum(dx[:18]))
    assert moments[0] > 0 > moments[1] and rates[0] < 0 < rates[1]
    # Static nearby states, not two members proved to be attained by the source.
    # The general cubic estimate also includes a nonzero relative mean.
    delta = tuple(_mp(mp, Q(j % 5 - 2, 10**5) + Q(1, 10000)) for j in range(18))
    psi = sum(delta) / 18
    residual = sum(mp.sin(value) for value in delta) - 18 * psi
    norm = mp.sqrt(sum(value * value for value in delta) / 2)
    assert abs(residual) <= sum(abs(value) ** 3 for value in delta) / 6
    assert (
        sum(abs(value) ** 3 for value in delta) / 6
        <= mp.sqrt(2) * norm**3 / 3
        < norm**3 / 2
    )


def test_new_storage_jump_and_complete_loss_use_both_boundary_channels(mp, report):
    x = tuple(
        _mp(mp, value)
        for value in _center(tuple(Q((j * 5) % 17 - 8, 10**5) for j in range(36)))
    )
    y = tuple(
        _mp(mp, value)
        for value in _center(tuple(Q((j * 3) % 11 - 5, 10**5) for j in range(36)))
    )
    weights = (_weights(mp, 0), _weights(mp, 1))
    hx, gx = _h_apply(mp, x, weights), _exchange(x)
    lx = tuple(a + 9 * b for a, b in zip(hx, gx, strict=True))
    cross = tuple(mp.sin(y[j + 18] - y[j]) for j in range(18))
    cross = cross + tuple(-value for value in cross)
    sine = tuple(a + 9 * b for a, b in zip(_sine(mp, y), cross, strict=True))
    dx, dy = _field(mp, x, y, weights)
    derivative = _dot(lx, dx) - _dot(sine, dy)
    assert abs(derivative + _dot(lx, lx) / 9) < mp.mpf("1e-200")
    gamma = 1 / (1023 * mp.pi)
    old_phase_row = tuple(gamma * value / 9 for value in hx)
    omitted_reciprocal_defect = (
        _dot(lx, dx) - _dot(sine, old_phase_row) + _dot(lx, lx) / 9
    )
    assert abs(omitted_reciprocal_defect) > mp.mpf("1e-15")
    c = 9 * (mp.mpf(13) / 27 - _mp(mp, DELTA) / (18 * mp.pi))
    common = _relative_common(x)
    centered = tuple(a - b for a, b in zip(x, common, strict=True))
    potential = sum(1 - mp.cos(y[j + 18] - y[j]) for j in range(18))
    jump = 9 * _dot(x, gx) / 2 + 9 * potential - c * _dot(common, common)
    resolved = (
        (9 - c) * _dot(common, common)
        + 9 * _dot(centered, _exchange(centered)) / 2
        + 9 * potential
    )
    assert abs(jump - resolved) < mp.mpf("1e-200")
    assert 0 < jump <= 9 * (_dot(x, x) + _dot(y, y))
    assert report.storage_jump_bounds == (Q(0), 9 * report.entry_norm_upper_bound**2)
    assert report.fresh_storage_certified


def test_two_coordinate_generator_contracts_with_paid_initial_phase_uncertainty(report):
    # All arithmetic here is rational, including the moving-gauge source bound.
    for q, v, gamma in ((Q(2), Q(-3), G), (Q(-5), Q(1), Q(1, 3300)), (Q(0), Q(4), G)):
        aq, av = -2 * q - 2 * gamma * v, 2 * gamma * q
        assert q * aq + v * av == -2 * q * q
    origins = (-DELTA / 2,) * 9 + (DELTA / 2,) * 9
    assert sum(origins) == sum(_turns()) == 0
    copy_errors = ((-3 * ETA,) * 18, (3 * ETA,) * 18)
    gauges = tuple(sum(values) / 18 for values in copy_errors)
    assert gauges[1] - gauges[0] == 6 * ETA
    gauge_shift_norm_squared = 18 * (3 * ETA) ** 2 + 18 * (3 * ETA) ** 2
    assert gauge_shift_norm_squared == (18 * ETA) ** 2
    assert report.phase_mean_error_upper_bound == 3 * ETA
    assert report.relative_phase_entry_error_upper_bound == 6 * ETA
    assert (
        report.collective_initialization_error_upper_bound
        == report.inherited_output_error_upper_bound + 54 * ETA
    )
    assert (
        report.collective_prediction_error_upper_bound
        == report.collective_initialization_error_upper_bound
        + report.nonlinear_output_error_upper_bound
        < NOISE
    )
    assert report.collective_prediction_certified


def test_new_joint_chart_and_registration_use_the_changed_complete_law(mp, report):
    entry = 4 * QHI + 200 * MATURE + 18 * ETA
    coupling = 4 * G * HORIZON
    upper = entry / (1 - coupling)
    assert upper == entry + coupling * upper
    assert report.entry_norm_upper_bound == entry
    assert report.norm_growth_denominator == 1 - coupling > 0
    assert report.post_norm_upper_bound == upper < Q(1, 100) < Q(1, 32)
    assert report.phase_drift_upper_bound == 6 * G * HORIZON * upper
    assert (
        report.marked_gap_upper_bound
        == QHI + 300 * MATURE + report.phase_drift_upper_bound
    )
    assert report.registration_margin == DELTA / 4 - report.marked_gap_upper_bound > 0
    assert report.finite_identity_certified
    x = tuple(
        _mp(mp, v) for v in _center(tuple(Q(j % 7 - 3, 10**5) for j in range(36)))
    )
    y = tuple(
        _mp(mp, v) for v in _center(tuple(Q(j % 5 - 2, 10**5) for j in range(36)))
    )
    dx, dy = _field(mp, x, y, (_weights(mp, 0), _weights(mp, 1)))
    assert _dot(x, dx) + _dot(y, dy) <= 4 * _mp(mp, G) * (_dot(x, x) + _dot(y, y))
    assert _dot(dy, dy) <= 16 * _mp(mp, G) ** 2 * _dot(x, x)


def test_same_source_scalar_effect_and_endpoint_errors_are_independently_priced(
    report, record_property
):
    allowance = report.inherited_output_error_upper_bound
    amplitude_upper = report.slow_amplitude_bounds[1]
    remainder = report.post_norm_upper_bound**3 / 2
    forcing = G * HORIZON * remainder
    uniform = amplitude_upper + allowance + 54 * ETA + forcing
    effect = 54 * G * ETA + 2 * G**2 * HORIZON * uniform + G * remainder / 2
    assert report.nonlinear_moment_remainder_upper_bound == remainder
    assert report.nonlinear_output_error_upper_bound == forcing
    assert report.aggregate_uniform_upper_bound == uniform
    assert report.relative_phase_upper_bound == 6 * ETA + 2 * G * HORIZON * uniform / 9
    assert report.scalar_effect_upper_bound == effect < NOISE
    assert report.scalar_initialization_error_upper_bound == allowance * DECAY
    assert (
        report.scalar_prediction_error_upper_bound == effect + allowance * DECAY < NOISE
    )
    assert (
        report.scalar_tolerance_margin
        == NOISE - report.scalar_prediction_error_upper_bound
        > 0
    )
    full_upper = (amplitude_upper + allowance) * DECAY
    assert report.form_only_true_bounds == (Q(0), full_upper)
    assert report.phase_sensitive_true_bounds == (-effect, full_upper + effect)
    assert full_upper + effect < NOISE
    assert report.phase_sensitive_record_bounds == (
        -effect - NOISE,
        full_upper + effect + NOISE,
    )
    assert report.form_only_record_bounds == (-NOISE, full_upper + NOISE)
    assert report.phase_sensitive_return_certified and report.form_only_return_certified
    assert report.scalar_effect_bound_certified and report.scalar_tolerance_certified
    assert report.same_threshold_classification_certified and report.certified
    for name in (
        "switch_age",
        "post_duration",
        "switch_time",
        "total_duration",
        "inherited_output_error_upper_bound",
        "relative_phase_entry_error_upper_bound",
        "entry_norm_upper_bound",
        "post_norm_upper_bound",
        "phase_drift_upper_bound",
        "registration_margin",
        "nonlinear_moment_remainder_upper_bound",
        "nonlinear_output_error_upper_bound",
        "collective_initialization_error_upper_bound",
        "collective_prediction_error_upper_bound",
        "scalar_effect_upper_bound",
        "scalar_initialization_error_upper_bound",
        "scalar_prediction_error_upper_bound",
        "scalar_tolerance_margin",
        "record_difference_absolute_upper_bound",
        "same_member_record_overlap_margin",
        "phase_sensitive_threshold_margin",
        "form_only_threshold_margin",
    ):
        record_property(name, str(getattr(report, name)))
    for name in (
        "storage_jump_bounds",
        "phase_sensitive_true_bounds",
        "form_only_true_bounds",
        "phase_sensitive_record_bounds",
        "form_only_record_bounds",
    ):
        record_property(name, json.dumps([str(v) for v in getattr(report, name)]))


def test_readout_overlap_is_sufficient_and_independent_of_true_tolerance(report):
    effect = report.scalar_effect_upper_bound
    assert report.record_difference_absolute_upper_bound == effect + 2 * NOISE
    assert report.same_member_record_overlap_margin == 2 * NOISE - effect > 0
    assert report.same_member_record_overlap_certified
    # Construct a common reading for any same-member true difference bounded
    # by effect. This does not assert that two arbitrary records must agree.
    for difference in (-effect, Q(0), effect):
        form, phase = Q(1, 10**11), Q(1, 10**11) + difference
        common = (form + phase) / 2
        assert abs(common - form) <= NOISE and abs(common - phase) <= NOISE
        assert abs((phase + NOISE) - (form - NOISE)) <= effect + 2 * NOISE
    zero = _assess(
        phase_sensitive_observation_error_bound=Q(0),
        form_only_observation_error_bound=Q(0),
    )
    assert not zero.same_member_record_overlap_certified
    assert zero.same_member_record_overlap_margin == -effect
    assert zero.scalar_tolerance_certified and zero.collective_prediction_certified
    assert zero.same_threshold_classification_certified and zero.certified
    assert zero.phase_sensitive_record_bounds == zero.phase_sensitive_true_bounds
    for name in (
        "phase_sensitive_observation_error_bound",
        "form_only_observation_error_bound",
    ):
        changed = _assess(**{name: NOISE + Q(1, 10**100)})
        assert not changed.observation_allowances_admitted and not changed.certified
        assert not changed.same_threshold_classification_certified
        assert (
            changed.scalar_tolerance_certified
            and changed.collective_prediction_certified
        )


def _replace_at(value, path, **changes):
    if not path:
        return replace(value, **changes)
    return replace(
        value, **{path[0]: _replace_at(getattr(value, path[0]), path[1:], **changes)}
    )


def test_consumed_source_bounds_clocks_and_phase_reset_are_readmitted(
    monkeypatch, report
):
    source = report.source
    cases = (
        ((), "source_association_certified", False),
        ((), "inherited_output_error_upper_bound", Q(0)),
        ((), "scalar_initialization_error_upper_bound", Q(0)),
        ((), "entry_aggregate_bounds", (Q(0), Q(0))),
        ((), "entry_norm_upper_bound", Q(0)),
        ((), "slow_amplitude_bounds", (Q(0), Q(1))),
        ((), "switch_age", Q(3375008)),
        ((), "activation_time", Q(0)),
        ((), "original_baseline_retained", False),
        ((), "state_reset_performed", True),
        ((), "full_observation_error_bound", NOISE),
        ((), "clock", "wrong renewed clock"),
        (("source",), "uniform_approximation_certified", False),
        (("source",), "total_output_error_upper_bound", Q(0)),
        (("source",) * 3, "uniform_tube_certified", False),
        (("source",) * 3, "nominal_norm_upper_bound", Q(0)),
        (("source",) * 3, "uniform_comparison_budget", Q(0)),
        (("source",) * 3, "original_phase_gauges_and_branches_retained", False),
        (("source",) * 3, "physical_capacities", (Q(0),) + (Q(1),) * 35),
        (("source",) * 4, "selected_mu", Q(2)),
        (("source",) * 4, "origin_reference", "different origin"),
    )
    for path, name, value in cases:
        changed = _replace_at(source, path, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_renewed_interface", lambda **kwargs: changed
        )
        assert not _assess().certified
    path = ("source",) * 5 + ("handoff", "prefix", "reference")
    reference = source.source.source.source.source.source.handoff.prefix.reference
    for reset in (Q(0), 4 * ETA):
        changed_original = replace(reference.sources[0], phase_reset_error_bound=reset)
        changed_reference = replace(
            reference,
            sources=(changed_original, reference.sources[1]),
            donor=replace(reference.donor, source=changed_original),
        )
        changed = _replace_at(
            source,
            path,
            sources=changed_reference.sources,
            donor=changed_reference.donor,
        )
        monkeypatch.setattr(
            owner, "_assess_mixed_renewed_interface", lambda **kwargs: changed
        )
        result = _assess()
        assert not result.relative_phase_registration_certified and not result.certified
    monkeypatch.setattr(
        owner, "_assess_mixed_renewed_interface", lambda **kwargs: source
    )
    registered = owner._derive_registered_phase_probe_geometry()
    altered = replace(
        registered,
        successor=replace(registered.successor, held_capacities=(Q(0),) + (Q(1),) * 17),
    )
    monkeypatch.setattr(
        owner, "_derive_registered_phase_probe_geometry", lambda: altered
    )
    assert not _assess().certified


def test_old_global_verdicts_do_not_substitute_for_the_new_admission(
    monkeypatch, report
):
    source = replace(
        report.source,
        certified=False,
        sum_closure_certified=False,
        scalar_response_certified=False,
        changed_port_storage_certified=False,
        finite_identity_certified=False,
        work_allowance_certified=False,
        full_return_certified=False,
        continued_response_certified=False,
        full_record_certified=False,
        continued_record_certified=False,
        independent_record_sets_separated=False,
        fixed_threshold_certified=False,
        full_true_bounds=(Q(0), Q(0)),
        full_record_bounds=(Q(0), Q(0)),
        continued_true_bounds=(Q(0), Q(0)),
        continued_record_bounds=(Q(0), Q(0)),
        post_norm_upper_bound=Q(0),
        storage_jump=Q(1),
        status="old_changed_law_verdict_unused",
        unresolved_conditions=("unused",),
    )
    monkeypatch.setattr(
        owner, "_assess_mixed_renewed_interface", lambda **kwargs: source
    )
    changed = _assess()
    assert changed.certified and changed.finite_identity_certified
    assert (
        changed.scalar_prediction_error_upper_bound
        == report.scalar_prediction_error_upper_bound
    )
    assert changed.storage_jump_bounds == report.storage_jump_bounds


def test_field_and_reader_reject_raw_values_before_response_work(monkeypatch):
    weights = ((Q(4, 5),) * 63, (Q(5, 6),) * 63)
    field = _new_field(weights)
    state = _state((Q(0),) * 36, (Q(0),) * 36)
    for name in ("form_error_bounds", "phase_error_bounds"):
        assert not _assess(**{name: (ERROR + Q(1, 10**100), ERROR)}).certified

    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached response or trigonometry")

    monkeypatch.setattr(owner, "_assess_mixed_renewed_interface", forbidden)
    for module in (owner, flow_owner):
        for name in ("sin", "jet_sin"):
            monkeypatch.setattr(module, name, forbidden)
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
        "phase_sensitive_observation_error_bound",
        "form_only_observation_error_bound",
    ):
        for bad in invalid:
            with pytest.raises((TypeError, ValueError, OverflowError)):
                _assess(**{name: bad})
    for bad in (
        state[:-1],
        state + (I(0),),
        repeat(I(0)),
        (True,) + state[1:],
        state[:-1] + (float("nan"),),
        (Jet((I(0), I(1))),) + state[1:],
    ):
        with pytest.raises((TypeError, ValueError, OverflowError)):
            field(bad)
    jets = tuple(Jet((value, I(0))) for value in state)
    with pytest.raises(ValueError):
        field(jets[:-1] + (Jet((I(0), I(0), I(1))),))
    for side in (0, 1):
        for bad in (
            weights[side][:-1],
            weights[side] + (Q(1),),
            repeat(Q(1)),
            weights[side][:-1] + (True,),
            weights[side][:-1] + (Q(3, 4) - Q(1, 10**100),),
            weights[side][:-1] + (Q(1) + Q(1, 10**100),),
        ):
            arguments = list(weights)
            arguments[side] = bad
            with pytest.raises((TypeError, ValueError, OverflowError)):
                _new_field(arguments)


def test_fixed_width_and_jet_contracts_preserve_the_frozen_protocol(mp, report):
    def fixed(value):
        return Q(np.int64(value.numerator), np.int64(value.denominator))

    assert (
        _assess(
            form_error_bounds=(fixed(ERROR) for _ in range(2)),
            phase_error_bounds=(fixed(ERROR) for _ in range(2)),
            phase_sensitive_observation_error_bound=fixed(NOISE),
            form_only_observation_error_bound=fixed(NOISE),
        )
        == report
    )
    weights = ((Q(4, 5),) * 63, (Q(5, 6),) * 63)
    field = _new_field(weights)
    fixed_field = _new_field(tuple((fixed(value) for value in row) for row in weights))
    forms = _center(tuple(Q(j % 7 - 3, 8192) for j in range(36)))
    phases = tuple(Q(j % 3 - 1, 8192) for j in range(36))
    state = _state(forms, phases)
    assert fixed_field(value for value in state) == field(state)
    directions = tuple(Q((j * 5) % 11 - 5, 8192) for j in range(36))
    jets = tuple(
        Jet((value, I(directions[j] if j < 36 else 0))) for j, value in enumerate(state)
    )
    jet_rates = field(jets)
    scalar_rates = field(state)
    d = tuple(_mp(mp, value) for value in directions)
    hd, gd = _h_apply(mp, d, weights), _exchange(d)
    ld = tuple(a + 9 * b for a, b in zip(hd, gd, strict=True))
    gamma = 1 / (1023 * mp.pi)
    expected = tuple(-value / 9 for value in ld) + tuple(
        gamma * value / 9 for value in ld
    )
    for jet, scalar, derivative in zip(jet_rates, scalar_rates, expected, strict=True):
        assert jet.coeffs[0] == scalar
        assert _mp(mp, jet.coeffs[1].lo) <= derivative <= _mp(mp, jet.coeffs[1].hi)
    assert report.switch_age == SWITCH_AGE and report.post_duration == HORIZON
    assert report.switch_time == 5385000 and report.total_duration == 5385008
    assert report.original_clock_horizon * Q(1023, 1024) == report.total_duration
    assert (
        report.whole_phase_gauge_only
        and not report.original_phase_means_preserved_after_switch
    )
    assert not report.initial_relative_phase_mean_assumed_zero
    assert report.original_phase_lifts_retained and report.same_actual_member_comparator
    assert (
        report.same_input_collective_prediction
        and report.form_only_is_boundary_effect_control
    )
    assert (
        report.original_baseline_retained
        and report.inherited_amplitude_rate_correlation_retained
    )
    assert (
        not report.full_nonlinear_state_closure
        and not report.exact_nonlinear_two_coordinate_closure
    )
    assert (
        not report.scalar_exact_under_phase_port
        and not report.phase_moment_claimed_zero
    )
    assert (
        report.independent_record_errors and not report.all_record_values_claimed_equal
    )
    assert (
        not report.observation_error_cancellation_assumed
        and not report.all_future_changed_port_identity_claimed
    )
    assert (
        not report.zero_switching_storage_jump_claimed
        and not report.zero_physical_switching_cost_claimed
    )
    assert not report.original_source_reacquired and not report.state_reset_performed
    assert not report.observation_rebased_at_switch and report.switch_observations == 0
    assert not report.source_trajectory_evaluated and not report.exponentials_evaluated
    assert not report.roots_numerically_evaluated and not report.empirical_validation
    assert (
        report.required_endpoint_records == 2 and report.newly_observed_aggregates == 0
    )
    with pytest.raises(FrozenInstanceError):
        report.post_duration = Q(0)
    for name in (
        "source",
        "horizon",
        "switch_age",
        "initial_phase",
        "phase_offset",
        "initial_reading",
        "threshold",
        "port_strength",
        "baseline",
        "phase_observation",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: Q(0)})
