"""A strict full-state estimate certifies a noise-resolved finite return band.

Independent edge states check algebra and its domain, not attained endpoints.
The early and late predictions refer to the same original actual history;
only the two aggregate readout errors are independent. No trajectory runs.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import repeat

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _mixed_finite_return as owner
from tnfr.physics import _mixed_reciprocal_partner as field_owner
from tnfr.physics import _sine_flow as flow_owner
from tnfr.physics import _sine_lyapunov as kernel_owner

ERROR = Q(1, 10**10)
DELTA = Q(1, 1024)
SECTION = 3 * DELTA / 16
ETA = Q(2, 10**69)
WEIGHT_ERROR = Q(2, 10**60)
SECTION_ERROR = Q(3, 2) * (14 * ETA + 2 * 10**10 * WEIGHT_ERROR)
MATURE = Q(1, 2**28)
GAMMA_LO, GAMMA_HI = Q(1, 3300), Q(1, 3000)
QLO, QHI = SECTION - SECTION_ERROR / 2, SECTION + SECTION_ERROR / 2


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
    support = {
        (a, b)
        for a in range(18)
        for b in range(a + 1, 18)
        if min((bins[b] - bins[a]) % 9, (bins[a] - bins[b]) % 9) in (1, 2)
        or (a // 9 != b // 9 and bins[a] == bins[b])
    }
    return old, tuple(sorted(support - set(old)))


def _edges(weights):
    old, new = _support()
    return tuple((a, b, Q(1)) for a, b in old) + tuple(
        (a, b, w) for (a, b), w in zip(new, weights, strict=True)
    )


def _nominal_weights(mp):
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + (-1 if j < 9 else 1) * _mp(mp, DELTA) / 2
        for j, turn in enumerate(_turns())
    )
    return tuple(
        1 - abs((theta[b] - theta[a] + mp.pi) % (2 * mp.pi) - mp.pi) / (2 * mp.pi)
        for a, b in _support()[1]
    )


def _actual_weights(mp, side):
    result = tuple(
        Q(mp.nstr(value, 210))
        + (1 if value < _mp(mp, Q(7, 8)) else -1)
        * WEIGHT_ERROR
        * Q(1 + (j + side) % 3, 6)
        for j, value in enumerate(_nominal_weights(mp))
    )
    assert all(Q(3, 4) <= value <= 1 for value in result)
    return result


def _offset(value):
    return (-value / 2,) * 9 + (value / 2,) * 9


def _exchange(values):
    first = tuple(values[j] - values[18 + j] for j in range(9)) + (0,) * 9
    return first + tuple(-value for value in first)


def _relative_common(values):
    whole = sum(values) / 36
    return tuple(
        sum(values[18 * p : 18 * (p + 1)]) / 18 - whole
        for p in (0, 1)
        for _ in range(18)
    )


def _h_apply(mp, values, weights):
    out = [mp.mpf(0)] * 36
    for p in (0, 1):
        for i, j, weight in _edges(weights[p]):
            a, b = i + 18 * p, j + 18 * p
            current = _mp(mp, weight) * (values[a] - values[b])
            out[a] += current
            out[b] -= current
    return tuple(out)


def _phase(mp, phases):
    return tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, y)
        for turn, y in zip(_turns() * 2, phases, strict=True)
    )


def _sine(mp, phases):
    theta = _phase(mp, phases)
    out = [mp.mpf(0)] * 36
    for p in (0, 1):
        for i, j, _ in _edges((Q(1),) * 63):
            a, b = i + 18 * p, j + 18 * p
            current = mp.sin(theta[b] - theta[a])
            out[a] += current
            out[b] -= current
    return tuple(out)


def _hessian_apply(mp, phases, vector):
    theta = _phase(mp, phases)
    out = [mp.mpf(0)] * 36
    for p in (0, 1):
        for i, j, _ in _edges((Q(1),) * 63):
            a, b = i + 18 * p, j + 18 * p
            current = mp.cos(theta[b] - theta[a]) * (vector[a] - vector[b])
            out[a] += current
            out[b] -= current
    return tuple(out)


def _potential(mp, phases):
    theta = _phase(mp, phases)
    return sum(
        1 - mp.cos(theta[18 * p + b] - theta[18 * p + a])
        for p in (0, 1)
        for a, b, _ in _edges((Q(1),) * 63)
    )


def _field(mp, forms, phases, weights):
    x = tuple(_mp(mp, value) for value in forms)
    hx, sine, gx = _h_apply(mp, x, weights), _sine(mp, phases), _exchange(x)
    gamma = 1 / (1023 * mp.pi)
    return tuple(
        (-h + gamma * s) / 9 - g for h, s, g in zip(hx, sine, gx, strict=True)
    ), tuple(gamma * h / 9 for h in hx)


def _assess(**changes):
    values = dict(
        form_error_bounds=(ERROR, ERROR),
        phase_error_bounds=(ERROR, ERROR),
        early_observation_error_bound=Q(1, 10**9),
        late_observation_error_bound=Q(1, 10**9),
    )
    values.update(changes)
    return owner._assess_mixed_finite_return(**values)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_prior_producer():
    factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("finite-return controls cannot advance or replay a source")

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
        with pytest.MonkeyPatch.context() as static:
            static.setattr(flow_owner, "_sine_rate_evaluator", factory)
            yield


@pytest.fixture(scope="module")
def mp():
    result = mpmath.mp.clone()
    result.dps = 240
    return result


@pytest.fixture(scope="module")
def report():
    return _assess()


EPSILON = Q(1, 2000)
NOISE = Q(1, 10**9)
THRESHOLD = Q(1, 10**8)
RATE = Q(1, 9801000000000)
DEADLINE = Q(627264000000000)
KERNEL_INPUTS = dict(
    storage_lower=Q(1, 5),
    storage_upper=Q(9),
    form_loss_lower=Q(1, 5),
    transport_norm_upper=Q(18),
    damping_norm_upper=Q(4),
    phase_stiffness_lower=Q(2, 5),
    gamma_lower=GAMMA_LO,
    gamma_upper=GAMMA_HI,
    loss_divisor=Q(9),
    cross_coefficient=EPSILON,
)


def _helper(**changes):
    values = dict(KERNEL_INPUTS)
    values.update(changes)
    return kernel_owner._sine_strict_energy_coefficients(**values)


def _storage(mp, x, y, weights):
    c = mp.mpf(13) / 3 - _mp(mp, DELTA) / (2 * mp.pi)
    hx = _h_apply(mp, x, weights)
    return (
        _dot(x, hx) / 2
        + c * _dot(x, _relative_common(x))
        + _potential(mp, y)
        - _potential(mp, (Q(0),) * 36)
    )


def test_actual_weighted_strict_balance_keeps_the_full_noncommuting_port(mp):
    forms = _center(tuple(Q((j * 7) % 13 - 6, 10**5) for j in range(36)))
    phases = tuple(
        value
        for p in (0, 1)
        for value in _center(tuple(Q((j * 3 + p) % 11 - 5, 10**5) for j in range(18)))
    )
    x, y = tuple(_mp(mp, v) for v in forms), tuple(_mp(mp, v) for v in phases)
    weights = (_actual_weights(mp, 0), _actual_weights(mp, 1))
    gamma = 1 / (1023 * mp.pi)
    c = mp.mpf(13) / 3 - _mp(mp, DELTA) / (2 * mp.pi)
    eps = _mp(mp, EPSILON)
    dx, dy = _field(mp, x, y, weights)
    hx, gx = _h_apply(mp, x, weights), _exchange(x)
    gradient = tuple(-s for s in _sine(mp, y))
    effort = tuple(h + 2 * c * p for h, p in zip(hx, _relative_common(x), strict=True))
    old_derivative = _dot(effort, dx) + _dot(gradient, dy)
    loss = _dot(hx, hx) / 9 + _dot(effort, gx)
    cross_derivative = eps * gamma * (_dot(dx, y) + _dot(x, dy))
    expected_cross = (
        eps
        * gamma
        * (
            -_dot(y, hx) / 9
            - _dot(y, gx)
            - gamma * _dot(y, gradient) / 9
            + gamma * _dot(x, hx) / 9
        )
    )
    assert abs(old_derivative + loss) < mp.mpf("1e-220")
    assert abs(cross_derivative - expected_cross) < mp.mpf("1e-220")
    derivative = old_derivative + cross_derivative
    _, _, qx, qy, _ = _helper()
    assert derivative <= -_mp(mp, qx) * _dot(x, x) - _mp(mp, qy) * _dot(y, y)
    commutator = tuple(
        a - b for a, b in zip(_h_apply(mp, gx, weights), _exchange(hx), strict=True)
    )
    assert max(abs(value) for value in commutator) > mp.mpf("1e-8")
    # Existing production field supplies the actual 72 rows at this static
    # state. The test does not advance it or claim this state was acquired.
    state = tuple(I(v) for v in forms) + tuple(
        2 * pi_interval() * turn + I(v)
        for turn, v in zip(_turns() * 2, phases, strict=True)
    )
    field = field_owner._mixed_reciprocal_partner_field(
        first_new_edge_conductances=weights[0], second_new_edge_conductances=weights[1]
    )
    for interval, value in zip(field(state), dx + dy, strict=True):
        assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def test_phase_only_state_needs_the_cross_term_to_show_strict_loss(mp):
    zero = (Q(0),) * 36
    phases = _offset(SECTION / 2) + _offset(-SECTION / 3)
    y = tuple(_mp(mp, v) for v in phases)
    weights = (_actual_weights(mp, 0), _actual_weights(mp, 1))
    dx, dy = _field(mp, zero, phases, weights)
    gradient = tuple(-s for s in _sine(mp, phases))
    assert all(v == 0 for v in dy)
    assert _dot(gradient, dy) == 0  # original form-only loss is zero here
    gamma = 1 / (1023 * mp.pi)
    strict_derivative = _mp(mp, EPSILON) * gamma * _dot(dx, y)
    assert strict_derivative < 0
    assert _dot(y, gradient) >= _mp(mp, Q(2, 5)) * _dot(y, y)
    assert abs(
        strict_derivative + _mp(mp, EPSILON) * gamma**2 * _dot(y, gradient) / 9
    ) < mp.mpf("1e-220")
    assert strict_derivative <= -_mp(mp, _helper()[3]) * _dot(y, y)


def test_metric_bounds_require_the_original_phase_gauges_and_whole_form_mean(mp):
    weights = (_actual_weights(mp, 0), _actual_weights(mp, 1))
    base = tuple(
        _mp(mp, v)
        for p in (0, 1)
        for v in _center(tuple(Q((j * 5 + p) % 13 - 6, 10**5) for j in range(18)))
    )
    common = (mp.mpf("1e-5"),) * 18 + (-mp.mpf("1e-5"),) * 18
    x = tuple(a + b for a, b in zip(base, common, strict=True))
    lower, upper, *_ = _helper()
    for sign in (-1, 1):
        y = tuple(sign * value for value in base)
        norm_squared = _dot(x, x) + _dot(y, y)
        energy = _storage(mp, x, y, weights)
        strict = energy + _mp(mp, EPSILON) * _dot(x, y) / (1023 * mp.pi)
        assert _mp(mp, lower) * norm_squared <= strict <= _mp(mp, upper) * norm_squared
    # A free phase gauge or whole form mean is a genuine zero direction.
    # These larger-class static states are outside the admitted mean leaves.
    zero = (Q(0),) * 36
    gauge = (Q(1, 10000),) * 18 + (-Q(1, 10000),) * 18
    assert _dot(gauge, gauge) > 0
    assert abs(_storage(mp, zero, gauge, weights)) < mp.mpf("1e-220")
    whole_mean = (Q(1, 10000),) * 36
    assert _dot(whole_mean, whole_mean) > 0
    assert abs(_storage(mp, whole_mean, zero, weights)) < mp.mpf("1e-220")


def test_donor_total_covector_has_norm_three_only_on_the_conserved_leaf(mp, report):
    projected = (-Q(1, 2),) * 18 + (Q(1, 2),) * 18
    assert _dot(projected, projected) == 9
    assert report.centered_aggregate_covector == projected + (Q(0),) * 36
    assert report.centered_aggregate_covector_norm_squared == 9
    assert (-sum(projected[:18])) ** 2 == 9 * _dot(projected, projected)
    forms = _center(tuple(Q((j * 3) % 11 - 5, 10**6) for j in range(36)))
    assert sum(forms) == 0 and sum(forms[:18]) != 0
    transfer = -sum(forms[:18])
    assert transfer == _dot(projected, forms)
    assert transfer**2 <= 9 * _dot(forms, forms)
    phases = _offset(SECTION / 3) + _offset(-SECTION / 9)
    dx, _ = _field(mp, forms, phases, (_actual_weights(mp, 0), _actual_weights(mp, 1)))
    current = sum(forms[:9]) - sum(forms[18:27])
    assert abs(sum(dx[:18]) + _mp(mp, current)) < mp.mpf("1e-220")
    assert abs(sum(dx)) < mp.mpf("1e-220")
    # The original covector has norm sqrt(18), but its whole-mean-free
    # projection has norm3. This is not an instantaneous-current precision.
    whole_mean = (Q(1),) * 36
    assert _dot(projected, whole_mean) == 0 != -sum(whole_mean[:18])


def test_shared_strict_coefficients_include_zero_and_noncoercive_candidates(report):
    lower = Q(1, 5) - EPSILON * GAMMA_HI / 2
    upper = 9 + EPSILON * GAMMA_HI / 2
    qx = Q(1, 10) - 2 * EPSILON * GAMMA_HI**2
    qy = EPSILON * GAMMA_LO**2 * (Q(2, 45) - 40 * EPSILON)
    assert _helper() == (lower, upper, qx, qy, min(qx, qy) / upper)
    assert Q(1, 6) < lower and upper < 10
    assert qx > Q(1, 12)
    assert qy > EPSILON * GAMMA_LO**2 / 45
    assert _helper()[4] > RATE
    assert report.strict_storage_lower_bound == lower
    assert report.strict_storage_upper_bound == upper
    assert report.strict_form_loss_lower_bound == qx
    assert report.strict_phase_loss_lower_bound == qy
    assert report.strict_decay_rate == _helper()[4]
    assert _helper(cross_coefficient=Q(0))[3:] == (Q(0), None)
    assert _helper(gamma_lower=Q(0))[3:] == (Q(0), None)
    assert _helper(phase_stiffness_lower=Q(0), damping_norm_upper=Q(0))[3:] == (
        Q(0),
        None,
    )
    assert _helper(cross_coefficient=Q(1, 900))[3:] == (Q(0), None)
    assert _helper(cross_coefficient=Q(1, 800))[3] < 0
    assert _helper(cross_coefficient=Q(1, 800))[4] is None
    assert _helper(cross_coefficient=Q(1200))[0] == 0
    assert _helper(cross_coefficient=Q(1201))[0] < 0
    assert _helper(cross_coefficient=Q(1201))[4] is None
    form_boundary = Q(1, 5) * 9 / (2 * EPSILON * GAMMA_HI**2)
    assert _helper(transport_norm_upper=form_boundary)[2] == 0
    assert _helper(transport_norm_upper=form_boundary)[4] is None
    assert _helper(transport_norm_upper=form_boundary + 1)[2] < 0
    assert _helper(transport_norm_upper=form_boundary + 1)[4] is None
    # This helper has its own energy hypotheses, not the old common-metric
    # kernel's unrelated upper coupling restriction gamma<=1/2.
    assert _helper(gamma_lower=Q(1), gamma_upper=Q(2))[4] > 0


def test_shared_kernel_normalizes_all_ten_arguments_before_domain_or_algebra(
    monkeypatch,
):
    fixed = {
        name: Q(np.int64(value.numerator), np.int64(value.denominator))
        for name, value in KERNEL_INPUTS.items()
    }
    assert kernel_owner._sine_strict_energy_coefficients(**fixed) == _helper()
    for name in KERNEL_INPUTS:
        for bad in (
            True,
            np.bool_(False),
            1,
            1.0,
            None,
            "1",
            float("nan"),
            float("inf"),
            complex(1, 2),
        ):
            with pytest.raises((TypeError, ValueError, OverflowError)):
                _helper(**{name: bad})
        with pytest.raises(ValueError):
            _helper(**{name: Q(-1)})
    for name in ("storage_lower", "storage_upper", "form_loss_lower", "loss_divisor"):
        with pytest.raises(ValueError):
            _helper(**{name: Q(0)})
    with pytest.raises(ValueError):
        _helper(storage_lower=Q(10))
    with pytest.raises(ValueError):
        _helper(gamma_lower=2 * GAMMA_HI)
    calls = []
    normalize = kernel_owner._quadratic_rational

    def capture(value, name):
        calls.append(name)
        return normalize(value, name)

    monkeypatch.setattr(kernel_owner, "_quadratic_rational", capture)
    with pytest.raises(TypeError):
        _helper(storage_lower=Q(-1), cross_coefficient=True)
    assert calls == list(KERNEL_INPUTS)


def test_rational_deadline_prices_the_original_source_and_all_later_ages(report):
    energy = Q(27, 2) * DELTA**2 + 486 * ETA + 18 * ETA**2 + 9 * MATURE**2
    assert RATE == EPSILON * GAMMA_LO**2 / 450
    assert DEADLINE == 64 / RATE
    assert report.conservative_decay_rate == RATE
    assert report.return_age == DEADLINE > Q(1, 2)
    assert report.decay_power == 64
    assert report.decay_upper_bound == Q(1, 2**64)
    assert report.initial_excess_storage_upper_bound == energy
    assert report.strict_initial_upper_bound == 2 * energy
    assert report.returned_norm_squared_upper_bound == 12 * energy / 2**64
    assert report.aggregate_squared_upper_bound == 108 * energy / 2**64 < NOISE**2
    assert report.true_return_tolerance == NOISE
    assert report.late_true_bounds == (-NOISE, NOISE)
    assert report.strict_decay_certified and report.finite_return_certified
    assert report.all_future_ages_covered
    assert report.total_return_time == 2010000 + DEADLINE
    assert report.original_clock_return_time == (2010000 + DEADLINE) * Q(1024, 1023)
    # e>1+1=2 gives exp(-64)<2^-64. No exponential, state evaluation,
    # endpoint acquisition or choice of a first-return time is needed.
    assert Q(1) + Q(1) + Q(1, 2) > 2


def test_independent_endpoint_errors_leave_an_absolute_late_band_and_decrement(report):
    early_lo, early_hi = report.early_true_bounds
    assert report.early_record_bounds == (early_lo - NOISE, early_hi + NOISE)
    assert report.late_record_bounds == (-2 * NOISE, 2 * NOISE)
    assert report.early_threshold_margin == early_lo - NOISE - THRESHOLD > 0
    assert report.late_threshold_margin == THRESHOLD - 2 * NOISE > 0
    assert report.record_separation_margin == early_lo - 3 * NOISE > 0
    assert report.returned_form_true_bounds == (early_lo - NOISE, early_hi + NOISE)
    assert report.returned_form_record_bounds == (
        early_lo - 3 * NOISE,
        early_hi + 3 * NOISE,
    )
    assert early_lo - NOISE > 0  # returned net amount on the same actual history
    assert report.independent_record_sets_separated and report.fixed_threshold_certified
    zero = _assess(
        early_observation_error_bound=Q(0), late_observation_error_bound=Q(0)
    )
    assert zero.early_record_bounds == zero.early_true_bounds
    assert zero.late_record_bounds == zero.late_true_bounds
    only_early = _assess(late_observation_error_bound=Q(0))
    only_late = _assess(early_observation_error_bound=Q(0))
    assert only_early.early_record_bounds == report.early_record_bounds
    assert only_early.late_record_bounds == report.late_true_bounds
    assert only_late.early_record_bounds == report.early_true_bounds
    assert only_late.late_record_bounds == report.late_record_bounds
    for name in ("early_observation_error_bound", "late_observation_error_bound"):
        rejected = _assess(**{name: NOISE + Q(1, 10**100)})
        assert not rejected.observation_allowances_admitted and not rejected.certified
        assert not rejected.fixed_threshold_certified


def test_source_metric_phase_leaf_and_original_clock_are_readmitted(
    monkeypatch, report
):
    source = report.source
    for name, value in (
        ("source_association_certified", False),
        ("broad_trap_certified", False),
        ("phase_convexity_certified", False),
        ("actual_dissipation_lower_bound", Q(1)),
        ("initial_excess_storage_upper_bound", Q(0)),
        ("whole_form_mean", Q(1)),
        ("activation_time", Q(0)),
        ("previously_reached_duration", Q(1)),
        ("clock", "unrescaled clock"),
        ("physical_capacities", (Q(2),) * 36),
        ("original_phase_gauges_and_branches_retained", False),
        ("state_reset_performed", True),
    ):
        changed = replace(source, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_persistent_contact", lambda **kwargs: changed
        )
        result = _assess()
        assert not result.certified
    # The early claim also consumes its old quantitative component premises;
    # failure there does not invalidate an independently rebuilt late rate.
    positive = source.source.source
    for name, value in (
        ("nominal_transfer_certified", False),
        ("positive_form_error_upper_bound", Q(0)),
        ("comparison_forcing_upper_bound", Q(0)),
    ):
        changed_positive = replace(positive, **{name: value})
        changed_storage = replace(source.source, source=changed_positive)
        changed = replace(source, source=changed_storage)
        monkeypatch.setattr(
            owner, "_assess_mixed_persistent_contact", lambda **kwargs: changed
        )
        result = _assess()
        assert not result.early_transfer_certified and not result.certified
        assert result.finite_return_certified
    monkeypatch.setattr(
        owner, "_assess_mixed_persistent_contact", lambda **kwargs: source
    )
    registered = owner._derive_registered_phase_probe_geometry()
    for name, value in (
        ("bare_gap_lower_bound", Q(3)),
        ("spectral_gap_certified", False),
    ):
        changed_geometry = replace(registered.successor, **{name: value})
        changed_registered = replace(registered, successor=changed_geometry)
        monkeypatch.setattr(
            owner, "_derive_registered_phase_probe_geometry", lambda: changed_registered
        )
        result = _assess()
        assert not result.certified and not result.finite_return_certified


def test_old_record_bands_and_global_verdicts_do_not_replace_new_premises(
    monkeypatch, report
):
    source = report.source
    positive = replace(
        source.source.source,
        certified=False,
        independent_record_sets_separated=False,
        fixed_threshold_certified=False,
        positive_record_bounds=(Q(0), Q(0)),
        cut_record_bounds=(Q(0), Q(0)),
        positive_true_bounds=(Q(0), Q(0)),
        status="old_record_unused",
        unresolved_conditions=("unused",),
    )
    storage = replace(
        source.source,
        source=positive,
        certified=False,
        nominal_classification_certified=False,
        fine_port_only_obstruction_certified=False,
    )
    changed = replace(
        source,
        source=storage,
        certified=False,
        all_time_identity_certified=False,
        full_relaxation_certified=False,
        vanishing_net_transfer_certified=False,
        nominal_cone_certified=False,
        uniform_tube_certified=False,
        status="old_global_verdict_unused",
        unresolved_conditions=("unused",),
    )
    monkeypatch.setattr(
        owner, "_assess_mixed_persistent_contact", lambda **kwargs: changed
    )
    result = _assess()
    assert result.certified
    assert result.strict_decay_certified and result.finite_return_certified
    assert result.early_true_bounds == report.early_true_bounds
    assert result.aggregate_squared_upper_bound == report.aggregate_squared_upper_bound


def test_raw_original_and_observation_errors_precede_source_work(monkeypatch):
    for name in ("form_error_bounds", "phase_error_bounds"):
        assert not _assess(**{name: (ERROR + Q(1, 10**100), ERROR)}).certified

    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached source or derived arithmetic")

    monkeypatch.setattr(owner, "_assess_mixed_persistent_contact", forbidden)
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
                value = tuple(bad if j == side else Q(0) for j in range(2))
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    _assess(**{name: value})
        for bad in ((0,), (0, 0, 0), {0, 1}, {0: 0, 1: 0}, repeat(0)):
            with pytest.raises((TypeError, ValueError)):
                _assess(**{name: bad})
    for name in ("early_observation_error_bound", "late_observation_error_bound"):
        for bad in invalid:
            with pytest.raises((TypeError, ValueError, OverflowError)):
                _assess(**{name: bad})


def test_one_shot_fixed_width_inputs_cannot_retune_or_reset_the_deadline(report):
    assert report.same_actual_history_required and report.independent_endpoint_errors
    assert not report.endpoint_error_cancellation_assumed
    assert not report.state_error_reinitialized
    assert not report.source_association_authenticated
    assert not report.observed_records_supplied
    assert not report.current_or_phase_precision_granted
    assert not report.first_return_time_identified
    assert not report.exact_finite_zero_arrival_claimed
    assert report.observed_relaxation_rate is None
    assert not report.aggregate_monotonicity_claimed
    assert not report.state_reset_performed
    assert not report.port_switch_performed
    assert not report.source_trajectory_evaluated
    assert not report.runtime_event_executed
    assert report.numerical_response_allowance == 0
    assert report.newly_observed_aggregates == 0
    assert report.required_endpoint_records == 2
    assert not report.physical_clock_identified
    assert not report.empirical_validation
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
    with pytest.raises(FrozenInstanceError):
        report.return_age = Q(0)
    for name in (
        "source",
        "observed_record",
        "early_record",
        "late_record",
        "current_error_bound",
        "duration",
        "return_age",
        "first_return_time",
        "phase_gauge",
        "endpoint",
        "cross_coefficient",
        "threshold",
        "capacity",
        "port_strength",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: Q(0)})
