"""All-time identity uses a moving reference and its finite total variation.

The edge oracle is independent of the reader. Its constructed chart states
check identities, not acquired endpoints or trajectories. Algebraic exponential
comparisons concern bounds only; no source, contact or operator is advanced.
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
from tnfr.physics import _mixed_persistent_contact as owner
from tnfr.physics import _mixed_reciprocal_partner as field_owner
from tnfr.physics import _sine_flow as flow_owner

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
    values = dict(form_error_bounds=(ERROR, ERROR), phase_error_bounds=(ERROR, ERROR))
    values.update(changes)
    return owner._assess_mixed_persistent_contact(**values)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_prior_producer():
    factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("persistence controls cannot advance or replay a source")

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


def test_actual_entry_storage_traps_chart_without_resetting_the_half_unit_state(report):
    initial = Q(27, 2) * DELTA**2 + 486 * ETA + 18 * ETA**2 + 9 * MATURE**2
    barrier = Q(1, 5) * Q(1, 32) ** 2
    assert report.initial_excess_storage_upper_bound == initial
    assert report.storage_barrier == barrier
    assert report.storage_margin == barrier - initial > 0
    assert 5 * initial < Q(1, 100) ** 2 < Q(1, 32) ** 2
    assert report.actual_dissipation_lower_bound == Q(1, 4) - 84 * WEIGHT_ERROR > 0
    assert report.broad_trap_certified
    # The broad bound is not itself the original much tighter gap test.
    assert Q(1, 100) > DELTA / 4


def test_all_nominal_cone_faces_include_the_generated_common_form(mp, report):
    gamma = 1 / (1023 * mp.pi)
    k = mp.mpf(13) / 27 - _mp(mp, DELTA) / (18 * mp.pi)
    coefficient = 1 + 2 * mp.cos(mp.pi / 9)
    stiffness = 4 * gamma**2 * k * coefficient / 9
    qhi = _mp(mp, QHI)
    for s, z in (
        (qhi, qhi),
        (qhi / 3, 2 * qhi / 3),
        (mp.mpf(0), qhi / 2),
        (qhi / 2, mp.mpf(0)),
    ):
        for b_fraction in (0, 1):
            for u_fraction in (0, 1):
                b, u = b_fraction * gamma * z, u_fraction * mp.mpf(6) / 5 * gamma * z
                zd = -4 * gamma * k * b
                ud = -u + b
                bd = (
                    u
                    - (2 * k + 1) * b
                    + 2 * gamma * coefficient * mp.sin(z / 2) * mp.cos(s / 2) / 9
                )
                if u_fraction:
                    assert ud - mp.mpf(6) / 5 * gamma * zd <= 0
                else:
                    assert ud >= 0
                if b_fraction:
                    assert bd - gamma * zd <= 0
                else:
                    assert bd >= 0
                for velocity_face in (0, 1):
                    sd = -velocity_face * gamma**2 * s
                    sdd = -2 * k * sd - 2 * stiffness * mp.sin(s / 2) * mp.cos(z / 2)
                    if velocity_face:
                        assert sdd + gamma**2 * sd >= 0
                    else:
                        assert sdd <= 0
                assert zd <= 0
    assert Q(6, 5) - 1 > Q(24, 5) * GAMMA_HI**2 * Q(1, 2)
    assert 2 * Q(4, 9) - Q(1, 5) - Q(1, 3) - 2 * GAMMA_HI**2 > 0
    assert Q(1) / (4 * Q(4, 9)) < 1  # inherited cut b <= gamma*z/(4*k)
    assert report.nominal_cone_certified


def test_reference_variation_and_form_integral_keep_common_mode_memory(mp, report):
    # Exact coordinate geometry supplies the constants in the L1 budgets.
    gamma = 1 / (1023 * mp.pi)
    k = mp.mpf(13) / 27 - _mp(mp, DELTA) / (18 * mp.pi)
    u, b, vsum = mp.mpf("2e-8"), mp.mpf("1e-8"), -mp.mpf("3e-8")
    a, r = -u, -b
    vd, vr = (vsum + r) / 2, (vsum - r) / 2
    forms = (
        (a / 2 - vd,) * 9 + (a / 2 + vd,) * 9 + (-a / 2 - vr,) * 9 + (-a / 2 + vr,) * 9
    )
    assert abs(_dot(forms, forms) - 9 * (u * u + b * b + vsum * vsum)) < mp.mpf(
        "1e-220"
    )
    qd, qr = 4 * gamma * k * vd, 4 * gamma * k * vr
    sd, zd = qd + qr, qd - qr
    phase_rate = _offset(qd) + _offset(qr)
    assert abs(
        _dot(phase_rate, phase_rate) - mp.mpf(9) / 4 * (sd * sd + zd * zd)
    ) < mp.mpf("1e-220")
    assert mp.sqrt(_dot(phase_rate, phase_rate)) <= mp.mpf(3) / 2 * (abs(sd) + abs(zd))
    # u'=-u+b, u(0)=0 gives integral(u)<=integral(b); the initial
    # condition belongs to original A activation, not a new half-unit reset.
    assert Q(9) / (4 * Q(4, 9)) < 10
    assert report.nominal_phase_variation_upper_bound == 3 * QHI
    assert report.nominal_form_integral_upper_bound == 10 * QHI / GAMMA_LO
    assert report.finite_variation_certified


def test_actual_metric_bregman_identity_uses_nominal_form_in_both_forcing_rows(mp):
    gamma = 1 / (1023 * mp.pi)
    c = mp.mpf(13) / 3 - _mp(mp, DELTA) / (2 * mp.pi)
    bar_x = (Q(1, 10**8),) * 9 + (-Q(3, 10**8),) * 9 + (Q(2, 10**8),) * 18
    bar_x = tuple(_mp(mp, value) for value in _center(bar_x))
    bar_y = _offset(SECTION / 2) + _offset(-SECTION / 8)
    ex = tuple(
        _mp(mp, value)
        for value in _center(tuple(Q((j * 7) % 13 - 6, 10**10) for j in range(36)))
    )
    ey = tuple(
        _mp(mp, value)
        for p in (0, 1)
        for value in _center(tuple(Q((j * 3 + p) % 11 - 5, 10**10) for j in range(18)))
    )
    x = tuple(a + b for a, b in zip(bar_x, ex, strict=True))
    y = tuple(_mp(mp, a) + b for a, b in zip(bar_y, ey, strict=True))
    actual_weights = (_actual_weights(mp, 0), _actual_weights(mp, 1))
    nominal_weights = (_nominal_weights(mp),) * 2
    dx, dy = _field(mp, x, y, actual_weights)
    bx, by = _field(mp, bar_x, bar_y, nominal_weights)
    exd, eyd = tuple(a - b for a, b in zip(dx, bx, strict=True)), tuple(
        a - b for a, b in zip(dy, by, strict=True)
    )
    hx = _h_apply(mp, ex, actual_weights)
    delta_bar = tuple(
        a - b
        for a, b in zip(
            _h_apply(mp, bar_x, actual_weights),
            _h_apply(mp, bar_x, nominal_weights),
            strict=True,
        )
    )
    delta_error = tuple(
        a - b for a, b in zip(hx, _h_apply(mp, ex, nominal_weights), strict=True)
    )
    fx, fy = tuple(-value / 9 for value in delta_bar), tuple(
        gamma * value / 9 for value in delta_bar
    )
    gradient = tuple(b - a for a, b in zip(_sine(mp, y), _sine(mp, bar_y), strict=True))
    effort = tuple(h + 2 * c * m for h, m in zip(hx, _relative_common(ex), strict=True))
    remainder = tuple(
        g - k for g, k in zip(gradient, _hessian_apply(mp, bar_y, ey), strict=True)
    )
    drift = _dot(remainder, by)
    derivative = _dot(effort, exd) + _dot(gradient, eyd) + drift
    dissipation = _dot(hx, hx) / 9 + _dot(effort, _exchange(ex))
    expected = -dissipation + _dot(effort, fx) + _dot(gradient, fy) + drift
    assert abs(derivative - expected) < mp.mpf("1e-210")
    assert max(abs(value) for value in delta_error) > mp.mpf("1e-100")
    # Swapping back to H0 in the homogeneous row while retaining this
    # reference-form forcing omits deltaH*ex and is a different identity.
    assert max(abs(value / 9) for value in delta_error) > mp.mpf("1e-101")
    assert abs(drift) > mp.mpf("1e-50")
    state = tuple(I(Q(mp.nstr(value, 210))) for value in x)
    state += tuple(
        2 * pi_interval() * turn + I(Q(mp.nstr(value, 210)))
        for turn, value in zip(_turns() * 2, y, strict=True)
    )
    field = field_owner._mixed_reciprocal_partner_field(
        first_new_edge_conductances=actual_weights[0],
        second_new_edge_conductances=actual_weights[1],
    )
    for interval, value in zip(field(state), dx + dy, strict=True):
        assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def test_bregman_coercivity_and_moving_reference_remainder_are_not_zero(mp):
    gamma = 1 / (1023 * mp.pi)
    c = mp.mpf(13) / 3 - _mp(mp, DELTA) / (2 * mp.pi)
    bx = tuple(
        _mp(mp, value)
        for value in _center(tuple(Q((j * 5) % 7 - 3, 10**5) for j in range(36)))
    )
    by = _offset(SECTION / 2) + _offset(-SECTION / 7)
    ex = tuple(
        _mp(mp, value)
        for value in _center(tuple(Q((j * 3) % 11 - 5, 10**8) for j in range(36)))
    )
    ey = tuple(
        _mp(mp, value)
        for p in (0, 1)
        for value in _center(tuple(Q((j + p) % 7 - 3, 10**7) for j in range(18)))
    )
    ay = tuple(_mp(mp, value) + error for value, error in zip(by, ey, strict=True))
    weights = (_actual_weights(mp, 0), _actual_weights(mp, 1))
    hx = _h_apply(mp, ex, weights)
    grad_bar = tuple(-value for value in _sine(mp, by))
    potential = _potential(mp, ay) - _potential(mp, by) - _dot(grad_bar, ey)
    energy = _dot(ex, hx) / 2 + c * _dot(ex, _relative_common(ex)) + potential
    norm_squared = _dot(ex, ex) + _dot(ey, ey)
    assert norm_squared / 5 <= energy <= 9 * norm_squared
    gradient = tuple(b - a for a, b in zip(_sine(mp, ay), _sine(mp, by), strict=True))
    remainder = tuple(
        a - b for a, b in zip(gradient, _hessian_apply(mp, by, ey), strict=True)
    )
    velocity = tuple(gamma * h / 9 for h in _h_apply(mp, bx, weights))
    vnorm = mp.sqrt(_dot(velocity, velocity))
    assert abs(_dot(remainder, velocity)) <= 18 * vnorm * _dot(ey, ey)
    assert abs(_dot(remainder, velocity)) > 0


def test_finite_gronwall_budget_keeps_the_original_marked_tolerance(mp, report):
    ly, lx = 3 * QHI, 10 * QHI / GAMMA_LO
    exponent = 45 * ly
    initial = 24 * MATURE
    delta = 14 * WEIGHT_ERROR
    upper = 3 * (initial + 6 * delta * lx) / (1 - exponent)
    assert 0 < exponent < 1
    assert mp.exp(_mp(mp, exponent)) < _mp(mp, 1 / (1 - exponent))
    assert report.bregman_growth_integral_upper_bound == exponent
    assert report.initial_bregman_root_upper_bound == initial
    assert report.weight_matrix_deviation_upper_bound == delta
    assert report.uniform_comparison_upper_bound == upper < 200 * MATURE
    assert report.uniform_comparison_budget == 200 * MATURE
    assert report.marked_gap_upper_bound == QHI + 300 * MATURE
    assert report.registration_margin == DELTA / 4 - report.marked_gap_upper_bound > 0
    assert report.bregman_comparison_certified
    assert report.uniform_tube_certified and report.all_time_identity_certified
    assert report.all_contact_durations_covered


def test_largest_zero_loss_invariant_set_needs_the_full_phase_row(mp, report):
    weights = (_actual_weights(mp, 0), _actual_weights(mp, 1))
    zero = (Q(0),) * 36
    noncritical = _offset(SECTION / 2) + _offset(Q(0))
    dx, dy = _field(mp, zero, noncritical, weights)
    assert all(value == 0 for value in dy)
    assert max(abs(value) for value in dx) > 0  # X=0 alone is not invariant
    # On the centered phase leaf the Hessian's strict lower bound excludes
    # another critical point in the acute chart, with gauges still fixed.
    vector = tuple(
        _mp(mp, value)
        for p in (0, 1)
        for value in _center(tuple(Q((j * 5 + p) % 13 - 6, 10**6) for j in range(18)))
    )
    assert _dot(vector, _hessian_apply(mp, noncritical, vector)) >= _mp(
        mp, Q(2, 5)
    ) * _dot(vector, vector)
    eq_dx, eq_dy = _field(mp, zero, zero, weights)
    assert max(abs(value) for value in eq_dx + eq_dy) < mp.mpf("1e-230")
    assert report.full_relaxation_certified
    assert report.phase_convexity_certified
    assert report.full_form_and_phase_relaxation_claimed


def test_vanishing_net_transfer_is_a_limit_not_a_return_time(mp, report):
    forms = tuple(
        _mp(mp, value)
        for value in _center(tuple(Q((j * 3) % 11 - 5, 10**6) for j in range(36)))
    )
    phases = _offset(SECTION / 3) + _offset(-SECTION / 9)
    dx, _ = _field(mp, forms, phases, (_actual_weights(mp, 0), _actual_weights(mp, 1)))
    current = sum(forms[:9]) - sum(forms[18:27])
    assert abs(sum(dx[:18]) + current) < mp.mpf("1e-220")
    assert abs(sum(dx)) < mp.mpf("1e-220")
    assert report.vanishing_net_transfer_certified
    assert report.net_transfer_limit == 0
    assert report.finite_relaxation_deadline is None
    assert report.exponential_relaxation_rate is None
    assert not report.monotone_net_transfer_claimed
    assert report.certified


def test_prior_record_and_storage_selection_verdicts_do_not_replace_premises(
    monkeypatch, report
):
    source = report.source
    positive = replace(
        source.source,
        certified=False,
        independent_record_sets_separated=False,
        fixed_threshold_certified=False,
        positive_record_bounds=(Q(0), Q(0)),
        cut_record_bounds=(Q(0), Q(0)),
        status="record_unused",
        unresolved_conditions=("unused",),
    )
    changed = replace(
        source,
        source=positive,
        certified=False,
        nominal_classification_certified=False,
        fine_port_only_obstruction_certified=False,
        status="unrelated_storage_claims_unused",
        unresolved_conditions=("unused",),
    )
    monkeypatch.setattr(
        owner, "_assess_mixed_common_form_storage", lambda **kwargs: changed
    )
    result = _assess()
    assert result.certified and result.all_time_identity_certified
    assert (
        result.uniform_comparison_upper_bound == report.uniform_comparison_upper_bound
    )


def test_actual_source_metric_and_activation_history_are_readmitted(
    monkeypatch, report
):
    source = report.source
    positive = source.source
    for name, value in (
        ("actual_dissipation_lower_bound", Q(1)),
        ("actual_dissipation_certified", False),
        ("transport_gap_lower_bound", Q(2)),
        ("fixed_port_strength", Q(2)),
    ):
        changed = replace(source, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_common_form_storage", lambda **kwargs: changed
        )
        result = _assess()
        assert not result.certified
        assert not result.all_contact_durations_covered
        assert not result.full_form_and_phase_relaxation_claimed
    for name, value in (
        ("handoff_time", Q(0)),
        ("positive_duration", Q(1)),
        ("initial_donor_form_total", Q(1)),
        ("state_reset_performed", True),
        ("contact_pairs", tuple((j, 18 + j) for j in range(18))),
    ):
        changed = replace(source, source=replace(positive, **{name: value}))
        monkeypatch.setattr(
            owner, "_assess_mixed_common_form_storage", lambda **kwargs: changed
        )
        result = _assess()
        assert not result.certified
        assert not result.all_contact_durations_covered
        assert not result.full_form_and_phase_relaxation_claimed
    old = positive.handoff.prefix.reference
    for name, value in (
        ("original_relative_phase_gap", SECTION),
        ("initial_form_means", (Q(0), Q(1))),
        ("actual_weight_deviation_upper_bound", 2 * WEIGHT_ERROR),
        ("inherited_donor_velocity_retained", False),
    ):
        changed_old = replace(old, **{name: value})
        changed_prefix = replace(positive.handoff.prefix, reference=changed_old)
        changed_handoff = replace(positive.handoff, prefix=changed_prefix)
        changed = replace(source, source=replace(positive, handoff=changed_handoff))
        monkeypatch.setattr(
            owner, "_assess_mixed_common_form_storage", lambda **kwargs: changed
        )
        result = _assess()
        assert not result.certified
        assert not result.all_contact_durations_covered
        if name == "inherited_donor_velocity_retained":
            # Phase relaxation follows the broad actual trap even when the
            # separate reference-cone premise for tight identity is absent.
            assert result.full_form_and_phase_relaxation_claimed
            assert result.vanishing_net_transfer_certified
        else:
            assert not result.full_form_and_phase_relaxation_claimed


def test_raw_original_pairs_precede_source_and_geometry_work(monkeypatch):
    for name in ("form_error_bounds", "phase_error_bounds"):
        assert not _assess(**{name: (ERROR + Q(1, 10**100), ERROR)}).certified

    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached source or derived arithmetic")

    monkeypatch.setattr(owner, "_assess_mixed_common_form_storage", forbidden)
    for name in ("form_error_bounds", "phase_error_bounds"):
        for bad in (
            True,
            np.bool_(False),
            None,
            "0",
            float("nan"),
            float("inf"),
            complex(1, 2),
            -Q(1, 10**100),
        ):
            for side in (0, 1):
                values = tuple(bad if j == side else Q(0) for j in range(2))
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    _assess(**{name: values})
        for bad in ((0,), (0, 0, 0), {0, 1}, {0: 0, 1: 0}, repeat(0)):
            with pytest.raises((TypeError, ValueError)):
                _assess(**{name: bad})


def test_one_shot_fixed_width_does_not_accept_a_reset_rate_or_endpoint(report):
    assert report.activation_time == 2010000
    assert report.previously_reached_duration == Q(1, 2)
    assert report.activation_relative_common_form == 0
    assert report.independently_observed_aggregates == 0
    assert report.numerical_response_allowance == 0
    assert not report.state_reset_performed
    assert not report.port_switch_performed
    assert not report.new_preparation_performed
    assert not report.source_trajectory_evaluated
    assert not report.runtime_event_executed
    assert not report.outer_object_declared
    assert not report.physical_energy_identified
    assert not report.empirical_validation
    error = Q(np.int64(ERROR.numerator), np.int64(ERROR.denominator))
    assert (
        _assess(
            form_error_bounds=(error for _ in range(2)),
            phase_error_bounds=(error for _ in range(2)),
        )
        == report
    )
    with pytest.raises(FrozenInstanceError):
        report.uniform_comparison_budget = Q(0)
    for name in (
        "source",
        "endpoint",
        "common_form",
        "donor_velocity",
        "elapsed_time",
        "duration",
        "relaxation_rate",
        "return_time",
        "observed_record",
        "phase_gauge",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: Q(0)})
