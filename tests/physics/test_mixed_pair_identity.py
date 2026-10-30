"""A marked pair, its invariant representation and oriented response task.

Coordinate covariance preserves the physical port. Erasing that association
loses task information; constructed closure witnesses are not acquired sources.
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
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _mixed_effective_interaction as effective_owner
from tnfr.physics import _mixed_pair_identity as owner
from tnfr.physics import _mixed_phase_sensitive_interface as phase_owner
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
    return owner._assess_mixed_pair_identity(**values)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_retained_producer():
    static_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("pair identity controls cannot replay or advance a source")

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
        pytest.fail("static pair identity proof cannot evaluate an exponential or root")

    with pytest.MonkeyPatch.context() as patch:
        for name in ("exp", "expm", "eig", "polyroots", "odefun", "invertlaplace"):
            patch.setattr(context, name, forbidden)
        yield context


@pytest.fixture(scope="module")
def report():
    return _assess()


def _copy_center(values):
    return _center(values[:18]) + _center(values[18:])


def _swap(values):
    return tuple(values[18:]) + tuple(values[:18])


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _static_state():
    forms = _copy_center(tuple(Q((j * 3) % 11 - 5, 10**8) for j in range(36)))
    common = (-Q(1, 10**7),) * 18 + (Q(1, 10**7),) * 18
    forms = tuple(a + b for a, b in zip(forms, common, strict=True))
    centered = _copy_center(tuple(Q((j * 5) % 7 - 3, 10**6) for j in range(36)))
    phases = tuple(
        value + (Q(1, 3) if j >= 18 else 0) for j, value in enumerate(centered)
    )
    return forms, phases


def _invariants(mp, aggregate, phase):
    return aggregate**2, aggregate * mp.sin(phase), mp.cos(phase)


def _invariant_rows(mp, aggregate, phase, residual=0):
    gamma = 1 / (1023 * mp.pi)
    u, v, w = _invariants(mp, aggregate, phase)
    return (
        -4 * u - 36 * gamma * v - 2 * gamma * aggregate * residual,
        -2 * v
        - 18 * gamma * (1 - w * w)
        + 2 * gamma * u * w / 9
        - gamma * residual * mp.sin(phase),
        -2 * gamma * v / 9,
    )


def test_membership_is_actual_state_and_maintenance_is_source_conditional(report):
    assert report.source_maintained_membership_certified
    assert report.transported_windings == (1, 2, 1, 2)
    assert len(report.marked_pairs) == len(set(report.marked_pairs)) == 18
    assert all((a < 18) == (b < 18) for a, b in report.marked_pairs)
    assert (
        report.scaled_radius_margin
        == Q(1, 500) ** 2 - report.scaled_norm_squared_upper_bound
        > 0
    )
    assert report.registration_margin == DELTA / 4 - report.marked_gap_upper_bound > 0
    assert report.phase_sector_bound == pi_interval() / 2 + Q(1, 1000)
    assert report.phase_lift_margin > 0
    assert (
        report.uniform_form_error_upper_bound
        == report.source.uniform_form_error_upper_bound
        < NOISE
    )
    assert (
        report.uniform_phase_error_upper_bound
        == report.source.uniform_phase_error_upper_bound
    )
    assert report.source_prediction_certified
    assert not report.arbitrary_member_prediction_certified
    assert not report.pair_acquisition_claimed
    # Actual-state membership has its own margins; the original source is
    # the admission domain for maintenance and associated prediction.


def test_complete_field_and_relative_chart_ignore_constant_common_origin(mp):
    forms, phases = _static_state()
    weights = (_weights(mp, 0), _weights(mp, 1))
    field = _new_field(weights)
    state = _state(forms, phases)
    shift = Q(7, 13)
    shifted = state[:36] + tuple(value + shift for value in state[36:])
    expected = _field(mp, forms, phases, weights)
    translated = _field(mp, forms, tuple(value + shift for value in phases), weights)
    for actual, other, left, right in zip(
        field(state),
        field(shifted),
        expected[0] + expected[1],
        translated[0] + translated[1],
        strict=True,
    ):
        _contains(mp, actual, left)
        _contains(mp, other, right)
        assert abs(left - right) < mp.mpf("1e-190")
    mean_difference = (sum(phases[18:]) - sum(phases[:18])) / 18
    moved_difference = (
        sum(value + shift for value in phases[18:])
        - sum(value + shift for value in phases[:18])
    ) / 18
    assert moved_difference == mean_difference
    assert -sum(forms[:18]) == Q(18, 10**7)
    # Only a constant origin change is used. A rotating frame would add
    # a common phase row and is not this same-law covariance statement.


def test_copy_exchange_transports_unequal_weights_and_all_rows(mp):
    forms, phases = _static_state()
    weights = (_weights(mp, 0), _weights(mp, 1))
    assert weights[0] != weights[1]
    before = _new_field(weights)(_state(forms, phases))
    after = _new_field(weights[::-1])(_state(_swap(forms), _swap(phases)))
    oracle = _field(mp, forms, phases, weights)
    moved = _field(mp, _swap(forms), _swap(phases), weights[::-1])
    for actual, expected in zip(before, oracle[0] + oracle[1], strict=True):
        _contains(mp, actual, expected)
    for actual, expected, old in zip(
        after, moved[0] + moved[1], _swap(oracle[0]) + _swap(oracle[1]), strict=True
    ):
        _contains(mp, actual, expected)
        assert abs(expected - old) < mp.mpf("1e-190")
    q = -sum(forms[:18])
    psi = (sum(phases[18:]) - sum(phases[:18])) / 18
    assert -sum(_swap(forms)[:18]) == -q
    assert (sum(_swap(phases)[18:]) - sum(_swap(phases)[:18])) / 18 == -psi
    # Weights, support slots, histories and port labels travel with the
    # components. This is not a new preparation with held labels.


def test_swapping_only_state_is_not_a_fixed_weight_symmetry(mp):
    forms = (Q(1, 10**6), -Q(1, 10**6)) + (Q(0),) * 34
    phases = (Q(0),) * 36
    weights = (_weights(mp, 0), _weights(mp, 1))
    state = _state(_swap(forms), _swap(phases))
    fixed = _new_field(weights)(state)
    transported = _new_field(weights[::-1])(state)
    fixed_oracle = _field(mp, _swap(forms), _swap(phases), weights)
    transported_oracle = _field(mp, _swap(forms), _swap(phases), weights[::-1])
    dx = tuple(
        a - b for a, b in zip(fixed_oracle[0], transported_oracle[0], strict=True)
    )
    dy = tuple(
        a - b for a, b in zip(fixed_oracle[1], transported_oracle[1], strict=True)
    )
    assert max(abs(value) for value in dx) > mp.mpf("1e-100")
    gamma = 1 / (1023 * mp.pi)
    assert max(abs(b + gamma * a) for a, b in zip(dx, dy, strict=True)) < mp.mpf(
        "1e-190"
    )
    for got, want in zip(fixed, fixed_oracle[0] + fixed_oracle[1], strict=True):
        _contains(mp, got, want)
    for got, want in zip(
        transported, transported_oracle[0] + transported_oracle[1], strict=True
    ):
        _contains(mp, got, want)
    # A legal unequal-weight larger-class row witness. No acquired
    # trajectory or numerical symmetry failure is inferred from it.


def test_oriented_readout_transports_the_physical_port_covector(mp):
    forms, phases = _static_state()
    physical_port = (-Q(1),) * 18 + (Q(0),) * 18
    assert sum(forms) == 0
    q = _dot(physical_port, forms)
    moved = _swap(forms)
    assert _dot(_swap(physical_port), moved) == q
    assert _dot(physical_port, moved) == -q
    psi = (sum(phases[18:]) - sum(phases[:18])) / 18
    for orientation in (-1, 1):
        assert orientation * q == (-orientation) * (-q)
        old = _invariants(mp, _mp(mp, q), _mp(mp, psi))
        new = _invariants(mp, -_mp(mp, q), -_mp(mp, psi))
        assert all(abs(a - b) < mp.mpf("1e-190") for a, b in zip(old, new, strict=True))
    # If the donor is identifiable from its retained history/metadata,
    # the requested port association is already available there.


def test_effective_invariant_rows_keep_constraint_and_actual_residuals(mp, report):
    gamma = 1 / (1023 * mp.pi)
    q, psi, residual = (mp.mpf(value) for value in ("0.0002", "0.3", "0.00000004"))
    reduced = effective_owner._mixed_effective_interaction_field()(
        (I(Q(1, 5000)), I(Q(3, 10)))
    )
    qdot = -2 * q - 18 * gamma * mp.sin(psi)
    psidot = 2 * gamma * q / 9
    _contains(mp, reduced[0], qdot)
    _contains(mp, reduced[1], psidot)
    negated = effective_owner._mixed_effective_interaction_field()(
        (I(-Q(1, 5000)), I(-Q(3, 10)))
    )
    _contains(mp, negated[0], -qdot)
    _contains(mp, negated[1], -psidot)
    u, v, w = _invariants(mp, q, psi)
    assert abs(v * v - u * (1 - w * w)) < mp.mpf("1e-190")
    for r in (mp.mpf(0), residual):
        derivative = (
            2 * q * (qdot - gamma * r),
            (qdot - gamma * r) * mp.sin(psi) + q * mp.cos(psi) * psidot,
            -mp.sin(psi) * psidot,
        )
        rows = _invariant_rows(mp, q, psi, r)
        assert all(
            abs(a - b) < mp.mpf("1e-190") for a, b in zip(derivative, rows, strict=True)
        )
        assert abs(
            2 * v * rows[1] - (1 - w * w) * rows[0] + 2 * u * w * rows[2]
        ) < mp.mpf("1e-190")
    actual = _invariant_rows(mp, q, psi, residual)
    effective = _invariant_rows(mp, q, psi)
    assert abs(actual[0] - effective[0] + 2 * gamma * q * residual) < mp.mpf("1e-190")
    assert abs(actual[1] - effective[1] + gamma * residual * mp.sin(psi)) < mp.mpf(
        "1e-190"
    )
    b = report.aggregate_absolute_upper_bound
    eq = report.uniform_form_error_upper_bound
    ep = report.uniform_phase_error_upper_bound
    assert report.invariant_prediction_error_bounds == (2 * b * eq, eq + b * ep, ep)
    aq, hq = _mp(mp, b / 4 + eq / 2), _mp(mp, b / 4)
    ap, hp = mp.mpf("0.2") + _mp(mp, ep / 2), mp.mpf("0.2")
    for actual, predicted, bound in zip(
        _invariants(mp, aq, ap),
        _invariants(mp, hq, hp),
        report.invariant_prediction_error_bounds,
        strict=True,
    ):
        assert abs(actual - predicted) <= _mp(mp, bound)
    # These observables have different units. Their bounds inherit no
    # single common 1e-9 target or exact actual closure claim.


def test_invariants_separate_sign_orbits_but_are_singular_at_the_origin(mp):
    points = tuple(
        (q, p) for q in (Q(-1, 10), Q(0), Q(1, 10)) for p in (Q(-1, 3), Q(0), Q(1, 3))
    )
    for q, p in points:
        inv = _invariants(mp, _mp(mp, q), _mp(mp, p))
        for r, s in points:
            other = _invariants(mp, _mp(mp, r), _mp(mp, s))
            coincident = all(
                abs(a - b) < mp.mpf("1e-190") for a, b in zip(inv, other, strict=True)
            )
            assert coincident is ((q, p) == (r, s) or (q, p) == (-r, -s))
        # At least one two-by-two minor is nonzero away from the only fixed
        # point in the admitted |psi|<pi sector; all derivatives vanish there.
        jac = (
            (2 * _mp(mp, q), mp.mpf(0)),
            (mp.sin(_mp(mp, p)), _mp(mp, q) * mp.cos(_mp(mp, p))),
            (mp.mpf(0), -mp.sin(_mp(mp, p))),
        )
        minors = tuple(
            jac[i][0] * jac[j][1] - jac[i][1] * jac[j][0]
            for i, j in ((0, 1), (0, 2), (1, 2))
        )
        if q == p == 0:
            assert all(value == 0 for row in jac for value in row)
        else:
            assert any(abs(value) > mp.mpf("1e-20") for value in minors)
    # Separation on a quotient does not make this three-coordinate
    # representation a globally regular coordinate system.


def test_existing_negative_band_prices_erased_port_association(report):
    lower, upper = report.inherited_registered_true_bounds
    assert lower <= upper < 0
    assert report.port_blind_error_lower_bound == -upper
    assert report.port_association_obstruction_margin == -upper - NOISE > 0
    assert report.port_association_obstruction_certified
    for q in (lower, (lower + upper) / 2, upper):
        for prediction in (q, Q(0), -q):
            assert (
                max(abs(prediction - q), abs(prediction + q))
                >= abs(q)
                >= -upper
                > NOISE
            )
    # Formal points in an enclosing band need not all be attainable.
    # The minimax inequality holds for every retained actual member.
    # Only the port-to-local-chart association is erased; with it, oQ
    # has the same value in both descriptions of the same physical member.


def test_hidden_offsets_limit_exact_closure_inside_identity_shaped_class(mp, report):
    amplitude = qstar = DELTA / 10000
    zeta = (amplitude, amplitude, -2 * amplitude) + (Q(0),) * 15
    centered = (Q(0),) * 18 + zeta
    forms = (-qstar / 18,) * 18 + (qstar / 18,) * 18
    baseline = (Q(0),) * 36
    assert sum(forms) == sum(centered[:18]) == sum(centered[18:]) == 0
    assert _copy_center(forms) == (Q(0),) * 36
    assert report.witness_qstar == report.witness_offset_amplitude == amplitude
    assert report.witness_matching_offsets == zeta
    assert report.witness_centered_phase == centered
    assert (
        report.witness_scaled_norm_squared
        == 4 * _dot(centered, centered)
        == 24 * amplitude**2
        < Q(1, 500) ** 2
    )
    assert report.witness_marked_gap_upper_bound == 3 * amplitude < DELTA / 4
    for a, b in report.marked_pairs:
        branch = _turns()[b % 18] - _turns()[a % 18]
        assert branch.denominator == 1
        assert abs(centered[b] - centered[a]) <= 3 * amplitude
    target = _turns() * 2
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, value)
        for turn, value in zip(target, centered, strict=True)
    )
    for offset, winding in ((0, 1), (9, 2), (18, 1), (27, 2)):
        turns = sum(
            (theta[offset + (j + 1) % 9] - theta[offset + j] + mp.pi) % (2 * mp.pi)
            - mp.pi
            for j in range(9)
        ) / (2 * mp.pi)
        assert abs(turns - winding) < mp.mpf("1e-190")
    weights = (_weights(mp, 0), _weights(mp, 1))
    field = _new_field(weights)
    rows = []
    for phase in (baseline, centered):
        actual = field(_state(forms, phase))
        oracle = _field(mp, forms, phase, weights)
        for got, want in zip(actual, oracle[0] + oracle[1], strict=True):
            _contains(mp, got, want)
        rows.append(-sum(oracle[0][:18]))
    a = _mp(mp, amplitude)
    moment = 2 * mp.sin(a) * (1 - mp.cos(a))
    assert moment > 0
    gamma = 1 / (1023 * mp.pi)
    assert abs(rows[1] - rows[0] + gamma * moment) < mp.mpf("1e-190")
    udifference = 2 * _mp(mp, qstar) * (rows[1] - rows[0])
    assert udifference < 0
    _contains(mp, report.witness_moment_difference_bounds, moment)
    _contains(mp, report.witness_form_rate_difference_bounds, -gamma * moment)
    _contains(mp, report.witness_invariant_u_rate_difference_bounds, udifference)
    # Q, psi and all three invariants coincide, while Q' and U' differ.
    # These members of the larger identity-shaped class are not asserted
    # to come from the acquired histories used by the approximation theorem.
    assert not report.witness_is_acquired_source
    assert not report.exact_actual_invariant_closure


def test_zero_primitive_resultant_and_lift_alias_keep_phase_unavailable(mp):
    # Every supplied C9 winding profile sums to zero, even after a
    # constant regional phase offset. No circular angle is defined there.
    shifts = (Q(1, 1000), Q(-1, 2000), Q(3, 10), Q(-2, 7))
    primitive = []
    for region in range(4):
        winding = 1 + region % 2
        angles = tuple(
            2 * mp.pi * winding * _mp(mp, Q(j - 4, 9)) + _mp(mp, shifts[region])
            for j in range(9)
        )
        resultant = sum(mp.cos(value) + mp.j * mp.sin(value) for value in angles)
        assert abs(resultant) < mp.mpf("1e-190")
        primitive.append(resultant)
    assert abs(sum(primitive)) < mp.mpf("1e-190")
    forms, phases = _static_state()
    weights = (_weights(mp, 0), _weights(mp, 1))
    state = _state(forms, phases)
    alias = list(state)
    alias[36 + 18] += 2 * pi_interval()
    original = _new_field(weights)(state)
    changed = _new_field(weights)(tuple(alias))
    lifted = tuple(
        _mp(mp, value) + (2 * mp.pi if j == 18 else 0) for j, value in enumerate(phases)
    )
    original_oracle = _field(mp, forms, phases, weights)
    alias_oracle = _field(mp, forms, lifted, weights)
    for first, second, left, right in zip(
        original,
        changed,
        original_oracle[0] + original_oracle[1],
        alias_oracle[0] + alias_oracle[1],
        strict=True,
    ):
        _contains(mp, first, left)
        _contains(mp, second, right)
        assert abs(left - right) < mp.mpf("1e-190")
    before = _mp(mp, (sum(phases[18:]) - sum(phases[:18])) / 18)
    after = (sum(lifted[18:]) - sum(lifted[:18])) / 18
    assert abs(after - before - mp.pi / 9) < mp.mpf("1e-190")
    lifted_centered = _copy_center(lifted)
    assert 4 * _dot(lifted_centered, lifted_centered) > _mp(mp, Q(1, 500) ** 2)
    # A circularly identical representative can leave the admitted mean
    # chart. Its lift/reference/history data cannot be discarded silently.


def _replace_at(value, path, **changes):
    if not path:
        return replace(value, **changes)
    return replace(
        value, **{path[0]: _replace_at(getattr(value, path[0]), path[1:], **changes)}
    )


def test_consumed_sources_are_readmitted_without_global_verdict_substitution(
    monkeypatch, report
):
    source = report.source
    cases = (
        ((), "source_association_certified", False),
        ((), "complete_laws_certified", False),
        ((), "source_comparison_upper_bound", 200 * MATURE),
        ((), "inherited_output_error_upper_bound", Q(0)),
        ((), "aggregate_absolute_upper_bound", Q(0)),
        ((), "registered_observation_error_bound", NOISE),
        ((), "initial_registrations_turns", (Q(0), Q(0))),
        ((), "whole_history_covariance_retained", False),
        ((), "source_correlated_initialization_retained", False),
        ((), "initial_reduced_form", "restart from a later actual reading"),
        ((), "actual_phase_supplied_to_predictor", True),
        ((), "physical_capacities", (Q(0),) + (Q(1),) * 35),
        ((), "whole_form_mean", Q(1)),
        ((), "common_storage_reference", "rotated storage reference"),
        ((), "switch_time", Q(5385008)),
        ((), "clock", "different current clock"),
        ((), "actual_form_equation", "Q_prime=-2*Q"),
        ((), "effective_phase_equation", "psihat_prime=0"),
        ((), "state_reset_performed", True),
        ((), "prior_domain_certified", False),
        ((), "reference_cone_certified", False),
        ((), "incremental_comparison_certified", False),
        ((), "scaled_norm_squared_upper_bound", Q(0)),
        ((), "signed_sensitivity_certified", False),
        ((), "central_entry_certified", False),
        ((), "residual_integral_certified", False),
        ((), "decay_rate", Q(0)),
        ((), "metric_root_upper_bound", Q(0)),
        ((), "reference_phase_squared_integral_upper_bound", Q(0)),
        ((), "comparison_phase_squared_integral_upper_bound", Q(0)),
        ((), "moment_integral_upper_bound", Q(0)),
        ((), "sensitivity_gain_upper_bound", Q(1)),
        ((), "uniform_form_error_upper_bound", Q(0)),
        ((), "uniform_phase_error_upper_bound", Q(0)),
        ((), "original_source_error_paid_once", False),
        ((), "switching_work_certified", False),
        ((), "phase_switching_work_allowance", Q(0)),
        (("source",), "forced_comparison_certified", False),
        (("source",), "cycle_windings_certified", False),
        (("source",), "marked_gap_upper_bound", Q(0)),
        (("source",), "registration_margin", Q(0)),
        (("source",), "metric_root_upper_bound", Q(0)),
        (("source",), "marked_pairs", ()),
        (("source",), "direct_entry_energy_adds_attachment_twice", True),
        (("source",) * 2, "phase_sector_bound", I(0)),
        (("source",) * 4, "pressure_selected_from_preparation_label", True),
        (("source",) * 5, "complete_precontact_history_translated", False),
        (("source",) * 5, "original_lifts_and_error_association_retained", False),
        (("source",) * 5, "registration_turns", Q(0)),
        (("source",) * 5, "centered_identity_certified", False),
        (("source",) * 5, "relative_phase_drift_certified", False),
        (("source",) * 5, "phase_moment_remainder_certified", False),
        (("source",) * 5, "post_duration", Q(16)),
        (("source",) * 5, "centered_norm_upper_bound", Q(0)),
        (("source",) * 5, "relative_phase_deviation_upper_bound", Q(0)),
        (("source",) * 5, "phase_moment_deficit_upper_bound", Q(0)),
        (("source",) * 5, "registered_true_bounds", (Q(0), Q(0))),
        (("source",) * 6, "phase_mean_error_upper_bound", Q(0)),
        (("source",) * 6, "original_phase_lifts_retained", False),
        (("source",) * 8, "same_actual_history_required", False),
        (("source",) * 8, "correlated_initialization_retained", False),
        (("source",) * 8, "early_age", SWITCH_AGE + 1),
        (("source",) * 10, "uniform_comparison_upper_bound", 200 * MATURE),
        (("source",) * 10, "original_phase_gauges_and_branches_retained", False),
        (("source",) * 12, "exact_zero_entry_total_from_prior_laws", False),
        (("source",) * 12, "prefix_cut_duration", Q(10001)),
    )
    for path, name, value in cases:
        changed = _replace_at(source, path, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_collective_sensitivity", lambda **kw: changed
        )
        assert not _assess().certified
    path = ("source",) * 12 + ("handoff", "prefix", "reference")
    reference = source
    for name in path:
        reference = getattr(reference, name)
    for name, value in (
        ("original_supplied_phase_origins", (Q(0), DELTA)),
        ("actual_weight_deviation_upper_bound", Q(0)),
        ("recipient_isolated_deadline", Q(10**10 - 1)),
        ("inherited_donor_velocity_retained", False),
        ("clock", "untransported source clock"),
    ):
        changed = _replace_at(source, path, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_collective_sensitivity", lambda **kw: changed
        )
        assert not _assess().certified
    original = replace(reference.sources[0], phase_reset_error_bound=4 * ETA)
    changed = _replace_at(
        source,
        path,
        sources=(original, reference.sources[1]),
        donor=replace(reference.donor, source=original),
    )
    monkeypatch.setattr(
        owner, "_assess_mixed_collective_sensitivity", lambda **kw: changed
    )
    assert not _assess().source_association_certified
    monkeypatch.setattr(
        owner, "_assess_mixed_collective_sensitivity", lambda **kw: source
    )
    geometry = owner._derive_registered_phase_probe_geometry()
    for changes in (
        dict(bare_gap_lower_bound=Q(3)),
        dict(weighted_gap_lower_bound=Q(2)),
        dict(spectral_gap_certified=False),
    ):
        altered = replace(geometry, successor=replace(geometry.successor, **changes))
        monkeypatch.setattr(
            owner, "_derive_registered_phase_probe_geometry", lambda: altered
        )
        assert not _assess().source_maintained_membership_certified
    monkeypatch.setattr(
        owner, "_derive_registered_phase_probe_geometry", lambda: geometry
    )
    stale = replace(
        source,
        certified=False,
        uniform_envelope_certified=False,
        form_accuracy_certified=False,
        phase_accuracy_certified=True,
        original_uniform_prediction_tolerance_certified=True,
        observation_allowances_admitted=False,
        all_postattachment_times_covered=False,
        status="unused old verdict",
        unresolved_conditions=("unused",),
        source=replace(
            source.source,
            certified=False,
            strict_marked_identity_certified=False,
            source=source.source.source,
        ),
    )
    stale = _replace_at(
        stale,
        ("source",) * 5,
        certified=False,
        registered_record_certified=False,
        aligned_record_certified=False,
        fixed_threshold_certified=False,
    )
    monkeypatch.setattr(
        owner, "_assess_mixed_collective_sensitivity", lambda **kw: stale
    )
    rebuilt = _assess()
    assert rebuilt.certified
    assert rebuilt.source_maintained_membership_certified
    assert rebuilt.source_prediction_certified
    assert (
        rebuilt.invariant_prediction_error_bounds
        == report.invariant_prediction_error_bounds
    )
    assert (
        rebuilt.port_association_obstruction_margin
        == report.port_association_obstruction_margin
    )
    assert source == report.source


def test_raw_fixed_width_and_immutable_contract_keep_prediction_scope(
    monkeypatch, report, record_property
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
    for name in ("form_error_bounds", "phase_error_bounds"):
        assert not _assess(**{name: (ERROR + Q(1, 10**100), ERROR)}).certified
    for name in (
        "registered_observation_error_bound",
        "aligned_observation_error_bound",
    ):
        outside = _assess(**{name: NOISE + Q(1, 10**100)})
        assert not outside.observation_allowances_admitted and not outside.certified
        assert (
            outside.source_maintained_membership_certified
            and outside.source_prediction_certified
        )
        assert (
            outside.port_association_obstruction_margin
            == report.port_association_obstruction_margin
        )
    exact = _assess(
        registered_observation_error_bound=0, aligned_observation_error_bound=0
    )
    assert (
        exact.registered_form_record_error_upper_bound
        == exact.aligned_form_record_error_upper_bound
        == report.uniform_form_error_upper_bound
    )
    assert (
        report.registered_form_record_error_upper_bound
        == report.aligned_form_record_error_upper_bound
        == report.uniform_form_error_upper_bound + NOISE
    )
    assert report.status == "conditional_marked_pair_and_oriented_prediction"
    assert report.invariant_coordinates == ("Q**2", "Q*sin(psi)", "cos(psi)")
    assert report.invariant_constraint == "V**2=U*(1-W**2)"
    assert (
        report.invariant_coordinate_count == 3
        and report.invariant_continuous_dimension == 2
    )
    assert report.invariant_singular_point == (Q(0), Q(0), Q(1))
    assert report.copy_chart_reversal == ("-Q", "-psi", "-R", "-o")
    assert (
        report.physical_port_observation
        == "o*Q; o transforms with the local copy chart"
    )
    assert report.full_state_dimension == 72 and report.reduced_state_dimension == 2
    assert report.physical_capacities == (Q(1),) * 36 and report.internal_divisor == 9
    assert report.whole_form_mean == report.original_donor_form_total == 0
    assert report.initial_registrations_turns == (Q(0), Q(1, 4))
    assert report.switch_age == SWITCH_AGE and report.switch_time == 5385000
    assert report.original_clock_start_time * Q(1023, 1024) == report.switch_time
    assert report.inherited_discrimination_age == 8
    assert report.intrinsic_form_tolerance == NOISE
    assert (
        report.registration_gap_limit == DELTA / 4
        and report.scaled_radius_limit == Q(1, 500)
    )
    assert (
        report.original_correlated_initializer
        == "(Q_s(3375000),alpha); alpha in{0,pi/2}"
    )
    assert report.circular_resultant_nonzero_lower_bound is None
    n0 = 4 * QHI + 200 * MATURE
    work = 972 * ETA + Q(9, 2) * n0**3
    assert report.phase_switching_work_allowance == work
    assert report.aligned_storage_jump_bounds == (Q(0), 9 * (n0 + 18 * ETA) ** 2)
    assert report.registered_storage_jump_bounds == (162 - work, 162 + work + 9 * n0**2)
    for name in (
        "source_membership_retained_for_all_times",
        "original_source_error_paid_once",
        "source_correlated_initialization_retained",
        "original_attachment_work_retained_once",
        "weights_support_cycles_and_history_transported",
        "physical_port_association_retained",
        "complete_law_transport_certified",
        "common_origin_invariance_certified",
        "transported_relabeling_covariance_certified",
        "constituent_exchange_covariance_certified",
        "invariant_chart_certified",
        "oriented_port_descent_certified",
        "inherited_negative_band_certified",
        "larger_class_exact_nonclosure_certified",
        "phase_resultant_boundary_certified",
        "switching_work_certified",
    ):
        assert getattr(report, name) is True
    for name in (
        "membership_implies_arbitrary_source_admission",
        "arbitrary_member_maintenance_certified",
        "arbitrary_member_prediction_certified",
        "arbitrary_age_predictor_restart_admitted",
        "exact_actual_invariant_closure",
        "globally_regular_invariant_chart",
        "invariant_coordinates_independently_free",
        "fixed_weight_swap_symmetry_claimed",
        "primitive_whole_pair_phase_defined",
        "independent_phase_representatives_quotiented",
        "time_dependent_phase_frame_admitted",
        "orientation_metadata_is_new_state_coordinate",
        "relabeling_is_physical_intervention",
        "original_noise_arguments_reassigned_to_port_readings",
        "invariant_observables_share_one_accuracy_target",
        "witness_is_acquired_source",
        "witness_refutes_source_approximation",
        "pair_acquisition_claimed",
        "autonomous_pair_formation_claimed",
        "intrinsic_pair_epi_or_phase_identified",
        "new_field_installed",
        "new_observation_acquired",
        "state_reset_performed",
        "source_trajectory_evaluated",
        "reduced_trajectory_evaluated",
        "original_source_replayed",
        "runtime_event_executed",
        "physical_comparator_installed",
        "physical_property_identified",
        "empirical_validation",
    ):
        assert getattr(report, name) is False
    assert not hasattr(report, "post_duration") and not hasattr(
        report, "actual_source_state"
    )
    with pytest.raises(FrozenInstanceError):
        report.uniform_form_error_upper_bound = Q(0)
    for name in (
        "source",
        "state",
        "weights",
        "actual_phase",
        "initial_phase",
        "initial_reading",
        "horizon",
        "port_orientation",
        "threshold",
        "registration",
        "clock",
        "initialization",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: Q(0)})
    # Only this one module fixture supplies reported acceptance properties.
    for name in (
        "switch_time",
        "inherited_discrimination_age",
        "source_comparison_upper_bound",
        "centered_entry_norm_upper_bound",
        "scaled_norm_squared_upper_bound",
        "scaled_radius_margin",
        "marked_gap_upper_bound",
        "registration_margin",
        "phase_lift_margin",
        "aggregate_absolute_upper_bound",
        "uniform_form_error_upper_bound",
        "uniform_phase_error_upper_bound",
        "registered_form_record_error_upper_bound",
        "aligned_form_record_error_upper_bound",
        "port_blind_error_lower_bound",
        "port_association_obstruction_margin",
        "phase_switching_work_allowance",
        "witness_qstar",
        "witness_offset_amplitude",
        "witness_scaled_norm_squared",
        "witness_marked_gap_upper_bound",
    ):
        record_property(name, str(getattr(report, name)))
    for name in (
        "invariant_prediction_error_bounds",
        "inherited_registered_true_bounds",
        "aligned_storage_jump_bounds",
        "registered_storage_jump_bounds",
    ):
        record_property(
            name, json.dumps([str(value) for value in getattr(report, name)])
        )
    for name in (
        "phase_sector_bound",
        "witness_moment_difference_bounds",
        "witness_form_rate_difference_bounds",
        "witness_invariant_u_rate_difference_bounds",
    ):
        interval = getattr(report, name)
        record_property(name, json.dumps([str(interval.lo), str(interval.hi)]))
    for name in (
        "source_maintained_membership_certified",
        "source_prediction_certified",
        "port_association_obstruction_certified",
        "larger_class_exact_nonclosure_certified",
        "arbitrary_member_prediction_certified",
        "exact_actual_invariant_closure",
        "pair_acquisition_claimed",
    ):
        record_property(name, str(getattr(report, name)))

    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached retained source work")

    monkeypatch.setattr(owner, "_assess_mixed_collective_sensitivity", forbidden)
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
