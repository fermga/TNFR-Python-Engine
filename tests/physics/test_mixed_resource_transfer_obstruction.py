"""Resource is present but the declared mutual target remains inaccessible.

Constructed points check the complete law and comparison algebra. They do not
replace the actual section donor or mature recipient, and no trajectory,
source replay, new preparation input or response evaluation is performed.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _mixed_resource_transfer_obstruction as owner
from tnfr.physics import _mixed_two_channel_circulation as field_owner
from tnfr.physics import _sine_coupling_successor as source_owner
from tnfr.physics import _sine_flow as flow_owner
from tnfr.physics import _sine_lyapunov as kernel_owner

ERROR = Q(1, 10**10)
ETA = Q(2, 10**69)
DELTA = Q(1, 1024)
AMPLITUDE = DELTA / 8
SECTION = 3 * DELTA / 16
REGISTRATION = DELTA / 4
WEIGHT_ERROR = Q(2, 10**60)
WAIT = Q(10**10)
MATURE = Q(1, 2**28)
PREFIX = 3 * MATURE / 2
SCALE = Q(6, 5)
SECTION_ERROR = Q(3, 2) * (14 * ETA + 2 * WAIT * WEIGHT_ERROR)


def _mp(mp, value):
    if isinstance(value, (int, Q, np.integer)):
        value = Q(value)
        return mp.mpf(value.numerator) / value.denominator
    return mp.mpf(value)


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
    assert interval.width < Q(1, 10**30)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _center(values):
    mean = sum(values) / len(values)
    return tuple(value - mean for value in values)


def _turns():
    return tuple(Q((1 + i // 9) * (i % 9 - 4), 9) for i in range(18))


def _support():
    bins = tuple(int((9 * value) % 9) for value in _turns())
    pairs = {
        (i, j)
        for i in range(18)
        for j in range(i + 1, 18)
        if min((bins[j] - bins[i]) % 9, (bins[i] - bins[j]) % 9) in (1, 2)
        or (i // 9 != j // 9 and bins[i] == bins[j])
    }
    old = tuple(
        sorted(
            tuple(sorted((9 * r + j, 9 * r + (j + 1) % 9)))
            for r in (0, 1)
            for j in range(9)
        )
    )
    return old, tuple(sorted(pairs - set(old)))


def _weights(side):
    return tuple(Q(3, 4) + Q((i + side) % 5, 16) for i in range(63))


def _edges(weights):
    old, new = _support()
    return tuple((a, b, Q(1)) for a, b in old) + tuple(
        (a, b, w) for (a, b), w in zip(new, weights, strict=True)
    )


def _nominal_weights(mp, gap=DELTA):
    phase = tuple(
        2 * mp.pi * _mp(mp, t) + (-1 if i < 9 else 1) * _mp(mp, gap) / 2
        for i, t in enumerate(_turns())
    )
    return tuple(
        1 - abs((phase[b] - phase[a] + mp.pi) % (2 * mp.pi) - mp.pi) / (2 * mp.pi)
        for a, b in _support()[1]
    )


def _laplacian(edges):
    rows = [[0] * 18 for _ in range(18)]
    for a, b, w in edges:
        rows[a][a] += w
        rows[b][b] += w
        rows[a][b] -= w
        rows[b][a] -= w
    return tuple(tuple(row) for row in rows)


def _matvec(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


def _product(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(18)) for j in range(18))
        for i in range(18)
    )


def _exchange(vector):
    first = tuple(a - b for a, b in zip(vector[:18], vector[18:]))
    return first + tuple(-value for value in first)


def _block(h, vector):
    return _matvec(h, vector[:18]) + _matvec(h, vector[18:])


def _offset(gap):
    return (-gap / 2,) * 9 + (gap / 2,) * 9


def _state(forms, errors, gauges=(Q(0), Q(0))):
    return tuple(I(x) for x in forms) + tuple(
        2 * pi_interval() * t + y + gauges[i // 18]
        for i, (t, y) in enumerate(zip(_turns() * 2, errors))
    )


def _oracle(mp, forms, errors, weights, exchange=1):
    x = tuple(_mp(mp, value) for value in forms)
    phase = tuple(
        2 * mp.pi * _mp(mp, t) + _mp(mp, y) for t, y in zip(_turns() * 2, errors)
    )
    gradient, sine = [mp.mpf(0)] * 36, [mp.mpf(0)] * 36
    for side in (0, 1):
        for a, b, w in _edges(weights[side]):
            a, b = a + 18 * side, b + 18 * side
            q = _mp(mp, w) * (x[a] - x[b])
            s = mp.sin(phase[b] - phase[a])
            gradient[a] += q
            gradient[b] -= q
            sine[a] += s
            sine[b] -= s
    gx = _exchange(x)
    gamma = 1 / (1023 * mp.pi)
    dx = tuple(
        (-q + gamma * s) / 9 - exchange * g for q, s, g in zip(gradient, sine, gx)
    )
    dy = tuple(gamma * q / 9 for q in gradient)
    return x, tuple(gradient), tuple(sine), dx, dy


def _assess(**changes):
    values = dict(form_error_bounds=(ERROR, ERROR), phase_error_bounds=(ERROR, ERROR))
    values.update(changes)
    return owner._assess_mixed_resource_transfer_obstruction(**values)


def _kernel_arguments(**changes):
    values = dict(
        gamma_lower=Q(1, 2750),
        gamma_upper=Q(1, 2500),
        cosine_lower=Q(5, 72),
        gap_lower=Q(3),
        capacity=Q(4, 9),
        exchange_norm_upper=Q(2),
    )
    values.update(changes)
    return values


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_prior_producer():
    factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("transfer obstruction controls cannot advance or replay a source")

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
    result.dps = 200
    return result


@pytest.fixture(scope="module")
def report():
    return _assess()


@pytest.fixture(scope="module")
def fields():
    full = field_owner._mixed_two_channel_field(
        first_new_edge_conductances=_weights(0),
        second_new_edge_conductances=_weights(1),
        law="full",
    )
    cut = tuple(
        source_owner._coupling_successor_field(
            completion="common_sense",
            new_edge_conductances=_weights(side),
            loss_normalization="bare_degree",
        )
        for side in (0, 1)
    )
    return full, cut


def test_complete_72_rows_keep_mutual_reactions_clock_and_zero_means(mp, fields):
    forms = tuple(
        value
        for side in (0, 1)
        for value in _center(tuple(Q((7 * i + side) % 17 - 8, 8192) for i in range(18)))
    )
    errors = _offset(SECTION) + _offset(Q(0))
    weights = (_weights(0), _weights(1))
    _, q, sine, dx, dy = _oracle(mp, forms, errors, weights)
    for interval, value in zip(fields[0](_state(forms, errors)), dx + dy):
        _contains(mp, interval, value)
    for side in (0, 1):
        assert abs(sum(dx[18 * side : 18 * side + 18])) < mp.mpf("1e-190")
        assert abs(sum(dy[18 * side : 18 * side + 18])) < mp.mpf("1e-190")
    storage_rate = _dot(q, dx) - _dot(sine, dy)
    expected_rate = -_dot(q, q) / 9 - _dot(
        q, _exchange(tuple(_mp(mp, x) for x in forms))
    )
    assert abs(storage_rate - expected_rate) < mp.mpf("1e-190")
    assert max(abs(value) for value in _exchange(forms)) > 0
    cut = _oracle(mp, forms, errors, weights, exchange=0)
    for side in (0, 1):
        a, b = 18 * side, 18 * side + 18
        state = tuple(I(v) for v in forms[a:b]) + tuple(
            2 * pi_interval() * t + y for t, y in zip(_turns(), errors[a:b])
        )
        for interval, value in zip(fields[1][side](state), cut[-2][a:b] + cut[-1][a:b]):
            _contains(mp, interval, value)
    for i, gx in enumerate(_exchange(tuple(_mp(mp, x) for x in forms))):
        assert abs(dx[i] - cut[-2][i] + gx) < mp.mpf("1e-190")
        assert dy[i] == cut[-1][i]
    clock = _mp(mp, Q(1023, 1024))
    original = tuple(
        -clock * qi / 9 + si / (1024 * mp.pi * 9) - clock * gi
        for qi, si, gi in zip(q, sine, _exchange(tuple(_mp(mp, x) for x in forms)))
    )
    original += tuple(qi / (1024 * mp.pi * 9) for qi in q)
    assert max(abs(clock * a - b) for a, b in zip(dx + dy, original)) < mp.mpf("1e-190")
    for interval, value in zip(
        fields[0](_state(forms, errors, (Q(2, 7), -Q(3, 11)))), dx + dy
    ):
        _contains(mp, interval, value)


def test_original_nominal_weights_and_inherited_velocity_give_paired_wedges(mp):
    weights = _nominal_weights(mp)
    rewritten = _nominal_weights(mp, SECTION)
    assert max(abs(a - b) for a, b in zip(weights, rewritten)) > _mp(mp, WEIGHT_ERROR)
    gamma = 1 / (1023 * mp.pi)
    k = mp.mpf(13) / 27 - _mp(mp, DELTA) / (18 * mp.pi)
    c = 1 + 2 * mp.cos(mp.pi / 9)
    factor = 4 * gamma**2 * k * c / 9
    q0 = _mp(mp, SECTION + SECTION_ERROR / 2)
    w0 = -(gamma**2) * q0 / 2
    assert w0 < 0
    # Initial donor forms retain the isolated reference's velocity;
    # the independently mature recipient reference has no centered state.
    donor_v = w0 / (4 * gamma * k)
    initial_forms = (-donor_v,) * 9 + (donor_v,) * 9 + (mp.mpf(0),) * 18
    initial = _oracle(
        mp, initial_forms, _offset(q0) + _offset(mp.mpf(0)), (weights, weights)
    )
    assert abs(initial[-1][9] - initial[-1][0] - w0) < mp.mpf("1e-190")
    assert initial[-1][27] == initial[-1][18] == 0
    for coupling in (0, 1):
        for total, difference in ((q0, q0), (q0 / 3, 2 * q0 / 3)):
            for first_face, second_face in ((0, 0), (0, 1), (1, 0), (1, 1)):
                st, zt = (
                    -first_face * gamma**2 * total,
                    -second_face * gamma**2 * difference,
                )
                qd, qr = (total + difference) / 2, (total - difference) / 2
                vd, vr = (st + zt) / (8 * gamma * k), (st - zt) / (8 * gamma * k)
                forms = (-vd,) * 9 + (vd,) * 9 + (-vr,) * 9 + (vr,) * 9
                _, _, _, dx, dy = _oracle(
                    mp,
                    forms,
                    _offset(qd) + _offset(qr),
                    (weights, weights),
                    exchange=coupling,
                )
                ds = 4 * gamma * k * (dx[9] + dx[27])
                dz = 4 * gamma * k * (dx[9] - dx[27])
                expected_s = -2 * k * st - 2 * factor * mp.sin(total / 2) * mp.cos(
                    difference / 2
                )
                expected_z = -(2 * k + 2 * coupling) * zt - 2 * factor * mp.sin(
                    difference / 2
                ) * mp.cos(total / 2)
                assert abs(ds - expected_s) < mp.mpf("1e-190")
                assert abs(dz - expected_z) < mp.mpf("1e-190")
                assert abs((dy[9] - dy[0]) + (dy[27] - dy[18]) - st) < mp.mpf("1e-190")
                for value, speed, acceleration, face in (
                    (total, st, ds, first_face),
                    (difference, zt, dz, second_face),
                ):
                    if face:
                        assert (
                            acceleration + gamma**2 * speed
                            > gamma**2 * (2 * k / 3 - gamma**2) * value
                            > 0
                        )
                    else:
                        assert acceleration < 0
                assert 0 <= qd <= q0 and abs(qr) <= q0 / 2


def test_added_metric_cancels_contact_cross_term_on_centered_leaves(mp):
    h = _laplacian(tuple((a, b, _mp(mp, w)) for a, b, w in _edges((Q(7, 8),) * 63)))
    zq = tuple(
        value
        for side in (0, 1)
        for value in _center(tuple(Q((i + 2 * side) % 7 - 3, 2**20) for i in range(18)))
    )
    z = tuple(_mp(mp, value) for value in zq)
    y = _block(h, z)
    assert abs(sum(y[:18])) < mp.mpf("1e-190") and abs(sum(y[18:])) < mp.mpf("1e-190")
    # y=H0*z realizes its centered inverse exactly without evaluating one.
    hgz, ghz = _block(h, _exchange(z)), _exchange(_block(h, z))
    assert max(abs(a - b) for a, b in zip(hgz, ghz)) < mp.mpf("1e-190")
    targets = _offset(SECTION) + _offset(Q(0))
    ks = []
    for side in (0, 1):
        target = tuple(
            2 * mp.pi * _mp(mp, t) + _mp(mp, b)
            for t, b in zip(_turns(), targets[18 * side : 18 * side + 18])
        )
        secants = []
        for a, b, _ in _edges((Q(7, 8),) * 63):
            displacement = y[18 * side + b] - y[18 * side + a]
            base = target[b] - target[a]
            coefficient = (
                (mp.sin(base + displacement) - mp.sin(base)) / displacement
                if displacement
                else mp.cos(base)
            )
            secants.append((a, b, coefficient))
        ks.append(_laplacian(secants))
    assert max(
        abs(a - b)
        for r, s in zip(_product(h, ks[0]), _product(ks[0], h))
        for a, b in zip(r, s)
    ) > mp.mpf("1e-6")
    ky = _matvec(ks[0], y[:18]) + _matvec(ks[1], y[18:])
    gamma, scale = 1 / (1023 * mp.pi), _mp(mp, SCALE)
    rho = _mp(mp, Q(5) / (216 * 2750**2 * (1 + Q(1, 2500))))
    for sign in (-1, 1):
        x = tuple(sign * value / 2 for value in y)
        hx, gx = _block(h, x), _exchange(x)
        dx = tuple((-a - gamma * b) / 9 - g for a, b, g in zip(hx, ky, gx))
        dy = tuple(gamma * a / 9 for a in hx)
        inverse_phase_rate = tuple(gamma * a / 9 for a in x)
        base = (
            mp.mpf(9)
            / 4
            * (_dot(x, x) + scale**2 * _dot(y, y) + 2 * scale**2 * gamma * _dot(x, y))
        )
        extra = mp.mpf(81) / 4 * scale**2 * _dot(y, _exchange(z))
        base_rate = (
            mp.mpf(9)
            / 2
            * (
                _dot(x, dx)
                + scale**2 * _dot(y, dy)
                + scale**2 * gamma * (_dot(dx, y) + _dot(x, dy))
            )
        )
        extra_rate = (
            mp.mpf(81)
            / 4
            * scale**2
            * (_dot(dy, _exchange(z)) + _dot(y, _exchange(inverse_phase_rate)))
        )
        assert abs(
            extra_rate - mp.mpf(9) / 2 * scale**2 * gamma * _dot(y, gx)
        ) < mp.mpf("1e-190")
        internal = (
            -(1 - scale**2 * gamma**2) * _dot(x, hx) / 2
            - gamma * _dot(x, ky) / 2
            - scale**2 * gamma**2 * _dot(y, ky) / 2
        )
        expected = internal - mp.mpf(9) / 2 * _dot(x, gx)
        assert abs(base_rate + extra_rate - expected) < mp.mpf("1e-190")
        assert 0 < extra < 7 * base
        assert base_rate + extra_rate < -rho * base <= -(rho / 8) * (base + extra)
        assert sign * _dot(x, y) > 0


def test_unequal_actual_matrices_do_not_commute_and_force_both_rows_on_actual_x(mp):
    nominal = _nominal_weights(mp)
    changes = tuple(
        tuple((-1) ** (i + side) * WEIGHT_ERROR for i in range(63)) for side in (0, 1)
    )
    actual = tuple(
        tuple(w + _mp(mp, d) for w, d in zip(nominal, changes[side])) for side in (0, 1)
    )
    ex = tuple(
        value
        for side in (0, 1)
        for value in _center(tuple(Q((i + side) % 7 - 3, 10**8) for i in range(18)))
    )
    ey = tuple(
        value
        for side in (0, 1)
        for value in _center(tuple(Q((i + side) % 5 - 2, 10**9) for i in range(18)))
    )
    reference_phase = _offset(SECTION) + _offset(Q(0))
    # With x_reference=0, the wrong deltaH*x_reference forcing would vanish.
    reference = _oracle(mp, (Q(0),) * 36, reference_phase, (nominal, nominal))
    reached = _oracle(mp, ex, tuple(a + b for a, b in zip(reference_phase, ey)), actual)
    h0 = _laplacian(tuple((a, b, _mp(mp, w)) for a, b, w in _edges(nominal)))
    delta = tuple(
        _laplacian(tuple((a, b, d) for (a, b), d in zip(_support()[1], changes[side])))
        for side in (0, 1)
    )
    force = _matvec(delta[0], ex[:18]) + _matvec(delta[1], ex[18:])
    assert max(abs(value) for value in force) > 0
    h0ex = _block(h0, tuple(_mp(mp, v) for v in ex))
    gx = _exchange(ex)
    gamma = 1 / (1023 * mp.pi)
    for i in range(36):
        expected_x = (
            -h0ex[i] + gamma * (reached[2][i] - reference[2][i]) - _mp(mp, force[i])
        ) / 9 - _mp(mp, gx[i])
        expected_y = gamma * (h0ex[i] + _mp(mp, force[i])) / 9
        assert abs(reached[-2][i] - reference[-2][i] - expected_x) < mp.mpf("1e-190")
        assert abs(reached[-1][i] - reference[-1][i] - expected_y) < mp.mpf("1e-190")
    dgx = _matvec(delta[0], gx[:18]) + _matvec(delta[1], gx[18:])
    gdex = _exchange(force)
    assert max(abs(a - b) for a, b in zip(dgx, gdex)) > 0
    assert all(
        max(sum(abs(x) for x in row) for row in dh) <= 14 * WEIGHT_ERROR for dh in delta
    )


def test_general_exchange_kernel_has_scalar_capacity_metric_and_zero_exchange_limit():
    gamma = Q(1, 4)
    x, y = (Q(1, 5), -Q(1, 7)), (Q(2, 9), Q(1, 11))
    hdiag, gdiag = (Q(2), Q(3)), (Q(1, 3), Q(1, 2))
    k = ((Q(1), Q(1, 8)), (Q(1, 8), Q(3, 2)))
    hx = tuple(h * a for h, a in zip(hdiag, x))
    ky = tuple(_dot(row, y) for row in k)
    gx = tuple(g * a for g, a in zip(gdiag, x))
    invy = tuple(a / h for a, h in zip(y, hdiag))
    # The two stiffness inequalities follow independently by 2x2 minors.
    assert Q(1, 3) * Q(1, 2) - Q(1, 8) ** 2 > 0
    assert Q(1) * Q(3, 2) - Q(1, 8) ** 2 > 0
    assert hdiag[0] * k[0][1] != k[0][1] * hdiag[1]
    for capacity in (Q(1, 4), Q(4, 9), Q(3, 2)):
        arguments = _kernel_arguments(
            gamma_lower=Q(1, 5),
            gamma_upper=gamma,
            cosine_lower=Q(1, 3),
            gap_lower=Q(2),
            capacity=capacity,
            exchange_norm_upper=Q(1, 2),
        )
        ell, up, rate = kernel_owner._sine_common_quadratic_exchange_coefficients(
            **arguments
        )
        base_lower, base_upper = (1 - gamma) / capacity, (1 + gamma) / capacity
        base_rate = capacity * Q(1, 5) ** 2 * Q(1, 3) * 2 / (4 * (1 + gamma))
        extra_upper = 4 * Q(1, 2) / (capacity**2 * 2)
        assert (ell, up, rate) == (
            base_lower,
            base_upper + extra_upper,
            base_rate * base_lower / (base_lower + extra_upper),
        )
        dx = tuple(-capacity * (a + gamma * b) / 4 - g for a, b, g in zip(hx, ky, gx))
        dy = tuple(gamma * capacity * a / 4 for a in hx)
        base = (_dot(x, x) + _dot(y, y) + 2 * gamma * _dot(x, y)) / capacity
        extra = 4 * sum(a * g * b for a, g, b in zip(y, gdiag, invy)) / capacity**2
        base_rate_actual = (
            2
            * (_dot(x, dx) + _dot(y, dy) + gamma * (_dot(dx, y) + _dot(x, dy)))
            / capacity
        )
        extra_rate = (
            8 * sum(a * g * b for a, g, b in zip(dy, gdiag, invy)) / capacity**2
        )
        internal = (
            -(1 - gamma**2) * _dot(x, hx) / 2
            - gamma * _dot(x, ky) / 2
            - gamma**2 * _dot(y, ky) / 2
        )
        assert base_rate_actual + extra_rate == internal - 2 * _dot(x, gx) / capacity
        assert (
            ell * (_dot(x, x) + _dot(y, y))
            <= base + extra
            <= up * (_dot(x, x) + _dot(y, y))
        )
        assert base_rate_actual + extra_rate <= -rate * (base + extra)
        zero = kernel_owner._sine_common_quadratic_exchange_coefficients(
            **{**arguments, "exchange_norm_upper": Q(0)}
        )
        assert zero == (base_lower, base_upper, base_rate)
    # Candidate scalar bounds cannot establish the missing matrix premise.
    badg = ((Q(1), Q(1, 2)), (Q(1, 2), Q(1)))
    assert hdiag[0] * badg[0][1] != badg[0][1] * hdiag[1]
    dy = tuple(gamma * a / 4 for a in hx)
    inverse_dy = tuple(a / h for a, h in zip(dy, hdiag))
    actual_extra_rate = 4 * (
        _dot(dy, tuple(_dot(row, invy) for row in badg))
        + _dot(y, tuple(_dot(row, inverse_dy) for row in badg))
    )
    presumed_cancel = 2 * gamma * _dot(y, tuple(_dot(row, x) for row in badg))
    assert actual_extra_rate != presumed_cancel
    for key in ("gamma_lower", "cosine_lower"):
        assert (
            kernel_owner._sine_common_quadratic_exchange_coefficients(
                **_kernel_arguments(**{key: Q(0)})
            )[2]
            == 0
        )


def test_shared_kernel_normalizes_every_primitive_before_base_work(monkeypatch):
    base = kernel_owner._sine_common_quadratic_coefficients
    observed = []

    def checked(**kwargs):
        assert all(
            type(value.numerator) is int and type(value.denominator) is int
            for value in kwargs.values()
        )
        observed.append(kwargs)
        return base(**kwargs)

    monkeypatch.setattr(kernel_owner, "_sine_common_quadratic_coefficients", checked)
    fixed = {
        name: Q(np.int64(value.numerator), np.int64(value.denominator))
        for name, value in _kernel_arguments().items()
    }
    expected = kernel_owner._sine_common_quadratic_exchange_coefficients(
        **_kernel_arguments()
    )
    assert (
        kernel_owner._sine_common_quadratic_exchange_coefficients(**fixed) == expected
    )
    assert observed[-1]["capacity_lower"] == observed[-1]["capacity_upper"] == Q(4, 9)

    def forbidden(*args, **kwargs):
        pytest.fail("malformed exchange metric primitive reached the base kernel")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(kernel_owner, "_sine_common_quadratic_coefficients", forbidden)
        for key in _kernel_arguments():
            for bad in (
                True,
                np.bool_(False),
                0,
                0.0,
                "0",
                None,
                float("nan"),
                float("inf"),
                complex(1, 2),
                (Q(1), Q(1)),
            ):
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    kernel_owner._sine_common_quadratic_exchange_coefficients(
                        **_kernel_arguments(**{key: bad})
                    )
        for changes in (
            {"capacity": Q(0)},
            {"capacity": -Q(1)},
            {"exchange_norm_upper": -Q(1)},
        ):
            with pytest.raises(ValueError):
                kernel_owner._sine_common_quadratic_exchange_coefficients(
                    **_kernel_arguments(**changes)
                )
    # Existing domain admission remains owned by the base kernel.
    for changes in (
        {"gamma_lower": -Q(1)},
        {"gamma_upper": Q(2, 3)},
        {"gamma_lower": Q(1, 3)},
        {"cosine_lower": Q(2)},
        {"gap_lower": Q(0)},
    ):
        with pytest.raises(ValueError):
            kernel_owner._sine_common_quadratic_exchange_coefficients(
                **_kernel_arguments(**changes)
            )


def test_actual_trap_precedes_uniform_comparison_and_retains_rate_factor_two(
    report, record_property
):
    energy = Q(27, 2) * DELTA**2 + 486 * ETA + 18 * ETA**2
    ell, up = Q(9, 4) * (1 - Q(1, 2500)), Q(9, 4) * (1 + Q(1, 2500))
    rho = Q(5) / (216 * 2750**2 * (1 + Q(1, 2500)))
    extra = Q(27, 2)
    assert report.initial_excess_storage_upper_bound == energy + 9 * MATURE**2
    assert report.initial_norm_squared_upper_bound == 5 * energy + MATURE**2
    assert (
        report.full_norm_squared_upper_bound
        == 5 * (energy + 9 * MATURE**2)
        < Q(1, 10000)
    )
    assert report.full_norm_upper_bound == Q(1, 100) < Q(1, 32)
    assert report.transport_half_difference_upper_bound == 14 * WEIGHT_ERROR
    assert report.dissipation_margin == 1 - 14 * WEIGHT_ERROR > 0
    assert report.actual_trap_certified
    assert (
        report.base_metric_lower_bound,
        report.base_metric_upper_bound,
        report.base_metric_decay_rate,
    ) == (ell, up, rho)
    assert (
        report.metric_lower_bound,
        report.metric_upper_bound,
        report.metric_decay_rate,
    ) == (ell, up + extra, rho * ell / (ell + extra))
    assert 1 + extra / ell < 8
    assert 8 * up * SCALE**2 / ell < 16
    assert report.metric_growth_squared_margin == 16 - SCALE**2 * (up + extra) / ell > 0
    assert report.conservative_metric_decay_rate == rho / 8
    assert report.metric_decay_rate > rho / 8 > 0
    e0, forcing = MATURE + SECTION_ERROR, 2 * WEIGHT_ERROR
    uniform = 4 * (e0 + 2 * forcing / (rho / 8))
    assert report.initial_comparison_norm_upper_bound == e0
    assert report.comparison_forcing_upper_bound == forcing
    assert (
        report.comparison_forcing_margin
        == forcing - 2 * 14 * WEIGHT_ERROR * Q(1, 100) / 9
        > 0
    )
    assert report.uniform_comparison_bound == uniform < 8 * MATURE
    assert report.uniform_comparison_upper_bound == 8 * MATURE
    assert report.uniform_comparison_margin == 8 * MATURE - uniform > 0
    assert report.comparison_certified and report.metric_certified
    assert report.nominal_q0_bounds == (
        SECTION - SECTION_ERROR / 2,
        SECTION + SECTION_ERROR / 2,
    )
    assert (
        report.marked_gap_upper_bound
        == SECTION + SECTION_ERROR / 2 + 12 * MATURE
        < REGISTRATION
    )
    assert (
        report.registration_margin == REGISTRATION - report.marked_gap_upper_bound > 0
    )
    assert report.all_time_identity_certified and report.all_contact_durations_covered
    record_property("actual_all_time_norm_bound", str(report.full_norm_upper_bound))
    record_property(
        "uniform_comparison_upper_bound", str(report.uniform_comparison_upper_bound)
    )
    record_property(
        "all_time_marked_gap_upper_bound", str(report.marked_gap_upper_bound)
    )
    record_property("comparison_norm_forcing_integral_factor", "2/(rho/8)")


def test_positive_donor_resource_and_unchanged_target_contrast_are_distinct(
    report, record_property
):
    mean = (-Q(1, 9),) * 9 + (Q(1, 9),) * 9
    contrast = mean + tuple(-value for value in mean)
    assert _dot(contrast, contrast) == Q(4, 9)
    target_min = AMPLITUDE - Q(1, 10**8)
    target = _offset(target_min) + _offset(-target_min)
    base = tuple(-3 * PREFIX * value / 2 for value in contrast)
    assert _dot(base, base) == PREFIX**2
    assert _dot(contrast, target) == 2 * target_min
    for sign in (-1, 1):
        point = tuple(sign * (a + b) for a, b in zip(target, base))
        assert abs(_dot(contrast, point)) == 2 * target_min - 2 * PREFIX / 3
        assert abs(_dot(contrast, point)) > 2 * target_min - PREFIX
    assert (
        report.actual_gap_contrast_upper_bound
        == SECTION + SECTION_ERROR / 2 + 8 * MATURE
    )
    assert report.target_gap_contrast_lower_bound == 2 * target_min - PREFIX
    assert (
        report.target_separation_margin
        == AMPLITUDE / 2 - 2 * Q(1, 10**8) - Q(19, 2) * MATURE - SECTION_ERROR / 2
        > 0
    )
    assert (
        report.actual_gap_contrast_upper_bound < report.target_gap_contrast_lower_bound
    )
    assert report.donor_resource_lower_bound == Q(9, 10) * SECTION**2
    assert report.comparison_phase_floor_upper_bound == Q(9, 5) * AMPLITUDE**2
    assert report.resource_floor_margin == Q(9, 40) * AMPLITUDE**2 > 0
    assert report.positive_donor_resource_certified
    assert (
        report.connected_target_unreachable and report.disconnected_target_unreachable
    )
    assert report.target_domain_certified and report.certified
    assert not report.zero_interaction_claimed
    assert not report.response_or_observational_equivalence_claimed
    record_property(
        "donor_section_resource_lower_bound", str(report.donor_resource_lower_bound)
    )
    record_property(
        "necessary_resource_floor_margin", str(report.resource_floor_margin)
    )
    record_property(
        "actual_gap_contrast_upper_bound", str(report.actual_gap_contrast_upper_bound)
    )
    record_property(
        "target_gap_contrast_lower_bound", str(report.target_gap_contrast_lower_bound)
    )
    record_property("target_separation_margin", str(report.target_separation_margin))
    record_property(
        "contact_horizon", "all nonnegative durations; no trajectory evaluated"
    )


def test_source_order_is_explicit_and_old_jump_or_response_verdicts_are_irrelevant(
    monkeypatch, report
):
    donor, ready = report.donor, report.recipient_readiness
    target = replace(
        ready.circulation,
        certified=False,
        source_association_certified=False,
        phase_intervention_certified=False,
        circulation_certified=False,
        independent_hypotheses_separated=False,
        work_certified=False,
        phase_jump_work_upper_bound=Q(100),
    )
    irrelevant_second = replace(
        ready.sources[1], source_certified=False, certified=False
    )
    altered_ready = replace(
        ready,
        sources=(ready.sources[0], irrelevant_second),
        source_certified=(True, False),
        readiness_certified=(True, False),
        circulation=target,
        handoff_association_certified=False,
        composed_response_certified=False,
        certified=False,
    )
    monkeypatch.setattr(
        owner,
        "_assess_mixed_registered_resource",
        lambda **kwargs: replace(donor, certified=False),
    )
    monkeypatch.setattr(
        owner, "_assess_mixed_circulation_readiness", lambda **kwargs: altered_ready
    )
    result = _assess()
    assert result.certified and result.sources == (donor.source, ready.sources[0])
    assert result.target_separation_margin == report.target_separation_margin
    altered_ready = replace(altered_ready, circulation=None)
    result = _assess()
    assert result.certified and result.target_domain_certified
    assert result.target_separation_margin == report.target_separation_margin
    # Original error pairs select donor and recipient independently.
    # Interchanging those associations is rejected even with stale True flags.
    result = _assess(
        form_error_bounds=(ERROR / 2, ERROR), phase_error_bounds=(ERROR, ERROR / 2)
    )
    assert not result.donor_source_certified and not result.recipient_source_certified
    assert not result.certified


def test_actual_section_maturity_and_original_nominal_associations_are_readmitted(
    monkeypatch, report
):
    donor, ready = report.donor, report.recipient_readiness
    for name, value in (
        ("comparison_error_upper_bound", 2 * SECTION_ERROR),
        ("section_mean_gap", 2 * SECTION),
        ("section_orientation_certified", False),
        ("charge_or_leak_prefix_applied", True),
        ("new_external_input_supplied", True),
    ):
        changed = replace(donor, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_registered_resource", lambda **kwargs: changed
        )
        result = _assess()
        assert (
            not result.donor_source_certified
            and not result.connected_target_unreachable
        )
    monkeypatch.setattr(
        owner, "_assess_mixed_registered_resource", lambda **kwargs: donor
    )
    for name, value in (
        ("waiting_duration", WAIT / 2),
        ("derived_mature_norm_upper_bound", 2 * MATURE),
        (
            "returned_norm_squared_upper_bounds",
            (MATURE**2, ready.returned_norm_squared_upper_bounds[1]),
        ),
    ):
        changed = replace(ready, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_circulation_readiness", lambda **kwargs: changed
        )
        result = _assess()
        assert result.donor_source_certified and not result.recipient_source_certified
        assert not result.certified
    monkeypatch.setattr(
        owner, "_assess_mixed_circulation_readiness", lambda **kwargs: ready
    )
    for side in (0, 1):
        source = report.sources[side]
        for name, value in (
            ("edge_weight_deviation_upper_bound", 2 * WEIGHT_ERROR),
            ("gauge_phase_deviation_upper_bound", 7 * ETA),
            ("relative_phase_gap", 2 * DELTA),
        ):
            changed = replace(source, **{name: value})
            with pytest.MonkeyPatch.context() as patch:
                if side == 0:
                    altered = replace(donor, source=changed)
                    patch.setattr(
                        owner,
                        "_assess_mixed_registered_resource",
                        lambda **kwargs: altered,
                    )
                else:
                    altered = replace(ready, sources=(changed, ready.sources[1]))
                    patch.setattr(
                        owner,
                        "_assess_mixed_circulation_readiness",
                        lambda **kwargs: altered,
                    )
                assert altered.certified and changed.certified
                assert not _assess().source_association_certified
        for name, value in (
            ("supplied_phase_origins", (-SECTION / 2, SECTION / 2)),
            ("initial_form_means", (Q(1), Q(0))),
        ):
            changed = replace(
                source,
                precontact_reference=replace(
                    source.precontact_reference, **{name: value}
                ),
            )
            with pytest.MonkeyPatch.context() as patch:
                if side == 0:
                    patch.setattr(
                        owner,
                        "_assess_mixed_registered_resource",
                        lambda **kwargs: replace(donor, source=changed),
                    )
                else:
                    patch.setattr(
                        owner,
                        "_assess_mixed_circulation_readiness",
                        lambda **kwargs: replace(
                            ready, sources=(changed, ready.sources[1])
                        ),
                    )
                assert not _assess().certified


def test_geometry_metric_and_target_domains_have_independent_failure_flags(
    monkeypatch, report
):
    registered = owner._derive_registered_phase_probe_geometry()
    changed = replace(
        registered,
        successor=replace(
            registered.successor,
            new_edge_pairs=registered.successor.new_edge_pairs[:-1],
        ),
    )
    monkeypatch.setattr(
        owner, "_derive_registered_phase_probe_geometry", lambda: changed
    )
    assert not _assess().source_association_certified
    monkeypatch.setattr(
        owner, "_derive_registered_phase_probe_geometry", lambda: registered
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            owner,
            "_sine_common_quadratic_exchange_coefficients",
            lambda **kwargs: (
                report.metric_lower_bound,
                100 * report.metric_upper_bound,
                report.metric_decay_rate,
            ),
        )
        result = _assess()
        assert result.actual_trap_certified
        assert not result.metric_certified and not result.comparison_certified
        assert not result.connected_target_unreachable
    fixed_target = owner._MixedTwoChannelCirculation
    for name, value in (
        ("phase_jump_amplitude", 2 * AMPLITUDE),
        ("phase_jump_error_cap", Q(2, 10**8)),
        ("contact_pairs", fixed_target.contact_pairs[:-1]),
    ):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(fixed_target, name, value)
            result = _assess()
        assert (
            result.source_association_certified and result.all_time_identity_certified
        )
        assert (
            not result.target_domain_certified
            and not result.connected_target_unreachable
        )
        assert not result.certified


def test_raw_pairs_precede_work_and_nonnegative_outside_caps_are_unavailable(
    monkeypatch,
):
    for name in ("form_error_bounds", "phase_error_bounds"):
        for side in (0, 1):
            values = tuple(
                ERROR + Q(1, 10**100) if i == side else ERROR for i in range(2)
            )
            result = _assess(**{name: values})
            assert result.status == "unavailable" and not result.certified
            assert not result.source_association_certified
            assert (
                not result.connected_target_unreachable
                and not result.disconnected_target_unreachable
            )

    def forbidden(*args, **kwargs):
        pytest.fail("invalid source pair reached source, geometry or arithmetic work")

    for name in (
        "_assess_mixed_registered_resource",
        "_assess_mixed_circulation_readiness",
        "_derive_registered_phase_probe_geometry",
        "_sine_common_quadratic_coefficients",
        "_sine_common_quadratic_exchange_coefficients",
        "pi_interval",
        "cos",
    ):
        monkeypatch.setattr(owner, name, forbidden)
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
                pair = tuple(bad if i == side else Q(0) for i in range(2))
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    _assess(**{name: pair})
        for bad in ((0,), (0, 0, 0), {0, 1}, {0: 0, 1: 0}):
            with pytest.raises((TypeError, ValueError)):
                _assess(**{name: bad})


def test_one_shot_fixed_width_and_no_fitted_state_age_resource_or_input(report):
    fixed = Q(np.int64(ERROR.numerator), np.int64(ERROR.denominator))
    assert (
        _assess(
            form_error_bounds=(fixed for _ in range(2)),
            phase_error_bounds=(fixed for _ in range(2)),
        )
        == report
    )
    assert report.status == "conditional_all_time_resource_transfer_obstruction"
    assert (
        report.full_state_dimension == 72 and report.physical_capacities == (Q(1),) * 36
    )
    assert report.contact_pairs == tuple((j, 18 + j) for j in range(18))
    assert report.initial_form_means == (Q(0), Q(0))
    assert report.mean_gap_contrast_covector_norm == Q(2, 3)
    assert report.inherited_donor_velocity_retained
    assert (
        report.inherited_nominal_velocity_relation
        == "-gamma**2*q0 <= w0 <= 0 at the same isolated age"
    )
    assert report.nominal_velocity_factor_upper_bound == Q(1, 3000) ** 2
    assert not report.actual_transport_commutation_assumed
    assert not report.donor_dwell_applied and not report.charge_or_leak_prefix_applied
    assert not report.source_synchronization_authenticated
    assert (
        not report.source_trajectory_evaluated
        and not report.runtime_source_authenticated
    )
    with pytest.raises(FrozenInstanceError):
        report.zero_interaction_claimed = True
    for name in (
        "source",
        "donor",
        "recipient",
        "readiness",
        "initial_state",
        "mature_norm_upper_bounds",
        "section_age",
        "donor_velocity",
        "contact_duration",
        "resource",
        "observed_transfer",
        "new_edge_conductances",
        "input",
        "phase_target",
        "phase_jump_error_bound",
        "readout_error_bound",
        "gamma",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: 0})
