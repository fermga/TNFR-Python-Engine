"""Independent controls for continuous form-driven phase preparation.

Static constructed points verify complete rows and inequalities, not acquired
endpoints. A positive finite residual replaces neither the actual source nor
its unknown common form and phase gauge. No trajectory or control fit is used.
"""

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
from tnfr.physics import _mixed_form_phase_preparation as owner
from tnfr.physics import _sine_coupling_successor as source_owner
from tnfr.physics import _sine_flow as flow_owner

ERROR = Q(1, 10**10)
DELTA = Q(1, 1024)
GAMMA = Q(1, 3000)
WEIGHT_ERROR = Q(2, 10**60)
RADIUS = Q(1, 32)
MATURE = Q(1, 2**28)
PREFIX = 3 * MATURE / 2
AMPLITUDE = Q(1, 8192)
TARGET_ERROR = Q(1, 10**8)
NOISE = Q(1, 10**6)
HOLD = Q(10**10)
SCALE = Q(6, 5)


def _mp(mp, value):
    if isinstance(value, (int, Q)):
        value = Q(value)
        return mp.mpf(value.numerator) / value.denominator
    return mp.mpf(value)


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
    assert interval.width < Q(1, 10**30)


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _center(values):
    mean = sum(values) / len(values)
    return tuple(value - mean for value in values)


def _turns():
    return tuple(Q((1 + i // 9) * (i % 9 - 4), 9) for i in range(18))


def _support():
    bins = tuple(int((9 * turn) % 9) for turn in _turns())
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


def _weights():
    return tuple(Q(3, 4) + Q(i % 5, 16) for i in range(63))


def _edges(weights=None):
    weights = _weights() if weights is None else weights
    old, new = _support()
    return tuple((a, b, Q(1)) for a, b in old) + tuple(
        (a, b, weight) for (a, b), weight in zip(new, weights, strict=True)
    )


def _nominal_weights(mp):
    phase = tuple(
        2 * mp.pi * _mp(mp, turn) + (-1 if i < 9 else 1) * _mp(mp, DELTA) / 2
        for i, turn in enumerate(_turns())
    )
    return tuple(
        1 - abs((phase[b] - phase[a] + mp.pi) % (2 * mp.pi) - mp.pi) / (2 * mp.pi)
        for a, b in _support()[1]
    )


def _laplacian(edges):
    matrix = [[0] * 18 for _ in range(18)]
    for a, b, weight in edges:
        matrix[a][a] += weight
        matrix[b][b] += weight
        matrix[a][b] -= weight
        matrix[b][a] -= weight
    return tuple(tuple(row) for row in matrix)


def _matvec(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


def _product(left, right):
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(18)) for j in range(18))
        for i in range(18)
    )


def _offset(gap):
    return (-gap / 2,) * 9 + (gap / 2,) * 9


def _state(forms, errors, gauge=Q(0)):
    return tuple(I(value) for value in forms) + tuple(
        2 * pi_interval() * turn + value + gauge
        for turn, value in zip(_turns(), errors)
    )


def _oracle(mp, forms, errors, gap, weights=None):
    x = tuple(_mp(mp, value) for value in forms)
    phase = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, value)
        for turn, value in zip(_turns(), errors)
    )
    q, sine = [mp.mpf(0)] * 18, [mp.mpf(0)] * 18
    for a, b, weight in _edges(weights):
        pressure = _mp(mp, weight) * (x[a] - x[b])
        current = mp.sin(phase[b] - phase[a])
        q[a] += pressure
        q[b] -= pressure
        sine[a] += current
        sine[b] -= current
    gamma = 1 / (1023 * mp.pi)
    cross = 1 + 2 * mp.cos(mp.pi / 9)
    first_input = -gamma * cross * mp.sin(_mp(mp, gap)) / 9
    inputs = (first_input,) * 9 + (-first_input,) * 9
    dx = tuple((-hi + gamma * si) / 9 + ui for hi, si, ui in zip(q, sine, inputs))
    phase_rate = tuple(gamma * hi / 9 for hi in q)
    return x, tuple(q), tuple(sine), inputs, dx, phase_rate


def _assess(**changes):
    arguments = dict(
        form_error_bounds=(ERROR, ERROR),
        phase_error_bounds=(ERROR, ERROR),
        target_gap_error_bound=TARGET_ERROR,
        readout_error_bound=NOISE,
    )
    arguments.update(changes)
    return owner._assess_mixed_form_phase_preparation(**arguments)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_prior_producer():
    static_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("phase preparation controls cannot advance or replay a source")

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
            static.setattr(flow_owner, "_sine_rate_evaluator", static_factory)
            yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 200
    return context


@pytest.fixture(scope="module")
def report():
    return _assess()


@pytest.fixture(scope="module")
def field():
    return owner._mixed_phase_bias_field(
        new_edge_conductances=_weights(), target_phase_gap=AMPLITUDE
    )


def test_complete_36_rows_and_both_clocks_keep_phase_live(mp, field):
    forms = tuple(Q(5, 1024) + Q((i % 7) - 3, 16384) for i in range(18))
    errors = tuple(Q((i % 5) - 2, 32768) for i in range(18))
    x, q, sine, inputs, dx, dy = _oracle(mp, forms, errors, AMPLITUDE)
    rates = field(_state(forms, errors))
    for interval, expected in zip(rates, dx + dy):
        _contains(mp, interval, expected)
    assert max(abs(value) for value in dy) > 0
    assert abs(sum(dx)) < mp.mpf("1e-190")
    assert abs(sum(dy)) < mp.mpf("1e-190")
    assert sum(forms) != 0 and abs(sum(inputs)) < mp.mpf("1e-190")
    clock = _mp(mp, Q(1023, 1024))
    original = tuple(
        -clock * hi / 9 + si / (1024 * mp.pi * 9) + clock * ui
        for hi, si, ui in zip(q, sine, inputs)
    ) + tuple(hi / (1024 * mp.pi * 9) for hi in q)
    assert max(abs(a / clock - b) for a, b in zip(original, dx + dy)) < mp.mpf("1e-190")
    translated = tuple(value + Q(3, 8) for value in forms)
    for interval, expected in zip(field(_state(translated, errors, Q(7, 16))), dx + dy):
        _contains(mp, interval, expected)
    assert _dot(x, x) > 0


def test_signed_target_equilibria_use_bare_sine_not_weighted_transport(mp):
    cross = 1 + 2 * mp.cos(mp.pi / 9)
    alternative = 1 + 2 * mp.cos(2 * mp.pi / 9) + 2 * mp.cos(4 * mp.pi / 9)
    assert abs(cross - alternative) < mp.mpf("1e-190")
    assert 2 < cross < 3
    for gap in (
        AMPLITUDE - TARGET_ERROR,
        AMPLITUDE + TARGET_ERROR,
        -AMPLITUDE + TARGET_ERROR,
        -AMPLITUDE - TARGET_ERROR,
    ):
        biased = owner._mixed_phase_bias_field(
            new_edge_conductances=_weights(), target_phase_gap=gap
        )
        forms, errors = (Q(5, 32),) * 18, _offset(gap)
        _, q, sine, inputs, dx, dy = _oracle(mp, forms, errors, gap)
        for i in range(18):
            expected_sine = (1 if i < 9 else -1) * cross * mp.sin(_mp(mp, gap))
            assert abs(sine[i] - expected_sine) < mp.mpf("1e-190")
            assert q[i] == 0 and dy[i] == 0
            assert abs(dx[i]) < mp.mpf("1e-190")
        for interval in biased(_state(forms, errors, Q(11, 32))):
            _contains(mp, interval, mp.mpf(0))
        # If the comparator were also used during preparation, this same
        # target would have nonzero form rates. Its switch is only at contact.
        assert all(value != 0 for value in inputs)
        assert max(abs(value) for value in inputs) > mp.mpf("1e-9")
        assert abs(sum(inputs)) < mp.mpf("1e-190")


def test_interval_and_jet_commands_add_no_phase_derivative(mp, field):
    forms = tuple(Q((i % 7) - 3, 16384) for i in range(18))
    errors = tuple(Q((i % 5) - 2, 32768) for i in range(18))
    state = _state(forms, errors)
    phase = tuple(2 * mp.pi * _mp(mp, t) + _mp(mp, y) for t, y in zip(_turns(), errors))
    gamma = 1 / (1023 * mp.pi)
    expected_constant = _oracle(mp, forms, errors, AMPLITUDE)[-2:]
    for coordinate in (4, 23):
        direction = tuple(Q(int(i == coordinate)) for i in range(36))
        jets = tuple(Jet((value, I(d))) for value, d in zip(state, direction))
        q, sine = [Q(0)] * 18, [mp.mpf(0)] * 18
        for a, b, weight in _edges():
            gradient = weight * (direction[a] - direction[b])
            current = mp.cos(phase[b] - phase[a]) * _mp(
                mp, direction[18 + b] - direction[18 + a]
            )
            q[a] += gradient
            q[b] -= gradient
            sine[a] += current
            sine[b] -= current
        derivative = tuple((-_mp(mp, hi) + gamma * si) / 9 for hi, si in zip(q, sine))
        derivative += tuple(gamma * _mp(mp, hi) / 9 for hi in q)
        for jet, constant, expected in zip(
            field(jets), sum(expected_constant, ()), derivative
        ):
            _contains(mp, jet.coeffs[0], constant)
            _contains(mp, jet.coeffs[1], expected)
        if coordinate == 4:
            assert max(abs(value) for value in derivative[18:]) > 0
        else:
            assert derivative[18:] == (mp.mpf(0),) * 18
    uncertain = list(state)
    uncertain[17] = I(forms[17] - Q(1, 10**20), forms[17] + Q(1, 10**20))
    rates = field(tuple(uncertain))
    for sign in (-1, 1):
        changed = forms[:17] + (forms[17] + sign * Q(1, 10**20),)
        oracle = _oracle(mp, changed, errors, AMPLITUDE)
        for interval, expected in zip(rates, oracle[-2] + oracle[-1]):
            assert _mp(mp, interval.lo) <= expected <= _mp(mp, interval.hi)


def test_nominal_regional_wedge_is_not_an_actual_fine_state_carrier(mp):
    weights = _nominal_weights(mp)
    gamma = 1 / (1023 * mp.pi)
    cross = 1 + 2 * mp.cos(mp.pi / 9)
    k = mp.mpf(13) / 27 - _mp(mp, DELTA) / (18 * mp.pi)
    assert _mp(mp, Q(4, 9)) < k < mp.mpf(1) / 2
    assert gamma**2 < 2 * k / 3
    for beta in (AMPLITUDE - TARGET_ERROR, AMPLITUDE + TARGET_ERROR):
        b = _mp(mp, beta)
        for fraction in (Q(0), Q(1, 3), Q(1)):
            q = b * _mp(mp, fraction)
            speed = gamma**2 * (b - q)
            z = -speed / (4 * gamma * k)
            forms = (z,) * 9 + (-z,) * 9
            _, _, _, _, dx, dy = _oracle(mp, forms, _offset(q), beta, weights)
            assert max(abs(value - dx[0]) for value in dx[:9]) < mp.mpf("1e-190")
            assert max(abs(value - dx[9]) for value in dx[9:]) < mp.mpf("1e-190")
            assert abs(dy[9] - dy[0] - speed) < mp.mpf("1e-190")
            acceleration = -4 * gamma * k * dx[0]
            expected = -2 * k * speed + 4 * gamma**2 * k * cross / 9 * (
                mp.sin(b) - mp.sin(q)
            )
            assert abs(acceleration - expected) < mp.mpf("1e-190")
            if fraction != 1:
                assert acceleration + gamma**2 * speed < 0
                assert 4 * gamma**2 * k * cross / 9 * (mp.sin(b) - mp.sin(q)) > 0
        assert gamma**2 * (-2 * k / 3 + gamma**2) < 0
    # A declared fine-weight box does not guarantee an exact transient
    # carrier. This constructed perturbation is not an acquired endpoint.
    new = _support()[1]
    edge = next(i for i, (a, b) in enumerate(new) if a < 9 <= b)
    perturbed = tuple(
        w + (_mp(mp, WEIGHT_ERROR) if i == edge else 0) for i, w in enumerate(weights)
    )
    forms = (Q(1, 4096),) * 9 + (-Q(1, 4096),) * 9
    rates = _oracle(mp, forms, _offset(AMPLITUDE / 2), AMPLITUDE, perturbed)
    a, _ = new[edge]
    other = next(i for i in range(9) if i != a)
    expected_difference = -_mp(mp, WEIGHT_ERROR) * _mp(mp, Q(2, 4096)) / 9
    assert abs(rates[-2][a] - rates[-2][other] - expected_difference) < mp.mpf("1e-190")
    assert abs(rates[-2][a] - rates[-2][other]) > mp.mpf("1e-70")


def test_shifted_metric_keeps_common_form_and_noncommuting_stiffness(mp, field):
    gap = AMPLITUDE
    z = _center(tuple(Q((i % 9) ** 2, 65536) for i in range(18)))
    errors = tuple(a + b for a, b in zip(_offset(gap), z))
    scale, gamma = _mp(mp, SCALE), 1 / (1023 * mp.pi)
    h = _laplacian(tuple((a, b, _mp(mp, w)) for a, b, w in _edges()))
    target = tuple(
        2 * mp.pi * _mp(mp, t) + _mp(mp, v) for t, v in zip(_turns(), _offset(gap))
    )
    secants = []
    for a, b, _ in _edges():
        base, delta = target[b] - target[a], _mp(mp, z[b] - z[a])
        coefficient = (
            (mp.sin(base + delta) - mp.sin(base)) / delta if delta else mp.cos(base)
        )
        secants.append((a, b, coefficient))
    k = _laplacian(secants)
    hk, kh = _product(h, k), _product(k, h)
    assert max(
        abs(a - b) for row1, row2 in zip(hk, kh) for a, b in zip(row1, row2)
    ) > mp.mpf("1e-6")
    for sign in (-1, 1):
        forms = tuple(Q(7, 128) + sign * value / 2 for value in z)
        x, q, sine, inputs, dx, dy = _oracle(mp, forms, errors, gap)
        centered = tuple(value - _mp(mp, Q(7, 128)) for value in x)
        zz = tuple(_mp(mp, value) for value in z)
        kz = _matvec(k, zz)
        assert max(
            abs(s + 9 * u / gamma + ki) for s, u, ki in zip(sine, inputs, kz)
        ) < mp.mpf("1e-190")
        assert sign * _dot(centered, zz) > 0
        assert abs(sum(dx)) < mp.mpf("1e-190")
        for interval, value in zip(field(_state(forms, errors)), dx + dy):
            _contains(mp, interval, value)
        metric = (
            mp.mpf(9)
            / 4
            * (
                _dot(centered, centered)
                + scale**2 * _dot(zz, zz)
                + 2 * scale**2 * gamma * _dot(centered, zz)
            )
        )
        derivative = (
            mp.mpf(9)
            / 2
            * (
                _dot(centered, dx)
                + scale**2 * _dot(zz, dy)
                + scale**2 * gamma * (_dot(dx, zz) + _dot(centered, dy))
            )
        )
        exact = (
            -(1 - scale**2 * gamma**2) * _dot(centered, q) / 2
            - gamma * _dot(centered, kz) / 2
            - scale**2 * gamma**2 * _dot(zz, kz) / 2
        )
        assert abs(derivative - exact) < mp.mpf("1e-190")
        rate = _mp(mp, Q(5) / (216 * 2750**2 * (1 + Q(1, 2500))))
        assert derivative < -rate * metric < 0
        for a, b, weight in _edges():
            assert -k[a][b] >= _mp(mp, Q(1, 10) * weight)
            assert -k[a][b] <= _mp(mp, Q(4, 3) * weight)


def test_weight_defect_forces_both_error_rows_without_feedback_fitting(mp):
    weights = _nominal_weights(mp)
    perturbations = tuple(((-1) ** i) * WEIGHT_ERROR for i in range(63))
    actual_weights = tuple(w + _mp(mp, d) for w, d in zip(weights, perturbations))
    reference_form = (Q(-1, 16384),) * 9 + (Q(1, 16384),) * 9
    reference_phase = _offset(AMPLITUDE / 2)
    ex = _center(tuple(Q((i % 7) - 3, 10**9) for i in range(18)))
    ey = _center(tuple(Q((i % 5) - 2, 10**9) for i in range(18)))
    actual_form = tuple(a + b for a, b in zip(reference_form, ex))
    actual_phase = tuple(a + b for a, b in zip(reference_phase, ey))
    actual = _oracle(mp, actual_form, actual_phase, AMPLITUDE, actual_weights)
    reference = _oracle(mp, reference_form, reference_phase, AMPLITUDE, weights)
    h = _laplacian(tuple((a, b, _mp(mp, w)) for a, b, w in _edges(actual_weights)))
    dh = _laplacian(
        tuple((a, b, _mp(mp, d)) for (a, b), d in zip(_support()[1], perturbations))
    )
    gamma = 1 / (1023 * mp.pi)
    hex_ = _matvec(h, tuple(_mp(mp, v) for v in ex))
    forcing = _matvec(dh, tuple(_mp(mp, v) for v in reference_form))
    assert max(abs(v) for v in forcing) > mp.mpf("1e-70")
    for i in range(18):
        ds = actual[2][i] - reference[2][i]
        expected_x = (-hex_[i] + gamma * ds - forcing[i]) / 9
        expected_y = gamma * (hex_[i] + forcing[i]) / 9
        assert abs(actual[-2][i] - reference[-2][i] - expected_x) < mp.mpf("1e-190")
        assert abs(actual[-1][i] - reference[-1][i] - expected_y) < mp.mpf("1e-190")
    # Exact edge-square and row-sum bounds apply to every perturbation sign.
    delta = _laplacian(
        tuple((a, b, d) for (a, b), d in zip(_support()[1], perturbations))
    )
    assert max(sum(abs(v) for v in row) for row in delta) <= 14 * WEIGHT_ERROR
    assert all(sum(row) == 0 for row in delta)
    for i in range(18):
        assert sum(i in edge for edge in _support()[1]) == 7


def test_continuous_work_has_a_primitive_but_absolute_input_is_not_free(mp, report):
    forms = tuple(Q(3, 64) + Q((i % 7) - 3, 8192) for i in range(18))
    errors = tuple(
        a + b
        for a, b in zip(
            _offset(AMPLITUDE / 2),
            _center(tuple(Q((i % 5) - 2, 16384) for i in range(18))),
        )
    )
    _, q, sine, inputs, dx, dy = _oracle(mp, forms, errors, AMPLITUDE)
    gamma = 1 / (1023 * mp.pi)
    power = _dot(q, inputs)
    original_storage_derivative = _dot(q, dx) - _dot(sine, dy)
    assert abs(original_storage_derivative + _dot(q, q) / 9 - power) < mp.mpf("1e-190")
    primitive_derivative = 9 * _dot(inputs, dy) / gamma
    assert abs(primitive_derivative - power) < mp.mpf("1e-190")
    target_sine = tuple(-9 * value / gamma for value in inputs)
    tilted_derivative = original_storage_derivative + _dot(target_sine, dy)
    assert abs(tilted_derivative + _dot(q, q) / 9) < mp.mpf("1e-190")
    # The primitive bounds signed net work only. A constant nonzero input
    # has a distinct positive L1 cost even when it holds an equilibrium.
    bmax = AMPLITUDE + TARGET_ERROR
    endpoint = Q(1, 2**30)
    for b1, b2 in (
        (AMPLITUDE - TARGET_ERROR, -bmax),
        (bmax, -AMPLITUDE + TARGET_ERROR),
    ):
        start = tuple(PREFIX * Q((i % 3) - 1, 12) for i in range(36))
        residual = tuple(endpoint * Q((i % 3) - 1, 12) for i in range(36))
        offset = _offset(b1) + _offset(b2)
        finish = tuple(a + b for a, b in zip(offset, residual))
        assert _dot(start, start) < PREFIX**2
        assert _dot(residual, residual) < endpoint**2
        commands = _oracle(mp, (Q(0),) * 18, _offset(b1), b1)[3]
        commands += _oracle(mp, (Q(0),) * 18, _offset(b2), b2)[3]
        displacement = tuple(_mp(mp, a - b) for a, b in zip(finish, start))
        net = 9 * _dot(commands, displacement) / gamma
        assert abs(net) < _mp(mp, report.bias_net_work_absolute_upper_bound)
        absolute_form_amount = _mp(mp, HOLD) * sum(abs(value) for value in commands)
        assert (
            0
            < absolute_form_amount
            < _mp(mp, report.bias_absolute_form_amount_upper_bound)
        )
    rate = Q(5) / (216 * 2750**2 * (1 + Q(1, 2500)))
    assert report.bias_node_amplitude_upper_bound == GAMMA * bmax / 3
    assert (
        report.bias_net_work_absolute_upper_bound
        == 18 * bmax * (3 * bmax + PREFIX + endpoint)
        < Q(1, 10**6)
    )
    # |q·u|<=18||X||*2g*bmax; ||X||<(3/2)/2048*exp(-rho*t/2).
    # Integrating the positive exponential to infinity gives this separate
    # absolute-work bound, without a simulated or fitted preparation trace.
    assert (
        report.bias_absolute_work_upper_bound == 108 * GAMMA * bmax / (2048 * rate) < 1
    )
    assert (
        report.bias_absolute_form_amount_upper_bound == 12 * HOLD * GAMMA * bmax < 5000
    )
    assert report.prefix_storage_work_upper_bound == 18 * Q(1, 32) * PREFIX**2
    assert report.contact_storage_work_upper_bound == 36 * 512 * Q(1, 1024) ** 2
    assert report.work_certified and not report.old_jump_work_included


def test_full_chart_registration_and_integer_endpoint_are_distinct(
    report, record_property
):
    bmax = AMPLITUDE + TARGET_ERROR
    lower, upper = Q(9, 4) * (1 - Q(1, 2500)), Q(9, 4) * (1 + Q(1, 2500))
    rate = Q(5) / (216 * 2750**2 * (1 + Q(1, 2500)))
    initial = PREFIX + 3 * bmax
    ratio = SCALE**2 * upper / lower
    common = Q(1, 32) * PREFIX
    assert initial < Q(1, 2048)
    assert ratio < 2 < Q(9, 4)
    assert report.metric_lower_bound == lower and report.metric_upper_bound == upper
    assert report.metric_decay_rate == rate
    assert report.physical_metric_ratio == ratio
    assert report.initial_centered_norm_upper_bound == initial
    assert report.centered_norm_upper_bound == Q(3, 2) * initial < Q(1, 1024)
    assert (
        report.original_norm_upper_bound
        == Q(3, 2) * initial + 3 * bmax + common
        < RADIUS
    )
    assert report.shifted_chart_margin == RADIUS - report.original_norm_upper_bound > 0
    assert (
        report.actual_reference_error_upper_bound
        == Q(3, 2) * (PREFIX + 2 * HOLD * WEIGHT_ERROR)
        < 2 * PREFIX
    )
    assert report.actual_reference_error_margin > 0
    assert report.weight_matrix_deviation_upper_bound == 14 * WEIGHT_ERROR
    assert report.nominal_form_norm_upper_bound == Q(27, 8) * GAMMA * bmax < Q(1, 1024)
    assert (
        report.nominal_wedge_margin
        == 2 * report.nominal_coupling_bounds[0] / 3 - GAMMA**2
        > 0
    )
    assert report.preparation_marked_gap_upper_bound == bmax + 3 * PREFIX < DELTA / 4
    assert report.preparation_registration_margin == DELTA / 4 - bmax - 3 * PREFIX > 0
    assert report.preparation_identity_certified and report.metric_domain_certified
    # Original post-leak mean is bounded by its mean projection, not reset.
    # For any x, ||P_mean P_A x||²=(sum_A x)²/18<=||P_A x||²/2.
    vector = tuple(Q((i % 5) - 2, 4096) for i in range(18))
    projected_square = sum(vector[:9]) ** 2 / 18
    assert projected_square <= _dot(vector[:9], vector[:9]) / 2
    assert report.carried_common_form_norm_upper_bound == common == 3 * MATURE / 64 > 0
    assert rate * HOLD > 30
    assert 1 + 1 + Q(1, 2) + Q(1, 6) > Q(5, 2)
    assert 5**10 > 2**23 and Q(5, 2) ** 30 > 2**39
    assert report.endpoint_prefactor == ratio * initial**2 < Q(1, 2**21)
    assert (
        report.centered_endpoint_error_squared_upper_bound
        == report.endpoint_prefactor * Q(1, 2**39)
        < Q(1, 2**60)
    )
    assert report.centered_endpoint_error_bound == Q(1, 2**30) == MATURE / 4
    assert report.centered_endpoint_error_bound**2 > 0
    assert report.endpoint_base_norm_upper_bound == Q(19, 64) * MATURE < PREFIX
    assert (
        report.endpoint_domain_margin
        == PREFIX - report.endpoint_base_norm_upper_bound
        > 0
    )
    assert report.endpoint_certified
    assert (
        report.original_clock_bias_duration * Q(1023, 1024)
        == report.bias_duration
        == HOLD
    )
    record_property("fixed_bias_duration", str(HOLD))
    record_property(
        "all_time_marked_gap_bound", str(report.preparation_marked_gap_upper_bound)
    )
    record_property(
        "positive_centered_endpoint_error", str(report.centered_endpoint_error_bound)
    )
    record_property(
        "full_endpoint_base_bound", str(report.endpoint_base_norm_upper_bound)
    )
    record_property(
        "absolute_net_work_bound", str(report.bias_net_work_absolute_upper_bound)
    )
    record_property("absolute_work_bound", str(report.bias_absolute_work_upper_bound))
    record_property(
        "supplied_absolute_form_amount",
        str(report.bias_absolute_form_amount_upper_bound),
    )


def test_endpoint_contact_domain_preserves_response_without_inventing_a_jump(
    report, record_property
):
    contact = report.contact_reference
    assert report.readiness.readiness_certified == (True, True)
    assert report.source_association_certified
    assert report.target_family_certified
    assert report.target_gap_bounds == (
        AMPLITUDE - TARGET_ERROR,
        AMPLITUDE + TARGET_ERROR,
    )
    assert report.prefix_charge_duration == report.prefix_leak_duration == Q(1, 32)
    assert report.prefix_charge_amplitude == 0
    assert (
        report.endpoint_base_norm_upper_bound
        < contact.prefix_norm_upper_bound
        == PREFIX
    )
    assert contact.contact_duration == 512 and contact.phase_jump_amplitude == AMPLITUDE
    assert contact.phase_jump_error_bound == TARGET_ERROR
    assert contact.readout_error_bound == contact.readout_error_cap == NOISE
    assert contact.contact_pairs == tuple((j, 18 + j) for j in range(18))
    assert contact.classifier_threshold == Q(1, 400000)
    assert contact.record_separation_margin >= Q(1503, 400000000) > 3 * NOISE
    assert min(contact.full_threshold_margin, contact.comparator_threshold_margin) > 0
    assert report.contact_domain_certified and report.contact_response_certified
    assert report.certified
    assert not report.phase_jump_performed and not report.exact_endpoint_claimed
    assert not report.common_form_reset and not report.comparator_prepares_source
    assert (
        not report.runtime_source_authenticated
        and not report.source_trajectory_evaluated
    )
    assert report.phase_coordinates_directly_actuated == 0
    assert report.actuated_form_coordinates == 36 and report.distinct_bias_commands == 2
    # Representation theta_end=base+zeta is algebraic. It supplies the same
    # contact domain, but does not claim the old actual jump history occurred.
    endpoint = report.centered_endpoint_error_bound
    x = (report.carried_common_form_norm_upper_bound / 6,) * 36
    residual = tuple(endpoint * Q((i % 3) - 1, 12) for i in range(36))
    assert _dot(x, x) <= report.carried_common_form_norm_upper_bound**2
    assert _dot(residual, residual) < endpoint**2
    assert _dot(x, x) + _dot(residual, residual) < PREFIX**2
    record_property(
        "independent_record_separation", str(contact.record_separation_margin)
    )
    record_property("classifier_threshold", str(contact.classifier_threshold))
    record_property(
        "endpoint_history", "continuous controlled preparation; no actual phase jump"
    )


def test_fresh_source_and_contact_association_cannot_be_replaced_by_success_flags(
    monkeypatch, report
):
    ready = report.readiness
    contact = ready.circulation
    # Irrelevant old jump-work and global summary verdicts are not consumed.
    ignored_contact = replace(
        contact,
        certified=False,
        work_certified=False,
        jump_work_margin=Q(-1),
        phase_jump_work_upper_bound=Q(100),
    )
    ignored = replace(
        ready,
        certified=False,
        composed_response_certified=False,
        circulation=ignored_contact,
    )
    monkeypatch.setattr(
        owner, "_assess_mixed_circulation_readiness", lambda **kwargs: ignored
    )
    independent = _assess()
    assert independent.certified
    assert (
        independent.bias_net_work_absolute_upper_bound,
        independent.bias_absolute_work_upper_bound,
        independent.bias_absolute_form_amount_upper_bound,
        independent.prefix_storage_work_upper_bound,
        independent.contact_storage_work_upper_bound,
    ) == (
        report.bias_net_work_absolute_upper_bound,
        report.bias_absolute_work_upper_bound,
        report.bias_absolute_form_amount_upper_bound,
        report.prefix_storage_work_upper_bound,
        report.contact_storage_work_upper_bound,
    )
    for field, value in (
        ("source_certified", (False, True)),
        ("all_time_source_identity_certified", (True, False)),
        ("readiness_certified", (False, True)),
        ("handoff_norm_upper_bounds", (MATURE, 2 * MATURE)),
        ("waiting_duration", HOLD - 1),
        (
            "returned_norm_squared_upper_bounds",
            (MATURE**2, ready.returned_norm_squared_upper_bounds[1]),
        ),
    ):
        altered = replace(ready, **{field: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_circulation_readiness", lambda **kwargs: altered
        )
        result = _assess()
        assert not result.source_association_certified and not result.certified
    for side, name, value in (
        (0, "edge_weight_deviation_upper_bound", 2 * WEIGHT_ERROR),
        (1, "form_error_bound", 2 * ERROR),
    ):
        changed = replace(ready.sources[side], **{name: value})
        altered = replace(
            ready,
            sources=tuple(changed if i == side else ready.sources[i] for i in range(2)),
        )
        assert altered.certified
        monkeypatch.setattr(
            owner, "_assess_mixed_circulation_readiness", lambda **kwargs: altered
        )
        result = _assess()
        assert not result.source_association_certified and not result.certified
    for name, value in (
        ("contact_duration", Q(511)),
        ("classifier_threshold", Q(1, 500000)),
        ("prefix_norm_upper_bound", 2 * PREFIX),
        ("readout_error_bound", 0),
    ):
        changed = replace(contact, **{name: value})
        altered = replace(ready, circulation=changed)
        monkeypatch.setattr(
            owner, "_assess_mixed_circulation_readiness", lambda **kwargs: altered
        )
        result = _assess()
        assert result.endpoint_certified and not result.contact_domain_certified
        assert not result.certified
    for side, name in (
        (0, "edge_weight_deviation_upper_bound"),
        (1, "gauge_phase_deviation_upper_bound"),
    ):
        source = contact.sources[side]
        changed = replace(source, **{name: 2 * getattr(source, name)})
        downstream = replace(
            contact,
            sources=tuple(
                changed if i == side else contact.sources[i] for i in range(2)
            ),
        )
        altered = replace(ready, circulation=downstream)
        assert altered.certified and downstream.certified
        monkeypatch.setattr(
            owner, "_assess_mixed_circulation_readiness", lambda **kwargs: altered
        )
        result = _assess()
        assert result.endpoint_certified and not result.contact_domain_certified
        assert not result.certified


def test_valid_outside_protocol_budgets_abstain_without_new_state_or_history():
    for name in ("form_error_bounds", "phase_error_bounds"):
        for side in (0, 1):
            values = tuple(
                ERROR + Q(1, 10**100) if i == side else ERROR for i in range(2)
            )
            unavailable = _assess(**{name: values})
            assert (
                not unavailable.source_association_certified
                and not unavailable.certified
            )
    target = _assess(target_gap_error_bound=TARGET_ERROR + Q(1, 10**100))
    assert target.source_association_certified
    assert not target.target_family_certified and not target.endpoint_certified
    assert not target.certified
    noise = _assess(readout_error_bound=NOISE + Q(1, 10**100))
    assert noise.endpoint_certified and noise.contact_domain_certified
    assert not noise.contact_response_certified and not noise.certified
    assert _assess(target_gap_error_bound=0, readout_error_bound=0).certified


def test_invalid_primitives_precede_readiness_geometry_and_metric(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid preparation primitive reached downstream work")

    for name in (
        "_assess_mixed_circulation_readiness",
        "_derive_registered_phase_probe_geometry",
        "_sine_common_quadratic_coefficients",
        "_bias_constants",
        "pi_interval",
    ):
        monkeypatch.setattr(owner, name, forbidden)
    for value in (
        True,
        np.bool_(False),
        "0",
        float("nan"),
        float("inf"),
        complex(0, 1),
        None,
    ):
        for side in (0, 1):
            pair = tuple(value if i == side else Q(0) for i in range(2))
            for name in ("form_error_bounds", "phase_error_bounds"):
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    _assess(**{name: pair})
        for name in ("target_gap_error_bound", "readout_error_bound"):
            with pytest.raises((TypeError, ValueError, OverflowError)):
                _assess(**{name: value})
    for name in ("form_error_bounds", "phase_error_bounds"):
        for values in (
            (0,),
            (0, 0, 0),
            repeat(Q(0)),
            {0, 1},
            {0: 0, 1: 0},
            (0, -Q(1, 10**100)),
        ):
            with pytest.raises((TypeError, ValueError)):
                _assess(**{name: values})
    for name in ("target_gap_error_bound", "readout_error_bound"):
        with pytest.raises(ValueError):
            _assess(**{name: -Q(1, 10**100)})


def test_bias_targets_weights_and_all_state_coordinates_are_admitted_first(
    monkeypatch, field
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid bias or state reached downstream arithmetic")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner, "_coupling_successor_field", forbidden)
        patch.setattr(owner, "_bias_constants", forbidden)
        for value in (
            True,
            np.bool_(False),
            "0",
            None,
            float("nan"),
            float("inf"),
            complex(0, 1),
            I(AMPLITUDE),
            Q(0),
            AMPLITUDE - TARGET_ERROR - Q(1, 10**100),
            -AMPLITUDE - TARGET_ERROR - Q(1, 10**100),
        ):
            with pytest.raises((TypeError, ValueError, OverflowError)):
                owner._mixed_phase_bias_field(
                    new_edge_conductances=_weights(), target_phase_gap=value
                )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner, "_bias_constants", forbidden)
        patch.setattr(source_owner, "_derive_coupling_successor_geometry", forbidden)
        for weights in (
            _weights()[:-1],
            _weights() + (Q(1),),
            repeat(Q(1)),
            {1},
            {0: 1},
            _weights()[:-1] + (True,),
            _weights()[:-1] + (float("nan"),),
            _weights()[:-1] + (Q(3, 4) - Q(1, 10**100),),
            _weights()[:-1] + (Q(1) + Q(1, 10**100),),
        ):
            with pytest.raises((TypeError, ValueError, OverflowError)):
                owner._mixed_phase_bias_field(
                    new_edge_conductances=weights, target_phase_gap=AMPLITUDE
                )
    state = _state((Q(0),) * 18, _offset(AMPLITUDE))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(flow_owner, "sin", forbidden)
        patch.setattr(flow_owner, "jet_sin", forbidden)
        bad = [state[:-1], state + (I(0),), repeat(I(0)), {0: 0}]
        bad += [
            state[:-1] + (value,) for value in (True, float("nan"), complex(0, 1), None)
        ]
        bad += [tuple(Jet((value, I(0))) for value in state[:-1]) + (I(0),)]
        bad += [tuple(Jet((value, I(0))) for value in state[:-1]) + (Jet((I(0),)),)]
        for values in bad:
            with pytest.raises((TypeError, ValueError, OverflowError)):
                field(values)


def test_one_shot_fixed_width_and_no_caller_endpoint_or_fitted_control(report, mp):
    def fixed(value):
        value = Q(value)
        return Q(np.int64(value.numerator), np.int64(value.denominator))

    normalized = _assess(
        form_error_bounds=(fixed(ERROR) for _ in range(2)),
        phase_error_bounds=(fixed(ERROR) for _ in range(2)),
        target_gap_error_bound=fixed(TARGET_ERROR),
        readout_error_bound=fixed(NOISE),
    )
    assert normalized == report
    field = owner._mixed_phase_bias_field(
        new_edge_conductances=(fixed(w) for w in _weights()),
        target_phase_gap=fixed(-AMPLITUDE),
    )
    forms = (Q(1, 32),) * 18
    expected = _oracle(mp, forms, _offset(-AMPLITUDE), -AMPLITUDE)
    for value, oracle in zip(
        field(v for v in _state(forms, _offset(-AMPLITUDE))),
        expected[-2] + expected[-1],
    ):
        _contains(mp, value, oracle)
    assert report.status == "conditional_form_preparation_and_circulation"
    with pytest.raises(FrozenInstanceError):
        report.phase_jump_performed = True
    for name in (
        "source",
        "readiness",
        "contact_reference",
        "initial_state",
        "endpoint",
        "mature_norm_upper_bounds",
        "bias_duration",
        "elapsed_time",
        "actual_transfer",
        "measured_phase",
        "bias_command",
        "phase_jump",
        "gamma",
        "capacity",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: 0})
