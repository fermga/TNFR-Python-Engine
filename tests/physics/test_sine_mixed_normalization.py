"""Same-source controls for two explicit dissipative normalizations.

The finite series below encloses a constant-initial-force analytical proxy.
Its separately proved live-phase remainder is retained; no nonlinear trajectory
or original preparation is evaluated. Generic row/energy points are law-level
controls, not assertions that those explicit points were reached by a source.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import repeat
from math import factorial

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _sine_flow as flow_owner
from tnfr.physics import _sine_mixed_normalization as owner

ERROR = Q(1, 10**10)
REGISTRATION_NOISE = Q(1, 10**6)
FORM_NOISE = Q(1, 10**13)
ETA = Q(2, 10**69)
DELTA = Q(1, 1024)
TIME = Q(1, 32)
TARGET = Q(5, 10**13)
WEIGHTS = (Q(1),) + (Q(5, 4),) * 8
LAWS = ("bare_degree", "transport_strength")


def _assess(**changes):
    values = dict(
        contact_conductances=WEIGHTS,
        form_error_bound=ERROR,
        phase_error_bound=ERROR,
        registration_error_bound=REGISTRATION_NOISE,
        form_readout_error_bound=FORM_NOISE,
    )
    values.update(changes)
    return owner._assess_sine_mixed_normalization(**values)


def _mp(context, value):
    value = Q(value)
    return context.mpf(value.numerator) / value.denominator


def _contains(context, interval, value):
    assert _mp(context, interval.lo) <= value <= _mp(context, interval.hi)
    assert _mp(context, interval.hi - interval.lo) < context.mpf("1e-30")


def _geometry(weights=WEIGHTS):
    permutation = tuple(
        next(k for k in range(9) if (2 * (k - 4) - (j - 4)) % 9 == 0) for j in range(9)
    )
    internal = tuple(
        (9 * r + j, 9 * r + (j + 1) % 9, Q(1)) for r in (0, 1) for j in range(9)
    )
    contacts = tuple((j, 9 + permutation[j], w) for j, w in enumerate(weights))
    return permutation, internal, contacts


def _state(context, forms=None, errors=None, gap=DELTA):
    forms = (Q(0),) * 18 if forms is None else forms
    errors = (Q(0),) * 18 if errors is None else errors
    phase = tuple(
        (1 + i // 9) * 2 * context.pi * (i % 9 - 4) / 9
        + (-1 if i < 9 else 1) * _mp(context, gap) / 2
        + _mp(context, errors[i])
        for i in range(18)
    )
    return tuple(_mp(context, x) for x in forms), phase


def _interval_state(forms, errors=None, gap=DELTA):
    errors = (Q(0),) * 18 if errors is None else errors
    phase = tuple(
        2 * pi_interval() * Q((1 + i // 9) * (i % 9 - 4), 9)
        + (-1 if i < 9 else 1) * gap / 2
        + errors[i]
        for i in range(18)
    )
    return tuple(I(value) for value in forms) + phase


def _edge_data(context, forms, phase, weights=WEIGHTS, joined=True):
    _, internal, contacts = _geometry(weights)
    edges = internal + (contacts if joined else ())
    gradient, sine, weighted_sine = (
        [context.mpf(0) for _ in range(18)] for _ in range(3)
    )
    strengths = [Q(0)] * 18
    energy = work = context.mpf(0)
    for i, j, raw_weight in edges:
        weight = _mp(context, raw_weight)
        difference = forms[i] - forms[j]
        gradient[i] += weight * difference
        gradient[j] -= weight * difference
        current = context.sin(phase[j] - phase[i])
        sine[i] += current
        sine[j] -= current
        weighted_sine[i] += weight * current
        weighted_sine[j] -= weight * current
        strengths[i] += raw_weight
        strengths[j] += raw_weight
        cost = weight * difference**2 / 2 + 1 - context.cos(phase[j] - phase[i])
        energy += cost
        if (i, j, raw_weight) in contacts:
            work += cost
    return (
        tuple(gradient),
        tuple(sine),
        tuple(weighted_sine),
        tuple(strengths),
        energy,
        work,
    )


def _rows(context, forms, phase, law, weights=WEIGHTS, joined=True):
    gradient, sine, _, strengths, *_ = _edge_data(
        context, forms, phase, weights, joined
    )
    gamma = 1 / (1023 * context.pi)
    degree = 3 if joined else 2
    denominators = strengths if law == "transport_strength" else (degree,) * 18
    dx = tuple(
        -g / _mp(context, d) + gamma * s / degree
        for g, s, d in zip(gradient, sine, denominators)
    )
    dt = tuple(gamma * g / degree for g in gradient)
    return dx, dt


def _matrix(edges, context=None):
    zero = Q(0) if context is None else context.mpf(0)
    result = [[zero for _ in range(18)] for _ in range(18)]
    for i, j, weight in edges:
        result[i][i] += weight
        result[j][j] += weight
        result[i][j] -= weight
        result[j][i] -= weight
    return tuple(tuple(row) for row in result)


def _apply(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def _multiply(left, right):
    return tuple(
        tuple(sum(a * b for a, b in zip(row, column)) for column in zip(*right))
        for row in left
    )


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _proxy(matrix, norm_bound):
    """Independent order-12 integral series and an absolute factorial tail."""
    power = (Q(1),) * 9 + (Q(-1),) * 9
    total = [Q(0)] * 18
    for n in range(13):
        factor = (-1) ** n * TIME ** (n + 1) / factorial(n + 1)
        total = [old + factor * new for old, new in zip(total, power)]
        power = _apply(matrix, power)
    tail = norm_bound**13 * TIME**14 / factorial(14) / (1 - norm_bound * TIME / 15)
    return tuple(total), tail


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_operator_or_source_replay():
    static_rate_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("normalization controls cannot evolve or replay a source")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(object_, name, forbidden)
        # Only static rows are admitted. Undo this local grant before restoring
        # the outer guards on trajectories, producers and source restoration.
        with pytest.MonkeyPatch.context() as static_patch:
            static_patch.setattr(
                flow_owner, "_sine_rate_evaluator", static_rate_factory
            )
            yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 200
    return context


@pytest.fixture(scope="module")
def report():
    return _assess()


def test_marked_strengths_change_only_loss_and_retain_actual_source(report):
    p, internal, contact = _geometry()
    expected = [Q(2)] * 18
    for j, weight in enumerate(WEIGHTS):
        expected[j] += weight
        expected[9 + p[j]] += weight
    assert report.contact_conductances == WEIGHTS
    assert report.strength_loss_mobilities == tuple(1 / value for value in expected)
    assert report.bare_degree_reference.weighted_edges == internal + contact
    assert report.bare_degree_reference.bare_degrees == (3,) * 18
    assert report.bare_degree_reference.conductance_strengths == tuple(expected)
    assert report.source_certified and report.isolated_waiting_certified
    assert tuple(
        item.target_winding
        for item in report.bare_degree_reference.precontact_reference.recoveries
    ) == (1, 2)
    assert report.waiting_radius == ETA
    assert report.clock == "tau=(1023/1024)*t; unchanged in every form and phase row"


def test_full_rows_match_independent_weighted_transport_and_bare_sine(mp):
    forms = tuple(
        Q((i % 9) - 4, 1000) + (Q(1, 100) if i < 9 else Q(-1, 100)) for i in range(18)
    )
    errors = tuple(Q((i % 9) - 4, 100000) for i in range(18))
    x, theta = _state(mp, forms, errors)
    gradient, sine, weighted_sine, strengths, *_ = _edge_data(mp, x, theta)
    gamma = 1 / (1023 * mp.pi)
    rows = {}
    for law in LAWS:
        field = owner._mixed_normalization_field(
            contact_conductances=WEIGHTS, loss_normalization=law
        )
        rows[law] = field(iter(_interval_state(forms, errors)))
        expected = sum(_rows(mp, x, theta, law), ())
        for interval, value in zip(rows[law], expected):
            _contains(mp, interval, value)
            _contains(mp, interval * Q(1023, 1024), value * 1023 / 1024)
    assert rows[LAWS[0]][18:] == rows[LAWS[1]][18:]
    for i in range(18):
        correction = (Q(1, 3) - 1 / strengths[i]) * _mp(mp, Q(1)) * gradient[i]
        _contains(mp, rows[LAWS[1]][i] - rows[LAWS[0]][i], correction)
    alternative = tuple(
        -g / _mp(mp, s) + gamma * a / 3
        for g, a, s in zip(gradient, weighted_sine, strengths)
    )
    true_strength = _rows(mp, x, theta, LAWS[1])[0]
    assert max(abs(a - b) for a, b in zip(true_strength, alternative)) > mp.mpf("1e-8")
    all_rows_strength_scaled = tuple(
        (-g + gamma * s) / _mp(mp, d) for g, s, d in zip(gradient, sine, strengths)
    )
    assert max(
        abs(a - b) for a, b in zip(true_strength, all_rows_strength_scaled)
    ) > mp.mpf("1e-9")


def test_strength_adapter_uses_complete_directed_shared_gradient(monkeypatch, mp):
    calls = []
    original = owner._laplacian

    def observed(edges, values):
        calls.append((edges, values))
        return original(edges, values)

    monkeypatch.setattr(owner, "_laplacian", observed)
    forms = tuple(Q(i - 8, 1000) for i in range(18))
    field = owner._mixed_normalization_field(
        contact_conductances=WEIGHTS, loss_normalization=LAWS[1]
    )
    state = _interval_state(forms)
    result = field(iter(state))
    _, internal, contact = _geometry()
    assert len(calls) == 1
    edges, values = calls[0]
    assert edges == tuple(
        row for a, b, w in internal + contact for row in ((a, b, w), (b, a, w))
    )
    assert values == state[:18]
    x, theta = _state(mp, forms)
    for interval, expected in zip(result, sum(_rows(mp, x, theta, LAWS[1]), ())):
        _contains(mp, interval, expected)


def test_jet_adapter_preserves_all_directional_form_and_phase_rows(mp):
    forms = tuple(Q(i - 8, 10000) for i in range(18))
    errors = tuple(Q((i % 9) - 4, 100000) for i in range(18))
    direction_x = tuple(Q((i % 5) - 2, 1000) for i in range(18))
    direction_phase = tuple(Q((i % 7) - 3, 10000) for i in range(18))
    state = _interval_state(forms, errors)
    jets = tuple(
        Jet((value, I(direction)))
        for value, direction in zip(state, direction_x + direction_phase)
    )
    x, theta = _state(mp, forms, errors)
    q = [mp.mpf(0)] * 18
    ds = [mp.mpf(0)] * 18
    _, internal, contact = _geometry()
    strengths = [Q(0)] * 18
    for a, b, w in internal + contact:
        flux = _mp(mp, w * (direction_x[a] - direction_x[b]))
        phase_flux = mp.cos(theta[b] - theta[a]) * _mp(
            mp, direction_phase[b] - direction_phase[a]
        )
        q[a] += flux
        q[b] -= flux
        ds[a] += phase_flux
        ds[b] -= phase_flux
        strengths[a] += w
        strengths[b] += w
    gamma = 1 / (1023 * mp.pi)
    for law in LAWS:
        result = owner._mixed_normalization_field(
            contact_conductances=WEIGHTS, loss_normalization=law
        )(jets)
        denominators = strengths if law == LAWS[1] else (Q(3),) * 18
        expected = tuple(
            -a / _mp(mp, d) + gamma * b / 3 for a, b, d in zip(q, ds, denominators)
        ) + tuple(gamma * a / 3 for a in q)
        for jet, constant, derivative in zip(
            result, sum(_rows(mp, x, theta, law), ()), expected
        ):
            assert isinstance(jet, Jet) and jet.order == 1
            _contains(mp, jet.coeffs[0], constant)
            _contains(mp, jet.coeffs[1], derivative)


def test_first_derivatives_agree_but_second_form_and_live_phase_do_not_vanish(mp):
    z = (Q(1),) * 9 + (Q(-1),) * 9
    state = _interval_state((Q(0),) * 18)
    directions = z + (Q(0),) * 18
    jets = tuple(
        Jet((value, I(direction))) for value, direction in zip(state, directions)
    )
    gamma = 1 / (1023 * mp.pi)
    forcing = gamma * mp.sin(_mp(mp, DELTA)) / 3
    _, internal, contact = _geometry()
    gradient = _apply(_matrix(internal + contact), z)
    _, _, _, strengths, *_ = _edge_data(mp, *_state(mp))
    derivatives = []
    for law in LAWS:
        result = owner._mixed_normalization_field(
            contact_conductances=WEIGHTS, loss_normalization=law
        )(jets)
        denominators = strengths if law == LAWS[1] else (Q(3),) * 18
        second_form = tuple(
            -forcing * _mp(mp, g / d) for g, d in zip(gradient, denominators)
        )
        derivatives.append(second_form)
        for i in range(18):
            _contains(mp, result[i].coeffs[0], forcing * z[i])
            _contains(mp, result[18 + i].coeffs[0], mp.mpf(0))
            _contains(mp, result[i].coeffs[1], -_mp(mp, gradient[i] / denominators[i]))
            _contains(mp, result[18 + i].coeffs[1], gamma * _mp(mp, gradient[i]) / 3)
        assert max(abs(forcing * gamma * _mp(mp, g) / 3) for g in gradient) > 0
    second_contrast = sum(derivatives[1][:9]) / 9 - sum(derivatives[0][:9]) / 9
    assert abs(second_contrast / (2 * forcing) - _mp(mp, Q(10, 351))) < mp.mpf("1e-190")
    assert second_contrast > 0


def test_full_storage_losses_share_exact_actual_attachment_work(mp, report):
    forms = tuple(ETA * Q((i % 9) - 4, 20) for i in range(18))
    errors = tuple(value * (1 if i < 9 else -1) for i, value in enumerate(forms))
    x, theta = _state(mp, forms, errors)
    q, sine, _, strengths, energy, work = _edge_data(mp, x, theta)
    cut_energy = _edge_data(mp, x, theta, joined=False)[-2]
    nominal = 9 * (1 - mp.cos(_mp(mp, DELTA)))
    assert abs(energy - cut_energy - work) < mp.mpf("1e-190")
    assert 0 < work - nominal < _mp(mp, Q(9, 2) * ETA**2)
    _contains(mp, report.bare_degree_reference.nominal_attachment_work_bounds, nominal)
    assert report.matched_attachment_work_certified
    for law in LAWS:
        dx, dt = _rows(mp, x, theta, law)
        derivative = _dot(q, dx) - _dot(sine, dt)
        denominators = strengths if law == LAWS[1] else (Q(3),) * 18
        expected = -sum(g**2 / _mp(mp, d) for g, d in zip(q, denominators))
        assert abs(derivative - expected) < mp.mpf("1e-190")
        assert derivative < 0
        cut = _rows(mp, x, theta, law, joined=False)
        other_cut = _rows(
            mp, x, theta, LAWS[1] if law == LAWS[0] else LAWS[0], joined=False
        )
        assert cut == other_cut


def test_strength_mean_drifts_while_mixed_inventory_and_phase_are_conserved(mp):
    forms = (Q(1, 1000),) + (Q(0),) * 17
    x, theta = _state(mp, forms)
    _, _, _, strengths, *_ = _edge_data(mp, x, theta)
    gamma = 1 / (1023 * mp.pi)
    degree = _rows(mp, x, theta, LAWS[0])
    strength = _rows(mp, x, theta, LAWS[1])
    assert abs(sum(degree[0])) < mp.mpf("1e-190")
    assert abs(sum(strength[0]) + _mp(mp, Q(2, 39000))) < mp.mpf("1e-190")
    assert abs(sum(strength[0])) > mp.mpf("1e-5")
    inventory_rate = sum(strength[0]) + 3 / gamma * sum(
        value / _mp(mp, s) for value, s in zip(strength[1], strengths)
    )
    assert abs(inventory_rate) < mp.mpf("1e-190")
    assert abs(sum(strength[1])) < mp.mpf("1e-190")
    assert degree[1] == strength[1]
    # A common form translates the limiting mode, without changing either row.
    shifted = tuple(value + _mp(mp, Q(7, 10)) for value in x)
    for law in LAWS:
        translated = _rows(mp, shifted, theta, law)
        original = _rows(mp, x, theta, law)
        assert max(
            abs(a - b) for a, b in zip(sum(translated, ()), sum(original, ()))
        ) < mp.mpf("1e-190")


def test_nonzero_strength_limit_is_a_quotient_statement(mp, report):
    errors = [Q(0)] * 18
    errors[0], errors[1] = ETA / 4, -ETA / 4
    _, theta = _state(mp, errors=errors)
    _, target = _state(mp, gap=Q(0))
    _, _, _, strengths, *_ = _edge_data(mp, (mp.mpf(0),) * 18, theta)
    gamma = 1 / (1023 * mp.pi)
    invariant = (
        3
        / gamma
        * sum((p - t) / _mp(mp, s) for p, t, s in zip(theta, target, strengths))
    )
    limit = invariant / 18
    assert limit > 0
    assert abs(limit - _mp(mp, ETA / 4) / (234 * gamma)) < mp.mpf("1e-185")
    assert abs(limit) <= _mp(mp, ETA) / (33 * gamma)
    assert report.strength_common_form_limit_bounds == (-100 * ETA, 100 * ETA)
    for law in LAWS:
        dx, dt = _rows(mp, (limit,) * 18, target, law)
        assert max(abs(value) for value in dx + dt) < mp.mpf("1e-190")
    assert report.strength_inventory == "sum(x)+(3/gamma)*sum(B*(theta-Theta))"


def test_modified_energy_derivative_uses_projected_rows_without_commutation(mp):
    forms = tuple(Q((i % 9) - 4, 10000) + Q(1, 1000) for i in range(18))
    errors = tuple(
        Q((i % 9) ** 2 - Q(68, 3), 100000) * (1 if i < 9 else -1) for i in range(18)
    )
    x, theta = _state(mp, forms, errors, gap=Q(0))
    _, target = _state(mp, gap=Q(0))
    v = tuple(value - sum(x) / 18 for value in x)
    y = tuple(a - b for a, b in zip(theta, target))
    q, sine, _, strengths, energy, _ = _edge_data(mp, x, theta)
    dx, dy = _rows(mp, x, theta, LAWS[1])
    dv = tuple(value - sum(dx) / 18 for value in dx)
    gamma = 1 / (1023 * mp.pi)
    _, internal, contacts = _geometry()
    weighted_laplacian = _matrix(
        tuple((i, j, _mp(mp, w)) for i, j, w in internal + contacts), mp
    )
    secants = []
    for i, j, _ in internal + contacts:
        d = y[j] - y[i]
        base = target[j] - target[i]
        coefficient = (mp.sin(base + d) - mp.sin(base)) / d if d else mp.cos(base)
        assert coefficient > mp.mpf("0.05")
        secants.append((i, j, coefficient))
    k = _matrix(tuple(secants), mp)
    ky = _apply(k, y)
    assert max(abs(a + b) for a, b in zip(ky, sine)) < mp.mpf("1e-190")
    hk, kh = _multiply(weighted_laplacian, k), _multiply(k, weighted_laplacian)
    assert max(abs(hk[i][j] - kh[i][j]) for i in range(18) for j in range(18)) > mp.mpf(
        "1e-3"
    )
    eprime = _dot(q, dx) - _dot(sine, dy)
    wprime = (
        eprime
        + gamma * (_dot(dv, y) + _dot(v, dy)) / 3
        + sum(a * b / _mp(mp, s) for a, b, s in zip(y, dy, strengths))
    )
    v_h_v, y_k_y = _dot(v, q), _dot(y, ky)
    identity = (
        -sum(value**2 / _mp(mp, s) for value, s in zip(q, strengths))
        + gamma**2 * (v_h_v - y_k_y) / 9
    )
    assert abs(wprime - identity) < mp.mpf("1e-190")
    assert wprime < -_mp(mp, Q(9, 100)) * v_h_v - _mp(mp, Q(1, 3300**2)) * y_k_y / 9 < 0
    baseline = _edge_data(mp, (mp.mpf(0),) * 18, target)[-2]
    excess = energy - baseline
    metric = (
        excess
        + gamma * _dot(v, y) / 3
        + sum(a**2 / _mp(mp, s) for a, s in zip(y, strengths)) / 2
    )
    joint_norm = _dot(v, v) + _dot(y, y)
    assert metric >= _mp(mp, Q(5997, 40000)) * joint_norm
    assert metric <= _mp(mp, 1 + Q(1, 2000)) * (v_h_v / 2 + _mp(mp, Q(210, 11)) * y_k_y)


def test_quotient_trap_and_finite_return_keep_same_late_registration(
    report, record_property
):
    initial = Q(9, 2) * DELTA**2 + 4 * ETA**2
    lower = Q(5997, 40000)
    rate = 11 * Q(1, 3300**2) / (1890 * (1 + Q(1, 2000)))
    returned = Q(7, 2) * initial / (lower * 2**14)
    assert report.initial_quotient_norm_squared_upper_bound == initial
    assert report.strength_post_attachment_excess_upper_bound == Q(13, 4) * initial
    assert report.strength_storage_margin == Q(1, 14400) - Q(13, 4) * initial > 0
    assert report.strength_radius_margin == Q(1, 12) ** 2 - initial > 0
    assert report.strength_metric_lower_bound == lower
    assert report.strength_initial_metric_upper_bound == Q(7, 2) * initial
    assert report.strength_decay_rate == rate
    assert report.strength_recovery_exponent_lower_bound == rate * 2 * 10**10 > 10
    assert sum(Q(1, factorial(n)) for n in range(6)) > Q(27, 10)
    assert Q(27, 10) ** 10 > 2**14
    assert report.strength_decay_upper_bound == Q(1, 2**14)
    assert (
        report.strength_returned_quotient_norm_squared_upper_bound
        == returned
        < DELTA**2 / 128
    )
    assert (
        report.strength_returned_registered_gap_squared_upper_bound
        == 2 * returned
        < (DELTA / 8) ** 2
    )
    assert (
        report.strength_joint_return_margin > 0
        and report.strength_registration_margin > 0
    )
    assert report.initial_identity_absent and report.both_joint_identities_certified
    assert report.registration_actual_contrast_bounds == (7 * DELTA / 8, 9 * DELTA / 8)
    bounds = (
        7 * DELTA / 8 - 2 * REGISTRATION_NOISE,
        9 * DELTA / 8 + 2 * REGISTRATION_NOISE,
    )
    assert report.registration_recorded_contrast_bounds == bounds
    assert report.registration_contrast_margin == bounds[0] - DELTA / 2 > 0
    assert report.registration_response_certified
    record_property("strength_recovery_exponent_lower_bound", str(rate * 2 * 10**10))
    record_property("strength_returned_quotient_ratio", str(returned / DELTA**2))
    record_property("recorded_registration_contrast_lower_bound", str(bounds[0]))


def test_independent_finite_markov_proxy_encloses_fixed_early_coefficient(
    mp, report, record_property
):
    _, internal, contacts = _geometry()
    laplacian = _matrix(internal + contacts)
    strengths = tuple(laplacian[i][i] for i in range(18))
    degree = tuple(tuple(value / 3 for value in row) for row in laplacian)
    strength = tuple(
        tuple(value / s for value in row) for row, s in zip(laplacian, strengths)
    )
    z = (Q(1),) * 9 + (Q(-1),) * 9
    for matrix, norm, first, second in (
        (degree, Q(13, 6), Q(5, 6), Q(65, 36)),
        (strength, Q(2), Q(10, 13), Q(20, 13)),
    ):
        assert all(sum(row) == 0 for row in matrix)
        assert all(matrix[i][j] <= 0 for i in range(18) for j in range(18) if i != j)
        assert max(sum(abs(value) for value in row) for row in matrix) <= norm
        assert max(map(abs, _apply(matrix, z))) <= first
        assert max(map(abs, _apply(matrix, _apply(matrix, z)))) <= second
    coefficient = sum(w / 3 - w / (2 + w) for w in WEIGHTS) / 9
    second_derivative = (
        sum(_apply(degree, z)[:9]) / 9 - sum(_apply(strength, z)[:9]) / 9
    )
    assert coefficient == second_derivative / 2 == Q(10, 351)
    assert report.early_quadratic_coefficient == coefficient
    d, dtail = _proxy(degree, Q(13, 6))
    t, ttail = _proxy(strength, Q(2))
    center, tail = sum(t[:9]) / 9 - sum(d[:9]) / 9, dtail + ttail
    coarse = (Q(65, 36) + Q(20, 13)) * TIME**3 / 6
    assert coefficient * TIME**2 - coarse <= center - tail
    assert center + tail <= coefficient * TIME**2 + coarse
    assert center - tail > 0
    forcing = mp.sin(_mp(mp, DELTA)) / (3069 * mp.pi)
    proxy_low = forcing * _mp(mp, center - tail)
    proxy_high = forcing * _mp(mp, center + tail)
    nominal_low, nominal_high = report.early_nominal_contrast_bounds
    assert _mp(mp, nominal_low) < proxy_low <= proxy_high < _mp(mp, nominal_high)
    record_property("constant_force_proxy_series_order", "12")
    record_property("constant_force_proxy_contrast_lower", mp.nstr(proxy_low, 24))
    record_property("constant_force_proxy_contrast_upper", mp.nstr(proxy_high, 24))


def test_live_phase_source_and_noise_bounds_keep_predeclared_positive_margin(
    mp, report, record_property
):
    forcing = mp.sin(_mp(mp, DELTA)) / (3069 * mp.pi)
    _contains(mp, report.early_forcing_bounds, forcing)
    assert forcing > _mp(mp, Q(1, 10240000))
    constant = (Q(65, 36) + Q(20, 13)) / 6
    feedback = Q(26, 9 * 3000**2)
    assert (
        report.early_constant_force_remainder_coefficient == constant == Q(1565, 2808)
    )
    assert report.early_phase_feedback_remainder_coefficient == feedback
    assert Q(10, 351) > Q(1, 36) and constant + feedback < Q(3, 5)
    assert Q(13, 9 * 3000**2) * TIME**2 < 1
    assert Q(13, 6) + 2 * Q(1, 3000) < 3
    assert 1 / (1 - 3 * TIME) < 2
    coarse_lower = Q(1, 10240000) * TIME**2 * (Q(1, 36) - TIME * Q(3, 5))
    assert coarse_lower == Q(13, 15099494400000) > Q(8, 10**13)
    assert coarse_lower - 4 * ETA - 2 * FORM_NOISE > TARGET
    nominal = report.early_nominal_contrast_bounds
    assert nominal[0] > coarse_lower
    assert report.early_source_error_upper_bound == 4 * ETA
    actual = (nominal[0] - 4 * ETA, nominal[1] + 4 * ETA)
    recorded = (actual[0] - 2 * FORM_NOISE, actual[1] + 2 * FORM_NOISE)
    assert report.early_actual_contrast_bounds == actual
    assert report.early_recorded_contrast_bounds == recorded
    assert report.early_contrast_margin == recorded[0] - TARGET > 0
    assert report.early_time == TIME and report.early_response_target == TARGET
    assert report.early_response_certified and not report.early_null_certified
    assert (
        report.certified and report.status == "certified_identity_and_nonzero_response"
    )
    assert not report.unresolved_conditions
    record_property("recorded_early_form_contrast_lower_bound", str(recorded[0]))
    record_property(
        "recorded_early_form_contrast_lower_display", mp.nstr(_mp(mp, recorded[0]), 24)
    )
    record_property("early_form_target", str(TARGET))
    record_property("early_actual_source_correction", str(4 * ETA))


def test_unit_same_source_null_is_exact_for_both_complete_rows(mp):
    weights = (Q(1),) * 9
    report = _assess(contact_conductances=weights)
    assert report.exact_law_equivalence and report.early_null_certified
    assert not report.early_response_certified
    assert (
        report.early_nominal_contrast_bounds
        == report.early_actual_contrast_bounds
        == (Q(0), Q(0))
    )
    assert report.early_source_error_upper_bound == 0
    assert report.early_recorded_contrast_bounds == (-2 * FORM_NOISE, 2 * FORM_NOISE)
    assert report.early_contrast_margin == TARGET - 2 * FORM_NOISE > 0
    assert report.certified and report.status == "certified_identity_and_exact_null"
    forms = tuple(ETA * Q((i % 9) - 4, 20) for i in range(18))
    errors = tuple(value * (1 if i < 9 else -1) for i, value in enumerate(forms))
    states = (
        _interval_state(forms, errors),
        tuple(
            Jet((value, I(Q((i % 3) - 1, 1000))))
            for i, value in enumerate(_interval_state(forms, errors))
        ),
    )
    for state in states:
        first, second = tuple(
            owner._mixed_normalization_field(
                contact_conductances=weights, loss_normalization=law
            )(state)
            for law in LAWS
        )
        assert first == second
    x, theta = _state(mp, forms, errors)
    assert _rows(mp, x, theta, LAWS[0], weights) == _rows(
        mp, x, theta, LAWS[1], weights
    )


def test_other_positive_profiles_retain_identity_without_claiming_selected_response():
    for weights in ((Q(3, 4),) * 9, (Q(5, 4),) * 9, tuple(reversed(WEIGHTS))):
        report = _assess(contact_conductances=weights)
        assert (
            report.both_joint_identities_certified
            and report.registration_response_certified
        )
        assert not report.early_profile_supported and not report.certified
        assert not report.exact_law_equivalence
        assert report.status == "identity_certified_discriminator_unsupported"
        for field in (
            "early_forcing_bounds",
            "early_quadratic_coefficient",
            "early_constant_force_remainder_coefficient",
            "early_phase_feedback_remainder_coefficient",
            "early_nominal_contrast_bounds",
            "early_source_error_upper_bound",
            "early_actual_contrast_bounds",
            "early_recorded_contrast_bounds",
            "early_contrast_margin",
        ):
            assert getattr(report, field) is None


def test_registration_and_form_readout_allowances_are_independent(report):
    excess = Q(1, 10**100)
    bad_form = _assess(form_readout_error_bound=FORM_NOISE + excess)
    assert (
        bad_form.both_joint_identities_certified
        and bad_form.registration_response_certified
    )
    assert not bad_form.early_response_certified and not bad_form.certified
    assert (
        bad_form.registration_recorded_contrast_bounds
        == report.registration_recorded_contrast_bounds
    )
    bad_registration = _assess(registration_error_bound=REGISTRATION_NOISE + excess)
    assert (
        bad_registration.both_joint_identities_certified
        and bad_registration.early_response_certified
    )
    assert (
        not bad_registration.registration_response_certified
        and not bad_registration.certified
    )
    assert (
        bad_registration.early_recorded_contrast_bounds
        == report.early_recorded_contrast_bounds
    )
    exact_reading = _assess(form_readout_error_bound=0, registration_error_bound=0)
    assert (
        exact_reading.early_recorded_contrast_bounds
        == report.early_actual_contrast_bounds
    )
    assert (
        exact_reading.registration_recorded_contrast_bounds
        == report.registration_actual_contrast_bounds
    )


def test_strength_proof_is_not_a_transfer_of_degree_return_or_cached_metric(
    monkeypatch, report
):
    original = owner._assess_sine_mixed_conductance

    def failed_degree_return(**kwargs):
        return replace(
            original(**kwargs),
            joint_identity_certified=False,
            certified=False,
            returned_joint_norm_squared_upper_bound=Q(10**100),
            status="unavailable",
        )

    monkeypatch.setattr(owner, "_assess_sine_mixed_conductance", failed_degree_return)
    result = _assess()
    assert result.strength_joint_identity_certified
    assert (
        result.strength_returned_quotient_norm_squared_upper_bound
        == report.strength_returned_quotient_norm_squared_upper_bound
    )
    assert not result.both_joint_identities_certified and not result.certified
    assert result.early_response_certified


def test_source_waiting_and_fresh_geometry_are_required_before_claims(monkeypatch):
    original = owner._assess_sine_mixed_conductance

    def corrupt_geometry(**kwargs):
        reference = original(**kwargs)
        return replace(
            reference,
            precontact_reference=replace(
                reference.precontact_reference,
                geometry=replace(
                    reference.precontact_reference.geometry,
                    matching_permutation=tuple(range(9)),
                ),
            ),
        )

    replacements = (
        lambda **kwargs: replace(original(**kwargs), source_certified=False),
        lambda **kwargs: replace(original(**kwargs), isolated_waiting_certified=False),
        lambda **kwargs: replace(original(**kwargs), waiting_radius=2 * ETA),
        corrupt_geometry,
    )
    for replacement in replacements:
        with monkeypatch.context() as patch:
            patch.setattr(owner, "_assess_sine_mixed_conductance", replacement)
            report = _assess()
            assert not report.strength_joint_identity_certified
            assert not report.early_response_certified and not report.certified
            assert report.status == "unavailable" and report.unresolved_conditions
    for field in ("form_error_bound", "phase_error_bound"):
        report = _assess(**{field: ERROR + Q(1, 10**100)})
        assert not report.source_certified and not report.certified
    unavailable_null = _assess(
        contact_conductances=(Q(1),) * 9, form_error_bound=2 * ERROR
    )
    assert unavailable_null.exact_law_equivalence
    assert unavailable_null.early_actual_contrast_bounds == (Q(0), Q(0))
    assert not unavailable_null.early_null_certified


def test_nonnegative_outside_contact_box_is_unavailable_and_field_rejects():
    for weights in (
        (Q(0),) * 9,
        (Q(3, 4) - Q(1, 10**100),) + (Q(1),) * 8,
        (Q(5, 4) + Q(1, 10**100),) + (Q(1),) * 8,
    ):
        report = _assess(contact_conductances=weights)
        assert not report.strength_constituent_identity_certified
        assert not report.both_joint_identities_certified and not report.certified
        assert report.status == "unavailable"
        for law in LAWS:
            with pytest.raises(ValueError):
                owner._mixed_normalization_field(
                    contact_conductances=weights, loss_normalization=law
                )


def test_malformed_inputs_reject_before_geometry_sources_fields_or_interval_work(
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached source, geometry or numerical work")

    for name in (
        "_assess_sine_mixed_conductance",
        "_derive_mixed_constitutive_pair",
        "_mixed_conductance_field",
        "_laplacian",
        "sin",
    ):
        monkeypatch.setattr(owner, name, forbidden)
    invalid_weights = (
        (),
        WEIGHTS[:-1],
        WEIGHTS + (Q(1),),
        repeat(Q(1)),
        {1: 2},
        "123456789",
        (True,) + (Q(1),) * 8,
        (np.bool_(False),) + (Q(1),) * 8,
        (Q(-1),) + (Q(1),) * 8,
        (float("nan"),) + (Q(1),) * 8,
        (float("inf"),) + (Q(1),) * 8,
        (1j,) + (Q(1),) * 8,
    )
    for value in invalid_weights:
        with pytest.raises((TypeError, ValueError, OverflowError)):
            _assess(contact_conductances=value)
        for law in LAWS:
            with pytest.raises((TypeError, ValueError, OverflowError)):
                owner._mixed_normalization_field(
                    contact_conductances=value, loss_normalization=law
                )
    for field in (
        "form_error_bound",
        "phase_error_bound",
        "registration_error_bound",
        "form_readout_error_bound",
    ):
        for value in (
            True,
            np.bool_(False),
            Q(-1),
            float("nan"),
            float("inf"),
            None,
            1j,
        ):
            with pytest.raises((TypeError, ValueError, OverflowError)):
                _assess(**{field: value})
    for label in (
        None,
        True,
        b"bare_degree",
        "BARE_DEGREE",
        "bare_degree ",
        "transport_ѕtrength",
        "native",
    ):
        with pytest.raises((TypeError, ValueError)):
            owner._mixed_normalization_field(
                contact_conductances=WEIGHTS, loss_normalization=label
            )


def test_state_admission_precedes_added_gradient_and_common_jet_order(monkeypatch):
    field = owner._mixed_normalization_field(
        contact_conductances=WEIGHTS, loss_normalization=LAWS[1]
    )

    def forbidden(*args, **kwargs):
        pytest.fail("invalid state reached the added shared gradient")

    monkeypatch.setattr(owner, "_laplacian", forbidden)
    invalid = (
        (0,) * 35,
        (0,) * 37,
        repeat(Q(0)),
        (True,) + (0,) * 35,
        (float("nan"),) + (0,) * 35,
        (Jet.constant(0, 1),) + (Q(0),) * 35,
        (Jet.constant(0, 1),) + (Jet.constant(0, 2),) * 35,
    )
    for state in invalid:
        with pytest.raises((TypeError, ValueError, OverflowError)):
            field(state)


def test_one_shot_fixed_width_inputs_and_no_cached_report_or_clock_override(report):
    normalized = _assess(
        contact_conductances=(
            Q(np.int64(w.numerator), np.int64(w.denominator)) for w in WEIGHTS
        ),
        form_error_bound=Q(np.int64(1), np.int64(10**10)),
        phase_error_bound=Q(np.int64(1), np.int64(10**10)),
        registration_error_bound=Q(np.int64(1), np.int64(10**6)),
        form_readout_error_bound=Q(np.int64(1), np.int64(10**13)),
    )
    assert normalized == report
    with pytest.raises(FrozenInstanceError):
        report.certified = False
    for extra in (
        dict(bare_degree_reference=report.bare_degree_reference),
        dict(recovery_duration=Q(2 * 10**10)),
        dict(early_time=TIME),
        dict(gamma=Q(1, 3000)),
        dict(capacity=(Q(1),) * 18),
    ):
        with pytest.raises(TypeError):
            _assess(**extra)
    ordinary = tuple(Q(i - 8, 1000) for i in range(36))
    fixed = tuple(
        Q(np.int64(value.numerator), np.int64(value.denominator)) for value in ordinary
    )
    for law in LAWS:
        field = owner._mixed_normalization_field(
            contact_conductances=(w for w in WEIGHTS), loss_normalization=law
        )
        assert field(iter(fixed)) == field(ordinary)
