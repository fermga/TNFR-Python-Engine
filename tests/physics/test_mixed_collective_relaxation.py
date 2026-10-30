"""Projected memory controls the aggregate without closing the fine state.

Independent static edge sums test the actual law and its discarded residuals.
Kernel and tangent identities use algebra, not trajectories, eigenvalue solvers
or exponential evaluations. Constructed states are not acquired endpoints.
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
from tnfr.physics import _mixed_collective_relaxation as owner
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
        full_observation_error_bound=Q(1, 10**9),
        form_only_observation_error_bound=Q(1, 10**9),
    )
    values.update(changes)
    return owner._assess_mixed_collective_relaxation(**values)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_prior_producer():
    factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("collective controls cannot advance or replay a source")

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


HORIZON = Q(64)
NOISE = Q(1, 10**9)
THRESHOLD = Q(1, 10**8)
COMPARISON = 200 * MATURE
CUT_ERROR = (
    8 * MATURE / (1 + Q(10000, 3)) + 6 * GAMMA_HI * 8 * MATURE + 6 * WEIGHT_ERROR
)


def _means(values):
    return tuple(sum(values[9 * j : 9 * (j + 1)]) / 9 for j in range(4))


def _regional(values):
    means = _means(values)
    return tuple((means[2 * p] + means[2 * p + 1]) / 2 for p in (0, 1)), tuple(
        (means[2 * p + 1] - means[2 * p]) / 2 for p in (0, 1)
    )


def _poly_add(a, b):
    return tuple(
        (a[j] if j < len(a) else 0) + (b[j] if j < len(b) else 0)
        for j in range(max(len(a), len(b)))
    )


def _poly_mul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for j, x in enumerate(a):
        for k, y in enumerate(b):
            out[j + k] += x * y
    return tuple(out)


def test_full_actual_projection_retains_weight_and_phase_defects_in_both_rows(mp):
    forms = _center(tuple(Q((j * 7) % 13 - 6, 10**5) for j in range(36)))
    hidden = tuple(
        value
        for p in range(4)
        for value in _center(tuple(Q((j * 3 + p) % 11 - 5, 10**6) for j in range(9)))
    )
    base = _offset(SECTION / 2) + _offset(-SECTION / 3)
    phases = tuple(a + b for a, b in zip(base, hidden, strict=True))
    x, y = tuple(_mp(mp, v) for v in forms), tuple(_mp(mp, v) for v in phases)
    actual = (_actual_weights(mp, 0), _actual_weights(mp, 1))
    nominal = (_nominal_weights(mp),) * 2
    gamma = 1 / (1023 * mp.pi)
    k = mp.mpf(13) / 27 - _mp(mp, DELTA) / (18 * mp.pi)
    coefficient = 1 + 2 * mp.cos(mp.pi / 9)
    dx, dy = _field(mp, forms, phases, actual)
    means, velocities = _regional(x)
    _, half_gaps = _regional(y)
    gaps = tuple(2 * v for v in half_gaps)
    md, vd = _regional(dx)
    _, half_qd = _regional(dy)
    delta_hx = tuple(
        a - b
        for a, b in zip(_h_apply(mp, x, actual), _h_apply(mp, x, nominal), strict=True)
    )
    _, half_weight = _regional(delta_hx)
    rw = tuple(-v / 9 for v in half_weight)
    _, projected_sine = _regional(_sine(mp, phases))
    rs = tuple(
        gamma * (v + coefficient * mp.sin(q)) / 9
        for v, q in zip(projected_sine, gaps, strict=True)
    )
    a, r = means[0] - means[1], velocities[0] - velocities[1]
    assert abs(md[0] - md[1] + a - r) < mp.mpf("1e-220")
    for p in (0, 1):
        expected = (
            -2 * k * velocities[p]
            - gamma * coefficient * mp.sin(gaps[p]) / 9
            + (1 if p == 0 else -1) * (a - r) / 2
            + rw[p]
            + rs[p]
        )
        assert abs(vd[p] - expected) < mp.mpf("1e-220")
        assert abs(
            2 * half_qd[p] - (4 * gamma * k * velocities[p] - 2 * gamma * rw[p])
        ) < mp.mpf("1e-220")
    assert max(abs(v) for v in rw) > mp.mpf("1e-80")
    assert max(abs(v) for v in rs) > mp.mpf("1e-30")
    u, b = -a, -r
    bd = -(vd[0] - vd[1])
    expected_b = (
        u
        - (2 * k + 1) * b
        + gamma * coefficient * (mp.sin(gaps[0]) - mp.sin(gaps[1])) / 9
        - rw[0]
        + rw[1]
        - rs[0]
        + rs[1]
    )
    assert abs(bd - expected_b) < mp.mpf("1e-220")
    state = tuple(I(v) for v in forms) + tuple(
        2 * pi_interval() * turn + I(v)
        for turn, v in zip(_turns() * 2, phases, strict=True)
    )
    field = field_owner._mixed_reciprocal_partner_field(
        first_new_edge_conductances=actual[0], second_new_edge_conductances=actual[1]
    )
    for interval, value in zip(field(state), dx + dy, strict=True):
        assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def test_hidden_phase_linear_cancellation_is_exact_but_common_forms_do_not_close(mp):
    q = _mp(mp, SECTION / 2)
    base = _offset(q) + (mp.mpf(0),) * 18
    eta = tuple(
        _mp(mp, v)
        for p in (0, 1)
        for v in _center(tuple(Q((j + 2 * p) % 5 - 2, 10**5) for j in range(9)))
    )
    theta = _phase(mp, base)
    linear = remainder = square = mp.mpf(0)
    for a, b, _ in _edges((Q(1),) * 63):
        if a < 9 <= b:
            angle, shift = theta[b] - theta[a], eta[b] - eta[a]
            linear += mp.cos(angle) * shift
            remainder += mp.sin(angle + shift) - mp.sin(angle) - mp.cos(angle) * shift
            square += shift**2
    assert abs(linear) < mp.mpf("1e-220")
    assert square <= 10 * _dot(eta, eta)
    gamma = 1 / (1023 * mp.pi)
    assert abs(gamma * remainder / 81) <= 5 * gamma * _dot(eta, eta) / 81
    assert abs(remainder) > mp.mpf("1e-30")
    # Identical forms give identical Q and Q', but distinct phase forces
    # give distinct b' and hence Q''. This larger-class witness is not
    # asserted to be a pair of reachable original source endpoints.
    forms = (Q(0),) * 36
    weights = (_actual_weights(mp, 0), _actual_weights(mp, 1))
    first, _ = _field(mp, forms, base, weights)
    second, _ = _field(mp, forms, (Q(0),) * 36, weights)
    assert abs(sum(first[:18]) - sum(second[:18])) < mp.mpf("1e-220")
    _, vfirst = _regional(first)
    _, vsecond = _regional(second)
    assert abs((vfirst[0] - vfirst[1]) - (vsecond[0] - vsecond[1])) > mp.mpf("1e-12")


def test_positive_memory_kernel_mass_and_forced_lower_envelope_are_algebraic(
    mp, report
):
    k = Q(4, 9)
    # M is Metzler and each unit-cube face is inward; its semigroup is
    # nonnegative with infinity norm<=1. The exact inverse supplies mass.
    for u, b in ((Q(0), Q(0)), (Q(0), Q(1)), (Q(1), Q(0)), (Q(1), Q(1))):
        ud, bd = -u + b, u - (2 * k + 1) * b
        assert ud >= 0 if u == 0 else ud <= 0
        assert bd >= 0 if b == 0 else bd <= 0
    inverse = ((-(2 * k + 1) / (2 * k), -1 / (2 * k)), (-1 / (2 * k), -1 / (2 * k)))
    matrix = ((-Q(1), Q(1)), (Q(1), -(2 * k + 1)))
    for i in (0, 1):
        for j in (0, 1):
            assert sum(
                matrix[i][column] * inverse[column][j] for column in (0, 1)
            ) == int(i == j)
    assert -9 * inverse[0][1] == Q(81, 8)
    km = mp.mpf(13) / 27 - _mp(mp, DELTA) / (18 * mp.pi)
    rlo, rhi = km + 1 - mp.sqrt(km * km + 1), km + 1 + mp.sqrt(km * km + 1)
    assert 0 < rlo < rhi
    assert abs(rlo * rhi - 2 * km) < mp.mpf("1e-220")
    assert abs((1 / rlo - 1 / rhi) / (rhi - rlo) - 1 / (2 * km)) < mp.mpf("1e-220")
    # With w=e^-t, w'=-w, these exact polynomial identities verify
    # b_F=F(1-w²)/2 and u_F=F(1-w)²/2. No exponential is evaluated.
    forcing = Q(7, 1000000000)
    for w in (Q(0), Q(1, 4), Q(1, 2), Q(1)):
        u, b = forcing * (1 - w) ** 2 / 2, forcing * (1 - w * w) / 2
        ud, bd = forcing * w * (1 - w), forcing * w * w
        assert ud == -u + b and bd == -2 * b + forcing
    assert report.kernel_impulse_upper_bound == 1
    assert report.kernel_mass_upper_bound == Q(81, 8)
    assert report.kernel_certified


def test_initial_projection_and_uniform_gain_price_source_error_only_once(report):
    d = (-Q(1, 9),) * 9 + (Q(1, 9),) * 9
    b_row = tuple(-v / 2 for v in d) + tuple(v / 2 for v in d)
    u_row = (-Q(1, 18),) * 18 + (Q(1, 18),) * 18
    assert _dot(b_row, b_row) == _dot(u_row, u_row) == Q(1, 9)
    assert _dot(tuple(9 * v for v in u_row), tuple(9 * v for v in u_row)) == 9
    # Independent per-copy zero entry means give e_u(0)=0 exactly;
    # e_b(0) is a carried bound, not a freshly measured velocity.
    ex = tuple(
        v
        for p in (0, 1)
        for v in _center(tuple(Q((j + p) % 7 - 3, 10**8) for j in range(18)))
    )
    assert _dot(u_row, ex) == 0
    assert _dot(b_row, ex) ** 2 <= _dot(ex, ex) / 9
    residual = (
        2 * GAMMA_HI * COMPARISON / 9
        + 5 * GAMMA_HI * COMPARISON**2 / 81
        + 14 * WEIGHT_ERROR / 2700
    )
    error = 3 * CUT_ERROR + Q(81, 8) * residual
    assert report.cut_form_error_upper_bound == CUT_ERROR
    assert report.full_state_comparison_upper_bound == COMPARISON
    assert report.forcing_error_upper_bound == residual
    assert report.uniform_output_error_upper_bound == error < NOISE
    assert (
        error
        == 3 * CUT_ERROR
        + Q(9, 4) * GAMMA_HI * COMPARISON
        + Q(5, 8) * GAMMA_HI * COMPARISON**2
        + Q(21, 400) * WEIGHT_ERROR
    )
    assert report.output_error_margin == NOISE - error > 0
    assert report.uniform_output_certified


def test_form_only_gap_uses_the_live_whole_mean_leaf_and_actual_weights(mp, report):
    k = mp.mpf(13) / 27 - _mp(mp, DELTA) / (18 * mp.pi)
    # Regional -M-I/3 is positive definite iff k>5/12. Hidden
    # regional-mean-zero directions retain H0/9>=1/3 and G_A>=0.
    assert Q(4, 9) > Q(5, 12)
    assert mp.mpf(2) / 3 > 0 and (mp.mpf(2) / 3) * (2 * k + mp.mpf(2) / 3) - 1 > 0
    x = tuple(
        _mp(mp, v)
        for v in _center(tuple(Q((j * 7) % 13 - 6, 10**5) for j in range(36)))
    )
    nominal = (_nominal_weights(mp),) * 2
    actual = (_actual_weights(mp, 0), _actual_weights(mp, 1))
    h0, hx, gx = _h_apply(mp, x, nominal), _h_apply(mp, x, actual), _exchange(x)
    d0 = tuple(h / 9 + g for h, g in zip(h0, gx, strict=True))
    da = tuple(h / 9 + g for h, g in zip(hx, gx, strict=True))
    assert _dot(x, d0) >= _dot(x, x) / 3
    gap = Q(1, 3) - 14 * WEIGHT_ERROR / 9
    assert _dot(x, da) >= _mp(mp, gap) * _dot(x, x)
    assert gap > Q(1, 4)
    assert report.actual_form_damping_lower_bound == gap
    assert report.nominal_form_damping_lower_bound == Q(1, 3)
    assert report.initial_form_norm_upper_bound == 9 * GAMMA_HI * QHI + CUT_ERROR
    assert (
        report.form_only_output_upper_bound
        == 3 * report.initial_form_norm_upper_bound / 2**16
    )
    # Same actual forms and port mean same initial outgoing current.
    # Omitting the phase force changes acceleration, not initial data.
    phases = _offset(SECTION / 2) + _offset(-SECTION / 3)
    full_dx, _ = _field(mp, x, phases, actual)
    assert abs(sum(full_dx[:18]) + sum(da[:18])) < mp.mpf("1e-220")
    assert max(abs(a + b) for a, b in zip(full_dx, da, strict=True)) > mp.mpf("1e-10")
    assert report.form_only_gap_certified


def test_tangent_cubic_requires_amplitudes_and_is_not_an_actual_rate(mp, report):
    gamma = 1 / (1023 * mp.pi)
    k = mp.mpf(13) / 27 - _mp(mp, DELTA) / (18 * mp.pi)
    c = 1 + 2 * mp.cos(mp.pi / 9)
    beta, chi = gamma * c / 9, 4 * gamma * k
    stiffness = beta * chi
    # Determinant of pI-A, independently assembled by polynomial products.
    determinant = _poly_add(
        _poly_mul((1, 1), _poly_add(_poly_mul((0, 1), (2 * k + 1, 1)), (stiffness,))),
        (0, -1),
    )
    expected = (stiffness, 2 * k + stiffness, 2 * k + 2, mp.mpf(1))
    assert all(
        abs(a - b) < mp.mpf("1e-220")
        for a, b in zip(determinant, expected, strict=True)
    )
    rho = stiffness / (2 * k)

    def decay_poly(r):
        return r**3 - (2 * k + 2) * r**2 + (2 * k + stiffness) * r - stiffness

    values = (rho, 2 * rho, mp.mpf(1) / 4, mp.mpf(1) / 2, mp.mpf(2), mp.mpf(3))
    assert tuple(1 if decay_poly(v) > 0 else -1 for v in values) == (
        -1,
        1,
        1,
        -1,
        -1,
        1,
    )
    assert 2 * rho < mp.mpf(1) / 4
    # Three disjoint sign-changing intervals exhaust a cubic and therefore
    # contain three real simple roots. Their amplitudes still need b0,z0.
    # An algebraic member of the admitted initial bounds, not a measured
    # or evaluated initialization of the original source.
    z0 = _mp(mp, QLO)
    b0 = gamma * z0 / (8 * k)
    for p in (mp.mpf(1), mp.mpf(2), mp.mpf(3)):
        polynomial = p**3 + (2 * k + 2) * p**2 + (2 * k + stiffness) * p + stiffness
        u = (p * b0 + beta * z0) / polynomial
        b = (p + 1) * u
        z = (z0 - chi * b) / p
        assert abs(-u + (p + 2 * k + 1) * b - beta * z - b0) < mp.mpf("1e-220")
        assert u != beta * z0 / polynomial  # inherited form velocity matters
        assert u != p * b0 / polynomial  # phase restoring amplitude matters
    # Every possible slow root is below2rho. The carried upper bound on b0
    # leaves a strictly positive numerator and hence a nonzero slow residue.
    assert beta * z0 - 2 * rho * gamma * z0 / (4 * k) > 0
    assert (mp.mpf(1) / 4 - 2 * rho) * (2 - 2 * rho) > 0
    assert report.tangent_certified
    assert report.nominal_slow_mode_excited_certified
    for interval, value in zip(
        report.characteristic_coefficients, reversed(expected), strict=True
    ):
        assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
    assert tuple(
        -1 if interval.hi < 0 else 1 for interval in report.tangent_root_sign_bounds
    ) == (-1, 1, 1, -1, -1, 1)


def test_fixed_age_discriminator_keeps_nonlinear_forcing_and_full_source_allowance(
    report,
    record_property,
):
    zmin = Q(233, 300) * QLO - 2 * GAMMA_HI**2 * HORIZON * QHI
    forcing = GAMMA_LO * Q(2, 9) * Q(99, 100) * zmin
    lower = 9 * forcing * (HORIZON / (1 + HORIZON)) ** 2 / 2
    upper = 9 * Q(6, 5) * GAMMA_HI * QHI
    error = report.uniform_output_error_upper_bound
    assert zmin > 0 and forcing > 0
    assert report.phase_difference_lower_bound == zmin
    assert report.nominal_forcing_lower_bound == forcing
    assert report.nominal_output_bounds == (lower, upper)
    assert report.full_true_bounds == (lower - error, upper + error)
    pure = 3 * (9 * GAMMA_HI * QHI + CUT_ERROR) / 2**16
    assert report.form_only_true_bounds == (-pure, pure)
    assert report.full_true_bounds[0] - NOISE > THRESHOLD
    assert pure + NOISE < THRESHOLD
    assert report.full_persistence_certified and report.certified
    # Save exact bounds from this one frozen assessment for the evidence
    # receipt; reporting them must not trigger another response evaluation.
    for name in (
        "horizon",
        "threshold",
        "uniform_output_error_upper_bound",
        "full_threshold_margin",
        "form_only_threshold_margin",
        "record_separation_margin",
    ):
        record_property(name, str(getattr(report, name)))
    for name in (
        "full_true_bounds",
        "full_record_bounds",
        "form_only_true_bounds",
        "form_only_record_bounds",
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


def test_separate_endpoint_errors_do_not_cancel_or_create_new_observations(report):
    lo, hi = report.full_true_bounds
    pure = report.form_only_output_upper_bound
    assert report.full_record_bounds == (lo - NOISE, hi + NOISE)
    assert report.form_only_record_bounds == (-pure - NOISE, pure + NOISE)
    assert report.full_threshold_margin == lo - NOISE - THRESHOLD > 0
    assert report.form_only_threshold_margin == THRESHOLD - pure - NOISE > 0
    assert report.record_separation_margin == lo - pure - 2 * NOISE > 0
    assert report.independent_record_sets_separated and report.fixed_threshold_certified
    zero = _assess(
        full_observation_error_bound=Q(0), form_only_observation_error_bound=Q(0)
    )
    assert zero.full_record_bounds == report.full_true_bounds
    assert zero.form_only_record_bounds == report.form_only_true_bounds
    for name in ("full_observation_error_bound", "form_only_observation_error_bound"):
        changed = _assess(**{name: NOISE + Q(1, 10**100)})
        assert not changed.observation_allowances_admitted and not changed.certified
        assert not changed.fixed_threshold_certified


def test_source_kernel_and_original_history_premises_are_readmitted(
    monkeypatch, report
):
    source = report.source
    for name, value in (
        ("source_association_certified", False),
        ("broad_trap_certified", False),
        ("uniform_tube_certified", False),
        ("nominal_cone_certified", False),
        ("uniform_comparison_budget", Q(0)),
        ("original_phase_gauges_and_branches_retained", False),
        ("activation_time", Q(0)),
        ("activation_relative_common_form", Q(1)),
        ("clock", "unrescaled clock"),
        ("physical_capacities", (Q(2),) * 36),
        ("whole_form_mean", Q(1)),
        ("selected_mu", Q(2)),
        ("state_reset_performed", True),
    ):
        changed = replace(source, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_persistent_contact", lambda **kwargs: changed
        )
        assert not _assess().certified
    for name, value in (
        ("storage_domain_certified", False),
        ("actual_dissipation_certified", False),
        ("mean_projector_rank", 2),
        ("port_norm_upper_bound", Q(4)),
        ("actual_weight_perturbation_upper_bound", Q(0)),
    ):
        changed = replace(source, source=replace(source.source, **{name: value}))
        monkeypatch.setattr(
            owner, "_assess_mixed_persistent_contact", lambda **kwargs: changed
        )
        result = _assess()
        assert not result.certified and not result.uniform_output_certified
    positive = source.source.source
    for name, value in (
        ("cut_form_error_upper_bound", Q(0)),
        ("handoff_phase_error_upper_bound", Q(0)),
        ("inherited_form_velocity_retained", False),
        ("clock", "changed positive-contact clock"),
        ("original_clock_horizon", Q(0)),
        ("contact_pairs", tuple((j, 18 + j) for j in range(18))),
    ):
        changed = replace(
            source,
            source=replace(source.source, source=replace(positive, **{name: value})),
        )
        monkeypatch.setattr(
            owner, "_assess_mixed_persistent_contact", lambda **kwargs: changed
        )
        assert not _assess().certified
    handoff = positive.handoff
    for altered in (
        replace(handoff, clock="changed carried clock"),
        replace(handoff, prefix=replace(handoff.prefix, clock="changed prefix clock")),
        replace(
            handoff,
            prefix=replace(
                handoff.prefix,
                reference=replace(
                    handoff.prefix.reference, clock="changed reference clock"
                ),
            ),
        ),
    ):
        changed = replace(
            source,
            source=replace(source.source, source=replace(positive, handoff=altered)),
        )
        monkeypatch.setattr(
            owner, "_assess_mixed_persistent_contact", lambda **kwargs: changed
        )
        assert not _assess().certified
    monkeypatch.setattr(
        owner, "_assess_mixed_persistent_contact", lambda **kwargs: source
    )
    registered = owner._derive_registered_phase_probe_geometry()
    for name, value in (
        ("bare_gap_lower_bound", Q(3)),
        ("weighted_gap_lower_bound", Q(2)),
        ("spectral_gap_certified", False),
    ):
        altered = replace(
            registered, successor=replace(registered.successor, **{name: value})
        )
        monkeypatch.setattr(
            owner, "_derive_registered_phase_probe_geometry", lambda: altered
        )
        assert not _assess().certified


def test_old_record_and_finite_return_verdicts_cannot_supply_a_collective_mode(
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
    )
    changed = replace(
        source,
        source=replace(source.source, source=positive, certified=False),
        certified=False,
        full_relaxation_certified=False,
        vanishing_net_transfer_certified=False,
        status="unrelated_old_verdict_unused",
        unresolved_conditions=("unused",),
    )
    monkeypatch.setattr(
        owner, "_assess_mixed_persistent_contact", lambda **kwargs: changed
    )
    result = _assess()
    assert result.certified
    assert (
        result.uniform_output_error_upper_bound
        == report.uniform_output_error_upper_bound
    )
    assert result.full_true_bounds == report.full_true_bounds


def test_raw_original_and_observation_errors_are_admitted_before_source_work(
    monkeypatch,
):
    for name in ("form_error_bounds", "phase_error_bounds"):
        assert not _assess(**{name: (ERROR + Q(1, 10**100), ERROR)}).certified

    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached the source or derived arithmetic")

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
                values = tuple(bad if j == side else Q(0) for j in range(2))
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    _assess(**{name: values})
        for bad in ((0,), (0, 0, 0), {0, 1}, {0: 0, 1: 0}, repeat(0)):
            with pytest.raises((TypeError, ValueError)):
                _assess(**{name: bad})
    for name in ("full_observation_error_bound", "form_only_observation_error_bound"):
        for bad in invalid:
            with pytest.raises((TypeError, ValueError, OverflowError)):
                _assess(**{name: bad})


def test_fixed_width_one_shot_inputs_cannot_fit_modes_or_reset_the_same_initial_state(
    report,
):
    assert report.horizon == HORIZON
    assert report.activation_time == 2010000
    assert report.total_duration == 2010000 + HORIZON
    assert report.original_clock_horizon == (2010000 + HORIZON) * Q(1024, 1023)
    assert report.initial_donor_form_total == report.initial_common_form_difference == 0
    assert report.nominal_state_dimension == 5
    assert report.tangent_output_subsystem_dimension == 3
    assert report.nonlinear_sum_phase_retained
    assert report.tangent_laplace_numerator == "p*b0+beta*z0"
    assert report.tangent_coefficients_correlated
    assert not report.all_tangent_modes_excited_claimed
    assert not report.numerical_eigenvalues_evaluated
    assert not report.observed_slow_rate_identified
    assert not report.actual_nonlinear_slow_timescale_identified
    assert report.same_actual_initial_state_required
    assert report.same_actual_transport_required
    assert report.independent_record_errors
    assert not report.observation_error_cancellation_assumed
    assert not report.exact_actual_five_coordinate_closure
    assert not report.source_error_reinitialized
    assert not report.source_association_authenticated
    assert not report.form_only_is_actual_complete_law
    assert not report.state_reset_performed
    assert not report.source_trajectory_evaluated
    assert report.numerical_response_allowance == 0
    assert report.newly_observed_aggregates == 0
    assert report.required_endpoint_records == 2
    assert not report.runtime_event_executed
    assert not report.physical_timescale_identified
    assert not report.empirical_validation
    error = Q(np.int64(ERROR.numerator), np.int64(ERROR.denominator))
    noise = Q(np.int64(NOISE.numerator), np.int64(NOISE.denominator))
    assert (
        _assess(
            form_error_bounds=(error for _ in range(2)),
            phase_error_bounds=(error for _ in range(2)),
            full_observation_error_bound=noise,
            form_only_observation_error_bound=noise,
        )
        == report
    )
    with pytest.raises(FrozenInstanceError):
        report.uniform_output_error_upper_bound = Q(0)
    for name in (
        "source",
        "horizon",
        "duration",
        "endpoint",
        "donor_velocity",
        "initial_current",
        "phase_gap",
        "slow_rate",
        "modal_amplitude",
        "observed_record",
        "threshold",
        "port_strength",
        "current_error_bound",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: Q(0)})
