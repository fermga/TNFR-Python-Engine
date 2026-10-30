"""Independent algebra for the frozen conditional joint-work proof.

These controls use free symbols and a complete three-node state witness.
They import no engine reader, select no source domain or horizon, and evaluate
no trajectory. Existing owners retain the relative-work and angle identities.
"""

from types import SimpleNamespace

import pytest
import sympy as s


def _zero(expression):
    assert s.cancel(s.trigsimp(s.expand_trig(s.expand(expression)))) == 0


def test_implicit_reference_speed_has_a_curvature_and_cauchy_square_remainder():
    q, C, S, V, Cdot, Sdot, Vdot = s.symbols("q C S V Cdot Sdot Vdot", real=True)
    gamma, kappa, m, h = s.symbols("gamma kappa m h", positive=True)
    force = s.sin(q) + gamma * (C * s.sin(gamma * q) + S * s.cos(gamma * q))
    # h is the actual derivative F_q at the admitted critical reference.
    qrate = -(s.diff(force, C) * Cdot + s.diff(force, S) * Sdot) / h
    rrate = s.diff(gamma * V / kappa, V) * Vdot
    work_squared = gamma**2 * (Vdot**2 / kappa + (Cdot**2 + Sdot**2) / m)
    reference_speed_squared = m * qrate**2 + kappa * rrate**2
    perpendicular = Cdot * s.cos(gamma * q) - Sdot * s.sin(gamma * q)
    remainder = (h**2 - m**2) * (Cdot**2 + Sdot**2) + m**2 * perpendicular**2
    _zero(m * h**2 * (work_squared - reference_speed_squared) / gamma**2 - remainder)
    excess = s.symbols("excess", nonnegative=True)
    _zero(
        remainder.subs(h, m + excess)
        - excess * (2 * m + excess) * (Cdot**2 + Sdot**2)
        - m**2 * perpendicular**2
    )
    # The derivative lower bound is essential, not merely a label on m.
    invalid = {h: m / 2, q: 0, Cdot: 0, Sdot: 1, Vdot: 0}
    _zero(
        (work_squared - reference_speed_squared).subs(invalid, simultaneous=True)
        + 3 * gamma**2 / m
    )


def test_reference_speed_conversion_retains_both_original_coordinate_scales():
    qrate, rrate = s.symbols("qrate rrate", real=True)
    lower, weight = s.symbols("lower weight", positive=True)
    phase_excess, form_excess = s.symbols("phase_excess form_excess", nonnegative=True)
    m = lower + phase_excess
    kappa = weight * (lower + form_excess)
    original_speed_squared = qrate**2 + weight * rrate**2
    _zero(
        m * qrate**2
        + kappa * rrate**2
        - lower * original_speed_squared
        - phase_excess * qrate**2
        - weight * form_excess * rrate**2
    )


@pytest.fixture(scope="module")
def budget():
    # a=sqrt(lambda), b=sqrt(Lambda), h=sqrt(m). No numerical metric is chosen.
    a, b, h, Q0, b0, low_radius, high_radius = s.symbols(
        "a b h Q0 b0 low_radius high_radius", positive=True
    )
    B0, j = s.symbols("B0 j", nonnegative=True)
    return SimpleNamespace(
        a=a,
        b=b,
        h=h,
        Q0=Q0,
        b0=b0,
        low_radius=low_radius,
        high_radius=high_radius,
        B0=B0,
        j=j,
        Qlo=Q0 - j,
        Qhi=Q0 + j,
        offset=B0 + j / a,
        clearance=b0 - j / h,
    )


def test_one_integral_gives_the_chart_and_angle_thresholds(budget):
    p = budget
    lower_error = p.Qlo / p.b
    chart_reserve = p.h * p.clearance - p.Qhi
    chart_threshold = (p.h * p.b0 - p.Q0) / 2
    _zero(chart_reserve - 2 * (chart_threshold - p.j))
    # Start from the original angular inequality lambda*e_min > Lambda*B.
    angle_reserve = p.a**2 * lower_error - p.b**2 * p.offset
    coefficient = p.a**2 + p.b**3 / p.a
    angle_threshold = (p.a**2 * p.Q0 - p.b**3 * p.B0) / coefficient
    _zero(p.b * angle_reserve - coefficient * (angle_threshold - p.j))
    # A positive angular threshold is strictly below Q0, so satisfying it
    # also retains the positive radial denominator required by that bridge.
    _zero(p.Q0 - angle_threshold - p.b**3 * (p.B0 + p.Q0 / p.a) / coefficient)
    assert (p.b**3 * (p.B0 + p.Q0 / p.a) / coefficient).is_positive


def test_independent_original_energy_limits_give_distinct_work_thresholds(budget):
    p = budget
    # low_radius=sqrt(2*E_low/c_E), high_radius=sqrt(2*E_high).
    # Derive from unsquared triangle bounds, preserving positive separation.
    lower_reserve = p.Qlo / p.b - p.offset - p.low_radius
    lower_coefficient = 1 / p.a + 1 / p.b
    lower_threshold = (p.Q0 / p.b - p.B0 - p.low_radius) / lower_coefficient
    _zero(lower_reserve - lower_coefficient * (lower_threshold - p.j))
    upper_reserve = p.high_radius - p.Qhi / p.a - p.offset
    upper_threshold = (p.a * (p.high_radius - p.B0) - p.Q0) / 2
    _zero(upper_reserve - 2 * (upper_threshold - p.j) / p.a)
    assert s.diff(lower_threshold, p.low_radius).is_negative
    assert s.diff(upper_threshold, p.high_radius).is_positive
    # Independent lower/upper identity limits cannot be silently identified.
    assert not lower_threshold.has(p.high_radius)
    assert not upper_threshold.has(p.low_radius)


def test_offset_ratio_increases_and_original_rotation_bound_decreases(budget):
    p = budget
    rho = p.b * p.offset / p.Qlo
    _zero(
        s.diff(rho, p.j)
        - p.b * (p.B0 + p.Q0 / p.a) / (p.Q0**2 - 2 * p.Q0 * p.j + p.j**2)
    )
    ratio = s.symbols("ratio", nonnegative=True)
    A = s.symbols("A", positive=True)
    omega = A * (p.a**2 - p.b**2 * ratio) / (1 + ratio) ** 2
    _zero(
        s.diff(omega, ratio)
        - A * (p.b**2 * ratio - p.b**2 - 2 * p.a**2) / (1 + ratio) ** 3
    )
    clearance = s.symbols("clearance", positive=True)
    numerator = p.b**2 * ratio - p.b**2 - 2 * p.a**2
    # ratio=(lambda/Lambda)*(1-clearance), with 0<clearance<=1,
    # parameterizes the admitted angular interval and proves its sign.
    admitted_numerator = numerator.subs(ratio, p.a**2 * (1 - clearance) / p.b**2)
    _zero(admitted_numerator + p.b**2 + p.a**2 * (1 + clearance))
    assert s.expand(admitted_numerator).is_negative


@pytest.fixture(scope="module")
def full_path():
    xa, xc, xj, ta, tc, tj = s.symbols("xa xc xj ta tc tj", real=True)
    ma, mc, mj, w, v = s.symbols("ma mc mj w v", positive=True)
    forms, phases = (xa, xc, xj), (ta, tc, tj)
    storage = (
        w * (xc - xa) ** 2 / 2
        + v * (xj - xa) ** 2 / 2
        + 1
        - s.cos(tc - ta)
        + 1
        - s.cos(tj - ta)
    )
    mobilities = (ma, mc, mj)
    form_rates = tuple(-m * s.diff(storage, t) for m, t in zip(mobilities, phases))
    phase_rates = tuple(m * s.diff(storage, x) for m, x in zip(mobilities, forms))
    coordinates, rates = forms + phases, form_rates + phase_rates

    def derivative(expression):
        return sum(s.diff(expression, z) * rate for z, rate in zip(coordinates, rates))

    gamma, mu = ma / (ma + mc), ma * mc / (ma + mc)
    X, theta = xa + gamma * (xc - xa), ta + gamma * (tc - ta)
    alpha = tj - theta
    # Positive mobilities correspond to nu_i=pi*d_i*m_i. The full law
    # keeps both bare phase edges and weighted form edges, without a host freeze.
    return SimpleNamespace(
        forms=forms,
        phases=phases,
        ma=ma,
        mc=mc,
        mj=mj,
        w=w,
        v=v,
        gamma=gamma,
        mu=mu,
        kappa=w + gamma**2 * v,
        form_rates=form_rates,
        phase_rates=phase_rates,
        derivative=derivative,
        X=X,
        theta=theta,
        alpha=alpha,
        V0=v * (X - xj),
        C=s.cos(alpha),
        S=s.sin(alpha),
    )


def test_coefficient_rates_follow_from_all_complete_rows_and_keep_mean_leaves(
    full_path,
):
    p = full_path
    xa, _, xj = p.forms
    ta, _, tj = p.phases
    V, U = p.v * (xa - xj), s.sin(tj - ta)
    alpha_rate = p.phase_rates[2] - p.mu * V
    _zero(p.derivative(p.V0) - p.v * p.mu * U + p.v * p.form_rates[2])
    _zero(p.derivative(p.C) + s.sin(p.alpha) * alpha_rate)
    _zero(p.derivative(p.S) - s.cos(p.alpha) * alpha_rate)
    # Both complete weighted means are first integrals. No projection or
    # reinitialization of them is introduced by the coefficient observation.
    mobilities = (p.ma, p.mc, p.mj)
    for rates in (p.form_rates, p.phase_rates):
        _zero(sum(rate / m for rate, m in zip(rates, mobilities)))


def test_signed_incidence_cancels_the_child_edge_before_environmental_bounds(full_path):
    p = full_path
    # Columns orient a->c and a->j. The full storage constructs the law;
    # these independently assembled matrices expose its signed coefficients.
    incidence = s.Matrix(((1, 1), (-1, 0), (0, -1)))
    mobility = s.diag(p.ma, p.mc, p.mj)
    laplacian = incidence * s.diag(p.w, p.v) * incidence.T
    form = s.Matrix(p.forms)
    phase = s.Matrix(p.phases)
    ell = s.Matrix([s.diff(p.V0, x) for x in p.forms])
    edge_coefficients = incidence.T * mobility * ell
    _zero(edge_coefficients[0])
    _zero(edge_coefficients[1] - p.v * (p.mu + p.mj))
    edge_phases = incidence.T * phase
    signed_rate = -(edge_coefficients.T * edge_phases.applyfunc(s.sin))[0]
    _zero(p.derivative(p.V0) - signed_rate)
    # Taking absolute node terms first would retain two positive bridge
    # terms whose exact difference vanishes. That loses a useful correlation.
    assert s.simplify(p.ma * ell[0]).is_positive
    _zero(p.ma * ell[0] - p.mc * ell[1])
    u = s.Matrix([s.diff(p.alpha, t) for t in p.phases])
    h = u.T * mobility * laplacian
    _zero(p.derivative(p.alpha) - (h * form)[0])
    _zero((u.T * phase)[0] - p.alpha)


def test_complex_coefficient_variation_retains_initial_rate_and_phase_change():
    angle, initial, rate, initial_rate = s.symbols(
        "angle initial rate initial_rate", real=True
    )
    phasor = s.cos(angle) + s.I * s.sin(angle)
    initial_phasor = phasor.subs(angle, initial)
    derivative = s.diff(phasor, angle) * rate
    initial_derivative = derivative.subs(
        {angle: initial, rate: initial_rate}, simultaneous=True
    )
    split = s.I * (
        phasor * (rate - initial_rate) + (phasor - initial_phasor) * initial_rate
    )
    _zero(derivative - initial_derivative - split)
    chord = phasor - initial_phasor
    _zero(
        s.expand_complex(chord * s.conjugate(chord))
        - 4 * s.sin((angle - initial) / 2) ** 2
    )
    # An unchanged angular rate does not remove the phase-change term.
    fixed_rate = (derivative - initial_derivative).subs(rate, initial_rate)
    _zero(fixed_rate - s.I * chord * initial_rate)
    assert (
        s.simplify(fixed_rate.subs({angle: s.pi / 2, initial: 0, initial_rate: 1})) != 0
    )


def test_zero_initial_work_norm_does_not_bound_the_evolving_environment(full_path):
    p = full_path
    q = s.symbols("q", real=True)
    xa, xc, xj = p.forms
    ta, tc, tj = p.phases
    state = {xa: 0, xc: 0, xj: 0, ta: 0, tc: q, tj: 0}
    for rate in p.phase_rates:
        _zero(rate.subs(state, simultaneous=True))
    for coefficient in (p.V0, p.C, p.S):
        _zero(p.derivative(coefficient).subs(state, simultaneous=True))
    alpha_rate = p.derivative(p.alpha)
    acceleration = p.derivative(alpha_rate).subs(state, simultaneous=True)
    expected = -(p.mj + p.mu) * p.v * p.ma * s.sin(q)
    _zero(acceleration - expected)
    # Differentiate the coefficient vector itself, retaining both moving
    # center and external-node rows, rather than prescribing its acceleration.
    C_acceleration = p.derivative(p.derivative(p.C)).subs(state, simultaneous=True)
    S_acceleration = p.derivative(p.derivative(p.S)).subs(state, simultaneous=True)
    _zero(C_acceleration**2 + S_acceleration**2 - expected**2)
    exact_angle = expected.subs(q, s.pi / 6)
    assert exact_angle.is_negative
    # A nonzero derivative of (Cdot,Sdot) at an initial zero makes that
    # vector nonzero for sufficiently small positive time by smoothness.
    # This is a complete-law state witness, not a retained-source prediction.
