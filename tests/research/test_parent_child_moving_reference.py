"""Independent symbolic controls for conditional moving-reference admission.

The parent-child-moving-reference protocol permits free symbolic algebra
and exact constructed domain/observation counterexamples. These controls
import no engine source reader and evaluate no retained preparation, root
search, numerical trajectory, event or new scientific parameter family.
"""

from types import SimpleNamespace

import pytest
import sympy as s


def _zero(expression):
    assert s.cancel(s.trigsimp(s.expand_trig(s.expand(expression)))) == 0


@pytest.fixture(scope="module")
def potential():
    r, q, qstar, V, C, S = s.symbols("r q qstar V C S", real=True)
    gamma, kappa, k = s.symbols("gamma kappa k", positive=True)
    Vdot, Cdot, Sdot, qstar_dot = s.symbols("Vdot Cdot Sdot qstar_dot", real=True)
    phase = 1 - s.cos(q) + C * (1 - s.cos(gamma * q)) + S * s.sin(gamma * q)
    energy = kappa * r**2 / 2 - gamma * V * r + phase
    rstar = gamma * V / kappa
    reference = energy.subs({r: rstar, q: qstar}, simultaneous=True)
    return SimpleNamespace(
        r=r,
        q=q,
        qstar=qstar,
        V=V,
        C=C,
        S=S,
        gamma=gamma,
        kappa=kappa,
        k=k,
        Vdot=Vdot,
        Cdot=Cdot,
        Sdot=Sdot,
        qstar_dot=qstar_dot,
        phase=phase,
        energy=energy,
        rstar=rstar,
        relative=energy - reference,
    )


def test_square_completion_and_hessian_preserve_the_phase_remainder(potential):
    p = potential
    _zero(
        p.relative
        - p.kappa * (p.r - p.rstar) ** 2 / 2
        - p.phase
        + p.phase.subs(p.q, p.qstar)
    )
    hessian = s.hessian(p.energy, (p.r, p.q))
    assert hessian[0, 0] == p.kappa
    assert hessian[0, 1] == hessian[1, 0] == 0
    _zero(
        hessian[1, 1]
        - s.cos(p.q)
        - p.gamma**2 * (p.C * s.cos(p.gamma * p.q) - p.S * s.sin(p.gamma * p.q))
    )
    # The integral Taylor remainder is fixed by its zero value/slope at
    # qstar and its second derivative on the entire intervening segment.
    # A curvature bound at qstar alone therefore cannot replace that domain.
    remainder = (
        p.phase
        - p.phase.subs(p.q, p.qstar)
        - s.diff(p.phase, p.q).subs(p.q, p.qstar) * (p.q - p.qstar)
    )
    assert remainder.subs(p.q, p.qstar) == 0
    assert s.diff(remainder, p.q).subs(p.q, p.qstar) == 0
    _zero(s.diff(remainder, p.q, 2) - hessian[1, 1])


def test_reference_rates_are_implicit_derivatives_of_criticality(potential):
    p = potential
    force = s.diff(p.phase, p.q).subs(p.q, p.qstar)
    curvature = s.diff(force, p.qstar)
    derivative = (
        curvature * p.qstar_dot
        + s.diff(force, p.C) * p.Cdot
        + s.diff(force, p.S) * p.Sdot
    )
    implicit_rate = s.solve(derivative, p.qstar_dot)[0]
    expected = (
        -p.gamma
        * (p.Cdot * s.sin(p.gamma * p.qstar) + p.Sdot * s.cos(p.gamma * p.qstar))
        / curvature
    )
    _zero(implicit_rate - expected)
    _zero(s.diff(p.rstar, p.V) * p.Vdot - p.gamma * p.Vdot / p.kappa)
    # This division is a conditional implicit-function identity. The
    # admitted branch requires curvature bounded strictly away from zero.
    _zero(derivative.subs(p.qstar_dot, expected))


def test_complete_chain_rule_retains_moving_coefficients_and_reference(potential):
    p = potential
    rates = {
        p.r: -p.k * s.diff(p.energy, p.q),
        p.q: p.k * s.diff(p.energy, p.r),
        p.V: p.Vdot,
        p.C: p.Cdot,
        p.S: p.Sdot,
        p.qstar: p.qstar_dot,
    }
    derivative = sum(
        s.diff(p.relative, coordinate) * rate for coordinate, rate in rates.items()
    )
    force_at_reference = s.diff(p.phase, p.q).subs(p.q, p.qstar)
    work = (
        -p.gamma * (p.r - p.rstar) * p.Vdot
        + (s.cos(p.gamma * p.qstar) - s.cos(p.gamma * p.q)) * p.Cdot
        + (s.sin(p.gamma * p.q) - s.sin(p.gamma * p.qstar)) * p.Sdot
    )
    _zero(derivative - work + force_at_reference * p.qstar_dot)
    # Criticality, not a frozen-reference assumption, removes the last
    # term. At qstar=S=0 the reference can move whenever Sdot is nonzero.
    _zero((derivative - work).subs({p.qstar: 0, p.S: 0}, simultaneous=True))
    hstar = s.diff(p.phase, p.q, 2).subs({p.q: 0, p.S: 0}, simultaneous=True)
    _zero(-p.gamma * p.Sdot / hstar + p.gamma * p.Sdot / (1 + p.gamma**2 * p.C))
    assert s.diff(work, p.Vdot) != 0
    assert s.diff(work, p.Cdot) != 0
    assert s.diff(work, p.Sdot) != 0


def test_work_bound_has_exact_trigonometric_and_weighted_cauchy_remainders(potential):
    p = potential
    left = s.cos(p.gamma * p.qstar) - s.cos(p.gamma * p.q)
    right = s.sin(p.gamma * p.q) - s.sin(p.gamma * p.qstar)
    _zero(left**2 + right**2 - 4 * s.sin(p.gamma * (p.q - p.qstar) / 2) ** 2)
    _zero(
        (left**2 + right**2) * (p.Cdot**2 + p.Sdot**2)
        - (left * p.Cdot + right * p.Sdot) ** 2
        - (left * p.Sdot - right * p.Cdot) ** 2
    )
    u, v, F, R = s.symbols("u v F R", real=True)
    lower = s.symbols("lower", positive=True)
    # Combined with |sin t|<=|t| and the segment coercivity premise, this
    # proves |Ldot|<=sqrt(2L)*gamma*sqrt(Vdot²/kappa+|Bdot|²/lower).
    _zero(
        p.kappa
        * lower
        * (
            (p.kappa * u**2 + lower * v**2) * (F**2 / p.kappa + R**2 / lower)
            - (u * F + v * R) ** 2
        )
        - (p.kappa * u * R - lower * v * F) ** 2
    )


def test_regularized_radial_work_avoids_dividing_by_zero_energy():
    energy = s.symbols("energy", nonnegative=True)
    epsilon, budget = s.symbols("epsilon budget", positive=True)
    regularized = s.sqrt(2 * energy + epsilon**2)
    extremal_rate = s.diff(regularized, energy) * s.sqrt(2 * energy) * budget
    _zero(1 - (extremal_rate / budget) ** 2 - epsilon**2 / (2 * energy + epsilon**2))
    assert (epsilon**2 / (2 * energy + epsilon**2)).is_positive
    assert extremal_rate.subs(energy, 0) == 0
    assert regularized.subs(energy, 0) == epsilon


def test_root_curvature_alone_and_nonminimal_critical_points_do_not_admit_a_chart(
    potential,
):
    p = potential
    phase = p.phase.subs({p.C: 0, p.S: 0})
    curvature = s.diff(phase, p.q, 2)
    assert curvature.subs(p.q, 0) == 1
    assert curvature.subs(p.q, 2 * s.pi / 3) == -s.Rational(1, 2)
    assert s.expand(s.series(phase - p.q**2 / 2, p.q, 0, 6).removeO()).coeff(
        p.q, 4
    ) == -s.Rational(1, 24)
    # Exact realizable phasor coefficient controls: gamma=1/2 and four or
    # eight external phases at pi give C=-4 or -8, S=0. Both qstar=0 points
    # are critical; neither supplies the required strict-minimum chart.
    for C, expected_curvature in ((-4, 0), (-8, -1)):
        changed = {p.gamma: s.Rational(1, 2), p.C: C, p.S: 0, p.q: 0}
        assert s.diff(p.phase, p.q).subs(changed, simultaneous=True) == 0
        assert (
            s.diff(p.phase, p.q, 2).subs(changed, simultaneous=True)
            == expected_curvature
        )


def test_original_angle_chain_rule_cancels_reference_velocity_but_centered_angle_does_not():
    eq, ev, zq, zv, zdq, zdv = s.symbols("eq ev zq zv zdq zdv", real=True)
    A, c, h = s.symbols("A c h", positive=True)
    error = s.Matrix((eq, ev))
    reference = s.Matrix((zq, zv))
    reference_rate = s.Matrix((zdq, zdv))
    original = error + reference
    actual_rate = A * s.Matrix((c * ev, -h * eq))
    error_rate = actual_rate - reference_rate

    def cross(left, right):
        return left[0] * right[1] - left[1] * right[0]

    via_moving_chart = cross(original, error_rate + reference_rate)
    expected = -A * (h * eq**2 + c * ev**2 + h * zq * eq + c * zv * ev)
    _zero(via_moving_chart - expected)
    assert not via_moving_chart.expand().has(zdq, zdv)
    _zero(
        cross(error, error_rate)
        - cross(error, actual_rate)
        + cross(error, reference_rate)
    )
    assert s.diff(cross(error, error_rate), zdq) == ev
    assert s.diff(cross(error, error_rate), zdv) == -eq


def test_direct_angle_bounds_have_exact_positive_semidefinite_remainders():
    eq, ev, zq, zv = s.symbols("eq ev zq zv", real=True)
    lower, upper, c, h = s.symbols("lower upper c h", positive=True)
    De = s.Matrix((h * eq, c * ev))
    reference = s.Matrix((zq, zv))
    _zero(
        reference.dot(reference) * De.dot(De)
        - reference.dot(De) ** 2
        - (zq * c * ev - zv * h * eq) ** 2
    )
    _zero(
        upper**2 * (eq**2 + ev**2)
        - De.dot(De)
        - (upper - h) * (upper + h) * eq**2
        - (upper - c) * (upper + c) * ev**2
    )
    da, db = s.symbols("da db", nonnegative=True)
    lower_gap = (h * eq**2 + c * ev**2 - lower * (eq**2 + ev**2)).subs(
        {h: lower + da, c: lower + db}
    )
    _zero(lower_gap - da * eq**2 - db * ev**2)
    assert s.expand(lower_gap).is_nonnegative
    # These identities justify the bound only with c,h in [lower,upper]
    # and |z|/|e|<lower/upper; they do not assert that a source meets it.


def test_original_radius_and_energy_bridge_keep_the_offset_and_original_origin():
    eq, ev, zq, zv, q, v = s.symbols("eq ev zq zv q v", real=True)
    _zero(
        (eq**2 + ev**2) * (zq**2 + zv**2)
        - (eq * zq + ev * zv) ** 2
        - (eq * zv - ev * zq) ** 2
    )
    energy = v**2 / 2 + 1 - s.cos(q)
    _zero(2 * energy - v**2 - 4 * s.sin(q / 2) ** 2)
    difference = (q**2 + v**2) / 2 - energy
    assert difference.subs(q, 0) == 0
    assert s.diff(difference, q).subs(q, 0) == 0
    _zero(s.diff(difference, q, 2) - (1 - s.cos(q)))
    # Triangle bounds concern |e+z|, not |e| alone. A lower energy bridge
    # additionally needs a declared principal chart and nonzero signal.


@pytest.fixture(scope="module")
def full_path():
    forms = s.symbols("xa xc xj", real=True)
    phases = s.symbols("theta_a theta_c theta_j", real=True)
    ma, mc, mj, w, external = s.symbols("ma mc mj w external", positive=True)
    storage = (
        w * (forms[1] - forms[0]) ** 2 / 2
        + external * (forms[2] - forms[0]) ** 2 / 2
        + 1
        - s.cos(phases[1] - phases[0])
        + 1
        - s.cos(phases[2] - phases[0])
    )
    form_rates = tuple(
        -m * s.diff(storage, theta) for m, theta in zip((ma, mc, mj), phases)
    )
    phase_rates = tuple(m * s.diff(storage, x) for m, x in zip((ma, mc, mj), forms))
    gamma = ma / (ma + mc)
    return SimpleNamespace(
        forms=forms,
        phases=phases,
        ma=ma,
        mc=mc,
        mj=mj,
        w=w,
        external=external,
        gamma=gamma,
        k=ma + mc,
        kappa=w + gamma**2 * external,
        form_rates=form_rates,
        phase_rates=phase_rates,
    )


def test_complete_primitive_state_can_have_positive_relative_storage_at_original_zero_signal(
    full_path,
):
    p = full_path
    host = s.symbols("host", real=True, nonzero=True)
    state = dict(zip(p.forms, (0, 0, host))) | dict.fromkeys(p.phases, 0)
    rstar = -p.gamma * p.external * host / p.kappa
    relative = p.kappa * rstar**2 / 2
    assert relative.is_positive
    original_r = (p.forms[1] - p.forms[0]).subs(state)
    original_q = (p.phases[1] - p.phases[0]).subs(state)
    assert p.w * original_r**2 / 2 + 1 - s.cos(original_q) == 0
    assert original_q**2 + p.w * original_r**2 == 0
    assert all(rate.subs(state) == 0 for rate in p.form_rates)
    _zero((p.phase_rates[1] - p.phase_rates[0]).subs(state) - p.ma * p.external * host)
    # Exact special case: ma=mc=w=external=1, host=-1. These are symbolic
    # mobility coefficients (capacities pi*d*m), never engine materialization.
    special = {p.ma: 1, p.mc: 1, p.w: 1, p.external: 1, host: -1}
    assert relative.subs(special) == s.Rational(1, 10)
    assert rstar.subs(special) == s.Rational(2, 5)


def test_complete_primitive_state_can_reverse_the_original_angular_sign(full_path):
    p = full_path
    b, c = s.symbols("b c", positive=True)
    V0 = p.kappa * (b + c) / p.gamma
    state = dict(
        zip(p.forms, (-p.gamma * c, (1 - p.gamma) * c, -V0 / p.external))
    ) | dict.fromkeys(p.phases, 0)
    rdot = (p.form_rates[1] - p.form_rates[0]).subs(state, simultaneous=True)
    qdot = (p.phase_rates[1] - p.phase_rates[0]).subs(state, simultaneous=True)
    assert rdot == 0
    _zero(qdot + p.k * p.kappa * b)
    original_cross = -s.sqrt(p.w) * c * qdot
    restoring_cross = s.sqrt(p.w) * b * qdot
    _zero(original_cross - p.k * p.kappa * s.sqrt(p.w) * b * c)
    _zero(restoring_cross + p.k * p.kappa * s.sqrt(p.w) * b**2)
    assert (p.k * p.kappa * s.sqrt(p.w) * b * c).is_positive
    assert (p.kappa * b**2 / 2).is_positive
    # This is a complete-law instantaneous state with C=1,S=0 and positive
    # local phase curvature 1+gamma². The restoring-field orientation is
    # not a centered-angle trajectory claim; the reference may also move.


def test_centered_closed_circle_does_not_imply_a_turn_about_the_original_origin():
    t = s.symbols("t", real=True)
    radius, gap = s.symbols("radius gap", positive=True)
    error = s.Matrix((radius * s.cos(t), radius * s.sin(t)))
    original = error + s.Matrix((radius + gap, 0))
    centered_cross = s.det(s.Matrix.hstack(error, error.diff(t)))
    original_cross = s.det(s.Matrix.hstack(original, original.diff(t)))
    _zero(centered_cross - radius**2)
    _zero(original_cross.subs(t, s.pi) + radius * gap)
    _zero(original[0] - gap - 2 * radius * s.cos(t / 2) ** 2)
    assert original.subs(t, 0) == original.subs(t, 2 * s.pi)
    # The whole original loop stays in q>=gap>0, so its continuously lifted
    # angle has zero net increment. This is ONLY an observation-geometry
    # counterexample, not a claimed solution of the selected network law.
