"""Independent algebra for the frozen signed boundary-transport theorem.

Free symbols and a complete four-node state supply the controls. No engine
reader, retained preparation, numerical solver or source trajectory is used.
The corrector is auxiliary signed work, not a replacement positive storage.
"""

from types import SimpleNamespace

import pytest
import sympy as s


def _zero(expression):
    assert s.cancel(s.trigsimp(s.expand_trig(s.expand(expression)))) == 0


@pytest.fixture(scope="module")
def full_tree():
    forms = s.symbols("xa xc xj xl", real=True)
    phases = s.symbols("ta tc tj tl", real=True)
    ma, mc, mj, ml, w, v, z = s.symbols("ma mc mj ml w v z", positive=True)
    mobilities = (ma, mc, mj, ml)
    edges = ((0, 1, w), (0, 2, v), (2, 3, z))
    storage = sum(
        weight * (forms[right] - forms[left]) ** 2 / 2
        + 1
        - s.cos(phases[right] - phases[left])
        for left, right, weight in edges
    )
    # Independent Hamiltonian assembly keeps bare phase storage and weighted
    # form storage separate. The remote j-l edge evolves both complete rows.
    form_rates = s.Matrix(
        [
            -mobility * s.diff(storage, phase)
            for mobility, phase in zip(mobilities, phases)
        ]
    )
    phase_rates = s.Matrix(
        [mobility * s.diff(storage, form) for mobility, form in zip(mobilities, forms)]
    )
    incidence = s.Matrix(((1, 1, 0), (-1, 0, 0), (0, -1, 1), (0, 0, -1)))
    mobility = s.diag(*mobilities)
    laplacian = incidence * s.diag(w, v, z) * incidence.T
    gamma, mu = ma / (ma + mc), ma * mc / (ma + mc)
    X = forms[0] + gamma * (forms[1] - forms[0])
    theta = phases[0] + gamma * (phases[1] - phases[0])

    def derivative(expression):
        return sum(
            s.diff(expression, coordinate) * rate
            for coordinate, rate in zip(
                forms + phases, tuple(form_rates) + tuple(phase_rates)
            )
        )

    return SimpleNamespace(
        forms=forms,
        phases=phases,
        mobilities=mobilities,
        ma=ma,
        mc=mc,
        mj=mj,
        ml=ml,
        w=w,
        v=v,
        z=z,
        storage=storage,
        form_rates=form_rates,
        phase_rates=phase_rates,
        B=incidence,
        M=mobility,
        laplacian=laplacian,
        gamma=gamma,
        mu=mu,
        k=ma + mc,
        kappa=w + gamma**2 * v,
        X=X,
        theta=theta,
        V=v * (X - forms[2]),
        alpha=phases[2] - theta,
        derivative=derivative,
    )


def test_complete_hamiltonian_rows_give_both_accelerations_and_conserved_means(
    full_tree,
):
    p = full_tree
    phi = p.B.T * s.Matrix(p.phases)
    currents = phi.applyfunc(s.sin)
    expected_form = -p.M * p.B * currents
    expected_phase = p.M * p.laplacian * s.Matrix(p.forms)
    for actual, expected in zip(p.form_rates, expected_form):
        _zero(actual - expected)
    for actual, expected in zip(p.phase_rates, expected_phase):
        _zero(actual - expected)

    form_acceleration = p.form_rates.jacobian(p.phases) * p.phase_rates
    phase_acceleration = p.phase_rates.jacobian(p.forms) * p.form_rates
    expected_form_acceleration = (
        -p.M
        * p.B
        * s.diag(*(s.cos(value) for value in phi))
        * p.B.T
        * p.M
        * p.laplacian
        * s.Matrix(p.forms)
    )
    expected_phase_acceleration = -p.M * p.laplacian * p.M * p.B * currents
    for actual, expected in zip(form_acceleration, expected_form_acceleration):
        _zero(actual - expected)
    for actual, expected in zip(phase_acceleration, expected_phase_acceleration):
        _zero(actual - expected)
    for rates in (p.form_rates, p.phase_rates):
        _zero(sum(rate / mobility for rate, mobility in zip(rates, p.mobilities)))


def test_coefficient_accelerations_keep_signed_child_cancellation_and_remote_rows(
    full_tree,
):
    p = full_tree
    ell = s.Matrix([s.diff(p.V, form) for form in p.forms])
    u = s.Matrix([s.diff(p.alpha, phase) for phase in p.phases])
    g = p.B.T * p.M * ell
    h = u.T * p.M * p.laplacian
    expected_h = (
        -p.v * (p.mj + p.mu),
        0,
        p.v * (p.mj + p.mu) + p.mj * p.z,
        -p.mj * p.z,
    )
    for actual, expected in zip(h, expected_h):
        _zero(actual - expected)
    _zero(g[0])
    _zero(h[1])
    assert expected_h[3].is_negative
    # The remote form enters alpha_dot even though it is absent from the
    # pair and the instantaneous coefficient values V, C and S.
    _zero(s.diff(p.derivative(p.alpha), p.forms[3]) + p.mj * p.z)

    phi = p.B.T * s.Matrix(p.phases)
    sine = phi.applyfunc(s.sin)
    Vdot = -(g.T * sine)[0]
    Vddot = -(
        g.T
        * s.diag(*(s.cos(value) for value in phi))
        * p.B.T
        * p.M
        * p.laplacian
        * s.Matrix(p.forms)
    )[0]
    alpha_dot = (h * s.Matrix(p.forms))[0]
    alpha_ddot = -(h * p.M * p.B * sine)[0]
    _zero(p.derivative(p.V) - Vdot)
    _zero(p.derivative(p.derivative(p.V)) - Vddot)
    _zero(p.derivative(p.alpha) - alpha_dot)
    _zero(p.derivative(p.derivative(p.alpha)) - alpha_ddot)

    # Differentiate the complex observation independently, then insert the
    # complete-law alpha rates above. Both acceleration and squared-speed
    # terms are essential; a frozen external phase would discard them.
    angle, speed, acceleration = s.symbols("angle speed acceleration", real=True)
    phasor = s.cos(angle) + s.I * s.sin(angle)
    first = s.diff(phasor, angle) * speed
    second = s.diff(first, angle) * speed + s.diff(first, speed) * acceleration
    _zero(second - phasor * (s.I * acceleration - speed**2))
    _zero(s.re(second) + s.cos(angle) * speed**2 + s.sin(angle) * acceleration)
    _zero(s.im(second) + s.sin(angle) * speed**2 - s.cos(angle) * acceleration)


def test_independent_relative_chain_rule_and_corrector_cancel_at_the_moving_minimum():
    p, q, root, delta, C, S = s.symbols("p q root delta C S", real=True)
    Cdot, Sdot, root_rate, r_rate, r_acceleration = s.symbols(
        "Cdot Sdot root_rate r_rate r_acceleration", real=True
    )
    gamma, kappa, k, h = s.symbols("gamma kappa k h", positive=True)
    potential = (
        1 - s.cos(delta) + C * (1 - s.cos(gamma * delta)) + S * s.sin(gamma * delta)
    )
    force = s.diff(potential, delta)
    actual_force = force.subs(delta, root + q)
    reference_force = force.subs(delta, root)
    relative = (
        kappa * p**2 / 2 + potential.subs(delta, root + q) - potential.subs(delta, root)
    )
    rates = {
        p: -k * actual_force - r_rate,
        q: k * kappa * p - root_rate,
        root: root_rate,
        C: Cdot,
        S: Sdot,
    }
    derivative = sum(
        s.diff(relative, coordinate) * rate for coordinate, rate in rates.items()
    )
    phase_work = (s.cos(gamma * root) - s.cos(gamma * (root + q))) * Cdot + (
        s.sin(gamma * (root + q)) - s.sin(gamma * root)
    ) * Sdot
    _zero(derivative + kappa * p * r_rate - phase_work + reference_force * root_rate)
    # Criticality removes the displayed reference_force term. The admitted
    # divided differences then give actual_force=h*q and phase_work=h*b*q.
    b, b_rate, db = s.symbols("b b_rate db", real=True)
    corrector = -(q * r_rate + p * b) / k
    corrector_rate = (
        s.diff(corrector, q) * (k * kappa * p - root_rate)
        + s.diff(corrector, p) * (-k * h * q - r_rate)
        + s.diff(corrector, r_rate) * r_acceleration
        + s.diff(corrector, b) * b_rate
    )
    relative_rate = -kappa * p * r_rate + q * h * b
    remainder = (q * (r_acceleration - r_rate * db) + p * b_rate) / k
    _zero((relative_rate - corrector_rate - remainder).subs(b, -root_rate + q * db))
    assert remainder.subs({p: 0, q: 0}) == 0
    _zero(corrector_rate.subs({p: 0, q: 0, b: -root_rate}, simultaneous=True))


def test_divided_differences_have_smooth_zero_limits_and_nonzero_reference_derivatives():
    delta, root, q, C, S, Cdot, Sdot = s.symbols(
        "delta root q C S Cdot Sdot", real=True
    )
    gamma = s.symbols("gamma", positive=True)
    potential = (
        1 - s.cos(delta) + C * (1 - s.cos(gamma * delta)) + S * s.sin(gamma * delta)
    )
    force = s.diff(potential, delta)
    coefficient_work = s.diff(potential, C) * Cdot + s.diff(potential, S) * Sdot
    h = (force.subs(delta, root + q) - force.subs(delta, root)) / q
    a = (
        coefficient_work.subs(delta, root + q) - coefficient_work.subs(delta, root)
    ) / q
    h0 = s.diff(force, delta).subs(delta, root)
    h1 = s.diff(force, delta, 2).subs(delta, root) / 2
    a0 = s.diff(coefficient_work, delta).subs(delta, root)
    a1 = s.diff(coefficient_work, delta, 2).subs(delta, root) / 2
    _zero(s.limit(h, q, 0) - h0)
    _zero(s.limit(a, q, 0) - a0)
    _zero(s.limit((h - h0) / q, q, 0) - h1)
    _zero(s.limit((a - a0) / q, q, 0) - a1)
    # Differentiate criticality independently. h0>=m>0 is a domain premise,
    # so quotient composition is smooth; no numerical q==0 fallback is used.
    root_rate = (
        -(s.diff(force, C) * Cdot + s.diff(force, S) * Sdot).subs(delta, root) / h0
    )
    _zero(a0 / h0 + root_rate)
    H0 = s.symbols("H0", positive=True)
    H1, A0, A1 = s.symbols("H1 A0 A1", real=True)
    ratio_jet = (A0 + A1 * q) / (H0 + H1 * q)
    db0 = s.limit((ratio_jet - A0 / H0) / q, q, 0)
    _zero(db0 - (A1 * H0 - A0 * H1) / H0**2)
    # The actual first coefficient is obtained from h1,a1 above, and need
    # not vanish. In particular b(0)=-root_rate is generally nonzero.
    _zero((a0 / h0).subs({root: 0, C: 1, S: 0}) - gamma * Sdot / (1 + gamma**2))


def test_corrector_and_remainder_bounds_have_weighted_square_remainders():
    q, p, r_rate, r_acceleration, b, b_rate, db = s.symbols(
        "q p r_rate r_acceleration b b_rate db", real=True
    )
    m, kappa, k = s.symbols("m kappa k", positive=True)
    metric_squared = m * q**2 + kappa * p**2
    for left, right in ((r_rate, b), (r_acceleration - r_rate * db, b_rate)):
        projection = (q * left + p * right) / k
        bound_squared = (left**2 / m + right**2 / kappa) / k**2
        _zero(
            m * kappa * k**2 * (metric_squared * bound_squared - projection**2)
            - (kappa * p * left - m * q * right) ** 2
        )
    # Coercivity gives metric_squared<=Q² on the same full domain. The two
    # bounds consume different rates: the second cannot omit accelerations
    # or the moving-reference quotient derivative.


def test_coefficient_displacement_bridge_uses_current_chart_and_signed_projection():
    delta, root, C, S, dC, dS, dV = s.symbols("delta root C S dC dS dV", real=True)
    gamma, kappa, m, h = s.symbols("gamma kappa m h", positive=True)
    force = s.sin(delta) + gamma * (C * s.sin(gamma * delta) + S * s.cos(gamma * delta))
    current = force.subs({C: C + dC, S: S + dS}, simultaneous=True)
    projection = dC * s.sin(gamma * root) + dS * s.cos(gamma * root)
    _zero((current - force).subs(delta, root) - gamma * projection)
    # Both endpoints are critical for their own coefficients. The current
    # force's positive secant h>=m gives h*Delta_root=-gamma*projection.
    root_displacement = -gamma * projection / h
    form_displacement = gamma * dV / kappa
    allowance = gamma**2 * (dV**2 / kappa + (dC**2 + dS**2) / m)
    actual_metric = m * root_displacement**2 + kappa * form_displacement**2
    perpendicular = dC * s.cos(gamma * root) - dS * s.sin(gamma * root)
    remainder = (h**2 - m**2) * (dC**2 + dS**2) + m**2 * perpendicular**2
    _zero(m * h**2 * (allowance - actual_metric) / gamma**2 - remainder)
    excess = s.symbols("excess", nonnegative=True)
    _zero(
        remainder.subs(h, m + excess)
        - excess * (2 * m + excess) * (dC**2 + dS**2)
        - m**2 * perpendicular**2
    )
    # This is a bound at each admitted time. A whole-window conclusion
    # requires displacement bounds at every intermediate time, not just
    # equality of the final coefficients with their initial values.


def test_complete_state_can_have_negative_corrected_storage_with_positive_excitation(
    full_tree,
):
    p = full_tree
    contrast = s.symbols("contrast", positive=True)
    parameters = dict.fromkeys(p.mobilities + (p.w, p.v, p.z), 1)
    # A common shift makes the complete weighted form mean zero. It changes
    # neither differences nor the actual complete field or coefficients.
    coordinates = dict(
        zip(
            p.forms,
            (
                -contrast / 2 - s.Rational(1, 4),
                contrast / 2 - s.Rational(1, 4),
                -s.Rational(1, 4),
                s.Rational(3, 4),
            ),
        )
    )
    coordinates.update(dict.fromkeys(p.phases, 0))

    def at_state(expression):
        return s.simplify(
            expression.subs(parameters).subs(coordinates, simultaneous=True)
        )

    gamma, kappa, k = (at_state(value) for value in (p.gamma, p.kappa, p.k))
    _zero(
        at_state(sum(form / mobility for form, mobility in zip(p.forms, p.mobilities)))
    )
    _zero(at_state(p.V))
    _zero(at_state(p.alpha))
    _zero(at_state(p.derivative(p.V)))
    alpha_rate = at_state(p.derivative(p.alpha))
    _zero(alpha_rate - 3 * contrast / 4 + 1)
    # Here C=1,S=0 and delta*=r*=0. The fixed chart is admitted by
    # sin(1)>1/2 and cos(1)>1/4; no source chart or trajectory is evaluated.
    curvature = 1 + gamma**2
    b = gamma * alpha_rate / curvature
    relative = kappa * contrast**2 / 2
    corrector = -contrast * b / k
    corrected = relative - corrector
    _zero(corrected - s.Rational(31, 40) * contrast**2 + contrast / 5)
    assert relative.subs(contrast, s.Rational(1, 10)) == s.Rational(1, 160)
    assert corrected.subs(contrast, s.Rational(1, 10)) == -s.Rational(49, 4000)


def test_signed_initial_corrector_gives_valid_quadratic_upper_and_conditional_lower_bounds():
    Q0, epsilon, remainder_budget = s.symbols(
        "Q0 epsilon remainder_budget", nonnegative=True
    )
    P0 = s.symbols("P0", real=True)
    A = epsilon + remainder_budget
    C0 = Q0**2 - 2 * P0
    radicand = A**2 + C0
    # |P0|<=epsilon*Q0 implies this nonnegative slack; C0 itself need not
    # be nonnegative because L-P has no general positivity theorem.
    slack = s.symbols("slack", nonnegative=True)
    admitted_radicand = radicand.subs(P0, epsilon * Q0 - slack)
    decomposition = (
        (Q0 - epsilon) ** 2
        + 2 * epsilon * remainder_budget
        + remainder_budget**2
        + 2 * slack
    )
    _zero(admitted_radicand - decomposition)
    assert decomposition.is_nonnegative
    U = A + s.sqrt(radicand)
    _zero(U**2 - 2 * A * U - C0)
    gap = s.symbols("gap", positive=True)
    _zero((U + gap) ** 2 - 2 * A * (U + gap) - C0 - gap * (2 * s.sqrt(radicand) + gap))
    # The supremum inequality Qsup²<=C0+2A*Qsup excludes Qsup>U.
    # Initial compatibility also places Q0 below this same upper root.
    _zero(
        (radicand - (Q0 - A) ** 2).subs(P0, epsilon * Q0 - slack)
        - 2 * (remainder_budget * Q0 + slack)
    )
    boundary = {Q0: 1, epsilon: 1, P0: 1, remainder_budget: 0}
    assert C0.subs(boundary) == -1
    assert radicand.subs(boundary) == 0
    assert U.subs(boundary) == 1

    D = C0 - 2 * remainder_budget * U
    Q = s.symbols("Q", nonnegative=True)
    corrector, integrated_remainder = s.symbols(
        "corrector integrated_remainder", real=True
    )
    transport_residual = Q**2 - C0 - 2 * corrector - 2 * integrated_remainder
    # Exact transport makes the residual zero. The two remaining terms
    # are nonnegative under P>=-epsilon*Q and integral(R)>=-R_T*U.
    _zero(
        Q**2
        + 2 * epsilon * Q
        - D
        - transport_residual
        - 2 * (corrector + epsilon * Q)
        - 2 * (integrated_remainder + remainder_budget * U)
    )
    positive_D = s.symbols("positive_D", positive=True)
    floor = s.sqrt(epsilon**2 + positive_D) - epsilon
    _zero(floor**2 + 2 * epsilon * floor - positive_D)
    _zero(floor - positive_D / (s.sqrt(epsilon**2 + positive_D) + epsilon))
    assert (positive_D / (s.sqrt(epsilon**2 + positive_D) + epsilon)).is_positive
    # For D<=0, Q=0 satisfies the inequality. No positive floor follows;
    # squaring a negative allowance would manufacture an invalid signal.
    deficit = s.symbols("deficit", nonnegative=True)
    assert (Q**2 + 2 * epsilon * Q + deficit).subs(Q, 0) == deficit
