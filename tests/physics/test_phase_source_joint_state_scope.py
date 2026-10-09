"""Conditional joint phase/form identities on the existing unit prism.

The fine nodal equation supplies the form row, not the phase-response row.
Real coefficients a,b below describe alternative declared linear responses;
none is selected as a TNFR law. The identities are exact for that assumed
linear family, or small-signal statements if only its linearization is given.
The full mean still obeys mu_dot=w*c(eta). No trajectory is integrated.
"""

import pytest

from tests.physics._internal_mode_fixture import (
    Q_MODE,
    P,
    _exact_generator,
    _graph,
    _phase_support,
)
from tnfr.physics.support_transport import observe_support_transport


def test_fine_form_row_leaves_source_area_and_acceleration_response_undetermined():
    s = pytest.importorskip("sympy")
    e, k = s.symbols("e k", positive=True)
    xr, xi, yr, yi, hr, hi, mu, common = s.symbols(
        "xr xi yr yi hr hi mu common", real=True
    )
    form, phase, phase_rate = s.Matrix([xr, xi]), s.Matrix([yr, yi]), s.Matrix([hr, hi])
    basis = s.Matrix.hstack(
        s.Matrix(P * 2) / s.sqrt(2), s.Matrix(Q_MODE * 2) / s.sqrt(6)
    )
    generator = s.Matrix(_exact_generator(observe_support_transport(_graph())))
    assert generator * basis == -basis
    assert basis.T * basis == 2 * s.eye(2)
    # On the repeated regular phase lift, c=Arg(sum exp(i*eta_j))/pi.
    # It is retained here as a readout symbol, not an independent source law.
    phase_source = -basis * phase / s.pi + common * s.ones(6, 1)
    fine_form = mu * s.ones(6, 1) + basis * form
    fine_rate = e * generator * fine_form + s.pi * k * phase_source
    form_rate = (basis.T * fine_rate / 2).applyfunc(s.simplify)
    assert form_rate == -e * form - k * phase
    assert s.simplify((s.ones(1, 6) * fine_rate)[0] / 6) == s.pi * k * common

    area = xr * yi - xi * yr
    joint = s.Matrix([xr, xi, yr, yi])
    area_rate = (s.Matrix([area]).jacobian(joint) * form_rate.col_join(phase_rate))[0]
    assert s.expand(area_rate + e * area - (xr * hi - xi * hr)) == 0

    # Two equivariant source responses at the SAME initial state agree on
    # the nodal velocity and differ at the next derivative. This is a jet
    # comparison, not a phase update or an executed trajectory.
    a = s.Symbol("a", positive=True)
    preparation = {xr: s.Rational(1, 8), xi: 0, yr: 0, yi: 0, mu: s.Rational(1, 2)}
    zero_source_rate = s.zeros(2, 1)
    responding_source_rate = a * form
    initial_velocity = form_rate.subs(preparation)
    assert initial_velocity == s.Matrix([-e / 8, 0])
    initial_field = fine_form.subs(preparation)
    assert all(0 < value < 1 for value in initial_field)
    acceleration_a = (-e * form_rate - k * zero_source_rate).subs(preparation)
    acceleration_b = (-e * form_rate - k * responding_source_rate).subs(preparation)
    assert acceleration_b - acceleration_a == s.Matrix([-k * a / 8, 0])


def test_real_joint_response_is_equivariant_without_fixing_form_mirror_crossings():
    s = pytest.importorskip("sympy")
    e, k = s.symbols("e k", positive=True)
    a, b = s.symbols("a b", real=True)
    basis = s.Matrix.hstack(s.Matrix(P) / s.sqrt(2), s.Matrix(Q_MODE) / s.sqrt(6))
    swap = s.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    cycle = s.Matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    reflection = (basis.T * swap * basis).applyfunc(s.simplify)
    rotation = (basis.T * cycle * basis).applyfunc(s.simplify)
    assert reflection == s.diag(-1, 1)  # z -> -conjugate(z)
    assert rotation == s.Matrix(
        [[-s.Rational(1, 2), -s.sqrt(3) / 2], [s.sqrt(3) / 2, -s.Rational(1, 2)]]
    )
    system = s.BlockMatrix(
        [[-e * s.eye(2), -k * s.eye(2)], [a * s.eye(2), b * s.eye(2)]]
    ).as_explicit()
    for spatial in (reflection, rotation):
        joint_action = s.diag(spatial, spatial)
        assert system * joint_action == joint_action * system

    radius, transverse = s.symbols("radius transverse", positive=True)
    state = s.Matrix([0, radius, transverse, 0])
    velocity = system * state
    assert reflection * state[:2, :] == state[:2, :]
    assert reflection * state[2:, :] != state[2:, :]
    assert velocity[0] == -k * transverse != 0
    # Only the joint fixed set (BOTH vectors on the axis) is invariant.
    # This does not install this candidate response or establish recurrence.
    x, y, u, v = s.symbols("x y u v", real=True)
    vectors = s.Matrix([[x, u], [y, v]])
    assert (rotation * vectors).det() == vectors.det()
    assert (reflection * vectors).det() == -vectors.det()
    # Signed area is reflection-odd; c(eta), a symmetric phasor sum readout,
    # is permutation-invariant and cannot be silently identified with it.


def test_linear_alternatives_have_distinct_damping_without_selected_coefficients():
    s = pytest.importorskip("sympy")
    e, k = s.symbols("e k", positive=True)
    a, b = s.symbols("a b", real=True)
    eigenvalue = s.Symbol("eigenvalue")
    matrix = s.Matrix([[-e, -k], [a, b]])
    polynomial = matrix.charpoly(eigenvalue).as_expr()
    expected = eigenvalue**2 + (e - b) * eigenvalue + k * a - e * b
    assert s.expand(polynomial - expected) == 0
    assert matrix.trace() == b - e
    assert matrix.det() == k * a - e * b
    discriminant = s.discriminant(expected, eigenvalue)
    assert s.expand(discriminant - ((e + b) ** 2 - 4 * k * a)) == 0

    damping, stiffness = s.symbols("damping stiffness", positive=True)
    hurwitz = expected.subs({a: (stiffness + e * (e - damping)) / k, b: e - damping})
    assert s.expand(hurwitz) == eigenvalue**2 + damping * eigenvalue + stiffness
    # For a real quadratic these positive coefficients are the strict
    # Hurwitz conditions b<e and ka>eb. Complex roots additionally require
    # (e+b)^2<4ka; pure imaginary roots require b=e and ka>e^2.
    omega = s.Symbol("omega", positive=True)
    neutral = expected.subs({b: e, a: (e**2 + omega**2) / k})
    assert s.expand(neutral) == eigenvalue**2 + omega**2
    for supplied_b, real_part in (
        (0, -s.Rational(1, 2)),
        (1, 0),
        (s.Rational(3, 2), s.Rational(1, 4)),
    ):
        alternative = matrix.subs({e: 1, k: 1, a: 2, b: supplied_b})
        eigenvalues = tuple(alternative.eigenvals())
        assert len(eigenvalues) == 2
        assert all(
            s.re(value) == real_part and s.im(value) != 0 for value in eigenvalues
        )
    # Stable, neutral, and unstable alternatives are logical controls of
    # the unselected family, not observed engine regimes or fitted laws.


def test_neutral_joint_family_has_positive_invariant_only_under_its_stated_gate():
    s = pytest.importorskip("sympy")
    e, k = s.symbols("e k", positive=True)
    a, b, x, y, u, v = s.symbols("a b x y u v", real=True)
    state = s.Matrix([x, y, u, v])
    velocity = s.Matrix([-e * x - k * u, -e * y - k * v, a * x + b * u, a * y + b * v])
    form_squared, phase_squared, pairing = x**2 + y**2, u**2 + v**2, x * u + y * v
    invariant = a * form_squared + 2 * e * pairing + k * phase_squared
    derivative = (s.Matrix([invariant]).jacobian(state) * velocity)[0]
    assert s.expand(derivative - 2 * (b - e) * (e * pairing + k * phase_squared)) == 0
    assert s.expand(derivative.subs(b, e)) == 0
    completed = (
        k * ((u + e * x / k) ** 2 + (v + e * y / k) ** 2)
        + (a - e**2 / k) * form_squared
    )
    assert s.expand(invariant - completed) == 0
    metric = s.Matrix([[a, e], [e, k]])
    assert metric.det() == k * a - e**2
    omega = s.Symbol("omega", positive=True)
    positive = invariant.subs(a, (e**2 + omega**2) / k)
    positive_completed = (
        k * ((u + e * x / k) ** 2 + (v + e * y / k) ** 2) + omega**2 * form_squared / k
    )
    assert s.expand(positive - positive_completed) == 0
    # k>0 and ka>e^2 make this strictly positive off the joint origin.
    # At equality a nonzero null direction prevents that promotion.
    boundary = invariant.subs({a: e**2 / k, u: -e * x / k, v: -e * y / k})
    assert s.simplify(boundary) == 0


def test_eliminating_phase_retains_response_memory_and_its_initial_condition():
    s = pytest.importorskip("sympy")
    e, k = s.symbols("e k", positive=True)
    a, b, initial_phase = s.symbols("a b initial_phase")
    time = s.Symbol("time", real=True)
    integration_time = s.Dummy("integration_time", real=True)
    form = s.Function("form")
    phase = s.Function("phase")
    second_derivative = -e * s.diff(form(time), time) - k * (
        a * form(time) + b * phase(time)
    )
    eliminated = second_derivative.subs(
        phase(time), -(s.diff(form(time), time) + e * form(time)) / k
    )
    assert (
        s.simplify(
            eliminated
            + (e - b) * s.diff(form(time), time)
            + (k * a - e * b) * form(time)
        )
        == 0
    )
    memory = s.exp(b * time) * (
        initial_phase
        + a
        * s.Integral(
            s.exp(-b * integration_time) * form(integration_time),
            (integration_time, 0, time),
        )
    )
    assert s.simplify(s.diff(memory, time) - a * form(time) - b * memory) == 0
    assert memory.subs(time, 0).doit() == initial_phase
    inherited_form_rate = -e * form(time) - k * memory
    assert inherited_form_rate.subs(time, 0).doit() == -e * form(0) - k * initial_phase
    assert (
        s.simplify(s.diff(inherited_form_rate, initial_phase) + k * s.exp(b * time))
        == 0
    )
    # The Volterra representation remembers the independently supplied
    # phase preparation and assumed a,b response; it does not derive them.
    # Neither this elimination nor Q removes the complete mean equation
    # mu_dot=w*c(eta), the regular phase chart, or primitive common phase.


def test_existing_phase_formula_with_fresh_pressure_has_real_damped_linear_modes():
    s = pytest.importorskip("sympy")
    e, k, alpha, beta, gamma, coupling = s.symbols(
        "e k alpha beta gamma coupling", positive=True
    )
    epsilon = s.Symbol("epsilon", real=True)
    x, y, u, v = s.symbols("x y u v", real=True)
    form, phase = s.Matrix([x, y]), s.Matrix([u, v])
    graph, rows = _phase_support()
    basis = s.Matrix.hstack(
        s.Matrix(P * 2) / s.sqrt(2), s.Matrix(Q_MODE * 2) / s.sqrt(6)
    )
    generator = s.Matrix(_exact_generator(observe_support_transport(graph)))
    assert generator * basis == -basis
    assert all(len(row) == 3 for row in rows)
    contrast = basis * phase
    # Differentiate the actual unique-neighbor mean-sine formula, including
    # the matched vertical edge (zero difference on repeated triples).
    current = s.Matrix(
        [
            sum(s.sin(epsilon * (contrast[j] - contrast[i])) for j in neighbors)
            / len(neighbors)
            for i, neighbors in enumerate(rows)
        ]
    )
    linear_current = current.diff(epsilon).subs(epsilon, 0).applyfunc(s.simplify)
    assert linear_current == -contrast
    # The canonical phasor common source starts at cubic order here. Its
    # fresh-pressure linearization is -e*B*z-k*B*zeta, not a free pressure.
    pressure = -e * basis * form - k * contrast
    optional_phase_rate = s.Matrix(
        [
            alpha * s.sin(s.pi * epsilon * pressure[i])
            + beta * epsilon * pressure[i]
            + gamma * coupling * current[i]
            for i in range(6)
        ]
    )
    projected = (
        basis.T * optional_phase_rate.diff(epsilon).subs(epsilon, 0) / 2
    ).applyfunc(s.simplify)
    h, g = alpha * s.pi + beta, gamma * coupling
    expected = -h * e * form - (h * k + g) * phase
    assert (projected - expected).applyfunc(s.simplify) == s.zeros(2, 1)
    matrix = s.Matrix([[-e, -k], [-h * e, -(h * k + g)]])
    total_damping = e + h * k + g
    discriminant = (e - h * k - g) ** 2 + 4 * k * h * e
    assert s.simplify(matrix.trace() + total_damping) == 0
    assert s.simplify(matrix.det() - e * g) == 0
    assert s.expand(matrix.trace() ** 2 - 4 * matrix.det() - discriminant) == 0
    assert s.expand(total_damping**2 - discriminant - 4 * e * g) == 0
    # Positivity gives 0<sqrt(discriminant)<total_damping: both eigenvalues
    # are real and strictly negative. Each repeats across the two modes.
    # This is the previously declared FRESH-PRESSURE comparison. The actual
    # optional runtime evolves pressure independently and is not certified
    # by this substitution or by a linearization of the reduced comparison.
