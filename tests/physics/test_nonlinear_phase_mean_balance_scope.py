"""Nonlinear mean accounting for the conditional fresh-pressure comparison.

The existing optional phase formula is paired hypothetically with refreshed
canonical pressure on the unit repeated prism. This is not the implemented
independent-pressure runtime. Exact domain bounds constrain the theorem,
not a new evolution law; no trajectory or forward invariance is asserted.
"""

import pytest

from tests.physics._internal_mode_fixture import (
    Q_MODE,
    _phase_support,
    _prepared_phase_geometry,
)


def test_complete_energy_chain_rule_retains_the_nonlinear_common_source_work():
    s = pytest.importorskip("sympy")
    e, k, alignment, alpha, beta = s.symbols("e k alignment alpha beta", positive=True)
    h0, h1, eta0, eta1, q, q_rate = s.symbols("h0 h1 eta0 eta1 q q_rate", real=True)
    h, eta = s.Matrix([h0, h1, -h0 - h1]), s.Matrix([eta0, eta1, -eta0 - eta1])
    _, rows = _phase_support()
    repeated = eta.col_join(eta)
    current_fine = s.Matrix(
        [
            sum(s.sin(repeated[j] - repeated[i]) for j in row) / len(row)
            for i, row in enumerate(rows)
        ]
    )
    current = current_fine[:3, :]
    assert current_fine[3:, :] == current
    assert s.expand_trig(sum(current)).simplify() == 0
    potential = sum(
        1 - s.cos(eta[i] - eta[j]) for i in range(3) for j in range(i + 1, 3)
    )
    for i, coordinate in enumerate((eta0, eta1)):
        assert (
            s.simplify(s.diff(potential, coordinate) + 3 * (current[i] - current[2]))
            == 0
        )

    def force(value):
        return alpha * s.sin(s.pi * value) + beta * value

    def primitive(value):
        return alpha * (1 - s.cos(s.pi * value)) / s.pi + beta * value**2 / 2

    pressure = h + q * s.ones(3, 1)
    response = pressure.applyfunc(force)
    projection = s.eye(3) - s.ones(3) / 3
    phase_rate = projection * response + alignment * current
    h_rate = -e * h - k * phase_rate
    assert s.expand(sum(phase_rate)) == 0
    assert s.expand(sum(h_rate)) == 0
    energy = (
        sum(primitive(value) for value in pressure)
        - 3 * primitive(q)
        + k * alignment * potential / 3
    )
    derivative = (
        sum(
            s.diff(energy, coordinate) * h_rate[i]
            for i, coordinate in enumerate((h0, h1))
        )
        + sum(
            s.diff(energy, coordinate) * phase_rate[i]
            for i, coordinate in enumerate((eta0, eta1))
        )
        + s.diff(energy, q) * q_rate
    )
    defect_coefficient = sum(response) - 3 * force(q)
    expected = (
        -e * h.dot(response)
        - k * phase_rate.dot(phase_rate)
        + defect_coefficient * q_rate
    )
    assert s.expand(derivative - expected) == 0
    assert s.simplify(defect_coefficient.subs(alpha, 0)) == 0
    # q is the actual pressure mean w*c(eta); in the displayed chain rule
    # q_rate must therefore be w*Dc(eta)[phase_rate], not an independent law.
    # The complete EPI mean has mu_rate=q and cannot be silently frozen.


def test_zero_instantaneous_mean_can_still_have_positive_or_negative_mean_work():
    s, _, rows, phases, center, _, _, response = _prepared_phase_geometry()
    alpha, beta, weight, alignment = s.symbols(
        "alpha beta weight alignment", positive=True
    )
    eta = s.Matrix([value - center for value in phases[:3]])
    assert sum(eta) == 0
    assert s.simplify(sum(s.sin(value) for value in eta)) == 0
    assert s.simplify(sum(s.cos(value) for value in eta)) == 1 + s.sqrt(3)
    # Reuse the exact canonical phase-response matrix, then take its mean
    # derivative on the repeated lift. The current value c=0 does not make
    # this derivative zero.
    repeated_lift = s.eye(3).col_join(s.eye(3))
    mean_derivative = (
        s.ones(1, 6) * (response - s.eye(6)) * repeated_lift / (6 * s.pi)
    ).applyfunc(s.simplify)
    coefficient = (s.sqrt(3) / 2 - 1) / (3 * s.pi * (1 + s.sqrt(3)))
    assert (mean_derivative - coefficient * s.Matrix([Q_MODE])).applyfunc(
        s.simplify
    ) == s.zeros(1, 3)
    assert coefficient < 0
    current = s.Matrix(
        [
            sum(s.sin(phases[j] - phases[i]) for j in rows[i]) / len(rows[i])
            for i in range(3)
        ]
    )
    assert s.simplify((mean_derivative * current)[0]) == 0

    def force(value):
        return alpha * s.sin(s.pi * value) + beta * value

    small = s.Rational(1, 8)
    f_small, f_double = force(small), force(2 * small)
    assert s.simplify(f_double - 2 * f_small) < 0
    assert s.simplify(f_double + f_small) > 0
    projection = s.eye(3) - s.ones(3) / 3
    for h, pair_factor, work_sign in (
        (s.Matrix([2 * small, -small, -small]), 1, 1),
        (s.Matrix([small, small, -2 * small]), 2, -1),
    ):
        assert sum(h) == 0 and all(abs(value) <= s.Rational(1, 4) for value in h)
        response_at_state = h.applyfunc(force)
        velocity = projection * response_at_state + alignment * current
        actual_mean_rate = weight * (mean_derivative * velocity)[0]
        assert (
            s.simplify(
                actual_mean_rate
                - weight * coefficient * pair_factor * (f_double + f_small)
            )
            == 0
        )
        defect = s.simplify(sum(response_at_state) * actual_mean_rate)
        assert s.simplify(work_sign * defect) > 0
    # Both witnesses are strict U3. Their defect has opposite signs even
    # though q=0 at each point, so a universal nonpositive sign is invalid.
    # This is an exact local accounting control, not an executed curve.


def test_taylor_simplex_and_young_bounds_give_a_sufficient_nonlinear_gate():
    s = pytest.importorskip("sympy")
    parameter = s.Symbol("parameter", real=True)
    q, displacement = s.symbols("q displacement", real=True)
    function = s.Function("function")
    along_segment = function(q + parameter * displacement)
    boundary_primitive = (1 - parameter) * s.diff(
        along_segment, parameter
    ) + along_segment
    assert (
        s.simplify(
            s.diff(boundary_primitive, parameter)
            - (1 - parameter) * s.diff(along_segment, parameter, 2)
        )
        == 0
    )
    boundary = boundary_primitive.subs(parameter, 1) - boundary_primitive.subs(
        parameter, 0
    )
    assert (
        s.simplify(
            boundary
            - function(q + displacement)
            + function(q)
            + displacement * s.diff(function(q), q)
        )
        == 0
    )
    # Applied to F with F'=f and f'>=ell, integration on [0,1]
    # gives the strong-convexity/Jensen remainder >=ell*h_i^2/2.
    # Applied to f with |f''|<=M, the same identity gives a remainder
    # <=M*h_i^2/2. Summing cancels linear terms because sum(h)=0.
    assert s.integrate(1 - parameter, (parameter, 0, 1)) == s.Rational(1, 2)

    a0, a1 = s.symbols("a0 a1", nonnegative=True)
    probabilities = (a0, a1, 1 - a0 - a1)
    pairs = sum(
        probabilities[i] * probabilities[j] for i in range(3) for j in range(i + 1, 3)
    )
    centered_squared = sum((value - s.Rational(1, 3)) ** 2 for value in probabilities)
    assert s.expand(centered_squared - s.Rational(2, 3) + 2 * pairs) == 0
    # On the strict phase chart, a_i=cos(eta_i-Arg S)/|S| are positive
    # and sum to one. All pair products are nonnegative, including the
    # declared third probability. Thus ||Dc||<=sqrt(2/3)/pi on 1-perp.

    damping, k, ratio = s.symbols("damping k ratio", positive=True)
    norm_h, norm_v = s.symbols("norm_h norm_v", nonnegative=True)
    cross = 2 * ratio * s.sqrt(damping * k)
    loss = damping * norm_h**2 + k * norm_v**2 - cross * norm_h * norm_v
    decomposed = (1 - ratio) * (damping * norm_h**2 + k * norm_v**2) + ratio * (
        s.sqrt(damping) * norm_h - s.sqrt(k) * norm_v
    ) ** 2
    assert s.expand(loss - decomposed) == 0
    # damping=e*ell; cross=w*M*C*H/2 follows from ||h||<=H.
    # cross^2<4*e*ell*k makes ratio<1 and strictly positive loss away
    # from h=v=0. This is a sufficient domain bound, not a new setting.


def test_monotone_half_pressure_gate_has_a_strict_margin_on_the_full_strict_chart():
    s = pytest.importorskip("sympy")
    pressure = s.Symbol("pressure", real=True)
    alpha, beta = s.Rational(1, 2), s.Rational(3, 20)
    e, weight = s.Rational(1, 2), s.Rational(1, 4)
    k = weight / s.pi
    force = alpha * s.sin(s.pi * pressure) + beta * pressure
    assert s.diff(force, pressure) == alpha * s.pi * s.cos(s.pi * pressure) + beta
    assert s.diff(force, pressure, 2) == -alpha * s.pi**2 * s.sin(s.pi * pressure)
    assert s.calculus.util.function_range(
        s.cos(s.pi * pressure),
        pressure,
        s.Interval(-s.Rational(1, 2), s.Rational(1, 2)),
    ) == s.Interval(0, 1)
    lower_slope, second_bound = beta, alpha * s.pi**2
    p0, p1, p2 = s.symbols("p0 p1 p2", real=True)
    mean = (p0 + p1 + p2) / 3
    variance = sum((value - mean) ** 2 for value in (p0, p1, p2))
    assert (
        s.expand(variance - sum(value**2 for value in (p0, p1, p2)) + 3 * mean**2) == 0
    )
    # |p_i|<=1/2 implies |q|<=1/2 and ||h||<=sqrt(3)/2. The segment
    # from q to each p_i remains in this monotonicity/convexity interval.
    h_bound, mean_gradient_bound = s.sqrt(3) / 2, s.sqrt(s.Rational(2, 3)) / s.pi
    cross = s.simplify(weight * second_bound * mean_gradient_bound * h_bound / 2)
    assert cross == s.sqrt(2) * s.pi / 32
    assert s.simplify(cross**2) == s.pi**2 / 512
    assert s.simplify(4 * e * lower_slope * k - lower_slope / (2 * s.pi)) == 0
    ratio_squared = s.simplify(cross**2 / (4 * e * lower_slope * k))
    assert ratio_squared == 5 * s.pi**3 / 192
    # The elementary bound pi<16/5 proves the strict criterion throughout
    # this monotonicity interval, without a selected small phase radius.
    assert s.pi < s.Rational(16, 5)
    assert s.Rational(5, 192) * s.Rational(16, 5) ** 3 == s.Rational(64, 75) < 1
    assert s.simplify(ratio_squared - s.Rational(64, 75)) < 0
    assert s.simplify(4 * e * lower_slope * k - cross**2) > 0
    # The decimal coefficients here are explicitly declared exact rationals;
    # this is not differentiation of a binary64 pipeline. Strict energy
    # decay excludes recurrent nonstationary contrast contained in this
    # domain, but neither the domain's invariance nor the full optional
    # runtime, source generation, or complete-state attraction is proved.
