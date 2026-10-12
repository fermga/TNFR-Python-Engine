"""Static controls for local dynamics at the organized constitutive transition.

The controls retain both evolution rows, the critical projection and the
original clock. They evaluate no source, trajectory or stationary root.
"""

import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation


@pytest.fixture(scope="module", autouse=True)
def no_scientific_regeneration():
    with forbid_organized_response_evaluation():
        yield


def test_critical_full_field_projection_retains_form_initialization():
    """A coupled relative-state example derives the left mode from both rows."""
    gamma, quartic = sp.symbols("gamma quartic", positive=True)
    phase = sp.Matrix(sp.symbols("phase0:3", real=True))
    form = sp.Matrix(sp.symbols("form0:3", real=True))
    laplacian = sp.Matrix([[3, 1, 0], [1, 2, 1], [0, 1, 2]])
    constraints = sp.Matrix([[-2, 1, 0], [1, 0, 1]])
    stiffness = constraints.T * sp.diag(2, 3) * constraints
    direction = sp.Matrix([1, 2, -1])
    norm = (direction.T * direction)[0]
    assert all(laplacian[:size, :size].det() > 0 for size in range(1, 4))
    assert stiffness.rank() == 2
    assert stiffness * direction == sp.zeros(3, 1)
    assert laplacian * stiffness != stiffness * laplacian

    coordinate = (direction.T * phase)[0] / norm
    potential = (phase.T * stiffness * phase)[0] / 2 + quartic * coordinate**4 / 4
    gradient = sp.Matrix([sp.diff(potential, value) for value in phase])
    field = (gamma * laplacian * form / 4).col_join(
        -(laplacian * form + gamma * gradient) / 4
    )
    state = phase.col_join(form)
    jacobian = field.jacobian(state).subs(dict.fromkeys(state, 0))
    expected = (sp.zeros(3).row_join(gamma * laplacian / 4)).col_join(
        (-gamma * stiffness / 4).row_join(-laplacian / 4)
    )
    assert jacobian == expected
    right = direction.col_join(sp.zeros(3, 1))
    left = direction.col_join(gamma * direction).T / norm
    assert jacobian * right == sp.zeros(6, 1)
    assert left * jacobian == sp.zeros(1, 6)
    assert (left * right)[0] == 1
    assert (
        sp.simplify(
            (left * field)[0] + gamma**2 * (direction.T * gradient)[0] / (4 * norm)
        )
        == 0
    )

    phase_acceleration = gamma * laplacian * field[3:, :] / 4
    assert sp.simplify(
        laplacian.inv() * phase_acceleration
        + field[:3, :] / 4
        + gamma**2 * gradient / 16
    ) == sp.zeros(3, 1)

    # Equal phases do not imply equal slow initial data. Removing form from
    # the proposed projection also destroys the left-nullvector identity.
    epsilon = sp.symbols("epsilon", positive=True)
    for sign in (-1, 1):
        preparation = sp.zeros(3, 1).col_join(sign * epsilon * direction)
        assert (left * preparation)[0] == sign * gamma * epsilon
    phase_only = direction.col_join(sp.zeros(3, 1)).T / norm
    assert phase_only * jacobian != sp.zeros(1, 6)
    assert (phase_only * preparation)[0] == 0


def test_actual_weighted_product_support_gives_origin_velocity_and_physical_norm():
    """The 54-node edge sum fixes the clock factor and nine-copy normalization."""
    eta, gamma, epsilon = sp.symbols("eta gamma epsilon", positive=True)
    shape = list(sp.symbols("shape0:4", real=True))
    shape.append(-sum(shape))
    wheel_direction = sp.Matrix(
        [sp.Rational(1, 6) + value for value in shape] + [-sp.Rational(5, 6)]
    )
    direction = sp.Matrix([value for value in wheel_direction for _ in range(9)])
    edges = (
        [
            (9 * child + slot, 9 * child + (slot + 1) % 9, 1)
            for child in range(6)
            for slot in range(9)
        ]
        + [
            (9 * child + slot, 9 * ((child + 1) % 5) + slot, 1)
            for child in range(5)
            for slot in range(9)
        ]
        + [
            (9 * child + slot, 45 + slot, eta)
            for child in range(5)
            for slot in range(9)
        ]
    )
    laplacian = sp.zeros(54)
    for left, right, weight in edges:
        laplacian[left, left] += weight
        laplacian[right, right] += weight
        laplacian[left, right] -= weight
        laplacian[right, left] -= weight
    assert len(edges) == 144
    assert laplacian == laplacian.T
    assert laplacian * sp.ones(54, 1) == sp.zeros(54, 1)
    origin = sp.Matrix([[sp.Rational(1, 45)] * 45 + [-sp.Rational(1, 9)] * 9])
    assert sp.simplify(sum(direction)) == 0
    assert sp.simplify((origin * direction)[0]) == 1
    norm = sp.expand((direction.T * direction)[0])
    shape_norm = sum(value**2 for value in shape)
    assert sp.simplify(norm - 9 * (sp.Rational(5, 6) + shape_norm)) == 0
    for sign in (-1, 1):
        # Both preparations have the identical critical phase; their form
        # perturbations alone produce opposite exact initial phase rates.
        phase_rate = sign * gamma * epsilon * laplacian * direction / 4
        assert (
            sp.simplify((origin * phase_rate)[0])
            == sign * 3 * gamma * eta * epsilon / 2
        )
    mobility = sp.factor(9 * gamma**2 / (4 * norm))
    assert (
        sp.simplify(mobility - gamma**2 / (4 * (sp.Rational(5, 6) + shape_norm))) == 0
    )
    assert mobility.subs(dict.fromkeys(shape[:4], 0)) == 3 * gamma**2 / 10
    assert (
        sp.simplify(
            mobility / (3 * gamma**2 / 10) - 1 / (1 + sp.Rational(6, 5) * shape_norm)
        )
        == 0
    )
    # Dropping the nine wheel copies from only one side changes the answer.
    assert sp.simplify(mobility - gamma**2 / (4 * norm)) != 0


def test_curved_stationary_valley_requires_two_row_invariance_jet():
    """Solve invariance coefficients, retaining the nonstationary shape lag."""
    q, transverse, parameter = sp.symbols("q transverse parameter", real=True)
    gamma, stiffness, cubic, onset, curvature = sp.symbols(
        "gamma stiffness cubic onset curvature", positive=True
    )
    laplacian = sp.Matrix([[2, 1], [1, 3]])
    potential = (
        stiffness * (transverse - 2 * q - curvature * q**2) ** 2 / 2
        + cubic * q**4 / 4
        - onset * parameter * q**2 / 2
    )
    gradient = sp.Matrix([sp.diff(potential, variable) for variable in (q, transverse)])
    hessian = sp.hessian(potential, (q, transverse)).subs(
        {q: 0, transverse: 0, parameter: 0}
    )
    direction = sp.Matrix([1, 2])
    assert hessian * direction == sp.zeros(2, 1)
    assert laplacian * hessian != hessian * laplacian
    phase2, phase3, phase_mixed = sp.symbols("phase2 phase3 phase_mixed")
    speed2, speed3, speed_mixed = sp.symbols("speed2 speed3 speed_mixed")
    form2 = sp.Matrix(sp.symbols("form2_0:2"))
    form3 = sp.Matrix(sp.symbols("form3_0:2"))
    form_mixed = sp.Matrix(sp.symbols("form_mixed0:2"))
    phase_curve = sp.Matrix(
        [q, 2 * q + phase2 * q**2 + phase3 * q**3 + phase_mixed * parameter * q]
    )
    form_curve = form2 * q**2 + form3 * q**3 + form_mixed * parameter * q
    speed = speed2 * q**2 + speed3 * q**3 + speed_mixed * parameter * q
    phase_residual = phase_curve.diff(q) * speed - gamma * laplacian * form_curve / 4
    form_residual = (
        form_curve.diff(q) * speed
        + (laplacian * form_curve + gamma * gradient.subs(transverse, phase_curve[1]))
        / 4
    )
    residual = phase_residual.col_join(form_residual)

    def coefficients(q_order, parameter_order, known):
        return [
            sp.expand(value.subs(known))
            .coeff(parameter, parameter_order)
            .coeff(q, q_order)
            for value in residual
        ]

    second = sp.solve(coefficients(2, 0, {}), [phase2, speed2, *form2], dict=True)
    assert len(second) == 1
    assert second[0][phase2] == curvature
    assert second[0][speed2] == 0
    assert form2.subs(second[0]) == sp.zeros(2, 1)
    third = sp.solve(coefficients(3, 0, second[0]), [phase3, speed3, *form3], dict=True)
    mixed = sp.solve(
        coefficients(1, 1, second[0]),
        [phase_mixed, speed_mixed, *form_mixed],
        dict=True,
    )
    assert len(third) == len(mixed) == 1
    norm = (direction.T * direction)[0]
    assert third[0][speed3] == -(gamma**2) * cubic / (4 * norm)
    assert mixed[0][speed_mixed] == gamma**2 * onset / (4 * norm)
    assert third[0][phase3] == 2 * cubic / (5 * stiffness)
    assert mixed[0][phase_mixed] == -2 * onset / (5 * stiffness)
    known = second[0] | third[0] | mixed[0]
    for orders in ((2, 0), (3, 0), (1, 1)):
        assert coefficients(*orders, known) == [0] * 4

    # The stationary valley has zero normal gradient, but its static form
    # balance does not make the phase velocity tangent to that valley.
    valley = sp.Matrix([q, 2 * q + curvature * q**2])
    static_gradient = gradient.subs({transverse: valley[1], parameter: 0})
    static_form = -gamma * laplacian.inv() * static_gradient
    static_velocity = sp.simplify(gamma * laplacian * static_form / 4)
    incompatibility = sp.expand((valley.diff(q).row_join(static_velocity)).det())
    assert incompatibility.coeff(q, 3) == gamma**2 * cubic / 2
    assert static_velocity[0].coeff(q, 3) != third[0][speed3]


def test_center_coefficients_critical_decay_branch_rates_and_clock():
    """The leading center polynomial separates decay, branch rates and clocks."""
    q, parameter, remainder = sp.symbols("q parameter remainder", real=True)
    gamma, amplitude, eta, shape_norm, restoring, onset = sp.symbols(
        "gamma amplitude eta shape_norm restoring onset", positive=True
    )
    norm = 9 * (sp.Rational(5, 6) + shape_norm)
    mobility = 9 * gamma**2 / (4 * norm)
    reduced_torque = amplitude * eta**5 * (-onset * parameter * q + restoring * q**3)
    # Nine copies enter the potential gradient; the full-state projection
    # divides by the physical norm, rather than the bare parent coordinate.
    speed = sp.expand(-(gamma**2) * 9 * reduced_torque / (4 * norm))
    linear = mobility * amplitude * eta**5 * onset
    cubic = mobility * amplitude * eta**5 * restoring
    assert sp.simplify(speed - (linear * parameter * q - cubic * q**3)) == 0
    assert sp.simplify(sp.diff(speed, q).subs(q, 0) - linear * parameter) == 0
    branch_square = linear * parameter / cubic
    branch_rate = sp.diff(speed, q).subs(q**2, branch_square)
    assert sp.simplify(branch_rate + 2 * linear * parameter) == 0
    assert sp.simplify(branch_square - onset * parameter / restoring) == 0
    # The existing joint remainder bounds enter the dynamic coefficients
    # through derivatives of the leading torque, not a fitted relaxation row.
    mu = sp.symbols("mu", real=True)
    leading_torque = sp.sin(5 * q) - mu * sp.sin(10 * q) / 2
    mixed_torque = sp.diff(leading_torque, q, mu).subs(q, 0)
    third_torque = sp.diff(leading_torque, q, 3).subs(q, 0)
    assert -mixed_torque - sp.Rational(1, 20) == sp.Rational(99, 20)
    assert -mixed_torque + sp.Rational(1, 20) == sp.Rational(101, 20)
    assert (third_torque.subs(mu, sp.Rational(99, 100)) - 40) / 6 == 55
    assert (third_torque.subs(mu, sp.Rational(101, 100)) + 40) / 6 == 70

    critical_speed = -cubic * q**3 + remainder * q**5
    reciprocal_rate = sp.expand(sp.diff(q**-2, q) * critical_speed)
    assert sp.simplify(reciprocal_rate - 2 * cubic + 2 * remainder * q**2) == 0
    assert sp.simplify(sp.limit(reciprocal_rate, q, 0) - 2 * cubic) == 0
    # tau=(1023/1024)*t_original transforms every rate, including the
    # reciprocal-square slope and branch exponents, in the same direction.
    clock = sp.Rational(1023, 1024)
    original_speed = clock * speed
    assert (
        sp.simplify(sp.diff(original_speed, q).subs(q, 0) - clock * linear * parameter)
        == 0
    )
    original_reciprocal_rate = sp.diff(q**-2, q) * clock * critical_speed
    assert (
        sp.simplify(sp.limit(original_reciprocal_rate, q, 0) - 2 * clock * cubic) == 0
    )
    assert sp.simplify(original_speed - speed / clock) != 0
