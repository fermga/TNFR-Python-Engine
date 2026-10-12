"""Exact controls for conditional collective boundary constraints and kinetics.

The examples check admitted quadratic forms, full rows and local constitutive
counterexamples. They use no unknown stationary state, trajectory or response.
"""

from fractions import Fraction as Q

import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics.linear_observation import derive_coordinate_memory
from tnfr.physics._sine_class_constitutive_comparison import _edge_law


@pytest.fixture(scope="module", autouse=True)
def no_scientific_regeneration():
    with forbid_organized_response_evaluation():
        yield


def _jacobian(laplacian, stiffness, gamma):
    size = laplacian.rows
    return (
        sp.zeros(size)
        .row_join(gamma * laplacian / 4)
        .col_join((-gamma * stiffness / 4).row_join(-laplacian / 4))
    )


def test_positive_constraints_and_normal_compliance_schur_identity():
    """Independent positive forms give the nullity and both Schur expressions."""
    old = sp.diag(0, 0, 2, 5)
    soft, normal = sp.eye(4)[:, :2], sp.eye(4)[:, 2:]
    boundary = sp.Matrix([[1, 0, 1, 2], [0, 1, -1, 1], [1, 1, 0, -1]])
    normal_stiffness = normal.T * old * normal
    for count in (1, 2, 3):
        trace = boundary[:count, :]
        weights = sp.diag(4, 9, 1)[:count, :count]
        joined = old + trace.T * weights * trace
        constraint, compliance = trace * soft, trace * normal
        normal_block = normal.T * joined * normal
        mixed = soft.T * joined * normal
        schur = soft.T * joined * soft - mixed * normal_block.inv() * mixed.T
        woodbury = (
            constraint.T
            * (weights.inv() + compliance * normal_stiffness.inv() * compliance.T).inv()
            * constraint
        )
        assert schur == woodbury
        assert len(joined.nullspace()) == 2 - constraint.rank()
        assert schur.nullspace() == constraint.nullspace()
        transform = soft.row_join(normal)
        elimination = sp.eye(4)
        elimination[2:, :2] = -normal_block.inv() * mixed.T
        assert (
            elimination.T * transform.T * joined * transform * elimination
            == sp.diag(schur, normal_block)
        )
        lost_stiffness = constraint.T * weights * constraint - schur
        assert lost_stiffness == mixed * normal_block.inv() * mixed.T
        assert lost_stiffness != sp.zeros(2)
        assert all(value >= 0 for value in lost_stiffness.diagonal())
        assert lost_stiffness.det() >= 0


def test_zero_or_signed_edge_curvature_changes_the_constraint_theorem():
    """Positive conductance does not guarantee positive phase curvature."""
    old, boundary = sp.diag(0, 0, 2), sp.Matrix([[1, 0, 1], [0, 1, -1]])
    weights, square_root = sp.diag(4, 0), sp.diag(2, 0)
    joined = old + boundary.T * weights * boundary
    true_constraints = old.col_join(square_root * boundary)
    assert joined.nullspace() == true_constraints.nullspace()
    assert len(joined.nullspace()) == 1
    assert old.col_join(boundary).nullspace() == []

    # Signed boundary curvature can cancel a positive old normal direction,
    # even when the resulting sum is still positive semidefinite.
    old_positive = sp.eye(2)
    signed_sum = old_positive + sp.diag(-1, 2)
    assert signed_sum == sp.diag(0, 3)
    assert len(signed_sum.nullspace()) == 1
    assert old_positive.nullspace() == []
    deformation = sp.symbols("deformation", nonnegative=True)
    _, opposite = _edge_law(sp.Integer(-1), sp.Integer(0), deformation)
    _, aligned = _edge_law(sp.Integer(1), sp.Integer(0), deformation)
    _, quadrature = _edge_law(sp.Integer(0), sp.Integer(1), sp.Integer(0))
    assert opposite[1] == -1 - 2 * deformation / 243
    assert aligned[1] == 1
    assert quadrature[1] == 0


def test_static_schur_is_not_the_dynamic_transfer_or_decay_rate():
    """The full pencil retains normal dynamics and the orientation of L."""
    spectral, gamma = sp.Symbol("spectral"), sp.Symbol("gamma", positive=True)
    boundary = sp.Matrix([[1, 2]])
    stiffness = sp.diag(0, 4) + 3 * boundary.T * boundary
    laplacian = sp.Matrix([[2, 1], [1, 2]])
    static = stiffness[0, 0] - stiffness[0, 1] ** 2 / stiffness[1, 1]
    assert static == sp.Rational(3, 4)
    assert static < stiffness[0, 0]
    pencil = (
        spectral**2 * laplacian.inv()
        + spectral * sp.eye(2) / 4
        + gamma**2 * stiffness / 16
    )
    transfer = sp.factor(pencil[0, 0] - pencil[0, 1] * pencil[1, 0] / pencil[1, 1])
    assert sp.simplify(transfer.subs(spectral, 0) - gamma**2 * static / 16) == 0
    # The static normal displacement is -3/8 times the soft displacement;
    # its physical damping is already present in the derivative of the pencil.
    assert sp.simplify(sp.diff(transfer, spectral).subs(spectral, 0)) == sp.Rational(
        73, 256
    )
    naive = spectral**2 * laplacian.inv()[0, 0] + spectral / 4 + gamma**2 * static / 16
    assert sp.factor(transfer - naive) != 0

    # Reuse the shared elimination owner. A hidden normal-phase preparation
    # has an immediate visible form effect despite identical visible state.
    rational_field = _jacobian(laplacian, stiffness, sp.Integer(1))
    admitted = tuple(
        tuple(Q(int(value.p), int(value.q)) for value in row)
        for row in rational_field.tolist()
    )
    memory = derive_coordinate_memory(admitted, (0, 2))
    assert memory.hidden_indices == (1, 3)
    assert sp.Matrix(memory.kernel_at_zero) == sp.Matrix(
        [
            [-sp.Rational(3, 8), -sp.Rational(1, 16)],
            [sp.Rational(3, 8), -sp.Rational(5, 16)],
        ]
    )
    hidden_preparation = sp.Matrix([0, 1, 0, 0])
    visible_rate = (rational_field * hidden_preparation).extract([0, 2], [0])
    assert visible_rate == sp.Matrix([0, -sp.Rational(3, 2)])
    assert visible_rate == sp.Matrix(memory.hidden_to_visible) * sp.Matrix([1, 0])

    # Equal kinetic spectra and the same stiffness still give different rates
    # when the kinetic and stiffness eigenspaces are differently aligned.
    stiffness = sp.diag(2, 5)
    diagonal = sp.diag(1, 3)
    assert diagonal.charpoly().as_expr() == laplacian.charpoly().as_expr()
    assert laplacian * stiffness != stiffness * laplacian
    first = _jacobian(diagonal, stiffness, gamma).charpoly(spectral).as_expr()
    second = _jacobian(laplacian, stiffness, gamma).charpoly(spectral).as_expr()
    assert sp.simplify(first - second) == 3 * gamma**2 * spectral**2 / 16
    for kinetic in (diagonal, laplacian):
        dynamic = _jacobian(kinetic, stiffness, gamma)
        exact_pencil = (
            spectral**2 * kinetic.inv()
            + spectral * sp.eye(2) / 4
            + gamma**2 * stiffness / 16
        )
        assert (
            sp.expand(
                dynamic.charpoly(spectral).as_expr()
                - kinetic.det() * exact_pencil.det()
            )
            == 0
        )


def test_full_multiple_zero_projection_retains_form_and_is_semisimple():
    """A nonorthogonal kernel basis normalizes both rows of the projector."""
    gamma, spectral = sp.Symbol("gamma", positive=True), sp.Symbol("spectral")
    kinetic = sp.Matrix([[2, 1, 0], [1, 3, 1], [0, 1, 4]])
    normal = sp.Matrix([[1, -1, 1]])
    stiffness = normal.T * normal
    modes = sp.Matrix([[1, -1], [1, 0], [0, 1]])
    gram = modes.T * modes
    assert gram != sp.eye(2)
    assert stiffness * modes == sp.zeros(3, 2)
    assert kinetic * stiffness != stiffness * kinetic
    dynamic = _jacobian(kinetic, stiffness, gamma)
    right = modes.col_join(sp.zeros(3, 2))
    phase_projection = gram.inv() * modes.T
    left = phase_projection.row_join(gamma * phase_projection)
    assert left * right == sp.eye(2)
    assert dynamic * right == sp.zeros(6, 2)
    assert left * dynamic == sp.zeros(2, 6)
    projector = right * left
    assert projector * projector == projector
    polynomial = sp.Poly(dynamic.charpoly(spectral).as_expr(), spectral)
    assert polynomial.nth(0) == polynomial.nth(1) == 0
    assert polynomial.nth(2).is_positive
    for column in range(2):
        initial = sp.zeros(3, 1).col_join(modes[:, column])
        assert left * initial == gamma * sp.eye(2)[:, column]
    assert phase_projection.row_join(sp.zeros(2, 3)) * dynamic != sp.zeros(2, 6)


def test_same_hessian_and_loss_allow_different_nonlinear_stability_and_order():
    """Nonnegative circular potentials can have a minimum or a maximum at zero."""
    q, transverse = sp.symbols("q transverse", real=True)
    gamma, normal_stiffness = sp.symbols("gamma normal_stiffness", positive=True)
    fourth = (1 - sp.cos(q)) ** 2
    sixth = (1 - sp.cos(q)) ** 3
    maximum = 4 - fourth
    kinetic = sp.Matrix([[2, 1], [1, 2]])
    phase = sp.Matrix([q, transverse])
    form = sp.Matrix(sp.symbols("form0:2", real=True))
    expected_hessian = sp.diag(0, normal_stiffness)
    jacobians = []
    for circular in (fourth, sixth, maximum):
        potential = circular + normal_stiffness * transverse**2 / 2
        gradient = sp.Matrix([sp.diff(potential, value) for value in phase])
        hessian = gradient.jacobian(phase).subs({q: 0, transverse: 0})
        assert hessian == expected_hessian
        phase_rate = gamma * kinetic * form / 4
        form_rate = -(kinetic * form + gamma * gradient) / 4
        loss = ((kinetic * form).T * form_rate + gradient.T * phase_rate)[0]
        assert sp.expand(loss + ((kinetic * form).T * kinetic * form)[0] / 4) == 0
        jacobians.append(
            phase_rate.col_join(form_rate)
            .jacobian(phase.col_join(form))
            .subs({q: 0, transverse: 0})
        )
    assert jacobians[0] == jacobians[1] == jacobians[2]
    assert sp.diff(fourth, q, 4).subs(q, 0) == 6
    assert sp.diff(sixth, q, 4).subs(q, 0) == 0
    assert sp.diff(sixth, q, 6).subs(q, 0) == 90
    assert sp.diff(maximum, q, 4).subs(q, 0) == -6
    # At zero form, the maximum's nearby storage lies below its equilibrium
    # value; nonincreasing storage then excludes attraction from those states.
    assert sp.simplify(maximum - maximum.subs(q, 0) + fourth) == 0


def test_actual_seven_child_one_and_split_boundary_projection():
    """Port placement changes independent constraints at equal total conductance."""
    e1, e2, total, gamma = sp.symbols("e1 e2 total gamma", positive=True)
    h = sp.symbols("h", positive=True)
    g = -sp.Rational(5, 6)
    u0, u1, u3 = sp.symbols("u0 u1 u3", real=True)
    mode = sp.Matrix(
        [
            sp.Rational(1, 6) + u0,
            sp.Rational(1, 6) + u1,
            h,
            sp.Rational(1, 6) + u3,
            sp.Rational(1, 3) - h - u0 - u1 - u3,
            g,
            0,
        ]
    )
    assert sum(mode) == 0
    norm = (mode.T * mode)[0]
    physical_norm = 9 * norm
    mobility = 9 * gamma**2 / (4 * physical_norm)
    relative = sp.Matrix([-sp.Rational(1, 7)] * 6 + [sp.Rational(6, 7)])
    mean = sp.Matrix([[-sp.Rational(1, 6)] * 6 + [1]])
    trace = sp.Matrix([[0, 0, -1, 0, 0, 0, 1], [0, 0, 0, 0, 0, -1, 1]])
    boundary = trace.T * sp.diag(e1, e2) * trace
    lift = sp.kronecker_product(sp.eye(7), sp.ones(9, 1))
    fine_trace = sp.zeros(18, 63)
    for port, child in enumerate((2, 5)):
        for slot in range(9):
            fine_trace[9 * port + slot, 9 * child + slot] = -1
            fine_trace[9 * port + slot, 54 + slot] = 1
    assert fine_trace * lift == sp.kronecker_product(trace, sp.ones(9, 1))
    fine_weights = sp.diag(*([e1] * 9 + [e2] * 9))
    assert lift.T * fine_trace.T * fine_weights * fine_trace * lift == 9 * boundary
    phase_columns = mode.row_join(sp.zeros(7, 1)).row_join(relative)
    form_columns = sp.zeros(7, 1).row_join(relative).row_join(sp.zeros(7, 1))
    left = (mode.T / norm).row_join(gamma * mode.T / norm)
    left = left.col_join(sp.zeros(1, 7).row_join(mean)).col_join(
        mean.row_join(sp.zeros(1, 7))
    )
    right = phase_columns.col_join(form_columns)
    assert sp.simplify(left * right) == sp.eye(3)
    projection = sp.simplify(left * _jacobian(boundary, boundary, gamma) * right)
    s0, s1, s2 = e1 + e2, e1 * h + e2 * g, e1 * h**2 + e2 * g**2
    factor = sp.Rational(7, 24)
    expected = sp.Matrix(
        [
            [-mobility * s2, 0, mobility * s1],
            [factor * gamma * s1, -factor * s0, -factor * gamma * s0],
            [0, factor * gamma * s0, 0],
        ]
    )
    assert sp.simplify(projection - expected) == sp.zeros(3)
    kernel_trace = trace * mode.row_join(relative)
    assert kernel_trace == sp.Matrix([[-h, 1], [-g, 1]])
    assert kernel_trace.det() == g - h
    assert kernel_trace[:1, :].rank() == 1
    assert kernel_trace.rank() == 2
    assert sp.factor(s0 * s2 - s1**2) == e1 * e2 * (6 * h + 5) ** 2 / 36
    assert (
        s0.subs({e1: total, e2: 0}) == s0.subs({e1: total / 2, e2: total / 2}) == total
    )
    # A balanced split can hide the initial linear mean signal while retaining
    # soft restoration; h remains the actual unknown admitted port slope.
    balanced = {e1: -g * total / (h - g), e2: h * total / (h - g)}
    assert sp.simplify(s1.subs(balanced)) == 0
    assert sp.simplify(s2.subs(balanced) + total * h * g) == 0


def test_split_contact_compliance_and_complete_leading_damping():
    """Eliminating normals and retaining three kinetic coordinates give distinct data."""
    t, d = sp.symbols("t d", real=True)
    e1, e2, h, gamma, mobility = sp.symbols("e1 e2 h gamma mobility", positive=True)
    g = -sp.Rational(5, 6)
    y = sp.Matrix(sp.symbols("normal0:2", real=True))
    normal_stiffness, port = sp.Matrix([[3, 1], [1, 2]]), sp.Matrix([1, 2])
    assert normal_stiffness.det() > 0
    compliance = (port.T * normal_stiffness.inv() * port)[0]
    host_projection = sp.eye(5) - sp.ones(5) / 5
    host_port = host_projection[:, 2]
    # The actual host admission A>I/5 gives chi<5*||P_5 e_2||^2=4.
    assert (host_port.T * host_port)[0] * 5 == 4
    harmonic = e1 * e2 / (e1 + e2)
    potential = (
        9
        * (
            (y.T * normal_stiffness * y)[0]
            + e1 * (d - h * t - (port.T * y)[0]) ** 2
            + e2 * (d - g * t) ** 2
        )
        / 2
    )
    variables = [d, *y]
    solution = sp.solve(
        [sp.diff(potential, value) for value in variables], variables, dict=True
    )
    assert len(solution) == 1
    relaxed = sp.factor(potential.subs(solution[0]))
    effective = sp.factor(sp.diff(relaxed, t, 2))
    expected = 9 * harmonic * (h - g) ** 2 / (1 + harmonic * compliance)
    assert sp.simplify(effective - expected) == 0
    assert sp.simplify(
        y.subs(solution[0])
        + harmonic
        * (h - g)
        * normal_stiffness.inv()
        * port
        * t
        / (1 + harmonic * compliance)
    ) == sp.zeros(2, 1)
    assert sp.factor(effective - 9 * harmonic * (h - g) ** 2) != 0

    s0, s1, s2, c = sp.symbols("s0 s1 s2 c", real=True)
    spectral = sp.Symbol("spectral")
    leading = sp.Matrix(
        [
            [-mobility * s2, 0, mobility * s1],
            [c * gamma * s1, -c * s0, -c * gamma * s0],
            [0, c * gamma * s0, 0],
        ]
    )
    polynomial = sp.Poly(leading.charpoly(spectral).as_expr(), spectral)
    quadratic, linear, constant = (
        polynomial.nth(2),
        polynomial.nth(1),
        polynomial.nth(0),
    )
    delta = s0 * s2 - s1**2
    assert sp.factor(constant) == c**2 * gamma**2 * mobility * s0 * delta
    positive_remainder = (
        c * mobility * s0 * s2 * (c * s0 + mobility * s2)
        + c**3 * gamma**2 * s0**3
        + mobility * c**2 * gamma**2 * s0 * s1**2
    )
    assert sp.expand(quadratic * linear - constant - positive_remainder) == 0
    single = {s0: e1, s1: e1 * h, s2: e1 * h**2}
    assert sp.simplify(constant.subs(single)) == 0
    split = {
        s0: e1 + e2,
        s1: e1 * h + e2 * g,
        s2: e1 * h**2 + e2 * g**2,
        c: sp.Rational(7, 24),
    }
    assert sp.factor(constant.subs(split)).is_positive
    assert quadratic.subs(split).is_positive
    assert linear.subs(split).is_positive
    assert positive_remainder.subs(split).is_positive
