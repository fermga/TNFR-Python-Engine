"""Exact constructed controls for graph-relative finite preparation errors.

No retained graph, source record, family response or error propagation is
consumed. These algebraic witnesses do not certify a target validation domain.
"""

from fractions import Fraction as Q
from itertools import product

import pytest
import sympy as s

from tnfr.mathematics._comparison_flow import MAX_COMPARISON_DIMENSION
from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_symmetric_semidefinite,
)
from tnfr.mathematics._rational_interval import I, sqrt
from tnfr.mathematics._validated_metric import certify_metric_logarithmic_bound


def _rational_matrix(matrix):
    return tuple(
        tuple(Q(int(value.p), int(value.q)) for value in row) for row in matrix.tolist()
    )


def _inverse(matrix):
    answer = s.Matrix(exact_matrix_inverse(_rational_matrix(matrix)))
    assert matrix * answer == answer * matrix == s.eye(matrix.rows)
    return answer


def _psd(matrix, *, strict=False):
    return exact_symmetric_semidefinite(_rational_matrix(matrix), strict=strict)


def _incidence(size, edges):
    matrix = s.zeros(size, len(edges))
    for column, (left, right) in enumerate(edges):
        matrix[left, column], matrix[right, column] = 1, -1
    return matrix


def _triangle():
    incidence = _incidence(3, ((0, 1), (1, 2), (0, 2)))
    K = incidence * s.diag(2, 3, 5) * incidence.T
    C0 = incidence * incidence.T
    beta = s.diag(1, 2, 3)
    rho = s.Matrix((1, s.Rational(1, 2), s.Rational(1, 3)))
    alpha = rho / sum(rho)
    rank_one = rho * rho.T / sum(rho)
    return incidence, K, C0, beta, rho, alpha, K + rank_one, C0 + rank_one


def test_complete_nonlinear_rows_preserve_both_unequal_mobility_means():
    B, K, _, beta, rho, alpha, _, _ = _triangle()
    x = s.Matrix(s.symbols("x0:3", real=True))
    theta = s.Matrix(s.symbols("theta0:3", real=True))
    ex = s.Matrix(s.symbols("u0:3", real=True))
    etheta = s.Matrix(s.symbols("v0:3", real=True))

    def rows(forms, phases):
        return -beta * B * (B.T * phases).applyfunc(s.sin), beta * K * forms

    nominal = rows(x, theta)
    perturbed = rows(x + ex, theta + etheta)
    assert all((rho.T * row)[0] == 0 for row in nominal + perturbed)
    P = s.eye(3) - s.ones(3, 1) * alpha.T
    assert P * P == P and alpha.T * P == s.zeros(1, 3)
    assert B.T * P == B.T and K * P == K
    differences = tuple(
        left - right for left, right in zip(perturbed, nominal, strict=True)
    )
    centered_perturbed = rows(x + P * ex, theta + P * etheta)
    centered_differences = tuple(
        left - right for left, right in zip(centered_perturbed, nominal, strict=True)
    )
    for original, centered in zip(differences, centered_differences, strict=True):
        assert (original - centered).applyfunc(s.trigsimp) == s.zeros(3, 1)
        assert (P * original - original).applyfunc(s.simplify) == s.zeros(3, 1)


def test_finite_segment_secant_is_smooth_and_differs_from_the_tangent():
    phi, parameter = s.symbols("phi parameter", real=True)
    eta = s.symbols("eta", real=True, nonzero=True)
    difference = s.sin(phi + eta) - s.sin(phi)
    secant = s.integrate(s.cos(phi + parameter * eta), (parameter, 0, 1))
    assert s.simplify(secant * eta - difference) == 0
    quotient = difference / eta
    assert s.limit(quotient, eta, 0) == s.cos(phi)
    assert s.integrate(s.cos(phi), (parameter, 0, 1)) == s.cos(phi)
    # Even an acute reference does not license replacing the finite chord
    # by its tangent. The end of this chord has a right-angle edge gap.
    assert quotient.subs({phi: 0, eta: s.pi / 2}) == 2 / s.pi
    assert s.cos(0) == 1
    assert s.limit(quotient, eta, 0).subs(phi, s.pi) == -1


def test_graph_quadratic_cancellation_retains_signed_secants_and_mean_terms():
    B, K, C0, beta, _, _, Qx, Qtheta = _triangle()
    coefficients = s.symbols("d0:3", real=True)
    C = B * s.diag(*coefficients) * B.T
    J = s.zeros(6)
    J[:3, 3:] = -beta * C
    J[3:, :3] = beta * K
    metric = s.diag(Qx, Qtheta)
    balance = metric * J + J.T * metric
    expected = s.zeros(6)
    expected[:3, 3:] = K * beta * (C0 - C)
    expected[3:, :3] = (C0 - C) * beta * K
    assert (balance - expected).applyfunc(s.expand) == s.zeros(6)
    assert _psd(Qx, strict=True) and _psd(Qtheta, strict=True)
    # Choosing the metric reference equal to the phase Hessian cancels all
    # restoring terms. No absolute-entry bound was substituted before this.
    assert balance.subs(dict(zip(coefficients, (1, 1, 1), strict=True))) == s.zeros(6)


def test_mean_augmented_inverse_gives_the_exact_quotient_and_dual_norms():
    _, K, C0, _, _, alpha, Qx, Qtheta = _triangle()
    inverse_x, inverse_theta = _inverse(Qx), _inverse(Qtheta)
    assert K * inverse_x * K == K
    assert C0 * inverse_theta * C0 == C0
    ell = s.Matrix((-1, 1, 0))
    assert alpha.T * inverse_x * ell == s.zeros(1, 1)
    assert alpha.T * inverse_theta * ell == s.zeros(1, 1)
    # Independent parallel/series effective conductance: 2 + 3*5/(3+5).
    assert (ell.T * inverse_x * ell)[0] == s.Rational(8, 31)
    assert (ell.T * inverse_theta * ell)[0] == s.Rational(2, 3)


def test_n_block_trace_and_edge_uncertainty_certify_all_signed_vertices():
    B, K, C0, beta, _, _, Qx, Qtheta = _triangle()
    inverse_x, inverse_theta = _inverse(Qx), _inverse(Qtheta)
    residual_center = (s.Rational(1, 4), s.Rational(4, 3), s.Rational(1, 5))
    radii = (Q(1, 20), Q(1, 30), Q(1, 40))
    D = B * s.diag(*residual_center) * B.T
    F = D * beta * K * beta * D
    trace_square = s.trace(inverse_theta * F)
    assert trace_square > 0 and _psd(trace_square * Qtheta - F)
    center_upper = sqrt(I(Q(int(trace_square.p), int(trace_square.q)))).hi
    edge_upper = []
    for column in range(B.cols):
        b = B[:, column]
        a_e, resistance = (b.T * beta * K * beta * b)[0], (b.T * inverse_theta * b)[0]
        H = K * beta * b * b.T
        # The rank-one Frobenius norm equals the product of its two exact
        # dual norms. This keeps the mobility and weighted form graph.
        assert s.trace(inverse_x * H * inverse_theta * H.T) == a_e * resistance
        square = a_e * resistance
        edge_upper.append(sqrt(I(Q(int(square.p), int(square.q)))).hi)
    upper = center_upper + sum(
        (r * bound for r, bound in zip(radii, edge_upper, strict=True)), Q(0)
    )
    assert upper > center_upper
    for signs in product((-1, 1), repeat=3):
        vertex = (
            D
            + B
            * s.diag(
                *(sign * radius for sign, radius in zip(signs, radii, strict=True))
            )
            * B.T
        )
        # n-block certificate only. A center-only certificate would omit the
        # independent uncertainty represented by these eight vertices.
        assert _psd(upper**2 * Qtheta - vertex * beta * K * beta * vertex)
    # Complete the square to derive the 2-block estimate without asking the
    # full-state metric validator to handle a larger dimension.
    x = s.Matrix(s.symbols("x0:3"))
    theta = s.Matrix(s.symbols("theta0:3"))
    rate = s.symbols("rate", positive=True)
    shifted = x - beta * D * theta / rate
    lhs = (
        rate * (x.T * K * x)[0]
        + rate * (theta.T * Qtheta * theta)[0]
        - 2 * (x.T * K * beta * D * theta)[0]
    )
    rhs = (
        rate * (shifted.T * K * shifted)[0]
        + (theta.T * (rate**2 * Qtheta - F) * theta)[0] / rate
    )
    assert s.expand(lhs - rhs) == 0


def test_connected_observation_uses_the_maximum_dual_coefficient_and_is_sharp():
    _, K, C0, _, _, _, Qx, Qtheta = _triangle()
    inverse_x, inverse_theta = _inverse(Qx), _inverse(Qtheta)
    ell = s.Matrix((-1, 1, 0))
    w = s.Integer(2)
    rx, rt = (ell.T * inverse_x * ell)[0], (ell.T * inverse_theta * ell)[0]
    coefficient = max(w * rx, rt)
    assert coefficient == s.Rational(2, 3) < w * rx + rt
    assert _psd(coefficient * Qx - w * ell * ell.T)
    assert _psd(coefficient * Qtheta - ell * ell.T)
    forms, phases = s.zeros(3, 1), inverse_theta * ell
    energy = (forms.T * K * forms)[0] + (phases.T * C0 * phases)[0]
    observed = w * (ell.T * forms)[0] ** 2 + (ell.T * phases)[0] ** 2
    assert observed == coefficient * energy > 0


def test_whole_original_box_keeps_two_host_means_and_both_child_constants():
    B = _incidence(3, ((0, 1), (1, 2)))
    K, C0 = B * s.diag(3, 5) * B.T, B * B.T
    rho = s.Matrix((s.Rational(1, 2), 1, s.Rational(1, 4)))
    alpha = rho / sum(rho)
    assert alpha == s.Matrix((s.Rational(2, 7), s.Rational(4, 7), s.Rational(1, 7)))
    Qx, Qt = K + rho * rho.T / sum(rho), C0 + rho * rho.T / sum(rho)
    b = s.Matrix((1, 0, 0)) - alpha
    rx, rt = (b.T * _inverse(Qx) * b)[0], (b.T * _inverse(Qt) * b)[0]
    assert rx == s.Rational(128, 735) and rt == s.Rational(26, 49)
    coefficient = max(2 * rx, rt)
    widths = (Q(1, 8), Q(1, 16), Q(1, 32), Q(1, 64))
    for signs in product((-1, 1), repeat=8):
        forms = tuple(signs[i] * widths[i] for i in range(4))
        phases = tuple(signs[4 + i] * widths[i] for i in range(4))
        hx, ht = s.Matrix((forms[0], forms[2], forms[3])), s.Matrix(
            (phases[0], phases[2], phases[3])
        )
        mx, mt = (alpha.T * hx)[0], (alpha.T * ht)[0]
        zx, zt = hx - mx * s.ones(3, 1), ht - mt * s.ones(3, 1)
        assert (alpha.T * zx)[0] == (alpha.T * zt)[0] == 0
        # The same full eight-coordinate initial state supplies connected
        # and cut portraits. Centering is an identity, never a projection.
        assert forms[1] - forms[0] == forms[1] - mx - (b.T * zx)[0]
        assert phases[1] - phases[0] == phases[1] - mt - (b.T * zt)[0]
        energy = (zx.T * K * zx)[0] + (zt.T * C0 * zt)[0]
        relative_observed = 2 * (b.T * zx)[0] ** 2 + (b.T * zt)[0] ** 2
        assert relative_observed <= coefficient * energy
    # Uniform host displacement has zero relative energy but nonzero marked
    # error against a child with another constant preparation. Dropping the
    # offset would assert a false zero bound even on this exact state.
    hx, ht = s.ones(3, 1), -2 * s.ones(3, 1)
    assert (hx.T * K * hx)[0] + (ht.T * C0 * ht)[0] == 0
    assert (-ht[0]) ** 2 + 2 * (-hx[0]) ** 2 == 6


def test_cycle_compensation_allows_a_nonacute_edge_but_a_nonpositive_bridge_does_not():
    B = _incidence(3, ((0, 1), (1, 2), (0, 2)))
    bare = B * B.T
    signed = B * s.diag(s.Rational(2, 3), s.Rational(2, 3), s.Rational(-1, 9)) * B.T
    assert s.Rational(-1, 9) == 2 * s.Rational(2, 3) ** 2 - 1
    first, second = s.Matrix((1, -2, 1)), s.Matrix((1, 0, -1))
    assert signed * first == 2 * first
    assert signed * second == s.Rational(4, 9) * second
    assert _psd(signed - s.Rational(4, 27) * bare)
    for signs in product((-1, 1), repeat=3):
        perturbed = (
            signed + B * s.diag(*(sign * s.Rational(2, 27) for sign in signs)) * B.T
        )
        assert _psd(perturbed - s.Rational(2, 27) * bare)
    bridge_B = _incidence(4, ((0, 1), (1, 2), (2, 3)))
    p, q, negative = s.symbols("p q negative", positive=True)
    bridge_laplacian = bridge_B * s.diag(p, -negative, q) * bridge_B.T
    across_cut = s.Matrix((0, 0, 1, 1))
    assert (across_cut.T * bridge_laplacian * across_cut)[0] == -negative


def test_antiphase_saddle_forbids_a_global_nonincreasing_positive_error_metric():
    m, w = s.symbols("m w", positive=True)
    r, eta = s.symbols("r eta", real=True)
    # Original gap is pi+eta; the complete two-node law has both nonzero
    # capacities and the actual conductance in its phase row.
    field = s.Matrix((2 * m * s.sin(eta), 2 * m * w * r))
    tangent = field.jacobian((r, eta)).subs({r: 0, eta: 0})
    growing = s.Matrix((1, s.sqrt(w)))
    eigenvalue = 2 * m * s.sqrt(w)
    assert tangent * growing == eigenvalue * growing
    a, b, c = s.symbols("a b c", real=True)
    metric = s.Matrix(((a, b), (b, c)))
    assert (
        s.expand(
            (growing.T * (metric * tangent + tangent.T * metric) * growing)[0]
            - 2 * eigenvalue * (growing.T * metric * growing)[0]
        )
        == 0
    )
    # If metric is positive definite, the last quadratic is strictly
    # positive; no such metric can make the balance nonpositive globally.
    assert (s.cos(eta) - 1).subs(eta, s.pi / 2) == -1


@pytest.mark.parametrize("host_nodes,constant_coordinates", ((21, 2), (20, 4)))
def test_mean_removal_keeps_full_dimension_without_extending_metric_policy(
    host_nodes, constant_coordinates
):
    relative_dimension = 2 * (host_nodes - 1)
    assert relative_dimension + constant_coordinates == 42
    assert host_nodes <= MAX_COMPARISON_DIMENSION < relative_dimension

    def forbidden(*args):
        pytest.fail("dimension admission must precede any field evaluation")

    with pytest.raises(ValueError, match="dimension outside"):
        certify_metric_logarithmic_bound(
            (I(0),) * relative_dimension,
            (),
            forbidden,
            rate_bounds=(0, 1),
            bisections=0,
        )
