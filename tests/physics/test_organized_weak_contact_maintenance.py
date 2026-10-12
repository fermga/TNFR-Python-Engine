"""Static arithmetic for the declared positive weak-spoke contact law.

These controls check complete weighted rows, storage and a sufficient
barrier. They run no source producer, stationary search or trajectory.
"""

from fractions import Fraction as Q

import mpmath
import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import cos, pi_interval

_OLD_EDGES = tuple(
    (9 * child + slot, 9 * child + (slot + 1) % 9)
    for child in range(6)
    for slot in range(9)
) + tuple(
    (9 * child + slot, 9 * ((child + 1) % 5) + slot)
    for child in range(5)
    for slot in range(9)
)
_SPOKES = tuple(
    (9 * child + slot, 45 + slot) for child in range(5) for slot in range(9)
)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _weighted_edges(eta):
    return tuple((left, right, 1) for left, right in _OLD_EDGES) + tuple(
        (left, right, eta) for left, right in _SPOKES
    )


def _product_laplacian(mp, eta):
    """Assemble I6 tensor L9 + L_wheel(eta) tensor I9 by node rows."""
    laplacian = mp.zeros(54)
    for child in range(6):
        for slot in range(9):
            index = 9 * child + slot
            laplacian[index, index] = 4 + eta if child < 5 else 2 + 5 * eta
            for neighbor_slot in ((slot - 1) % 9, (slot + 1) % 9):
                laplacian[index, 9 * child + neighbor_slot] = -1
            if child < 5:
                for neighbor_child in ((child - 1) % 5, (child + 1) % 5):
                    laplacian[index, 9 * neighbor_child + slot] = -1
                laplacian[index, 45 + slot] = -eta
            else:
                for neighbor_child in range(5):
                    laplacian[index, 9 * neighbor_child + slot] = -eta
    return laplacian


@pytest.mark.parametrize("eta_value", (Q(1, 100000), Q(1)))
def test_weak_contact_complete_weighted_rows_and_nodal_loss(mp, eta_value):
    """Independent edge storage and node rows retain the same spoke weights."""
    tolerance = mp.mpf("1e-85")
    eta, gamma = _mp(mp, eta_value), 1 / (1023 * mp.pi)
    edges = _weighted_edges(eta)
    assert len(_OLD_EDGES) == 99
    assert len(_SPOKES) == 45
    assert len({tuple(sorted((a, b))) for a, b, _ in edges}) == 144

    raw_form = tuple(mp.mpf((7 * index) % 17 - 8) / 19 for index in range(54))
    form = mp.matrix([value - sum(raw_form) / 54 for value in raw_form])
    phase = mp.matrix([mp.mpf((11 * index) % 23 - 11) / 13 for index in range(54)])
    form_gradient, current = mp.zeros(54, 1), mp.zeros(54, 1)
    for left, right, weight in edges:
        exchange = weight * (form[left] - form[right])
        form_gradient[left] += exchange
        form_gradient[right] -= exchange
        pressure = weight * mp.sin(phase[right] - phase[left])
        current[left] += pressure
        current[right] -= pressure

    laplacian = _product_laplacian(mp, eta)
    assert max(abs(value) for value in laplacian * form - form_gradient) < tolerance
    assert max(abs(value) for value in laplacian * mp.ones(54, 1)) < tolerance
    assert abs(sum(form)) < tolerance
    assert abs(sum(current)) < tolerance
    form_rate = (-form_gradient + gamma * current) / 4
    phase_rate = gamma * laplacian * form / 4
    assert abs(sum(form_rate)) < tolerance
    assert abs(sum(phase_rate)) < tolerance

    def storage_on_tangent(parameter, velocity=phase_rate):
        forms = form + parameter * form_rate
        phases = phase + parameter * velocity
        return sum(
            weight
            * ((forms[b] - forms[a]) ** 2 / 2 + 1 - mp.cos(phases[b] - phases[a]))
            for a, b, weight in edges
        )

    loss = -(form_gradient.T * form_gradient)[0] / 4
    assert abs(mp.diff(storage_on_tangent, 0) - loss) < tolerance
    if eta_value != 1:
        # Leaving unit spokes in the phase row breaks the declared loss identity.
        unit_laplacian = _product_laplacian(mp, mp.mpf(1))
        wrong_velocity = gamma * unit_laplacian * form / 4
        defect = gamma * (current.T * (laplacian - unit_laplacian) * form)[0] / 4
        wrong_rate = mp.diff(lambda t: storage_on_tangent(t, wrong_velocity), 0)
        assert abs(defect) > mp.mpf("1e-10")
        assert abs(wrong_rate - loss - defect) < tolerance


@pytest.mark.parametrize("beta_turns", (Q(0), Q(1, 14)))
def test_weak_contact_reference_circle_energy_and_nonstationarity(mp, beta_turns):
    """The low-energy circle is a comparison set, not an equilibrium family."""
    tolerance = mp.mpf("1e-85")
    eta = mp.mpf(1) / 100000
    alpha, beta = 2 * mp.pi / 9, 2 * mp.pi * _mp(mp, beta_turns)
    origins = tuple(2 * mp.pi * (child - 2) / 5 for child in range(5)) + (beta,)
    phase = tuple(
        alpha * (slot - 4) + origins[child] for child in range(6) for slot in range(9)
    )
    current = [mp.mpf(0)] * 54
    old_storage = sum(1 - mp.cos(phase[b] - phase[a]) for a, b in _OLD_EDGES)
    spoke_storage = sum(1 - mp.cos(phase[b] - phase[a]) for a, b in _SPOKES)
    for left, right, weight in _weighted_edges(eta):
        exchange = weight * mp.sin(phase[right] - phase[left])
        current[left] += exchange
        current[right] -= exchange
    v9, v5 = 9 * (1 - mp.cos(alpha)), 5 * (1 - mp.cos(2 * mp.pi / 5))
    assert abs(old_storage - (6 * v9 + 9 * v5)) < tolerance
    assert abs(spoke_storage - 45) < tolerance
    assert (
        abs(old_storage + eta * spoke_storage - (6 * v9 + 9 * v5 + 45 * eta))
        < tolerance
    )
    for child in range(5):
        for slot in range(9):
            assert (
                abs(current[9 * child + slot] - eta * mp.sin(beta - origins[child]))
                < tolerance
            )
    assert max(abs(value) for value in current[45:]) < tolerance
    assert max(abs(value) for value in current[:45]) > eta / 2


def test_weak_contact_old_cycle_faces_give_strict_rational_barrier(mp):
    """Exact outward arithmetic checks feasible faces and the spare gap."""
    pi = pi_interval()
    margins = []
    for size in (9, 5):
        ground = size * (1 - cos(2 * pi / size))
        positive_face = 1 + (size - 1) * (1 - cos(3 * pi / (2 * (size - 1))))
        margin = positive_face - ground
        margins.append(margin)
        # Independent saturated-face gaps sum to the same winding-one lift.
        plus_gaps = (mp.pi / 2,) + (3 * mp.pi / (2 * (size - 1)),) * (size - 1)
        minus_gaps = (-mp.pi / 2,) + (5 * mp.pi / (2 * (size - 1)),) * (size - 1)
        expected_ground = size * (1 - mp.cos(2 * mp.pi / size))
        expected_margin = sum(1 - mp.cos(gap) for gap in plus_gaps) - expected_ground
        assert abs(sum(plus_gaps) - 2 * mp.pi) < mp.mpf("1e-85")
        assert abs(sum(minus_gaps) - 2 * mp.pi) < mp.mpf("1e-85")
        assert _mp(mp, margin.lo) < expected_margin < _mp(mp, margin.hi)
        if size == 9:
            negative_face = 1 + (size - 1) * (1 - cos(5 * pi / (2 * (size - 1))))
            assert negative_face.lo > positive_face.hi
            assert all(abs(gap) <= mp.pi / 2 for gap in minus_gaps)
        else:
            # A negative C5 face cannot reach winding one with four acute gaps.
            assert -mp.pi / 2 + (size - 1) * mp.pi / 2 < 2 * mp.pi
            assert any(abs(gap) > mp.pi / 2 for gap in minus_gaps)
    assert margins[1].hi < margins[0].lo
    barrier_lower = min(value.lo for value in margins)
    barrier_upper = min(value.hi for value in margins)
    assert barrier_lower > Q(7, 500)
    assert 45 * Q(1, 100000) < barrier_lower
    # The sufficient weak-contact bound does not admit the original unit point.
    # Its failure is not a nonexistence or instability verdict at that point.
    assert 45 * Q(1) > barrier_upper


def test_weak_contact_source_budget_uses_only_retained_analytic_allowances():
    """Recombine exact majorants; do not run or replace their source producers."""
    gamma_upper, form_allowance, duration = Q(1, 3000), Q(1, 100), Q(1, 5000)
    reached_error, eta = Q(1, 10**70), Q(1, 100000)
    form_bound = form_allowance + gamma_upper * duration
    form_drift = (2 * form_bound + gamma_upper) * duration
    phase_drift = 2 * gamma_upper * form_bound * duration
    host_increase = 360 * form_bound * form_drift + 180 * phase_drift
    donor_excess = 4 * reached_error**2
    attachment = 45 * eta * ((form_bound + reached_error) ** 2 / 2 + 2)
    total_excess = Q(11, 1000) + host_increase + donor_excess + attachment
    assert host_increase < Q(1, 50000)
    assert attachment < Q(901, 10**6)
    assert total_excess < Q(3, 250)
    assert Q(7, 500) - total_excess > Q(1, 500)


def _wheel_matrix(matrix, rim_weights, spoke_weights):
    """Assemble six-node edge quadratic forms, without a parity formula."""
    for child in range(5):
        for neighbor, weight in (
            ((child + 1) % 5, rim_weights[child]),
            (5, spoke_weights[child]),
        ):
            matrix[child, child] += weight
            matrix[neighbor, neighbor] += weight
            matrix[child, neighbor] -= weight
            matrix[neighbor, child] -= weight
    return matrix


def test_weak_contact_reflected_branch_exact_parity_determinants():
    """Polynomial identities retain both reflection parities and all five modes."""
    radius, strength = sp.symbols("radius strength", positive=True)
    cubic = radius**3 - (strength + 1) * radius + strength**2
    cosine_v = -(strength / radius + strength - 1) / 2
    cosine_u = (strength + cosine_v) / radius
    cosine_difference = (1 + strength * cosine_v) / radius
    eta = strength - 1

    def reduced_numerator(expression):
        numerator = sp.fraction(sp.cancel(expression))[0]
        return sp.factor(sp.rem(numerator, cubic, strength))

    # The reconstructed sine/cosine pairs have unit length, and both reflected
    # stationary equations hold. The cubic is an equality, not a root estimate.
    assert reduced_numerator(cosine_u**2 + (1 - cosine_v**2) / radius**2 - 1) == 0
    assert sp.simplify(strength / radius + 2 * cosine_v + eta) == 0
    assert sp.simplify(cosine_u - cosine_v / radius - strength / radius) == 0

    wheel = _wheel_matrix(
        sp.zeros(6),
        (cosine_difference, cosine_u, cosine_u, cosine_difference, 2 * cosine_v**2 - 1),
        tuple(eta * value for value in (cosine_v, cosine_u, 1, cosine_u, cosine_v)),
    )
    even_basis = sp.Matrix(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]]
    )
    odd_basis = sp.Matrix([[-1, 0], [0, -1], [0, 0], [0, 1], [1, 0], [0, 0]])
    assert even_basis.row_join(odd_basis).row_join(sp.ones(6, 1)).det() != 0
    assert even_basis.T * wheel * odd_basis == sp.zeros(3, 2)
    even = sp.simplify(even_basis.T * wheel * even_basis / 2)
    odd = sp.simplify(odd_basis.T * wheel * odd_basis / 2)
    even_expected = sp.Matrix(
        [
            [cosine_difference + eta * cosine_v, -cosine_difference, 0],
            [-cosine_difference, radius, -cosine_u],
            [0, -cosine_u, cosine_u + eta / 2],
        ]
    )
    odd_expected = sp.Matrix(
        [
            [
                cosine_difference + 2 * (2 * cosine_v**2 - 1) + eta * cosine_v,
                -cosine_difference,
            ],
            [-cosine_difference, radius],
        ]
    )
    assert all(reduced_numerator(value) == 0 for value in even - even_expected)
    assert all(reduced_numerator(value) == 0 for value in odd - odd_expected)
    even_determinant = (strength - 1) * (radius**2 + radius - 1) ** 4 / (8 * radius**4)
    odd_determinant = (1 - cosine_v**2) * (strength + 1 - 3 * radius**2) / radius
    assert reduced_numerator(even.det() - even_determinant) == 0
    assert reduced_numerator(odd.det() - odd_determinant) == 0

    anchor = {strength: 1, radius: (sp.sqrt(5) - 1) / 2}
    cosine_twist = (sp.sqrt(5) - 1) / 4
    path_laplacian = sp.Matrix([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])
    assert sp.simplify(even.subs(anchor) - cosine_twist * path_laplacian) == sp.zeros(3)
    assert sp.simplify(
        odd.subs(anchor) - cosine_twist * sp.Matrix([[3, -1], [-1, 2]])
    ) == sp.zeros(2)


def test_weak_contact_reflected_branch_fixed_algebraic_state_and_full_hessian(mp):
    """One exact state checks all 54 rows; it is not a larger-eta source claim."""
    radius_exact = sp.Rational(2, 3)
    strength_exact = (3 + sp.sqrt(39)) / 9
    assert (
        sp.simplify(
            radius_exact**3 - (strength_exact + 1) * radius_exact + strength_exact**2
        )
        == 0
    )
    assert sp.simplify(strength_exact + 1 - 3 * radius_exact**2).is_positive
    assert (strength_exact - 1).is_positive
    assert (sp.Rational(107, 100) - strength_exact).is_positive

    cosine_v_exact = -(strength_exact / radius_exact + strength_exact - 1) / 2
    cosine_u_exact = (strength_exact + cosine_v_exact) / radius_exact
    difference_exact = (1 + strength_exact * cosine_v_exact) / radius_exact
    eta_exact = strength_exact - 1
    even_exact = sp.Matrix(
        [
            [difference_exact + eta_exact * cosine_v_exact, -difference_exact, 0],
            [-difference_exact, radius_exact, -cosine_u_exact],
            [0, -cosine_u_exact, cosine_u_exact + eta_exact / 2],
        ]
    )
    odd_exact = sp.Matrix(
        [
            [
                difference_exact
                + 2 * (2 * cosine_v_exact**2 - 1)
                + eta_exact * cosine_v_exact,
                -difference_exact,
            ],
            [-difference_exact, radius_exact],
        ]
    )
    for block in (even_exact, odd_exact):
        for size in range(1, block.rows + 1):
            assert sp.simplify(block[:size, :size].det()).is_positive

    tolerance = mp.mpf("1e-85")
    radius, strength = mp.mpf(2) / 3, (3 + mp.sqrt(39)) / 9
    eta, alpha = strength - 1, 2 * mp.pi / 9
    cosine_v = -(strength / radius + strength - 1) / 2
    u, v = mp.acos((strength + cosine_v) / radius), mp.acos(cosine_v)
    origins = (-v, -u, mp.mpf(0), u, v, mp.mpf(0))
    phase = tuple(
        alpha * (slot - 4) + origins[child] for child in range(6) for slot in range(9)
    )
    current, hessian = mp.zeros(54, 1), mp.zeros(54)
    gaps = {}
    for left, right, weight in _weighted_edges(eta):
        difference = phase[right] - phase[left]
        exchange = weight * mp.sin(difference)
        current[left] += exchange
        current[right] -= exchange
        curvature = weight * mp.cos(difference)
        hessian[left, left] += curvature
        hessian[right, right] += curvature
        hessian[left, right] -= curvature
        hessian[right, left] -= curvature
        gaps[left, right] = mp.atan2(mp.sin(difference), mp.cos(difference))
    assert max(abs(value) for value in current) < tolerance
    for child in range(6):
        cycle = tuple(
            gaps[9 * child + slot, 9 * child + (slot + 1) % 9] for slot in range(9)
        )
        assert abs(sum(cycle) - 2 * mp.pi) < tolerance
        assert all(0 < gap < mp.pi for gap in cycle)
    for slot in range(9):
        cycle = tuple(
            gaps[9 * child + slot, 9 * ((child + 1) % 5) + slot] for child in range(5)
        )
        assert abs(sum(cycle) - 2 * mp.pi) < tolerance
        assert all(0 < gap < mp.pi for gap in cycle)
    for child in range(5):
        for slot in range(9):
            left, right = 9 * child + slot, 9 * ((child + 1) % 5) + slot
            up_left, up_right = (
                9 * child + (slot + 1) % 9,
                9 * ((child + 1) % 5) + (slot + 1) % 9,
            )
            circulation = (
                gaps[left, right]
                + gaps[right, up_right]
                - gaps[up_left, up_right]
                - gaps[left, up_left]
            )
            assert abs(circulation) < tolerance

    wheel = _wheel_matrix(
        mp.zeros(6),
        tuple(mp.cos(origins[(child + 1) % 5] - origins[child]) for child in range(5)),
        tuple(eta * mp.cos(origins[child]) for child in range(5)),
    )
    tensor = mp.zeros(54)
    for child in range(6):
        for slot in range(9):
            index = 9 * child + slot
            for neighbor_child in range(6):
                tensor[index, 9 * neighbor_child + slot] += wheel[child, neighbor_child]
            tensor[index, index] += 2 * mp.cos(alpha)
            for neighbor_slot in ((slot - 1) % 9, (slot + 1) % 9):
                tensor[index, 9 * child + neighbor_slot] -= mp.cos(alpha)
    assert max(abs(value) for value in hessian - tensor) < tolerance
    assert max(abs(value) for value in wheel * mp.ones(6, 1)) < tolerance
    grounded_factor = mp.cholesky(wheel[:5, :5])
    assert all(grounded_factor[index, index] > 0 for index in range(5))
    # All eight nonconstant C9 Fourier sectors retain all six wheel directions.
    for mode in range(1, 9):
        shift = 2 * mp.cos(alpha) * (1 - mp.cos(2 * mp.pi * mode / 9))
        assert shift > 0
        factor = mp.cholesky(wheel + shift * mp.eye(6))
        assert all(factor[index, index] > 0 for index in range(6))


def test_weak_contact_reflected_fold_exact_bracket_and_nonacute_closing_gap():
    """Rational signs locate the unique fold without computing any root."""
    strength, displacement, radius = sp.symbols("strength displacement radius")
    polynomial = 27 * strength**4 - 4 * (strength + 1) ** 3
    cubic = radius**3 - (strength + 1) * radius + strength**2
    assert sp.expand(sp.discriminant(cubic, radius) + polynomial) == 0
    assert polynomial.subs(strength, sp.Rational(107, 100)) == -sp.Rational(
        8747973, 10**8
    )
    assert polynomial.subs(strength, sp.Rational(27, 25)) == sp.Rational(288107, 390625)
    # Strictly positive coefficients give monotonicity for every strength>=1.
    shifted_derivative = sp.expand(
        sp.diff(polynomial, strength).subs(strength, 1 + displacement) / 12
    )
    assert (
        shifted_derivative
        == 9 * displacement**3 + 26 * displacement**2 + 23 * displacement + 5
    )
    assert Q(207, 100) / 3 > Q(83, 100) ** 2
    assert Q(108, 83) + Q(2, 25) < Q(7, 5)
    assert Q(7, 5) ** 2 < 2
    # Thus at the fold eta+a/R<sqrt(2), cos(v)>-1/sqrt(2), and
    # the closing gap exceeds pi/2. It remains below the marked pi cut.
    assert Q(11, 10) / Q(3, 5) + Q(1, 10) < 2


def test_weak_contact_reflected_explicit_partner_is_a_cut_free_saddle(mp):
    """The other positive root is supplied algebraically, never searched for."""
    tolerance = mp.mpf("1e-85")
    strength = (3 + mp.sqrt(39)) / 9
    radius = -mp.mpf(1) / 3 + mp.sqrt(strength + mp.mpf(2) / 3)
    eta, alpha = strength - 1, 2 * mp.pi / 9
    assert abs(radius**3 - (strength + 1) * radius + strength**2) < tolerance
    assert strength + 1 - 3 * radius**2 < 0
    cosine_v = -(strength / radius + eta) / 2
    u, v = mp.acos((strength + cosine_v) / radius), mp.acos(cosine_v)
    difference_cosine = mp.cos(v - u)
    origins = (-v, -u, mp.mpf(0), u, v, mp.mpf(0))
    phase = tuple(
        alpha * (slot - 4) + origins[child] for child in range(6) for slot in range(9)
    )
    rim_gaps = (v - u, u, u, v - u, 2 * mp.pi - 2 * v)
    assert all(0 < gap < mp.pi for gap in rim_gaps)
    assert abs(sum(rim_gaps) - 2 * mp.pi) < tolerance
    # Every internal gap is alpha, and additive slot/child phases make all
    # mixed host periods zero. Replicate a negative collective odd direction.
    wheel_witness = (
        -mp.mpf(1),
        -difference_cosine / radius,
        mp.mpf(0),
        difference_cosine / radius,
        mp.mpf(1),
        mp.mpf(0),
    )
    witness = tuple(value for value in wheel_witness for _ in range(9))
    current, quadratic = [mp.mpf(0)] * 54, mp.mpf(0)
    for left, right, weight in _weighted_edges(eta):
        gap = phase[right] - phase[left]
        exchange = weight * mp.sin(gap)
        current[left] += exchange
        current[right] -= exchange
        quadratic += weight * mp.cos(gap) * (witness[right] - witness[left]) ** 2
    expected = 18 * (1 - cosine_v**2) * (strength + 1 - 3 * radius**2) / radius**2
    assert max(abs(value) for value in current) < tolerance
    assert abs(sum(witness)) < tolerance
    assert abs(quadratic - expected) < tolerance
    assert quadratic < 0


@pytest.mark.parametrize("eta_value", (Q(1, 100000), Q(1)))
def test_weak_contact_carrier_marked_cut_has_negative_full_curvature(mp, eta_value):
    """A static cut point checks the boundary obstruction, not a target verdict."""
    eta, alpha, tolerance = _mp(mp, eta_value), 2 * mp.pi / 9, mp.mpf("1e-85")
    origins = (mp.mpf(0), mp.pi, 5 * mp.pi / 4, 3 * mp.pi / 2, 7 * mp.pi / 4, mp.pi / 7)
    phase = tuple(
        alpha * (slot - 4) + origins[child] for child in range(6) for slot in range(9)
    )
    witness = tuple(
        1 if child == 0 else -1 if child == 1 else 0
        for child in range(6)
        for _ in range(9)
    )
    quadratic = sum(
        weight * mp.cos(phase[b] - phase[a]) * (witness[b] - witness[a]) ** 2
        for a, b, weight in _weighted_edges(eta)
    )
    assert sum(witness) == 0
    assert abs(quadratic - 9 * (-4 + mp.sqrt(2))) < tolerance
    assert quadratic < -18
