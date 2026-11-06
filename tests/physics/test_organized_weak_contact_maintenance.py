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

    # The actual source circle closes by a full turn on the nine donor
    # vertices. Both complete rows and storage descend to that circular state;
    # the relative lift still records the turn used in the degree argument.
    shifted_phase = phase + mp.matrix([0] * 45 + [2 * mp.pi] * 9)
    shifted_current, potential_change = mp.zeros(54, 1), mp.mpf(0)
    for left, right, weight in edges:
        gap = shifted_phase[right] - shifted_phase[left]
        exchange = weight * mp.sin(gap)
        shifted_current[left] += exchange
        shifted_current[right] -= exchange
        potential_change += weight * (mp.cos(phase[right] - phase[left]) - mp.cos(gap))
    shifted_form_rate = (-form_gradient + gamma * shifted_current) / 4
    assert max(abs(value) for value in shifted_form_rate - form_rate) < tolerance
    assert abs(potential_change) < tolerance
    assert (
        abs(shifted_phase[45] - shifted_phase[0] - phase[45] + phase[0] - 2 * mp.pi)
        < tolerance
    )
    assert abs(sum(shifted_phase - phase) / 54 - mp.pi / 3) < tolerance

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


def test_weak_contact_antipodal_saddle_branch_rational_domain():
    """The continued lower root stays in an acute rim on the whole source range."""
    radius, strength = sp.symbols("radius strength", real=True)
    cubic = radius**3 - (strength + 1) * radius + strength**2
    anchor_radius = (sp.sqrt(5) - 1) / 2
    assert cubic.subs(
        {radius: sp.Rational(3, 5), strength: sp.Rational(999, 1000)}
    ) == (sp.Rational(14601, 10**6))
    assert (
        sp.simplify(
            cubic.subs(radius, anchor_radius)
            - (strength - 1) * (strength + 1 - anchor_radius)
        )
        == 0
    )
    assert (sp.Rational(5, 8) - anchor_radius).is_positive
    assert 3 * Q(5, 8) ** 2 - Q(1999, 1000) < 0
    # The reconstructed cos(v) lies strictly between -5/6 and -3/4.
    assert Q(99999, 100000) / Q(5, 8) - Q(1, 100000) > Q(3, 2)
    assert Q(1) / Q(3, 5) == Q(5, 3)
    assert Q(99999, 100000) - Q(5, 6) > 0
    assert 1 - Q(5, 6) > 0
    assert 2 * Q(3, 4) ** 2 - 1 == Q(1, 8)
    assert Q(1, 5) - 6 * Q(1, 100000) > 0
    assert 90 * Q(1, 100000) < Q(3, 250)


def test_weak_contact_positive_spokes_admit_an_acute_low_energy_saddle(mp):
    """One exact algebraic saddle uses positive weights in every actual row.

    The supplied radius fixes the state before evaluation. No stationary root
    search, source history or trajectory selects it. Its tiny negative mode
    needs the declared 100-digit static arithmetic, not a float64 verdict.
    """
    radius_exact = (sp.sqrt(5) - 1) / 2 - sp.Rational(1, 10**6)
    strength_exact = (
        radius_exact + sp.sqrt(radius_exact**2 - 4 * radius_exact**3 + 4 * radius_exact)
    ) / 2

    def exact_cubic(strength):
        return sp.simplify(
            radius_exact**3 - (strength + 1) * radius_exact + strength**2
        )

    assert exact_cubic(strength_exact) == 0
    assert exact_cubic(sp.Rational(99999, 100000)).is_negative
    assert exact_cubic(1).is_positive
    # The positive quadratic root in strength is therefore in this interval.
    radius = (mp.sqrt(5) - 1) / 2 - mp.mpf(1) / 10**6
    strength = (radius + mp.sqrt(radius**2 - 4 * radius**3 + 4 * radius)) / 2
    eta, alpha, tolerance = 1 - strength, 2 * mp.pi / 9, mp.mpf("1e-85")
    assert 0 < eta < mp.mpf(1) / 100000
    cosine_v = -(strength / radius + strength - 1) / 2
    u, v = mp.acos((strength + cosine_v) / radius), mp.acos(cosine_v)
    raw_origins = (-v, -u, mp.mpf(0), u, v, mp.pi)
    origins = tuple(value - mp.pi / 6 for value in raw_origins)
    phase = tuple(
        alpha * (slot - 4) + origins[child] for child in range(6) for slot in range(9)
    )
    assert abs(sum(phase)) < tolerance
    carrier_profile = mp.matrix(
        [alpha * (slot - 4) for _ in range(6) for slot in range(9)]
    )
    assert (
        max(abs(value) for value in _fine_projection(mp, phase) - carrier_profile)
        < tolerance
    )

    current, hessian, storage, gaps = mp.zeros(54, 1), mp.zeros(54), mp.mpf(0), {}
    for left, right, weight in _weighted_edges(eta):
        assert weight > 0
        difference = phase[right] - phase[left]
        exchange = weight * mp.sin(difference)
        current[left] += exchange
        current[right] -= exchange
        curvature = weight * mp.cos(difference)
        hessian[left, left] += curvature
        hessian[right, right] += curvature
        hessian[left, right] -= curvature
        hessian[right, left] -= curvature
        storage += weight * (1 - mp.cos(difference))
        gaps[left, right] = mp.atan2(mp.sin(difference), mp.cos(difference))
    assert max(abs(value) for value in current) < tolerance
    # At zero form, these are also the complete form and phase stationary rows.
    laplacian = _product_laplacian(mp, eta)
    positive_factor = mp.cholesky(laplacian[:53, :53])
    assert all(positive_factor[index, index] > 0 for index in range(53))
    for child in range(6):
        cycle = tuple(
            gaps[9 * child + slot, 9 * child + (slot + 1) % 9] for slot in range(9)
        )
        assert abs(sum(cycle) - 2 * mp.pi) < tolerance
        assert all(0 < gap < mp.pi / 2 for gap in cycle)
    for slot in range(9):
        cycle = tuple(
            gaps[9 * child + slot, 9 * ((child + 1) % 5) + slot] for child in range(5)
        )
        assert abs(sum(cycle) - 2 * mp.pi) < tolerance
        assert all(0 < gap < mp.pi / 2 for gap in cycle)
    for child in range(5):
        for slot in range(9):
            left, right = 9 * child + slot, 9 * ((child + 1) % 5) + slot
            up_left = 9 * child + (slot + 1) % 9
            up_right = 9 * ((child + 1) % 5) + (slot + 1) % 9
            assert (
                abs(
                    gaps[left, right]
                    + gaps[right, up_right]
                    - gaps[up_left, up_right]
                    - gaps[left, up_left]
                )
                < tolerance
            )
    ground = 54 * (1 - mp.cos(alpha)) + 45 * (1 - mp.cos(2 * mp.pi / 5))
    assert 0 < storage - ground < 90 * eta < mp.mpf(3) / 250
    # Independent full-edge assembly retains one negative collective mode,
    # one common-rotation zero, and all 52 remaining positive phase modes.
    eigenvalues = mp.eigsy(hessian, eigvals_only=True)
    assert eigenvalues[0] < -mp.mpf("1e-40")
    assert abs(eigenvalues[1]) < tolerance
    assert all(value > tolerance for value in eigenvalues[2:])


def test_weak_contact_collective_shape_global_convexity_and_complex_disk():
    """Exact majorants admit a unique analytic host shape over the whole circle."""

    def cosine_upper(value):
        return 1 - value**2 / 2 + value**4 / 24 - value**6 / 720 + value**8 / 40320

    gap = Q(29, 20)
    remaining_gap_lower = (2 * Q(157, 50) - gap) / 4
    excess_lower = (
        5 * Q(309, 1000) - cosine_upper(gap) - 4 * cosine_upper(remaining_gap_lower)
    )
    assert Q(2236, 1000) ** 2 < 5
    assert excess_lower > Q(1, 750)
    assert cos(gap).lo > Q(1, 10)
    assert Q(1, 10) - Q(1, 100000) > 0

    cosine_twist = (sp.sqrt(5) - 1) / 4
    cycle = sp.zeros(5)
    incidence = sp.zeros(5)
    for index in range(5):
        incidence[index, index], incidence[index, (index + 1) % 5] = -1, 1
    cycle = incidence.T * incidence
    first_row = tuple(map(sp.Rational, (Q(2, 5), 0, Q(-1, 5), Q(-1, 5), 0)))
    inverse = sp.Matrix(
        5, 5, lambda left, right: first_row[(right - left) % 5] / cosine_twist
    )
    assert sp.simplify(cosine_twist * cycle * inverse) == sp.eye(5) - sp.ones(5) / 5
    inverse_norm = sp.simplify(sum(abs(value) for value in inverse.row(0)))
    difference_norm = sp.simplify(
        sum(abs(value) for value in (inverse * incidence.T).row(0))
    )
    assert sp.simplify(inverse_norm - 4 / (5 * cosine_twist)) == 0
    assert sp.simplify(difference_norm - 6 / (5 * cosine_twist)) == 0
    assert (sp.Rational(259, 100) - inverse_norm).is_positive
    assert (sp.Rational(389, 100) - difference_norm).is_positive
    first_mode_inverse = sp.simplify(1 / (cosine_twist * (2 - 2 * cosine_twist)))
    assert sp.simplify(first_mode_inverse - (1 + 3 * sp.sqrt(5) / 5)) == 0
    assert (sp.Rational(47, 20) - first_mode_inverse).is_positive
    first_mode_difference = sp.sqrt(10 - 2 * sp.sqrt(5)) * first_mode_inverse / 2
    assert (sp.Rational(69, 25) - first_mode_difference).is_positive

    disk, ball, inverse_bound, difference_bound = (
        Q(3, 200),
        Q(1, 100),
        Q(259, 100),
        Q(389, 100),
    )
    first_shape, first_difference = disk * Q(47, 20), disk * Q(69, 25)
    shape, difference = Q(23, 500), Q(31, 500)
    assert first_shape == Q(141, 4000)
    assert first_difference == Q(207, 5000)
    assert first_shape + ball < shape
    assert first_difference + 2 * ball < difference

    def cosh_remainder(value):
        return value**2 / (2 * (1 - value**2 / 12))

    def sinh_upper(value):
        return value / (1 - value**2 / 6)

    def sinh_remainder(value):
        return value**3 / (6 * (1 - value**2 / 20))

    sine_bound, cosine_bound = Q(24, 25), Q(31, 100)
    assert (sp.Rational(31, 100) - cosine_twist).is_positive
    assert (sp.Rational(24, 25) - sp.sqrt(10 + 2 * sp.sqrt(5)) / 4).is_positive
    contraction = 2 * difference_bound * (
        sine_bound * sinh_upper(difference) + cosine_bound * cosh_remainder(difference)
    ) + inverse_bound * disk / (1 - shape)
    center_residual = difference_bound * (
        sine_bound * cosh_remainder(first_difference)
        + cosine_bound * sinh_remainder(first_difference)
    ) + inverse_bound * disk * first_shape / (1 - first_shape)
    assert contraction < Q(13, 25)
    assert center_residual < Q(47, 10000)
    assert Q(47, 10000) + Q(13, 25) * ball < ball
    assert 5 * disk / (1 - shape) < Q(2, 25)


def test_weak_contact_collective_fifth_order_pinning_and_global_fourier_tail():
    """The exact parity curvature fixes the leading sign; all higher modes fit."""
    phi = (1 + sp.sqrt(5)) / 2
    radius = 1 / phi
    radius_derivative = sp.simplify((2 - radius) / (2 - 3 * radius**2))
    assert sp.simplify(radius_derivative - phi) == 0
    vanishing_factor_derivative = sp.simplify((2 * radius + 1) * radius_derivative)
    determinant_leading = sp.simplify(vanishing_factor_derivative**4 / (8 * radius**4))
    assert sp.simplify(determinant_leading - 25 * phi**8 / 8) == 0

    # Fixing the five-host mean uses l=(2,2,1), not the unweighted
    # coordinate mean of the three-dimensional reflection-even block.
    cosine_twist = (sp.sqrt(5) - 1) / 4
    even_anchor = cosine_twist * sp.Matrix([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])
    mean_weights = sp.Matrix([2, 2, 1])
    denominator = sp.simplify(
        (mean_weights.T * even_anchor.adjugate() * mean_weights)[0]
    )
    curvature_leading = sp.simplify(50 * determinant_leading / denominator)
    assert sp.simplify(curvature_leading - 25 * phi**10) == 0
    assert (5 * phi**10 - 600).is_positive

    ratio = sp.symbols("ratio", real=True)
    # Sum (n+7)(n+12) ratio**n using differentiated geometric series.
    geometric = 1 / (1 - ratio)
    first_moment = ratio * sp.diff(geometric, ratio)
    second_moment = ratio * sp.diff(first_moment, ratio)
    series = sp.factor(second_moment + 19 * first_moment + 84 * geometric)
    assert (
        sp.simplify(series - (84 - 148 * ratio + 66 * ratio**2) / (1 - ratio) ** 3) == 0
    )
    disk, eta, torque_bound = Q(3, 200), Q(1, 100000), Q(2, 25)
    endpoint = eta / disk
    tail = (
        torque_bound
        * eta**2
        / disk**7
        * (84 - 148 * endpoint + 66 * endpoint**2)
        / (25 * (1 - endpoint) ** 3)
    )
    assert tail < 200 < 600


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


def _fine_projection(mp, values):
    return mp.matrix(
        [
            values[9 * child + slot] - sum(values[9 * child : 9 * child + 9]) / 9
            for child in range(6)
            for slot in range(9)
        ]
    )


@pytest.fixture(scope="module")
def bounded_fine_state(mp):
    """One declared synthetic state; this is no source or stationary candidate."""
    eta, gamma, alpha = mp.mpf(1) / 100000, 1 / (1023 * mp.pi), 2 * mp.pi / 9
    origins = tuple(2 * mp.pi * (child - 2) / 5 for child in range(5)) + (mp.pi / 7,)
    means = tuple(mp.mpf(2 * child - 5) / 10000 for child in range(6))
    fine_phase, fine_form = [], []
    for child in range(6):
        raw_phase = tuple(Q((7 * child + 5 * slot) % 13 - 6, 2000) for slot in range(9))
        raw_form = tuple(Q((3 * child + 7 * slot) % 17 - 8, 4000) for slot in range(9))
        phase_row = tuple(value - sum(raw_phase) / 9 for value in raw_phase)
        form_row = tuple(value - sum(raw_form) / 9 for value in raw_form)
        assert sum(phase_row) == sum(form_row) == 0
        fine_phase.extend(_mp(mp, value) for value in phase_row)
        fine_form.extend(_mp(mp, value) for value in form_row)
    fine_phase, fine_form = mp.matrix(fine_phase), mp.matrix(fine_form)
    carrier = mp.matrix(
        [alpha * (slot - 4) + origins[child] for child in range(6) for slot in range(9)]
    )
    phase = carrier + fine_phase
    form = mp.matrix([value for value in means for _ in range(9)]) + fine_form
    return {
        "eta": eta,
        "gamma": gamma,
        "alpha": alpha,
        "origins": origins,
        "means": means,
        "q": fine_phase,
        "z": fine_form,
        "carrier": carrier,
        "phase": phase,
        "form": form,
        "edges": _weighted_edges(eta),
        "laplacian": _product_laplacian(mp, eta),
    }


def test_weak_contact_fine_capture_strict_gap_and_decay_constants(mp):
    """Outward bounds exclude a pi/3 internal gap from the admitted sublevel."""
    pi = pi_interval()
    ground = 9 * (1 - cos(2 * pi / 9))
    positive_face = Q(1, 2) + 8 * (1 - cos(5 * pi / 24))
    negative_face = Q(1, 2) + 8 * (1 - cos(7 * pi / 24))
    excess = positive_face - ground
    assert excess.lo > Q(1, 25) > Q(3, 250)
    assert negative_face.lo > positive_face.hi
    for sign, enclosure in ((1, positive_face), (-1, negative_face)):
        gaps = (sign * mp.pi / 3,) + ((2 * mp.pi - sign * mp.pi / 3) / 8,) * 8
        assert abs(sum(gaps) - 2 * mp.pi) < mp.mpf("1e-85")
        assert all(abs(gap) < mp.pi / 2 for gap in gaps)
        value = sum(1 - mp.cos(gap) for gap in gaps)
        assert _mp(mp, enclosure.lo) < value < _mp(mp, enclosure.hi)

    cycle_gap = 2 - 2 * cos(2 * pi / 9)
    assert cycle_gap.lo > Q(2, 5)
    coercivity, gamma_upper = Q(19, 100), Q(1, 3000)
    assert cycle_gap.lo / 2 - 6 * Q(1, 100000) > coercivity
    # These are the Young remainder and norm constants for the common F.
    assert Q(2, 5) - 200 * gamma_upper**2 / coercivity > coercivity / 2
    assert Q(3, 2) * gamma_upper**2 < 1
    assert coercivity / 8 == Q(19, 800)


def test_weak_contact_fine_capture_projection_and_unequal_spoke_control(mp):
    """Exact product commutation needs slot-independent contact weights."""
    eta = sp.Rational(1, 100000)
    projector = sp.eye(54) - sp.kronecker_product(sp.eye(6), sp.ones(9) / 9)
    laplacian = _product_laplacian(sp, eta)
    edge_laplacian = sp.zeros(54)
    for left, right, weight in _weighted_edges(eta):
        edge_laplacian[left, left] += weight
        edge_laplacian[right, right] += weight
        edge_laplacian[left, right] -= weight
        edge_laplacian[right, left] -= weight
    assert laplacian == edge_laplacian
    assert projector**2 == projector
    assert projector.rank() == 48
    assert projector * laplacian == laplacian * projector

    coarse_form = sp.Matrix(
        [sp.Rational(2 * child - 5, 10000) for child in range(6) for _ in range(9)]
    )
    assert projector * laplacian * coarse_form == sp.zeros(54, 1)
    spoke = sp.zeros(54, 1)
    spoke[0], spoke[45] = 1, -1
    unequal = laplacian + eta * spoke * spoke.T
    assert projector * unequal != unequal * projector
    expected = eta * (coarse_form[0] - coarse_form[45]) * projector * spoke
    assert projector * unequal * coarse_form == expected != sp.zeros(54, 1)

    # The same single changed spoke breaks the normal-current closure.
    alpha, strength = 2 * mp.pi / 9, mp.mpf(1) / 100000
    origins = tuple(2 * mp.pi * (child - 2) / 5 for child in range(5)) + (mp.pi / 7,)
    carrier = tuple(
        alpha * (slot - 4) + origins[child] for child in range(6) for slot in range(9)
    )
    current = mp.zeros(54, 1)
    for left, right, weight in _weighted_edges(strength):
        exchange = weight * mp.sin(carrier[right] - carrier[left])
        current[left] += exchange
        current[right] -= exchange
    assert max(abs(value) for value in _fine_projection(mp, current)) < mp.mpf("1e-85")
    extra = strength * mp.sin(carrier[45] - carrier[0])
    current[0] += extra
    current[45] -= extra
    assert max(abs(value) for value in _fine_projection(mp, current)) > strength / 10


def test_weak_contact_fine_capture_common_lyapunov_identity(mp, bounded_fine_state):
    """Differentiate F along the independent full edge field, not a trajectory."""
    state, tolerance = bounded_fine_state, mp.mpf("1e-85")
    phase, form = state["phase"], state["form"]
    q, z, gamma, laplacian = state["q"], state["z"], state["gamma"], state["laplacian"]
    for child in range(6):
        for slot in range(9):
            gap = phase[9 * child + (slot + 1) % 9] - phase[9 * child + slot]
            gap += 2 * mp.pi if slot == 8 else 0
            assert abs(gap) < mp.pi / 3
    for child in range(5):
        for slot in range(9):
            gap = phase[9 * ((child + 1) % 5) + slot] - phase[9 * child + slot]
            gap += 2 * mp.pi if child == 4 else 0
            assert abs(gap) < mp.pi / 2

    gradient, carrier_gradient, form_gradient = (mp.zeros(54, 1) for _ in range(3))
    storage = mp.mpf(0)
    for left, right, weight in state["edges"]:
        gap, form_gap = phase[right] - phase[left], form[right] - form[left]
        phase_exchange = weight * mp.sin(gap)
        carrier_exchange = weight * mp.sin(
            state["carrier"][right] - state["carrier"][left]
        )
        gradient[left] -= phase_exchange
        gradient[right] += phase_exchange
        carrier_gradient[left] -= carrier_exchange
        carrier_gradient[right] += carrier_exchange
        form_gradient[left] -= weight * form_gap
        form_gradient[right] += weight * form_gap
        storage += weight * (form_gap**2 / 2 + 1 - mp.cos(gap))
    ground = 54 * (1 - mp.cos(state["alpha"])) + 45 * (1 - mp.cos(2 * mp.pi / 5))
    assert 0 < storage - ground < mp.mpf(3) / 250
    assert (
        max(abs(value) for value in _fine_projection(mp, carrier_gradient)) < tolerance
    )
    assert max(abs(value) for value in form_gradient - laplacian * form) < tolerance
    g = _fine_projection(mp, gradient)
    q_squared, z_squared = (q.T * q)[0], (z.T * z)[0]
    coercivity = mp.mpf(19) / 100
    assert (q.T * g)[0] > coercivity * q_squared
    assert (g.T * g)[0] <= 100 * q_squared
    assert (z.T * laplacian * z)[0] > mp.mpf(2) / 5 * z_squared

    form_rate = (-form_gradient - gamma * gradient) / 4
    phase_rate = gamma * form_gradient / 4
    assert (
        max(
            abs(value)
            for value in _fine_projection(mp, form_rate)
            + (laplacian * z + gamma * g) / 4
        )
        < tolerance
    )
    assert (
        max(
            abs(value)
            for value in _fine_projection(mp, phase_rate) - gamma * laplacian * z / 4
        )
        < tolerance
    )

    def auxiliary_on_tangent(parameter):
        fine_phase = _fine_projection(
            mp, phase + parameter * phase_rate - state["carrier"]
        )
        fine_form = _fine_projection(mp, form + parameter * form_rate)
        mixed = fine_phase + gamma * fine_form
        return ((mixed.T * mixed)[0] + gamma**2 * (fine_form.T * fine_form)[0]) / 2

    auxiliary = auxiliary_on_tangent(0)
    bracket = (q.T * g)[0] + (z.T * laplacian * z)[0] + 2 * gamma * (z.T * g)[0]
    derivative = mp.diff(auxiliary_on_tangent, 0)
    assert abs(derivative + gamma**2 * bracket / 4) < tolerance
    assert q_squared / 4 <= auxiliary <= q_squared + z_squared
    assert bracket >= coercivity * (q_squared + z_squared) / 2
    assert derivative <= -19 * gamma**2 * auxiliary / 800


def test_weak_contact_fine_capture_mean_rows_keep_nonzero_residual(
    mp, bounded_fine_state
):
    """A quadratic transient remainder is retained rather than exact closure."""
    state, tolerance = bounded_fine_state, mp.mpf("1e-85")
    eta, gamma, phase, form = (
        state["eta"],
        state["gamma"],
        state["phase"],
        state["form"],
    )
    q, origins, means = state["q"], state["origins"], mp.matrix(state["means"])
    wheel_edges = tuple(
        (child, (child + 1) % 5, mp.mpf(1)) for child in range(5)
    ) + tuple((child, 5, eta) for child in range(5))
    wheel_laplacian = mp.zeros(6)
    wheel_current, residual = mp.zeros(6, 1), mp.zeros(6, 1)
    fine_edge_square = mp.mpf(0)
    for left, right, weight in wheel_edges:
        wheel_laplacian[left, left] += weight
        wheel_laplacian[right, right] += weight
        wheel_laplacian[left, right] -= weight
        wheel_laplacian[right, left] -= weight
        gap = origins[right] - origins[left]
        exchange = weight * mp.sin(gap)
        wheel_current[left] += exchange
        wheel_current[right] -= exchange
        fine_gaps = tuple(q[9 * right + slot] - q[9 * left + slot] for slot in range(9))
        assert abs(sum(fine_gaps)) < tolerance
        remainder = (
            weight
            * sum(mp.sin(gap + fine_gap) - mp.sin(gap) for fine_gap in fine_gaps)
            / 9
        )
        residual[left] += remainder
        residual[right] -= remainder
        fine_edge_square += weight * sum(value**2 for value in fine_gaps)

    current, form_gradient = mp.zeros(54, 1), mp.zeros(54, 1)
    for left, right, weight in state["edges"]:
        exchange = weight * mp.sin(phase[right] - phase[left])
        current[left] += exchange
        current[right] -= exchange
        form_exchange = weight * (form[left] - form[right])
        form_gradient[left] += form_exchange
        form_gradient[right] -= form_exchange
    form_rate = (-form_gradient + gamma * current) / 4
    phase_rate = gamma * form_gradient / 4
    mean_form_rate = mp.matrix(
        [sum(form_rate[9 * child : 9 * child + 9]) / 9 for child in range(6)]
    )
    mean_phase_rate = mp.matrix(
        [sum(phase_rate[9 * child : 9 * child + 9]) / 9 for child in range(6)]
    )
    expected_form_rate = (
        -wheel_laplacian * means + gamma * (wheel_current + residual)
    ) / 4
    expected_phase_rate = gamma * wheel_laplacian * means / 4
    assert max(abs(value) for value in mean_form_rate - expected_form_rate) < tolerance
    assert (
        max(abs(value) for value in mean_phase_rate - expected_phase_rate) < tolerance
    )
    residual_norm = sum(abs(value) for value in residual)
    assert residual_norm > mp.mpf("1e-8")
    assert abs(sum(residual)) < tolerance
    q_squared, z = (q.T * q)[0], state["z"]
    mixed = q + gamma * z
    auxiliary = ((mixed.T * mixed)[0] + gamma**2 * (z.T * z)[0]) / 2
    assert residual_norm <= fine_edge_square / 9 <= mp.mpf(2) / 3 * q_squared
    assert residual_norm <= 8 * auxiliary / 3


@pytest.mark.parametrize("origin_index", (0, 2))
def test_weak_contact_source_basin_relative_rows_retain_fine_torque(
    mp, bounded_fine_state, origin_index
):
    """Full weighted edges fix the donor sign and the corrected-origin row."""
    state, tolerance = bounded_fine_state, mp.mpf("1e-85")
    eta, gamma = state["eta"], state["gamma"]
    phase, form = mp.matrix(state["phase"]), state["form"]
    beta = mp.pi / 7 + 2 * origin_index * mp.pi / 5
    for slot in range(9):
        phase[45 + slot] += 2 * origin_index * mp.pi / 5

    current, form_gradient = mp.zeros(54, 1), mp.zeros(54, 1)
    for left, right, weight in state["edges"]:
        exchange = weight * mp.sin(phase[right] - phase[left])
        current[left] += exchange
        current[right] -= exchange
        form_exchange = weight * (form[left] - form[right])
        form_gradient[left] += form_exchange
        form_gradient[right] -= form_exchange
    form_rate = (-form_gradient + gamma * current) / 4
    phase_rate = gamma * form_gradient / 4

    def host_minus_donor(values):
        return sum(values[:45]) / 45 - sum(values[45:]) / 9

    origin = host_minus_donor(phase)
    chi = gamma * host_minus_donor(form)
    origin_rate = host_minus_donor(phase_rate)
    chi_rate = gamma * host_minus_donor(form_rate)
    donor_torque = (
        eta
        * sum(
            mp.sin(phase[9 * child + slot] - phase[45 + slot])
            for child in range(5)
            for slot in range(9)
        )
        / 9
    )
    assert abs(origin + beta) < tolerance
    assert abs(sum(current[45:]) / 9 - donor_torque) < tolerance
    assert abs(origin_rate - 3 * eta * chi / 2) < tolerance
    assert (
        abs(chi_rate + 3 * eta * chi / 2 + 3 * gamma**2 * donor_torque / 10) < tolerance
    )
    assert abs(origin_rate + chi_rate + 3 * gamma**2 * donor_torque / 10) < tolerance

    # The five uniform host origins cancel the mean-only spoke torque. The
    # supplied, nonuniform fine arrays do not: dropping them changes y'.
    means = tuple(sum(phase[9 * child : 9 * child + 9]) / 9 for child in range(6))
    mean_torque = eta * sum(mp.sin(means[child] - means[5]) for child in range(5))
    remainder = donor_torque - mean_torque
    fine_square = sum(
        (state["q"][9 * child + slot] - state["q"][45 + slot]) ** 2
        for child in range(5)
        for slot in range(9)
    )
    q_squared = (state["q"].T * state["q"])[0]
    assert abs(mean_torque) < tolerance
    assert abs(remainder) > mp.mpf("1e-14")
    assert abs(remainder) <= eta * fine_square / 18 <= eta * q_squared / 3
    assert abs(origin_rate + chi_rate) > gamma**2 * mp.mpf("1e-15")


def test_weak_contact_source_basin_moving_shape_lyapunov_identity():
    """Differentiate the quadratic with moving u and an independent fine row."""
    gamma, origin_rate = sp.symbols("gamma origin_rate", real=True)
    error = sp.Matrix(sp.symbols("e0:2", real=True))
    form = sp.Matrix(sp.symbols("p0:2", real=True))
    gradient = sp.Matrix(sp.symbols("g0:2", real=True))
    shape_derivative = sp.Matrix(sp.symbols("u0:2", real=True))
    remainder = sp.Matrix(sp.symbols("r0:2", real=True))
    diagonal, coupling = sp.symbols("diagonal coupling", real=True)
    laplacian = sp.Matrix([[diagonal, coupling], [coupling, diagonal]])
    error_rate = gamma * laplacian * form / 4 - shape_derivative * origin_rate
    form_rate = (-laplacian * form - gamma * gradient + gamma * remainder) / 4
    mixed = error + gamma * form
    quadratic = ((mixed.T * mixed)[0] + gamma**2 * (form.T * form)[0]) / 2
    differentiated = sum(
        sp.diff(quadratic, variable) * rate
        for variable, rate in zip(
            tuple(error) + tuple(form), tuple(error_rate) + tuple(form_rate)
        )
    )
    expected = (
        -(gamma**2)
        * (
            (error.T * gradient)[0]
            + (form.T * laplacian * form)[0]
            + 2 * gamma * (form.T * gradient)[0]
        )
        / 4
        - (mixed.T * shape_derivative)[0] * origin_rate
        + gamma**2 * ((error + 2 * gamma * form).T * remainder)[0] / 4
    )
    assert sp.expand(differentiated - expected) == 0
    # Both omitted terms are genuine polynomial terms, not a fixed-shape
    # closure or a silently unforced approximation.
    assert sp.diff(differentiated, shape_derivative[0]) != 0
    assert sp.diff(differentiated, remainder[0]) != 0


def test_weak_contact_source_basin_coarse_source_and_tracking_barrier():
    """Derive the fixed-strength constants from source, Young and mean bounds."""
    eta, gamma_upper = Q(1, 100000), Q(1, 3000)
    duration, initial_form, zeta = Q(1, 5000), Q(1, 100), Q(1, 10**6)
    form_bound = initial_form + gamma_upper * duration
    phase_drift = 2 * gamma_upper * form_bound * duration
    phase_fine = 2 * (zeta + phase_drift)
    form_fine = 2 * form_bound
    assert phase_fine < Q(1, 400000)
    assert gamma_upper * form_fine < Q(7, 10**6)
    source_fine = (
        54
        * ((phase_fine + gamma_upper * form_fine) ** 2 + (gamma_upper * form_fine) ** 2)
        / 2
    )
    fine_bound, shape_bound = Q(1, 10**8), Q(1, 5)
    assert source_fine < fine_bound
    # Jensen and the host-shape Hessian, not a substituted equilibrium,
    # control h; p uses the conserved zero initial host form mean.
    host_shape_squared = 2 * Q(1, 750) / Q(1, 10)
    assert host_shape_squared < Q(1, 6) ** 2
    assert 5 * (Q(46, 15) * eta) ** 2 < (7 * eta) ** 2
    assert 5 * form_bound**2 < Q(1, 40) ** 2
    assert Q(1, 6) + 7 * eta + gamma_upper / 40 < shape_bound

    coercivity, gradient_norm = Q(9, 100), Q(5)
    assert Q(1, 10) - eta > coercivity
    assert 4 + eta < gradient_norm
    assert 5 < (25 * coercivity) ** 2
    assert 1 - 2 * gradient_norm**2 * gamma_upper**2 / coercivity > coercivity / 2
    shape_decay = coercivity / 16
    moving_coefficient = Q(28)
    assert (25 * Q(3, 2)) ** 2 < 2 * moving_coefficient**2
    mean_gain, mean_damping, torque_error = Q(3, 10), Q(3, 2), Q(5)
    assert 4 * 5 < torque_error**2
    weight, decay, fine_decay = Q(20), Q(1, 200), Q(19, 800)
    torque_bound, fine_torque, fine_forcing = Q(1000), Q(4, 3), Q(8, 3)
    # The positive root of r^2-r-1 is below 13/8. Combine its fifth-order
    # coefficient with the already certified Fourier-tail allowance.
    assert Q(13, 8) ** 2 - Q(13, 8) - 1 > 0
    assert 5 * Q(13, 8) ** 10 + 200 < torque_bound
    assert shape_decay - weight * mean_gain * torque_error * eta**2 > decay
    chi_coercivity = weight * mean_damping - moving_coefficient
    assert chi_coercivity * eta > weight * decay * gamma_upper**2
    combined_fine = fine_forcing + weight * mean_gain * fine_torque * eta**2
    assert combined_fine < 3
    plateau = weight * mean_gain * torque_bound * eta**6 / decay
    transient_convolution = Q(3) / (fine_decay - decay)
    assert transient_convolution == 160
    transient = shape_bound + transient_convolution * fine_bound
    assert transient < Q(21, 100)
    chi_bound = (
        mean_gain
        * gamma_upper**2
        * (
            torque_bound * eta**4
            + torque_error * (transient + plateau)
            + fine_torque * fine_bound
        )
        / mean_damping
    )
    assert chi_bound < gamma_upper**2 / 4 < Q(1, 10**6)

    impulse = (
        mean_gain
        * eta
        * (torque_error * transient / decay + fine_torque * fine_bound / fine_decay)
    )
    assert impulse < Q(1, 1000)
    inner, outer, chi_margin = Q(3, 5), Q(5, 8), Q(1, 10**6)
    pi_lower, pi_upper = Q(157, 50), Q(22, 7)
    sine_argument_lower = pi_lower - 5 * (outer + chi_margin)
    sine_argument_upper = pi_upper - 5 * (inner - chi_margin)
    assert 0 < sine_argument_lower < sine_argument_upper < 1
    sine_lower = Q(5, 6) * sine_argument_lower
    assert sine_lower > Q(1, 100)
    restoring_lower = (600 - 200) * eta**5 * sine_lower
    permanent_error = torque_error * eta * plateau
    assert permanent_error < 4 * eta**5 < restoring_lower
    assert impulse < outer - inner
    assert outer + chi_margin < pi_lower / 5
