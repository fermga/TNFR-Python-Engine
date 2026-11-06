"""Exact whole-support controls for the maintained-target Lyapunov obstruction.

These test the auxiliary functional of the admitted sine-law ratio family,
not another constitutive energy, an acute-entry prohibition or a trajectory
prediction. Support, held capacities and storage scale retain their premises.
"""

from fractions import Fraction as Q
from itertools import combinations

import networkx as nx
import pytest

from tests.physics.test_relational_sine_formation import _support


@pytest.fixture(scope="module")
def algebra():
    s = pytest.importorskip("sympy")
    edges, neighbors = _support()
    degrees = tuple(map(len, neighbors))
    capacity = tuple(
        Q(3, 2) if i == 6 else Q(1, 2) if i == 9 else Q(1) for i in range(11)
    )
    k = tuple(s.Rational(nu / degree) for nu, degree in zip(capacity, degrees))
    incidence = s.zeros(len(edges), 11)
    for row, (i, j) in enumerate(edges):
        incidence[row, i], incidence[row, j] = -1, 1
    return s, edges, neighbors, capacity, incidence, s.diag(*k)


def test_auxiliary_derivative_from_all_fine_rows_and_global_cosine_remainder(algebra):
    s, edges, neighbors, _, incidence, k = algebra
    x = s.Matrix(s.symbols("x0:11", real=True))
    sine = s.Matrix(s.symbols("s0:12", real=True))
    cosine = s.symbols("c0:12", real=True)
    a, e = s.symbols("a e", positive=True)
    coefficient = s.Rational(4, 5)
    mixed_coefficient = a / (2 * e)
    laplacian = incidence.T * incidence
    hessian = incidence.T * s.diag(*cosine) * incidence
    q, current = laplacian * x, -incidence.T * sine
    xdot = k * (-e * q + a * current)
    phasedot = a * k * q

    # Differentiate the original full neighbor sums, retaining all receiver
    # rows. Edge sine/cosine symbols represent their exact chain-rule jets.
    qdot = s.Matrix(
        [sum(xdot[i] - xdot[j] for j in row) for i, row in enumerate(neighbors)]
    )
    current_dot = s.zeros(11, 1)
    for edge, (i, j) in enumerate(edges):
        flux_derivative = cosine[edge] * (phasedot[j] - phasedot[i])
        current_dot[i] += flux_derivative
        current_dot[j] -= flux_derivative
    assert (current_dot + hessian * phasedot).applyfunc(s.expand) == s.zeros(11, 1)
    assert s.expand(sum(q)) == s.expand(sum(current)) == 0
    form_rate = q.dot(xdot)
    potential_rate = sine.dot(incidence * phasedot)
    assert s.expand(form_rate + potential_rate + e * q.dot(k * q)) == 0
    derivative = coefficient * form_rate + potential_rate
    derivative -= mixed_coefficient * (qdot.dot(k * current) + q.dot(k * current_dot))
    upper_quadratic = -coefficient * e * q.dot(k * q)
    upper_quadratic += a * (coefficient - 1) * q.dot(k * current)
    upper_quadratic += mixed_coefficient * e * q.dot(k * laplacian * k * current)
    upper_quadratic -= mixed_coefficient * a * current.dot(k * laplacian * k * current)
    upper_quadratic += mixed_coefficient * a * q.dot(k * laplacian * k * q)
    v = k * q
    remainder = (
        mixed_coefficient
        * a
        * sum(
            (1 - cosine[edge]) * (v[j] - v[i]) ** 2 for edge, (i, j) in enumerate(edges)
        )
    )
    # Every actual cosine is <=1, including nonacute/negative ones; no
    # donor-winding preservation or positive Hessian is assumed.
    assert s.expand(upper_quadratic - derivative - remainder) == 0


def test_full_support_supplies_both_positive_spectral_endpoint_bounds(algebra):
    s, edges, neighbors, capacity, incidence, k = algebra
    graph = nx.Graph()
    graph.add_nodes_from(range(11))
    graph.add_edges_from(edges)
    loads = {frozenset(edge): 0 for edge in edges}
    for i, j in combinations(graph, 2):
        path = nx.shortest_path(graph, i, j)
        length = len(path) - 1
        for first, second in zip(path, path[1:]):
            loads[frozenset((first, second))] += length
    # Sum over all-pairs path Cauchy--Schwarz yields lambda2(L)>=n/maxload.
    assert max(loads.values()) == 121
    combinatorial_gap = Q(len(graph), max(loads.values()))
    assert combinatorial_gap == Q(1, 11)
    assert min(k.diagonal()) == s.Rational(1, 4)
    assert min(k.diagonal()) * combinatorial_gap == s.Rational(1, 44)
    # Nonzero spectra of sqrt(K)Lsqrt(K) and sqrt(L)Ksqrt(L) coincide;
    # the latter is >=L/4 on the correct constant-orthogonal subspace.

    v = s.Matrix(s.symbols("v0:11", real=True))
    root_k = s.diag(*(s.sqrt(value) for value in k.diagonal()))
    b = root_k * incidence.T * incidence * root_k
    complementary_squares = sum(
        (root_k[i, i] * v[i] + root_k[j, j] * v[j]) ** 2 for i, j in edges
    )
    complementary_squares += sum(
        (3 - 2 * s.Rational(nu)) * v[i] ** 2 for i, nu in enumerate(capacity)
    )
    assert all(3 - 2 * nu >= 0 for nu in capacity)
    assert s.expand(3 * v.dot(v) - v.dot(b * v) - complementary_squares) == 0
    assert tuple(graph.degree[i] for i in graph) == tuple(map(len, neighbors))


def test_mode_quadratic_is_strict_on_the_entire_admitted_spectral_interval(algebra):
    s = algebra[0]
    lam, a, e, rho = s.symbols("lambda a e rho", positive=True)
    c, h = s.Rational(4, 5), a / (2 * e)
    mixed = a * (c - 1) + h * e * lam
    matrix = s.Matrix(((c * e - h * a * lam, -mixed / 2), (-mixed / 2, h * a * lam)))
    determinant_polynomial = (1 - c) ** 2 - (c + 1) * lam
    determinant_polynomial += (s.Rational(1, 4) + (a / e) ** 2) * lam**2
    assert s.expand(-4 * matrix.det() / a**2 - determinant_polynomial) == 0

    # 0<w/e<3/2 and pi>3 imply rho=a/e<1/2. This is a declared
    # sufficient domain, not a search for the largest admissible ratio.
    ratio_polynomial = determinant_polynomial.subs(a, rho * e)
    upper = ratio_polynomial.subs(rho, s.Rational(1, 2))
    assert s.diff(upper, lam, 2) > 0
    assert upper.subs(lam, s.Rational(1, 44)) < 0
    assert upper.subs(lam, 3) < 0
    assert c - s.Rational(3, 8) == s.Rational(17, 40) > 0
    # The original equal-half-weight certificate is recovered exactly.
    assert h.subs(e, s.Rational(1, 2)) == a
    original = (1 - c) ** 2 - s.Rational(9, 5) * lam
    original += (s.Rational(1, 4) + 4 * a**2) * lam**2
    assert s.expand(determinant_polynomial.subs(e, s.Rational(1, 2)) - original) == 0


def test_complete_field_functional_and_action_have_consistent_clock_covariance(algebra):
    s, _, _, _, incidence, k = algebra
    x = s.Matrix(s.symbols("clock_x0:11", real=True))
    sine = s.Matrix(s.symbols("clock_s0:12", real=True))
    q, current = incidence.T * incidence * x, -incidence.T * sine
    e, w, clock, maximum, allowance = s.symbols(
        "e w clock maximum allowance", positive=True
    )
    a = w / s.pi

    def full_rows(loss, exchange):
        return (k * (-loss * q + exchange * current)).col_join(exchange * k * q)

    field = full_rows(e, a)
    transformed = full_rows(e / clock, a / clock)
    assert (transformed - field / clock).applyfunc(s.simplify) == s.zeros(22, 1)
    coefficient = a / (2 * e)
    transformed_coefficient = (a / clock) / (2 * e / clock)
    assert s.simplify(transformed_coefficient - coefficient) == 0
    potential = s.Symbol("V", real=True)
    form_storage = x.dot(incidence.T * incidence * x) / 2
    auxiliary = s.Rational(4, 5) * form_storage + potential
    auxiliary -= coefficient * q.dot(k * current)
    assert (
        s.simplify(
            auxiliary.subs({e: e / clock, w: w / clock}, simultaneous=True) - auxiliary
        )
        == 0
    )
    # Unchanged W and a uniformly rescaled full vector field imply Wdot/clock.
    gradient = s.Matrix(s.symbols("W_gradient0:22", real=True))
    assert s.expand(gradient.dot(transformed) - gradient.dot(field) / clock) == 0

    action_time = e * s.pi**2 / (a**2 * maximum * allowance)
    transformed_time = action_time.subs({e: e / clock, w: w / clock}, simultaneous=True)
    assert s.simplify(transformed_time - clock * action_time) == 0
    # Scaling raw weights before the model's normalization is a different
    # operation: it leaves the effective pair unchanged, not field/clock.
    assert s.cancel((e / clock) / ((e + w) / clock) - e / (e + w)) == 0
    assert s.cancel((w / clock) / ((e + w) / clock) - w / (e + w)) == 0


def test_exact_preparation_and_maintained_target_are_separated_for_all_forms(algebra):
    s, edges, _, _, incidence, k = algebra
    donor = s.symbols("D0:5", real=True)
    mediator, common = s.symbols("H common", real=True)
    a, e = s.symbols("a e", positive=True)
    form = s.Matrix(donor + (0,) * 5 + (mediator,))
    donor_phase = tuple(2 * s.pi * j / 5 for j in range(5))
    initial_phase = s.Matrix(donor_phase + (0,) * 6)
    target_phase = s.Matrix(donor_phase + donor_phase + (0,))
    target_form = s.ones(11, 1) * common
    laplacian = incidence.T * incidence

    def quantities(x, theta):
        gaps = incidence * theta
        current = -incidence.T * gaps.applyfunc(s.sin)
        potential = sum(1 - s.cos(gap) for gap in gaps)
        storage = s.expand(x.dot(laplacian * x) / 2)
        auxiliary = s.Rational(4, 5) * storage + potential
        auxiliary -= a * (laplacian * x).dot(k * current) / (2 * e)
        return (
            storage,
            s.simplify(potential),
            current.applyfunc(s.simplify),
            s.simplify(auxiliary),
        )

    f, phase, current, initial = quantities(form, initial_phase)
    target_f, target_v, target_current, target = quantities(target_form, target_phase)
    v5 = (25 - 5 * s.sqrt(5)) / 4
    assert current == target_current == s.zeros(11, 1)
    assert target_f == 0 and phase == v5 and target_v == 2 * v5
    assert s.simplify(initial - (s.Rational(4, 5) * f + v5)) == 0
    assert s.simplify(target - 2 * v5) == 0
    assert s.simplify(f - sum((form[j] - form[i]) ** 2 for i, j in edges) / 2) == 0
    # sqrt(5)<9/4 proves a uniform gap without selecting a donor direction.
    assert s.Rational(9, 4) ** 2 > 5
    assert (25 - 5 * s.Rational(9, 4)) / 4 - s.Rational(16, 5) == s.Rational(19, 80) > 0


def test_receiver_only_target_and_lifted_invariants_do_not_imply_accessibility(algebra):
    s, _, _, _, incidence, k = algebra
    donor = s.symbols("transfer_D0:5", real=True)
    mediator = s.Symbol("transfer_H", real=True)
    initial_form = s.Matrix(donor + (0,) * 5 + (mediator,))
    initial_turns = s.Matrix(tuple(s.Rational(j, 5) for j in range(5)) + (0,) * 6)
    target_turns = s.Matrix((0,) * 5 + tuple(s.Rational(j, 5) for j in range(5)) + (0,))
    weights = k.inv().diagonal().T
    mass = sum(weights)
    assert mass == s.Rational(76, 3)
    assert weights.dot(initial_turns) == 4
    assert weights.dot(target_turns) == s.Rational(82, 15)
    common_form = weights.dot(initial_form) / mass
    # The terminal lifted representative is not selected by this admission.
    turns = s.Matrix(s.symbols("terminal_turn0:11", integer=True))
    common_turn = weights.dot(initial_turns - target_turns - turns) / mass
    assert s.simplify(common_turn.subs(dict.fromkeys(turns, 0))) == -s.Rational(11, 190)
    final_form = s.ones(11, 1) * common_form
    final_phase = 2 * s.pi * (target_turns + turns + s.ones(11, 1) * common_turn)
    assert s.simplify(weights.dot(final_form - initial_form)) == 0
    assert s.simplify(weights.dot(final_phase / (2 * s.pi) - initial_turns)) == 0

    laplacian = incidence.T * incidence
    q = laplacian * final_form
    gaps = incidence * final_phase
    current = (-incidence.T * gaps.applyfunc(s.sin)).applyfunc(s.simplify)
    assert q == current == s.zeros(11, 1)
    assert all(s.simplify(s.cos(gap)).is_positive for gap in gaps)
    potential = s.simplify(sum(1 - s.cos(gap) for gap in gaps))
    initial_gaps = 2 * s.pi * incidence * initial_turns
    initial_potential = s.simplify(sum(1 - s.cos(gap) for gap in initial_gaps))
    assert potential == initial_potential
    form_storage = initial_form.dot(laplacian * initial_form) / 2
    initial_current = (-incidence.T * initial_gaps.applyfunc(s.sin)).applyfunc(
        s.simplify
    )
    initial_auxiliary = s.Rational(4, 5) * form_storage + initial_potential
    initial_auxiliary -= (laplacian * initial_form).dot(k * initial_current) / (
        2 * s.pi
    )
    final_form_storage = final_form.dot(laplacian * final_form) / 2
    final_auxiliary = s.Rational(4, 5) * final_form_storage + potential
    final_auxiliary -= q.dot(k * current) / (2 * s.pi)
    assert (
        s.simplify(
            final_auxiliary - initial_auxiliary + s.Rational(4, 5) * form_storage
        )
        == 0
    )
    assert s.simplify(final_form_storage + potential - initial_potential) == 0
    # These endpoint deficits permit a transfer budget; they do not provide
    # a trajectory or a source lying in the target's recovery neighborhood.


def test_two_winding_changes_consume_disjoint_full_node_phase_action(algebra):
    s, edges, _, _, _, k = algebra
    donor_edges = tuple((i, j) for i, j in edges if i < 5 and j < 5)
    receiver_edges = tuple((i, j) for i, j in edges if 5 <= i < 10 and 5 <= j < 10)
    donor_gap_over_pi = tuple(s.Rational(2 * (j - i), 5) for i, j in donor_edges)
    # Every continuous donor edge lift must move at least 3*pi/5 to an
    # antipode; the initially flat receiver requires a full pi excursion.
    assert all(
        min(abs(gap - odd) for odd in (-3, -1, 1, 3)) == s.Rational(3, 5)
        for gap in donor_gap_over_pi
    )
    assert min(abs(odd) for odd in (-3, -1, 1, 3)) == 1
    donor_max = max(k[i, i] + k[j, j] for i, j in donor_edges)
    receiver_max = max(k[i, i] + k[j, j] for i, j in receiver_edges)
    assert donor_max == 1 and receiver_max == s.Rational(5, 4)

    q = s.Matrix(s.symbols("action_q0:11", real=True))
    regional_losses = []
    for indices, region_edges in (
        (range(5), donor_edges),
        (range(5, 10), receiver_edges),
    ):
        loss_without_e = sum(k[node, node] * q[node] ** 2 for node in indices)
        regional_losses.append(loss_without_e)
        for i, j in region_edges:
            velocity_without_b = k[j, j] * q[j] - k[i, i] * q[i]
            residual = (k[i, i] + k[j, j]) * loss_without_e - velocity_without_b**2
            squares = k[i, i] * k[j, j] * (q[i] + q[j]) ** 2
            squares += (k[i, i] + k[j, j]) * sum(
                k[node, node] * q[node] ** 2 for node in indices if node not in (i, j)
            )
            assert s.expand(residual - squares) == 0
    # Separate regional expenditures may be added without spending the
    # intermediary loss twice, regardless of which ring crosses first.
    assert s.expand(q.dot(k * q) - sum(regional_losses) - k[10, 10] * q[10] ** 2) == 0
    e, b, time = s.symbols("action_e action_b time", positive=True)
    action = e * ((3 * s.pi / 5) ** 2 / donor_max + s.pi**2 / receiver_max) / b**2
    assert s.simplify(action - s.Rational(29, 25) * e * s.pi**2 / b**2) == 0
    assert s.simplify(action.subs({e: s.Rational(1, 2), b: 1 / (2 * s.pi)})) == (
        s.Rational(58, 25) * s.pi**4
    )
    # Independently check the temporal Cauchy remainder for a nonconstant
    # affine velocity; constant velocity is the equality case. Each actual
    # crossing precedes product entry, so its time is strictly smaller.
    t, mean, slope = s.symbols("t mean slope", real=True)
    velocity = mean + slope * (t - time / 2)
    displacement = s.integrate(velocity, (t, 0, time))
    action_integral = s.integrate(velocity**2, (t, 0, time))
    assert s.expand(time * action_integral - displacement**2) == slope**2 * time**4 / 12


def test_transfer_product_face_cost_uses_one_twist_not_the_coexistence_allowance(
    algebra,
):
    s = algebra[0]
    v5 = 5 * (1 - s.cos(2 * s.pi / 5))
    # Jensen's convex edge cost on the acute chart minimizes the four free
    # increments at their mean: -pi/8 for winding0, 3*pi/8 for winding1.
    face_zero = 1 + 4 * (1 - s.cos(s.pi / 8))
    face_one = 1 + 4 * (1 - s.cos(3 * s.pi / 8))
    assert s.simplify(face_zero - 1).is_positive
    assert s.simplify(v5 - 3).is_positive
    assert s.simplify(s.Rational(7, 2) - face_one).is_positive
    assert s.simplify(face_one - v5).is_positive
    # If the donor hits its winding0 face, the receiver already costs V5;
    # if the receiver hits its winding1 face, the donor's minimum is zero.
    assert s.simplify(face_zero + v5 - face_one).is_positive
    form_budget = s.Symbol("initial_F", nonnegative=True)
    initial_storage = form_budget + v5
    transfer_allowance = initial_storage - face_one
    coexistence_allowance = initial_storage - (v5 + face_one)
    assert s.simplify(transfer_allowance - coexistence_allowance - v5) == 0
    assert s.simplify(transfer_allowance.subs(form_budget, face_one - v5)) == 0


def test_auxiliary_sublevel_connectivity_is_not_a_dynamical_transfer(algebra):
    s, _, _, _, incidence, k = algebra
    laplacian = incidence.T * incidence
    x, current = (
        s.Matrix(s.symbols(prefix + "0:11", real=True)) for prefix in ("x", "S")
    )
    potential, c, h = s.symbols("V c h", positive=True)
    auxiliary = c * x.dot(laplacian * x) / 2 + potential
    auxiliary -= h * (laplacian * x).dot(k * current)
    shifted = x - h * k * current / c
    phase_floor = potential - h**2 * current.dot(k * laplacian * k * current) / (2 * c)
    assert (
        s.expand(auxiliary - c * shifted.dot(laplacian * shifted) / 2 - phase_floor)
        == 0
    )

    # On each phase-only leg one ring is flat and the other has phases j*t.
    # The common phase shift enforces the original invariant without changing
    # any edge current or storage; it is not an added driving term.
    weights = k.inv().diagonal().T
    initial_turns = s.Matrix(tuple(s.Rational(j, 5) for j in range(5)) + (0,) * 6)
    initial_phase_sum = 2 * s.pi * weights.dot(initial_turns)
    t, common_form = s.symbols("phase_leg_t conserved_form", real=True)
    storage = 5 - 4 * s.cos(t) - s.cos(4 * t)
    derivative = s.diff(storage, t)
    assert s.trigsimp(derivative - 8 * s.sin(5 * t / 2) * s.cos(3 * t / 2)) == 0
    assert storage.subs(t, 0) == 0
    assert storage.subs(t, s.pi / 3) == s.Rational(7, 2)
    v5 = 5 * (1 - s.cos(2 * s.pi / 5))
    assert s.simplify(storage.subs(t, 2 * s.pi / 5) - v5) == 0
    # On [0,2*pi/5], sin(5*t/2)>=0 and cos(3*t/2) changes sign once,
    # at pi/3. Thus this is the exact maximum, without scanning a path.
    for offset in (0, 5):
        base = s.zeros(11, 1)
        for j in range(5):
            base[offset + j] = j * t
        origin = (initial_phase_sum - weights.dot(base)) / sum(weights)
        phase = base + s.ones(11, 1) * origin
        assert s.simplify(weights.dot(phase) - initial_phase_sum) == 0
        assert (incidence * phase - incidence * base).applyfunc(s.expand) == s.zeros(
            12, 1
        )
        assert (
            s.trigsimp(sum(1 - s.cos(gap) for gap in incidence * phase) - storage) == 0
        )
        q = laplacian * (s.ones(11, 1) * common_form)
        assert q == s.zeros(11, 1)
        assert phase.diff(t) != s.zeros(11, 1)
    # q=0 forces the actual phase row to vanish. The moving geometric legs
    # therefore cannot be interpreted as trajectories or accessibility proof.


def test_complete_critical_catalog_has_a_regular_strip_above_the_donor_minimum(algebra):
    from tnfr.physics.phase_cycle_geometry import (
        classify_c5_sine_critical_set,
        derive_phase_cycle_geometry,
    )

    s, edges, _, _, _, _ = algebra
    graph = nx.Graph()
    graph.add_nodes_from(range(11))
    graph.add_edges_from(edges)
    family = classify_c5_sine_critical_set(
        derive_phase_cycle_geometry(graph),
        cycles=(tuple(range(5)), tuple(range(5, 10))),
    )
    # These are evaluated from the complete exact branches, not an acute-only
    # target list. The prior full-field proof puts every critical point of W
    # at q=S=0, where W equals this phase potential.
    values = {
        s.simplify(sum(1 - s.cos(2 * s.pi * s.Rational(t)) for t in option))
        for option in family.cycle_edge_turn_options
    }
    v5 = s.simplify(5 * (1 - s.cos(2 * s.pi / 5)))
    critical_height = s.Rational(7, 2)
    assert 0 in values and v5 in values and critical_height in values
    assert all(
        value in (0, v5) or s.simplify(value - critical_height).is_nonnegative
        for value in values
    )
    assert s.simplify(v5 - 2).is_positive
    assert s.simplify(critical_height - v5).is_positive
    assert s.simplify(2 * v5 - critical_height).is_positive
    # Bridge saddles at 2 lie BELOW the newly born one-twist minima. They
    # cannot be silently omitted, but add no critical value in (V5,7/2).
    lower_ring_values = (s.Integer(0), v5)
    for donor in lower_ring_values:
        for receiver in lower_ring_values:
            for bridge_cost in (0, 2, 4):
                value = donor + receiver + bridge_cost
                assert (
                    s.simplify(v5 - value).is_nonnegative
                    or s.simplify(value - critical_height).is_nonnegative
                )
    # Component persistence across this strip is the separate compact
    # deformation argument in the proof, not a sampled-path assertion.


def test_form_convexity_preserves_the_strict_sublevel_at_the_equality_boundary(algebra):
    s, _, _, _, incidence, k = algebra
    laplacian = incidence.T * incidence
    x = s.Matrix(s.symbols("retention_x0:11", real=True))
    current = s.Matrix(s.symbols("retention_S0:11", real=True))
    t, potential, c, h = s.symbols("t potential c h", real=True)
    weights = k.inv().diagonal().T
    common = weights.dot(x) / sum(weights)
    uniform = s.ones(11, 1) * common

    def functional(form):
        return (
            c * form.dot(laplacian * form) / 2
            + potential
            - h * (laplacian * form).dot(k * current)
        )

    form_storage = x.dot(laplacian * x) / 2
    segment = (1 - t) * x + t * uniform
    assert s.simplify(weights.dot(segment - x)) == 0
    assert s.simplify(functional(uniform) - potential) == 0
    assert (
        s.expand(
            functional(segment)
            - (1 - t) * functional(x)
            - t * potential
            + c * t * (1 - t) * form_storage
        )
        == 0
    )
    # c>0 and 0<=t<=1 put the entire segment below the maximum of its
    # endpoints. At the exact initial phase critical point S=0, shrinking
    # form deviations also connects every preparation to its donor minimum.
    initial_value = functional(segment).subs(dict.fromkeys(current, 0))
    assert s.expand(initial_value - potential - c * (1 - t) ** 2 * form_storage) == 0


@pytest.mark.parametrize(
    "donor,mediator",
    [((Q(0),) * 5, Q(7, 30)), ((Q(1, 8),) + (Q(0),) * 4, Q(0))],
)
def test_nonsilent_full_support_preparations_have_exact_donor_retention(
    algebra, donor, mediator
):
    from tnfr.dynamics.relational import RelationalExchangeModel
    from tnfr.physics.relational_sine_formation import assess_sine_mediated_formation

    s, _, _, _, incidence, k = algebra
    laplacian = incidence.T * incidence
    form = s.Matrix(tuple(map(s.Rational, donor)) + (0,) * 5 + (s.Rational(mediator),))
    phase = s.Matrix(tuple(2 * s.pi * j / 5 for j in range(5)) + (0,) * 6)
    q = laplacian * form
    current = (-incidence.T * (incidence * phase).applyfunc(s.sin)).applyfunc(
        s.simplify
    )
    storage = form.dot(q) / 2
    assert current == s.zeros(11, 1)
    assert q != s.zeros(11, 1)
    assert q.dot(k * q) > 0
    # Neither preparation is a stationary source or the silent donor
    # sign-reflection subspace. Its actual phase row is already nonzero.
    assert k * q / (2 * s.pi) != s.zeros(11, 1)
    assert mediator != 0 or donor[0] != 0

    v5 = s.simplify(sum(1 - s.cos(gap) for gap in incidence * phase))
    critical_height = s.Rational(7, 2)
    critical_form_storage = (critical_height - v5) * s.Rational(5, 4)
    acute_face = 5 - 4 * s.cos(3 * s.pi / 8)
    assert s.simplify(storage + v5 - acute_face).is_positive
    assert s.simplify(critical_form_storage - storage).is_positive
    if mediator:
        # This control also clears the stronger bare energy barrier. The
        # mixed functional, not an insufficient total budget, retains it.
        assert s.simplify(storage + v5 - critical_height).is_positive
    exact_margin = 3125 - (16 * storage + 55) ** 2
    assert exact_margin > 0

    source = assess_sine_mediated_formation(
        amplitude=mediator,
        capacity_contrast=Q(1, 2),
        profile="explicit",
        donor_epi=donor,
        model=RelationalExchangeModel(1, phase_domain="regular"),
    )
    retained = source.donor_well_retention()
    assert retained.initial_form_storage == Q(storage)
    assert retained.exact_retention_polynomial_margin == Q(exact_margin)
    assert retained.status == "certified"
    assert retained.relative_donor_pattern_convergence_certified
    assert retained.receiver_only_targets_excluded
    assert retained.retained_phase_turns == tuple(
        Q(value / (2 * s.pi)) for value in phase
    )

    # Fixing all five receiver forms to zero leaves a positive definite
    # six-coordinate storage quadratic. Strict inequalities above therefore
    # hold on a full-dimensional neighborhood, not only the chosen line.
    coordinates = (0, 1, 2, 3, 4, 10)
    restriction = laplacian.extract(coordinates, coordinates)
    _, diagonal = restriction.LDLdecomposition()
    assert all(value > 0 for value in diagonal.diagonal())


def test_short_time_full_matrix_envelopes_and_exact_functional_loss(algebra):
    s, edges, _, _, incidence, k = algebra
    root_k = s.diag(*(s.sqrt(value) for value in k.diagonal()))
    b = root_k * incidence.T * incidence * root_k
    cosine = s.symbols("capture_cos0:12", real=True)
    current_jacobian = root_k * incidence.T * s.diag(*cosine) * incidence * root_k
    v = s.Matrix(s.symbols("capture_v0:11", real=True))
    for sign in (-1, 1):
        remainder = sum(
            (1 + sign * cosine[row]) * (root_k[j, j] * v[j] - root_k[i, i] * v[i]) ** 2
            for row, (i, j) in enumerate(edges)
        )
        assert s.expand(v.dot((b + sign * current_jacobian) * v) - remainder) == 0
    # Together with the full-support B<=3I identity above, |cos|<=1
    # bounds BOTH signs of the live Hessian, even outside an acute chart.
    eye = s.eye(11)
    cross_operator = -eye / 5 + b / 2
    assert cross_operator + eye / 5 == b / 2
    assert s.Rational(13, 10) * eye - cross_operator == (3 * eye - b) / 2
    a = s.Symbol("capture_a", positive=True)
    diagonal = s.Rational(2, 5) * eye - a**2 * b
    assert (
        diagonal
        - s.Rational(19, 60) * eye
        - (3 * eye - b) / 36
        - (s.Rational(1, 36) - a**2) * b
    ).applyfunc(s.expand) == s.zeros(11)

    # At a=1/(2*pi)<1/6, Y<=C*Y0 and Z<=C*t*Y0/2 follow
    # from the coupled integral norm inequalities, with C=11/10.
    horizon, c = s.Rational(1, 4), s.Rational(11, 10)
    argument_squared = (horizon / 2) ** 2
    geometric_cosh_upper = 1 / (1 - argument_squared / 2)
    assert geometric_cosh_upper < c
    t = s.Symbol("capture_t", nonnegative=True)
    semigroup_lower = 1 - 3 * t / 2 - c * t**2 / 8
    lower = 1 - 2 * t
    difference = s.factor(semigroup_lower - lower)
    assert s.expand(difference - t * (40 - 11 * t) / 80) == 0
    assert 40 - 11 * horizon > 0 and lower.subs(t, horizon) > 0
    # The damped-semigroup lower estimate avoids differentiating a norm at
    # zero; the positive lower bound justifies squaring throughout the window.
    rate_lower = s.Rational(19, 60) * lower**2 - s.Rational(13, 60) * c**2 * t / 2
    integrated = s.integrate(rate_lower, (t, 0, horizon))
    assert integrated == s.Rational(48481, 1152000)
    assert integrated > s.Rational(1, 25)


def test_phase_homotopy_and_form_gradient_share_full_support_bounds(algebra):
    s, _, _, _, incidence, k = algebra
    laplacian = incidence.T * incidence
    root_k = s.diag(*(s.sqrt(value) for value in k.diagonal()))
    b = root_k * laplacian * root_k
    x = s.Matrix(s.symbols("capture_form0:11", real=True))
    q = laplacian * x
    form_storage, norm_squared = x.dot(q) / 2, q.dot(k * q)
    # This is the dual full-support bound to B<=3I. It does not assume
    # uniform degrees or discard the unequal receiver capacities.
    assert 3 * k.inv() - laplacian == root_k.inv() * (3 * s.eye(11) - b) * root_k.inv()
    assert (
        s.expand(
            6 * form_storage
            - norm_squared
            - x.dot((3 * laplacian - laplacian * k * laplacian) * x)
        )
        == 0
    )
    # PSD of 3L-LKL also has this independent exact matrix check. Pinning
    # one node removes its common-form null direction without changing signs.
    complementary = 3 * laplacian - laplacian * k * laplacian
    assert complementary * s.ones(11, 1) == s.zeros(11, 1)
    _, diagonal = complementary[:10, :10].LDLdecomposition()
    assert all(value > 0 for value in diagonal.diagonal())

    integrated_y = s.Matrix(s.symbols("integrated_y0:11", real=True))
    a = s.Symbol("phase_a", positive=True)
    displacement = a * root_k * integrated_y
    assert (
        s.expand(
            displacement.dot(laplacian * displacement)
            - a**2 * integrated_y.dot(b * integrated_y)
        )
        == 0
    )
    phase = s.Matrix(tuple(2 * s.pi * j / 5 for j in range(5)) + (0,) * 6)
    delta = s.Matrix(s.symbols("phase_delta0:11", real=True))
    fraction = s.Symbol("phase_fraction", real=True)
    gaps, gap_delta = incidence * phase, incidence * delta
    potential = sum(
        1 - s.cos(gap + fraction * change) for gap, change in zip(gaps, gap_delta)
    )
    assert s.simplify(s.diff(potential, fraction).subs(fraction, 0)) == 0
    # Check the global Hessian inequality with arbitrary edge phases as well
    # as the exact source-gradient cancellation above.
    arbitrary_gaps = s.symbols("path_gap0:12", real=True)
    potential = sum(
        1 - s.cos(gap + fraction * change)
        for gap, change in zip(arbitrary_gaps, gap_delta)
    )
    hessian_remainder = sum(
        change**2 * (1 - s.cos(gap + fraction * change))
        for gap, change in zip(arbitrary_gaps, gap_delta)
    )
    assert (
        s.expand(
            delta.dot(laplacian * delta)
            - s.diff(potential, fraction, 2)
            - hessian_remainder
        )
        == 0
    )
    # Taylor's integral remainder therefore bounds the entire straight phase
    # path, not merely its endpoint. At T=1/4 its coefficient is exact.
    coefficient = (
        s.Rational(3, 2)
        * s.Rational(1, 36)
        * s.Rational(11, 10) ** 2
        * s.Rational(1, 4) ** 2
    )
    assert coefficient == s.Rational(121, 38400)
    # For every admitted source N<=6F implies A>=14F/25 and B<=121F/6400.
    # Thus endpoint entry A<D automatically puts the phase path below D.
    assert s.Rational(121, 6400) / s.Rational(14, 25) == s.Rational(121, 3584) < 1


def test_fixed_above_barrier_source_enters_donor_well_by_early_dissipation(algebra):
    from tnfr.dynamics.relational import RelationalExchangeModel
    from tnfr.physics.relational_sine_formation import assess_sine_mediated_formation

    s, _, _, _, incidence, k = algebra
    form = s.zeros(11, 1)
    form[10] = s.Rational(1, 4)
    laplacian = incidence.T * incidence
    q = laplacian * form
    f, n = form.dot(q) / 2, q.dot(k * q)
    assert f == s.Rational(1, 16) and n == s.Rational(1, 6)
    v5 = (25 - 5 * s.sqrt(5)) / 4
    critical_gap = s.Rational(7, 2) - v5
    assert s.simplify(s.Rational(4, 5) * f - critical_gap).is_positive
    allowance = s.Rational(4, 5) * f - n / 25
    phase_allowance = s.Rational(121, 38400) * n
    assert allowance == s.Rational(13, 300)
    assert s.simplify(critical_gap - allowance).is_positive
    assert s.simplify(critical_gap - phase_allowance).is_positive
    assert 125 - (11 + 4 * allowance) ** 2 > 0
    assert 125 - (11 + 4 * phase_allowance) ** 2 > 0

    source = assess_sine_mediated_formation(
        amplitude=Q(1, 4),
        capacity_contrast=Q(1, 2),
        model=RelationalExchangeModel(1, phase_domain="regular"),
    )
    assert (
        not source.donor_well_retention().relative_donor_pattern_convergence_certified
    )
    captured = source.donor_dissipative_capture()
    assert captured.initial_form_storage == Q(f)
    assert captured.initial_dissipative_norm_squared == Q(n)
    assert captured.horizon == Q(1, 4)
    assert captured.functional_drop_lower_bound == Q(n / 25)
    assert captured.endpoint_storage_allowance == Q(allowance)
    assert captured.phase_path_storage_allowance == Q(phase_allowance)
    assert captured.donor_component_entry_certified
    assert captured.relative_donor_pattern_convergence_certified
    assert captured.receiver_only_targets_excluded

    # The same form storage need not give the same short-time certificate:
    # the retained full gradient geometry supplies a separate loss budget.
    alternate_form = s.Matrix(
        (s.Rational(3, 10),) * 5 + (0,) * 5 + (s.Rational(7, 20),)
    )
    alternate_q = laplacian * alternate_form
    alternate_f = alternate_form.dot(alternate_q) / 2
    alternate_n = alternate_q.dot(k * alternate_q)
    assert alternate_f == f
    assert alternate_n == s.Rational(73, 600)
    alternate_allowance = s.Rational(4, 5) * alternate_f - alternate_n / 25
    alternate_phase_allowance = s.Rational(121, 38400) * alternate_n
    assert s.simplify(alternate_allowance - critical_gap).is_positive
    assert s.simplify(critical_gap - alternate_phase_allowance).is_positive
    alternate = assess_sine_mediated_formation(
        amplitude=Q(7, 20),
        capacity_contrast=Q(1, 2),
        profile="explicit",
        donor_epi=(Q(3, 10),) * 5,
        model=source.model,
    ).donor_dissipative_capture()
    assert alternate.initial_form_storage == Q(alternate_f)
    assert alternate.initial_dissipative_norm_squared == Q(alternate_n)
    assert alternate.endpoint_exact_polynomial_margin < 0
    assert alternate.phase_path_exact_polynomial_margin > 0
    assert alternate.status == "not_certified"
    assert not alternate.relative_donor_pattern_convergence_certified
    # This is a limitation of this bound, not a different endpoint prediction.


def test_receiver_internal_storage_uses_full_nodal_loss_and_signed_port_work(algebra):
    s, edges, _, _, incidence, k = algebra
    x = s.Matrix(s.symbols("receiver_x0:11", real=True))
    edge_sine = s.Matrix(s.symbols("receiver_sin0:12", real=True))
    e, a, beta = s.symbols("receiver_e receiver_a beta", positive=True)
    q = incidence.T * incidence * x
    current = -incidence.T * edge_sine
    xdot = k * (-e * q + a * current)
    phasedot = a * k * q / beta
    receiver = tuple(range(5, 10))
    rows = tuple(
        row for row, (i, j) in enumerate(edges) if i in receiver and j in receiver
    )
    receiver_incidence = incidence.extract(rows, range(11))
    internal_q = receiver_incidence.T * receiver_incidence * x
    internal_current = -receiver_incidence.T * edge_sine.extract(rows, (0,))
    storage_rate = internal_q.dot(xdot) - beta * internal_current.dot(phasedot)
    full_loss = e * sum(k[i, i] * q[i] ** 2 for i in receiver)
    bridge = edges.index((5, 10))
    port_work = (x[10] - x[5]) * xdot[5] + beta * edge_sine[bridge] * phasedot[5]
    assert s.expand(storage_rate - port_work + full_loss) == 0
    # This is the negative of the mediator ledger's work at port5, not
    # the work summed over both donor and receiver ports.
    mediator_port_work = (x[5] - x[10]) * xdot[5] - beta * edge_sine[bridge] * phasedot[
        5
    ]
    assert s.expand(port_work + mediator_port_work) == 0

    intrinsic_loss = e * internal_q.dot(k * internal_q)
    contrast = x[5] - x[10]
    assert (
        s.expand(
            full_loss
            - intrinsic_loss
            - e * k[5, 5] * (2 * internal_q[5] * contrast + contrast**2)
        )
        == 0
    )
    assert s.expand(full_loss - intrinsic_loss) != 0
    # All other receiver rows retain their actual capacities and full phase
    # velocities. Zero cumulative loss would force every phase stationary.
    phase_action_loss = (
        e * beta**2 * sum(phasedot[i] ** 2 / k[i, i] for i in receiver) / a**2
    )
    assert s.expand(full_loss - phase_action_loss) == 0

    # The phase-action bound uses the INTERNAL cycle Laplacian with the
    # ACTUAL full-support mobilities, including port degree3 and unequal nu.
    root_k = s.diag(*(s.sqrt(value) for value in k.diagonal()))
    receiver_laplacian = receiver_incidence.T * receiver_incidence
    receiver_b = root_k * receiver_laplacian * root_k
    v = s.Matrix(s.symbols("receiver_action_v0:11", real=True))
    spectral_squares = sum(
        (root_k[i, i] * v[i] + root_k[j, j] * v[j]) ** 2
        for row, (i, j) in enumerate(edges)
        if row in rows
    )
    coefficients = tuple(3 - 2 * receiver_laplacian[i, i] * k[i, i] for i in receiver)
    assert all(value >= 0 for value in coefficients)
    spectral_squares += sum(
        value * v[i] ** 2 for i, value in zip(receiver, coefficients)
    )
    assert (
        s.expand(
            3 * sum(v[i] ** 2 for i in receiver)
            - v.dot(receiver_b * v)
            - spectral_squares
        )
        == 0
    )
    # V_R<=eta.T*L_C*eta/2 and time Cauchy--Schwarz therefore yield
    # tau*D_R>=7e/(3b^2) at the first V_R=7/2 crossing. This uses unit
    # phase cost V_R; beta multiplies the storage barrier, not V_R again.
    action_product_lower = 7 * e * beta**2 / (3 * a**2)
    assert (
        s.simplify(
            action_product_lower.subs({e: s.Rational(1, 2), a: 1 / (2 * s.pi), beta: 1})
            - 14 * s.pi**2 / 3
        )
        == 0
    )

    donor = s.symbols("receiver_initial_D0:5", real=True)
    hidden = s.Symbol("receiver_initial_H", real=True)
    initial_form = s.Matrix(donor + (0,) * 5 + (hidden,))
    initial_phase = s.Matrix(tuple(2 * s.pi * j / 5 for j in range(5)) + (0,) * 6)
    substitutions = dict(zip(x, initial_form))
    substitutions.update(
        dict(zip(edge_sine, (incidence * initial_phase).applyfunc(s.sin)))
    )
    assert s.simplify(storage_rate.subs(substitutions)) == 0
    assert s.simplify(port_work.subs(substitutions) - e * hidden**2 / 3) == 0
    assert s.simplify(full_loss.subs(substitutions) - e * hidden**2 / 3) == 0
    # Positive instantaneous supply can initially pay exactly for nodal
    # dissipation. It does not certify accumulated receiver phase energy.


def test_receiver_minimax_supply_and_first_barrier_order_are_conditional(algebra):
    s = algebra[0]
    v5 = (25 - 5 * s.sqrt(5)) / 4
    barrier = s.Rational(7, 2)
    assert s.simplify(barrier - v5).is_positive
    # The prior complete critical-set/component proof gives the same
    # minimax for either direction and either twist handedness. Its explicit
    # comparison path attains the barrier; it is not a dynamical trajectory.
    t = s.Symbol("receiver_phase_path", real=True)
    path_cost = 5 - 4 * s.cos(t) - s.cos(4 * t)
    assert path_cost.subs(t, 0) == 0
    assert s.simplify(path_cost.subs(t, 2 * s.pi / 5) - v5) == 0
    assert path_cost.subs(t, s.pi / 3) == barrier
    assert s.simplify(path_cost.subs(t, -t) - path_cost) == 0

    receiver_form = s.Symbol("receiver_form_at_crossing", nonnegative=True)
    receiver_loss = s.Symbol("receiver_loss_until_crossing", positive=True)
    signed_supply = barrier + receiver_form + receiver_loss
    assert (signed_supply - barrier).is_positive
    # Strict receiver loss follows from the phase-action identity above:
    # reaching the nonflat barrier requires a nonconstant receiver phase.
    initial_f = s.Symbol("initial_form_budget", nonnegative=True)
    total_loss = s.Symbol("total_loss_until_crossing", positive=True)
    other_storage = s.Symbol("non_donor_non_receiver_phase_storage", nonnegative=True)
    donor_phase_at_crossing = initial_f + v5 - barrier - total_loss - other_storage
    assert (v5 - donor_phase_at_crossing.subs(initial_f, barrier)).is_positive
    # For all F<=7/2 this upper bound is strictly below V5, incompatible
    # with remaining in the donor's initial sublevel component. Therefore
    # the donor first crosses its own barrier before receiver acquisition.
    simultaneous_budget_threshold = 2 * barrier - v5
    assert s.simplify(simultaneous_budget_threshold - barrier).is_positive
    simultaneous_storage = 2 * barrier + other_storage
    necessary_initial_form = simultaneous_storage + total_loss - v5
    assert s.simplify(
        necessary_initial_form - simultaneous_budget_threshold
    ).is_positive
    # Only the necessary strict condition F>7-V5 follows for simultaneous
    # first crossings; there is no event time or basin-selection assertion.
