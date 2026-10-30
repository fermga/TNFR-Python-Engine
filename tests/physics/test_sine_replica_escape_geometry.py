"""Exact full-state controls for one nonacute doubled-cycle critical geometry.

These are stationary algebra and instantaneous-field controls, not trajectories
or a new runtime report. The nonlinear elimination is stationary, not an
autonomous model. Its remainder radius is existential in the reviewed proof.
"""

from fractions import Fraction as Q
from itertools import product
from math import factorial

import pytest
import sympy as sp

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_matrix_product,
)
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.mathematics.krylov import exact_rank
from tnfr.physics._sine_flow import _held_sine_neighbor_field
from tnfr.physics.phase_cycle_geometry import _derive, reconstruct_circular_phase_state
from tnfr.physics.relational_sine_resonance import _conservative_full_tangent


def _edges():
    return tuple(
        sorted(
            {
                tuple(sorted((2 * i + p, 2 * ((i + 1) % 5) + q)))
                for i in range(5)
                for p, q in product((0, 1), repeat=2)
            }
        )
    )


def _neighbors():
    edges = _edges()
    return tuple(
        tuple(j for j in range(10) if tuple(sorted((i, j))) in edges) for i in range(10)
    )


def _fine_rows(forms, phases):
    """Independent edge sums of both original-time nodal rows."""
    neighbors = _neighbors()
    return sp.Matrix(
        [
            sum(sp.sin(phases[j] - phases[i]) for j in row) / (4 * sp.pi)
            for i, row in enumerate(neighbors)
        ]
        + [
            sum(forms[i] - forms[j] for j in row) / (4 * sp.pi)
            for i, row in enumerate(neighbors)
        ]
    )


def _fine_storage(forms, phases):
    return sum(
        sp.Rational(1, 2) * (forms[i] - forms[j]) ** 2
        + 1
        - sp.cos(phases[j] - phases[i])
        for i, j in _edges()
    )


def _rational(matrix):
    return tuple(
        tuple(Q(int(value.p), int(value.q)) for value in row)
        for row in sp.Matrix(matrix).tolist()
    )


def _embedding():
    """Fine state from ordered (X, Theta, u, delta), all twenty coordinates."""
    matrix = sp.zeros(20)
    for i in range(5):
        for member, sign in ((0, 1), (1, -1)):
            node = 2 * i + member
            matrix[node, i] = matrix[10 + node, 5 + i] = 1
            matrix[node, 10 + i] = matrix[10 + node, 15 + i] = sign
    return matrix


def _laplacian(weights):
    matrix = sp.zeros(5)
    for i, weight in enumerate(weights):
        j = (i + 1) % 5
        matrix[i, i] += weight
        matrix[j, j] += weight
        matrix[i, j] -= weight
        matrix[j, i] -= weight
    return matrix


def _lift_phases(gaps, splits):
    means = [sp.S(0)]
    for gap in gaps[:4]:
        means.append(means[-1] + gap)
    return tuple(means[i] + sign * splits[i] for i in range(5) for sign in (1, -1))


def _clean(value):
    return sp.simplify(sp.expand_trig(value))


@pytest.fixture(scope="module")
def exact_candidate():
    forms, phases = sp.symbols("x0:10", real=True), sp.symbols("t0:10", real=True)
    state = forms + phases
    target_phases = tuple(sp.Rational(i // 2, 3) * sp.pi for i in range(10))
    target = dict(zip(state, (sp.S(0),) * 10 + target_phases))
    energy = _fine_storage(forms, phases)
    rows = _fine_rows(forms, phases)
    hessian = sp.hessian(energy, state).subs(target)
    jacobian = rows.jacobian(state).subs(target)
    turns = tuple(Q(i // 2, 6) for i in range(10))
    edge_turns = tuple(
        (turns[j] - turns[i] + Q(1, 2)) % 1 - Q(1, 2) for i, j in _edges()
    )
    cosines = tuple(
        Q(1, 2) if abs(turn) == Q(1, 6) else Q(-1, 2) for turn in edge_turns
    )
    shared = _conservative_full_tangent(_edges(), (Q(1, 4),) * 10, cosines)
    return dict(
        forms=forms,
        phases=phases,
        state=state,
        target=target,
        target_phases=target_phases,
        energy=energy,
        rows=rows,
        hessian=hessian,
        jacobian=jacobian,
        shared=shared,
        edge_turns=edge_turns,
        turns=turns,
    )


def test_exact_candidate_cancels_every_fine_row_and_has_storage_fourteen(
    exact_candidate,
):
    data = exact_candidate
    assert len(_edges()) == 20 and all(len(row) == 4 for row in _neighbors())
    geometry = _derive(tuple(range(10)), _edges())
    target = reconstruct_circular_phase_state(geometry, edge_turns=data["edge_turns"])
    assert target.sine_balance_status == "proved_by_period_reflection_cancellation"
    assert not any(target.symbolic_sine_coefficients)
    assert data["rows"].subs(data["target"]) == sp.zeros(20, 1)
    assert data["energy"].subs(data["target"]) == 14
    assert sp.Matrix(
        [sp.diff(data["energy"], variable) for variable in data["state"]]
    ).subs(data["target"]) == sp.zeros(20, 1)
    assert set(map(abs, data["edge_turns"])) == {Q(1, 6), Q(1, 3)}
    # Exact circular turns represent this hypothetical critical target;
    # no captured binary64 radian state is promoted to that target.
    assert "symbolic_exact_angles_are_not_binary64_radian_equilibria" in target.scope


def test_storage_hessian_and_both_row_tangent_keep_the_original_clock(exact_candidate):
    data = exact_candidate
    laplacian, phase_hessian, scaled_tangent, metric = data["shared"]
    assert data["hessian"] == sp.Matrix(metric)
    assert data["hessian"][:10, :10] == sp.Matrix(laplacian)
    assert data["hessian"][10:, 10:] == sp.Matrix(phase_hessian)
    assert data["jacobian"] == sp.Matrix(scaled_tangent) / sp.pi
    weighted = data["hessian"] * data["jacobian"]
    assert weighted + weighted.T == sp.zeros(20)
    # Independently exercise the existing complete numerical owner on a
    # rational instantaneous control. No approximation of the target is used.
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    neighbor_rows = tuple(tuple((j, 1, Q(1)) for j in row) for row in _neighbors())
    flow = _held_sine_neighbor_field(model, (4,) * 10, (Q(1),) * 10, neighbor_rows)
    values = flow((Q(1), Q(-1)) + (Q(0),) * 18)
    assert all(
        value.contains(0) and value.width < Q(1, 2**100) for value in values[:10]
    )
    for index, value in enumerate(values[10:]):
        expected = Q(1) if index == 0 else Q(-1) if index == 1 else Q(0)
        assert (value * pi_interval()).contains(expected)


def test_full_coordinate_congruence_and_similarity_retain_all_twenty_directions(
    exact_candidate,
):
    data = exact_candidate
    transform = _embedding()
    exact_transform = _rational(transform)
    inverse = exact_matrix_inverse(exact_transform)
    assert exact_rank(exact_transform) == 20
    assert sp.Matrix(inverse) * transform == sp.eye(20)
    laplacian = _laplacian((sp.S(1),) * 5)
    signed = _laplacian((sp.Rational(1, 2),) * 4 + (-sp.Rational(1, 2),))
    internal = sp.diag(0, 1, 1, 1, 0)
    expected_hessian = sp.diag(4 * laplacian, 4 * signed, 8 * sp.eye(5), 4 * internal)
    congruence = exact_matrix_product(
        _rational(transform.T), exact_matrix_product(data["shared"][3], exact_transform)
    )
    assert sp.Matrix(congruence) == transform.T * data["hessian"] * transform
    assert sp.Matrix(congruence) == expected_hessian
    similarity = sp.Matrix(
        exact_matrix_product(
            inverse, exact_matrix_product(data["shared"][2], exact_transform)
        )
    )
    expected_tangent = sp.zeros(20)
    expected_tangent[:5, 5:10] = -signed / 2
    expected_tangent[5:10, :5] = laplacian / 2
    expected_tangent[10:15, 15:20] = -internal / 2
    expected_tangent[15:20, 10:15] = sp.eye(5)
    assert similarity == expected_tangent
    bracket = sp.zeros(20)
    for first, second in ((0, 5), (10, 15)):
        bracket[first : first + 5, second : second + 5] = -sp.eye(5) / 8
        bracket[second : second + 5, first : first + 5] = sp.eye(5) / 8
    assert bracket * expected_hessian == similarity
    assert transform * similarity / sp.pi == data["jacobian"] * transform


def test_exact_inertia_spectrum_and_zero_jordan_chains_distinguish_internal_modes(
    exact_candidate,
):
    data = exact_candidate
    transform = _embedding()
    metric = transform.T * data["hessian"] * transform
    relative = sp.zeros(5, 4)
    for i in range(4):
        relative[i, i], relative[4, i] = 1, -1
    form_reduced = relative.T * metric[:5, :5] * relative
    phase_reduced = relative.T * metric[5:10, 5:10] * relative
    lf, df = form_reduced.LDLdecomposition(hermitian=False)
    lp, dp = phase_reduced.LDLdecomposition(hermitian=False)
    assert lf * df * lf.T == form_reduced and lp * dp * lp.T == phase_reduced
    assert all(value > 0 for value in df.diagonal())
    assert sum(bool(value > 0) for value in dp.diagonal()) == 3
    assert sum(bool(value < 0) for value in dp.diagonal()) == 1
    assert metric[10:15, 10:15] == 8 * sp.eye(5)
    assert tuple(metric[15:20, 15:20].diagonal()) == (0, 4, 4, 4, 0)
    assert exact_rank(_rational(data["hessian"])) == 16
    # Four positive collective forms, five internal forms, three positive
    # collective phases and three internal phases: fifteen positive directions.
    assert 4 + 5 + 3 + 3 == 15
    tangent = sp.Matrix(data["shared"][2])
    assert exact_rank(_rational(tangent)) == 16
    assert exact_rank(_rational(tangent * tangent)) == 14
    assert exact_rank(_rational(tangent**3)) == 14
    paired = transform.inv() * tangent * transform
    for index in (0, 4):
        form, phase = sp.eye(20)[:, 10 + index], sp.eye(20)[:, 15 + index]
        assert paired * form == phase and paired * phase == sp.zeros(20, 1)
    for block in (0, 5):
        origin = sp.zeros(20, 1)
        origin[block : block + 5, 0] = sp.ones(5, 1)
        assert paired * origin == sp.zeros(20, 1)
    variable = sp.Symbol("lambda")
    characteristic = tangent.charpoly(variable).as_expr()
    expected = (
        variable**6
        * (variable**2 + sp.Rational(1, 2)) ** 3
        * (variable**4 + sp.Rational(15, 8) * variable**2 + sp.Rational(25, 64))
        * (variable**4 + sp.Rational(3, 8) * variable**2 - sp.Rational(15, 64))
    )
    assert sp.expand(characteristic - expected) == 0
    squared_rates = (
        (-sp.Rational(1, 2),) * 3
        + tuple((-15 + sign * 5 * sp.sqrt(5)) / 16 for sign in (-1, 1))
        + tuple((-3 + sign * sp.sqrt(69)) / 16 for sign in (-1, 1))
    )
    assert sum(value.is_negative for value in squared_rates) == 6
    assert sum(value.is_positive for value in squared_rates) == 1


def test_frozen_mean_null_slice_keeps_quadratic_collective_and_cubic_internal_forces():
    a, b = sp.symbols("a b", real=True)
    root = sp.sqrt(3) / 2
    gaps = (sp.pi / 3,) * 4 + (2 * sp.pi / 3,)
    phases = _lift_phases(gaps, (a, 0, 0, 0, b))
    storage = _clean(_fine_storage((0,) * 10, phases))
    assert _clean(storage - 14 - 2 * (1 - sp.cos(a)) * (1 - sp.cos(b))) == 0
    force = _embedding().inv() * _fine_rows((0,) * 10, phases)
    currents = (sp.cos(a), 1, 1, sp.cos(b), sp.cos(a) * sp.cos(b))
    for i in range(5):
        expected = root * (currents[i] - currents[(i - 1) % 5]) / (2 * sp.pi)
        assert _clean(force[i] - expected) == 0
    assert _clean(force[10] + sp.sin(a) * (1 - sp.cos(b)) / (4 * sp.pi)) == 0
    assert _clean(force[14] + sp.sin(b) * (1 - sp.cos(a)) / (4 * sp.pi)) == 0
    assert all(_clean(value) == 0 for value in tuple(force[5:10]) + tuple(force[15:20]))
    assert _clean(storage.subs(b, 0)) == _clean(storage.subs(a, 0)) == 14
    axis_force = _clean(force[1].subs(b, 0))
    assert _clean(axis_force - sp.sqrt(3) * (1 - sp.cos(a)) / (4 * sp.pi)) == 0
    assert sp.diff(axis_force, a, 2).subs(a, 0) == sp.sqrt(3) / (4 * sp.pi)
    assert _clean(sp.diff(force[10], a, 1, b, 2).subs({a: 0, b: 0})) == -1 / (4 * sp.pi)


def test_exact_adjusted_mean_branches_cancel_both_rows_with_a_local_domain():
    a, h = sp.symbols("a h", real=True)
    root = sp.sqrt(3) / 2
    branches = (
        ((h, sp.pi / 3, sp.pi / 3, sp.pi / 3, sp.pi - h), (a, 0, 0, 0, 0)),
        ((sp.pi / 3, sp.pi / 3, sp.pi / 3, h, sp.pi - h), (0, 0, 0, 0, a)),
    )
    for gaps, splits in branches:
        phases = _lift_phases(gaps, splits)
        rows = _fine_rows((0,) * 10, phases)
        for value in rows:
            reduced = sp.expand_trig(value).subs(sp.sin(h), root / sp.cos(a))
            assert sp.simplify(reduced) == 0
        assert _clean(_fine_storage((0,) * 10, phases)) == 14
        assert sum(gaps) == 2 * sp.pi
    assert sp.cos(sp.pi / 6) == root
    assert sp.asin(root) == sp.pi / 3
    assert sp.asin(1) == sp.pi / 2
    # On |a|<pi/6, cos(a)>sqrt(3)/2>0; h is real and acute.
    # The chart bound is a branch domain, not a uniform remainder radius.
    r, angle = sp.symbols("r angle", real=True)
    assert (
        sp.expand_trig((r * sp.cos(angle)) ** 2 + (r * sp.sin(angle)) ** 2).simplify()
        == r**2
    )
    # Vanishing neighbor resultant requires equal lengths 1 and cos(b).
    # In the local positive-cosine chart this forces the other split to zero.
    assert sp.simplify(sp.factor(1 - r**2) - (1 - r) * (1 + r)) == 0


def test_stationary_elimination_has_mixed_quartic_two_and_only_an_existential_remainder():
    current, r, s = sp.symbols("J R S", real=True)
    root = sp.sqrt(3) / 2
    closure = (
        sp.asin(current / r)
        + 2 * sp.asin(current)
        + sp.asin(current / s)
        - sp.asin(current / (r * s))
        - sp.pi
    )
    base = {current: root, r: 1, s: 1}
    assert sp.simplify(closure.subs(base)) == 0
    assert sp.simplify(sp.diff(closure, current).subs(base)) == 6
    a, b, epsilon = sp.symbols("a b epsilon", real=True)
    correction = (root * a * a, 0, 0, root * b * b, -root * (a * a + b * b))
    gaps0 = (sp.pi / 3,) * 4 + (2 * sp.pi / 3,)
    gaps = tuple(value + epsilon**2 * q for value, q in zip(gaps0, correction))
    assert sp.simplify(sum(correction)) == 0
    splits = (epsilon * a, 0, 0, 0, epsilon * b)
    phases = _lift_phases(gaps, splits)
    # Differentiate the independent twenty-edge potential along the stationary
    # correction. This includes its quadratic and mixed collective costs.
    potential = _fine_storage((0,) * 10, phases)
    coefficient = sp.expand(
        sp.diff(potential, epsilon, 4).subs(epsilon, 0) / factorial(4)
    )
    assert sp.simplify(coefficient - 2 * a * a * b * b) == 0
    assert sp.simplify(sp.diff(potential, epsilon, 2).subs(epsilon, 0)) == 0
    for index, gap in enumerate(gaps):
        flux = sp.sin(gap) * sp.cos(splits[index]) * sp.cos(splits[(index + 1) % 5])
        assert sp.simplify(sp.diff(flux, epsilon, 2).subs(epsilon, 0)) == 0
    c = (sp.Rational(1, 2),) * 4 + (-sp.Rational(1, 2),)
    quadratic = sp.expand(2 * sum(weight * q * q for weight, q in zip(c, correction)))
    t = (a * a, 0, 0, b * b, a * a + b * b)
    coupling = sp.expand(-2 * root * sum(value * q for value, q in zip(t, correction)))
    assert quadratic == -sp.Rational(3, 2) * a * a * b * b
    assert coupling == 3 * a * a * b * b
    # Independent parity plus the exact constant axes implies divisibility
    # by a^2 b^2. The implicit analytic theorem supplies a local C and radius;
    # this finite coefficient control does not certify a numerical radius.
    assert sp.cos(-a) == sp.cos(a) and sp.cos(-b) == sp.cos(b)
    stationary_potential = 20 - 4 * (
        sp.sqrt(r * r - current * current)
        + 2 * sp.sqrt(1 - current * current)
        + sp.sqrt(s * s - current * current)
        - sp.sqrt(r * r * s * s - current * current)
    )
    assert sp.simplify(stationary_potential.subs({current: root, s: 1})) == 14
    assert sp.simplify(stationary_potential.subs({current: root, r: 1})) == 14


def test_label_pair_origin_covariance_and_forward_five_cycles_preserve_winding(
    exact_candidate,
):
    data = exact_candidate
    m, c = sp.symbols("m c", real=True)
    shifted = dict(
        zip(
            data["state"],
            (m,) * 10 + tuple(c + angle for angle in data["target_phases"]),
        )
    )
    assert data["rows"].subs(shifted) == sp.zeros(20, 1)
    assert data["energy"].subs(shifted) == 14
    permutations = []
    for pair in range(5):
        permutation = list(range(10))
        permutation[2 * pair], permutation[2 * pair + 1] = (
            permutation[2 * pair + 1],
            permutation[2 * pair],
        )
        permutations.append(permutation)
    permutations.append([2 * (4 - (i // 2)) + i % 2 for i in range(10)])
    tangent = sp.Matrix(data["shared"][2])
    for permutation in permutations:
        assert {
            tuple(sorted((permutation[i], permutation[j]))) for i, j in _edges()
        } == set(_edges())
        block = sp.zeros(10)
        for i, j in enumerate(permutation):
            block[i, j] = 1
        full = sp.diag(block, block)
        assert full * data["hessian"] * full.T == data["hessian"]
        assert full * tangent * full.T == tangent
    perturbations = sp.symbols("e0:10", real=True)
    radius = Q(1, 24)  # turns: a fine phase radius pi/12
    for members in product((0, 1), repeat=5):
        nodes = tuple(2 * i + member for i, member in enumerate(members))
        principal = []
        for index, node in enumerate(nodes):
            other = nodes[(index + 1) % 5]
            offset = int(index == 4)
            turn = data["turns"][other] - data["turns"][node] + offset
            assert (I(turn) + I(-2 * radius, 2 * radius)).abs_max < Q(1, 2)
            principal.append(
                sp.Rational(turn.numerator, turn.denominator)
                + perturbations[other]
                - perturbations[node]
            )
        assert sp.simplify(sum(principal)) == 1
    assert Q(1, 3) + 2 * radius == Q(5, 12) < Q(1, 2)
