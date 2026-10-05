"""Exact restrictions on mixed storage with the fixed native form row.

The phase completion is unknown throughout the nonlinear obstruction tests.
At a phase-critical state it contributes no storage work, whatever its finite
value. These controls reject named candidates; they do not prove general
separability, select physical energy or install another production law.
"""

from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.physics.phase_response import derive_phase_response


@pytest.fixture(scope="module")
def symbolic():
    return pytest.importorskip("sympy")


def test_mixed_tangent_family_can_match_the_fixed_loss_with_a_different_phase_row(
    symbolic,
):
    s = symbolic
    e, w, c, degree = s.symbols("e w c degree", positive=True)
    capacity = s.Symbol("capacity", nonnegative=True)
    a, b, q, p, t = s.symbols("a b q p t", real=True)
    k = w / s.pi
    # q=Bx and p=Btheta. At the quadratic storage's phase-critical row,
    # p=-b*q/c, no finite choice of phase rate can change the energy balance.
    form_gradient = a * q + b * p
    phase_gradient = b * q + c * p
    form_rate = -capacity * (e * q + k * p) / degree
    loss = e * capacity * q**2 / degree
    critical = {p: -b * q / c}
    assert phase_gradient.subs(critical) == 0
    defect = s.factor((form_gradient * form_rate + loss).subs(critical))
    expected = capacity * q**2 / degree * (e - (a - b**2 / c) * (e - k * b / c))
    assert s.simplify(defect - expected) == 0

    # For t<e/k, sigma>0 is the Schur complement of the mixed Hessian.
    # The phase row below is a comparison completion, not F_rel by assumption.
    sigma = e / (e - k * t)
    mixed_a, mixed_b = sigma + c * t**2, c * t
    u, v = k * sigma / c + e * t, k * t
    comparison_phase_rate = capacity * (u * q + v * p) / degree
    joint_work = (mixed_a * q + mixed_b * p) * form_rate + (
        mixed_b * q + c * p
    ) * comparison_phase_rate
    assert s.simplify(joint_work + loss) == 0
    assert s.simplify(mixed_a - mixed_b**2 / c - sigma) == 0
    assert s.simplify(sigma * (e - k * t) - e) == 0
    x, theta = s.symbols("x theta", real=True)
    quadratic = (mixed_a * x**2 + 2 * mixed_b * x * theta + c * theta**2) / 2
    assert s.simplify(quadratic - sigma * x**2 / 2 - c * (theta + t * x) ** 2 / 2) == 0
    # One normalized native coefficient pair gives a genuinely mixed positive
    # Hessian. Consensus information alone therefore cannot rule mixing out.
    prepared = {e: s.Rational(1, 2), w: s.Rational(1, 2), c: 1, t: -s.pi}
    hessian = s.Matrix(((mixed_a, mixed_b), (mixed_b, c))).subs(prepared)
    assert hessian[0, 1] == -s.pi
    assert hessian[0, 0].is_positive and hessian.det() == s.Rational(1, 2)


def test_periodic_shifted_cosine_lift_fails_at_an_acute_phase_critical_pair(symbolic):
    s = symbolic
    e, w, c = s.symbols("e w c", positive=True)
    total_capacity = s.Symbol("total_capacity", positive=True)
    r, delta, t, unknown_phase_rate = s.symbols(
        "r delta t unknown_phase_rate", real=True
    )
    k = w / s.pi
    sigma = e / (e - k * t)
    cost = sigma * r**2 / 2 + c * (1 - s.cos(delta + t * r))
    form_gradient, phase_gradient = s.diff(cost, r), s.diff(cost, delta)
    form_rate = -total_capacity * (e * r + k * delta)
    loss = e * total_capacity * r**2
    work = form_gradient * form_rate + phase_gradient * unknown_phase_rate
    # On the full regular P2 domain, |delta|<pi. Both critical branches are
    # admitted when 0<t*r<pi; their form rates differ by a native phase source.
    assert s.simplify(phase_gradient.subs(delta, -t * r)) == 0
    assert s.simplify((work + loss).subs(delta, -t * r)) == 0
    critical = {delta: s.pi - t * r}
    assert s.simplify(phase_gradient.subs(critical)) == 0
    defect = s.simplify((work + loss).subs(critical))
    assert s.simplify(defect + total_capacity * w * sigma * r) == 0
    assert not defect.has(unknown_phase_rate)

    # Nonnegative periodic cost, normalized e+w=1, sigma>0, and a strictly
    # acute actual phase gap. The unobserved phase law cannot repair this.
    prepared = {
        e: s.Rational(1, 2),
        w: s.Rational(1, 2),
        c: 1,
        t: -s.pi,
        r: -s.Rational(2, 3),
        delta: s.pi / 3,
        total_capacity: 3,
    }
    assert sigma.subs(prepared) == s.Rational(1, 2)
    assert (delta + t * r).subs(prepared) == s.pi
    assert s.cos(prepared[delta]).is_positive
    assert s.simplify(cost.subs(prepared)) == s.Rational(19, 9)
    assert phase_gradient.subs(prepared) == 0
    assert s.simplify(work.subs(prepared)) == -s.Rational(1, 6)
    assert loss.subs(prepared) == s.Rational(2, 3)
    assert s.simplify(defect.subs(prepared)) == s.Rational(1, 2)


def test_form_dependent_cosine_amplitude_passes_pair_check_but_fails_cycle_balance(
    symbolic,
):
    s = symbolic
    alpha = s.Symbol("alpha", nonnegative=True)
    beta, e, total_capacity = s.symbols("beta e total_capacity", positive=True)
    r, delta = s.symbols("r delta", real=True)
    cost = r**2 / 2 + (beta + alpha * r**2) * (1 - s.cos(delta))
    phase_derivative = s.diff(cost, delta)
    assert s.simplify(phase_derivative / s.sin(delta)) == beta + alpha * r**2
    assert (beta + alpha * r**2).is_positive
    # P2 has no other phase-critical point in its regular open branch.
    assert phase_derivative.subs(delta, 0) == 0
    pair_form_gradient = s.diff(cost, r).subs(delta, 0)
    assert pair_form_gradient == r
    assert pair_form_gradient * (-e * total_capacity * r) == -e * total_capacity * r**2

    graph = nx.cycle_graph(6)
    forms = s.symbols("x0:6", real=True)
    phases = s.symbols("theta0:6", real=True)
    amplitude = s.Symbol("amplitude", nonzero=True, real=True)
    capacities = s.symbols("nu0:6", positive=True)
    prepared_forms = tuple((-1) ** node * amplitude for node in graph)
    prepared_phases = tuple(node * s.pi / 3 for node in graph)
    prepared = dict(zip(forms + phases, prepared_forms + prepared_phases, strict=True))
    joint = sum(
        cost.subs({r: forms[i] - forms[j], delta: phases[i] - phases[j]})
        for i, j in graph.edges()
    )
    form_gradient = s.Matrix(
        [s.simplify(s.diff(joint, coordinate).subs(prepared)) for coordinate in forms]
    )
    phase_gradient = s.Matrix(
        [s.simplify(s.diff(joint, coordinate).subs(prepared)) for coordinate in phases]
    )
    assert phase_gradient == s.zeros(6, 1)
    rows = tuple(tuple(graph[node]) for node in graph)
    gram = tuple(
        tuple(Q(s.cos(left - right)) for right in prepared_phases)
        for left in prepared_phases
    )
    reference = derive_phase_response(
        cosine_gram=gram,
        mean_neighbors=rows,
        receiver_sources=tuple((node,) for node in graph),
        phase_factor=1,
    )
    assert reference.mean_resultant_squared == (1,) * 6
    for node, row in enumerate(rows):
        gaps = [prepared_phases[other] - prepared_phases[node] for other in row]
        assert s.simplify(sum(map(s.sin, gaps))) == 0
        assert s.simplify(sum(map(s.cos, gaps))) == 1
    # The exact native phase source is zero on this regular winding-one cycle.
    q = s.Matrix(
        [
            sum(prepared_forms[node] - prepared_forms[other] for other in row)
            for node, row in enumerate(rows)
        ]
    )
    assert (form_gradient - (1 + alpha) * q).applyfunc(s.simplify) == s.zeros(6, 1)
    rates = s.Matrix(
        [-e * nu * value / 2 for nu, value in zip(capacities, q, strict=True)]
    )
    loss = e * sum(nu * value**2 / 2 for nu, value in zip(capacities, q, strict=True))
    unknown_phase_rates = s.Matrix(s.symbols("phase_rate0:6", real=True))
    defect = s.factor(
        form_gradient.dot(rates) + phase_gradient.dot(unknown_phase_rates) + loss
    )
    assert s.simplify(defect + alpha * loss) == 0
    assert s.simplify(loss - 8 * e * amplitude**2 * sum(capacities)) == 0
    assert defect.subs(
        {
            alpha: 1,
            e: s.Rational(1, 2),
            amplitude: s.Rational(1, 4),
            **dict.fromkeys(capacities, 1),
        }
    ) == -s.Rational(3, 2)
    # Pair agreement was only a necessary check. At this phase-critical cycle,
    # no finite phase row can restore the prescribed Dirichlet loss for alpha>0.


def test_joint_reflection_completion_isolates_work_without_own_capacity_freezing(
    symbolic,
):
    s = symbolic
    r, delta, coupling = s.symbols("r delta coupling", real=True)
    beta = s.Symbol("beta", positive=True)
    # This globally nonnegative cost has an r*sin(delta) cross term: reversal
    # is simultaneous, with no separate-evenness assumption smuggled in.
    cost = (r + coupling * s.sin(delta)) ** 2 / 2 + beta * (1 - s.cos(delta))
    derivatives = (s.diff(cost, r), s.diff(cost, delta))
    assert s.simplify(cost.subs({r: -r, delta: -delta}) - cost) == 0
    assert all(
        s.simplify(value.subs({r: -r, delta: -delta}) + value) == 0
        for value in derivatives
    )
    graph = nx.star_graph(2)
    for node, form, phase in zip(
        graph,
        (s.Rational(1, 3), -s.Rational(1, 5), s.Rational(2, 7)),
        (0, s.pi / 6, -s.pi / 3),
        strict=True,
    ):
        graph.nodes[node].update(EPI=form, theta=phase, nu_f=int(node == 0))
    completed = deepcopy(graph)
    for neighbor in graph.neighbors(0):
        parent = graph.nodes[neighbor]
        reflected = (
            2 * parent["EPI"] - graph.nodes[0]["EPI"],
            2 * parent["theta"] - graph.nodes[0]["theta"],
        )
        aligned = (parent["EPI"], parent["theta"])
        for slot, (form, phase) in enumerate((reflected, aligned, aligned)):
            leaf = (neighbor, slot)
            completed.add_node(leaf, EPI=form, theta=phase, nu_f=0)
            completed.add_edge(neighbor, leaf)
    assert nx.utils.graphs_equal(graph, completed.subgraph(tuple(graph)))
    assert tuple(graph[0]) == tuple(completed[0])
    gradient_rows = []
    for node in completed:
        row = []
        for derivative in derivatives:
            row.append(
                s.simplify(
                    sum(
                        derivative.subs(
                            {
                                r: completed.nodes[node]["EPI"]
                                - completed.nodes[other]["EPI"],
                                delta: completed.nodes[node]["theta"]
                                - completed.nodes[other]["theta"],
                            }
                        )
                        for other in completed[node]
                    )
                )
            )
        gradient_rows.append(tuple(row))
    for neighbor in graph.neighbors(0):
        assert gradient_rows[tuple(completed).index(neighbor)] == (0, 0)
        gaps = [
            completed.nodes[other]["theta"] - completed.nodes[neighbor]["theta"]
            for other in completed[neighbor]
        ]
        assert s.simplify(sum(map(s.sin, gaps))) == 0
        assert s.simplify(sum(map(s.cos, gaps))).is_positive

    zero_star_rate = s.Symbol("zero_star_rate", real=True)
    assert s.solve(zero_star_rate - 2 * zero_star_rate, zero_star_rate) == [0]
    phase_rates = s.Matrix(s.symbols(f"phase_rate0:{len(completed)}", real=True))
    for index, node in enumerate(completed):
        if node not in graph:
            ball = {node, *completed.neighbors(node)}
            assert 0 not in ball
            assert all(completed.nodes[item]["nu_f"] == 0 for item in ball)
            phase_rates[index] = 0
    # F(2*0)=2F(0), together with locality, forces these entirely inactive
    # leaf-star rates to zero. Neighbor rows seeing the active root remain
    # arbitrary, even though their own capacity is zero.
    phase_gradient = s.Matrix([row[1] for row in gradient_rows])
    assert (
        s.simplify(phase_gradient.dot(phase_rates) - phase_gradient[0] * phase_rates[0])
        == 0
    )
    assert all(completed.nodes[node]["nu_f"] == 0 for node in completed if node != 0)


def test_aligned_padding_obstructs_nonzero_phase_resultant_at_zero_joint_torque(
    symbolic,
):
    s = symbolic
    inverse_padding = s.Symbol("inverse_padding", positive=True)
    original_degree = s.Symbol("original_degree", positive=True, integer=True)
    real, imaginary = s.symbols("real imaginary", real=True)
    normalized_angle = (original_degree + 1 / inverse_padding) * s.atan(
        imaginary * inverse_padding / (1 + real * inverse_padding)
    )
    expansion = s.series(normalized_angle, inverse_padding, 0, 2).removeO().expand()
    assert expansion.coeff(inverse_padding, 0) == imaginary
    assert (
        s.simplify(
            expansion.coeff(inverse_padding, 1) - (original_degree - real) * imaginary
        )
        == 0
    )
    gaps = s.symbols("delta0:3", real=True)
    defect = len(gaps) - sum(map(s.cos, gaps))
    squares = 2 * sum(s.sin(gap / 2) ** 2 for gap in gaps)
    assert s.trigsimp(defect - squares) == 0
    assert squares.is_nonnegative
    # If the imaginary sum is nonzero, some gap is unaligned, making this
    # coefficient nonzero. For an exact two-leaf preparation it is positive.
    coefficient = ((original_degree - real) * imaginary).subs(
        {original_degree: 2, real: 1 + s.sqrt(3) / 2, imaginary: s.Rational(1, 2)}
    )
    assert coefficient.is_positive
    assert s.simplify(coefficient - (2 - s.sqrt(3)) / 4) == 0
    # Appended (r,delta)=(0,0) observations do not alter q, W_r or W_delta.
    # Constancy over all sufficiently large integer paddings would contradict
    # this nonzero asymptotic coefficient unless the imaginary sum vanishes.
    form_sum = s.Symbol("form_sum", nonzero=True, real=True)
    form_gradient = s.Symbol("form_gradient", real=True)
    e = s.Symbol("e", positive=True)
    assert s.solve(form_gradient * e * form_sum - e * form_sum**2, form_gradient) == [
        form_sum
    ]
    # This last implication is conditional on the isolated zero-torque work
    # identity; finite examples are not the all-support storage theorem.
