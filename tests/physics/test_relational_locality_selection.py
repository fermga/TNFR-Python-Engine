"""Exact controls for conditional uniform primitive-local phase/source selection.

The theorems quantify over every finite simple support on their declared
regular or full phase domains. These finite/symbolic controls verify the
cancellation mechanisms and scope; they install no support-birth law or executor.
"""

from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import pytest

from tests.physics._internal_mode_fixture import _metric_differential
from tests.physics.test_relational_exchange_selection import _exchange
from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
)
from tnfr.physics.phase_response import (
    derive_phase_response,
    observe_phase_source_geometry,
)


@pytest.fixture(scope="module")
def symbolic():
    return pytest.importorskip("sympy")


def _completion(graph, root):
    """Keep the induced root ball; cancel every neighboring phase gradient."""
    retained = {root, *graph.neighbors(root)}
    ball = graph.subgraph(retained).copy()
    completed = deepcopy(ball)
    for neighbor in tuple(ball.neighbors(root)):
        phase = ball.nodes[neighbor]["theta"]
        for other in tuple(ball.neighbors(neighbor)):
            reflected = 2 * phase - ball.nodes[other]["theta"]
            for slot, leaf_phase in enumerate((reflected, phase, phase)):
                leaf = ("auxiliary", neighbor, other, slot)
                completed.add_node(leaf, EPI=0, theta=leaf_phase, nu_f=0)
                completed.add_edge(neighbor, leaf, weight=1.0)
    return completed


def _triangle(s, *, balanced_root=False):
    graph = nx.complete_graph(3)
    nx.set_edge_attributes(graph, 1.0, "weight")
    phases = (0, s.pi / 6, -s.pi / 6 if balanced_root else s.pi / 3)
    for node, form, phase, capacity in zip(
        graph,
        (s.Rational(1, 3), 0, -s.Rational(1, 3)),
        phases,
        (1, 2, 3),
        strict=True,
    ):
        graph.nodes[node].update(EPI=form, theta=phase, nu_f=capacity)
    return graph


def _relative_resultant(s, graph, node):
    gaps = [graph.nodes[j]["theta"] - graph.nodes[node]["theta"] for j in graph[node]]
    return tuple(
        s.simplify(sum(function(gap) for gap in gaps)) for function in (s.cos, s.sin)
    )


def _gradient(s, graph):
    return s.Matrix([-_relative_resultant(s, graph, node)[1] for node in graph])


def _balanced_star(s, gaps):
    """Pad the supplied phase gaps with m+1 aligned positive-capacity leaves."""
    phases = (s.S.Zero, *gaps, *((s.S.Zero,) * (len(gaps) + 1)))
    graph = nx.star_graph(len(phases) - 1)
    for node, phase in zip(graph, phases, strict=True):
        graph.nodes[node].update(EPI=node - 1, theta=phase, nu_f=node + 1)
    return graph


def test_mirrored_and_aligned_leaves_cancel_work_with_positive_resultant(symbolic):
    s = symbolic
    gap = s.Symbol("gap", real=True)
    imaginary = s.sin(gap) + s.sin(-gap) + 2 * s.sin(0)
    real = s.cos(gap) + s.cos(-gap) + 2 * s.cos(0)
    assert s.simplify(imaginary) == 0
    assert s.simplify(real - 4 * s.cos(gap / 2) ** 2) == 0
    # For a wrapped non-antipodal gap, |gap/2|<pi/2, hence this square
    # is strictly positive. The exact nonacute probe needs the aligned leaves.
    assert real.subs(gap, 2 * s.pi / 3) == 1
    assert (real - 2).subs(gap, 2 * s.pi / 3) == -1
    assert real.subs(gap, s.pi) == 0


def test_completion_preserves_induced_ball_but_not_remote_metadata(symbolic):
    s = symbolic
    original = _triangle(s)
    original.add_node(3, EPI=7, theta=s.pi / 4, nu_f=5)
    original.add_edge(1, 3, weight=1.0)
    saved = deepcopy(original)
    completed = _completion(original, 0)
    retained = {0, *original.neighbors(0)}
    assert nx.utils.graphs_equal(
        original.subgraph(retained), completed.subgraph(retained)
    )
    assert set(completed.neighbors(0)) == set(original.neighbors(0))
    assert completed.has_edge(1, 2)
    assert completed.degree[0] == original.degree[0]
    assert completed.degree[1] != original.degree[1]
    assert len(completed) != len(original)
    assert 3 not in completed
    assert nx.utils.graphs_equal(original, saved)
    assert all(
        completed.nodes[node]["nu_f"] == 0 for node in completed if node not in retained
    )


def test_closed_form_balance_selects_source_on_an_induced_triangle(symbolic):
    s = symbolic
    original = _triangle(s)
    completed = _completion(original, 0)
    nodes = tuple(completed)
    positions = {node: index for index, node in enumerate(nodes)}
    sine = -_gradient(s, completed)
    degrees = s.Matrix([completed.degree[node] for node in nodes])
    assert nx.utils.graphs_equal(original, completed.subgraph(original))
    assert completed.has_edge(1, 2)
    assert all(sine[positions[node]] == 0 for node in original[0])

    def form_gradient(forms):
        return s.Matrix(
            [
                sum(
                    forms[positions[node]] - forms[positions[j]]
                    for j in completed[node]
                )
                for node in nodes
            ]
        )

    e, w, beta = s.symbols("e w beta", positive=True)
    unknown_sources = s.Matrix(s.symbols(f"g0:{len(nodes)}", real=True))
    # Separate zero-capacity preparations derive S_j=0 => g_j=0. The
    # surviving phase rate is arbitrary, without assuming separability.
    for neighbor in original[0]:
        index = positions[neighbor]
        isolated_capacity = s.Matrix([int(node == neighbor) for node in nodes])
        q = form_gradient(isolated_capacity)
        form_rate = s.Matrix(
            [
                isolated_capacity[k] * (-e * q[k] / degrees[k] + w * unknown_sources[k])
                for k in range(len(nodes))
            ]
        )
        phase_rate = s.Symbol("finite_phase_rate", real=True) * isolated_capacity
        loss = e * sum(
            isolated_capacity[k] * q[k] ** 2 / degrees[k] for k in range(len(nodes))
        )
        residual = s.simplify(q.dot(form_rate) - beta * sine.dot(phase_rate) + loss)
        assert q[index] > 0
        assert s.simplify(residual - w * q[index] * unknown_sources[index]) == 0
        assert s.solve(residual, unknown_sources[index]) == [0]

    kernel = s.Function("f")

    def singleton(gap):
        assert abs(gap) < s.pi  # This exact preparation needs no branch limit.
        return s.sign(gap) * kernel(abs(gap))  # Only singleton oddness is imposed.

    root_source = s.Symbol("root_source", real=True)
    sources = {0: root_source, 1: s.S.Zero, 2: s.S.Zero}
    for leaf in nodes:
        if leaf not in original:
            (neighbor,) = completed[leaf]
            sources[leaf] = singleton(
                completed.nodes[neighbor]["theta"] - completed.nodes[leaf]["theta"]
            )
    # Reflections of the retained neighbor-neighbor edge cancel in pairs.
    internal_leaf_sources = (
        sources[leaf] for leaf in sources if isinstance(leaf, tuple) and leaf[2] != 0
    )
    assert s.simplify(sum(internal_leaf_sources)) == 0
    root_singletons = sum(
        singleton(original.nodes[node]["theta"] - original.nodes[0]["theta"])
        for node in original[0]
    )
    weighted_source = sum(completed.degree[node] * sources[node] for node in nodes)
    assert s.simplify(weighted_source - degrees[0] * root_source + root_singletons) == 0

    # Q uses a new, wholly positive capacity assignment. No inverse of the
    # isolated or helper's zero capacities enters this conservation audit.
    for index, node in enumerate(nodes):
        completed.nodes[node]["nu_f"] = index + 1
    capacity = s.Matrix([completed.nodes[node]["nu_f"] for node in nodes])
    assert all(value > 0 for value in capacity)
    forms = s.Matrix(s.symbols(f"x0:{len(nodes)}", real=True))
    q = form_gradient(forms)
    form_rate = s.Matrix(
        [
            capacity[k] * (-e * q[k] / degrees[k] + w * sources[node])
            for k, node in enumerate(nodes)
        ]
    )
    charge_rate = sum(
        degrees[k] * form_rate[k] / capacity[k] for k in range(len(nodes))
    )
    assert s.simplify(charge_rate - w * weighted_source) == 0
    assert s.solve(charge_rate, root_source) == [root_singletons / degrees[0]]


def test_global_balance_isolates_root_without_capacity_separability(symbolic):
    s = symbolic
    original = _triangle(s)
    capacities = s.symbols("nu0:3", nonnegative=True)
    forms = s.symbols("x0:3", real=True)
    for node in original:
        original.nodes[node].update(EPI=forms[node], nu_f=capacities[node])
    completed = _completion(original, 0)
    gradient = _gradient(s, completed)
    corrections = s.Matrix(s.symbols(f"z0:{len(completed)}", real=True))
    # Neighbor rows remain completely arbitrary: no restriction on how they
    # depend on any admitted capacity is inserted into this work identity.
    for index, node in enumerate(completed):
        if node not in original:
            corrections[index] = 0
    root = tuple(completed).index(0)
    assert gradient[root] == -(1 + s.sqrt(3)) / 2
    assert all(gradient[tuple(completed).index(node)] == 0 for node in (1, 2))
    power = s.simplify(gradient.dot(corrections))
    assert s.simplify(power - gradient[root] * corrections[root]) == 0
    assert s.solve(power, corrections[root]) == [0]
    assert all(
        _relative_resultant(s, completed, node)[0].is_positive for node in completed
    )


def test_regular_nonacute_original_star_has_a_regular_completed_support(symbolic):
    s = symbolic
    graph = nx.star_graph(3)
    for node, phase in enumerate((0, 0, 0, 2 * s.pi / 3)):
        graph.nodes[node].update(EPI=node, theta=phase, nu_f=node + 1)
    completed = _completion(graph, 0)
    assert _relative_resultant(s, completed, 0) == (s.Rational(3, 2), s.sqrt(3) / 2)
    for neighbor in graph.neighbors(0):
        real, imaginary = _relative_resultant(s, completed, neighbor)
        assert real.is_positive and imaginary == 0
    # Degree-one auxiliaries may have negative real resultant; nonzero modulus
    # and a non-antipodal displacement suffice for the full regular chart.
    for leaf in completed:
        if leaf not in graph:
            real, imaginary = _relative_resultant(s, completed, leaf)
            assert s.simplify(real**2 + imaginary**2) == 1
            assert not (real == -1 and imaginary == 0)


def test_antipodal_edge_requires_density_limit_not_singular_evaluation(symbolic):
    s = symbolic
    graph = nx.Graph(((0, 1), (0, 2), (0, 3), (3, 4), (3, 5)))
    for node, phase in enumerate((0, 0, 0, s.pi, s.pi, s.pi)):
        graph.nodes[node].update(EPI=node, theta=phase, nu_f=1)
    assert all(_relative_resultant(s, graph, node) == (1, 0) for node in graph)
    assert _relative_resultant(s, _completion(graph, 0), 3) == (0, 0)
    gap = s.Symbol("gap", real=True)
    epsilon = s.Symbol("epsilon", positive=True)
    completed_real = 2 + 2 * s.cos(gap)
    assert completed_real.subs(gap, s.pi) == 0
    perturbed = s.simplify(completed_real.subs(gap, s.pi - epsilon))
    assert s.simplify(perturbed - 4 * s.sin(epsilon / 2) ** 2) == 0
    assert s.limit(perturbed, epsilon, 0, dir="+") == 0
    # This globally regular original graph has root neighbor phases (0,0,pi)
    # and resultant +1 despite that edge. Its correction is fixed only by
    # continuity from admissible completions, never by evaluating the singular
    # completion at epsilon=0. Nonzero root work is dense at its zero as well.
    root_phase = s.Symbol("root_phase", real=True)
    root_gradient = 2 * s.sin(root_phase) + s.sin(root_phase - s.pi)
    assert root_gradient.subs(root_phase, 0) == 0
    assert s.diff(root_gradient, root_phase).subs(root_phase, 0) == 1


def test_native_root_row_survives_completion_while_mediated_row_does_not(symbolic):
    s = symbolic
    original = _triangle(s, balanced_root=True)
    nx.set_node_attributes(original, 1, "nu_f")
    completed = _completion(original, 0)
    extras = []
    fields = []
    for graph in (original, completed):
        order = tuple(graph)
        adjacency = s.Matrix(
            [[int(graph.has_edge(i, j)) for j in order] for i in order]
        )
        form, phase, capacity = (
            s.Matrix([graph.nodes[node][name] for node in order])
            for name in ("EPI", "theta", "nu_f")
        )
        _, gradient, extra = _exchange(s, adjacency, form, phase, capacity)
        assert s.simplify(gradient.dot(extra)) == 0
        extras.append(extra[order.index(0)])
        materialized = deepcopy(graph)
        for _, data in materialized.nodes(data=True):
            data.update({name: float(data[name]) for name in ("EPI", "theta", "nu_f")})
        fields.append(
            evaluate_relational_exchange(
                materialized, model=RelationalExchangeModel(storage_scale=1.0)
            )
        )
    assert extras == [(1 + s.sqrt(3)) / 64, 0]
    before, after = fields
    for attribute in (
        "phase_source",
        "phase_metric",
        "form_gradient",
        "form_rate",
        "phase_rate",
    ):
        assert getattr(before, attribute)[0] == getattr(after, attribute)[0]
    # The old alternative still balances work. It fails the uniform local
    # information contract because changing remote state changes this root row.


def test_aligned_padding_admits_balanced_stars_including_obtuse_gaps(symbolic):
    s = symbolic
    gaps = s.symbols("delta0:3", real=True)
    real, imaginary = _relative_resultant(s, _balanced_star(s, gaps), 0)
    nonnegative = 2 * sum(s.cos(gap / 2) ** 2 for gap in gaps)
    assert s.trigsimp(real - 1 - nonnegative) == 0
    assert nonnegative.is_nonnegative
    assert s.simplify(imaginary - sum(map(s.sin, gaps))) == 0
    graph = _balanced_star(s, (2 * s.pi / 3, -s.pi / 3))
    assert _relative_resultant(s, graph, 0) == (3, 0)
    for leaf in tuple(graph)[1:]:
        real, imaginary = _relative_resultant(s, graph, leaf)
        assert s.simplify(real**2 + imaginary**2) == 1
        assert not (real == -1 and imaginary == 0)
    assert all(graph.nodes[node]["nu_f"] > 0 for node in graph)


def test_pair_symmetry_and_mirrored_star_remove_leaf_attribute_freedom(symbolic):
    s = symbolic
    pair = nx.path_graph(2)
    for node, phase in enumerate((0, s.pi / 6)):
        pair.nodes[node].update(EPI=node, theta=phase, nu_f=node + 1)
    left, right = s.symbols("left right", real=True)
    assert s.solve(_gradient(s, pair).dot(s.Matrix((left, right))), left) == [right]

    star = _balanced_star(s, (s.pi / 6, -s.pi / 6))
    modified = deepcopy(star)
    modified.nodes[1].update(EPI=-17, nu_f=s.Rational(13, 7))
    first, changed, fixed = s.symbols("first changed fixed", real=True)
    constraints = []
    for graph, response in ((star, first), (modified, changed)):
        values = s.Matrix(s.symbols(f"z0:{len(graph)}", real=True))
        values[1], values[2] = response, fixed
        constraints.append(s.simplify(_gradient(s, graph).dot(values)))
    assert s.solve(constraints, (first, changed)) == {first: fixed, changed: fixed}
    # The opposite leaf's primitive data and degree-one information are fixed.
    # Its response cannot read the changed leaf's form/capacity. Arbitrary root
    # and aligned-leaf responses disappear because their gradients are zero.
    # Pair symmetry transfers the same attribute independence to the parent.


def test_balanced_stars_supply_sine_fibers_and_local_additivity(symbolic):
    s = symbolic
    graph = _balanced_star(s, (2 * s.pi / 3, -s.pi / 3))
    obtuse, acute = s.symbols("obtuse acute", real=True)
    values = s.Matrix(s.symbols(f"z0:{len(graph)}", real=True))
    values[1], values[2] = obtuse, acute
    assert s.solve(_gradient(s, graph).dot(values), obtuse) == [acute]

    u, v = s.symbols("u v", positive=True)
    # The three-leaf construction is admitted for u,v>0 and u+v<1. The
    # reflected-star identity already makes f(-s)=-f(s), where f(s)=s*h(s).
    graph = _balanced_star(s, (s.asin(u), s.asin(v), -s.asin(u + v)))
    f_u, f_v, f_sum = s.symbols("f_u f_v f_sum", real=True)
    values = s.Matrix(s.symbols(f"z0:{len(graph)}", real=True))
    values[1], values[2], values[3] = f_u / u, f_v / v, f_sum / (u + v)
    gradient = _gradient(s, graph)
    assert gradient[0] == 0
    assert s.simplify(gradient.dot(values) - f_u - f_v + f_sum) == 0
    assert s.solve(gradient.dot(values), f_sum) == [f_u + f_v]
    # An even, symmetric-looking leaf response h=1+sin(delta)^2 fails this
    # necessary star identity. Its residual cannot be repaired by the center.
    nonlinear = [1 + s.sin(graph.nodes[node]["theta"]) ** 2 for node in graph]
    defect = s.factor(gradient.dot(s.Matrix(nonlinear)))
    assert defect == -3 * u * v * (u + v)
    assert defect.subs({u: s.Rational(1, 4), v: s.Rational(1, 3)}) == -s.Rational(7, 48)
    # The continuous bounded-Cauchy implication for arbitrary functions is an
    # analytic proof in the owner, not a consequence of these finite controls.


def test_completion_isolates_the_clock_with_nonzero_auxiliary_capacities(symbolic):
    s = symbolic
    original = _triangle(s)
    completed = _completion(original, 0)
    clock = s.Symbol("clock", real=True)
    values = s.Matrix(s.symbols(f"z0:{len(completed)}", real=True))
    for index, node in enumerate(completed):
        if node not in original:
            completed.nodes[node].update(EPI=index - 5, nu_f=s.Rational(index + 1, 7))
            values[index] = clock
    assert all(completed.nodes[node]["nu_f"] > 0 for node in completed)
    gradient = _gradient(s, completed)
    root = tuple(completed).index(0)
    assert all(gradient[tuple(completed).index(node)] == 0 for node in (1, 2))
    assert s.simplify(sum(gradient)) == 0
    power = s.simplify(gradient.dot(values))
    assert s.simplify(power - gradient[root] * (values[root] - clock)) == 0
    assert s.solve(power, values[root]) == [clock]
    # Auxiliary rows equal the universal degree-one clock. After subtracting
    # it their work vanishes, irrespective of their strictly positive capacity.


def test_shared_phase_geometry_retains_only_relative_equivalence_of_a_clock(symbolic):
    s = symbolic
    graph = _triangle(s)
    phases = (0, s.pi / 3, 0)
    for node, phase in zip(graph, phases, strict=True):
        graph.nodes[node]["theta"] = phase
    graph.nodes[1]["nu_f"] = 0
    rows = tuple(tuple(graph[node]) for node in graph)
    metric, _, source, symbolic_jacobian = _metric_differential(s, rows, phases)
    gram = tuple(tuple(Q(s.cos(left - right)) for right in phases) for left in phases)
    geometry = observe_phase_source_geometry(
        derive_phase_response(
            cosine_gram=gram,
            mean_neighbors=rows,
            receiver_sources=((0,), (1,), (2,)),
            phase_factor=1,
        )
    )
    jacobian = s.Matrix(geometry.scaled_source_jacobian) / s.pi
    assert (jacobian - symbolic_jacobian).applyfunc(s.simplify) == s.zeros(3)
    assert geometry.only_common_rotation
    gradient = _gradient(s, graph)
    assert (gradient + metric * source).applyfunc(s.simplify) == s.zeros(3, 1)
    clock = s.Symbol("clock", nonzero=True)
    rotation = clock * s.ones(len(graph), 1)
    capacity = s.diag(*(graph.nodes[node]["nu_f"] for node in graph))
    form_gradient = s.Matrix(
        [
            sum(
                graph.nodes[node]["EPI"] - graph.nodes[other]["EPI"]
                for other in graph[node]
            )
            for node in graph
        ]
    )
    w, beta = s.symbols("w beta", positive=True)
    reference = (w / beta) * metric.inv() * capacity * form_gradient
    with_clock = reference + rotation
    assert reference[1] == 0 and with_clock[1] == clock
    assert s.simplify(gradient.dot(rotation)) == 0
    assert (
        s.simplify(
            w * form_gradient.dot(capacity * source) + beta * gradient.dot(with_clock)
        )
        == 0
    )
    assert jacobian * rotation == s.zeros(3, 1)
    assert w * capacity * jacobian * rotation == s.zeros(3, 1)
    assert all(
        s.simplify(with_clock[i] - with_clock[j] - reference[i] + reference[j]) == 0
        for i, j in graph.edges()
    )
    assert s.matrix_multiply_elementwise(gradient, rotation) != s.zeros(3, 1)
    # This is a supplied phase-reference change in the declared model, not a
    # new production clock or proof that absolute phase is always unobservable.
    scale = s.Symbol("scale", positive=True)
    scaled_capacity_law = scale * reference + rotation
    defect = (scaled_capacity_law - scale * with_clock).applyfunc(s.simplify)
    assert defect == (1 - scale) * rotation
    assert defect.subs(scale, 2) != s.zeros(3, 1)
    # Fixed-model capacity homogeneity excludes the extra clock; changing
    # clock units could instead transform it and is a different premise.
