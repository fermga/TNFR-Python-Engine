"""Independent exact controls for the full two-C5 critical classification.

These verify constructive geometry and full sine rows. Long-time convergence
uses the separate compactness/connected-limit-set proof, not a trajectory test.
"""

from collections import Counter
from fractions import Fraction as Q
from functools import lru_cache

import networkx as nx
import pytest

from tests.physics.test_relational_sine_formation import _support
from tnfr.physics.phase_cycle_geometry import (
    classify_c5_sine_critical_set,
    derive_phase_cycle_geometry,
    reconstruct_phase_cycle_state,
)


@pytest.fixture(scope="module")
def critical_family():
    edges, _ = _support()
    graph = nx.Graph()
    graph.add_nodes_from(range(11))
    graph.add_edges_from(edges)
    geometry = derive_phase_cycle_geometry(graph)
    cycles = (tuple(range(5)), tuple(range(5, 10)))
    return graph, classify_c5_sine_critical_set(geometry, cycles=cycles)


def _principal_sine_turn(turn):
    """Independent exact inverse-sine branch, using only circular identities."""
    value = (turn + Q(1, 2)) % 1 - Q(1, 2)
    if value > Q(1, 4):
        return Q(1, 2) - value
    if value < -Q(1, 4):
        return -Q(1, 2) - value
    return value


def test_exact_branch_masks_close_uniquely_including_nonacute_and_antipodal(
    critical_family,
):
    _, family = critical_family
    options = family.cycle_edge_turn_options
    phase_shapes, branch_counts = set(), Counter()
    for option in options:
        assert len(option) == 5
        assert sum(option).denominator == 1
        principals = tuple(_principal_sine_turn(value) for value in option)
        assert len(set(principals)) == 1
        alpha = principals[0]
        assert abs(alpha) < Q(1, 4)
        negative = tuple(
            abs((value + Q(1, 2)) % 1 - Q(1, 2)) > Q(1, 4) for value in option
        )
        k = sum(negative)
        assert all(
            (value - (Q(1, 2) - alpha if second else alpha)).denominator == 1
            for value, second in zip(option, negative)
        )
        # The complete inverse-sine closure, rather than a fitted residual,
        # determines the integer and excludes a zero odd-cycle denominator.
        closure_integer = (5 - 2 * k) * alpha + Q(k, 2)
        assert closure_integer.denominator == 1
        assert alpha == (2 * closure_integer - k) / (2 * (5 - 2 * k))
        branch_counts[k] += 1
        phase = [Q(0)]
        for gap in option[:-1]:
            phase.append((phase[-1] + gap) % 1)
        assert (phase[0] - phase[-1] - option[-1]).denominator == 1
        shape = tuple(phase)
        assert shape not in phase_shapes
        phase_shapes.add(shape)
    assert branch_counts == {0: 3, 1: 10, 2: 10, 4: 5, 5: 2}
    assert len(phase_shapes) == 30
    assert family.relative_state_count == len(phase_shapes) ** 2 * 4
    # All sixteen zero-current Ising states are retained, including exact
    # antipodes; the two all-negative-cosine twists are retained as well.
    assert (
        sum(all(_principal_sine_turn(t) == 0 for t in option) for option in options)
        == 16
    )
    assert len({tuple((-t) % 1 for t in option) for option in options}) == 30
    assert {tuple(t % 1 for t in option) for option in options} == {
        tuple((-t) % 1 for t in option) for option in options
    }


def test_reconstructed_nonacute_patterns_annul_every_complete_fine_row(critical_family):
    s = pytest.importorskip("sympy")
    graph, family = critical_family
    e, w, beta, common = s.symbols("loss exchange beta common", positive=True)
    capacity = tuple(s.Rational(i + 1, 7) for i in range(11))

    @lru_cache(None)
    def sine(turn):
        return s.simplify(s.sin(2 * s.pi * s.Rational(turn % 1)))

    seen_bridges = set()
    # Every ring option appears in both roles; the four bridge combinations
    # appear too. The oracle evaluates the original full neighbor rows.
    for index in range(30):
        bridges = (Q(index % 2, 2), Q((index // 2) % 2, 2))
        state = family.reconstruct(
            cycle_choices=(index, (7 * index + 3) % 30), bridge_turns=bridges
        )
        seen_bridges.add(bridges)
        phase = dict(zip(state.geometry.nodes, state.nodal_turns))
        for node in graph:
            gradient = sum(common - common for _ in graph[node])
            current = s.simplify(
                sum(sine(phase[neighbor] - phase[node]) for neighbor in graph[node])
            )
            degree = graph.degree[node]
            form_rate = capacity[node] * (
                -e * gradient / degree + w * current / (s.pi * degree)
            )
            phase_rate = w * capacity[node] * gradient / (beta * s.pi * degree)
            assert current == form_rate == phase_rate == 0
        for (i, j), turn, offset in zip(
            state.geometry.edges, state.edge_turns, state.edge_integer_offsets
        ):
            assert state.nodal_turns[j] - state.nodal_turns[i] - turn == offset
        assert state.sine_balance_status == "proved_by_period_reflection_cancellation"
    assert seen_bridges == {
        (Q(0), Q(0)),
        (Q(0), Q(1, 2)),
        (Q(1, 2), Q(0)),
        (Q(1, 2), Q(1, 2)),
    }


def test_cycle_orientation_and_node_labels_only_reparameterize_the_shapes(
    critical_family,
):
    graph, family = critical_family
    consensus = next(
        i for i, option in enumerate(family.cycle_edge_turn_options) if not any(option)
    )

    def donor_shapes(report, node_names):
        result = set()
        for index in range(30):
            state = report.reconstruct(
                cycle_choices=(index, consensus), bridge_turns=(0, 0)
            )
            phase = dict(zip(state.geometry.nodes, state.nodal_turns))
            anchor = phase[node_names[10]]
            result.add(tuple((phase[node_names[i]] - anchor) % 1 for i in range(11)))
        return result

    original = donor_shapes(family, tuple(range(11)))
    labels = tuple(("constituent", 10 - i) for i in range(11))
    renamed = nx.relabel_nodes(graph, dict(enumerate(labels)))
    reversed_family = classify_c5_sine_critical_set(
        derive_phase_cycle_geometry(renamed),
        cycles=(
            tuple(labels[i] for i in (0, 4, 3, 2, 1)),
            tuple(labels[i] for i in range(5, 10)),
        ),
    )
    assert donor_shapes(reversed_family, labels) == original


def test_antipodal_bridge_is_a_sine_equilibrium_outside_native_arg_domain(
    critical_family,
):
    s = pytest.importorskip("sympy")
    graph, family = critical_family
    consensus = next(
        i for i, option in enumerate(family.cycle_edge_turn_options) if not any(option)
    )
    state = family.reconstruct(
        cycle_choices=(consensus, consensus), bridge_turns=(0, Q(1, 2))
    )
    phase = dict(zip(state.geometry.nodes, state.nodal_turns))
    # The mediator sees opposite port phasors: its argument is undefined,
    # while its complete sine row is exactly zero and remains meaningful.
    resultant = sum(
        s.exp(2 * s.pi * s.I * s.Rational(phase[j] - phase[10])) for j in graph[10]
    )
    assert s.simplify(resultant) == 0
    assert all(
        s.simplify(
            sum(s.sin(2 * s.pi * s.Rational(phase[j] - phase[i])) for j in graph[i])
        )
        == 0
        for i in graph
    )
    with pytest.raises(ValueError, match="acute"):
        reconstruct_phase_cycle_state(state.geometry, edge_turns=state.edge_turns)


def test_odd_cycle_and_positive_capacity_premises_cannot_be_dropped():
    s = pytest.importorskip("sympy")
    alpha, common = s.symbols("alpha common", real=True)
    # An even cycle has an actual continuum: alternating branches close for
    # every alpha. This is why a finite C5 argument is not generic topology.
    phase = (0, alpha, s.pi, s.pi + alpha)
    for node in range(4):
        neighbors = ((node - 1) % 4, (node + 1) % 4)
        current = sum(s.sin(phase[j] - phase[node]) for j in neighbors)
        assert s.trigsimp(current) == 0
        assert sum(common - common for _ in neighbors) == 0
    assert s.diff(phase[1] - phase[0], alpha) == 1
    assert s.exp(5 * s.pi * s.I / 2) != 1
    assert s.exp(-5 * s.pi * s.I / 2) != 1
    # With all capacities zero, every state is stationary, irrespective of
    # its two gradients. Thus the classified finite set is not exhaustive
    # on that excluded boundary of the complete law.
    _, neighbors = _support()
    form = (1,) + (0,) * 10
    phase = (alpha,) + (0,) * 10
    q = tuple(sum(form[i] - form[j] for j in row) for i, row in enumerate(neighbors))
    currents = tuple(
        sum(s.sin(phase[j] - phase[i]) for j in row) for i, row in enumerate(neighbors)
    )
    assert q[0] == 3 and s.simplify(currents[0]) == -3 * s.sin(alpha)
    e, a, b = s.symbols("e a b", positive=True)
    capacity = s.Symbol("capacity", nonnegative=True)
    for i, row in enumerate(neighbors):
        form_rate = capacity * (-e * q[i] + a * currents[i]) / len(row)
        phase_rate = b * capacity * q[i] / len(row)
        assert form_rate.subs(capacity, 0) == phase_rate.subs(capacity, 0) == 0


def test_phase_index_follows_constrained_edge_congruence_not_negative_edge_count(
    critical_family,
):
    s = pytest.importorskip("sympy")
    _, family = critical_family
    tangent = s.eye(5)[:, :4]
    tangent[4, :] = s.ones(1, 4) * -1
    consensus = next(
        i for i, option in enumerate(family.cycle_edge_turn_options) if not any(option)
    )
    by_mask_count, index_counts = {}, Counter()
    for index, option in enumerate(family.cycle_edge_turn_options):
        signs = tuple(
            -1 if abs((turn + Q(1, 2)) % 1 - Q(1, 2)) > Q(1, 4) else 1
            for turn in option
        )
        negative_edges = signs.count(-1)
        diagonal = s.diag(*signs)
        complement = s.Matrix(signs)
        change = tangent.row_join(complement)
        reduced = tangent.T * diagonal * tangent
        assert change.det() != 0
        assert change.T * diagonal * change == s.diag(reduced, sum(signs))
        # Compute the actual exact four-dimensional inertia once per branch
        # type, independently of the implementation's combinatorial rule.
        if negative_edges not in by_mask_count:
            characteristic = reduced.charpoly().as_poly()
            assert characteristic.eval(0) != 0
            _, factors = characteristic.factor_list()
            by_mask_count[negative_edges] = (
                sum(
                    multiplicity * factor.count_roots(0, s.oo)
                    for factor, multiplicity in factors
                ),
                sum(
                    multiplicity * factor.count_roots(-s.oo, 0)
                    for factor, multiplicity in factors
                ),
            )
        positive, negative = by_mask_count[negative_edges]
        index_counts[negative] += 1
        report = family.phase_hessian_inertia(
            cycle_choices=(index, consensus), bridge_turns=(0, 0)
        )
        assert report.relative_inertia == (positive + 6, negative, 0)
        assert report.common_phase_nullity == 1
    assert by_mask_count[4] == (1, 3)
    assert by_mask_count[5] == (0, 4)
    z = s.Symbol("z")
    ring_polynomial = sum(count * z**index for index, count in index_counts.items())
    whole = s.Poly(s.expand(ring_polynomial**2 * (1 + z) ** 2), z)
    histogram = tuple(int(whole.nth(index)) for index in range(11))
    assert family.phase_hessian_index_counts == histogram
    assert histogram[0] == 9 and sum(histogram[1:]) == 3591


def test_full_hessian_retains_bridge_and_both_cycle_variation_spaces(critical_family):
    s = pytest.importorskip("sympy")
    _, family = critical_family
    options = family.cycle_edge_turn_options

    def choice(negative_count):
        return next(
            i
            for i, option in enumerate(options)
            if sum(abs((t + Q(1, 2)) % 1 - Q(1, 2)) > Q(1, 4) for t in option)
            == negative_count
        )

    selection = (choice(1), choice(4))
    report = family.phase_hessian_inertia(
        cycle_choices=selection, bridge_turns=(0, Q(1, 2))
    )
    state = report.state
    hessian = s.zeros(11)
    for left, right in state.geometry.edges:
        angle = (
            2 * s.pi * s.Rational(state.nodal_turns[right] - state.nodal_turns[left])
        )
        weight = s.simplify(s.cos(angle))
        hessian[left, left] += weight
        hessian[right, right] += weight
        hessian[left, right] -= weight
        hessian[right, left] -= weight
    # Four independent edge differences per ring and two bridge variations
    # span the whole ten-dimensional phase quotient with mediator fixed.
    change = s.zeros(11, 10)
    for offset, first_column, bridge_column in ((0, 0, 8), (5, 4, 9)):
        for j in range(5):
            change[offset + j, bridge_column] = 1
            for edge in range(j):
                change[offset + j, first_column + edge] = 1
    assert change[:10, :].det() != 0
    ring_tangent = s.eye(5)[:, :4]
    ring_tangent[4, :] = s.ones(1, 4) * -1
    blocks = []
    for option_index in selection:
        weights = tuple(
            s.simplify(s.cos(2 * s.pi * s.Rational(t))) for t in options[option_index]
        )
        blocks.append(ring_tangent.T * s.diag(*weights) * ring_tangent)
    assert (change.T * hessian * change - s.diag(*blocks, 1, -1)).applyfunc(
        s.simplify
    ) == s.zeros(10)
    assert hessian * s.ones(11, 1) == s.zeros(11, 1)
    assert report.relative_inertia == (5, 5, 0)


@pytest.mark.parametrize("unstable", (False, True))
def test_full_reciprocal_linearization_matches_index_with_noncommuting_mobility(
    critical_family, unstable
):
    mp = pytest.importorskip("mpmath")
    from tnfr.dynamics.relational import RelationalExchangeModel
    from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange

    graph = critical_family[0].copy()
    capacities = (
        Q(1),
        Q(1, 2),
        Q(3, 2),
        Q(2),
        Q(3, 4),
        Q(5, 4),
        Q(7, 4),
        Q(1),
        Q(1, 2),
        Q(3, 4),
        Q(3, 2),
    )
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=capacities[node])
    model = RelationalExchangeModel(
        Q(7, 4), epi_weight=1, phase_weight=3, phase_domain="regular"
    )
    source = bound_relational_sine_exchange(graph, reference_model=model)
    asymptotic = source.asymptotic_equilibria(
        cycles=(tuple(range(5)), tuple(range(5, 10)))
    )
    family = asymptotic.critical_set
    options = family.cycle_edge_turn_options
    choices = tuple(
        next(
            i
            for i, option in enumerate(options)
            if sum(abs((t + Q(1, 2)) % 1 - Q(1, 2)) > Q(1, 4) for t in option) == count
        )
        for count in ((1, 4) if unstable else (0, 0))
    )
    report = asymptotic.classify_equilibrium(
        cycle_choices=choices, bridge_turns=(0, Q(1, 2) if unstable else 0)
    )
    with mp.workdps(65):

        def number(value):
            value = Q(value)
            return mp.mpf(value.numerator) / value.denominator

        e, w = map(number, model.effective_weights)
        beta = number(model.storage_scale)
        a, b = w / mp.pi, w / (beta * mp.pi)
        mobility = tuple(
            number(nu) / graph.degree[i] for i, nu in enumerate(capacities)
        )
        phase = tuple(
            2 * mp.pi * number(turn) for turn in report.phase_hessian.state.nodal_turns
        )
        center = [mp.mpf(0)] * 11 + list(phase)

        def fine_row(values, row):
            node = row % 11
            q = mp.fsum(values[node] - values[j] for j in graph[node])
            if row >= 11:
                return b * mobility[node] * q
            current = mp.fsum(
                mp.sin(values[11 + j] - values[11 + node]) for j in graph[node]
            )
            return mobility[node] * (-e * q + a * current)

        full = mp.matrix(22)
        for column in range(22):
            for row in range(22):

                def varied(value):
                    values = list(center)
                    values[column] = value
                    return fine_row(values, row)

                full[row, column] = mp.diff(varied, center[column])
        # Restrict both independently conserved weighted means. This spans
        # the same dynamics as fixing form mean and removing common phase.
        leaf = mp.matrix(22, 20)
        for block in range(2):
            for column in range(10):
                leaf[11 * block + column, 10 * block + column] = 1
                leaf[11 * block + 10, 10 * block + column] = (
                    -mobility[10] / mobility[column]
                )
        image = full * leaf
        rows = tuple(range(10)) + tuple(range(11, 21))
        reduced = mp.matrix(
            [[image[row, column] for column in range(20)] for row in rows]
        )
        assert max(abs(v) for v in image - leaf * reduced) < mp.mpf("1e-55")
        # The two common directions remain neutral in the full state, not in
        # the admitted twenty-dimensional relative/conserved-mean leaf.
        for block in range(2):
            common = mp.matrix(
                [int(block * 11 <= i < (block + 1) * 11) for i in range(22)]
            )
            assert max(abs(v) for v in full * common) < mp.mpf("1e-55")
        laplacian, hessian = mp.matrix(11), mp.matrix(11)
        for i, j in graph.edges:
            cosine = mp.cos(phase[j] - phase[i])
            for matrix, weight in ((laplacian, 1), (hessian, cosine)):
                matrix[i, i] += weight
                matrix[j, j] += weight
                matrix[i, j] -= weight
                matrix[j, i] -= weight
        root_k = mp.diag([mp.sqrt(value) for value in mobility])
        B, C = root_k * laplacian * root_k, root_k * hessian * root_k
        assert max(abs(v) for v in B * C - C * B) > mp.mpf("1e-3")
        eigenvalues = mp.eig(reduced, left=False, right=False)
        assert all(abs(mp.re(value)) > mp.mpf("1e-30") for value in eigenvalues)
        positives = [value for value in eigenvalues if mp.re(value) > 0]
        assert all(abs(mp.im(value)) < mp.mpf("1e-50") for value in positives)
        assert (
            len(positives)
            == report.phase_hessian.relative_inertia[1]
            == (5 if unstable else 0)
        )
        assert report.relative_unstable_modes == len(positives)
        assert report.relative_stable_modes == 20 - len(positives)
        assert report.relative_center_modes == 0
        assert report.local_exponential_attraction_certified is (not unstable)
        assert report.nonlinear_instability_certified is unstable


def test_same_functional_excludes_all_four_attractive_double_twists_only(
    critical_family,
):
    s = pytest.importorskip("sympy")
    _, family = critical_family
    options = tuple(
        i
        for i, option in enumerate(family.cycle_edge_turn_options)
        if all(abs((t + Q(1, 2)) % 1 - Q(1, 2)) < Q(1, 4) for t in option)
    )
    v5 = 5 * (1 - s.cos(2 * s.pi / 5))
    excluded, compatible = 0, 0
    for donor in options:
        for receiver in options:
            hessian = family.phase_hessian_inertia(
                cycle_choices=(donor, receiver), bridge_turns=(0, 0)
            )
            assert hessian.relative_inertia == (10, 0, 0)
            state = hessian.state
            # Uniform form gives q=0, so the old W cross term vanishes.
            # Evaluate the actual full phase potential, including bridges.
            potential = s.simplify(
                sum(
                    1
                    - s.cos(
                        2
                        * s.pi
                        * s.Rational(state.nodal_turns[j] - state.nodal_turns[i])
                    )
                    for i, j in state.geometry.edges
                )
            )
            twisted_rings = sum(
                any(family.cycle_edge_turn_options[index])
                for index in (donor, receiver)
            )
            assert s.simplify(potential - twisted_rings * v5) == 0
            if twisted_rings == 2:
                assert s.simplify(
                    potential - (v5 + s.Rational(16, 5)) - s.Rational(1, 4)
                ).is_positive
                excluded += 1
            else:
                assert s.simplify(v5 - potential).is_nonnegative
                compatible += 1
    assert (excluded, compatible) == (4, 5)
    # The other five merely pass this endpoint obstruction. This does not
    # assign a basin to the six-coordinate preparation or exclude saddle limits.
