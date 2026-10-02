"""Exact cycle-sector and regular-chart boundaries, without trajectories.

The derived skip-two graph is an observation, not additional physical support.
Symbolic controls separate an execution cutoff from actual field singularities.
"""

import math
import pickle
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.phase_resultant_sectors import derive_cycle_resultant_sector
from tnfr.physics.winding_certificates import certify_phase_winding


def _cycle(turns):
    graph = nx.cycle_graph(len(turns))
    graph.graph["untouched"] = {"values": [1, 2]}
    return graph, dict(enumerate(turns))


def _derive(turns):
    graph, phases = _cycle(turns)
    return derive_cycle_resultant_sector(
        graph, cycle_nodes=tuple(graph), phase_turns=phases
    )


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def test_acute_c5_consensus_and_twist_have_different_resultant_sectors():
    consensus = _derive((Q(0),) * 5)
    twist = _derive(tuple(Q(i, 5) for i in range(5)))
    assert consensus.support_winding == 0
    assert consensus.auxiliary_winding == (0,)
    assert twist.support_winding == 1
    assert twist.auxiliary_cycles == ((0, 2, 4, 1, 3),)
    assert twist.auxiliary_winding == (2,)
    assert twist.edge_turns == (Q(1, 5),) * 5
    assert consensus.strict_acute_edges and twist.strict_acute_edges
    assert consensus.regular_phase_chart and twist.regular_phase_chart
    assert consensus.resultant_sector_available and twist.resultant_sector_available


def test_even_cycle_keeps_separate_parity_sectors():
    twist = _derive(tuple(Q(i, 6) for i in range(6)))
    assert twist.support_winding == 1
    assert twist.auxiliary_cycles == ((0, 2, 4), (1, 3, 5))
    assert twist.auxiliary_winding == (1, 1)
    assert twist.strict_acute_edges and twist.regular_phase_chart


def test_original_edge_antipode_is_regular_and_does_not_fix_original_winding():
    boundary = _derive((Q(0), Q(1, 8), Q(1, 4), Q(3, 8), Q(1, 2)))
    assert boundary.support_winding is None
    assert boundary.auxiliary_winding == (0,)
    assert boundary.zero_resultant_nodes == ()
    assert boundary.negative_real_resultant_nodes == ()
    assert boundary.regular_phase_chart and boundary.resultant_sector_available
    assert not boundary.strict_acute_edges
    before = _derive(tuple(Q(3, 5) * Q(i, 5) for i in range(5)))
    after = _derive(tuple(Q(13, 20) * Q(i, 5) for i in range(5)))
    assert (before.support_winding, after.support_winding) == (0, 1)
    assert before.auxiliary_winding == after.auxiliary_winding == (0,)
    assert before.regular_phase_chart and after.regular_phase_chart


def test_zero_resultant_and_negative_real_branch_are_distinct():
    zero = _derive((Q(0), Q(1, 4), Q(1, 2), Q(2, 3), Q(5, 6)))
    assert zero.support_winding == 1
    assert zero.zero_resultant_nodes == (1,)
    assert zero.negative_real_resultant_nodes == ()
    assert zero.auxiliary_winding == (None,)
    assert not zero.resultant_sector_available and not zero.regular_phase_chart
    branch = _derive(tuple(Q(2 * i, 5) for i in range(5)))
    assert branch.support_winding == 2
    assert branch.zero_resultant_nodes == ()
    assert branch.negative_real_resultant_nodes == tuple(range(5))
    assert branch.auxiliary_winding == (-1,)
    assert branch.resultant_sector_available and not branch.regular_phase_chart


def test_exact_turns_preserve_caller_orientation_and_leave_graph_untouched():
    labels = ("first", ("pair", 1), 7, "last", 2)
    graph = nx.Graph()
    graph.add_nodes_from(reversed(labels))
    graph.add_edges_from(zip(labels, labels[1:] + labels[:1]))
    graph.graph["payload"] = [1, 2]
    phases = {node: Q(i, 5) + 7 for i, node in enumerate(labels)}
    before = _snapshot(graph)
    report = derive_cycle_resultant_sector(
        graph, cycle_nodes=labels, phase_turns=phases
    )
    reverse = derive_cycle_resultant_sector(
        graph, cycle_nodes=tuple(reversed(labels)), phase_turns=phases
    )
    assert _snapshot(graph) == before
    assert report.geometry.nodes == tuple(reversed(labels))
    assert report.cycle_nodes == labels
    assert report.phase_turns == tuple(Q(i, 5) for i in range(5))
    assert report.support_winding == 1 and report.auxiliary_winding == (2,)
    assert reverse.support_winding == -1 and reverse.auxiliary_winding == (-2,)


@pytest.mark.parametrize(
    "issue", ("chord", "missing_edge", "short_cycle", "wrong_order", "duplicate_order")
)
def test_wrong_topology_or_order_is_rejected_without_mutation(issue):
    graph, phases = _cycle(tuple(Q(i, 5) for i in range(5)))
    order = tuple(graph)
    if issue == "chord":
        graph.add_edge(0, 2)
    elif issue == "missing_edge":
        graph.remove_edge(0, 4)
    elif issue == "short_cycle":
        graph, phases = _cycle((Q(0),) * 4)
        order = tuple(graph)
    elif issue == "wrong_order":
        order = (0, 2, 1, 3, 4)
    else:
        order = (0, 1, 2, 3, 3)
    before = _snapshot(graph)
    with pytest.raises(ValueError):
        derive_cycle_resultant_sector(graph, cycle_nodes=order, phase_turns=phases)
    assert _snapshot(graph) == before


@pytest.mark.parametrize("issue", ("missing", "extra", "float", "bool", "unordered"))
def test_exact_phase_mapping_and_order_admission(issue):
    graph, phases = _cycle(tuple(Q(i, 5) for i in range(5)))
    order = tuple(graph)
    if issue == "missing":
        phases.pop(4)
    elif issue == "extra":
        phases[5] = Q(0)
    elif issue == "float":
        phases[0] = 0.0
    elif issue == "bool":
        phases[0] = False
    else:
        order = set(order)
    with pytest.raises((TypeError, ValueError)):
        derive_cycle_resultant_sector(graph, cycle_nodes=order, phase_turns=phases)


def test_numpy_rational_components_do_not_overflow_before_modulo_one():
    np = pytest.importorskip("numpy")
    denominator = np.int64(2**62)
    declared = Q(np.int64(2**62 - 1), denominator)
    report = _derive((declared, Q(0), Q(0), Q(0), Q(0)))
    assert report.phase_turns[0] == Q(2**62 - 1, 2**62)
    assert report.support_winding == 0 and report.strict_acute_edges
    integers = _derive(tuple(np.int64(2**62 + i) for i in range(5)))
    assert integers.phase_turns == (Q(0),) * 5


@pytest.fixture(scope="module")
def symbolic():
    return pytest.importorskip("sympy")


def test_sharp_c5_zero_resultant_energy_barrier_has_independent_algebra(symbolic):
    s = symbolic
    turns = (0, s.Rational(1, 4), s.Rational(1, 2), s.Rational(2, 3), s.Rational(5, 6))
    phase = tuple(2 * s.pi * value for value in turns)
    energy = sum(1 - s.cos(phase[(i + 1) % 5] - phase[i]) for i in range(5))
    assert s.simplify(energy) == s.Rational(7, 2)
    assert s.simplify(s.exp(s.I * phase[0]) + s.exp(s.I * phase[2])) == 0
    # Two incident costs are exactly 2 at a zero resultant. The remaining
    # three edge increments sum to pi modulo 2*pi and have cosine sum <= 3/2.
    t, v = s.symbols("t v", real=True)
    cosine_sum = s.cos(t + v) + s.cos(t - v) + s.cos(s.pi - 2 * t)
    assert s.trigsimp(cosine_sum - (2 * s.cos(t) * s.cos(v) - s.cos(2 * t))) == 0
    u, z = s.symbols("u z", nonnegative=True)
    deficit = s.Rational(3, 2) - (2 * u * (1 - z) - (2 * u**2 - 1))
    certificate = 2 * (u - s.Rational(1, 2)) ** 2 + 2 * u * z
    assert s.expand(deficit - certificate) == 0
    assert certificate.is_nonnegative
    twist_minimum = 5 * (1 - s.cos(2 * s.pi / 5))
    assert s.simplify(s.Rational(7, 2) - twist_minimum).is_positive


def test_regular_metric_survives_an_original_antipodal_edge_and_acute_cutoff(symbolic):
    s = symbolic
    displacement = 5 * s.pi / 8
    radius = 2 * s.cos(3 * s.pi / 8)
    metric = s.pi * radius * s.sin(displacement) / displacement
    assert s.simplify(metric - 4 * s.sqrt(2) / 5) == 0
    assert s.simplify(metric).is_positive
    # P2 at exactly pi/2 lies outside the strict acute execution contract,
    # while its conditional full-chart metric is positive and nonsingular.
    p2_metric = s.pi * s.sin(s.pi / 2) / (s.pi / 2)
    assert p2_metric == 2
    antipodal_edge_energy = 2 + 4 * (1 - s.cos(s.pi / 4))
    assert s.simplify(s.Rational(7, 2) - antipodal_edge_energy).is_positive
    assert s.simplify(s.pi * s.sin(s.pi) / s.pi) == 0


def _paired_cycles(delta, form=None):
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edges_from(((0, 5), (1, 6)))
    graph.graph.update(
        GAMMA={"type": "none"},
        DNFR_WEIGHTS={"epi": 0.5, "phase": 0.5, "vf": 0.0, "topo": 0.0},
    )
    phases = (0.0, -4 * delta, -3 * delta, -2 * delta, -delta)
    for i in graph:
        graph.nodes[i].update(
            EPI=0.0 if form is None else float(form[i % 5]),
            theta=phases[i % 5],
            nu_f=1.0,
        )
    return graph


def test_two_bridge_path_remains_regular_while_original_cycle_winding_changes(symbolic):
    s = symbolic
    delta = s.symbols("delta", real=True)
    port_real = 1 + s.cos(delta) + s.cos(4 * delta)
    assert s.trigsimp(port_real - s.cos(delta) - 2 * s.cos(2 * delta) ** 2) == 0
    assert s.cos(2 * s.pi / 5).is_positive
    symbolic_phases = (0, -4 * delta, -3 * delta, -2 * delta, -delta) * 2
    support = _paired_cycles(0.0)
    path_storage = sum(
        1 - s.cos(symbolic_phases[j] - symbolic_phases[i]) for i, j in support.edges
    )
    assert s.simplify(path_storage.subs(delta, s.pi / 3)) == 7
    assert s.simplify(s.diff(path_storage, delta, 2).subs(delta, s.pi / 3)) == -12
    # For 0 <= delta <= 2*pi/5 the ports have real part >= cos(delta) > 0;
    # all other relative resultants are 2*cos(delta) > 0. This is a geometric
    # path with supplied phases, not a solution of the complete nodal law.
    for parameter, expected in ((math.pi / 8, 0), (3 * math.pi / 10, 1)):
        graph = _paired_cycles(parameter)
        assert tuple(
            certify_phase_winding(graph, nodes).winding
            for nodes in (tuple(range(5)), tuple(range(5, 10)))
        ) == (expected, expected)
    endpoint = _paired_cycles(2 * math.pi / 5)
    field = evaluate_relational_exchange(endpoint, model=RelationalExchangeModel(1.0))
    assert max(map(abs, field.phase_source)) < 1e-15
    assert field.phase_rate == (0.0,) * 10
    assert tuple(
        certify_phase_winding(endpoint, nodes).winding
        for nodes in (tuple(range(5)), tuple(range(5, 10)))
    ) == (1, 1)


def test_actual_bridge_neighbors_remove_the_path_specific_pair_cancellation(symbolic):
    s = symbolic
    phases = (0, -4 * s.pi / 3, -s.pi, -2 * s.pi / 3, -s.pi / 3) * 2
    graph = _paired_cycles(math.pi / 3)

    def resultant(node):
        return s.simplify(
            sum(
                s.cos(phases[neighbor] - phases[node])
                + s.I * s.sin(phases[neighbor] - phases[node])
                for neighbor in graph[node]
            )
        )

    assert tuple(resultant(node) for node in graph) == (1,) * 10
    graph.remove_edge(1, 6)
    assert resultant(1) == resultant(6) == 0
    assert resultant(0) == resultant(5) == 1
    # This supplied path needs both interfaces at this point. It does not
    # exclude every possible path on a graph with only one bridge.


def test_two_bridge_crossing_tangent_satisfies_conditional_rows_and_native_pressure(
    symbolic,
):
    s = symbolic
    root = s.sqrt(2)
    laplacian = s.Matrix(
        (
            (2, -1, 0, 0, -1),
            (-1, 2, -1, 0, 0),
            (0, -1, 2, -1, 0),
            (0, 0, -1, 2, -1),
            (-1, 0, 0, -1, 2),
        )
    )
    q = s.Matrix((8 * root, -8 * root, -2 * s.pi * root, 0, 2 * s.pi * root))
    form = (laplacian + s.ones(5) / 5).inv() * q
    assert (laplacian * form - q).applyfunc(s.simplify) == s.zeros(5, 1)
    assert s.simplify(sum(form)) == 0
    graph = _paired_cycles(math.pi / 4, form)
    phases = (0, -s.pi, -3 * s.pi / 4, -s.pi / 2, -s.pi / 4) * 2
    metrics, sources = [], []
    for node in graph:
        real = s.simplify(sum(s.cos(phases[j] - phases[node]) for j in graph[node]))
        imag = s.simplify(sum(s.sin(phases[j] - phases[node]) for j in graph[node]))
        assert real.is_positive
        displacement = s.atan(imag / real)
        sinc = 1 if displacement == 0 else s.sin(displacement) / displacement
        metrics.append(s.simplify(s.pi * s.sqrt(real**2 + imag**2) * sinc))
        sources.append(s.simplify(displacement / s.pi))
    expected_metric = (2 * root, 2 * root, s.pi * root, s.pi * root, s.pi * root)
    expected_source = (-s.Rational(1, 4), s.Rational(1, 4), 0, 0, 0)
    assert tuple(metrics) == expected_metric * 2
    assert tuple(sources) == expected_source * 2
    metric = s.diag(*metrics[:5])
    full_form = tuple(form) * 2
    actual_q = s.Matrix(
        [sum(full_form[node] - full_form[j] for j in graph[node]) for node in graph]
    )
    assert (actual_q - q.col_join(q)).applyfunc(s.simplify) == s.zeros(10, 1)
    velocity = s.Matrix((0, -4, -3, -2, -1))
    common = s.simplify(-(s.ones(1, 5) * metric * velocity)[0] / sum(metric.diagonal()))
    assert common == 2
    phase_rate = s.Rational(1, 2) * metric.inv() * q
    assert phase_rate == velocity + common * s.ones(5, 1)
    assert phase_rate[1] - phase_rate[0] == -4
    source = s.Matrix(sources[:5])
    degrees = s.diag(3, 3, 2, 2, 2)
    pressure = -s.Rational(1, 2) * degrees.inv() * q + s.Rational(1, 2) * source
    gradient = -metric * source
    work = 2 * (q.dot(pressure) + gradient.dot(phase_rate))
    loss = (q.T * degrees.inv() * q)[0]
    assert s.simplify(work + loss) == 0
    assert s.simplify(loss - (s.Rational(256, 3) + 8 * s.pi**2)) == 0
    # The same form in both rings makes both bridge form differences zero.
    # Native pressure remains usable; the default relational executor
    # deliberately retains its stricter acute-edge admission.
    captured = capture_non_epi_forcing(graph)
    expected = tuple(map(float, pressure)) * 2
    assert tuple(map(float, captured.full_kernel_pressure)) == pytest.approx(
        expected, abs=1e-12
    )
    before = _snapshot(graph)
    with pytest.raises(ValueError, match="acute"):
        evaluate_relational_exchange(graph, model=RelationalExchangeModel(1.0))
    assert _snapshot(graph) == before


def test_reflected_two_ring_reduction_has_independent_laplacian_and_phasors(symbolic):
    s = symbolic
    a, b, amplitude, contrast = s.symbols("a b A B", real=True)
    graph = _paired_cycles(0.0)
    form = (amplitude, -amplitude, -contrast, 0, contrast) * 2
    phase = (a, -a, -b, 0, b) * 2
    q, r = 3 * amplitude - contrast, 2 * contrast - amplitude
    gradient = tuple(
        s.expand(sum(form[node] - form[j] for j in graph[node])) for node in graph
    )
    assert gradient == (q, -q, -r, 0, r) * 2
    real = tuple(
        s.trigsimp(sum(s.cos(phase[j] - phase[node]) for j in graph[node]))
        for node in graph
    )
    imag = tuple(
        s.trigsimp(sum(s.sin(phase[j] - phase[node]) for j in graph[node]))
        for node in graph
    )
    port_real = 1 + s.cos(2 * a) + s.cos(a - b)
    port_imag = -s.sin(2 * a) - s.sin(a - b)
    interior_real = 2 * s.cos(a / 2) * s.cos(a / 2 - b)
    interior_imag = 2 * s.cos(a / 2) * s.sin(a / 2 - b)
    expected_real = (port_real, port_real, interior_real, 2 * s.cos(b), interior_real)
    expected_imag = (port_imag, -port_imag, -interior_imag, 0, interior_imag)
    assert all(
        s.trigsimp(value - target) == 0
        for value, target in zip(real, expected_real * 2)
    )
    assert all(
        s.trigsimp(value - target) == 0
        for value, target in zip(imag, expected_imag * 2)
    )
    energy = sum((form[i] - form[j]) ** 2 / 2 for i, j in graph.edges)
    assert (
        s.expand(
            energy - (6 * amplitude**2 - 4 * amplitude * contrast + 4 * contrast**2)
        )
        == 0
    )


def test_actual_regular_field_preserves_reflected_four_coordinate_subspace():
    # This state is outside the acute edge chamber, but every local resultant
    # has positive real part. Only this field is evaluated; no step is taken.
    a, b, amplitude, contrast, center = 2.25, 1.0, 0.375, 0.625, 0.125
    graph = _paired_cycles(0.0, (amplitude, -amplitude, -contrast, 0.0, contrast))
    phase = (a, -a, -b, 0.0, b)
    for node in graph:
        graph.nodes[node]["theta"] = center + phase[node % 5]
    before = _snapshot(graph)
    model = RelationalExchangeModel(
        1.25, epi_weight=0.3, phase_weight=0.7, phase_domain="positive_resultant"
    )
    field = evaluate_relational_exchange(graph, model=model)
    assert _snapshot(graph) == before
    e, w = model.effective_weights
    q, r = 3 * amplitude - contrast, 2 * contrast - amplitude
    real = 1 + math.cos(2 * a) + math.cos(a - b)
    imag = -math.sin(2 * a) - math.sin(a - b)
    alpha = math.atan2(imag, real)
    displacement = a / 2 - b
    h0 = math.pi * math.hypot(real, imag) * math.sin(alpha) / alpha
    h4 = 2 * math.pi * math.cos(a / 2) * math.sin(displacement) / displacement
    adot = w * q / (model.storage_scale * h0)
    bdot = w * r / (model.storage_scale * h4)
    amplitude_dot = -e * q / 3 + w * alpha / math.pi
    contrast_dot = -e * r / 2 + w * displacement / math.pi
    assert field.form_gradient == pytest.approx((q, -q, -r, 0.0, r) * 2, abs=1e-14)
    assert field.phase_metric == pytest.approx(
        (h0, h0, h4, 2 * math.pi * math.cos(b), h4) * 2, abs=1e-13
    )
    assert field.phase_rate == pytest.approx(
        (adot, -adot, -bdot, 0.0, bdot) * 2, abs=1e-14
    )
    assert field.form_rate == pytest.approx(
        (amplitude_dot, -amplitude_dot, -contrast_dot, 0.0, contrast_dot) * 2, abs=1e-14
    )
    assert min(field.resultant_real_lower_bounds) > 0


def test_reflected_central_resultant_cancellation_has_sharp_phase_barrier(symbolic):
    s = symbolic
    a, b = s.symbols("a b", real=True)
    graph = _paired_cycles(0.0)
    phase = (a, -a, -b, 0, b) * 2
    energy = sum(1 - s.cos(phase[j] - phase[i]) for i, j in graph.edges)
    formula = 10 - 2 * s.cos(2 * a) - 4 * s.cos(a - b) - 4 * s.cos(b)
    assert s.trigsimp(energy - formula) == 0
    for direction in (-1, 1):
        boundary = energy.subs(b, direction * s.pi / 2)
        certificate = 7 + 4 * (s.sin(a) - direction * s.Rational(1, 2)) ** 2
        assert s.trigsimp(boundary - certificate) == 0
        assert (certificate - 7).is_nonnegative
    assert s.simplify(energy.subs({a: s.pi / 6, b: s.pi / 2})) == 7
    # This is an energy cost at a reflected central-node cancellation, not a
    # reachability proof, global regular-domain barrier or basin certificate.
    target = s.simplify(energy.subs({a: 4 * s.pi / 5, b: 2 * s.pi / 5}))
    assert s.simplify(7 - target).is_positive


def test_capture_rectangle_boundary_costs_follow_from_full_support(symbolic):
    s = symbolic
    a, b = s.symbols("a b", real=True)
    graph = _paired_cycles(0.0)
    phase = (a, -a, -b, 0, b) * 2
    energy = sum(1 - s.cos(phase[j] - phase[i]) for i, j in graph.edges)
    faces = (
        (energy.subs(a, 2 * s.pi / 3), 7 + 4 * (1 - s.cos(b - s.pi / 3))),
        (energy.subs(a, s.pi), s.Integer(8)),
        (energy.subs(b, 0), 8 - 4 * s.cos(a) * (s.cos(a) + 1)),
        (energy.subs(b, s.pi / 2), 7 + 4 * (s.sin(a) - s.Rational(1, 2)) ** 2),
    )
    assert all(s.trigsimp(actual - expected) == 0 for actual, expected in faces)
    # On 2*pi/3 <= a <= pi, -1 <= cos(a) <= -1/2. Therefore the
    # b=0 face is >=8; the remaining displayed face costs are >=7.
    # Re(z0) >= 2*cos(a)**2 + cos(a) > 0 in the open rectangle,
    # since 0 < a-b < a < pi. This polynomial factors with the required sign.
    cosine = s.symbols("cosine", real=True)
    assert s.factor(2 * cosine**2 + cosine) == cosine * (2 * cosine + 1)
    z0_real = 1 + s.cos(2 * a) + s.cos(a - b)
    assert s.trigsimp(z0_real - (2 * s.cos(a) ** 2 + s.cos(a - b))) == 0


def test_capture_rectangle_has_unique_critical_geometry_and_positive_target_hessian(
    symbolic,
):
    s = symbolic
    a, b, cosine = s.symbols("a b cosine", real=True)
    graph = _paired_cycles(0.0)
    phase = (a, -a, -b, 0, b) * 2
    energy = sum(1 - s.cos(phase[j] - phase[i]) for i, j in graph.edges)
    assert s.trigsimp(s.diff(energy, b) - 8 * s.cos(a / 2) * s.sin(b - a / 2)) == 0
    # Inside the rectangle, cos(a/2)>0 and -pi/2 < b-a/2 < pi/2,
    # so every critical point has b=a/2. The remaining condition is cubic
    # in cosine=cos(a/2), which lies strictly between 0 and 1/2.
    a_gradient = s.diff(energy, a).subs(b, a / 2)
    assert s.trigsimp(a_gradient - 8 * s.sin(5 * a / 4) * s.cos(3 * a / 4)) == 0
    polynomial = 8 * cosine**3 - 4 * cosine + 1
    assert (
        s.expand(polynomial - (2 * cosine - 1) * (4 * cosine**2 + 2 * cosine - 1)) == 0
    )
    target_cosine = (s.sqrt(5) - 1) / 4
    other_cosine = -(s.sqrt(5) + 1) / 4
    assert target_cosine.is_positive
    assert (s.Rational(1, 2) - target_cosine).is_positive
    assert other_cosine.is_negative
    assert s.simplify(polynomial.subs(cosine, target_cosine)) == 0
    assert s.simplify(s.cos(2 * s.pi / 5) - target_cosine) == 0
    hessian = s.hessian(energy, (a, b)).subs({a: 4 * s.pi / 5, b: 2 * s.pi / 5})
    expected = target_cosine * s.Matrix(((12, -4), (-4, 8)))
    assert (hessian - expected).applyfunc(s.simplify) == s.zeros(2)
    assert expected[0, 0].is_positive and s.simplify(expected.det()).is_positive


def test_subthreshold_storage_protects_both_regular_landscape_components(symbolic):
    s = symbolic
    a, b = s.symbols("a b", real=True)
    graph = _paired_cycles(0.0)
    phase = (a, -a, -b, 0, b) * 2
    energy = sum(1 - s.cos(phase[j] - phase[i]) for i, j in graph.edges)
    real = tuple(
        sum(s.cos(phase[j] - phase[node]) for j in graph[node]) for node in graph
    )
    common_lower = (8 - energy) / 4
    assert s.trigsimp(real[0] - common_lower - s.cos(a) ** 2 - (1 - s.cos(b))) == 0
    assert s.trigsimp(real[4] - common_lower - s.sin(a) ** 2) == 0
    assert s.trigsimp(real[3] - 2 * s.cos(b)) == 0
    # V<7 therefore gives Re(z0), Re(z4)>1/4 everywhere in the reflected
    # plane. The center is also positive for |b|<pi/2, including the central
    # consensus rectangle |a|<2*pi/3 and the adjacent twist rectangle.
    for direction in (-1, 1):
        assert (
            s.trigsimp(
                energy.subs(a, direction * 2 * s.pi / 3)
                - (7 + 4 * (1 - s.cos(b - direction * s.pi / 3)))
            )
            == 0
        )
        assert (
            s.trigsimp(
                energy.subs(b, direction * s.pi / 2)
                - (7 + 4 * (s.sin(a) - direction * s.Rational(1, 2)) ** 2)
            )
            == 0
        )
    # In the central rectangle, dV/db=0 again forces b=a/2. Factoring
    # dV/da there gives sin(a/2)*(2*cos(a/2)-1)*(4*cos(a/2)**2+2*cos(a/2)-1).
    # Since cos(a/2)>1/2, the latter two factors are positive; only a=b=0
    # remains. This critical point has V=0 and positive definite Hessian.
    assert s.simplify(energy.subs({a: 0, b: 0})) == 0
    hessian = s.hessian(energy, (a, b)).subs({a: 0, b: 0})
    assert hessian == s.Matrix(((12, -4), (-4, 8)))
    assert hessian[0, 0] > 0 and hessian.det() > 0


def test_saddle_linearization_has_the_symmetric_quadratic_pencil(symbolic):
    s = symbolic
    a, b, amplitude, contrast = s.symbols("a b A B", real=True)
    e, w, beta, rate = s.symbols("e w beta rate", positive=True)
    real = 1 + s.cos(2 * a) + s.cos(a - b)
    imag = -s.sin(2 * a) - s.sin(a - b)
    alpha = s.atan(imag / real)
    displacement = a / 2 - b
    h0 = s.pi * s.sqrt(real**2 + imag**2) * s.sinc(alpha)
    h4 = 2 * s.pi * s.cos(a / 2) * s.sinc(displacement)
    q, r = 3 * amplitude - contrast, 2 * contrast - amplitude
    flow = s.Matrix(
        (
            w * q / (beta * h0),
            w * r / (beta * h4),
            -e * q / 3 + w * alpha / s.pi,
            -e * r / 2 + w * displacement / s.pi,
        )
    )
    saddle = {a: 2 * s.pi / 3, b: s.pi / 3}
    assert s.simplify(h0.subs(saddle)) == s.pi
    assert s.simplify(h4.subs(saddle)) == s.pi
    # Substitute zero form first: phase derivatives of the metric multiply
    # q=r=0, so their apparent sinc derivatives at zero do not enter J.
    jacobian = (
        flow.jacobian((a, b, amplitude, contrast))
        .subs({amplitude: 0, contrast: 0})
        .subs(saddle)
        .applyfunc(s.simplify)
    )
    k = s.Matrix(((3, -1), (-1, 2)))
    g = s.Matrix(((s.Rational(1, 2), s.Rational(1, 2)), (s.Rational(1, 2), -1)))
    damping = s.diag(s.Rational(1, 3), s.Rational(1, 2))
    coupling = w / (beta * s.pi)
    expected = (s.zeros(2).row_join(coupling * k)).col_join(
        ((w / s.pi) * g).row_join(-e * damping * k)
    )
    assert jacobian == expected
    assert k[0, 0] > 0 and k.det() == 5
    assert g.det() == -s.Rational(3, 4)
    pencil = rate**2 * k.inv() + e * rate * damping - w**2 * g / (beta * s.pi**2)
    assert s.simplify((rate * s.eye(4) - jacobian).det() - k.det() * pencil.det()) == 0
    assert s.diff(pencil, rate) == 2 * rate * k.inv() + e * damping
    # Both terms in P'(rate) are positive definite for rate>=0 (e>0),
    # while P(0) is indefinite. The increasing eigenvalue crossing is unique.
    assert s.simplify(pencil.subs(rate, 0).det()).is_negative


def test_upper_corner_preparation_has_low_storage_and_inward_initial_b_acceleration(
    symbolic,
):
    s = symbolic
    epsilon = s.symbols("epsilon", positive=True)
    a = s.pi / 2 - epsilon
    graph = _paired_cycles(0.0)
    phase = (a, -a, -a, 0, a) * 2
    form = (
        s.Rational(2, 5),
        -s.Rational(2, 5),
        -s.Rational(1, 5),
        0,
        s.Rational(1, 5),
    ) * 2
    gradient = tuple(sum(form[node] - form[j] for j in graph[node]) for node in graph)
    assert gradient == (1, -1, 0, 0, 0) * 2
    form_energy = sum((form[i] - form[j]) ** 2 / 2 for i, j in graph.edges)
    phase_energy = sum(1 - s.cos(phase[j] - phase[i]) for i, j in graph.edges)
    assert form_energy == s.Rational(4, 5)
    expected_energy = s.Rational(44, 5) - 4 * s.sin(epsilon) - 4 * s.sin(epsilon) ** 2
    assert s.trigsimp(form_energy + phase_energy - expected_energy) == 0
    real = tuple(
        sum(s.cos(phase[j] - phase[node]) for j in graph[node]) for node in graph
    )
    imag = tuple(
        sum(s.sin(phase[j] - phase[node]) for j in graph[node]) for node in graph
    )
    assert s.trigsimp(real[0] - (2 - s.cos(2 * epsilon))) == 0
    assert s.trigsimp(imag[0] + s.sin(2 * epsilon)) == 0
    assert s.trigsimp(real[4] - (1 + s.sin(epsilon))) == 0
    assert s.trigsimp(imag[4] + s.cos(epsilon)) == 0
    assert s.trigsimp(real[3] - 2 * s.sin(epsilon)) == 0
    chi = s.atan(s.sin(2 * epsilon) / (2 - s.cos(2 * epsilon)))
    g0 = -chi / s.pi
    g4 = -s.Rational(1, 4) + epsilon / (2 * s.pi)
    amplitude_rate = -s.Rational(1, 6) + g0 / 2
    contrast_rate = g4 / 2
    q_rate = 3 * amplitude_rate - contrast_rate
    r_rate = 2 * contrast_rate - amplitude_rate
    assert s.simplify(q_rate + s.Rational(3, 8) + (6 * chi + epsilon) / (4 * s.pi)) == 0
    assert s.simplify(r_rate + s.Rational(1, 12) - (epsilon + chi) / (2 * s.pi)) == 0
    h0 = s.pi * s.sin(2 * epsilon) / chi
    h4 = 2 * s.pi * s.cos(epsilon) / a
    assert s.limit(q_rate, epsilon, 0, dir="+") == -s.Rational(3, 8)
    assert s.limit(r_rate, epsilon, 0, dir="+") == -s.Rational(1, 12)
    assert s.limit(1 / (2 * h0), epsilon, 0, dir="+") == 1 / (2 * s.pi)
    assert s.limit(r_rate / (2 * h4), epsilon, 0, dir="+") == -s.Rational(1, 96)
    # The limit is a boundary calculation, not an admitted singular initial
    # state or a trajectory. The actual declared preparation has epsilon=1/64.


def test_upper_corner_short_crossing_box_has_exact_rational_margin():
    epsilon, horizon = Q(1, 64), Q(1, 4)
    real_lower = 1 - Q(3, 32) ** 2 / 2
    imag_upper = Q(3, 64)
    assert real_lower == Q(2039, 2048)
    # atan(u)<=u and 3<pi<22/7 give the phase-source and metric bounds.
    assert imag_upper / real_lower / 3 < Q(1, 60)
    assert 3 * real_lower > Q(5, 2)
    assert Q(22, 7) * (1 + 2 * Q(1, 16) ** 2 + imag_upper) < Q(7, 2)
    g0_bound, g4_lower, g4_upper = Q(1, 60), -Q(1, 4), -Q(11, 48)
    q_rate_lower = -Q(1, 2) - Q(1, 32) + (-3 * g0_bound - g4_upper) / 2
    q_rate_upper = -Q(3, 8) + (3 * g0_bound - g4_lower) / 2
    r_rate_lower = Q(1, 8) + g4_lower - g0_bound / 2
    r_zero_face_upper = Q(1, 6) + g4_upper + g0_bound / 2
    assert q_rate_lower == -Q(53, 120) > -Q(1, 2)
    assert q_rate_upper == -Q(9, 40) < 0
    assert r_rate_lower == -Q(2, 15) > -Q(1, 6)
    assert r_zero_face_upper == -Q(13, 240) < 0
    assert 1 - horizon / 2 > Q(3, 4)
    assert -horizon / 6 > -Q(1, 8)
    assert horizon / 5 < Q(1, 16) + epsilon
    assert horizon**2 / 60 < epsilon
    crossing_bound = epsilon / Q(3, 28)
    assert crossing_bound == Q(7, 48) < horizon
    # Initial derivative signs at the fixed epsilon use 0<chi<2*epsilon.
    assert -Q(1, 12) + 3 * epsilon / 6 == -Q(29, 384) < 0


def test_acute_sector_target_and_bridge_square_supply_the_full_cycle_basis(symbolic):
    s = symbolic
    graph = _paired_cycles(0.0)
    edges = tuple(graph.edges)
    cycles = (tuple(range(5)), tuple(range(5, 10)), (0, 1, 6, 5))
    incidence = s.zeros(len(graph), len(edges))
    for column, (left, right) in enumerate(edges):
        incidence[left, column], incidence[right, column] = -1, 1
    cycle_rows = []
    reference = (
        s.Rational(4, 5),
        -s.Rational(4, 5),
        -s.Rational(2, 5),
        0,
        s.Rational(2, 5),
    ) * 2
    periods = []
    for cycle in cycles:
        row = [0] * len(edges)
        gaps = []
        for left, right in zip(cycle, cycle[1:] + cycle[:1]):
            column = next(
                index for index, edge in enumerate(edges) if set(edge) == {left, right}
            )
            row[column] = 1 if edges[column] == (left, right) else -1
            gap = (reference[right] - reference[left] + 1) % 2 - 1
            assert abs(gap) < s.Rational(1, 2)
            gaps.append(gap)
        cycle_rows.append(row)
        periods.append(sum(gaps) / 2)
    cycle_matrix = s.Matrix(cycle_rows)
    assert incidence * cycle_matrix.T == s.zeros(10, 3)
    assert cycle_matrix.rank() == len(edges) - len(graph) + 1 == 3
    assert tuple(periods) == (1, 1, 0)
    # A four-cycle with strictly acute edges has total absolute angle <2*pi,
    # so its integer period must be zero for every admitted state, not only
    # for this reference. The two ring periods then fix the lifted component.


def test_acute_cycle_boundary_energy_has_the_declared_exact_barrier(symbolic):
    s = symbolic
    cycle = nx.cycle_graph(5)
    boundary = (0, s.pi / 2, 7 * s.pi / 8, 5 * s.pi / 4, 13 * s.pi / 8)
    face_cost = sum(1 - s.cos(boundary[j] - boundary[i]) for i, j in cycle.edges)
    target = tuple(2 * s.pi * index / 5 for index in range(5))
    target_cost = sum(1 - s.cos(target[j] - target[i]) for i, j in cycle.edges)
    assert s.simplify(face_cost - (5 - 4 * s.cos(3 * s.pi / 8))) == 0
    assert s.simplify(target_cost - 5 * (1 - s.cos(2 * s.pi / 5))) == 0
    barrier = s.simplify(face_cost + target_cost)
    radical = (45 - 5 * s.sqrt(5)) / 4 - 2 * s.sqrt(2 - s.sqrt(2))
    assert s.simplify(barrier - radical) == 0
    # Jensen's four remaining acute gaps must sum to 3*pi/2 when one gap
    # equals +pi/2. The negative face would require an impossible 5*pi/2.
    assert 2 * s.pi + s.pi / 2 > 4 * s.pi / 2
    assert s.pi / 2 + 4 * (3 * s.pi / 8) == 2 * s.pi
    # The bridge-face lower bound exceeds the ring-face bound: cos(3*pi/8)
    # is larger than cos(2*pi/5), and 1-cos(2*pi/5)>0.
    difference = 1 + target_cost - face_cost
    remainder = 4 * (s.cos(3 * s.pi / 8) - s.cos(2 * s.pi / 5))
    assert s.simplify(difference - (1 - s.cos(2 * s.pi / 5)) - remainder) == 0
    assert 0 < s.Rational(3, 8) < s.Rational(2, 5) < 1


def test_acute_sector_barrier_rational_lower_bound_uses_exact_radicals():
    twist_cosine_upper = Q(309017, 1000000)
    face_cosine_upper = Q(382684, 1000000)
    # cos(2*pi/5)=(sqrt(5)-1)/4. Squaring positive quantities proves
    # sqrt(5)<4*upper+1, independently of any floating cosine evaluation.
    assert (4 * twist_cosine_upper + 1) ** 2 > 5
    # cos(3*pi/8)=sqrt(2-sqrt(2))/2. The following positive rational lies
    # below sqrt(2), hence the chosen squared cosine bound is strictly above.
    lower_for_root_two = 2 - 4 * face_cosine_upper**2
    assert lower_for_root_two > 0 and lower_for_root_two**2 < 2
    barrier_lower = 10 - 5 * twist_cosine_upper - 4 * face_cosine_upper
    assert barrier_lower == Q(6924179, 1000000)
    # This is a sufficient support-wide threshold. The two bridges can keep
    # the simultaneous one-ring Jensen equalities from being attained.


def test_sector_deficit_has_the_exact_jensen_slope_bound(symbolic):
    s = symbolic
    t = s.symbols("t", real=True)
    profile = 1 - s.cos(t) + 4 * (1 - s.cos((2 * s.pi - t) / 4))
    derivative = s.diff(profile, t)
    assert s.simplify(derivative - (s.sin(t) - s.sin((2 * s.pi - t) / 4))) == 0
    assert s.simplify(derivative.subs(t, 2 * s.pi / 5)) == 0
    assert s.simplify(derivative.subs(t, s.pi / 2) - (1 - s.sin(3 * s.pi / 8))) == 0
    curvature = s.diff(profile, t, 2)
    assert s.simplify(curvature - s.cos(t) - s.cos((2 * s.pi - t) / 4) / 4) == 0
    # Both cosines are nonnegative on [2*pi/5,pi/2], with the second
    # positive. The boundary derivative is therefore the strict upper bound.
    # sin(3*pi/8)>12/13 follows from sqrt(2)>238/169 exactly.
    root_two_lower = 4 * Q(12, 13) ** 2 - 2
    assert root_two_lower == Q(238, 169)
    assert 0 < root_two_lower and root_two_lower**2 < 2
    # Thus d=integral F' < pi/130; eta<=d gives 13eta<pi/10.
    assert 13 * Q(1, 130) == Q(1, 10) < Q(1, 6)


def test_reduced_flux_coordinates_reuse_full_support_storage_and_loss(symbolic):
    s = symbolic
    q, r = s.symbols("q r", real=True)
    A, B = (2 * q + r) / 5, (q + 3 * r) / 5
    form = s.Matrix([A, -A, -B, 0, B] * 2)
    graph = _paired_cycles(0.0)
    storage = sum((form[j] - form[i]) ** 2 / 2 for i, j in graph.edges)
    assert s.simplify(storage - (4 * q**2 + 4 * q * r + 6 * r**2) / 5) == 0
    flux = [sum(form[i] - form[j] for j in graph[i]) for i in graph]
    loss = sum(flux[i] ** 2 / graph.degree(i) for i in graph)
    assert s.simplify(loss - (4 * q**2 / 3 + 2 * r**2)) == 0
    assert s.simplify(3 * A - B - q) == s.simplify(2 * B - A - r) == 0
