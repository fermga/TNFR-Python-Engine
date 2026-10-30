"""Independent full-node controls for finite conservative regional winding.

No trajectory is evaluated. The global bounded sine row gives a continuous
Taylor enclosure, and one integer branch choice holds on the entire window.
"""

from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._phase_resultant_chamber import certified_cosine_bounds
from tnfr.mathematics._rational_interval import I, cos, pi_interval
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_entry import (
    analyze_sine_conservative_phase_transport,
    certify_sine_conservative_winding_entry,
)
from tnfr.physics.relational_sine_symmetry import assess_sine_cycle_symmetry

RECEIVER = tuple(range(5, 10))
WINDOW = (Q(1, 4), Q(1, 3))


def _preparation(*, return_edge=True, contrast=24):
    graph = nx.Graph()
    graph.add_nodes_from(range(11))
    graph.add_edges_from(
        [(offset + i, offset + (i + 1) % 5) for offset in (0, 5) for i in range(5)]
        + [(0, 10), (10, 5)]
        + ([(1, 6)] if return_edge else [])
    )
    graph.graph["GAMMA"] = {"type": "none"}
    for node in graph:
        graph.nodes[node].update(
            EPI=contrast if node == 10 else -contrast if node == 1 else 0,
            theta=0,
            nu_f=1,
        )
    source = bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )
    return graph, source


@pytest.fixture(scope="module")
def preparation():
    return _preparation()


@pytest.fixture(scope="module")
def report(preparation):
    _, source = preparation
    return certify_sine_conservative_winding_entry(
        source,
        cycle=RECEIVER,
        scaled_window=WINDOW,
        edge_turn_offsets=(1, 0, 0, 0, 0),
    )


def _full_initial_velocity(graph, source):
    # Independent complete graph rows; the receiver port degrees include
    # both boundary edges, and the environmental forms are retained.
    form = dict(zip(source.nodes, source.epi))
    return {
        node: sum((form[node] - form[neighbor] for neighbor in graph[node]), Q(0))
        / graph.degree[node]
        for node in graph
    }


def test_initial_jet_and_storage_use_every_node_and_boundary_edge(preparation, report):
    graph, source = preparation
    velocity = _full_initial_velocity(graph, source)
    assert (
        tuple(velocity[node] for node in source.nodes) == report.initial_phase_velocity
    )
    assert tuple(velocity[node] for node in RECEIVER) == (-8, 8, 0, 0, 0)
    assert sum(graph.degree[node] * velocity[node] for node in graph) == 0
    form = dict(zip(source.nodes, source.epi))
    energy = sum(((form[j] - form[i]) ** 2 / 2 for i, j in graph.edges), Q(0))
    assert report.initial_total_storage == energy == 1440
    assert all(form[node] == 0 and source.phase[node] == 0 for node in RECEIVER)


def test_global_sine_bound_controls_acceleration_and_continuous_remainder(
    preparation, report
):
    graph, source = preparation
    # Each normalized sine current is a convex average of values in [-1,1].
    # Every complete KL row has absolute coefficient sum two, so |theta''|
    # is at most two for every state, independently of the environmental form.
    for node in graph:
        row = tuple(
            (
                Q(1)
                if other == node
                else -Q(1, graph.degree[node]) if graph.has_edge(node, other) else Q(0)
            )
            for other in source.nodes
        )
        assert sum(row) == 0
        assert sum(map(abs, row)) == 2
    end = WINDOW[1]
    acceleration_bound = Q(2)
    integrated_phase_error = acceleration_bound * end**2 / 2
    assert report.node_form_remainder_bound == end
    assert report.node_phase_remainder_bound == integrated_phase_error
    assert report.cycle_gap_remainder_bound == 2 * integrated_phase_error


def test_whole_window_branches_prove_winding_without_sampling_a_trajectory(
    preparation, report
):
    graph, source = preparation
    velocity = _full_initial_velocity(graph, source)
    start, end = WINDOW
    remainder = 2 * end**2
    expected, slopes = [], []
    for i, j in zip(RECEIVER, RECEIVER[1:] + RECEIVER[:1]):
        slope = velocity[j] - velocity[i]
        slopes.append(slope)
        endpoints = (slope * start, slope * end)
        expected.append(I(min(endpoints) - remainder, max(endpoints) + remainder))
    assert report.cycle_raw_gap_bounds == tuple(expected)
    assert sum(slopes) == 0
    pi = pi_interval()
    first, *remaining = expected
    assert first.lo > pi.hi and first.hi < 3 * pi.lo
    assert all(value.lo > -pi.lo and value.hi < pi.lo for value in remaining)
    # Continuous real cycle increments telescope to zero exactly. On this
    # entire window, exactly one edge requires subtracting 2*pi, yielding -1.
    offsets = (1,) + (0,) * len(remaining)
    assert report.edge_turn_offsets == offsets
    assert report.initial_winding == 0
    assert report.declared_winding == report.certified_winding == -sum(offsets)
    assert report.acquisition_certified
    assert report.status == "certified_finite_acquisition"
    assert report.branch_margin_lower_bound > 0
    assert report.scaled_window == WINDOW
    assert report.original_time_window_bounds == tuple(pi * t for t in WINDOW)


def test_finite_sector_does_not_certify_an_acute_pattern(report):
    pi = pi_interval()
    # Two complete edge intervals lie below -pi/2 even after enclosure.
    # Nonzero winding and finite branch retention are weaker than acuteness.
    for index in (1, 4):
        assert report.cycle_principal_gap_bounds[index].hi < -pi.hi / 2
    assert report.scaled_window[1] - report.scaled_window[0] == Q(1, 12)


def test_adjacent_port_structure_imposes_early_acute_winding_obstruction(
    preparation, report
):
    graph, _ = preparation
    environment = tuple(node for node in graph if node not in RECEIVER)
    # A common form shift sets the uniform initial receiver form to zero.
    # These complete KL rows retain arbitrary environmental amplitudes;
    # equality of two coefficient rows therefore means their initial gap
    # velocity is zero for every environmental preparation in this class.
    rows = {
        node: tuple(
            -Q(1, graph.degree[node]) if graph.has_edge(node, other) else Q(0)
            for other in environment
        )
        for node in RECEIVER
    }
    receiver_edges = tuple(zip(RECEIVER, RECEIVER[1:] + RECEIVER[:1]))
    slow_edges = tuple((i, j) for i, j in receiver_edges if rows[i] == rows[j])
    assert slow_edges == ((7, 8), (8, 9))
    slow_count = len(slow_edges)
    fast_count = len(receiver_edges) - slow_count
    assert fast_count > 0
    # The two slow increments share node8. Sum them before bounding: the
    # full KL row difference for theta9-theta7 cancels that shared coordinate.
    complete_rows = {
        node: tuple(
            (
                Q(1)
                if other == node
                else -Q(1, graph.degree[node]) if graph.has_edge(node, other) else Q(0)
            )
            for other in graph
        )
        for node in (7, 9)
    }
    path_row = tuple(b - a for a, b in zip(complete_rows[7], complete_rows[9]))
    path_acceleration_bound = sum(map(abs, path_row))
    assert path_acceleration_bound == 3
    path_remainder_coefficient = path_acceleration_bound / 2
    # If all principal gaps were acute, the three remaining gaps contribute
    # strictly less than 3*pi/2. The correlated inner path contributes at
    # most 3*tau^2/2. Unit winding still requires absolute sum at least 2*pi.
    threshold_over_pi = (Q(2) - Q(fast_count, 2)) / path_remainder_coefficient
    assert threshold_over_pi == Q(1, 3)
    assert Q(fast_count, 2) + path_remainder_coefficient * threshold_over_pi == 2
    # At and below that limit the slow raw increments are already inside
    # the principal branch; hidden wrap offsets cannot evade the inequality.
    assert 2 * threshold_over_pi < 1
    pi = pi_interval()
    assert report.scaled_window[1] ** 2 < threshold_over_pi * pi.lo


def test_full_nonlinear_initial_jet_starts_internal_phase_redistribution(preparation):
    graph, source = preparation
    velocity = _full_initial_velocity(graph, source)
    # At exactly equal phases, S=0 and cos(theta_j-theta_i)=1. Differentiate
    # the full neighbor sine rows directly, before applying the phase row.
    form_second = {
        node: sum((velocity[j] - velocity[node] for j in graph[node]), Q(0))
        / graph.degree[node]
        for node in graph
    }
    phase_third = {
        node: sum((form_second[node] - form_second[j] for j in graph[node]), Q(0))
        / graph.degree[node]
        for node in graph
    }
    contrasts = tuple(
        tuple(Q(node == j) - Q(node == i) for node in source.nodes)
        for i, j in ((7, 8), (8, 9), (7, 9))
    )
    transport = analyze_sine_conservative_phase_transport(source, contrasts=contrasts)
    assert transport.initial_phase_velocity == tuple(velocity[i] for i in source.nodes)
    assert transport.initial_phase_acceleration == (0,) * len(source.nodes)
    assert transport.initial_form_acceleration == tuple(
        form_second[i] for i in source.nodes
    )
    assert transport.initial_phase_jerk == tuple(phase_third[i] for i in source.nodes)
    h = source.epi[10]
    assert tuple(phase_third[i] for i in (7, 8, 9)) == (5 * h / 9, 0, -5 * h / 9)
    assert transport.contrast_initial_velocity == (0, 0, 0)
    assert transport.contrast_initial_jerk == (-5 * h / 9, -5 * h / 9, -10 * h / 9)
    assert transport.contrast_acceleration_bounds[-1] == 3
    assert transport.contrast_quadratic_remainder_coefficients[-1] == Q(3, 2)
    assert transport.clock == "tau=t/pi"
    # Nonzero cubic coefficients rule out permanently flat inner nodes.
    # They are not a Taylor-error certificate at a later entry time.


def test_sharper_initial_endpoint_requires_returned_work_before_low_storage_organization(
    preparation,
):
    graph, source = preparation
    velocity = _full_initial_velocity(graph, source)
    at = WINDOW[0]
    remainder = 2 * at**2
    endpoint_gaps = tuple(
        I(
            at * (velocity[j] - velocity[i]) - remainder,
            at * (velocity[j] - velocity[i]) + remainder,
        )
        for i, j in zip(RECEIVER, RECEIVER[1:] + RECEIVER[:1])
    )
    pi = pi_interval()
    first = endpoint_gaps[0]
    side = endpoint_gaps[1]
    assert first.lo > pi.hi and first.hi < 2 * pi.lo
    assert -side.hi > pi.hi / 2 and -side.lo < pi.lo
    assert endpoint_gaps[4] == side
    # Cosine increases on (pi,2*pi) and decreases on (0,pi). Its certified
    # endpoint bounds give a lower potential bound over whole gap intervals.
    _, first_cosine_upper = certified_cosine_bounds(first.hi, terms=32, bits=128)
    _, side_cosine_upper = certified_cosine_bounds(-side.hi, terms=32, bits=128)
    regional_storage_lower = 1 - first_cosine_upper + 2 * (1 - side_cosine_upper)
    assert regional_storage_lower > 4

    edge_count = graph.subgraph(RECEIVER).number_of_edges()
    # In a unit-winding C5 acute cell, Jensen's inequality gives the equal-gap
    # minimum. On a face, one gap is pi/2 and the other four sum to 3*pi/2;
    # convexity gives their minimum at 3*pi/8. A negative face cannot close.
    uniform_angle_over_pi = Q(2, edge_count)
    remaining_face_mean_over_pi = (Q(2) - Q(1, 2)) / (edge_count - 1)
    assert -Q(1, 2) + Q(edge_count - 1, 2) < 2
    minimum = edge_count * (1 - cos(uniform_angle_over_pi * pi))
    barrier = 1 + (edge_count - 1) * (1 - cos(remaining_face_mean_over_pi * pi))
    assert minimum.hi < barrier.lo < barrier.hi < Q(7, 2)
    # Conditional on a later E_R strictly below this barrier, the exact
    # regional balance requires the intervening signed boundary work to be
    # less than -1/2. This does not assert that such an endpoint is reached.
    necessary_returned_work_lower = regional_storage_lower - barrier.hi
    assert necessary_returned_work_lower > Q(1, 2)


def test_acute_winding_does_not_by_itself_impose_low_regional_storage(preparation):
    graph, _ = preparation
    turns = {node: -Q(i, len(RECEIVER)) for i, node in enumerate(RECEIVER)}
    form = {node: Q(4 if node == 7 else 0) for node in RECEIVER}
    gaps = tuple(
        (turns[j] - turns[i] + Q(1, 2)) % 1 - Q(1, 2)
        for i, j in zip(RECEIVER, RECEIVER[1:] + RECEIVER[:1])
    )
    assert all(abs(gap) < Q(1, 4) for gap in gaps)
    assert sum(gaps) == -1
    internal_form_storage = sum(
        ((form[j] - form[i]) ** 2 / 2 for i, j in graph.subgraph(RECEIVER).edges), Q(0)
    )
    assert internal_form_storage > Q(7, 2)
    # Nonnegative phase storage only raises this energy. The returned-work
    # bound for E_R<B cannot be reused to exclude this acute phase geometry.


def test_same_support_zero_environment_is_exactly_stationary_and_has_no_local_seed(
    preparation,
):
    graph, source = preparation
    quiet_graph, quiet = _preparation(contrast=0)
    assert set(quiet_graph.edges) == set(graph.edges)
    assert quiet.degrees == source.degrees
    assert all(quiet.phase[i] == source.phase[i] == 0 for i in range(11))
    assert all(quiet.epi[i] == source.epi[i] == 0 for i in RECEIVER)
    assert set(_full_initial_velocity(quiet_graph, quiet).values()) == {0}
    # Every sine difference and every form difference is exactly zero;
    # both complete autonomous rows vanish, hence uniqueness keeps this state.
    for i, j in quiet_graph.edges:
        assert quiet.epi[i] - quiet.epi[j] == quiet.phase[i] - quiet.phase[j] == 0
    assert quiet.form_storage == 0


def test_zero_receiver_storage_grows_from_boundary_exchange_at_second_order(
    preparation,
):
    graph, source = preparation
    velocity = _full_initial_velocity(graph, source)
    internal_edges = tuple(graph.subgraph(RECEIVER).edges)
    # Flat initial phases make every x' vanish. Differentiating the internal
    # phase potential twice therefore supplies the entire receiver E_R''(0).
    receiver_second_derivative = sum(
        ((velocity[j] - velocity[i]) ** 2 for i, j in internal_edges), Q(0)
    )
    assert receiver_second_derivative == Q(2, 3) * source.epi[10] ** 2 > 0
    receiver_initial_storage = sum(
        ((source.epi[j] - source.epi[i]) ** 2 / 2 for i, j in internal_edges), Q(0)
    )
    assert receiver_initial_storage == 0


def test_single_port_receiver_reflection_excludes_winding_for_arbitrary_environment():
    graph, source = _preparation(return_edge=False)
    permutation = tuple({6: 9, 9: 6, 7: 8, 8: 7}.get(i, i) for i in range(11))
    original = {frozenset(edge) for edge in graph.edges}
    mapped = {frozenset((permutation[i], permutation[j])) for i, j in graph.edges}
    assert mapped == original
    assert tuple(source.epi[i] for i in permutation) == source.epi
    assert tuple(source.phase[i] for i in permutation) == source.phase
    assert all(permutation[i] == i for i in (0, 1, 2, 3, 4, 10))
    obstruction = assess_sine_cycle_symmetry(
        source, permutation_indices=permutation, cycle=RECEIVER
    )
    assert obstruction.trajectory_symmetry_certified
    assert obstruction.zero_winding_when_nonantipodal
    assert obstruction.cycle_orientation_reversed
    # The return edge destroys this full-support symmetry, despite leaving
    # the receiver's initial form and phase reflection-fixed.
    _, with_return = _preparation()
    changed = assess_sine_cycle_symmetry(
        with_return, permutation_indices=permutation, cycle=RECEIVER
    )
    assert not changed.support_automorphism
    assert not changed.zero_winding_when_nonantipodal
