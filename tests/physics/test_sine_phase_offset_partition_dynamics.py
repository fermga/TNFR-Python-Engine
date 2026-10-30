"""Independent fine-row and storage checks for compatible phase geometry.

These are algebraic evaluations of supplied families, never trajectories or
evidence that a source has formed, approached or entered such a family.
"""

from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, cos, pi_interval
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_partition import assess_sine_phase_offset_partition


def _source(graph):
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    return bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )


def _contains(interval, value):
    assert interval.lo <= value <= interval.hi


def _phase_rows(graph, values):
    return tuple(
        sum((values[node] - values[neighbor] for neighbor in graph[node]), Q(0))
        / graph.degree[node]
        for node in graph
    )


def _quarter_sine(turns):
    quarter = 4 * (turns % 1)
    assert quarter.denominator == 1
    return (Q(0), Q(1), Q(0), Q(-1))[int(quarter)]


def _quarter_form_rows(graph, turns):
    return tuple(
        sum(
            (_quarter_sine(turns[neighbor] - turns[node]) for neighbor in graph[node]),
            Q(0),
        )
        / graph.degree[node]
        for node in graph
    )


def test_moving_c5_identity_uses_live_environment_and_full_energy_balance():
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    graph.add_edges_from((i, (i + 1) % 5) for i in range(5))
    graph.add_edges_from((i, i + 5) for i in range(5))
    source = _source(graph)
    offsets = tuple(Q(i, 5) for i in range(5)) * 2
    family = assess_sine_phase_offset_partition(
        source,
        blocks=(tuple(range(5)), tuple(range(5, 10))),
        phase_offset_turns=offsets,
    )
    assert family.invariance_certified
    state = family.evaluate(
        collective_form=(2, -1), collective_phase_turns=(0, Q(1, 4))
    )

    # Two internal sine currents cancel. The retained private contact adds
    # +1 to each receiver current and -1 to each environmental current.
    expected_form = (Q(1, 3),) * 5 + (Q(-1),) * 5
    expected_phase = _phase_rows(graph, state.fine_form)
    assert expected_phase == (1,) * 5 + (-3,) * 5
    assert state.full_phase_rates == expected_phase
    for bound, rate in zip(state.full_form_rates, expected_form):
        _contains(bound, rate)
    assert state.block_phase_rates == (1, -3)
    for bound, rate in zip(state.block_form_rates, (Q(1, 3), Q(-1))):
        _contains(bound, rate)

    # Reconstruct the full quadratic storage directly from all graph edges.
    form_storage = sum(
        ((state.fine_form[i] - state.fine_form[j]) ** 2 / 2 for i, j in graph.edges),
        Q(0),
    )
    assert state.full_form_storage == state.cross_form_storage == form_storage
    assert form_storage == Q(45, 2)
    assert state.internal_form_storage == (0, 0)
    v5 = 5 * (I(1) - cos(2 * pi_interval() / 5))
    assert state.internal_phase_storage_bounds[0].lo <= v5.hi
    assert state.internal_phase_storage_bounds[0].hi >= v5.lo
    _contains(state.internal_phase_storage_bounds[1], 0)
    _contains(state.cross_phase_storage_bounds, 5)

    # The receiver has zero internal power despite nonzero full rows. The
    # cross-edge form and phase powers are opposite, and both are nonzero.
    form_power = sum(
        (state.fine_form[i] - state.fine_form[j])
        * (expected_form[i] - expected_form[j])
        for i, j in graph.edges
    )
    phase_power = sum(expected_phase[i + 5] - expected_phase[i] for i in range(5))
    assert form_power == 20 and phase_power == -20
    assert sum(graph.degree[i] * expected_form[i] for i in graph) == 0
    assert sum(graph.degree[i] * expected_phase[i] for i in graph) == 0
    assert all(expected_phase[(i + 1) % 5] - expected_phase[i] == 0 for i in range(5))
    # An offset-family assessment does not relabel the captured flat state
    # as an already acquired winding state.
    assert source.phase == (0,) * 10 and source.epi == (0,) * 10


def test_complex_contact_moment_is_retained_in_both_reconstructed_rows():
    graph = nx.Graph()
    graph.add_nodes_from(range(8))
    graph.add_edges_from(
        (start + i, start + (i + 1) % 4) for start in (0, 4) for i in range(4)
    )
    graph.add_edges_from((i, i + 4) for i in range(4))
    offsets = tuple(Q(i, 4) for i in range(4)) + tuple(Q(i + 1, 4) for i in range(4))
    family = assess_sine_phase_offset_partition(
        _source(graph), blocks=((0, 1, 2, 3), (4, 5, 6, 7)), phase_offset_turns=offsets
    )
    assert family.invariance_certified
    state = family.evaluate(collective_form=(2, -1), collective_phase_turns=(0, 0))
    fine_turns = dict(enumerate(offsets))
    expected_form = _quarter_form_rows(graph, fine_turns)
    expected_phase = _phase_rows(graph, state.fine_form)
    assert expected_form == (Q(1, 3),) * 4 + (Q(-1, 3),) * 4
    assert expected_phase == (1,) * 4 + (-1,) * 4
    for interval, rate in zip(state.full_form_rates, expected_form):
        _contains(interval, rate)
    assert state.full_phase_rates == expected_phase
    # All phases have zero collective displacement here, but the imaginary
    # contact moment still causes form exchange. A real bare graph quotient
    # would incorrectly return zero form rates.
    assert all(interval.lo > 0 for interval in state.block_form_rates[:1])
    assert state.full_form_storage == 18
    _contains(state.full_phase_storage_bounds, 12)
    _contains(state.full_storage_bounds, 30)
    form_power = sum(
        (state.fine_form[i] - state.fine_form[j])
        * (expected_form[i] - expected_form[j])
        for i, j in graph.edges
    )
    phase_power = sum(
        _quarter_sine(fine_turns[j] - fine_turns[i])
        * (expected_phase[j] - expected_phase[i])
        for i, j in graph.edges
    )
    assert form_power == 8 and phase_power == -8


def test_internal_cosine_equality_is_not_a_condition_for_the_consumed_rows():
    graph = nx.Graph()
    graph.add_nodes_from(range(6))
    graph.add_edges_from(
        ((0, 1), (1, 2), (3, 4), (3, 5), (0, 3), (1, 3), (1, 4), (2, 5))
    )
    offsets = (Q(0), Q(0), Q(1, 2), Q(0), Q(0), Q(1, 2))
    family = assess_sine_phase_offset_partition(
        _source(graph), blocks=((0, 1, 2), (3, 4, 5)), phase_offset_turns=offsets
    )
    assert family.invariance_certified
    state = family.evaluate(
        collective_form=(2, -1), collective_phase_turns=(0, Q(1, 4))
    )
    turns = dict(
        enumerate(offsets[:3] + tuple(value + Q(1, 4) for value in offsets[3:]))
    )
    expected_form = _quarter_form_rows(graph, turns)
    assert expected_form == (Q(1, 2),) * 3 + (Q(-1, 2),) * 3
    for interval, rate in zip(state.full_form_rates, expected_form):
        _contains(interval, rate)
    assert state.full_phase_rates == _phase_rows(graph, state.fine_form)
    # Internal cosine means differ although every internal sine is zero.
    cosine = lambda turn: (Q(1), Q(0), Q(-1), Q(0))[int(4 * (turn % 1))]
    internal_real = tuple(
        sum(
            (cosine(offsets[j] - offsets[i]) for j in graph[i] if (j < 3) == (i < 3)),
            Q(0),
        )
        / graph.degree[i]
        for i in graph
    )
    assert internal_real == (Q(1, 2), 0, Q(-1, 2), 0, Q(1, 2), Q(-1, 2))


def test_matching_neighbor_counts_do_not_close_the_nonlinear_form_row():
    graph = nx.Graph()
    graph.add_nodes_from(range(3))
    graph.add_edges_from(((0, 2), (1, 2)))
    turns = (Q(0), Q(1, 4), Q(0))
    family = assess_sine_phase_offset_partition(
        _source(graph), blocks=((0, 1), (2,)), phase_offset_turns=turns
    )
    assert family.status == "excluded" and not family.invariance_certified
    # The phase rows agree for uniform receiver form. Nevertheless even the
    # chosen zero collective phases give different receiver form rates.
    assert _phase_rows(graph, (1, 1, 0))[:2] == (1, 1)
    assert _quarter_form_rows(graph, dict(enumerate(turns)))[:2] == (0, -1)


def test_matching_complex_moments_do_not_close_the_linear_phase_row():
    graph = nx.Graph()
    graph.add_nodes_from(range(6))
    graph.add_edges_from(((0, 1), (2, 3), (0, 4), (0, 5), (2, 4), (2, 5)))
    turns = (Q(0), Q(0), Q(1, 2), Q(1, 2), Q(0), Q(1, 2))
    family = assess_sine_phase_offset_partition(
        _source(graph), blocks=((0, 1, 2, 3), (4, 5)), phase_offset_turns=turns
    )
    assert family.status == "excluded" and not family.invariance_certified
    # Every cross-block complex moment vanishes by opposite phase pairs;
    # internal currents vanish too. Different retained boundary fractions
    # still produce unequal phase velocities for nonzero collective form.
    assert _quarter_form_rows(graph, dict(enumerate(turns))) == (0,) * 6
    assert _phase_rows(graph, (1, 1, 1, 1, 0, 0))[:4] == (Q(2, 3), 0, Q(2, 3), 0)


def test_unrecognized_exact_root_of_unity_cancellation_remains_unavailable():
    graph = nx.complete_bipartite_graph(4, 3)
    turns = (Q(0), Q(1, 4), Q(1, 2), Q(3, 4), Q(0), Q(1, 3), Q(2, 3))
    family = assess_sine_phase_offset_partition(
        _source(graph), blocks=((0, 1, 2, 3), (4, 5, 6)), phase_offset_turns=turns
    )
    # Both blocks contain all roots of unity of their respective orders,
    # so every cross moment is zero. Oddness/reflection alone does not
    # establish the remaining cubic-root identity
    # 2*sin(pi/6)-sin(pi/2)=0; an interval overlap is not failure.
    assert family.status == "unavailable" and not family.invariance_certified
    with pytest.raises(ValueError):
        family.evaluate(collective_form=(0, 0), collective_phase_turns=(0, 0))
