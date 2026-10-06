"""Admission and exact-coordinate controls for the conservative bridge memory."""

from dataclasses import replace
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._exact_linear_algebra import exact_matrix_product as product
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_bridge_memory import assess_sine_bridge_memory
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange

LEFT, RIGHT = tuple(range(6)), tuple(range(6, 12))


def _source(*, beta=1, capacity=1, loss=0, reverse_order=False, extra_edge=False):
    edges = list(nx.cycle_graph(6).edges)
    edges += [(i + 6, j + 6) for i, j in edges]
    graph = nx.Graph()
    graph.add_nodes_from(reversed(range(12)) if reverse_order else range(12))
    graph.add_edges_from(edges + [(0, 6)])
    if extra_edge:
        graph.add_edge(0, 2)
    graph.graph["GAMMA"] = {"type": "none"}
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=capacity)
    return bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            beta, epi_weight=loss, phase_weight=1, phase_domain="regular"
        ),
    )


def _memory(source, **changes):
    arguments = dict(
        left_cycle=LEFT,
        right_cycle=RIGHT,
        target_phase_turns=tuple(Q(node % 6, 6) for node in source.nodes),
    )
    arguments.update(changes)
    return assess_sine_bridge_memory(source, **arguments)


@pytest.fixture(scope="module")
def source():
    return _source()


@pytest.fixture(scope="module")
def report(source):
    return _memory(source)


def test_shell_chart_is_an_exact_section_and_full_linear_observation(report):
    p, r = report.projection_rows, report.nodal_lift
    j, full = report.coordinate_generator, report.full_tangent_generator
    identity = tuple(tuple(Q(int(i == k)) for k in range(8)) for i in range(8))
    assert product(p, r) == identity
    assert product(p, full) == product(j, p)
    assert product(full, r) == product(r, j)
    assert report.visible_observation.dimension == 8
    assert report.visible_observation.extra_coordinates == 6
    assert report.coordinate_memory.visible_indices == (0, 4)
    assert report.coordinate_memory.hidden_indices == (1, 2, 3, 5, 6, 7)


def test_generator_and_metric_retain_both_full_nodal_channels(report):
    a = (
        (Q(2, 3), Q(-2, 3), 0, 0),
        (Q(-2, 3), Q(7, 6), Q(-1, 2), 0),
        (0, Q(-1, 2), 1, Q(-1, 2)),
        (0, 0, Q(-1, 2), Q(3, 2)),
    )
    for i in range(4):
        for j in range(4):
            assert report.coordinate_generator[i + 4][j] == a[i][j]
            assert report.coordinate_generator[i][j + 4] == -a[i][j] * (
                1 if j == 0 else Q(1, 2)
            )
    expected_weights = (1, 1, 1, 1, 1, Q(1, 2), Q(1, 2), Q(1, 2))
    assert report.energy_metric == tuple(
        tuple(Q(expected_weights[i]) if i == j else Q(0) for j in range(8))
        for i in range(8)
    )
    weighted = product(report.energy_metric, report.coordinate_generator)
    assert all(weighted[i][j] == -weighted[j][i] for i in range(8) for j in range(8))


def test_kernel_jets_recover_channel_difference_without_a_local_loss_term(report):
    memory = report.coordinate_memory
    assert memory.visible_generator == ((0, Q(-2, 3)), (Q(2, 3), 0))
    assert memory.kernel_at_zero == ((Q(-2, 9), 0), (0, Q(-4, 9)))
    a_squared = product(memory.visible_generator, memory.visible_generator)
    assert a_squared[0][0] + memory.kernel_at_zero[0][0] == Q(-2, 3)
    assert a_squared[1][1] + memory.kernel_at_zero[1][1] == Q(-8, 9)
    assert report.static_visible_generator == ((0, Q(-2, 13)), (Q(2, 13), 0))
    assert report.low_frequency_rate_matrix == (
        (Q(309, 169), 0),
        (0, Q(449, 169)),
    )


def test_initial_source_is_a_map_of_declared_hidden_perturbations(report):
    # A retained internal form increment changes phase rate at the bridge,
    # despite zero visible initial state. No source sample is supplied to API.
    initial = tuple((Q(int(i == 1)),) for i in range(8))
    nodal = product(report.nodal_lift, initial)
    source = product(report.initial_source_rows, nodal)
    assert source == ((Q(0),), (Q(-2, 3),))
    hidden = product(report.hidden_initial_projection_rows, nodal)
    assert source == product(report.coordinate_memory.hidden_to_visible, hidden)
    visible_initial = (initial[0], initial[4])
    assert visible_initial == ((0,), (0,))


def test_cycle_orientation_winding_sign_and_node_order_do_not_select_new_memory(source):
    normal = _memory(source)
    reverse_cycles = _memory(
        source,
        left_cycle=(0, 5, 4, 3, 2, 1),
        right_cycle=(6, 11, 10, 9, 8, 7),
        target_phase_turns=tuple(-Q(node % 6, 6) for node in source.nodes),
    )
    reordered = _memory(_source(reverse_order=True))
    for report in (reverse_cycles, reordered):
        assert report.coordinate_generator == normal.coordinate_generator
        assert report.coordinate_memory == normal.coordinate_memory
        assert report.static_visible_generator == normal.static_visible_generator


def test_source_state_and_forged_derived_cache_do_not_become_tangent_preparation(
    source,
):
    clean = _memory(source)
    forged = replace(
        source,
        epi=tuple(float(i) for i in range(12)),
        phase=(Q(1, 8),) * 12,
        form_rates=(I(999),) * 12,
        phase_rates=(I(999),) * 12,
        storage=I(999),
    )
    changed = _memory(forged)
    assert changed.channels.source.epi == tuple(map(Q, range(12)))
    assert changed.coordinate_generator == clean.coordinate_generator
    assert changed.initial_source_rows == clean.initial_source_rows
    assert changed.coordinate_memory == clean.coordinate_memory


@pytest.mark.parametrize(
    "changes",
    [
        {"left_cycle": LEFT[:-1]},
        {"left_cycle": (0, 1, 2, 3, 4, 4)},
        {"left_cycle": (0, 1, 2, 3, 4, 99)},
        {"right_cycle": LEFT},
        {"left_cycle": (1, 2, 3, 4, 5, 0)},
        {"left_cycle": (0, 2, 1, 3, 4, 5)},
        {"target_phase_turns": (0,) * 12},
    ],
)
def test_full_cycle_and_target_declarations_are_required(source, changes):
    with pytest.raises((TypeError, ValueError)):
        _memory(source, **changes)


@pytest.mark.parametrize(
    "changes", [{"beta": 4}, {"capacity": 2}, {"loss": 1}, {"extra_edge": True}]
)
def test_specialized_law_and_complete_support_are_not_silently_relaxed(changes):
    with pytest.raises(ValueError):
        _memory(_source(**changes))


def test_source_primitive_admission_is_not_replaced_by_python_equality(source):
    with pytest.raises((ValueError, TypeError)):
        _memory(replace(source, capacity=(True,) + source.capacity[1:]))
    with pytest.raises(ValueError):
        _memory(replace(source, law="native"))
    with pytest.raises(TypeError):
        assess_sine_bridge_memory(
            object(), left_cycle=LEFT, right_cycle=RIGHT, target_phase_turns=(0,) * 12
        )


def test_export_detaches_exact_coordinate_source_and_model_premises(report):
    payload = report.to_dict()
    assert payload["schema"] == "tnfr.relational-sine-bridge-memory.v1"
    body = payload["report"]
    assert body["static_visible_generator"][0][1] == {
        "numerator": -2,
        "denominator": 13,
    }
    assert body["coordinate_memory"]["hidden_indices"] == [1, 2, 3, 5, 6, 7]
    body["coordinate_generator"][0][4]["numerator"] = 999
    assert report.coordinate_generator[0][4] == Q(-2, 3)
