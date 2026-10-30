"""Full-law controls for the conservative bridge channel distinction."""

from dataclasses import replace
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_resonance import (
    _sine_target_tangent_action,
    assess_sine_bridge_channels,
)

TARGET = tuple(Q(i % 6, 6) for i in range(12))


def _source(*, beta=1, loss=0, capacity=1, graph=None):
    if graph is None:
        graph = nx.disjoint_union(nx.cycle_graph(6), nx.cycle_graph(6))
        graph.add_edge(0, 6)
    graph.graph["GAMMA"] = {"type": "none"}
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=capacity)
    model = RelationalExchangeModel(
        beta, epi_weight=loss, phase_weight=1, phase_domain="regular"
    )
    return graph, bound_relational_sine_exchange(graph, reference_model=model)


@pytest.fixture(scope="module")
def source():
    return _source()[1]


def _report(source, target=TARGET, bridge=(0, 6)):
    return assess_sine_bridge_channels(source, target_phase_turns=target, bridge=bridge)


def _near_zero(interval):
    assert interval.contains(0)
    assert interval.abs_max < Q(1, 10**30)


def test_actual_c6_bridge_has_unequal_second_jets_and_a_physical_lift(source):
    report = _report(source)
    assert report.channel_difference_is_positive
    assert report.channel_difference_bounds.contains(Q(2, 9))
    assert report.channel_difference_bounds.lo > 0
    assert report.first_jet_bounds[1][0].contains(Q(2, 3))
    assert report.first_jet_bounds[0][1].contains(Q(-2, 3))
    assert report.second_jet_diagonal_bounds[0].contains(Q(-2, 3))
    assert report.second_jet_diagonal_bounds[1].contains(Q(-8, 9))
    assert report.source.phase == (0,) * 12
    assert report.target_phase_turns != report.source.phase
    geometry = report.target_geometry.geometry
    for j in range(len(geometry.edges)):
        assert sum(
            geometry.incidence[i][j] * report.nodal_bridge_lift[i] for i in range(12)
        ) == int(j == report.bridge_edge_index)
    assert sum(q / k for q, k in zip(report.nodal_bridge_lift, report.mobility)) == 0


def test_bridge_jets_match_two_derivatives_of_the_shared_complete_nodal_rows(source):
    report = _report(source)
    geometry = report.target_geometry.geometry
    neighbors = [[] for _ in source.nodes]
    cosine_rows = [[] for _ in source.nodes]
    for i, (left, right) in enumerate(geometry.edges):
        # Independent exact C6 cosine, not the report's calculated enclosure.
        cosine = I(1 if i == report.bridge_edge_index else Q(1, 2))
        for node, other in ((left, right), (right, left)):
            neighbors[node].append(other)
            cosine_rows[node].append(cosine)
    zero = (I(0),) * 12
    lift = tuple(I(value) for value in report.nodal_bridge_lift)
    for channel, initial in enumerate(((lift, zero), (zero, lift))):
        current = initial
        for order in (1, 2):
            current = _sine_target_tangent_action(
                source.reference_model,
                neighbors,
                source.capacity,
                cosine_rows,
                *current,
            )
            current = tuple(
                tuple(value * pi_interval() for value in row) for row in current
            )
            observed = tuple(row[6] - row[0] for row in current)
            for output in range(2):
                expected = (
                    report.first_jet_bounds[output][channel]
                    if order == 1
                    else (
                        report.second_jet_diagonal_bounds[output]
                        if output == channel
                        else I(0)
                    )
                )
                _near_zero(observed[output] - expected)


def test_consensus_and_geometrically_remote_bridge_do_not_claim_a_distinction(source):
    report = _report(source, (0,) * 12)
    assert not report.channel_difference_is_positive
    assert report.channel_difference_bounds == I(0)
    assert report.second_jet_diagonal_bounds[0] == report.second_jet_diagonal_bounds[1]
    graph = nx.cycle_graph(6)
    graph.add_edges_from(((0, 6), (6, 7)))
    _, remote = _source(graph=graph)
    remote_report = _report(remote, tuple(Q(i, 6) for i in range(6)) + (0, 0), (6, 7))
    assert not remote_report.channel_difference_is_positive
    assert remote_report.channel_difference_bounds == I(0)


def test_bridge_orientation_and_common_target_turn_shift_preserve_the_result(source):
    normal = _report(source)
    changed = _report(source, tuple(value + Q(3, 7) for value in TARGET), (6, 0))
    assert normal.bridge == changed.bridge
    assert normal.nodal_bridge_lift == changed.nodal_bridge_lift
    assert normal.first_jet_bounds == changed.first_jet_bounds
    assert normal.second_jet_diagonal_bounds == changed.second_jet_diagonal_bounds
    assert normal.channel_difference_bounds == changed.channel_difference_bounds


def test_heterogeneous_capacities_and_storage_scale_both_channels():
    _, source = _source(beta=4)
    capacity = list(source.capacity)
    capacity[0], capacity[6] = Q(2), Q(3)
    report = _report(replace(source, capacity=tuple(capacity)))
    assert report.first_jet_bounds[1][0].contains(Q(5, 6))
    assert report.second_jet_diagonal_bounds[0].contains(Q(-19, 18))
    assert report.second_jet_diagonal_bounds[1].contains(Q(-17, 12))
    assert report.channel_difference_bounds.contains(Q(13, 36))
    assert sum(q / k for q, k in zip(report.nodal_bridge_lift, report.mobility)) == 0


def test_source_cache_cannot_supply_jets_and_normalized_primitives_are_retained(source):
    forged = replace(
        source,
        epi=tuple(float(value) for value in source.epi),
        storage=I(999),
        form_rates=(I(999),) * 12,
        form_gradient=(Q(999),) * 12,
        continuous_loss=Q(999),
    )
    report = _report(forged)
    clean = _report(source)
    assert all(isinstance(value, Q) for value in report.source.epi)
    assert report.first_jet_bounds == clean.first_jet_bounds
    assert report.second_jet_diagonal_bounds == clean.second_jet_diagonal_bounds
    assert report.channel_difference_bounds == clean.channel_difference_bounds
    assert report.to_dict()["schema"] == "tnfr.relational-sine-bridge-channels.v1"


@pytest.mark.parametrize("bridge", [(0, 1), (0, 7), (0,), (0, 6, 7)])
def test_bridge_must_be_an_actual_cut_edge(source, bridge):
    with pytest.raises(ValueError):
        _report(source, (0,) * 12, bridge)


@pytest.mark.parametrize(
    "target",
    [
        tuple(Q(i % 6, 3) for i in range(12)),
        (Q(1, 20),) + TARGET[1:],
        (True,) + TARGET[1:],
        tuple(float(value) for value in TARGET),
    ],
)
def test_target_requires_exact_acute_critical_phase_geometry(source, target):
    with pytest.raises(ValueError):
        _report(source, target)


@pytest.mark.parametrize("changes", [{"loss": 1}, {"capacity": 0}])
def test_admission_keeps_the_conservative_positive_capacity_premises(changes):
    _, invalid = _source(**changes)
    with pytest.raises(ValueError):
        _report(invalid)


def test_forged_primitive_state_and_law_are_readmitted(source):
    with pytest.raises((ValueError, TypeError)):
        _report(replace(source, capacity=(True,) + source.capacity[1:]))
    with pytest.raises(ValueError):
        _report(replace(source, law="native"))
    with pytest.raises(ValueError):
        _report(replace(source, degrees=(1,) + source.degrees[1:]))
    with pytest.raises(TypeError):
        _report(object())


def test_subgrid_storage_and_capacity_remain_exact_premises():
    _, tiny_storage = _source(beta=Q(1, 10**100))
    storage_report = _report(tiny_storage)
    assert storage_report.channel_difference_bounds.contains(
        Q(2, 9) / Q(tiny_storage.reference_model.storage_scale)
    )
    assert storage_report.channel_difference_is_positive
    _, tiny_capacity = _source(capacity=Q(1, 10**100))
    capacity_report = _report(tiny_capacity)
    assert capacity_report.channel_difference_is_positive
    assert capacity_report.channel_difference_bounds.contains(
        Q(2, 9) * tiny_capacity.capacity[0] ** 2
    )
    assert capacity_report.channel_difference_bounds.lo == 0
