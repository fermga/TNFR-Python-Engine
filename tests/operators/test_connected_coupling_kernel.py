"""An isolated supplied partner participates in the existing UM neighborhood map."""

from __future__ import annotations

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_THETA, ALIAS_VF
from tnfr.errors import TNFRValueError
from tnfr.operators._coupling_potential import resolve_potential_coupling
from tnfr.operators._coupling_stage_kernel import propose_coupling_stage
from tnfr.utils import angle_diff

FACTORS = {"UM_theta_push": 0.25, "UM_vf_sync": 0.5, "UM_dnfr_reduction": 0.5}


def _graph(graph_type=nx.Graph):
    graph = graph_type(
        UM_FUNCTIONAL_LINKS=True,
        UM_COMPAT_THRESHOLD=0.0,
        UM_BIDIRECTIONAL=True,
        UM_SYNC_VF=True,
        UM_STABILIZE_DNFR=True,
    )
    for node, phase, capacity in ((0, 0.0, 1.0), (1, 0.4, 5.0), (2, -0.4, 3.0)):
        graph.add_node(node, EPI=0.6, nu_f=capacity, theta=phase, delta_nfr=0.2)
    graph.add_edge(0, 2, weight=0.37, length=2.0, marker={"preserve": [1, 2]})
    return graph


def _state(graph):
    edges = (
        tuple(graph.edges(keys=True, data=True))
        if graph.is_multigraph()
        else tuple(graph.edges(data=True))
    )
    return deepcopy((dict(graph.nodes(data=True)), edges, graph.graph))


def _proposal(graph, factors=FACTORS, **kwargs):
    return propose_coupling_stage(graph, (0,), factors, potential_partner=1, **kwargs)


@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
@pytest.mark.parametrize("bidirectional", [False, True])
def test_augmented_map_equals_separate_realized_reference_without_provisional_edge(
    graph_type, bidirectional
):
    graph = _graph(graph_type)
    graph.graph["UM_BIDIRECTIONAL"] = bidirectional
    if graph.is_multigraph():
        graph.add_edge(0, 2, weight=0.0, marker="second retained edge")
    graph.add_node(3, theta=2.5, EPI="unconsumed", nu_f=-1.0)
    graph.add_edge(0, 3, weight=4.0)
    before = _state(graph)
    reference = deepcopy(graph)
    reference.add_edge(0, 1)
    reference.graph["UM_FUNCTIONAL_LINKS"] = False

    proposal = _proposal(graph)
    realized = propose_coupling_stage(reference, (0,), FACTORS)

    assert proposal.node_updates == realized.node_updates
    assert proposal.target_proposals[0].compatible_neighbors == (2, 1)
    assert (
        proposal.target_proposals[0].phase_proposals
        == realized.target_proposals[0].phase_proposals
    )
    assert len(proposal.edges) == 1
    assert (proposal.edges[0].left, proposal.edges[0].right) == (0, 1)
    assert _state(graph) == before
    assert not graph.has_edge(0, 1)
    assert graph.degree(1) == 0


@pytest.mark.parametrize("bidirectional", [False, True])
def test_augmented_capacity_pressure_and_phase_have_independent_oracles(bidirectional):
    graph = _graph()
    graph.graph["UM_BIDIRECTIONAL"] = bidirectional
    proposal = _proposal(graph)
    updates = {update.node: update for update in proposal.node_updates}
    remaining_gap = 0.3 if bidirectional else 0.4

    assert updates[0].theta_after == pytest.approx(0.0, abs=1e-15)
    assert updates[0].vf_after == 2.5  # 1 + .5*((3+5)/2 - 1), not pair-only 3.
    assert updates[0].dnfr_after == pytest.approx(0.1 + 0.1 * remaining_gap / math.pi)
    if bidirectional:
        assert angle_diff(updates[1].theta_after, 0.0) == pytest.approx(0.3)
        assert angle_diff(updates[2].theta_after, 0.0) == pytest.approx(-0.3)
        assert updates[1].vf_after is updates[2].vf_after is None
        assert updates[1].dnfr_after is updates[2].dnfr_after is None
    else:
        assert tuple(updates) == (0,)


def test_initial_old_compatibility_cannot_hide_incompatible_requested_partner():
    graph = _graph()
    graph.nodes[1]["theta"] = math.pi
    before = _state(graph)
    with pytest.raises(TNFRValueError, match="phase gate"):
        _proposal(graph)
    assert _state(graph) == before


def test_multineighbor_unidirectional_request_rejects_lost_final_partner_u3():
    graph = _graph()
    limit = math.pi / 3
    graph.graph.update(DELTA_PHI_MAX=limit, UM_BIDIRECTIONAL=False)
    graph.nodes[1]["theta"] = -limit
    graph.nodes[2]["theta"] = limit
    graph.add_node(3, theta=limit, EPI=0.6, nu_f=3.0)
    graph.add_edge(0, 3, weight=0.7)
    before = _state(graph)
    assert resolve_potential_coupling(graph, 0, 1).neighbors == (2, 3, 1)
    # Consensus of two +pi/3 and one -pi/3 is pi/6. Moving the
    # target there leaves the fixed candidate at distance pi/2 > limit.
    with pytest.raises(TNFRValueError, match="final U3 gate"):
        _proposal(graph, {**FACTORS, "UM_theta_push": 1.0})
    assert _state(graph) == before


def test_connected_target_can_have_no_compatible_old_neighbor():
    graph = _graph()
    graph.nodes[2].update(theta=math.pi, EPI="unconsumed", nu_f=-1)
    proposal = _proposal(graph)
    assert proposal.target_proposals[0].compatible_neighbors == (1,)
    assert {update.node for update in proposal.node_updates} == {0, 1}
    assert proposal.node_updates[0].vf_after == 3.0


@pytest.mark.parametrize("bad_phase", [None, True, math.nan, math.inf])
def test_all_old_outgoing_phases_are_admitted_before_compatible_filtering(bad_phase):
    graph = _graph()
    graph.nodes[2][ALIAS_THETA[0]] = bad_phase
    graph.nodes[2][ALIAS_THETA[1]] = 0.0
    before = _state(graph)
    with pytest.raises(TNFRValueError):
        _proposal(graph)
    assert _state(graph) == before


@pytest.mark.parametrize("sync", [False, True])
def test_old_capacity_is_consumed_only_when_selected_sync_branch_uses_it(sync):
    graph = _graph()
    graph.graph["UM_SYNC_VF"] = sync
    graph.nodes[2][ALIAS_VF[0]] = math.nan
    graph.nodes[2][ALIAS_VF[1]] = 1.0
    if sync:
        with pytest.raises(TNFRValueError):
            _proposal(graph)
    else:
        proposal = _proposal(graph)
        assert all(update.vf_after is None for update in proposal.node_updates)


def test_incoming_only_target_support_is_retained_but_not_a_um_participant():
    graph = _graph(nx.DiGraph)
    graph.remove_edge(0, 2)
    graph.add_edge(2, 0, weight=0.31)
    graph.nodes[2].update(theta=math.nan, EPI="unconsumed", nu_f=-1)
    before = _state(graph)
    proposal = _proposal(graph)
    assert proposal.target_proposals[0].compatible_neighbors == (1,)
    assert {update.node for update in proposal.node_updates} == {0, 1}
    assert [(edge.left, edge.right) for edge in proposal.edges] == [(0, 1)]
    assert _state(graph) == before


@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
def test_existing_target_loop_preserves_ordinary_multiplicity_and_deduplicates_writes(
    graph_type,
):
    graph = _graph(graph_type)
    graph.add_edge(0, 0, weight=0.0)
    if graph.is_multigraph():
        graph.add_edge(0, 0, weight=0.81)
    reference = deepcopy(graph)
    reference.add_edge(0, 1)
    reference.graph["UM_FUNCTIONAL_LINKS"] = False
    proposal = _proposal(graph)
    assert proposal.target_proposals[0].compatible_neighbors == (2, 0, 1)
    assert (
        proposal.node_updates
        == propose_coupling_stage(reference, (0,), FACTORS).node_updates
    )
    assert len({update.node for update in proposal.node_updates}) == len(
        proposal.node_updates
    )


def test_connected_request_consumes_no_candidate_sampler_or_unrelated_phase():
    graph = _graph()
    reference = _proposal(graph)
    graph.graph.update(
        _node_sample=object(), UM_CANDIDATE_COUNT="unused", UM_CANDIDATE_MODE=object()
    )
    graph.add_node(3, theta=math.nan)
    assert _proposal(graph, resolved_seed=object(), node_offsets=object()) == reference


def test_failed_partner_affinity_rejects_even_when_old_neighbor_is_admitted():
    graph = _graph()
    graph.graph["UM_COMPAT_THRESHOLD"] = 1.0
    graph.nodes[1]["EPI"] = -0.6
    before = _state(graph)
    with pytest.raises(TNFRValueError, match="final link admission"):
        _proposal(graph)
    assert _state(graph) == before


def test_selection_does_not_require_parent_child_metadata():
    graph = _graph()
    assert "parent_node" not in graph.nodes[1]
    assert "sub_nodes" not in graph.nodes[0]
    assert resolve_potential_coupling(graph, 0, 1).neighbors == (2, 1)
