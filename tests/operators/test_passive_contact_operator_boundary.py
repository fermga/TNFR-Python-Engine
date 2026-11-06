"""Finite operator events distinguish passive budgets from operator realization.

These synthetic controls execute actual admitted stages and detached readers.
They install no weighted nonlinear executor, trajectory, selector or clock map.
A mismatch rules out the direct identifications tested, not every possible
operator composition, restriction or continuous limit.
"""

from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_SI
from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    _phase_edge_storage,
    evaluate_relational_exchange,
)
from tnfr.operators.definitions import Coupling, Resonance
from tnfr.operators.network_stage import execute_coupling_stage, execute_neighbor_stage
from tnfr.physics.passive_contact import evaluate_passive_contact_budget
from tnfr.physics.support_transport import observe_support_transport

_EPSILON = Q(1, 32)
_MODEL = RelationalExchangeModel(storage_scale=1)


def _source(*, block_phase=0, existing_weight=None):
    graph = nx.Graph()
    graph.add_nodes_from(range(4))
    graph.add_weighted_edges_from(((0, 1, 1), (2, 3, 1)))
    if existing_weight is not None:
        graph.add_edge(1, 2, weight=existing_weight)
    graph.graph.update(
        RANDOM_SEED=17,
        UM_FUNCTIONAL_LINKS=True,
        UM_BIDIRECTIONAL=True,
        UM_CANDIDATE_COUNT=0,
        _node_sample=(2,),
        # Isolate the mandatory phase gate from optional compatibility/activity
        # policy. Target 1 has a compatible existing neighbor in every case.
        UM_COMPAT_THRESHOLD=0,
        UM_MIN_EPI=0,
        UM_SYNC_VF=False,
        UM_STABILIZE_DNFR=False,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_REJECTION_MODE="raise",
        GLYPH_FACTORS={"UM_theta_push": 1 / 8},
        GAMMA={"type": "none"},
        _t=0.0,
    )
    for node, form, phase in zip(
        range(4), (0, _EPSILON, 0, 0), (0, 0, block_phase, block_phase), strict=True
    ):
        graph.nodes[node].update(
            EPI=float(form),
            theta=float(phase),
            nu_f=1.0,
            delta_nfr=0.0,
            dEPI=0.0,
            glyph_history=["AL"],
        )
        graph.nodes[node][ALIAS_SI[0]] = 0.8
    return graph


def _weighted_losses(graph):
    """Reevaluate e*nu*q**2/d via the shared weighted form reader, e=1/2."""
    state = observe_support_transport(graph)
    assert state.nodes == (0, 1, 2, 3)
    strengths = [Q(0)] * len(state.nodes)
    for source, _, weight in state.conductance:
        strengths[source] += weight
    return tuple(
        Q(1, 2) * capacity * gradient**2 / strength
        for capacity, gradient, strength in zip(
            state.capacity, state.dirichlet_gradient, strengths, strict=True
        )
    )


def _budgets(graph, losses):
    # The chosen gaps are 0 or 2 radians in one principal chart. The shared
    # phase-storage owner uses represented trigonometry; this is not exact cos(2).
    form_gap = Q(graph.nodes[1]["EPI"]) - Q(graph.nodes[2]["EPI"])
    phase_gap = Q(graph.nodes[1]["theta"]) - Q(graph.nodes[2]["theta"])
    cost = form_gap**2 / 2 + _phase_edge_storage(phase_gap)
    return tuple(
        evaluate_passive_contact_budget(
            node_losses=losses,
            candidates=((1, 2),),
            contact_costs=(cost,),
            port_shares=((1, 1),),
            storage_scale=1,
            law=law,
        )
        for law in ("reciprocal", "squared_reciprocal")
    )


@pytest.mark.parametrize("block_phase, admitted", ((2, False), (0, True)))
def test_positive_passive_rate_does_not_admit_um_candidate(block_phase, admitted):
    graph = _source(block_phase=block_phase)
    before = deepcopy(graph)
    losses = _weighted_losses(graph)
    # Unit disconnected components are individually in the native law's domain.
    # Its public field independently authenticates these initial loss values.
    native_losses = tuple(
        loss
        for nodes in ((0, 1), (2, 3))
        for loss in evaluate_relational_exchange(
            graph.subgraph(nodes).copy(), model=_MODEL
        ).work.dissipation
    )
    assert losses == native_losses == (_EPSILON**2 / 2,) * 2 + (0, 0)
    assert all(report.contact_rates[0] > 0 for report in _budgets(graph, losses))
    assert nx.utils.graphs_equal(graph, before)

    execution = execute_coupling_stage(graph, Coupling(), (1,))

    assert execution.glyph == "UM" and execution.nodes_processed == 1
    assert tuple(graph.nodes[1]["glyph_history"]) == ("AL", "UM")
    assert graph.graph["_t"] == before.graph["_t"] == 0
    assert graph.has_edge(1, 2) is admitted
    assert tuple(graph.nodes[node]["theta"] for node in graph) == (
        0,
        0,
        block_phase,
        block_phase,
    )
    assert tuple(graph.nodes[node]["EPI"] for node in graph) == (0, _EPSILON, 0, 0)
    if admitted:
        assert graph.edges[1, 2]["weight"] > 0
    else:
        assert set(graph.edges) == {(0, 1), (2, 3)}


def test_um_does_not_reinforce_existing_passive_contact():
    graph = _source(existing_weight=Q(1, 4))
    before = deepcopy(graph)
    losses = _weighted_losses(graph)
    assert all(report.contact_rates[0] > 0 for report in _budgets(graph, losses))
    assert nx.utils.graphs_equal(graph, before)

    execution = execute_coupling_stage(graph, Coupling(), (1,))

    assert execution.glyph == "UM" and execution.nodes_processed == 1
    assert tuple(graph.nodes[1]["glyph_history"]) == ("AL", "UM")
    assert graph.edges[1, 2]["weight"] == before.edges[1, 2]["weight"] == Q(1, 4)
    assert set(graph.edges) == set(before.edges)
    assert graph.graph["_t"] == 0


def test_ra_phase_consensus_is_not_native_form_driven_phase_exchange():
    graph = _source().subgraph((0, 1)).copy()
    for node, form in enumerate((1 / 4, 1 / 2)):
        graph.nodes[node]["EPI"] = form
    graph.graph["GLYPH_FACTORS"] = {
        "RA_epi_diff": 1 / 4,
        "RA_vf_amplification": 1 / 8,
        "RA_phase_coupling": 1 / 8,
    }
    field = evaluate_relational_exchange(graph, model=_MODEL)
    assert field.phase_rate[0] < 0 < field.phase_rate[1]
    before = deepcopy(graph)

    execution = execute_neighbor_stage(graph, Resonance(), (0,))

    assert execution.glyph == "RA" and execution.nodes_processed == 1
    assert tuple(graph.nodes[0]["glyph_history"]) == ("AL", "RA")
    assert tuple(graph.nodes[node]["theta"] for node in graph) == (0, 0)
    assert graph.nodes[0]["EPI"] != before.nodes[0]["EPI"]
    assert graph.graph["_t"] == 0
