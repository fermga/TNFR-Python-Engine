"""Pure UM pair-domain controls; supplied access does not imply emergence."""

from __future__ import annotations

import math
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.errors import TNFRValueError
from tnfr.operators._coupling_potential import resolve_potential_coupling
from tnfr.operators._coupling_stage_kernel import propose_coupling_stage
from tnfr.utils import angle_diff

FACTORS = {"UM_theta_push": 0.25, "UM_vf_sync": 0.5, "UM_dnfr_reduction": 0.5}


def _pair(graph_type=nx.Graph):
    graph = graph_type(UM_FUNCTIONAL_LINKS=True)
    graph.add_node(0, EPI=0.6, nu_f=1.0, theta=0.0, delta_nfr=0.2)
    graph.add_node(1, EPI=0.6, nu_f=2.0, theta=0.4, delta_nfr=-0.3)
    return graph


def _state(graph):
    return deepcopy((dict(graph.nodes(data=True)), tuple(graph.edges), graph.graph))


def _proposal(graph, **kwargs):
    return propose_coupling_stage(graph, (0,), FACTORS, potential_partner=1, **kwargs)


@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
@pytest.mark.parametrize("bidirectional", [False, True])
def test_pair_reuses_the_actual_existing_relation_map(graph_type, bidirectional):
    potential = _pair(graph_type)
    potential.graph["UM_BIDIRECTIONAL"] = bidirectional
    before = _state(potential)
    # This separate reference already has support. No provisional edge is
    # inserted into the graph consumed by the potential-pair action.
    realized_reference = deepcopy(potential)
    realized_reference.add_edge(0, 1)
    realized_reference.graph["UM_FUNCTIONAL_LINKS"] = False

    proposed = _proposal(potential)
    reference = propose_coupling_stage(realized_reference, (0,), FACTORS)

    assert proposed.node_updates == reference.node_updates
    assert proposed.target_proposals[0].potential_partner == 1
    assert reference.target_proposals[0].potential_partner is None
    assert len(proposed.edges) == 1
    assert (proposed.edges[0].left, proposed.edges[0].right) == (0, 1)
    assert proposed.edges[0].weight == pytest.approx(1 - 0.3 / (2 * math.pi))
    assert _state(potential) == before


def test_omitted_partner_keeps_existing_relation_admission():
    isolated = _pair()
    with pytest.raises(TNFRValueError, match="phase gate"):
        propose_coupling_stage(isolated, (0,), FACTORS)

    realized = _pair()
    realized.add_edge(0, 1)
    realized.graph["UM_FUNCTIONAL_LINKS"] = False
    assert propose_coupling_stage(realized, (0,), FACTORS) == propose_coupling_stage(
        realized, (0,), FACTORS, potential_partner=None
    )


@pytest.mark.parametrize("targets", [(), (0, 1)])
def test_potential_pair_rejects_non_single_target_stage(targets):
    graph = _pair()
    before = _state(graph)
    with pytest.raises(TNFRValueError, match="exactly one target"):
        propose_coupling_stage(graph, targets, FACTORS, potential_partner=1)
    assert _state(graph) == before


@pytest.mark.parametrize("partner", [0, 2, []], ids=["self", "unknown", "unhashable"])
def test_potential_pair_requires_distinct_existing_identities(partner):
    graph = _pair()
    before = _state(graph)
    with pytest.raises(TNFRValueError):
        propose_coupling_stage(graph, (0,), FACTORS, potential_partner=partner)
    assert _state(graph) == before


@pytest.mark.parametrize("edge", [(0, 1), (1, 0), (2, 1), (1, 2), (1, 1)])
def test_partner_total_isolation_includes_incoming_edges_and_self_loops(edge):
    graph = _pair(nx.DiGraph)
    graph.add_node(2, EPI=0.6, nu_f=1.0, theta=0.0)
    graph.add_edge(*edge, weight=0.0)
    before = _state(graph)
    with pytest.raises(TNFRValueError, match="totally isolated"):
        _proposal(graph)
    assert _state(graph) == before


@pytest.mark.parametrize("endpoint", [0, 1])
@pytest.mark.parametrize("aliases", [ALIAS_EPI, ALIAS_VF, ALIAS_THETA])
@pytest.mark.parametrize(
    "invalid",
    [True, math.nan, Fraction(1, 10**400)],
    ids=["boolean", "nan", "underflow"],
)
def test_pair_state_admission_rejects_authoritative_invalid_alias(
    endpoint, aliases, invalid
):
    graph = _pair()
    graph.graph.update(UM_SYNC_VF=False, UM_STABILIZE_DNFR=False)
    graph.nodes[endpoint][aliases[0]] = invalid
    graph.nodes[endpoint][aliases[1]] = 0.5
    before = _state(graph)
    with pytest.raises(TNFRValueError):
        _proposal(graph)
    assert _state(graph) == before


@pytest.mark.parametrize("endpoint", [0, 1])
def test_pair_capacity_remains_nonnegative_when_sync_is_disabled(endpoint):
    graph = _pair()
    graph.graph["UM_SYNC_VF"] = False
    graph.nodes[endpoint][ALIAS_VF[0]] = -1.0
    with pytest.raises(TNFRValueError, match="nonnegative"):
        _proposal(graph)


def test_valid_secondary_aliases_admit_the_same_pair():
    graph = _pair()
    reference = _proposal(graph)
    for endpoint in graph:
        for aliases in (ALIAS_EPI, ALIAS_THETA):
            graph.nodes[endpoint][aliases[1]] = graph.nodes[endpoint].pop(aliases[0])
    assert _proposal(graph) == reference


@pytest.mark.parametrize(
    "phase, config",
    [(math.pi, {}), (0.4, {"UM_MAX_PHASE_DIFF": 0.3}), (0.4, {"DELTA_PHI_MAX": 2.0})],
)
def test_u3_cannot_be_bypassed_by_potential_access_or_affinity(phase, config):
    graph = _pair()
    graph.nodes[1][ALIAS_THETA[0]] = phase
    graph.graph.update(UM_COMPAT_THRESHOLD=0.0, UM_STRICT_PHASE_CHECK=False, **config)
    before = _state(graph)
    with pytest.raises(TNFRValueError, match="phase gate"):
        _proposal(graph)
    assert _state(graph) == before


@pytest.mark.parametrize("phase", [0.0, math.pi / 2, -math.pi / 2, math.tau - 0.01])
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("gain", [0.0, 0.25, 1.0])
def test_pair_circular_contraction_and_final_u3(phase, bidirectional, gain):
    graph = _pair()
    graph.nodes[1][ALIAS_THETA[0]] = phase
    graph.graph["UM_BIDIRECTIONAL"] = bidirectional
    graph.graph["UM_COMPAT_THRESHOLD"] = 0.0
    initial_gap = angle_diff(phase, 0.0)
    factors = {**FACTORS, "UM_theta_push": gain}

    if gain == 0.0 and bidirectional and phase == -math.pi / 2:
        # Canonical normalization of -pi/2 to 3pi/2 introduces a represented
        # gap one ulp above the hard limit. Both actual maps must reject;
        # the exact-real contraction identity cannot weaken final U3.
        before = _state(graph)
        realized_reference = deepcopy(graph)
        realized_reference.add_edge(0, 1)
        realized_reference.graph["UM_FUNCTIONAL_LINKS"] = False
        with pytest.raises(TNFRValueError, match="final U3 gate"):
            propose_coupling_stage(graph, (0,), factors, potential_partner=1)
        with pytest.raises(TNFRValueError, match="final U3 gate"):
            propose_coupling_stage(realized_reference, (0,), factors)
        assert _state(graph) == before
        return

    stage = propose_coupling_stage(graph, (0,), factors, potential_partner=1)

    final_phases = {node: data[ALIAS_THETA[0]] for node, data in graph.nodes(data=True)}
    for update in stage.node_updates:
        if update.theta_after is not None:
            final_phases[update.node] = update.theta_after
    final_gap = angle_diff(final_phases[1], final_phases[0])
    assert final_gap == pytest.approx((1 - gain) * initial_gap)
    assert abs(final_gap) <= math.pi / 2


@pytest.mark.parametrize("rejection", ["disabled", "affinity"])
def test_creation_request_must_actually_propose_its_link(rejection):
    graph = _pair()
    if rejection == "disabled":
        graph.graph["UM_FUNCTIONAL_LINKS"] = False
    else:
        graph.nodes[1][ALIAS_EPI[0]] = -0.6
    before = _state(graph)
    with pytest.raises(TNFRValueError):
        _proposal(graph)
    assert _state(graph) == before


def test_supplied_pair_does_not_consume_candidate_search_or_unrelated_phases():
    graph = _pair()
    reference = _proposal(graph)
    graph.graph.update(
        _node_sample=object(),
        UM_CANDIDATE_COUNT="not consumed",
        UM_CANDIDATE_MODE=object(),
        RANDOM_SEED="not consumed by this pure kernel",
    )
    graph.add_node(2, theta=math.nan, EPI="not consumed", nu_f=-1.0)
    before = _state(graph)

    stage = _proposal(graph, resolved_seed=object(), node_offsets=object())

    assert stage == reference
    # Ignore identity-only sentinels when comparing the otherwise unchanged graph.
    assert tuple(graph.edges) == before[1]
    assert dict(graph.nodes(data=True)) == before[0]
    assert "_node_cache" not in graph.graph


def test_potential_selection_is_an_intended_relation_not_a_preinserted_edge():
    graph = _pair()
    selection = resolve_potential_coupling(graph, 0, 1)
    assert selection.neighbors == (1,)
    assert tuple(graph.neighbors(0)) == ()
    assert tuple(graph.neighbors(1)) == ()
    assert not graph.has_edge(0, 1)
