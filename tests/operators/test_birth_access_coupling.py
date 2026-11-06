"""Actual THOL association supplies UM access and continued EN/RA input.

Preparation, coefficients and invocation times are supplied. The access policy
does not infer a clock, authenticate arbitrary metadata or rank several children.
"""

from copy import deepcopy
from fractions import Fraction as Q
from math import pi

import networkx as nx
import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.errors import TNFRValueError
from tnfr.operators.birth_access_coupling import (
    execute_birth_access_coupling,
    observe_birth_association_access,
)
from tnfr.operators.coherence import Coherence
from tnfr.operators.coupling import Coupling
from tnfr.operators.dissonance import Dissonance
from tnfr.operators.emission_episode import execute_emission_episode
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import (
    STAGE_SCHEDULE_KEY,
    execute_dissonance_stage,
    execute_pointwise_stage,
)
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.preconditions.reception import validate_reception_strict
from tnfr.operators.reception import Reception
from tnfr.operators.resonance import Resonance
from tnfr.operators.self_organization_selection import (
    execute_eligible_self_organization_stage,
)
from tnfr.types import require_finite_real_scalar_epi

PARENT, PEER = "parent", "peer"
POLICY = "unique_isolated_birth_child"


def _form(graph, node):
    return Q.from_float(
        require_finite_real_scalar_epi(
            get_attr(
                graph.nodes[node], ALIAS_EPI, strict=True, conv=lambda value: value
            )
        )
    )


def _pressure(graph, node):
    return Q.from_float(get_attr(graph.nodes[node], ALIAS_DNFR))


def _cohere(graph, targets):
    execute_pointwise_stage(
        graph,
        Coherence(),
        targets,
        phase_locking_coefficient=0.0,
        require_exact_operator=True,
        validate_preconditions=True,
    )


def _prepare_pressure(graph, targets, factor):
    _cohere(graph, targets)
    graph.graph["GLYPH_FACTORS"]["OZ_dnfr_factor"] = factor
    execute_dissonance_stage(
        graph,
        Dissonance(),
        targets,
        propagate_to_network=False,
        validate_preconditions=True,
    )


def _source(labels=(PARENT, PEER)):
    parent, peer = labels
    graph = nx.Graph()
    graph.add_edge(parent, peer, weight=1.0)
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        INTEGRATOR_METHOD="euler",
        GAMMA={"type": "none"},
        RANDOM_SEED=71,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        EDGE_AWARE_ENABLED=True,
        OZ_NOISE_MODE=False,
        IL_PRECONDITIONS={"warn_zero_dnfr": False},
        CLIP_MODE="hard",
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        THOL_METABOLIC_ENABLED=False,
        THOL_PROPAGATION_ENABLED=False,
        UM_FUNCTIONAL_LINKS=True,
        UM_BIDIRECTIONAL=True,
        UM_SYNC_VF=True,
        UM_STABILIZE_DNFR=True,
        UM_COMPAT_THRESHOLD=0.5,
        history={},
    )
    graph.graph["GLYPH_FACTORS"].update(
        IL_dnfr_factor=0.5,
        OZ_dnfr_factor=2.0,
        THOL_accel=0.125,
        UM_theta_push=0.5,
        UM_vf_sync=0.25,
        UM_dnfr_reduction=0.25,
        EN_mix=0.25,
        RA_epi_diff=0.5,
        RA_vf_amplification=0.125,
    )
    for node in labels:
        graph.nodes[node].update(
            EPI=0.25,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.5,
            Si=0.0,
            epi_kind="form",
        )
        set_attr(graph.nodes[node], ALIAS_DEPI, 0.0)
        set_attr(graph.nodes[node], ALIAS_D2EPI, 0.0)
    _prepare_pressure(graph, labels, 2.0)
    first = execute_emission_episode(
        graph, labels, duration=0.125, method="euler", n_jobs=1
    )
    assert first.exact_rates == (Q(1, 2), Q(1, 2))
    assert first.exact_epi_after == (Q(5, 16), Q(5, 16))
    _prepare_pressure(graph, (parent,), 2.5)
    second = execute_emission_episode(
        graph, labels, duration=0.125, method="euler", n_jobs=1
    )
    assert second.exact_rates == (Q(5, 8), Q(1, 2))
    assert second.exact_epi_after == (Q(25, 64), Q(3, 8))
    for node, endpoint, acceleration, steps in (
        (parent, Q(25, 64), 1.0, 6),
        (peer, Q(3, 8), 0.0, 4),
    ):
        data = graph.nodes[node]
        assert tuple(data["epi_time_history"]) == (
            (0.0, 0.25),
            (0.125, 0.3125),
            (0.25, float(endpoint)),
        )
        observed = observe_structural_acceleration(graph, node)
        assert observed.available and observed.source == "epi_time_history"
        assert observed.value == acceleration
        grammar = read_grammar_state(data)
        assert grammar.operator_step == steps and grammar.prior_coherence
        assert data["_structural_lineage"]["activation_count"] == 2
    assert graph.graph["_t"] == 0.25
    return graph


def _birth(graph, parent=PARENT, *, count=1):
    dispatch = execute_eligible_self_organization_stage(
        graph, execution_kwargs={"tau": 0.5, "validate_preconditions": True}
    )
    assert dispatch.eligibility.eligible_nodes == (parent,)
    ((actual_parent, child),) = dispatch.parent_children
    assert actual_parent == parent and graph.degree(child) == 0
    assert graph.nodes[parent]["sub_nodes"][-1] == child
    assert len(graph.nodes[parent]["sub_nodes"]) == count
    assert graph.graph["hierarchy"][parent] == graph.nodes[parent]["sub_nodes"]
    record = graph.nodes[parent]["sub_epis"][-1]
    assert record["node_id"] == child and record["timestamp"] == 6 + count
    assert _pressure(graph, parent) == Q(5 + count, 8)
    assert read_grammar_state(graph.nodes[parent]).operator_step == 6 + count
    data = graph.nodes[child]
    assert data["parent_node"] == parent and _form(graph, child) == Q(15, 128)
    assert get_attr(data, ALIAS_VF) == 0.95 and _pressure(graph, child) == 0
    assert get_attr(data, ALIAS_THETA) == 0.0
    assert read_grammar_state(data).history == ()
    assert data["epi_history"] == [15 / 128]
    assert "epi_time_history" not in data and "_structural_lineage" not in data
    assert graph.graph["_t"] == 0.25
    return child


def _state(graph):
    """Declared structural/lifecycle observation, not a complete checkpoint."""
    return (
        tuple(graph),
        tuple((u, v, deepcopy(data)) for u, v, data in graph.edges(data=True)),
        tuple(
            (
                node,
                _form(graph, node),
                _pressure(graph, node),
                get_attr(data, ALIAS_VF),
                get_attr(data, ALIAS_THETA),
                read_grammar_state(data),
                deepcopy(data.get("epi_time_history")),
                deepcopy(data.get("_structural_lineage")),
                tuple(data.get("sub_nodes", ())),
            )
            for node, data in graph.nodes(data=True)
        ),
        deepcopy(graph.graph.get("hierarchy")),
        deepcopy(graph.graph.get(STAGE_SCHEDULE_KEY)),
        graph.graph["_t"],
    )


def _automatic_contact(graph, parent, child):
    before_step = read_grammar_state(graph.nodes[parent]).operator_step
    result = execute_birth_access_coupling(graph, parent)
    assert result.executed and result.stage_result.nodes_processed == 1
    assert result.access.policy == POLICY and result.access.parent == parent
    assert result.access.status == "selected"
    assert result.access.selected_partner == child
    assert result.access.isolated_children == (child,)
    assert graph.has_edge(parent, child)
    assert read_grammar_state(graph.nodes[parent]).operator_step == before_step + 1
    assert read_grammar_state(graph.nodes[parent]).history[-1] == "UM"
    assert graph.graph[STAGE_SCHEDULE_KEY]["potential_pair"] == {
        "target": parent,
        "partner": child,
        "origin": "explicit_request",
    }
    return result


def _contact_state(graph):
    """Primary channels, support and grammar; no incidental diagnostic clock."""
    return (
        tuple(
            (
                node,
                _form(graph, node),
                _pressure(graph, node),
                get_attr(data, ALIAS_VF),
                get_attr(data, ALIAS_THETA),
                read_grammar_state(data),
            )
            for node, data in graph.nodes(data=True)
        ),
        tuple((u, v, deepcopy(data)) for u, v, data in graph.edges(data=True)),
    )


def _respond(graph, parent, peer, child, *, connected, peer_only=False):
    physical = {
        node: deepcopy(graph.nodes[node]["epi_time_history"]) for node in (parent, peer)
    }
    expected_pressure = Q(9, 16) if connected else Q(3, 4)
    assert _pressure(graph, parent) == expected_pressure
    expected_capacity = Q(8950904259398861, 2**53) if connected else Q(1)
    assert Q.from_float(get_attr(graph.nodes[parent], ALIAS_VF)) == expected_capacity
    for count in range(1, 4):
        _cohere(graph, (parent,))
        assert _pressure(graph, parent) == expected_pressure / 2**count
        if count == 2:
            if connected:
                validate_reception_strict(graph, parent)
            else:
                with pytest.raises(TNFRValueError, match="DNFR"):
                    validate_reception_strict(graph, parent)
    validate_reception_strict(graph, parent)
    child_before = deepcopy(graph.nodes[child])
    kwargs = {"source_nodes": (peer,)} if peer_only else {}
    Reception()(graph, parent, track_sources=False, **kwargs)
    Resonance()(graph, peer)
    includes_child = connected and not peer_only
    expected = (
        (Q(363, 1024), Q(747, 2048)) if includes_child else (Q(99, 256), Q(195, 512))
    )
    observed = (_form(graph, parent), _form(graph, peer))
    assert observed == expected
    assert read_grammar_state(graph.nodes[parent]).operator_step == 11 + int(connected)
    assert read_grammar_state(graph.nodes[peer]).operator_step == 5
    assert get_attr(graph.nodes[peer], ALIAS_VF) == 9 / 8
    assert graph.nodes[child] == child_before
    for node in (parent, peer):
        assert graph.nodes[node]["epi_time_history"] == physical[node]
        assert graph.nodes[node]["_structural_lineage"]["activation_count"] == 2
    assert graph.graph["_t"] == 0.25
    return observed


def _unrelated_isolate(graph):
    """Declared compatible control, inserted before the actual child exists."""
    node = "unrelated"
    graph.add_node(
        node,
        EPI=25 / 64,
        nu_f=0.95,
        theta=0.0,
        delta_nfr=0.0,
        Si=0.0,
        epi_kind="form",
    )
    return node


def test_actual_birth_access_matches_explicit_contact_and_changes_response():
    outcomes = {}
    contact_states = {}
    for arm in ("derived", "explicit", "no_contact", "derived_peer_only"):
        graph = _source()
        child = _birth(graph)
        if arm in {"derived", "derived_peer_only"}:
            result = _automatic_contact(graph, PARENT, child)
            assert result.access.associated_children == (child,)
        elif arm == "explicit":
            Coupling()(graph, PARENT, potential_partner=child)
        if arm in {"derived", "explicit"}:
            contact_states[arm] = _contact_state(graph)
        assert graph.edges[PARENT, PEER] == {"weight": 1.0}
        outcomes[arm] = _respond(
            graph,
            PARENT,
            PEER,
            child,
            connected=arm != "no_contact",
            peer_only=arm == "derived_peer_only",
        )
    assert contact_states["derived"] == contact_states["explicit"]
    assert outcomes["derived"] == outcomes["explicit"]
    assert outcomes["derived_peer_only"] == outcomes["no_contact"]
    assert outcomes["derived"][1] - outcomes["no_contact"][1] == -Q(33, 2048)


def test_actual_birth_access_excludes_earlier_more_compatible_unrelated_isolate():
    graph = _source()
    unrelated = _unrelated_isolate(graph)
    child = _birth(graph)
    assert tuple(graph).index(unrelated) < tuple(graph).index(child)
    before = deepcopy(graph.nodes[unrelated])
    result = _automatic_contact(graph, PARENT, child)
    assert result.access.associated_children == (child,)
    assert graph.degree(unrelated) == 0 and graph.nodes[unrelated] == before


def test_selected_actual_child_rejection_does_not_fall_back_to_unrelated_isolate():
    graph = _source()
    unrelated = _unrelated_isolate(graph)
    child = _birth(graph)
    # Deliberate state perturbation, not an acquired phase trajectory.
    set_attr(graph.nodes[child], ALIAS_THETA, pi)
    access = observe_birth_association_access(graph, PARENT)
    assert access.selected_partner == child
    before = _state(graph)
    with pytest.raises(OperatorPreconditionError, match="U3"):
        execute_birth_access_coupling(graph, PARENT)
    assert _state(graph) == before
    assert graph.degree(child) == graph.degree(unrelated) == 0
    assert graph.nodes[PARENT]["sub_nodes"] == [child]


def test_actual_prebirth_source_has_no_candidate_and_acquires_no_um():
    graph = _source()
    before = _state(graph)
    access = observe_birth_association_access(graph, PARENT)
    assert access.status == "no_associated_child" and access.selected_partner is None
    assert access.associated_children == access.isolated_children == ()
    result = execute_birth_access_coupling(graph, PARENT)
    assert (
        result.access == access and not result.executed and result.stage_result is None
    )
    assert _state(graph) == before


def test_two_actual_supplied_birth_occasions_are_ambiguous_not_ranked():
    graph = _source()
    first = _birth(graph)
    second = _birth(graph, count=2)
    before = _state(graph)
    result = execute_birth_access_coupling(graph, PARENT)
    assert result.access.associated_children == (first, second)
    assert result.access.isolated_children == (first, second)
    assert (
        result.access.status == "ambiguous" and result.access.selected_partner is None
    )
    assert not result.executed and result.stage_result is None
    assert _state(graph) == before


def test_actual_contact_of_one_child_leaves_unique_live_access_to_its_sibling():
    graph = _source()
    first = _birth(graph)
    second = _birth(graph, count=2)
    assert observe_birth_association_access(graph, PARENT).status == "ambiguous"
    Coupling()(graph, PARENT, potential_partner=first)
    access = observe_birth_association_access(graph, PARENT)
    assert access.associated_children == (first, second)
    assert access.connected_children == (first,) and access.isolated_children == (
        second,
    )
    assert access.selected_partner == second
    result = _automatic_contact(graph, PARENT, second)
    assert result.access == access
    assert graph.has_edge(PARENT, first) and graph.has_edge(PARENT, second)
    assert graph.edges[PARENT, PEER] == {"weight": 1.0}
    assert _pressure(graph, PARENT) == Q(63, 128)
    assert read_grammar_state(graph.nodes[PARENT]).operator_step == 10
    assert graph.graph["_t"] == 0.25


def test_old_passive_selection_does_not_replay_actual_contact():
    graph = _source()
    child = _birth(graph)
    old_access = observe_birth_association_access(graph, PARENT)
    assert old_access.selected_partner == child
    Coupling()(graph, PARENT, potential_partner=child)
    before = _state(graph)
    result = execute_birth_access_coupling(graph, PARENT)
    assert result.access.status == "no_isolated_child"
    assert result.access.connected_children == (child,)
    assert (
        result.access.isolated_children == () and result.access.selected_partner is None
    )
    assert not result.executed and result.stage_result is None
    assert old_access.selected_partner == child
    assert _state(graph) == before


@pytest.mark.parametrize("labels", [("renamed_parent", "renamed_peer"), (31, 17)])
def test_actual_access_and_response_under_consistent_initial_relabeling(labels):
    parent, peer = labels
    graph = _source(labels)
    child = _birth(graph, parent)
    result = _automatic_contact(graph, parent, child)
    assert result.access.associated_children == (child,)
    assert graph.nodes[child]["parent_node"] == parent
    assert _respond(graph, parent, peer, child, connected=True) == (
        Q(363, 1024),
        Q(747, 2048),
    )
