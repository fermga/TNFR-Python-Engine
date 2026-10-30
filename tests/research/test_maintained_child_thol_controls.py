"""Synthetic nested THOL controls; no retained source or reserved event.

The two-node graph and three samples below are manufactured independently of
the acquired child. They test canonical event wiring, not acquisition, U5 or
physical predictions. No numerical flow or remaining SHA event is executed.
"""

from copy import deepcopy
from fractions import Fraction as Q
from math import pi

import networkx as nx
import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators.definitions import Dissonance, SelfOrganization, Silence
from tnfr.operators.grammar_debt import node_debt
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.grammar_types import StructuralGrammarError
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.operators.preconditions import OperatorPreconditionError, validate_silence
from tnfr.operators.preconditions.self_organization import (
    validate_self_organization_strict,
)
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.physics.relational_sine_support import (
    bound_relational_sine_frozen_isolates,
    bound_relational_sine_support_exchange,
)
from tnfr.research.local_capacity_history import MODEL, refresh_pressure
from tnfr.research.maintained_parent_thol_execution import refresh_frozen_successor
from tnfr.research.retained_graph_state import record_graph_state

ROOT = "synthetic_root"
ACTOR = "synthetic_branch"
CHILD = ACTOR + "_sub_0"


def _word():
    return ValidatedSequence(
        (Dissonance(), SelfOrganization(), Silence()),
        context={"initial_epi_nonzero": True, "compatibility_profile": "core"},
    )


def _graph():
    graph = nx.Graph(
        _t=12.0,
        GAMMA={"type": "none"},
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        EDGE_AWARE_ENABLED=False,
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        hierarchy={ROOT: [ACTOR]},
    )
    for node, epi, phase, capacity in (
        (ROOT, 5 / 8, 1 / 4, 2.5),
        (ACTOR, 3 / 8, 0.0, 7.3),
    ):
        graph.add_node(
            node,
            **{
                ALIAS_EPI[0]: epi,
                ALIAS_THETA[0]: phase,
                ALIAS_VF[0]: capacity,
                ALIAS_DNFR[0]: 0.0,
                "epi_history": [epi],
                "glyph_history": [],
            },
        )
    graph.nodes[ROOT]["sub_nodes"] = [ACTOR]
    graph.nodes[ACTOR].update(
        parent_node=ROOT,
        hierarchy_level=1,
        _bifurcation_level=1,
        _hierarchy_path=[ROOT],
        glyph_history=["OZ"],
        _operator_step=1,
        _grammar_u2_debt=1,
        _grammar_prior_coherence=False,
        epi_time_history=[(10.0, 1 / 4), (11.0, 1 / 4), (12.0, 3 / 8)],
    )
    graph.add_edge(ROOT, ACTOR, weight=3 / 4)
    refresh_pressure(graph)
    return graph


@pytest.fixture(autouse=True)
def forbid_retained_readers_flow_and_closure(monkeypatch):
    from tnfr.dynamics import integrators
    from tnfr.operators import al_sha_stage_proposals
    from tnfr.research import (
        maintained_child_history_acquisition,
        maintained_child_sample_certificate,
        maintained_sha_motion_source,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("synthetic THOL controls cannot read or execute retained evidence")

    monkeypatch.setattr(maintained_sha_motion_source, "load_sha_successor", forbidden)
    monkeypatch.setattr(
        maintained_child_sample_certificate,
        "prepare_child_sample_certificate",
        forbidden,
    )
    monkeypatch.setattr(
        maintained_child_history_acquisition, "evaluate_child_history", forbidden
    )
    monkeypatch.setattr(integrators, "rk4_coupled_step", forbidden)
    monkeypatch.setattr(al_sha_stage_proposals, "propose_silence_stage", forbidden)


def _primary(graph, nodes):
    return tuple(
        tuple(
            get_attr(graph.nodes[node], alias)
            for alias in (ALIAS_EPI, ALIAS_THETA, ALIAS_VF)
        )
        for node in nodes
    )


def _run(graph, *, refresh=None):
    return execute_network_operator_stage(
        graph,
        SelfOrganization(),
        (ACTOR,),
        sequence_context=_word().step(1),
        compute_delta_nfr=refresh,
    )


def test_complete_nested_jump_has_rounded_capacity_and_no_inherited_history():
    graph = _graph()
    old_nodes = tuple(graph)
    primary = _primary(graph, old_nodes)
    edges = deepcopy(tuple(graph.edges(data=True)))
    old_pressure = get_attr(graph.nodes[ACTOR], ALIAS_DNFR)
    root_before = deepcopy(graph.nodes[ROOT])
    history = deepcopy(graph.nodes[ACTOR]["epi_time_history"])
    observation = observe_structural_acceleration(graph, ACTOR)
    assert observation.available and observation.value == 1 / 8
    validate_self_organization_strict(graph, ACTOR)
    assert validate_candidate(
        graph, ACTOR, "THOL", sequence_context=_word().step(1)
    ).allowed

    stage = _run(graph)

    assert (stage.glyph, stage.schedule, stage.nodes_processed) == (
        "THOL",
        "two_phase_jacobi",
        1,
    )
    assert tuple(graph) == old_nodes + (CHILD,)
    assert _primary(graph, old_nodes) == primary
    assert tuple(graph.edges(data=True)) == edges
    assert graph.nodes[ROOT] == root_before
    assert get_attr(graph.nodes[ACTOR], ALIAS_DNFR) == old_pressure + (1 / (4 * pi)) * (
        1 / 8
    )
    child = graph.nodes[CHILD]
    # Independently bound the singleton-neighbor formula: ideal circular
    # variance is zero; the small allowance admits runtime trig roundoff.
    ideal_form = Q(3 / 8) * Q(0.3) + Q(1 / 4) * Q(1 / (2 * pi))
    assert abs(Q(get_attr(child, ALIAS_EPI)) - ideal_form) < Q(1, 2**44)
    exact_capacity = Q(7.3) * Q(0.95)
    represented_capacity = Q(float(exact_capacity))
    assert represented_capacity != exact_capacity
    assert Q(get_attr(child, ALIAS_VF)) == represented_capacity
    assert get_attr(child, ALIAS_THETA) == 0
    assert get_attr(child, ALIAS_DNFR) == 0
    assert graph.degree(CHILD) == 0
    assert child["parent_node"] == ACTOR
    assert child["hierarchy_level"] == child["_bifurcation_level"] == 2
    assert child["_hierarchy_path"] == [ROOT, ACTOR]
    assert graph.graph["hierarchy"] == {ROOT: [ACTOR], ACTOR: [CHILD]}
    actor = graph.nodes[ACTOR]
    assert actor["sub_nodes"] == [CHILD]
    assert tuple(actor["glyph_history"]) == ("OZ", "THOL")
    assert actor["_operator_step"] == 2
    assert node_debt(actor) == 0
    assert actor["epi_time_history"] == history
    assert graph.graph["_t"] == 12.0
    record = actor["sub_epis"][0]
    assert record["node_id"] == CHILD and record["timestamp"] == 2
    assert record["d2_epi"] == 1 / 8 and record["tau"] == 0.1
    assert record["bifurcation_level"] == 2
    assert record["hierarchy_path"] == [ROOT, ACTOR]
    assert record["metabolized"] is True
    assert record["network_signals"]["neighbor_count"] == 1
    assert record["network_signals"]["epi_gradient"] == 1 / 4
    assert actor["_mutation_context"]["destabilizer_operator"] == "dissonance"
    assert actor["_mutation_context"]["destabilizer_distance"] == 1
    assert actor["_thol_subepi_amplitude_alignment"] == 0.0
    assert child["glyph_history"] == []
    assert child["epi_history"] == [get_attr(child, ALIAS_EPI)]
    assert "epi_time_history" not in child and "_operator_step" not in child
    assert "_grammar_u2_debt" not in child
    assert not observe_structural_acceleration(graph, CHILD).available


def test_explicit_frozen_isolate_refresh_preserves_old_law_and_admits_remaining_sha():
    graph = _graph()
    old_nodes = tuple(graph)
    old_pressure = tuple(get_attr(graph.nodes[node], ALIAS_DNFR) for node in old_nodes)
    before = bound_relational_sine_support_exchange(graph, reference_model=MODEL)
    raw = []

    def handoff(live):
        raw.append((tuple(live), get_attr(live.nodes[ACTOR], ALIAS_DNFR)))
        assert tuple(live.nodes[ACTOR]["glyph_history"]) == ("OZ", "THOL")
        refresh_frozen_successor(live)

    _run(graph, refresh=handoff)

    assert len(raw) == 1 and raw[0][0] == old_nodes + (CHILD,)
    assert raw[0][1] > old_pressure[old_nodes.index(ACTOR)]
    assert (
        tuple(get_attr(graph.nodes[node], ALIAS_DNFR) for node in old_nodes)
        == old_pressure
    )
    successor = bound_relational_sine_frozen_isolates(graph, reference_model=MODEL)
    restriction = bound_relational_sine_support_exchange(
        graph.subgraph(old_nodes).copy(), reference_model=MODEL
    )
    assert restriction.form_rates == before.form_rates
    assert restriction.phase_rates == before.phase_rates
    assert successor.form_storage == before.form_storage
    assert successor.phase_storage == before.phase_storage
    index = successor.nodes.index(CHILD)
    assert successor.form_rates[index].lo == successor.form_rates[index].hi == 0
    assert successor.phase_rates[index].lo == successor.phase_rates[index].hi == 0
    with pytest.raises(ValueError, match="connected|isolat"):
        bound_relational_sine_support_exchange(graph, reference_model=MODEL)
    before_read = record_graph_state(graph)
    validate_silence(graph, ACTOR)
    assert validate_candidate(
        graph, ACTOR, "SHA", sequence_context=_word().step(2)
    ).allowed
    assert record_graph_state(graph) == before_read
    assert tuple(graph.nodes[ACTOR]["glyph_history"]) == ("OZ", "THOL")
    assert "latent" not in graph.nodes[ACTOR]


def test_connected_only_callback_rejects_and_rolls_back_nested_birth():
    graph = _graph()
    before = record_graph_state(graph)
    with pytest.raises(ValueError, match="connected|isolat"):
        _run(graph, refresh=refresh_pressure)
    assert record_graph_state(graph) == before


def test_identifier_collision_is_skipped_without_overwriting_existing_node():
    graph = _graph()
    graph.add_node(CHILD, marker="unrelated existing identity")
    _run(graph)
    next_child = ACTOR + "_sub_1"
    assert tuple(graph) == (ROOT, ACTOR, CHILD, next_child)
    assert graph.nodes[CHILD] == {"marker": "unrelated existing identity"}
    assert graph.nodes[ACTOR]["sub_nodes"] == [next_child]
    assert graph.nodes[ACTOR]["sub_epis"][0]["node_id"] == next_child
    assert graph.graph["hierarchy"][ACTOR] == [next_child]
    assert graph.nodes[next_child]["parent_node"] == ACTOR


def test_depth_limit_preserves_pressure_action_without_creating_a_descendant():
    graph = _graph()
    graph.graph["THOL_MAX_BIFURCATION_DEPTH"] = 1
    pressure = get_attr(graph.nodes[ACTOR], ALIAS_DNFR)
    _run(graph)
    assert tuple(graph) == (ROOT, ACTOR)
    assert get_attr(graph.nodes[ACTOR], ALIAS_DNFR) > pressure
    assert graph.graph["hierarchy"] == {ROOT: [ACTOR]}
    assert "sub_nodes" not in graph.nodes[ACTOR]
    assert graph.nodes[ACTOR]["_thol_no_bifurcation_expected"] is True
    assert graph.graph["thol_depth_limits"] == [
        {"node": ACTOR, "depth": 1, "max_depth": 1}
    ]


@pytest.mark.parametrize(
    "defect",
    (
        "conflicting_hierarchy",
        "cyclic_path",
        "boolean_depth",
        "invalid_primary_alias",
        "stale_physical_history",
        "insufficient_active_history",
        "zero_pressure",
        "inactive_capacity",
        "missing_recent_destabilizer",
        "forbidden_propagation",
        "invalid_child_bounds",
    ),
)
def test_invalid_live_conditions_reject_before_any_nested_commit(defect):
    graph = _graph()
    actor = graph.nodes[ACTOR]
    if defect == "conflicting_hierarchy":
        graph.graph["hierarchy"][ACTOR] = [ROOT]
    elif defect == "cyclic_path":
        actor["_hierarchy_path"].append(ACTOR)
    elif defect == "boolean_depth":
        actor["_bifurcation_level"] = True
    elif defect == "invalid_primary_alias":
        actor[ALIAS_VF[0]] = None
        actor[ALIAS_VF[1]] = 7.3
    elif defect == "stale_physical_history":
        actor["epi_time_history"][-1] = (12.0, 1 / 4)
        actor["epi_history"] = [0.0, 1 / 4, 3 / 8]
    elif defect == "insufficient_active_history":
        actor["epi_time_history"] = actor["epi_time_history"][-2:]
        actor["epi_history"] = [0.0, 1 / 4, 3 / 8]
    elif defect == "zero_pressure":
        set_attr(actor, ALIAS_DNFR, 0.0)
    elif defect == "inactive_capacity":
        set_attr(actor, ALIAS_VF, 0.0)
    elif defect == "missing_recent_destabilizer":
        actor["glyph_history"] = ["IL", "SHA", "NAV"]
    elif defect == "forbidden_propagation":
        graph.graph["THOL_PROPAGATION_ENABLED"] = True
    elif defect == "invalid_child_bounds":
        graph.graph["EPI_MAX"] = -0.1
    before = record_graph_state(graph)
    expected_error = (
        StructuralGrammarError
        if defect == "missing_recent_destabilizer"
        else OperatorPreconditionError
    )
    with pytest.raises(expected_error):
        _run(graph)
    assert record_graph_state(graph) == before
