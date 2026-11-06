"""Bounded request, collateral U3 and rollback controls for connected UM.

All histories in the ordinary fixture are acquired through actual IL. The
explicit corrupted grammar controls below are admission fixtures, not claims
of an acquired organization. No source trajectory or old producer is replayed.
"""

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.config import inject_defaults
from tnfr.constants.aliases import ALIAS_THETA
from tnfr.errors import TNFRValueError
from tnfr.operators import Coherence, Coupling, Reception
from tnfr.operators._coupling_potential import resolve_potential_coupling
from tnfr.operators._coupling_stage_kernel import propose_coupling_stage
from tnfr.operators._phase_gate import U3PhaseGateError
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors
from tnfr.operators.grammar_application import enforce_canonical_grammar
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.grammar_types import StructuralGrammarError
from tnfr.operators.network_stage import (
    STAGE_SCHEDULE_KEY,
    TWO_PHASE_JACOBI,
    execute_coupling_stage,
    execute_pointwise_stage,
)
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.types import Glyph
from tnfr.utils._structural_signature import structural_proof_signature

A, LEFT, B, RIGHT = "a", "left", "b", "right"
SCOPE = "disjoint_components"


def _graph(graph_type=nx.Graph):
    graph = graph_type()
    for node, capacity in ((A, 1.0), (LEFT, 3.0), (B, 5.0), (RIGHT, 7.0)):
        graph.add_node(
            node,
            EPI=0.5,
            nu_f=capacity,
            theta=0.0,
            delta_nfr=0.4,
            Si=0.0,
            epi_kind="form",
        )
    graph.add_edge(A, LEFT, weight=0.37, marker={"retained": [1, 2]})
    graph.add_edge(B, RIGHT, weight=0.71, length=2.0)
    inject_defaults(graph)
    graph.graph.update(
        RANDOM_SEED=71,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        IL_PRECONDITIONS={"warn_zero_dnfr": False},
        UM_FUNCTIONAL_LINKS=True,
        UM_BIDIRECTIONAL=True,
        UM_SYNC_VF=True,
        UM_STABILIZE_DNFR=True,
        UM_COMPAT_THRESHOLD=0.5,
    )
    graph.graph["GLYPH_FACTORS"].update(
        IL_dnfr_factor=0.5,
        UM_theta_push=0.5,
        UM_vf_sync=0.25,
        UM_dnfr_reduction=0.25,
    )
    execute_pointwise_stage(
        graph,
        Coherence(),
        tuple(graph),
        phase_locking_coefficient=0.0,
        require_exact_operator=True,
    )
    return graph


def _state(graph):
    edges = (
        tuple(graph.edges(keys=True, data=True))
        if graph.is_multigraph()
        else tuple(graph.edges(data=True))
    )
    return structural_proof_signature(
        (
            graph.graph,
            tuple(graph.nodes(data=True)),
            edges,
            hasattr(graph, "_last_operator_applied"),
            getattr(graph, "_last_operator_applied", None),
        ),
        opaque_references=(graph,),
    )


def _execute(graph, route="stage", **kwargs):
    options = {"potential_partner": B, "potential_scope": SCOPE, **kwargs}
    if route == "public":
        return Coupling()(graph, A, **options)
    if route == "dispatcher":
        return execute_network_operator_stage(graph, Coupling(), (A,), **options)
    return execute_coupling_stage(graph, Coupling(), (A,), **options)


def _factors(graph):
    return resolve_runtime_operator_factors(
        graph.graph["GLYPH_FACTORS"], Glyph.UM, graph.graph
    )


@pytest.mark.parametrize("route", ["public", "stage", "dispatcher"])
def test_supplied_connected_participant_uses_real_um_and_only_target_context(route):
    graph = _graph()
    old_edges = {
        (left, right): deepcopy(data) for left, right, data in graph.edges(data=True)
    }
    before_contexts = {node: read_grammar_state(graph.nodes[node]) for node in graph}
    result = _execute(graph, route)
    if route == "public":
        assert result is None
    else:
        assert (result.schedule, result.nodes_processed) == (TWO_PHASE_JACOBI, 1)
    assert graph.has_edge(A, B) and graph.edges[A, B] == {"weight": 1.0}
    for edge, data in old_edges.items():
        assert graph.edges[edge] == data
    assert graph.nodes[A]["nu_f"] == 1.75  # 1+.25*((3+5)/2-1).
    assert graph.nodes[A]["delta_nfr"] == pytest.approx(0.15)
    assert all(graph.nodes[node]["EPI"] == 0.5 for node in graph)
    assert all(graph.nodes[node]["theta"] == 0.0 for node in graph)
    current = read_grammar_state(graph.nodes[A])
    assert current.history == (*before_contexts[A].history, "UM")
    assert current.operator_step == before_contexts[A].operator_step + 1
    for node in (LEFT, B, RIGHT):
        assert read_grammar_state(graph.nodes[node]) == before_contexts[node]
    assert graph.graph[STAGE_SCHEDULE_KEY]["potential_pair"] == {
        "target": A,
        "partner": B,
        "origin": "explicit_request",
        "scope": SCOPE,
    }


@pytest.mark.parametrize("bidirectional", [False, True])
def test_new_scope_reuses_same_map_as_separately_realized_reference(bidirectional):
    graph = _graph()
    graph.graph["UM_BIDIRECTIONAL"] = bidirectional
    graph.nodes[LEFT]["theta"] = -0.25
    graph.nodes[B]["theta"] = 0.25
    graph.nodes[RIGHT]["theta"] = 0.25
    before = _state(graph)
    reference = deepcopy(graph)
    reference.add_edge(A, B)
    reference.graph["UM_FUNCTIONAL_LINKS"] = False
    proposed = propose_coupling_stage(
        graph,
        (A,),
        _factors(graph),
        potential_partner=B,
        potential_scope=SCOPE,
    )
    realized = propose_coupling_stage(reference, (A,), _factors(reference))
    assert proposed.node_updates == realized.node_updates
    assert (
        proposed.target_proposals[0].phase_proposals
        == realized.target_proposals[0].phase_proposals
    )
    assert proposed.target_proposals[0].compatible_neighbors == (LEFT, B)
    assert proposed.target_proposals[0].affected_relations == (
        ((A, LEFT), (B, RIGHT), (A, B)) if bidirectional else ((A, LEFT), (A, B))
    )
    assert _state(graph) == before


def test_default_isolated_request_stays_unchanged_and_rejects_connected_partner():
    graph = _graph()
    before = _state(graph)
    with pytest.raises(OperatorPreconditionError, match="isolated"):
        Coupling()(graph, A, potential_partner=B)
    assert _state(graph) == before
    graph.remove_edge(B, RIGHT)
    explicit_default = deepcopy(graph)
    Coupling()(graph, A, potential_partner=B)
    Coupling()(explicit_default, A, potential_partner=B, potential_scope="isolated")
    assert tuple(graph.nodes(data=True)) == tuple(explicit_default.nodes(data=True))
    assert tuple(graph.edges(data=True)) == tuple(explicit_default.edges(data=True))
    assert graph.graph[STAGE_SCHEDULE_KEY]["potential_pair"] == {
        "target": A,
        "partner": B,
        "origin": "explicit_request",
    }


@pytest.mark.parametrize("route", ["public", "stage", "dispatcher"])
@pytest.mark.parametrize(
    "scope,error", [(None, TypeError), (False, TypeError), ("connected", ValueError)]
)
def test_scope_admission_precedes_graph_effects(route, scope, error):
    graph = _graph()
    before = _state(graph)
    with pytest.raises(error):
        _execute(graph, route, potential_scope=scope)
    assert _state(graph) == before


@pytest.mark.parametrize("route", ["public", "stage", "dispatcher"])
def test_nondefault_scope_requires_partner_even_before_ordinary_path(route):
    graph = _graph()
    before = _state(graph)
    with pytest.raises(ValueError, match="requires potential_partner"):
        _execute(graph, route, potential_partner=None)
    assert _state(graph) == before


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_new_scope_rejects_unsupported_support_without_changing_legacy_scope(
    graph_type,
):
    graph = _graph(graph_type)
    before = _state(graph)
    with pytest.raises(OperatorPreconditionError, match="undirected non-multigraph"):
        _execute(graph)
    assert _state(graph) == before
    graph.remove_edge(B, RIGHT)
    Coupling()(graph, A, potential_partner=B)
    assert graph.has_edge(A, B)


@pytest.mark.parametrize(
    "invalid", ["same_component", "existing_link", "singleton", "loop_only"]
)
def test_complete_components_and_absent_link_are_live_admission(invalid):
    graph = _graph()
    if invalid == "same_component":
        graph.add_edge(LEFT, RIGHT)
    elif invalid == "existing_link":
        graph.add_edge(A, B)
    else:
        graph.remove_edge(B, RIGHT)
        if invalid == "loop_only":
            graph.add_edge(B, B)
    before = _state(graph)
    with pytest.raises(OperatorPreconditionError, match="distinct components"):
        _execute(graph)
    assert _state(graph) == before


def test_zero_weight_support_and_loops_in_nontrivial_components_are_admitted():
    graph = _graph()
    graph.edges[A, LEFT]["weight"] = 0.0
    graph.edges[B, RIGHT]["weight"] = 0.0
    graph.add_edge(B, B, weight=0.0, marker="retained")
    _execute(graph)
    assert graph.has_edge(A, B)
    assert graph.edges[B, B] == {"weight": 0.0, "marker": "retained"}


@pytest.mark.parametrize("location", ["partner", "moved_old_neighbor"])
def test_initially_admitted_collateral_relation_can_fail_after_actual_map(location):
    graph = _graph()
    if location == "partner":
        graph.nodes[B]["theta"] = 1.0
        graph.nodes[RIGHT]["theta"] = 2.5
    else:
        graph.nodes[LEFT]["theta"] = 1.0
        graph.add_node("outer", theta=2.5)
        graph.add_edge(LEFT, "outer", weight=0.37)
    selection = resolve_potential_coupling(graph, A, B, potential_scope=SCOPE)
    assert selection.neighbors == (LEFT, B)
    before = _state(graph)
    with pytest.raises(TNFRValueError, match="final U3 gate"):
        _execute(graph)
    assert _state(graph) == before


@pytest.mark.parametrize("phase", [math.pi, None, True, math.nan, math.inf])
def test_initial_collateral_phase_is_mandatory_and_authoritative(phase):
    graph = _graph()
    graph.nodes[RIGHT][ALIAS_THETA[0]] = phase
    graph.nodes[RIGHT][ALIAS_THETA[1]] = 0.0
    before = _state(graph)
    with pytest.raises(OperatorPreconditionError):
        _execute(graph)
    assert _state(graph) == before


def test_unidirectional_request_does_not_consume_unchanged_partner_outer_phase():
    graph = _graph()
    graph.graph["UM_BIDIRECTIONAL"] = False
    graph.nodes[RIGHT]["theta"] = math.nan
    graph.nodes[RIGHT]["EPI"] = "not a consumed source"
    graph.add_node("remote", theta=math.nan)
    _execute(graph)
    assert graph.has_edge(A, B)
    assert math.isnan(graph.nodes[RIGHT]["theta"])


def test_request_grammar_uses_new_scope_without_waiving_target_debt():
    graph = _graph()
    assert validate_candidate(
        graph, A, "UM", potential_partner=B, potential_scope=SCOPE
    ).allowed
    assert (
        enforce_canonical_grammar(
            graph, A, "UM", potential_partner=B, potential_scope=SCOPE
        )
        == "UM"
    )
    # Deliberately supplied invalid-context control, not an acquired past.
    graph.nodes[A].update(glyph_history=["OZ"] * 3, _grammar_u2_debt=3)
    before = _state(graph)
    with pytest.raises(StructuralGrammarError) as error:
        _execute(graph)
    assert error.value.rule == "U2"
    assert _state(graph) == before


def test_secondary_partner_history_is_not_a_second_um_admission_or_occurrence():
    graph = _graph()
    graph.nodes[B].update(glyph_history=["OZ"] * 3, _grammar_u2_debt=3)
    partner = read_grammar_state(graph.nodes[B])
    _execute(graph)
    assert read_grammar_state(graph.nodes[B]) == partner


def test_new_scope_requires_one_canonical_um_target_and_never_other_glyphs():
    graph = _graph()
    before = _state(graph)
    with pytest.raises(ValueError, match="exactly one"):
        execute_coupling_stage(
            graph, Coupling(), (A, LEFT), potential_partner=B, potential_scope=SCOPE
        )
    with pytest.raises(ValueError, match="only for Coupling"):
        execute_network_operator_stage(
            graph, Reception(), (A,), potential_partner=B, potential_scope=SCOPE
        )
    with pytest.raises(ValueError, match="only for UM"):
        validate_candidate(graph, A, "EN", potential_partner=B, potential_scope=SCOPE)
    assert _state(graph) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "phase",
        "old_edge",
        "new_edge",
        "remove_edge",
        "add_edge",
        "node",
        "widen_limit",
        "tighten_limit",
        "change_flags",
    ],
)
def test_late_refresh_cannot_invalidate_original_phase_boundary_or_support(mutation):
    graph = _graph()
    graph.graph["DELTA_PHI_MAX"] = 0.5
    before = _state(graph)

    def refresh(live):
        if mutation == "phase":
            live.nodes[RIGHT]["theta"] = 2.0
        elif mutation == "old_edge":
            live.edges[A, LEFT]["marker"]["retained"].append(3)
        elif mutation == "new_edge":
            live.edges[A, B]["weight"] = 0.1
        elif mutation == "remove_edge":
            live.remove_edge(B, RIGHT)
        elif mutation == "add_edge":
            live.add_edge(LEFT, RIGHT)
        elif mutation == "node":
            live.add_node("new identity")
        elif mutation == "widen_limit":
            live.graph.update(DELTA_PHI_MAX=1.0, UM_MAX_PHASE_DIFF=1.0)
            live.nodes[B]["theta"] = live.nodes[RIGHT]["theta"] = 0.75
        elif mutation == "tighten_limit":
            live.graph["DELTA_PHI_MAX"] = 0.1
            live.nodes[B]["theta"] = live.nodes[RIGHT]["theta"] = 0.2
        else:
            live.graph["UM_BIDIRECTIONAL"] = False
            live.nodes[RIGHT]["theta"] = 2.0

    with pytest.raises((RuntimeError, U3PhaseGateError)):
        _execute(graph, compute_delta_nfr=refresh)
    assert _state(graph) == before


def test_late_explicit_failure_preserves_complete_entry_state():
    graph = _graph()
    before = _state(graph)

    def fail_after_commit(live):
        assert live.has_edge(A, B)
        assert read_grammar_state(live.nodes[A]).history[-1] == "UM"
        raise RuntimeError("actual UM prefix failed its refresh")

    with pytest.raises(RuntimeError, match="actual UM prefix"):
        _execute(graph, compute_delta_nfr=fail_after_commit)
    assert _state(graph) == before
