"""Bounded synthetic admission controls for ordinary UM relation preservation.

Each graph is supplied explicitly and receives one actual IL preparation.
Corrupted aliases, grammar and callbacks below are labelled fault controls;
they are not acquired source history or a replay of the maintenance studies.
"""

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.config import inject_defaults
from tnfr.constants.aliases import ALIAS_EPI_KIND, ALIAS_THETA
from tnfr.errors import TNFRValueError
from tnfr.operators import Coherence, Coupling, Reception
from tnfr.operators._phase_gate import U3PhaseGateError
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import (
    STAGE_SCHEDULE_KEY,
    TWO_PHASE_JACOBI,
    execute_coupling_stage,
    execute_pointwise_stage,
)
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.utils import angle_diff
from tnfr.utils._structural_signature import structural_proof_signature

A, R, W = "target", "participant", "outer"


def _graph(*, phases=None, edges=None, graph_type=nx.Graph):
    graph = graph_type()
    phases = {A: 0.0, R: 0.25, W: 0.5} if phases is None else phases
    edges = ((A, R), (R, W)) if edges is None else edges
    for index, (node, phase) in enumerate(phases.items()):
        graph.add_node(
            node,
            EPI=0.5,
            nu_f=float(index + 1),
            theta=phase,
            delta_nfr=0.4,
            Si=0.0,
            **{ALIAS_EPI_KIND[0]: "form"},
        )
    for left, right in edges:
        graph.add_edge(left, right, weight=0.375, marker={"retained": [1, 2]})
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=71,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        IL_PRECONDITIONS={"warn_zero_dnfr": False, "warn_isolated": False},
        UM_FUNCTIONAL_LINKS=False,
        UM_BIDIRECTIONAL=True,
        UM_SYNC_VF=True,
        UM_STABILIZE_DNFR=True,
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


def _boundary_graph():
    return _graph(phases={A: 0.25, R: 0.0, W: -math.pi / 2 + 1 / 32})


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


def _apply(graph, *, route="stage", targets=(A,), **options):
    options.setdefault("preserve_relations", True)
    if route == "public":
        assert targets == (A,)
        return Coupling()(graph, A, **options)
    if route == "dispatcher":
        return execute_network_operator_stage(graph, Coupling(), targets, **options)
    return execute_coupling_stage(graph, Coupling(), targets, **options)


def _gap(graph, left, right):
    return abs(angle_diff(graph.nodes[left]["theta"], graph.nodes[right]["theta"]))


def _edge_state(graph):
    return structural_proof_signature(
        tuple(graph.edges(data=True)), opaque_references=(graph,)
    )


def _primary(graph):
    return tuple(
        (
            node,
            graph.nodes[node]["EPI"],
            graph.nodes[node]["nu_f"],
            graph.nodes[node]["delta_nfr"],
            graph.nodes[node]["theta"],
            read_grammar_state(graph.nodes[node]),
        )
        for node in graph
    )


@pytest.mark.parametrize("route", ["stage", "public", "dispatcher"])
def test_guarded_routes_share_actual_um_map_and_only_target_context(route):
    graph = _graph()
    # Builtin cyclic edge data belongs to the existing transaction domain.
    edge_data = graph.edges[A, R]
    edge_data["self"] = edge_data
    cycle = {}
    cycle["self"] = cycle
    edge_data["cycle"] = cycle
    legacy = deepcopy(graph)
    before = {node: read_grammar_state(graph.nodes[node]) for node in graph}
    old_edges = _edge_state(graph)
    result = _apply(graph, route=route)
    _apply(legacy, route=route, preserve_relations=False)
    if route == "public":
        assert result is None
    else:
        assert (result.schedule, result.nodes_processed) == (TWO_PHASE_JACOBI, 1)
    assert _primary(graph) == _primary(legacy)
    assert _edge_state(graph) == old_edges
    assert graph.nodes[A]["nu_f"] == 1.25
    assert graph.nodes[A]["theta"] == pytest.approx(1 / 16, rel=0.0, abs=1e-15)
    assert graph.nodes[R]["theta"] == pytest.approx(3 / 16, rel=0.0, abs=1e-15)
    assert 0.0 < graph.nodes[A]["delta_nfr"] < 0.2
    current = read_grammar_state(graph.nodes[A])
    assert current.operator_step == before[A].operator_step + 1
    assert current.history == (*before[A].history, "UM")
    for node in (R, W):
        assert read_grammar_state(graph.nodes[node]) == before[node]
    assert graph.graph[STAGE_SCHEDULE_KEY]["nodes_processed"] == 1
    assert graph.graph["_t"] == 0.0


def test_omitted_and_false_keep_legacy_collateral_counterexample():
    graph = _boundary_graph()
    explicit_false = deepcopy(graph)
    execute_coupling_stage(graph, Coupling(), (A,))
    _apply(explicit_false, preserve_relations=False)
    assert _primary(graph) == _primary(explicit_false)
    assert tuple(graph.edges(data=True)) == tuple(explicit_false.edges(data=True))
    assert _gap(graph, A, R) < math.pi / 2
    assert _gap(graph, R, W) > math.pi / 2
    assert read_grammar_state(graph.nodes[A]).history == ("IL", "UM")


def test_actual_merged_phase_crossing_rejects_and_restores_whole_invocation():
    graph = _boundary_graph()
    assert all(_gap(graph, left, right) < math.pi / 2 for left, right in graph.edges)
    before = _state(graph)
    with pytest.raises((TNFRValueError, U3PhaseGateError), match="U3|phase|relation"):
        _apply(graph)
    assert _state(graph) == before


def test_filtered_target_edge_still_belongs_to_initial_boundary():
    graph = _graph(phases={A: 0.0, R: 0.0, W: 2.0}, edges=((A, R), (A, W)))
    legacy = deepcopy(graph)
    before = _state(graph)
    with pytest.raises((TNFRValueError, U3PhaseGateError), match="U3|phase|relation"):
        _apply(graph)
    assert _state(graph) == before
    _apply(legacy, preserve_relations=False)
    assert read_grammar_state(legacy.nodes[A]).history == ("IL", "UM")


def test_zero_phase_displacement_does_not_erase_secondary_writer_boundary():
    graph = _graph(phases={A: 0.0, R: 0.0, W: 2.0})
    # Equal input phases give zero displacement with a strictly positive gain.
    before = _state(graph)
    with pytest.raises((TNFRValueError, U3PhaseGateError), match="U3|phase|relation"):
        _apply(graph)
    assert _state(graph) == before


def test_one_way_action_does_not_guard_unchanged_neighbor_outer_relation():
    graph = _graph(phases={A: 0.0, R: 0.25, W: 2.0})
    graph.graph["UM_BIDIRECTIONAL"] = False
    outer = deepcopy(graph.nodes[W])
    participant_phase = graph.nodes[R]["theta"]
    _apply(graph)
    assert graph.nodes[R]["theta"] == participant_phase
    assert graph.nodes[W] == outer
    assert _gap(graph, A, R) < math.pi / 2
    assert _gap(graph, R, W) > math.pi / 2


def test_finite_incompatible_remote_edge_is_not_global_phase_repair():
    graph = _graph(
        phases={A: 0.0, R: 0.25, "remote_a": 0.0, "remote_b": 2.0},
        edges=((A, R), ("remote_a", "remote_b")),
    )
    remote = deepcopy((graph.nodes["remote_a"], graph.nodes["remote_b"]))
    _apply(graph)
    assert (graph.nodes["remote_a"], graph.nodes["remote_b"]) == remote
    assert _gap(graph, "remote_a", "remote_b") > math.pi / 2


def test_joint_merge_cancellation_is_admitted_in_both_target_orders():
    second = "second_target"
    graph = _graph(
        phases={A: 0.5, second: -0.5, R: 0.0, W: -math.pi / 2 + 1 / 16},
        edges=((A, R), (second, R), (R, W)),
    )
    reversed_graph = deepcopy(graph)
    _apply(graph, targets=(A, second))
    _apply(reversed_graph, targets=(second, A))
    assert _primary(graph) == _primary(reversed_graph)
    assert abs(angle_diff(graph.nodes[R]["theta"], 0.0)) < 2e-15
    assert abs(angle_diff(graph.nodes[A]["theta"], 3 / 8)) < 2e-15
    assert abs(angle_diff(graph.nodes[second]["theta"], -3 / 8)) < 2e-15
    assert all(_gap(graph, left, right) <= math.pi / 2 for left, right in graph.edges)
    for node in (A, second):
        assert read_grammar_state(graph.nodes[node]).history == ("IL", "UM")
    for node in (R, W):
        assert read_grammar_state(graph.nodes[node]).history == ("IL",)


def test_closed_phase_limit_zero_weight_and_self_loop_are_preserved():
    graph = _graph(phases={A: 0.0, R: 0.0, W: 0.5})
    graph.graph["DELTA_PHI_MAX"] = 0.5
    graph.edges[R, W]["weight"] = 0.0
    graph.add_edge(R, R, weight=0.0, marker={"self": True})
    edges = deepcopy(tuple(graph.edges(data=True)))
    _apply(graph)
    assert _gap(graph, R, W) == 0.5
    assert tuple(graph.edges(data=True)) == edges


def test_unsupported_graph_types_reject_only_enabled_contract():
    for graph_type in (nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph):
        graph = _graph(graph_type=graph_type)
        before = _state(graph)
        with pytest.raises(ValueError, match="undirected non-multigraph"):
            _apply(graph)
        assert _state(graph) == before
        _apply(graph, preserve_relations=False)
        assert read_grammar_state(graph.nodes[A]).history == ("IL", "UM")


def test_invalid_boolean_requests_reject_before_each_public_entry_effects():
    for route, value in (("public", None), ("stage", 1), ("dispatcher", "true")):
        graph = _graph()
        before = _state(graph)
        with pytest.raises(OperatorPreconditionError, match="boolean"):
            _apply(graph, route=route, preserve_relations=value)
        assert _state(graph) == before


def test_enabled_contract_rejects_link_requests_empty_targets_and_other_glyphs():
    for mode in ("functional_links", "potential_partner", "empty", "other_glyph"):
        graph = _graph()
        options = {}
        if mode == "functional_links":
            graph.graph["UM_FUNCTIONAL_LINKS"] = True
        elif mode == "potential_partner":
            options["potential_partner"] = W
        elif mode == "empty":
            options["targets"] = ()
        before = _state(graph)
        with pytest.raises(ValueError, match="preserve_relations|ordinary|nonempty"):
            if mode == "other_glyph":
                execute_network_operator_stage(
                    graph, Reception(), (A,), preserve_relations=True
                )
            else:
                _apply(graph, **options)
        assert _state(graph) == before


def test_grammar_substitution_cannot_satisfy_requested_um_contract():
    graph = _graph()
    # Supplied corrupt-context control, not an acquired operator past.
    graph.nodes[A].update(glyph_history=["OZ"] * 3, _grammar_u2_debt=3)
    graph.graph["GRAMMAR_REJECTION_MODE"] = "fallback"
    before = _state(graph)
    with pytest.raises(ValueError, match="UM grammar admission|substituted"):
        _apply(graph)
    assert _state(graph) == before


def test_noncanonical_bindings_cannot_claim_guarded_stage(monkeypatch):
    import tnfr.operators.coupling as owner

    canonical = owner.Coupling

    class Derived(canonical):
        pass

    for mode in ("subclass", "execute", "class_alias"):
        graph = _graph()
        before = _state(graph)
        with monkeypatch.context() as patch:
            operator = Derived() if mode in {"subclass", "class_alias"} else canonical()
            if mode == "execute":
                patch.setattr(canonical, "_execute", lambda self, *args, **kw: None)
            elif mode == "class_alias":
                patch.setattr(owner, "Coupling", Derived)
            with pytest.raises(ValueError, match="canonical"):
                execute_coupling_stage(graph, operator, (A,), preserve_relations=True)
        assert _state(graph) == before


def test_invalid_authoritative_boundary_phases_do_not_use_fallback_alias():
    for value in (None, True, math.nan, math.inf):
        graph = _graph()
        graph.nodes[W][ALIAS_THETA[0]] = value
        graph.nodes[W][ALIAS_THETA[1]] = 0.5
        before = _state(graph)
        with pytest.raises(
            (TNFRValueError, U3PhaseGateError), match="phase|finite|boolean"
        ):
            _apply(graph)
        assert _state(graph) == before


def test_postcommit_monitor_boundary_corruption_rolls_back_actual_prefix():
    graph = _graph()
    observed = []

    class Monitor:
        def before_operator(self, snapshot, node):
            assert node == A

        def after_operator(self, live, node, operator_name):
            assert operator_name == "coupling"
            observed.append(read_grammar_state(live.nodes[node]).history)
            live.nodes[W]["theta"] = math.pi

    graph.graph["integrity_monitor"] = Monitor()
    before = _state(graph)
    with pytest.raises(U3PhaseGateError, match="U3"):
        _apply(graph)
    assert observed == [("IL", "UM")]
    assert _state(graph) == before


def test_refresh_cannot_widen_original_limit_or_shrink_original_write_set():
    graph = _graph(phases={A: 0.0, R: 0.0, W: 0.0})
    graph.graph.update(DELTA_PHI_MAX=0.5, UM_MAX_PHASE_DIFF=0.5)
    before = _state(graph)

    def refresh(live):
        assert read_grammar_state(live.nodes[A]).history[-1] == "UM"
        live.graph.update(
            DELTA_PHI_MAX=1.0, UM_MAX_PHASE_DIFF=1.0, UM_BIDIRECTIONAL=False
        )
        live.nodes[W]["theta"] = 0.75

    with pytest.raises(U3PhaseGateError, match="U3"):
        _apply(graph, compute_delta_nfr=refresh)
    assert _state(graph) == before


def test_stricter_live_limit_applies_to_returned_boundary():
    graph = _graph(phases={A: 0.0, R: 0.0, W: 0.25})
    before = _state(graph)

    def refresh(live):
        live.graph["UM_MAX_PHASE_DIFF"] = 0.125

    with pytest.raises(U3PhaseGateError, match="U3"):
        _apply(graph, compute_delta_nfr=refresh)
    assert _state(graph) == before


def test_late_support_or_edge_data_changes_cannot_certify_original_boundary():
    for mutation in ("add_edge", "remove_edge", "edge_data", "node"):
        graph = _graph()
        before = _state(graph)

        def refresh(live):
            assert read_grammar_state(live.nodes[A]).history[-1] == "UM"
            if mutation == "add_edge":
                live.add_edge(A, W, weight=0.0)
            elif mutation == "remove_edge":
                live.remove_edge(R, W)
            elif mutation == "edge_data":
                live.edges[R, W]["marker"]["retained"].append(3)
            else:
                live.add_node("new_identity", theta=0.0)

        with pytest.raises(RuntimeError, match="support|edge data|identit"):
            _apply(graph, compute_delta_nfr=refresh)
        assert _state(graph) == before


def test_explicit_late_failure_preserves_original_error_and_complete_state():
    graph = _graph()
    before = _state(graph)
    observed = []

    def refresh(live):
        observed.append(live.nodes[A]["nu_f"])
        assert read_grammar_state(live.nodes[A]).history == ("IL", "UM")
        raise RuntimeError("declared late refresh failure")

    with pytest.raises(RuntimeError, match="declared late refresh failure"):
        _apply(graph, compute_delta_nfr=refresh)
    assert observed == [1.25]
    assert _state(graph) == before


def test_preflight_changes_do_not_replace_original_support_or_limit(monkeypatch):
    from tnfr.operators import network_stage as owner

    real_preflight = owner._preflight_two_phase_stage
    for mutation in ("support", "limit"):
        graph = _graph()
        graph.graph["DELTA_PHI_MAX"] = 0.5
        before = _state(graph)

        def preflight(live, *args, **kwargs):
            result = real_preflight(live, *args, **kwargs)
            # Deliberate private-boundary fault; no replacement operator is run.
            if mutation == "support":
                live.edges[R, W]["marker"]["retained"].append(3)
            else:
                live.graph["DELTA_PHI_MAX"] = 1.0
            return result

        with monkeypatch.context() as patch:
            patch.setattr(owner, "_preflight_two_phase_stage", preflight)
            with pytest.raises(RuntimeError, match="support|edge data|source"):
                _apply(graph)
        assert _state(graph) == before


def test_stricter_still_admissible_live_limit_is_not_blanket_rejected():
    graph = _graph(phases={A: 0.0, R: 0.0, W: 0.25})

    def refresh(live):
        live.graph["UM_MAX_PHASE_DIFF"] = 0.5

    _apply(graph, compute_delta_nfr=refresh)
    assert _gap(graph, R, W) == 0.25
    assert graph.graph["UM_MAX_PHASE_DIFF"] == 0.5
    assert read_grammar_state(graph.nodes[A]).history == ("IL", "UM")
