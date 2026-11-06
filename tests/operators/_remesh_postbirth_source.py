"""New finite acquired source shared only by the post-birth REMESH contracts.

This module performs no work at import time and imports no historical producer.
Sampling occasions and the operator word are supplied, not autonomously derived.
"""

from fractions import Fraction as Q

import networkx as nx

from tnfr.alias import get_attr, set_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_D2EPI, ALIAS_DEPI, ALIAS_DNFR, ALIAS_EPI
from tnfr.dynamics.remesh_history import append_remesh_epi_history_snapshot
from tnfr.operators.birth_access_coupling import execute_birth_access_coupling
from tnfr.operators.coherence import Coherence
from tnfr.operators.dissonance import Dissonance
from tnfr.operators.emission_episode import execute_emission_episode
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import (
    execute_dissonance_stage,
    execute_pointwise_stage,
)
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.operators.self_organization_selection import (
    execute_eligible_self_organization_stage,
)
from tnfr.types import require_finite_real_scalar_epi


def form(graph, node):
    """Read the admitted signed scalar, including its uniform BEPI form."""
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


def build_prebirth_source():
    """Acquire actual parent/peer AL trajectories without delayed memory."""
    parent, peer = "parent", "peer"
    nodes = (parent, peer)
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
        BIFURCATION_THRESHOLD_TAU=0.5,
        THOL_METABOLIC_ENABLED=False,
        THOL_PROPAGATION_ENABLED=False,
        UM_FUNCTIONAL_LINKS=True,
        UM_BIDIRECTIONAL=True,
        UM_SYNC_VF=True,
        UM_STABILIZE_DNFR=True,
        UM_COMPAT_THRESHOLD=0.5,
        REMESH_TAU_LOCAL=1,
        REMESH_TAU_GLOBAL=1,
        REMESH_ALPHA=0.5,
        REMESH_ALPHA_HARD=True,
        REMESH_LOG_EVENTS=False,
        history={},
    )
    graph.graph["GLYPH_FACTORS"].update(
        IL_dnfr_factor=0.5,
        OZ_dnfr_factor=2.0,
        THOL_accel=0.125,
        UM_theta_push=0.5,
        UM_vf_sync=0.25,
        UM_dnfr_reduction=0.25,
        AL_boost=0.125,
        EN_mix=0.5,
        RA_epi_diff=0.5,
        RA_vf_amplification=0.125,
    )
    for node in nodes:
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
        assert read_grammar_state(graph.nodes[node]).history == ()
    _prepare_pressure(graph, nodes, 2.0)
    first = execute_emission_episode(
        graph, nodes, duration=0.125, method="euler", n_jobs=1
    )
    assert first.exact_rates == (Q(1, 2), Q(1, 2))
    assert first.exact_epi_after == (Q(5, 16), Q(5, 16))
    _prepare_pressure(graph, (parent,), 2.5)
    second = execute_emission_episode(
        graph, nodes, duration=0.125, method="euler", n_jobs=1
    )
    assert second.exact_rates == (Q(5, 8), Q(1, 2))
    assert second.exact_epi_after == (Q(25, 64), Q(3, 8))
    for result in (first, second):
        assert result.flow_result.events == ()
        assert result.flow_result.pressure_refresh_callback_invocations == 0
        assert (
            result.flow_result.all_positive_flow_intervals_binary64_held_pressure_identified
        )
    for node, endpoint, acceleration, word in (
        (parent, Q(25, 64), 1.0, ("IL", "OZ", "AL", "IL", "OZ", "AL")),
        (peer, Q(3, 8), 0.0, ("IL", "OZ", "AL", "AL")),
    ):
        assert form(graph, node) == endpoint
        assert tuple(graph.nodes[node]["epi_time_history"]) == (
            (0.0, 0.25),
            (0.125, 0.3125),
            (0.25, float(endpoint)),
        )
        observed = observe_structural_acceleration(graph, node)
        assert observed.available and observed.source == "epi_time_history"
        assert observed.value == acceleration
        grammar = read_grammar_state(graph.nodes[node])
        assert grammar.history == word and grammar.operator_step == len(word)
        assert grammar.prior_coherence and grammar.u2_debt == 1
        assert graph.nodes[node]["_structural_lineage"]["activation_count"] == 2
    assert graph.graph["_t"] == 0.25 and "_epi_hist" not in graph.graph
    return graph, parent, peer


def build_source(*, capture_prebirth=False):
    """Acquire birth/contact and each participant's own continuation context."""
    graph, parent, peer = build_prebirth_source()
    if capture_prebirth:
        captured = append_remesh_epi_history_snapshot(graph)
        assert tuple(node for node, _ in captured.snapshot_items) == (parent, peer)
        assert captured.history_length_before == 0
        assert captured.history_length_after == 1
    birth = execute_eligible_self_organization_stage(
        graph, execution_kwargs={"tau": 0.5, "validate_preconditions": True}
    )
    assert birth.eligibility.eligible_nodes == (parent,)
    ((actual_parent, child),) = birth.parent_children
    assert actual_parent == parent and graph.degree(child) == 0
    assert tuple(graph) == (parent, peer, child)
    assert form(graph, child) == Q(15, 128)
    assert read_grammar_state(graph.nodes[child]).history == ()
    assert "epi_time_history" not in graph.nodes[child]
    assert graph.nodes[child]["epi_history"] == [15 / 128]
    assert "_emission_activated" not in graph.nodes[child]
    assert _pressure(graph, parent) == Q(3, 4)

    contact = execute_birth_access_coupling(graph, parent)
    assert contact.executed and contact.access.selected_partner == child
    assert contact.access.policy == "unique_isolated_birth_child"
    assert contact.stage_result.nodes_processed == 1
    assert graph.has_edge(parent, child)
    assert read_grammar_state(graph.nodes[child]).history == ()
    assert _pressure(graph, parent) == Q(9, 16)
    for _ in range(3):
        _cohere(graph, (parent,))
    for _ in range(2):
        _cohere(graph, (peer,))
    _cohere(graph, (child,))
    assert tuple(form(graph, node) for node in graph) == (
        Q(25, 64),
        Q(3, 8),
        Q(15, 128),
    )
    assert tuple(_pressure(graph, node) for node in graph) == (Q(9, 128), Q(1, 8), Q(0))
    for node, count in ((parent, 11), (peer, 6), (child, 1)):
        grammar = read_grammar_state(graph.nodes[node])
        assert grammar.operator_step == count and grammar.prior_coherence
    assert read_grammar_state(graph.nodes[child]).history == ("IL",)
    assert graph.graph["_t"] == 0.25
    assert ("_epi_hist" in graph.graph) is capture_prebirth
    if capture_prebirth:
        assert len(graph.graph["_epi_hist"]) == 1
        assert child not in graph.graph["_epi_hist"][0]
    return graph, parent, peer, child
