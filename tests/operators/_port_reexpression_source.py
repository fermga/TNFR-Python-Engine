"""Pure helper definitions for actual acquired port-source preparation.

No engine call occurs at import. Inputs, occasions and UM candidates are supplied;
all successful source history, child form and support are acquired by operators.
"""

from fractions import Fraction as Q

import networkx as nx

from tnfr.alias import get_attr, set_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.operators.birth_access_coupling import execute_birth_access_coupling
from tnfr.operators.coherence import Coherence
from tnfr.operators.coupling import Coupling
from tnfr.operators.dissonance import Dissonance
from tnfr.operators.emission import Emission
from tnfr.operators.emission_episode import execute_emission_episode
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import (
    execute_coupling_stage,
    execute_dissonance_stage,
    execute_neighbor_stage,
    execute_pointwise_stage,
)
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.operators.reception import Reception
from tnfr.operators.self_organization_selection import (
    execute_eligible_self_organization_stage,
)
from tnfr.types import require_finite_real_scalar_epi

A, B, U, V, R, W = "A", "B", "input-U", "input-V", "receiver", "observer"
INITIAL_NODES = (A, B, U, V, R, W)
INITIAL_EDGES = ((A, B), (U, A), (V, B), (A, R), (B, R), (R, W))


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


def _support(graph):
    return {frozenset((left, right)) for left, right in graph.edges}


def _held_channels(graph):
    return tuple(
        (
            node,
            get_attr(data, ALIAS_VF),
            get_attr(data, ALIAS_DNFR),
            get_attr(data, ALIAS_THETA),
        )
        for node, data in graph.nodes(data=True)
    )


def _cohere(graph, targets):
    execute_pointwise_stage(
        graph,
        Coherence(),
        targets,
        phase_locking_coefficient=0.0,
        require_exact_operator=True,
        validate_preconditions=True,
    )


def _dissonance(graph, targets, factor):
    graph.graph["GLYPH_FACTORS"]["OZ_dnfr_factor"] = factor
    execute_dissonance_stage(
        graph,
        Dissonance(),
        targets,
        propagate_to_network=False,
        validate_preconditions=True,
    )


def _emit(graph, node, boost):
    graph.graph["GLYPH_FACTORS"]["AL_boost"] = float(boost)
    before = _form(graph, node)
    Emission()(graph, node, validate_preconditions=True)
    assert _form(graph, node) == before + boost
    assert graph.graph["_t"] == 0.25


def _receive(graph, targets, sources):
    return execute_neighbor_stage(
        graph,
        Reception(),
        targets,
        source_nodes=sources,
        track_sources=False,
        validate_preconditions=True,
    )


def _source(*, refined):
    graph = nx.Graph()
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
        UM_CANDIDATE_COUNT=0,
        history={},
    )
    graph.graph["GLYPH_FACTORS"].update(
        IL_dnfr_factor=0.5,
        OZ_dnfr_factor=2.0,
        THOL_accel=0.125,
        UM_theta_push=0.5,
        UM_vf_sync=0.25,
        UM_dnfr_reduction=0.25,
        EN_mix=0.5,
        RA_epi_diff=0.5,
        RA_vf_amplification=0.125,
    )
    for node, epi in (
        (A, Q(1, 4)),
        (B, Q(1, 4)),
        (U, Q(1, 8)),
        (V, Q(1, 8)),
        (R, Q(1, 4)),
        (W, Q(3, 8)),
    ):
        graph.add_node(
            node,
            EPI=float(epi),
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.5 if node in (A, B) else 0.0,
            Si=0.0,
            epi_kind="form",
        )
        set_attr(graph.nodes[node], ALIAS_DEPI, 0.0)
        set_attr(graph.nodes[node], ALIAS_D2EPI, 0.0)
        assert read_grammar_state(graph.nodes[node]).history == ()
    for left, right in INITIAL_EDGES:
        graph.add_edge(left, right, weight=1.0)
    _cohere(graph, INITIAL_NODES)
    _dissonance(graph, (A, B), 2.0)
    first = execute_emission_episode(
        graph, (A, B), duration=0.125, method="euler", n_jobs=1
    )
    assert first.exact_rates == (Q(1, 2), Q(1, 2), Q(0), Q(0), Q(0), Q(0))
    assert first.exact_epi_after[:2] == (Q(5, 16), Q(5, 16))
    _cohere(graph, (A,))
    _dissonance(graph, (A,), 2.5)
    second = execute_emission_episode(
        graph, (A, B), duration=0.125, method="euler", n_jobs=1
    )
    assert second.exact_rates == (Q(5, 8), Q(1, 2), Q(0), Q(0), Q(0), Q(0))
    assert second.exact_epi_after[:2] == (Q(25, 64), Q(3, 8))
    for episode in (first, second):
        assert episode.flow_result.events == ()
        assert episode.flow_result.pressure_refresh_callback_invocations == 0
        assert (
            episode.flow_result.all_positive_flow_intervals_binary64_held_pressure_identified
        )
    for node, acceleration, count in ((A, 1.0, 6), (B, 0.0, 4)):
        observed = observe_structural_acceleration(graph, node)
        assert observed.available and observed.source == "epi_time_history"
        assert observed.value == acceleration
        assert len(graph.nodes[node]["epi_time_history"]) == 3
        assert read_grammar_state(graph.nodes[node]).operator_step == count
        assert graph.nodes[node]["_structural_lineage"]["activation_count"] == 2
    assert tuple(graph) == INITIAL_NODES and graph.graph["_t"] == 0.25
    assert tuple(_form(graph, n) for n in (U, V, R, W)) == (
        Q(1, 8),
        Q(1, 8),
        Q(1, 4),
        Q(3, 8),
    )
    child = None
    group = (A,)
    if refined:
        birth = execute_eligible_self_organization_stage(
            graph, execution_kwargs={"tau": 0.5, "validate_preconditions": True}
        )
        assert birth.eligibility.eligible_nodes == (A,)
        ((parent, child),) = birth.parent_children
        assert parent == A and graph.degree(child) == 0
        assert _form(graph, child) == Q(15, 128)
        assert graph.nodes[child]["epi_history"] == [15 / 128]
        assert "epi_time_history" not in graph.nodes[child]
        assert "_structural_lineage" not in graph.nodes[child]
        assert read_grammar_state(graph.nodes[child]).history == ()
        contact = execute_birth_access_coupling(graph, A)
        assert contact.executed and contact.access.selected_partner == child
        assert read_grammar_state(graph.nodes[child]).history == ()
        assert _pressure(graph, A) == Q(9, 16)
        _cohere(graph, (child,))
        # Explicit ordinary functional-link candidates, not a disjoint scope.
        graph.graph["_node_sample"] = (U, R)
        linked = execute_coupling_stage(
            graph, Coupling(), (child,), validate_preconditions=True
        )
        assert linked.nodes_processed == 1 and linked.glyph == "UM"
        assert _support(graph) == {
            *map(frozenset, INITIAL_EDGES),
            frozenset((A, child)),
            frozenset((U, child)),
            frozenset((R, child)),
        }
        assert tuple(graph.neighbors(W)) == (R,)
        assert all(graph.edges[child, n]["weight"] >= 0.75 for n in (A, U, R))
        child_input = _receive(graph, (child,), (A,))
        assert _form(graph, child) == Q(65, 256)
        assert (
            get_attr(graph.nodes[child], ALIAS_EPI_KIND, strict=True, conv=str)
            == "form"
        )
        assert child_input.reception_observations[0].target_epi_kind_after == "form"
        assert read_grammar_state(graph.nodes[child]).history == ("IL", "UM", "EN")
        assert "epi_time_history" not in graph.nodes[child]
        group = (A, child)
    for _ in range(3):
        _cohere(graph, (A,))
    for _ in range(2):
        _cohere(graph, (B,))
    if refined:
        assert _form(graph, A) == Q(25, 64) < Q(1, 2)
        _emit(graph, A, Q(35, 256))
        assert _form(graph, A) == Q(135, 256)
        assert (_form(graph, A) - _form(graph, child)) / 2 == Q(35, 256)
        assert read_grammar_state(graph.nodes[child]).operator_step == 3
        assert "_structural_lineage" not in graph.nodes[child]
    assert sum((_form(graph, n) for n in group), Q(0)) / len(group) == Q(25, 64)
    assert _form(graph, B) == Q(3, 8)
    assert _pressure(graph, A) == (Q(9, 128) if refined else Q(5, 64))
    assert _pressure(graph, B) == Q(1, 8)
    assert tuple(graph.neighbors(W)) == (R,)
    assert all(get_attr(data, ALIAS_THETA) == 0.0 for _, data in graph.nodes(data=True))
    assert graph.graph["_t"] == 0.25
    return graph, group, child
