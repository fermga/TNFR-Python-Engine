"""Actual retained synthesis and finite delayed boundary-response controls.

Equal exported form and one response are not equality of complete state. All
occasions and initial states are declared; no member deletion is proposed.
"""

from copy import deepcopy
from fractions import Fraction as Q

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
from tnfr.operators.birth_access_coupling import execute_birth_access_coupling
from tnfr.operators.coherence import Coherence
from tnfr.operators.contraction import Contraction
from tnfr.operators.dissonance import Dissonance
from tnfr.operators.emission_episode import execute_emission_episode
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import (
    execute_dissonance_stage,
    execute_pointwise_stage,
)
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.operators.preconditions.reception import validate_reception_strict
from tnfr.operators.reception import Reception
from tnfr.operators.resonance import Resonance
from tnfr.operators.retained_organization import execute_retained_scale_stage
from tnfr.operators.retained_synthesis import (
    capture_retained_pair,
    execute_retained_synthesis_stage,
)
from tnfr.operators.self_organization_selection import (
    execute_eligible_self_organization_stage,
)
from tnfr.types import require_finite_real_scalar_epi

CORE, MEMBER, PEER = "core", "member", "peer"


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


def _graph():
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
        NUL_scale=0.5,
        EN_mix=0.5,
        RA_epi_diff=0.5,
        RA_vf_amplification=0.125,
    )
    # Derive the reciprocal from this scale, not the injected default scale.
    graph.graph["GLYPH_FACTORS"].pop("NUL_densification_factor")
    return graph


def _node(graph, node, form, pressure):
    graph.add_node(
        node,
        EPI=float(form),
        nu_f=1.0,
        theta=0.0,
        delta_nfr=float(pressure),
        Si=0.0,
        epi_kind="form",
    )
    set_attr(graph.nodes[node], ALIAS_DEPI, 0.0)
    set_attr(graph.nodes[node], ALIAS_D2EPI, 0.0)


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


def _edges(graph):
    return tuple(
        (left, right, deepcopy(data)) for left, right, data in graph.edges(data=True)
    )


def _fragment(graph, member, *, compound):
    pair = capture_retained_pair(graph, (CORE, member))
    old_edges = _edges(graph)
    old_peer = deepcopy(graph.nodes[PEER])
    old_nodes = tuple(graph)
    if compound:
        result = execute_retained_synthesis_stage(
            graph,
            pair,
            contraction_kwargs={"validate_preconditions": True},
            reception_kwargs={"validate_preconditions": True},
        )
        assert result.contraction_result.stage_result.nodes_processed == 2
        assert result.reception_result.nodes_processed == 1
        observed = result.reception_result.reception_observations[0]
        assert observed.node == CORE and observed.source_nodes == (member,)
        assert observed.neighbors == (member,)
        assert not observed.source_tracking_enabled
        assert result.after.members == (CORE, member)
    else:
        result = execute_retained_scale_stage(
            graph,
            pair.organization,
            Contraction(),
            execution_kwargs={"validate_preconditions": True},
        )
        assert result.stage_result.nodes_processed == 2
    assert tuple(graph) == old_nodes and _edges(graph) == old_edges
    assert graph.nodes[PEER] == old_peer
    assert all(get_attr(graph.nodes[node], ALIAS_THETA) == 0.0 for node in graph)


def _continue(graph, member, expected):
    """Actual RA(peer), ordinary EN(core), RA(peer), with all members retained."""
    old_member = deepcopy(graph.nodes[member])
    old_edges = _edges(graph)
    old_time = graph.graph["_t"]
    Resonance()(graph, PEER)
    first = _form(graph, PEER)
    assert first == expected[0]
    assert get_attr(graph.nodes[PEER], ALIAS_VF) == 9 / 8
    validate_reception_strict(graph, CORE)
    Reception()(graph, CORE, track_sources=False)
    middle = _form(graph, CORE)
    assert middle == expected[1]
    Resonance()(graph, PEER)
    last = _form(graph, PEER)
    assert last == expected[2]
    assert get_attr(graph.nodes[PEER], ALIAS_VF) == 81 / 64
    assert graph.nodes[member] == old_member
    assert _edges(graph) == old_edges and graph.graph["_t"] == old_time
    assert all(get_attr(graph.nodes[node], ALIAS_THETA) == 0.0 for node in graph)
    return first, middle, last


def _declared_response_source(core_form, member_form):
    graph = _graph()
    for node, form in ((CORE, core_form), (MEMBER, member_form), (PEER, Q(3, 8))):
        _node(graph, node, form, Q(1, 32))
    graph.add_edge(CORE, MEMBER, weight=1.0)
    graph.add_edge(PEER, CORE, weight=1.0)
    _cohere(graph, (CORE, MEMBER, PEER))
    for node in graph:
        assert _pressure(graph, node) == Q(1, 64)
        assert read_grammar_state(graph.nodes[node]).history == ("IL",)
    return graph


def test_equal_first_boundary_response_does_not_close_later_nul_en_continuation():
    observations = []
    for core, member, contracted_member, expected in (
        (Q(3, 4), Q(1, 4), Q(1, 8), (Q(5, 16), Q(15, 64), Q(35, 128))),
        (Q(1, 2), Q(1, 2), Q(1, 4), (Q(5, 16), Q(17, 64), Q(37, 128))),
    ):
        graph = _declared_response_source(core, member)
        _fragment(graph, MEMBER, compound=True)
        assert _form(graph, CORE) == Q(1, 4)
        assert _form(graph, MEMBER) == contracted_member
        assert _pressure(graph, CORE) == _pressure(graph, MEMBER) == Q(1, 32)
        assert get_attr(graph.nodes[CORE], ALIAS_VF) == 0.5
        assert get_attr(graph.nodes[MEMBER], ALIAS_VF) == 0.5
        assert read_grammar_state(graph.nodes[CORE]).history == ("IL", "NUL", "EN")
        assert read_grammar_state(graph.nodes[MEMBER]).history == ("IL", "NUL")
        observations.append(_continue(graph, MEMBER, expected))
        assert read_grammar_state(graph.nodes[CORE]).operator_step == 4
        assert read_grammar_state(graph.nodes[PEER]).operator_step == 3
        assert read_grammar_state(graph.nodes[MEMBER]).operator_step == 2
        assert graph.graph["_t"] == 0.0
        assert all("epi_time_history" not in graph.nodes[node] for node in graph)
    assert observations[0][0] == observations[1][0]
    assert observations[1][2] - observations[0][2] == Q(1, 64)


def _acquired_source():
    graph = _graph()
    for node in (CORE, PEER):
        _node(graph, node, Q(1, 4), Q(1, 2))
    graph.add_edge(CORE, PEER, weight=1.0)
    _prepare_pressure(graph, (CORE, PEER), 2.0)
    first = execute_emission_episode(
        graph, (CORE, PEER), duration=0.125, method="euler", n_jobs=1
    )
    assert first.exact_rates == (Q(1, 2), Q(1, 2))
    assert first.exact_epi_after == (Q(5, 16), Q(5, 16))
    _prepare_pressure(graph, (CORE,), 2.5)
    second = execute_emission_episode(
        graph, (CORE, PEER), duration=0.125, method="euler", n_jobs=1
    )
    assert second.exact_rates == (Q(5, 8), Q(1, 2))
    assert second.exact_epi_after == (Q(25, 64), Q(3, 8))
    for node, endpoint, acceleration in (
        (CORE, Q(25, 64), 1.0),
        (PEER, Q(3, 8), 0.0),
    ):
        assert tuple(graph.nodes[node]["epi_time_history"]) == (
            (0.0, 0.25),
            (0.125, 0.3125),
            (0.25, float(endpoint)),
        )
        measured = observe_structural_acceleration(graph, node)
        assert measured.available and measured.source == "epi_time_history"
        assert measured.value == acceleration
    birth = execute_eligible_self_organization_stage(
        graph, execution_kwargs={"tau": 0.5, "validate_preconditions": True}
    )
    assert birth.eligibility.eligible_nodes == (CORE,)
    ((parent, child),) = birth.parent_children
    assert parent == CORE and graph.degree(child) == 0
    assert graph.nodes[CORE]["sub_nodes"] == [child]
    assert graph.nodes[child]["parent_node"] == CORE
    assert _form(graph, child) == Q(15, 128)
    assert get_attr(graph.nodes[child], ALIAS_VF) == 0.95
    assert _pressure(graph, child) == 0
    assert read_grammar_state(graph.nodes[child]).history == ()
    access = execute_birth_access_coupling(graph, CORE)
    assert access.executed and access.access.selected_partner == child
    assert _pressure(graph, CORE) == Q(9, 16)
    assert Q.from_float(get_attr(graph.nodes[CORE], ALIAS_VF)) == Q(
        8950904259398861, 2**53
    )
    for _ in range(3):
        _cohere(graph, (CORE,))
    assert _pressure(graph, CORE) == Q(9, 128)
    assert read_grammar_state(graph.nodes[CORE]).operator_step == 11
    assert read_grammar_state(graph.nodes[PEER]).operator_step == 4
    assert graph.graph["_t"] == 0.25
    return graph, child


def test_acquired_birth_access_reaches_synthesis_and_actual_boundary_continuation():
    outcomes = {}
    for compound, expected_core, expected_response in (
        (True, Q(65, 512), (Q(257, 1024), Q(577, 4096), Q(1605, 8192))),
        (False, Q(25, 128), (Q(73, 256), Q(47, 256), Q(15, 64))),
    ):
        graph, child = _acquired_source()
        physical = {
            node: deepcopy(graph.nodes[node]["epi_time_history"])
            for node in (CORE, PEER)
        }
        _fragment(graph, child, compound=compound)
        assert _form(graph, CORE) == expected_core
        assert _form(graph, child) == Q(15, 256)
        assert _pressure(graph, CORE) == Q(9, 64)
        assert _pressure(graph, child) == 0
        validate_reception_strict(graph, CORE)
        assert Q.from_float(get_attr(graph.nodes[CORE], ALIAS_VF)) == Q(
            8950904259398861, 2**54
        )
        assert get_attr(graph.nodes[child], ALIAS_VF) == 0.95 / 2
        assert read_grammar_state(graph.nodes[child]).history == ("NUL",)
        assert not read_grammar_state(graph.nodes[child]).prior_coherence
        assert "epi_time_history" not in graph.nodes[child]
        assert "_structural_lineage" not in graph.nodes[child]
        # Capturing a retained pair does not make the old physical endpoint fresh.
        with pytest.raises(TNFRValueError, match="final EPI"):
            observe_structural_acceleration(graph, CORE)
        outcomes[compound] = _continue(graph, child, expected_response)
        assert read_grammar_state(graph.nodes[CORE]).operator_step == 13 + int(compound)
        assert read_grammar_state(graph.nodes[PEER]).operator_step == 6
        assert read_grammar_state(graph.nodes[child]).operator_step == 1
        for node in (CORE, PEER):
            assert graph.nodes[node]["epi_time_history"] == physical[node]
            assert graph.nodes[node]["_structural_lineage"]["activation_count"] == 2
        assert graph.graph["_t"] == 0.25
    assert outcomes[True][0] - outcomes[False][0] == -Q(35, 1024)
    assert outcomes[True][2] - outcomes[False][2] == -Q(315, 8192)
