"""Admission boundaries of an acquired AL-history/THOL birth/UM composition.

The source history is produced by actual IL/OZ/AL execution. Supplied times,
factors, eligibility policy and partner access are finite interventions; they
do not establish autonomous generation or inherited child execution history.
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
    ALIAS_VF,
)
from tnfr.errors import TNFRValueError
from tnfr.operators.coherence import Coherence
from tnfr.operators.coupling import Coupling
from tnfr.operators.dissonance import Dissonance
from tnfr.operators.emission_episode import execute_emission_episode
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import (
    STAGE_CONTRACT_KEY,
    STAGE_SCHEDULE_KEY,
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

PARENT, PEER = "parent", "peer"
NODES = (PARENT, PEER)


def _form(graph, node):
    return require_finite_real_scalar_epi(
        get_attr(graph.nodes[node], ALIAS_EPI, strict=True, conv=lambda value: value)
    )


def _state(graph):
    """State relevant to the composed actions, excluding adapter caches."""
    keys = (
        "_t",
        "history",
        "hybrid_event_log",
        "hierarchy",
        "recognized_coherence_patterns",
        STAGE_CONTRACT_KEY,
        STAGE_SCHEDULE_KEY,
    )
    return deepcopy(
        (
            tuple((node, dict(data)) for node, data in graph.nodes(data=True)),
            tuple(graph.edges(data=True)),
            tuple((key, graph.graph[key]) for key in keys if key in graph.graph),
            (
                hasattr(graph, "_last_operator_applied"),
                getattr(graph, "_last_operator_applied", None),
            ),
        )
    )


def _change_pressure(graph, targets, factor):
    graph.graph["GLYPH_FACTORS"]["OZ_dnfr_factor"] = factor
    execute_pointwise_stage(
        graph,
        Coherence(),
        targets,
        phase_locking_coefficient=0.0,
        require_exact_operator=True,
        validate_preconditions=True,
    )
    execute_dissonance_stage(
        graph,
        Dissonance(),
        targets,
        propagate_to_network=False,
        validate_preconditions=True,
    )


def _episode(graph):
    return execute_emission_episode(
        graph, NODES, duration=0.125, method="euler", n_jobs=1
    )


def _source(*, complete=True):
    graph = nx.Graph()
    graph.add_edge(PARENT, PEER, weight=1.0)
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        INTEGRATOR_METHOD="euler",
        GAMMA={"type": "none"},
        RANDOM_SEED=67,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        EDGE_AWARE_ENABLED=True,
        OZ_NOISE_MODE=False,
        IL_PRECONDITIONS={"warn_zero_dnfr": False},
        CLIP_MODE="hard",
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        EPI_LATENT_MAX=0.5,
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
    )
    for node in NODES:
        graph.nodes[node].update(
            EPI=0.25,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.5,
            epi_kind="form",
        )
        set_attr(graph.nodes[node], ALIAS_DEPI, 0.0)
        set_attr(graph.nodes[node], ALIAS_D2EPI, 0.0)
    _change_pressure(graph, NODES, 2.0)
    assert all(
        read_grammar_state(graph.nodes[node]).history == ("IL", "OZ") for node in NODES
    )
    assert all("epi_time_history" not in graph.nodes[node] for node in NODES)
    _episode(graph)
    assert all(_form(graph, node) == 5 / 16 for node in NODES)
    if complete:
        _change_pressure(graph, (PARENT,), 2.5)
        _episode(graph)
        assert _form(graph, PARENT) == 25 / 64
        assert _form(graph, PEER) == 3 / 8
        assert get_attr(graph.nodes[PARENT], ALIAS_DNFR) == 5 / 8
        assert get_attr(graph.nodes[PEER], ALIAS_DNFR) == 1 / 2
        assert observe_structural_acceleration(graph, PARENT).value == 1.0
        assert observe_structural_acceleration(graph, PEER).value == 0.0
    count = 2 if complete else 1
    assert graph.graph["_t"] == count / 8
    for node in NODES:
        assert len(graph.nodes[node]["epi_time_history"]) == count + 1
        assert graph.nodes[node]["_structural_lineage"]["activation_count"] == count
    return graph


def _dispatch(graph, tau=0.5):
    return execute_eligible_self_organization_stage(
        graph, execution_kwargs={"tau": tau, "validate_preconditions": True}
    )


def _birth(graph):
    old_step = read_grammar_state(graph.nodes[PARENT]).operator_step
    old_samples = deepcopy(graph.nodes[PARENT]["epi_time_history"])
    result = _dispatch(graph)
    assert result.policy == "all_eligible_once"
    assert result.eligibility.eligible_nodes == (PARENT,)
    assert result.stage_result.nodes_processed == 1
    ((parent, child),) = result.parent_children
    assert parent == PARENT and graph.nodes[child]["parent_node"] == PARENT
    assert tuple(graph) == (*NODES, child)
    assert tuple(graph.edges) == ((PARENT, PEER),)
    assert graph.degree(child) == 0
    assert Q.from_float(_form(graph, child)) == Q(15, 128)
    assert get_attr(graph.nodes[child], ALIAS_VF) == 0.95
    assert get_attr(graph.nodes[child], ALIAS_DNFR) == 0.0
    assert graph.nodes[child]["theta"] == 0.0
    assert read_grammar_state(graph.nodes[child]).history == ()
    assert read_grammar_state(graph.nodes[child]).operator_step == 0
    assert graph.nodes[child]["epi_history"] == [_form(graph, child)]
    assert "epi_time_history" not in graph.nodes[child]
    assert not observe_structural_acceleration(graph, child).available
    assert graph.nodes[PARENT]["epi_time_history"] == old_samples
    record = graph.nodes[PARENT]["sub_epis"][-1]
    assert record["node_id"] == child
    assert record["timestamp"] == old_step + 1 == 7
    assert record["timestamp"] != graph.graph["_t"] == 0.25
    assert get_attr(graph.nodes[PARENT], ALIAS_DNFR) == 3 / 4
    return child


def test_short_acquired_episode_history_abstains_without_a_birth_or_empty_stage():
    graph = _source(complete=False)
    before = _state(graph)
    result = _dispatch(graph)
    assert result.eligibility.eligible_nodes == ()
    assert result.stage_result is None and result.parent_children == ()
    for candidate in result.eligibility.candidates:
        assert candidate.acceleration.source == "epi_time_history"
        assert candidate.acceleration.history_length == 2
        assert candidate.acceleration.available is False
        assert candidate.acceleration.value is None
        assert not candidate.eligible
    assert _state(graph) == before
    assert tuple(graph) == NODES


def test_exact_threshold_equality_abstains_on_the_full_acquired_source():
    graph = _source()
    before = _state(graph)
    result = _dispatch(graph, tau=1.0)
    assert result.eligibility.eligible_nodes == ()
    assert result.stage_result is None and result.parent_children == ()
    candidates = {item.node: item for item in result.eligibility.candidates}
    assert candidates[PARENT].acceleration.value == 1.0
    assert candidates[PEER].acceleration.value == 0.0
    assert all(item.threshold_crossed is False for item in candidates.values())
    assert all(item.birth_proposed is False for item in candidates.values())
    assert _state(graph) == before
    assert get_attr(graph.nodes[PARENT], ALIAS_DNFR) == 5 / 8


def test_late_um_failure_rolls_back_contact_and_preserves_the_completed_birth():
    graph = _source()
    child = _birth(graph)
    child_storage = graph.nodes[child]
    before = _state(graph)
    calls = []

    def fail_after_actual_contact(subject):
        calls.append(child)
        assert subject.has_edge(PARENT, child)
        assert read_grammar_state(subject.nodes[PARENT]).history[-1] == "UM"
        assert get_attr(subject.nodes[PARENT], ALIAS_DNFR) == 9 / 16
        assert get_attr(subject.nodes[PARENT], ALIAS_VF) < 1.0
        subject.nodes[PEER]["late_contact_marker"] = True
        subject.graph["late_contact_marker"] = True
        raise RuntimeError("declared late acquired-birth UM failure")

    with pytest.raises(RuntimeError, match="late acquired-birth UM failure"):
        execute_coupling_stage(
            graph,
            Coupling(),
            (PARENT,),
            potential_partner=child,
            compute_delta_nfr=fail_after_actual_contact,
            validate_preconditions=True,
        )
    assert calls == [child]
    assert _state(graph) == before
    assert graph.nodes[child] is child_storage
    assert graph.degree(child) == 0
    assert "late_contact_marker" not in graph.graph
    assert get_attr(graph.nodes[PARENT], ALIAS_DNFR) == 3 / 4
    assert read_grammar_state(graph.nodes[PARENT]).history[-1] == "THOL"
    assert graph.nodes[PARENT]["_structural_lineage"]["activation_count"] == 2


def test_child_cannot_receive_from_ancestry_before_an_actual_edge_exists():
    graph = _source()
    child = _birth(graph)
    before = _state(graph)
    with pytest.raises(ValueError, match="source"):
        execute_neighbor_stage(
            graph,
            Reception(),
            (child,),
            source_nodes=(PARENT,),
            track_sources=False,
            validate_preconditions=True,
        )
    assert _state(graph) == before
    assert read_grammar_state(graph.nodes[child]).history == ()
    assert "epi_time_history" not in graph.nodes[child]
    assert graph.degree(child) == 0


def test_stale_acquired_parent_endpoint_is_reported_without_a_false_birth():
    graph = _source()
    # This is a labelled fault after acquisition, not another produced history.
    set_attr(graph.nodes[PARENT], ALIAS_EPI, 0.375)
    before = _state(graph)
    with pytest.raises(TNFRValueError, match="final EPI"):
        observe_structural_acceleration(graph, PARENT)
    result = _dispatch(graph)
    assert result.eligibility.eligible_nodes == ()
    assert result.stage_result is None and result.parent_children == ()
    candidates = {item.node: item for item in result.eligibility.candidates}
    assert candidates[PARENT].acceleration is None
    assert "final EPI" in candidates[PARENT].acceleration_error
    assert candidates[PARENT].proposal_valid is False
    assert candidates[PEER].acceleration.value == 0.0
    assert candidates[PEER].threshold_crossed is False
    assert _state(graph) == before
    assert tuple(graph) == NODES
