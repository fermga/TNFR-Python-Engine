"""Finite constituent/port continuations; no producer, flow or scalar closure.

Each birth source is built here from three supplied temporal samples. Tests
invoke a bounded list of actual operators and never import a research fixture.
"""

from __future__ import annotations

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators.contraction import Contraction
from tnfr.operators.coupling import Coupling
from tnfr.operators.expansion import Expansion
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import (
    TWO_PHASE_JACOBI,
    execute_coupling_stage,
    execute_neighbor_stage,
    execute_pointwise_stage,
    execute_self_organization_stage,
)
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.reception import Reception
from tnfr.operators.resonance import Resonance
from tnfr.operators.retained_organization import (
    capture_retained_organization,
    execute_retained_scale_stage,
    retained_organization_matches_graph,
)
from tnfr.operators.self_organization import SelfOrganization
from tnfr.types import require_finite_real_scalar_epi
from tnfr.utils._structural_signature import structural_proof_signature

_PARENT = "parent"
_PEER = "external-peer"
_VAL_SCALE = 65.0 / 64.0
_NUL_SCALE = 0.5


def _form(graph, node):
    return require_finite_real_scalar_epi(get_attr(graph.nodes[node], ALIAS_EPI))


def _state(graph, node):
    return (
        _form(graph, node),
        get_attr(graph.nodes[node], ALIAS_VF),
        get_attr(graph.nodes[node], ALIAS_THETA),
        get_attr(graph.nodes[node], ALIAS_DNFR),
    )


def _edges(graph):
    if graph.is_multigraph():
        return tuple(
            (a, b, k, deepcopy(d)) for a, b, k, d in graph.edges(keys=True, data=True)
        )
    return tuple((a, b, deepcopy(d)) for a, b, d in graph.edges(data=True))


def _plain(graph):
    metadata = dict(graph.graph)
    if "_node_cache" in metadata:
        metadata["_node_cache"] = {
            node: id(wrapper) for node, wrapper in metadata["_node_cache"].items()
        }
    return structural_proof_signature(
        (
            tuple(graph.nodes(data=True)),
            _edges(graph),
            metadata,
            hasattr(graph, "_last_operator_applied"),
            getattr(graph, "_last_operator_applied", None),
        ),
        opaque_references=(graph,),
    )


def _birth_source(graph_type=nx.Graph, *, receive=True, admit_zero_pressure=False):
    graph = graph_type(
        THOL_METABOLIC_ENABLED=False,
        THOL_PROPAGATION_ENABLED=False,
        THOL_ALLOW_ISOLATED=False,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_REJECTION_MODE="raise",
        EDGE_AWARE_ENABLED=True,
        CLIP_MODE="hard",
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        _t=2.0,
        RANDOM_SEED=17,
        UM_FUNCTIONAL_LINKS=True,
        UM_BIDIRECTIONAL=True,
        UM_SYNC_VF=True,
        UM_STABILIZE_DNFR=True,
        UM_COMPAT_THRESHOLD=0.5,
        GLYPH_FACTORS={
            "THOL_accel": 0.25,
            "UM_theta_push": 0.5,
            "UM_vf_sync": 0.25,
            "UM_dnfr_reduction": 0.25,
            "EN_mix": 0.25,
            "RA_epi_diff": 0.5,
            "VAL_scale": _VAL_SCALE,
            "NUL_scale": _NUL_SCALE,
        },
    )
    if admit_zero_pressure:
        # A declared optional VAL policy, not pressure injected into a child.
        graph.graph["VAL_MIN_DNFR"] = 0.0
    graph.add_node(
        _PARENT,
        EPI=0.875,
        nu_f=1.0,
        theta=0.25,
        delta_nfr=0.05,
        Si=0.5,
        epi_kind="prepared-form",
        epi_time_history=[(0.0, 0.5), (1.0, 0.625), (2.0, 0.875)],
        glyph_history=["IL", "OZ"],
        _grammar_u2_debt=1,
        _grammar_prior_coherence=True,
        _operator_step=2,
    )
    graph.add_node(
        _PEER,
        EPI=0.125,
        nu_f=3.0,
        theta=0.75,
        delta_nfr=0.05,
        Si=0.5,
        epi_kind="prepared-form",
        glyph_history=["AL", "IL"],
        _grammar_u2_debt=0,
        _grammar_prior_coherence=True,
        _operator_step=2,
    )
    graph.add_edge(_PARENT, _PEER, weight=0.37, supplied="old-support")
    if graph.is_multigraph():
        graph.add_edge(_PARENT, _PEER, weight=0.0, supplied="parallel-zero")
    execute_self_organization_stage(graph, SelfOrganization(), (_PARENT,), tau=0.1)
    (child,) = graph.nodes[_PARENT]["sub_nodes"]
    assert graph.nodes[child]["parent_node"] == _PARENT
    assert graph.degree(child) == 0
    assert _form(graph, child) == 0.3 * 0.875
    assert get_attr(graph.nodes[child], ALIAS_VF) == 0.95
    assert read_grammar_state(graph.nodes[child]).history == ()
    assert "epi_time_history" not in graph.nodes[child]
    execute_coupling_stage(graph, Coupling(), (_PARENT,), potential_partner=child)
    if receive:
        execute_neighbor_stage(
            graph,
            Reception(),
            (child,),
            source_nodes=(_PARENT,),
            track_sources=False,
        )
        assert _form(graph, child) == (0.3 * 0.875) * 0.75 + 0.875 * 0.25
        assert read_grammar_state(graph.nodes[child]).history == ("EN",)
    assert get_attr(graph.nodes[child], ALIAS_DNFR) == 0.0
    return graph, child


@pytest.mark.parametrize(
    "operator_type,scale", [(Expansion, _VAL_SCALE), (Contraction, _NUL_SCALE)]
)
@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
def test_real_birth_members_scale_like_scalar_stage_and_preserve_individual_state(
    operator_type, scale, graph_type
):
    graph, child = _birth_source(
        graph_type, admit_zero_pressure=operator_type is Expansion
    )
    control, control_child = _birth_source(
        graph_type, admit_zero_pressure=operator_type is Expansion
    )
    assert child == control_child
    members = (_PARENT, child)
    before = {node: _state(graph, node) for node in members}
    contexts = {node: read_grammar_state(graph.nodes[node]) for node in members}
    peer_before = deepcopy(graph.nodes[_PEER])
    old_edges = _edges(graph)
    old_nodes = tuple(graph)
    time_history = deepcopy(graph.nodes[_PARENT]["epi_time_history"])
    record = capture_retained_organization(graph, members)
    dispatch = execute_retained_scale_stage(graph, record, operator_type())
    scalar = execute_pointwise_stage(control, operator_type(), members)

    assert dispatch.before.members == record.members
    assert dispatch.before.member_states == record.member_states
    assert dispatch.before.incoming_ports == record.incoming_ports
    assert dispatch.before.outgoing_ports == record.outgoing_ports
    assert dispatch.stage_result.schedule == scalar.schedule == TWO_PHASE_JACOBI
    assert dispatch.stage_result.nodes_processed == 2
    assert dispatch.after.members == members
    assert retained_organization_matches_graph(graph, dispatch.after)
    assert not retained_organization_matches_graph(graph, record)
    assert tuple(graph) == old_nodes
    assert _edges(graph) == old_edges
    assert graph.nodes[_PEER] == peer_before
    assert graph.nodes[_PARENT]["epi_time_history"] == time_history
    assert "epi_time_history" not in graph.nodes[child]
    assert not observe_structural_acceleration(graph, child).available
    assert graph.graph["_t"] == 2.0

    for node in members:
        x, capacity, phase, pressure = before[node]
        expected_pressure = pressure if operator_type is Expansion else pressure * 2.0
        assert _state(graph, node) == (
            x * scale,
            capacity * scale,
            phase,
            expected_pressure,
        )
        assert _state(graph, node) == _state(control, node)
        assert read_grammar_state(graph.nodes[node]) == read_grammar_state(
            control.nodes[node]
        )
        assert (
            read_grammar_state(graph.nodes[node]).operator_step
            == contexts[node].operator_step + 1
        )
        assert (
            read_grammar_state(graph.nodes[node]).history[-1]
            == operator_type.glyph.value
        )
    assert not read_grammar_state(graph.nodes[child]).prior_coherence
    assert read_grammar_state(graph.nodes[_PARENT]).prior_coherence


@pytest.mark.parametrize(
    "operator_type,scale", [(Expansion, _VAL_SCALE), (Contraction, _NUL_SCALE)]
)
def test_retained_scale_changes_actual_en_and_ra_through_original_member_ports(
    operator_type, scale
):
    responses = []
    for scaled in (False, True):
        graph, child = _birth_source(admit_zero_pressure=operator_type is Expansion)
        child_form = (0.3 * 0.875) * 0.75 + 0.875 * 0.25
        if scaled:
            dispatch = execute_retained_scale_stage(
                graph,
                capture_retained_organization(graph, (_PARENT, child)),
                operator_type(),
            )
            assert dispatch.after.members == (_PARENT, child)
        factor = scale if scaled else 1.0
        expected_en = (0.875 * factor) * 0.75 + 0.25 * (
            (0.125 + child_form * factor) / 2.0
        )
        execute_neighbor_stage(graph, Reception(), (_PARENT,), track_sources=False)
        assert _form(graph, _PARENT) == expected_en
        expected_ra = 0.125 + 0.5 * (expected_en - 0.125)
        execute_neighbor_stage(graph, Resonance(), (_PEER,))
        assert _form(graph, _PEER) == expected_ra
        assert _form(graph, child) == child_form * factor
        assert "epi_time_history" not in graph.nodes[child]
        assert graph.graph["_t"] == 2.0
        responses.append(expected_ra)
    assert responses[0] != responses[1]


def test_fresh_child_does_not_inherit_parent_context_and_own_en_invalidates_old_record():
    graph, child = _birth_source(receive=False)
    record = capture_retained_organization(graph, (_PARENT, child))
    assert read_grammar_state(graph.nodes[child]).history == ()
    assert read_grammar_state(graph.nodes[child]).u2_debt == 0
    assert not read_grammar_state(graph.nodes[child]).prior_coherence
    assert read_grammar_state(graph.nodes[_PARENT]).prior_coherence
    execute_neighbor_stage(
        graph, Reception(), (child,), source_nodes=(_PARENT,), track_sources=False
    )
    assert read_grammar_state(graph.nodes[child]).history == ("EN",)
    assert not retained_organization_matches_graph(graph, record)
    before = _plain(graph)
    with pytest.raises((ValueError, RuntimeError)):
        execute_retained_scale_stage(graph, record, Contraction())
    assert _plain(graph) == before
    refreshed = capture_retained_organization(graph, (_PARENT, child))
    execute_retained_scale_stage(graph, refreshed, Contraction())
    assert read_grammar_state(graph.nodes[child]).history == ("EN", "NUL")
    assert "epi_time_history" not in graph.nodes[child]


@pytest.mark.parametrize("order", ["parent-first", "child-first"])
def test_default_val_pressure_gate_rejects_mixed_real_members_atomically(order):
    graph, child = _birth_source()
    members = (_PARENT, child) if order == "parent-first" else (child, _PARENT)
    assert "VAL_MIN_DNFR" not in graph.graph
    assert get_attr(graph.nodes[child], ALIAS_DNFR) == 0.0
    record = capture_retained_organization(graph, members)
    before = _plain(graph)
    with pytest.raises(OperatorPreconditionError):
        execute_retained_scale_stage(graph, record, Expansion())
    assert _plain(graph) == before
    assert retained_organization_matches_graph(graph, record)
    assert read_grammar_state(graph.nodes[child]).history == ("EN",)


@pytest.mark.parametrize("rejection_mode", ["raise", "fallback"])
def test_one_member_debt_cannot_be_hidden_in_aggregate_or_replaced_by_another_glyph(
    rejection_mode,
):
    graph, child = _birth_source(admit_zero_pressure=True)
    graph.graph["GRAMMAR_REJECTION_MODE"] = rejection_mode
    graph.nodes[child]["_grammar_u2_debt"] = 3
    record = capture_retained_organization(graph, (_PARENT, child))
    before = _plain(graph)
    with pytest.raises((ValueError, RuntimeError, OperatorPreconditionError)):
        execute_retained_scale_stage(graph, record, Expansion())
    assert _plain(graph) == before
    assert read_grammar_state(graph.nodes[child]).u2_debt == 3


def _form_port_graph(left, right):
    graph = nx.Graph(
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_REJECTION_MODE="raise",
        GLYPH_FACTORS={"RA_epi_diff": 0.5, "EN_mix": 0.25},
    )
    for node, form in (("left", left), ("right", right), ("outside", 0.125)):
        graph.add_node(
            node,
            EPI=form,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.1,
            Si=0.5,
            epi_kind="form",
            glyph_history=["AL", "IL"],
        )
    graph.add_edges_from((("left", "right"), ("outside", "left")))
    return graph


def test_same_scalar_mean_and_member_count_lose_actual_external_port_response():
    # Supplied collision controls, not an asserted clone or birth history.
    responses = []
    for left, right in ((0.75, 0.25), (0.25, 0.75)):
        graph = _form_port_graph(left, right)
        record = capture_retained_organization(graph, ("left", "right"))
        assert (_form(graph, "left") + _form(graph, "right")) / 2.0 == 0.5
        assert len(record.members) == 2
        execute_neighbor_stage(graph, Resonance(), ("outside",))
        expected = 0.125 + 0.5 * (left - 0.125)
        assert _form(graph, "outside") == expected
        responses.append(expected)
    assert responses == [0.4375, 0.1875]


def test_parent_en_leaves_the_initial_actual_thol_form_proportionality():
    graph, child = _birth_source(receive=False)
    parent_before = _form(graph, _PARENT)
    child_before = _form(graph, child)
    assert child_before == 0.3 * parent_before
    record = capture_retained_organization(graph, (_PARENT, child))
    execute_neighbor_stage(graph, Reception(), (_PARENT,), track_sources=False)
    expected_parent = 0.75 * parent_before + 0.25 * ((0.125 + child_before) / 2.0)
    assert _form(graph, _PARENT) == expected_parent
    assert _form(graph, child) == child_before
    defect = child_before - 0.3 * expected_parent
    ideal_defect = 0.3 * 0.25 * ((1.0 - 0.3 / 2.0) * parent_before - 0.125 / 2.0)
    assert defect == pytest.approx(ideal_defect, rel=0.0, abs=1e-16)
    assert defect > 0.0
    assert not retained_organization_matches_graph(graph, record)
    assert read_grammar_state(graph.nodes[child]).history == ()


def test_exact_dyadic_response_contrasts_retain_two_member_form_directions():
    # Three supplied finite states test the declared affine word. They are not
    # a trajectory or a numerical proof of rank on a general state domain.
    outputs = []
    for left, right in ((0.5, 0.25), (0.625, 0.25), (0.5, 0.375)):
        graph = _form_port_graph(left, right)
        execute_neighbor_stage(graph, Resonance(), ("outside",))
        expected_ra = 0.125 + 0.5 * (left - 0.125)
        assert _form(graph, "outside") == expected_ra
        execute_neighbor_stage(graph, Reception(), ("left",), track_sources=False)
        expected_en = 0.75 * left + 0.25 * ((expected_ra + right) / 2.0)
        assert _form(graph, "left") == expected_en
        outputs.append((expected_ra, expected_en))
    first = tuple(
        value - base for value, base in zip(outputs[1], outputs[0], strict=True)
    )
    second = tuple(
        value - base for value, base in zip(outputs[2], outputs[0], strict=True)
    )
    assert first == (0.0625, 0.1015625)
    assert second == (0.0, 0.015625)
    assert first[0] * second[1] - second[0] * first[1] == 1.0 / 1024.0


def test_duplicating_one_supplied_donor_changes_en_mean_despite_edge_conductances():
    # A multiplicity counterexample; no operation here is labeled VAL cloning.
    responses = []
    for duplicate in (False, True):
        graph = nx.DiGraph(
            VALIDATE_OPERATOR_PRECONDITIONS=True,
            GRAMMAR_REJECTION_MODE="raise",
            GLYPH_FACTORS={"EN_mix": 1.0},
        )
        forms = {"receiver": 0.0, "positive": 1.0, "negative": -1.0}
        if duplicate:
            forms["positive-copy"] = 1.0
        for node, form in forms.items():
            graph.add_node(
                node,
                EPI=form,
                nu_f=1.0,
                theta=0.0,
                delta_nfr=0.1,
                Si=0.5,
                epi_kind="form",
                glyph_history=["AL"],
            )
            if node != "receiver":
                graph.add_edge(node, "receiver", weight=100.0 if form < 0 else 0.0)
        execute_neighbor_stage(graph, Reception(), ("receiver",), track_sources=False)
        expected = 1.0 / 3.0 if duplicate else 0.0
        assert _form(graph, "receiver") == expected
        responses.append(expected)
    assert responses == [0.0, 1.0 / 3.0]
