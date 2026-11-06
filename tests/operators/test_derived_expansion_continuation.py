"""Independent finite VAL derivation/UM/EN/RA controls without a producer."""

from __future__ import annotations

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators.coupling import Coupling
from tnfr.operators.derived_expansion import (
    DerivedExpansionPolicy,
    execute_derived_expansion_stage,
)
from tnfr.operators.expansion import Expansion
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import (
    TWO_PHASE_JACOBI,
    execute_coupling_stage,
    execute_neighbor_stage,
    execute_pointwise_stage,
)
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.reception import Reception
from tnfr.operators.resonance import Resonance
from tnfr.operators.retained_organization import (
    capture_retained_organization,
    retained_organization_matches_graph,
)
from tnfr.types import require_finite_real_scalar_epi
from tnfr.utils._structural_signature import structural_proof_signature

_PARENT = "core"
_PEER = "outside"
_CHILD = "supplied-new-identity"
_ERRORS = (ValueError, RuntimeError, OperatorPreconditionError)


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
    return structural_proof_signature(
        (
            graph.graph,
            tuple(graph.nodes(data=True)),
            _edges(graph),
            getattr(graph, "_last_operator_applied", None),
        ),
        opaque_references=(graph,),
    )


def _source(
    graph_type=nx.Graph, *, form=0.5, capacity=1.0, factor=1.5, edge_aware=True
):
    graph = graph_type(
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_REJECTION_MODE="raise",
        EDGE_AWARE_ENABLED=edge_aware,
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        UM_FUNCTIONAL_LINKS=True,
        UM_BIDIRECTIONAL=True,
        UM_SYNC_VF=False,
        UM_STABILIZE_DNFR=False,
        UM_COMPAT_THRESHOLD=0.5,
        _t=2.0,
        RANDOM_SEED=11,
        GLYPH_FACTORS={
            "VAL_scale": factor,
            "UM_theta_push": 0.5,
            "EN_mix": 0.25,
            "RA_epi_diff": 0.5,
        },
    )
    graph.add_node(
        _PARENT,
        EPI=form,
        nu_f=capacity,
        theta=0.25,
        delta_nfr=0.125,
        Si=0.5,
        epi_kind="same-kind",
        glyph_history=["AL", "IL"],
        _grammar_u2_debt=0,
        _grammar_prior_coherence=True,
        _operator_step=2,
    )
    graph.add_node(
        _PEER,
        EPI=0.125,
        nu_f=2.0,
        theta=0.5,
        delta_nfr=0.125,
        Si=0.5,
        epi_kind="same-kind",
        glyph_history=["AL", "IL"],
        _grammar_u2_debt=0,
        _grammar_prior_coherence=True,
        _operator_step=2,
    )
    graph.add_edge(_PEER, _PARENT, weight=0.375, purpose="original-port")
    if graph.is_multigraph():
        graph.add_edge(_PEER, _PARENT, weight=0.0, purpose="parallel-zero")
    return graph


def _derive(graph, order, *, form_ratio=0.5, capacity_ratio=0.5):
    policy = DerivedExpansionPolicy(
        source_order=order, form_ratio=form_ratio, capacity_ratio=capacity_ratio
    )
    source = capture_retained_organization(graph, (_PARENT,))
    return execute_derived_expansion_stage(graph, source, _CHILD, policy)


def _connect(graph):
    return execute_coupling_stage(
        graph, Coupling(), (_PARENT,), potential_partner=_CHILD
    )


def _en_ra(graph):
    expected_en = 0.75 * _form(graph, _PARENT) + 0.25 * (
        (_form(graph, _PEER) + _form(graph, _CHILD)) / 2.0
    )
    execute_neighbor_stage(graph, Reception(), (_PARENT,), track_sources=False)
    assert _form(graph, _PARENT) == expected_en
    expected_ra = 0.125 + 0.5 * (expected_en - 0.125)
    execute_neighbor_stage(graph, Resonance(), (_PEER,))
    assert _form(graph, _PEER) == expected_ra
    return expected_en, expected_ra


def test_plain_public_val_remains_scalar_without_derived_request():
    graph = _source()
    old_edges = _edges(graph)
    assert Expansion()(graph, _PARENT) is None
    assert tuple(graph) == (_PARENT, _PEER)
    assert _edges(graph) == old_edges
    assert _state(graph, _PARENT) == (0.75, 1.5, 0.25, 0.125)
    assert read_grammar_state(graph.nodes[_PARENT]).history == ("AL", "IL", "VAL")


@pytest.mark.parametrize(
    "graph_type", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
@pytest.mark.parametrize(
    "order,child_form,child_capacity",
    [("pre_val", 0.25, 0.5), ("post_val", 0.375, 0.75)],
)
def test_new_child_uses_declared_boundary_and_parent_keeps_actual_scalar_val(
    graph_type, order, child_form, child_capacity
):
    graph = _source(graph_type)
    ordinary = _source(graph_type)
    old_edges = _edges(graph)
    old_peer = deepcopy(graph.nodes[_PEER])
    assert not observe_structural_acceleration(graph, _PARENT).available
    result = _derive(graph, order)
    ordinary_stage = execute_pointwise_stage(
        ordinary, Expansion(), (_PARENT,), require_exact_operator=True
    )

    assert result.parent == _PARENT and result.child == _CHILD
    assert result.policy.source_order == order
    assert result.after.members == (_PARENT, _CHILD)
    assert retained_organization_matches_graph(graph, result.after)
    assert (
        result.scale_result.stage_result.schedule
        == ordinary_stage.schedule
        == TWO_PHASE_JACOBI
    )
    assert result.scale_result.stage_result.nodes_processed == 1
    assert (
        _state(graph, _PARENT) == _state(ordinary, _PARENT) == (0.75, 1.5, 0.25, 0.125)
    )
    assert read_grammar_state(graph.nodes[_PARENT]) == read_grammar_state(
        ordinary.nodes[_PARENT]
    )
    assert tuple(ordinary) == (_PARENT, _PEER)
    assert tuple(graph) == (_PARENT, _PEER, _CHILD)
    assert _edges(graph) == _edges(ordinary) == old_edges
    assert graph.nodes[_PEER] == old_peer
    assert graph.degree(_CHILD) == 0
    assert _state(graph, _CHILD) == (child_form, child_capacity, 0.25, 0.0)
    selected_x, selected_capacity = (0.5, 1.0) if order == "pre_val" else (0.75, 1.5)
    assert result.inheritance_source.node == _PARENT
    assert result.inheritance_source.epi == selected_x
    assert result.inheritance_source.capacity == selected_capacity
    assert graph.nodes[_CHILD]["derived_from"] == _PARENT
    assert graph.nodes[_CHILD]["epi_kind"] == "same-kind"
    child_context = read_grammar_state(graph.nodes[_CHILD])
    assert child_context.history == () and child_context.u2_debt == 0
    assert not child_context.prior_coherence and child_context.operator_step == 0
    assert graph.nodes[_CHILD]["epi_history"] == [child_form]
    for key in (
        "epi_time_history",
        "parent_node",
        "sub_nodes",
        "sub_epis",
        "_hierarchy_path",
    ):
        assert key not in graph.nodes[_CHILD]
    assert not observe_structural_acceleration(graph, _CHILD).available
    assert read_grammar_state(graph.nodes[_PARENT]).history == ("AL", "IL", "VAL")
    assert graph.graph["_t"] == 2.0

    # Declared-map accounting, without assuming conservation of either sum.
    assert _form(graph, _PARENT) + _form(graph, _CHILD) == 0.75 + child_form
    assert (
        get_attr(graph.nodes[_PARENT], ALIAS_VF)
        + get_attr(graph.nodes[_CHILD], ALIAS_VF)
        == 1.5 + child_capacity
    )
    assert (
        sum(
            get_attr(graph.nodes[n], ALIAS_VF) * get_attr(graph.nodes[n], ALIAS_DNFR)
            for n in (_PARENT, _CHILD)
        )
        == 1.5 * 0.125
    )


def test_fixed_ratios_same_final_parent_have_distinct_actual_en_ra_response():
    outputs = []
    for order in ("pre_val", "post_val"):
        graph = _source()
        result = _derive(graph, order)
        child_before = _state(graph, _CHILD)
        assert not graph.has_edge(_PARENT, _CHILD)
        with pytest.raises(_ERRORS):
            execute_neighbor_stage(
                graph,
                Reception(),
                (_PARENT,),
                source_nodes=(_CHILD,),
                track_sources=False,
            )
        _connect(graph)
        assert graph.has_edge(_PARENT, _CHILD)
        assert get_attr(graph.nodes[_PARENT], ALIAS_VF) == 1.5
        assert get_attr(graph.nodes[_PARENT], ALIAS_DNFR) == 0.125
        assert _form(graph, _CHILD) == child_before[0]
        assert get_attr(graph.nodes[_CHILD], ALIAS_VF) == child_before[1]
        assert not retained_organization_matches_graph(graph, result.after)
        outputs.append(_en_ra(graph))
        assert read_grammar_state(graph.nodes[_CHILD]).history == ()
    assert outputs == [(39.0 / 64.0, 47.0 / 128.0), (5.0 / 8.0, 3.0 / 8.0)]
    assert outputs[1][1] - outputs[0][1] == 1.0 / 128.0


def test_reparametrized_orders_can_have_same_primitives_and_downstream_response():
    outputs = []
    for order, ratio in (("pre_val", 0.5), ("post_val", 0.25)):
        graph = _source(form=0.25, factor=2.0)
        result = _derive(graph, order, form_ratio=ratio, capacity_ratio=ratio)
        assert result.policy.source_order == order
        assert _state(graph, _PARENT) == (0.5, 2.0, 0.25, 0.125)
        assert _state(graph, _CHILD) == (0.125, 0.5, 0.25, 0.0)
        _connect(graph)
        outputs.append(_en_ra(graph))
    assert outputs[0] == outputs[1]
    # Agreement of observed coordinates does not identify the supplied policy.


def test_capacity_only_parent_val_removes_form_contrast_but_not_capacity_contrast():
    capacities = []
    for order in ("pre_val", "post_val"):
        graph = _source(edge_aware=False)
        _derive(graph, order)
        assert _state(graph, _PARENT) == (0.5, 1.5, 0.25, 0.125)
        assert _form(graph, _CHILD) == 0.25
        capacities.append(get_attr(graph.nodes[_CHILD], ALIAS_VF))
    assert capacities == [0.5, 0.75]


def test_separately_enabled_um_capacity_sync_changes_the_parent_comparison():
    capacities = []
    for order, child_capacity in (("pre_val", 0.5), ("post_val", 0.75)):
        graph = _source()
        graph.graph["UM_SYNC_VF"] = True
        graph.graph["GLYPH_FACTORS"]["UM_vf_sync"] = 0.25
        _derive(graph, order)
        _connect(graph)
        expected = 1.5 + 0.25 * ((2.0 + child_capacity) / 2.0 - 1.5)
        assert get_attr(graph.nodes[_PARENT], ALIAS_VF) == expected
        assert get_attr(graph.nodes[_CHILD], ALIAS_VF) == child_capacity
        capacities.append(expected)
    assert capacities == [1.4375, 1.46875]


@pytest.mark.parametrize("order,expected", [("pre_val", -0.25), ("post_val", -0.375)])
def test_declared_signed_parent_domain_keeps_signed_child_without_thol_amplitude_clamp(
    order, expected
):
    graph = _source(form=-0.5)
    graph.graph["VAL_MIN_EPI"] = -1.0
    _derive(graph, order)
    assert _form(graph, _PARENT) == -0.75
    assert _form(graph, _CHILD) == expected
    assert get_attr(graph.nodes[_CHILD], ALIAS_DNFR) == 0.0
    # UM's admitted activity policy uses magnitude; it adds no sign prohibition.
    _connect(graph)
    assert graph.has_edge(_PARENT, _CHILD)
    assert _form(graph, _CHILD) == expected


def test_child_bound_rejection_is_atomic_and_does_not_clip_to_selected_minimum():
    graph = _source()
    graph.graph["EPI_MIN"] = 0.4
    before = _plain(graph)
    with pytest.raises(_ERRORS):
        _derive(graph, "pre_val", form_ratio=0.75)
    assert _plain(graph) == before and _CHILD not in graph
    # Same supplied source: post-scale .75*.75 is inside the selected interval.
    _derive(graph, "post_val", form_ratio=0.75)
    assert _form(graph, _CHILD) == 0.5625


@pytest.mark.parametrize("rejection_mode", ["raise", "fallback"])
def test_parent_u2_rejection_cannot_create_child_or_substitute_another_action(
    rejection_mode,
):
    graph = _source()
    graph.graph["GRAMMAR_REJECTION_MODE"] = rejection_mode
    graph.nodes[_PARENT]["_grammar_u2_debt"] = 3
    before = _plain(graph)
    with pytest.raises(_ERRORS):
        _derive(graph, "post_val")
    assert _plain(graph) == before and _CHILD not in graph


@pytest.mark.parametrize("failure", ["stale", "existing-child", "default-pressure"])
def test_invalid_request_restores_parent_support_and_context(failure):
    graph = _source()
    if failure == "default-pressure":
        graph.nodes[_PARENT]["delta_nfr"] = 0.0
    source = capture_retained_organization(graph, (_PARENT,))
    child = _CHILD
    if failure == "stale":
        graph.nodes[_PARENT]["EPI"] = 0.625
    elif failure == "existing-child":
        child = _PEER
    before = _plain(graph)
    with pytest.raises(_ERRORS):
        execute_derived_expansion_stage(
            graph,
            source,
            child,
            DerivedExpansionPolicy(
                source_order="post_val", form_ratio=0.5, capacity_ratio=0.5
            ),
        )
    assert _plain(graph) == before


def test_new_nonzero_child_can_receive_own_en_without_borrowing_parent_il():
    graph = _source()
    _derive(graph, "pre_val")
    _connect(graph)
    expected = 0.25 * 0.75 + 0.75 * 0.25
    execute_neighbor_stage(
        graph, Reception(), (_CHILD,), source_nodes=(_PARENT,), track_sources=False
    )
    assert _form(graph, _CHILD) == expected
    own = read_grammar_state(graph.nodes[_CHILD])
    assert own.history == ("EN",) and not own.prior_coherence
    assert get_attr(graph.nodes[_CHILD], ALIAS_DNFR) == 0.0
    assert "epi_time_history" not in graph.nodes[_CHILD]


def test_new_zero_form_child_retains_u1a_even_after_real_um_connection():
    graph = _source(form=0.0)
    graph.graph["VAL_MIN_EPI"] = 0.0
    graph.graph["UM_MIN_EPI"] = 0.0
    _derive(graph, "pre_val")
    assert _form(graph, _CHILD) == 0.0
    _connect(graph)
    before = _plain(graph)
    with pytest.raises(_ERRORS):
        execute_neighbor_stage(
            graph, Reception(), (_CHILD,), source_nodes=(_PARENT,), track_sources=False
        )
    assert _plain(graph) == before
    assert read_grammar_state(graph.nodes[_CHILD]).history == ()


def test_zero_capacity_creation_does_not_bypass_actual_um_capacity_policy():
    graph = _source(capacity=0.0)
    _derive(graph, "post_val")
    assert get_attr(graph.nodes[_PARENT], ALIAS_VF) == 0.0
    assert get_attr(graph.nodes[_CHILD], ALIAS_VF) == 0.0
    before = _plain(graph)
    with pytest.raises(OperatorPreconditionError):
        _connect(graph)
    assert _plain(graph) == before


def test_actual_um_rechecks_child_phase_and_parent_debt_after_birth():
    for failure in ("phase", "debt"):
        graph = _source()
        _derive(graph, "post_val")
        if failure == "phase":
            graph.nodes[_CHILD]["theta"] = 3.0
        else:
            graph.nodes[_PARENT]["_grammar_u2_debt"] = 3
        before = _plain(graph)
        with pytest.raises(_ERRORS):
            _connect(graph)
        assert _plain(graph) == before and graph.degree(_CHILD) == 0


def test_directed_child_port_does_not_become_parent_incoming_reception_support():
    graph = _source(nx.DiGraph)
    _derive(graph, "pre_val")
    _connect(graph)
    assert graph.has_edge(_PARENT, _CHILD) and not graph.has_edge(_CHILD, _PARENT)
    before = _plain(graph)
    with pytest.raises(_ERRORS):
        execute_neighbor_stage(
            graph, Reception(), (_PARENT,), source_nodes=(_CHILD,), track_sources=False
        )
    assert _plain(graph) == before
    execute_neighbor_stage(graph, Reception(), (_PARENT,), track_sources=False)
    assert _form(graph, _PARENT) == 0.75 * 0.75 + 0.125 * 0.25
    execute_neighbor_stage(
        graph, Reception(), (_CHILD,), source_nodes=(_PARENT,), track_sources=False
    )
    assert _form(graph, _CHILD) == 0.25 * 0.75 + (0.75 * 0.75 + 0.125 * 0.25) * 0.25
    assert read_grammar_state(graph.nodes[_CHILD]).history == ("EN",)
