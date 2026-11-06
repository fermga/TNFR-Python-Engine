"""Independent finite NUL/EN concentration and real operator continuations."""

from __future__ import annotations

import math
import sys
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators.contraction import Contraction
from tnfr.operators.coupling import Coupling
from tnfr.operators.derived_expansion import (
    DerivedExpansionPolicy,
    execute_derived_expansion_stage,
)
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import (
    TWO_PHASE_JACOBI,
    execute_coupling_stage,
    execute_neighbor_stage,
    execute_pointwise_stage,
)
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.reception import Reception
from tnfr.operators.resonance import Resonance
from tnfr.operators.retained_organization import (
    capture_retained_organization,
    retained_organization_matches_graph,
)
from tnfr.operators.retained_synthesis import (
    capture_retained_pair,
    execute_retained_synthesis_stage,
)
from tnfr.types import require_finite_real_scalar_epi
from tnfr.utils._structural_signature import structural_proof_signature

_CORE = "core"
_MEMBER = "retained-derived-member"
_PEER = "external-peer"
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
            (a, b, key, deepcopy(data))
            for a, b, key, data in graph.edges(keys=True, data=True)
        )
    return tuple((a, b, deepcopy(data)) for a, b, data in graph.edges(data=True))


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


def _graph(graph_type=nx.Graph, *, mix=0.25, edge_aware=True):
    return graph_type(
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_REJECTION_MODE="raise",
        EDGE_AWARE_ENABLED=edge_aware,
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        # Explicit admission: NUL doubles p before EN. The default is tested.
        DNFR_RECEPTION_MAX=0.75,
        UM_FUNCTIONAL_LINKS=True,
        UM_BIDIRECTIONAL=True,
        UM_SYNC_VF=False,
        UM_STABILIZE_DNFR=False,
        UM_COMPAT_THRESHOLD=0.5,
        _t=2.0,
        RANDOM_SEED=11,
        GLYPH_FACTORS={
            "VAL_scale": 1.5,
            "NUL_scale": 0.5,
            "UM_theta_push": 0.5,
            "EN_mix": mix,
            "RA_epi_diff": 0.5,
        },
    )


def _node(graph, node, form, *, capacity=1.0, pressure=0.0, phase=0.25):
    graph.add_node(
        node,
        EPI=form,
        nu_f=capacity,
        theta=phase,
        delta_nfr=pressure,
        Si=0.5,
        epi_kind="shared-form",
        glyph_history=["AL", "IL"],
        _grammar_u2_debt=0,
        _grammar_prior_coherence=True,
        _operator_step=2,
    )


def _prepared(graph_type=nx.Graph):
    """A real local derived-VAL event and UM, without a producer fixture."""
    graph = _graph(graph_type)
    _node(graph, _CORE, 0.5, pressure=0.125)
    _node(graph, _PEER, 0.125, capacity=2.0, pressure=0.125, phase=0.5)
    graph.add_edge(_PEER, _CORE, weight=0.375, purpose="old-external-port")
    if graph.is_multigraph():
        graph.add_edge(_PEER, _CORE, weight=0.0, purpose="parallel-zero")
    source = capture_retained_organization(graph, (_CORE,))
    execute_derived_expansion_stage(
        graph,
        source,
        _MEMBER,
        DerivedExpansionPolicy(
            source_order="pre_val", form_ratio=0.5, capacity_ratio=0.5
        ),
    )
    assert graph.degree(_MEMBER) == 0
    assert _state(graph, _CORE) == (0.75, 1.5, 0.25, 0.125)
    assert _state(graph, _MEMBER) == (0.25, 0.5, 0.25, 0.0)
    execute_coupling_stage(graph, Coupling(), (_CORE,), potential_partner=_MEMBER)
    return graph


def _supplied(core=0.75, member=0.25, *, mix=0.25, edge_aware=True):
    """Separately supplied collision controls, not asserted formation histories."""
    graph = _graph(mix=mix, edge_aware=edge_aware)
    _node(graph, _CORE, core)
    _node(graph, _MEMBER, member)
    _node(graph, _PEER, 0.125)
    graph.add_edge(_CORE, _MEMBER, weight=0.75, purpose="internal-port")
    graph.add_edge(_PEER, _CORE, weight=0.375, purpose="external-port")
    return graph


def _pair(graph, members=(_CORE, _MEMBER)):
    return capture_retained_pair(graph, members)


def _concentrate(graph, pair):
    return execute_retained_synthesis_stage(
        graph,
        pair,
        contraction_kwargs={"validate_preconditions": True},
        reception_kwargs={"validate_preconditions": True},
    )


def _forms(graph, members=(_CORE, _MEMBER)):
    return tuple(_form(graph, node) for node in members)


def _chart_is(pair, mean, contrast):
    assert pair.form_mean == mean
    assert pair.form_contrast == contrast
    core, member = pair.organization.member_states
    assert mean + contrast == Fraction.from_float(core.epi)
    assert mean - contrast == Fraction.from_float(member.epi)


@pytest.mark.parametrize("graph_type", [nx.Graph, nx.MultiGraph])
def test_actual_derived_members_match_independent_nul_en_composition(graph_type):
    graph = _prepared(graph_type)
    ordinary = _prepared(graph_type)
    original = _pair(graph)
    old_edges = _edges(graph)
    old_peer = deepcopy(graph.nodes[_PEER])
    phases = tuple(get_attr(graph.nodes[n], ALIAS_THETA) for n in graph)
    child_context = read_grammar_state(graph.nodes[_MEMBER])
    assert child_context.history == () and not child_context.prior_coherence
    assert read_grammar_state(graph.nodes[_CORE]).u2_debt == 1

    result = _concentrate(graph, original)
    ordinary_nul = execute_pointwise_stage(
        ordinary,
        Contraction(),
        (_CORE, _MEMBER),
        require_exact_operator=True,
        validate_preconditions=True,
    )
    ordinary_en = execute_neighbor_stage(
        ordinary,
        Reception(),
        (_CORE,),
        source_nodes=(_MEMBER,),
        track_sources=False,
        validate_preconditions=True,
    )

    assert _forms(graph) == _forms(ordinary) == (5 / 16, 1 / 8)
    assert result.before.members == result.after.members == (_CORE, _MEMBER)
    _chart_is(result.before, Fraction(1, 2), Fraction(1, 4))
    _chart_is(result.contracted, Fraction(1, 4), Fraction(1, 8))
    _chart_is(result.after, Fraction(7, 32), Fraction(3, 32))
    assert result.contraction_result.stage_result.schedule == TWO_PHASE_JACOBI
    assert result.reception_result.schedule == TWO_PHASE_JACOBI
    assert result.contraction_result.stage_result.nodes_processed == 2
    assert ordinary_nul.nodes_processed == 2
    assert result.reception_result.nodes_processed == ordinary_en.nodes_processed == 1
    observed = result.reception_result.reception_observations[0]
    assert observed.node == _CORE and observed.source_nodes == (_MEMBER,)
    assert observed.neighbors == (_MEMBER,)
    assert observed.neighbor_epi_values == (1 / 8,)
    assert observed.target_epi_before == 3 / 8
    assert observed.target_epi_after == 5 / 16
    assert not observed.source_tracking_enabled
    assert observed._proof_fields_are_intact()
    assert _edges(graph) == old_edges
    assert graph.nodes[_PEER] == old_peer
    assert tuple(get_attr(graph.nodes[n], ALIAS_THETA) for n in graph) == phases
    assert retained_organization_matches_graph(graph, result.after.organization)
    assert not retained_organization_matches_graph(graph, original.organization)
    for node in (_CORE, _MEMBER):
        assert _state(graph, node) == _state(ordinary, node)
        assert read_grammar_state(graph.nodes[node]) == read_grammar_state(
            ordinary.nodes[node]
        )
    assert _state(graph, _CORE)[1::2] == (0.75, 0.25)
    assert _state(graph, _MEMBER)[1::2] == (0.25, 0.0)
    assert 0.75 * 0.25 == 1.5 * 0.125 == 3 / 16
    assert read_grammar_state(graph.nodes[_CORE]).history == (
        "AL",
        "IL",
        "VAL",
        "UM",
        "NUL",
        "EN",
    )
    assert read_grammar_state(graph.nodes[_CORE]).u2_debt == 1
    assert read_grammar_state(graph.nodes[_MEMBER]).history == ("NUL",)
    assert not read_grammar_state(graph.nodes[_MEMBER]).prior_coherence
    assert "epi_time_history" not in graph.nodes[_MEMBER]
    assert graph.graph["_t"] == 2.0


def test_actual_peer_ra_distinguishes_compound_from_nul_only():
    responses = []
    for concentrate in (False, True):
        graph = _prepared()
        if concentrate:
            _concentrate(graph, _pair(graph))
        else:
            execute_pointwise_stage(
                graph,
                Contraction(),
                (_CORE, _MEMBER),
                require_exact_operator=True,
                validate_preconditions=True,
            )
        assert _form(graph, _MEMBER) == 1 / 8
        execute_neighbor_stage(graph, Resonance(), (_PEER,))
        responses.append(_form(graph, _PEER))
    assert responses == [1 / 4, 7 / 32]
    assert responses[1] - responses[0] == -1 / 32


def test_two_supplied_fragments_then_default_nul_form_rejection():
    graph = _prepared()
    first = _concentrate(graph, _pair(graph))
    second = _concentrate(graph, first.after)
    assert _forms(graph) == (17 / 128, 1 / 16)
    _chart_is(second.after, Fraction(25, 256), Fraction(9, 256))
    assert _state(graph, _CORE)[1::2] == (3 / 8, 1 / 2)
    assert _state(graph, _MEMBER)[1::2] == (1 / 8, 0.0)
    assert (3 / 8) * (1 / 2) == 3 / 16
    assert read_grammar_state(graph.nodes[_CORE]).u2_debt == 1
    assert read_grammar_state(graph.nodes[_MEMBER]).history == ("NUL", "NUL")
    assert "NUL_MIN_EPI" not in graph.graph
    before = _plain(graph)
    with pytest.raises(OperatorPreconditionError, match="EPI.*minimum"):
        _concentrate(graph, second.after)
    assert _plain(graph) == before
    assert retained_organization_matches_graph(graph, second.after.organization)


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiDiGraph])
def test_actual_directed_birth_access_requires_correct_reception_role(graph_type):
    graph = _prepared(graph_type)
    assert graph.has_edge(_CORE, _MEMBER)
    assert not graph.has_edge(_MEMBER, _CORE)
    before = _plain(graph)
    with pytest.raises(ValueError, match="incoming"):
        _concentrate(graph, _pair(graph))
    assert _plain(graph) == before
    edges = _edges(graph)
    result = _concentrate(graph, _pair(graph, (_MEMBER, _CORE)))
    assert result.before.core == _MEMBER and result.before.member == _CORE
    assert _forms(graph, (_MEMBER, _CORE)) == (3 / 16, 3 / 8)
    _chart_is(result.after, Fraction(9, 32), Fraction(-3, 32))
    assert _edges(graph) == edges
    assert not graph.has_edge(_MEMBER, _CORE)
    assert read_grammar_state(graph.nodes[_MEMBER]).history == ("NUL", "EN")
    assert read_grammar_state(graph.nodes[_CORE]).history[-1:] == ("NUL",)


def test_same_mean_and_same_member_count_lose_actual_continuation_response():
    responses = []
    for core, member in ((0.75, 0.25), (0.5, 0.5)):
        graph = _supplied(core, member)
        before = _pair(graph)
        assert before.form_mean == Fraction(1, 2)
        assert len(before.members) == 2
        _concentrate(graph, before)
        execute_neighbor_stage(graph, Resonance(), (_PEER,))
        responses.append(_form(graph, _PEER))
    assert responses == [7 / 32, 3 / 16]


@pytest.mark.parametrize("rejection_mode", ["raise", "fallback"])
def test_equal_primitive_chart_does_not_pool_member_grammar_debt(rejection_mode):
    graph = _prepared()
    graph.graph["GRAMMAR_REJECTION_MODE"] = rejection_mode
    admitted = _pair(graph)
    graph.nodes[_MEMBER]["_grammar_u2_debt"] = 3
    indebted = _pair(graph)
    assert admitted.form_mean == indebted.form_mean
    assert admitted.form_contrast == indebted.form_contrast
    before = _plain(graph)
    with pytest.raises(_ERRORS):
        _concentrate(graph, indebted)
    assert _plain(graph) == before
    assert read_grammar_state(graph.nodes[_MEMBER]).u2_debt == 3
    assert read_grammar_state(graph.nodes[_CORE]).history == ("AL", "IL", "VAL", "UM")


def test_default_en_pressure_gate_rejects_real_source_after_valid_nul_atomically():
    graph = _prepared()
    del graph.graph["DNFR_RECEPTION_MAX"]
    pair = _pair(graph)
    before = _plain(graph)
    with pytest.raises(_ERRORS, match="DNFR|pressure"):
        _concentrate(graph, pair)
    assert _plain(graph) == before
    assert _state(graph, _CORE)[1::2] == (1.5, 0.125)
    assert read_grammar_state(graph.nodes[_MEMBER]).history == ()
    assert retained_organization_matches_graph(graph, pair.organization)


def test_late_en_saturation_rejection_restores_valid_capacity_only_nul():
    graph = _supplied(0.95, 0.25, edge_aware=False)
    pair = _pair(graph)
    before = _plain(graph)
    with pytest.raises(_ERRORS, match="EPI|Reception"):
        _concentrate(graph, pair)
    assert _plain(graph) == before
    assert _forms(graph) == (0.95, 0.25)
    assert get_attr(graph.nodes[_CORE], ALIAS_VF) == 1.0
    assert read_grammar_state(graph.nodes[_CORE]).history == ("AL", "IL")


def test_capacity_only_nul_keeps_its_distinct_form_factor():
    graph = _supplied(edge_aware=False)
    result = _concentrate(graph, _pair(graph))
    _chart_is(result.contracted, Fraction(1, 2), Fraction(1, 4))
    assert _forms(graph) == (5 / 8, 1 / 4)
    _chart_is(result.after, Fraction(7, 16), Fraction(3, 16))
    assert get_attr(graph.nodes[_CORE], ALIAS_VF) == 0.5
    assert get_attr(graph.nodes[_MEMBER], ALIAS_VF) == 0.5


def test_mix_one_erases_previous_core_form_from_current_forms_only():
    reports = []
    for core in (0.75, 0.5):
        graph = _supplied(core, 0.25, mix=1.0)
        report = _concentrate(graph, _pair(graph))
        assert _forms(graph) == (1 / 8, 1 / 8)
        _chart_is(report.after, Fraction(1, 8), Fraction(0))
        assert report.after.members == (_CORE, _MEMBER)
        assert read_grammar_state(graph.nodes[_CORE]).history[-2:] == ("NUL", "EN")
        assert read_grammar_state(graph.nodes[_MEMBER]).history[-1:] == ("NUL",)
        reports.append(report)
    assert reports[0].before.form_contrast != reports[1].before.form_contrast
    assert reports[0].after.form_mean == reports[1].after.form_mean
    assert reports[0].after.form_contrast == reports[1].after.form_contrast


def test_zero_en_mix_remains_invalid_and_does_not_replace_no_en_control():
    graph = _supplied(mix=0.0)
    before = _plain(graph)
    with pytest.raises(_ERRORS, match="EN_mix"):
        _concentrate(graph, _pair(graph))
    assert _plain(graph) == before


@pytest.mark.parametrize(
    "core,member,mean,contrast",
    [
        (-0.75, -0.25, Fraction(-7, 32), Fraction(-3, 32)),
        (0.5, -0.5, Fraction(-1, 16), Fraction(3, 16)),
        (0.0, 0.25, Fraction(5, 64), Fraction(-3, 64)),
    ],
)
def test_signed_and_zero_forms_keep_actual_roles_without_ratio_assumption(
    core, member, mean, contrast
):
    graph = _supplied(core, member)
    graph.graph["NUL_MIN_EPI"] = -1.0
    result = _concentrate(graph, _pair(graph))
    _chart_is(result.after, mean, contrast)
    assert _forms(graph) == (0.5 * (0.75 * core + 0.25 * member), 0.5 * member)
    if core < 0 and member < 0:
        assert abs(result.after.form_contrast / result.after.form_mean) < abs(
            result.before.form_contrast / result.before.form_mean
        )
    if core == -member:
        assert result.before.form_mean == 0
        assert result.after.form_mean != 0


def test_opposite_sign_normalized_contrast_can_increase():
    graph = _supplied(0.75, -0.25, mix=0.375)
    graph.graph["NUL_MIN_EPI"] = -1.0
    result = _concentrate(graph, _pair(graph))
    assert result.before.form_contrast / result.before.form_mean == 2
    _chart_is(result.after, Fraction(1, 32), Fraction(5, 32))
    assert result.after.form_contrast / result.after.form_mean == 5
    assert abs(result.after.form_contrast) < abs(result.before.form_contrast)


def test_active_hard_projection_is_retained_instead_of_ideal_linear_formula():
    graph = _supplied()
    graph.graph["EPI_MIN"] = 0.25
    graph.nodes[_PEER]["EPI"] = 0.25
    result = _concentrate(graph, _pair(graph))
    assert tuple(s.epi for s in result.contracted.organization.member_states) == (
        3 / 8,
        1 / 4,
    )
    assert _forms(graph) == (11 / 32, 1 / 4)
    _chart_is(result.after, Fraction(19, 64), Fraction(3, 64))
    assert result.after.form_mean != Fraction(7, 32)


def test_exact_chart_retains_half_subnormal_without_executing_on_it():
    smallest = math.ulp(0.0)
    graph = _supplied(smallest, 0.0)
    before = _plain(graph)
    pair = _pair(graph)
    half = Fraction.from_float(smallest) / 2
    assert half != 0 and float(half) == 0.0
    _chart_is(pair, half, half)
    assert _plain(graph) == before


def test_exact_chart_avoids_overflow_in_finite_mean_without_action():
    maximum = sys.float_info.max
    graph = _supplied(maximum, maximum)
    before = _plain(graph)
    pair = _pair(graph)
    _chart_is(pair, Fraction.from_float(maximum), Fraction(0))
    assert math.isinf(maximum + maximum)
    assert _plain(graph) == before


def test_exact_real_chart_does_not_claim_signed_zero_encoding():
    graph = _supplied(-0.0, 0.0)
    assert math.copysign(1.0, graph.nodes[_CORE]["EPI"]) == -1.0
    before = _plain(graph)
    pair = _pair(graph)
    _chart_is(pair, Fraction(0), Fraction(0))
    # The scalar semantic reader may normalize zero; raw retained state must not.
    assert math.copysign(1.0, graph.nodes[_CORE]["EPI"]) == -1.0
    assert math.copysign(1.0, graph.nodes[_MEMBER]["EPI"]) == 1.0
    assert _plain(graph) == before
