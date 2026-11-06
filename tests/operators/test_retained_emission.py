"""Exact retained AL admission on supplied small states and unit contexts.

These are not acquired child histories. The separate continuation fixture owns
the actual THOL/UM source, broadcast obstruction and EN/RA response family.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EMISSION_TIMESTAMP,
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_VF,
)
from tnfr.operators import retained_emission as owner
from tnfr.operators.emission import Emission
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import execute_pointwise_stage
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.retained_emission import execute_retained_emission_stage
from tnfr.operators.retained_organization import (
    capture_retained_organization,
    retained_organization_matches_graph,
)
from tnfr.types import require_finite_real_scalar_epi
from tnfr.utils._structural_signature import structural_proof_signature

A, C, R = "a", "child", "receiver"


def _graph():
    graph = nx.Graph(
        _t=0.25,
        RANDOM_SEED=17,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        GLYPH_FACTORS={"AL_boost": 0.125},
    )
    for node, value in ((A, 0.25), (C, 0.125), (R, 0.375)):
        graph.add_node(
            node,
            EPI=value,
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0625,
            epi_kind="form",
            glyph_history=["IL"],
            private_memory={"node": node},
        )
    graph.add_edge(A, C, weight=1.0, role="interior")
    graph.add_edge(A, R, weight=0.25, role="output")
    graph.add_edge(C, R, weight=0.5, role="output")
    return graph


def _form(graph, node):
    return require_finite_real_scalar_epi(
        get_attr(graph.nodes[node], ALIAS_EPI, strict=True, conv=lambda value: value)
    )


def _checkpoint(graph):
    return structural_proof_signature(
        (
            graph.graph,
            tuple(graph.nodes(data=True)),
            tuple(graph.edges(data=True)),
            hasattr(graph, "_last_operator_applied"),
            getattr(graph, "_last_operator_applied", None),
        ),
        opaque_references=(graph,),
    )


def _record(graph, members=(A, C)):
    return capture_retained_organization(graph, members)


def _execute(graph, record=None, index=1, gain=Q(1, 16), **kwargs):
    return execute_retained_emission_stage(
        graph,
        _record(graph) if record is None else record,
        index,
        mean_gain=gain,
        **kwargs,
    )


@pytest.mark.parametrize("reactivation", [False, True])
def test_exact_singleton_actual_al_and_own_activation(reactivation):
    graph = _graph()
    graph.graph["GLYPH_FACTORS"]["AL_boost"] = 0.0625
    if reactivation:
        execute_pointwise_stage(graph, Emission(), (A,), require_exact_operator=True)
    record = _record(graph, (A,))
    before_form = _form(graph, A)
    before_context = read_grammar_state(graph.nodes[A])
    origin = graph.nodes[A].get("_emission_origin")
    other_nodes = {node: deepcopy(graph.nodes[node]) for node in (C, R)}
    factor_source = graph.graph["GLYPH_FACTORS"]
    direct = deepcopy(graph)
    result = _execute(graph, record, index=0)
    execute_pointwise_stage(direct, Emission(), (A,), require_exact_operator=True)
    assert result.member == A and result.member_index == 0
    assert result.mean_gain == result.realized_mean_gain == Q(1, 16)
    assert result.local_boost == 0.0625
    assert _form(graph, A) == _form(direct, A) == before_form + 0.0625
    assert read_grammar_state(graph.nodes[A]) == read_grammar_state(direct.nodes[A])
    assert (
        read_grammar_state(graph.nodes[A]).operator_step
        == before_context.operator_step + 1
    )
    lineage = graph.nodes[A]["_structural_lineage"]
    assert lineage["activation_count"] == (2 if reactivation else 1)
    assert lineage["parent_emission"] is None
    assert lineage["origin"] == graph.nodes[A]["_emission_origin"]
    assert (
        get_attr(graph.nodes[A], ALIAS_EMISSION_TIMESTAMP, conv=str)
        == lineage["origin"]
    )
    if reactivation:
        assert lineage["origin"] == origin
    assert all(graph.nodes[node] == data for node, data in other_nodes.items())
    assert graph.graph["GLYPH_FACTORS"] is factor_source
    assert graph.graph["_t"] == 0.25
    assert retained_organization_matches_graph(graph, result.after)
    assert not retained_organization_matches_graph(graph, record)


def test_unequal_group_selects_only_supplied_latent_member():
    graph = _graph()
    set_attr(graph.nodes[A], ALIAS_EPI, 0.75)
    other = deepcopy(graph.nodes[A])
    result = _execute(graph)
    assert result.member == C and result.realized_mean_gain == Q(1, 16)
    assert _form(graph, C) == 0.25 and graph.nodes[A] == other
    assert read_grammar_state(graph.nodes[C]).history == ("IL", "AL")
    assert graph.nodes[C]["_structural_lineage"]["activation_count"] == 1


def test_al_preserves_actual_kind_aliases_none_and_absence():
    for value in (None, "absent"):
        graph = _graph()
        if value == "absent":
            del graph.nodes[C]["epi_kind"]
        else:
            graph.nodes[C]["epi_kind"] = None
        before = {
            key: graph.nodes[C][key] for key in ALIAS_EPI_KIND if key in graph.nodes[C]
        }
        _execute(graph)
        assert {
            key: graph.nodes[C][key] for key in ALIAS_EPI_KIND if key in graph.nodes[C]
        } == before


@pytest.mark.parametrize("field", ["form", "history", "support"])
def test_stale_retained_source_rejects_without_undoing_entry_changes(field):
    graph = _graph()
    record = _record(graph)
    if field == "form":
        set_attr(graph.nodes[C], ALIAS_EPI, 0.25)
    elif field == "history":
        graph.nodes[C]["epi_history"] = [0.125]
    else:
        graph.edges[A, C]["weight"] = 0.75
    before = _checkpoint(graph)
    with pytest.raises(ValueError, match="foreign|stale|unsealed"):
        _execute(graph, record)
    assert _checkpoint(graph) == before


def test_foreign_altered_records_and_inconsistent_summary_are_not_authority():
    graph = _graph()
    record = _record(graph)
    before = _checkpoint(graph)
    for invalid in (_record(deepcopy(graph)), replace(record, members=(C, A))):
        with pytest.raises(ValueError):
            _execute(graph, invalid)
        assert _checkpoint(graph) == before
    actual = _execute(graph, record)
    with pytest.raises(ValueError):
        replace(actual, member_index=0)
    with pytest.raises(ValueError):
        replace(actual, mean_gain=Q(1, 8))
    uncertified = replace(actual.stage_result, neighbor_epi_jump_certificate=object())
    with pytest.raises(ValueError):
        replace(actual, stage_result=uncertified)
    object.__setattr__(actual.stage_result, "reception_observations", ("invalid",))
    with pytest.raises((TypeError, ValueError)):
        replace(actual, stage_result=actual.stage_result)


def test_invalid_member_ordinal_has_no_search_or_negative_index_fallback():
    graph = _graph()
    record = _record(graph)
    before = _checkpoint(graph)
    for index in (True, -1, 2, 1.0, "1"):
        with pytest.raises((TypeError, IndexError)):
            _execute(graph, record, index=index)
        assert _checkpoint(graph) == before


def test_invalid_gain_and_consumed_factor_are_errors_not_abstention():
    for value in (
        True,
        float("nan"),
        float("inf"),
        0.0,
        -0.125,
        "0.0625",
        Q(1, 10**400),
    ):
        graph = _graph()
        before = _checkpoint(graph)
        with pytest.raises((TypeError, ValueError)):
            _execute(graph, gain=value)
        assert _checkpoint(graph) == before
    for value in (True, float("nan"), float("inf"), 0.0, -0.125):
        graph = _graph()
        graph.graph["GLYPH_FACTORS"]["AL_boost"] = value
        before = _checkpoint(graph)
        with pytest.raises((TypeError, ValueError)):
            _execute(graph)
        assert _checkpoint(graph) == before
    graph = _graph()
    before = _checkpoint(graph)

    class ActiveGain(float):
        def __float__(self):
            set_attr(graph.nodes[R], ALIAS_EPI, 0.75)
            return float.__float__(self)

    with pytest.raises(RuntimeError, match="admission changed source"):
        _execute(graph, gain=ActiveGain(0.0625))
    assert _checkpoint(graph) == before


def test_configured_factor_must_match_declared_mean_gain():
    graph = _graph()
    before = _checkpoint(graph)
    with pytest.raises(ValueError, match="AL_boost must equal"):
        _execute(graph, gain=Q(1, 32))
    assert _checkpoint(graph) == before


def test_binary64_rounding_is_not_reported_as_exact_mean_translation():
    graph = _graph()
    graph.graph["GLYPH_FACTORS"]["AL_boost"] = 0.1
    set_attr(graph.nodes[C], ALIAS_EPI, 0.25)
    assert Q.from_float(0.1) == 2 * Q.from_float(0.05)
    before = _checkpoint(graph)
    with pytest.raises(ValueError, match="rounded translations"):
        _execute(graph, gain=0.05)
    assert _checkpoint(graph) == before


def test_projection_rejects_only_the_exact_gain_contract():
    graph = _graph()
    graph.graph["EPI_MAX"] = 0.1875
    before = _checkpoint(graph)
    with pytest.raises(ValueError, match="projected"):
        _execute(graph)
    assert _checkpoint(graph) == before
    # Ordinary AL retains its wider projected map and its actual activation.
    execute_pointwise_stage(graph, Emission(), (C,), require_exact_operator=True)
    assert _form(graph, C) == 0.1875
    assert graph.nodes[C]["_structural_lineage"]["activation_count"] == 1


@pytest.mark.parametrize("field,value", [(ALIAS_EPI, 0.5), (ALIAS_VF, 0.25)])
def test_strict_entry_admission_is_not_disabled_by_global_optional_flag(field, value):
    graph = _graph()
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = False
    set_attr(graph.nodes[C], field, value)
    before = _checkpoint(graph)
    with pytest.raises(ValueError, match="AL precondition"):
        _execute(graph)
    assert _checkpoint(graph) == before


def test_local_policy_and_option_shape_are_mandatory():
    cases = (
        ({}, {"validate_preconditions": False}),
        ({"GRAMMAR_CANON": {"enabled": False}}, None),
        ({"GRAMMAR_REJECTION_MODE": "fallback"}, None),
        ({}, {"AL_boost": 0.125}),
        ({}, {"window": True}),
        ({}, {"sequence_context": ("IL",)}),
    )
    for updates, options in cases:
        graph = _graph()
        graph.graph.update(updates)
        before = _checkpoint(graph)
        with pytest.raises((TypeError, ValueError, OperatorPreconditionError)):
            _execute(graph, execution_kwargs=options)
        assert _checkpoint(graph) == before


def test_modified_canonical_execution_and_precondition_methods_reject(monkeypatch):
    graph = _graph()
    record = _record(graph)
    before = _checkpoint(graph)
    for method in owner._METHODS:
        with monkeypatch.context() as patch:
            patch.setattr(Emission, method, lambda *args, **kwargs: None)
            with pytest.raises(TypeError, match="canonical bindings"):
                _execute(graph, record)
        assert _checkpoint(graph) == before

    class InheritedEmission(Emission):
        pass

    with monkeypatch.context() as patch:
        patch.setattr(owner, "Emission", InheritedEmission)
        with pytest.raises(TypeError, match="exact canonical class"):
            _execute(graph, record)
    assert _checkpoint(graph) == before


def test_legacy_activated_flag_without_lineage_cannot_claim_counted_realization():
    graph = _graph()
    graph.nodes[C]["_emission_activated"] = True
    before = _checkpoint(graph)
    with pytest.raises(ValueError, match="counted activation lineage"):
        _execute(graph)
    assert _checkpoint(graph) == before


@pytest.mark.parametrize("fault", ["form", "pressure", "nonselected", "lifecycle"])
def test_silent_post_stage_corruption_rolls_back_actual_al(monkeypatch, fault):
    # The lifecycle family also checks the separately owned grammar advance.
    for subfault in (("count", "grammar") if fault == "lifecycle" else (fault,)):
        graph = _graph()
        record = _record(graph)
        before = _checkpoint(graph)
        actual_stage = owner.execute_pointwise_stage

        def corrupt(*args, **kwargs):
            result = actual_stage(*args, **kwargs)
            assert _form(graph, C) == 0.25
            assert graph.nodes[C]["_structural_lineage"]["activation_count"] == 1
            if subfault == "form":
                set_attr(graph.nodes[C], ALIAS_EPI, 0.5)
            elif subfault == "pressure":
                set_attr(graph.nodes[C], ALIAS_DNFR, 0.5)
            elif subfault == "nonselected":
                graph.nodes[A]["private_memory"]["extra"] = True
            elif subfault == "count":
                graph.nodes[C]["_structural_lineage"]["activation_count"] = 99
            else:
                graph.nodes[C]["_operator_step"] += 1
            return result

        with monkeypatch.context() as patch:
            patch.setattr(owner, "execute_pointwise_stage", corrupt)
            with pytest.raises((RuntimeError, ValueError)):
                _execute(graph, record)
        assert _checkpoint(graph) == before
        assert retained_organization_matches_graph(graph, record)


@pytest.mark.parametrize("raises", [False, True])
def test_options_mapping_cannot_retarget_or_change_configured_factor(raises):
    graph = _graph()
    record = _record(graph)
    before = _checkpoint(graph)

    class Options(Mapping):
        def __len__(self):
            return 1

        def __getitem__(self, key):
            if key != "collect_metrics":
                raise KeyError(key)
            return False

        def __iter__(self):
            set_attr(graph.nodes[C], ALIAS_EPI, 0.75)
            graph.graph["GLYPH_FACTORS"]["AL_boost"] = 0.5
            graph.remove_edge(A, C)
            if raises:
                raise RuntimeError("options materialization")
            yield "collect_metrics"

    if raises:
        with pytest.raises(RuntimeError, match="options materialization"):
            _execute(graph, record, execution_kwargs=Options())
        assert _checkpoint(graph) == before
    else:
        result = _execute(graph, record, execution_kwargs=Options())
        assert result.local_boost == 0.125 and result.realized_mean_gain == Q(1, 16)
        assert _form(graph, C) == 0.25 and graph.has_edge(A, C)
        assert graph.graph["GLYPH_FACTORS"]["AL_boost"] == 0.125
