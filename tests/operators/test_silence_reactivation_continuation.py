"""Prospective SHA/NAV continuation, with own memory and addressed real ports.

Five finite cases follow silence-reactivation-20261103/math-protocol.md.
Schedules and members are supplied; equality of response is not state equality.
"""

from copy import deepcopy
from datetime import datetime
from fractions import Fraction as Q

import networkx as nx
import pytest

from tests.operators._port_reexpression_source import A, B, R, U, V, W, _source
from tnfr.alias import get_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.errors import TNFRValueError
from tnfr.operators.coherence import Coherence
from tnfr.operators.emission import Emission
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import execute_neighbor_stage, execute_pointwise_stage
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.reception import Reception
from tnfr.operators.resonance import Resonance
from tnfr.operators.retained_emission import execute_retained_emission_stage
from tnfr.operators.retained_organization import capture_retained_organization
from tnfr.operators.retained_reception import (
    execute_retained_output_reception_stage,
    retained_output_ports,
)
from tnfr.operators.silence import Silence
from tnfr.operators.transition import Transition
from tnfr.types import require_finite_real_scalar_epi
from tnfr.utils._structural_signature import structural_proof_signature

C = "supplied-member-C"
SYNTHETIC_NODES = (A, C, U, R, W)


def _form(graph, node):
    return Q.from_float(
        require_finite_real_scalar_epi(
            get_attr(graph.nodes[node], ALIAS_EPI, strict=True, conv=lambda x: x)
        )
    )


def _scalar(graph, node, alias):
    return Q.from_float(get_attr(graph.nodes[node], alias, strict=True))


def _mean(graph, group):
    return sum((_form(graph, n) for n in group), Q(0)) / len(group)


def _support(graph):
    return deepcopy(tuple(graph.edges(data=True)))


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


def _stage(graph, operator, targets, **kwargs):
    before = {n: read_grammar_state(graph.nodes[n]) for n in targets}
    clock = graph.graph["_t"]
    result = execute_pointwise_stage(
        graph,
        operator,
        targets,
        require_exact_operator=True,
        validate_preconditions=True,
        **kwargs,
    )
    assert result.glyph == operator.glyph.value
    assert result.nodes_processed == len(targets)
    assert graph.graph["_t"] == clock
    for node in targets:
        after = read_grammar_state(graph.nodes[node])
        assert after.operator_step == before[node].operator_step + 1
        assert after.history[-1] == operator.glyph.value
    return result


def _cohere(graph, targets):
    before = {
        node: (
            _form(graph, node),
            _scalar(graph, node, ALIAS_VF),
            _scalar(graph, node, ALIAS_THETA),
            _scalar(graph, node, ALIAS_DNFR),
        )
        for node in targets
    }
    _stage(graph, Coherence(), targets, phase_locking_coefficient=0.0)
    for node, (form, capacity, phase, pressure) in before.items():
        assert _form(graph, node) == form
        assert _scalar(graph, node, ALIAS_VF) == capacity
        assert _scalar(graph, node, ALIAS_THETA) == phase
        assert _scalar(graph, node, ALIAS_DNFR) == pressure / 2


def _configure_continuation(graph):
    graph.graph.update(NAV_STRICT=True, NAV_RANDOM=False)
    graph.graph["GLYPH_FACTORS"].update(
        NAV_jitter=0.0,
        IL_dnfr_factor=0.5,
        EN_mix=0.5,
        RA_epi_diff=0.5,
        RA_vf_amplification=0.125,
    )


def _synthetic_source():
    graph = nx.Graph()
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        IL_PRECONDITIONS={"warn_zero_dnfr": False},
        CLIP_MODE="hard",
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
    )
    _configure_continuation(graph)
    for node, value in zip(
        SYNTHETIC_NODES, (Q(3, 4), Q(1, 8), Q(1, 2), Q(1, 4), Q(1, 4)), strict=True
    ):
        graph.add_node(
            node,
            EPI=float(value),
            nu_f=15 / 16 if node == C else 1.0,
            theta=0.0,
            delta_nfr=1 / 8 if node == C else 0.0,
            Si=0.0,
            epi_kind="form",
        )
        assert read_grammar_state(graph.nodes[node]).history == ()
    for left, right in ((A, C), (U, C), (A, R), (C, R), (R, W)):
        graph.add_edge(left, right, weight=1.0)
    _cohere(graph, SYNTHETIC_NODES)
    graph.graph["GLYPH_FACTORS"]["AL_boost"] = 1 / 8
    _stage(graph, Emission(), (C,))
    assert _form(graph, C) == Q(1, 4)
    assert _scalar(graph, C, ALIAS_DNFR) == Q(1, 16)
    assert _scalar(graph, C, ALIAS_VF) == Q(15, 16)
    assert _mean(graph, (A, C)) == Q(1, 2)
    assert read_grammar_state(graph.nodes[C]).history == ("IL", "AL")
    assert graph.nodes[C]["_structural_lineage"]["activation_count"] == 1
    assert "epi_time_history" not in graph.nodes[C]
    assert tuple(graph.neighbors(W)) == (R,)
    return graph


def _silence(graph, node, factor):
    graph.graph["GLYPH_FACTORS"]["SHA_vf_factor"] = float(factor)
    before = (
        _form(graph, node),
        _scalar(graph, node, ALIAS_VF),
        _scalar(graph, node, ALIAS_DNFR),
        _scalar(graph, node, ALIAS_THETA),
    )
    others = {n: deepcopy(data) for n, data in graph.nodes(data=True) if n != node}
    lineage = deepcopy(graph.nodes[node].get("_structural_lineage"))
    origin = graph.nodes[node].get("_emission_origin")
    support = _support(graph)
    _stage(graph, Silence(), (node,))
    form, capacity, pressure, phase = before
    assert _form(graph, node) == form
    assert _scalar(graph, node, ALIAS_VF) == Q.from_float(
        float(capacity) * float(factor)
    )
    assert _scalar(graph, node, ALIAS_DNFR) == pressure
    assert _scalar(graph, node, ALIAS_THETA) == phase
    assert _support(graph) == support
    assert all(graph.nodes[n] == data for n, data in others.items())
    data = graph.nodes[node]
    assert data.get("_structural_lineage") == lineage
    assert data.get("_emission_origin") == origin
    assert data["latent"] is True and data["preserved_epi"] == float(form)
    assert data["silence_duration"] == 0.0
    assert data["was_initial_on_silence"] is False
    timestamp = datetime.fromisoformat(data["latency_start_time"])
    assert timestamp.tzinfo is not None and timestamp.utcoffset() is not None


def _transition(graph, node):
    data = graph.nodes[node]
    form, support = _form(graph, node), _support(graph)
    capacity = _scalar(graph, node, ALIAS_VF)
    pressure = _scalar(graph, node, ALIAS_DNFR)
    lineage, origin = deepcopy(data.get("_structural_lineage")), data.get(
        "_emission_origin"
    )
    event_count = len(graph.graph.get("_nav_transitions", ()))
    _stage(graph, Transition(), (node,), phase_shift=0.125)
    assert _form(graph, node) == form and _support(graph) == support
    assert _scalar(graph, node, ALIAS_THETA) == Q(1, 8)
    assert data.get("_structural_lineage") == lineage
    assert data.get("_emission_origin") == origin
    assert all(
        key not in data for key in ("latent", "latency_start_time", "preserved_epi")
    )
    assert data["silence_duration"] >= 0.0
    assert data["was_initial_on_silence"] is False
    assert data["_regime_before"] == "latent"
    assert "_rng_jitter_progress" not in data
    assert len(graph.graph["_nav_transitions"]) == event_count + 1
    event = graph.graph["_nav_transitions"][-1]
    assert event["node"] == node and event["regime_origin"] == "latent"
    assert event["vf_before"] == float(capacity)
    assert event["dnfr_before"] == float(pressure)
    assert event["dnfr_handler_after"] == float(capacity) != float(pressure)
    assert event["phase_shift_requested"] == 1 / 8


def _gain(graph, group, index):
    node = group[index]
    graph.graph["GLYPH_FACTORS"]["AL_boost"] = len(group) / 16
    record = capture_retained_organization(graph, group)
    before, mean = _form(graph, node), _mean(graph, group)
    context = read_grammar_state(graph.nodes[node])
    origin = graph.nodes[node].get("_emission_origin")
    count = graph.nodes[node].get("_structural_lineage", {}).get("activation_count", 0)
    held = tuple(_scalar(graph, node, a) for a in (ALIAS_VF, ALIAS_DNFR, ALIAS_THETA))
    kind = get_attr(graph.nodes[node], ALIAS_EPI_KIND, strict=True, conv=str)
    support = _support(graph)
    result = execute_retained_emission_stage(graph, record, index, mean_gain=Q(1, 16))
    assert result.member == node and result.stage_result.glyph == "AL"
    assert result.stage_result.nodes_processed == 1
    assert result.realized_mean_gain == result.mean_gain == Q(1, 16)
    assert _form(graph, node) == before + Q(len(group), 16)
    assert _mean(graph, group) == mean + Q(1, 16)
    assert (
        tuple(_scalar(graph, node, a) for a in (ALIAS_VF, ALIAS_DNFR, ALIAS_THETA))
        == held
    )
    assert get_attr(graph.nodes[node], ALIAS_EPI_KIND, strict=True, conv=str) == kind
    assert _support(graph) == support
    assert (
        read_grammar_state(graph.nodes[node]).operator_step == context.operator_step + 1
    )
    assert read_grammar_state(graph.nodes[node]).history[-1] == "AL"
    lineage = graph.nodes[node]["_structural_lineage"]
    assert lineage["activation_count"] == count + 1
    assert lineage["origin"] == graph.nodes[node]["_emission_origin"]
    if origin is not None:
        assert graph.nodes[node]["_emission_origin"] == origin
    else:
        assert lineage["parent_emission"] is None and lineage["derived_nodes"] == []


def _receive(graph, targets, sources):
    result = execute_neighbor_stage(
        graph,
        Reception(),
        targets,
        source_nodes=sources,
        track_sources=False,
        validate_preconditions=True,
    )
    assert result.glyph == "EN" and result.nodes_processed == len(targets)
    assert all(row.source_nodes == sources for row in result.reception_observations)
    return result


def _output(graph, group):
    record = capture_retained_organization(graph, group)
    (port,) = tuple(p for p in retained_output_ports(record) if p.receiver == R)
    before = {node: deepcopy(graph.nodes[node]) for node in group}
    result = execute_retained_output_reception_stage(graph, record, port.ordinal)
    assert result.stage_result.glyph == "EN"
    (observation,) = result.stage_result.reception_observations
    assert observation.node == R and observation.source_nodes == group
    assert all(graph.nodes[n] == data for n, data in before.items())


def _resonate(graph):
    assert tuple(graph.neighbors(W)) == (R,)
    before = {
        node: deepcopy(data) for node, data in graph.nodes(data=True) if node != W
    }
    context = read_grammar_state(graph.nodes[W])
    Resonance()(graph, W, validate_preconditions=True)
    after = read_grammar_state(graph.nodes[W])
    assert (
        after.operator_step == context.operator_step + 1 and after.history[-1] == "RA"
    )
    assert _scalar(graph, W, ALIAS_VF) == Q(9, 8)
    assert all(graph.nodes[n] == data for n, data in before.items())


def test_positive_silence_reactivation_preserves_the_marked_response_not_state():
    observed = {}
    for recovery in (False, True):
        graph = _synthetic_source()
        support = _support(graph)
        origin = graph.nodes[C]["_emission_origin"]
        if recovery:
            _silence(graph, C, Q(1, 2))
            assert _scalar(graph, C, ALIAS_VF) == Q(15, 32) < Q(1, 2)
            _transition(graph, C)
            assert _scalar(graph, C, ALIAS_VF) == Q(9, 16)
            assert _scalar(graph, C, ALIAS_DNFR) == Q(21, 64)
        _cohere(graph, (C,))
        _cohere(graph, (C,))
        assert _scalar(graph, C, ALIAS_DNFR) == (Q(21, 256) if recovery else Q(1, 64))
        _gain(graph, (A, C), 1)
        assert _form(graph, C) == Q(3, 8)
        _receive(graph, (C,), (U,))
        assert _form(graph, C) == Q(7, 16)
        assert _mean(graph, (A, C)) == Q(19, 32)
        _output(graph, (A, C))
        assert _form(graph, R) == Q(27, 64)
        _resonate(graph)
        assert _form(graph, W) == Q(43, 128)
        assert _support(graph) == support and graph.graph["_t"] == 0.0
        assert graph.nodes[C]["_emission_origin"] == origin
        assert graph.nodes[C]["_structural_lineage"]["activation_count"] == 2
        expected = (
            ("IL", "AL", "SHA", "NAV", "IL", "IL", "AL", "EN")
            if recovery
            else ("IL", "AL", "IL", "IL", "AL", "EN")
        )
        state = read_grammar_state(graph.nodes[C])
        assert state.history == expected[-7:] and state.operator_step == len(expected)
        assert state.prior_coherence and state.u2_debt == 0
        for node, count in ((A, 1), (U, 1), (R, 2), (W, 2)):
            assert read_grammar_state(graph.nodes[node]).operator_step == count
        assert _scalar(graph, C, ALIAS_VF) == (Q(9, 16) if recovery else Q(15, 16))
        assert _scalar(graph, C, ALIAS_THETA) == (Q(1, 8) if recovery else Q(0))
        assert "epi_time_history" not in graph.nodes[C]
        observed[recovery] = (_form(graph, R), _form(graph, W))
    assert observed[False] == observed[True] == (Q(27, 64), Q(43, 128))


def test_silence_capacity_is_a_sufficient_strict_al_obstruction():
    graph = _synthetic_source()
    _silence(graph, C, Q(1, 2))
    graph.graph["GLYPH_FACTORS"]["AL_boost"] = 1 / 8
    record = capture_retained_organization(graph, (A, C))
    assert _scalar(graph, C, ALIAS_VF) == Q(15, 32)
    before = _checkpoint(graph)
    with pytest.raises(TNFRValueError, match="frequency too low"):
        execute_retained_emission_stage(graph, record, 1, mean_gain=Q(1, 16))
    assert _checkpoint(graph) == before
    assert graph.nodes[C]["latent"] is True
    assert graph.nodes[C]["_structural_lineage"]["activation_count"] == 1


def test_one_il_after_nav_allows_al_but_reception_still_rejects_pressure():
    graph = _synthetic_source()
    _silence(graph, C, Q(1, 2))
    _transition(graph, C)
    _cohere(graph, (C,))
    assert _scalar(graph, C, ALIAS_DNFR) == Q(21, 128)
    _gain(graph, (A, C), 1)
    assert _form(graph, C) == Q(3, 8)
    before = _checkpoint(graph)
    with pytest.raises(TNFRValueError, match="Stored DNFR exceeds"):
        _receive(graph, (C,), (U,))
    assert _checkpoint(graph) == before
    assert graph.nodes[C]["_structural_lineage"]["activation_count"] == 2
    assert read_grammar_state(graph.nodes[C]).history == (
        "IL",
        "AL",
        "SHA",
        "NAV",
        "IL",
        "AL",
    )


def test_zero_silence_capacity_rejects_nav_and_al_without_erasing_memory():
    graph = _synthetic_source()
    _silence(graph, C, Q(0))
    assert _scalar(graph, C, ALIAS_VF) == Q(0)
    before = _checkpoint(graph)
    with pytest.raises(OperatorPreconditionError, match="frequency too low"):
        _stage(graph, Transition(), (C,), phase_shift=0.125)
    assert _checkpoint(graph) == before
    graph.graph["GLYPH_FACTORS"]["AL_boost"] = 1 / 8
    record = capture_retained_organization(graph, (A, C))
    before = _checkpoint(graph)
    with pytest.raises(TNFRValueError, match="frequency too low"):
        execute_retained_emission_stage(graph, record, 1, mean_gain=Q(1, 16))
    assert _checkpoint(graph) == before
    assert graph.nodes[C]["latent"] is True
    assert graph.nodes[C]["_structural_lineage"]["activation_count"] == 1
    assert read_grammar_state(graph.nodes[C]).history == ("IL", "AL", "SHA")


def test_one_acquired_coarse_refined_pair_survives_the_reactivation_prefix():
    observed = {}
    for refined in (False, True):
        graph, group, child = _source(refined=refined)
        _configure_continuation(graph)
        target, index = (child, 1) if refined else (A, 0)
        initial_forms = tuple(_form(graph, n) for n in group)
        support = _support(graph)
        assert Q.from_float(0.95) <= _scalar(graph, target, ALIAS_VF) <= Q(1)
        if not refined:
            assert _scalar(graph, target, ALIAS_VF) == Q(1)
        _silence(graph, target, Q(15, 32))
        assert Q(44, 100) < _scalar(graph, target, ALIAS_VF) < Q(47, 100) < Q(1, 2)
        _transition(graph, target)
        assert Q(1, 2) < Q(53, 100) < _scalar(graph, target, ALIAS_VF) < Q(57, 100)
        _cohere(graph, (target,))
        _cohere(graph, (target,))
        assert Q(0) < _scalar(graph, target, ALIAS_DNFR) < Q(83, 1000) < Q(15, 100)
        assert tuple(_form(graph, n) for n in group) == initial_forms
        _gain(graph, group, index)
        assert _mean(graph, group) == Q(29, 64)
        for donor, boost, expected in ((U, Q(1, 8), Q(1, 4)), (V, Q(3, 8), Q(1, 2))):
            graph.graph["GLYPH_FACTORS"]["AL_boost"] = float(boost)
            _stage(graph, Emission(), (donor,))
            assert _form(graph, donor) == expected
        _receive(graph, group, (U,))
        _receive(graph, (B,), (V,))
        assert _mean(graph, group) == Q(45, 128)
        assert _form(graph, B) == Q(7, 16)
        if refined:
            assert (_form(graph, A) - _form(graph, child)) / 2 == Q(19, 512)
        _output(graph, group)
        assert _form(graph, R) == Q(77, 256)
        first = _form(graph, R)
        _output(graph, (B,))
        assert _form(graph, R) == Q(189, 512)
        second = _form(graph, R)
        _resonate(graph)
        assert _form(graph, W) == Q(381, 1024)
        assert _support(graph) == support and graph.graph["_t"] == 0.25
        for node, count in (
            (A, 13 if refined else 15),
            (B, 7),
            (U, 2),
            (V, 2),
            (R, 3),
            (W, 2),
        ):
            state = read_grammar_state(graph.nodes[node])
            assert state.operator_step == count and state.prior_coherence
        assert graph.nodes[A]["_structural_lineage"]["activation_count"] == 3
        if refined:
            state = read_grammar_state(graph.nodes[child])
            assert (
                state.operator_step == 9
                and state.prior_coherence
                and state.u2_debt == 0
            )
            assert graph.nodes[child]["_structural_lineage"]["activation_count"] == 1
            assert "epi_time_history" not in graph.nodes[child]
        observed[refined] = (first, second, _form(graph, W))
    assert observed[False] == observed[True] == (Q(77, 256), Q(189, 512), Q(381, 1024))
