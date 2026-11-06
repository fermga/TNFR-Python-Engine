"""Bounded actual UM renewal controls; no autonomous or continuous-time claim.

The prospective maintenance-return-20261103 protocol owns the three cases.
All source context is acquired here; no previous continuation is imported.
"""

import math
from copy import deepcopy
from datetime import datetime
from fractions import Fraction as Q

import networkx as nx
import pytest

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
from tnfr.operators.coupling import Coupling
from tnfr.operators.emission import Emission
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import (
    execute_coupling_stage,
    execute_neighbor_stage,
    execute_pointwise_stage,
)
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

A, C, U, R, W = "member-A", "member-C", "input-U", "receiver-R", "observer-W"
NODES = (A, C, U, R, W)
EDGES = ((A, C), (U, C), (A, R), (C, R), (R, W))
ABS_ERROR = 2.0**-40


def _form(graph, node):
    return Q.from_float(
        require_finite_real_scalar_epi(
            get_attr(graph.nodes[node], ALIAS_EPI, strict=True, conv=lambda x: x)
        )
    )


def _scalar(graph, node, alias):
    return Q.from_float(get_attr(graph.nodes[node], alias, strict=True))


def _close(actual, expected):
    assert float(actual) == pytest.approx(float(expected), rel=0.0, abs=ABS_ERROR)


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


def _step(graph, operator, targets, **options):
    before = {node: read_grammar_state(graph.nodes[node]) for node in targets}
    protected = {
        n: deepcopy(data) for n, data in graph.nodes(data=True) if n not in targets
    }
    result = execute_pointwise_stage(
        graph,
        operator,
        targets,
        require_exact_operator=True,
        validate_preconditions=True,
        **options,
    )
    assert result.glyph == operator.glyph.value and result.nodes_processed == len(
        targets
    )
    for node in targets:
        after = read_grammar_state(graph.nodes[node])
        assert after.operator_step == before[node].operator_step + 1
        assert after.history[-1] == operator.glyph.value
    assert all(graph.nodes[n] == data for n, data in protected.items())
    assert graph.graph["_t"] == 0.0


def _source():
    graph = nx.Graph()
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        NAV_STRICT=True,
        NAV_RANDOM=False,
        UM_BIDIRECTIONAL=False,
        UM_SYNC_VF=True,
        UM_STABILIZE_DNFR=True,
        UM_FUNCTIONAL_LINKS=False,
        CLIP_MODE="hard",
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
    )
    graph.graph["GLYPH_FACTORS"].update(
        AL_boost=0.125,
        SHA_vf_factor=0.5,
        NAV_jitter=0.0,
        IL_dnfr_factor=0.5,
        UM_vf_sync=0.25,
        UM_theta_push=0.5,
        UM_dnfr_reduction=0.25,
        EN_mix=0.5,
        RA_epi_diff=0.5,
        RA_vf_amplification=0.125,
    )
    for node, value in zip(
        NODES, (Q(3, 4), Q(1, 8), Q(1, 4), Q(1, 4), Q(1, 4)), strict=True
    ):
        graph.add_node(
            node,
            EPI=float(value),
            nu_f=15 / 16 if node == C else 1.0,
            theta=0.0,
            delta_nfr=1 / 8 if node == C else 0.0,
            EPI_kind="form",
            Si=0.0,
        )
        assert read_grammar_state(graph.nodes[node]).history == ()
    for left, right in EDGES:
        graph.add_edge(left, right, weight=1.0)
    _step(graph, Coherence(), NODES, phase_locking_coefficient=0.0)
    _step(graph, Emission(), (C,))
    assert _form(graph, C) == Q(1, 4)
    assert _scalar(graph, C, ALIAS_VF) == Q(15, 16)
    assert _scalar(graph, C, ALIAS_DNFR) == Q(1, 16)
    assert read_grammar_state(graph.nodes[C]).history == ("IL", "AL")
    assert graph.nodes[C]["_structural_lineage"]["activation_count"] == 1
    assert "epi_time_history" not in graph.nodes[C]
    assert tuple(graph.neighbors(C)) == (A, U, R)
    assert tuple(graph.neighbors(W)) == (R,)
    return graph


def _participants(graph):
    assert all(
        get_attr(graph.nodes[node], ALIAS_EPI_KIND, strict=True, conv=str) == "form"
        for node in NODES
    )
    for node in (A, U, R):
        assert _scalar(graph, node, ALIAS_VF) == Q(1)
        assert _scalar(graph, node, ALIAS_THETA) == Q(0)
        assert _scalar(graph, node, ALIAS_DNFR) == Q(0)
    assert _form(graph, A) == Q(3, 4) and _form(graph, U) == Q(1, 4)
    assert tuple(graph.neighbors(C)) == (A, U, R)
    assert {frozenset(edge) for edge in graph.edges} == set(map(frozenset, EDGES))
    assert all(data == {"weight": 1.0} for _, _, data in graph.edges(data=True))


def _domain(graph):
    _participants(graph)
    assert Q(5, 8) <= _scalar(graph, C, ALIAS_VF) <= Q(15, 16)
    assert Q(1, 4) <= _form(graph, C) <= Q(7, 16)
    assert Q(0) <= _scalar(graph, C, ALIAS_DNFR) <= Q(1, 8)
    assert Q(0) <= _scalar(graph, C, ALIAS_THETA) <= Q(1, 16)
    assert Q(1, 4) <= _form(graph, R) <= Q(5, 8)
    assert _form(graph, W) == Q(1, 4)
    assert _scalar(graph, W, ALIAS_VF) == Q(1)
    assert _scalar(graph, W, ALIAS_THETA) == _scalar(graph, W, ALIAS_DNFR) == Q(0)
    assert graph.graph["_t"] == 0.0


def _renewal_prefix(graph, um_count):
    start_c = _scalar(graph, C, ALIAS_VF)
    start_p = _scalar(graph, C, ALIAS_DNFR)
    start_phase = _scalar(graph, C, ALIAS_THETA)
    form, support = _form(graph, C), _support(graph)
    origin = graph.nodes[C]["_emission_origin"]
    lineage = deepcopy(graph.nodes[C]["_structural_lineage"])
    event_count = len(graph.graph.get("_nav_transitions", ()))
    _step(graph, Silence(), (C,))
    assert _scalar(graph, C, ALIAS_VF) == start_c / 2
    assert _scalar(graph, C, ALIAS_DNFR) == start_p
    assert _scalar(graph, C, ALIAS_THETA) == start_phase
    assert graph.nodes[C]["latent"] is True
    assert graph.nodes[C]["preserved_epi"] == float(form)
    assert graph.nodes[C]["silence_duration"] == 0.0
    stamp = datetime.fromisoformat(graph.nodes[C]["latency_start_time"])
    assert stamp.tzinfo is not None and stamp.utcoffset() is not None
    _step(graph, Transition(), (C,), phase_shift=0.125)
    _close(_scalar(graph, C, ALIAS_VF), Q(3, 5) * start_c)
    _close(_scalar(graph, C, ALIAS_DNFR), Q(7, 20) * start_c)
    _close(_scalar(graph, C, ALIAS_THETA), start_phase + Q(1, 8))
    assert len(graph.graph["_nav_transitions"]) == event_count + 1
    event = graph.graph["_nav_transitions"][-1]
    assert event["node"] == C and event["regime_origin"] == "latent"
    assert event["dnfr_handler_after"] == float(start_c / 2) != float(start_p)
    assert all(
        key not in graph.nodes[C]
        for key in ("latent", "latency_start_time", "preserved_epi")
    )
    assert graph.nodes[C]["silence_duration"] >= 0.0
    assert graph.nodes[C]["was_initial_on_silence"] is False
    for _ in range(2):
        old_p = _scalar(graph, C, ALIAS_DNFR)
        old_c = _scalar(graph, C, ALIAS_VF)
        old_phase = _scalar(graph, C, ALIAS_THETA)
        _step(graph, Coherence(), (C,), phase_locking_coefficient=0.0)
        assert _scalar(graph, C, ALIAS_DNFR) == old_p / 2
        assert _scalar(graph, C, ALIAS_VF) == old_c
        assert _scalar(graph, C, ALIAS_THETA) == old_phase
    assert _scalar(graph, C, ALIAS_DNFR) < Q(15, 100)
    ideal_pressure = float(Q(7, 80) * start_c)
    for index in range(um_count):
        old_c = _scalar(graph, C, ALIAS_VF)
        old_p = _scalar(graph, C, ALIAS_DNFR)
        old_phase = _scalar(graph, C, ALIAS_THETA)
        prior = read_grammar_state(graph.nodes[C])
        others = {n: deepcopy(data) for n, data in graph.nodes(data=True) if n != C}
        result = execute_coupling_stage(
            graph, Coupling(), (C,), validate_preconditions=True
        )
        assert result.glyph == "UM" and result.nodes_processed == 1
        assert (
            read_grammar_state(graph.nodes[C]).operator_step == prior.operator_step + 1
        )
        assert read_grammar_state(graph.nodes[C]).history[-1] == "UM"
        assert all(graph.nodes[n] == data for n, data in others.items())
        _close(_scalar(graph, C, ALIAS_VF), Q(3, 4) * old_c + Q(1, 4))
        _close(_scalar(graph, C, ALIAS_THETA), old_phase / 2)
        assert 0 <= _scalar(graph, C, ALIAS_DNFR) <= old_p
        # Independent ideal affine affinity, evaluated after each phase step.
        ideal_phase = (start_phase + Q(1, 8)) / (2 ** (index + 1))
        ideal_pressure *= 0.75 + float(ideal_phase) / (4 * math.pi)
        _close(_scalar(graph, C, ALIAS_DNFR), ideal_pressure)
        _participants(graph)
    assert _form(graph, C) == form and _support(graph) == support
    assert graph.nodes[C]["_emission_origin"] == origin
    assert graph.nodes[C]["_structural_lineage"] == lineage
    assert graph.graph["_t"] == 0.0


def _complete_return(graph):
    before_form = _form(graph, C)
    before_count = graph.nodes[C]["_structural_lineage"]["activation_count"]
    origin = graph.nodes[C]["_emission_origin"]
    result = execute_retained_emission_stage(
        graph, capture_retained_organization(graph, (A, C)), 1, mean_gain=Q(1, 16)
    )
    assert result.member == C and result.stage_result.glyph == "AL"
    assert result.realized_mean_gain == Q(1, 16)
    assert _form(graph, C) == before_form + Q(1, 8)
    assert graph.nodes[C]["_structural_lineage"]["activation_count"] == before_count + 1
    assert graph.nodes[C]["_emission_origin"] == origin
    received = execute_neighbor_stage(
        graph,
        Reception(),
        (C,),
        source_nodes=(U,),
        track_sources=False,
        validate_preconditions=True,
    )
    assert received.glyph == "EN" and received.nodes_processed == 1
    (row,) = received.reception_observations
    assert row.node == C and row.source_nodes == (U,)
    assert _form(graph, C) == before_form / 2 + Q(3, 16)
    record = capture_retained_organization(graph, (A, C))
    (port,) = tuple(p for p in retained_output_ports(record) if p.receiver == R)
    previous_r = _form(graph, R)
    sources = {n: deepcopy(graph.nodes[n]) for n in (A, C)}
    output = execute_retained_output_reception_stage(graph, record, port.ordinal)
    assert output.stage_result.glyph == "EN"
    (row,) = output.stage_result.reception_observations
    assert row.node == R and row.source_nodes == (A, C)
    assert _form(graph, R) == previous_r / 2 + (Q(3, 4) + _form(graph, C)) / 4
    assert all(graph.nodes[n] == data for n, data in sources.items())
    _participants(graph)


def _reject_al(graph):
    record = capture_retained_organization(graph, (A, C))
    before = _checkpoint(graph)
    assert _scalar(graph, C, ALIAS_VF) < Q(1, 2)
    with pytest.raises(TNFRValueError, match="frequency too low"):
        execute_retained_emission_stage(graph, record, 1, mean_gain=Q(1, 16))
    assert _checkpoint(graph) == before


def _own_memory(graph, um_count, completed, failed, expected_step):
    prefix = ("SHA", "NAV", "IL", "IL") + ("UM",) * um_count
    word = ("IL", "AL") + (prefix + ("AL", "EN")) * completed + prefix * failed
    state = read_grammar_state(graph.nodes[C])
    assert state.operator_step == len(word) == expected_step
    assert state.history == word[-7:]
    assert state.prior_coherence and state.u2_debt == 0
    assert graph.nodes[C]["_structural_lineage"]["activation_count"] == 1 + completed
    assert len(graph.graph["_nav_transitions"]) == completed + failed
    assert graph.nodes[C]["silence_duration"] >= 0.0
    assert graph.nodes[C]["was_initial_on_silence"] is False
    assert "latent" not in graph.nodes[C]
    assert "epi_time_history" not in graph.nodes[C]
    receiver = read_grammar_state(graph.nodes[R])
    assert receiver.operator_step == 1 + completed
    assert receiver.history == ("IL",) + ("EN",) * completed
    for node in (A, U, W):
        assert read_grammar_state(graph.nodes[node]).history == ("IL",)
    assert graph.graph["_t"] == 0.0


def test_two_actual_um_actions_return_all_channels_and_preserve_the_output():
    graph = _source()
    _domain(graph)
    origin, support = graph.nodes[C]["_emission_origin"], _support(graph)
    for capacity, phase, form, receiver in (
        (Q(193, 256), Q(1, 32), Q(5, 16), Q(25, 64)),
        (Q(14171, 20480), Q(5, 128), Q(11, 32), Q(15, 32)),
    ):
        _renewal_prefix(graph, 2)
        _close(_scalar(graph, C, ALIAS_VF), capacity)
        if capacity == Q(193, 256):
            assert _scalar(graph, C, ALIAS_VF) == capacity
        _close(_scalar(graph, C, ALIAS_THETA), phase)
        _complete_return(graph)
        assert _form(graph, C) == form and _form(graph, R) == receiver
        _domain(graph)
    assert graph.nodes[C]["_emission_origin"] == origin and _support(graph) == support
    _own_memory(graph, 2, completed=2, failed=0, expected_step=18)
    # RA is one final probe, outside the repeated return map and its W capacity.
    protected = {n: deepcopy(data) for n, data in graph.nodes(data=True) if n != W}
    assert tuple(graph.neighbors(W)) == (R,)
    Resonance()(graph, W, validate_preconditions=True)
    assert _form(graph, W) == Q(23, 64)
    assert _scalar(graph, W, ALIAS_VF) == Q(9, 8)
    assert read_grammar_state(graph.nodes[W]).history == ("IL", "RA")
    assert all(graph.nodes[n] == data for n, data in protected.items())
    assert _support(graph) == support and graph.graph["_t"] == 0.0


def test_no_renewal_rejects_the_second_al_and_preserves_its_finished_prefix():
    graph = _source()
    _renewal_prefix(graph, 0)
    assert _scalar(graph, C, ALIAS_VF) == Q(9, 16)
    _complete_return(graph)
    assert _form(graph, C) == Q(5, 16) and _form(graph, R) == Q(25, 64)
    _renewal_prefix(graph, 0)
    _close(_scalar(graph, C, ALIAS_VF), Q(27, 80))
    _close(_scalar(graph, C, ALIAS_THETA), Q(1, 4))
    _reject_al(graph)
    assert _form(graph, C) == Q(5, 16) and _form(graph, R) == Q(25, 64)
    _own_memory(graph, 0, completed=1, failed=1, expected_step=12)


def test_one_actual_um_delays_but_does_not_remove_the_capacity_obstruction():
    graph = _source()
    for capacity, phase, form, receiver in (
        (Q(43, 64), Q(1, 16), Q(5, 16), Q(25, 64)),
        (Q(707, 1280), Q(3, 32), Q(11, 32), Q(15, 32)),
    ):
        _renewal_prefix(graph, 1)
        _close(_scalar(graph, C, ALIAS_VF), capacity)
        if capacity == Q(43, 64):
            assert _scalar(graph, C, ALIAS_VF) == capacity
        _close(_scalar(graph, C, ALIAS_THETA), phase)
        _complete_return(graph)
        assert _form(graph, C) == form and _form(graph, R) == receiver
    _renewal_prefix(graph, 1)
    _close(_scalar(graph, C, ALIAS_VF), Q(12763, 25600))
    _close(_scalar(graph, C, ALIAS_THETA), Q(7, 64))
    _reject_al(graph)
    assert _form(graph, C) == Q(11, 32) and _form(graph, R) == Q(15, 32)
    _own_memory(graph, 1, completed=2, failed=1, expected_step=21)
