"""Two finite reciprocal-return controls from the reviewed prospective protocol.

Support, timing and selection are supplied. Actual neighbors move; W capacity
is retained rather than hidden inside a bounded maintenance claim.
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
from tnfr.utils.numeric import angle_diff

A, C, U, R, W = "member-A", "member-C", "input-U", "receiver-R", "observer-W"
NODES = (A, C, U, R, W)
EDGES = ((A, C), (U, C), (A, R), (C, R), (R, W))
ABS_ERROR = 2.0**-40
BETA = 1.0 / (4.0 * math.pi)


def _form(graph, node):
    return Q.from_float(
        require_finite_real_scalar_epi(
            get_attr(graph.nodes[node], ALIAS_EPI, strict=True, conv=lambda x: x)
        )
    )


def _scalar(graph, node, alias):
    return Q.from_float(get_attr(graph.nodes[node], alias, strict=True))


def _phase(graph, node):
    return float(_scalar(graph, node, ALIAS_THETA))


def _gap(graph, left, right):
    return angle_diff(_phase(graph, left), _phase(graph, right))


def _close(actual, expected):
    assert float(actual) == pytest.approx(float(expected), abs=ABS_ERROR, rel=0.0)


def _phase_close(actual, expected):
    _close(angle_diff(float(actual), float(expected)), 0.0)


def _g(value):
    return math.atan2(math.sin(value), 3.0 + math.cos(value))


def _advance(value):
    return (_g(value) + _g(value / 2.0)) / 2.0


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


def _pointwise(graph, operator, targets, **options):
    before = {n: read_grammar_state(graph.nodes[n]) for n in targets}
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


def _source(*, boundary=False):
    graph = nx.Graph()
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        NAV_STRICT=True,
        NAV_RANDOM=False,
        UM_BIDIRECTIONAL=True,
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
    # Retain the resolved default, rather than tune a tracking gain.
    assert graph.graph["GLYPH_FACTORS"]["RA_phase_coupling"] == BETA
    for node, value in zip(
        NODES, (Q(3, 4), Q(1, 8), Q(1, 4), Q(1, 4), Q(1, 4)), strict=True
    ):
        phase = -math.pi / 2.0 + 1 / 128 if boundary and node == W else 0.0
        graph.add_node(
            node,
            EPI=float(value),
            nu_f=15 / 16 if node == C else 1.0,
            theta=phase,
            delta_nfr=1 / 8 if node == C else 0.0,
            EPI_kind="form",
            Si=0.0,
        )
        assert read_grammar_state(graph.nodes[node]).history == ()
    for left, right in EDGES:
        graph.add_edge(left, right, weight=1.0)
    _pointwise(graph, Coherence(), NODES, phase_locking_coefficient=0.0)
    _pointwise(graph, Emission(), (C,))
    assert _form(graph, C) == Q(1, 4)
    assert _scalar(graph, C, ALIAS_DNFR) == Q(1, 16)
    assert _scalar(graph, C, ALIAS_VF) == Q(15, 16)
    assert read_grammar_state(graph.nodes[C]).history == ("IL", "AL")
    assert graph.nodes[C]["_structural_lineage"]["activation_count"] == 1
    assert tuple(graph.neighbors(C)) == (A, U, R)
    assert tuple(graph.neighbors(W)) == (R,)
    if boundary:
        _close(_gap(graph, R, W), math.pi / 2 - 1 / 128)
    return graph


def _lifecycle_prefix(graph):
    capacity, pressure = _scalar(graph, C, ALIAS_VF), _scalar(graph, C, ALIAS_DNFR)
    phase, form, support = _phase(graph, C), _form(graph, C), _support(graph)
    lineage = deepcopy(graph.nodes[C]["_structural_lineage"])
    _pointwise(graph, Silence(), (C,))
    assert _scalar(graph, C, ALIAS_VF) == capacity / 2
    assert _scalar(graph, C, ALIAS_DNFR) == pressure
    assert graph.nodes[C]["latent"] is True
    assert graph.nodes[C]["preserved_epi"] == float(form)
    assert graph.nodes[C]["silence_duration"] == 0.0
    timestamp = datetime.fromisoformat(graph.nodes[C]["latency_start_time"])
    assert timestamp.tzinfo is not None and timestamp.utcoffset() is not None
    _pointwise(graph, Transition(), (C,), phase_shift=0.125)
    _close(_scalar(graph, C, ALIAS_VF), Q(3, 5) * capacity)
    _close(_scalar(graph, C, ALIAS_DNFR), Q(7, 20) * capacity)
    _phase_close(_phase(graph, C), phase + 1 / 8)
    assert all(
        key not in graph.nodes[C]
        for key in ("latent", "latency_start_time", "preserved_epi")
    )
    assert graph.nodes[C]["silence_duration"] >= 0.0
    assert graph.nodes[C]["was_initial_on_silence"] is False
    for _ in range(2):
        before = _scalar(graph, C, ALIAS_DNFR)
        phase = _phase(graph, C)
        _pointwise(graph, Coherence(), (C,), phase_locking_coefficient=0.0)
        assert _scalar(graph, C, ALIAS_DNFR) == before / 2
        _phase_close(_phase(graph, C), phase)
    assert _scalar(graph, C, ALIAS_DNFR) < Q(15, 100)
    assert _form(graph, C) == form and _support(graph) == support
    assert graph.nodes[C]["_structural_lineage"] == lineage


def _um(graph):
    common, gap = _phase(graph, R), _gap(graph, C, R)
    assert 0 < gap < math.pi / 2
    for node in (A, U):
        _phase_close(_phase(graph, node), common)
    neighbors = {
        n: deepcopy(
            {
                key: value
                for key, value in graph.nodes[n].items()
                if key not in ALIAS_THETA
            }
        )
        for n in (A, U, R)
    }
    untouched_w, support = deepcopy(graph.nodes[W]), _support(graph)
    before = read_grammar_state(graph.nodes[C])
    capacity, pressure, form = (
        _scalar(graph, C, ALIAS_VF),
        _scalar(graph, C, ALIAS_DNFR),
        _form(graph, C),
    )
    result = execute_coupling_stage(
        graph, Coupling(), (C,), validate_preconditions=True
    )
    assert result.glyph == "UM" and result.nodes_processed == 1
    for node in (A, U, R):
        _phase_close(_phase(graph, node), common + _g(gap) / 2)
        assert {
            key: value
            for key, value in graph.nodes[node].items()
            if key not in ALIAS_THETA
        } == neighbors[node]
    _close(_gap(graph, C, R), gap / 2)
    _close(_scalar(graph, C, ALIAS_VF), Q(3, 4) * capacity + Q(1, 4))
    _close(
        _scalar(graph, C, ALIAS_DNFR), float(pressure) * (0.75 + gap / (8 * math.pi))
    )
    assert _form(graph, C) == form and _support(graph) == support
    assert graph.nodes[W] == untouched_w
    after = read_grammar_state(graph.nodes[C])
    assert after.operator_step == before.operator_step + 1 and after.history[-1] == "UM"
    # This is the local UM contract. R--W is checked separately by each case.
    assert all(abs(_gap(graph, C, n)) < math.pi / 2 for n in (A, U, R))


def _member_and_output(graph):
    before = _form(graph, C)
    count = graph.nodes[C]["_structural_lineage"]["activation_count"]
    phases = tuple(_phase(graph, n) for n in NODES)
    result = execute_retained_emission_stage(
        graph, capture_retained_organization(graph, (A, C)), 1, mean_gain=Q(1, 16)
    )
    assert result.member == C and result.stage_result.glyph == "AL"
    assert result.realized_mean_gain == Q(1, 16)
    assert _form(graph, C) == before + Q(1, 8)
    assert graph.nodes[C]["_structural_lineage"]["activation_count"] == count + 1
    received = execute_neighbor_stage(
        graph,
        Reception(),
        (C,),
        source_nodes=(U,),
        track_sources=False,
        validate_preconditions=True,
    )
    assert received.glyph == "EN" and received.nodes_processed == 1
    assert received.reception_observations[0].source_nodes == (U,)
    assert _form(graph, C) == before / 2 + Q(3, 16)
    record = capture_retained_organization(graph, (A, C))
    (port,) = tuple(p for p in retained_output_ports(record) if p.receiver == R)
    old_r = _form(graph, R)
    output = execute_retained_output_reception_stage(graph, record, port.ordinal)
    assert output.stage_result.glyph == "EN"
    assert output.stage_result.reception_observations[0].source_nodes == (A, C)
    assert _form(graph, R) == old_r / 2 + (Q(3, 4) + _form(graph, C)) / 4
    assert tuple(_phase(graph, n) for n in NODES) == phases


def _ra(graph):
    before_form, before_capacity = _form(graph, W), _scalar(graph, W, ALIAS_VF)
    phase, lag = _phase(graph, W), _gap(graph, R, W)
    assert 0 <= lag < math.pi / 2
    protected = {n: deepcopy(data) for n, data in graph.nodes(data=True) if n != W}
    prior = read_grammar_state(graph.nodes[W])
    result = execute_neighbor_stage(
        graph, Resonance(), (W,), validate_preconditions=True
    )
    assert result.glyph == "RA" and result.nodes_processed == 1
    assert _form(graph, W) == (before_form + _form(graph, R)) / 2
    assert _scalar(graph, W, ALIAS_VF) == Q(9, 8) * before_capacity
    _phase_close(_phase(graph, W), phase + BETA * lag)
    assert _scalar(graph, W, ALIAS_DNFR) == Q(0)
    state = read_grammar_state(graph.nodes[W])
    assert state.operator_step == prior.operator_step + 1 and state.history[-1] == "RA"
    assert all(graph.nodes[n] == data for n, data in protected.items())


def _domain(graph):
    assert Q(5, 8) <= _scalar(graph, C, ALIAS_VF) <= Q(15, 16)
    assert Q(1, 4) <= _form(graph, C) <= Q(7, 16)
    assert Q(0) <= _scalar(graph, C, ALIAS_DNFR) <= Q(1, 8)
    assert 0 <= _gap(graph, C, R) <= 1 / 16
    assert 0 <= _gap(graph, R, W) <= 1 / 2
    for node in (A, U, R):
        _phase_close(_phase(graph, node), _phase(graph, R))
        assert _scalar(graph, node, ALIAS_VF) == Q(1)
        assert _scalar(graph, node, ALIAS_DNFR) == Q(0)
    assert _form(graph, A) == Q(3, 4) and _form(graph, U) == Q(1, 4)
    for node in (R, W):
        assert Q(1, 4) <= _form(graph, node) <= Q(5, 8)
    assert _scalar(graph, W, ALIAS_VF) >= Q(1)  # No upper bound is hidden.
    assert _scalar(graph, W, ALIAS_DNFR) == Q(0)
    assert all(abs(_gap(graph, left, right)) < math.pi / 2 for left, right in EDGES)
    assert all(
        get_attr(graph.nodes[n], ALIAS_EPI_KIND, strict=True, conv=str) == "form"
        for n in NODES
    )
    assert graph.graph["_t"] == 0.0


def test_reciprocal_phase_return_tracks_the_port_and_records_growing_capacity():
    graph = _source()
    _domain(graph)
    support, origin = _support(graph), graph.nodes[C]["_emission_origin"]
    expected_common, expected_w_phase = 0.0, 0.0
    for turn, (
        incoming_gap,
        gap,
        capacity,
        form,
        receiver,
        observer,
        observer_capacity,
    ) in enumerate(
        (
            (Q(1, 8), Q(1, 32), Q(193, 256), Q(5, 16), Q(25, 64), Q(41, 128), Q(9, 8)),
            (
                Q(5, 32),
                Q(5, 128),
                Q(14171, 20480),
                Q(11, 32),
                Q(15, 32),
                Q(101, 256),
                Q(81, 64),
            ),
        ),
        start=1,
    ):
        _lifecycle_prefix(graph)
        _close(_gap(graph, C, R), incoming_gap)
        _um(graph)
        assert all(abs(_gap(graph, left, right)) < math.pi / 2 for left, right in EDGES)
        _um(graph)
        assert all(abs(_gap(graph, left, right)) < math.pi / 2 for left, right in EDGES)
        expected_common += _advance(float(incoming_gap))
        _phase_close(_phase(graph, R), expected_common)
        _close(_gap(graph, C, R), gap)
        _close(_scalar(graph, C, ALIAS_VF), capacity)
        if turn == 1:
            assert _scalar(graph, C, ALIAS_VF) == Q(193, 256)
        _member_and_output(graph)
        assert _form(graph, C) == form and _form(graph, R) == receiver
        expected_w_phase = (1 - BETA) * expected_w_phase + BETA * expected_common
        _ra(graph)
        _phase_close(_phase(graph, W), expected_w_phase)
        assert _form(graph, W) == observer
        assert _scalar(graph, W, ALIAS_VF) == observer_capacity
        _domain(graph)
        assert _support(graph) == support
        assert graph.nodes[C]["_emission_origin"] == origin
        assert graph.nodes[C]["_structural_lineage"]["activation_count"] == 1 + turn
        assert read_grammar_state(graph.nodes[C]).operator_step == 2 + 8 * turn
        assert read_grammar_state(graph.nodes[R]).history == ("IL",) + ("EN",) * turn
        assert read_grammar_state(graph.nodes[W]).history == ("IL",) + ("RA",) * turn
        assert read_grammar_state(graph.nodes[R]).operator_step == 1 + turn
        assert read_grammar_state(graph.nodes[W]).operator_step == 1 + turn
        for node in (A, U):
            assert read_grammar_state(graph.nodes[node]).history == ("IL",)
    state = read_grammar_state(graph.nodes[C])
    assert state.history == ("NAV", "IL", "IL", "UM", "UM", "AL", "EN")
    assert state.operator_step == 18 and state.prior_coherence and state.u2_debt == 0
    assert len(graph.graph["_nav_transitions"]) == 2
    assert "epi_time_history" not in graph.nodes[C]


def test_local_um_contraction_can_leave_the_external_ra_port_inadmissible():
    graph = _source(boundary=True)
    support, origin = _support(graph), graph.nodes[C]["_emission_origin"]
    initial_w = _phase(graph, W)
    assert all(abs(_gap(graph, left, right)) < math.pi / 2 for left, right in EDGES)
    _lifecycle_prefix(graph)
    _um(graph)
    _close(_gap(graph, C, R), Q(1, 16))
    _phase_close(_phase(graph, R), _g(1 / 8) / 2)
    assert _scalar(graph, C, ALIAS_VF) == Q(43, 64)
    assert _phase(graph, W) == initial_w
    _close(_gap(graph, R, W), math.pi / 2 - 1 / 128 + _g(1 / 8) / 2)
    assert math.pi / 2 < _gap(graph, R, W) < math.pi
    assert all(abs(_gap(graph, C, n)) < math.pi / 2 for n in (A, U, R))
    before = _checkpoint(graph)
    with pytest.raises(
        (TNFRValueError, OperatorPreconditionError), match=r"(?i)(u3|phase|compatible)"
    ):
        execute_neighbor_stage(graph, Resonance(), (W,), validate_preconditions=True)
    assert _checkpoint(graph) == before
    assert _support(graph) == support and graph.graph["_t"] == 0.0
    assert graph.nodes[C]["_emission_origin"] == origin
    assert graph.nodes[C]["_structural_lineage"]["activation_count"] == 1
    state = read_grammar_state(graph.nodes[C])
    assert state.history == ("IL", "AL", "SHA", "NAV", "IL", "IL", "UM")
    assert state.operator_step == 7 and state.prior_coherence and state.u2_debt == 0
    for node in (A, U, R, W):
        assert read_grammar_state(graph.nodes[node]).history == ("IL",)
        assert read_grammar_state(graph.nodes[node]).operator_step == 1
    assert _form(graph, R) == _form(graph, W) == Q(1, 4)
    assert _scalar(graph, W, ALIAS_VF) == Q(1)
