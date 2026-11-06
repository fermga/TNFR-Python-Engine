"""Detached adapter controls; no selected trajectory or certificate is evaluated.

Synthetic wiring replaces both scientific producers. The field controls use a
different three-node graph and evaluate only its instantaneous complete rows.
"""

from dataclasses import dataclass, replace
from fractions import Fraction as Q
from math import pi, sin
from types import SimpleNamespace

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.operators.grammar_debt import reset_grammar_state_from_history
from tnfr.research import local_capacity_history as study
from tnfr.research.artifact_io import decode_exact_tree, encode_exact_tree
from tnfr.research.retained_graph_state import (
    record_graph_state,
    restore_recorded_graph_state,
)
from tnfr.utils.io import json_dumps, json_loads


def _independent_source():
    graph = nx.Graph(GAMMA={"type": "none"})
    for node, x, theta, capacity in (
        (8, -0.25, -0.5, 2.0),
        (3, 0.5, 0.25, 0.5),
        (11, 0.75, 1.0, 3.0),
    ):
        graph.add_node(node, EPI=x, theta=theta, nu_f=capacity)
    graph.add_edge(8, 3, weight=0.25)
    graph.add_edge(3, 11, weight=0.0)
    return study.capture(graph)


def test_pressure_precision_rebuilds_primitive_map_instead_of_consuming_wide_cache(
    monkeypatch,
):
    graph = nx.path_graph(3)
    for node, phase in enumerate((0.0, 5.0, 4.5)):
        graph.nodes[node].update(EPI=0.25, theta=phase, nu_f=1.0)
    source = study.capture(graph)
    coarse = replace(source, pressure=(I(-1, 1),) * 3)
    monkeypatch.setattr(study, "capture", lambda _: coarse)
    study.refresh_pressure(graph)
    assert get_attr(graph.nodes[0], ALIAS_DNFR) == pytest.approx(sin(5.0) / pi)
    assert all(
        (I(Q(get_attr(graph.nodes[node], ALIAS_DNFR))) - bound).abs_max
        <= study.PRESSURE_ERROR_BUDGET
        for node, bound in zip(graph, study._fresh_pressure_bounds(source), strict=True)
    )


def test_numerical_field_retains_bare_phase_weighted_form_and_nonuniform_capacity():
    source = _independent_source()
    field = study.numerical_field(source)
    state = tuple(float(x) for x in source.epi + source.phase)
    # Zero conductance suppresses the second form-gradient contribution but
    # does not remove that edge from phase pressure or bare degree.
    expected = (
        2 * sin(0.75) / pi,
        0.0,
        -3 * sin(0.75) / pi,
        -0.375 / pi,
        0.046875 / pi,
        0.0,
    )
    assert field(7.0, state) == pytest.approx(expected, abs=2e-16)
    altered = state[:3] + (-0.625, 0.25, 1.125)
    assert field(8.0, altered) != field(7.0, state)
    assert field(8.0, altered)[3:] == field(7.0, state)[3:]
    assert state[3] == -0.5  # No input wrapping or mutation.


def test_numerical_field_rebuilds_cached_support_derived_values():
    source = _independent_source()
    forged = replace(
        source,
        pressure=(I(99),) * 3,
        form_gradient=(Q(99),) * 3,
        degrees=(99,) * 3,
    )
    state = tuple(float(x) for x in source.epi + source.phase)
    assert study.numerical_field(forged)(0.0, state) == study.numerical_field(source)(
        0.0, state
    )
    with pytest.raises(ValueError, match="both nodal rows"):
        study.numerical_field(source)(0.0, state[:-1])


@pytest.mark.parametrize("primitive", ("capacity", "conductances"))
def test_numerical_field_rejects_lost_nonzero_primitive(primitive):
    source = _independent_source()
    values = getattr(source, primitive)
    changed = replace(source, **{primitive: (Q(1, 2**2000),) + values[1:]})
    with pytest.raises(ValueError):
        study.numerical_field(changed)


def test_numerical_field_rejects_another_storage_scale():
    changed = replace(
        _independent_source(),
        reference_model=RelationalExchangeModel(
            2, epi_weight=0, phase_domain="regular"
        ),
    )
    with pytest.raises(ValueError, match="e=0, w=beta=1"):
        study.numerical_field(changed)


@dataclass(frozen=True)
class _NestedEvidence:
    interval: I
    unavailable: object
    record: dict


def test_evidence_projection_roundtrips_intervals_and_tagged_graph_context():
    graph = nx.Graph(_t=Q(7), nested={"negative_zero": -0.0})
    graph.add_node(2, EPI=0.3125, theta=-0.25, nu_f=128.0)
    graph.nodes[2]["epi_time_history"] = ((7.0, 0.3125),)
    record = record_graph_state(graph)
    evidence = _NestedEvidence(I(Q(1, 3)), None, record)
    projected = study.encode_result(evidence)
    decoded = decode_exact_tree(json_loads(json_dumps(encode_exact_tree(projected))))
    assert decoded["interval"] == {
        "lo": evidence.interval.lo,
        "hi": evidence.interval.hi,
    }
    assert decoded["unavailable"] is None
    restored = restore_recorded_graph_state(decoded["record"]).graph
    assert restored.graph["nested"]["negative_zero"].hex() == "-0x0.0p+0"
    assert restored.nodes[2]["epi_time_history"] == ((7.0, 0.3125),)
    assert restored.nodes[2]["theta"] == -0.25


@dataclass(frozen=True)
class _SyntheticCertificate:
    endpoint_budget_certified: bool = True
    trigger_lower_bound: Q = Q(1, 5)
    true_parent_pressure_bounds: tuple = (I(1),) * 3


@dataclass(frozen=True)
class _SyntheticStage:
    glyph: str = "VAL"
    nodes_processed: int = 1


@pytest.fixture
def synthetic_wiring(monkeypatch):
    """Replace both numerical response and validation; do not fit a model."""
    graph = nx.Graph(
        _t=7.0,
        GAMMA={"type": "none"},
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        EDGE_AWARE_ENABLED=False,
        GLYPH_FACTORS={"VAL_scale": 128.0},
    )
    for node in range(20):
        graph.add_node(
            node,
            EPI=0.3125,
            theta=0.25,
            nu_f=0.5 if node < 2 else 1.0,
            delta_nfr=0.25,
            glyph_history=[],
        )
        reset_grammar_state_from_history(graph.nodes[node])
    graph.add_edges_from((node, node + 1) for node in range(19))
    calls = []

    def capture(subject):
        return SimpleNamespace(
            nodes=tuple(subject),
            epi=tuple(Q(get_attr(subject.nodes[n], ALIAS_EPI)) for n in subject),
            phase=tuple(Q(get_attr(subject.nodes[n], ALIAS_THETA)) for n in subject),
            capacity=tuple(Q(get_attr(subject.nodes[n], ALIAS_VF)) for n in subject),
            edges=tuple(subject.edges),
            conductances=(Q(1),) * 19,
            degrees=tuple(subject.degree(node) for node in subject),
            form_rates=(I(0),) * 20,
            phase_rates=(I(0),) * 20,
            storage=I(0),
            pressure=(I(Q(1, 4)),) * 20,
        )

    def stage(subject, operator, targets, **kwargs):
        assert targets == (2,) and operator.glyph.value == "VAL"
        calls.append("synthetic_stage")
        subject.nodes[2]["nu_f"] = 128.0
        subject.nodes[2]["glyph_history"] = ["VAL"]
        reset_grammar_state_from_history(subject.nodes[2])
        return _SyntheticStage()

    def numerical_step(state, field, *, t, dt):
        # These deliberately manufactured values only exercise adapter wiring.
        # They are never claimed to solve the actual sine law.
        assert dt == 2**-20 and callable(field)
        calls.append(("synthetic_step", Q(t), Q(dt)))
        # Exercise recording of the four callback positions without solving
        # any scientific field or retaining a numerical response.
        for time in (t, t + dt / 2, t + dt / 2, t + dt):
            assert field(time, state) == (0.0,) * 40
        result = list(state)
        index = 1 if t == 7.0 else 2
        result[2] = 0.3125 + index * index * 2**-43
        result[30] = -float(index) / 8
        return tuple(result)

    def certify(source, times, states):
        calls.append("synthetic_certificate")
        assert source.capacity[2] == 128
        assert times == study.TIMES and len(states) == 3
        assert states[1][30] == -0.125 and states[2][30] == -0.25
        assert states[2][2] == graph.nodes[2]["EPI"]
        assert graph.graph["_t"] == float(times[-1])
        return _SyntheticCertificate()

    def marker(time, state):
        return (0.0,) * 40

    monkeypatch.setattr(study, "capture", capture)
    monkeypatch.setattr(study, "refresh_pressure", lambda subject: capture(subject))
    monkeypatch.setattr(study, "admit_source", lambda source: {"synthetic": True})
    monkeypatch.setattr(study, "execute_network_operator_stage", stage)
    monkeypatch.setattr(study, "numerical_field", lambda source: marker)
    monkeypatch.setattr(study, "rk4_coupled_step", numerical_step)
    monkeypatch.setattr(study, "certify_history", certify)
    return graph, calls


def test_synthetic_wiring_associates_records_clock_and_read_only_checks(
    synthetic_wiring,
):
    graph, calls = synthetic_wiring
    result = study.evaluate_local_history(graph)
    assert calls == [
        "synthetic_stage",
        ("synthetic_step", Q(7), Q(1, 2**20)),
        ("synthetic_step", Q(7) + Q(1, 2**20), Q(1, 2**20)),
        "synthetic_certificate",
    ]
    assert result["all_checks_passed"]
    assert result["states"][-1][30] == Q(-1, 4)
    assert result["parent_history"] == tuple(
        (time, row[2]) for time, row in zip(study.TIMES, result["states"], strict=True)
    )
    decoded = decode_exact_tree(json_loads(json_dumps(encode_exact_tree(result))))
    restored = restore_recorded_graph_state(decoded["endpoint"]).graph
    assert restored.nodes[2]["epi_time_history"] == graph.nodes[2]["epi_time_history"]
    assert restored.nodes[2]["glyph_history"] == ["VAL"]
    assert restored.nodes[10]["theta"] == -0.25


def test_invalid_initial_parent_context_rejects_before_producers(synthetic_wiring):
    graph, calls = synthetic_wiring
    graph.nodes[2]["glyph_history"] = ["OZ"]
    with pytest.raises(ValueError, match="borrow"):
        study.evaluate_local_history(graph)
    assert calls == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("GRAMMAR_REJECTION_MODE", "fallback"),
        ("VALIDATE_OPERATOR_PRECONDITIONS", False),
        ("VALIDATE_OPERATOR_PRECONDITIONS", 1),
        ("EDGE_AWARE_ENABLED", True),
        ("EDGE_AWARE_ENABLED", 0),
        ("THOL_BIFURCATION_THRESHOLD", 0.0),
        ("VAL_MIN_DNFR", 0.0),
        ("GLYPH_FACTORS", {"VAL_scale": 256.0}),
        ("GLYPH_FACTORS", {"VAL_scale": True}),
        ("GLYPH_FACTORS", {"VAL_scale": 128.0, "THOL_gain": 1.0}),
        ("_t", True),
        ("_t", 7.5),
    ],
)
def test_changed_frozen_settings_reject_before_any_producer(
    synthetic_wiring, key, value
):
    graph, calls = synthetic_wiring
    graph.graph[key] = value
    with pytest.raises((TypeError, ValueError)):
        study.evaluate_local_history(graph)
    assert calls == []


@pytest.mark.parametrize(
    "certificate,failed_check",
    [
        (
            _SyntheticCertificate(endpoint_budget_certified=False),
            "same_law_parent_endpoint_accuracy",
        ),
        (
            _SyntheticCertificate(trigger_lower_bound=Q(1, 20)),
            "certified_represented_trigger",
        ),
        (
            _SyntheticCertificate(true_parent_pressure_bounds=(I(-1),) * 3),
            "fresh_parent_pressure_positive",
        ),
    ],
)
def test_passing_synthetic_reader_cannot_override_failed_certificate(
    synthetic_wiring, monkeypatch, certificate, failed_check
):
    graph, _ = synthetic_wiring
    monkeypatch.setattr(study, "certify_history", lambda *args: certificate)
    result = study.evaluate_local_history(graph)
    assert result["checks"]["recorded_trigger"]
    assert not result["checks"][failed_check]
    assert not result["all_checks_passed"]
