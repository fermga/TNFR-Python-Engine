"""Public, legacy and diagnostic THOL gates share one read-only contract."""

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.mathematics import BEPIElement
from tnfr.operators.definitions import SelfOrganization
from tnfr.operators.preconditions import (
    OperatorPreconditionError,
    validate_self_organization,
)
from tnfr.operators.preconditions.self_organization import (
    validate_self_organization_strict,
)
from tnfr.types import ensure_bepi, serialize_bepi
from tnfr.validation import TNFRValidator


def _graph():
    graph = nx.Graph(THOL_METABOLIC_ENABLED=False)
    graph.add_node(
        0,
        **{
            ALIAS_EPI[0]: 0.6,
            ALIAS_DNFR[0]: 0.2,
            ALIAS_VF[0]: 1.0,
            "theta": 0.0,
            "epi_history": [0.0, 0.1, 0.6],
            "glyph_history": ["OZ"],
        },
    )
    graph.add_node(
        1,
        **{
            ALIAS_EPI[0]: 0.4,
            ALIAS_DNFR[0]: 0.1,
            ALIAS_VF[0]: 1.0,
            "theta": 0.0,
        },
    )
    graph.add_edge(0, 1)
    return graph


def _state(graph):
    return deepcopy(
        (
            graph.graph,
            tuple((node, dict(data)) for node, data in graph.nodes(data=True)),
            tuple(graph.edges(data=True)),
        )
    )


GATES = (
    validate_self_organization,
    SelfOrganization()._validate_preconditions,
    validate_self_organization_strict,
)


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize(
    "history",
    [
        [0.0, 0.1, 0.6],
        [0.0, 0.3, 0.6],
        [0.0, 0.9, 0.6],
    ],
)
def test_gate_accepts_both_acceleration_signs_and_closed_window_without_writes(
    gate, history
):
    graph = _graph()
    graph.nodes[0]["epi_history"] = history
    graph.nodes[0]["_mutation_context"] = {"keep": [1]}
    graph.nodes[0]["_thol_no_bifurcation_expected"] = "existing diagnostic"
    graph.nodes[0]["D2_EPI"] = 12345.0
    before = _state(graph)
    assert gate(graph, 0) is None
    assert _state(graph) == before


@pytest.mark.parametrize(
    "representation",
    [
        0.6,
        ensure_bepi(0.6),
        serialize_bepi(ensure_bepi(0.6)),
    ],
)
def test_all_gates_accept_the_same_signed_scalar_embedding(representation):
    for gate in GATES:
        graph = _graph()
        graph.nodes[0][ALIAS_EPI[0]] = representation
        before = _state(graph)
        gate(graph, 0)
        assert _state(graph) == before


@pytest.mark.parametrize(
    "key,value",
    [
        ("THOL_MIN_EPI", float("nan")),
        ("THOL_MIN_EPI", -0.1),
        ("THOL_MIN_VF", True),
        ("THOL_MIN_VF", float("inf")),
        ("THOL_MIN_DEGREE", 1.5),
        ("THOL_MIN_DEGREE", True),
        ("THOL_ALLOW_ISOLATED", 1),
        ("THOL_METABOLIC_ENABLED", "false"),
        ("THOL_MIN_HISTORY_LENGTH", 2),
        ("THOL_MIN_HISTORY_LENGTH", 3.0),
        ("THOL_MIN_HISTORY_LENGTH", True),
        ("THOL_MIN_HISTORY_LENGTH", 4),
        ("BIFURCATION_THRESHOLD_TAU", float("nan")),
        ("THOL_BIFURCATION_THRESHOLD", -0.1),
    ],
)
def test_invalid_configuration_has_identical_rejection_and_no_execution_writes(
    key, value
):
    reasons = []
    for gate in GATES:
        graph = _graph()
        graph.graph[key] = value
        before = _state(graph)
        with pytest.raises(OperatorPreconditionError) as error:
            gate(graph, 0)
        reasons.append(error.value.reason)
        assert _state(graph) == before
    assert len(set(reasons)) == 1


@pytest.mark.parametrize(
    "key,value",
    [
        (ALIAS_EPI[0], -0.6),
        (ALIAS_EPI[0], True),
        (ALIAS_DNFR[0], 0.0),
        (ALIAS_DNFR[0], -0.1),
        (ALIAS_DNFR[0], float("nan")),
        (ALIAS_VF[0], float("inf")),
        (ALIAS_VF[0], -1.0),
        (ALIAS_VF[0], True),
        (ALIAS_EPI[0], BEPIElement((0.5, 0.6), (0.5, 0.5), (0.0, 1.0))),
    ],
)
def test_invalid_parent_state_is_rejected_identically(key, value):
    reasons = []
    for gate in GATES:
        graph = _graph()
        graph.nodes[0][key] = value
        before = _state(graph)
        with pytest.raises(OperatorPreconditionError) as error:
            gate(graph, 0)
        reasons.append(error.value.reason)
        assert _state(graph) == before
    assert len(set(reasons)) == 1


@pytest.mark.parametrize(
    "history",
    [
        [],
        [(0.0, 0.0), (1.0, 0.6)],
        [(0.0, 0.0), (0.0, 0.1), (1.0, 0.6)],
        [(0.0, 0.0), (1.0, 0.1), (2.0, 0.5)],
        [(0.0, 0.0), (1.0, float("nan")), (2.0, 0.6)],
    ],
)
def test_physical_history_rejection_never_falls_back_to_legacy_or_cached_curvature(
    history,
):
    for gate in GATES:
        graph = _graph()
        graph.nodes[0]["epi_time_history"] = history
        graph.nodes[0]["D2_EPI"] = 100.0
        before = _state(graph)
        with pytest.raises(OperatorPreconditionError):
            gate(graph, 0)
        assert _state(graph) == before


def test_gate_does_not_imply_grammar_admission_or_depth_readiness():
    graph = _graph()
    graph.nodes[0]["glyph_history"] = []
    graph.graph["THOL_MAX_BIFURCATION_DEPTH"] = "unchecked proposal field"
    before = _state(graph)
    validate_self_organization_strict(graph, 0)
    from tnfr.operators.grammar_dynamics import validate_candidate

    assert not validate_candidate(graph, 0, "THOL").allowed
    assert _state(graph) == before


def test_explicit_tau_and_nonblocking_warning_use_the_same_resolver(caplog):
    graph = _graph()
    graph.nodes[0]["epi_history"] = [0.6, 0.6, 0.6]
    graph.graph["BIFURCATION_THRESHOLD_TAU"] = "inactive override"
    before = _state(graph)
    validate_self_organization_strict(graph, 0, tau=0.0, emit_warnings=False)
    assert not caplog.records
    SelfOrganization()._validate_preconditions(graph, 0, tau=0.0)
    assert any("does not exceed tau" in record.message for record in caplog.records)
    assert _state(graph) == before


def test_isolation_permission_does_not_supply_metabolic_neighbors():
    for gate in GATES:
        graph = _graph()
        graph.remove_edge(0, 1)
        graph.graph["THOL_ALLOW_ISOLATED"] = True
        gate(graph, 0)
        graph.graph["THOL_METABOLIC_ENABLED"] = True
        before = _state(graph)
        with pytest.raises(OperatorPreconditionError, match="metabolic"):
            gate(graph, 0)
        assert _state(graph) == before


def test_public_commit_still_records_telemetry_after_success():
    graph = _graph()
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    validate_self_organization(graph, 0)
    assert "_mutation_context" not in graph.nodes[0]
    assert "_thol_no_bifurcation_expected" not in graph.nodes[0]
    SelfOrganization()(graph, 0)
    assert graph.nodes[0]["_mutation_context"]["destabilizer_operator"] == "dissonance"
    assert graph.nodes[0]["_thol_no_bifurcation_expected"] is False
    assert len(graph.nodes[0]["sub_nodes"]) == 1


@pytest.mark.parametrize("options", [{}, {"validate_preconditions": False}])
def test_disabled_optional_gate_is_not_enabled_by_centralization(options):
    graph = _graph()
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = bool(options)
    graph.nodes[0][ALIAS_DNFR[0]] = -0.2
    with pytest.raises(OperatorPreconditionError, match="positive"):
        validate_self_organization(graph, 0)
    SelfOrganization()(graph, 0, **options)
    assert len(graph.nodes[0]["sub_nodes"]) == 1
    assert "_mutation_context" not in graph.nodes[0]


def test_validator_facade_uses_the_same_read_only_gate():
    graph = _graph()
    before = _state(graph)
    assert TNFRValidator().validate_operator_preconditions(
        graph, 0, "self_organization"
    )
    assert _state(graph) == before


def test_warning_toggle_is_strictly_boolean():
    with pytest.raises(OperatorPreconditionError, match="emit_warnings"):
        validate_self_organization_strict(_graph(), 0, emit_warnings=1)
