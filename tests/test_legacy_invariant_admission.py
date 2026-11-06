"""Legacy check identifiers retain canonical primitive admission and narrow scope."""

import math
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.mathematics import BEPIElement
from tnfr.validation.invariants import (
    Invariant1_EPIOnlyThroughOperators,
    Invariant2_VfInHzStr,
    Invariant3_DNFRSemantics,
    Invariant4_OperatorClosure,
    Invariant5_ExplicitPhaseChecks,
    Invariant6_NodeBirthCollapse,
    Invariant9_StructuralMetrics,
    InvariantSeverity,
)
from tnfr.validation.validator import TNFRValidator


def _graph():
    graph = nx.Graph()
    graph.add_node(0, EPI=-0.5, nu_f=0, theta=0, delta_nfr=3)
    return graph


def test_uniform_negative_form_zero_capacity_and_pressure_are_jointly_admitted():
    graph = _graph()
    graph.nodes[0]["EPI"] = BEPIElement((-0.5, -0.5), (-0.5,), (0, 1))
    before = deepcopy(graph.nodes[0])

    for check in (
        Invariant1_EPIOnlyThroughOperators(),
        Invariant2_VfInHzStr(),
        Invariant3_DNFRSemantics(),
        Invariant5_ExplicitPhaseChecks(),
    ):
        assert check.validate(graph) == []
    assert graph.nodes[0] == before
    # Check 4 recognizes supported alias spellings; only the separate hook
    # metadata advisory remains on this unconfigured graph.
    assert all(
        "missing required" not in violation.description
        for violation in Invariant4_OperatorClosure().validate(graph)
    )
    result = TNFRValidator(enable_graph_validation=False).validate(
        graph, include_graph_structure=False
    )
    assert result["passed"]


@pytest.mark.parametrize(
    "value",
    (
        1j,
        BEPIElement((0.2, 0.9), (0.2,), (0, 1)),
        {"continuous": (0.2, 0.9), "discrete": (0.2,), "grid": (0, 1)},
    ),
)
def test_form_validation_rejects_projection_of_complex_or_nonuniform_state(value):
    graph = _graph()
    graph.nodes[0]["EPI"] = value
    graph.nodes[0]["epi"] = 0.5

    violations = Invariant1_EPIOnlyThroughOperators().validate(graph)

    assert len(violations) == 1
    assert violations[0].invariant_id == 1
    assert violations[0].severity is InvariantSeverity.CRITICAL
    assert "invalid scalar domain" in violations[0].description


@pytest.mark.parametrize(
    "factory,aliases",
    (
        (Invariant1_EPIOnlyThroughOperators, ALIAS_EPI),
        (Invariant2_VfInHzStr, ALIAS_VF),
        (Invariant3_DNFRSemantics, ALIAS_DNFR),
        (Invariant5_ExplicitPhaseChecks, ALIAS_THETA),
    ),
)
@pytest.mark.parametrize(
    "invalid", (True, np.bool_(False), math.nan, math.inf, "0", Fraction(1, 10**400))
)
def test_invalid_authoritative_scalar_cannot_fall_through_to_valid_alias(
    factory, aliases, invalid
):
    graph = _graph()
    graph.nodes[0][aliases[0]] = invalid
    graph.nodes[0][aliases[1]] = 0.5

    violations = factory().validate(graph)

    assert any(v.severity is InvariantSeverity.CRITICAL for v in violations)
    assert graph.nodes[0][aliases[0]] is invalid


def test_zero_capacity_remains_subject_to_explicit_configured_policy():
    graph = _graph()
    graph.graph["VF_MIN"] = 0.25

    violations = Invariant2_VfInHzStr().validate(graph)

    assert len(violations) == 1
    assert violations[0].severity is InvariantSeverity.ERROR
    assert "configured bounds" in violations[0].description


@pytest.mark.parametrize("key", ("EPI_MIN", "VF_MAX"))
@pytest.mark.parametrize("invalid", (math.nan, True, Fraction(1, 10**400)))
def test_invalid_configured_bounds_reject_even_empty_graphs(key, invalid):
    graph = nx.Graph()
    graph.graph[key] = invalid
    check = (
        Invariant1_EPIOnlyThroughOperators()
        if key.startswith("EPI")
        else Invariant2_VfInHzStr()
    )
    with pytest.raises((TypeError, ValueError)):
        check.validate(graph)


@pytest.mark.parametrize(
    "invalid", (math.nan, math.inf, True, -1, "0", Fraction(1, 10**400))
)
def test_phase_threshold_cannot_silently_certify_an_empty_domain(invalid):
    with pytest.raises((TypeError, ValueError)):
        Invariant5_ExplicitPhaseChecks(invalid)
    check = Invariant5_ExplicitPhaseChecks()
    check.phase_coupling_threshold = invalid
    with pytest.raises((TypeError, ValueError)):
        check.validate(nx.Graph())


def test_phase_aliases_use_wrapped_difference_and_allow_zero_advisory_threshold():
    graph = nx.path_graph(2)
    graph.nodes[0]["phase"] = 0
    graph.nodes[1]["phase"] = math.pi
    violations = Invariant5_ExplicitPhaseChecks(0).validate(graph)
    edge = [
        item
        for item in violations
        if item.description == "Large phase difference between coupled nodes"
    ]
    assert len(edge) == 1
    assert edge[0].actual_value == pytest.approx(math.pi)


def test_activity_advisory_does_not_claim_zero_capacity_destroys_a_node():
    violations = Invariant6_NodeBirthCollapse().validate(_graph())
    assert len(violations) == 1
    assert violations[0].severity is InvariantSeverity.WARNING
    assert violations[0].description == "Capacity below the selected activity threshold"
    assert "does not decide NFR existence" in violations[0].suggestion


@pytest.mark.parametrize("invalid", (np.bool_(False), Fraction(-1, 10**400)))
def test_exposed_sense_index_does_not_erase_invalid_source_as_zero(invalid):
    graph = _graph()
    graph.nodes[0]["Si"] = invalid
    graph.nodes[0]["si"] = 0
    violations = Invariant9_StructuralMetrics().validate(graph)
    assert any("invalid scalar domain" in item.description for item in violations)


def test_operator_marker_is_a_monitor_warning_not_a_continuous_flow_prohibition():
    graph = _graph()
    check = Invariant1_EPIOnlyThroughOperators()
    assert check.validate(graph) == []
    graph._last_operator_applied = False
    graph.nodes[0]["EPI"] = -0.25

    violations = check.validate(graph)

    assert len(violations) == 1
    assert violations[0].severity is InvariantSeverity.WARNING
    assert "does not identify the cause" in violations[0].suggestion
