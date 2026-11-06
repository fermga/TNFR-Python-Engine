"""THOL births retain inherited coordinates and fresh child execution context.

These single-event controls test the existing public/stage consumer. They do
not estimate dimension, prove a rank theorem for binary64, or acquire a child
identity, trajectory, independent input, or physical organization.
"""

from __future__ import annotations

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.operators._thol_constants import THOL_CHILD_VF_DAMPING, THOL_SUB_EPI_SCALING
from tnfr.operators.definitions import SelfOrganization
from tnfr.operators.grammar_debt import (
    OPERATOR_STEP_KEY,
    PRIOR_COHERENCE_KEY,
    U2_DEBT_KEY,
    read_grammar_state,
)
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.network_stage import (
    TWO_PHASE_JACOBI,
    execute_self_organization_stage,
)
from tnfr.operators.nodal_equation import observe_structural_acceleration


def _graph() -> nx.Graph:
    return nx.Graph(
        THOL_METABOLIC_ENABLED=False,
        THOL_PROPAGATION_ENABLED=False,
        THOL_ALLOW_ISOLATED=True,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_REJECTION_MODE="raise",
        GLYPH_FACTORS={"THOL_accel": 0.25},
    )


def _channels(data: dict) -> tuple[float, float, float, float]:
    return tuple(
        get_attr(data, aliases, strict=True)
        for aliases in (ALIAS_EPI, ALIAS_VF, ALIAS_THETA, ALIAS_DNFR)
    )


def _add_parent(
    graph: nx.Graph,
    node: int | str,
    *,
    epi: float = 0.875,
    capacity: float = 1.0,
    phase: float = 0.25,
) -> None:
    # The two slopes are 1/8 and 1/4, so the physical three-point
    # acceleration is 1/8 for every translated form value in this family.
    graph.add_node(
        node,
        EPI=epi,
        nu_f=capacity,
        theta=phase,
        delta_nfr=0.125,
        epi_time_history=[(0.0, epi - 0.375), (1.0, epi - 0.25), (2.0, epi)],
        glyph_history=["IL", "OZ"],
        **{U2_DEBT_KEY: 1, PRIOR_COHERENCE_KEY: True, OPERATOR_STEP_KEY: 2},
    )


def _birth(graph: nx.Graph, node: int | str, route: str) -> str:
    if route == "public":
        SelfOrganization()(graph, node, tau=0.1)
    else:
        result = execute_self_organization_stage(
            graph, SelfOrganization(), (node,), tau=0.1
        )
        assert result.schedule == TWO_PHASE_JACOBI
        assert result.nodes_processed == 1
    (child,) = graph.nodes[node]["sub_nodes"]
    return child


def _assert_fresh_child_context(graph: nx.Graph, child: str) -> None:
    data = graph.nodes[child]
    assert data["glyph_history"] == []
    assert data["epi_history"] == [data["EPI"]]
    assert "epi_time_history" not in data
    assert not any(alias in data for alias in ALIAS_D2EPI)
    state = read_grammar_state(data)
    assert state.history == ()
    assert state.u2_debt == 0
    assert state.prior_coherence is False
    assert state.operator_step == 0
    acceleration = observe_structural_acceleration(graph, child)
    assert acceleration.available is False
    assert acceleration.value is None
    assert acceleration.history_length == 1
    assert acceleration.reason == "insufficient_history"
    # Parent lineage is not a previously executed child Coherence or OZ.
    mutation = validate_candidate(graph, child, "ZHIR")
    assert mutation.allowed is False
    assert any(violation.rule == "U4b" for violation in mutation.violations)


@pytest.mark.parametrize("route", ["public", "stage"])
@pytest.mark.parametrize(
    "epi, capacity, phase",
    [(0.8125, 0.5, 0.125), (0.875, 1.25, 0.25), (0.9375, 2.0, 0.5)],
)
def test_birth_obeys_inherited_state_relations_and_fresh_context(
    route: str, epi: float, capacity: float, phase: float
) -> None:
    graph = _graph()
    _add_parent(graph, "parent", epi=epi, capacity=capacity, phase=phase)
    physical_history = deepcopy(graph.nodes["parent"]["epi_time_history"])

    child = _birth(graph, "parent", route)

    parent, born = graph.nodes["parent"], graph.nodes[child]
    assert (parent["EPI"], parent["nu_f"], parent["theta"]) == (epi, capacity, phase)
    assert parent["delta_nfr"] == 0.125 + 0.25 * 0.125
    assert parent["epi_time_history"] == physical_history
    # Algebraic expectations use the declared represented coefficients, not
    # the production amplitude composer or a numerically estimated Jacobian.
    assert _channels(born) == (
        THOL_SUB_EPI_SCALING * epi,
        THOL_CHILD_VF_DAMPING * capacity,
        phase,
        0.0,
    )
    assert tuple(graph) == ("parent", child)
    assert graph.degree(child) == 0
    assert born["parent_node"] == "parent"
    assert born["_hierarchy_path"] == ["parent"]
    assert graph.graph["hierarchy"]["parent"] == [child]
    (record,) = parent["sub_epis"]
    assert record["node_id"] == child
    assert record["epi"] == born["EPI"]
    assert record["vf"] == get_attr(born, ALIAS_VF, strict=True)
    assert record["timestamp"] == 3
    assert record["d2_epi"] == 0.125
    assert record["metabolized"] is False
    assert record["network_signals"] is None
    parent_state = read_grammar_state(parent)
    assert parent_state.history == ("IL", "OZ", "THOL")
    assert parent_state.prior_coherence is True
    assert parent_state.operator_step == 3
    _assert_fresh_child_context(graph, child)


@pytest.mark.parametrize("route", ["public", "stage"])
@pytest.mark.parametrize("changed, other_value", [("nu_f", 2.0), ("theta", 0.5)])
def test_equal_parent_epi_does_not_determine_complete_child_state(
    route: str, changed: str, other_value: float
) -> None:
    first, second = _graph(), _graph()
    for graph in (first, second):
        _add_parent(graph, "parent")
    second.nodes["parent"][changed] = other_value

    first_child = _birth(first, "parent", route)
    second_child = _birth(second, "parent", route)

    left, right = first.nodes[first_child], second.nodes[second_child]
    assert first.nodes["parent"]["EPI"] == second.nodes["parent"]["EPI"]
    assert left["EPI"] == right["EPI"]
    aliases = ALIAS_VF if changed == "nu_f" else ALIAS_THETA
    assert get_attr(left, aliases, strict=True) != get_attr(right, aliases, strict=True)
    assert left["parent_node"] == right["parent_node"] == "parent"


def test_multi_parent_stage_retains_each_inherited_child_association() -> None:
    graph = _graph()
    _add_parent(graph, "left", capacity=0.5, phase=0.125)
    _add_parent(graph, "right", capacity=2.0, phase=0.5)
    result = execute_self_organization_stage(
        graph, SelfOrganization(), ("right", "left"), tau=0.1
    )

    assert result.nodes_processed == 2
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 0
    for node, capacity, phase in (("left", 0.5, 0.125), ("right", 2.0, 0.5)):
        (child,) = graph.nodes[node]["sub_nodes"]
        data = graph.nodes[child]
        assert data["parent_node"] == node
        assert data["_hierarchy_path"] == [node]
        assert data["EPI"] == THOL_SUB_EPI_SCALING * 0.875
        assert get_attr(data, ALIAS_VF, strict=True) == THOL_CHILD_VF_DAMPING * capacity
        assert data["theta"] == phase
        assert graph.graph["hierarchy"][node] == [child]
        assert graph.nodes[node]["sub_epis"][0]["node_id"] == child
        _assert_fresh_child_context(graph, child)


@pytest.mark.parametrize("route", ["public", "stage"])
def test_metabolic_birth_requires_retaining_the_consumed_neighbor_form(
    route: str,
) -> None:
    child_forms = []
    for neighbor_epi in (0.75, 1.0):
        graph = _graph()
        _add_parent(graph, "parent")
        graph.add_node(
            "boundary", EPI=neighbor_epi, nu_f=1.0, theta=0.25, delta_nfr=0.0
        )
        graph.add_edge("parent", "boundary")
        graph.graph.update(
            THOL_METABOLIC_ENABLED=True,
            THOL_METABOLIC_GRADIENT_WEIGHT=0.25,
            THOL_METABOLIC_COMPLEXITY_WEIGHT=0.0,
        )
        child = _birth(graph, "parent", route)
        born = graph.nodes[child]
        expected = THOL_SUB_EPI_SCALING * 0.875 + 0.25 * (neighbor_epi - 0.875)
        assert 0.0 < expected < 1.0
        assert born["EPI"] == expected
        assert graph.degree(child) == 0
        assert set(graph.edges) == {("parent", "boundary")}
        (record,) = graph.nodes["parent"]["sub_epis"]
        assert record["metabolized"] is True
        assert record["network_signals"]["mean_neighbor_epi"] == neighbor_epi
        child_forms.append(born["EPI"])
    assert child_forms[0] != child_forms[1]
