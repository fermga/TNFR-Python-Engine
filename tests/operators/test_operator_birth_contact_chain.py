"""Finite supplied birth/contact/propagation; no autonomous source acquisition."""

from __future__ import annotations

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.dynamics.runtime import _record_mutation_flow_boundary
from tnfr.errors import TNFRValueError
from tnfr.operators.definitions import (
    Coherence,
    Coupling,
    Resonance,
    SelfOrganization,
    Silence,
)
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.network_stage import execute_self_organization_stage
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.types import require_finite_real_scalar_epi

_MIX = 0.25
_PARENT = "source"


def _source() -> nx.Graph:
    graph = nx.Graph(
        THOL_METABOLIC_ENABLED=False,
        THOL_PROPAGATION_ENABLED=False,
        THOL_ALLOW_ISOLATED=True,
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_REJECTION_MODE="raise",
        _t=2.0,
        UM_FUNCTIONAL_LINKS=True,
        UM_BIDIRECTIONAL=True,
        UM_SYNC_VF=True,
        UM_STABILIZE_DNFR=True,
        UM_COMPAT_THRESHOLD=0.5,
        EPI_LATENT_MAX=1.0,
        SHA_MIN_VF=0.1,
        GLYPH_FACTORS={
            "THOL_accel": 0.25,
            "UM_theta_push": 0.5,
            "UM_vf_sync": 0.25,
            "UM_dnfr_reduction": 0.25,
            "RA_epi_diff": _MIX,
            "RA_phase_coupling": 0.5,
            "RA_vf_amplification": 0.125,
            "SHA_vf_factor": 0.5,
        },
    )
    # The supplied adjacent slopes are 1/8 and 1/4: A=1/8>tau=1/10.
    # Samples match the current form; this is not an acquired trajectory.
    graph.add_node(
        _PARENT,
        EPI=0.875,
        nu_f=1.0,
        theta=0.25,
        delta_nfr=0.125,
        epi_time_history=[(0.0, 0.5), (1.0, 0.625), (2.0, 0.875)],
        glyph_history=["IL", "OZ"],
        _grammar_u2_debt=1,
        _grammar_prior_coherence=True,
        _operator_step=2,
        epi_kind="prepared-form",
    )
    return graph


def _apply(graph: nx.Graph, operator, route: str, **kwargs) -> None:
    if route == "public":
        operator(graph, _PARENT, **kwargs)
    else:
        execute = (
            execute_self_organization_stage
            if isinstance(operator, SelfOrganization)
            else execute_network_operator_stage
        )
        stage = execute(graph, operator, (_PARENT,), **kwargs)
        assert stage.nodes_processed == 1


def _birth(graph: nx.Graph, route: str):
    _apply(graph, SelfOrganization(), route, tau=0.1)
    (record,) = graph.nodes[_PARENT]["sub_epis"]
    child = record["node_id"]
    assert graph.nodes[_PARENT]["sub_nodes"] == [child]
    assert graph.nodes[child]["parent_node"] == _PARENT
    assert graph.degree(_PARENT) == graph.degree(child) == 0
    return child


def _unchanged_fresh_child(
    graph: nx.Graph, child, expected, *, observed: bool = False
) -> None:
    # Bidirectional UM synchronizes the alias even for equal phases, preserving
    # the birth channels and the child's own execution context in this case.
    current = dict(graph.nodes[child])
    if observed:
        assert list(current.pop("epi_time_history")) == [(2.0, expected["EPI"])]
    assert current == {**expected, "phase": expected["theta"]}
    state = read_grammar_state(graph.nodes[child])
    assert state.history == ()
    assert state.u2_debt == 0 and state.prior_coherence is False
    acceleration = observe_structural_acceleration(graph, child)
    assert not acceleration.available and acceleration.history_length == 1
    if not observed:
        assert "epi_time_history" not in graph.nodes[child]


@pytest.mark.parametrize("route", ["public", "stage"])
def test_birth_contact_then_resonance_uses_actual_child_without_inherited_history(
    route: str,
) -> None:
    graph = _source()
    parent_history = deepcopy(graph.nodes[_PARENT]["epi_time_history"])
    child = _birth(graph, route)
    child_state = deepcopy(graph.nodes[child])
    parent_form = graph.nodes[_PARENT]["EPI"]
    child_form = graph.nodes[child]["EPI"]
    expected = (1.0 - _MIX) * parent_form + _MIX * child_form

    _apply(graph, Coupling(), route, potential_partner=child)

    assert graph.has_edge(_PARENT, child) and graph.number_of_edges() == 1
    assert graph.nodes[_PARENT]["EPI"] == parent_form
    assert graph.nodes[_PARENT]["epi_time_history"] == parent_history
    _unchanged_fresh_child(graph, child, child_state)
    assert read_grammar_state(graph.nodes[_PARENT]).history[-2:] == ("THOL", "UM")

    _apply(graph, Resonance(), route)

    observed_form = require_finite_real_scalar_epi(graph.nodes[_PARENT]["EPI"])
    assert child_form < observed_form == expected < parent_form
    assert graph.nodes[_PARENT]["epi_kind"] == "prepared-form"
    # Raw operators do not own runtime sampling. Their form jump makes the
    # old endpoint unusable as current acceleration evidence until recording.
    assert graph.nodes[_PARENT]["epi_time_history"] == parent_history
    with pytest.raises(TNFRValueError, match="final EPI must match"):
        observe_structural_acceleration(graph, _PARENT)
    _unchanged_fresh_child(graph, child, child_state)

    # Existing observer at the unchanged clock: no integrator or flow runs.
    # It restarts the parent's history and creates the child's first sample.
    _record_mutation_flow_boundary(graph)
    assert list(graph.nodes[_PARENT]["epi_time_history"]) == [(2.0, expected)]
    assert not observe_structural_acceleration(graph, _PARENT).available
    assert graph.graph["_t"] == 2.0
    _unchanged_fresh_child(graph, child, child_state, observed=True)
    assert read_grammar_state(graph.nodes[_PARENT]).history[-3:] == (
        "THOL",
        "UM",
        "RA",
    )

    _apply(graph, Coherence(), route)
    _apply(graph, Silence(), route)
    assert require_finite_real_scalar_epi(graph.nodes[_PARENT]["EPI"]) == expected
    assert graph.has_edge(_PARENT, child)
    assert graph.nodes[_PARENT]["latent"] is True
    assert read_grammar_state(graph.nodes[_PARENT]).history[-5:] == (
        "THOL",
        "UM",
        "RA",
        "IL",
        "SHA",
    )
    _unchanged_fresh_child(graph, child, child_state, observed=True)


@pytest.mark.parametrize("route", ["public", "stage"])
def test_matched_birth_without_coupling_cannot_propagate_by_resonance(
    route: str,
) -> None:
    graph = _source()
    child = _birth(graph, route)
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_history = read_grammar_state(graph.nodes[_PARENT])

    with pytest.raises(OperatorPreconditionError, match="U3|neighbor|coupling"):
        _apply(graph, Resonance(), route)

    assert dict(graph.nodes(data=True)) == before_nodes
    assert read_grammar_state(graph.nodes[_PARENT]) == before_history
    assert graph.degree(_PARENT) == graph.degree(child) == 0
    assert graph.graph["_t"] == 2.0
