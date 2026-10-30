"""Static VAL access controls, separate from maintained-source evidence.

The small graph has explicitly supplied positive stored pressures in the
successful controls. It is not a prepared organization or a sine trajectory.
Fresh constitutive pressure is admitted separately in the rejection controls.
Only one canonical VAL stage is executed; no THOL event or closed word follows.
"""

from math import pi, sin

import networkx as nx
import pytest

from tests.operators.test_collective_reception import MODEL, _SuppliedClock
from tnfr.alias import get_attr, set_dnfr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.operators import network_stage
from tnfr.operators.definitions import Expansion
from tnfr.operators.grammar_debt import node_debt, reset_grammar_state_from_history
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.sdk._state import copy_graph_state


def _source(factor, *, capacity_only=True):
    graph = nx.Graph(
        RANDOM_SEED=17,
        _t=7.0,
        GAMMA={"type": "none"},
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        GLYPH_FACTORS={"VAL_scale": factor},
    )
    if capacity_only:
        graph.graph["EDGE_AWARE_ENABLED"] = False
    for node, form, capacity, phase in (
        (0, 5 / 16, 1 / 2, 0.0),
        (1, 3 / 8, 1.0, 1 / 8),
        (2, 7 / 16, 1.0, 1 / 4),
    ):
        graph.add_node(
            node,
            EPI=form,
            nu_f=capacity,
            theta=phase,
            delta_nfr=1 / 8,
            EPI_kind="supplied_static_control",
            glyph_history=["AL", "UM", "SHA"] if node == 0 else ["AL"],
        )
        reset_grammar_state_from_history(graph.nodes[node])
    graph.add_edge(0, 1, weight=1.0, length=2.0)
    graph.add_edge(1, 2, weight=3 / 4, length=3.0)
    return graph


def _stage(graph, targets=(0, 1, 2)):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(network_stage, "datetime", _SuppliedClock)
        return execute_network_operator_stage(graph, Expansion(), targets)


def _values(graph, aliases):
    return tuple(get_attr(graph.nodes[node], aliases) for node in graph)


@pytest.mark.parametrize("factor", (128.0, 256.0))
def test_supplied_positive_pressure_admits_only_the_declared_capacity_map(factor):
    graph = _source(factor)
    before = copy_graph_state(graph)
    for node in graph:
        assert node_debt(graph.nodes[node]) == 0
        candidate = validate_candidate(graph, node, "THOL")
        assert not candidate.allowed
        assert {violation.rule for violation in candidate.violations} == {"U4b"}

    stage = _stage(graph)

    assert stage.glyph == "VAL"
    assert stage.schedule == "two_phase_jacobi"
    assert stage.nodes_processed == 3
    assert _values(graph, ALIAS_VF) == (factor / 2, factor, factor)
    assert (
        tuple(
            after / initial
            for after, initial in zip(
                _values(graph, ALIAS_VF), _values(before, ALIAS_VF)
            )
        )
        == (factor,) * 3
    )
    for aliases in (ALIAS_EPI, ALIAS_THETA, ALIAS_DNFR):
        assert _values(graph, aliases) == _values(before, aliases)
    assert tuple(graph.edges(data=True)) == tuple(before.edges(data=True))
    assert graph.graph["_t"] == before.graph["_t"]
    assert "edge_aware_interventions" not in graph.graph
    for node in graph:
        assert graph.nodes[node]["EPI_kind"] == before.nodes[node]["EPI_kind"]
        assert tuple(graph.nodes[node]["glyph_history"]) == (
            *before.nodes[node]["glyph_history"],
            "VAL",
        )
        assert node_debt(graph.nodes[node]) == 1
        assert validate_candidate(graph, node, "THOL").allowed
        # Grammar context is now available, but the stage supplied no measured
        # temporal evidence and did not perform child creation.
        assert not observe_structural_acceleration(graph, node).available
        assert "sub_epis" not in graph.nodes[node]
    assert "hierarchy" not in graph.graph
    # The default strict maximum is a before-state admission check. Its
    # unchanged value does not silently clip the finite capacity proposal.
    assert "VAL_MAX_VF" not in graph.graph
    assert min(_values(graph, ALIAS_VF)) > 10.0


@pytest.mark.parametrize("factor", (128.0, 256.0))
def test_fresh_sine_pressure_blocks_all_node_val_without_partial_commit(factor):
    graph = _source(factor)
    report = bound_relational_sine_support_exchange(graph, reference_model=MODEL)
    for node, pressure in zip(report.nodes, report.pressure):
        set_dnfr(graph, node, float(pressure.midpoint))
    expected = (sin(1 / 8) / pi, 0.0, -sin(1 / 8) / pi)
    assert _values(graph, ALIAS_DNFR) == pytest.approx(expected, abs=1e-17)
    assert sum(graph.degree(node) * expected[node] for node in graph) == 0.0
    assert get_attr(graph.nodes[0], ALIAS_DNFR) > 1e-6
    assert get_attr(graph.nodes[2], ALIAS_DNFR) < 0.0
    assert "VAL_MIN_DNFR" not in graph.graph
    before = copy_graph_state(graph)

    # The first target passes the unchanged scalar policy; a later target
    # fails. Atomic preflight must preserve even the first target's history.
    with pytest.raises(OperatorPreconditionError, match="VAL minimum"):
        _stage(graph, targets=(0, 2, 1))

    assert graph.graph == before.graph
    assert dict(graph.nodes) == dict(before.nodes)
    assert dict(graph.edges) == dict(before.edges)
    assert not hasattr(graph, "_last_operator_applied")
    assert all(node_debt(graph.nodes[node]) == 0 for node in graph)


@pytest.mark.parametrize("factor", (2.0, 128.0))
def test_default_edge_aware_val_is_a_different_complete_reset(factor):
    graph = _source(factor, capacity_only=False)
    before = copy_graph_state(graph)
    assert "EDGE_AWARE_ENABLED" not in graph.graph

    _stage(graph)

    assert _values(graph, ALIAS_VF) == (factor / 2, factor, factor)
    assert _values(graph, ALIAS_EPI) == tuple(
        min(factor * form, 1.0) for form in _values(before, ALIAS_EPI)
    )
    assert _values(graph, ALIAS_EPI) != _values(before, ALIAS_EPI)
    assert _values(graph, ALIAS_THETA) == _values(before, ALIAS_THETA)
    assert _values(graph, ALIAS_DNFR) == _values(before, ALIAS_DNFR)
    assert tuple(graph.edges(data=True)) == tuple(before.edges(data=True))
