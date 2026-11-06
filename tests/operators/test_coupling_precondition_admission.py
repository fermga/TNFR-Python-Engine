"""Selected UM activity admission rejects malformed inputs before execution.

These are runtime boundary controls on explicitly constructed graphs, not
acquired endpoints. Successful calls stop at policy/state readers; public and
atomic calls are rejection controls with proposal and execution guards.
"""

from __future__ import annotations

import math
import pickle
from decimal import Decimal
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF
from tnfr.mathematics import BEPIElement
from tnfr.operators import _coupling_stage_kernel, network_stage
from tnfr.operators.definitions import Coupling
from tnfr.operators.preconditions import (
    OperatorPreconditionError,
    _coupling_activity_policy,
    validate_coupling,
)
from tnfr.types import serialize_bepi


def _graph():
    graph = nx.path_graph(2)
    for node in graph:
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: 0.25,
                ALIAS_VF[0]: 1.0,
                "theta": 0.0,
                "delta_nfr": 0.0,
            }
        )
    graph.graph.update(
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_REJECTION_MODE="raise",
        RANDOM_SEED=17,
    )
    # Nonzero form needs no invented AL history to pass the U1a source check.
    return graph


def _state(graph):
    # Read-only NetworkX views may cache themselves on the graph object. Retain
    # graph-owned metadata, node state and support without treating such view
    # materialization as an event write.
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


@pytest.fixture(autouse=True)
def _no_event(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("UM admission control reached a proposal or execution")

    monkeypatch.setattr(Coupling, "_execute", forbidden)
    monkeypatch.setattr(_coupling_stage_kernel, "propose_coupling_stage", forbidden)
    monkeypatch.setattr(network_stage, "_commit_coupling_structure", forbidden)


def _invoke(graph, path):
    if path == "direct":
        validate_coupling(graph, 0)
    elif path == "public":
        Coupling()(graph, 0)
    else:
        network_stage.execute_coupling_stage(graph, Coupling(), (0, 1))


def test_policy_reader_preserves_defaults_and_inclusive_zero():
    assert _coupling_activity_policy({}) == (0.05, 0.01)
    assert _coupling_activity_policy({"UM_MIN_EPI": 0, "UM_MIN_VF": 0}) == (
        0.0,
        0.0,
    )
    assert _coupling_activity_policy({"UM_MIN_EPI": Q(1, 8), "UM_MIN_VF": Q(1, 4)}) == (
        0.125,
        0.25,
    )
    graph = _graph()
    before = _state(graph)
    assert _coupling_activity_policy(graph.graph) == (0.05, 0.01)
    assert _state(graph) == before


@pytest.mark.parametrize(
    "invalid",
    (
        True,
        False,
        math.nan,
        math.inf,
        -math.inf,
        -0.01,
        "0.1",
        complex(1, 0),
        Decimal("1e-1000"),
        Q(10**400),
    ),
)
def test_policy_and_direct_guard_share_strict_threshold_admission(invalid):
    for key in ("UM_MIN_EPI", "UM_MIN_VF"):
        graph = _graph()
        graph.graph[key] = invalid
        before = _state(graph)
        with pytest.raises(OperatorPreconditionError, match=key):
            _coupling_activity_policy(graph.graph)
        with pytest.raises(OperatorPreconditionError, match=key):
            validate_coupling(graph, 0)
        assert _state(graph) == before


@pytest.mark.parametrize("path", ("public", "atomic"))
@pytest.mark.parametrize("key", ("UM_MIN_EPI", "UM_MIN_VF"))
def test_nan_policy_cannot_bypass_public_or_atomic_admission(path, key):
    graph = _graph()
    graph.graph[key] = math.nan
    if key == "UM_MIN_EPI":
        graph.nodes[0][ALIAS_EPI[0]] = 2.0**-240
    else:
        graph.nodes[0][ALIAS_VF[0]] = 0.0
    before = _state(graph)
    with pytest.raises(OperatorPreconditionError, match=key):
        _invoke(graph, path)
    assert _state(graph) == before
    assert "_last_operator_applied" not in graph.__dict__


@pytest.mark.parametrize("invalid", (True, math.nan, math.inf, "0.5", Q(1, 10**400)))
@pytest.mark.parametrize("aliases", (ALIAS_EPI, ALIAS_VF))
def test_invalid_authoritative_state_cannot_fall_through_any_path(aliases, invalid):
    for path in ("direct", "public", "atomic"):
        graph = _graph()
        graph.nodes[0][aliases[0]] = invalid
        graph.nodes[0][aliases[1]] = 0.5
        before = _state(graph)
        with pytest.raises(OperatorPreconditionError):
            _invoke(graph, path)
        assert _state(graph) == before
        assert "_last_operator_applied" not in graph.__dict__


def test_negative_capacity_rejects_even_when_selected_minimum_is_zero():
    for path in ("direct", "public", "atomic"):
        graph = _graph()
        graph.graph["UM_MIN_VF"] = 0
        graph.nodes[0][ALIAS_VF[0]] = -Q(1, 8)
        before = _state(graph)
        with pytest.raises(OperatorPreconditionError, match="nu_f"):
            _invoke(graph, path)
        assert _state(graph) == before


def test_invalid_later_target_rejects_before_any_atomic_proposal():
    graph = _graph()
    graph.nodes[1][ALIAS_VF[0]] = math.nan
    graph.nodes[1][ALIAS_VF[1]] = 1.0
    before = _state(graph)
    with pytest.raises(OperatorPreconditionError, match="nu_f"):
        _invoke(graph, "atomic")
    assert _state(graph) == before
    assert "_last_operator_applied" not in graph.__dict__


def test_all_selected_inputs_admit_before_activity_comparisons():
    graph = _graph()
    graph.nodes[0][ALIAS_EPI[0]] = 0.0
    graph.graph["UM_MIN_VF"] = math.nan
    with pytest.raises(OperatorPreconditionError, match="UM_MIN_VF"):
        validate_coupling(graph, 0)
    graph.graph["UM_MIN_VF"] = 0.01
    graph.nodes[0][ALIAS_VF[0]] = math.nan
    with pytest.raises(OperatorPreconditionError, match="nu_f"):
        validate_coupling(graph, 0)


def test_inclusive_signed_form_and_capacity_boundaries_remain_read_only():
    for form in (0.05, -0.05):
        graph = _graph()
        graph.nodes[0][ALIAS_EPI[0]] = form
        graph.nodes[0][ALIAS_VF[0]] = 0.01
        before = _state(graph)
        validate_coupling(graph, 0)
        assert _state(graph) == before
        graph.nodes[0][ALIAS_EPI[0]] = math.copysign(math.nextafter(0.05, 0.0), form)
        with pytest.raises(OperatorPreconditionError, match="EPI too low"):
            validate_coupling(graph, 0)
        graph.nodes[0][ALIAS_EPI[0]] = form
        graph.nodes[0][ALIAS_VF[0]] = math.nextafter(0.01, 0.0)
        with pytest.raises(OperatorPreconditionError, match="frequency too low"):
            validate_coupling(graph, 0)


def test_explicit_zero_minima_remain_valid_without_waiving_live_u3():
    graph = _graph()
    graph.graph.update(UM_MIN_EPI=0, UM_MIN_VF=0)
    graph.nodes[0][ALIAS_EPI[0]] = 0.0
    graph.nodes[0][ALIAS_VF[0]] = 0.0
    before = _state(graph)
    validate_coupling(graph, 0)
    assert _state(graph) == before
    graph.nodes[1]["theta"] = math.pi
    with pytest.raises(OperatorPreconditionError, match="U3"):
        validate_coupling(graph, 0)


def test_signed_uniform_bepi_admits_without_projecting_nonuniform_form():
    graph = _graph()
    for form in (
        BEPIElement((-0.5, -0.5), (-0.5, -0.5), (0.0, 1.0)),
        serialize_bepi(-0.5),
    ):
        graph.nodes[0][ALIAS_EPI[0]] = form
        before = _state(graph)
        validate_coupling(graph, 0)
        assert graph.nodes[0][ALIAS_EPI[0]] is form
        assert _state(graph) == before
    graph.nodes[0][ALIAS_EPI[0]] = BEPIElement((-0.5, -0.4), (-0.5, -0.5), (0.0, 1.0))
    with pytest.raises(OperatorPreconditionError, match="EPI"):
        validate_coupling(graph, 0)


@pytest.mark.parametrize("selection", ("absent", "graph_false", "call_false"))
def test_inactive_policy_is_not_evaluated_by_application_reader(selection):
    graph = _graph()
    graph.graph.update(UM_MIN_EPI=math.nan, UM_MIN_VF=math.nan)
    kwargs = {}
    if selection == "absent":
        del graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"]
    elif selection == "graph_false":
        graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = False
    else:
        kwargs["validate_preconditions"] = False
    before = _state(graph)
    Coupling()._validate_application_preconditions(graph, 0, **kwargs)
    assert _state(graph) == before
    # Policy selection never disables the separate mandatory hard invariant.
    graph.nodes[1]["theta"] = math.pi
    with pytest.raises(OperatorPreconditionError, match="U3"):
        Coupling()._validate_hard_invariants(graph, 0)
