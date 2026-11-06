"""Direct NAV admission shares the public/staged policy and rejects invalid evidence."""

import warnings
from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.operators.definitions import Transition
from tnfr.operators.network_stage import execute_pointwise_stage
from tnfr.operators.preconditions import OperatorPreconditionError, validate_transition


def _graph():
    graph = nx.Graph()
    graph.add_node(0, EPI=0.5, nu_f=1.0, delta_nfr=0.0, theta=0.0, glyph_history=["AL"])
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    graph.graph["GRAMMAR_REJECTION_MODE"] = "raise"
    return graph


def _apply(graph, route):
    if route == "validator":
        validate_transition(graph, 0)
    elif route == "public":
        Transition()(graph, 0)
    else:
        execute_pointwise_stage(graph, Transition(), (0,))


@pytest.mark.parametrize("route", ("validator", "public", "stage"))
@pytest.mark.parametrize(
    "key", ("NAV_MIN_VF", "NAV_MAX_DNFR", "NAV_MIN_EPI_FROM_LATENCY")
)
@pytest.mark.parametrize("value", (float("nan"), float("inf"), True, -1, Q(1, 10**400)))
def test_invalid_policy_rejected_without_state_or_warning(route, key, value):
    graph = _graph()
    graph.graph[key] = value
    before = deepcopy(graph)
    with warnings.catch_warnings(record=True) as captured:
        with pytest.raises(OperatorPreconditionError, match=key):
            _apply(graph, route)
    assert not captured
    assert nx.utils.graphs_equal(graph, before)


@pytest.mark.parametrize("route", ("validator", "public", "stage"))
@pytest.mark.parametrize("value", (1, "false", None))
def test_invalid_optional_sequence_flag_has_same_boundary(route, value):
    graph = _graph()
    graph.graph["NAV_STRICT_SEQUENCE_CHECK"] = value
    before = deepcopy(graph)
    with pytest.raises(OperatorPreconditionError, match="NAV_STRICT_SEQUENCE_CHECK"):
        _apply(graph, route)
    assert nx.utils.graphs_equal(graph, before)


@pytest.mark.parametrize("key", ("nu_f", "delta_nfr", "EPI"))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), True, Q(1, 10**400)))
def test_direct_reader_rejects_invalid_consumed_state(key, value):
    graph = _graph()
    graph.nodes[0][key] = value
    before = deepcopy(graph)
    error = ValueError if key == "EPI" else OperatorPreconditionError
    with pytest.raises(error):
        validate_transition(graph, 0)
    assert nx.utils.graphs_equal(graph, before)


def test_direct_reader_does_not_fall_through_invalid_authoritative_alias():
    graph = _graph()
    graph.nodes[0].update(EPI=float("nan"), epi=0.5)
    with pytest.raises(ValueError):
        validate_transition(graph, 0)


def test_inclusive_policy_boundaries_and_signed_pressure_remain_valid():
    for pressure in (-1.0, 1.0):
        graph = _graph()
        graph.nodes[0].update(nu_f=0.01, delta_nfr=pressure)
        before = deepcopy(graph)
        validate_transition(graph, 0)
        assert nx.utils.graphs_equal(graph, before)
    graph.nodes[0]["delta_nfr"] = 1.01
    with pytest.raises(OperatorPreconditionError, match="too high"):
        validate_transition(graph, 0)


def test_predecessor_policy_is_advisory_and_consumes_only_enabled_history():
    graph = _graph()
    graph.nodes[0]["glyph_history"] = ["OZ"]
    with warnings.catch_warnings(record=True) as captured:
        validate_transition(graph, 0)
    assert not captured
    graph.graph["NAV_STRICT_SEQUENCE_CHECK"] = True
    before = deepcopy(graph)
    with pytest.warns(UserWarning, match="NAV applied after"):
        validate_transition(graph, 0)
    assert nx.utils.graphs_equal(graph, before)
    graph.nodes[0]["glyph_history"] = None
    with pytest.raises(OperatorPreconditionError, match="glyph_history"):
        validate_transition(graph, 0)


def test_latency_flag_is_not_coerced_by_truthiness():
    graph = _graph()
    graph.nodes[0]["latent"] = "false"
    with pytest.raises(OperatorPreconditionError, match="latent"):
        validate_transition(graph, 0)
