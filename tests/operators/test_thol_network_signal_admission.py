"""THOL admits consumed network values before deriving lossy signal summaries."""

from __future__ import annotations

import math
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA
from tnfr.operators import apply_glyph
from tnfr.operators.definitions import SelfOrganization
from tnfr.operators.metabolism import capture_network_signals, compose_subepi_amplitude
from tnfr.operators.network_stage import execute_self_organization_stage
from tnfr.operators.preconditions import OperatorPreconditionError


def _graph() -> nx.Graph:
    graph = nx.Graph(THOL_METABOLIC_ENABLED=True, THOL_PROPAGATION_ENABLED=False)
    graph.add_node(
        0,
        EPI=0.6,
        nu_f=1.0,
        theta=0.1,
        delta_nfr=0.2,
        epi_history=[0.0, 0.1, 0.6],
        glyph_history=["OZ"],
    )
    graph.add_node(1, EPI=0.4, nu_f=1.0, theta=0.12, delta_nfr=0.1)
    graph.add_edge(0, 1)
    return graph


def _state(graph: nx.Graph) -> tuple[object, ...]:
    return deepcopy(
        (
            dict(graph.nodes(data=True)),
            tuple(graph.edges(data=True)),
            {
                key: value
                for key, value in graph.graph.items()
                if key not in {"_node_cache", "_node_cache_weak", "integrity_monitor"}
            },
        )
    )


def _run(graph: nx.Graph, route: str) -> object:
    if route == "helper":
        return capture_network_signals(graph, 0)
    if route == "public":
        return SelfOrganization()(graph, 0, tau=0.1)
    return execute_self_organization_stage(graph, SelfOrganization(), (0,), tau=0.1)


@pytest.mark.parametrize("route", ["helper", "public", "stage"])
@pytest.mark.parametrize("target", [0, 1], ids=["parent", "neighbor"])
@pytest.mark.parametrize(
    "invalid",
    [True, "0.25", math.nan, math.inf, Fraction(1, 10**400)],
    ids=["boolean", "text", "nan", "infinity", "underflow"],
)
def test_authoritative_phase_rejects_without_using_secondary_alias(
    route: str, target: int, invalid: object
) -> None:
    graph = _graph()
    graph.nodes[target][ALIAS_THETA[0]] = invalid
    graph.nodes[target][ALIAS_THETA[1]] = 0.75
    before = _state(graph)

    with pytest.raises(OperatorPreconditionError):
        _run(graph, route)

    assert _state(graph) == before


@pytest.mark.parametrize("route", ["helper", "public", "stage"])
@pytest.mark.parametrize("target", [0, 1], ids=["parent", "neighbor"])
def test_valid_secondary_phase_alias_preserves_the_existing_map(
    route: str, target: int
) -> None:
    graph = _graph()
    expected_signals = capture_network_signals(graph, 0)
    expected_amplitude = compose_subepi_amplitude(0.6, expected_signals)
    graph.nodes[target][ALIAS_THETA[1]] = graph.nodes[target].pop(ALIAS_THETA[0])

    result = _run(graph, route)

    if route == "helper":
        assert result == expected_signals
    else:
        record = graph.nodes[0]["sub_epis"][0]
        assert record["epi"] == expected_amplitude
        assert graph.nodes[record["node_id"]][ALIAS_EPI[0]] == expected_amplitude
        assert graph.nodes[0][ALIAS_EPI[0]] == 0.6


@pytest.mark.parametrize("route", ["helper", "public", "stage"])
def test_neighbor_epi_admission_stays_in_shared_capture(route: str) -> None:
    graph = _graph()
    graph.nodes[1][ALIAS_EPI[0]] = "0.4"
    graph.nodes[1][ALIAS_EPI[1]] = 0.4
    before = _state(graph)

    with pytest.raises(OperatorPreconditionError):
        _run(graph, route)

    assert _state(graph) == before


@pytest.mark.parametrize("route", ["helper", "public", "stage"])
def test_finite_neighbor_forms_do_not_admit_an_overflowed_mean(route: str) -> None:
    graph = _graph()
    graph.nodes[1][ALIAS_EPI[0]] = 1e308
    graph.add_node(2, EPI=1e308, nu_f=1.0, theta=0.12, delta_nfr=0.0)
    graph.add_edge(0, 2)
    before = _state(graph)

    with pytest.warns(RuntimeWarning, match="overflow"):
        with pytest.raises(OperatorPreconditionError, match="mean neighbor EPI"):
            _run(graph, route)

    assert _state(graph) == before


def test_finite_mean_does_not_admit_an_overflowed_gradient() -> None:
    graph = _graph()
    graph.nodes[0][ALIAS_EPI[0]] = -1e308
    graph.nodes[1][ALIAS_EPI[0]] = 1e308
    before = _state(graph)

    with pytest.raises(OperatorPreconditionError, match="EPI gradient"):
        capture_network_signals(graph, 0)

    assert _state(graph) == before


def test_isolated_capture_has_no_consumed_parent_channel() -> None:
    graph = nx.Graph()
    graph.add_node(0, EPI="unused", theta=math.nan)
    before = _state(graph)

    assert capture_network_signals(graph, 0) is None
    assert _state(graph) == before


def test_missing_phase_still_uses_the_declared_zero_default() -> None:
    graph = _graph()
    for node in graph:
        for alias in ALIAS_THETA:
            graph.nodes[node].pop(alias, None)

    signals = capture_network_signals(graph, 0)

    assert signals["phase_variance"] == 0.0
    assert signals["coupling_strength_mean"] == 1.0


def test_capture_retains_directed_outgoing_support_and_circular_statistics() -> None:
    graph = nx.DiGraph()
    graph.add_node(0, EPI=0.25, theta=0.0)
    graph.add_node(1, EPI=0.5, theta=0.25)
    graph.add_node(2, EPI=1.0, theta=math.tau - 0.25)
    graph.add_node(3, EPI="not consumed", theta=math.nan)
    graph.add_edges_from([(0, 1), (0, 2), (3, 0)])

    signals = capture_network_signals(graph, 0)

    assert signals == pytest.approx(
        {
            "neighbor_count": 2,
            "mean_neighbor_epi": 0.75,
            "epi_gradient": 0.5,
            "phase_variance": 1.0 - math.cos(0.25),
            "coupling_strength_mean": 1.0 - 0.25 / math.pi,
        }
    )


def test_primitive_thol_has_no_network_signal_read() -> None:
    ordinary = _graph()
    unrelated_invalid_neighbor = _graph()
    unrelated_invalid_neighbor.nodes[1][ALIAS_THETA[0]] = math.nan

    apply_glyph(ordinary, 0, "THOL")
    apply_glyph(unrelated_invalid_neighbor, 0, "THOL")

    assert unrelated_invalid_neighbor.nodes[0] == ordinary.nodes[0]
    assert get_attr(ordinary.nodes[0], ALIAS_DNFR) != 0.2
    assert tuple(unrelated_invalid_neighbor.nodes) == (0, 1)
    assert "sub_epis" not in unrelated_invalid_neighbor.nodes[0]
