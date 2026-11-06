"""NAV cannot create zero capacity through loss of a positive product."""

from __future__ import annotations

import math
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.operators.definitions import Transition
from tnfr.operators.network_stage import execute_pointwise_stage
from tnfr.operators.preconditions import OperatorPreconditionError


def _graph(*, capacity: float = 0.5, epi: float = 0.5) -> nx.Graph:
    graph = nx.Graph(
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        NAV_MIN_VF=0.0,
    )
    graph.add_node(
        0,
        EPI=epi,
        nu_f=capacity,
        delta_nfr=0.125,
        theta=0.125,
        glyph_history=["AL", "IL"],
    )
    return graph


def _snapshot(graph: nx.Graph) -> tuple[object, ...]:
    return deepcopy(
        (
            dict(graph.nodes(data=True)),
            tuple(graph.edges(data=True)),
            {
                key: value
                for key, value in graph.graph.items()
                if key not in {"_node_cache", "_node_cache_weak"}
            },
        )
    )


def _apply(graph: nx.Graph, route: str, factor: float) -> None:
    operator = Transition()
    if route == "public":
        operator(graph, 0, vf_factor=factor)
    elif route == "stage":
        execute_pointwise_stage(graph, operator, (0,), vf_factor=factor)
    else:
        operator._apply_structural_transition(
            graph, 0, operator._detect_regime(graph, 0), vf_factor=factor
        )


@pytest.mark.parametrize("route", ["public", "stage", "compatibility"])
def test_positive_capacity_product_lost_to_zero_rejects_before_writes(
    route: str,
) -> None:
    graph = _graph()
    factor = math.ulp(0.0)
    assert Fraction(graph.nodes[0]["nu_f"]) * Fraction(factor) > 0
    assert graph.nodes[0]["nu_f"] * factor == 0.0
    before = _snapshot(graph)

    with pytest.raises(OperatorPreconditionError, match="nu_f proposal.*underflows"):
        _apply(graph, route, factor)

    assert _snapshot(graph) == before


@pytest.mark.parametrize("route", ["public", "stage", "compatibility"])
@pytest.mark.parametrize("factor", [0.0, 2 * math.ulp(0.0), 0.5, 1.5])
def test_deliberate_zero_and_representable_products_retain_existing_map(
    route: str, factor: float
) -> None:
    graph = _graph()
    expected = 0.5 * factor

    _apply(graph, route, factor)

    assert graph.nodes[0]["nu_f"] == expected
    assert graph.nodes[0]["EPI"] == 0.5
    assert graph.graph["_nav_transitions"][-1]["vf_after"] == expected


@pytest.mark.parametrize("route", ["public", "stage", "compatibility"])
@pytest.mark.parametrize(
    "capacity, epi, multiplier",
    [(0.0, 0.5, 1.2), (0.025, 0.5, 1.2), (1.0, 0.75, 0.95)],
)
def test_other_regimes_keep_their_multiplier_and_zero_source_policy(
    route: str, capacity: float, epi: float, multiplier: float
) -> None:
    graph = _graph(capacity=capacity, epi=epi)

    _apply(graph, route, math.ulp(0.0))

    assert graph.nodes[0]["nu_f"] == capacity * multiplier
    assert graph.nodes[0]["EPI"] == epi


def test_later_target_underflow_rolls_back_the_complete_stage() -> None:
    graph = _graph(capacity=1.0)
    graph.add_node(1, **deepcopy(_graph().nodes[0]))
    factor = math.ulp(0.0)
    assert graph.nodes[0]["nu_f"] * factor > 0.0
    assert graph.nodes[1]["nu_f"] * factor == 0.0
    before = _snapshot(graph)

    with pytest.raises(OperatorPreconditionError, match="nu_f proposal.*underflows"):
        execute_pointwise_stage(graph, Transition(), (0, 1), vf_factor=factor)

    assert _snapshot(graph) == before
