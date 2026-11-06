"""Executable contracts for the structural-emergence heuristics."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.mathematics import BEPIElement
from tnfr.metrics.emergence import (
    compute_bifurcation_rate,
    compute_emergence_index,
    compute_metabolic_efficiency,
)
from tnfr.metrics.learning_metrics import compute_learning_efficiency
from tnfr.types import ensure_bepi, serialize_bepi


def _node(
    *,
    epi: float,
    initial_epi: float,
    sub_epi_count: int,
    thol_count: int,
) -> nx.Graph:
    graph = nx.Graph()
    history = ["THOL"] * thol_count
    graph.add_node(
        0,
        EPI=epi,
        epi_initial=initial_epi,
        glyph_history=history,
        sub_epis=[{"timestamp": len(history)} for _ in range(sub_epi_count)],
    )
    return graph


def test_emergence_index_is_exact_zero_when_a_factor_is_absent() -> None:
    graph = _node(epi=0.8, initial_epi=0.2, sub_epi_count=0, thol_count=2)

    assert compute_emergence_index(graph, 0) == 0.0


def test_emergence_index_clamps_net_loss_to_real_zero() -> None:
    graph = _node(epi=0.2, initial_epi=0.8, sub_epi_count=1, thol_count=1)

    result = compute_emergence_index(graph, 0)

    assert isinstance(result, float)
    assert result == 0.0


def test_emergence_index_is_cube_root_of_declared_positive_product() -> None:
    graph = _node(epi=0.9, initial_epi=0.1, sub_epi_count=2, thol_count=2)

    # complexity=2, rate=2/10, efficiency=(0.9-0.1)/2=0.4
    assert compute_emergence_index(graph, 0) == pytest.approx(0.16 ** (1.0 / 3.0))


@pytest.mark.parametrize("window", [0, -1, True, 1.5])
def test_bifurcation_rate_rejects_invalid_windows(window: object) -> None:
    graph = _node(epi=0.9, initial_epi=0.1, sub_epi_count=1, thol_count=1)

    with pytest.raises(ValueError, match="positive integer"):
        compute_bifurcation_rate(graph, 0, window=window)


def test_bifurcation_rate_uses_monotonic_step_after_trace_eviction() -> None:
    graph = _node(epi=0.9, initial_epi=0.1, sub_epi_count=0, thol_count=2)
    graph.nodes[0]["_operator_step"] = 50
    graph.nodes[0]["sub_epis"] = [
        {"timestamp": 35},
        {"timestamp": 45},
        {"timestamp": 50},
    ]

    assert compute_bifurcation_rate(graph, 0, window=10) == pytest.approx(0.2)


def test_operator_step_counter_survives_bounded_glyph_history() -> None:
    from tnfr.glyph_history import current_operator_step, push_glyph

    data: dict[str, object] = {}
    for glyph in ("AL", "IL", "OZ", "THOL", "SHA"):
        push_glyph(data, glyph, window=2)

    assert list(data["glyph_history"]) == ["THOL", "SHA"]
    assert current_operator_step(data) == 5


@pytest.mark.parametrize("count,value", [(10, 1e308), (1, math.ulp(0.0))])
def test_emergence_root_stays_finite_when_intermediate_product_does_not(count, value):
    graph = _node(epi=value, initial_epi=0.0, sub_epi_count=count, thol_count=1)
    with localcontext() as context:
        context.prec = 90
        exact_product = Decimal(count * count) * Decimal.from_float(value) / 10
        expected = float(exact_product ** (Decimal(1) / 3))
    actual = compute_emergence_index(graph, 0)
    assert math.isfinite(actual) and actual > 0
    assert actual == pytest.approx(expected, rel=2e-14, abs=0)


def test_metabolic_division_precedes_unrepresentable_materialization():
    graph = _node(epi=1e308, initial_epi=-1e308, sub_epi_count=1, thol_count=2)
    assert compute_metabolic_efficiency(graph, 0) == 1e308
    graph.nodes[0]["glyph_history"] = ["THOL"]
    with pytest.raises(ValueError, match="finite float range"):
        compute_metabolic_efficiency(graph, 0)
    # The composite can still be representable without materializing that ratio.
    assert math.isfinite(compute_emergence_index(graph, 0))


@pytest.mark.parametrize(
    "reader",
    [
        compute_metabolic_efficiency,
        compute_emergence_index,
        compute_learning_efficiency,
    ],
)
@pytest.mark.parametrize("key", ["EPI", "epi_initial"])
@pytest.mark.parametrize(
    "invalid",
    [
        math.nan,
        math.inf,
        True,
        "0.5",
        BEPIElement((-2.0, -2.0), (-2.0, -1.0), (0.0, 1.0)),
    ],
)
def test_emergence_heuristics_reject_invalid_scalar_form(reader, key, invalid):
    graph = _node(epi=0.5, initial_epi=0.0, sub_epi_count=1, thol_count=1)
    graph.nodes[0][key] = invalid
    # A secondary valid alias must not hide an invalid authoritative EPI.
    graph.nodes[0]["epi"] = 0.5
    with pytest.raises(ValueError):
        reader(graph, 0)


@pytest.mark.parametrize("encode", [ensure_bepi, serialize_bepi])
def test_metabolic_efficiency_reuses_signed_live_and_serialized_chart(encode):
    graph = _node(epi=-0.25, initial_epi=-0.75, sub_epi_count=1, thol_count=2)
    for key in ("EPI", "epi_initial"):
        graph.nodes[0][key] = encode(graph.nodes[0][key])
    assert compute_metabolic_efficiency(graph, 0) == 0.25


def test_explicit_operator_clock_is_not_moved_to_admit_future_records():
    graph = _node(epi=0.5, initial_epi=0.0, sub_epi_count=1, thol_count=1)
    graph.nodes[0].update(_operator_step=2, sub_epis=[{"timestamp": 100}])
    with pytest.raises(ValueError, match="exceeds"):
        compute_bifurcation_rate(graph, 0)
    assert graph.nodes[0]["_operator_step"] == 2


def test_learning_ratio_retains_representable_large_endpoint_change():
    graph = _node(epi=1e308, initial_epi=-1e308, sub_epi_count=0, thol_count=2)
    assert compute_learning_efficiency(graph, 0) == 1e308
    graph.nodes[0]["glyph_history"] = ["AL"]
    with pytest.raises(ValueError):
        compute_learning_efficiency(graph, 0)


@pytest.mark.parametrize("encode", [ensure_bepi, serialize_bepi])
def test_learning_ratio_uses_absolute_signed_change_without_mutation(encode):
    graph = _node(epi=-0.75, initial_epi=0.25, sub_epi_count=0, thol_count=2)
    for key in ("EPI", "epi_initial"):
        graph.nodes[0][key] = encode(graph.nodes[0][key])
    before = dict(graph.nodes[0])
    assert compute_learning_efficiency(graph, 0) == 0.5
    assert graph.nodes[0] == before


def test_learning_ratio_rejects_underflowing_input_and_output():
    graph = _node(
        epi=Fraction(1, 2**1075), initial_epi=0.0, sub_epi_count=0, thol_count=2
    )
    with pytest.raises(ValueError):
        compute_learning_efficiency(graph, 0)
    graph.nodes[0]["EPI"] = math.ulp(0.0)
    with pytest.raises(ValueError, match="underflows"):
        compute_learning_efficiency(graph, 0)
    graph.nodes[0]["glyph_history"] = []
    assert compute_learning_efficiency(graph, 0) == 0.0
