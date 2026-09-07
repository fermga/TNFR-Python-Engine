"""Executable contracts for the structural-emergence heuristics."""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.metrics.emergence import (
    compute_bifurcation_rate,
    compute_emergence_index,
)


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
        sub_epis=[
            {"timestamp": len(history)}
            for _ in range(sub_epi_count)
        ],
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