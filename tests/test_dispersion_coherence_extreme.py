"""Signed pressure dispersion retains its scale at finite binary64 extremes."""

from __future__ import annotations

import math
import sys

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.metrics import coherence


def _pair(values: tuple[object, object]) -> nx.Graph:
    graph = nx.path_graph(2)
    for node, value in enumerate(values):
        graph.nodes[node][ALIAS_DNFR[0]] = value
    return graph


@pytest.mark.parametrize("use_numpy", [True, False])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize(
    "scale", [1.0, 1e200, 1e-200, math.ulp(0.0), sys.float_info.max / 2.0]
)
def test_dispersion_preserves_scale_and_global_sign_at_extremes(
    monkeypatch: pytest.MonkeyPatch, use_numpy: bool, sign: float, scale: float
) -> None:
    if not use_numpy:
        monkeypatch.setattr(coherence, "np", None)
    graph = _pair((sign * scale, sign * 2.0 * scale))

    # Population std(m,2m)=|m|/2 and max magnitude=2|m|, hence C=3/4.
    assert coherence.compute_global_coherence(graph) == 0.75
    assert coherence.compute_local_coherence(graph, 0) == 0.75


@pytest.mark.parametrize("use_numpy", [True, False])
@pytest.mark.parametrize("scale", [math.ulp(0.0), 1.0, sys.float_info.max])
def test_opposed_extreme_pressures_have_maximal_normalized_dispersion(
    monkeypatch: pytest.MonkeyPatch, use_numpy: bool, scale: float
) -> None:
    if not use_numpy:
        monkeypatch.setattr(coherence, "np", None)
    graph = _pair((-scale, scale))

    assert coherence.compute_global_coherence(graph) == 0.0
    assert coherence.compute_local_coherence(graph, 0) == 0.0


@pytest.mark.parametrize("values", [(0.0, 0.0), (-2.0, -2.0), (1e308, 1e308)])
def test_uniform_and_empty_pressure_conventions_are_preserved(
    values: tuple[float, float],
) -> None:
    graph = _pair(values)

    assert coherence.compute_global_coherence(graph) == 1.0
    assert coherence.compute_local_coherence(graph, 0) == 1.0
    assert coherence.compute_global_coherence(nx.Graph()) == 1.0
    assert coherence.compute_local_coherence(nx.empty_graph(1), 0) == 1.0


@pytest.mark.parametrize("use_numpy", [True, False])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "1"])
def test_invalid_provided_pressures_are_rejected_by_both_wrappers(
    monkeypatch: pytest.MonkeyPatch, use_numpy: bool, value: object
) -> None:
    if not use_numpy:
        monkeypatch.setattr(coherence, "np", None)
    graph = _pair((0.0, value))
    graph.nodes[1][ALIAS_DNFR[1]] = 0.0

    with pytest.raises((TypeError, ValueError)):
        coherence.compute_global_coherence(graph)
    with pytest.raises((TypeError, ValueError)):
        coherence.compute_local_coherence(graph, 0)


def test_local_dispersion_only_reads_the_selected_neighborhood() -> None:
    graph = nx.path_graph(4)
    for node, value in enumerate((1e200, 2e200, 0.0, float("nan"))):
        graph.nodes[node][ALIAS_DNFR[0]] = value

    assert coherence.compute_local_coherence(graph, 0, radius=1) == 0.75
    with pytest.raises(ValueError, match="finite"):
        coherence.compute_local_coherence(graph, 0, radius=3)
