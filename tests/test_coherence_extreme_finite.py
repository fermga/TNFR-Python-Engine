"""Extreme finite-value regressions for coherence similarities."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from tnfr.metrics.coherence import coherence_matrix


@pytest.mark.parametrize("use_numpy", [True, False])
def test_extreme_finite_epi_similarity_is_bounded_and_backend_invariant(
    use_numpy: bool,
) -> None:
    maximum = np.finfo(float).max
    graph = nx.Graph()
    graph.add_edge(0, 1)
    for node, epi in enumerate((maximum, -maximum)):
        graph.nodes[node].update(EPI=epi, nu_f=1.0, phase=0.0, Si=1.0)

    _, matrix = coherence_matrix(graph, use_numpy=use_numpy)

    assert matrix is not None
    values = [entry[2] for entry in matrix]
    assert values == pytest.approx([0.67, 0.67])
    assert all(math.isfinite(value) for value in values)
