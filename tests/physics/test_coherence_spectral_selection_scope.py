"""The declared numerical mode cutoff can skip a positive structural gap.

These fixed P4 controls independently identify all four exact eigenvalues.
They change no estimator threshold and make no trajectory or correlation claim.
"""

import math
from fractions import Fraction as F

import networkx as nx
import pytest

from tnfr.mathematics.krylov import exact_rank
from tnfr.physics.fields import estimate_coherence_length_with_provenance
from tnfr.physics.structural_diffusion import structural_eigenvalues


def _graph(epsilon):
    graph = nx.path_graph(4)
    for left, right in graph.edges:
        graph.edges[left, right].update(
            weight=float(epsilon) if (left, right) == (1, 2) else 1.0,
            length=1.0,
        )
    nx.set_node_attributes(graph, 0.0, "delta_nfr")
    return graph


def _exact_spectrum(epsilon):
    """Reflection-even and reflection-odd eigenvectors of this weighted P4."""
    reciprocal = 1 / (1 + epsilon)
    gap = epsilon * reciprocal
    matrix = (
        (F(1), F(-1), F(0), F(0)),
        (-reciprocal, F(1), -gap, F(0)),
        (F(0), -gap, F(1), -reciprocal),
        (F(0), F(0), F(-1), F(1)),
    )
    vectors = (
        (F(1), F(1), F(1), F(1)),
        (F(1), reciprocal, -reciprocal, F(-1)),
        (F(1), -reciprocal, -reciprocal, F(1)),
        (F(1), F(-1), F(1), F(-1)),
    )
    eigenvalues = (F(0), gap, 2 - gap, F(2))
    assert exact_rank(vectors) == 4
    for vector, eigenvalue in zip(vectors, eigenvalues, strict=True):
        assert tuple(
            sum((a * b for a, b in zip(row, vector, strict=True)), F(0))
            for row in matrix
        ) == tuple(eigenvalue * value for value in vector)
    assert 0 < gap < 1 < eigenvalues[2] < 2
    return eigenvalues


def test_weak_connection_can_make_the_selected_scale_skip_the_actual_gap():
    eigenvalues = _exact_spectrum(F(1, 2**40))
    graph = _graph(F(1, 2**40))
    observed = structural_eigenvalues(graph)
    assert tuple(observed) == pytest.approx(tuple(map(float, eigenvalues)), abs=1e-14)
    assert 0 < eigenvalues[1] < F(1, 10**9)
    assert 0 < observed[1] < 1e-9 < observed[2]

    estimate = estimate_coherence_length_with_provenance(graph)
    # Four nodes give only six unordered pairs, below the fit's ten-pair
    # requirement. A flat pressure field provides no decay fit either.
    assert estimate.method == "spectral_gap"
    assert estimate.fit_available is False
    assert estimate.fit_quality == "autocorrelation fit unavailable"
    assert estimate.positive_mode_selection == "smallest eigenvalue above 1e-9"
    assert "dimensionless" in estimate.distance_weighting
    assert estimate.graph_regime == "undirected; connected"
    assert estimate.value == pytest.approx(
        1 / math.sqrt(float(eigenvalues[2])), rel=1e-13
    )
    slowest_mode_scale = 1 / math.sqrt(float(eigenvalues[1]))
    assert slowest_mode_scale > 10**6
    assert 0 < estimate.value < 1


def test_gap_above_the_cutoff_is_selected_with_the_same_provenance():
    eigenvalues = _exact_spectrum(F(1))
    assert eigenvalues == (F(0), F(1, 2), F(3, 2), F(2))
    estimate = estimate_coherence_length_with_provenance(_graph(F(1)))
    assert estimate.method == "spectral_gap"
    assert estimate.positive_mode_selection == "smallest eigenvalue above 1e-9"
    assert estimate.value == pytest.approx(math.sqrt(2), rel=1e-13)
