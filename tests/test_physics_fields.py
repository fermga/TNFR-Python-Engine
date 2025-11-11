import math

import networkx as nx

from tnfr.physics.fields import (
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
)


def test_phase_gradient_line_monotonic():
    # Line: 0-1-2 with linear phase ramp
    G = nx.path_graph(3)
    phases = {0: 0.0, 1: 0.1, 2: 0.2}
    for n, p in phases.items():
        G.nodes[n]["phase"] = p
    grad = compute_phase_gradient(G)
    # Middle node sees two neighbors with symmetric deltas → larger average than ends with one neighbor?
    # Ends have one neighbor delta=0.1; middle has deltas 0.1 and 0.1 → average also 0.1
    assert math.isclose(grad[0], 0.1, rel_tol=1e-6)
    assert math.isclose(grad[1], 0.1, rel_tol=1e-6)
    assert math.isclose(grad[2], 0.1, rel_tol=1e-6)


def test_phase_curvature_constant_zero():
    # Any graph with constant phase should have zero curvature
    G = nx.cycle_graph(6)
    for n in G.nodes():
        G.nodes[n]["phase"] = 1.234
    curv = compute_phase_curvature(G)
    assert all(abs(v) < 1e-12 for v in curv.values())


def test_coherence_length_ring_exponential():
    # Build a ring and set coherence = exp(-d/xi_true) from a seed
    n = 20
    xi_true = 3.5
    G = nx.cycle_graph(n)
    seed = 0
    # Compute distances on ring
    dist = nx.single_source_shortest_path_length(G, seed)
    for i in G.nodes():
        d = dist[i]
        G.nodes[i]["coherence"] = math.exp(-d / xi_true)
    xi_est = estimate_coherence_length(G)
    # Expect estimation within ~30% for small n
    assert xi_est > 0
    assert abs(xi_est - xi_true) / xi_true < 0.35
