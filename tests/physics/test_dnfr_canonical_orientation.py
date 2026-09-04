"""Canonicity regression tests: directed orientation and weighted EPI channel.

These lock in the 2026-09-04 canonicity audit fixes (invariant #1):

* The EPI channel of ΔNFR realizes the **outgoing** random-walk Laplacian
  L_out = I − D⁻¹W (node i receives the weighted mean of the nodes it points
  to) — exactly the operator of
  :func:`tnfr.physics.structural_diffusion.structural_diffusion_operator`.
* Edge weights reach the EPI channel; unit weights reproduce the unweighted
  result bitwise; mutating a weight takes effect immediately (no stale cache).

The default (vectorized/fused) ΔNFR path is the canonical, audit-measured
path and the one used by every numpy-enabled runtime (including the
number-theory residue digraphs).
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.dynamics.fused_dnfr import compute_fused_gradients_symmetric
from tnfr.physics.structural_diffusion import structural_diffusion_operator

_EPI = {0: 1.0, 1: 0.3, 2: -0.5, 3: 0.8, 4: -0.2}


def _seed_nodes(G):
    """Uniform phase/νf so only the EPI channel drives ΔNFR."""
    for n in G.nodes():
        G.nodes[n]["EPI"] = _EPI[n]
        G.nodes[n]["theta"] = 0.0
        G.nodes[n]["vf"] = 1.0
    G.graph["_dnfr_weights"] = {"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0}
    return G


def _epi_vector(G, nodes):
    return np.array([float(G.nodes[n]["EPI"]) for n in nodes], dtype=float)


def _read_dnfr(G, nodes):
    return np.array(
        [get_attr(G.nodes[n], ALIAS_DNFR, 0.0) for n in nodes], dtype=float
    )


def _lrw_gradient(G):
    """Canonical EPI-channel gradient g = −L_rw·EPI (outgoing L_out)."""
    nodes, lap = structural_diffusion_operator(G)
    return nodes, -(lap @ _epi_vector(G, nodes))


def _asymmetric_digraph():
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (3, 1)])
    return _seed_nodes(G)


def test_dnfr_epi_matches_outgoing_lrw_digraph():
    """Default ΔNFR EPI channel == −L_out·EPI on an asymmetric DiGraph."""
    G = _asymmetric_digraph()
    nodes, g_out = _lrw_gradient(G)
    default_compute_delta_nfr(G)
    g_dyn = _read_dnfr(G, nodes)
    assert np.max(np.abs(g_dyn - g_out)) < 1e-12


def test_dnfr_epi_rejects_incoming_transpose():
    """The dynamics must NOT realize the incoming transpose L_in = L_out^T."""
    G = _asymmetric_digraph()
    nodes = list(G.nodes())
    # L_in on G is L_out on the reversed graph (predecessors become successors).
    Grev = G.reverse(copy=True)
    _, lap_in = structural_diffusion_operator(Grev)
    g_in = -(lap_in @ _epi_vector(G, nodes))
    default_compute_delta_nfr(G)
    g_dyn = _read_dnfr(G, nodes)
    # The asymmetric graph makes L_in and L_out genuinely different.
    assert np.max(np.abs(g_dyn - g_in)) > 0.1


def test_weighted_dnfr_matches_weighted_lrw():
    """Non-uniform edge weights reproduce the weighted L_out on a DiGraph."""
    G = _asymmetric_digraph()
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0 + (u + 2 * v) % 3
    nodes, g_out = _lrw_gradient(G)
    default_compute_delta_nfr(G)
    g_dyn = _read_dnfr(G, nodes)
    assert np.max(np.abs(g_dyn - g_out)) < 1e-12


def test_weighted_dnfr_matches_weighted_lrw_undirected():
    """Weighted undirected graph also matches the weighted L_rw."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    _seed_nodes(G)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0 + (u + v) % 3
    nodes, g_out = _lrw_gradient(G)
    default_compute_delta_nfr(G)
    g_dyn = _read_dnfr(G, nodes)
    assert np.max(np.abs(g_dyn - g_out)) < 1e-12


def test_undirected_default_matches_lrw():
    """Undirected default path is unchanged: L_out == L_in == L_rw."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    _seed_nodes(G)
    nodes, g_out = _lrw_gradient(G)
    default_compute_delta_nfr(G)
    g_dyn = _read_dnfr(G, nodes)
    assert np.max(np.abs(g_dyn - g_out)) < 1e-12


def test_unit_weights_preserve_legacy_result():
    """edge_weight of ones is bitwise-identical to the unweighted kernel."""
    G = _asymmetric_digraph()
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    src, dst = [], []
    for node in nodes:
        for nb in G.neighbors(node):
            src.append(idx[node])
            dst.append(idx[nb])
    edge_src = np.asarray(src, dtype=np.intp)
    edge_dst = np.asarray(dst, dtype=np.intp)
    phase = np.zeros(len(nodes))
    epi = _epi_vector(G, nodes)
    vf = np.ones(len(nodes))
    weights = {"w_phase": 0.0, "w_epi": 1.0, "w_vf": 0.0, "w_topo": 0.0}

    g_none = compute_fused_gradients_symmetric(
        edge_src=edge_src, edge_dst=edge_dst, phase=phase, epi=epi, vf=vf,
        weights=weights, edge_weight=None, accumulate_both_directions=False,
        use_jit=False,
    )
    g_ones = compute_fused_gradients_symmetric(
        edge_src=edge_src, edge_dst=edge_dst, phase=phase, epi=epi, vf=vf,
        weights=weights, edge_weight=np.ones(edge_src.shape[0]),
        accumulate_both_directions=False, use_jit=False,
    )
    assert np.array_equal(g_none, g_ones)


def test_cache_invalidates_on_weight_change():
    """Mutating an edge weight changes ΔNFR on the next computation."""
    G = _asymmetric_digraph()
    nodes = list(G.nodes())
    default_compute_delta_nfr(G)
    before = _read_dnfr(G, nodes)

    # Add a non-unit weight without changing topology.
    G[0][1]["weight"] = 5.0
    default_compute_delta_nfr(G)
    after = _read_dnfr(G, nodes)

    assert np.max(np.abs(after - before)) > 1e-9
    # And it matches the freshly weighted L_out.
    _, g_out = _lrw_gradient(G)
    assert np.max(np.abs(after - g_out)) < 1e-12
