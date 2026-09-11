"""Canonicity regression: xi_C has a single public kernel (invariant #5).

The 2026-09-04 audit found ``telemetry()["canonical"]["xi_c"]`` returning NaN
on a flat connected field while ``tetrad().xi_c`` returned a finite spectral
fallback.  Both public paths must now call the one canonical kernel
:func:`estimate_coherence_length` (ADR-005), so they agree and never diverge.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from tnfr.dynamics import default_compute_delta_nfr
from tnfr.physics.canonical import estimate_coherence_length
from tnfr.physics.fields import compute_unified_telemetry


def _flat_graph(G):
    for nd in G.nodes():
        G.nodes[nd]["EPI"] = 0.5
        G.nodes[nd]["theta"] = 0.0
        G.nodes[nd]["vf"] = 1.0
    default_compute_delta_nfr(G)
    return G


def test_xi_c_public_api_parity_flat_connected():
    """telemetry xi_C == tetrad-kernel xi_C, and finite, on a flat ring."""
    G = _flat_graph(nx.cycle_graph(8))
    xi_tel = compute_unified_telemetry(G)["canonical"]["xi_c"]
    xi_kernel = estimate_coherence_length(G)
    assert np.isfinite(xi_tel)
    assert abs(float(xi_tel) - float(xi_kernel)) < 1e-12


def test_xi_c_no_nan_on_connected_nonflat():
    """A non-flat connected graph also yields a finite, matching xi_C."""
    G = nx.path_graph(10)
    for i, nd in enumerate(G.nodes()):
        G.nodes[nd]["EPI"] = 0.1 * i
        G.nodes[nd]["theta"] = 0.0
        G.nodes[nd]["vf"] = 1.0
    default_compute_delta_nfr(G)
    xi_tel = compute_unified_telemetry(G)["canonical"]["xi_c"]
    xi_kernel = estimate_coherence_length(G)
    assert np.isfinite(xi_tel)
    assert abs(float(xi_tel) - float(xi_kernel)) < 1e-12


def test_xi_c_disconnected_policy_is_consistent():
    """On a disconnected graph both public paths return the same value."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    _flat_graph(G)
    xi_tel = compute_unified_telemetry(G)["canonical"]["xi_c"]
    xi_kernel = estimate_coherence_length(G)
    both_nan = np.isnan(float(xi_tel)) and np.isnan(float(xi_kernel))
    assert both_nan or abs(float(xi_tel) - float(xi_kernel)) < 1e-9
