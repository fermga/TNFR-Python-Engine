"""Regression smoke tests for physics.signatures.

This module was previously broken by a cascade of pre-existing bugs (a cache
decorator called without its required ``level``/``dependencies`` arguments, an
invalid ``coherence_key`` kwarg passed to ``estimate_coherence_length``, and a
hard import of the removed ``examples_utils`` helper). These tests ensure the
module imports and both signature functions run, so the cascade cannot silently
reappear.
"""

from __future__ import annotations

import networkx as nx

from tnfr.physics.signatures import compute_au_like_signature, compute_element_signature


def _ring(n: int = 10) -> nx.Graph:
    G = nx.cycle_graph(n)
    for node in G.nodes():
        G.nodes[node]["delta_nfr"] = 0.1
        G.nodes[node]["theta"] = 0.0
        G.nodes[node]["EPI"] = 0.5
        G.nodes[node]["nu_f"] = 1.0
    return G


def test_compute_element_signature_runs():
    sig = compute_element_signature(_ring())
    assert {"xi_c", "signature_class", "phi_s_drift"} <= set(sig)
    assert isinstance(sig["xi_c"], float)


def test_synthetic_step_degrades_gracefully():
    # apply_synthetic_step=True used to raise ModuleNotFoundError
    # (examples_utils was removed); it is now a soft dependency.
    sig = compute_element_signature(_ring(), apply_synthetic_step=True)
    assert isinstance(sig["phi_s_drift"], float)


def test_compute_au_like_signature_runs():
    au = compute_au_like_signature(_ring())
    assert isinstance(au, dict) and au
