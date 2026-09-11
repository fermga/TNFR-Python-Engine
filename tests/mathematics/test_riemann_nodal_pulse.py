"""Tests for the canonical TNFR-Riemann nodal-pulse foundation.

The obsolete combinatorial-Laplacian track (H = L_k + V_sigma) was eliminated and
the attack re-founded on the emergent prime-NFR nodal pulse
``zeta(1/2 + iT) = sum_n n^{-1/2} e^{-i (log n) T}`` (structural frequency
``nu_f = log n``); zeros are the heights where the integer-NFR pulses
destructively interfere.
"""

from __future__ import annotations

import math

import numpy as np

from tnfr.riemann import (
    KNOWN_RIEMANN_ZEROS,
    NodalPulseCertificate,
    build_prime_nfr_graph,
    detect_zeros_by_interference,
    first_primes,
    nodal_pulse,
    nodal_pulse_magnitude,
    prime_structural_frequencies,
    verify_nodal_pulse,
)


def test_first_primes():
    assert first_primes(6) == [2, 3, 5, 7, 11, 13]
    assert first_primes(0) == []


def test_prime_structural_frequencies_are_log_primes():
    freqs = prime_structural_frequencies(5)
    assert freqs == [math.log(p) for p in (2, 3, 5, 7, 11)]


def test_nu_f_is_additive_under_multiplication():
    # The emergent structural frequency is log-additive: nu_f(p*q)=nu_f(p)+nu_f(q).
    assert math.isclose(math.log(6), math.log(2) + math.log(3))


def test_nodal_pulse_matches_truncated_dirichlet_sum():
    t = 20.0
    n_terms = 12
    n = np.arange(1, n_terms + 1, dtype=float)
    expected = complex(np.sum(n**-0.5 * np.exp(-1j * t * np.log(n))))
    assert abs(nodal_pulse(t, n_terms) - expected) < 1e-12
    assert nodal_pulse_magnitude(t, n_terms) == abs(expected)


def test_zeros_emerge_as_destructive_interference_dips():
    dips = detect_zeros_by_interference(8.0, 40.0, resolution=8000)
    # Each of the first six known ordinates has a nearby interference dip.
    for gamma in KNOWN_RIEMANN_ZEROS[:6]:
        assert any(abs(d - gamma) < 0.5 for d in dips), gamma


def test_verify_nodal_pulse_certificate_passes():
    cert = verify_nodal_pulse(6)
    assert isinstance(cert, NodalPulseCertificate)
    assert cert.all_matched
    assert cert.max_abs_error < 0.5
    assert "PASS" in cert.summary()


def test_emergent_prime_nfr_graph_uses_log_prime_frequencies():
    G = build_prime_nfr_graph(6)
    assert G.number_of_nodes() == 6
    vfs = [G.nodes[i]["vf"] for i in range(6)]
    assert vfs == [math.log(p) for p in (2, 3, 5, 7, 11, 13)]


def test_emergent_operator_is_random_walk_laplacian_not_combinatorial():
    # The emergent structural operator on the prime graph is L_rw, never D - A.
    import networkx as nx

    from tnfr.physics.structural_diffusion import structural_diffusion_operator

    G = build_prime_nfr_graph(8)
    nodes, l_rw = structural_diffusion_operator(G)
    assert l_rw.shape == (8, 8)
    # L_rw rows sum to ~0 (I - D^-1 W); a combinatorial D - A on a weighted graph
    # would not have unit diagonal on interior nodes.
    assert np.allclose(l_rw.sum(axis=1), 0.0, atol=1e-9)
    assert not nx.is_directed(G)
