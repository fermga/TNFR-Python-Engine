"""Regression tests for the ΔNFR computation paths.

Guards against a structural-pressure freeze bug in which the default
vectorized fused gradient path silently produced ΔNFR ≡ 0 because the
caller passed bare weight keys (``phase``/``epi``/``vf``/``topo``) while
``compute_fused_gradients_symmetric`` reads ``w_``-prefixed keys
(``w_phase`` ...). A zero ΔNFR violates the nodal equation
``∂EPI/∂t = νf · ΔNFR(t)`` by removing the structural-pressure term, yet
passes ``isfinite``/``isinstance`` assertions — hence these explicit
magnitude and cross-path parity checks.
"""

from __future__ import annotations

import random

import networkx as nx
import numpy as np

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.dynamics.dnfr import default_compute_delta_nfr


def _build_canonical_graph(n: int = 40, seed: int = 3) -> nx.Graph:
    """Watts-Strogatz graph with canonical phase/EPI/νf attributes."""
    rng = random.Random(seed)
    G = nx.watts_strogatz_graph(n, 4, 0.3, seed=seed)
    for node in G.nodes():
        G.nodes[node]["theta"] = rng.uniform(0.0, 2.0 * np.pi)
        G.nodes[node]["EPI"] = rng.uniform(0.2, 0.8)
        G.nodes[node]["nu_f"] = rng.uniform(0.5, 1.5)
    return G


def _dnfr_vector(G: nx.Graph) -> np.ndarray:
    return np.array(
        [get_attr(G.nodes[n], ALIAS_DNFR, 0.0) for n in G.nodes()],
        dtype=float,
    )


class TestDeltaNFRComputationPaths:
    """ΔNFR must be non-zero and path-independent under canonical weights."""

    def test_vectorized_path_produces_nonzero_dnfr(self) -> None:
        """Default (vectorized fused) path must not freeze ΔNFR to zero.

        Regression: bare-key weights zeroed every component, producing
        ΔNFR ≡ 0 on every canonical step.
        """
        G = _build_canonical_graph()
        default_compute_delta_nfr(G)
        dnfr = _dnfr_vector(G)

        assert np.any(dnfr != 0.0), "ΔNFR is identically zero (pressure frozen)"
        assert float(np.mean(np.abs(dnfr))) > 1e-6
        assert np.all(np.isfinite(dnfr))

    def test_vectorized_matches_fallback(self) -> None:
        """Vectorized fused path must equal the non-vectorized fallback.

        Both paths implement the same canonical ΔNFR assembly; they must
        agree to floating-point precision.
        """
        G_vec = _build_canonical_graph()
        default_compute_delta_nfr(G_vec)
        dnfr_vec = _dnfr_vector(G_vec)

        G_fallback = _build_canonical_graph()
        G_fallback.graph["vectorized_dnfr"] = False
        default_compute_delta_nfr(G_fallback)
        dnfr_fallback = _dnfr_vector(G_fallback)

        assert dnfr_vec.shape == dnfr_fallback.shape
        np.testing.assert_allclose(dnfr_vec, dnfr_fallback, rtol=1e-9, atol=1e-12)
        # Both paths must be meaningfully non-zero (not trivially equal at 0).
        assert float(np.mean(np.abs(dnfr_fallback))) > 1e-6
