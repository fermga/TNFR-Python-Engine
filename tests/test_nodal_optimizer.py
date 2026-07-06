"""Regression tests for the nodal-optimizer spectral path.

Guards two canonical properties of
:mod:`tnfr.dynamics.nodal_optimizer`:

1. ``precompute_spectral_basis`` executes without error. Before the
   structural-Laplacian canonicity fix it called
   ``get_laplacian_spectrum(G, normalized=True, cache_key=...)`` with keyword
   arguments the function never accepted, so the whole spectral path raised
   ``TypeError`` and was silently dead / untested.
2. The precomputed spectrum is the canonical EMERGENT operator -- the symmetric
   normalized Laplacian ``L_sym`` (which shares the spectrum of the random-walk
   diffusion operator ``L_rw = I - D^-1 W`` that the canonical ΔNFR realises) --
   not the imposed combinatorial graph Laplacian ``D - A``.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics.nodal_optimizer import (
    HAS_SPECTRAL,
    NodalEquationOptimizer,
    NodalOptimizationState,
)

pytestmark = pytest.mark.skipif(
    not HAS_SPECTRAL, reason="spectral dependencies (scipy) required"
)


def _graph() -> nx.Graph:
    G = nx.karate_club_graph()
    for node in G.nodes():
        G.nodes[node]["EPI"] = float(node % 5) / 5.0
        G.nodes[node]["vf"] = 1.0
        G.nodes[node]["theta"] = 0.0
    return G


def test_precompute_spectral_basis_executes() -> None:
    """The spectral path runs (regression: previously raised TypeError)."""
    opt = NodalEquationOptimizer(enable_cache=False)
    state = opt.precompute_spectral_basis(_graph())

    assert isinstance(state, NodalOptimizationState)
    n = 34  # karate club
    assert state.eigenvalues.shape == (n,)
    assert state.eigenvectors.shape == (n, n)
    assert state.node_index and len(state.node_index) == n


def test_precomputed_spectrum_is_canonical_emergent_operator() -> None:
    """The spectrum equals L_sym (emergent), not combinatorial D - A."""
    from tnfr.physics.structural_diffusion import symmetric_normalized_laplacian

    G = _graph()
    opt = NodalEquationOptimizer(enable_cache=False)
    state = opt.precompute_spectral_basis(G)

    _, l_sym = symmetric_normalized_laplacian(G)
    sym_spectrum = np.sort(np.linalg.eigvalsh(l_sym))
    comb_spectrum = np.sort(
        np.linalg.eigvalsh(nx.laplacian_matrix(G).toarray().astype(float))
    )

    got = np.sort(np.real(state.eigenvalues))
    # Matches the canonical emergent L_sym spectrum ...
    assert np.allclose(got, sym_spectrum, atol=1e-8)
    # ... and is genuinely a different operator from combinatorial D - A.
    assert not np.allclose(got, comb_spectrum, atol=1e-6)


def test_spectral_basis_is_cached_by_topology() -> None:
    """A second call on an unchanged topology reuses the cached state."""
    G = _graph()
    opt = NodalEquationOptimizer(enable_cache=True)
    first = opt.precompute_spectral_basis(G)
    second = opt.precompute_spectral_basis(G)
    assert first is second
