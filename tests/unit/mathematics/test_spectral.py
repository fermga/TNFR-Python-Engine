"""Unit tests for Spectral Graph Theory utilities."""

import numpy as np
import pytest

try:
    import networkx as nx
except ImportError:
    nx = None

from tnfr.mathematics.spectral import (
    get_laplacian_spectrum,
    gft,
    igft,
    heat_diffusion,
    compute_spectral_smoothness
)

@pytest.mark.skipif(nx is None, reason="NetworkX required")
def test_laplacian_spectrum_cycle() -> None:
    """Test spectrum of a cycle graph (known analytical solution)."""
    N = 10
    G = nx.cycle_graph(N)
    evals, evecs = get_laplacian_spectrum(G)

    # Smallest eigenvalue should be 0
    assert np.isclose(evals[0], 0.0)

    # Eigenvalues should be non-negative (PSD Laplacian)
    assert np.all(evals >= -1e-10)

    # Check shape
    assert evals.shape == (N,)
    assert evecs.shape == (N, N)


@pytest.mark.skipif(nx is None, reason="NetworkX required")
def test_gft_igft_reconstruction() -> None:
    """Test that IGFT(GFT(signal)) reconstructs the signal."""
    N = 15
    G = nx.path_graph(N)
    evals, evecs = get_laplacian_spectrum(G)

    # Random signal
    np.random.seed(42)
    signal = np.random.randn(N)

    hat_signal = gft(signal, evecs)
    reconstructed = igft(hat_signal, evecs)

    assert np.allclose(signal, reconstructed)


@pytest.mark.skipif(nx is None, reason="NetworkX required")
def test_heat_diffusion_physics() -> None:
    """Test physical properties of heat diffusion."""
    N = 20
    G = nx.path_graph(N)
    evals, evecs = get_laplacian_spectrum(G)

    # Delta at center
    signal = np.zeros(N)
    center = N // 2
    signal[center] = 1.0

    # Diffuse
    t = 1.0
    diffused = heat_diffusion(signal, evecs, evals, t)

    # Conservation of mass (for standard Laplacian on regular graph? No, L=D-A conserves mass sum(f)=const if 1 is eigenvector with eval 0)
    # Yes, 1 is eigenvector for eval 0. exp(-0*t) = 1. So DC component is preserved.
    assert np.isclose(np.sum(signal), np.sum(diffused))

    # Smoothing: Peak should decrease
    assert diffused[center] < signal[center]

    # Spreading: Neighbors should increase
    assert diffused[center - 1] > signal[center - 1]
    assert diffused[center + 1] > signal[center + 1]


@pytest.mark.skipif(nx is None, reason="NetworkX required")
def test_spectral_smoothness() -> None:
    """Test smoothness metric."""
    N = 10
    G = nx.path_graph(N)
    L = nx.laplacian_matrix(G)
    
    # Constant signal -> Smoothness 0
    const_sig = np.ones(N)
    assert np.isclose(compute_spectral_smoothness(const_sig, L), 0.0)
    
    # Alternating signal -> High smoothness value (energy)
    alt_sig = np.array([(-1)**i for i in range(N)])
    smoothness = compute_spectral_smoothness(alt_sig, L)
    assert smoothness > 0.0
