"""Spectral Graph Theory utilities for TNFR.

This module implements the "FFT arithmetic" of TNFR dynamics by providing
tools for Spectral Graph Theory. It leverages the repository's caching
infrastructure to store expensive spectral decompositions (eigenvalues/vectors).

The Graph Fourier Transform (GFT) allows analyzing EPI and ΔNFR signals in the
frequency domain of the network structure, which is natural for the resonant
dynamics of TNFR.

Key Features:
- Cached Laplacian diagonalization
- Graph Fourier Transform (GFT) and Inverse GFT
- Spectral filtering and convolution
- Heat kernel diffusion
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np
import scipy.linalg
import scipy.sparse.linalg

try:
    import networkx as nx
except ImportError:
    nx = None

from ..utils.cache import cache_tnfr_computation, CacheLevel


@cache_tnfr_computation(
    level=CacheLevel.GRAPH_STRUCTURE,
    dependencies={'graph_topology'}
)
def get_laplacian_spectrum(
    G: Any,
    weight: Optional[str] = "weight",
    k: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute and cache the Laplacian spectrum of the graph.
    
    Args:
        G: The graph (NetworkX or compatible).
        weight: Edge attribute to use as weight.
        k: Number of eigenvalues/vectors to compute (for sparse/large graphs).
           If None, computes full spectrum.
           
    Returns:
        Tuple (eigenvalues, eigenvectors).
        eigenvalues: Array of shape (N,) sorted ascending.
        eigenvectors: Array of shape (N, N) or (N, k), where column i is the eigenvector for eval i.
    """
    if nx is None:
        raise ImportError("NetworkX is required for spectral analysis.")
        
    # Get Laplacian Matrix
    # normalized=True is often better for spectral clustering, but for
    # physical diffusion (heat equation), the combinatorial or random-walk
    # Laplacian is often used. TNFR usually assumes standard Laplacian L = D - A.
    L = nx.laplacian_matrix(G, weight=weight)
    
    # Convert to dense if small enough or if full spectrum requested
    N = L.shape[0]
    
    if k is None or k >= N - 1:
        # Full diagonalization
        # L is symmetric (for undirected graphs).
        # TODO: Handle directed graphs (using singular values or non-symmetric eigensolvers)
        if nx.is_directed(G):
            # For directed graphs, we might use the symmetrized Laplacian
            # or just the standard one (which has complex eigenvalues).
            # For now, we assume undirected or symmetrized for GFT basis.
            L_dense = L.toarray()
            evals, evecs = scipy.linalg.eig(L_dense)
            # Sort by real part
            idx = np.argsort(np.real(evals))
            evals = evals[idx]
            evecs = evecs[:, idx]
        else:
            L_dense = L.toarray()
            evals, evecs = scipy.linalg.eigh(L_dense)
    else:
        # Sparse partial diagonalization
        # 'SM' = Smallest Magnitude (eigenvalues near 0)
        # Note: eigsh finds largest by default, so we use 'SA' (Smallest Algebraic)
        # or shift-invert mode.
        evals, evecs = scipy.sparse.linalg.eigsh(L, k=k, which='SM')

    return evals, evecs


def gft(signal: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Compute the Graph Fourier Transform of a signal.

    Args:
        signal: Node signal array of shape (N,).
        U: Eigenvectors matrix of shape (N, N) (columns are eigenvectors).

    Returns:
        Spectral coefficients (hat_signal) of shape (N,).
    """
    # GFT is projection onto eigenvectors: \hat{f} = U^T f
    return U.T @ signal


def igft(hat_signal: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Compute the Inverse Graph Fourier Transform.

    Args:
        hat_signal: Spectral coefficients of shape (N,).
        U: Eigenvectors matrix of shape (N, N).

    Returns:
        Reconstructed signal of shape (N,).
    """
    # IGFT is reconstruction: f = U \hat{f}
    return U @ hat_signal


def spectral_filter(
    signal: np.ndarray,
    U: np.ndarray,
    evals: np.ndarray,
    filter_func: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Apply a spectral filter to a signal.

    Args:
        signal: Input signal (N,).
        U: Eigenvectors (N, N).
        evals: Eigenvalues (N,).
        filter_func: Function taking eigenvalues and returning filter coefficients.

    Returns:
        Filtered signal.
    """
    # 1. GFT
    hat_f = gft(signal, U)

    # 2. Apply filter
    h = filter_func(evals)
    hat_f_filtered = hat_f * h

    # 3. IGFT
    return igft(hat_f_filtered, U)


def heat_diffusion(
    signal: np.ndarray,
    U: np.ndarray,
    evals: np.ndarray,
    t: float
) -> np.ndarray:
    """Simulate heat diffusion on the graph for time t.

    Solves ∂f/∂t = -L f.
    Solution: f(t) = U exp(-Λt) U^T f(0).

    Args:
        signal: Initial state f(0).
        U: Eigenvectors.
        evals: Eigenvalues.
        t: Time parameter.

    Returns:
        Diffused signal f(t).
    """
    return spectral_filter(signal, U, evals, lambda lam: np.exp(-lam * t))


def compute_spectral_smoothness(signal: np.ndarray, L: Any) -> float:
    """Compute the smoothness of a signal on the graph (Dirichlet energy).

    E = f^T L f = Σ (f_i - f_j)^2

    Args:
        signal: Node signal (N,).
        L: Laplacian matrix (or precomputed).

    Returns:
        Scalar smoothness value.
    """
    if scipy.sparse.issparse(L):
        return float(signal.T @ (L @ signal))
    else:
        return float(signal.T @ np.dot(L, signal))
