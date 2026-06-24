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

from typing import Any, Callable

import scipy.linalg
import scipy.sparse.linalg

from ..errors import TNFRValueError
from .unified_numerical import np

try:
    import networkx as nx
except ImportError:
    nx = None

# Import GPU-aware mathematics backend
try:
    from .backend import get_backend

    HAS_GPU_BACKENDS = True
except ImportError:
    HAS_GPU_BACKENDS = False

from .unified_cache import CacheLevel, cache_tnfr_computation


@cache_tnfr_computation(
    level=CacheLevel.GRAPH_STRUCTURE, dependencies={"graph_topology"}
)
def get_laplacian_spectrum(
    G: Any, weight: str | None = "weight", k: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute and cache the Laplacian spectrum of the graph.

    Args:
        G: The graph (NetworkX or compatible).
        weight: Edge attribute to use as weight.
        k: Number of eigenvalues/vectors to compute (for sparse/large graphs).
           If None, computes full spectrum.

    Returns:
        tuple (eigenvalues, eigenvectors).
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
        # Full diagonalization with GPU backend support
        L_dense = L.toarray()

        # Use GPU backend if available and beneficial
        if HAS_GPU_BACKENDS and N > 100:  # GPU beneficial for larger matrices
            try:
                backend = get_backend()
                if backend.supports_autodiff and hasattr(backend, "eigh"):
                    # Convert to backend format
                    L_tensor = backend.as_array(L_dense)

                    if nx.is_directed(G):
                        # Use general eigenvalue solver for directed graphs
                        evals_tensor, evecs_tensor = backend.eig(L_tensor)
                        # Convert back to numpy and sort
                        evals = backend.to_numpy(evals_tensor)
                        evecs = backend.to_numpy(evecs_tensor)
                        idx = np.argsort(np.real(evals))
                        evals = evals[idx]
                        evecs = evecs[:, idx]
                    else:
                        # Use Hermitian solver for undirected graphs
                        evals_tensor, evecs_tensor = backend.eigh(L_tensor)
                        evals = backend.to_numpy(evals_tensor)
                        evecs = backend.to_numpy(evecs_tensor)
                else:
                    raise TNFRValueError(
                        "Backend doesn't support eigendecomposition",
                        context={"backend": backend.name},
                        suggestion="Use a backend that supports eigendecomposition (e.g., numpy, torch, jax).",
                    )
            except Exception:
                # Fallback to CPU implementation
                if nx.is_directed(G):
                    evals, evecs = scipy.linalg.eig(L_dense)
                    idx = np.argsort(np.real(evals))
                    evals = evals[idx]
                    evecs = evecs[:, idx]
                else:
                    evals, evecs = scipy.linalg.eigh(L_dense)
        else:
            # CPU implementation
            if nx.is_directed(G):
                evals, evecs = scipy.linalg.eig(L_dense)
                idx = np.argsort(np.real(evals))
                evals = evals[idx]
                evecs = evecs[:, idx]
            else:
                evals, evecs = scipy.linalg.eigh(L_dense)
    else:
        # Sparse partial diagonalization
        # 'SM' = Smallest Magnitude (eigenvalues near 0)
        # Note: eigsh finds largest by default, so we use 'SA' (Smallest Algebraic)
        # or shift-invert mode.
        evals, evecs = scipy.sparse.linalg.eigsh(L, k=k, which="SM")

    return evals, evecs


def gft(signal: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Compute the Graph Fourier Transform of a signal.

    Args:
        signal: Node signal array of shape (N,).
        U: Eigenvectors matrix of shape (N, N) (columns are eigenvectors).

    Returns:
        Spectral coefficients (hat_signal) of shape (N,).
    """
    # Use GPU backend for large matrices
    if HAS_GPU_BACKENDS and U.shape[0] > 100:
        try:
            backend = get_backend()
            if backend.supports_autodiff:
                U_tensor = backend.as_array(U)
                signal_tensor = backend.as_array(signal)
                result_tensor = backend.matmul(
                    backend.conjugate_transpose(U_tensor), signal_tensor
                )
                return backend.to_numpy(result_tensor)
        except Exception:
            pass  # Fallback to CPU

    # CPU implementation: GFT is projection onto eigenvectors: \hat{f} = U^T f
    return U.T @ signal


def igft(hat_signal: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Compute the Inverse Graph Fourier Transform.

    Args:
        hat_signal: Spectral coefficients of shape (N,).
        U: Eigenvectors matrix of shape (N, N).

    Returns:
        Reconstructed signal of shape (N,).
    """
    # Use GPU backend for large matrices
    if HAS_GPU_BACKENDS and U.shape[0] > 100:
        try:
            backend = get_backend()
            if backend.supports_autodiff:
                U_tensor = backend.as_array(U)
                hat_signal_tensor = backend.as_array(hat_signal)
                result_tensor = backend.matmul(U_tensor, hat_signal_tensor)
                return backend.to_numpy(result_tensor)
        except Exception:
            pass  # Fallback to CPU

    # CPU implementation: IGFT is reconstruction: f = U \hat{f}
    return U @ hat_signal


def spectral_filter(
    signal: np.ndarray,
    U: np.ndarray,
    evals: np.ndarray,
    filter_func: Callable[[np.ndarray], np.ndarray],
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
    signal: np.ndarray, U: np.ndarray, evals: np.ndarray, t: float
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
