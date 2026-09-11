"""Spectral graph utilities for TNFR.

This module implements the "FFT arithmetic" of TNFR dynamics by providing
tools for Spectral Graph Theory. It leverages the repository's caching
infrastructure to store expensive spectral decompositions (eigenvalues/vectors).

The Graph Fourier Transform (GFT) expands a node signal in a Laplacian
eigenbasis.  A dense generic-graph eigendecomposition and its matrix transforms
are not a one-dimensional FFT and do not have ``O(N log N)`` complexity.
Diagonalization is normally cubic; projection in an orthonormal basis is
quadratic, while the uncached solve for a general right basis is normally
cubic.  The utilities here also do not turn a modewise product into the
pointwise nodal product ``νf * ΔNFR``.

Key Features:
- Cached Laplacian diagonalization
- Graph Fourier Transform (GFT) and Inverse GFT
- Spectral filtering and convolution
- Heat kernel diffusion
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Callable, Literal

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


def _build_structural_laplacian(
    G: Any, operator: str, weight: str | None
) -> np.ndarray:
    """Build the dense structural Laplacian for the requested operator.

    ``"symmetric"`` and ``"random_walk"`` reuse the canonical builders in
    :mod:`tnfr.physics.structural_diffusion` (the single source of truth for
    L_sym / L_rw); ``"combinatorial"`` returns the generic ``D − A``.
    """
    if operator in ("symmetric", "random_walk"):
        # Lazy import: structural_diffusion is a physics module; importing it at
        # module load would risk an import cycle in this widely-imported utility.
        from ..physics.structural_diffusion import (
            structural_diffusion_operator,
            symmetric_normalized_laplacian,
        )

        if operator == "symmetric":
            _, lap = symmetric_normalized_laplacian(G)
        else:
            _, lap = structural_diffusion_operator(G)
        return np.asarray(lap, dtype=float)
    if operator == "combinatorial":
        return nx.laplacian_matrix(G, weight=weight).toarray().astype(float)
    raise TNFRValueError(
        f"Unknown Laplacian operator: {operator!r}",
        context={"operator": operator},
        suggestion=(
            "Use 'symmetric' (canonical L_sym), 'random_walk' (canonical L_rw), "
            "or 'combinatorial' (generic D - A)."
        ),
    )


@cache_tnfr_computation(
    level=CacheLevel.GRAPH_STRUCTURE, dependencies={"graph_topology"}
)
def _get_laplacian_spectrum_cached(
    G: Any,
    weight: str | None = "weight",
    k: int | None = None,
    operator: Literal["symmetric", "random_walk", "combinatorial"] = "symmetric",
    normalized: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute and cache the structural Laplacian spectrum of the graph.

    TNFR's canonical structural operator is the random-walk Laplacian
    ``L_rw = I - D^{-1} W`` (:mod:`tnfr.physics.structural_diffusion`): the EPI
    channel of the nodal equation is exactly ``dEPI/dt = -vf * L_rw * EPI``, the
    topology-only comparison/fallback for coherence length is
    ``1/sqrt(lambda_2)`` of ``L_rw``, while the primary fitted ``xi_C`` is
    state-dependent. The emergent
    pulse is ``omega_k = sqrt(lambda_k)`` of ``L_rw``.  Its symmetric twin
    ``L_sym = I - D^{-1/2} W D^{-1/2}`` shares that spectrum and has an
    orthonormal eigenbasis, so it is the default GFT chart.  A complete
    ``L_rw`` right basis instead uses its biorthogonal inverse; a partial
    non-orthonormal basis is not a valid transform chart.  The combinatorial
    Laplacian ``D - A`` is a *different*
    operator with a *different* spectrum (generic graph signal processing); it is
    exposed only for explicitly non-structural uses.

    Args:
        G: The graph (NetworkX or compatible).
        weight: Edge attribute to use as weight (``"combinatorial"`` operator only).
        k: Number of eigenvalues/vectors to compute (for sparse/large graphs).
           If None, computes full spectrum.
        operator: Which structural operator to diagonalise -- ``"symmetric"``
           (default, canonical L_sym), ``"random_walk"`` (canonical L_rw;
           non-symmetric) or ``"combinatorial"`` (generic ``D - A``).
        normalized: Backward-compatible convenience alias. ``True`` selects the
           canonical ``"symmetric"`` operator, ``False`` selects
           ``"combinatorial"``; ``None`` (default) defers to ``operator``.

    Returns:
        tuple (eigenvalues, eigenvectors).
        eigenvalues: Array of shape (N,) sorted ascending.
        eigenvectors: Array of shape (N, N) or (N, k), with eigenvectors in
            columns.
    """
    if nx is None:
        raise ImportError("NetworkX is required for spectral analysis.")
    if k is not None and (
        isinstance(k, bool) or not isinstance(k, Integral) or k <= 0
    ):
        raise TNFRValueError(
            "k must be a positive integer or None",
            context={"k": k},
            suggestion="Request at least one mode, or use None for the full spectrum.",
        )

    if k is not None:
        k = int(k)
    if normalized is not None:
        operator = "symmetric" if normalized else "combinatorial"

    # Build the canonical structural Laplacian (dense) for the chosen operator.
    L_dense = _build_structural_laplacian(G, operator, weight)
    N = L_dense.shape[0]
    # L_rw is non-symmetric, and any directed graph yields a non-symmetric
    # matrix -> use the general (non-Hermitian) eigensolver in those cases.
    use_general_eig = bool(nx.is_directed(G)) or operator == "random_walk"

    if k is None or k >= N:
        # Full diagonalization with GPU backend support

        # Use GPU backend if available and beneficial
        if HAS_GPU_BACKENDS and N > 100:  # GPU beneficial for larger matrices
            try:
                backend = get_backend()
                if backend.supports_autodiff and hasattr(backend, "eigh"):
                    # Convert to backend format
                    L_tensor = backend.as_array(L_dense)

                    if use_general_eig:
                        # Use general eigenvalue solver for non-symmetric operators
                        evals_tensor, evecs_tensor = backend.eig(L_tensor)
                        # Convert back to numpy and sort
                        evals = backend.to_numpy(evals_tensor)
                        evecs = backend.to_numpy(evecs_tensor)
                        idx = np.argsort(np.real(evals))
                        evals = evals[idx]
                        evecs = evecs[:, idx]
                    else:
                        # Use Hermitian solver for symmetric operators
                        evals_tensor, evecs_tensor = backend.eigh(L_tensor)
                        evals = backend.to_numpy(evals_tensor)
                        evecs = backend.to_numpy(evecs_tensor)
                else:
                    raise TNFRValueError(
                        "Backend doesn't support eigendecomposition",
                        context={"backend": backend.name},
                        suggestion=(
                            "Use a backend that supports eigendecomposition "
                            "(e.g., numpy, torch, jax)."
                        ),
                    )
            except Exception:
                # Fallback to CPU implementation
                if use_general_eig:
                    evals, evecs = scipy.linalg.eig(L_dense)
                    idx = np.argsort(np.real(evals))
                    evals = evals[idx]
                    evecs = evecs[:, idx]
                else:
                    evals, evecs = scipy.linalg.eigh(L_dense)
        else:
            # CPU implementation
            if use_general_eig:
                evals, evecs = scipy.linalg.eig(L_dense)
                idx = np.argsort(np.real(evals))
                evals = evals[idx]
                evecs = evecs[:, idx]
            else:
                evals, evecs = scipy.linalg.eigh(L_dense)
    else:
        if use_general_eig:
            # ``eigsh`` assumes a real-symmetric/Hermitian matrix.  L_rw and
            # directed Laplacians do not satisfy that contract.  Compute the
            # general spectrum, then retain the modes nearest zero.  This is a
            # dense fallback, not a fast partial transform.
            all_evals, all_evecs = scipy.linalg.eig(L_dense)
            nearest = np.argsort(np.abs(all_evals))[:k]
            order = nearest[
                np.lexsort((np.imag(all_evals[nearest]), np.real(all_evals[nearest])))
            ]
            evals = all_evals[order]
            evecs = all_evecs[:, order]
        else:
            # 'SM' = eigenvalues with the smallest magnitude.
            evals, evecs = scipy.sparse.linalg.eigsh(L_dense, k=k, which="SM")
            order = np.argsort(evals)
            evals = evals[order]
            evecs = evecs[:, order]

    return evals, evecs


def get_laplacian_spectrum(
    G: Any,
    weight: str | None = "weight",
    k: int | None = None,
    operator: Literal["symmetric", "random_walk", "combinatorial"] = "symmetric",
    normalized: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return detached arrays from the cached Laplacian eigensystem."""

    eigenvalues, eigenvectors = _get_laplacian_spectrum_cached(
        G,
        weight=weight,
        k=k,
        operator=operator,
        normalized=normalized,
    )
    return np.array(eigenvalues, copy=True), np.array(eigenvectors, copy=True)


# Retain the complete public documentation on the ownership-safe facade.
get_laplacian_spectrum.__doc__ = _get_laplacian_spectrum_cached.__doc__


def _validate_transform_shapes(
    signal: np.ndarray, U: np.ndarray, *, inverse: bool
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Validate a spectral transform and identify an orthonormal basis."""
    signal = np.asarray(signal)
    U = np.asarray(U)
    if signal.ndim != 1 or U.ndim != 2:
        raise TNFRValueError(
            "Graph spectral transforms require a vector and a rank-2 basis",
            context={"signal_shape": signal.shape, "basis_shape": U.shape},
            suggestion="Pass a one-dimensional signal and a node-by-mode basis.",
        )
    expected = U.shape[1] if inverse else U.shape[0]
    if signal.shape[0] != expected:
        raise TNFRValueError(
            "Signal and spectral basis dimensions are incompatible",
            context={
                "signal_size": signal.shape[0],
                "expected_size": expected,
                "basis_shape": U.shape,
            },
            suggestion=(
                "Use coefficients matching the basis columns and node signals "
                "matching its rows."
            ),
        )
    gram = U.conj().T @ U
    identity = np.eye(U.shape[1], dtype=gram.dtype)
    orthonormal = bool(np.allclose(gram, identity, rtol=1e-10, atol=1e-12))
    if not orthonormal and U.shape[0] != U.shape[1]:
        raise TNFRValueError(
            "A partial graph spectral basis must have orthonormal columns",
            context={"basis_shape": U.shape},
            suggestion=(
                "Use the symmetric Laplacian for a partial GFT, or request the full "
                "right-eigenvector basis for a random-walk Laplacian."
            ),
        )
    if not orthonormal and np.linalg.matrix_rank(U) != U.shape[0]:
        raise TNFRValueError(
            "The full graph spectral basis is singular",
            context={"basis_shape": U.shape},
            suggestion="Use a complete diagonalizable Laplacian basis.",
        )
    return signal, U, orthonormal


def gft(signal: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Expand a node signal in a graph eigenbasis.

    Args:
        signal: Node signal array of shape (N,).
        U: Node-by-mode eigenvector matrix.  Orthonormal columns use the usual
           conjugate projection.  A full non-orthonormal right-eigenvector basis
           (for example from ``L_rw``) uses ``solve(U, signal)`` so that the
           corresponding left basis is respected.

    Returns:
        Spectral coefficients (hat_signal) of shape (N,).
    """
    signal, U, orthonormal = _validate_transform_shapes(signal, U, inverse=False)

    # GPU projection is valid only for an orthonormal basis.  A general right
    # eigenbasis requires a linear solve (equivalently, the biorthogonal left
    # eigenvectors), which stays on the CPU path here.
    if orthonormal and HAS_GPU_BACKENDS and U.shape[0] > 100:
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

    if orthonormal:
        return U.conj().T @ signal
    return scipy.linalg.solve(U, signal, assume_a="gen")


def igft(hat_signal: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Reconstruct a node signal from graph spectral coefficients.

    Args:
        hat_signal: Spectral coefficients of shape (N,).
        U: Eigenvectors matrix of shape (N, N).

    Returns:
        Reconstructed signal of shape (N,).
    """
    hat_signal, U, _ = _validate_transform_shapes(hat_signal, U, inverse=True)

    # Reconstruction is U @ coefficients for both orthonormal and full
    # biorthogonal decompositions.
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
    """Apply a diagonal multiplier in a graph eigenbasis.

    Args:
        signal: Input signal (N,).
        U: Eigenvectors (N, N).
        evals: Eigenvalues (N,).
        filter_func: Function taking eigenvalues and returning filter coefficients.

    Returns:
        Filtered signal.
    """
    evals = np.asarray(evals)
    U = np.asarray(U)
    if evals.ndim != 1 or U.ndim != 2 or len(evals) != U.shape[1]:
        raise TNFRValueError(
            "Eigenvalue and basis dimensions are incompatible",
            context={"eigenvalue_shape": evals.shape, "basis_shape": U.shape},
            suggestion="Provide one eigenvalue for each spectral basis column.",
        )

    # 1. GFT
    hat_f = gft(signal, U)

    # 2. Apply filter
    h = np.asarray(filter_func(evals))
    if h.shape != evals.shape:
        raise TNFRValueError(
            "A spectral filter must return one multiplier per eigenvalue",
            context={"filter_shape": h.shape, "eigenvalue_shape": evals.shape},
            suggestion="Return a vector with the same shape as the eigenvalues.",
        )
    hat_f_filtered = hat_f * h

    # 3. IGFT
    return igft(hat_f_filtered, U)


def heat_diffusion(
    signal: np.ndarray, U: np.ndarray, evals: np.ndarray, t: float
) -> np.ndarray:
    """Evaluate a diagonalizable graph heat semigroup for time ``t``.

    Solves ∂f/∂t = -L f.
    For an orthonormal basis the solution is
    ``U exp(-Λt) Uᴴ f(0)``.  For a full non-orthonormal right basis it is
    ``U exp(-Λt) U⁻¹ f(0)``.  A partial orthonormal basis evaluates the
    corresponding projected/truncated semigroup; at ``t = 0`` it returns the
    projection ``UUᴴf(0)``.  Partial non-orthonormal bases are rejected by
    :func:`gft` because they do not define that inverse.

    Args:
        signal: Initial state f(0).
        U: Eigenvectors.
        evals: Eigenvalues.
        t: Time parameter.

    Returns:
        Diffused signal f(t).
    """
    if isinstance(t, (bool, np.bool_)):
        raise TNFRValueError(
            "Heat-diffusion time must be finite and nonnegative",
            context={"time": t},
            suggestion="Use a finite scalar time greater than or equal to zero.",
        )
    try:
        time = float(t)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TNFRValueError(
            "Heat-diffusion time must be finite and nonnegative",
            context={"time": t},
            suggestion="Use a finite scalar time greater than or equal to zero.",
        ) from exc
    if not np.isfinite(time) or time < 0.0:
        raise TNFRValueError(
            "Heat-diffusion time must be finite and nonnegative",
            context={"time": t},
            suggestion="Use a finite scalar time greater than or equal to zero.",
        )
    result = spectral_filter(signal, U, evals, lambda lam: np.exp(-lam * time))
    return np.real_if_close(result)


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
