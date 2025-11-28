"""Liouvillian spectrum computation and analysis for TNFR dynamics.

This module provides utilities to compute, store, and retrieve Liouvillian
eigenvalue spectra from TNFR graphs. The spectrum is critical for U6 temporal
ordering analysis, particularly for extracting the slow relaxation mode.

Key Functions
-------------
- compute_liouvillian_spectrum: Compute eigenvalues from Lindblad generator
- store_liouvillian_spectrum: Store spectrum in graph metadata
- get_liouvillian_spectrum: Retrieve cached spectrum from graph
- get_slow_relaxation_mode: Extract slowest decay eigenvalue

Theoretical Background
----------------------
The Liouvillian superoperator L governs density matrix evolution:

    dρ/dt = L[ρ]

For Lindblad form:

    L[ρ] = -i[H, ρ] + Σ_k (L_k ρ L_k† - 1/2{L_k†L_k, ρ})

Eigenvalue spectrum properties:
- All eigenvalues have Re(λ) ≤ 0 (contractivity)
- λ = 0 corresponds to steady state
- Smallest |Re(λ)| > 0 is the slow relaxation mode
- τ_relax = 1/|Re(λ_slow)| is the relaxation timescale

See Also
--------
tnfr.mathematics.generators.build_lindblad_delta_nfr : Lindblad generator construction
tnfr.operators.metrics_u6.measure_tau_relax_observed : U6 relaxation time telemetry
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .backend import ensure_array, ensure_numpy, get_backend

__all__ = [
    "compute_liouvillian_spectrum",
    "store_liouvillian_spectrum",
    "get_liouvillian_spectrum",
    "get_slow_relaxation_mode",
]


def compute_liouvillian_spectrum(
    liouvillian: np.ndarray | Sequence[Sequence[complex]],
    *,
    sort: bool = True,
    validate_contractivity: bool = True,
    atol: float = 1e-9,
) -> np.ndarray:
    """Compute eigenvalue spectrum of a Liouvillian superoperator.

    Parameters
    ----------
    liouvillian : array_like
        Liouvillian matrix in vectorized density operator basis (dim² × dim²).
        Expected to be the output of `build_lindblad_delta_nfr()`.
    sort : bool, default=True
        Sort eigenvalues by ascending real part (most negative first).
        Useful for identifying slow relaxation modes.
    validate_contractivity : bool, default=True
        Verify all eigenvalues have Re(λ) ≤ atol (contractivity requirement).
        Raises ValueError if violated.
    atol : float, default=1e-9
        Absolute tolerance for contractivity validation.

    Returns
    -------
    eigenvalues : np.ndarray
        Complex eigenvalues of the Liouvillian, shape (dim²,).
        If sorted, ordered by ascending real part.

    Raises
    ------
    ValueError
        If `validate_contractivity=True` and spectrum contains positive
        real eigenvalues beyond tolerance.

    Examples
    --------
    >>> from tnfr.mathematics.generators import build_lindblad_delta_nfr
    >>> from tnfr.mathematics.liouville import compute_liouvillian_spectrum
    >>>
    >>> # Construct Lindblad generator for 2-level system
    >>> H = np.array([[1, 0], [0, -1]])
    >>> L1 = np.array([[0, 1], [0, 0]])  # decay operator
    >>> liouv = build_lindblad_delta_nfr(hamiltonian=H, collapse_operators=[L1])
    >>>
    >>> # Compute spectrum
    >>> eigs = compute_liouvillian_spectrum(liouv)
    >>> print(f"Eigenvalues: {eigs}")
    >>> print(f"All Re(λ) ≤ 0: {np.all(eigs.real <= 1e-9)}")

    Notes
    -----
    **Computational Notes**:

    - Uses backend-specific eigenvalue solver (NumPy/JAX/PyTorch)
    - For large systems (dim > 10), consider sparse solvers
    - Eigenvalue ordering is by real part to facilitate slow-mode extraction

    **Physical Interpretation**:

    - λ = 0: Steady-state eigenvalue (always present for trace-preserving L)
    - Re(λ) < 0: Decay modes with rate |Re(λ)|
    - λ_slow = min{|Re(λ)| : Re(λ) < 0}: Slowest relaxation mode
    - τ_relax = 1/|Re(λ_slow)|: Characteristic relaxation timescale

    See Also
    --------
    get_slow_relaxation_mode : Extract slowest decay eigenvalue
    build_lindblad_delta_nfr : Construct Liouvillian from Hamiltonian and collapse ops
    """
    backend = get_backend()
    liouv_array = ensure_array(np.asarray(liouvillian, dtype=np.complex128), backend=backend)

    # Compute eigenvalues using backend-specific solver
    eigenvalues_backend, _ = backend.eig(liouv_array)
    eigenvalues = ensure_numpy(eigenvalues_backend, backend=backend)

    if validate_contractivity:
        max_real = np.max(eigenvalues.real)
        if max_real > atol:
            raise ValueError(
                f"Liouvillian spectrum violates contractivity: "
                f"max(Re(λ)) = {max_real:.3e} > {atol:.3e}"
            )

    if sort:
        # Sort by ascending real part (most negative first)
        eigenvalues = eigenvalues[np.argsort(eigenvalues.real)]

    return eigenvalues


def store_liouvillian_spectrum(
    G: Any,
    eigenvalues: np.ndarray | Sequence[complex],
    *,
    key: str = "LIOUVILLIAN_EIGS",
) -> None:
    """Store Liouvillian eigenvalue spectrum in graph metadata.

    Parameters
    ----------
    G : Graph-like
        Graph with `.graph` metadata dictionary attribute.
    eigenvalues : array_like
        Complex eigenvalues to store.
    key : str, default="LIOUVILLIAN_EIGS"
        Metadata key for storage. Use consistent key across codebase.

    Examples
    --------
    >>> import networkx as nx
    >>> from tnfr.mathematics.liouville import store_liouvillian_spectrum
    >>>
    >>> G = nx.Graph()
    >>> eigs = np.array([0.0+0j, -1.2+0.3j, -3.5-0.1j])
    >>> store_liouvillian_spectrum(G, eigs)
    >>> assert "LIOUVILLIAN_EIGS" in G.graph

    Notes
    -----
    Eigenvalues are converted to a plain Python list for JSON serialization
    compatibility. Complex numbers are preserved.
    """
    G.graph[key] = [complex(z) for z in eigenvalues]


def get_liouvillian_spectrum(
    G: Any,
    *,
    key: str = "LIOUVILLIAN_EIGS",
    default: Any = None,
) -> np.ndarray | None:
    """Retrieve cached Liouvillian spectrum from graph metadata.

    Parameters
    ----------
    G : Graph-like
        Graph with `.graph` metadata dictionary attribute.
    key : str, default="LIOUVILLIAN_EIGS"
        Metadata key to retrieve.
    default : Any, default=None
        Fallback value if key not found.

    Returns
    -------
    eigenvalues : np.ndarray | None
        Complex eigenvalues array, or `default` if not found.

    Examples
    --------
    >>> from tnfr.mathematics.liouville import get_liouvillian_spectrum
    >>>
    >>> eigs = get_liouvillian_spectrum(G)
    >>> if eigs is not None:
    ...     print(f"Cached spectrum: {eigs}")
    ... else:
    ...     print("No cached spectrum; compute on demand")
    """
    cached = G.graph.get(key, default)
    if cached is None:
        return default
    return np.asarray(cached, dtype=np.complex128)


def get_slow_relaxation_mode(
    eigenvalues: np.ndarray | Sequence[complex],
    *,
    tolerance: float = 1e-12,
) -> complex | None:
    """Extract the slowest relaxation eigenvalue from Liouvillian spectrum.

    The slow mode is the eigenvalue with the smallest magnitude negative
    real part (closest to zero without being zero).

    Parameters
    ----------
    eigenvalues : array_like
        Complex eigenvalues from Liouvillian spectrum.
    tolerance : float, default=1e-12
        Threshold for excluding near-zero eigenvalues (steady state).

    Returns
    -------
    lambda_slow : complex | None
        Slowest relaxation eigenvalue, or None if no valid modes found.

    Examples
    --------
    >>> eigs = np.array([0.0+0j, -0.1+0.05j, -2.3-0.1j, -5.0+0j])
    >>> slow = get_slow_relaxation_mode(eigs)
    >>> print(f"Slow mode: λ = {slow}")
    >>> print(f"Relaxation time: τ = {1.0/abs(slow.real):.2f}")

    Notes
    -----
    **Selection Criteria**:

    - Excludes eigenvalues with |Re(λ)| < tolerance (steady states)
    - Selects eigenvalue with min(|Re(λ)|) among remaining
    - Returns None if no valid eigenvalues found

    **Physical Significance**:

    The slow relaxation mode determines the longest timescale for the
    system to approach steady state after perturbation. It's critical
    for U6 temporal ordering validation.

    See Also
    --------
    compute_liouvillian_spectrum : Compute full spectrum
    """
    eigs = np.asarray(eigenvalues, dtype=np.complex128)

    # Filter eigenvalues: exclude near-zero (steady state)
    valid_mask = np.abs(eigs.real) > tolerance
    valid_eigs = eigs[valid_mask]

    if len(valid_eigs) == 0:
        return None

    # Find eigenvalue with smallest magnitude negative real part
    # (closest to zero while still being a decay mode)
    real_parts = valid_eigs.real
    negative_mask = real_parts < 0
    negative_eigs = valid_eigs[negative_mask]

    if len(negative_eigs) == 0:
        return None

    # Slow mode = least negative (closest to zero)
    slow_idx = np.argmax(negative_eigs.real)  # max of negatives = least negative
    return complex(negative_eigs[slow_idx])
