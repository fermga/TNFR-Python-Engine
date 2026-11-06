"""Liouvillian spectrum computation and analysis for TNFR dynamics.

This module computes, stores, and retrieves supplied Liouvillian eigenvalue
spectra. A selected decay rate is auxiliary telemetry in the generator's clock;
it does not independently establish a U6 ordering or physical relaxation time.

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

For an admitted finite-dimensional GKSL generator:
- All eigenvalues have Re(λ) ≤ 0 (spectral non-expansion)
- A zero eigenvalue admits stationary modes; a zero real part alone does not
- Smallest |Re(λ)| among negative real parts selects the slow decay mode
- τ_relax = 1/|Re(λ_slow)| is the selected mode's exponential decay timescale

See Also
--------
tnfr.mathematics.generators.build_lindblad_delta_nfr : Lindblad generator construction
tnfr.operators.metrics_u6.measure_tau_relax_observed : U6 relaxation snapshot telemetry
"""

from __future__ import annotations

from typing import Any, Sequence

from .._exact_time import finite_represented_real
from ..config.parsing import parse_bool
from ._complex_admission import finite_represented_complex
from ._complex_arrays import numpy_complex_array
from .backend import ensure_array, ensure_numpy, get_backend
from .unified_numerical import TNFRValueError, np

__all__ = [
    "compute_liouvillian_spectrum",
    "store_liouvillian_spectrum",
    "get_liouvillian_spectrum",
    "get_slow_relaxation_mode",
]


def _nonnegative_tolerance(value: Any, label: str) -> float:
    resolved, _ = finite_represented_real(value, label)
    if resolved < 0:
        raise TNFRValueError(f"{label} must be nonnegative")
    return resolved


def _finite_complex_array(values: Any, label: str) -> np.ndarray:
    """Validate original channels before materializing a detached complex array."""
    if type(values) is np.ndarray and values.dtype.kind in "iufc":
        try:
            admitted_array = numpy_complex_array(values, label=label)
        except TNFRValueError:
            # Preserve the scalar reader's exception class and first bad index.
            pass
        else:
            array = np.array(admitted_array, dtype=np.complex128, copy=True, order="C")
            # The scalar reader canonicalizes both represented zero channels.
            array.real[array.real == 0] = 0.0
            array.imag[array.imag == 0] = 0.0
            return array
    original = np.asarray(values, dtype=object)
    admitted = [
        finite_represented_complex(value, f"{label}[{index}]")
        for index, value in enumerate(original.flat)
    ]
    return np.array(admitted, dtype=np.complex128).reshape(original.shape)


def _spectrum_array(values: Any) -> np.ndarray:
    spectrum = _finite_complex_array(values, "eigenvalues")
    if spectrum.ndim != 1:
        raise TNFRValueError("eigenvalues must be a one-dimensional spectrum")
    return spectrum


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
        Finite nonnegative represented tolerance for the spectral sign check.

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

    The validation flag checks eigenvalue real parts only. It does not prove
    a Lindblad representation or contraction in an arbitrary norm. Rates and
    timescales retain the supplied generator's clock.

    **Physical Interpretation**:

    - λ = 0: Steady-state eigenvalue (always present for trace-preserving L)
    - Re(λ) < 0: Decay modes with rate |Re(λ)|
    - λ_slow: Eigenvalue with negative real part closest to zero, subject to
      the selector's strict tolerance cutoff
    - τ_relax = 1/|Re(λ_slow)|: Selected mode's exponential decay timescale

    See Also
    --------
    get_slow_relaxation_mode : Extract slowest decay eigenvalue
    build_lindblad_delta_nfr : Construct Liouvillian from Hamiltonian and collapse ops
    """
    atol = _nonnegative_tolerance(atol, "atol")
    sort = parse_bool(sort)
    validate_contractivity = parse_bool(validate_contractivity)
    matrix = _finite_complex_array(liouvillian, "liouvillian")
    if matrix.ndim != 2 or not matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        raise TNFRValueError("liouvillian must be a nonempty square matrix")
    backend = get_backend()
    liouv_array = ensure_array(matrix, backend=backend)

    # Compute eigenvalues using backend-specific solver
    eigenvalues_backend, _ = backend.eig(liouv_array)
    eigenvalues = _spectrum_array(ensure_numpy(eigenvalues_backend, backend=backend))

    if validate_contractivity:
        max_real = np.max(eigenvalues.real)
        if max_real > atol:
            raise TNFRValueError(
                "Liouvillian spectrum violates contractivity.",
                context={"max_real_eigenvalue": max_real, "tolerance": atol},
                suggestion="Ensure the Liouvillian represents a valid dissipative process.",
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
    Original real/imaginary components are admitted before metadata is changed.
    Values are stored as a detached list of Python complex scalars; a separate
    complex encoding is required for JSON serialization.
    """
    G.graph[key] = _spectrum_array(eigenvalues).tolist()


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
    return _spectrum_array(cached)


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
        Finite nonnegative represented threshold for excluding near-zero real
        parts. A near-zero real part alone does not establish a steady state.

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

    - Retains eigenvalues with Re(λ) < -tolerance
    - Selects the least negative real part, preserving the first supplied tie
    - Returns None if no valid eigenvalues found

    **Scope**:

    The reciprocal decay rate uses the supplied generator's clock. A selected
    eigenvalue does not prove convergence of every initial state: peripheral
    modes, observability and defective-mode prefactors need separate analysis.
    It supplies neither a physical clock bridge nor a U6 ordering certificate.

    See Also
    --------
    compute_liouvillian_spectrum : Compute full spectrum
    """
    tolerance = _nonnegative_tolerance(tolerance, "tolerance")
    eigs = _spectrum_array(eigenvalues)

    # Finite real parts and a nonnegative tolerance admit one combined filter.
    negative_eigs = eigs[eigs.real < -tolerance]

    if len(negative_eigs) == 0:
        return None

    # Slow mode = least negative (closest to zero)
    slow_idx = np.argmax(negative_eigs.real)  # max of negatives = least negative
    return complex(negative_eigs[slow_idx])
