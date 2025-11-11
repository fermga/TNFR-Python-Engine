"""ΔNFR generator construction utilities."""

from __future__ import annotations

from typing import Final, Sequence

import numpy as np
from numpy.random import Generator

from .backend import ensure_array, ensure_numpy, get_backend

__all__ = ["build_delta_nfr", "build_lindblad_delta_nfr"]

_TOPOLOGIES: Final[set[str]] = {"laplacian", "adjacency"}


def _ring_adjacency(dim: int) -> np.ndarray:
    """Return the adjacency matrix for a coherent ring topology."""

    adjacency: np.ndarray = np.zeros((dim, dim), dtype=float)
    if dim == 1:
        return adjacency

    indices = np.arange(dim)
    adjacency[indices, (indices + 1) % dim] = 1.0
    adjacency[(indices + 1) % dim, indices] = 1.0
    return adjacency


def _laplacian_from_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """Construct a Laplacian operator from an adjacency matrix."""

    degrees = adjacency.sum(axis=1)
    laplacian = np.diag(degrees) - adjacency
    return laplacian


def _hermitian_noise(dim: int, rng: Generator) -> np.ndarray:
    """Generate a Hermitian noise matrix with reproducible statistics."""

    real = rng.standard_normal((dim, dim))
    imag = rng.standard_normal((dim, dim))
    noise = real + 1j * imag
    return 0.5 * (noise + noise.conj().T)


def _as_square_matrix(
    matrix: Sequence[Sequence[complex]] | np.ndarray,
    *,
    expected_dim: int | None = None,
    label: str = "matrix",
) -> np.ndarray:
    """Return ``matrix`` as a square :class:`numpy.ndarray` with validation."""

    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{label} must be a square matrix.")
    if expected_dim is not None and array.shape[0] != expected_dim:
        raise ValueError(
            f"{label} dimension mismatch: expected {expected_dim}, received {array.shape[0]}."
        )
    return array


def build_delta_nfr(
    dim: int,
    *,
    topology: str = "laplacian",
    nu_f: float = 1.0,
    scale: float = 1.0,
    rng: Generator | None = None,
) -> np.ndarray:
    """Construct a Hermitian ΔNFR generator using canonical TNFR topologies.

    Parameters
    ----------
    dim:
        Dimensionality of the Hilbert space supporting the ΔNFR operator.
    topology:
        Requested canonical topology. Supported values are ``"laplacian"``
        and ``"adjacency"``.
    nu_f:
        Structural frequency scaling applied to the resulting operator.
    scale:
        Additional scaling applied uniformly to the operator amplitude.
    rng:
        Optional NumPy :class:`~numpy.random.Generator` used to inject
        reproducible Hermitian noise.
    """

    if dim <= 0:
        raise ValueError("ΔNFR generators require a positive dimensionality.")

    if topology not in _TOPOLOGIES:
        allowed = ", ".join(sorted(_TOPOLOGIES))
        raise ValueError(f"Unknown ΔNFR topology: {topology}. Expected one of: {allowed}.")

    adjacency = _ring_adjacency(dim)
    if topology == "laplacian":
        base = _laplacian_from_adjacency(adjacency)
    else:
        base = adjacency

    matrix: np.ndarray = base.astype(np.complex128, copy=False)

    if rng is not None:
        noise = _hermitian_noise(dim, rng)
        matrix = matrix + (1.0 / np.sqrt(dim)) * noise

    matrix *= nu_f * scale
    hermitian = 0.5 * (matrix + matrix.conj().T)
    backend = get_backend()
    return np.asarray(
        ensure_numpy(ensure_array(hermitian, backend=backend), backend=backend),
        dtype=np.complex128,
    )


def build_lindblad_delta_nfr(
    *,
    hamiltonian: Sequence[Sequence[complex]] | np.ndarray | None = None,
    collapse_operators: Sequence[Sequence[Sequence[complex]] | np.ndarray] | None = None,
    dim: int | None = None,
    nu_f: float = 1.0,
    scale: float = 1.0,
    ensure_trace_preserving: bool = True,
    ensure_contractive: bool = True,
    atol: float = 1e-9,
) -> np.ndarray:
    """Construct a Lindblad ΔNFR generator in Liouville space.

    The resulting matrix acts on vectorised density operators using the
    canonical column-major flattening.  The construction follows the standard
    Gorini–Kossakowski–Sudarshan–Lindblad prescription while exposing TNFR
    semantics through ``ν_f`` and ``scale``.

    Parameters
    ----------
    hamiltonian:
        Optional coherent component.  When ``None`` a null Hamiltonian is
        assumed.
    collapse_operators:
        Iterable with the dissipative operators driving the contractive
        semigroup.  Each entry must be square with the same dimension as the
        Hamiltonian.  When ``None`` the generator reduces to the coherent part.
    dim:
        Explicit Hilbert-space dimension.  Only required if neither
        ``hamiltonian`` nor ``collapse_operators`` are provided.  When supplied,
        it must match the dimension inferred from the Hamiltonian and collapse
        operators.
    nu_f, scale:
        Structural frequency scaling applied uniformly to the final generator.
    ensure_trace_preserving:
        When ``True`` (default) the resulting superoperator is validated to
        leave the identity invariant.
    ensure_contractive:
        When ``True`` (default) the spectrum is required to have non-positive
        real parts within ``atol``.
    atol:
        Absolute tolerance used for Hermiticity, trace and spectral checks.
    """

    operators = list(collapse_operators or [])

    inferred_dim: int | None = dim
    if hamiltonian is not None:
        hermitian = _as_square_matrix(hamiltonian, label="hamiltonian")
        inferred_dim = hermitian.shape[0]
    elif operators:
        inferred_dim = _as_square_matrix(operators[0], label="collapse operator[0]").shape[0]

    if inferred_dim is None:
        raise ValueError("dim must be supplied when no operators are provided.")

    if inferred_dim <= 0:
        raise ValueError("ΔNFR generators require a positive dimension.")

    dimension = inferred_dim

    if dim is not None and dim != dimension:
        raise ValueError(
            "Provided dim is inconsistent with the supplied operators: "
            f"expected {dimension}, received {dim}."
        )

    if hamiltonian is None:
        hermitian = np.zeros((dimension, dimension), dtype=np.complex128)
    else:
        hermitian = _as_square_matrix(hamiltonian, expected_dim=dimension, label="hamiltonian")
        if not np.allclose(hermitian, hermitian.conj().T, atol=atol):
            raise ValueError("Hamiltonian component must be Hermitian within tolerance.")

    dissipators = [
        _as_square_matrix(operator, expected_dim=dimension, label=f"collapse operator[{index}]")
        for index, operator in enumerate(operators)
    ]

    identity = np.eye(dimension, dtype=np.complex128)
    liouvillian = -1j * (np.kron(identity, hermitian) - np.kron(hermitian.T, identity))

    for operator in dissipators:
        adjoint_product = operator.conj().T @ operator
        liouvillian += np.kron(operator.conj(), operator)
        liouvillian -= 0.5 * np.kron(identity, adjoint_product)
        liouvillian -= 0.5 * np.kron(adjoint_product.T, identity)

    liouvillian *= nu_f * scale

    if ensure_trace_preserving:
        identity_vec = identity.reshape(dimension * dimension, order="F")
        left_residual = identity_vec.conj().T @ liouvillian
        if not np.allclose(left_residual, np.zeros_like(left_residual), atol=10 * atol):
            raise ValueError("Lindblad generator must preserve the trace of density operators.")

    backend = get_backend()
    liouvillian_backend = ensure_array(liouvillian, backend=backend)

    if ensure_contractive:
        eigenvalues_backend, _ = backend.eig(liouvillian_backend)
        eigenvalues = ensure_numpy(eigenvalues_backend, backend=backend)
        if np.max(eigenvalues.real) > atol:
            raise ValueError(
                "Lindblad generator is not contractive: spectrum has positive real components."
            )

    return np.asarray(ensure_numpy(liouvillian_backend, backend=backend), dtype=np.complex128)
