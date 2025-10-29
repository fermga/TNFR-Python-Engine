"""ΔNFR generator construction utilities."""
from __future__ import annotations

from typing import Final

import numpy as np
from numpy.random import Generator

__all__ = ["build_delta_nfr"]

_TOPOLOGIES: Final[set[str]] = {"laplacian", "adjacency"}


def _ring_adjacency(dim: int) -> np.ndarray:
    """Return the adjacency matrix for a coherent ring topology."""

    adjacency = np.zeros((dim, dim), dtype=float)
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

    matrix = base.astype(np.complex128, copy=False)

    if rng is not None:
        noise = _hermitian_noise(dim, rng)
        matrix = matrix + (1.0 / np.sqrt(dim)) * noise

    matrix *= (nu_f * scale)
    hermitian = 0.5 * (matrix + matrix.conj().T)
    return np.asarray(hermitian, dtype=np.complex128)
