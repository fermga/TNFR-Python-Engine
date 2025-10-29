"""Mathematical spaces supporting the TNFR canonical paradigm."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class HilbertSpace:
    """Finite section of :math:`\ell^2(\mathbb{N}) \otimes L^2(\mathbb{R})`.

    The space models the discrete spectral component of the TNFR paradigm.  The
    canonical orthonormal basis corresponds to the standard coordinate vectors
    and the inner product is sesquilinear, implemented through
    :func:`numpy.vdot`.  Projection returns expansion coefficients for any
    supplied orthonormal basis.
    """

    dimension: int
    dtype: np.dtype = np.complex128

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("Hilbert spaces require a positive dimension.")

    @property
    def basis(self) -> np.ndarray:
        """Return the canonical orthonormal basis as identity vectors."""

        return np.eye(self.dimension, dtype=self.dtype)

    def _as_vector(self, value: Sequence[complex] | np.ndarray) -> np.ndarray:
        vector = np.asarray(value, dtype=self.dtype)
        if vector.shape != (self.dimension,):
            raise ValueError(
                f"Vector must have shape ({self.dimension},), got {vector.shape!r}."
            )
        return vector

    def inner_product(
        self, vector_a: Sequence[complex] | np.ndarray, vector_b: Sequence[complex] | np.ndarray
    ) -> complex:
        """Compute the sesquilinear inner product ``⟨a, b⟩``."""

        vec_a = self._as_vector(vector_a)
        vec_b = self._as_vector(vector_b)
        return np.vdot(vec_a, vec_b)

    def norm(self, vector: Sequence[complex] | np.ndarray) -> float:
        """Return the Hilbert norm induced by the inner product."""

        value = self.inner_product(vector, vector)
        magnitude = max(value.real, 0.0)
        return float(np.sqrt(magnitude))

    def is_normalized(
        self, vector: Sequence[complex] | np.ndarray, *, atol: float = 1e-9
    ) -> bool:
        """Check whether a vector has unit norm within a tolerance."""

        return np.isclose(self.norm(vector), 1.0, atol=atol)

    def _validate_basis(self, basis: Sequence[Sequence[complex] | np.ndarray]) -> np.ndarray:
        basis_list = list(basis)
        if len(basis_list) == 0:
            raise ValueError("An orthonormal basis must contain at least one vector.")

        basis_vectors = [self._as_vector(vector) for vector in basis_list]
        matrix = np.vstack(basis_vectors)
        gram = matrix @ matrix.conj().T
        identity = np.eye(matrix.shape[0], dtype=self.dtype)
        if not np.allclose(gram, identity, atol=1e-10):
            raise ValueError("Provided basis is not orthonormal within tolerance.")
        return matrix

    def project(
        self,
        vector: Sequence[complex] | np.ndarray,
        basis: Sequence[Sequence[complex] | np.ndarray] | None = None,
    ) -> np.ndarray:
        """Return coefficients ``⟨b_k|ψ⟩`` for the chosen orthonormal basis."""

        vec = self._as_vector(vector)
        if basis is None:
            return vec.astype(self.dtype, copy=True)

        basis_matrix = self._validate_basis(basis)
        coefficients = basis_matrix.conj() @ vec
        return coefficients.astype(self.dtype, copy=False)


class BanachSpaceEPI:
    """Banach space for :math:`C^0([0, 1],\mathbb{C}) \oplus \ell^2(\mathbb{N})`.

    Elements are represented by a pair ``(f, a)`` where ``f`` samples the
    continuous field over a uniform grid ``x_grid`` and ``a`` is the discrete
    spectral tail.  The coherence norm combines the supremum of ``f``, the
    :math:`\ell^2` norm of ``a`` and a derivative-based functional capturing
    the local stability of ``f``.
    """

    @staticmethod
    def _as_array(values: Sequence[complex] | np.ndarray, *, dtype: np.dtype) -> np.ndarray:
        array = np.asarray(values, dtype=dtype)
        if array.ndim != 1:
            raise ValueError("Inputs must be one-dimensional arrays.")
        if not np.all(np.isfinite(array)):
            raise ValueError("Inputs must not contain NaNs or infinities.")
        return array

    def validate_domain(
        self,
        f_continuous: Sequence[complex] | np.ndarray,
        a_discrete: Sequence[complex] | np.ndarray,
        x_grid: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        """Validate dimensionality and sampling grid compatibility."""

        f_array = self._as_array(f_continuous, dtype=np.complex128)
        self._as_array(a_discrete, dtype=np.complex128)

        if x_grid is None:
            return

        grid = np.asarray(x_grid, dtype=float)
        if grid.ndim != 1:
            raise ValueError("x_grid must be one-dimensional.")
        if grid.size != f_array.size:
            raise ValueError("x_grid length must match continuous component.")
        if grid.size < 2:
            raise ValueError("x_grid must contain at least two points.")
        if not np.all(np.isfinite(grid)):
            raise ValueError("x_grid must not contain NaNs or infinities.")

        spacings = np.diff(grid)
        if np.any(spacings <= 0):
            raise ValueError("x_grid must be strictly increasing.")
        if not np.allclose(spacings, spacings[0], rtol=1e-9, atol=1e-12):
            raise ValueError("x_grid must be uniform for finite-difference stability.")

    def compute_coherence_functional(
        self,
        f_continuous: Sequence[complex] | np.ndarray,
        x_grid: Sequence[float] | np.ndarray,
    ) -> float:
        """Approximate :math:`\int |f'|^2 dx / (1 + \int |f|^2 dx)`."""

        f_array = self._as_array(f_continuous, dtype=np.complex128)
        grid = np.asarray(x_grid, dtype=float)
        self.validate_domain(f_array, np.array([0.0], dtype=np.complex128), grid)

        derivative = np.gradient(
            f_array,
            grid,
            edge_order=2 if f_array.size > 2 else 1,
        )
        numerator = np.trapz(np.abs(derivative) ** 2, grid)
        denominator = 1.0 + np.trapz(np.abs(f_array) ** 2, grid)
        if denominator <= 0:
            raise ValueError("Denominator of coherence functional must be positive.")
        return float(np.real_if_close(numerator / denominator))

    def coherence_norm(
        self,
        f_continuous: Sequence[complex] | np.ndarray,
        a_discrete: Sequence[complex] | np.ndarray,
        *,
        x_grid: Sequence[float] | np.ndarray,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ) -> float:
        """Return ``α‖f‖_∞ + β‖a‖_2 + γ CF(f)`` for positive weights."""

        if alpha <= 0 or beta <= 0 or gamma <= 0:
            raise ValueError("alpha, beta and gamma must be strictly positive.")

        f_array = self._as_array(f_continuous, dtype=np.complex128)
        a_array = self._as_array(a_discrete, dtype=np.complex128)
        self.validate_domain(f_array, a_array, x_grid)

        sup_norm = float(np.max(np.abs(f_array))) if f_array.size else 0.0
        l2_norm = float(np.linalg.norm(a_array))
        coherence_functional = self.compute_coherence_functional(f_array, x_grid)

        value = alpha * sup_norm + beta * l2_norm + gamma * coherence_functional
        return float(np.real_if_close(value))
