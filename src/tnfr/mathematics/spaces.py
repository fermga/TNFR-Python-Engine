"""Mathematical spaces supporting the TNFR canonical paradigm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .epi import BEPIElement, _EPIValidators


@dataclass(frozen=True)
class HilbertSpace:
    r"""Finite section of :math:`\ell^2(\mathbb{N}) \otimes L^2(\mathbb{R})`.

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
            raise ValueError(f"Vector must have shape ({self.dimension},), got {vector.shape!r}.")
        return vector

    def inner_product(
        self,
        vector_a: Sequence[complex] | np.ndarray,
        vector_b: Sequence[complex] | np.ndarray,
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

    def is_normalized(self, vector: Sequence[complex] | np.ndarray, *, atol: float = 1e-9) -> bool:
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


class BanachSpaceEPI(_EPIValidators):
    r"""Banach space for :math:`C^0([0, 1],\mathbb{C}) \oplus \ell^2(\mathbb{N})`.

    Elements are represented by a pair ``(f, a)`` where ``f`` samples the
    continuous field over a uniform grid ``x_grid`` and ``a`` is the discrete
    spectral tail.  The coherence norm combines the supremum of ``f``, the
    :math:`\ell^2` norm of ``a`` and a derivative-based functional capturing
    the local stability of ``f``.
    """

    def element(
        self,
        f_continuous: Sequence[complex] | np.ndarray,
        a_discrete: Sequence[complex] | np.ndarray,
        *,
        x_grid: Sequence[float] | np.ndarray,
    ) -> BEPIElement:
        """Create a :class:`~tnfr.mathematics.epi.BEPIElement` with validated data."""

        self.validate_domain(f_continuous, a_discrete, x_grid)
        return BEPIElement(f_continuous, a_discrete, x_grid)

    def zero_element(
        self,
        *,
        continuous_size: int,
        discrete_size: int,
        x_grid: Sequence[float] | np.ndarray | None = None,
    ) -> BEPIElement:
        """Return the neutral element for the direct sum."""

        if continuous_size < 2:
            raise ValueError("continuous_size must be at least two samples.")
        grid = (
            np.asarray(x_grid, dtype=float)
            if x_grid is not None
            else np.linspace(0.0, 1.0, continuous_size, dtype=float)
        )
        zeros_f = np.zeros(continuous_size, dtype=np.complex128)
        zeros_a = np.zeros(discrete_size, dtype=np.complex128)
        return self.element(zeros_f, zeros_a, x_grid=grid)

    def canonical_basis(
        self,
        *,
        continuous_size: int,
        discrete_size: int,
        continuous_index: int = 0,
        discrete_index: int = 0,
        x_grid: Sequence[float] | np.ndarray | None = None,
    ) -> BEPIElement:
        """Generate a canonical basis element for the Banach space."""

        if continuous_size < 2:
            raise ValueError("continuous_size must be at least two samples.")
        if not (0 <= continuous_index < continuous_size):
            raise ValueError("continuous_index out of range.")
        if not (0 <= discrete_index < discrete_size):
            raise ValueError("discrete_index out of range.")

        grid = (
            np.asarray(x_grid, dtype=float)
            if x_grid is not None
            else np.linspace(0.0, 1.0, continuous_size, dtype=float)
        )

        f_vector = np.zeros(continuous_size, dtype=np.complex128)
        a_vector = np.zeros(discrete_size, dtype=np.complex128)
        f_vector[continuous_index] = 1.0 + 0.0j
        a_vector[discrete_index] = 1.0 + 0.0j
        return self.element(f_vector, a_vector, x_grid=grid)

    def direct_sum(self, left: BEPIElement, right: BEPIElement) -> BEPIElement:
        """Delegate direct sums to the underlying EPI element."""

        return left.direct_sum(right)

    def adjoint(self, element: BEPIElement) -> BEPIElement:
        """Return the adjoint element of the supplied operand."""

        return element.adjoint()

    def compose(
        self,
        element: BEPIElement,
        transform: Callable[[np.ndarray], np.ndarray],
        *,
        spectral_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> BEPIElement:
        """Compose an element with the provided transforms."""

        return element.compose(transform, spectral_transform=spectral_transform)

    def tensor_with_hilbert(
        self,
        element: BEPIElement,
        hilbert_space: HilbertSpace,
        vector: Sequence[complex] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the tensor product against a :class:`HilbertSpace` vector."""

        raw_vector = hilbert_space.basis[0] if vector is None else vector
        hilbert_vector = hilbert_space._as_vector(raw_vector)  # pylint: disable=protected-access
        return element.tensor(hilbert_vector)

    def compute_coherence_functional(
        self,
        f_continuous: Sequence[complex] | np.ndarray,
        x_grid: Sequence[float] | np.ndarray,
    ) -> float:
        r"""Approximate :math:`\int |f'|^2 dx / (1 + \int |f|^2 dx)`."""

        f_array, _, grid = self.validate_domain(
            f_continuous, np.array([0.0], dtype=np.complex128), x_grid
        )
        if grid is None:
            raise ValueError("x_grid must be provided for coherence evaluations.")

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

        f_array, a_array, grid = self.validate_domain(f_continuous, a_discrete, x_grid)
        if grid is None:
            raise ValueError("x_grid must be supplied when evaluating the norm.")

        sup_norm = float(np.max(np.abs(f_array))) if f_array.size else 0.0
        l2_norm = float(np.linalg.norm(a_array))
        coherence_functional = self.compute_coherence_functional(f_array, grid)

        value = alpha * sup_norm + beta * l2_norm + gamma * coherence_functional
        return float(np.real_if_close(value))
