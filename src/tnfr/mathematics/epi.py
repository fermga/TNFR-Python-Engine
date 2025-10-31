"""EPI elements and algebraic helpers for the TNFR Banach space."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


class _EPIValidators:
    """Shared validation helpers for EPI Banach constructions."""

    _complex_dtype = np.complex128

    @staticmethod
    def _as_array(values: Sequence[complex] | np.ndarray, *, dtype: np.dtype) -> np.ndarray:
        array = np.asarray(values, dtype=dtype)
        if array.ndim != 1:
            raise ValueError("Inputs must be one-dimensional arrays.")
        if not np.all(np.isfinite(array)):
            raise ValueError("Inputs must not contain NaNs or infinities.")
        return array

    @classmethod
    def _validate_grid(cls, grid: Sequence[float] | np.ndarray, expected_size: int) -> np.ndarray:
        array = np.asarray(grid, dtype=float)
        if array.ndim != 1:
            raise ValueError("x_grid must be one-dimensional.")
        if array.size != expected_size:
            raise ValueError("x_grid length must match continuous component.")
        if array.size < 2:
            raise ValueError("x_grid must contain at least two points.")
        if not np.all(np.isfinite(array)):
            raise ValueError("x_grid must not contain NaNs or infinities.")

        spacings = np.diff(array)
        if np.any(spacings <= 0):
            raise ValueError("x_grid must be strictly increasing.")
        if not np.allclose(spacings, spacings[0], rtol=1e-9, atol=1e-12):
            raise ValueError("x_grid must be uniform for finite-difference stability.")
        return array

    @classmethod
    def validate_domain(
        cls,
        f_continuous: Sequence[complex] | np.ndarray,
        a_discrete: Sequence[complex] | np.ndarray,
        x_grid: Sequence[float] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Validate dimensionality and sampling grid compatibility."""

        f_array = cls._as_array(f_continuous, dtype=cls._complex_dtype)
        a_array = cls._as_array(a_discrete, dtype=cls._complex_dtype)

        if x_grid is None:
            return f_array, a_array, None

        grid_array = cls._validate_grid(x_grid, f_array.size)
        return f_array, a_array, grid_array


@dataclass(frozen=True)
class BEPIElement(_EPIValidators):
    """Concrete :math:`C^0([0,1]) \oplus \ell^2` element with TNFR operations."""

    f_continuous: Sequence[complex] | np.ndarray
    a_discrete: Sequence[complex] | np.ndarray
    x_grid: Sequence[float] | np.ndarray

    def __post_init__(self) -> None:
        f_array, a_array, grid = self.validate_domain(self.f_continuous, self.a_discrete, self.x_grid)
        if grid is None:
            raise ValueError("x_grid is mandatory for BEPIElement instances.")
        object.__setattr__(self, "f_continuous", f_array)
        object.__setattr__(self, "a_discrete", a_array)
        object.__setattr__(self, "x_grid", grid)

    def _assert_compatible(self, other: BEPIElement) -> None:
        if self.f_continuous.shape != other.f_continuous.shape:
            raise ValueError("Continuous components must share shape for direct sums.")
        if self.a_discrete.shape != other.a_discrete.shape:
            raise ValueError("Discrete tails must share shape for direct sums.")
        if not np.allclose(self.x_grid, other.x_grid, rtol=1e-12, atol=1e-12):
            raise ValueError("x_grid must match to combine EPI elements.")

    def direct_sum(self, other: BEPIElement) -> BEPIElement:
        """Return the algebraic direct sum ``self ⊕ other``."""

        self._assert_compatible(other)
        return BEPIElement(self.f_continuous + other.f_continuous, self.a_discrete + other.a_discrete, self.x_grid)

    def tensor(self, vector: Sequence[complex] | np.ndarray) -> np.ndarray:
        """Return the tensor product between the discrete tail and a Hilbert vector."""

        hilbert_vector = self._as_array(vector, dtype=self._complex_dtype)
        return np.outer(self.a_discrete, hilbert_vector)

    def adjoint(self) -> BEPIElement:
        """Return the conjugate element representing the ``*`` operation."""

        return BEPIElement(np.conjugate(self.f_continuous), np.conjugate(self.a_discrete), self.x_grid)

    @staticmethod
    def _apply_transform(transform: Callable[[np.ndarray], np.ndarray], values: np.ndarray) -> np.ndarray:
        result = np.asarray(transform(values), dtype=np.complex128)
        if result.shape != values.shape:
            raise ValueError("Transforms must preserve the element shape.")
        if not np.all(np.isfinite(result)):
            raise ValueError("Transforms must return finite values.")
        return result

    def compose(
        self,
        transform: Callable[[np.ndarray], np.ndarray],
        *,
        spectral_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> BEPIElement:
        """Compose the element with linear transforms on both components."""

        new_f = self._apply_transform(transform, self.f_continuous)
        spectral_fn = spectral_transform or transform
        new_a = self._apply_transform(spectral_fn, self.a_discrete)
        return BEPIElement(new_f, new_a, self.x_grid)

    def _max_magnitude(self) -> float:
        mags = []
        if self.f_continuous.size:
            mags.append(float(np.max(np.abs(self.f_continuous))))
        if self.a_discrete.size:
            mags.append(float(np.max(np.abs(self.a_discrete))))
        return float(max(mags)) if mags else 0.0

    def __float__(self) -> float:
        return self._max_magnitude()

    def __abs__(self) -> float:
        return self._max_magnitude()

