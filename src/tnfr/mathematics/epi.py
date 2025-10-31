"""EPI elements and algebraic helpers for the TNFR Banach space."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "BEPIElement",
    "CoherenceEvaluation",
    "evaluate_coherence_transform",
]


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


@dataclass(frozen=True)
class CoherenceEvaluation:
    """Container describing the outcome of a coherence transform evaluation."""

    element: BEPIElement
    transformed: BEPIElement
    coherence_before: float
    coherence_after: float
    kappa: float
    tolerance: float
    satisfied: bool
    required: float
    deficit: float
    ratio: float


def evaluate_coherence_transform(
    element: BEPIElement,
    transform: Callable[[BEPIElement], BEPIElement],
    *,
    kappa: float = 1.0,
    tolerance: float = 1e-9,
    space: "BanachSpaceEPI" | None = None,
    norm_kwargs: Mapping[str, float] | None = None,
) -> CoherenceEvaluation:
    """Apply ``transform`` to ``element`` and verify a coherence inequality.

    Parameters
    ----------
    element:
        The :class:`BEPIElement` subject to the transformation.
    transform:
        Callable receiving ``element`` and returning the transformed
        :class:`BEPIElement`.  The callable is expected to preserve the
        structural sampling grid and dimensionality of the element.
    kappa:
        Factor on the right-hand side of the inequality ``C(T(EPI)) ≥ κ·C(EPI)``.
    tolerance:
        Non-negative slack applied to the inequality.  When
        ``C(T(EPI)) + tolerance`` exceeds ``κ·C(EPI)`` the check succeeds.
    space:
        Optional :class:`~tnfr.mathematics.spaces.BanachSpaceEPI` instance used
        to compute the coherence norm.  When omitted, a local instance is
        constructed to avoid circular imports at module import time.
    norm_kwargs:
        Optional keyword arguments forwarded to
        :meth:`BanachSpaceEPI.coherence_norm`.

    Returns
    -------
    CoherenceEvaluation
        Dataclass capturing the before/after coherence values together with the
        inequality verdict.
    """

    if kappa < 0:
        raise ValueError("kappa must be non-negative.")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")

    if norm_kwargs is None:
        norm_kwargs = {}

    from .spaces import BanachSpaceEPI  # Local import to avoid circular dependency

    working_space = space if space is not None else BanachSpaceEPI()

    coherence_before = working_space.coherence_norm(
        element.f_continuous,
        element.a_discrete,
        x_grid=element.x_grid,
        **norm_kwargs,
    )

    transformed = transform(element)
    if not isinstance(transformed, BEPIElement):
        raise TypeError("transform must return a BEPIElement instance.")

    coherence_after = working_space.coherence_norm(
        transformed.f_continuous,
        transformed.a_discrete,
        x_grid=transformed.x_grid,
        **norm_kwargs,
    )

    required = kappa * coherence_before
    satisfied = coherence_after + tolerance >= required
    deficit = max(0.0, required - coherence_after)

    if coherence_before > 0:
        ratio = coherence_after / coherence_before
    elif coherence_after > tolerance:
        ratio = float("inf")
    else:
        ratio = 1.0

    return CoherenceEvaluation(
        element=element,
        transformed=transformed,
        coherence_before=coherence_before,
        coherence_after=coherence_after,
        kappa=kappa,
        tolerance=tolerance,
        satisfied=satisfied,
        required=required,
        deficit=deficit,
        ratio=ratio,
    )

