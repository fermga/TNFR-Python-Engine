"""EPI elements and algebraic helpers for the TNFR Banach space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

from .unified_numerical import TNFRValueError, np

if TYPE_CHECKING:
    from .spaces import BanachSpaceEPI

COMPOSITE_EPI_REGULARITY_KIND = "composite_epi_regularity"
COMPOSITE_EPI_REGULARITY_PROVENANCE = (
    "tnfr.mathematics.spaces.BanachSpaceEPI.composite_epi_regularity"
)

__all__ = [
    "BEPIElement",
    "COMPOSITE_EPI_REGULARITY_KIND",
    "COMPOSITE_EPI_REGULARITY_PROVENANCE",
    "CompositeEPIRegularityEvaluation",
    "evaluate_composite_epi_regularity_transform",
    # Historical compatibility aliases; neither denotes canonical C(t).
    "CoherenceEvaluation",
    "evaluate_coherence_transform",
]


class _EPIValidators:
    """Shared validation helpers for EPI Banach constructions."""

    _complex_dtype = np.complex128

    @staticmethod
    def _as_array(
        values: Sequence[complex] | np.ndarray, *, dtype: np.dtype
    ) -> np.ndarray:
        array = np.asarray(values, dtype=dtype)
        if array.ndim != 1:
            raise TNFRValueError(
                "Inputs must be one-dimensional arrays.",
                context={"ndim": array.ndim},
                suggestion="Provide a 1D array.",
            )
        if not np.all(np.isfinite(array)):
            raise TNFRValueError(
                "Inputs must not contain NaNs or infinities.",
                context={"finite": False},
                suggestion="Check input data for validity.",
            )
        return array

    @classmethod
    def _validate_grid(
        cls, grid: Sequence[float] | np.ndarray, expected_size: int
    ) -> np.ndarray:
        array = np.asarray(grid, dtype=float)
        if array.ndim != 1:
            raise TNFRValueError(
                "x_grid must be one-dimensional.",
                context={"ndim": array.ndim},
                suggestion="Provide a 1D grid.",
            )
        if array.size != expected_size:
            raise TNFRValueError(
                "x_grid length must match continuous component.",
                context={"grid_size": array.size, "expected_size": expected_size},
                suggestion="Ensure grid size matches data size.",
            )
        if array.size < 2:
            raise TNFRValueError(
                "x_grid must contain at least two points.",
                context={"grid_size": array.size},
                suggestion="Provide a grid with at least 2 points.",
            )
        if not np.all(np.isfinite(array)):
            raise TNFRValueError(
                "x_grid must not contain NaNs or infinities.",
                context={"finite": False},
                suggestion="Check grid data for validity.",
            )

        spacings = np.diff(array)
        if np.any(spacings <= 0):
            raise TNFRValueError(
                "x_grid must be strictly increasing.",
                context={"monotonic": False},
                suggestion="Ensure grid points are sorted and unique.",
            )
        if not np.allclose(spacings, spacings[0], rtol=1e-9, atol=1e-12):
            raise TNFRValueError(
                "x_grid must be uniform for finite-difference stability.",
                context={"uniform": False},
                suggestion="Provide a uniformly spaced grid.",
            )
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
    r"""Concrete :math:`C^0([0,1]) \oplus \ell^2` element with TNFR operations."""

    f_continuous: Sequence[complex] | np.ndarray
    a_discrete: Sequence[complex] | np.ndarray
    x_grid: Sequence[float] | np.ndarray

    def __post_init__(self) -> None:
        f_array, a_array, grid = self.validate_domain(
            self.f_continuous, self.a_discrete, self.x_grid
        )
        if grid is None:
            raise TNFRValueError(
                "x_grid is mandatory for BEPIElement instances.",
                context={"grid": None},
                suggestion="Provide a valid x_grid.",
            )
        object.__setattr__(self, "f_continuous", f_array)
        object.__setattr__(self, "a_discrete", a_array)
        object.__setattr__(self, "x_grid", grid)

    def _assert_compatible(self, other: BEPIElement) -> None:
        if self.f_continuous.shape != other.f_continuous.shape:
            raise TNFRValueError(
                "Continuous components must share shape for direct sums.",
                context={
                    "self_shape": self.f_continuous.shape,
                    "other_shape": other.f_continuous.shape,
                },
                suggestion="Ensure continuous components have matching shapes.",
            )
        if self.a_discrete.shape != other.a_discrete.shape:
            raise TNFRValueError(
                "Discrete tails must share shape for direct sums.",
                context={
                    "self_shape": self.a_discrete.shape,
                    "other_shape": other.a_discrete.shape,
                },
                suggestion="Ensure discrete components have matching shapes.",
            )
        if not np.allclose(self.x_grid, other.x_grid, rtol=1e-12, atol=1e-12):
            raise TNFRValueError(
                "x_grid must match to combine EPI elements.",
                context={"grid_match": False},
                suggestion="Ensure both elements share the same grid.",
            )

    def direct_sum(self, other: BEPIElement) -> BEPIElement:
        """Return the algebraic direct sum ``self ⊕ other``."""

        self._assert_compatible(other)
        return BEPIElement(
            self.f_continuous + other.f_continuous,
            self.a_discrete + other.a_discrete,
            self.x_grid,
        )

    def tensor(self, vector: Sequence[complex] | np.ndarray) -> np.ndarray:
        """Return the tensor product between the discrete tail and a Hilbert vector."""

        hilbert_vector = self._as_array(vector, dtype=self._complex_dtype)
        return np.outer(self.a_discrete, hilbert_vector)

    def adjoint(self) -> BEPIElement:
        """Return the conjugate element representing the ``*`` operation."""

        return BEPIElement(
            np.conjugate(self.f_continuous), np.conjugate(self.a_discrete), self.x_grid
        )

    @staticmethod
    def _apply_transform(
        transform: Callable[[np.ndarray], np.ndarray], values: np.ndarray
    ) -> np.ndarray:
        result = np.asarray(transform(values), dtype=np.complex128)
        if result.shape != values.shape:
            raise TNFRValueError(
                "Transforms must preserve the element shape.",
                context={"input_shape": values.shape, "output_shape": result.shape},
                suggestion="Ensure transform preserves shape.",
            )
        if not np.all(np.isfinite(result)):
            raise TNFRValueError(
                "Transforms must return finite values.",
                context={"finite": False},
                suggestion="Check transform for singularities.",
            )
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
        """Return the non-negative component supremum used by ``abs``."""

        mags = []
        if self.f_continuous.size:
            mags.append(float(np.max(np.abs(self.f_continuous))))
        if self.a_discrete.size:
            mags.append(float(np.max(np.abs(self.a_discrete))))
        return float(max(mags)) if mags else 0.0

    def real_scalar_embedding(self) -> float | None:
        """Return the exact signed scalar embedding, if one is present.

        Both BEPI components must be non-empty and every entry in both must be
        exactly the same finite real number.  Returning ``None`` distinguishes
        richer elements from the valid scalar value zero.
        """

        if not self.f_continuous.size or not self.a_discrete.size:
            return None
        representative = complex(self.f_continuous[0])
        if (
            representative.imag == 0.0
            and bool(np.isfinite(representative.real))
            and bool(np.all(self.f_continuous == representative))
            and bool(np.all(self.a_discrete == representative))
        ):
            return float(representative.real)
        return None

    def scalar_projection(self) -> float:
        """Return the canonical scalar read-out of this EPI element.

        A uniform real element is the trivial embedding produced by
        :func:`tnfr.types.ensure_bepi` for a scalar.  Its signed value is the
        scalar EPI carried by the graph engine.  A genuinely non-scalar or
        complex element has no signed one-dimensional representative, so its
        established maximum-component magnitude remains the projection.

        Uniformity is intentionally exact.  Approximate equality would silently
        discard resolved structural variation and could change sign near zero.
        """

        scalar = self.real_scalar_embedding()
        if scalar is not None:
            return scalar
        return self._max_magnitude()

    def __float__(self) -> float:
        return self.scalar_projection()

    def __abs__(self) -> float:
        return self._max_magnitude()

    def __getstate__(self) -> dict[str, tuple[complex, ...] | tuple[float, ...]]:
        """Serialize BEPIElement to a JSON-compatible dict with real/imag pairs.

        This method enables pickle, JSON, and YAML serialization while preserving
        TNFR invariant #1 (Nodal Equation Integrity, EPI as coherent form) and #3 (Multi-Scale Fractality).
        """
        # Convert numpy arrays to lists for serialization
        continuous = self.f_continuous.tolist()
        discrete = self.a_discrete.tolist()
        grid = self.x_grid.tolist()

        return {
            "continuous": tuple(continuous),
            "discrete": tuple(discrete),
            "grid": tuple(grid),
        }

    def __setstate__(
        self, state: dict[str, tuple[complex, ...] | tuple[float, ...]]
    ) -> None:
        """Deserialize BEPIElement from a dict representation.

        Restores the structural integrity by validating and converting back to numpy arrays.
        """
        f_array, a_array, grid = self.validate_domain(
            state["continuous"], state["discrete"], state["grid"]
        )
        if grid is None:
            raise TNFRValueError(
                "x_grid is mandatory for BEPIElement instances.",
                context={"grid": None},
                suggestion="Provide a valid x_grid.",
            )
        object.__setattr__(self, "f_continuous", f_array)
        object.__setattr__(self, "a_discrete", a_array)
        object.__setattr__(self, "x_grid", grid)

    def __add__(self, other: BEPIElement | float | int) -> BEPIElement:
        """Add a scalar or another BEPIElement to this element."""
        if isinstance(other, (int, float)):
            # Scalar addition: broadcast to all components
            scalar = complex(other)
            return BEPIElement(
                self.f_continuous + scalar, self.a_discrete + scalar, self.x_grid
            )
        elif isinstance(other, BEPIElement):
            # Element addition: use direct_sum
            return self.direct_sum(other)
        return NotImplemented

    def __radd__(self, other: float | int) -> BEPIElement:
        """Support reversed addition (scalar + BEPIElement)."""
        return self.__add__(other)

    def __sub__(self, other: BEPIElement | float | int) -> BEPIElement:
        """Subtract a scalar or another BEPIElement from this element."""
        if isinstance(other, (int, float)):
            scalar = complex(other)
            return BEPIElement(
                self.f_continuous - scalar, self.a_discrete - scalar, self.x_grid
            )
        elif isinstance(other, BEPIElement):
            self._assert_compatible(other)
            return BEPIElement(
                self.f_continuous - other.f_continuous,
                self.a_discrete - other.a_discrete,
                self.x_grid,
            )
        return NotImplemented

    def __rsub__(self, other: float | int) -> BEPIElement:
        """Support reversed subtraction (scalar - BEPIElement)."""
        if isinstance(other, (int, float)):
            scalar = complex(other)
            return BEPIElement(
                scalar - self.f_continuous, scalar - self.a_discrete, self.x_grid
            )
        return NotImplemented

    def __mul__(self, other: float | int) -> BEPIElement:
        """Multiply this element by a scalar."""
        if isinstance(other, (int, float)):
            scalar = complex(other)
            return BEPIElement(
                self.f_continuous * scalar, self.a_discrete * scalar, self.x_grid
            )
        return NotImplemented

    def __rmul__(self, other: float | int) -> BEPIElement:
        """Support reversed multiplication (scalar * BEPIElement)."""
        return self.__mul__(other)

    def __truediv__(self, other: float | int) -> BEPIElement:
        """Divide this element by a scalar."""
        if isinstance(other, (int, float)):
            scalar = complex(other)
            if scalar == 0:
                raise ZeroDivisionError("Cannot divide BEPIElement by zero")
            return BEPIElement(
                self.f_continuous / scalar, self.a_discrete / scalar, self.x_grid
            )
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        """Check equality with another BEPIElement or numeric value.

        Numeric comparison uses the same canonical scalar projection as
        :class:`float`.  Thus scalar embeddings retain their sign while richer
        BEPI elements retain the established maximum-magnitude comparison.
        """
        if isinstance(other, BEPIElement):
            return (
                np.allclose(
                    self.f_continuous, other.f_continuous, rtol=1e-12, atol=1e-12
                )
                and np.allclose(
                    self.a_discrete, other.a_discrete, rtol=1e-12, atol=1e-12
                )
                and np.allclose(self.x_grid, other.x_grid, rtol=1e-12, atol=1e-12)
            )
        elif isinstance(other, (int, float)):
            return abs(self.scalar_projection() - float(other)) < 1e-12
        return NotImplemented


@dataclass(frozen=True)
class CompositeEPIRegularityEvaluation:
    """Result of a composite EPI regularity lower-bound assessment.

    The evaluated functional is unbounded and rises with amplitude or sampled
    roughness.  ``satisfied`` only reports the declared lower-bound inequality;
    it does not certify canonical structural coherence ``C(t)``.
    """

    element: BEPIElement
    transformed: BEPIElement
    regularity_before: float
    regularity_after: float
    kappa: float
    tolerance: float
    satisfied: bool
    required: float
    deficit: float
    ratio: float
    metric_kind: str = COMPOSITE_EPI_REGULARITY_KIND
    provenance: str = COMPOSITE_EPI_REGULARITY_PROVENANCE

    @property
    def coherence_before(self) -> float:
        """Compatibility alias for :attr:`regularity_before`.

        This historical property name does not denote canonical ``C(t)``.
        """

        return self.regularity_before

    @property
    def coherence_after(self) -> float:
        """Compatibility alias for :attr:`regularity_after`.

        This historical property name does not denote canonical ``C(t)``.
        """

        return self.regularity_after


def evaluate_composite_epi_regularity_transform(
    element: BEPIElement,
    transform: Callable[[BEPIElement], BEPIElement],
    *,
    kappa: float = 1.0,
    tolerance: float = 1e-9,
    space: "BanachSpaceEPI | None" = None,
    regularity_kwargs: Mapping[str, float] | None = None,
) -> CompositeEPIRegularityEvaluation:
    r"""Apply ``transform`` and assess regularity retention.

    The check is
    ``R(T(EPI)) + tolerance >= kappa * R(EPI)``, where ``R`` is
    :meth:`BanachSpaceEPI.composite_epi_regularity`.  It preserves the legacy
    lower-bound contract, but a larger ``R`` can mean greater amplitude or
    derivative energy.  The verdict therefore makes no claim about canonical
    structural coherence ``C(t)``.
    """

    if not np.isfinite(kappa) or kappa < 0:
        raise TNFRValueError(
            "kappa must be finite and non-negative.",
            context={"kappa": kappa},
            suggestion="Provide a finite, non-negative kappa.",
        )
    if not np.isfinite(tolerance) or tolerance < 0:
        raise TNFRValueError(
            "tolerance must be finite and non-negative.",
            context={"tolerance": tolerance},
            suggestion="Provide a finite, non-negative tolerance.",
        )

    if regularity_kwargs is None:
        regularity_kwargs = {}

    from .spaces import BanachSpaceEPI  # Local import avoids circular dependency.

    working_space = space if space is not None else BanachSpaceEPI()
    regularity_before = working_space.composite_epi_regularity(
        element.f_continuous,
        element.a_discrete,
        x_grid=element.x_grid,
        **regularity_kwargs,
    )

    transformed = transform(element)
    if not isinstance(transformed, BEPIElement):
        raise TypeError("transform must return a BEPIElement instance.")

    regularity_after = working_space.composite_epi_regularity(
        transformed.f_continuous,
        transformed.a_discrete,
        x_grid=transformed.x_grid,
        **regularity_kwargs,
    )

    required = kappa * regularity_before
    satisfied = regularity_after + tolerance >= required
    deficit = max(0.0, required - regularity_after)

    if regularity_before > 0:
        ratio = regularity_after / regularity_before
    elif regularity_after > tolerance:
        ratio = float("inf")
    else:
        ratio = 1.0

    return CompositeEPIRegularityEvaluation(
        element=element,
        transformed=transformed,
        regularity_before=regularity_before,
        regularity_after=regularity_after,
        kappa=kappa,
        tolerance=tolerance,
        satisfied=satisfied,
        required=required,
        deficit=deficit,
        ratio=ratio,
    )


# Historical type alias.  Its fields now use the honest regularity terminology;
# ``coherence_before`` and ``coherence_after`` remain read-only property aliases.
CoherenceEvaluation = CompositeEPIRegularityEvaluation


def evaluate_coherence_transform(
    element: BEPIElement,
    transform: Callable[[BEPIElement], BEPIElement],
    *,
    kappa: float = 1.0,
    tolerance: float = 1e-9,
    space: "BanachSpaceEPI | None" = None,
    norm_kwargs: Mapping[str, float] | None = None,
) -> CompositeEPIRegularityEvaluation:
    """Compatibility alias for regularity-transform assessment.

    The historical function name and ``norm_kwargs`` parameter are retained for
    callers.  The result assesses the unbounded composite EPI regularity and
    does not evaluate canonical structural coherence ``C(t)``.
    """

    return evaluate_composite_epi_regularity_transform(
        element,
        transform,
        kappa=kappa,
        tolerance=tolerance,
        space=space,
        regularity_kwargs=norm_kwargs,
    )
