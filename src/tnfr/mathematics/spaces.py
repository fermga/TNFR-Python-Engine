"""Mathematical spaces supporting the TNFR canonical paradigm."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from .._exact_time import finite_represented_real
from .._spectral_expectation import positive_spectral_dimension
from ..errors import TNFRValueError
from ._complex_arrays import (
    finite_complex_norm,
    nonnegative_tolerance,
    numpy_complex_array,
)
from .epi import BEPIElement, _EPIValidators
from .unified_numerical import np, trapezoid


@dataclass(frozen=True)
class HilbertSpace:
    r"""Finite section of :math:`\ell^2(\mathbb{N}) \otimes L^2(\mathbb{R})`.

    The space models the discrete spectral component of the TNFR paradigm.  The
    canonical orthonormal basis corresponds to the standard coordinate vectors
    and the inner product is sesquilinear, accumulated in complex128 through
    :func:`numpy.vdot`. Projection returns expansion coefficients for any
    supplied orthonormal family, including a partial family. The storage dtype
    must be floating or complex; real storage accepts only zero-imaginary
    coordinates. Materialization must preserve finite, nonzero channels.
    """

    dimension: int
    dtype: np.dtype = np.complex128

    def __post_init__(self) -> None:
        dimension = positive_spectral_dimension(
            self.dimension, label="Hilbert space dimension"
        )
        try:
            dtype = np.dtype(self.dtype)
        except (TypeError, ValueError) as exc:
            raise TNFRValueError("Hilbert dtype must be floating or complex.") from exc
        if dtype.kind not in "fc":
            raise TNFRValueError("Hilbert dtype must be floating or complex.")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "dtype", dtype)

    @property
    def basis(self) -> np.ndarray:
        """Return the canonical orthonormal basis as identity vectors."""

        return np.eye(self.dimension, dtype=self.dtype)

    def _as_vector(self, value: Sequence[complex] | np.ndarray) -> np.ndarray:
        vector = numpy_complex_array(value, label="Hilbert vector")
        if vector.shape != (self.dimension,):
            raise TNFRValueError(
                f"Vector must have shape ({self.dimension},), got {vector.shape!r}.",
                context={
                    "expected_shape": (self.dimension,),
                    "actual_shape": vector.shape,
                },
                suggestion="Ensure the vector shape matches the Hilbert space dimension.",
            )
        return self._materialize(vector)

    def _materialize(self, values: np.ndarray) -> np.ndarray:
        """Cast admitted coordinates without losing an entire nonzero channel."""
        if self.dtype.kind == "f":
            if np.any(values.imag != 0):
                raise TNFRValueError(
                    "Real Hilbert dtype cannot discard nonzero imaginary coordinates."
                )
            source = values.real
        else:
            source = values
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            materialized = np.asarray(source, dtype=self.dtype)
        if not np.all(np.isfinite(materialized)):
            raise TNFRValueError("Hilbert dtype cannot represent finite coordinates.")
        if np.any((values.real != 0) & (materialized.real == 0)) or np.any(
            (values.imag != 0) & (materialized.imag == 0)
        ):
            raise TNFRValueError("Hilbert dtype loses nonzero coordinate channels.")
        return materialized

    def inner_product(
        self,
        vector_a: Sequence[complex] | np.ndarray,
        vector_b: Sequence[complex] | np.ndarray,
    ) -> complex:
        """Compute a finite complex128 sesquilinear product ``⟨a, b⟩``.

        This floating evaluation can round; unresolved overflow is rejected.
        """

        vec_a = np.asarray(self._as_vector(vector_a), dtype=np.complex128)
        vec_b = np.asarray(self._as_vector(vector_b), dtype=np.complex128)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            value = np.vdot(vec_a, vec_b)
        if not np.isfinite(value):
            raise TNFRValueError("Hilbert inner product is not finite.")
        return complex(value)

    def norm(self, vector: Sequence[complex] | np.ndarray) -> float:
        """Return the Hilbert norm induced by the inner product."""

        return finite_complex_norm(self._as_vector(vector), label="Hilbert vector")

    def is_normalized(
        self, vector: Sequence[complex] | np.ndarray, *, atol: float = 1e-9
    ) -> bool:
        """Check whether a vector has unit norm within a tolerance."""

        tolerance = nonnegative_tolerance(atol)
        return bool(np.isclose(self.norm(vector), 1.0, atol=tolerance))

    def _validate_basis(
        self, basis: Sequence[Sequence[complex] | np.ndarray]
    ) -> np.ndarray:
        basis_list = list(basis)
        if len(basis_list) == 0:
            raise TNFRValueError(
                "An orthonormal basis must contain at least one vector.",
                context={"basis_length": 0},
                suggestion="Provide a non-empty basis.",
            )

        basis_vectors = [self._as_vector(vector) for vector in basis_list]
        matrix = np.asarray(np.vstack(basis_vectors), dtype=np.complex128)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            gram = matrix @ matrix.conj().T
        identity = np.eye(matrix.shape[0], dtype=np.complex128)
        if not np.all(np.isfinite(gram)) or not np.allclose(gram, identity, atol=1e-10):
            raise TNFRValueError(
                "Provided basis is not orthonormal within tolerance.",
                context={"tolerance": 1e-10},
                suggestion="Ensure the basis vectors are orthonormal.",
            )
        return matrix

    def project(
        self,
        vector: Sequence[complex] | np.ndarray,
        basis: Sequence[Sequence[complex] | np.ndarray] | None = None,
    ) -> np.ndarray:
        """Return finite coefficients ``⟨b_k|ψ⟩`` in the configured dtype.

        A supplied partial orthonormal family returns only its coefficients.
        Products accumulate in complex128 before checked materialization.
        """

        vec = self._as_vector(vector)
        if basis is None:
            return vec.astype(self.dtype, copy=True)

        basis_matrix = self._validate_basis(basis)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            coefficients = basis_matrix.conj() @ np.asarray(vec, dtype=np.complex128)
        if not np.all(np.isfinite(coefficients)):
            raise TNFRValueError("Hilbert projection coefficients are not finite.")
        return self._materialize(coefficients)


class BanachSpaceEPI(_EPIValidators):
    r"""Finite sampled model inspired by a continuous/discrete direct sum.

    Elements are represented by a pair ``(f, a)`` where ``f`` samples the
    continuous field over a supplied uniform grid ``x_grid`` and ``a`` is a
    finite coefficient array. No interpolation, infinite tail or particular
    grid interval is inferred. :meth:`composite_epi_regularity` combines the
    supremum of ``f``, the
    :math:`\ell^2` norm of ``a`` and a derivative-energy quotient.  It is an
    unbounded regularity functional: larger amplitude or a rougher continuous
    field can increase it.  It is neither the canonical TNFR structural
    coherence ``C(t)`` nor, because of its quotient term, a mathematical norm.
    """

    def element(
        self,
        f_continuous: Sequence[complex] | np.ndarray,
        a_discrete: Sequence[complex] | np.ndarray,
        *,
        x_grid: Sequence[float] | np.ndarray,
    ) -> BEPIElement:
        """Create a :class:`~tnfr.mathematics.epi.BEPIElement` with validated data."""

        validator = self.validate_domain
        if (
            type(self) is not BanachSpaceEPI
            or getattr(validator, "__func__", None)
            is not _EPIValidators.validate_domain.__func__
        ):
            # Preserve user-defined admission hooks. The standard factory
            # delegates its single shared validation to BEPIElement.
            validator(f_continuous, a_discrete, x_grid)
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
            raise TNFRValueError(
                "continuous_size must be at least two samples.",
                context={"continuous_size": continuous_size},
                suggestion="Provide a continuous_size of at least 2.",
            )
        grid = (
            x_grid
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
        """Return a paired unit entry in each of the two stored components.

        Despite the historical name, these paired entries alone do not form
        a basis of the full direct sum: both component sums are always equal.
        """

        if continuous_size < 2:
            raise TNFRValueError(
                "continuous_size must be at least two samples.",
                context={"continuous_size": continuous_size},
                suggestion="Provide a continuous_size of at least 2.",
            )
        if not (0 <= continuous_index < continuous_size):
            raise TNFRValueError(
                "continuous_index out of range.",
                context={
                    "continuous_index": continuous_index,
                    "continuous_size": continuous_size,
                },
                suggestion="Ensure continuous_index is within [0, continuous_size).",
            )
        if not (0 <= discrete_index < discrete_size):
            raise TNFRValueError(
                "discrete_index out of range.",
                context={
                    "discrete_index": discrete_index,
                    "discrete_size": discrete_size,
                },
                suggestion="Ensure discrete_index is within [0, discrete_size).",
            )

        grid = (
            x_grid
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
        hilbert_vector = hilbert_space._as_vector(
            raw_vector
        )  # pylint: disable=protected-access
        return element.tensor(hilbert_vector)

    def derivative_regularity(
        self,
        f_continuous: Sequence[complex] | np.ndarray,
        x_grid: Sequence[float] | np.ndarray,
    ) -> float:
        r"""Return the sampled derivative-energy quotient.

        The functional is

        .. math::

            R_D(f) = \frac{\int |f'(x)|^2\,dx}
                           {1 + \int |f(x)|^2\,dx}.

        ``R_D`` is nonnegative and unbounded.  At comparable amplitude,
        increasing spatial oscillation usually increases its numerator and
        therefore its value.  It is a roughness/regularity read-out, not the
        bounded canonical TNFR structural coherence ``C(t)``.
        """

        f_array, _, grid = self.validate_domain(
            f_continuous, np.array([0.0], dtype=np.complex128), x_grid
        )
        if grid is None:
            raise TNFRValueError(
                "x_grid must be provided for derivative-regularity evaluations.",
                context={"x_grid": x_grid},
                suggestion="Provide a valid x_grid.",
            )

        # Cancel a common large amplitude before squaring. Scaling real and
        # imaginary channels separately avoids complex division overflow.
        scale = max(
            1.0,
            float(np.max(np.abs(f_array.real), initial=0.0)),
            float(np.max(np.abs(f_array.imag), initial=0.0)),
        )
        scaled = f_array.real / scale + 1j * (f_array.imag / scale)
        with np.errstate(
            over="ignore", under="ignore", invalid="ignore", divide="ignore"
        ):
            derivative = np.gradient(
                scaled,
                grid,
                edge_order=2 if f_array.size > 2 else 1,
            )
            numerator = float(trapezoid(np.abs(derivative) ** 2, grid))
            denominator = (1.0 / scale) ** 2 + float(
                trapezoid(np.abs(scaled) ** 2, grid)
            )
        if (
            not math.isfinite(numerator)
            or not math.isfinite(denominator)
            or denominator <= 0
        ):
            raise TNFRValueError(
                "Derivative regularity requires finite energy arithmetic "
                "and a positive denominator."
            )
        value = numerator / denominator
        if not math.isfinite(value):
            raise TNFRValueError("Derivative regularity must be finite.")
        if value == 0 and np.any(f_array != f_array[0]):
            raise TNFRValueError("Nonzero derivative regularity is not representable.")
        return value

    def composite_epi_regularity(
        self,
        f_continuous: Sequence[complex] | np.ndarray,
        a_discrete: Sequence[complex] | np.ndarray,
        *,
        x_grid: Sequence[float] | np.ndarray,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ) -> float:
        r"""Return ``α‖f‖∞ + β‖a‖₂ + γ R_D(f)``.

        The weights must be finite and strictly positive.  This composite EPI
        regularity is an unbounded amplitude-and-roughness functional.  A
        larger value can reflect more amplitude, more discrete-tail energy, or
        more derivative energy; it must not be interpreted as improved
        structural coherence ``C(t)``.
        """

        try:
            alpha, beta, gamma = (
                finite_represented_real(value, name)[0]
                for name, value in (("alpha", alpha), ("beta", beta), ("gamma", gamma))
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise TNFRValueError(
                "alpha, beta and gamma must be finite and strictly positive."
            ) from exc
        if min(alpha, beta, gamma) <= 0:
            raise TNFRValueError(
                "alpha, beta and gamma must be finite and strictly positive.",
                context={"alpha": alpha, "beta": beta, "gamma": gamma},
                suggestion="Provide finite, strictly positive weights.",
            )

        f_array, a_array, grid = self.validate_domain(f_continuous, a_discrete, x_grid)
        if grid is None:
            raise TNFRValueError(
                "x_grid must be supplied when evaluating EPI regularity.",
                context={"x_grid": x_grid},
                suggestion="Provide a valid x_grid.",
            )

        sup_norm = float(np.max(np.abs(f_array))) if f_array.size else 0.0
        l2_norm = finite_complex_norm(a_array, label="Discrete EPI component")
        derivative_term = self.derivative_regularity(f_array, grid)
        try:
            value = math.fsum(
                (alpha * sup_norm, beta * l2_norm, gamma * derivative_term)
            )
        except OverflowError as exc:
            raise TNFRValueError("Composite EPI regularity must be finite.") from exc
        if not math.isfinite(value):
            raise TNFRValueError("Composite EPI regularity must be finite.")
        return value

    def compute_coherence_functional(
        self,
        f_continuous: Sequence[complex] | np.ndarray,
        x_grid: Sequence[float] | np.ndarray,
    ) -> float:
        """Compatibility alias for :meth:`derivative_regularity`.

        The historical name is retained for callers, but this method does not
        compute canonical structural coherence ``C(t)``.
        """

        return self.derivative_regularity(f_continuous, x_grid)

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
        """Compatibility alias for :meth:`composite_epi_regularity`.

        The historical name is retained for callers, but the returned value is
        unbounded and is neither canonical structural coherence ``C(t)`` nor a
        mathematical norm.
        """

        return self.composite_epi_regularity(
            f_continuous,
            a_discrete,
            x_grid=x_grid,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
