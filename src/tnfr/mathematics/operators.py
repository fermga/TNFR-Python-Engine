"""Auxiliary Hermitian spectral-expectation and frequency operators.

The expectation ``<psi|A|psi>`` is an unbounded real observable in the units of
``A``.  It is deliberately separate from canonical structural coherence
``C(t)`` and is never a source for ``history['C_steps']``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

from .._spectral_expectation import finite_spectral_real
from ..constants.canonical import (
    MATH_SPECTRAL_EXPECTATION_FLOOR_DEFAULT,
    MATH_TOLERANCE_CANONICAL,
)
from ..errors import TNFRValueError
from ._complex_arrays import (
    backend_complex_array,
    nonnegative_tolerance,
    normalized_complex_vector,
)
from .backend import MathematicsBackend, ensure_array, ensure_numpy, get_backend
from .unified_numerical import np

if TYPE_CHECKING:  # pragma: no cover - typing imports only
    # We import numpy directly for typing to ensure mypy understands the types
    import numpy as _np_typing
    import numpy.typing as npt

    ComplexVector = npt.NDArray[
        _np_typing.complexfloating[_np_typing.float64, _np_typing.float64]
    ]
    ComplexMatrix = npt.NDArray[
        _np_typing.complexfloating[_np_typing.float64, _np_typing.float64]
    ]
else:  # pragma: no cover - runtime alias
    ComplexVector = np.ndarray
    ComplexMatrix = np.ndarray

__all__ = [
    "SpectralExpectationOperator",
    "CoherenceOperator",
    "FrequencyOperator",
    "DEFAULT_SPECTRAL_EXPECTATION_FLOOR",
    "DEFAULT_C_MIN",
]

DEFAULT_SPECTRAL_EXPECTATION_FLOOR: float = MATH_SPECTRAL_EXPECTATION_FLOOR_DEFAULT
# Compatibility alias for the historical spectral API.  This value has never
# been a canonical C(t) threshold.
DEFAULT_C_MIN: float = DEFAULT_SPECTRAL_EXPECTATION_FLOOR
_C_MIN_UNSET = object()


def _as_complex_vector(
    vector: Sequence[complex] | np.ndarray | Any,
    *,
    backend: MathematicsBackend,
) -> Any:
    arr = backend_complex_array(vector, backend=backend, label="Spectral state")
    if getattr(arr, "ndim", len(getattr(arr, "shape", ()))) != 1:
        raise TNFRValueError(
            "Vector input must be one-dimensional.",
            context={"ndim": getattr(arr, "ndim", len(getattr(arr, "shape", ())))},
            suggestion="Provide a 1D vector.",
        )
    return arr


def _make_diagonal(values: Any, *, backend: MathematicsBackend) -> Any:
    dim = int(getattr(values, "shape")[0])
    identity = ensure_array(np.eye(dim, dtype=np.complex128), backend=backend)
    return backend.einsum("i,ij->ij", values, identity)


@dataclass(slots=True)
class SpectralExpectationOperator:
    r"""Hermitian operator defining an auxiliary spectral expectation.

    The observable is ``<psi|A|psi>`` for a Hermitian matrix ``A``. For a
    unit-norm state its exact-real value lies in the spectral interval of ``A``;
    an unnormalized state scales that interval by ``||psi||**2``. Values may
    lie below zero or above one and have no structural-coherence meaning. In
    particular, it is not ``C(t) = 1/(1 + mean|DeltaNFR| + mean|dEPI|)`` and
    must not be recorded in ``C_steps`` or compared with canonical ``[0, 1]``
    coherence bands.

    Construction accepts either an explicit matrix or an eigenvalue vector.
    ``expectation_floor`` names the optional auxiliary comparison floor.  The
    historical keyword and attribute ``c_min`` remain compatibility aliases;
    when neither is supplied, the minimum real eigenvalue is stored.

    When instantiated under an automatic differentiation backend (JAX, PyTorch)
    the spectral decomposition remains differentiable provided the supplied
    operator is non-defective.  NumPy callers receive ``numpy.ndarray`` outputs
    and all tolerance checks match the historical semantics.

    Construction owns a copy of the supplied matrix. ``matrix``, ``eigenvalues``
    and ``spectrum()`` return detached arrays; construct a new operator to change
    the law instead of assigning to those read-only properties.
    """

    metric_kind: ClassVar[str] = "spectral_operator_expectation"
    canonical_coherence_certified: ClassVar[bool] = False
    canonical_history_key: ClassVar[None] = None

    c_min: float
    backend: MathematicsBackend = field(init=False, repr=False)
    _matrix_backend: Any = field(init=False, repr=False)
    _matrix_numpy: ComplexMatrix = field(init=False, repr=False)
    _eigenvalues_numpy: ComplexVector = field(init=False, repr=False)

    def __init__(
        self,
        operator: Sequence[Sequence[complex]] | Sequence[complex] | np.ndarray | Any,
        *,
        c_min: float | object = _C_MIN_UNSET,
        expectation_floor: float | object = _C_MIN_UNSET,
        ensure_hermitian: bool = True,
        atol: float = 1e-9,
        backend: MathematicsBackend | None = None,
    ) -> None:
        if c_min is not _C_MIN_UNSET and expectation_floor is not _C_MIN_UNSET:
            raise TNFRValueError(
                "Provide either expectation_floor or legacy c_min, not both.",
                context={"c_min": c_min, "expectation_floor": expectation_floor},
                suggestion="Use expectation_floor in new spectral code.",
            )
        atol = nonnegative_tolerance(atol)
        resolved_backend = backend or get_backend()
        operand = backend_complex_array(
            operator, backend=resolved_backend, label="Spectral operator", owned=True
        )
        if getattr(operand, "ndim", len(getattr(operand, "shape", ()))) == 1:
            eigenvalues_numpy = np.array(
                ensure_numpy(operand, backend=resolved_backend), copy=True
            )
            if ensure_hermitian:
                imag = eigenvalues_numpy.imag
                if not np.allclose(imag, 0.0, atol=atol):
                    raise TNFRValueError(
                        "Hermitian operators require real eigenvalues.",
                        context={"max_imag": float(np.max(np.abs(imag))), "atol": atol},
                        suggestion="Ensure eigenvalues are real.",
                    )
            matrix_backend = _make_diagonal(operand, backend=resolved_backend)
            matrix_numpy = np.array(
                ensure_numpy(matrix_backend, backend=resolved_backend), copy=True
            )
        else:
            shape = getattr(operand, "shape", None)
            if shape is None or len(shape) != 2 or shape[0] != shape[1]:
                raise TNFRValueError(
                    "Operator matrix must be square.",
                    context={"shape": shape},
                    suggestion="Provide a square matrix.",
                )
            matrix_backend = operand
            matrix_numpy = np.array(
                ensure_numpy(matrix_backend, backend=resolved_backend), copy=True
            )
            if ensure_hermitian and not self._check_hermitian(matrix_numpy, atol=atol):
                raise TNFRValueError(
                    "Spectral expectation operator must be Hermitian.",
                    context={"is_hermitian": False},
                    suggestion="Ensure the operator matrix is Hermitian.",
                )
            if ensure_hermitian:
                eigenvalues_backend, _ = resolved_backend.eigh(matrix_backend)
            else:
                eigenvalues_backend, _ = resolved_backend.eig(matrix_backend)
            eigenvalues_numpy = np.array(
                ensure_numpy(eigenvalues_backend, backend=resolved_backend), copy=True
            )

        self.backend = resolved_backend
        self._matrix_backend = matrix_backend
        self._matrix_numpy = matrix_numpy
        self._eigenvalues_numpy = eigenvalues_numpy
        derived_c_min = float(np.min(self._eigenvalues_numpy.real))
        requested_floor = (
            expectation_floor if expectation_floor is not _C_MIN_UNSET else c_min
        )
        if requested_floor is _C_MIN_UNSET:
            self.c_min = derived_c_min
        else:
            self.c_min = finite_spectral_real(
                requested_floor, label="Spectral expectation floor"
            )

    @property
    def matrix(self) -> ComplexMatrix:
        """Return a detached copy of the owned operator matrix."""
        return self._matrix_numpy.copy()

    @property
    def eigenvalues(self) -> ComplexVector:
        """Return a detached copy of the spectrum of the owned matrix."""
        return self._eigenvalues_numpy.copy()

    @property
    def expectation_floor(self) -> float:
        """Return the auxiliary spectral comparison floor.

        ``c_min`` remains available as the historical attribute name.
        """

        return self.c_min

    @staticmethod
    def _check_hermitian(
        matrix: Any,
        *,
        atol: float = 1e-9,
        backend: MathematicsBackend | None = None,
    ) -> bool:
        atol = nonnegative_tolerance(atol)
        matrix_np = (
            np.asarray(matrix)
            if backend is None
            else ensure_numpy(matrix, backend=backend)
        )
        return bool(np.allclose(matrix_np, matrix_np.conj().T, atol=atol))

    def is_hermitian(self, *, atol: float = 1e-9) -> bool:
        """Return ``True`` when the operator matches its adjoint."""

        return self._check_hermitian(self._matrix_numpy, atol=atol)

    def is_positive_semidefinite(self, *, atol: float = 1e-9) -> bool:
        """Check Hermiticity and non-negative eigenvalues within ``atol``."""

        atol = nonnegative_tolerance(atol)
        return self.is_hermitian(atol=atol) and bool(
            np.all(self._eigenvalues_numpy.real >= -atol)
        )

    def spectrum(self) -> ComplexVector:
        """Return the complex eigenvalue spectrum."""

        return np.array(self._eigenvalues_numpy, dtype=np.complex128, copy=True)

    def spectral_radius(self) -> float:
        """Return the largest magnitude eigenvalue (spectral radius)."""

        return float(np.max(np.abs(self._eigenvalues_numpy)))

    def spectral_bandwidth(self) -> float:
        """Return the real bandwidth ``max(λ) - min(λ)``."""

        eigvals = self._eigenvalues_numpy.real
        return float(np.max(eigvals) - np.min(eigvals))

    def expectation(
        self,
        state: Sequence[complex] | np.ndarray,
        *,
        normalise: bool = True,
        atol: float = 1e-9,
    ) -> float:
        atol = nonnegative_tolerance(atol)
        vector_backend = _as_complex_vector(state, backend=self.backend)
        if vector_backend.shape != (self._matrix_numpy.shape[0],):
            raise TNFRValueError(
                "State vector dimension mismatch with operator.",
                context={
                    "operator_shape": self._matrix_numpy.shape,
                    "vector_shape": vector_backend.shape,
                },
                suggestion="Ensure vector dimension matches operator dimension.",
            )
        working = vector_backend
        if normalise:
            working = normalized_complex_vector(
                working, backend=self.backend, atol=1e-8, label="state vector"
            )
        column = working[..., None]
        bra = self.backend.conjugate_transpose(column)
        evolved = self.backend.matmul(self._matrix_backend, column)
        expectation_backend = self.backend.matmul(bra, evolved)
        expectation = ensure_numpy(expectation_backend, backend=self.backend)
        expectation_scalar = complex(np.asarray(expectation).reshape(()))
        if not np.isfinite(expectation_scalar):
            raise TNFRValueError("Spectral expectation must be finite")
        if abs(expectation_scalar.imag) > atol:
            raise TNFRValueError(
                "Expectation value carries an imaginary component beyond tolerance.",
                context={"imag": expectation_scalar.imag, "atol": atol},
                suggestion="Check operator hermiticity or state vector validity.",
            )
        eps = np.finfo(float).eps
        tol = (
            max(MATH_TOLERANCE_CANONICAL, float(atol / eps))
            if atol > 0
            else MATH_TOLERANCE_CANONICAL
        )
        real_expectation = np.real_if_close(expectation_scalar, tol=tol)
        if np.iscomplexobj(real_expectation):
            raise TNFRValueError(
                "Expectation remained complex after coercion.",
                context={"value": expectation_scalar},
                suggestion="Ensure the expectation value is real.",
            )
        return float(real_expectation)


CoherenceOperator = SpectralExpectationOperator
"""Backward-compatible alias for :class:`SpectralExpectationOperator`."""


class FrequencyOperator(SpectralExpectationOperator):
    """Operator encoding the structural frequency distribution.

    The frequency operator reuses the Hermitian expectation machinery but
    enforces a real
    spectrum representing the structural hertz (νf) each mode contributes.  Its
    helpers therefore constrain outputs to the real axis and expose projections
    suited for telemetry collection.
    """

    metric_kind: ClassVar[str] = "frequency_operator_expectation"

    def __init__(
        self,
        operator: Sequence[Sequence[complex]] | Sequence[complex] | np.ndarray | Any,
        *,
        ensure_hermitian: bool = True,
        atol: float = 1e-9,
        backend: MathematicsBackend | None = None,
    ) -> None:
        super().__init__(
            operator,
            ensure_hermitian=ensure_hermitian,
            atol=atol,
            backend=backend,
        )

    def spectrum(self) -> np.ndarray:
        """Return the real-valued structural frequency spectrum."""

        return np.array(self._eigenvalues_numpy.real, dtype=float, copy=True)

    def is_positive_semidefinite(self, *, atol: float = 1e-9) -> bool:
        """Frequency spectra must be non-negative to preserve νf semantics."""

        return super().is_positive_semidefinite(atol=atol)

    def project_frequency(
        self,
        state: Sequence[complex] | np.ndarray,
        *,
        normalise: bool = True,
        atol: float = 1e-9,
    ) -> float:
        return self.expectation(state, normalise=normalise, atol=atol)
