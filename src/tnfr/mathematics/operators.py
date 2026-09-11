"""Auxiliary Hermitian spectral-expectation and frequency operators.

The expectation ``<psi|A|psi>`` is an unbounded real observable in the units of
``A``.  It is deliberately separate from canonical structural coherence
``C(t)`` and is never a source for ``history['C_steps']``.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

from ..compat.dataclass import dataclass
from ..constants.canonical import (
    MATH_SPECTRAL_EXPECTATION_FLOOR_DEFAULT,
    MATH_TOLERANCE_CANONICAL,
)
from ..errors import TNFRValueError
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

DEFAULT_SPECTRAL_EXPECTATION_FLOOR: float = (
    MATH_SPECTRAL_EXPECTATION_FLOOR_DEFAULT
)
# Compatibility alias for the historical spectral API.  This value has never
# been a canonical C(t) threshold.
DEFAULT_C_MIN: float = DEFAULT_SPECTRAL_EXPECTATION_FLOOR
_C_MIN_UNSET = object()


def _as_complex_vector(
    vector: Sequence[complex] | np.ndarray | Any,
    *,
    backend: MathematicsBackend,
) -> Any:
    arr = ensure_array(vector, dtype=np.complex128, backend=backend)
    if getattr(arr, "ndim", len(getattr(arr, "shape", ()))) != 1:
        raise TNFRValueError(
            "Vector input must be one-dimensional.",
            context={"ndim": getattr(arr, "ndim", len(getattr(arr, "shape", ())))},
            suggestion="Provide a 1D vector.",
        )
    return arr


def _as_complex_matrix(
    matrix: Sequence[Sequence[complex]] | np.ndarray | Any,
    *,
    backend: MathematicsBackend,
) -> Any:
    arr = ensure_array(matrix, dtype=np.complex128, backend=backend)
    shape = getattr(arr, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        raise TNFRValueError(
            "Operator matrix must be square.",
            context={"shape": shape},
            suggestion="Provide a square matrix.",
        )
    return arr


def _make_diagonal(values: Any, *, backend: MathematicsBackend) -> Any:
    dim = int(getattr(values, "shape")[0])
    identity = ensure_array(np.eye(dim, dtype=np.complex128), backend=backend)
    return backend.einsum("i,ij->ij", values, identity)


@dataclass(slots=True)
class SpectralExpectationOperator:
    r"""Hermitian operator defining an auxiliary spectral expectation.

    The observable is ``<psi|A|psi>`` for a Hermitian matrix ``A``.  Its range
    is the real spectral interval of ``A`` and may lie below zero or above one;
    it therefore has no canonical structural-coherence interpretation.  In
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
    """

    metric_kind: ClassVar[str] = "spectral_operator_expectation"
    canonical_coherence_certified: ClassVar[bool] = False
    canonical_history_key: ClassVar[None] = None

    matrix: ComplexMatrix
    eigenvalues: ComplexVector
    c_min: float
    backend: MathematicsBackend = field(init=False, repr=False)
    _matrix_backend: Any = field(init=False, repr=False)
    _eigenvalues_backend: Any = field(init=False, repr=False)

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
        resolved_backend = backend or get_backend()
        operand = ensure_array(operator, dtype=np.complex128, backend=resolved_backend)
        if getattr(operand, "ndim", len(getattr(operand, "shape", ()))) == 1:
            eigvals_backend = _as_complex_vector(operand, backend=resolved_backend)
            if ensure_hermitian:
                imag = ensure_numpy(eigvals_backend.imag, backend=resolved_backend)
                if not np.allclose(imag, 0.0, atol=atol):
                    raise TNFRValueError(
                        "Hermitian operators require real eigenvalues.",
                        context={"max_imag": float(np.max(np.abs(imag))), "atol": atol},
                        suggestion="Ensure eigenvalues are real.",
                    )
            matrix_backend = _make_diagonal(eigvals_backend, backend=resolved_backend)
            eigenvalues_backend = eigvals_backend
        else:
            matrix_backend = _as_complex_matrix(operand, backend=resolved_backend)
            if ensure_hermitian and not self._check_hermitian(
                matrix_backend, atol=atol, backend=resolved_backend
            ):
                raise TNFRValueError(
                    "Spectral expectation operator must be Hermitian.",
                    context={"is_hermitian": False},
                    suggestion="Ensure the operator matrix is Hermitian.",
                )
            if ensure_hermitian:
                eigenvalues_backend, _ = resolved_backend.eigh(matrix_backend)
            else:
                eigenvalues_backend, _ = resolved_backend.eig(matrix_backend)

        self.backend = resolved_backend
        self._matrix_backend = matrix_backend
        self._eigenvalues_backend = eigenvalues_backend
        self.matrix = ensure_numpy(matrix_backend, backend=resolved_backend)
        self.eigenvalues = ensure_numpy(eigenvalues_backend, backend=resolved_backend)
        derived_c_min = float(np.min(self.eigenvalues.real))
        requested_floor = (
            expectation_floor
            if expectation_floor is not _C_MIN_UNSET
            else c_min
        )
        if requested_floor is _C_MIN_UNSET:
            self.c_min = derived_c_min
        else:
            self.c_min = float(requested_floor)

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
        backend: MathematicsBackend,
    ) -> bool:
        matrix_np = ensure_numpy(matrix, backend=backend)
        return bool(np.allclose(matrix_np, matrix_np.conj().T, atol=atol))

    def is_hermitian(self, *, atol: float = 1e-9) -> bool:
        """Return ``True`` when the operator matches its adjoint."""

        return self._check_hermitian(
            self._matrix_backend, atol=atol, backend=self.backend
        )

    def is_positive_semidefinite(self, *, atol: float = 1e-9) -> bool:
        """Check that all eigenvalues are non-negative within ``atol``."""

        return bool(np.all(self.eigenvalues.real >= -atol))

    def spectrum(self) -> ComplexVector:
        """Return the complex eigenvalue spectrum."""

        return np.asarray(self.eigenvalues, dtype=np.complex128)

    def spectral_radius(self) -> float:
        """Return the largest magnitude eigenvalue (spectral radius)."""

        return float(np.max(np.abs(self.eigenvalues)))

    def spectral_bandwidth(self) -> float:
        """Return the real bandwidth ``max(λ) - min(λ)``."""

        eigvals = self.eigenvalues.real
        return float(np.max(eigvals) - np.min(eigvals))

    def expectation(
        self,
        state: Sequence[complex] | np.ndarray,
        *,
        normalise: bool = True,
        atol: float = 1e-9,
    ) -> float:
        vector_backend = _as_complex_vector(state, backend=self.backend)
        if vector_backend.shape != (self.matrix.shape[0],):
            raise TNFRValueError(
                "State vector dimension mismatch with operator.",
                context={
                    "operator_shape": self.matrix.shape,
                    "vector_shape": vector_backend.shape,
                },
                suggestion="Ensure vector dimension matches operator dimension.",
            )
        working = vector_backend
        if normalise:
            norm_value = ensure_numpy(self.backend.norm(working), backend=self.backend)
            norm = float(norm_value)
            if np.isclose(norm, 0.0):
                raise TNFRValueError(
                    "Cannot normalise a null state vector.",
                    context={"norm": norm},
                    suggestion="Provide a non-zero state vector.",
                )
            working = working / norm
        column = working[..., None]
        bra = self.backend.conjugate_transpose(column)
        evolved = self.backend.matmul(self._matrix_backend, column)
        expectation_backend = self.backend.matmul(bra, evolved)
        expectation = ensure_numpy(expectation_backend, backend=self.backend)
        expectation_scalar = complex(np.asarray(expectation).reshape(()))
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

        return np.asarray(self.eigenvalues.real, dtype=float)

    def is_positive_semidefinite(self, *, atol: float = 1e-9) -> bool:
        """Frequency spectra must be non-negative to preserve νf semantics."""

        return bool(np.all(self.spectrum() >= -atol))

    def project_frequency(
        self,
        state: Sequence[complex] | np.ndarray,
        *,
        normalise: bool = True,
        atol: float = 1e-9,
    ) -> float:
        return self.expectation(state, normalise=normalise, atol=atol)
