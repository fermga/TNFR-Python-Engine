"""Spectral operators modelling coherence and frequency dynamics."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from ..compat.dataclass import dataclass
from .backend import MathematicsBackend, ensure_array, ensure_numpy, get_backend

if TYPE_CHECKING:  # pragma: no cover - typing imports only
    import numpy.typing as npt

    ComplexVector = npt.NDArray[np.complexfloating[np.float64, np.float64]]
    ComplexMatrix = npt.NDArray[np.complexfloating[np.float64, np.float64]]
else:  # pragma: no cover - runtime alias
    ComplexVector = np.ndarray
    ComplexMatrix = np.ndarray

__all__ = ["CoherenceOperator", "FrequencyOperator"]

DEFAULT_C_MIN: float = 0.1
_C_MIN_UNSET = object()


def _as_complex_vector(
    vector: Sequence[complex] | np.ndarray | Any,
    *,
    backend: MathematicsBackend,
) -> Any:
    arr = ensure_array(vector, dtype=np.complex128, backend=backend)
    if getattr(arr, "ndim", len(getattr(arr, "shape", ()))) != 1:
        raise ValueError("Vector input must be one-dimensional.")
    return arr


def _as_complex_matrix(
    matrix: Sequence[Sequence[complex]] | np.ndarray | Any,
    *,
    backend: MathematicsBackend,
) -> Any:
    arr = ensure_array(matrix, dtype=np.complex128, backend=backend)
    shape = getattr(arr, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Operator matrix must be square.")
    return arr


def _make_diagonal(values: Any, *, backend: MathematicsBackend) -> Any:
    dim = int(getattr(values, "shape")[0])
    identity = ensure_array(np.eye(dim, dtype=np.complex128), backend=backend)
    return backend.einsum("i,ij->ij", values, identity)


@dataclass(slots=True)
class CoherenceOperator:
    """Hermitian operator capturing coherence redistribution.

    The operator encapsulates how a TNFR EPI redistributes coherence across
    its spectral components.  It supports construction either from an explicit
    matrix expressed on the canonical basis or from a pre-computed list of
    eigenvalues (interpreted as already diagonalised).  The minimal eigenvalue
    ``c_min`` is tracked explicitly so structural stability thresholds are easy
    to evaluate during simulations.  The precedence for determining the stored
    threshold is: an explicit ``c_min`` wins, otherwise the spectral floor
    (minimum real eigenvalue) is used, with ``0.1`` acting as the canonical
    fallback for callers that still wish to supply a fixed number.

    When instantiated under an automatic differentiation backend (JAX, PyTorch)
    the spectral decomposition remains differentiable provided the supplied
    operator is non-defective.  NumPy callers receive ``numpy.ndarray`` outputs
    and all tolerance checks match the historical semantics.
    """

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
        ensure_hermitian: bool = True,
        atol: float = 1e-9,
        backend: MathematicsBackend | None = None,
    ) -> None:
        resolved_backend = backend or get_backend()
        operand = ensure_array(operator, dtype=np.complex128, backend=resolved_backend)
        if getattr(operand, "ndim", len(getattr(operand, "shape", ()))) == 1:
            eigvals_backend = _as_complex_vector(operand, backend=resolved_backend)
            if ensure_hermitian:
                imag = ensure_numpy(eigvals_backend.imag, backend=resolved_backend)
                if not np.allclose(imag, 0.0, atol=atol):
                    raise ValueError("Hermitian operators require real eigenvalues.")
            matrix_backend = _make_diagonal(eigvals_backend, backend=resolved_backend)
            eigenvalues_backend = eigvals_backend
        else:
            matrix_backend = _as_complex_matrix(operand, backend=resolved_backend)
            if ensure_hermitian and not self._check_hermitian(
                matrix_backend, atol=atol, backend=resolved_backend
            ):
                raise ValueError("Coherence operator must be Hermitian.")
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
        if c_min is _C_MIN_UNSET:
            self.c_min = derived_c_min
        else:
            self.c_min = float(c_min)

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

        return self._check_hermitian(self._matrix_backend, atol=atol, backend=self.backend)

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
            raise ValueError("State vector dimension mismatch with operator.")
        working = vector_backend
        if normalise:
            norm_value = ensure_numpy(self.backend.norm(working), backend=self.backend)
            norm = float(norm_value)
            if np.isclose(norm, 0.0):
                raise ValueError("Cannot normalise a null state vector.")
            working = working / norm
        column = working[..., None]
        bra = self.backend.conjugate_transpose(column)
        evolved = self.backend.matmul(self._matrix_backend, column)
        expectation_backend = self.backend.matmul(bra, evolved)
        expectation = ensure_numpy(expectation_backend, backend=self.backend)
        expectation_scalar = complex(np.asarray(expectation).reshape(()))
        if abs(expectation_scalar.imag) > atol:
            raise ValueError("Expectation value carries an imaginary component beyond tolerance.")
        eps = np.finfo(float).eps
        tol = max(1000.0, float(atol / eps)) if atol > 0 else 1000.0
        real_expectation = np.real_if_close(expectation_scalar, tol=tol)
        if np.iscomplexobj(real_expectation):
            raise ValueError("Expectation remained complex after coercion.")
        return float(real_expectation)


class FrequencyOperator(CoherenceOperator):
    """Operator encoding the structural frequency distribution.

    The frequency operator reuses the coherence machinery but enforces a real
    spectrum representing the structural hertz (νf) each mode contributes.  Its
    helpers therefore constrain outputs to the real axis and expose projections
    suited for telemetry collection.
    """

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
