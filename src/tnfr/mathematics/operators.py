"""Spectral operators modelling coherence and frequency dynamics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing imports only
    import numpy.typing as npt

    ComplexVector = npt.NDArray[np.complexfloating[np.float64, np.float64]]
    ComplexMatrix = npt.NDArray[np.complexfloating[np.float64, np.float64]]
else:  # pragma: no cover - runtime alias
    ComplexVector = np.ndarray
    ComplexMatrix = np.ndarray

__all__ = ["CoherenceOperator", "FrequencyOperator"]


def _as_complex_vector(vector: Sequence[complex] | np.ndarray) -> ComplexVector:
    arr = np.asarray(vector, dtype=np.complex128)
    if arr.ndim != 1:
        raise ValueError("Vector input must be one-dimensional.")
    return arr


def _as_complex_matrix(matrix: Sequence[Sequence[complex]] | np.ndarray) -> ComplexMatrix:
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Operator matrix must be square.")
    return arr


@dataclass(slots=True)
class CoherenceOperator:
    """Hermitian operator capturing coherence redistribution.

    The operator encapsulates how a TNFR EPI redistributes coherence across
    its spectral components.  It supports construction either from an explicit
    matrix expressed on the canonical basis or from a pre-computed list of
    eigenvalues (interpreted as already diagonalised).  The minimal eigenvalue
    ``c_min`` is tracked explicitly so structural stability thresholds are easy
    to evaluate during simulations.
    """

    matrix: ComplexMatrix
    eigenvalues: ComplexVector
    c_min: float

    def __init__(
        self,
        operator: Sequence[Sequence[complex]] | Sequence[complex] | np.ndarray,
        *,
        ensure_hermitian: bool = True,
        atol: float = 1e-9,
    ) -> None:
        if np.ndim(operator) == 1:
            eigvals = _as_complex_vector(operator)
            if ensure_hermitian and not np.allclose(eigvals.imag, 0.0, atol=atol):
                raise ValueError("Hermitian operators require real eigenvalues.")
            self.matrix = np.diag(eigvals)
            self.eigenvalues = eigvals
        else:
            matrix = _as_complex_matrix(operator)
            if ensure_hermitian and not self._check_hermitian(matrix, atol=atol):
                raise ValueError("Coherence operator must be Hermitian.")
            self.matrix = matrix
            if ensure_hermitian:
                self.eigenvalues = np.linalg.eigvalsh(self.matrix)
            else:
                self.eigenvalues = np.linalg.eigvals(self.matrix)
        self.c_min = float(np.min(self.eigenvalues.real))

    @staticmethod
    def _check_hermitian(matrix: ComplexMatrix, *, atol: float = 1e-9) -> bool:
        return np.allclose(matrix, matrix.conj().T, atol=atol)

    def is_hermitian(self, *, atol: float = 1e-9) -> bool:
        """Return ``True`` when the operator matches its adjoint."""

        return self._check_hermitian(self.matrix, atol=atol)

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
    ) -> complex:
        vector = _as_complex_vector(state)
        if vector.shape != (self.matrix.shape[0],):
            raise ValueError("State vector dimension mismatch with operator.")
        if normalise:
            norm = np.linalg.norm(vector)
            if np.isclose(norm, 0.0):
                raise ValueError("Cannot normalise a null state vector.")
            vector = vector / norm
        return vector.conj().T @ (self.matrix @ vector)


class FrequencyOperator(CoherenceOperator):
    """Operator encoding the structural frequency distribution.

    The frequency operator reuses the coherence machinery but enforces a real
    spectrum representing the structural hertz (νf) each mode contributes.  Its
    helpers therefore constrain outputs to the real axis and expose projections
    suited for telemetry collection.
    """

    def __init__(
        self,
        operator: Sequence[Sequence[complex]] | Sequence[complex] | np.ndarray,
        *,
        ensure_hermitian: bool = True,
        atol: float = 1e-9,
    ) -> None:
        super().__init__(operator, ensure_hermitian=ensure_hermitian, atol=atol)

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
    ) -> float:
        expectation = self.expectation(state, normalise=normalise)
        return float(expectation.real)
