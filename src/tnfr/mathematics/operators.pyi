from __future__ import annotations

import numpy as np
from .backend import MathematicsBackend
from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy.typing as npt

__all__ = ["CoherenceOperator", "FrequencyOperator"]

ComplexMatrix = npt.NDArray[np.complexfloating[np.float64, np.float64]]
ComplexVector = npt.NDArray[np.complexfloating[np.float64, np.float64]]

@dataclass
class CoherenceOperator:
    matrix: ComplexMatrix
    eigenvalues: ComplexVector
    c_min: float
    backend: MathematicsBackend = field(init=False, repr=False)
    def __init__(
        self,
        operator: Sequence[Sequence[complex]] | Sequence[complex] | np.ndarray | Any,
        *,
        c_min: float | object = ...,
        ensure_hermitian: bool = True,
        atol: float = 1e-09,
        backend: MathematicsBackend | None = None,
    ) -> None: ...
    def is_hermitian(self, *, atol: float = 1e-09) -> bool: ...
    def is_positive_semidefinite(self, *, atol: float = 1e-09) -> bool: ...
    def spectrum(self) -> ComplexVector: ...
    def spectral_radius(self) -> float: ...
    def spectral_bandwidth(self) -> float: ...
    def expectation(
        self,
        state: Sequence[complex] | np.ndarray,
        *,
        normalise: bool = True,
        atol: float = 1e-09,
    ) -> float: ...

class FrequencyOperator(CoherenceOperator):
    def __init__(
        self,
        operator: Sequence[Sequence[complex]] | Sequence[complex] | np.ndarray | Any,
        *,
        ensure_hermitian: bool = True,
        atol: float = 1e-09,
        backend: MathematicsBackend | None = None,
    ) -> None: ...
    def spectrum(self) -> np.ndarray: ...
    def is_positive_semidefinite(self, *, atol: float = 1e-09) -> bool: ...
    def project_frequency(
        self,
        state: Sequence[complex] | np.ndarray,
        *,
        normalise: bool = True,
        atol: float = 1e-09,
    ) -> float: ...
