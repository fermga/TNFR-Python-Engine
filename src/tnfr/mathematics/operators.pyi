from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from .backend import MathematicsBackend

__all__ = ["SpectralExpectationOperator", "CoherenceOperator", "FrequencyOperator"]

ComplexMatrix = npt.NDArray[np.complexfloating[np.float64, np.float64]]
ComplexVector = npt.NDArray[np.complexfloating[np.float64, np.float64]]

@dataclass
class SpectralExpectationOperator:
    matrix: ComplexMatrix
    eigenvalues: ComplexVector
    c_min: float
    metric_kind: str
    canonical_coherence_certified: bool
    canonical_history_key: None
    backend: MathematicsBackend = field(init=False, repr=False)
    def __init__(
        self,
        operator: Sequence[Sequence[complex]] | Sequence[complex] | np.ndarray | Any,
        *,
        c_min: float | object = ...,
        expectation_floor: float | object = ...,
        ensure_hermitian: bool = True,
        atol: float = 1e-09,
        backend: MathematicsBackend | None = None,
    ) -> None: ...
    @property
    def expectation_floor(self) -> float: ...
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

CoherenceOperator = SpectralExpectationOperator

class FrequencyOperator(SpectralExpectationOperator):
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
