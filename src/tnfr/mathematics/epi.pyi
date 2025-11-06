from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

__all__ = ["BEPIElement", "CoherenceEvaluation", "evaluate_coherence_transform"]

class _EPIValidators:
    @classmethod
    def validate_domain(
        cls,
        f_continuous: Sequence[complex] | np.ndarray,
        a_discrete: Sequence[complex] | np.ndarray,
        x_grid: Sequence[float] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]: ...

@dataclass(frozen=True)
class BEPIElement(_EPIValidators):
    f_continuous: Sequence[complex] | np.ndarray
    a_discrete: Sequence[complex] | np.ndarray
    x_grid: Sequence[float] | np.ndarray
    def __post_init__(self) -> None: ...
    def direct_sum(self, other: BEPIElement) -> BEPIElement: ...
    def tensor(self, vector: Sequence[complex] | np.ndarray) -> np.ndarray: ...
    def adjoint(self) -> BEPIElement: ...
    def compose(
        self,
        transform: Callable[[np.ndarray], np.ndarray],
        *,
        spectral_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> BEPIElement: ...
    def __float__(self) -> float: ...
    def __abs__(self) -> float: ...
    def __add__(self, other: BEPIElement | float | int) -> BEPIElement: ...
    def __radd__(self, other: float | int) -> BEPIElement: ...
    def __sub__(self, other: BEPIElement | float | int) -> BEPIElement: ...
    def __rsub__(self, other: float | int) -> BEPIElement: ...
    def __mul__(self, other: float | int) -> BEPIElement: ...
    def __rmul__(self, other: float | int) -> BEPIElement: ...
    def __truediv__(self, other: float | int) -> BEPIElement: ...
    def __eq__(self, other: object) -> bool: ...

@dataclass(frozen=True)
class CoherenceEvaluation:
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
    tolerance: float = 1e-09,
    space: BanachSpaceEPI | None = None,
    norm_kwargs: Mapping[str, float] | None = None,
) -> CoherenceEvaluation: ...
