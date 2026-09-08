from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .spaces import BanachSpaceEPI

COMPOSITE_EPI_REGULARITY_KIND: str
COMPOSITE_EPI_REGULARITY_PROVENANCE: str

__all__ = [
    "BEPIElement",
    "COMPOSITE_EPI_REGULARITY_KIND",
    "COMPOSITE_EPI_REGULARITY_PROVENANCE",
    "CompositeEPIRegularityEvaluation",
    "evaluate_composite_epi_regularity_transform",
    "CoherenceEvaluation",
    "evaluate_coherence_transform",
]

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
    def real_scalar_embedding(self) -> float | None: ...
    def scalar_projection(self) -> float: ...
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
class CompositeEPIRegularityEvaluation:
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
    metric_kind: str = ...
    provenance: str = ...
    @property
    def coherence_before(self) -> float: ...
    @property
    def coherence_after(self) -> float: ...

CoherenceEvaluation = CompositeEPIRegularityEvaluation

def evaluate_composite_epi_regularity_transform(
    element: BEPIElement,
    transform: Callable[[BEPIElement], BEPIElement],
    *,
    kappa: float = 1.0,
    tolerance: float = 1e-09,
    space: BanachSpaceEPI | None = None,
    regularity_kwargs: Mapping[str, float] | None = None,
) -> CompositeEPIRegularityEvaluation: ...

def evaluate_coherence_transform(
    element: BEPIElement,
    transform: Callable[[BEPIElement], BEPIElement],
    *,
    kappa: float = 1.0,
    tolerance: float = 1e-09,
    space: BanachSpaceEPI | None = None,
    norm_kwargs: Mapping[str, float] | None = None,
) -> CompositeEPIRegularityEvaluation: ...
