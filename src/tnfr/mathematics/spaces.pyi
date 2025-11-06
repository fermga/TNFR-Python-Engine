from __future__ import annotations

import numpy as np
from .epi import BEPIElement as BEPIElement, _EPIValidators
from dataclasses import dataclass
from typing import Callable, Sequence

@dataclass(frozen=True)
class HilbertSpace:
    dimension: int
    dtype: np.dtype = ...
    def __post_init__(self) -> None: ...
    @property
    def basis(self) -> np.ndarray: ...
    def inner_product(
        self,
        vector_a: Sequence[complex] | np.ndarray,
        vector_b: Sequence[complex] | np.ndarray,
    ) -> complex: ...
    def norm(self, vector: Sequence[complex] | np.ndarray) -> float: ...
    def is_normalized(
        self, vector: Sequence[complex] | np.ndarray, *, atol: float = 1e-09
    ) -> bool: ...
    def project(
        self,
        vector: Sequence[complex] | np.ndarray,
        basis: Sequence[Sequence[complex] | np.ndarray] | None = None,
    ) -> np.ndarray: ...

class BanachSpaceEPI(_EPIValidators):
    def element(
        self,
        f_continuous: Sequence[complex] | np.ndarray,
        a_discrete: Sequence[complex] | np.ndarray,
        *,
        x_grid: Sequence[float] | np.ndarray,
    ) -> BEPIElement: ...
    def zero_element(
        self,
        *,
        continuous_size: int,
        discrete_size: int,
        x_grid: Sequence[float] | np.ndarray | None = None,
    ) -> BEPIElement: ...
    def canonical_basis(
        self,
        *,
        continuous_size: int,
        discrete_size: int,
        continuous_index: int = 0,
        discrete_index: int = 0,
        x_grid: Sequence[float] | np.ndarray | None = None,
    ) -> BEPIElement: ...
    def direct_sum(self, left: BEPIElement, right: BEPIElement) -> BEPIElement: ...
    def adjoint(self, element: BEPIElement) -> BEPIElement: ...
    def compose(
        self,
        element: BEPIElement,
        transform: Callable[[np.ndarray], np.ndarray],
        *,
        spectral_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> BEPIElement: ...
    def tensor_with_hilbert(
        self,
        element: BEPIElement,
        hilbert_space: HilbertSpace,
        vector: Sequence[complex] | np.ndarray | None = None,
    ) -> np.ndarray: ...
    def compute_coherence_functional(
        self,
        f_continuous: Sequence[complex] | np.ndarray,
        x_grid: Sequence[float] | np.ndarray,
    ) -> float: ...
    def coherence_norm(
        self,
        f_continuous: Sequence[complex] | np.ndarray,
        a_discrete: Sequence[complex] | np.ndarray,
        *,
        x_grid: Sequence[float] | np.ndarray,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ) -> float: ...
