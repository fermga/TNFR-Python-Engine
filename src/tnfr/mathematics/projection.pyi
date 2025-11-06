from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Protocol
import numpy.typing as npt

__all__ = ["StateProjector", "BasicStateProjector"]

ComplexVector = npt.NDArray[np.complexfloating[np.float64, np.float64]]

class StateProjector(Protocol):
    def __call__(
        self,
        epi: float,
        nu_f: float,
        theta: float,
        dim: int,
        rng: np.random.Generator | None = None,
    ) -> ComplexVector: ...

@dataclass
class BasicStateProjector:
    dtype: np.dtype[np.complexfloating[np.float64, np.float64]] = ...
    atol: float = ...
    def __call__(
        self,
        epi: float,
        nu_f: float,
        theta: float,
        dim: int,
        rng: np.random.Generator | None = None,
    ) -> ComplexVector: ...
