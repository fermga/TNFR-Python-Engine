from __future__ import annotations

from typing import Sequence

import numpy as np

from .operators import CoherenceOperator, SpectralExpectationOperator

__all__ = ["spectral_weighted_angle", "dcoh"]

def spectral_weighted_angle(
    psi1: Sequence[complex] | np.ndarray,
    psi2: Sequence[complex] | np.ndarray,
    operator: SpectralExpectationOperator,
    *,
    normalise: bool = True,
    atol: float = 1e-09,
) -> float: ...

def dcoh(
    psi1: Sequence[complex] | np.ndarray,
    psi2: Sequence[complex] | np.ndarray,
    operator: CoherenceOperator,
    *,
    normalise: bool = True,
    atol: float = 1e-09,
) -> float: ...