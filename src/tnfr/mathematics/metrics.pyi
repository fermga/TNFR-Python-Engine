from __future__ import annotations

import numpy as np
from .operators import CoherenceOperator
from typing import Sequence

__all__ = ["dcoh"]

def dcoh(
    psi1: Sequence[complex] | np.ndarray,
    psi2: Sequence[complex] | np.ndarray,
    operator: CoherenceOperator,
    *,
    normalise: bool = True,
    atol: float = 1e-09,
) -> float: ...
