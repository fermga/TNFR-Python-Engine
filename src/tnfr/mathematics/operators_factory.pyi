from __future__ import annotations

import numpy as np
from .operators import CoherenceOperator, FrequencyOperator

__all__ = ["make_coherence_operator", "make_frequency_operator"]

def make_coherence_operator(
    dim: int, *, spectrum: np.ndarray | None = None, c_min: float = 0.1
) -> CoherenceOperator: ...
def make_frequency_operator(matrix: np.ndarray) -> FrequencyOperator: ...
