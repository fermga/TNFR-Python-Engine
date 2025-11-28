from __future__ import annotations

import numpy as np
from .operators import CoherenceOperator, FrequencyOperator
from .spaces import HilbertSpace
from typing import Sequence

__all__ = [
    "normalized",
    "coherence",
    "frequency_positive",
    "stable_unitary",
    "coherence_expectation",
    "frequency_expectation",
]

def normalized(
    state: Sequence[complex] | np.ndarray,
    hilbert_space: HilbertSpace,
    *,
    atol: float = 1e-09,
    label: str = "state",
) -> tuple[bool, float]: ...
def coherence_expectation(
    state: Sequence[complex] | np.ndarray,
    operator: CoherenceOperator,
    *,
    normalise: bool = True,
    atol: float = 1e-09,
) -> float: ...
def coherence(
    state: Sequence[complex] | np.ndarray,
    operator: CoherenceOperator,
    threshold: float,
    *,
    normalise: bool = True,
    atol: float = 1e-09,
    label: str = "state",
) -> tuple[bool, float]: ...
def frequency_expectation(
    state: Sequence[complex] | np.ndarray,
    operator: FrequencyOperator,
    *,
    normalise: bool = True,
    atol: float = 1e-09,
) -> float: ...
def frequency_positive(
    state: Sequence[complex] | np.ndarray,
    operator: FrequencyOperator,
    *,
    normalise: bool = True,
    enforce: bool = True,
    atol: float = 1e-09,
    label: str = "state",
) -> dict[str, float | bool]: ...
def stable_unitary(
    state: Sequence[complex] | np.ndarray,
    operator: CoherenceOperator,
    hilbert_space: HilbertSpace,
    *,
    normalise: bool = True,
    atol: float = 1e-09,
    label: str = "state",
) -> tuple[bool, float]: ...
