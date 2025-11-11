from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..mathematics.operators import CoherenceOperator, FrequencyOperator
from ..mathematics.spaces import HilbertSpace
from . import ValidationOutcome, Validator

class NFRValidator(Validator[np.ndarray]):
    hilbert_space: HilbertSpace
    coherence_operator: CoherenceOperator
    coherence_threshold: float
    frequency_operator: FrequencyOperator | None
    atol: float

    def __init__(
        self,
        hilbert_space: HilbertSpace,
        coherence_operator: CoherenceOperator,
        *,
        coherence_threshold: float,
        frequency_operator: FrequencyOperator | None = ...,
        atol: float = ...,
    ) -> None: ...
    def validate(
        self,
        subject: Sequence[complex] | np.ndarray,
        /,
        *,
        enforce_frequency_positivity: bool | None = ...,
    ) -> ValidationOutcome[np.ndarray]: ...  # type: ignore[override]
    def validate_state(
        self,
        state: Sequence[complex] | np.ndarray,
        *,
        enforce_frequency_positivity: bool | None = ...,
    ) -> tuple[bool, Mapping[str, Any]]: ...
    def report(self, outcome: ValidationOutcome[np.ndarray]) -> str: ...

__all__ = ("NFRValidator",)
