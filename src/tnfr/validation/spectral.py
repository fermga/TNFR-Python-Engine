"""Spectral validation helpers aligned with the TNFR canonical interface."""

from __future__ import annotations

from ..compat.dataclass import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..mathematics.operators import CoherenceOperator, FrequencyOperator
from ..mathematics.spaces import HilbertSpace
from ..mathematics.runtime import (
    coherence as runtime_coherence,
    frequency_positive as runtime_frequency_positive,
    normalized as runtime_normalized,
    stable_unitary as runtime_stable_unitary,
)
from .base import ValidationOutcome, Validator

__all__ = ("NFRValidator",)


@dataclass(slots=True)
class NFRValidator(Validator[np.ndarray]):
    """Validate spectral states against TNFR canonical invariants."""

    hilbert_space: HilbertSpace
    coherence_operator: CoherenceOperator
    coherence_threshold: float
    frequency_operator: FrequencyOperator | None = None
    atol: float = 1e-9

    def _compute_summary(
        self,
        state: Sequence[complex] | np.ndarray,
        *,
        enforce_frequency_positivity: bool | None = None,
    ) -> tuple[bool, dict[str, Any], np.ndarray]:
        vector = self.hilbert_space.project(state)

        normalized_passed, norm_value = runtime_normalized(
            vector, self.hilbert_space, atol=self.atol
        )
        if np.isclose(norm_value, 0.0, atol=self.atol):
            raise ValueError("Cannot normalise a null state vector.")
        normalised_vector = vector / norm_value

        coherence_passed, coherence_value = runtime_coherence(
            normalised_vector,
            self.coherence_operator,
            self.coherence_threshold,
            normalise=False,
            atol=self.atol,
        )

        frequency_summary: dict[str, Any] | None = None
        freq_ok = True
        if self.frequency_operator is not None:
            if enforce_frequency_positivity is None:
                enforce_frequency_positivity = True

            runtime_summary = runtime_frequency_positive(
                normalised_vector,
                self.frequency_operator,
                normalise=False,
                enforce=enforce_frequency_positivity,
                atol=self.atol,
            )
            freq_ok = bool(runtime_summary["passed"])
            frequency_summary = {
                **runtime_summary,
                "enforced": runtime_summary["enforce"],
            }
            frequency_summary.pop("enforce", None)
        elif enforce_frequency_positivity:
            raise ValueError("Frequency positivity enforcement requested without operator.")

        unitary_passed, unitary_norm = runtime_stable_unitary(
            normalised_vector,
            self.coherence_operator,
            self.hilbert_space,
            normalise=False,
            atol=self.atol,
        )

        summary: dict[str, Any] = {
            "normalized": bool(normalized_passed),
            "coherence": {
                "passed": bool(coherence_passed),
                "value": coherence_value,
                "threshold": self.coherence_threshold,
            },
            "frequency": frequency_summary,
            "unitary_stability": {
                "passed": bool(unitary_passed),
                "norm_after": unitary_norm,
            },
        }

        overall = bool(normalized_passed and coherence_passed and freq_ok and unitary_passed)
        return overall, summary, normalised_vector

    def validate(
        self,
        subject: Sequence[complex] | np.ndarray,
        /,
        *,
        enforce_frequency_positivity: bool | None = None,
    ) -> ValidationOutcome[np.ndarray]:
        """Return :class:`ValidationOutcome` for ``subject``."""

        overall, summary, normalised_vector = self._compute_summary(
            subject, enforce_frequency_positivity=enforce_frequency_positivity
        )
        artifacts = {"normalised_state": normalised_vector}
        return ValidationOutcome(
            subject=normalised_vector,
            passed=overall,
            summary=summary,
            artifacts=artifacts,
        )

    def validate_state(
        self,
        state: Sequence[complex] | np.ndarray,
        *,
        enforce_frequency_positivity: bool | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Backward compatible validation returning ``(passed, summary)``."""

        overall, summary, _ = self._compute_summary(
            state, enforce_frequency_positivity=enforce_frequency_positivity
        )
        return overall, summary

    def report(self, outcome: ValidationOutcome[np.ndarray]) -> str:
        """Return a human-readable report naming failed conditions."""

        summary = outcome.summary
        failed_checks: list[str] = []
        if not summary.get("normalized", False):
            failed_checks.append("normalization")

        coherence_summary = summary.get("coherence", {})
        if not coherence_summary.get("passed", False):
            failed_checks.append("coherence threshold")

        frequency_summary = summary.get("frequency")
        if isinstance(frequency_summary, Mapping) and not frequency_summary.get("passed", False):
            failed_checks.append("frequency positivity")

        unitary_summary = summary.get("unitary_stability", {})
        if not unitary_summary.get("passed", False):
            failed_checks.append("unitary stability")

        if not failed_checks:
            return "All validation checks passed."
        return "Failed checks: " + ", ".join(failed_checks) + "."
