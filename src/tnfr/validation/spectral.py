"""Validation of auxiliary Hermitian spectral expectations.

These checks are separate from canonical structural coherence ``C(t)`` and
never certify or populate ``history['C_steps']``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..compat.dataclass import dataclass
from ..errors import TNFRValueError
from ..mathematics.operators import CoherenceOperator, FrequencyOperator
from ..mathematics.runtime import (
    meets_spectral_expectation_threshold as runtime_spectral_threshold,
)
from ..mathematics.runtime import frequency_positive as runtime_frequency_positive
from ..mathematics.runtime import normalized as runtime_normalized
from ..mathematics.runtime import stable_unitary as runtime_stable_unitary
from ..mathematics.spaces import HilbertSpace
from ..mathematics.unified_numerical import np
from .base import ValidationOutcome, Validator

__all__ = ("NFRValidator",)


@dataclass(slots=True)
class NFRValidator(Validator[np.ndarray]):
    """Validate a normalized state against auxiliary spectral contracts.

    ``coherence_operator`` and ``coherence_threshold`` are retained public
    compatibility names.  Their canonical meanings are ``spectral_operator``
    and ``spectral_expectation_threshold``; the resulting expectation is an
    unbounded Hermitian observable and is not structural ``C(t)``.
    """

    hilbert_space: HilbertSpace
    coherence_operator: CoherenceOperator
    coherence_threshold: float
    frequency_operator: FrequencyOperator | None = None
    atol: float = 1e-9

    @property
    def spectral_operator(self) -> CoherenceOperator:
        """Return the operator under its canonical auxiliary name."""

        return self.coherence_operator

    @property
    def spectral_expectation_threshold(self) -> float:
        """Return the unbounded auxiliary comparison floor."""

        return self.coherence_threshold

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
            raise TNFRValueError(
                "Cannot normalise a null state vector.",
                context={"norm_value": float(norm_value)},
                suggestion="Ensure the state vector is non-zero.",
            )
        normalised_vector = vector / norm_value

        expectation_passed, expectation_value = runtime_spectral_threshold(
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
            raise TNFRValueError(
                "Frequency positivity enforcement requested without operator.",
                context={"enforce_frequency_positivity": enforce_frequency_positivity},
                suggestion="Provide a frequency_operator to NFRValidator.",
            )

        unitary_passed, unitary_norm = runtime_stable_unitary(
            normalised_vector,
            self.coherence_operator,
            self.hilbert_space,
            normalise=False,
            atol=self.atol,
        )

        expectation_summary: dict[str, Any] = {
            "passed": bool(expectation_passed),
            "value": expectation_value,
            "threshold": self.coherence_threshold,
            "metric_kind": "spectral_operator_expectation",
            "range": "unbounded_real",
            "canonical_coherence_certified": False,
            "records_to_C_steps": False,
        }
        summary: dict[str, Any] = {
            "normalized": bool(normalized_passed),
            "spectral_operator_expectation": expectation_summary,
            # Historical result key retained for callers.  It points to the
            # explicitly scoped auxiliary payload above, never to C(t).
            "coherence": expectation_summary,
            "frequency": frequency_summary,
            "unitary_stability": {
                "passed": bool(unitary_passed),
                "norm_after": unitary_norm,
            },
        }

        overall = bool(
            normalized_passed and expectation_passed and freq_ok and unitary_passed
        )
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

        expectation_summary = summary.get("spectral_operator_expectation", {})
        if not expectation_summary.get("passed", False):
            failed_checks.append("spectral expectation threshold")

        frequency_summary = summary.get("frequency")
        if isinstance(frequency_summary, Mapping) and not frequency_summary.get(
            "passed", False
        ):
            failed_checks.append("frequency positivity")

        unitary_summary = summary.get("unitary_stability", {})
        if not unitary_summary.get("passed", False):
            failed_checks.append("unitary stability")

        if not failed_checks:
            return "All validation checks passed."
        return "Failed checks: " + ", ".join(failed_checks) + "."
