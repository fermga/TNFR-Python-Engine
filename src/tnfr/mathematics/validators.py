"""Validation helpers for TNFR spectral states."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .operators import CoherenceOperator, FrequencyOperator
from .spaces import HilbertSpace


@dataclass(slots=True)
class NFRValidator:
    """Validate spectral states against TNFR canonical invariants.

    The validator orchestrates consistency checks that mirror the canonical
    constraints expected for a resonant TNFR state:

    * **Normalization** – the state must live on the Hilbert sphere so coherence
      measures remain meaningful.
    * **Coherence threshold** – the expectation value of the coherence operator
      must stay above a configurable structural bound to ensure the node remains
      phase-coupled to its environment.
    * **Frequency positivity** – when a frequency operator is supplied the
      projected structural hertz (νf) must stay non-negative.
    * **One-step unitary stability** – evolving the state through the
      coherence-driven unitary ``exp(-i·C)`` preserves normalization, signalling
      that the local dynamics do not collapse the node.
    """

    hilbert_space: HilbertSpace
    coherence_operator: CoherenceOperator
    coherence_threshold: float
    frequency_operator: FrequencyOperator | None = None
    atol: float = 1e-9

    def _as_vector(self, state: Sequence[complex] | np.ndarray) -> np.ndarray:
        vector = np.asarray(state, dtype=self.hilbert_space.dtype)
        expected_shape = (self.hilbert_space.dimension,)
        if vector.shape != expected_shape:
            raise ValueError(
                "State vector dimension mismatch: "
                f"expected {expected_shape}, received {vector.shape}."
            )
        return vector

    def _normalise(self, vector: np.ndarray) -> np.ndarray:
        norm = self.hilbert_space.norm(vector)
        if np.isclose(norm, 0.0, atol=self.atol):
            raise ValueError("Cannot normalise a null state vector.")
        return vector / norm

    def _unitary_step(self, vector: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(self.coherence_operator.matrix)
        phases = np.exp(-1j * eigenvalues)
        unitary = (eigenvectors * phases) @ eigenvectors.conj().T
        return unitary @ vector

    def validate_state(
        self,
        state: Sequence[complex] | np.ndarray,
        *,
        enforce_frequency_positivity: bool | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Validate ``state`` and return a success flag alongside diagnostics."""

        vector = self._as_vector(state)
        normalized = bool(self.hilbert_space.is_normalized(vector, atol=self.atol))
        normalised_vector = self._normalise(vector)

        expectation = self.coherence_operator.expectation(normalised_vector, normalise=False)
        coherence_value = float(expectation.real)
        coherence_ok = bool(coherence_value + self.atol >= self.coherence_threshold)

        frequency_summary: dict[str, Any] | None = None
        freq_ok = True
        if self.frequency_operator is not None:
            if enforce_frequency_positivity is None:
                enforce_frequency_positivity = True

            spectrum = self.frequency_operator.spectrum()
            spectrum_psd = bool(
                self.frequency_operator.is_positive_semidefinite(atol=self.atol)
            )
            min_frequency = float(np.min(spectrum)) if spectrum.size else float("inf")

            frequency_value = float(
                self.frequency_operator.project_frequency(normalised_vector, normalise=False)
            )
            projection_non_negative = bool(frequency_value + self.atol >= 0.0)

            freq_ok = bool(
                spectrum_psd
                and (projection_non_negative or not enforce_frequency_positivity)
            )

            frequency_summary = {
                "passed": bool(freq_ok),
                "value": frequency_value,
                "enforced": enforce_frequency_positivity,
                "spectrum_psd": spectrum_psd,
                "spectrum_min": min_frequency,
                "projection_passed": projection_non_negative,
            }
        elif enforce_frequency_positivity:
            raise ValueError("Frequency positivity enforcement requested without operator.")

        unitary_vector = self._unitary_step(normalised_vector)
        unitary_norm = self.hilbert_space.norm(unitary_vector)
        unitary_stable = bool(np.isclose(unitary_norm, 1.0, atol=self.atol))

        summary: dict[str, Any] = {
            "normalized": bool(normalized),
            "coherence": {
                "passed": bool(coherence_ok),
                "value": coherence_value,
                "threshold": self.coherence_threshold,
            },
            "frequency": frequency_summary,
            "unitary_stability": {
                "passed": bool(unitary_stable),
                "norm_after": unitary_norm,
            },
        }

        overall = bool(normalized and coherence_ok and freq_ok and unitary_stable)
        return overall, summary

    def report(self, summary: Mapping[str, Any]) -> str:
        """Return a human-readable report naming failed conditions."""

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
