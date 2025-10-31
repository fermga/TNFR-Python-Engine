"""Runtime helpers capturing TNFR spectral performance metrics."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import get_flags
from ..utils import get_logger
from .operators import CoherenceOperator, FrequencyOperator
from .spaces import HilbertSpace

__all__ = [
    "normalized",
    "coherence",
    "frequency_positive",
    "stable_unitary",
    "coherence_expectation",
    "frequency_expectation",
]


LOGGER = get_logger(__name__)


def _as_vector(state: Sequence[complex] | np.ndarray, *, dimension: int) -> np.ndarray:
    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1 or vector.shape[0] != dimension:
        raise ValueError(
            "State vector dimension mismatch: "
            f"expected ({dimension},), received {vector.shape!r}."
        )
    return vector


def _maybe_log(metric: str, payload: dict[str, object]) -> None:
    if not get_flags().log_performance:
        return
    LOGGER.debug("%s: %s", metric, payload)


def normalized(
    state: Sequence[complex] | np.ndarray,
    hilbert_space: HilbertSpace,
    *,
    atol: float = 1e-9,
    label: str = "state",
) -> tuple[bool, float]:
    """Return normalization status and norm for ``state``."""

    vector = _as_vector(state, dimension=hilbert_space.dimension)
    norm = hilbert_space.norm(vector)
    passed = bool(np.isclose(norm, 1.0, atol=atol))
    _maybe_log("normalized", {"label": label, "norm": norm, "passed": passed})
    return passed, float(norm)


def coherence_expectation(
    state: Sequence[complex] | np.ndarray,
    operator: CoherenceOperator,
    *,
    normalise: bool = True,
    atol: float = 1e-9,
) -> float:
    """Return the coherence expectation value for ``state``."""

    return float(operator.expectation(state, normalise=normalise, atol=atol))


def coherence(
    state: Sequence[complex] | np.ndarray,
    operator: CoherenceOperator,
    threshold: float,
    *,
    normalise: bool = True,
    atol: float = 1e-9,
    label: str = "state",
) -> tuple[bool, float]:
    """Evaluate coherence expectation against ``threshold``."""

    value = coherence_expectation(state, operator, normalise=normalise, atol=atol)
    passed = bool(value + atol >= threshold)
    _maybe_log(
        "coherence",
        {"label": label, "value": value, "threshold": threshold, "passed": passed},
    )
    return passed, value


def frequency_expectation(
    state: Sequence[complex] | np.ndarray,
    operator: FrequencyOperator,
    *,
    normalise: bool = True,
    atol: float = 1e-9,
) -> float:
    """Return the structural frequency projection for ``state``."""

    return float(operator.project_frequency(state, normalise=normalise, atol=atol))


def frequency_positive(
    state: Sequence[complex] | np.ndarray,
    operator: FrequencyOperator,
    *,
    normalise: bool = True,
    enforce: bool = True,
    atol: float = 1e-9,
    label: str = "state",
) -> dict[str, float | bool]:
    """Return summary ensuring structural frequency remains non-negative."""

    spectrum = operator.spectrum()
    spectrum_psd = bool(operator.is_positive_semidefinite(atol=atol))
    value = frequency_expectation(state, operator, normalise=normalise, atol=atol)
    projection_ok = bool(value + atol >= 0.0)
    passed = bool(spectrum_psd and (projection_ok or not enforce))
    summary = {
        "passed": passed,
        "value": value,
        "enforce": enforce,
        "spectrum_psd": spectrum_psd,
        "spectrum_min": float(np.min(spectrum)) if spectrum.size else float("inf"),
        "projection_passed": projection_ok,
    }
    _maybe_log("frequency_positive", {"label": label, **summary})
    return summary


def stable_unitary(
    state: Sequence[complex] | np.ndarray,
    operator: CoherenceOperator,
    hilbert_space: HilbertSpace,
    *,
    normalise: bool = True,
    atol: float = 1e-9,
    label: str = "state",
) -> tuple[bool, float]:
    """Return whether a one-step unitary preserves the Hilbert norm."""

    vector = _as_vector(state, dimension=hilbert_space.dimension)
    if normalise:
        norm = hilbert_space.norm(vector)
        if np.isclose(norm, 0.0, atol=atol):
            raise ValueError("Cannot normalise a null state vector.")
        vector = vector / norm
    eigenvalues, eigenvectors = np.linalg.eigh(operator.matrix)
    phases = np.exp(-1j * eigenvalues)
    unitary = (eigenvectors * phases) @ eigenvectors.conj().T
    evolved = unitary @ vector
    norm_after = hilbert_space.norm(evolved)
    passed = bool(np.isclose(norm_after, 1.0, atol=atol))
    _maybe_log("stable_unitary", {"label": label, "norm_after": norm_after, "passed": passed})
    return passed, float(norm_after)
