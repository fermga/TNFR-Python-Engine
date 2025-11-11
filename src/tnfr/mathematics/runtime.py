"""Runtime helpers capturing TNFR spectral performance metrics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..config import get_flags
from ..utils import get_logger
from .backend import ensure_array, ensure_numpy, get_backend
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


def _as_vector(
    state: Sequence[complex] | np.ndarray,
    *,
    dimension: int,
    backend=None,
) -> Any:
    resolved_backend = backend or get_backend()
    vector = ensure_array(state, dtype=np.complex128, backend=resolved_backend)
    if (
        getattr(vector, "ndim", len(getattr(vector, "shape", ()))) != 1
        or vector.shape[0] != dimension
    ):
        raise ValueError(
            "State vector dimension mismatch: "
            f"expected ({dimension},), received {vector.shape!r}."
        )
    return vector


def _resolve_operator_backend(operator: CoherenceOperator) -> tuple[Any, Any]:
    backend = getattr(operator, "backend", None) or get_backend()
    matrix_backend = getattr(operator, "_matrix_backend", None)
    if matrix_backend is None:
        matrix_backend = ensure_array(operator.matrix, dtype=np.complex128, backend=backend)
    return backend, matrix_backend


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

    backend = get_backend()
    vector = _as_vector(state, dimension=hilbert_space.dimension, backend=backend)
    norm_backend = backend.norm(vector)
    norm = float(np.asarray(ensure_numpy(norm_backend, backend=backend)))
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

    backend, matrix_backend = _resolve_operator_backend(operator)
    vector = _as_vector(state, dimension=hilbert_space.dimension, backend=backend)
    if normalise:
        norm_backend = backend.norm(vector)
        norm = float(np.asarray(ensure_numpy(norm_backend, backend=backend)))
        if np.isclose(norm, 0.0, atol=atol):
            raise ValueError("Cannot normalise a null state vector.")
        vector = vector / norm
    generator = -1j * matrix_backend
    unitary = backend.matrix_exp(generator)
    evolved_backend = backend.matmul(unitary, vector[..., None]).reshape((hilbert_space.dimension,))
    evolved = np.asarray(ensure_numpy(evolved_backend, backend=backend))
    norm_after = hilbert_space.norm(evolved)
    passed = bool(np.isclose(norm_after, 1.0, atol=atol))
    _maybe_log("stable_unitary", {"label": label, "norm_after": norm_after, "passed": passed})
    return passed, float(norm_after)
