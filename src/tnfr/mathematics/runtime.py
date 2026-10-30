"""Runtime helpers for auxiliary spectral observables."""

from __future__ import annotations

from typing import Any, Sequence

from .._spectral_expectation import finite_spectral_real
from ..config import get_flags
from ..errors import TNFRValueError
from ..utils import get_logger
from ._complex_arrays import (
    backend_complex_array,
    finite_complex_norm,
    nonnegative_tolerance,
    normalized_complex_vector,
)
from .backend import get_backend
from .operators import CoherenceOperator, FrequencyOperator, SpectralExpectationOperator
from .spaces import HilbertSpace
from .unified_numerical import np

__all__ = [
    "normalized",
    "spectral_operator_expectation",
    "meets_spectral_expectation_threshold",
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
    vector = backend_complex_array(
        state, backend=resolved_backend, label="State vector"
    )
    if (
        getattr(vector, "ndim", len(getattr(vector, "shape", ()))) != 1
        or vector.shape[0] != dimension
    ):
        raise TNFRValueError(
            "State vector dimension mismatch: "
            f"expected ({dimension},), received {vector.shape!r}.",
            context={"expected_dimension": dimension, "actual_shape": vector.shape},
            suggestion="Ensure the state vector matches the expected dimension.",
        )
    return vector


def _resolve_operator_backend(operator: CoherenceOperator) -> tuple[Any, Any]:
    backend = getattr(operator, "backend", None) or get_backend()
    matrix_backend = getattr(operator, "_matrix_backend", None)
    if matrix_backend is None:
        matrix_backend = backend_complex_array(
            operator.matrix, backend=backend, label="Spectral operator matrix"
        )
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
    """Return unit-norm status and a finite standard Euclidean observation.

    ``hilbert_space`` supplies the vector dimension; this observation uses the
    standard complex coordinate norm, independent of its materialization dtype.
    """

    tolerance = nonnegative_tolerance(atol)
    backend = get_backend()
    vector = _as_vector(state, dimension=hilbert_space.dimension, backend=backend)
    norm = finite_complex_norm(vector, backend=backend)
    passed = bool(np.isclose(norm, 1.0, atol=tolerance))
    _maybe_log("normalized", {"label": label, "norm": norm, "passed": passed})
    return passed, float(norm)


def spectral_operator_expectation(
    state: Sequence[complex] | np.ndarray,
    operator: SpectralExpectationOperator,
    *,
    normalise: bool = True,
    atol: float = 1e-9,
) -> float:
    r"""Return the auxiliary Hermitian expectation ``<psi|A|psi>``.

    The result is unbounded and does not represent canonical structural
    ``C(t)``.  This helper never writes ``history['C_steps']``.
    """

    return float(operator.expectation(state, normalise=normalise, atol=atol))


def meets_spectral_expectation_threshold(
    state: Sequence[complex] | np.ndarray,
    operator: SpectralExpectationOperator,
    threshold: float,
    *,
    normalise: bool = True,
    atol: float = 1e-9,
    label: str = "state",
) -> tuple[bool, float]:
    """Compare an auxiliary spectral expectation with a finite real floor."""

    tolerance = nonnegative_tolerance(atol)
    floor = finite_spectral_real(threshold, label="Spectral expectation threshold")
    value = spectral_operator_expectation(
        state, operator, normalise=normalise, atol=tolerance
    )
    passed = bool(value + tolerance >= floor)
    _maybe_log(
        "spectral_operator_expectation",
        {
            "label": label,
            "value": value,
            "threshold": floor,
            "passed": passed,
            "canonical_coherence_certified": False,
        },
    )
    return passed, value


def coherence_expectation(
    state: Sequence[complex] | np.ndarray,
    operator: CoherenceOperator,
    *,
    normalise: bool = True,
    atol: float = 1e-9,
) -> float:
    """Compatibility alias for :func:`spectral_operator_expectation`."""

    return spectral_operator_expectation(
        state, operator, normalise=normalise, atol=atol
    )


def coherence(
    state: Sequence[complex] | np.ndarray,
    operator: CoherenceOperator,
    threshold: float,
    *,
    normalise: bool = True,
    atol: float = 1e-9,
    label: str = "state",
) -> tuple[bool, float]:
    """Compatibility alias for an auxiliary spectral-threshold comparison."""

    return meets_spectral_expectation_threshold(
        state,
        operator,
        threshold,
        normalise=normalise,
        atol=atol,
        label=label,
    )


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

    tolerance = nonnegative_tolerance(atol)
    spectrum = operator.spectrum()
    spectrum_psd = bool(operator.is_positive_semidefinite(atol=tolerance))
    value = frequency_expectation(state, operator, normalise=normalise, atol=tolerance)
    projection_ok = bool(value + tolerance >= 0.0)
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
    operator: SpectralExpectationOperator,
    hilbert_space: HilbertSpace,
    *,
    normalise: bool = True,
    atol: float = 1e-9,
    label: str = "state",
) -> tuple[bool, float]:
    """Observe whether one unitary step ends with unit Euclidean norm.

    Native backend arithmetic retains the differentiable state and operator;
    the returned Boolean and float are detached observations. With
    ``normalise=False``, the input is left at its supplied amplitude.
    ``hilbert_space`` supplies the dimension of the standard coordinate norm.
    """

    tolerance = nonnegative_tolerance(atol)
    backend, matrix_backend = _resolve_operator_backend(operator)
    vector = _as_vector(state, dimension=hilbert_space.dimension, backend=backend)
    if normalise:
        vector = normalized_complex_vector(
            vector, backend=backend, atol=tolerance, label="state vector"
        )
    generator = -1j * matrix_backend
    unitary = backend.matrix_exp(generator)
    evolved_backend = backend.matmul(unitary, vector[..., None]).reshape(
        (hilbert_space.dimension,)
    )
    norm_after = finite_complex_norm(evolved_backend, backend=backend)
    passed = bool(np.isclose(norm_after, 1.0, atol=tolerance))
    _maybe_log(
        "stable_unitary", {"label": label, "norm_after": norm_after, "passed": passed}
    )
    return passed, float(norm_after)
