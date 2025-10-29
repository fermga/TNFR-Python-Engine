"""Factory helpers to assemble TNFR coherence and frequency operators."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .operators import CoherenceOperator, FrequencyOperator

__all__ = [
    "build_coherence_operator",
    "build_frequency_operator",
    "as_coherence_operator",
    "as_frequency_operator",
]


def _as_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim not in (1, 2):
        raise ValueError("Operator specification must be 1D or 2D array-like.")
    if array.ndim == 2 and array.shape[0] != array.shape[1]:
        raise ValueError("Operator matrix must be square.")
    return array


def _symmetrise(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.conj().T)


def build_coherence_operator(
    operator: CoherenceOperator | Sequence[Sequence[complex]] | Sequence[complex] | np.ndarray,
    *,
    c_min: float | None = None,
    ensure_hermitian: bool = True,
    atol: float = 1e-9,
) -> CoherenceOperator:
    """Return a :class:`CoherenceOperator` from ``operator`` specification."""

    if isinstance(operator, CoherenceOperator):
        return operator
    array = _as_array(operator)
    if ensure_hermitian:
        array = _symmetrise(array)
    kwargs: dict[str, Any] = {"ensure_hermitian": ensure_hermitian, "atol": atol}
    if c_min is not None:
        kwargs["c_min"] = float(c_min)
    return CoherenceOperator(array, **kwargs)


def build_frequency_operator(
    operator: FrequencyOperator | Sequence[Sequence[complex]] | Sequence[complex] | np.ndarray,
    *,
    ensure_positive: bool = True,
    ensure_hermitian: bool = True,
    atol: float = 1e-9,
) -> FrequencyOperator:
    """Return a :class:`FrequencyOperator` ensuring Hermitian PSD behaviour."""

    if isinstance(operator, FrequencyOperator):
        return operator
    array = _as_array(operator)
    if ensure_hermitian:
        array = _symmetrise(array)
    if ensure_positive:
        array = array @ array.conj().T
    return FrequencyOperator(array, ensure_hermitian=True, atol=atol)


def as_coherence_operator(
    operator: CoherenceOperator | Sequence[Sequence[complex]] | Sequence[complex] | np.ndarray | None,
    params: Mapping[str, Any] | None = None,
) -> CoherenceOperator | None:
    """Coerce ``operator`` into a :class:`CoherenceOperator` when provided."""

    if operator is None:
        return None
    params = dict(params or {})
    return build_coherence_operator(operator, **params)


def as_frequency_operator(
    operator: FrequencyOperator | Sequence[Sequence[complex]] | Sequence[complex] | np.ndarray | None,
    params: Mapping[str, Any] | None = None,
) -> FrequencyOperator | None:
    """Coerce ``operator`` into a :class:`FrequencyOperator` when provided."""

    if operator is None:
        return None
    params = dict(params or {})
    return build_frequency_operator(operator, **params)
