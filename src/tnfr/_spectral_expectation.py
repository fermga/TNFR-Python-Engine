"""Shared contracts for auxiliary spectral-expectation telemetry.

Spectral expectations are real Hermitian observables whose range is determined
by the operator spectrum. They are not canonical structural coherence C(t).
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

from .errors import TNFRValueError

SPECTRAL_EXPECTATION_METRIC_KIND = "spectral_operator_expectation"
SPECTRAL_EXPECTATION_RANGE = "unbounded_real"


def finite_spectral_real(value: Any, *, label: str) -> float:
    """Return a finite non-boolean real used by spectral telemetry."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TNFRValueError(f"{label} must be a finite real scalar")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(
            f"{label} must be a finite real scalar"
        ) from exc
    if not math.isfinite(result):
        raise TNFRValueError(f"{label} must be finite")
    return result


def positive_spectral_dimension(value: Any, *, label: str) -> int:
    """Return a strict positive integer Hilbert dimension."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TNFRValueError(f"{label} must be a positive integer")
    resolved = int(value)
    if resolved <= 0:
        raise TNFRValueError(f"{label} must be a positive integer")
    return resolved


def validate_spectral_operator(
    operator: Any,
    *,
    dimension: int | None = None,
    label: str = "spectral_operator",
) -> Any:
    """Validate one finite Hermitian-observable instance and its dimension."""

    from .mathematics.operators import SpectralExpectationOperator

    if not isinstance(operator, SpectralExpectationOperator):
        raise TNFRValueError(
            f"{label} must be a SpectralExpectationOperator"
        )

    matrix_shape = tuple(getattr(operator.matrix, "shape", ()))
    if len(matrix_shape) != 2 or matrix_shape[0] != matrix_shape[1]:
        raise TNFRValueError(f"{label} must be a square operator")
    if not operator.is_hermitian():
        raise TNFRValueError(f"{label} must be Hermitian")
    if dimension is not None:
        resolved_dimension = positive_spectral_dimension(
            dimension, label="Hilbert space dimension"
        )
        if matrix_shape != (resolved_dimension, resolved_dimension):
            raise TNFRValueError(
                f"{label} dimension must match the Hilbert space dimension"
            )

    for eigenvalue in operator.eigenvalues:
        try:
            scalar = complex(eigenvalue)
        except (OverflowError, TypeError, ValueError) as exc:
            raise TNFRValueError(
                f"{label} spectrum must contain finite scalars"
            ) from exc
        if not (math.isfinite(scalar.real) and math.isfinite(scalar.imag)):
            raise TNFRValueError(f"{label} spectrum must be finite")

    finite_spectral_real(
        operator.expectation_floor,
        label=f"{label} expectation floor",
    )
    return operator


def resolve_compatibility_value(
    canonical: Any,
    legacy: Any,
    *,
    canonical_name: str,
    legacy_name: str,
) -> Any:
    """Resolve equal canonical/legacy inputs and reject contradictions."""

    if canonical is None:
        return legacy
    if legacy is None or canonical is legacy:
        return canonical
    if isinstance(canonical, bool) != isinstance(legacy, bool):
        raise TNFRValueError(
            f"Provide either {canonical_name} or legacy {legacy_name}, not both"
        )
    try:
        comparison = canonical == legacy
        all_values = getattr(comparison, "all", None)
        equal = bool(all_values()) if callable(all_values) else bool(comparison)
        if equal:
            return canonical
    except (TypeError, ValueError):
        pass
    raise TNFRValueError(
        f"Provide either {canonical_name} or legacy {legacy_name}, not both"
    )


def spectral_expectation_metadata(*, provenance: str) -> dict[str, Any]:
    """Return canonical scope metadata for a spectral expectation."""

    if not isinstance(provenance, str) or not provenance.strip():
        raise TNFRValueError("Spectral expectation provenance must be non-empty")
    return {
        "metric_kind": SPECTRAL_EXPECTATION_METRIC_KIND,
        "range": SPECTRAL_EXPECTATION_RANGE,
        "bounded": False,
        "provenance": provenance,
        "canonical_coherence": False,
        "canonical_coherence_certified": False,
        "records_to_C_steps": False,
    }


def spectral_expectation_payload(
    *,
    value: Any,
    threshold: Any,
    passed: Any,
    provenance: str,
    operator: Any | None = None,
) -> dict[str, Any]:
    """Build the canonical payload for one spectral-threshold observation."""

    payload: dict[str, Any] = {
        "value": finite_spectral_real(
            value, label="spectral operator expectation"
        ),
        "threshold": finite_spectral_real(
            threshold, label="spectral expectation threshold"
        ),
        "passed": bool(passed),
        **spectral_expectation_metadata(provenance=provenance),
    }
    if operator is not None:
        payload["operator_type"] = type(operator).__name__
    return payload


__all__ = [
    "SPECTRAL_EXPECTATION_METRIC_KIND",
    "SPECTRAL_EXPECTATION_RANGE",
    "finite_spectral_real",
    "positive_spectral_dimension",
    "validate_spectral_operator",
    "resolve_compatibility_value",
    "spectral_expectation_metadata",
    "spectral_expectation_payload",
]
