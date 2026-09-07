"""Pure validators shared by public operator request boundaries."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from numbers import Integral, Real
from typing import Any, NoReturn


def reject_operator_argument(operator: str, detail: str) -> NoReturn:
    """Raise the canonical precondition error without creating import cycles."""

    from .preconditions import OperatorPreconditionError

    raise OperatorPreconditionError(operator, detail)


def finite_real(
    value: Any,
    *,
    operator: str,
    label: str,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    """Return one non-boolean finite real inside an optional closed interval."""

    if isinstance(value, bool) or not isinstance(value, Real):
        reject_operator_argument(
            operator, f"{label} must be a finite real scalar, got {value!r}"
        )
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        reject_operator_argument(
            operator, f"{label} must be representable as a finite scalar, got {value!r}"
        )
    if not math.isfinite(result):
        reject_operator_argument(operator, f"{label} must be finite, got {value!r}")
    if lower is not None and result < lower:
        reject_operator_argument(
            operator, f"{label} must be >= {lower!r}, got {result!r}"
        )
    if upper is not None and result > upper:
        reject_operator_argument(
            operator, f"{label} must be <= {upper!r}, got {result!r}"
        )
    return result


def finite_node_real(
    node_data: dict[str, Any],
    aliases: Iterable[str],
    default: Any,
    *,
    operator: str,
    label: str,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    """Read one scalar alias without permissive fall-through, then validate it."""

    raw = default
    for key in aliases:
        if key in node_data:
            raw = node_data[key]
            break
    return finite_real(
        raw,
        operator=operator,
        label=label,
        lower=lower,
        upper=upper,
    )


def strict_bool(value: Any, *, operator: str, label: str) -> bool:
    """Return a real boolean, rejecting truthy compatibility coercions."""

    if not isinstance(value, bool):
        reject_operator_argument(
            operator, f"{label} must be a boolean, got {value!r}"
        )
    return value


def nonnegative_integer(value: Any, *, operator: str, label: str) -> int:
    """Return a nonnegative non-boolean integral request value."""

    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        reject_operator_argument(
            operator, f"{label} must be a nonnegative integer, got {value!r}"
        )
    return int(value)


def require_list_sink(
    mapping: Mapping[str, Any], key: str, *, operator: str
) -> None:
    """Validate an existing append-only telemetry sink before state changes."""

    if key in mapping and not isinstance(mapping[key], list):
        reject_operator_argument(
            operator, f"{key} must be a list when already configured"
        )


def validate_common_execution_arguments(
    graph_data: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    *,
    operator: str,
) -> None:
    """Validate common flags and an active nodal-equation time step."""

    for key in (
        "validate_preconditions",
        "collect_metrics",
        "validate_nodal_equation",
        "validate_postconditions",
    ):
        if key in kwargs:
            strict_bool(kwargs[key], operator=operator, label=key)

    for key in (
        "VALIDATE_OPERATOR_PRECONDITIONS",
        "COLLECT_OPERATOR_METRICS",
        "VALIDATE_NODAL_EQUATION",
        "VALIDATE_OPERATOR_POSTCONDITIONS",
    ):
        if key in graph_data:
            strict_bool(graph_data[key], operator=operator, label=key)

    validate_equation = bool(kwargs.get("validate_nodal_equation", False)) or bool(
        graph_data.get("VALIDATE_NODAL_EQUATION", False)
    )
    if validate_equation:
        if "NODAL_EQUATION_STRICT" in graph_data:
            strict_bool(
                graph_data["NODAL_EQUATION_STRICT"],
                operator=operator,
                label="NODAL_EQUATION_STRICT",
            )
        finite_real(
            kwargs.get("dt", 1.0),
            operator=operator,
            label="dt",
            lower=math.nextafter(0.0, math.inf),
        )


__all__ = [
    "finite_node_real",
    "finite_real",
    "nonnegative_integer",
    "reject_operator_argument",
    "require_list_sink",
    "strict_bool",
    "validate_common_execution_arguments",
]
