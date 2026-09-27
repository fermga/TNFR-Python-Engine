"""Exact JSON projections of detached relational model reports.

Projection preserves rational evidence and node order. It neither authenticates
the report nor reconstructs a live graph, model admission or execution history.
"""

from __future__ import annotations

import math
from dataclasses import fields, is_dataclass
from fractions import Fraction
from typing import Any

__all__ = ("relational_report_to_dict",)


def _validate_label(value: Any) -> None:
    if isinstance(value, tuple):
        for item in value:
            _validate_label(item)
    elif value is None or type(value) in (bool, int, str):
        return
    elif type(value) is float and math.isfinite(value):
        return
    else:
        raise TypeError(
            "report export requires JSON scalar node labels or tuples of them"
        )


def _project(value: Any) -> Any:
    if isinstance(value, Fraction):
        return {
            "numerator": int(value.numerator),
            "denominator": int(value.denominator),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _project(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_project(item) for item in value]
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("relational report contains a nonfinite float")
        return value
    raise TypeError(
        f"unsupported relational report value or node label: {type(value).__name__}; "
        "use JSON scalar labels or tuples of them"
    )


def relational_report_to_dict(report: Any) -> dict[str, Any]:
    """Project a relational field, step, pattern, capture or continuous-transit report.

    Use ``export_to_json(relational_report_to_dict(report), path)`` to save the
    result atomically. Fractions become ``{numerator, denominator}`` records;
    tuples become arrays, including tuple node labels. Opaque node labels are
    rejected rather than rendered with potentially ambiguous ``str``/``repr``.
    The result is detached, but is not a typed round trip or resumable checkpoint.
    Public dataclass construction is not proof of scientific provenance.
    Capture and continuous-transit certificates retain their exact rational
    bounds and independent theorem admission; projection never combines their
    verdicts or authenticates a publicly constructed report.
    """
    from ..dynamics.relational import RelationalExchangeField, RelationalExchangeStep
    from ..physics.relational_capture import (
        RelationalCaptureCertificate,
        RelationalLocalCaptureCertificate,
        RelationalSectorCaptureCertificate,
    )
    from ..physics.relational_observations import RelationalPatternObservation
    from ..physics.relational_transit import RelationalTransitCertificate

    if not isinstance(
        report,
        (
            RelationalExchangeField,
            RelationalExchangeStep,
            RelationalPatternObservation,
            RelationalCaptureCertificate,
            RelationalLocalCaptureCertificate,
            RelationalSectorCaptureCertificate,
            RelationalTransitCertificate,
        ),
    ):
        raise TypeError(
            "expected a relational field, step, pattern, capture or continuous-transit report"
        )
    observation = (
        report.initial if isinstance(report, RelationalTransitCertificate) else report
    )
    if isinstance(observation, RelationalExchangeStep):
        states = (observation.before, observation.after)
    elif isinstance(
        observation,
        (
            RelationalPatternObservation,
            RelationalCaptureCertificate,
            RelationalLocalCaptureCertificate,
            RelationalSectorCaptureCertificate,
        ),
    ):
        states = (observation.field,)
        # Undefined cycles can retain nodes absent from the admitted graph.
        # Those labels must not bypass admission through dataclass projection.
        for certificate in observation.winding:
            for node in certificate.cycle_nodes:
                _validate_label(node)
    else:
        states = (observation,)
    for state in states:
        for node in state.nodes:
            _validate_label(node)
    return {
        "schema": "tnfr.relational-report.v1",
        "report_type": type(report).__name__,
        "report": _project(report),
    }
