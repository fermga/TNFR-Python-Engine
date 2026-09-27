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


def _validate_support_change_labels(report, states, bridges, cuts, partitions=()):
    for state in states:
        for edge in state.edges:
            for node in edge:
                _validate_label(node)
    for labels in (*bridges, *partitions):
        for node in labels:
            _validate_label(node)
    for port in report.ports:
        _validate_label(port.before.node)
        _validate_label(port.after.node)
    for snapshot in (report.transport_reset.before, report.transport_reset.after):
        for node in snapshot.nodes:
            _validate_label(node)
    for cut in cuts:
        for labels in (cut.nodes, cut.region, cut.environment):
            for node in labels:
                _validate_label(node)
        for a, b, _ in cut.cut_edges:
            _validate_label(a)
            _validate_label(b)


def relational_report_to_dict(report: Any) -> dict[str, Any]:
    """Project a relational field, step, observation or scoped certificate.

    Use ``export_to_json(relational_report_to_dict(report), path)`` to save the
    result atomically. Fractions become ``{numerator, denominator}`` records;
    tuples become arrays, including tuple node labels. Opaque node labels are
    rejected rather than rendered with potentially ambiguous ``str``/``repr``.
    The result is detached, but is not a typed round trip or resumable checkpoint.
    Public dataclass construction is not proof of scientific provenance.
    Capture and continuous-transit certificates retain their exact rational
    bounds and independent theorem admission; projection never combines their
    verdicts or authenticates a publicly constructed report.
    Attachment observations retain both component fields, the hypothetical
    joined field and supplied ports without representing a committed event.
    Relocation observations retain the old/new fields and the unchanged
    component partition with the same support-change accounting semantics.
    Supply assessments compare declared work with represented storage; they
    neither authenticate that work nor select or certify an actual event.
    """
    from ..dynamics.relational import RelationalExchangeField, RelationalExchangeStep
    from ..physics.relational_capture import (
        RelationalCaptureCertificate,
        RelationalLocalCaptureCertificate,
        RelationalSectorCaptureCertificate,
    )
    from ..physics.relational_observations import (
        RelationalAttachmentObservation,
        RelationalAttachmentSupplyAssessment,
        RelationalPatternObservation,
        RelationalRelocationObservation,
    )
    from ..physics.relational_transit import RelationalTransitCertificate

    if not isinstance(
        report,
        (
            RelationalExchangeField,
            RelationalExchangeStep,
            RelationalPatternObservation,
            RelationalAttachmentObservation,
            RelationalAttachmentSupplyAssessment,
            RelationalRelocationObservation,
            RelationalCaptureCertificate,
            RelationalLocalCaptureCertificate,
            RelationalSectorCaptureCertificate,
            RelationalTransitCertificate,
        ),
    ):
        raise TypeError(
            "expected a relational field, step, pattern, attachment, relocation, supply assessment, "
            "capture or continuous-transit report"
        )
    observation = (
        report.initial if isinstance(report, RelationalTransitCertificate) else report
    )
    if isinstance(observation, RelationalAttachmentSupplyAssessment):
        states = ()
    elif isinstance(observation, RelationalAttachmentObservation):
        states = (*observation.components, observation.joined)
        _validate_support_change_labels(
            observation, states, (observation.bridge,), (observation.cut,)
        )
    elif isinstance(observation, RelationalRelocationObservation):
        states = (observation.before, observation.after)
        _validate_support_change_labels(
            observation,
            states,
            (observation.remove_bridge, observation.add_bridge),
            (observation.cut_before, observation.cut_after),
            observation.components,
        )
    elif isinstance(observation, RelationalExchangeStep):
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
    projected = _project(report)
    if isinstance(
        report, (RelationalAttachmentObservation, RelationalRelocationObservation)
    ):
        projected["continuous_loss_change"] = _project(report.continuous_loss_change)
        projected["represented_zero_supply_passive"] = (
            report.represented_zero_supply_passive
        )
    return {
        "schema": "tnfr.relational-report.v1",
        "report_type": type(report).__name__,
        "report": projected,
    }
