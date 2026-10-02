"""One read-only optional THOL precondition gate for public and legacy callers.

This checks structural prerequisites, not grammar admission or complete birth
readiness. Crossing the acceleration threshold is not required to execute the
pressure action. Depth, hierarchy, metabolic amplitude, child identity and
transactional side effects remain with the public SelfOrganization proposal.
No validator call records execution telemetry or selects a target.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from typing import Any

from ...constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from ...errors import TNFRValueError
from ...types import TNFRGraph, real_scalar_epi
from .._argument_validation import (
    finite_node_real,
    finite_real,
    nonnegative_integer,
    reject_operator_argument,
    strict_bool,
)
from .._thol_config import resolve_thol_bifurcation_threshold

__all__ = ["validate_self_organization_strict"]

_OPERATOR = "Self-organization"


def _finite_epi_value(
    value: Any,
    *,
    label: str,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    """Validate a real scalar or canonical BEPI representation."""

    if isinstance(value, bool):
        reject_operator_argument(_OPERATOR, f"{label} must be a real EPI value")
    try:
        result = real_scalar_epi(value)
    except (OverflowError, TypeError, ValueError):
        reject_operator_argument(
            _OPERATOR, f"{label} must be a scalar or uniform-real BEPI value"
        )
    if result is None:
        reject_operator_argument(
            _OPERATOR, f"{label} must have an exact signed scalar embedding"
        )
    return finite_real(
        result,
        operator=_OPERATOR,
        label=label,
        lower=lower,
        upper=upper,
    )


def _finite_node_epi(
    node_data: Mapping[str, Any], *, label: str, default: Any = 0.0
) -> float:
    raw = default
    for key in ALIAS_EPI:
        if key in node_data:
            raw = node_data[key]
            break
    return _finite_epi_value(raw, label=label)


def _active_acceleration_history_length(node_data: Mapping[str, Any]) -> int:
    """Read only the size selected by the shared acceleration implementation."""

    from ..nodal_equation import _select_acceleration_history

    source, history = _select_acceleration_history(node_data)
    if history is None:
        return 0
    _require_indexed_acceleration_history(history, source)
    try:
        return len(history)
    except (OverflowError, TypeError):
        reject_operator_argument(_OPERATOR, f"{source} must be a sized history")


def _require_indexed_acceleration_history(history: Any, source: str) -> None:
    """Keep THOL's sequence contract separate from numerical compatibility."""
    if isinstance(history, (str, bytes, bytearray, Mapping, Iterator)):
        reject_operator_argument(_OPERATOR, f"{source} must be an indexed history")


def validate_self_organization_strict(
    G: TNFRGraph,
    node: Any,
    *,
    tau=None,
    emit_warnings: bool = True,
) -> None:
    """Check the optional public THOL gate without changing graph state.

    Calling this function explicitly always checks the gate. Public execution
    invokes it only when its existing precondition switches enable it. Exact
    signed scalar and uniform-real BEPI readings, active-history precedence,
    strict finite configuration and positive stored pressure match the public
    operator. An acceleration at or below tau warns but is not rejected.

    Passing proves neither grammar admission nor a future birth. No history,
    pressure, cached acceleration, context or no-birth flag is written. Those
    execution records belong to a successful operator commit.
    """
    emit_warnings = strict_bool(
        emit_warnings,
        operator=_OPERATOR,
        label="emit_warnings",
    )
    data = G.nodes[node]
    epi = _finite_node_epi(data, label="EPI")
    dnfr = finite_node_real(data, ALIAS_DNFR, 0.0, operator=_OPERATOR, label="DeltaNFR")
    vf = finite_node_real(
        data,
        ALIAS_VF,
        0.0,
        operator=_OPERATOR,
        label="nu_f",
        lower=0.0,
    )
    min_epi = finite_real(
        G.graph.get("THOL_MIN_EPI", 0.2),
        operator=_OPERATOR,
        label="THOL_MIN_EPI",
        lower=0.0,
    )
    min_vf = finite_real(
        G.graph.get("THOL_MIN_VF", 0.1),
        operator=_OPERATOR,
        label="THOL_MIN_VF",
        lower=0.0,
    )
    if epi < min_epi:
        reject_operator_argument(
            _OPERATOR, f"EPI too low for bifurcation ({epi!r} < {min_epi!r})"
        )
    if dnfr <= 0.0:
        reject_operator_argument(
            _OPERATOR, "DeltaNFR must be positive for self-organization"
        )
    if vf < min_vf:
        reject_operator_argument(
            _OPERATOR,
            f"nu_f too low for reorganization ({vf!r} < {min_vf!r})",
        )

    min_degree = nonnegative_integer(
        G.graph.get("THOL_MIN_DEGREE", 1),
        operator=_OPERATOR,
        label="THOL_MIN_DEGREE",
    )
    allow_isolated = strict_bool(
        G.graph.get("THOL_ALLOW_ISOLATED", False),
        operator=_OPERATOR,
        label="THOL_ALLOW_ISOLATED",
    )
    degree = G.degree(node)
    if degree < min_degree and not allow_isolated:
        reject_operator_argument(
            _OPERATOR,
            f"node degree {degree} is below THOL_MIN_DEGREE {min_degree}",
        )

    from ..nodal_equation import observe_structural_acceleration

    try:
        observation = observe_structural_acceleration(G, node)
    except TNFRValueError as exc:
        reject_operator_argument(_OPERATOR, str(exc))
    min_history = nonnegative_integer(
        G.graph.get("THOL_MIN_HISTORY_LENGTH", 3),
        operator=_OPERATOR,
        label="THOL_MIN_HISTORY_LENGTH",
    )
    if min_history < 3:
        reject_operator_argument(
            _OPERATOR, "THOL_MIN_HISTORY_LENGTH must be at least 3"
        )
    if observation.source is not None:
        _require_indexed_acceleration_history(
            data[observation.source],
            observation.source,
        )
    history_length = observation.history_length
    if history_length < min_history:
        reject_operator_argument(
            _OPERATOR,
            f"active EPI history has {history_length} samples; "
            f"{min_history} required",
        )
    d2_epi = finite_real(
        observation.value,
        operator=_OPERATOR,
        label="signed EPI acceleration",
    )

    metabolic = strict_bool(
        G.graph.get("THOL_METABOLIC_ENABLED", True),
        operator=_OPERATOR,
        label="THOL_METABOLIC_ENABLED",
    )
    if metabolic and degree == 0:
        reject_operator_argument(
            _OPERATOR, "metabolic THOL requires at least one neighbour"
        )

    tau = resolve_thol_bifurcation_threshold(G.graph, tau)
    logger = logging.getLogger(__name__)
    if emit_warnings and abs(d2_epi) <= tau:
        logger.warning(
            "Node %r: THOL acceleration magnitude %.6g does not exceed tau %.6g; "
            "no sub-EPI will be generated.",
            node,
            abs(d2_epi),
            tau,
        )
