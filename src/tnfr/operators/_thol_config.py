"""One validated threshold convention for public THOL and metabolic callers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..config.defaults_core import CORE_DEFAULTS
from ._argument_validation import finite_real


def resolve_thol_bifurcation_threshold(
    graph_data: Mapping[str, Any],
    requested: Any = None,
    *,
    operator: str = "Self-organization",
) -> float:
    """Resolve explicit tau, generic alias, THOL alias, then the core default.

    ``None`` means no explicit/generic override. A present THOL-specific value
    must itself be valid; invalid values never silently fall through. The
    threshold is nonnegative, finite and non-boolean. This resolves configured
    policy, not a threshold derived from the nodal equation or a birth promise.
    OZ and ZHIR have different contracts and are outside this resolver.
    """
    value = requested
    if value is None:
        value = graph_data.get("BIFURCATION_THRESHOLD_TAU")
    if value is None:
        value = graph_data.get(
            "THOL_BIFURCATION_THRESHOLD",
            CORE_DEFAULTS["THOL_BIFURCATION_THRESHOLD"],
        )
    return finite_real(value, operator=operator, label="tau", lower=0.0)
