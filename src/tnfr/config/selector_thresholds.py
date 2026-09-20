"""Shared resolution of the configured selector and Sense Index policy.

These cutoffs are operational choices, not thresholds derived from the nodal
equation. Partial mappings overlay the selector defaults. ``GLYPH_THRESHOLDS``
belongs to a separate policy and is not an alias for this configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from .._exact_time import finite_represented_real
from ..types import SelectorThresholds, TNFRGraph
from .defaults_core import SELECTOR_THRESHOLD_DEFAULTS


def resolve_selector_thresholds(graph: TNFRGraph) -> SelectorThresholds:
    """Return a detached, finite unit-interval selector threshold mapping.

    Resolve on every call so in-place overrides cannot leave a stale cache and
    callers cannot mutate a shared cached policy. Invalid values are rejected,
    never clipped into a different policy. Equal or overlapping bands remain
    permitted; each consumer retains its documented comparison order.
    """
    from . import get_param

    configured = get_param(graph, "SELECTOR_THRESHOLDS")
    if not isinstance(configured, Mapping):
        raise ValueError("SELECTOR_THRESHOLDS must be a mapping")
    resolved: dict[str, float] = {}
    for key, default in SELECTOR_THRESHOLD_DEFAULTS.items():
        value: Any = configured.get(key, default)
        label = f"SELECTOR_THRESHOLDS[{key!r}]"
        try:
            number = finite_represented_real(value, label)[0]
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"{label} must be a finite real number in [0, 1]")
        resolved[key] = number
    return cast(SelectorThresholds, resolved)
