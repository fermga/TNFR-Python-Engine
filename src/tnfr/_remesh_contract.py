"""Shared strict inputs for delayed REMESH execution and evidence."""

from __future__ import annotations

import math
import sys
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal

from .errors import TNFRValueError

__all__ = [
    "DelayedRemeshConfiguration",
    "materialize_delayed_remesh_configuration",
    "materialize_positive_diagonal_metric",
    "positive_remesh_delay",
    "remesh_history_maxlen",
]


def _finite_metric_weight(value: Any, label: str) -> float:
    if isinstance(value, (bool, str, bytes, bytearray, complex)):
        raise TNFRValueError(f"{label} must be a finite real scalar")
    if not isinstance(value, Real):
        raise TNFRValueError(f"{label} must be a finite real scalar")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(f"{label} must be a finite real scalar") from exc
    if not math.isfinite(result):
        raise TNFRValueError(f"{label} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class DelayedRemeshConfiguration:
    """Frozen deterministic configuration for one delayed REMESH map."""

    tau_local: int
    tau_global: int
    history_maxlen: int
    alpha: float
    alpha_source: str
    epi_min: float
    epi_max: float
    clip_mode: Literal["hard", "soft"]


def positive_remesh_delay(value: Any, label: str) -> int:
    """Return one strict positive integral REMESH delay."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TNFRValueError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise TNFRValueError(f"{label} must be a positive integer")
    return result


def remesh_history_maxlen(
    tau_global: Any,
    tau_local: Any,
) -> int:
    """Return the canonical bounded-history capacity if Python can allocate it."""

    global_delay = positive_remesh_delay(
        tau_global,
        "REMESH_TAU_GLOBAL",
    )
    local_delay = positive_remesh_delay(
        tau_local,
        "REMESH_TAU_LOCAL",
    )
    delay = max(global_delay, local_delay)
    if delay > (sys.maxsize - 5) // 2:
        raise TNFRValueError(
            "REMESH delays exceed the materializable history capacity"
        )
    return max(2 * delay + 5, 64)


def materialize_delayed_remesh_configuration(
    *,
    tau_local: Any,
    tau_global: Any,
    alpha: Any,
    alpha_source: Any,
    epi_min: Any,
    epi_max: Any,
    clip_mode: Any,
) -> DelayedRemeshConfiguration:
    """Validate and freeze every deterministic delayed-REMESH control."""

    local_delay = positive_remesh_delay(tau_local, "REMESH_TAU_LOCAL")
    global_delay = positive_remesh_delay(tau_global, "REMESH_TAU_GLOBAL")
    history_maxlen = remesh_history_maxlen(global_delay, local_delay)
    alpha_value = _finite_metric_weight(alpha, "REMESH_alpha")
    if not 0.0 < alpha_value <= 1.0:
        raise TNFRValueError("REMESH_alpha must be in the interval (0, 1]")
    lower = _finite_metric_weight(epi_min, "EPI_MIN")
    upper = _finite_metric_weight(epi_max, "EPI_MAX")
    if lower > upper:
        raise TNFRValueError("EPI_MIN must not exceed EPI_MAX")
    if clip_mode not in ("hard", "soft"):
        raise TNFRValueError("CLIP_MODE must be 'hard' or 'soft'")
    if not isinstance(alpha_source, str) or not alpha_source:
        raise TNFRValueError("alpha_source must be a non-empty string")
    return DelayedRemeshConfiguration(
        tau_local=local_delay,
        tau_global=global_delay,
        history_maxlen=history_maxlen,
        alpha=alpha_value,
        alpha_source=alpha_source,
        epi_min=lower,
        epi_max=upper,
        clip_mode=clip_mode,
    )


def materialize_positive_diagonal_metric(
    raw: Mapping[Hashable, Any] | Sequence[Any] | None,
    nodes: Sequence[Hashable],
) -> tuple[float, ...]:
    """Freeze a full-support strictly positive metric in declared node order."""

    node_order = tuple(nodes)
    if raw is None:
        values = tuple(1.0 for _ in node_order)
    elif isinstance(raw, Mapping):
        try:
            keys = tuple(raw)
            support = frozenset(keys)
            node_support = frozenset(node_order)
        except (TypeError, ValueError) as exc:
            raise TNFRValueError(
                "metric_weights must have hashable node keys"
            ) from exc
        if len(keys) != len(node_order) or support != node_support:
            raise TNFRValueError(
                "metric_weights support must equal the current graph node "
                "support"
            )
        values = tuple(raw[node] for node in node_order)
    elif isinstance(raw, Sequence) and not isinstance(
        raw,
        (str, bytes, bytearray),
    ):
        if len(raw) != len(node_order):
            raise TNFRValueError(
                "metric_weights length must equal the current graph order"
            )
        values = tuple(raw)
    else:
        raise TNFRValueError(
            "metric_weights must be a node mapping or indexed sequence"
        )

    weights = tuple(
        _finite_metric_weight(value, f"metric_weights[{index}]")
        for index, value in enumerate(values)
    )
    if any(weight <= 0.0 for weight in weights):
        raise TNFRValueError("metric_weights must be strictly positive")
    return weights
