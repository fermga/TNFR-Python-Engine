"""Shared configured OZ policies for execution and detached admission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._argument_validation import finite_real, nonnegative_integer


@dataclass(frozen=True, slots=True)
class OZStrictPolicy:
    """Admitted scalar thresholds; history and grammar remain separate gates."""

    min_epi: float
    max_dnfr: float
    min_vf: float
    min_degree: int


def resolve_oz_strict_policy(graph_data: Mapping[str, Any]) -> OZStrictPolicy:
    """Resolve the strict OZ thresholds without inspecting or mutating a node.

    The minimum form remains signed. Pressure magnitude, capacity and degree
    thresholds are nonnegative; degree is integral. Present invalid values
    reject instead of falling through to defaults. The degree threshold controls
    a warning, not a blocking gate. These are supplied execution policies.
    """
    return OZStrictPolicy(
        min_epi=finite_real(
            graph_data.get("OZ_MIN_EPI", 0.2),
            operator="Dissonance",
            label="OZ_MIN_EPI",
        ),
        max_dnfr=finite_real(
            graph_data.get("OZ_MAX_DNFR", 0.8),
            operator="Dissonance",
            label="OZ_MAX_DNFR",
            lower=0.0,
        ),
        min_vf=finite_real(
            graph_data.get("OZ_MIN_VF", 0.1),
            operator="Dissonance",
            label="OZ_MIN_VF",
            lower=0.0,
        ),
        min_degree=nonnegative_integer(
            graph_data.get("OZ_MIN_DEGREE", 1),
            operator="Dissonance",
            label="OZ_MIN_DEGREE",
        ),
    )
