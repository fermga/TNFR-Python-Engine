from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

from .._compat import TypeAlias
from ._delayed_remesh_kernel import (
    DelayedRemeshNodeProposal as DelayedRemeshNodeProposal,
    DelayedRemeshPlan as DelayedRemeshPlan,
    DelayedRemeshResult as DelayedRemeshResult,
    DelayedRemeshStabilityEvidence as DelayedRemeshStabilityEvidence,
)

__all__ = [
    "DelayedRemeshNodeProposal",
    "DelayedRemeshPlan",
    "DelayedRemeshResult",
    "DelayedRemeshStabilityEvidence",
    "apply_network_remesh",
    "plan_network_remesh",
    "apply_topological_remesh",
    "apply_remesh_if_globally_stable",
]

CommunityGraph: TypeAlias = Any

def plan_network_remesh(
    G: CommunityGraph,
    *,
    include_stability_evidence: bool = False,
    metric_weights: Mapping[Hashable, Any] | Sequence[Any] | None = None,
) -> DelayedRemeshPlan: ...
def apply_network_remesh(
    G: CommunityGraph,
    *,
    include_stability_evidence: bool = False,
    metric_weights: Mapping[Hashable, Any] | Sequence[Any] | None = None,
) -> DelayedRemeshResult: ...
def apply_topological_remesh(
    G: CommunityGraph,
    mode: str | None = None,
    *,
    k: int | None = None,
    p_rewire: float = 0.2,
    seed: int | None = None,
) -> None: ...
def apply_remesh_if_globally_stable(
    G: CommunityGraph, stable_step_window: int | None = None, **kwargs: Any
) -> None: ...