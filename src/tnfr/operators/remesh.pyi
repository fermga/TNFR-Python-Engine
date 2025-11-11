from __future__ import annotations

from typing import Any

from .._compat import TypeAlias

__all__ = [
    "apply_network_remesh",
    "apply_topological_remesh",
    "apply_remesh_if_globally_stable",
]

CommunityGraph: TypeAlias = Any

def apply_network_remesh(G: CommunityGraph) -> None: ...
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
