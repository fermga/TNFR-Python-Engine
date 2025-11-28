from __future__ import annotations

from ..types import NodeId, Phase, TNFRGraph
from collections.abc import Mapping, Sequence

__all__ = ["coordinate_global_local_phase"]

ChunkArgs = tuple[
    Sequence[NodeId],
    Mapping[NodeId, Phase],
    Mapping[NodeId, float],
    Mapping[NodeId, float],
    Mapping[NodeId, Sequence[NodeId]],
    float,
    float,
    float,
]

def coordinate_global_local_phase(
    G: TNFRGraph,
    global_force: float | None = None,
    local_force: float | None = None,
    *,
    n_jobs: int | None = None,
) -> None: ...
