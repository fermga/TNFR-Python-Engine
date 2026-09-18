from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..errors import TNFRValueError
from ..mathematics.phasor_resultant import RepresentedPhasorResultant
from ..types import NodeId, Phase, TNFRGraph

__all__ = [
    "coordinate_global_local_phase",
    "GlobalPhaseCoordinationEvidence",
    "UndefinedGlobalPhaseError",
]

class UndefinedGlobalPhaseError(TNFRValueError): ...

@dataclass(frozen=True)
class GlobalPhaseCoordinationEvidence:
    version: str
    status: str
    nodes: tuple[NodeId, ...]
    neighbor_order: tuple[tuple[NodeId, tuple[NodeId, ...]], ...]
    primitive_phases: tuple[float, ...]
    resultant: RepresentedPhasorResultant | None
    requested_global_force: float | None
    requested_local_force: float | None
    effective_global_force: float
    effective_local_force: float
    gain_mode: str
    global_target: float | None
    global_term_active: bool
    local_targets: tuple[float, ...]
    raw_proposals: tuple[float, ...]
    realized_phases: tuple[float, ...]
    execution_path: str
    scope: str

ChunkArgs = tuple[
    Sequence[NodeId],
    Mapping[NodeId, Phase],
    Mapping[NodeId, float],
    Mapping[NodeId, float],
    Mapping[NodeId, Sequence[NodeId]],
    float | None,
    float,
    float,
]

def coordinate_global_local_phase(
    G: TNFRGraph,
    global_force: float | None = None,
    local_force: float | None = None,
    *,
    n_jobs: int | None = None,
    global_reduction: str = "legacy",
) -> GlobalPhaseCoordinationEvidence | None: ...
