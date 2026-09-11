from __future__ import annotations

import networkx as nx

from ..operators.event_remesh_causal_runtime import (
    EventRemeshCycleExecutionSpec,
)
from .binary64_p2_reception_stability import (
    P2HalfReceptionRemeshStabilityCertificate,
)
from .runtime_p2_reception_remesh_sequence import (
    ExecutedP2HalfReceptionRemeshSequenceCertificate,
)

def execute_p2_half_reception_remesh_policy_invocation(
    graph: nx.Graph,
    kernel_certificate: P2HalfReceptionRemeshStabilityCertificate,
    specs: tuple[EventRemeshCycleExecutionSpec, ...],
    *,
    metric_weights: tuple[float, float],
    suppress_birth_warnings: bool = ...,
) -> ExecutedP2HalfReceptionRemeshSequenceCertificate: ...

__all__: tuple[str, ...]
