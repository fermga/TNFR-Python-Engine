"""Partition metadata structures and planner for Paley graph factorization."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np

try:
    from tnfr.metrics.common import compute_coherence
    from tnfr.metrics.sense_index import compute_Si
    HAS_PARTITION_METRICS = True
except ImportError:  # pragma: no cover - optional dependency in minimal installs
    HAS_PARTITION_METRICS = False
    compute_coherence = None  # type: ignore
    compute_Si = None  # type: ignore


@dataclass
class PaleyPartitionTelemetry:
    """Telemetry snapshot collected for a single partition."""

    phi_s: float = 0.0
    phase_gradient: float = 0.0
    phase_curvature: float = 0.0
    coherence_length: float = 0.0
    coherence: float = 0.0
    sense_index: float = 0.0
    notes: str = ""


def _mean_from_iter(values: Iterable[float]) -> float:
    data = [float(v) for v in values if isinstance(v, (int, float))]
    if not data:
        return 0.0
    try:
        return float(fmean(data))
    except Exception:  # pragma: no cover - fmean fallback
        return float(sum(data) / len(data))


def _measure_partition_health(graph: nx.Graph, nodes: Sequence[int]) -> Tuple[float, float]:
    if not nodes or graph is None or not HAS_PARTITION_METRICS:
        return 0.0, 0.0
    subgraph = graph.subgraph(nodes).copy()
    coherence_value = 0.0
    sense_value = 0.0
    if compute_coherence:
        try:
            coherence_value = float(compute_coherence(subgraph))
        except Exception:  # pragma: no cover - telemetry fallback
            coherence_value = 0.0
    if compute_Si:
        try:
            si_payload = compute_Si(subgraph, inplace=False)
            if isinstance(si_payload, dict):
                sense_value = _mean_from_iter(si_payload.values())
            else:
                arr = np.asarray(si_payload, dtype=float)
                sense_value = float(arr.mean()) if arr.size else 0.0
        except Exception:  # pragma: no cover - telemetry fallback
            sense_value = 0.0
    return coherence_value, sense_value


@dataclass
class PaleyPartition:
    """Describes a partition of the Paley graph used during factorization."""

    partition_id: str
    node_indices: List[int]
    boundary_nodes: List[int] = field(default_factory=list)
    phase_reference: float = 0.0
    telemetry: PaleyPartitionTelemetry | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    candidate_factors: List[int] = field(default_factory=list)

    def to_mapping(self) -> Dict[str, Any]:
        """Serialize the partition for certificates or telemetry."""

        payload: Dict[str, Any] = {
            "id": self.partition_id,
            "node_indices": list(self.node_indices),
            "boundary_nodes": list(self.boundary_nodes),
            "phase_reference": self.phase_reference,
        }
        if self.telemetry is not None:
            payload["telemetry"] = vars(self.telemetry)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        if self.candidate_factors:
            payload["candidate_factors"] = list(self.candidate_factors)
        return payload


@dataclass
class PartitionedPaleyGraph:
    """Container describing the partitioning layout for a Paley graph."""

    modulus: int
    partitions: List[PaleyPartition]
    overlap_policy: str = "phase_reference"
    notes: str = ""

    def iter_partitions(self) -> Iterable[PaleyPartition]:
        """Iterate over partitions in deterministic order."""

        yield from self.partitions

    def summary(self) -> Mapping[str, Any]:
        """Return a summary suitable for certificates or logging."""

        return {
            "modulus": self.modulus,
            "partition_count": len(self.partitions),
            "overlap_policy": self.overlap_policy,
            "notes": self.notes,
        }

    @classmethod
    def single_partition(cls, modulus: int, graph: nx.Graph | None = None) -> PartitionedPaleyGraph:
        """Convenience helper that models the entire graph as one partition."""

        nodes = list(graph.nodes()) if graph is not None else []
        partition = PaleyPartition(partition_id="all", node_indices=nodes)
        return cls(modulus=modulus, partitions=[partition])


@dataclass
class PartitionPlannerConfig:
    """Configuration parameters for the Paley partition planner."""

    target_size: int = 256
    boundary_overlap: int = 4
    notes: str = "auto"


def plan_paley_partitions(
    graph: nx.Graph,
    modulus: int,
    *,
    phi_s: float,
    phase_gradient: float,
    phase_curvature: float,
    coherence_length: float,
    config: PartitionPlannerConfig | None = None,
) -> PartitionedPaleyGraph:
    """Compute a simple deterministic partitioning of the Paley graph."""

    if graph.number_of_nodes() == 0:
        return PartitionedPaleyGraph.single_partition(modulus, graph)

    cfg = config or PartitionPlannerConfig()
    target_size = max(1, cfg.target_size)
    overlap = max(0, cfg.boundary_overlap)

    nodes = sorted(graph.nodes())
    total_nodes = len(nodes)
    step = max(1, target_size - overlap)

    partitions: List[PaleyPartition] = []
    for idx, start in enumerate(range(0, total_nodes, step)):
        chunk = nodes[start : start + target_size]
        chunk_set = set(chunk)
        boundary_nodes = [
            node
            for node in chunk
            if any(neigh not in chunk_set for neigh in graph.neighbors(node))
        ]

        ratio = len(chunk) / total_nodes if total_nodes else 0.0
        coherence_value, sense_value = _measure_partition_health(graph, chunk)
        telemetry = PaleyPartitionTelemetry(
            phi_s=phi_s * ratio,
            phase_gradient=phase_gradient,
            phase_curvature=phase_curvature,
            coherence_length=coherence_length * ratio,
            coherence=coherence_value,
            sense_index=sense_value,
            notes=f"planner={cfg.notes}",
        )

        partition = PaleyPartition(
            partition_id=f"p{idx}",
            node_indices=chunk,
            boundary_nodes=boundary_nodes,
            phase_reference=0.0,
            telemetry=telemetry,
            metadata={
                "range": [chunk[0], chunk[-1]] if chunk else [],
                "ratio": ratio,
                "size": len(chunk),
            },
        )
        partitions.append(partition)

    return PartitionedPaleyGraph(
        modulus=modulus,
        partitions=partitions,
        overlap_policy="boundary_overlap",
        notes=f"target={target_size}",
    )


@dataclass
class PartitionAggregation:
    """Aggregated telemetry derived from all partitions."""

    partition_count: int
    node_total: int
    boundary_fraction: float
    phi_s_weighted: float
    phi_s_ratio: float
    phase_gradient_max: float
    phase_gradient_ratio: float
    phase_curvature_max: float
    phase_curvature_ratio: float
    coherence_length_weighted: float
    coherence_ratio: float
    coverage_ratio: float
    candidate_total: int
    candidate_ratio: float
    partition_candidates: Dict[str, List[int]]
    empty_partitions: List[str]
    notes: str = ""

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "partition_count": self.partition_count,
            "node_total": self.node_total,
            "boundary_fraction": self.boundary_fraction,
            "phi_s_weighted": self.phi_s_weighted,
            "phi_s_ratio": self.phi_s_ratio,
            "phase_gradient_max": self.phase_gradient_max,
            "phase_gradient_ratio": self.phase_gradient_ratio,
            "phase_curvature_max": self.phase_curvature_max,
            "phase_curvature_ratio": self.phase_curvature_ratio,
            "coherence_length_weighted": self.coherence_length_weighted,
            "coherence_ratio": self.coherence_ratio,
            "coverage_ratio": self.coverage_ratio,
            "candidate_total": self.candidate_total,
            "candidate_ratio": self.candidate_ratio,
            "partition_candidates": self.partition_candidates,
            "empty_partitions": self.empty_partitions,
            "notes": self.notes,
        }


def aggregate_partition_metrics(
    partitioned: PartitionedPaleyGraph,
    *,
    parent_phi_s: float,
    parent_phase_gradient: float,
    parent_phase_curvature: float,
    parent_coherence_length: float,
    total_candidate_count: int,
) -> PartitionAggregation:
    """Combine per-partition telemetry into global aggregates."""

    partitions = list(partitioned.iter_partitions())
    if not partitions:
        return PartitionAggregation(
            partition_count=0,
            node_total=0,
            boundary_fraction=0.0,
            phi_s_weighted=0.0,
            phi_s_ratio=0.0,
            phase_gradient_max=0.0,
            phase_gradient_ratio=0.0,
            phase_curvature_max=0.0,
            phase_curvature_ratio=0.0,
            coherence_length_weighted=0.0,
            coherence_ratio=0.0,
            coverage_ratio=0.0,
            candidate_total=0,
            candidate_ratio=0.0,
            partition_candidates={},
            empty_partitions=[],
            notes="no-partitions",
        )

    node_total = sum(len(part.node_indices) for part in partitions)
    unique_nodes = len({node for part in partitions for node in part.node_indices})
    boundary_nodes = {node for part in partitions for node in part.boundary_nodes}

    phi_sum = 0.0
    coherence_sum = 0.0
    gradient_max = 0.0
    curvature_max = 0.0
    candidate_total = 0
    partition_candidates: Dict[str, List[int]] = {}
    empty_partitions: List[str] = []

    for part in partitions:
        telemetry = part.telemetry
        if telemetry is not None:
            phi_sum += telemetry.phi_s
            coherence_sum += telemetry.coherence_length
            gradient_max = max(gradient_max, telemetry.phase_gradient)
            curvature_max = max(curvature_max, telemetry.phase_curvature)

        metadata_candidates = list(part.metadata.get("candidate_factors", []))
        candidate_list = list(part.candidate_factors)
        combined = metadata_candidates + candidate_list
        if combined:
            deduped = sorted(set(combined))
            partition_candidates[part.partition_id] = deduped
            candidate_total += len(deduped)
        else:
            partition_candidates[part.partition_id] = []
            empty_partitions.append(part.partition_id)

    phi_ratio = (phi_sum / parent_phi_s) if parent_phi_s else 0.0
    coherence_ratio = (coherence_sum / parent_coherence_length) if parent_coherence_length else 0.0
    gradient_ratio = (gradient_max / parent_phase_gradient) if parent_phase_gradient else 0.0
    curvature_ratio = (curvature_max / parent_phase_curvature) if parent_phase_curvature else 0.0

    coverage_ratio = unique_nodes / partitioned.modulus if partitioned.modulus else 0.0
    boundary_fraction = (len(boundary_nodes) / unique_nodes) if unique_nodes else 0.0
    candidate_ratio = (candidate_total / total_candidate_count) if total_candidate_count else 0.0

    return PartitionAggregation(
        partition_count=len(partitions),
        node_total=node_total,
        boundary_fraction=boundary_fraction,
        phi_s_weighted=phi_sum,
        phi_s_ratio=phi_ratio,
        phase_gradient_max=gradient_max,
        phase_gradient_ratio=gradient_ratio,
        phase_curvature_max=curvature_max,
        phase_curvature_ratio=curvature_ratio,
        coherence_length_weighted=coherence_sum,
        coherence_ratio=coherence_ratio,
        coverage_ratio=coverage_ratio,
        candidate_total=candidate_total,
        candidate_ratio=candidate_ratio,
        partition_candidates=partition_candidates,
        empty_partitions=empty_partitions,
        notes="aggregated",
    )


def annotate_partition_candidates(
    partitioned: PartitionedPaleyGraph,
    candidates: Sequence[int],
) -> Dict[str, List[int]]:
    """Distribute candidate factors across partitions deterministically."""

    partitions = list(partitioned.iter_partitions())
    if not partitions or not candidates:
        return {part.partition_id: [] for part in partitions}

    assignment: Dict[str, List[int]] = {part.partition_id: [] for part in partitions}
    for idx, candidate in enumerate(candidates):
        target = partitions[idx % len(partitions)]
        target.candidate_factors.append(candidate)
        assignment[target.partition_id].append(candidate)
        target.metadata.setdefault("candidate_factors", []).append(candidate)

    return assignment
