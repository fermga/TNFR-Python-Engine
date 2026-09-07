"""GPU strategy implementations for TNFR operators.

This module provides GPU-assisted wrappers around the canonical
TNFR AL and RA operators that integrate with the strategy registry
and auto-scaler recommendations.

Key Features:
- Optional read-only GPU backend preview (JAX/PyTorch/CuPy)
- Canonical state commits through validated public operators
- Integration with existing telemetry and validation
- Preserves all TNFR operator contracts and grammar rules

Usage:
    >>> from tnfr.operators.strategies import gpu_strategies
    >>> gpu_strategies.register_all_gpu_strategies()
    >>> # GPU strategies are now available in the registry
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

try:
    from ...engines.computation.unified_gpu_system import (
        TNFRUnifiedGPUSystem,
        get_unified_gpu_system,
    )

    HAS_GPU_ENGINE = True
except Exception:
    HAS_GPU_ENGINE = False
    TNFRUnifiedGPUSystem = None
    get_unified_gpu_system = None

from ...alias import get_attr
from ...constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..network_stage import (
    GraphTransactionSnapshot,
    OPERATOR_MAJOR_GAUSS_SEIDEL,
    TWO_PHASE_JACOBI,
    execute_neighbor_stage,
    record_gauss_seidel_stage,
)
from .strategy import (
    OperationResult,
    PartitionBlock,
    PreparedBlock,
    ResourceEstimate,
    StrategyContext,
    StrategyRegistry,
)


_CONTEXT_FACTOR_OVERRIDE_KEYS = frozenset(
    {"GLYPH_FACTORS", "glyph_factors", "operator_factors"}
)
# Private compatibility alias; the implementation now has one shared home.
_GraphTransactionSnapshot = GraphTransactionSnapshot


@dataclass(frozen=True, slots=True)
class _AuxiliaryGPUPreview:
    """Validated read-only response and its actual backend provenance."""

    available: bool
    mean: float
    warning: str | None
    backend_used: str | None
    fallback_used: bool | None


def _gpu_preview_graph(graph: PartitionBlock) -> PartitionBlock:
    """Return a detached graph without invoking the subclass constructor."""

    # deepcopy preserves directed/multigraph semantics and subclasses whose
    # constructors require domain arguments. The preview cannot mutate the
    # canonical graph, including nested node, edge, or graph attributes.
    return deepcopy(graph)


def _reject_context_factor_overrides(ctx: StrategyContext) -> None:
    """Keep the graph's canonical factor registry as the only runtime source."""

    conflicts = tuple(
        sorted(_CONTEXT_FACTOR_OVERRIDE_KEYS.intersection(ctx.dispatcher_capabilities))
    )
    if conflicts:
        raise ValueError(
            "GPU strategies do not accept factor overrides through "
            f"StrategyContext ({', '.join(conflicts)}); configure the block "
            "graph's GLYPH_FACTORS mapping so canonical factor validation applies"
        )


def _compute_auxiliary_gpu_preview(
    engine: Any, graph: PartitionBlock
) -> _AuxiliaryGPUPreview:
    """Read a backend-specific diagnostic that never authorizes TNFR writes.

    Legacy or custom GPU readers may not implement every weighted/directed
    convention of the canonical CPU ΔNFR kernel.  A preview is never used to
    authorize a state transition, even when its reader is fully canonical.
    """

    try:
        raw = engine.compute_delta_nfr_from_graph(_gpu_preview_graph(graph))
        if not isinstance(raw, Mapping):
            raise TypeError("Auxiliary GPU preview must return a node mapping")
        values: list[float] = []
        for node in graph.nodes:
            value = raw[node]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(
                    f"Auxiliary GPU preview for node {node!r} must be real"
                )
            resolved = float(value)
            if not math.isfinite(resolved):
                raise ValueError(
                    f"Auxiliary GPU preview for node {node!r} must be finite"
                )
            values.append(resolved)
        mean = math.fsum(values) / len(values) if values else 0.0
        backend_used = getattr(raw, "backend_used", None)
        fallback_used = getattr(raw, "fallback_used", None)
        return _AuxiliaryGPUPreview(
            available=True,
            mean=mean,
            warning=None,
            backend_used=(str(backend_used) if backend_used is not None else None),
            fallback_used=(
                bool(fallback_used) if fallback_used is not None else None
            ),
        )
    except Exception as exc:
        return _AuxiliaryGPUPreview(
            available=False,
            mean=0.0,
            warning=f"Auxiliary GPU preview failed: {exc}",
            backend_used=None,
            fallback_used=None,
        )


def _history_ends_with(graph: PartitionBlock, node: Any, glyph: str) -> bool:
    history = graph.nodes[node].get("glyph_history")
    if history is None:
        return False
    try:
        items = list(history)
    except TypeError:
        return False
    if not items:
        return False
    token = getattr(items[-1], "value", items[-1])
    token = str(token)
    if token.startswith("Glyph."):
        token = token.split(".", 1)[1]
    return token.upper() == glyph


def _discard_pending_monitor(graph: PartitionBlock) -> None:
    monitor = graph.graph.get("integrity_monitor")
    discard = getattr(monitor, "discard_pending_operator", None)
    if callable(discard):
        try:
            discard()
        except Exception:
            pass


def _apply_canonical_block(
    graph: PartitionBlock,
    glyph: str,
    *,
    transaction_snapshot: GraphTransactionSnapshot | None = None,
) -> int:
    """Apply one canonical block under its explicit network schedule."""

    from ..definitions import Emission, Resonance

    if glyph not in {"AL", "RA"}:
        raise ValueError(f"Unsupported canonical GPU block glyph: {glyph!r}")
    operator_type = Emission if glyph == "AL" else Resonance
    snapshot = transaction_snapshot or GraphTransactionSnapshot(graph)
    nodes = tuple(graph.nodes)
    try:
        operator = operator_type()
        if glyph == "RA":
            execute_neighbor_stage(
                graph,
                operator,
                nodes,
                compute_delta_nfr=graph.graph.get("compute_delta_nfr"),
                transaction_snapshot=snapshot,
            )
        else:
            for node in nodes:
                operator(graph, node)
            record_gauss_seidel_stage(graph, operator, len(nodes))
        for node in nodes:
            if not _history_ends_with(graph, node, glyph):
                raise RuntimeError(
                    f"Canonical grammar did not accept {glyph} for node {node!r}"
                )
    except BaseException:
        _discard_pending_monitor(graph)
        snapshot.restore(graph)
        raise
    return len(nodes)


def _preview_telemetry(preview: _AuxiliaryGPUPreview) -> dict[str, Any]:
    """Describe the diagnostic/commit boundary without guessing provenance."""

    return {
        # The state transition below always uses canonical public operators.
        "gpu_acceleration": False,
        "backend": "canonical-cpu",
        "auxiliary_gpu_preview": preview.available,
        "auxiliary_gpu_preview_backend": preview.backend_used,
        "auxiliary_gpu_preview_fallback_used": preview.fallback_used,
        "auxiliary_gpu_preview_mean": (
            preview.mean if preview.available else None
        ),
    }


def _operation_proof_hash(
    graph: PartitionBlock,
    ctx: StrategyContext,
    glyph: str,
    *,
    outcome: str,
) -> str:
    """Bind a proof to its partition, position, operator, and resulting state."""

    from ...dynamics.multi_modal_cache import cache_signature_digest

    state_digest = cache_signature_digest(graph)
    proof_scope = (
        ctx.partition_id,
        ctx.operator_sequence_position,
        glyph,
        outcome,
        ctx.seed,
        state_digest,
    )
    return hashlib.sha3_256(repr(proof_scope).encode("utf-8")).hexdigest()


def _auxiliary_preview_resource_estimate(ctx: StrategyContext) -> ResourceEstimate:
    """Return only the materialized memory bound derivable from block size."""

    node_count = max(0, int(ctx.block_size))
    # One dense float64 operator plus the EPI input and pressure output.
    memory_bytes = 8 * (node_count * node_count + 2 * node_count)
    return ResourceEstimate(
        memory_bytes=memory_bytes,
        time_ms=None,
        delta_nfr=None,
        phi_s_drift=None,
        failure_risk="unknown",
    )


class GPUEmissionStrategy:
    """Canonical Emission with an optional auxiliary GPU preview."""

    operator = "AL"

    def __init__(self):
        self.gpu_engine: TNFRUnifiedGPUSystem | None = None
        if HAS_GPU_ENGINE:
            try:
                self.gpu_engine = get_unified_gpu_system()
            except Exception:
                pass

    def supports(self, ctx: StrategyContext) -> bool:
        """Check whether the optional GPU preview is available for this block."""
        if not HAS_GPU_ENGINE or self.gpu_engine is None:
            return False

        # Reserve the auxiliary preview for networks above its setup threshold
        if ctx.block_size < 200:
            return False

        return ctx.backend == "gpu" and self.gpu_engine.is_available

    def resource_estimate(self, ctx: StrategyContext) -> ResourceEstimate:
        """Estimate auxiliary-preview resources for an Emission block."""
        return _auxiliary_preview_resource_estimate(ctx)

    def prepare(self, ctx: StrategyContext, block: PartitionBlock) -> PreparedBlock:
        """Prepare graph block for GPU processing."""
        if not self.supports(ctx):
            raise RuntimeError("GPU strategy not supported for this context")
        _reject_context_factor_overrides(ctx)

        # Extract graph from block
        graph = block  # Assume block is the graph for now

        # Pre-compute GPU-friendly representations
        prepared = {
            "graph": graph,
            "gpu_ready": True,
            "context": ctx,
            "engine": self.gpu_engine,
        }

        return prepared

    def apply(self, prepared: PreparedBlock) -> OperationResult:
        """Apply canonical Emission after an optional read-only GPU preview."""
        graph = prepared["graph"]
        engine = prepared["engine"]
        ctx = prepared["context"]

        preview = _compute_auxiliary_gpu_preview(engine, graph)
        warnings = [] if preview.warning is None else [preview.warning]
        transaction = GraphTransactionSnapshot(graph)
        try:
            nodes_processed = _apply_canonical_block(
                graph, "AL", transaction_snapshot=transaction
            )
            gpu_available = engine.is_available
            telemetry = {
                "operator": "AL",
                "strategy": "gpu_emission",
                "canonical_commit": True,
                "update_schedule": OPERATOR_MAJOR_GAUSS_SEIDEL,
                "nodes_processed": nodes_processed,
                "gpu_available": gpu_available,
                **_preview_telemetry(preview),
            }

            return OperationResult(
                block=graph,
                telemetry=telemetry,
                warnings=warnings,
                proof_hash=_operation_proof_hash(graph, ctx, "AL", outcome="committed"),
            )

        except Exception as e:
            _discard_pending_monitor(graph)
            transaction.restore(graph)
            warnings.append(f"Canonical Emission transaction failed: {e}")
            return OperationResult(
                block=graph,
                telemetry={
                    "operator": "AL",
                    "strategy": "gpu_emission",
                    "gpu_acceleration": False,
                    "auxiliary_gpu_preview": preview.available,
                    "canonical_commit": False,
                    "rolled_back": True,
                    "error": str(e),
                },
                warnings=warnings,
                proof_hash=_operation_proof_hash(graph, ctx, "AL", outcome="rejected"),
            )

    def cleanup(self, prepared: PreparedBlock) -> None:
        """Clean up GPU resources."""
        # GPU memory cleanup is handled by the engine


class GPUResonanceStrategy:
    """Canonical Resonance with an optional auxiliary GPU preview."""

    operator = "RA"

    def __init__(self):
        self.gpu_engine: TNFRUnifiedGPUSystem | None = None
        if HAS_GPU_ENGINE:
            try:
                self.gpu_engine = get_unified_gpu_system()
            except Exception:
                pass

    def supports(self, ctx: StrategyContext) -> bool:
        """Check whether the optional GPU preview is available for this block."""
        if not HAS_GPU_ENGINE or self.gpu_engine is None:
            return False

        # Reserve the auxiliary preview for resonance-sized matrix workloads
        if ctx.block_size < 100:
            return False

        return ctx.backend == "gpu" and self.gpu_engine.is_available

    def resource_estimate(self, ctx: StrategyContext) -> ResourceEstimate:
        """Estimate auxiliary-preview resources for a Resonance block."""
        return _auxiliary_preview_resource_estimate(ctx)

    def prepare(self, ctx: StrategyContext, block: PartitionBlock) -> PreparedBlock:
        """Prepare for GPU resonance processing."""
        if not self.supports(ctx):
            raise RuntimeError("GPU strategy not supported for this context")
        _reject_context_factor_overrides(ctx)
        return {
            "graph": block,
            "gpu_ready": True,
            "context": ctx,
            "engine": self.gpu_engine,
        }

    def apply(self, prepared: PreparedBlock) -> OperationResult:
        """Apply canonical Resonance after an optional read-only GPU preview."""
        graph = prepared["graph"]
        engine = prepared["engine"]
        ctx = prepared["context"]

        preview = _compute_auxiliary_gpu_preview(engine, graph)
        warnings = [] if preview.warning is None else [preview.warning]
        transaction = GraphTransactionSnapshot(graph)
        try:
            frequency_before = {
                node: float(get_attr(graph.nodes[node], ALIAS_VF, 0.0))
                for node in graph
            }
            nodes_processed = _apply_canonical_block(
                graph, "RA", transaction_snapshot=transaction
            )
            amplified_nodes = tuple(
                node
                for node, before in frequency_before.items()
                if float(get_attr(graph.nodes[node], ALIAS_VF, 0.0)) > before
            )
            telemetry = {
                "operator": "RA",
                "strategy": "gpu_resonance",
                "canonical_commit": True,
                "update_schedule": TWO_PHASE_JACOBI,
                "nodes_processed": nodes_processed,
                "resonance_amplification": bool(amplified_nodes),
                "resonance_amplified_nodes": len(amplified_nodes),
                "phase_sync_computed": True,
                **_preview_telemetry(preview),
            }

            return OperationResult(
                block=graph,
                telemetry=telemetry,
                warnings=warnings,
                proof_hash=_operation_proof_hash(graph, ctx, "RA", outcome="committed"),
            )

        except Exception as e:
            _discard_pending_monitor(graph)
            transaction.restore(graph)
            warnings.append(f"Canonical Resonance transaction failed: {e}")
            return OperationResult(
                block=graph,
                telemetry={
                    "operator": "RA",
                    "strategy": "gpu_resonance",
                    "gpu_acceleration": False,
                    "auxiliary_gpu_preview": preview.available,
                    "canonical_commit": False,
                    "rolled_back": True,
                    "error": str(e),
                },
                warnings=warnings,
                proof_hash=_operation_proof_hash(graph, ctx, "RA", outcome="rejected"),
            )

    def cleanup(self, prepared: PreparedBlock) -> None:
        """Clean up GPU resources."""


def register_all_gpu_strategies() -> None:
    """Register all GPU strategies with the strategy registry."""
    if not HAS_GPU_ENGINE:
        return  # Skip if GPU engine not available

    try:
        # Register GPU emission strategy
        StrategyRegistry.register(
            operator="AL", name="gpu_emission", factory=lambda: GPUEmissionStrategy()
        )

        # Register GPU resonance strategy
        StrategyRegistry.register(
            operator="RA", name="gpu_resonance", factory=lambda: GPUResonanceStrategy()
        )

        print("GPU strategies registered successfully")

    except Exception as e:
        print(f"Failed to register GPU strategies: {e}")


def get_gpu_strategy_recommendations(graph_size: int) -> dict[str, Any]:
    """Describe CPU commit and optional preview without unmeasured estimates."""

    if isinstance(graph_size, bool) or not isinstance(graph_size, Integral):
        raise TypeError("graph_size must be a nonnegative integer")
    if graph_size < 0:
        raise ValueError("graph_size must be nonnegative")

    recommendations: dict[str, Any] = {
        "use_gpu": False,
        "preferred_operators": [],
        "memory_estimate_mb": None,
        "speedup_factor": None,
        "canonical_execution_backend": "canonical-cpu",
        "auxiliary_preview_available": False,
        "auxiliary_preview_backend": None,
        "auxiliary_preview_memory_estimate_mb": None,
    }

    if not HAS_GPU_ENGINE:
        recommendations["reason"] = "GPU adapter is not available"
        return recommendations
    if graph_size < 100:
        recommendations["reason"] = (
            "The optional read-only preview is disabled below 100 nodes"
        )
        return recommendations

    try:
        engine = get_unified_gpu_system()
        if not engine.is_available:
            recommendations["reason"] = "GPU hardware is not available"
            return recommendations

        # Runtime backend and resource evidence belong to the preview result.
        # No speedup, peak-memory value, or GPU state transition is inferred.
        recommendations["auxiliary_preview_available"] = True
        recommendations["reason"] = (
            "AL and RA commit through canonical CPU operators; an optional "
            "read-only preview may be collected, with backend provenance "
            "reported only after execution"
        )
    except Exception as exc:
        recommendations["reason"] = f"GPU adapter initialization failed: {exc}"

    return recommendations
