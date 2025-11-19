"""TNFR Unified Telemetry Engine

Provides structured, batched emission of TNFR telemetry data preserving
canonical semantics while enabling efficient monitoring and analysis.

Design Principles:
- Read-only: Never mutates EPI, operators, or graph state
- Batched: Collects events in memory, flushes periodically
- Structured: JSONL with correlation IDs for trace analysis
- Canonical: Uses official field computations (Φ_s, |∇φ|, K_φ, ξ_C)
- Minimal overhead: Optional/configurable, non-blocking
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from collections import deque

try:
    from ..physics.fields import (
        compute_structural_potential,
        compute_phase_gradient,
        compute_phase_curvature,
        estimate_coherence_length
    )
    from ..operators.metrics import compute_coherence, compute_sense_index
    from ..constants.aliases import ALIAS_THETA, ALIAS_DNFR
    _FIELDS_AVAILABLE = True
except ImportError:
    _FIELDS_AVAILABLE = False


@dataclass
class TelemetryEvent:
    """Single telemetry event with TNFR semantics."""
    event_id: str
    correlation_id: str
    timestamp: float
    event_type: str

    # Graph state
    num_nodes: int
    num_edges: int

    # Canonical fields (if computed)
    phi_s_mean: Optional[float] = None
    phi_s_max: Optional[float] = None
    phi_s_drift: Optional[float] = None

    phase_grad_mean: Optional[float] = None
    phase_grad_max: Optional[float] = None
    phase_grad_over_threshold: Optional[int] = None

    phase_curv_mean: Optional[float] = None
    phase_curv_max: Optional[float] = None
    phase_curv_abs_max: Optional[float] = None

    coherence_length: Optional[float] = None

    # Core metrics
    coherence: Optional[float] = None
    sense_index: Optional[float] = None

    # Node-level aggregates
    nu_f_mean: Optional[float] = None
    nu_f_min: Optional[float] = None

    dnfr_mean: Optional[float] = None
    dnfr_max: Optional[float] = None
    dnfr_std: Optional[float] = None

    phase_mean: Optional[float] = None
    phase_std: Optional[float] = None

    # Operator context
    last_operator: Optional[str] = None
    sequence_step: Optional[int] = None

    # Custom metadata
    metadata: Optional[Dict[str, Any]] = None


class TelemetryEmitter:
    """Unified TNFR telemetry collection and emission."""

    def __init__(
        self,
        output_path: Optional[Union[str, Path]] = None,
        buffer_size: int = 100,
        auto_flush: bool = True,
        compute_fields: bool = True,
        field_thresholds: Optional[Dict[str, float]] = None,
        *,
        batch_size: Optional[int] = None,
    ):
        """Initialize telemetry emitter.

        Parameters
        ----------
        output_path : str or Path, optional
            JSONL output file path. If None, uses in-memory only.
        buffer_size : int
            Number of events to buffer before flushing
        batch_size : int, optional
            Deprecated alias for buffer_size (accepted for backward
            compatibility)
        auto_flush : bool
            Whether to auto-flush buffer when full
        compute_fields : bool
            Whether to compute structural fields (Φ_s, |∇φ|, K_φ, ξ_C)
        field_thresholds : dict, optional
            Thresholds for field warnings (phase_grad: 0.38, etc.)
        """
        self.output_path = Path(output_path) if output_path else None
        # Support deprecated alias "batch_size" to maintain backwards
        # compatibility
        if batch_size is not None:
            self.buffer_size = int(batch_size)
        else:
            self.buffer_size = int(buffer_size)
        self.auto_flush = auto_flush
        self.compute_fields = compute_fields and _FIELDS_AVAILABLE

        # Default field thresholds from canonical research
        default_thresholds = {
            'phase_grad': 0.38,  # |∇φ| stability threshold
            'phase_curv_abs': 3.0,  # |K_φ| confinement threshold
            'phi_s_drift': 2.0,  # Φ_s escape threshold (U6)
        }
        if field_thresholds:
            default_thresholds.update(field_thresholds)
        self.thresholds = default_thresholds

        self.event_buffer: deque = deque()
        self.correlation_id = str(uuid.uuid4())
        self.sequence_step = 0

        # Performance tracking
        self.stats = {
            'events_emitted': 0,
            'flushes': 0,
            'field_computations': 0,
            'warnings_generated': 0
        }

    def new_correlation(self) -> str:
        """Start new correlation ID for sequence boundary."""
        self.correlation_id = str(uuid.uuid4())
        self.sequence_step = 0
        return self.correlation_id

    def emit_graph_snapshot(
        self,
        graph: Any,
        event_type: str = "graph_snapshot",
        last_operator: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        phi_s_baseline: Optional[Dict[Any, float]] = None
    ) -> str:
        """Emit complete graph state snapshot with all canonical fields.

        Parameters
        ----------
        graph : TNFRGraph
            Graph to snapshot
        event_type : str
            Type of event ("graph_snapshot", "operator_applied", etc.)
        last_operator : str, optional
            Name of last applied operator
        metadata : dict, optional
            Additional metadata
        phi_s_baseline : dict, optional
            Baseline Φ_s values for drift computation

        Returns
        -------
        str
            Event ID of emitted event
        """
        event_id = str(uuid.uuid4())
        timestamp = time.time()

        # Basic graph properties
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        event = TelemetryEvent(
            event_id=event_id,
            correlation_id=self.correlation_id,
            timestamp=timestamp,
            event_type=event_type,
            num_nodes=num_nodes,
            num_edges=num_edges,
            last_operator=last_operator,
            sequence_step=self.sequence_step,
            metadata=metadata
        )

        if num_nodes == 0:
            # Empty graph - just emit basic event
            self._add_event(event)
            return event_id

        # Extract node-level data
        self._populate_node_aggregates(graph, event)

        # Compute canonical fields if enabled
        if self.compute_fields:
            self._populate_canonical_fields(graph, event, phi_s_baseline)
            self.stats['field_computations'] += 1

        # Core TNFR metrics
        self._populate_core_metrics(graph, event)

        # Check thresholds and generate warnings
        self._check_thresholds(event)

        self._add_event(event)
        self.sequence_step += 1

        return event_id

    def _populate_node_aggregates(self, graph: Any, event: TelemetryEvent) -> None:
        """Populate node-level aggregate statistics."""
        import numpy as np

        # Extract node attributes using TNFR aliases
        nu_f_values = []
        dnfr_values = []
        phase_values = []

        for node in graph.nodes():
            node_data = graph.nodes[node]

            # Structural frequency (νf)
            nu_f = node_data.get('nu_f', 0.0)
            nu_f_values.append(nu_f)

            # ΔNFR using alias system
            dnfr = 0.0
            for alias in ALIAS_DNFR:
                if alias in node_data:
                    dnfr = float(node_data[alias])
                    break
            dnfr_values.append(dnfr)

            # Phase using alias system
            phase = 0.0
            for alias in ALIAS_THETA:
                if alias in node_data:
                    phase = float(node_data[alias])
                    break
            phase_values.append(phase)

        # Compute aggregates
        if nu_f_values:
            event.nu_f_mean = float(np.mean(nu_f_values))
            event.nu_f_min = float(np.min(nu_f_values))

        if dnfr_values:
            event.dnfr_mean = float(np.mean(dnfr_values))
            event.dnfr_max = float(np.max(dnfr_values))
            event.dnfr_std = float(np.std(dnfr_values))

        if phase_values:
            event.phase_mean = float(np.mean(phase_values))
            event.phase_std = float(np.std(phase_values))

    def _populate_canonical_fields(
        self,
        graph: Any,
        event: TelemetryEvent,
        phi_s_baseline: Optional[Dict[Any, float]] = None
    ) -> None:
        """Populate canonical structural fields."""
        import numpy as np

        try:
            # Structural potential Φ_s [CANONICAL]
            phi_s = compute_structural_potential(graph)
            if phi_s:
                phi_s_vals = list(phi_s.values())
                event.phi_s_mean = float(np.mean(phi_s_vals))
                event.phi_s_max = float(np.max(phi_s_vals))

                # Compute drift if baseline provided
                if phi_s_baseline:
                    drifts = [
                        abs(phi_s[node] - phi_s_baseline.get(node, 0.0))
                        for node in phi_s.keys()
                    ]
                    event.phi_s_drift = float(np.mean(drifts))

            # Phase gradient |∇φ| [CANONICAL]
            phase_grad = compute_phase_gradient(graph)
            if phase_grad:
                grad_vals = list(phase_grad.values())
                event.phase_grad_mean = float(np.mean(grad_vals))
                event.phase_grad_max = float(np.max(grad_vals))
                # Count nodes over threshold
                threshold = self.thresholds.get('phase_grad', 0.38)
                event.phase_grad_over_threshold = sum(
                    1 for v in grad_vals if v >= threshold
                )

            # Phase curvature K_φ [CANONICAL]
            phase_curv = compute_phase_curvature(graph)
            if phase_curv:
                curv_vals = list(phase_curv.values())
                event.phase_curv_mean = float(np.mean(curv_vals))
                event.phase_curv_max = float(np.max(curv_vals))
                event.phase_curv_abs_max = float(np.max(np.abs(curv_vals)))

            # Coherence length ξ_C [CANONICAL]
            try:
                xi_c = estimate_coherence_length(graph)
                if xi_c is not None and np.isfinite(xi_c):
                    event.coherence_length = float(xi_c)
            except Exception:
                # ξ_C computation can fail for small/degenerate graphs
                pass

        except Exception as e:
            # Field computation failed - continue without fields
            if event.metadata is None:
                event.metadata = {}
            event.metadata['field_error'] = str(e)

    def _populate_core_metrics(self, graph: Any, event: TelemetryEvent) -> None:
        """Populate core TNFR metrics."""
        try:
            # Total coherence C(t)
            event.coherence = float(compute_coherence(graph))
        except Exception:
            pass

        try:
            # Sense index Si
            event.sense_index = float(compute_sense_index(graph))
        except Exception:
            pass

    def _check_thresholds(self, event: TelemetryEvent) -> None:
        """Check field values against thresholds and generate warnings."""
        warnings = []

        # Phase gradient threshold
        if (event.phase_grad_max is not None and
            event.phase_grad_max >= self.thresholds.get('phase_grad', 0.38)):
            warnings.append(f"High phase gradient: {event.phase_grad_max:.3f}")

        # Phase curvature threshold
        if (event.phase_curv_abs_max is not None and
            event.phase_curv_abs_max >= self.thresholds.get('phase_curv_abs', 3.0)):
            warnings.append(f"High phase curvature: {event.phase_curv_abs_max:.3f}")

        # Structural potential drift (U6)
        if (event.phi_s_drift is not None and
            event.phi_s_drift >= self.thresholds.get('phi_s_drift', 2.0)):
            warnings.append(f"Φ_s drift exceeds U6 threshold: {event.phi_s_drift:.3f}")

        if warnings:
            if event.metadata is None:
                event.metadata = {}
            event.metadata['warnings'] = warnings
            self.stats['warnings_generated'] += len(warnings)

    def _add_event(self, event: TelemetryEvent) -> None:
        """Add event to buffer and auto-flush if needed."""
        self.event_buffer.append(event)
        self.stats['events_emitted'] += 1

        if self.auto_flush and len(self.event_buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> int:
        """Flush event buffer to output file.

        Returns
        -------
        int
            Number of events flushed
        """
        if not self.event_buffer:
            return 0

        events_to_flush = list(self.event_buffer)
        self.event_buffer.clear()

        if self.output_path:
            # Write JSONL format
            with open(self.output_path, 'a', encoding='utf-8') as f:
                for event in events_to_flush:
                    json_line = json.dumps(asdict(event), separators=(',', ':'))
                    f.write(json_line + '\n')

        self.stats['flushes'] += 1
        return len(events_to_flush)

    def get_stats(self) -> Dict[str, Any]:
        """Get telemetry statistics."""
        return {
            **self.stats,
            'buffer_size': len(self.event_buffer),
            'correlation_id': self.correlation_id,
            'sequence_step': self.sequence_step,
            'compute_fields': self.compute_fields,
            'thresholds': self.thresholds.copy()
        }

    def close(self) -> None:
        """Close emitter and flush remaining events."""
        self.flush()


# Global default emitter instance
_default_emitter: Optional[TelemetryEmitter] = None


def get_default_emitter() -> TelemetryEmitter:
    """Get or create default telemetry emitter."""
    global _default_emitter
    if _default_emitter is None:
        _default_emitter = TelemetryEmitter()
    return _default_emitter


def emit_snapshot(
    graph: Any,
    event_type: str = "snapshot",
    **kwargs
) -> str:
    """Emit graph snapshot using default emitter."""
    return get_default_emitter().emit_graph_snapshot(
        graph, event_type=event_type, **kwargs
    )


def configure_telemetry(
    output_path: Optional[str] = None,
    **kwargs
) -> TelemetryEmitter:
    """Configure global telemetry emitter."""
    global _default_emitter
    _default_emitter = TelemetryEmitter(output_path=output_path, **kwargs)
    return _default_emitter


__all__ = [
    "TelemetryEvent",
    "TelemetryEmitter",
    "get_default_emitter",
    "emit_snapshot",
    "configure_telemetry"
]