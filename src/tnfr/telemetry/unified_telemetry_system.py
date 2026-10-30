"""Buffered storage for caller-supplied structural, performance and failure events.

The three channels share detached payload capture and atomic UTF-8 JSON/JSONL
batch writing. Accepted buffers survive failed writes; each successful batch
has its own filename. This is an optional event sink, not a field calculator,
trajectory validator or replacement for every specialized telemetry owner.

Wall-clock timestamps describe collection, not the model's structural clock.
Callers must retain scientific provenance and unavailable-field evidence.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .._exact_time import finite_represented_real
from ..config import get_config
from ..utils.io import _reject_duplicate_json_keys, json_dumps, safe_write
from ..validation.window import validate_window

logger = logging.getLogger(__name__)


@dataclass
class TelemetryConfiguration:
    """Configuration for unified telemetry system."""

    # Collection settings
    enable_telemetry: bool = True
    enable_structural_telemetry: bool = True
    enable_performance_telemetry: bool = True
    enable_failure_telemetry: bool = True

    # Batching and storage
    batch_size: int = 100
    flush_interval_seconds: float = 30.0
    max_memory_buffer_mb: float = 50.0  # Reserved; no memory eviction policy.

    # Output configuration
    output_directory: Path = Path("results/telemetry")
    file_format: str = "jsonl"  # Supported: "jsonl", "json".
    enable_correlation_tracking: bool = True

    # Performance tuning
    async_emission: bool = True
    compression_enabled: bool = True  # Reserved; output is uncompressed.

    # Reserved filtering controls; events are not filtered by these fields.
    min_event_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    event_type_filters: list[str] = field(default_factory=list)


@dataclass
class StructuralTelemetryEvent:
    """Telemetry event for TNFR structural measurements."""

    # Event metadata
    event_id: str
    correlation_id: str
    timestamp: float
    node_id: str | None = None

    # Supplied field observations; None retains missing/unavailable evidence.
    phi_s: float | None = None  # Structural potential
    phase_gradient: float | None = None  # |∇φ|
    phase_curvature: float | None = None  # K_φ
    coherence_length: float | None = None  # ξ_C

    # Core TNFR metrics
    coherence: float | None = None  # C(t)
    sense_index: float | None = None  # Si
    delta_nfr: float | None = None  # ΔNFR
    vf: float | None = None  # νf
    phase: float | None = None  # φ/θ
    epi: float | None = None  # EPI

    # System state
    operator_sequence: list[str] | None = None
    system_status: str = "normal"

    # Additional context
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceTelemetryEvent:
    """Telemetry event for performance monitoring."""

    # Event metadata
    event_id: str
    correlation_id: str
    timestamp: float

    # Performance metrics
    operation_name: str
    duration_ms: float
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0

    # Throughput metrics
    operations_per_second: float | None = None
    data_throughput_mbps: float | None = None

    # Resource utilization
    backend_used: str = "unknown"
    device_used: str | None = None
    cache_hit_rate: float | None = None

    # Quality metrics
    success: bool = True
    error_message: str | None = None

    # Context
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureTelemetryEvent:
    """Telemetry event for failure analysis."""

    # Event metadata
    event_id: str
    correlation_id: str
    timestamp: float

    # Failure details
    failure_type: str  # "computational", "memory", "convergence", "validation"
    error_code: str | None = None
    error_message: str = ""
    stack_trace: str | None = None

    # System context at failure
    system_state: dict[str, Any] = field(default_factory=dict)
    operation_context: dict[str, Any] = field(default_factory=dict)

    # Recovery information
    recovery_attempted: bool = False
    recovery_successful: bool = False
    fallback_used: str | None = None

    # Impact assessment
    severity: str = "medium"  # "low", "medium", "high", "critical"
    affected_operations: list[str] = field(default_factory=list)

    # Additional context
    metadata: dict[str, Any] = field(default_factory=dict)


class TNFRUnifiedTelemetrySystem:
    """Collect supplied events in three channels with shared batching and storage.

    Emission captures detached JSON-compatible data. Manual writes raise on
    failure and retain accepted events for retry; timer failures are logged.
    Cleanup stops future emission/timers and flushes retained data. A failed
    cleanup can be retried. The shared writer syncs and replaces each file;
    buffered events do not have a process-crash recovery guarantee. Metadata
    must remain finite JSON-compatible data.

    Usage:
        # Optional event sink
        telemetry = TNFRUnifiedTelemetrySystem()

        # Structural measurements
        telemetry.emit_structural_event(
            phi_s=structural_potential,
            coherence=coherence_value,
            correlation_id=session_id
        )

        # Performance monitoring
        telemetry.emit_performance_event(
            operation_name="delta_nfr_computation",
            duration_ms=computation_time,
            backend_used="torch"
        )

        # Failure tracking
        telemetry.emit_failure_event(
            failure_type="memory",
            error_message="GPU out of memory",
            system_state=current_state
        )

    Memory limits, compression and severity/type filters in the configuration
    are reserved compatibility fields, not implemented guarantees.
    """

    def __init__(self, config: TelemetryConfiguration | None = None):
        """Initialize unified telemetry system."""
        self.config = config or TelemetryConfiguration()

        # Event buffers for batching
        self._structural_buffer: deque = deque()
        self._performance_buffer: deque = deque()
        self._failure_buffer: deque = deque()

        # Threading for async emission
        self._flush_lock = threading.RLock()
        self._flush_timer: threading.Timer | None = None
        self._closed = False

        # Correlation tracking
        self._active_correlations: dict[str, dict[str, Any]] = {}
        self._correlation_counters: dict[str, int] = {}

        # Performance statistics
        self._emission_stats = {
            "total_events": 0,
            "structural_events": 0,
            "performance_events": 0,
            "failure_events": 0,
            "flush_count": 0,
            "bytes_emitted": 0,
        }

        # Integration with unified systems
        self.global_config = get_config()

        self._file_format()
        validate_window(self.config.batch_size, positive=True)
        # A disabled sink creates neither files nor a periodic worker.
        if self._enabled("enable_telemetry") and self._flag("async_emission"):
            self._schedule_flush()

        logger.info(
            f"Initialized unified telemetry system: {self.config.output_directory}"
        )

    def _flag(self, name: str) -> bool:
        value = getattr(self.config, name)
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean")
        return value

    def _enabled(self, channel: str) -> bool:
        with self._flush_lock:
            if self._closed:
                raise RuntimeError("telemetry sink is closed")
            return self._flag("enable_telemetry") and self._flag(channel)

    def _file_format(self) -> str:
        if self.config.file_format not in ("jsonl", "json"):
            raise ValueError("file_format must be 'jsonl' or 'json'")
        return self.config.file_format

    def _enqueue(self, channel: str, event: Any) -> str:
        """Capture, count and flush one event through the common channel path."""
        with self._flush_lock:
            if not self._enabled(f"enable_{channel}_telemetry"):
                return ""
            batch_size = validate_window(self.config.batch_size, positive=True)
            self._file_format()
            track = self._flag("enable_correlation_tracking")
            # asdict detaches nested caller metadata before it enters a buffer.
            payload = asdict(event)
            encoded = json_dumps(payload, allow_nan=False, ensure_ascii=False)
            encoded.encode("utf-8")
            json.loads(encoded, object_pairs_hook=_reject_duplicate_json_keys)
            buffer = getattr(self, f"_{channel}_buffer")
            buffer.append(payload)
            self._emission_stats[f"{channel}_events"] += 1
            self._emission_stats["total_events"] += 1
            if track:
                self._update_correlation_tracking(
                    event.correlation_id, channel, event.event_id
                )
            if channel == "failure" or len(buffer) >= batch_size:
                self._flush_events(channel)
            return event.event_id

    def emit_structural_event(
        self,
        correlation_id: str | None = None,
        node_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Emit structural telemetry event with TNFR field measurements.

        Parameters
        ----------
        correlation_id : str, optional
            Correlation ID for event tracking (auto-generated if not provided)
        node_id : str, optional
            Node identifier for the measurement
        **kwargs
            Structural field values (phi_s, phase_gradient, coherence, etc.)

        Returns
        -------
        str
            Event ID for the emitted event
        """
        if not self._enabled("enable_structural_telemetry"):
            return ""

        event_id = str(uuid.uuid4())
        correlation_id = self._resolve_correlation_id(correlation_id, "structural")

        event = StructuralTelemetryEvent(
            event_id=event_id,
            correlation_id=correlation_id,
            timestamp=time.time(),
            node_id=node_id,
            **kwargs,
        )

        return self._enqueue("structural", event)

    def emit_performance_event(
        self,
        operation_name: str,
        duration_ms: float,
        correlation_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Emit performance telemetry event for operation monitoring.

        Parameters
        ----------
        operation_name : str
            Name of the operation being monitored
        duration_ms : float
            Operation duration in milliseconds
        correlation_id : str, optional
            Correlation ID for event tracking
        **kwargs
            Additional performance metrics

        Returns
        -------
        str
            Event ID for the emitted event
        """
        if not self._enabled("enable_performance_telemetry"):
            return ""

        event_id = str(uuid.uuid4())
        correlation_id = self._resolve_correlation_id(correlation_id, "performance")

        event = PerformanceTelemetryEvent(
            event_id=event_id,
            correlation_id=correlation_id,
            timestamp=time.time(),
            operation_name=operation_name,
            duration_ms=duration_ms,
            **kwargs,
        )

        return self._enqueue("performance", event)

    def emit_failure_event(
        self,
        failure_type: str,
        error_message: str = "",
        correlation_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Emit failure telemetry event for error analysis.

        Parameters
        ----------
        failure_type : str
            type of failure (computational, memory, convergence, etc.)
        error_message : str
            Description of the failure
        correlation_id : str, optional
            Correlation ID for event tracking
        **kwargs
            Additional failure context and metadata

        Returns
        -------
        str
            Event ID for the emitted event
        """
        if not self._enabled("enable_failure_telemetry"):
            return ""

        event_id = str(uuid.uuid4())
        correlation_id = self._resolve_correlation_id(correlation_id, "failure")

        event = FailureTelemetryEvent(
            event_id=event_id,
            correlation_id=correlation_id,
            timestamp=time.time(),
            failure_type=failure_type,
            error_message=error_message,
            **kwargs,
        )

        return self._enqueue("failure", event)

    def start_correlation(
        self, correlation_name: str, context: dict[str, Any] | None = None
    ) -> str:
        """Start a new correlation session for tracking related events.

        Parameters
        ----------
        correlation_name : str
            Human-readable name for the correlation
        context : dict, optional
            Initial context for the correlation

        Returns
        -------
        str
            Correlation ID for tracking events
        """
        with self._flush_lock:
            if not self._enabled("enable_correlation_tracking"):
                return ""
            correlation_id = self._generate_correlation_id(correlation_name)
            self._active_correlations[correlation_id] = {
                "name": correlation_name,
                "start_time": time.time(),
                "context": deepcopy(context) if context is not None else {},
                "event_count": 0,
                "event_types": set(),
            }
            return correlation_id

    def end_correlation(self, correlation_id: str) -> dict[str, Any]:
        """End a correlation session and return summary.

        Parameters
        ----------
        correlation_id : str
            Correlation ID to end

        Returns
        -------
        dict
            Correlation summary with event statistics
        """
        with self._flush_lock:
            if correlation_id not in self._active_correlations:
                return {"error": "correlation_not_found"}
            correlation = self._active_correlations.pop(correlation_id)
            correlation["end_time"] = time.time()
            correlation["duration"] = (
                correlation["end_time"] - correlation["start_time"]
            )
            return correlation

    def flush_all(self) -> None:
        """Flush all telemetry buffers to storage immediately."""
        with self._flush_lock:
            self._flush_structural_events()
            self._flush_performance_events()
            self._flush_failure_events()
            self._emission_stats["flush_count"] += 1

    def _generate_correlation_id(self, prefix: str) -> str:
        """Generate a unique correlation ID with prefix."""
        with self._flush_lock:
            counter = self._correlation_counters.get(prefix, 0) + 1
            self._correlation_counters[prefix] = counter
            return f"{prefix}_{int(time.time())}_{counter}_{str(uuid.uuid4())[:8]}"

    def _resolve_correlation_id(self, value: str | None, prefix: str) -> str:
        """Admit caller IDs before tracking can modify an accepted event."""
        if value is not None and not isinstance(value, str):
            raise TypeError("correlation_id must be a string or None")
        return str(value) if value else self._generate_correlation_id(prefix)

    def _update_correlation_tracking(
        self, correlation_id: str, event_type: str, event_id: str
    ) -> None:
        """Update correlation tracking with new event."""
        if correlation_id in self._active_correlations:
            correlation = self._active_correlations[correlation_id]
            correlation["event_count"] += 1
            correlation["event_types"].add(event_type)
            correlation["last_event_id"] = event_id
            correlation["last_event_time"] = time.time()

    def _flush_structural_events(self) -> None:
        """Flush structural events to storage."""
        self._flush_events("structural")

    def _flush_performance_events(self) -> None:
        """Flush performance events to storage."""
        self._flush_events("performance")

    def _flush_failure_events(self) -> None:
        """Flush failure events to storage."""
        self._flush_events("failure")

    def _flush_events(self, channel: str) -> None:
        """Remove captured events only after their atomic write succeeds."""
        with self._flush_lock:
            buffer = getattr(self, f"_{channel}_buffer")
            if not buffer:
                return
            file_format = self._file_format()
            events = list(buffer)
            filename = Path(self.config.output_directory) / (
                f"{channel}_telemetry_{int(time.time())}_{uuid.uuid4().hex}.{file_format}"
            )
            self._write_events_to_file(events, filename)
            for _ in events:
                buffer.popleft()

    def _write_events_to_file(self, events: list[Any], filename: Path) -> None:
        """Serialize before opening a destination and reuse the shared writer."""
        if self._file_format() == "jsonl":
            payload = "".join(
                json_dumps(event, allow_nan=False, ensure_ascii=False) + "\n"
                for event in events
            )
        else:
            payload = json_dumps(events, allow_nan=False, ensure_ascii=False, indent=2)
        encoded = payload.encode("utf-8")
        safe_write(filename, lambda stream: stream.write(encoded), mode="wb")
        self._emission_stats["bytes_emitted"] += len(encoded)

    def _schedule_flush(self) -> None:
        """Schedule periodic flushing of telemetry buffers."""
        with self._flush_lock:
            if (
                self._closed
                or not self._flag("enable_telemetry")
                or not self._flag("async_emission")
            ):
                return
            interval = finite_represented_real(
                self.config.flush_interval_seconds, "flush_interval_seconds"
            )[0]
            if interval <= 0.0:
                raise ValueError("flush_interval_seconds must be positive")
            if self._flush_timer:
                self._flush_timer.cancel()
            self._flush_timer = threading.Timer(interval, self._periodic_flush)
            self._flush_timer.daemon = True
            self._flush_timer.start()

    def _periodic_flush(self) -> None:
        """Periodic flush callback."""
        try:
            self.flush_all()
        except Exception as e:
            logger.error(f"Periodic flush failed: {e}")
        finally:
            # Schedule next flush
            self._schedule_flush()

    def get_statistics(self) -> dict[str, Any]:
        """Get telemetry system statistics."""
        with self._flush_lock:
            stats = self._emission_stats.copy()
            stats.update(
                {
                    "buffer_sizes": {
                        "structural": len(self._structural_buffer),
                        "performance": len(self._performance_buffer),
                        "failure": len(self._failure_buffer),
                    },
                    "active_correlations": len(self._active_correlations),
                    "correlation_types": len(self._correlation_counters),
                    "config": asdict(self.config),
                    "closed": self._closed,
                }
            )
            return stats

    def cleanup(self) -> None:
        """Stop collection and flush; failed writes retain their buffers for retry."""
        with self._flush_lock:
            self._closed = True
            if self._flush_timer:
                self._flush_timer.cancel()
                self._flush_timer = None
            self.flush_all()
        logger.info("Unified telemetry system cleanup completed")


# ============================================================================
# PUBLIC API - Unified Telemetry Interface
# ============================================================================

# Global unified telemetry system instance
_unified_telemetry_system: TNFRUnifiedTelemetrySystem | None = None


def get_unified_telemetry_system(
    config: TelemetryConfiguration | None = None,
) -> TNFRUnifiedTelemetrySystem:
    """Get or create global unified telemetry system.

    Configuration is consumed on first construction. This optional event sink
    remains separate from field computation and specialized telemetry owners.

    Parameters
    ----------
    config : TelemetryConfiguration, optional
        Configuration for system (only used on first call)

    Returns
    -------
    TNFRUnifiedTelemetrySystem
        Global unified telemetry system instance
    """
    global _unified_telemetry_system

    if _unified_telemetry_system is None:
        _unified_telemetry_system = TNFRUnifiedTelemetrySystem(config)
        logger.info("Created global unified telemetry system")

    return _unified_telemetry_system


# Convenience functions for direct telemetry operations
def emit_structural_telemetry(**kwargs: Any) -> str:
    """Emit structural telemetry - convenience function."""
    return get_unified_telemetry_system().emit_structural_event(**kwargs)


def emit_performance_telemetry(
    operation_name: str, duration_ms: float, **kwargs: Any
) -> str:
    """Emit performance telemetry - convenience function."""
    return get_unified_telemetry_system().emit_performance_event(
        operation_name, duration_ms, **kwargs
    )


def emit_failure_telemetry(
    failure_type: str, error_message: str = "", **kwargs: Any
) -> str:
    """Emit failure telemetry - convenience function."""
    return get_unified_telemetry_system().emit_failure_event(
        failure_type, error_message, **kwargs
    )


def flush_unified_telemetry() -> None:
    """Flush unified telemetry buffers - convenience function."""
    if _unified_telemetry_system is not None:
        _unified_telemetry_system.flush_all()


def get_unified_telemetry_stats() -> dict[str, Any]:
    """Get unified telemetry statistics - convenience function."""
    if _unified_telemetry_system is not None:
        return _unified_telemetry_system.get_statistics()
    return {"status": "system_not_initialized"}
