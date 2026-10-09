"""Telemetry event storage, cache observations and reorganization-count estimates.

The optional event sink records caller-supplied values; it neither calculates
the tetrad nor authenticates model dynamics. Specialized graph/cache/count
observers retain their own contracts and are exposed alongside the sink.

Main Components:
- TNFRUnifiedTelemetrySystem: Consolidated event collection
- Structural telemetry: Tetrad field measurements
- Performance telemetry: Operation monitoring
- Failure telemetry: Error analysis
- Correlation tracking: Event relationship analysis

Usage:
```python
from tnfr.telemetry import get_unified_telemetry_system, emit_structural_telemetry
telemetry = get_unified_telemetry_system()
telemetry.emit_structural_event(coherence=0.85, phi_s=0.6)
# Or use convenience function
emit_structural_telemetry(coherence=0.85, phi_s=0.6)
```
"""

# Specialized observation owners
from .cache_metrics import (
    CacheMetricsSnapshot,
    CacheTelemetryPublisher,
    ensure_cache_metrics_publisher,
    publish_graph_cache_metrics,
)
from .nu_f import (
    NuFSnapshot,
    NuFTelemetryAccumulator,
    NuFWindow,
    ensure_nu_f_telemetry,
    record_nu_f_window,
)

# Optional buffered event sink
from .unified_telemetry_system import (
    FailureTelemetryEvent,
    PerformanceTelemetryEvent,
    StructuralTelemetryEvent,
    TelemetryConfiguration,
    TNFRUnifiedTelemetrySystem,
    emit_failure_telemetry,
    emit_performance_telemetry,
    emit_structural_telemetry,
    flush_unified_telemetry,
    get_unified_telemetry_stats,
    get_unified_telemetry_system,
)
from .verbosity import (
    TELEMETRY_VERBOSITY_DEFAULT,
    TELEMETRY_VERBOSITY_LEVELS,
    TelemetryVerbosity,
)

__all__ = [
    "TNFRUnifiedTelemetrySystem",
    "TelemetryConfiguration",
    "StructuralTelemetryEvent",
    "PerformanceTelemetryEvent",
    "FailureTelemetryEvent",
    "get_unified_telemetry_system",
    "emit_structural_telemetry",
    "emit_performance_telemetry",
    "emit_failure_telemetry",
    "flush_unified_telemetry",
    "get_unified_telemetry_stats",
    "CacheMetricsSnapshot",
    "CacheTelemetryPublisher",
    "ensure_cache_metrics_publisher",
    "publish_graph_cache_metrics",
    "NuFWindow",
    "NuFSnapshot",
    "NuFTelemetryAccumulator",
    "ensure_nu_f_telemetry",
    "record_nu_f_window",
    "TelemetryVerbosity",
    "TELEMETRY_VERBOSITY_DEFAULT",
    "TELEMETRY_VERBOSITY_LEVELS",
]
