"""TNFR Unified Telemetry System - Consolidated Event Collection and Monitoring.

Provides unified telemetry emission for all TNFR dynamics analysis.
Telemetry captures observable manifestations of nodal equation evolution
without perturbing underlying TNFR coherence.

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

# Legacy telemetry components (for compatibility)
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

# Unified telemetry system (primary interface)
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
