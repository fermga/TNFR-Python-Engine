"""Compatibility exports for canonical metric defaults.

Values and dataclass definitions are owned by ``tnfr.config.defaults_metric``.
The legacy public helper imports remain available for import compatibility.
"""

from __future__ import annotations

from ..config.defaults_metric import (
    COHERENCE,
    DIAGNOSIS,
    GRAMMAR_CANON,
    METRIC_DEFAULTS,
    METRICS,
    SIGMA,
    TRACE,
    Any,
    MappingProxyType,
    MetricDefaults,
    asdict,
    dataclass,
    field,
)
