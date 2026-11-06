"""Compatibility exports for canonical metric defaults.

Values and dataclass definitions are owned by ``tnfr.config.defaults_metric``.
The legacy public helper imports remain available for import compatibility.
"""

from __future__ import annotations

from ..config.defaults_metric import COHERENCE as COHERENCE
from ..config.defaults_metric import DIAGNOSIS as DIAGNOSIS
from ..config.defaults_metric import GRAMMAR_CANON as GRAMMAR_CANON
from ..config.defaults_metric import METRIC_DEFAULTS as METRIC_DEFAULTS
from ..config.defaults_metric import METRICS as METRICS
from ..config.defaults_metric import SIGMA as SIGMA
from ..config.defaults_metric import TRACE as TRACE
from ..config.defaults_metric import Any as Any
from ..config.defaults_metric import MappingProxyType as MappingProxyType
from ..config.defaults_metric import MetricDefaults as MetricDefaults
from ..config.defaults_metric import asdict as asdict
from ..config.defaults_metric import dataclass as dataclass
from ..config.defaults_metric import field as field

# Preserve the complete historical wildcard surface, including helper imports.
__all__ = [
    "annotations",
    "COHERENCE",
    "DIAGNOSIS",
    "GRAMMAR_CANON",
    "METRIC_DEFAULTS",
    "METRICS",
    "SIGMA",
    "TRACE",
    "Any",
    "MappingProxyType",
    "MetricDefaults",
    "asdict",
    "dataclass",
    "field",
]
