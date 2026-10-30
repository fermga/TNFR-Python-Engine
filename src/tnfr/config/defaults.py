"""Consolidated TNFR configuration defaults.

This module provides all default configuration values organized by subsystem,
following TNFR structural coherence principles.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from ..types import TNFRConfigValue
from .defaults_core import CORE_DEFAULTS, REMESH_DEFAULTS
from .defaults_init import INIT_DEFAULTS
from .defaults_metric import (
    COHERENCE,
    DIAGNOSIS,
    GRAMMAR_CANON,
    METRIC_DEFAULTS,
    METRICS,
    SIGMA,
    TRACE,
)

# Exported sections organized by subsystem
DEFAULT_SECTIONS: Mapping[str, Mapping[str, TNFRConfigValue]] = MappingProxyType(
    {
        "core": CORE_DEFAULTS,
        "init": INIT_DEFAULTS,
        "remesh": REMESH_DEFAULTS,
        "metric": METRIC_DEFAULTS,
    }
)

# Combined defaults with priority: CORE < INIT < REMESH < METRIC
# METRIC_DEFAULTS has highest priority to match previous ChainMap behavior
DEFAULTS: Mapping[str, TNFRConfigValue] = MappingProxyType(
    CORE_DEFAULTS | INIT_DEFAULTS | REMESH_DEFAULTS | METRIC_DEFAULTS
)

__all__ = (
    "DEFAULTS",
    "DEFAULT_SECTIONS",
    "CORE_DEFAULTS",
    "INIT_DEFAULTS",
    "REMESH_DEFAULTS",
    "METRIC_DEFAULTS",
    "SIGMA",
    "TRACE",
    "METRICS",
    "GRAMMAR_CANON",
    "COHERENCE",
    "DIAGNOSIS",
)
