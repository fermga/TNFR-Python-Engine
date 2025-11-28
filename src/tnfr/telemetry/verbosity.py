"""Canonical telemetry verbosity presets for TNFR structures.

Each level expresses how much structural context is exported in traces and
metrics:

* ``basic`` preserves lightweight coherence checks for quick health probes.
* ``detailed`` adds phase alignment and coupling diagnostics to map resonance.
* ``debug`` captures the full glyph narrative for deep structural forensics.
"""

from __future__ import annotations

from enum import Enum


class TelemetryVerbosity(str, Enum):
    """Enumerated verbosity tiers shared by trace and metrics pipelines."""

    BASIC = "basic"
    DETAILED = "detailed"
    DEBUG = "debug"


TELEMETRY_VERBOSITY_LEVELS: tuple[str, ...] = tuple(level.value for level in TelemetryVerbosity)
"""Ordered tuple of canonical telemetry verbosity identifiers."""

TELEMETRY_VERBOSITY_DEFAULT: str = TelemetryVerbosity.DEBUG.value
"""Default telemetry verbosity preserving complete structural capture."""

__all__ = [
    "TelemetryVerbosity",
    "TELEMETRY_VERBOSITY_LEVELS",
    "TELEMETRY_VERBOSITY_DEFAULT",
]
