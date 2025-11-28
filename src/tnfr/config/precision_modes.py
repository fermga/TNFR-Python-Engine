"""Global precision, telemetry, and diagnostics configuration.

This module provides non-physical runtime configuration for:
- Numerical precision modes (standard/high/research)
- Telemetry density (low/medium/high)
- Diagnostics level (off/basic/rich)

CRITICAL CONSTRAINTS:
- Must NOT alter TNFR physics or grammar semantics (U1-U6)
- Must NOT change operator contracts or coherence evolution
- Only adjusts numeric details (dtypes, algorithms) and observational richness

These settings tune HOW we measure and report, never WHAT the physics does.
"""

from __future__ import annotations
from typing import Literal

# Type aliases for configuration modes
PrecisionMode = Literal["standard", "high", "research"]
TelemetryDensity = Literal["low", "medium", "high"]
DiagnosticsLevel = Literal["off", "basic", "rich"]

# Global state (module-level, mutable for runtime configuration)
_precision_mode: PrecisionMode = "standard"
_telemetry_density: TelemetryDensity = "low"
_diagnostics_level: DiagnosticsLevel = "off"


def get_precision_mode() -> PrecisionMode:
    """Return current precision mode.
    
    Returns
    -------
    PrecisionMode
        Current mode: "standard", "high", or "research".
    """
    return _precision_mode


def set_precision_mode(mode: PrecisionMode) -> None:
    """Set precision mode for canonical field computations.
    
    Parameters
    ----------
    mode : PrecisionMode
        Target mode: "standard" (default), "high" (extended precision),
        or "research" (maximum precision for publication-grade numerics).
    
    Raises
    ------
    ValueError
        If mode is not one of the valid precision modes.
    
    Notes
    -----
    Physics invariant: Changing precision mode must not alter:
    - Grammar validation (U1-U6)
    - Operator contracts
    - Coherence evolution semantics
    
    Only affects numeric implementation details (dtypes, algorithms).
    """
    global _precision_mode
    valid_modes: tuple[PrecisionMode, ...] = ("standard", "high", "research")
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid precision mode '{mode}'. Must be one of {valid_modes}"
        )
    _precision_mode = mode


def get_telemetry_density() -> TelemetryDensity:
    """Return current telemetry density.
    
    Returns
    -------
    TelemetryDensity
        Current density: "low", "medium", or "high".
    """
    return _telemetry_density


def set_telemetry_density(density: TelemetryDensity) -> None:
    """Set telemetry density for field snapshots and time-series.
    
    Parameters
    ----------
    density : TelemetryDensity
        Target density: "low" (default, minimal overhead), "medium"
        (standard research), or "high" (dense spatiotemporal sampling).
    
    Raises
    ------
    ValueError
        If density is not one of the valid telemetry densities.
    
    Notes
    -----
    Physics invariant: Telemetry is purely observational.
    Changes density of measurement, never the dynamics being measured.
    """
    global _telemetry_density
    valid_densities: tuple[TelemetryDensity, ...] = ("low", "medium", "high")
    if density not in valid_densities:
        raise ValueError(
            f"Invalid telemetry density '{density}'. "
            f"Must be one of {valid_densities}"
        )
    _telemetry_density = density


def get_diagnostics_level() -> DiagnosticsLevel:
    """Return current diagnostics level.
    
    Returns
    -------
    DiagnosticsLevel
        Current level: "off", "basic", or "rich".
    """
    return _diagnostics_level


def set_diagnostics_level(level: DiagnosticsLevel) -> None:
    """Set diagnostics level for stability and safety checks.
    
    Parameters
    ----------
    level : DiagnosticsLevel
        Target level: "off" (default, production), "basic" (light validation),
        or "rich" (comprehensive research-grade diagnostics).
    
    Raises
    ------
    ValueError
        If level is not one of the valid diagnostics levels.
    
    Notes
    -----
    Physics invariant: Diagnostics are read-only checks.
    Increasing level adds more safety validation and logging,
    but never changes computational results.
    """
    global _diagnostics_level
    valid_levels: tuple[DiagnosticsLevel, ...] = ("off", "basic", "rich")
    if level not in valid_levels:
        raise ValueError(
            f"Invalid diagnostics level '{level}'. "
            f"Must be one of {valid_levels}"
        )
    _diagnostics_level = level


__all__ = [
    "PrecisionMode",
    "TelemetryDensity",
    "DiagnosticsLevel",
    "get_precision_mode",
    "set_precision_mode",
    "get_telemetry_density",
    "set_telemetry_density",
    "get_diagnostics_level",
    "set_diagnostics_level",
]
