"""Canonical TNFR configuration system.

This package provides the unified configuration system for TNFR, consolidating:
- TNFRConfig class with structural invariant validation
- Secure configuration management (moved from secure_config.py)
- All default configurations organized by subsystem
- TNFR semantic mapping (νf, θ, ΔNFR)

Single import path philosophy:
    from tnfr.config import TNFRConfig, DEFAULTS, get_param

Key Changes (Phase 3):
- Consolidated constants from constants/ package
- Integrated secure_config functionality
- Added TNFR invariant validation
- Explicit structural coherence principles
"""

from __future__ import annotations

from .defaults import (
    COHERENCE,
    CORE_DEFAULTS,
    DEFAULT_SECTIONS,
    DEFAULTS,
    DIAGNOSIS,
    GRAMMAR_CANON,
    INIT_DEFAULTS,
    METRIC_DEFAULTS,
    METRICS,
    REMESH_DEFAULTS,
    SIGMA,
    TRACE,
)
from .feature_flags import context_flags, get_flags
from .init import apply_config, load_config
from .parsing import parse_bool as _parse_bool
from .precision_modes import (
    DiagnosticsLevel,
    PrecisionMode,
    TelemetryDensity,
    get_diagnostics_level,
    get_precision_mode,
    get_telemetry_density,
    set_diagnostics_level,
    set_precision_mode,
    set_telemetry_density,
)
from .thresholds import (
    EPI_LATENT_MAX,
    EPSILON_MIN_EMISSION,
    MIN_NETWORK_DEGREE_COUPLING,
    VF_BASAL_THRESHOLD,
)
from .tnfr_config import (
    ALIASES,
    CANONICAL_STATE_TOKENS,
    D2EPI_PRIMARY,
    D2VF_PRIMARY,
    DNFR_KEY,
    DNFR_PRIMARY,
    EPI_KIND_PRIMARY,
    EPI_PRIMARY,
    SI_PRIMARY,
    STATE_DISSONANT,
    STATE_STABLE,
    STATE_TRANSITION,
    THETA_KEY,
    THETA_PRIMARY,
    VF_KEY,
    VF_PRIMARY,
    TNFRConfig,
    TNFRConfigError,
    dEPI_PRIMARY,
    dSI_PRIMARY,
    dVF_PRIMARY,
    get_aliases,
    normalise_state_token,
)

# Import compatibility utilities from constants (for backward compat)
# These will be re-exported through constants/__init__.py
try:
    from ..utils import ensure_node_offset_map as _ensure_node_offset_map
except ImportError:
    _ensure_node_offset_map = None

ensure_node_offset_map = _ensure_node_offset_map

_GLOBAL_CONFIG = None


def get_config() -> TNFRConfig:
    """Get the global TNFR configuration singleton."""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        _GLOBAL_CONFIG = TNFRConfig(defaults=DEFAULTS)
    return _GLOBAL_CONFIG


# Legacy function wrappers that use TNFRConfig internally
def inject_defaults(G, defaults=None, override=False):
    """Inject defaults into graph (backward compatible wrapper).

    Uses TNFRConfig internally for validation.
    """
    selected = DEFAULTS if defaults is None else defaults
    get_config().inject_defaults(G, defaults=selected, override=override)


def merge_overrides(G, **overrides):
    """Apply specific overrides to graph configuration.

    Parameters
    ----------
    G : GraphLike
        The graph whose configuration should be updated.
    **overrides
        Keyword arguments mapping parameter names to new values.

    Raises
    ------
    KeyError
        If any parameter name is not present in DEFAULTS.
    """
    for key in overrides:
        if key not in DEFAULTS:
            raise KeyError(f"Unknown parameter: '{key}'")
    updates = get_config()._prepare_updates(G.graph, overrides, override=True)
    G.graph.update(updates)


def get_param(G, key: str):
    """Retrieve parameter from graph or defaults.

    Parameters
    ----------
    G : GraphLike
        Graph containing configuration.
    key : str
        Parameter name.

    Returns
    -------
    TNFRConfigValue
        Graph-owned value, or an isolated copy of a mutable default.

    Raises
    ------
    KeyError
        If key not found in graph or DEFAULTS.
    """
    return get_config().get_param_with_fallback(G.graph, key)


def get_graph_param(G, key: str, cast_fn=float):
    """Return parameter from graph applying cast function.

    Parameters
    ----------
    G : GraphLike
        Graph containing configuration.
    key : str
        Parameter name.
    cast_fn : callable, default=float
        Function to cast value (e.g., float, int, bool).

    Returns
    -------
    Any
        Casted parameter value, or None if value is None.
    """
    val = get_param(G, key)
    if val is None:
        return None
    return _parse_bool(val) if cast_fn is bool else cast_fn(val)


__all__ = (
    # Main configuration class
    "TNFRConfig",
    "TNFRConfigError",
    # File-based configuration
    "load_config",
    "apply_config",
    # Feature flags
    "get_flags",
    "context_flags",
    # Precision/telemetry/diagnostics modes
    "PrecisionMode",
    "TelemetryDensity",
    "DiagnosticsLevel",
    "get_precision_mode",
    "set_precision_mode",
    "get_telemetry_density",
    "set_telemetry_density",
    "get_diagnostics_level",
    "set_diagnostics_level",
    # Defaults and sections
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
    # Operator precondition thresholds
    "EPI_LATENT_MAX",
    "VF_BASAL_THRESHOLD",
    "EPSILON_MIN_EMISSION",
    "MIN_NETWORK_DEGREE_COUPLING",
    # TNFR semantic aliases
    "ALIASES",
    "VF_KEY",
    "THETA_KEY",
    "DNFR_KEY",
    "VF_PRIMARY",
    "THETA_PRIMARY",
    "DNFR_PRIMARY",
    "EPI_PRIMARY",
    "EPI_KIND_PRIMARY",
    "SI_PRIMARY",
    "dEPI_PRIMARY",
    "D2EPI_PRIMARY",
    "dVF_PRIMARY",
    "D2VF_PRIMARY",
    "dSI_PRIMARY",
    # State tokens
    "STATE_STABLE",
    "STATE_TRANSITION",
    "STATE_DISSONANT",
    "CANONICAL_STATE_TOKENS",
    # Utility functions
    "get_aliases",
    "normalise_state_token",
    "inject_defaults",
    "merge_overrides",
    "get_param",
    "get_graph_param",
    "ensure_node_offset_map",
    "get_config",
)
