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
    EPSILON_MIN_EMISSION,
    EPI_LATENT_MAX,
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
    TNFRConfig,
    TNFRConfigError,
    VF_KEY,
    VF_PRIMARY,
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


# Legacy function wrappers that use TNFRConfig internally
def inject_defaults(G, defaults=None, override=False):
    """Inject defaults into graph (backward compatible wrapper).

    Uses TNFRConfig internally for validation.
    """
    config = TNFRConfig(defaults=defaults or DEFAULTS, validate_invariants=True)
    config.inject_defaults(G, defaults=defaults or DEFAULTS, override=override)


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
    import copy
    from ..immutable import _is_immutable
    from ..types import TNFRConfigValue
    from typing import cast

    for key, value in overrides.items():
        if key not in DEFAULTS:
            raise KeyError(f"Unknown parameter: '{key}'")
        G.graph[key] = (
            value if _is_immutable(value) else cast(TNFRConfigValue, copy.deepcopy(value))
        )


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
        Configuration value.

    Raises
    ------
    KeyError
        If key not found in graph or DEFAULTS.
    """
    if key in G.graph:
        return G.graph[key]
    if key not in DEFAULTS:
        raise KeyError(f"Unknown parameter: '{key}'")
    return DEFAULTS[key]


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
    return None if val is None else cast_fn(val)


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
)
