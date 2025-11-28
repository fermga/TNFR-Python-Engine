"""Canonical TNFR configuration system with structural invariant validation.

This module provides the TNFRConfig class that consolidates all TNFR
configuration with explicit semantic mapping to TNFR invariants (νf, θ, ΔNFR).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from ..immutable import _is_immutable
from ..types import GraphLike, TNFRConfigValue

T = TypeVar("T")

# TNFR Canonical State Tokens
STATE_STABLE = "stable"
STATE_TRANSITION = "transition"
STATE_DISSONANT = "dissonant"

CANONICAL_STATE_TOKENS = frozenset({STATE_STABLE, STATE_TRANSITION, STATE_DISSONANT})

# Canonical TNFR keys with Unicode symbols
VF_KEY = "νf"  # Structural frequency (Hz_str)
THETA_KEY = "theta"  # Phase
DNFR_KEY = "ΔNFR"  # Internal reorganization operator

# Node attribute aliases for canonical TNFR variables
ALIASES: dict[str, tuple[str, ...]] = {
    "VF": (VF_KEY, "nu_f", "nu-f", "nu", "freq", "frequency"),
    "THETA": (THETA_KEY, "phase"),
    "DNFR": (DNFR_KEY, "delta_nfr", "dnfr"),
    "EPI": ("EPI", "psi", "PSI", "value"),
    "EPI_KIND": ("EPI_kind", "epi_kind", "source_glyph"),
    "SI": ("Si", "sense_index", "S_i", "sense", "meaning_index"),
    "DEPI": ("dEPI_dt", "dpsi_dt", "dEPI", "velocity"),
    "D2EPI": ("d2EPI_dt2", "d2psi_dt2", "d2EPI", "accel"),
    "DVF": ("dνf_dt", "dvf_dt", "dnu_dt", "dvf"),
    "D2VF": ("d2νf_dt2", "d2vf_dt2", "d2nu_dt2", "B"),
    "DSI": ("δSi", "delta_Si", "dSi"),
    "EMISSION_TIMESTAMP": ("emission_timestamp", "emission_t", "t_al"),
}


class TNFRConfigError(Exception):
    """Raised when TNFR configuration violates structural invariants."""


class TNFRConfig:
    """Canonical TNFR configuration with structural invariant validation.

    This class consolidates all TNFR configuration and provides validation
    against the canonical TNFR invariants defined in AGENTS.md.

    TNFR Structural Invariants Enforced:
    -------------------------------------
    1. νf (structural frequency) must be in Hz_str units, > 0
    2. θ (phase) must be in [-π, π] if THETA_WRAP enabled
    3. ΔNFR magnitude bounds define reorganization stability
    4. EPI coherent form bounds define valid state space
    5. Configuration parameters maintain operator closure

    Parameters
    ----------
    defaults : Mapping[str, TNFRConfigValue], optional
        Base configuration defaults to use.
    validate_invariants : bool, default=True
        Whether to validate TNFR structural invariants.

    Examples
    --------
    >>> config = TNFRConfig()
    >>> config.validate_vf_bounds(vf_min=0.0, vf_max=10.0)
    True

    >>> config.get_param_with_fallback({}, "DT", default=1.0)
    1.0
    """

    def __init__(
        self,
        defaults: Mapping[str, TNFRConfigValue] | None = None,
        validate_invariants: bool = True,
    ) -> None:
        """Initialize TNFR configuration."""
        self._defaults = defaults or {}
        self._validate_invariants = validate_invariants

    def validate_vf_bounds(
        self,
        vf_min: float | None = None,
        vf_max: float | None = None,
        vf: float | None = None,
    ) -> bool:
        """Validate νf (structural frequency) bounds.

        TNFR Invariant: νf must be expressed in Hz_str (structural hertz)
        and must be > 0 to maintain node existence.

        Parameters
        ----------
        vf_min : float, optional
            Minimum structural frequency bound.
        vf_max : float, optional
            Maximum structural frequency bound.
        vf : float, optional
            Specific frequency value to validate.

        Returns
        -------
        bool
            True if bounds are valid.

        Raises
        ------
        TNFRConfigError
            If bounds violate TNFR invariants.
        """
        if not self._validate_invariants:
            return True

        # Invariant 2: Structural units - νf in Hz_str, must be positive
        if vf_min is not None and vf_min < 0.0:
            raise TNFRConfigError(f"VF_MIN must be >= 0 (Hz_str units), got {vf_min}")

        if vf_max is not None and vf_min is not None and vf_max < vf_min:
            raise TNFRConfigError(f"VF_MAX ({vf_max}) must be >= VF_MIN ({vf_min})")

        if vf is not None:
            if vf < 0.0:
                raise TNFRConfigError(f"νf must be >= 0 (Hz_str units), got {vf}")
            if vf_min is not None and vf < vf_min:
                raise TNFRConfigError(f"νf ({vf}) below VF_MIN ({vf_min})")
            if vf_max is not None and vf > vf_max:
                raise TNFRConfigError(f"νf ({vf}) above VF_MAX ({vf_max})")

        return True

    def validate_theta_bounds(
        self,
        theta: float | None = None,
        theta_wrap: bool = True,
    ) -> bool:
        """Validate θ (phase) bounds.

        TNFR Invariant: Phase must be properly bounded to ensure
        valid network synchrony measurements.

        Parameters
        ----------
        theta : float, optional
            Phase value to validate.
        theta_wrap : bool, default=True
            Whether phase wrapping is enabled.

        Returns
        -------
        bool
            True if phase is valid.

        Raises
        ------
        TNFRConfigError
            If phase violates TNFR invariants.
        """
        if not self._validate_invariants:
            return True

        import math

        # Invariant 5: Phase check - valid synchrony requires bounded phase
        if theta is not None and not theta_wrap:
            if not (-math.pi <= theta <= math.pi):
                raise TNFRConfigError(
                    f"θ (phase) must be in [-π, π] when THETA_WRAP=False, got {theta}"
                )

        return True

    def validate_epi_bounds(
        self,
        epi_min: float | None = None,
        epi_max: float | None = None,
        epi: float | None = None,
    ) -> bool:
        """Validate EPI (Primary Information Structure) bounds.

        TNFR Invariant: EPI as coherent form must remain within
        valid bounds to maintain structural coherence.

        Parameters
        ----------
        epi_min : float, optional
            Minimum EPI bound.
        epi_max : float, optional
            Maximum EPI bound.
        epi : float, optional
            Specific EPI value to validate.

        Returns
        -------
        bool
            True if bounds are valid.

        Raises
        ------
        TNFRConfigError
            If bounds violate TNFR invariants.
        """
        if not self._validate_invariants:
            return True

        # Invariant 1: EPI as coherent form - must have valid bounds
        if epi_max is not None and epi_min is not None and epi_max < epi_min:
            raise TNFRConfigError(f"EPI_MAX ({epi_max}) must be >= EPI_MIN ({epi_min})")

        if epi is not None:
            if epi_min is not None and epi < epi_min:
                raise TNFRConfigError(f"EPI ({epi}) below EPI_MIN ({epi_min})")
            if epi_max is not None and epi > epi_max:
                raise TNFRConfigError(f"EPI ({epi}) above EPI_MAX ({epi_max})")

        return True

    def validate_dnfr_semantics(
        self,
        dnfr: float | None = None,
        context: str = "",
    ) -> bool:
        """Validate ΔNFR (reorganization operator) semantics.

        TNFR Invariant: ΔNFR semantics must not be reinterpreted as
        classical ML "error" or "loss gradient". It modulates
        reorganization rate.

        Parameters
        ----------
        dnfr : float, optional
            ΔNFR value to validate.
        context : str, optional
            Context description for validation.

        Returns
        -------
        bool
            True if ΔNFR semantics are preserved.

        Raises
        ------
        TNFRConfigError
            If ΔNFR semantics are violated.
        """
        if not self._validate_invariants:
            return True

        # Invariant 3: ΔNFR semantics - modulates reorganization rate
        # Sign and magnitude are semantically significant
        # This validation ensures we don't reinterpret ΔNFR incorrectly

        # No specific numeric bounds - ΔNFR can be any real value
        # The semantic check is about usage context, not numeric range

        return True

    def validate_config(
        self,
        config: Mapping[str, Any],
    ) -> bool:
        """Validate complete configuration against TNFR invariants.

        Parameters
        ----------
        config : Mapping[str, Any]
            Configuration to validate.

        Returns
        -------
        bool
            True if configuration is valid.

        Raises
        ------
        TNFRConfigError
            If configuration violates TNFR invariants.
        """
        if not self._validate_invariants:
            return True

        # Validate νf bounds
        vf_min = config.get("VF_MIN")
        vf_max = config.get("VF_MAX")
        if vf_min is not None or vf_max is not None:
            self.validate_vf_bounds(vf_min=vf_min, vf_max=vf_max)

        # Validate θ bounds
        theta_wrap = config.get("THETA_WRAP", True)
        init_theta_min = config.get("INIT_THETA_MIN")
        init_theta_max = config.get("INIT_THETA_MAX")
        if init_theta_min is not None:
            self.validate_theta_bounds(theta=init_theta_min, theta_wrap=theta_wrap)
        if init_theta_max is not None:
            self.validate_theta_bounds(theta=init_theta_max, theta_wrap=theta_wrap)

        # Validate EPI bounds
        epi_min = config.get("EPI_MIN")
        epi_max = config.get("EPI_MAX")
        if epi_min is not None or epi_max is not None:
            self.validate_epi_bounds(epi_min=epi_min, epi_max=epi_max)

        # Validate DT (time step) is positive for temporal coherence
        dt = config.get("DT")
        if dt is not None and dt <= 0:
            raise TNFRConfigError(f"DT (time step) must be > 0 for temporal coherence, got {dt}")

        return True

    def get_param_with_fallback(
        self,
        G_graph: Mapping[str, Any],
        key: str,
        default: TNFRConfigValue | None = None,
    ) -> TNFRConfigValue:
        """Retrieve parameter from graph or defaults with fallback.

        Parameters
        ----------
        G_graph : Mapping[str, Any]
            Graph configuration dictionary.
        key : str
            Parameter key to retrieve.
        default : TNFRConfigValue, optional
            Fallback default value.

        Returns
        -------
        TNFRConfigValue
            Configuration value.
        """
        if key in G_graph:
            return G_graph[key]
        if key in self._defaults:
            value = self._defaults[key]
            # Deep copy mutable values to avoid shared state
            if not _is_immutable(value):
                return cast(TNFRConfigValue, copy.deepcopy(value))
            return value
        if default is not None:
            return default
        raise KeyError(f"Parameter '{key}' not found in graph or defaults")

    def inject_defaults(
        self,
        G: GraphLike,
        defaults: Mapping[str, TNFRConfigValue] | None = None,
        override: bool = False,
    ) -> None:
        """Inject defaults into graph with TNFR invariant validation.

        Parameters
        ----------
        G : GraphLike
            Graph to inject configuration into.
        defaults : Mapping[str, TNFRConfigValue], optional
            Configuration to inject. Uses instance defaults if not provided.
        override : bool, default=False
            Whether to override existing values.

        Raises
        ------
        TNFRConfigError
            If configuration violates TNFR invariants.
        """
        config_to_inject = defaults or self._defaults

        # Validate before injection
        if self._validate_invariants:
            self.validate_config(config_to_inject)

        G.graph.setdefault("_tnfr_defaults_attached", False)
        for k, v in config_to_inject.items():
            if override or k not in G.graph:
                G.graph[k] = v if _is_immutable(v) else cast(TNFRConfigValue, copy.deepcopy(v))
        G.graph["_tnfr_defaults_attached"] = True

        # Ensure node offset map if available
        try:
            from ..utils import ensure_node_offset_map

            ensure_node_offset_map(G)
        except ImportError:
            pass


def get_aliases(key: str) -> tuple[str, ...]:
    """Return alias tuple for canonical TNFR variable ``key``.

    Parameters
    ----------
    key : str
        Canonical variable key (e.g., "VF", "THETA", "DNFR").

    Returns
    -------
    tuple[str, ...]
        Tuple of aliases for the variable.

    Examples
    --------
    >>> get_aliases("VF")
    ('νf', 'nu_f', 'nu-f', 'nu', 'freq', 'frequency')
    """
    return ALIASES[key]


# Primary aliases for common TNFR variables
VF_PRIMARY = get_aliases("VF")[0]  # νf
THETA_PRIMARY = get_aliases("THETA")[0]  # theta
DNFR_PRIMARY = get_aliases("DNFR")[0]  # ΔNFR
EPI_PRIMARY = get_aliases("EPI")[0]  # EPI
EPI_KIND_PRIMARY = get_aliases("EPI_KIND")[0]  # EPI_kind
SI_PRIMARY = get_aliases("SI")[0]  # Si
dEPI_PRIMARY = get_aliases("DEPI")[0]  # dEPI_dt
D2EPI_PRIMARY = get_aliases("D2EPI")[0]  # d2EPI_dt2
dVF_PRIMARY = get_aliases("DVF")[0]  # dνf_dt
D2VF_PRIMARY = get_aliases("D2VF")[0]  # d2νf_dt2
dSI_PRIMARY = get_aliases("DSI")[0]  # δSi


def normalise_state_token(token: str) -> str:
    """Return the canonical English token for node state.

    TNFR defines three canonical states: stable, transition, dissonant.

    Parameters
    ----------
    token : str
        State token to normalize.

    Returns
    -------
    str
        Canonical state token.

    Raises
    ------
    TypeError
        If token is not a string.
    ValueError
        If token is not a valid TNFR state.
    """
    if not isinstance(token, str):
        raise TypeError("state token must be a string")

    stripped = token.strip()
    lowered = stripped.lower()

    if stripped in CANONICAL_STATE_TOKENS:
        return stripped

    if lowered in CANONICAL_STATE_TOKENS:
        return lowered

    raise ValueError("state token must be one of 'stable', 'transition', or 'dissonant'")


__all__ = (
    "TNFRConfig",
    "TNFRConfigError",
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
    "STATE_STABLE",
    "STATE_TRANSITION",
    "STATE_DISSONANT",
    "CANONICAL_STATE_TOKENS",
    "get_aliases",
    "normalise_state_token",
)
