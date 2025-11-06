"""Input validation for TNFR structural operators.

This module provides security-focused input validation for structural operator
parameters to prevent injection attacks and ensure canonical TNFR invariants
are preserved. All validation functions respect TNFR structural semantics:

- EPI (Primary Information Structure) bounds
- νf (structural frequency) Hz_str units and ranges
- θ (phase) normalization to [-π, π]
- ΔNFR (reorganization operator) magnitude constraints
- Type safety for TNFRGraph, NodeId, and Glyph enumerations
"""

from __future__ import annotations

import math
import re
import warnings
from typing import Any, Mapping

import networkx as nx

from ..constants import DEFAULTS
from ..types import Glyph, TNFRGraph

# Check HzStr availability once at module level
_HZ_STR_AVAILABLE = False
try:
    from ..operators.structural_units import HzStr

    _HZ_STR_AVAILABLE = True
except ImportError:
    HzStr = None  # type: ignore[assignment, misc]

__all__ = [
    "ValidationError",
    "validate_epi_value",
    "validate_vf_value",
    "validate_theta_value",
    "validate_dnfr_value",
    "validate_node_id",
    "validate_glyph",
    "validate_tnfr_graph",
    "validate_glyph_factors",
    "validate_operator_parameters",
]


class ValidationError(ValueError):
    """Raised when input validation fails for structural operators."""

    __slots__ = ("parameter", "value", "constraint")

    def __init__(
        self,
        message: str,
        *,
        parameter: str | None = None,
        value: Any = None,
        constraint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.parameter = parameter
        self.value = value
        self.constraint = constraint


def _get_bound(config: Mapping[str, Any] | None, key: str, default: float) -> float:
    """Retrieve a numeric bound from configuration or DEFAULTS.

    Parameters
    ----------
    config : Mapping[str, Any] | None
        Graph configuration mapping (typically from G.graph).
    key : str
        Configuration key (e.g., "EPI_MIN", "VF_MAX").
    default : float
        Fallback value when key is not in config or DEFAULTS.

    Returns
    -------
    float
        The configured bound or default value.

    Notes
    -----
    DEFAULTS is a mapping defined in tnfr.constants containing canonical
    TNFR parameter bounds and configuration values.
    """
    if config is not None and key in config:
        try:
            return float(config[key])
        except (TypeError, ValueError):
            pass
    return float(DEFAULTS.get(key, default))


def validate_epi_value(
    value: Any,
    *,
    config: Mapping[str, Any] | None = None,
    allow_complex: bool = True,
) -> float | complex:
    """Validate EPI (Primary Information Structure) value.

    EPI represents the coherent form of a node and must remain within
    canonical bounds to preserve structural stability.

    Parameters
    ----------
    value : Any
        EPI value to validate. Must be numeric (float or complex if allowed).
    config : Mapping[str, Any] | None
        Graph configuration containing EPI_MIN and EPI_MAX bounds.
    allow_complex : bool
        Whether to allow complex EPI values (default: True).

    Returns
    -------
    float or complex
        Validated EPI value.

    Raises
    ------
    ValidationError
        If value is not numeric, contains special floats (nan, inf),
        or violates bounds.

    Examples
    --------
    >>> validate_epi_value(0.5)
    0.5
    >>> validate_epi_value(1.5, config={"EPI_MAX": 1.0})
    Traceback (most recent call last):
        ...
    ValidationError: EPI value 1.5 exceeds maximum bound 1.0
    """
    if not isinstance(value, (int, float, complex)):
        raise ValidationError(
            f"EPI must be numeric, got {type(value).__name__}",
            parameter="EPI",
            value=value,
            constraint="numeric type",
        )

    if isinstance(value, complex):
        if not allow_complex:
            raise ValidationError(
                "EPI must be real-valued in this context",
                parameter="EPI",
                value=value,
                constraint="real-valued",
            )
        # Check both real and imaginary parts for special values
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValidationError(
                f"EPI cannot contain nan or inf: {value}",
                parameter="EPI",
                value=value,
                constraint="finite",
            )
        magnitude = abs(value)
    else:
        if not math.isfinite(float(value)):
            raise ValidationError(
                f"EPI cannot be nan or inf: {value}",
                parameter="EPI",
                value=value,
                constraint="finite",
            )
        magnitude = abs(float(value))

    epi_min = _get_bound(config, "EPI_MIN", 0.0)
    epi_max = _get_bound(config, "EPI_MAX", 1.0)

    if magnitude < epi_min:
        raise ValidationError(
            f"EPI magnitude {magnitude} below minimum bound {epi_min}",
            parameter="EPI",
            value=value,
            constraint=f"magnitude >= {epi_min}",
        )

    if magnitude > epi_max:
        raise ValidationError(
            f"EPI magnitude {magnitude} exceeds maximum bound {epi_max}",
            parameter="EPI",
            value=value,
            constraint=f"magnitude <= {epi_max}",
        )

    return value


def validate_vf_value(
    value: Any,
    *,
    config: Mapping[str, Any] | None = None,
    enforce_hz_str: bool = False,
) -> float:
    """Validate νf (structural frequency) value in Hz_str units.

    Structural frequency represents the reorganization rate and must be
    positive and within canonical bounds. Optionally enforces Hz_str type.

    Parameters
    ----------
    value : Any
        νf value to validate. Can be numeric or HzStr instance.
    config : Mapping[str, Any] | None
        Graph configuration containing VF_MIN and VF_MAX bounds.
    enforce_hz_str : bool, default False
        If True, warns when value is not a HzStr instance (encourages
        canonical unit usage without breaking existing code).

    Returns
    -------
    float
        Validated νf value.

    Raises
    ------
    ValidationError
        If value is not numeric, not finite, negative, or violates bounds.

    Examples
    --------
    >>> validate_vf_value(1.0)
    1.0
    >>> validate_vf_value(-0.5)
    Traceback (most recent call last):
        ...
    ValidationError: νf must be non-negative, got -0.5

    Notes
    -----
    When enforce_hz_str is True and value is not a HzStr instance, logs a
    warning to encourage canonical unit usage but still accepts the value.
    """
    # Accept HzStr instances (canonical form)
    if _HZ_STR_AVAILABLE and HzStr is not None and isinstance(value, HzStr):
        value = float(value)  # Extract numeric value
    elif enforce_hz_str and _HZ_STR_AVAILABLE:
        # Log warning but don't raise (soft enforcement for migration)
        warnings.warn(
            f"νf should use Hz_str units for TNFR canonical compliance. "
            f"Got {type(value).__name__} instead.",
            UserWarning,
            stacklevel=2,
        )

    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"νf must be numeric, got {type(value).__name__}",
            parameter="vf",
            value=value,
            constraint="numeric type",
        )

    value = float(value)

    if not math.isfinite(value):
        raise ValidationError(
            f"νf cannot be nan or inf: {value}",
            parameter="vf",
            value=value,
            constraint="finite",
        )

    if value < 0:
        raise ValidationError(
            f"νf must be non-negative, got {value}",
            parameter="vf",
            value=value,
            constraint=">= 0",
        )

    vf_min = _get_bound(config, "VF_MIN", 0.0)
    vf_max = _get_bound(config, "VF_MAX", 1.0)  # Match DEFAULTS["VF_MAX"]

    if value < vf_min:
        raise ValidationError(
            f"νf value {value} below minimum bound {vf_min}",
            parameter="vf",
            value=value,
            constraint=f">= {vf_min}",
        )

    if value > vf_max:
        raise ValidationError(
            f"νf value {value} exceeds maximum bound {vf_max}",
            parameter="vf",
            value=value,
            constraint=f"<= {vf_max}",
        )

    return value


def validate_theta_value(
    value: Any,
    *,
    normalize: bool = True,
) -> float:
    """Validate θ (phase) value.

    Phase represents synchrony with the network and is normalized to [-π, π].

    Parameters
    ----------
    value : Any
        θ value to validate. Must be a real number.
    normalize : bool
        Whether to normalize phase to [-π, π] (default: True).

    Returns
    -------
    float
        Validated (and possibly normalized) θ value.

    Raises
    ------
    ValidationError
        If value is not numeric or not finite.

    Examples
    --------
    >>> validate_theta_value(math.pi / 2)
    1.5707963267948966
    >>> validate_theta_value(4 * math.pi, normalize=True)
    0.0
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"θ must be numeric, got {type(value).__name__}",
            parameter="theta",
            value=value,
            constraint="numeric type",
        )

    value = float(value)

    if not math.isfinite(value):
        raise ValidationError(
            f"θ cannot be nan or inf: {value}",
            parameter="theta",
            value=value,
            constraint="finite",
        )

    if normalize:
        # Normalize to [-π, π]
        value = (value + math.pi) % (2 * math.pi) - math.pi

    return value


def validate_dnfr_value(
    value: Any,
    *,
    config: Mapping[str, Any] | None = None,
) -> float:
    """Validate ΔNFR (reorganization operator) magnitude.

    ΔNFR represents the internal reorganization differential and should be
    bounded to prevent excessive structural changes.

    Parameters
    ----------
    value : Any
        ΔNFR value to validate. Must be a real number.
    config : Mapping[str, Any] | None
        Graph configuration containing DNFR_MAX bound.

    Returns
    -------
    float
        Validated ΔNFR value.

    Raises
    ------
    ValidationError
        If value is not numeric, not finite, or exceeds bounds.

    Examples
    --------
    >>> validate_dnfr_value(0.1)
    0.1
    >>> validate_dnfr_value(2.0, config={"DNFR_MAX": 1.0})
    Traceback (most recent call last):
        ...
    ValidationError: |ΔNFR| value 2.0 exceeds maximum bound 1.0
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"ΔNFR must be numeric, got {type(value).__name__}",
            parameter="dnfr",
            value=value,
            constraint="numeric type",
        )

    value = float(value)

    if not math.isfinite(value):
        raise ValidationError(
            f"ΔNFR cannot be nan or inf: {value}",
            parameter="dnfr",
            value=value,
            constraint="finite",
        )

    dnfr_max = _get_bound(config, "DNFR_MAX", 1.0)

    if abs(value) > dnfr_max:
        raise ValidationError(
            f"|ΔNFR| value {abs(value)} exceeds maximum bound {dnfr_max}",
            parameter="dnfr",
            value=value,
            constraint=f"|ΔNFR| <= {dnfr_max}",
        )

    return value


def validate_node_id(value: Any) -> Any:
    """Validate NodeId value.

    NodeId must be hashable and not contain special characters that could
    enable injection attacks.

    Parameters
    ----------
    value : Any
        NodeId to validate.

    Returns
    -------
    Any
        Validated NodeId (unchanged if valid).

    Raises
    ------
    ValidationError
        If value is not hashable or contains suspicious patterns.

    Examples
    --------
    >>> validate_node_id("node_1")
    'node_1'
    >>> validate_node_id(42)
    42
    >>> validate_node_id(['list'])
    Traceback (most recent call last):
        ...
    ValidationError: NodeId must be hashable
    """
    try:
        hash(value)
    except TypeError:
        raise ValidationError(
            "NodeId must be hashable",
            parameter="node",
            value=value,
            constraint="hashable",
        )

    # For string node IDs, check for suspicious patterns
    if isinstance(value, str):
        # Disallow control characters and common injection patterns
        if any(ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValidationError(
                "NodeId cannot contain control characters",
                parameter="node",
                value=value,
                constraint="printable",
            )

        # Check for common injection patterns (basic protection)
        suspicious_patterns = [
            r"<script",  # XSS attempts
            r"javascript:",  # JavaScript protocol
            r"on\w+\s*=",  # Event handlers
            r"\$\{",  # Template injection
            r"`",  # Backticks for command injection
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise ValidationError(
                    f"NodeId contains suspicious pattern: {pattern}",
                    parameter="node",
                    value=value,
                    constraint="safe string",
                )

    return value


def validate_glyph(value: Any) -> Glyph:
    """Validate Glyph enumeration value or structural operator name.

    Accepts both Glyph codes (e.g., "AL", "IL") and structural operator names
    (e.g., "emission", "coherence") following TNFR canonical grammar.

    Parameters
    ----------
    value : Any
        Value to validate as a Glyph. Can be:
        - A Glyph enum instance (e.g., Glyph.AL)
        - A glyph code string (e.g., "AL", "IL")
        - A structural operator name (e.g., "emission", "coherence")

    Returns
    -------
    Glyph
        Validated Glyph enumeration.

    Raises
    ------
    ValidationError
        If value is not a valid Glyph or structural operator name.

    Examples
    --------
    >>> validate_glyph(Glyph.AL)
    <Glyph.AL: 'AL'>
    >>> validate_glyph("AL")
    <Glyph.AL: 'AL'>
    >>> validate_glyph("emission")
    <Glyph.AL: 'AL'>
    >>> validate_glyph("INVALID")
    Traceback (most recent call last):
        ...
    ValidationError: Invalid glyph value: 'INVALID'
    """
    if isinstance(value, Glyph):
        return value

    if isinstance(value, str):
        # Try direct Glyph code first
        try:
            return Glyph(value)
        except ValueError:
            pass

        # Try structural operator name mapping
        try:
            from ..operators.grammar import function_name_to_glyph

            glyph = function_name_to_glyph(value)
            if glyph is not None:
                return glyph
        except Exception:
            # If grammar module import fails, continue to error
            pass

    raise ValidationError(
        f"Invalid glyph value: {value!r}",
        parameter="glyph",
        value=value,
        constraint="valid Glyph enumeration or structural operator name",
    )


def validate_tnfr_graph(value: Any) -> TNFRGraph:
    """Validate TNFRGraph instance.

    Parameters
    ----------
    value : Any
        Value to validate as a TNFRGraph.

    Returns
    -------
    TNFRGraph
        Validated TNFRGraph instance.

    Raises
    ------
    ValidationError
        If value is not a valid TNFRGraph.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.DiGraph()
    >>> validate_tnfr_graph(G)
    <networkx.classes.digraph.DiGraph object at ...>
    >>> validate_tnfr_graph("not a graph")
    Traceback (most recent call last):
        ...
    ValidationError: Expected TNFRGraph, got str
    """
    if not isinstance(value, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise ValidationError(
            f"Expected TNFRGraph, got {type(value).__name__}",
            parameter="G",
            value=value,
            constraint="networkx.Graph instance",
        )

    # Ensure required graph attributes exist
    if not hasattr(value, "graph"):
        raise ValidationError(
            "TNFRGraph must have 'graph' attribute",
            parameter="G",
            value=value,
            constraint="graph attribute",
        )

    return value


def validate_glyph_factors(
    factors: Any,
    *,
    required_keys: set[str] | None = None,
) -> dict[str, float]:
    """Validate glyph factors dictionary.

    Glyph factors contain operator-specific coefficients that modulate
    structural transformations.

    Parameters
    ----------
    factors : Any
        Glyph factors to validate.
    required_keys : set[str] | None
        Required factor keys. If None, no specific keys are required.

    Returns
    -------
    dict[str, float]
        Validated glyph factors.

    Raises
    ------
    ValidationError
        If factors is not a mapping or contains invalid values.

    Examples
    --------
    >>> validate_glyph_factors({"AL_boost": 0.1, "EN_mix": 0.25})
    {'AL_boost': 0.1, 'EN_mix': 0.25}
    >>> validate_glyph_factors("not a dict")
    Traceback (most recent call last):
        ...
    ValidationError: Glyph factors must be a mapping
    """
    if not isinstance(factors, Mapping):
        raise ValidationError(
            "Glyph factors must be a mapping",
            parameter="glyph_factors",
            value=factors,
            constraint="mapping type",
        )

    validated: dict[str, float] = {}

    for key, value in factors.items():
        if not isinstance(key, str):
            raise ValidationError(
                f"Glyph factor key must be string, got {type(key).__name__}",
                parameter=f"glyph_factors[{key!r}]",
                value=value,
                constraint="string key",
            )

        if not isinstance(value, (int, float)):
            raise ValidationError(
                f"Glyph factor value must be numeric, got {type(value).__name__}",
                parameter=f"glyph_factors[{key!r}]",
                value=value,
                constraint="numeric value",
            )

        if not math.isfinite(float(value)):
            raise ValidationError(
                f"Glyph factor value cannot be nan or inf: {value}",
                parameter=f"glyph_factors[{key!r}]",
                value=value,
                constraint="finite",
            )

        validated[key] = float(value)

    if required_keys is not None:
        missing = required_keys - set(validated.keys())
        if missing:
            raise ValidationError(
                f"Missing required glyph factor keys: {sorted(missing)}",
                parameter="glyph_factors",
                value=factors,
                constraint=f"required keys: {sorted(required_keys)}",
            )

    return validated


def validate_operator_parameters(
    parameters: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate structural operator parameters.

    This function validates common operator parameters including EPI, νf, θ,
    and ΔNFR values to ensure they respect canonical bounds.

    Parameters
    ----------
    parameters : Mapping[str, Any]
        Operator parameters to validate.
    config : Mapping[str, Any] | None
        Graph configuration for bound checking.

    Returns
    -------
    dict[str, Any]
        Validated parameters.

    Raises
    ------
    ValidationError
        If any parameter fails validation.

    Examples
    --------
    >>> validate_operator_parameters({"epi": 0.5, "vf": 1.0, "theta": 0.0})
    {'epi': 0.5, 'vf': 1.0, 'theta': 0.0}
    """
    validated: dict[str, Any] = {}

    for key, value in parameters.items():
        if key in ("epi", "EPI"):
            validated[key] = validate_epi_value(value, config=config)
        elif key in ("vf", "VF", "nu_f"):
            validated[key] = validate_vf_value(value, config=config)
        elif key in ("theta", "THETA", "phase"):
            validated[key] = validate_theta_value(value)
        elif key in ("dnfr", "DNFR", "delta_nfr"):
            validated[key] = validate_dnfr_value(value, config=config)
        elif key == "node":
            validated[key] = validate_node_id(value)
        elif key == "glyph":
            validated[key] = validate_glyph(value)
        elif key == "G" or key == "graph":
            validated[key] = validate_tnfr_graph(value)
        elif key == "glyph_factors":
            validated[key] = validate_glyph_factors(value)
        else:
            # Pass through other parameters unchanged
            validated[key] = value

    return validated
