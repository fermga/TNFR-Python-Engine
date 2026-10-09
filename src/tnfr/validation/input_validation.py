"""Value-returning input adapters shared by the public validation pipeline.

Scalar EPI and pressure reuse their represented-real admission owners;
frequency and phase adapt configured reports from the unified input validator.
Glyph, identifier and graph-interface helpers retain their local contracts.
These checks do not establish operator admission or a valid nodal trajectory.
"""

from __future__ import annotations

from typing import Any

from .._exact_time import finite_represented_real
from ..types import Glyph, NodeId, TNFRGraph, require_finite_real_scalar_epi
from .unified_validation_system import (
    ValidationError,
    ValidationResult,
    get_unified_validation_system,
)

__all__ = (
    "ValidationError",
    "ValidationResult",
    "validate_glyph",
    "validate_node_id",
    "validate_tnfr_graph",
    "validate_epi_value",
    "validate_vf_value",
    "validate_theta_value",
    "validate_dnfr_value",
    "validate_glyph_factors",
    "validate_operator_parameters",
)


def validate_glyph(value: Any) -> Glyph:
    """Validate and convert a value to a Glyph.

    Parameters
    ----------
    value : Any
        Value to validate as a Glyph

    Returns
    -------
    Glyph
        Validated Glyph instance

    Raises
    ------
    ValidationError
        If value cannot be converted to a valid Glyph
    """
    if isinstance(value, Glyph):
        return value

    try:
        return Glyph(str(value))
    except ValueError as e:
        raise ValidationError(f"Invalid glyph value: {value}") from e


def validate_node_id(value: Any) -> NodeId:
    """Validate a node ID value.

    Parameters
    ----------
    value : Any
        Value to validate as NodeId

    Returns
    -------
    NodeId
        Validated node ID

    Raises
    ------
    ValidationError
        If value is not a valid node ID
    """
    if value is None:
        raise ValidationError("Node ID cannot be None")

    # NodeId can be int, str, or other hashable types
    # Basic validation - ensure it's hashable
    try:
        hash(value)
    except TypeError as e:
        raise ValidationError(
            f"Node ID must be hashable, got {type(value).__name__}"
        ) from e

    return value


def validate_tnfr_graph(graph: Any) -> TNFRGraph:
    """Validate that an object is a valid TNFR graph.

    Parameters
    ----------
    graph : Any
        Object to validate as TNFRGraph

    Returns
    -------
    TNFRGraph
        Validated graph

    Raises
    ------
    ValidationError
        If object is not a valid TNFR graph
    """
    if graph is None:
        raise ValidationError("Graph cannot be None")

    # Check for required graph interface
    required_attrs = ("nodes", "edges", "graph")
    missing = [attr for attr in required_attrs if not hasattr(graph, attr)]

    if missing:
        raise ValidationError(
            f"Object does not have required graph attributes: {missing}"
        )

    return graph


def validate_epi_value(value: Any, field_name: str = "epi") -> float:
    """Validate an EPI (Primary Information Structure) value.

    This scalar input adapter preserves a signed uniform-real EPI embedding.
    Richer form states require their own model admission; they cannot be
    replaced by a magnitude here.

    Parameters
    ----------
    value : Any
        Value to validate
    field_name : str
        Field name for error messages

    Returns
    -------
    float
        Validated EPI value

    Raises
    ------
    ValidationError
        If value is invalid
    """
    try:
        return require_finite_real_scalar_epi(value, field_name)
    except (TypeError, ValueError) as exc:
        raise ValidationError(str(exc)) from exc


def validate_vf_value(value: Any, field_name: str = "vf") -> float:
    """Validate a structural frequency (νf) value.

    Parameters
    ----------
    value : Any
        Value to validate
    field_name : str
        Field name for error messages

    Returns
    -------
    float
        Validated structural frequency value

    Raises
    ------
    ValidationError
        If value is invalid
    """
    validator = get_unified_validation_system()
    result = validator.validate_structural_frequency(value, field_name)

    if not result.is_valid:
        raise ValidationError("; ".join(result.error_messages))

    return result.validated_value


def validate_theta_value(value: Any, field_name: str = "theta") -> float:
    """Validate a phase (θ) value.

    Parameters
    ----------
    value : Any
        Value to validate
    field_name : str
        Field name for error messages

    Returns
    -------
    float
        Validated phase value

    Raises
    ------
    ValidationError
        If value is invalid
    """
    validator = get_unified_validation_system()
    result = validator.validate_phase_value(value, field_name)

    if not result.is_valid:
        raise ValidationError("; ".join(result.error_messages))

    return result.validated_value


def validate_dnfr_value(value: Any, field_name: str = "dnfr") -> float:
    """Validate a finite represented-real ΔNFR pressure value.

    Parameters
    ----------
    value : Any
        Value to validate
    field_name : str
        Field name for error messages

    Returns
    -------
    float
        Validated ΔNFR value

    Raises
    ------
    ValidationError
        If value is invalid
    """
    try:
        return finite_represented_real(value, field_name)[0]
    except (TypeError, ValueError) as exc:
        raise ValidationError(str(exc)) from exc


def validate_glyph_factors(
    factors: Any, *, glyph: Glyph | str | None = None
) -> dict[str, Any]:
    """Validate glyph factors dictionary.

    Parameters
    ----------
    factors : Any
        Value to validate as glyph factors.
    glyph : Glyph or str, optional
        Operator context. When supplied, only that operator's known factors
        are validated; extension-owned keys remain untouched.

    Returns
    -------
    dict
        Validated factors dictionary

    Raises
    ------
    ValidationError
        If factors is invalid
    """
    from ..operators.factor_contracts import GlyphFactorValidationError
    from ..operators.factor_contracts import (
        validate_glyph_factors as validate_factor_mapping,
    )

    try:
        return validate_factor_mapping(factors, glyph=glyph)
    except GlyphFactorValidationError as exc:
        raise ValidationError(str(exc)) from exc


def validate_operator_parameters(**params: Any) -> dict:
    """Validate operator parameters.

    Parameters
    ----------
    **params : Any
        Parameters to validate

    Returns
    -------
    dict
        Validated parameters

    Raises
    ------
    ValidationError
        If any parameter is invalid
    """
    validated = {}

    for key, value in params.items():
        if value is None:
            validated[key] = value
            continue

        # Check for common parameters
        if key in ("epi", "EPI"):
            validated[key] = validate_epi_value(value, key)
        elif key in ("vf", "nu_f", "structural_frequency"):
            validated[key] = validate_vf_value(value, key)
        elif key in ("theta", "phase", "phi"):
            validated[key] = validate_theta_value(value, key)
        elif key in ("dnfr", "DNFR", "delta_nfr"):
            validated[key] = validate_dnfr_value(value, key)
        else:
            # Pass through other parameters
            validated[key] = value

    return validated
