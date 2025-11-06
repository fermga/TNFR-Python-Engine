"""Canonical TNFR nodal equation implementation.

This module provides the explicit, canonical implementation of the fundamental
TNFR nodal equation as specified in the theory:

    ∂EPI/∂t = νf · ΔNFR(t)

Where:
  - EPI: Primary Information Structure (coherent form)
  - νf: Structural frequency in Hz_str (structural hertz)
  - ΔNFR: Nodal gradient (reorganization operator)
  - t: Structural time (not chronological time)

This implementation ensures theoretical fidelity to the TNFR paradigm by:
  1. Making the canonical equation explicit in code
  2. Validating dimensional consistency (Hz_str units)
  3. Providing clear mapping between theory and implementation
  4. Maintaining reproducibility and traceability

TNFR Invariants (from AGENTS.md):
  - EPI as coherent form: changes only via structural operators
  - Structural units: νf expressed in Hz_str (structural hertz)
  - ΔNFR semantics: sign and magnitude modulate reorganization rate
  - Operator closure: composition yields valid TNFR states

References:
  - TNFR.pdf: Canonical nodal equation specification
  - AGENTS.md: Section 3 (Canonical invariants)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from ..types import GraphLike

__all__ = (
    "NodalEquationResult",
    "compute_canonical_nodal_derivative",
    "validate_structural_frequency",
    "validate_nodal_gradient",
)


class NodalEquationResult(NamedTuple):
    """Result of canonical nodal equation evaluation.

    Attributes:
        derivative: ∂EPI/∂t computed from νf · ΔNFR(t)
        nu_f: Structural frequency (Hz_str) used in computation
        delta_nfr: Nodal gradient (ΔNFR) used in computation
        validated: Whether units and bounds were validated
    """

    derivative: float
    nu_f: float
    delta_nfr: float
    validated: bool


def compute_canonical_nodal_derivative(
    nu_f: float,
    delta_nfr: float,
    *,
    validate_units: bool = True,
    graph: GraphLike | None = None,
) -> NodalEquationResult:
    """Compute ∂EPI/∂t using the canonical TNFR nodal equation.

    This is the explicit implementation of the fundamental equation:
        ∂EPI/∂t = νf · ΔNFR(t)

    The function computes the time derivative of the Primary Information
    Structure (EPI) as the product of:
      - νf: structural frequency (reorganization rate in Hz_str)
      - ΔNFR: nodal gradient (reorganization need/operator)

    Args:
        nu_f: Structural frequency in Hz_str (must be non-negative)
        delta_nfr: Nodal gradient (reorganization operator)
        validate_units: If True, validates that inputs are in valid ranges
        graph: Optional graph for context-aware validation

    Returns:
        NodalEquationResult containing the computed derivative and metadata

    Raises:
        ValueError: If validation is enabled and inputs are invalid

    Notes:
        - This function is the canonical reference implementation
        - The result represents the instantaneous rate of EPI evolution
        - Units: [∂EPI/∂t] = Hz_str (structural reorganization rate)
        - The product νf·ΔNFR must preserve TNFR operator closure

    Examples:
        >>> # Basic computation
        >>> result = compute_canonical_nodal_derivative(1.0, 0.5)
        >>> result.derivative
        0.5

        >>> # With explicit validation
        >>> result = compute_canonical_nodal_derivative(
        ...     nu_f=1.2,
        ...     delta_nfr=-0.3,
        ...     validate_units=True
        ... )
        >>> result.validated
        True
    """
    validated = False

    if validate_units:
        nu_f = validate_structural_frequency(nu_f, graph=graph)
        delta_nfr = validate_nodal_gradient(delta_nfr, graph=graph)
        validated = True

    # Canonical TNFR nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
    derivative = float(nu_f) * float(delta_nfr)

    return NodalEquationResult(
        derivative=derivative,
        nu_f=nu_f,
        delta_nfr=delta_nfr,
        validated=validated,
    )


def validate_structural_frequency(
    nu_f: float,
    *,
    graph: GraphLike | None = None,
) -> float:
    """Validate that structural frequency is in valid range.

    Structural frequency (νf) must satisfy TNFR constraints:
      - Non-negative (νf ≥ 0)
      - Expressed in Hz_str (structural hertz)
      - Finite and well-defined

    Args:
        nu_f: Structural frequency to validate
        graph: Optional graph for context-aware bounds checking

    Returns:
        Validated structural frequency value

    Raises:
        ValueError: If nu_f is negative, infinite, or NaN
        TypeError: If nu_f cannot be converted to float

    Notes:
        - νf = 0 is valid and represents structural silence
        - Units must be Hz_str (not classical Hz)
        - For Hz↔Hz_str conversion, use tnfr.units module
    """
    try:
        value = float(nu_f)
    except TypeError as exc:
        # Non-convertible type (e.g., None, object())
        raise TypeError(
            f"Structural frequency must be numeric, got {type(nu_f).__name__}"
        ) from exc
    except ValueError as exc:
        # Invalid string value (e.g., "invalid")
        raise ValueError(
            f"Structural frequency must be a valid number, got {nu_f!r}"
        ) from exc

    # Check for NaN or infinity using math.isfinite
    if not math.isfinite(value):
        raise ValueError(f"Structural frequency must be finite, got νf={value}")

    if value < 0:
        raise ValueError(f"Structural frequency must be non-negative, got νf={value}")

    return value


def validate_nodal_gradient(
    delta_nfr: float,
    *,
    graph: GraphLike | None = None,
) -> float:
    """Validate that nodal gradient is well-defined.

    The nodal gradient (ΔNFR) represents the internal reorganization
    operator and must be:
      - Finite and well-defined
      - Sign indicates reorganization direction
      - Magnitude indicates reorganization intensity

    Args:
        delta_nfr: Nodal gradient to validate
        graph: Optional graph for context-aware validation

    Returns:
        Validated nodal gradient value

    Raises:
        ValueError: If delta_nfr is infinite or NaN
        TypeError: If delta_nfr cannot be converted to float

    Notes:
        - ΔNFR can be positive (expansion) or negative (contraction)
        - ΔNFR = 0 indicates equilibrium (no reorganization)
        - Do NOT reinterpret as classical "error gradient"
        - Semantics: operator over EPI, not optimization target
    """
    try:
        value = float(delta_nfr)
    except TypeError as exc:
        # Non-convertible type (e.g., None, object())
        raise TypeError(
            f"Nodal gradient must be numeric, got {type(delta_nfr).__name__}"
        ) from exc
    except ValueError as exc:
        # Invalid string value (e.g., "invalid")
        raise ValueError(
            f"Nodal gradient must be a valid number, got {delta_nfr!r}"
        ) from exc

    # Check for NaN or infinity using math.isfinite
    if not math.isfinite(value):
        raise ValueError(f"Nodal gradient must be finite, got ΔNFR={value}")

    return value
