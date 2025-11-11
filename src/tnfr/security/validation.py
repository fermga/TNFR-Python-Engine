"""Input validation utilities for TNFR structural data.

This module provides validation functions for TNFR-specific data types
to ensure structural coherence when persisting or querying data.

TNFR Structural Invariants
---------------------------
These validators enforce TNFR canonical invariants:
1. Structural frequency (νf) in Hz_str units
2. Phase (φ) in valid range [0, 2π]
3. Coherence C(t) as non-negative value
4. Sense index Si validation

Example
-------
>>> validate_structural_frequency(0.5)  # Valid
0.5
>>> validate_phase_value(3.14)  # Valid
3.14
>>> validate_structural_frequency(-1.0)  # doctest: +SKIP
Traceback (most recent call last):
    ...
ValueError: Structural frequency must be non-negative
"""

from __future__ import annotations

import math
from typing import Any


def validate_structural_frequency(nu_f: float) -> float:
    """Validate structural frequency (νf) value.

    Structural frequency must be non-negative and expressed in Hz_str
    (structural hertz), representing the nodal reorganization rate.

    Parameters
    ----------
    nu_f : float
        Structural frequency value to validate

    Returns
    -------
    float
        The validated frequency value

    Raises
    ------
    ValueError
        If the frequency is negative, NaN, or infinite

    Example
    -------
    >>> validate_structural_frequency(0.5)
    0.5
    >>> validate_structural_frequency(0.0)  # Silence operator
    0.0
    >>> validate_structural_frequency(-0.1)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: Structural frequency must be non-negative, got -0.1
    """
    if not isinstance(nu_f, (int, float)):
        raise ValueError(f"Structural frequency must be numeric, got {type(nu_f).__name__}")

    if math.isnan(nu_f):
        raise ValueError("Structural frequency cannot be NaN")

    if math.isinf(nu_f):
        raise ValueError("Structural frequency cannot be infinite")

    if nu_f < 0:
        raise ValueError(f"Structural frequency must be non-negative, got {nu_f}")

    return float(nu_f)


def validate_phase_value(phase: float, *, allow_wrap: bool = True) -> float:
    """Validate phase (φ) value.

    Phase represents synchronization in the network and should be in the
    range [0, 2π]. If allow_wrap is True, values outside this range are
    automatically wrapped.

    Parameters
    ----------
    phase : float
        Phase value to validate
    allow_wrap : bool, optional
        If True, wrap phase to [0, 2π] range (default: True)

    Returns
    -------
    float
        The validated (and possibly wrapped) phase value

    Raises
    ------
    ValueError
        If phase is NaN, infinite, or outside valid range (when allow_wrap=False)

    Example
    -------
    >>> validate_phase_value(1.57)  # π/2
    1.57
    >>> result = validate_phase_value(7.0)  # Wrapped to [0, 2π]
    >>> 0.0 <= result <= 6.3
    True
    >>> validate_phase_value(7.0, allow_wrap=False)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: Phase must be in range [0, 2π], got 7.0
    """
    if not isinstance(phase, (int, float)):
        raise ValueError(f"Phase must be numeric, got {type(phase).__name__}")

    if math.isnan(phase):
        raise ValueError("Phase cannot be NaN")

    if math.isinf(phase):
        raise ValueError("Phase cannot be infinite")

    two_pi = 2 * math.pi

    if allow_wrap:
        # Wrap phase to [0, 2π] range
        phase = phase % two_pi
    else:
        if not 0 <= phase <= two_pi:
            raise ValueError(f"Phase must be in range [0, 2π], got {phase}")

    return float(phase)


def validate_coherence_value(coherence: float) -> float:
    """Validate coherence C(t) value.

    Coherence represents the total structural stability and must be
    non-negative.

    Parameters
    ----------
    coherence : float
        Coherence value to validate

    Returns
    -------
    float
        The validated coherence value

    Raises
    ------
    ValueError
        If coherence is negative, NaN, or infinite

    Example
    -------
    >>> validate_coherence_value(0.8)
    0.8
    >>> validate_coherence_value(0.0)  # Minimum coherence
    0.0
    >>> validate_coherence_value(-0.1)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: Coherence must be non-negative, got -0.1
    """
    if not isinstance(coherence, (int, float)):
        raise ValueError(f"Coherence must be numeric, got {type(coherence).__name__}")

    if math.isnan(coherence):
        raise ValueError("Coherence cannot be NaN")

    if math.isinf(coherence):
        raise ValueError("Coherence cannot be infinite")

    if coherence < 0:
        raise ValueError(f"Coherence must be non-negative, got {coherence}")

    return float(coherence)


def validate_sense_index(si: float) -> float:
    """Validate sense index (Si) value.

    Sense index represents the capacity to generate stable reorganization.
    Valid range is typically [0, 1] but can exceed 1 in high-coherence networks.

    Parameters
    ----------
    si : float
        Sense index value to validate

    Returns
    -------
    float
        The validated sense index value

    Raises
    ------
    ValueError
        If Si is negative, NaN, or infinite

    Example
    -------
    >>> validate_sense_index(0.7)
    0.7
    >>> validate_sense_index(1.2)  # High coherence
    1.2
    >>> validate_sense_index(-0.1)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: Sense index must be non-negative, got -0.1
    """
    if not isinstance(si, (int, float)):
        raise ValueError(f"Sense index must be numeric, got {type(si).__name__}")

    if math.isnan(si):
        raise ValueError("Sense index cannot be NaN")

    if math.isinf(si):
        raise ValueError("Sense index cannot be infinite")

    if si < 0:
        raise ValueError(f"Sense index must be non-negative, got {si}")

    return float(si)


def validate_nodal_input(data: dict[str, Any]) -> dict[str, Any]:
    """Validate a complete nodal data structure.

    This function validates all common NFR node attributes to ensure
    structural coherence before database persistence.

    Parameters
    ----------
    data : dict
        Dictionary containing nodal attributes

    Returns
    -------
    dict
        The validated data dictionary

    Raises
    ------
    ValueError
        If any attribute fails validation

    Example
    -------
    >>> data = {"nu_f": 0.5, "phase": 1.57, "delta_nfr": 0.1}
    >>> validated = validate_nodal_input(data)
    >>> validated["nu_f"]
    0.5
    """
    validated = {}

    if "nu_f" in data:
        validated["nu_f"] = validate_structural_frequency(data["nu_f"])

    if "phase" in data:
        validated["phase"] = validate_phase_value(data["phase"])

    if "coherence" in data:
        validated["coherence"] = validate_coherence_value(data["coherence"])

    if "si" in data or "sense_index" in data:
        si_key = "si" if "si" in data else "sense_index"
        validated[si_key] = validate_sense_index(data[si_key])

    # Pass through other fields without validation
    # (e.g., node_id, epi arrays, etc.)
    for key, value in data.items():
        if key not in validated:
            validated[key] = value

    return validated


__all__ = (
    "validate_coherence_value",
    "validate_nodal_input",
    "validate_phase_value",
    "validate_sense_index",
    "validate_structural_frequency",
)
