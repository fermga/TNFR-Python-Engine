"""Structural units and types for TNFR canonical measurements.

This module provides type-safe wrappers for TNFR structural units,
ensuring canonical unit enforcement throughout the engine.

Key structural units:
- Hz_str: Structural hertz (reorganization rate, νf)
- rad_str: Structural radians (phase, θ)
"""

from __future__ import annotations

from typing import Any, Union

__all__ = [
    "HzStr",
    "StructuralFrequency",
    "ensure_hz_str",
    "hz_to_hz_str",
]

# Default tolerance for structural frequency validation
MIN_STRUCTURAL_FREQUENCY = 0.0


class HzStr:
    """Structural frequency in Hz_str (structural hertz) units.

    Hz_str represents the rate of structural reorganization, not physical
    frequency. It measures how rapidly a node's Primary Information Structure
    (EPI) evolves according to the nodal equation:

        ∂EPI/∂t = νf · ΔNFR(t)

    Where νf is the structural frequency in Hz_str units.

    Attributes
    ----------
    value : float
        Magnitude of the structural frequency
    unit : str
        Always "Hz_str" to maintain unit clarity

    Notes
    -----
    Hz_str is distinct from physical Hz (cycles per second). It represents
    the rate of structural change in TNFR's reorganization phase space.
    Conversion from physical Hz depends on the domain context (biological,
    quantum, social, etc.).
    """

    __slots__ = ("value", "unit")

    def __init__(self, value: float) -> None:
        """Initialize structural frequency.

        Parameters
        ----------
        value : float
            Structural frequency magnitude (must be non-negative)

        Raises
        ------
        ValueError
            If value is negative (structural frequencies are non-negative)
        """
        if value < MIN_STRUCTURAL_FREQUENCY:
            raise ValueError(
                f"Structural frequency must be >= {MIN_STRUCTURAL_FREQUENCY} Hz_str, got {value}"
            )
        self.value = float(value)
        self.unit = "Hz_str"

    def __float__(self) -> float:
        """Convert to float for numerical operations."""
        return self.value

    def __repr__(self) -> str:
        """String representation."""
        return f"HzStr({self.value})"

    def __str__(self) -> str:
        """Human-readable string."""
        return f"{self.value} Hz_str"

    def __eq__(self, other: Any) -> bool:
        """Equality comparison."""
        if isinstance(other, HzStr):
            return abs(self.value - other.value) < 1e-10
        if isinstance(other, (int, float)):
            return abs(self.value - float(other)) < 1e-10
        return NotImplemented

    def __lt__(self, other: Any) -> bool:
        """Less than comparison."""
        if isinstance(other, HzStr):
            return self.value < other.value
        if isinstance(other, (int, float)):
            return self.value < float(other)
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        """Less than or equal comparison."""
        if isinstance(other, HzStr):
            return self.value <= other.value
        if isinstance(other, (int, float)):
            return self.value <= float(other)
        return NotImplemented

    def __gt__(self, other: Any) -> bool:
        """Greater than comparison."""
        if isinstance(other, HzStr):
            return self.value > other.value
        if isinstance(other, (int, float)):
            return self.value > float(other)
        return NotImplemented

    def __ge__(self, other: Any) -> bool:
        """Greater than or equal comparison."""
        if isinstance(other, HzStr):
            return self.value >= other.value
        if isinstance(other, (int, float)):
            return self.value >= float(other)
        return NotImplemented

    def __add__(self, other: Any) -> HzStr:
        """Addition."""
        if isinstance(other, HzStr):
            return HzStr(self.value + other.value)
        if isinstance(other, (int, float)):
            return HzStr(self.value + float(other))
        return NotImplemented

    def __radd__(self, other: Any) -> HzStr:
        """Right addition."""
        return self.__add__(other)

    def __sub__(self, other: Any) -> HzStr:
        """Subtraction."""
        if isinstance(other, HzStr):
            return HzStr(self.value - other.value)
        if isinstance(other, (int, float)):
            return HzStr(self.value - float(other))
        return NotImplemented

    def __rsub__(self, other: Any) -> HzStr:
        """Right subtraction."""
        if isinstance(other, (int, float)):
            return HzStr(float(other) - self.value)
        return NotImplemented

    def __mul__(self, other: Any) -> Union[HzStr, float]:
        """Multiplication.

        When multiplied by another HzStr or dimensionless number, returns HzStr.
        When multiplied by ΔNFR (dimensionless), returns float (∂EPI/∂t rate).
        """
        if isinstance(other, HzStr):
            return HzStr(self.value * other.value)
        if isinstance(other, (int, float)):
            # This is typically νf · ΔNFR in nodal equation
            return self.value * float(other)
        return NotImplemented

    def __rmul__(self, other: Any) -> Union[HzStr, float]:
        """Right multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> HzStr:
        """Division."""
        if isinstance(other, HzStr):
            return HzStr(self.value / other.value)
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError(
                    f"Cannot divide structural frequency {self.value} Hz_str by zero"
                )
            return HzStr(self.value / float(other))
        return NotImplemented

    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        return hash((self.value, self.unit))


# Alias for clarity in type hints
StructuralFrequency = HzStr


def ensure_hz_str(value: Union[float, HzStr]) -> HzStr:
    """Ensure value is in Hz_str units.

    Parameters
    ----------
    value : float or HzStr
        Value to convert to Hz_str

    Returns
    -------
    HzStr
        Value as structural frequency

    Examples
    --------
    >>> ensure_hz_str(1.5)
    HzStr(1.5)
    >>> ensure_hz_str(HzStr(2.0))
    HzStr(2.0)
    """
    if isinstance(value, HzStr):
        return value
    return HzStr(float(value))


def hz_to_hz_str(
    hz_value: float,
    context: str = "default",
) -> HzStr:
    """Convert physical Hz to structural Hz_str with domain-specific scaling.

    Different physical domains have different relationships between physical
    frequency and structural reorganization rate. This function provides
    context-aware conversion factors.

    Parameters
    ----------
    hz_value : float
        Physical frequency in Hz (cycles per second)
    context : str, default "default"
        Domain context for conversion:
        - "default": Direct 1:1 mapping
        - "biological": Biological systems (slower structural reorganization)
        - "quantum": Quantum systems (faster structural reorganization)
        - "social": Social systems (much slower structural reorganization)
        - "neural": Neural systems (moderate structural reorganization)

    Returns
    -------
    HzStr
        Structural frequency in Hz_str units

    Notes
    -----
    Conversion factors are based on typical timescales in each domain:
    - Biological: 0.1 (10 Hz physical → 1 Hz_str)
    - Quantum: 1e12 (1 Hz physical → 1 THz_str)
    - Social: 1e-6 (1 MHz physical → 1 Hz_str)
    - Neural: 1.0 (1 Hz physical → 1 Hz_str, matched to firing rates)

    Examples
    --------
    >>> hz_to_hz_str(10.0, "biological")
    HzStr(1.0)
    >>> hz_to_hz_str(1.0, "quantum")
    HzStr(1000000000000.0)
    """
    CONVERSION_FACTORS = {
        "default": 1.0,
        "biological": 0.1,
        "quantum": 1e12,
        "social": 1e-6,
        "neural": 1.0,
    }

    factor = CONVERSION_FACTORS.get(context, 1.0)
    hz_str_value = hz_value * factor

    return HzStr(hz_str_value)
