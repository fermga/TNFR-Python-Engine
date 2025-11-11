"""Contextual error handling for TNFR operations.

This module provides enhanced error messages that guide users to solutions
while maintaining TNFR theoretical compliance. All errors include:

1. Clear explanation of the violation
2. Actionable suggestions for resolution
3. Links to relevant documentation
4. Context about the structural operation that failed

Canonical Invariants Preserved
------------------------------
These errors enforce TNFR invariants from AGENTS.md:
- Operator closure and sequence validity
- Phase synchrony requirements for coupling
- Frequency (νf) bounds in Hz_str units
- ΔNFR semantic correctness
- EPI coherence preservation
"""

from __future__ import annotations

from difflib import get_close_matches
from typing import Optional, List, Dict, Any

__all__ = [
    "TNFRUserError",
    "OperatorSequenceError",
    "NetworkConfigError",
    "PhaseError",
    "CoherenceError",
    "FrequencyError",
]


class TNFRUserError(Exception):
    """Base class for user-facing TNFR errors with helpful context.

    All TNFR errors inherit from this class and provide:
    - Human-readable error messages
    - Actionable suggestions
    - Documentation links
    - Structural context

    Parameters
    ----------
    message : str
        Primary error message describing what went wrong.
    suggestion : str, optional
        Specific suggestion for how to fix the issue.
    docs_url : str, optional
        URL to relevant documentation section.
    context : dict, optional
        Additional context about the failed operation (node IDs, values, etc).

    Examples
    --------
    >>> raise TNFRUserError(
    ...     "Invalid structural frequency",
    ...     suggestion="νf must be positive in Hz_str units",
    ...     docs_url="https://tnfr.readthedocs.io/api/core.html#frequency"
    ... )
    """

    def __init__(
        self,
        message: str,
        suggestion: Optional[str] = None,
        docs_url: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.suggestion = suggestion
        self.docs_url = docs_url
        self.context = context or {}

        # Build comprehensive error message
        full_message = f"\n{'='*70}\n"
        full_message += f"TNFR Error: {message}\n"
        full_message += f"{'='*70}\n"

        if suggestion:
            full_message += f"\n💡 Suggestion: {suggestion}\n"

        if context:
            full_message += "\n📊 Context:\n"
            for key, value in context.items():
                full_message += f"   • {key}: {value}\n"

        if docs_url:
            full_message += f"\n📚 Documentation: {docs_url}\n"

        full_message += f"{'='*70}\n"

        super().__init__(full_message)


class OperatorSequenceError(TNFRUserError):
    """Error raised when operator sequence violates TNFR grammar.

    TNFR operators must be applied in valid sequences that respect
    structural coherence. This error provides:
    - The invalid sequence attempted
    - Which operator violated the grammar
    - Valid next operators
    - Fuzzy matching for typos

    Enforces Invariant #4: Operator closure from AGENTS.md

    Parameters
    ----------
    invalid_operator : str
        The operator that violated the grammar.
    sequence_so_far : list of str
        Operators successfully applied before the error.
    valid_next : list of str, optional
        Valid operators that can follow the current sequence.

    Examples
    --------
    >>> raise OperatorSequenceError(
    ...     "emision",
    ...     ["reception", "coherence"],
    ...     ["emission", "recursivity"]
    ... )
    """

    # Valid TNFR operators (13 canonical operators)
    VALID_OPERATORS = {
        "emission",
        "reception",
        "coherence",
        "dissonance",
        "coupling",
        "resonance",
        "silence",
        "expansion",
        "contraction",
        "self_organization",
        "mutation",
        "transition",
        "recursivity",
    }

    # Operator aliases for user convenience
    OPERATOR_ALIASES = {
        "emit": "emission",
        "receive": "reception",
        "cohere": "coherence",
        "couple": "coupling",
        "resonate": "resonance",
        "silent": "silence",
        "expand": "expansion",
        "contract": "contraction",
        "self_organize": "self_organization",
        "mutate": "mutation",
        "recurse": "recursivity",
    }

    def __init__(
        self,
        invalid_operator: str,
        sequence_so_far: Optional[List[str]] = None,
        valid_next: Optional[List[str]] = None,
    ):
        sequence_so_far = sequence_so_far or []

        # Try fuzzy matching for typos
        all_valid = list(self.VALID_OPERATORS) + list(self.OPERATOR_ALIASES.keys())
        matches = get_close_matches(invalid_operator, all_valid, n=3, cutoff=0.6)

        suggestion_parts = []
        if matches:
            suggestion_parts.append(f"Did you mean one of: {', '.join(matches)}?")

        if valid_next:
            suggestion_parts.append(f"Valid next operators: {', '.join(valid_next)}")
        else:
            suggestion_parts.append(
                f"Use one of the 13 canonical operators: "
                f"{', '.join(sorted(self.VALID_OPERATORS))}"
            )

        suggestion = " ".join(suggestion_parts) if suggestion_parts else None

        context = {
            "invalid_operator": invalid_operator,
            "sequence_so_far": (" → ".join(sequence_so_far) if sequence_so_far else "empty"),
            "operator_count": len(sequence_so_far),
        }

        super().__init__(
            message=f"Invalid operator sequence: '{invalid_operator}' cannot be applied",
            suggestion=suggestion,
            docs_url="https://github.com/fermga/Teoria-de-la-naturaleza-fractal-resonante-TNFR-/blob/main/docs/source/api/operators.md",
            context=context,
        )


class NetworkConfigError(TNFRUserError):
    """Error raised when network configuration violates TNFR constraints.

    This error validates configuration parameters and provides valid ranges
    with physical/structural meaning.

    Enforces multiple invariants:
    - Invariant #2: Structural units (νf in Hz_str)
    - Invariant #5: Phase check requirements
    - Invariant #6: Node birth/collapse conditions

    Parameters
    ----------
    parameter : str
        The configuration parameter that is invalid.
    value : any
        The invalid value provided.
    valid_range : tuple, optional
        Valid range for the parameter (min, max).
    reason : str, optional
        Structural reason for the constraint.

    Examples
    --------
    >>> raise NetworkConfigError(
    ...     "vf",
    ...     -0.5,
    ...     (0.01, 100.0),
    ...     "Structural frequency must be positive (Hz_str units)"
    ... )
    """

    # Valid parameter ranges with structural meaning
    PARAMETER_CONSTRAINTS = {
        "vf": {
            "range": (0.01, 100.0),
            "unit": "Hz_str",
            "description": "Structural frequency (reorganization rate)",
        },
        "phase": {
            "range": (0.0, 2 * 3.14159),
            "unit": "radians",
            "description": "Phase angle for network synchrony",
        },
        "coherence": {
            "range": (0.0, 1.0),
            "unit": "dimensionless",
            "description": "Structural stability measure C(t)",
        },
        "delta_nfr": {
            "range": (-10.0, 10.0),
            "unit": "dimensionless",
            "description": "Internal reorganization gradient ΔNFR",
        },
        "epi": {
            "range": (0.0, 1.0),
            "unit": "dimensionless",
            "description": "Primary Information Structure magnitude",
        },
        "edge_probability": {
            "range": (0.0, 1.0),
            "unit": "probability",
            "description": "Network edge connection probability",
        },
        "num_nodes": {
            "range": (1, 100000),
            "unit": "count",
            "description": "Number of nodes in network",
        },
    }

    def __init__(
        self,
        parameter: str,
        value: Any,
        valid_range: Optional[tuple] = None,
        reason: Optional[str] = None,
    ):
        # Get constraint info if available
        constraint_info = self.PARAMETER_CONSTRAINTS.get(parameter)

        if constraint_info and not valid_range:
            valid_range = constraint_info["range"]
            reason = reason or constraint_info["description"]

        suggestion_parts = []
        if valid_range:
            min_val, max_val = valid_range
            suggestion_parts.append(f"'{parameter}' must be in range [{min_val}, {max_val}]")

        if constraint_info:
            suggestion_parts.append(f"Unit: {constraint_info['unit']}")

        if reason:
            suggestion_parts.append(f"Structural meaning: {reason}")

        context = {
            "parameter": parameter,
            "provided_value": value,
            "valid_range": (f"[{valid_range[0]}, {valid_range[1]}]" if valid_range else "see docs"),
        }

        super().__init__(
            message=f"Invalid network configuration for '{parameter}'",
            suggestion=" | ".join(suggestion_parts) if suggestion_parts else None,
            docs_url="https://github.com/fermga/Teoria-de-la-naturaleza-fractal-resonante-TNFR-/blob/main/docs/source/api/overview.md",
            context=context,
        )


class PhaseError(TNFRUserError):
    """Error raised when phase synchrony is violated.

    TNFR requires explicit phase checking before coupling operations.
    This error indicates phase incompatibility between nodes.

    Enforces Invariant #5: Phase check from AGENTS.md

    Parameters
    ----------
    node1 : str
        First node ID.
    node2 : str
        Second node ID.
    phase1 : float
        Phase of first node (radians).
    phase2 : float
        Phase of second node (radians).
    threshold : float
        Phase difference threshold for coupling.

    Examples
    --------
    >>> raise PhaseError("n1", "n2", 0.5, 2.8, 0.5)
    """

    def __init__(
        self,
        node1: str,
        node2: str,
        phase1: float,
        phase2: float,
        threshold: float = 0.5,
    ):
        phase_diff = abs(phase1 - phase2)

        suggestion = (
            f"Nodes cannot couple: phase difference ({phase_diff:.3f} rad) "
            f"exceeds threshold ({threshold:.3f} rad). "
            f"Apply phase synchronization or adjust threshold."
        )

        context = {
            "node1": node1,
            "node2": node2,
            "phase1": f"{phase1:.3f} rad",
            "phase2": f"{phase2:.3f} rad",
            "phase_difference": f"{phase_diff:.3f} rad",
            "threshold": f"{threshold:.3f} rad",
        }

        super().__init__(
            message=f"Phase synchrony violation between nodes '{node1}' and '{node2}'",
            suggestion=suggestion,
            docs_url="https://github.com/fermga/Teoria-de-la-naturaleza-fractal-resonante-TNFR-/blob/main/GLOSSARY.md#phase",
            context=context,
        )


class CoherenceError(TNFRUserError):
    """Error raised when coherence operations violate monotonicity.

    Coherence operator must not decrease C(t) except in controlled
    dissonance tests. This error indicates unexpected coherence loss.

    Enforces Invariant #1: EPI coherent form from AGENTS.md

    Parameters
    ----------
    operation : str
        The operation that caused coherence decrease.
    before : float
        Coherence C(t) before operation.
    after : float
        Coherence C(t) after operation.
    node_id : str, optional
        Node ID if the error is node-specific.

    Examples
    --------
    >>> raise CoherenceError("coherence", 0.85, 0.42)
    """

    def __init__(
        self,
        operation: str,
        before: float,
        after: float,
        node_id: Optional[str] = None,
    ):
        decrease = before - after
        percent_loss = (decrease / before * 100) if before > 0 else 0

        suggestion = (
            f"Coherence decreased by {decrease:.3f} ({percent_loss:.1f}%). "
            f"This violates the coherence monotonicity invariant. "
            f"Check if this is a controlled dissonance test or if "
            f"there's an unexpected structural instability."
        )

        context = {
            "operation": operation,
            "coherence_before": f"{before:.3f}",
            "coherence_after": f"{after:.3f}",
            "decrease": f"{decrease:.3f}",
            "percent_loss": f"{percent_loss:.1f}%",
        }

        if node_id:
            context["node_id"] = node_id

        super().__init__(
            message=f"Unexpected coherence decrease during '{operation}'",
            suggestion=suggestion,
            docs_url="https://github.com/fermga/Teoria-de-la-naturaleza-fractal-resonante-TNFR-/blob/main/AGENTS.md#canonical-invariants",
            context=context,
        )


class FrequencyError(TNFRUserError):
    """Error raised when structural frequency νf is invalid.

    Structural frequency must be positive and expressed in Hz_str
    (structural hertz) units. This error indicates frequency violations.

    Enforces Invariant #2: Structural units from AGENTS.md

    Parameters
    ----------
    node_id : str
        Node ID with invalid frequency.
    vf : float
        The invalid frequency value.
    operation : str, optional
        Operation that triggered the check.

    Examples
    --------
    >>> raise FrequencyError("n1", -0.5, "emission")
    """

    def __init__(
        self,
        node_id: str,
        vf: float,
        operation: Optional[str] = None,
    ):
        if vf <= 0:
            suggestion = (
                f"Structural frequency νf must be positive (Hz_str units). "
                f"Set νf > 0 for node '{node_id}'. "
                f"Typical range: 0.1 to 10.0 Hz_str."
            )
        elif vf > 100:
            suggestion = (
                f"Structural frequency νf = {vf:.3f} Hz_str is very high. "
                f"Typical range: 0.1 to 10.0 Hz_str. "
                f"Verify this is intentional."
            )
        else:
            suggestion = f"Verify structural frequency for node '{node_id}'."

        context = {
            "node_id": node_id,
            "vf": f"{vf:.3f} Hz_str",
            "valid_range": "[0.01, 100.0] Hz_str",
        }

        if operation:
            context["operation"] = operation

        super().__init__(
            message=f"Invalid structural frequency for node '{node_id}'",
            suggestion=suggestion,
            docs_url="https://github.com/fermga/Teoria-de-la-naturaleza-fractal-resonante-TNFR-/blob/main/GLOSSARY.md#structural-frequency",
            context=context,
        )
