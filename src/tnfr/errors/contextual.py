"""Contextual error handling for TNFR operations.

This module provides enhanced error messages that guide users to solutions
while maintaining TNFR theoretical compliance. All errors include:

1. Clear explanation of the violation
2. Actionable suggestions for resolution
3. Links to relevant documentation
4. Context about the structural operation that failed

Scope
-----
Errors describe the caller's failed validation. They do not independently
validate a state or establish a universal monotonicity/stability theorem.
Configured bounds must be supplied by the actual consumer; the generic nodal
domains below must not replace them with arbitrary finite cutoffs.
"""

from __future__ import annotations

from difflib import get_close_matches
from typing import Any

_REFERENCE_ROOT = "https://github.com/fermga/TNFR-Python-Engine/blob/main/"

__all__ = [
    "TNFRUserError",
    "OperatorSequenceError",
    "NetworkConfigError",
    "PhaseError",
    "CoherenceError",
    "FrequencyError",
    "TNFRValueError",
    "TNFRSecurityError",
    "TNFRSecurityWarning",
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
    ...     suggestion="νf must be a finite nonnegative real scalar"
    ... )
    """

    def __init__(
        self,
        message: str,
        suggestion: str | None = None,
        docs_url: str | None = None,
        context: dict[str, Any] | None = None,
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

    Enforces Invariant #4: Grammar Compliance (operator closure) from AGENTS.md

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
        sequence_so_far: list[str] | None = None,
        valid_next: list[str] | None = None,
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
            "sequence_so_far": (
                " → ".join(sequence_so_far) if sequence_so_far else "empty"
            ),
            "operator_count": len(sequence_so_far),
        }

        super().__init__(
            message=f"Invalid operator sequence: '{invalid_operator}' cannot be applied",
            suggestion=suggestion,
            docs_url=_REFERENCE_ROOT + "docs/API_CONTRACTS.md",
            context=context,
        )


class NetworkConfigError(TNFRUserError):
    """Error raised when network configuration violates TNFR constraints.

    This error validates configuration parameters and provides valid ranges
    with physical/structural meaning.

    Enforces multiple invariants:
    - Invariant #5: Structural Metrology (νf in Hz_str)
    - Invariant #2: Phase-Coherent Coupling (phase check requirements)
    - Invariant #1: Nodal Equation Integrity (node birth/collapse conditions)

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
    ...     reason="Capacity must be finite and nonnegative"
    ... )
    """

    # Valid parameter ranges with structural meaning
    PARAMETER_CONSTRAINTS = {
        "vf": {
            "range": None,
            "unit": "Hz_str",
            "description": "Finite nonnegative reorganization capacity; zero is valid",
        },
        "phase": {
            "range": None,
            "unit": "radians",
            "description": "Finite circular phase representative; comparison uses wrapped separation",
        },
        "coherence": {
            "range": (0.0, 1.0),
            "unit": "dimensionless",
            "description": "Configured coherence readout C(t); not a stability certificate",
        },
        "delta_nfr": {
            "range": None,
            "unit": "[EPI]/([nu_f][time])",
            "description": "Finite signed reorganization pressure; model bounds require explicit policy",
        },
        "epi": {
            "range": None,
            "unit": "declared structural chart",
            "description": "Finite signed scalar EPI; clipping bounds belong to the configured solver",
        },
        "edge_probability": {
            "range": (0.0, 1.0),
            "unit": "probability",
            "description": "Network edge connection probability",
        },
        "num_nodes": {
            "range": None,
            "unit": "count",
            "description": "Nonnegative integer count; individual builders may require nonempty support",
        },
    }

    def __init__(
        self,
        parameter: str,
        value: Any,
        valid_range: tuple | None = None,
        reason: str | None = None,
    ):
        # Get constraint info if available
        constraint_info = self.PARAMETER_CONSTRAINTS.get(parameter)

        if constraint_info and not valid_range:
            valid_range = constraint_info["range"]
            reason = reason or constraint_info["description"]

        suggestion_parts = []
        if valid_range:
            min_val, max_val = valid_range
            suggestion_parts.append(
                f"'{parameter}' must be in range [{min_val}, {max_val}]"
            )

        if constraint_info:
            suggestion_parts.append(f"Unit: {constraint_info['unit']}")

        if reason:
            suggestion_parts.append(f"Structural meaning: {reason}")

        context = {
            "parameter": parameter,
            "provided_value": value,
            "valid_range": (
                f"[{valid_range[0]}, {valid_range[1]}]" if valid_range else "see docs"
            ),
        }

        super().__init__(
            message=f"Invalid network configuration for '{parameter}'",
            suggestion=" | ".join(suggestion_parts) if suggestion_parts else None,
            docs_url=_REFERENCE_ROOT
            + "docs/API_CONTRACTS.md#nodal-solver-input-clock-and-output-boundaries",
            context=context,
        )


class PhaseError(TNFRUserError):
    """Error raised when phase synchrony is violated.

    TNFR requires explicit phase checking before coupling operations.
    This error indicates phase incompatibility between nodes.

    Enforces Invariant #2: Phase-Coherent Coupling from AGENTS.md

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
    threshold : float, optional
        Phase difference threshold for coupling. Defaults to the canonical
        U3 gate ``DELTA_PHI_MAX = π/2``.

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
        threshold: float | None = None,
    ):
        # Keep imports lazy: this foundational error module is imported while
        # constants, types, and numeric utilities are still being initialized.
        from ..constants.canonical import DELTA_PHI_MAX
        from ..utils import angle_diff

        if threshold is None:
            threshold = DELTA_PHI_MAX
        phase_diff = abs(angle_diff(phase1, phase2))

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
            docs_url=_REFERENCE_ROOT + "AGENTS.md#structural-triad",
            context=context,
        )


class CoherenceError(TNFRUserError):
    """Report a decrease against a caller's declared coherence postcondition.

    Nondecrease must belong to the actual operator/model contract and observation
    scope. It is not a consequence of the nodal identity for every trajectory;
    constructing this error does not independently validate those premises.

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
        node_id: str | None = None,
    ):
        decrease = before - after
        percent_loss = (decrease / before * 100) if before > 0 else 0

        suggestion = (
            f"Coherence decreased by {decrease:.3f} ({percent_loss:.1f}%). "
            "Check the caller's nondecrease postcondition and the scope of both "
            "observations. General nodal evolution does not guarantee monotone C(t)."
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
            docs_url=_REFERENCE_ROOT + "theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md",
            context=context,
        )


class FrequencyError(TNFRUserError):
    """Error raised when structural frequency νf is invalid.

    Capacity must be a finite nonnegative real scalar, with no boolean/text
    coercion or underflow of a nonzero input to represented zero. Zero is
    admitted; no finite upper cutoff is imposed. The resulting product must
    still be representable. Numeric admission does not calibrate units.

    Parameters
    ----------
    vf : Any
        The invalid frequency value.
    node_id : str, optional
        Node ID with invalid frequency.
    operation : str, optional
        Operation that triggered the check.

    Examples
    --------
    >>> raise FrequencyError(-0.5, node_id="n1", operation="validation")
    """

    def __init__(
        self,
        vf: Any,
        node_id: str | None = None,
        operation: str | None = None,
    ):
        node_msg = f" for node '{node_id}'" if node_id else ""

        suggestion = (
            f"Set capacity νf{node_msg} to a finite nonnegative real scalar. "
            "Zero is valid; a nonzero input must remain nonzero in binary64. "
            "Do not supply booleans or numeric text."
        )

        context = {
            "vf": vf,
            "valid_range": "finite represented real values >= 0 (zero admitted)",
        }

        if node_id:
            context["node_id"] = node_id
        if operation:
            context["operation"] = operation

        super().__init__(
            message=f"Invalid structural frequency{node_msg}",
            suggestion=suggestion,
            docs_url=_REFERENCE_ROOT + "theory/NODAL_PARAMETER_FOUNDATIONS.md",
            context=context,
        )


class TNFRValueError(TNFRUserError, ValueError):
    """Error raised when an operation receives an argument with inappropriate value.

    This is a drop-in replacement for ValueError that adds TNFR context.
    It inherits from both TNFRUserError and ValueError, allowing it to be
    caught by existing exception handlers while providing enhanced diagnostics.

    Parameters
    ----------
    message : str
        Primary error message.
    suggestion : str, optional
        Actionable suggestion for resolution.
    docs_url : str, optional
        Link to relevant documentation.
    context : dict, optional
        Additional context about the error.

    Examples
    --------
    >>> raise TNFRValueError(
    ...     "Invalid dimension size",
    ...     suggestion="Dimension must be positive",
    ...     context={"dim": -1}
    ... )
    """

    def __init__(
        self,
        message: str,
        suggestion: str | None = None,
        docs_url: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            suggestion=suggestion,
            docs_url=docs_url,
            context=context,
        )


class TNFRSecurityError(TNFRValueError):
    """Security validation error for input sanitization or integrity failures.

    Raised when:
    - Input contains forbidden patterns (injection attempts)
    - Cache signatures are invalid (tampering detected)
    - Path traversal attempts are detected
    """

    def __init__(self, message: str, suspicious_input: str | None = None, **kwargs):
        context = kwargs.get("context", {})
        if suspicious_input:
            context["suspicious_input"] = suspicious_input
        kwargs["context"] = context

        super().__init__(message, **kwargs)
        self.suspicious_input = suspicious_input


class TNFRSecurityWarning(UserWarning):
    """Issued when potentially unsafe serialization is used without signing."""
