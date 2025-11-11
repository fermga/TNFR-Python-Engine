"""Postcondition validators for TNFR structural operators.

Each operator has specific guarantees that must be verified after execution
to ensure TNFR structural invariants are maintained. This package provides
postcondition validators for operators that need strict verification.

Postconditions ensure that operators fulfill their contracts and maintain
canonical TNFR physics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__ = [
    "OperatorContractViolation",
]


class OperatorContractViolation(Exception):
    """Raised when an operator's postconditions are violated."""

    def __init__(self, operator: str, reason: str) -> None:
        """Initialize contract violation error.

        Parameters
        ----------
        operator : str
            Name of the operator that violated its contract
        reason : str
            Description of why the contract was violated
        """
        self.operator = operator
        self.reason = reason
        super().__init__(f"{operator} contract violation: {reason}")
