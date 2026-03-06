"""Postcondition validators for TNFR structural operators.

Each operator has specific guarantees that must be verified after execution
to ensure TNFR structural invariants are maintained. This package provides
postcondition validators for operators that need strict verification.

Postconditions ensure that operators fulfill their contracts and maintain
canonical TNFR physics.

Exception hierarchy
-------------------
``OperatorContractViolation`` inherits from
``StructuralIntegrityViolation`` (physics.integrity) so that a single
``except StructuralIntegrityViolation`` catch handles both conservation-law
and operator-contract failures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from ...physics.integrity import StructuralIntegrityViolation

__all__ = [
    "OperatorContractViolation",
]

class OperatorContractViolation(StructuralIntegrityViolation):
    """Raised when an operator's postconditions are violated.

    Inherits from ``StructuralIntegrityViolation`` so callers can catch
    both conservation and contract failures with a single base class.
    """

    def __init__(self, operator: str, reason: str) -> None:
        self.reason = reason
        super().__init__(
            operator=operator,
            violation_type="postcondition",
            details={"reason": reason},
        )
