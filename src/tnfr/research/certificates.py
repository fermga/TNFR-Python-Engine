r"""Numerical certificates for floating-point research claims.

A floating-point measurement is only admissible with a certificate that ties its
verdict to numerical error rather than a hand-picked cut: the derived tolerance,
the backward error and the condition number of the problem, and the working
precision.  This complements the exact invariant-subspace certificate of
:mod:`tnfr.physics.spectral_certificates`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

__all__ = ["NumericalCertificate", "certify_within_tolerance"]


@dataclass(frozen=True)
class NumericalCertificate:
    """A pass/fail verdict for ``|value| < tolerance`` with numerical context."""

    quantity: str
    value: float
    tolerance: float
    passed: bool
    backward_error: float | None = None
    condition_number: float | None = None
    precision: str = "float64"

    def to_dict(self) -> dict:
        return {
            "quantity": self.quantity,
            "value": self.value,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "backward_error": self.backward_error,
            "condition_number": self.condition_number,
            "precision": self.precision,
        }

    def validate_for_admission(self) -> None:
        """Require numerical context before admitting a public result."""
        if not self.quantity or not self.precision:
            raise ValueError("quantity and precision are required")
        if self.precision not in {"float32", "float64", "longdouble", "exact"}:
            raise ValueError("precision must name a supported numerical model")
        values = (self.value, self.tolerance)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("value and tolerance must be finite")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be nonnegative")
        if self.backward_error is None or self.condition_number is None:
            raise ValueError(
                "backward_error and condition_number are required for admission"
            )
        if not math.isfinite(self.backward_error) or self.backward_error < 0.0:
            raise ValueError("backward_error must be finite and nonnegative")
        if not math.isfinite(self.condition_number) or self.condition_number < 0.0:
            raise ValueError("condition_number must be finite and nonnegative")


def certify_within_tolerance(
    quantity: str,
    value: float,
    tolerance: float,
    *,
    backward_error: float | None = None,
    condition_number: float | None = None,
    precision: str = "float64",
) -> NumericalCertificate:
    """Certify ``|value| < tolerance`` with the supporting numerical context."""
    return NumericalCertificate(
        quantity=quantity,
        value=float(value),
        tolerance=float(tolerance),
        passed=abs(float(value)) < float(tolerance),
        backward_error=backward_error,
        condition_number=condition_number,
        precision=precision,
    )
