"""Tests verifying @runtime_checkable protocols support isinstance checks.

This test module validates that TNFR protocols used with isinstance() have the
@runtime_checkable decorator applied and function correctly at runtime.
"""

from __future__ import annotations

import pytest

from tnfr.validation import Validator, ValidationOutcome


def test_validator_protocol_is_runtime_checkable() -> None:
    """Verify Validator protocol supports isinstance checks."""

    class MockValidator:
        """Mock implementation conforming to Validator protocol."""

        def validate(self, subject, **kwargs):
            """Mock validate method."""
            return ValidationOutcome(
                subject=subject,
                passed=True,
                summary={"test": "passed"},
            )

        def report(self, outcome):
            """Mock report method."""
            return f"Validation result: {outcome.passed}"

    mock = MockValidator()

    # This should work because Validator has @runtime_checkable
    assert isinstance(
        mock, Validator
    ), "MockValidator should satisfy Validator protocol"


def test_validator_protocol_rejects_incomplete_implementation() -> None:
    """Verify Validator protocol correctly identifies incomplete implementations."""

    class IncompleteValidator:
        """Implementation missing the report method."""

        def validate(self, subject, **kwargs):
            """Only validate method, missing report."""
            return ValidationOutcome(
                subject=subject,
                passed=True,
                summary={},
            )

    incomplete = IncompleteValidator()

    # Should not satisfy protocol due to missing report method
    assert not isinstance(
        incomplete, Validator
    ), "IncompleteValidator should not satisfy Validator protocol"


def test_validator_protocol_with_protocol_methods() -> None:
    """Verify protocol checking works with proper method signatures."""

    class ConformingValidator:
        """Full implementation of Validator protocol."""

        def validate(self, subject, **kwargs):
            """Validate subject and return outcome."""
            return ValidationOutcome(
                subject=subject,
                passed=True,
                summary={"checks": 1, "errors": 0},
                artifacts=None,
            )

        def report(self, outcome):
            """Generate textual report from outcome."""
            status = "✓" if outcome.passed else "✗"
            return f"{status} Validation completed"

    validator = ConformingValidator()

    # Full conformance to protocol
    assert isinstance(validator, Validator)

    # Can also call protocol methods
    outcome = validator.validate("test_subject")
    assert outcome.passed
    assert outcome.subject == "test_subject"

    report = validator.report(outcome)
    assert "✓" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
