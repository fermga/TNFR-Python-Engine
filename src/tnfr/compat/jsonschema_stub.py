"""Lightweight stub for jsonschema when it's not installed.

This stub provides minimal type compatibility for jsonschema when it's not
installed, allowing type checking to succeed.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Draft7Validator", "exceptions"]


class _NotInstalledError(RuntimeError):
    """Raised when trying to use jsonschema operations without jsonschema installed."""

    def __init__(self, operation: str = "jsonschema operation") -> None:
        super().__init__(
            f"Cannot perform {operation}: jsonschema is not installed. "
            "Install it with: pip install jsonschema"
        )


class SchemaError(Exception):
    """Stub for jsonschema.exceptions.SchemaError."""


class ValidationError(Exception):
    """Stub for jsonschema.exceptions.ValidationError."""


class _ExceptionsStub:
    """Stub for jsonschema.exceptions module."""

    SchemaError = SchemaError
    ValidationError = ValidationError


class Draft7Validator:
    """Stub for jsonschema.Draft7Validator."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise _NotInstalledError("Draft7Validator creation")

    @classmethod
    def check_schema(cls, schema: Any) -> None:
        raise _NotInstalledError("Draft7Validator.check_schema")

    def validate(self, instance: Any) -> None:
        raise _NotInstalledError("Draft7Validator.validate")

    def iter_errors(self, instance: Any) -> Any:
        raise _NotInstalledError("Draft7Validator.iter_errors")


# Module-level stubs
exceptions = _ExceptionsStub()
