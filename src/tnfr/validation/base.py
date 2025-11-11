"""Base validation types and protocols.

This module defines the core validation abstractions used throughout the
validation subsystem. It must remain import-free to avoid circular dependencies.
"""

from __future__ import annotations

from typing import Any, Generic, Mapping, Protocol, TypeVar, runtime_checkable

from ..compat.dataclass import dataclass

SubjectT = TypeVar("SubjectT")


@dataclass(slots=True)
class ValidationOutcome(Generic[SubjectT]):
    """Result emitted by all canonical TNFR validators."""

    subject: SubjectT
    """The validated subject in canonical form."""

    passed: bool
    """Whether the validation succeeded without invariant violations."""

    summary: Mapping[str, Any]
    """Structured diagnostics describing the performed checks."""

    artifacts: Mapping[str, Any] | None = None
    """Optional artefacts (e.g. clamped nodes, normalised vectors)."""


@runtime_checkable
class Validator(Protocol[SubjectT]):
    """Contract implemented by runtime and spectral validators."""

    def validate(self, subject: SubjectT, /, **kwargs: Any) -> ValidationOutcome[SubjectT]:
        """Validate ``subject`` returning a :class:`ValidationOutcome`."""

    def report(self, outcome: "ValidationOutcome[SubjectT]") -> str:
        """Produce a concise textual explanation for ``outcome``."""


__all__ = (
    "SubjectT",
    "ValidationOutcome",
    "Validator",
)
