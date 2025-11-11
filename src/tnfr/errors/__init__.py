"""Contextual error messages for TNFR operations.

This module provides user-friendly error messages with:
- Clear explanations of what went wrong
- Suggestions for how to fix the issue
- Links to relevant documentation
- Fuzzy matching for common typos

Examples
--------
>>> from tnfr.errors import OperatorSequenceError
>>> raise OperatorSequenceError(
...     "Invalid operator 'emision'",
...     suggestion="Did you mean 'emission'?",
...     docs_url="https://tnfr.readthedocs.io/operators.html"
... )
"""

from __future__ import annotations

__all__ = [
    "TNFRUserError",
    "OperatorSequenceError",
    "NetworkConfigError",
    "PhaseError",
    "CoherenceError",
    "FrequencyError",
]

from .contextual import (
    TNFRUserError,
    OperatorSequenceError,
    NetworkConfigError,
    PhaseError,
    CoherenceError,
    FrequencyError,
)
