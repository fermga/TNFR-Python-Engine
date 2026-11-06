"""Security utilities for TNFR.

This module provides security utilities including SQL injection prevention,
input validation, secure database query patterns, and secure subprocess execution.
These utilities are designed to be used proactively should database functionality
be added to TNFR, and to prevent command injection in script execution.

Structural Context (TNFR)
-------------------------
These security utilities maintain TNFR structural coherence by ensuring:
- Data persistence preserves EPI integrity
- Query operations maintain nodal coherence
- Input validation respects structural frequency constraints
- Database interactions preserve operational fractality
- External command execution maintains system integrity

Example
-------
>>> from tnfr.security import SecureQueryBuilder, validate_identifier
>>> from tnfr.security import run_command_safely, validate_git_ref
>>> # Validate database identifiers
>>> table = validate_identifier("nfr_nodes")  # Safe
>>> # Use parameterized queries
>>> builder = SecureQueryBuilder()
>>> query, params = builder.select("nodes", ["epi", "nu_f"]).where("id = ?", 123).build()
>>> # Execute commands safely
>>> result = run_command_safely(["git", "status"])
"""

from __future__ import annotations

from .database import (
    SecureQueryBuilder,
    execute_parameterized_query,
    sanitize_string_input,
    validate_identifier,
)
from .subprocess import (
    CommandValidationError,
    PathTraversalError,
    resolve_safe_path,
    run_command_safely,
    validate_file_path,
    validate_git_ref,
    validate_path_safe,
    validate_version_string,
)
from .validation import (
    validate_nodal_input,
    validate_phase_value,
    validate_structural_frequency,
)

__all__ = (
    "SecureQueryBuilder",
    "execute_parameterized_query",
    "sanitize_string_input",
    "validate_identifier",
    "validate_nodal_input",
    "validate_phase_value",
    "validate_structural_frequency",
    "CommandValidationError",
    "PathTraversalError",
    "run_command_safely",
    "validate_git_ref",
    "validate_path_safe",
    "validate_file_path",
    "resolve_safe_path",
    "validate_version_string",
)
