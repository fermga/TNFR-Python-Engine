"""Security utilities for TNFR.

This module provides security utilities including SQL injection prevention,
input validation, and secure database query patterns. These utilities are
designed to be used proactively should database functionality be added to TNFR.

Structural Context (TNFR)
-------------------------
These security utilities maintain TNFR structural coherence by ensuring:
- Data persistence preserves EPI integrity
- Query operations maintain nodal coherence
- Input validation respects structural frequency constraints
- Database interactions preserve operational fractality

Example
-------
>>> from tnfr.security import SecureQueryBuilder, validate_identifier
>>> # Validate database identifiers
>>> table = validate_identifier("nfr_nodes")  # Safe
>>> # Use parameterized queries
>>> builder = SecureQueryBuilder()
>>> query, params = builder.select("nodes", ["epi", "nu_f"]).where("id = ?", 123).build()
"""

from __future__ import annotations

from .database import (
    SecureQueryBuilder,
    execute_parameterized_query,
    sanitize_string_input,
    validate_identifier,
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
)
