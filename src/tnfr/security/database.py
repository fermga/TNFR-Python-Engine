"""Secure database query utilities for SQL injection prevention.

This module provides utilities for safe database interactions, ensuring that
SQL queries are properly parameterized and identifiers are validated. These
utilities should be used whenever database functionality is added to TNFR.

Security Principles
-------------------
1. **Parameterized Queries**: Always use placeholders (?, :name) for values
2. **Identifier Validation**: Validate table/column names against whitelist
3. **No String Concatenation**: Never build queries with string concatenation
4. **Input Sanitization**: Validate and sanitize all user inputs

TNFR Structural Context
------------------------
These utilities preserve TNFR structural coherence when persisting:
- EPI (Estructura Primaria de Información)
- NFR (Nodo Fractal Resonante) metadata
- Structural frequency (νf) measurements
- Phase relationships (φ) between nodes
- Coherence metrics C(t) and sense index Si

Example
-------
>>> # Safe parameterized query
>>> builder = SecureQueryBuilder()
>>> query, params = builder.select("nfr_nodes", ["id", "epi", "nu_f"])\\
...     .where("nu_f > ?", 0.5)\\
...     .order_by("nu_f", "DESC")\\
...     .build()
>>> print(query)
SELECT id, epi, nu_f FROM nfr_nodes WHERE nu_f > ? ORDER BY nu_f DESC
>>> print(params)
[0.5]

>>> # Validate identifiers before use
>>> table_name = validate_identifier("nfr_nodes")
>>> column_name = validate_identifier("nu_f")
"""

from __future__ import annotations

import re
from typing import Any

# Valid SQL identifier pattern: alphanumeric and underscores only, 1-64 chars
_VALID_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$")

# Common SQL keywords that should not be used as identifiers
_SQL_KEYWORDS = frozenset(
    {
        "SELECT",
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "CREATE",
        "ALTER",
        "TRUNCATE",
        "EXEC",
        "EXECUTE",
        "UNION",
        "FROM",
        "WHERE",
        "AND",
        "OR",
        "NOT",
        "IN",
        "EXISTS",
        "JOIN",
        "TABLE",
        "DATABASE",
        "GRANT",
        "REVOKE",
    }
)


class SQLInjectionError(ValueError):
    """Exception raised when potential SQL injection is detected."""


def validate_identifier(identifier: str, *, allow_keywords: bool = False) -> str:
    """Validate a SQL identifier (table or column name).

    This function ensures that identifiers are safe to use in SQL queries
    by checking against a whitelist pattern and optionally rejecting SQL keywords.

    Parameters
    ----------
    identifier : str
        The identifier to validate (table name, column name, etc.)
    allow_keywords : bool, optional
        If False (default), reject SQL keywords as identifiers

    Returns
    -------
    str
        The validated identifier

    Raises
    ------
    SQLInjectionError
        If the identifier is invalid or potentially unsafe

    Example
    -------
    >>> validate_identifier("nfr_nodes")
    'nfr_nodes'
    >>> validate_identifier("nu_f_measurements")
    'nu_f_measurements'
    >>> validate_identifier("DROP")  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    SQLInjectionError: Identifier 'DROP' is a SQL keyword
    >>> validate_identifier("invalid-name")  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    SQLInjectionError: Invalid identifier 'invalid-name'
    """
    if not isinstance(identifier, str):
        raise SQLInjectionError(f"Identifier must be a string, got {type(identifier).__name__}")

    if not identifier:
        raise SQLInjectionError("Identifier cannot be empty")

    if not _VALID_IDENTIFIER_PATTERN.match(identifier):
        raise SQLInjectionError(
            f"Invalid identifier '{identifier}': must contain only alphanumeric "
            f"characters and underscores, start with letter or underscore, "
            f"and be 1-64 characters long"
        )

    if not allow_keywords and identifier.upper() in _SQL_KEYWORDS:
        raise SQLInjectionError(f"Identifier '{identifier}' is a SQL keyword and cannot be used")

    return identifier


def sanitize_string_input(value: str, *, max_length: int = 1000) -> str:
    """Sanitize string input for safe database operations.

    This function validates string inputs to prevent SQL injection and ensure
    reasonable length constraints. Note: This is NOT a replacement for
    parameterized queries, but an additional validation layer.

    Parameters
    ----------
    value : str
        The string value to sanitize
    max_length : int, optional
        Maximum allowed length (default: 1000)

    Returns
    -------
    str
        The validated string value

    Raises
    ------
    SQLInjectionError
        If the value is too long or contains suspicious patterns

    Example
    -------
    >>> sanitize_string_input("valid string")
    'valid string'
    >>> sanitize_string_input("a" * 10000)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    SQLInjectionError: Input exceeds maximum length...
    """
    if not isinstance(value, str):
        raise SQLInjectionError(f"Value must be a string, got {type(value).__name__}")

    if len(value) > max_length:
        raise SQLInjectionError(
            f"Input exceeds maximum length of {max_length} characters " f"(got {len(value)})"
        )

    # Check for null bytes which can truncate strings in some contexts
    if "\x00" in value:
        raise SQLInjectionError("Input contains null bytes which are not allowed")

    return value


class SecureQueryBuilder:
    """Builder for constructing safe, parameterized SQL queries.

    This class provides a fluent interface for building SQL queries with
    proper parameterization to prevent SQL injection. All values are
    automatically parameterized, and identifiers are validated.

    TNFR Context
    ------------
    Use this builder when persisting TNFR structural data:
    - NFR node states (EPI, νf, phase)
    - Network topology and couplings
    - Coherence measurements C(t)
    - Sense index Si values
    - Operator application history

    Example
    -------
    >>> builder = SecureQueryBuilder()
    >>> query, params = builder.select("nfr_nodes", ["id", "epi"])\\
    ...     .where("nu_f > ?", 0.5)\\
    ...     .where("phase BETWEEN ? AND ?", 0.0, 3.14)\\
    ...     .order_by("nu_f", "DESC")\\
    ...     .limit(10)\\
    ...     .build()
    >>> print(query)  # doctest: +NORMALIZE_WHITESPACE
    SELECT id, epi FROM nfr_nodes WHERE nu_f > ? AND phase BETWEEN ? AND ?
    ORDER BY nu_f DESC LIMIT 10
    >>> print(params)
    [0.5, 0.0, 3.14]
    """

    def __init__(self) -> None:
        """Initialize a new query builder."""
        self._query_parts: list[str] = []
        self._params: list[Any] = []
        self._operation: str | None = None

    def select(self, table: str, columns: list[str] | None = None) -> SecureQueryBuilder:
        """Start a SELECT query.

        Parameters
        ----------
        table : str
            Table name (will be validated)
        columns : list[str], optional
            List of column names to select (default: all columns with *)

        Returns
        -------
        SecureQueryBuilder
            Self for method chaining
        """
        self._operation = "SELECT"
        table = validate_identifier(table)

        if columns is None:
            cols = "*"
        else:
            validated_cols = [validate_identifier(col) for col in columns]
            cols = ", ".join(validated_cols)

        self._query_parts.append(f"SELECT {cols} FROM {table}")
        return self

    def insert(self, table: str, columns: list[str]) -> SecureQueryBuilder:
        """Start an INSERT query.

        Parameters
        ----------
        table : str
            Table name (will be validated)
        columns : list[str]
            List of column names

        Returns
        -------
        SecureQueryBuilder
            Self for method chaining
        """
        self._operation = "INSERT"
        table = validate_identifier(table)
        validated_cols = [validate_identifier(col) for col in columns]
        cols_str = ", ".join(validated_cols)
        placeholders = ", ".join(["?"] * len(columns))
        self._query_parts.append(f"INSERT INTO {table} ({cols_str}) VALUES ({placeholders})")
        return self

    def update(self, table: str) -> SecureQueryBuilder:
        """Start an UPDATE query.

        Parameters
        ----------
        table : str
            Table name (will be validated)

        Returns
        -------
        SecureQueryBuilder
            Self for method chaining
        """
        self._operation = "UPDATE"
        table = validate_identifier(table)
        self._query_parts.append(f"UPDATE {table}")
        return self

    def set(self, **columns: Any) -> SecureQueryBuilder:
        """Add SET clause for UPDATE query.

        Parameters
        ----------
        **columns
            Column-value pairs to update

        Returns
        -------
        SecureQueryBuilder
            Self for method chaining
        """
        if self._operation != "UPDATE":
            raise SQLInjectionError("SET clause can only be used with UPDATE")

        validated_cols = [validate_identifier(col) for col in columns.keys()]
        set_parts = [f"{col} = ?" for col in validated_cols]
        self._params.extend(columns.values())
        self._query_parts.append("SET " + ", ".join(set_parts))
        return self

    def delete(self, table: str) -> SecureQueryBuilder:
        """Start a DELETE query.

        Parameters
        ----------
        table : str
            Table name (will be validated)

        Returns
        -------
        SecureQueryBuilder
            Self for method chaining
        """
        self._operation = "DELETE"
        table = validate_identifier(table)
        self._query_parts.append(f"DELETE FROM {table}")
        return self

    def where(self, condition: str, *values: Any) -> SecureQueryBuilder:
        """Add a WHERE clause with parameterized values.

        The condition should use '?' placeholders for values.

        Parameters
        ----------
        condition : str
            WHERE condition with '?' placeholders
        *values
            Values to bind to placeholders

        Returns
        -------
        SecureQueryBuilder
            Self for method chaining
        """
        # Validate that condition doesn't contain suspicious patterns
        if ";" in condition or "--" in condition or "/*" in condition:
            raise SQLInjectionError("WHERE condition contains suspicious SQL patterns")

        # Count expected placeholders
        expected_params = condition.count("?")
        if len(values) != expected_params:
            raise SQLInjectionError(
                f"WHERE condition expects {expected_params} parameters, " f"got {len(values)}"
            )

        prefix = "WHERE" if not any("WHERE" in part for part in self._query_parts) else "AND"
        self._query_parts.append(f"{prefix} {condition}")
        self._params.extend(values)
        return self

    def order_by(self, column: str, direction: str = "ASC") -> SecureQueryBuilder:
        """Add an ORDER BY clause.

        Parameters
        ----------
        column : str
            Column name to order by (will be validated)
        direction : str, optional
            Sort direction: 'ASC' or 'DESC' (default: 'ASC')

        Returns
        -------
        SecureQueryBuilder
            Self for method chaining
        """
        column = validate_identifier(column)
        direction = direction.upper()
        if direction not in ("ASC", "DESC"):
            raise SQLInjectionError(f"Invalid sort direction: {direction}")

        self._query_parts.append(f"ORDER BY {column} {direction}")
        return self

    def limit(self, count: int) -> SecureQueryBuilder:
        """Add a LIMIT clause.

        Parameters
        ----------
        count : int
            Maximum number of rows to return

        Returns
        -------
        SecureQueryBuilder
            Self for method chaining
        """
        if not isinstance(count, int) or count < 0:
            raise SQLInjectionError("LIMIT count must be a non-negative integer")

        self._query_parts.append(f"LIMIT {count}")
        return self

    def build(self) -> tuple[str, list[Any]]:
        """Build the final query and parameter list.

        Returns
        -------
        tuple[str, list[Any]]
            A tuple of (query_string, parameters)
        """
        if not self._query_parts:
            raise SQLInjectionError("Cannot build empty query")

        query = " ".join(self._query_parts)
        return query, self._params


def execute_parameterized_query(
    query: str, params: list[Any] | tuple[Any, ...] | None = None
) -> None:
    """Execute a parameterized query safely.

    This is a placeholder function that demonstrates the pattern for
    executing parameterized queries. In actual usage, this would be
    implemented with a real database connection.

    Parameters
    ----------
    query : str
        The SQL query with '?' placeholders
    params : list or tuple, optional
        Parameters to bind to the query

    Raises
    ------
    SQLInjectionError
        If the query appears to contain unparameterized values

    Warning
    -------
    This is a demonstration function. Real implementations should:
    1. Use database-specific parameterization (e.g., sqlite3.execute)
    2. Handle connection pooling and transactions
    3. Implement proper error handling and logging
    4. Use context managers for resource cleanup

    Example
    -------
    >>> # Safe parameterized query
    >>> execute_parameterized_query(
    ...     "SELECT * FROM nfr_nodes WHERE nu_f > ?",
    ...     [0.5]
    ... )  # doctest: +SKIP

    >>> # UNSAFE: Never do this!
    >>> # execute_parameterized_query(
    >>> #     f"SELECT * FROM nfr_nodes WHERE id = {user_input}"
    >>> # )
    """
    if params is None:
        params = []

    # Basic validation: check for suspicious patterns
    if "'" in query or '"' in query:
        # Allow quoted identifiers if query has standard SQL keywords
        if not any(
            keyword in query.upper() for keyword in ["SELECT", "INSERT", "UPDATE", "DELETE"]
        ):
            raise SQLInjectionError(
                "Query contains quoted strings. Use parameterized queries instead."
            )

    # Verify that the number of placeholders matches the number of parameters
    placeholder_count = query.count("?")
    if placeholder_count != len(params):
        raise SQLInjectionError(
            f"Query has {placeholder_count} placeholders but {len(params)} " f"parameters provided"
        )

    # In a real implementation, this would execute the query
    # For example, with sqlite3:
    # cursor.execute(query, params)


__all__ = (
    "SQLInjectionError",
    "SecureQueryBuilder",
    "execute_parameterized_query",
    "sanitize_string_input",
    "validate_identifier",
)
