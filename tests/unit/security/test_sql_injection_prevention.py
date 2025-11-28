"""Tests for SQL injection prevention utilities.

These tests verify that the database security utilities properly prevent
SQL injection attacks and enforce safe query patterns.
"""

from __future__ import annotations

import pytest

from tnfr.security.database import (
    SQLInjectionError,
    SecureQueryBuilder,
    execute_parameterized_query,
    sanitize_string_input,
    validate_identifier,
)


class TestValidateIdentifier:
    """Tests for identifier validation."""

    def test_valid_identifiers(self) -> None:
        """Test that valid identifiers pass validation."""
        assert validate_identifier("nfr_nodes") == "nfr_nodes"
        assert validate_identifier("nu_f") == "nu_f"
        assert validate_identifier("_private_table") == "_private_table"
        assert validate_identifier("Table123") == "Table123"
        assert validate_identifier("epi_measurements") == "epi_measurements"

    def test_invalid_identifiers(self) -> None:
        """Test that invalid identifiers are rejected."""
        # Hyphens not allowed
        with pytest.raises(SQLInjectionError, match="Invalid identifier"):
            validate_identifier("invalid-name")

        # Spaces not allowed
        with pytest.raises(SQLInjectionError, match="Invalid identifier"):
            validate_identifier("invalid name")

        # Special characters not allowed
        with pytest.raises(SQLInjectionError, match="Invalid identifier"):
            validate_identifier("table@name")

        # SQL injection attempts
        with pytest.raises(SQLInjectionError, match="Invalid identifier"):
            validate_identifier("users; DROP TABLE users;--")

        # Empty string
        with pytest.raises(SQLInjectionError, match="cannot be empty"):
            validate_identifier("")

    def test_sql_keywords_rejected(self) -> None:
        """Test that SQL keywords are rejected by default."""
        with pytest.raises(SQLInjectionError, match="is a SQL keyword"):
            validate_identifier("SELECT")

        with pytest.raises(SQLInjectionError, match="is a SQL keyword"):
            validate_identifier("DROP")

        with pytest.raises(SQLInjectionError, match="is a SQL keyword"):
            validate_identifier("delete")  # Case insensitive

    def test_sql_keywords_allowed_with_flag(self) -> None:
        """Test that SQL keywords can be allowed with flag."""
        # This is generally not recommended but supported
        assert validate_identifier("SELECT", allow_keywords=True) == "SELECT"
        assert validate_identifier("drop", allow_keywords=True) == "drop"

    def test_too_long_identifier(self) -> None:
        """Test that overly long identifiers are rejected."""
        long_name = "a" * 65  # 65 characters (max is 64)
        with pytest.raises(SQLInjectionError, match="Invalid identifier"):
            validate_identifier(long_name)

    def test_non_string_identifier(self) -> None:
        """Test that non-string identifiers are rejected."""
        with pytest.raises(SQLInjectionError, match="must be a string"):
            validate_identifier(123)  # type: ignore

        with pytest.raises(SQLInjectionError, match="must be a string"):
            validate_identifier(None)  # type: ignore


class TestSanitizeStringInput:
    """Tests for string input sanitization."""

    def test_valid_strings(self) -> None:
        """Test that valid strings pass sanitization."""
        assert sanitize_string_input("valid string") == "valid string"
        assert sanitize_string_input("") == ""
        assert sanitize_string_input("Special: #@!") == "Special: #@!"

    def test_length_limit(self) -> None:
        """Test that length limits are enforced."""
        # Within limit
        short_string = "a" * 100
        assert sanitize_string_input(short_string, max_length=100) == short_string

        # Exceeds limit
        long_string = "a" * 1001
        with pytest.raises(SQLInjectionError, match="exceeds maximum length"):
            sanitize_string_input(long_string, max_length=1000)

    def test_null_bytes_rejected(self) -> None:
        """Test that null bytes are rejected."""
        with pytest.raises(SQLInjectionError, match="null bytes"):
            sanitize_string_input("string\x00with null")

    def test_non_string_rejected(self) -> None:
        """Test that non-strings are rejected."""
        with pytest.raises(SQLInjectionError, match="must be a string"):
            sanitize_string_input(123)  # type: ignore

        with pytest.raises(SQLInjectionError, match="must be a string"):
            sanitize_string_input(None)  # type: ignore


class TestSecureQueryBuilder:
    """Tests for the secure query builder."""

    def test_simple_select(self) -> None:
        """Test building a simple SELECT query."""
        builder = SecureQueryBuilder()
        query, params = builder.select("nfr_nodes", ["id", "nu_f"]).build()

        assert query == "SELECT id, nu_f FROM nfr_nodes"
        assert params == []

    def test_select_all_columns(self) -> None:
        """Test SELECT with all columns (*)."""
        builder = SecureQueryBuilder()
        query, params = builder.select("nfr_nodes").build()

        assert query == "SELECT * FROM nfr_nodes"
        assert params == []

    def test_select_with_where(self) -> None:
        """Test SELECT with WHERE clause."""
        builder = SecureQueryBuilder()
        query, params = builder.select("nfr_nodes", ["id", "nu_f"]).where("nu_f > ?", 0.5).build()

        assert query == "SELECT id, nu_f FROM nfr_nodes WHERE nu_f > ?"
        assert params == [0.5]

    def test_select_with_multiple_where(self) -> None:
        """Test SELECT with multiple WHERE conditions."""
        builder = SecureQueryBuilder()
        query, params = (
            builder.select("nfr_nodes", ["id"])
            .where("nu_f > ?", 0.5)
            .where("phase < ?", 3.14)
            .build()
        )

        assert query == "SELECT id FROM nfr_nodes WHERE nu_f > ? AND phase < ?"
        assert params == [0.5, 3.14]

    def test_select_with_order_by(self) -> None:
        """Test SELECT with ORDER BY clause."""
        builder = SecureQueryBuilder()
        query, params = builder.select("nfr_nodes", ["id", "nu_f"]).order_by("nu_f", "DESC").build()

        assert query == "SELECT id, nu_f FROM nfr_nodes ORDER BY nu_f DESC"
        assert params == []

    def test_select_with_limit(self) -> None:
        """Test SELECT with LIMIT clause."""
        builder = SecureQueryBuilder()
        query, params = builder.select("nfr_nodes", ["id"]).limit(10).build()

        assert query == "SELECT id FROM nfr_nodes LIMIT 10"
        assert params == []

    def test_complex_select(self) -> None:
        """Test a complex SELECT query with multiple clauses."""
        builder = SecureQueryBuilder()
        query, params = (
            builder.select("nfr_nodes", ["id", "nu_f", "phase"])
            .where("nu_f > ?", 0.5)
            .where("phase BETWEEN ? AND ?", 0.0, 3.14)
            .order_by("nu_f", "DESC")
            .limit(10)
            .build()
        )

        expected = (
            "SELECT id, nu_f, phase FROM nfr_nodes "
            "WHERE nu_f > ? AND phase BETWEEN ? AND ? "
            "ORDER BY nu_f DESC LIMIT 10"
        )
        assert query == expected
        assert params == [0.5, 0.0, 3.14]

    def test_insert_query(self) -> None:
        """Test building an INSERT query."""
        builder = SecureQueryBuilder()
        query, params = builder.insert("nfr_nodes", ["id", "nu_f", "phase"]).build()

        assert query == "INSERT INTO nfr_nodes (id, nu_f, phase) VALUES (?, ?, ?)"
        assert params == []

    def test_update_query(self) -> None:
        """Test building an UPDATE query."""
        builder = SecureQueryBuilder()
        query, params = (
            builder.update("nfr_nodes").set(nu_f=0.8, phase=1.57).where("id = ?", 123).build()
        )

        assert query == "UPDATE nfr_nodes SET nu_f = ?, phase = ? WHERE id = ?"
        assert params == [0.8, 1.57, 123]

    def test_delete_query(self) -> None:
        """Test building a DELETE query."""
        builder = SecureQueryBuilder()
        query, params = builder.delete("nfr_nodes").where("nu_f < ?", 0.1).build()

        assert query == "DELETE FROM nfr_nodes WHERE nu_f < ?"
        assert params == [0.1]

    def test_invalid_table_name(self) -> None:
        """Test that invalid table names are rejected."""
        builder = SecureQueryBuilder()
        with pytest.raises(SQLInjectionError):
            builder.select("invalid-table").build()

    def test_invalid_column_name(self) -> None:
        """Test that invalid column names are rejected."""
        builder = SecureQueryBuilder()
        with pytest.raises(SQLInjectionError):
            builder.select("nfr_nodes", ["valid", "in-valid"]).build()

    def test_where_param_mismatch(self) -> None:
        """Test that WHERE parameter count is validated."""
        builder = SecureQueryBuilder()
        with pytest.raises(SQLInjectionError, match="expects 2 parameters"):
            builder.select("nfr_nodes").where("a = ? AND b = ?", 1).build()

    def test_where_suspicious_patterns(self) -> None:
        """Test that suspicious WHERE patterns are rejected."""
        builder = SecureQueryBuilder()

        # Semicolon (SQL command separator)
        with pytest.raises(SQLInjectionError, match="suspicious"):
            builder.select("nfr_nodes").where("id = 1; DROP TABLE users", 1).build()

        # SQL comment
        with pytest.raises(SQLInjectionError, match="suspicious"):
            builder.select("nfr_nodes").where("id = ? --", 1).build()

        # Block comment
        with pytest.raises(SQLInjectionError, match="suspicious"):
            builder.select("nfr_nodes").where("id = ? /* comment */", 1).build()

    def test_invalid_order_direction(self) -> None:
        """Test that invalid ORDER BY directions are rejected."""
        builder = SecureQueryBuilder()
        with pytest.raises(SQLInjectionError, match="Invalid sort direction"):
            builder.select("nfr_nodes").order_by("id", "INVALID").build()

    def test_invalid_limit(self) -> None:
        """Test that invalid LIMIT values are rejected."""
        builder = SecureQueryBuilder()

        # Negative limit
        with pytest.raises(SQLInjectionError, match="non-negative integer"):
            builder.select("nfr_nodes").limit(-1).build()

        # Non-integer limit
        with pytest.raises(SQLInjectionError, match="non-negative integer"):
            builder.select("nfr_nodes").limit(10.5).build()  # type: ignore

    def test_set_without_update(self) -> None:
        """Test that SET can only be used with UPDATE."""
        builder = SecureQueryBuilder()
        builder.select("nfr_nodes")
        with pytest.raises(SQLInjectionError, match="can only be used with UPDATE"):
            builder.set(nu_f=0.5).build()

    def test_empty_query(self) -> None:
        """Test that empty queries are rejected."""
        builder = SecureQueryBuilder()
        with pytest.raises(SQLInjectionError, match="Cannot build empty query"):
            builder.build()


class TestExecuteParameterizedQuery:
    """Tests for parameterized query execution."""

    def test_valid_query(self) -> None:
        """Test that valid parameterized queries are accepted."""
        # Should not raise
        execute_parameterized_query("SELECT * FROM nfr_nodes WHERE nu_f > ?", [0.5])

    def test_param_count_validation(self) -> None:
        """Test that parameter count is validated."""
        with pytest.raises(SQLInjectionError, match="placeholders"):
            execute_parameterized_query(
                "SELECT * FROM nfr_nodes WHERE nu_f > ? AND phase < ?",
                [0.5],  # Missing one parameter
            )

    def test_no_params_needed(self) -> None:
        """Test queries with no parameters."""
        # Should not raise
        execute_parameterized_query("SELECT * FROM nfr_nodes")

    def test_suspicious_quoted_strings(self) -> None:
        """Test detection of potentially unsafe queries."""
        # Note: This is a basic check, real implementations should be more sophisticated
        # Queries with SELECT/INSERT/UPDATE/DELETE are allowed to have quotes
        execute_parameterized_query("SELECT name FROM users WHERE name = ?", ["O'Brien"])


class TestSecurityIntegration:
    """Integration tests for security utilities."""

    def test_complete_workflow(self) -> None:
        """Test a complete workflow with validation and query building."""
        # Validate inputs
        table = validate_identifier("nfr_nodes")
        nu_f_threshold = 0.5

        # Sanitize string input (if any)
        node_name = sanitize_string_input("test_node")

        # Build secure query
        builder = SecureQueryBuilder()
        query, params = (
            builder.select(table, ["id", "nu_f", "phase"])
            .where("nu_f > ?", nu_f_threshold)
            .where("name = ?", node_name)
            .order_by("nu_f", "DESC")
            .limit(10)
            .build()
        )

        # Verify query structure
        assert "SELECT" in query
        assert "FROM nfr_nodes" in query
        assert "WHERE" in query
        assert len(params) == 2
        assert params[0] == nu_f_threshold
        assert params[1] == node_name

    def test_tnfr_specific_queries(self) -> None:
        """Test queries specific to TNFR data structures."""
        # Query for high-frequency nodes
        builder = SecureQueryBuilder()
        query, params = (
            builder.select("nfr_nodes", ["id", "epi", "nu_f"])
            .where("nu_f > ?", 0.7)
            .order_by("nu_f", "DESC")
            .build()
        )
        assert "nu_f" in query
        assert params == [0.7]

        # Query for phase-synchronized nodes
        builder = SecureQueryBuilder()
        query, params = (
            builder.select("nfr_nodes", ["id", "phase"])
            .where("phase BETWEEN ? AND ?", 0.0, 1.57)
            .build()
        )
        assert "phase BETWEEN" in query
        assert params == [0.0, 1.57]

        # Update node structural frequency
        builder = SecureQueryBuilder()
        query, params = builder.update("nfr_nodes").set(nu_f=0.8).where("id = ?", 123).build()
        assert "UPDATE nfr_nodes" in query
        assert "SET nu_f = ?" in query
        assert params == [0.8, 123]
