"""Unit tests for TNFR security utilities.

These tests validate input sanitization and subprocess execution security
to prevent command injection vulnerabilities.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tnfr.security.subprocess import (
    CommandValidationError,
    run_command_safely,
    validate_git_ref,
    validate_path_safe,
    validate_version_string,
)


class TestValidateGitRef:
    """Tests for git reference validation."""

    def test_valid_branch_names(self) -> None:
        """Test that valid branch names pass validation."""
        assert validate_git_ref("main") == "main"
        assert validate_git_ref("feature/new-operator") == "feature/new-operator"
        assert validate_git_ref("develop") == "develop"
        assert validate_git_ref("hotfix-123") == "hotfix-123"
        assert validate_git_ref("user_branch") == "user_branch"

    def test_valid_tag_names(self) -> None:
        """Test that valid tag names pass validation."""
        assert validate_git_ref("v1.0.0") == "v1.0.0"
        assert validate_git_ref("v16.2.3") == "v16.2.3"
        assert validate_git_ref("release-1.0") == "release-1.0"

    def test_valid_commit_sha(self) -> None:
        """Test that commit SHAs pass validation."""
        assert validate_git_ref("abc123def") == "abc123def"
        assert validate_git_ref("1234567890abcdef") == "1234567890abcdef"

    def test_rejects_path_traversal(self) -> None:
        """Test that path traversal patterns are rejected."""
        with pytest.raises(CommandValidationError, match="path traversal"):
            validate_git_ref("../etc/passwd")

        with pytest.raises(CommandValidationError, match="path traversal"):
            validate_git_ref("foo/../bar")

    def test_rejects_absolute_paths(self) -> None:
        """Test that absolute paths are rejected."""
        with pytest.raises(CommandValidationError, match="path traversal"):
            validate_git_ref("/etc/passwd")

        with pytest.raises(CommandValidationError, match="Invalid git reference"):
            validate_git_ref("~/secrets")

    def test_rejects_invalid_characters(self) -> None:
        """Test that invalid characters are rejected."""
        with pytest.raises(CommandValidationError, match="Invalid git reference"):
            validate_git_ref("branch; rm -rf /")

        with pytest.raises(CommandValidationError, match="Invalid git reference"):
            validate_git_ref("branch$malicious")

        with pytest.raises(CommandValidationError, match="Invalid git reference"):
            validate_git_ref("branch`whoami`")

    def test_rejects_empty_ref(self) -> None:
        """Test that empty references are rejected."""
        with pytest.raises(CommandValidationError, match="cannot be empty"):
            validate_git_ref("")


class TestValidateVersionString:
    """Tests for version string validation."""

    def test_valid_versions(self) -> None:
        """Test that valid semantic versions pass validation."""
        assert validate_version_string("1.0.0") == "1.0.0"
        assert validate_version_string("v16.2.3") == "v16.2.3"
        assert validate_version_string("2.0.0-beta.1") == "2.0.0-beta.1"
        assert validate_version_string("3.1.4-alpha") == "3.1.4-alpha"

    def test_rejects_invalid_versions(self) -> None:
        """Test that invalid version strings are rejected."""
        with pytest.raises(CommandValidationError, match="Invalid version"):
            validate_version_string("1.0")

        with pytest.raises(CommandValidationError, match="Invalid version"):
            validate_version_string("not-a-version")

        with pytest.raises(CommandValidationError, match="Invalid version"):
            validate_version_string("1.0.0; rm -rf /")

    def test_rejects_empty_version(self) -> None:
        """Test that empty version strings are rejected."""
        with pytest.raises(CommandValidationError, match="cannot be empty"):
            validate_version_string("")


class TestValidatePathSafe:
    """Tests for path safety validation."""

    def test_valid_relative_paths(self) -> None:
        """Test that valid relative paths pass validation."""
        assert validate_path_safe("src/tnfr/core.py") == Path("src/tnfr/core.py")
        assert validate_path_safe("tests/unit") == Path("tests/unit")
        assert validate_path_safe("README.md") == Path("README.md")

    def test_accepts_path_objects(self) -> None:
        """Test that Path objects are accepted."""
        assert validate_path_safe(Path("src/tnfr")) == Path("src/tnfr")

    def test_rejects_absolute_paths(self) -> None:
        """Test that absolute paths are rejected."""
        with pytest.raises(CommandValidationError, match="Absolute paths not allowed"):
            validate_path_safe("/etc/passwd")

        with pytest.raises(CommandValidationError, match="Absolute paths not allowed"):
            validate_path_safe("/tmp/exploit")

    def test_rejects_path_traversal(self) -> None:
        """Test that path traversal is rejected."""
        with pytest.raises(CommandValidationError, match="Path traversal not allowed"):
            validate_path_safe("../etc/passwd")

        with pytest.raises(CommandValidationError, match="Path traversal not allowed"):
            validate_path_safe("foo/../../../etc/passwd")

    def test_rejects_invalid_characters(self) -> None:
        """Test that paths with invalid characters are rejected."""
        with pytest.raises(CommandValidationError, match="invalid characters"):
            validate_path_safe("path; rm -rf /")

        with pytest.raises(CommandValidationError, match="invalid characters"):
            validate_path_safe("path$var")


class TestRunCommandSafely:
    """Tests for safe command execution."""

    def test_executes_valid_git_command(self) -> None:
        """Test that valid git commands execute successfully."""
        result = run_command_safely(["git", "--version"], check=True)
        assert result.returncode == 0
        assert "git version" in result.stdout.lower()

    def test_executes_python_command(self) -> None:
        """Test that Python commands execute successfully."""
        result = run_command_safely(
            [sys.executable, "-c", "print('hello')"], check=True
        )
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_rejects_non_allowlisted_command(self) -> None:
        """Test that non-allowlisted commands are rejected."""
        with pytest.raises(CommandValidationError, match="not in allowlist"):
            run_command_safely(["curl", "http://example.com"], check=True)

        with pytest.raises(CommandValidationError, match="not in allowlist"):
            run_command_safely(["wget", "http://example.com"], check=True)

    def test_rejects_empty_command(self) -> None:
        """Test that empty commands are rejected."""
        with pytest.raises(CommandValidationError, match="cannot be empty"):
            run_command_safely([], check=True)

    def test_rejects_non_string_arguments(self) -> None:
        """Test that non-string arguments are rejected."""
        with pytest.raises(CommandValidationError, match="must be strings"):
            run_command_safely(["git", 123], check=True)  # type: ignore[list-item]

    def test_shell_is_always_false(self) -> None:
        """Test that commands never use shell=True."""
        # This test verifies the function signature and implementation
        # don't allow shell=True by checking that command injection doesn't work
        result = run_command_safely(["git", "--version; echo injected"], check=False)
        # If shell were True, "injected" would appear in output
        # With shell=False, git treats the entire string as one argument and fails
        assert result.returncode != 0
        assert "injected" not in result.stdout

    def test_timeout_protection(self) -> None:
        """Test that timeout protection works."""
        with pytest.raises(subprocess.TimeoutExpired):
            run_command_safely(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                timeout=1,
                check=False,
            )

    def test_check_parameter_raises_on_failure(self) -> None:
        """Test that check=True raises on non-zero exit."""
        with pytest.raises(subprocess.CalledProcessError):
            run_command_safely(["git", "invalid-subcommand"], check=True)

    def test_check_false_returns_error_code(self) -> None:
        """Test that check=False returns error code without raising."""
        result = run_command_safely(["git", "invalid-subcommand"], check=False)
        assert result.returncode != 0

    def test_captures_output(self) -> None:
        """Test that output is captured correctly."""
        result = run_command_safely(
            [sys.executable, "-c", "print('test output')"], check=True
        )
        assert "test output" in result.stdout

    def test_respects_cwd_parameter(self, tmp_path: Path) -> None:
        """Test that cwd parameter is respected."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = run_command_safely(["git", "init"], cwd=str(tmp_path), check=True)
        assert result.returncode == 0
        assert (tmp_path / ".git").exists()


class TestSecurityIntegration:
    """Integration tests for security features."""

    def test_prevents_command_injection_via_arguments(self) -> None:
        """Test that command injection via arguments is prevented."""
        # Attempt injection via git ref
        with pytest.raises(CommandValidationError):
            ref = "main; rm -rf /"
            validate_git_ref(ref)

    def test_prevents_path_traversal_in_file_operations(self) -> None:
        """Test that path traversal in file operations is prevented."""
        with pytest.raises(CommandValidationError):
            path = "../../../etc/passwd"
            validate_path_safe(path)

    def test_command_allowlist_is_restrictive(self) -> None:
        """Test that only known-safe commands are allowed."""
        # These should be rejected
        dangerous_commands = ["rm", "curl", "wget", "nc", "telnet", "ssh"]
        for cmd in dangerous_commands:
            with pytest.raises(CommandValidationError, match="not in allowlist"):
                run_command_safely([cmd, "--help"], check=False)
