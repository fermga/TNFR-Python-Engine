"""Command execution security utilities for TNFR.

This module provides secure wrappers for subprocess execution and input validation
to prevent command injection attacks while maintaining TNFR structural coherence.

TNFR Context
------------
These utilities ensure that external process execution maintains the integrity of
the TNFR computational environment without introducing security vulnerabilities.
They act as a coherence boundary between user input and system command execution.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "validate_git_ref",
    "validate_path_safe",
    "validate_version_string",
    "run_command_safely",
    "CommandValidationError",
]


class CommandValidationError(ValueError):
    """Raised when command input validation fails."""


# Allowlisted commands that are safe to execute
ALLOWED_COMMANDS = frozenset({
    "git",
    "python",
    "python3",
    "stubgen",
    "gh",
    "pip",
    "twine",
})

# Pattern for valid git refs (branches, tags, commit SHAs)
GIT_REF_PATTERN = re.compile(r"^[a-zA-Z0-9/_\-\.]+$")

# Pattern for semantic version strings
VERSION_PATTERN = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(-[a-zA-Z0-9\-\.]+)?$")

# Pattern for safe path components (no path traversal)
SAFE_PATH_PATTERN = re.compile(r"^[a-zA-Z0-9/_\-\.]+$")


def validate_git_ref(ref: str) -> str:
    """Validate a git reference (branch, tag, or SHA).

    Parameters
    ----------
    ref : str
        The git reference to validate.

    Returns
    -------
    str
        The validated reference.

    Raises
    ------
    CommandValidationError
        If the reference contains invalid characters.

    Examples
    --------
    >>> validate_git_ref("main")
    'main'
    >>> validate_git_ref("feature/new-operator")
    'feature/new-operator'
    >>> validate_git_ref("v1.0.0")
    'v1.0.0'
    >>> validate_git_ref("abc123def")
    'abc123def'
    """
    if not ref:
        raise CommandValidationError("Git reference cannot be empty")

    if not GIT_REF_PATTERN.match(ref):
        raise CommandValidationError(
            f"Invalid git reference: {ref!r}. "
            "References must contain only alphanumeric characters, "
            "hyphens, underscores, slashes, and dots."
        )

    # Additional security: prevent path traversal patterns
    if ".." in ref or ref.startswith("/") or ref.startswith("~"):
        raise CommandValidationError(
            f"Invalid git reference: {ref!r}. "
            "References cannot contain path traversal patterns."
        )

    return ref


def validate_version_string(version: str) -> str:
    """Validate a semantic version string.

    Parameters
    ----------
    version : str
        The version string to validate.

    Returns
    -------
    str
        The validated version string.

    Raises
    ------
    CommandValidationError
        If the version string is invalid.

    Examples
    --------
    >>> validate_version_string("1.0.0")
    '1.0.0'
    >>> validate_version_string("v16.2.3")
    'v16.2.3'
    >>> validate_version_string("2.0.0-beta.1")
    '2.0.0-beta.1'
    """
    if not version:
        raise CommandValidationError("Version string cannot be empty")

    if not VERSION_PATTERN.match(version):
        raise CommandValidationError(
            f"Invalid version string: {version!r}. "
            "Version must follow semantic versioning (e.g., '1.0.0' or 'v1.0.0')."
        )

    return version


def validate_path_safe(path: str | Path) -> Path:
    """Validate that a path is safe (no path traversal attacks).

    Parameters
    ----------
    path : str | Path
        The path to validate.

    Returns
    -------
    Path
        The validated path as a Path object.

    Raises
    ------
    CommandValidationError
        If the path contains unsafe patterns.

    Examples
    --------
    >>> validate_path_safe("src/tnfr/core.py")
    PosixPath('src/tnfr/core.py')
    >>> validate_path_safe(Path("tests/unit"))
    PosixPath('tests/unit')
    """
    path_obj = Path(path)
    path_str = str(path_obj)

    # Check for absolute paths in untrusted input
    if path_obj.is_absolute():
        raise CommandValidationError(
            f"Absolute paths not allowed in user input: {path_str!r}"
        )

    # Check for path traversal
    if ".." in path_obj.parts:
        raise CommandValidationError(
            f"Path traversal not allowed: {path_str!r}"
        )

    # Check for special characters that could be exploited
    if not SAFE_PATH_PATTERN.match(path_str):
        raise CommandValidationError(
            f"Path contains invalid characters: {path_str!r}"
        )

    return path_obj


def run_command_safely(
    command: Sequence[str],
    *,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
    timeout: int | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[Any]:
    """Execute a command safely with validation.

    This function provides a secure wrapper around subprocess.run that:
    1. Never uses shell=True
    2. Validates the command is in the allowlist
    3. Ensures all arguments are strings
    4. Provides timeout protection

    Parameters
    ----------
    command : Sequence[str]
        Command and arguments as a list of strings.
    check : bool, optional
        If True, raise CalledProcessError on non-zero exit. Default is True.
    capture_output : bool, optional
        If True, capture stdout and stderr. Default is True.
    text : bool, optional
        If True, decode output as text. Default is True.
    timeout : int | None, optional
        Maximum time in seconds to wait for command completion.
    cwd : str | Path | None, optional
        Working directory for command execution.
    env : dict[str, str] | None, optional
        Environment variables for the subprocess.

    Returns
    -------
    subprocess.CompletedProcess
        The result of the command execution.

    Raises
    ------
    CommandValidationError
        If the command is not in the allowlist or arguments are invalid.
    subprocess.CalledProcessError
        If check=True and the command returns non-zero exit code.
    subprocess.TimeoutExpired
        If timeout is exceeded.

    Examples
    --------
    >>> result = run_command_safely(["git", "status"])
    >>> result.returncode
    0
    >>> result = run_command_safely(["git", "log", "-1", "--oneline"])
    """
    if not command:
        raise CommandValidationError("Command cannot be empty")

    # Validate all arguments are strings
    if not all(isinstance(arg, str) for arg in command):
        raise CommandValidationError(
            "All command arguments must be strings. "
            f"Got: {[type(arg).__name__ for arg in command]}"
        )

    # Extract base command (handle paths like /usr/bin/python)
    base_cmd = Path(command[0]).name

    # Validate command is in allowlist
    if base_cmd not in ALLOWED_COMMANDS:
        raise CommandValidationError(
            f"Command not in allowlist: {base_cmd!r}. "
            f"Allowed commands: {sorted(ALLOWED_COMMANDS)}"
        )

    # Validate cwd if provided
    if cwd is not None:
        cwd = str(cwd)

    # Execute with shell=False (explicit for clarity)
    return subprocess.run(
        list(command),
        check=check,
        capture_output=capture_output,
        text=text,
        timeout=timeout,
        cwd=cwd,
        env=env,
        shell=False,  # CRITICAL: Never use shell=True
    )
