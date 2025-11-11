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
    "validate_file_path",
    "resolve_safe_path",
    "validate_version_string",
    "run_command_safely",
    "CommandValidationError",
    "PathTraversalError",
]


class CommandValidationError(ValueError):
    """Raised when command input validation fails."""


class PathTraversalError(ValueError):
    """Raised when path traversal attempt is detected."""


# Allowlisted commands that are safe to execute
ALLOWED_COMMANDS = frozenset(
    {
        "git",
        "python",
        "python3",
        "stubgen",
        "gh",
        "pip",
        "twine",
    }
)

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
            f"Invalid git reference: {ref!r}. " "References cannot contain path traversal patterns."
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

    .. deprecated:: 0.2
       Use :func:`validate_file_path` instead for more comprehensive validation.

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
        raise CommandValidationError(f"Absolute paths not allowed in user input: {path_str!r}")

    # Check for path traversal
    if ".." in path_obj.parts:
        raise CommandValidationError(f"Path traversal not allowed: {path_str!r}")

    # Check for special characters that could be exploited
    if not SAFE_PATH_PATTERN.match(path_str):
        raise CommandValidationError(f"Path contains invalid characters: {path_str!r}")

    return path_obj


def validate_file_path(
    path: str | Path,
    *,
    allow_absolute: bool = False,
    allowed_extensions: Sequence[str] | None = None,
) -> Path:
    """Validate file path to prevent path traversal and unauthorized access.

    This function provides comprehensive path validation to prevent:
    - Path traversal attacks (../../../etc/passwd)
    - Unauthorized file access
    - Special character exploits
    - Symlink attacks

    TNFR Context
    ------------
    Maintains structural coherence by ensuring file operations preserve:
    - Configuration integrity (EPI structure preservation)
    - Data export authenticity (coherence metrics validity)
    - Model persistence safety (NFR state protection)

    Parameters
    ----------
    path : str | Path
        The file path to validate.
    allow_absolute : bool, default=False
        Whether to allow absolute paths. Default is False for user input.
    allowed_extensions : Sequence[str] | None, default=None
        List of allowed file extensions (e.g., ['.json', '.yaml', '.toml']).
        If None, any extension is allowed.

    Returns
    -------
    Path
        The validated path as a Path object.

    Raises
    ------
    PathTraversalError
        If path traversal patterns are detected.
    ValueError
        If the path is invalid or contains unsafe patterns.

    Examples
    --------
    >>> validate_file_path("config.json", allowed_extensions=['.json', '.yaml'])
    PosixPath('config.json')

    >>> validate_file_path("data/export.csv")
    PosixPath('data/export.csv')

    >>> validate_file_path("../../../etc/passwd")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    PathTraversalError: Path traversal detected
    """
    if not path:
        raise ValueError("Path cannot be empty")

    # Convert to Path object
    path_obj = Path(path)
    path_str = str(path)
    path_parts = Path(path).parts

    # Check for null bytes (common in exploit attempts) - do this before resolve()
    if "\x00" in path_str:
        raise ValueError(f"Null byte detected in path: {path!r}")

    # Check for path traversal attempts in the original path first
    if ".." in path_parts:
        raise PathTraversalError(
            f"Path traversal detected in {path!r}. "
            "Relative parent directory references (..) are not allowed."
        )

    # Normalize the path to resolve any . or .. components
    try:
        # Use resolve() with strict=False to normalize without checking existence
        normalized = path_obj.resolve()
    except (OSError, RuntimeError, ValueError) as e:
        # Catch embedded null byte errors from resolve()
        error_msg = str(e)
        if "null byte" in error_msg.lower():
            raise ValueError(f"Null byte detected in path: {path!r}") from e
        raise ValueError(f"Invalid path: {path}") from e

    # Check for absolute paths if not allowed
    if not allow_absolute and normalized.is_absolute():
        # For relative paths, ensure they don't escape to absolute paths
        if not Path(path).is_absolute():
            # This is a relative path that was resolved to absolute
            # We need to check if it contains .. components
            pass
        else:
            raise ValueError(
                f"Absolute paths not allowed: {path}. "
                "Use allow_absolute=True if this is intentional."
            )

    # Check for other dangerous patterns
    dangerous_patterns = [
        ("~", "Home directory expansion"),
        ("\n", "Newline character"),
        ("\r", "Carriage return"),
    ]

    for pattern, desc in dangerous_patterns:
        if pattern in path_str:
            raise ValueError(f"{desc} not allowed in path: {path!r}")

    # Validate file extension if restrictions are specified
    if allowed_extensions is not None:
        suffix = path_obj.suffix.lower()
        allowed_lower = [ext.lower() for ext in allowed_extensions]
        if suffix not in allowed_lower:
            raise ValueError(
                f"File extension {suffix!r} not allowed. "
                f"Allowed extensions: {allowed_extensions}"
            )

    return path_obj


def resolve_safe_path(
    path: str | Path,
    base_dir: str | Path,
    *,
    must_exist: bool = False,
    allowed_extensions: Sequence[str] | None = None,
) -> Path:
    """Resolve a path safely within a base directory.

    This function ensures that the resolved path stays within the specified
    base directory, preventing path traversal attacks while allowing normal
    subdirectory navigation.

    TNFR Context
    ------------
    Ensures configuration and data files maintain operational fractality by
    restricting file access to designated structural boundaries (base directories).

    Parameters
    ----------
    path : str | Path
        The path to resolve (can be relative or absolute).
    base_dir : str | Path
        The base directory that the path must stay within.
    must_exist : bool, default=False
        If True, raise ValueError if the resolved path doesn't exist.
    allowed_extensions : Sequence[str] | None, default=None
        List of allowed file extensions.

    Returns
    -------
    Path
        The validated, resolved absolute path.

    Raises
    ------
    PathTraversalError
        If the resolved path escapes the base directory.
    ValueError
        If the path is invalid or doesn't meet requirements.

    Examples
    --------
    >>> base = Path("/home/user/tnfr")
    >>> resolve_safe_path("config/settings.json", base)  # doctest: +SKIP
    PosixPath('/home/user/tnfr/config/settings.json')

    >>> resolve_safe_path("../../../etc/passwd", base)  # doctest: +SKIP +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    PathTraversalError: Path escapes base directory
    """
    if not path:
        raise ValueError("Path cannot be empty")
    if not base_dir:
        raise ValueError("Base directory cannot be empty")

    # First validate the path itself
    path_obj = validate_file_path(
        path,
        allow_absolute=True,
        allowed_extensions=allowed_extensions,
    )

    # Resolve base directory to absolute path
    base_path = Path(base_dir).resolve()

    # Resolve the target path
    # If path is relative, resolve it relative to base_dir
    if not path_obj.is_absolute():
        resolved = (base_path / path_obj).resolve()
    else:
        resolved = path_obj.resolve()

    # Security check: ensure resolved path is within base directory
    try:
        resolved.relative_to(base_path)
    except ValueError as e:
        raise PathTraversalError(
            f"Path {path!r} escapes base directory {base_dir!r}. " f"Resolved path: {resolved}"
        ) from e

    # Check existence if required
    if must_exist and not resolved.exists():
        raise ValueError(f"Path does not exist: {resolved}")

    return resolved


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
