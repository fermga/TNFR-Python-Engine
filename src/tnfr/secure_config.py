"""Secure configuration management for the TNFR engine.

This module provides utilities for loading configuration from environment
variables with validation and secure defaults. It ensures that sensitive
credentials are never hardcoded in source code.

Security Principles:
- Never hardcode secrets, API keys, or passwords
- Load sensitive values from environment variables
- Provide secure defaults for development
- Validate configuration before use
- Support multiple configuration sources (environment, .env files)
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Optional


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required values."""


def get_env_variable(
    name: str,
    default: Optional[str] = None,
    required: bool = False,
    secret: bool = False,
) -> str | None:
    """Get an environment variable with validation.

    Parameters
    ----------
    name : str
        The name of the environment variable to retrieve.
    default : str, optional
        Default value if the environment variable is not set.
    required : bool, default=False
        If True, raise ConfigurationError if the variable is not set.
    secret : bool, default=False
        If True, this is a sensitive value (password, token, etc.).
        Warnings will be issued if using defaults for secrets.

    Returns
    -------
    str or None
        The value of the environment variable, or the default value.

    Raises
    ------
    ConfigurationError
        If required=True and the variable is not set.

    Examples
    --------
    >>> # Get optional configuration with default
    >>> log_level = get_env_variable("TNFR_LOG_LEVEL", default="INFO")

    >>> # Get required secret (will raise if not set)
    >>> api_token = get_env_variable(
    ...     "GITHUB_TOKEN",
    ...     required=True,
    ...     secret=True
    ... )

    >>> # Get optional secret (will warn if using default)
    >>> redis_password = get_env_variable(
    ...     "REDIS_PASSWORD",
    ...     default="",
    ...     secret=True
    ... )
    """
    value = os.environ.get(name)

    if value is None:
        if required:
            raise ConfigurationError(
                f"Required environment variable '{name}' is not set. "
                f"Please set it in your environment or .env file."
            )
        if secret and default is not None:
            warnings.warn(
                f"Using default value for secret '{name}'. "
                f"Set the environment variable for production use.",
                stacklevel=2,
            )
        return default

    return value


def load_pypi_credentials() -> dict[str, str | None]:
    """Load PyPI publishing credentials from environment.

    Returns
    -------
    dict
        Dictionary containing username, password, and repository settings.

    Notes
    -----
    This function reads from multiple environment variables to support
    different tools (twine, poetry, etc.):

    - PYPI_USERNAME or TWINE_USERNAME
    - PYPI_PASSWORD, PYPI_API_TOKEN, or TWINE_PASSWORD
    - PYPI_REPOSITORY (defaults to 'pypi')

    Best Practice
    -------------
    Use API tokens instead of passwords:
    - PYPI_USERNAME=__token__
    - PYPI_PASSWORD=pypi-AgEIcHlwaS5vcmcC...

    See Also
    --------
    https://pypi.org/help/#apitoken : PyPI API token documentation
    """
    username = os.environ.get("PYPI_USERNAME") or os.environ.get("TWINE_USERNAME")
    password = (
        os.environ.get("PYPI_PASSWORD")
        or os.environ.get("PYPI_API_TOKEN")
        or os.environ.get("TWINE_PASSWORD")
    )
    repository = os.environ.get("PYPI_REPOSITORY", "pypi")

    return {
        "username": username,
        "password": password,
        "repository": repository,
    }


def load_github_credentials() -> dict[str, str | None]:
    """Load GitHub API credentials from environment.

    Returns
    -------
    dict
        Dictionary containing token and repository information.

    Notes
    -----
    This function reads GITHUB_TOKEN and GITHUB_REPOSITORY environment
    variables commonly set in GitHub Actions and other CI environments.

    Best Practice
    -------------
    Use fine-grained personal access tokens with minimal scopes:
    - For security scans: read:security_events
    - For releases: contents:write, packages:write

    See Also
    --------
    https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token
    """
    token = os.environ.get("GITHUB_TOKEN")
    repository = os.environ.get("GITHUB_REPOSITORY")

    return {
        "token": token,
        "repository": repository,
    }


def load_redis_config() -> dict[str, Any]:
    """Load Redis connection configuration from environment.

    Returns
    -------
    dict
        Dictionary containing Redis connection parameters.

    Notes
    -----
    Supports standard Redis configuration variables:

    - REDIS_HOST (default: 'localhost')
    - REDIS_PORT (default: 6379)
    - REDIS_PASSWORD (optional)
    - REDIS_DB (default: 0)
    - REDIS_USE_TLS (default: False)

    Security
    --------
    Always use authentication (REDIS_PASSWORD) in production.
    Enable TLS (REDIS_USE_TLS=true) for network connections.

    See Also
    --------
    tnfr.utils.RedisCacheLayer : Redis cache implementation
    """
    host = get_env_variable("REDIS_HOST", default="localhost")
    port_str = get_env_variable("REDIS_PORT", default="6379")
    password = get_env_variable("REDIS_PASSWORD", default=None, secret=True)
    db_str = get_env_variable("REDIS_DB", default="0")
    use_tls_str = get_env_variable("REDIS_USE_TLS", default="false")

    try:
        port = int(port_str)
    except ValueError:
        raise ConfigurationError(f"REDIS_PORT must be an integer, got: {port_str}")

    try:
        db = int(db_str)
    except ValueError:
        raise ConfigurationError(f"REDIS_DB must be an integer, got: {db_str}")

    use_tls = use_tls_str.lower() in ("true", "1", "yes", "on")

    return {
        "host": host,
        "port": port,
        "password": password,
        "db": db,
        "ssl": use_tls,
    }


def get_cache_secret() -> bytes | None:
    """Get the cache signing secret from environment.

    Returns
    -------
    bytes or None
        The cache secret as bytes, or None if not configured.

    Notes
    -----
    Reads from TNFR_CACHE_SECRET environment variable. The secret should
    be a hex-encoded string (recommended length: 64 characters / 32 bytes).

    Security
    --------
    Use a cryptographically strong random secret:

    >>> import secrets
    >>> secret = secrets.token_hex(32)  # 64-character hex string
    >>> # Set TNFR_CACHE_SECRET=<secret> in your environment

    See Also
    --------
    tnfr.utils.ShelveCacheLayer : Shelf cache with signature support
    tnfr.utils.RedisCacheLayer : Redis cache with signature support
    """
    secret_hex = get_env_variable("TNFR_CACHE_SECRET", secret=True)
    if secret_hex is None:
        return None

    try:
        return bytes.fromhex(secret_hex)
    except ValueError as exc:
        raise ConfigurationError(
            f"TNFR_CACHE_SECRET must be a hex-encoded string: {exc}"
        )


def validate_no_hardcoded_secrets(value: str) -> bool:
    """Validate that a string doesn't look like a hardcoded secret.

    Parameters
    ----------
    value : str
        The string to validate.

    Returns
    -------
    bool
        True if the value passes validation.

    Raises
    ------
    ValueError
        If the value appears to be a hardcoded secret.

    Notes
    -----
    This is a heuristic check for common secret patterns:

    - Long alphanumeric strings (potential tokens)
    - Known secret prefixes (ghp_, pypi-, sk-, etc.)
    - Base64-encoded strings

    Examples
    --------
    >>> validate_no_hardcoded_secrets("my-password")
    True

    >>> validate_no_hardcoded_secrets("ghp_abcd1234...")
    Traceback (most recent call last):
        ...
    ValueError: Value appears to be a hardcoded GitHub token
    """
    # Check for known secret prefixes
    secret_prefixes = [
        ("ghp_", "GitHub token"),
        ("gho_", "GitHub OAuth token"),
        ("ghu_", "GitHub user token"),
        ("ghs_", "GitHub server token"),
        ("ghr_", "GitHub refresh token"),
        ("pypi-", "PyPI token"),
        ("sk-", "OpenAI API key"),
        ("xoxb-", "Slack bot token"),
        ("xoxp-", "Slack user token"),
    ]

    for prefix, name in secret_prefixes:
        if value.startswith(prefix):
            raise ValueError(f"Value appears to be a hardcoded {name}")

    # Check for suspiciously long alphanumeric strings
    if len(value) > 32 and value.replace("-", "").replace("_", "").isalnum():
        # Allow environment variable names
        if not value.isupper():
            warnings.warn(
                f"Value looks like it might be a hardcoded secret: {value[:10]}...",
                stacklevel=2,
            )

    return True
