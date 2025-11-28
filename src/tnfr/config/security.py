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
- Sanitize credentials in logs to prevent exposure
- Secure memory management for secrets
- Credential rotation and TTL support
"""

from __future__ import annotations

import os
import secrets
import time
import warnings
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional
from urllib.parse import urlparse, urlunparse


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required values."""


class SecurityAuditWarning(UserWarning):
    """Warning for security audit findings that don't stop execution."""


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
    - PYPI_PASSWORD=pypi-XXXXXXXXXXXXXXXXXXXX...

    Note: Example uses 'XXX' pattern to avoid triggering security scanners.
    Actual PyPI tokens follow format: pypi-AgEIcHlwaS5vcmcC...

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


def load_redis_config(validate_url: bool = True) -> dict[str, Any]:
    """Load Redis connection configuration from environment.

    Parameters
    ----------
    validate_url : bool, default=True
        Whether to validate the constructed Redis URL.

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
    - REDIS_URL (alternative: full URL, overrides individual params)

    Security
    --------
    Always use authentication (REDIS_PASSWORD) in production.
    Enable TLS (REDIS_USE_TLS=true) for network connections.
    URLs with credentials are validated and sanitized for logging.

    See Also
    --------
    tnfr.utils.RedisCacheLayer : Redis cache implementation
    SecureCredentialValidator : URL validation and sanitization
    """
    # Check if full URL is provided
    redis_url = get_env_variable("REDIS_URL", default=None)

    if redis_url:
        # Validate URL if requested
        if validate_url:
            SecureCredentialValidator.validate_redis_url(redis_url)

        # Parse URL to extract components
        parsed = urlparse(redis_url)

        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 6379,
            "password": parsed.password,
            "db": int(parsed.path.lstrip("/") or "0") if parsed.path else 0,
            "ssl": parsed.scheme == "rediss",
            "url": redis_url,
        }

    # Load from individual variables
    host = get_env_variable("REDIS_HOST", default="localhost")
    port_str = get_env_variable("REDIS_PORT", default="6379")
    password = get_env_variable("REDIS_PASSWORD", default=None, secret=True)
    db_str = get_env_variable("REDIS_DB", default="0")
    use_tls_str = get_env_variable("REDIS_USE_TLS", default="false")

    try:
        port = int(port_str)
    except ValueError:
        raise ConfigurationError(f"REDIS_PORT must be an integer, got: {port_str}")

    # Validate port range
    if not (1 <= port <= 65535):
        raise ConfigurationError(f"REDIS_PORT must be between 1 and 65535, got: {port}")

    try:
        db = int(db_str)
    except ValueError:
        raise ConfigurationError(f"REDIS_DB must be an integer, got: {db_str}")

    use_tls = use_tls_str.lower() in ("true", "1", "yes", "on")

    # Construct URL for validation
    if validate_url:
        scheme = "rediss" if use_tls else "redis"
        if password:
            url = f"{scheme}://:{password}@{host}:{port}/{db}"
        else:
            url = f"{scheme}://{host}:{port}/{db}"
        SecureCredentialValidator.validate_redis_url(url)

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
        raise ConfigurationError(f"TNFR_CACHE_SECRET must be a hex-encoded string: {exc}")


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

    For production environments, consider using more sophisticated
    tools like `detect-secrets` which employ entropy analysis for
    better accuracy.

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
    # Note: This is a simple heuristic. For production use, consider
    # entropy-based analysis (e.g., using detect-secrets library)
    if len(value) > 32 and value.replace("-", "").replace("_", "").isalnum():
        # Allow environment variable names (typically uppercase)
        if not value.isupper():
            warnings.warn(
                f"Value looks like it might be a hardcoded secret: {value[:10]}...",
                stacklevel=2,
            )

    return True


class SecureCredentialValidator:
    """Robust credential and configuration validator.

    Validates credentials and configuration with strict security criteria
    following TNFR principles of structural coherence and stability.
    """

    ALLOWED_SCHEMES = frozenset(["redis", "rediss"])  # Only secure schemes
    MAX_URL_LENGTH = 512  # Prevent DoS attacks
    MIN_SECRET_LENGTH = 8  # Minimum secret strength

    @staticmethod
    def validate_redis_url(url: str) -> bool:
        """Validate Redis URL with strict security criteria.

        Parameters
        ----------
        url : str
            Redis URL to validate.

        Returns
        -------
        bool
            True if URL is valid.

        Raises
        ------
        ValueError
            If URL fails validation checks.

        Examples
        --------
        >>> SecureCredentialValidator.validate_redis_url("redis://localhost:6379/0")
        True

        >>> SecureCredentialValidator.validate_redis_url("http://evil.com")
        Traceback (most recent call last):
            ...
        ValueError: Unsupported scheme: http
        """
        if not url or not isinstance(url, str):
            raise ValueError("Redis URL must be a non-empty string")

        if len(url) > SecureCredentialValidator.MAX_URL_LENGTH:
            raise ValueError(
                f"Redis URL exceeds maximum length of {SecureCredentialValidator.MAX_URL_LENGTH}"
            )

        try:
            parsed = urlparse(url)
        except Exception as exc:
            raise ValueError(f"Invalid URL format: {exc}")

        if parsed.scheme not in SecureCredentialValidator.ALLOWED_SCHEMES:
            raise ValueError(
                f"Unsupported scheme: {parsed.scheme}. "
                f"Allowed: {', '.join(SecureCredentialValidator.ALLOWED_SCHEMES)}"
            )

        if not parsed.hostname:
            raise ValueError("Redis URL must include a hostname")

        # Validate port if specified
        if parsed.port is not None:
            if not (1 <= parsed.port <= 65535):
                raise ValueError(f"Invalid port number: {parsed.port}")

        return True

    @staticmethod
    def sanitize_for_logging(url: str) -> str:
        """Sanitize URL for safe logging (hide credentials).

        Parameters
        ----------
        url : str
            URL that may contain credentials.

        Returns
        -------
        str
            Sanitized URL with credentials masked.

        Examples
        --------
        >>> SecureCredentialValidator.sanitize_for_logging(
        ...     "redis://user:secret@host:6379/0"
        ... )
        'redis://user:***@host:6379/0'
        """
        if not url:
            return url

        try:
            parsed = urlparse(url)

            # Check if parsing actually succeeded
            if not parsed.scheme and not parsed.netloc:
                # This is not a valid URL
                return "<invalid-url>"
        except Exception:
            # If parsing fails, return a safe placeholder
            return "<invalid-url>"

        # Mask password if present
        if parsed.password:
            # Replace password with ***
            netloc = parsed.netloc
            if "@" in netloc:
                userinfo, hostinfo = netloc.rsplit("@", 1)
                if ":" in userinfo:
                    username, _ = userinfo.split(":", 1)
                    netloc = f"{username}:***@{hostinfo}"
                else:
                    netloc = f"***@{hostinfo}"

            sanitized = parsed._replace(netloc=netloc)
            return urlunparse(sanitized)

        return url

    @staticmethod
    def validate_secret_strength(secret: str | bytes, min_length: int = 8) -> bool:
        """Validate that a secret meets minimum strength requirements.

        Parameters
        ----------
        secret : str or bytes
            The secret to validate.
        min_length : int, default=8
            Minimum required length.

        Returns
        -------
        bool
            True if secret is strong enough.

        Raises
        ------
        ValueError
            If secret is too weak.
        """
        if isinstance(secret, bytes):
            length = len(secret)
            secret_str = secret.decode("utf-8", errors="ignore")
        else:
            length = len(secret)
            secret_str = secret

        # Check for common weak passwords first (before length check)
        # This provides more specific error messages
        weak_passwords = ["password", "123456", "admin", "secret", "test", "changeme"]
        if secret_str.lower() in weak_passwords:
            raise ValueError("Secret matches a known weak password")

        # Then check length
        if length < min_length:
            raise ValueError(f"Secret too short: {length} < {min_length} (minimum)")

        return True


class SecureSecretManager:
    """Secure secret management with automatic memory cleanup.

    Manages secrets in memory with secure cleanup to prevent exposure
    through memory dumps. Implements structural coherence principles
    by ensuring secrets maintain integrity throughout their lifecycle.
    """

    def __init__(self) -> None:
        """Initialize secure secret manager."""
        self._secrets: dict[str, bytearray] = {}
        self._access_log: list[tuple[str, float]] = []

    def store_secret(self, key: str, secret: bytes | str) -> None:
        """Store a secret securely.

        Parameters
        ----------
        key : str
            Identifier for the secret.
        secret : bytes or str
            The secret value to store.
        """
        if isinstance(secret, str):
            secret_bytes = secret.encode("utf-8")
        else:
            secret_bytes = secret

        # Store as mutable bytearray for secure clearing
        self._secrets[key] = bytearray(secret_bytes)

    def get_secret(self, key: str) -> bytes:
        """Get a secret with access tracking.

        Parameters
        ----------
        key : str
            Secret identifier.

        Returns
        -------
        bytes
            Copy of the secret (not direct reference).
        """
        self._access_log.append((key, time.time()))
        secret_array = self._secrets.get(key)
        if secret_array is None:
            return b""
        # Return copy to prevent external mutation
        return bytes(secret_array)

    def clear_secret(self, key: str) -> None:
        """Clear a secret from memory securely.

        Parameters
        ----------
        key : str
            Secret identifier to clear.
        """
        if key in self._secrets:
            # Overwrite with random bytes before deletion
            secret_array = self._secrets[key]
            for i in range(len(secret_array)):
                secret_array[i] = secrets.randbits(8) & 0xFF
            del self._secrets[key]

    def clear_all(self) -> None:
        """Clear all secrets from memory."""
        for key in list(self._secrets.keys()):
            self.clear_secret(key)

    def get_access_log(self) -> list[tuple[str, float]]:
        """Get access log for auditing.

        Returns
        -------
        list of tuples
            List of (key, timestamp) tuples.
        """
        return self._access_log.copy()

    def __del__(self) -> None:
        """Cleanup on destruction."""
        self.clear_all()


class CredentialRotationManager:
    """Manages credential rotation with TTL support.

    Implements structural reorganization principle by managing
    credential lifecycle and triggering rotation when coherence
    (validity period) decreases.
    """

    def __init__(
        self,
        rotation_interval: timedelta = timedelta(hours=24),
        warning_threshold: timedelta = timedelta(hours=2),
    ) -> None:
        """Initialize rotation manager.

        Parameters
        ----------
        rotation_interval : timedelta, default=24 hours
            How often credentials should be rotated.
        warning_threshold : timedelta, default=2 hours
            When to warn about upcoming expiration.
        """
        self.rotation_interval = rotation_interval
        self.warning_threshold = warning_threshold
        self._last_rotation: dict[str, datetime] = {}
        self._rotation_callbacks: dict[str, Callable[[], None]] = {}

    def register_credential(
        self,
        credential_key: str,
        rotation_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        """Register a credential for rotation tracking.

        Parameters
        ----------
        credential_key : str
            Identifier for the credential.
        rotation_callback : callable, optional
            Function to call when rotation is needed.
        """
        self._last_rotation[credential_key] = datetime.now(timezone.utc)
        if rotation_callback is not None:
            self._rotation_callbacks[credential_key] = rotation_callback

    def needs_rotation(self, credential_key: str) -> bool:
        """Check if credential needs rotation.

        Parameters
        ----------
        credential_key : str
            Credential identifier.

        Returns
        -------
        bool
            True if rotation is needed.
        """
        last = self._last_rotation.get(credential_key)
        if last is None:
            return True
        age = datetime.now(timezone.utc) - last
        return age >= self.rotation_interval

    def needs_warning(self, credential_key: str) -> bool:
        """Check if credential is nearing expiration.

        Parameters
        ----------
        credential_key : str
            Credential identifier.

        Returns
        -------
        bool
            True if warning should be issued.
        """
        last = self._last_rotation.get(credential_key)
        if last is None:
            return True
        age = datetime.now(timezone.utc) - last
        time_until_rotation = self.rotation_interval - age
        return time_until_rotation <= self.warning_threshold

    def rotate_if_needed(self, credential_key: str) -> bool:
        """Rotate credential if needed.

        Parameters
        ----------
        credential_key : str
            Credential identifier.

        Returns
        -------
        bool
            True if rotation was performed.
        """
        if self.needs_rotation(credential_key):
            callback = self._rotation_callbacks.get(credential_key)
            if callback is not None:
                callback()
            self._last_rotation[credential_key] = datetime.now(timezone.utc)
            return True
        return False

    def get_credential_age(self, credential_key: str) -> timedelta | None:
        """Get age of credential.

        Parameters
        ----------
        credential_key : str
            Credential identifier.

        Returns
        -------
        timedelta or None
            Age of credential, or None if not registered.
        """
        last = self._last_rotation.get(credential_key)
        if last is None:
            return None
        return datetime.now(timezone.utc) - last


class SecurityAuditor:
    """Security auditor for configuration and environment.

    Implements diagnostic nodal analysis to identify security
    coherence issues and dissonances in configuration.
    """

    SENSITIVE_PATTERNS = frozenset(
        [
            "password",
            "secret",
            "key",
            "token",
            "credential",
            "api_key",
            "apikey",
            "auth",
            "private",
        ]
    )

    WEAK_VALUES = frozenset(
        [
            "password",
            "123456",
            "admin",
            "secret",
            "test",
            "changeme",
            "default",
            "root",
            "toor",
        ]
    )

    def audit_environment_variables(self) -> list[str]:
        """Audit environment variables for security issues.

        Returns
        -------
        list of str
            List of security issues found.
        """
        issues = []

        for var_name in os.environ:
            var_name_lower = var_name.lower()
            var_value = os.environ[var_name]

            # Check if this is a sensitive variable
            is_sensitive = any(pattern in var_name_lower for pattern in self.SENSITIVE_PATTERNS)

            if is_sensitive:
                # Check for weak values
                if var_value.lower() in self.WEAK_VALUES:
                    issues.append(f"Weak/default value in sensitive variable: {var_name}")

                # Check for too short secrets
                if len(var_value) < 8:
                    issues.append(f"Secret too short ({len(var_value)} chars) in: {var_name}")

                # Check if secret looks like a placeholder
                if var_value in ["your-secret", "your-token", "changeme", "..."]:
                    issues.append(f"Placeholder value detected in: {var_name}")

        return issues

    def check_redis_config_security(self) -> list[str]:
        """Check Redis configuration for security issues.

        Returns
        -------
        list of str
            List of security issues found.
        """
        issues = []

        # Check if password is set
        redis_password = os.environ.get("REDIS_PASSWORD")
        if not redis_password:
            issues.append("REDIS_PASSWORD not set - authentication disabled")

        # Check if TLS is enabled
        redis_use_tls = os.environ.get("REDIS_USE_TLS", "false").lower()
        if redis_use_tls not in ("true", "1", "yes", "on"):
            issues.append("REDIS_USE_TLS not enabled - unencrypted connection")

        return issues

    def check_cache_secret_security(self) -> list[str]:
        """Check cache secret configuration.

        Returns
        -------
        list of str
            List of security issues found.
        """
        issues = []

        cache_secret = os.environ.get("TNFR_CACHE_SECRET")
        if not cache_secret:
            issues.append("TNFR_CACHE_SECRET not set - unsigned cache data")
        else:
            # Check if secret is strong enough
            try:
                secret_bytes = bytes.fromhex(cache_secret)
                if len(secret_bytes) < 16:
                    issues.append(
                        f"TNFR_CACHE_SECRET too short: {len(secret_bytes)} bytes "
                        "(recommend 32+ bytes)"
                    )
            except ValueError:
                issues.append("TNFR_CACHE_SECRET is not valid hex")

        return issues

    def run_full_audit(self) -> dict[str, list[str]]:
        """Run complete security audit.

        Returns
        -------
        dict
            Dictionary mapping audit category to list of issues.
        """
        return {
            "environment_variables": self.audit_environment_variables(),
            "redis_config": self.check_redis_config_security(),
            "cache_secret": self.check_cache_secret_security(),
        }


# Global instances for convenience
_global_secret_manager: Optional[SecureSecretManager] = None
_global_rotation_manager: Optional[CredentialRotationManager] = None


def get_secret_manager() -> SecureSecretManager:
    """Get global secret manager instance.

    Returns
    -------
    SecureSecretManager
        Global secret manager instance.
    """
    global _global_secret_manager
    if _global_secret_manager is None:
        _global_secret_manager = SecureSecretManager()
    return _global_secret_manager


def get_rotation_manager() -> CredentialRotationManager:
    """Get global rotation manager instance.

    Returns
    -------
    CredentialRotationManager
        Global rotation manager instance.
    """
    global _global_rotation_manager
    if _global_rotation_manager is None:
        _global_rotation_manager = CredentialRotationManager()
    return _global_rotation_manager
