"""Secure configuration management (backward compatibility wrapper).

This module re-exports from tnfr.config.security for backward compatibility.
New code should import directly from tnfr.config.security.

Migration Path:
    Old: from tnfr.secure_config import get_env_variable
    New: from tnfr.config.security import get_env_variable
"""

from __future__ import annotations

# Re-export all secure config functionality from tnfr.config.security
from .config.security import (
    ConfigurationError,
    CredentialRotationManager,
    SecureCredentialValidator,
    SecureSecretManager,
    SecurityAuditWarning,
    SecurityAuditor,
    get_cache_secret,
    get_env_variable,
    get_rotation_manager,
    get_secret_manager,
    load_github_credentials,
    load_pypi_credentials,
    load_redis_config,
    validate_no_hardcoded_secrets,
)

__all__ = (
    "ConfigurationError",
    "SecurityAuditWarning",
    "get_env_variable",
    "load_pypi_credentials",
    "load_github_credentials",
    "load_redis_config",
    "get_cache_secret",
    "validate_no_hardcoded_secrets",
    "SecureCredentialValidator",
    "SecureSecretManager",
    "CredentialRotationManager",
    "SecurityAuditor",
    "get_secret_manager",
    "get_rotation_manager",
)
