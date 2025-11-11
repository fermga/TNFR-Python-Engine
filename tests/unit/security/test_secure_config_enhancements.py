"""Tests for secure_config security enhancements.

This module tests the security improvements added to secure_config.py:
- URL validation and sanitization
- Secure secret management
- Credential rotation
- Security auditing
"""

from __future__ import annotations

import os
import time
from datetime import timedelta

import pytest

from tnfr.secure_config import (
    ConfigurationError,
    CredentialRotationManager,
    SecureCredentialValidator,
    SecureSecretManager,
    SecurityAuditor,
    SecurityAuditWarning,
    get_rotation_manager,
    get_secret_manager,
    load_redis_config,
)


class TestSecureCredentialValidator:
    """Test suite for SecureCredentialValidator."""

    def test_validate_redis_url_valid_redis(self):
        """Test validation accepts valid redis:// URL."""
        url = "redis://localhost:6379/0"
        assert SecureCredentialValidator.validate_redis_url(url) is True

    def test_validate_redis_url_valid_rediss(self):
        """Test validation accepts valid rediss:// URL (TLS)."""
        url = "rediss://secure-host.com:6380/1"
        assert SecureCredentialValidator.validate_redis_url(url) is True

    def test_validate_redis_url_with_auth(self):
        """Test validation accepts URL with authentication."""
        url = "redis://user:password@localhost:6379/0"
        assert SecureCredentialValidator.validate_redis_url(url) is True

    def test_validate_redis_url_rejects_http(self):
        """Test validation rejects non-Redis schemes."""
        with pytest.raises(ValueError, match="Unsupported scheme: http"):
            SecureCredentialValidator.validate_redis_url("http://localhost:6379")

    def test_validate_redis_url_rejects_empty(self):
        """Test validation rejects empty URLs."""
        with pytest.raises(ValueError, match="non-empty string"):
            SecureCredentialValidator.validate_redis_url("")

    def test_validate_redis_url_rejects_too_long(self):
        """Test validation rejects excessively long URLs."""
        long_url = "redis://" + "a" * 600
        with pytest.raises(ValueError, match="exceeds maximum length"):
            SecureCredentialValidator.validate_redis_url(long_url)

    def test_validate_redis_url_rejects_no_hostname(self):
        """Test validation rejects URL without hostname."""
        with pytest.raises(ValueError, match="must include a hostname"):
            SecureCredentialValidator.validate_redis_url("redis://:6379")

    def test_validate_redis_url_rejects_invalid_port(self):
        """Test validation rejects invalid port numbers."""
        with pytest.raises(ValueError, match="Port out of range"):
            SecureCredentialValidator.validate_redis_url("redis://localhost:99999")

    def test_sanitize_for_logging_masks_password(self):
        """Test URL sanitization hides passwords."""
        url = "redis://user:secret_password@localhost:6379/0"
        sanitized = SecureCredentialValidator.sanitize_for_logging(url)

        assert "secret_password" not in sanitized
        assert "***" in sanitized
        assert "localhost" in sanitized
        assert "user" in sanitized

    def test_sanitize_for_logging_preserves_no_password(self):
        """Test sanitization preserves URLs without passwords."""
        url = "redis://localhost:6379/0"
        sanitized = SecureCredentialValidator.sanitize_for_logging(url)
        assert sanitized == url

    def test_sanitize_for_logging_handles_invalid_url(self):
        """Test sanitization handles invalid URLs gracefully."""
        sanitized = SecureCredentialValidator.sanitize_for_logging("not a url")
        assert sanitized == "<invalid-url>"

    def test_sanitize_for_logging_empty_url(self):
        """Test sanitization handles empty URLs."""
        assert SecureCredentialValidator.sanitize_for_logging("") == ""

    def test_validate_secret_strength_accepts_strong(self):
        """Test secret validation accepts strong secrets."""
        strong_secret = "this-is-a-strong-secret-key"
        assert SecureCredentialValidator.validate_secret_strength(strong_secret) is True

    def test_validate_secret_strength_rejects_short(self):
        """Test secret validation rejects short secrets."""
        with pytest.raises(ValueError, match="Secret too short"):
            SecureCredentialValidator.validate_secret_strength("short")

    def test_validate_secret_strength_rejects_weak_passwords(self):
        """Test secret validation rejects known weak passwords."""
        weak_passwords = ["password", "123456", "admin", "secret"]
        for weak in weak_passwords:
            with pytest.raises(ValueError, match="known weak password"):
                SecureCredentialValidator.validate_secret_strength(weak)

    def test_validate_secret_strength_accepts_bytes(self):
        """Test secret validation works with bytes."""
        secret_bytes = b"strong-secret-bytes-here"
        assert SecureCredentialValidator.validate_secret_strength(secret_bytes) is True

    def test_validate_secret_strength_custom_length(self):
        """Test secret validation with custom minimum length."""
        secret = "medium"

        # Should pass with lower threshold
        assert SecureCredentialValidator.validate_secret_strength(secret, min_length=4) is True

        # Should fail with higher threshold
        with pytest.raises(ValueError, match="Secret too short"):
            SecureCredentialValidator.validate_secret_strength(secret, min_length=20)


class TestSecureSecretManager:
    """Test suite for SecureSecretManager."""

    def test_store_and_retrieve_secret(self):
        """Test storing and retrieving secrets."""
        manager = SecureSecretManager()
        manager.store_secret("test_key", b"test_secret")

        retrieved = manager.get_secret("test_key")
        assert retrieved == b"test_secret"

    def test_store_secret_as_string(self):
        """Test storing secret as string."""
        manager = SecureSecretManager()
        manager.store_secret("str_key", "string_secret")

        retrieved = manager.get_secret("str_key")
        assert retrieved == b"string_secret"

    def test_get_nonexistent_secret(self):
        """Test retrieving non-existent secret returns empty bytes."""
        manager = SecureSecretManager()
        assert manager.get_secret("nonexistent") == b""

    def test_clear_secret(self):
        """Test clearing a secret."""
        manager = SecureSecretManager()
        manager.store_secret("clear_me", b"sensitive")

        manager.clear_secret("clear_me")
        assert manager.get_secret("clear_me") == b""

    def test_clear_nonexistent_secret(self):
        """Test clearing non-existent secret doesn't raise."""
        manager = SecureSecretManager()
        manager.clear_secret("nonexistent")  # Should not raise

    def test_clear_all_secrets(self):
        """Test clearing all secrets."""
        manager = SecureSecretManager()
        manager.store_secret("key1", b"secret1")
        manager.store_secret("key2", b"secret2")

        manager.clear_all()

        assert manager.get_secret("key1") == b""
        assert manager.get_secret("key2") == b""

    def test_access_log(self):
        """Test access logging."""
        manager = SecureSecretManager()
        manager.store_secret("logged", b"secret")

        # Access the secret a few times
        manager.get_secret("logged")
        time.sleep(0.01)
        manager.get_secret("logged")

        log = manager.get_access_log()
        assert len(log) >= 2
        assert all(key == "logged" for key, _ in log)
        assert all(isinstance(ts, float) for _, ts in log)

    def test_get_secret_returns_copy(self):
        """Test that get_secret returns a copy, not reference."""
        manager = SecureSecretManager()
        manager.store_secret("original", b"secret")

        retrieved1 = manager.get_secret("original")
        retrieved2 = manager.get_secret("original")

        # Both should be equal but not the same object
        assert retrieved1 == retrieved2
        assert retrieved1 is not retrieved2

    def test_cleanup_on_destruction(self):
        """Test secrets are cleared on manager destruction."""
        manager = SecureSecretManager()
        manager.store_secret("temp", b"will_be_cleared")

        # Trigger destruction
        del manager

        # Create new manager - old secrets should not leak
        new_manager = SecureSecretManager()
        assert new_manager.get_secret("temp") == b""


class TestCredentialRotationManager:
    """Test suite for CredentialRotationManager."""

    def test_register_credential(self):
        """Test registering a credential."""
        manager = CredentialRotationManager()
        manager.register_credential("api_key")

        # Should not need rotation immediately after registration
        assert manager.needs_rotation("api_key") is False

    def test_needs_rotation_unregistered(self):
        """Test unregistered credentials need rotation."""
        manager = CredentialRotationManager()
        assert manager.needs_rotation("unknown") is True

    def test_needs_rotation_after_interval(self):
        """Test rotation needed after interval passes."""
        # Use very short interval for testing
        manager = CredentialRotationManager(rotation_interval=timedelta(seconds=0.1))
        manager.register_credential("short_lived")

        # Should not need rotation immediately
        assert manager.needs_rotation("short_lived") is False

        # Wait for interval to pass
        time.sleep(0.15)

        # Now should need rotation
        assert manager.needs_rotation("short_lived") is True

    def test_needs_warning(self):
        """Test warning threshold detection."""
        manager = CredentialRotationManager(
            rotation_interval=timedelta(seconds=1),
            warning_threshold=timedelta(seconds=0.5),
        )
        manager.register_credential("warn_me")

        # Should not need warning immediately
        assert manager.needs_warning("warn_me") is False

        # Wait until warning threshold
        time.sleep(0.6)

        # Should trigger warning but not rotation
        assert manager.needs_warning("warn_me") is True
        assert manager.needs_rotation("warn_me") is False

    def test_rotate_if_needed_no_callback(self):
        """Test rotation without callback."""
        manager = CredentialRotationManager(rotation_interval=timedelta(seconds=0.1))
        manager.register_credential("simple")

        time.sleep(0.15)

        # Should perform rotation
        assert manager.rotate_if_needed("simple") is True

        # After rotation, should not need rotation again
        assert manager.needs_rotation("simple") is False

    def test_rotate_if_needed_with_callback(self):
        """Test rotation calls callback."""
        callback_called = []

        def rotation_callback():
            callback_called.append(True)

        manager = CredentialRotationManager(rotation_interval=timedelta(seconds=0.1))
        manager.register_credential("with_callback", rotation_callback)

        time.sleep(0.15)

        # Should call callback during rotation
        manager.rotate_if_needed("with_callback")
        assert len(callback_called) == 1

    def test_get_credential_age(self):
        """Test getting credential age."""
        manager = CredentialRotationManager()
        manager.register_credential("aged")

        time.sleep(0.1)

        age = manager.get_credential_age("aged")
        assert age is not None
        assert age.total_seconds() >= 0.1

    def test_get_credential_age_unregistered(self):
        """Test age of unregistered credential."""
        manager = CredentialRotationManager()
        assert manager.get_credential_age("unknown") is None


class TestSecurityAuditor:
    """Test suite for SecurityAuditor."""

    def test_audit_environment_variables_weak_value(self, monkeypatch):
        """Test auditor detects weak values in sensitive variables."""
        monkeypatch.setenv("API_PASSWORD", "password")

        auditor = SecurityAuditor()
        issues = auditor.audit_environment_variables()

        assert any("weak" in issue.lower() for issue in issues)

    def test_audit_environment_variables_short_secret(self, monkeypatch):
        """Test auditor detects short secrets."""
        monkeypatch.setenv("API_KEY", "short")

        auditor = SecurityAuditor()
        issues = auditor.audit_environment_variables()

        assert any("too short" in issue.lower() for issue in issues)

    def test_audit_environment_variables_placeholder(self, monkeypatch):
        """Test auditor detects placeholder values."""
        monkeypatch.setenv("API_TOKEN", "your-token")

        auditor = SecurityAuditor()
        issues = auditor.audit_environment_variables()

        assert any("placeholder" in issue.lower() for issue in issues)

    def test_audit_environment_variables_clean(self, monkeypatch):
        """Test auditor with proper configuration."""
        # Set strong secrets
        monkeypatch.setenv("API_KEY", "strong-secret-key-value-here")

        auditor = SecurityAuditor()
        issues = auditor.audit_environment_variables()

        # Should not report issues for strong secrets
        api_key_issues = [i for i in issues if "API_KEY" in i]
        assert len(api_key_issues) == 0

    def test_check_redis_config_security_no_password(self, monkeypatch):
        """Test auditor detects missing Redis password."""
        monkeypatch.delenv("REDIS_PASSWORD", raising=False)

        auditor = SecurityAuditor()
        issues = auditor.check_redis_config_security()

        assert any("authentication disabled" in issue.lower() for issue in issues)

    def test_check_redis_config_security_no_tls(self, monkeypatch):
        """Test auditor detects disabled TLS."""
        monkeypatch.setenv("REDIS_USE_TLS", "false")

        auditor = SecurityAuditor()
        issues = auditor.check_redis_config_security()

        assert any("unencrypted" in issue.lower() for issue in issues)

    def test_check_cache_secret_security_not_set(self, monkeypatch):
        """Test auditor detects missing cache secret."""
        monkeypatch.delenv("TNFR_CACHE_SECRET", raising=False)

        auditor = SecurityAuditor()
        issues = auditor.check_cache_secret_security()

        assert any("not set" in issue.lower() for issue in issues)

    def test_check_cache_secret_security_too_short(self, monkeypatch):
        """Test auditor detects short cache secret."""
        # 10 hex chars = 5 bytes (too short)
        monkeypatch.setenv("TNFR_CACHE_SECRET", "0123456789")

        auditor = SecurityAuditor()
        issues = auditor.check_cache_secret_security()

        assert any("too short" in issue.lower() for issue in issues)

    def test_check_cache_secret_security_invalid_hex(self, monkeypatch):
        """Test auditor detects invalid hex in cache secret."""
        monkeypatch.setenv("TNFR_CACHE_SECRET", "not-valid-hex")

        auditor = SecurityAuditor()
        issues = auditor.check_cache_secret_security()

        assert any("not valid hex" in issue.lower() for issue in issues)

    def test_run_full_audit(self, monkeypatch):
        """Test running complete audit."""
        monkeypatch.setenv("API_PASSWORD", "weak")
        monkeypatch.delenv("REDIS_PASSWORD", raising=False)
        monkeypatch.delenv("TNFR_CACHE_SECRET", raising=False)

        auditor = SecurityAuditor()
        results = auditor.run_full_audit()

        assert "environment_variables" in results
        assert "redis_config" in results
        assert "cache_secret" in results

        # Should have findings in all categories
        assert len(results["environment_variables"]) > 0
        assert len(results["redis_config"]) > 0
        assert len(results["cache_secret"]) > 0


class TestLoadRedisConfigEnhancements:
    """Test enhancements to load_redis_config function."""

    def test_load_redis_config_from_url(self, monkeypatch):
        """Test loading Redis config from REDIS_URL."""
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

        config = load_redis_config()

        assert config["host"] == "localhost"
        assert config["port"] == 6379
        assert config["db"] == 0
        assert config["ssl"] is False

    def test_load_redis_config_url_with_auth(self, monkeypatch):
        """Test loading Redis URL with authentication."""
        monkeypatch.setenv("REDIS_URL", "redis://:mypassword@localhost:6379/1")

        config = load_redis_config()

        assert config["password"] == "mypassword"
        assert config["db"] == 1

    def test_load_redis_config_url_with_tls(self, monkeypatch):
        """Test loading Redis URL with TLS."""
        monkeypatch.setenv("REDIS_URL", "rediss://secure.redis.com:6380/0")

        config = load_redis_config()

        assert config["host"] == "secure.redis.com"
        assert config["port"] == 6380
        assert config["ssl"] is True

    def test_load_redis_config_url_validation_fails(self, monkeypatch):
        """Test that invalid URLs are rejected."""
        monkeypatch.setenv("REDIS_URL", "http://not-redis:6379")

        with pytest.raises(ValueError, match="Unsupported scheme"):
            load_redis_config()

    def test_load_redis_config_invalid_port_range(self, monkeypatch):
        """Test that invalid port ranges are rejected."""
        monkeypatch.setenv("REDIS_PORT", "99999")
        monkeypatch.delenv("REDIS_URL", raising=False)

        with pytest.raises(ConfigurationError, match="must be between"):
            load_redis_config()

    def test_load_redis_config_validation_disabled(self, monkeypatch):
        """Test disabling URL validation."""
        # This would normally fail validation
        monkeypatch.setenv("REDIS_URL", "http://localhost:6379")

        # Should not raise when validation is disabled
        config = load_redis_config(validate_url=False)
        assert config is not None


class TestGlobalInstances:
    """Test global manager instance helpers."""

    def test_get_secret_manager(self):
        """Test getting global secret manager."""
        manager1 = get_secret_manager()
        manager2 = get_secret_manager()

        # Should return same instance
        assert manager1 is manager2

    def test_get_rotation_manager(self):
        """Test getting global rotation manager."""
        manager1 = get_rotation_manager()
        manager2 = get_rotation_manager()

        # Should return same instance
        assert manager1 is manager2
