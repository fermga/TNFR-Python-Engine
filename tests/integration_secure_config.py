#!/usr/bin/env python3
"""Integration test demonstrating security features."""

import os
import sys
from datetime import timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from tnfr.secure_config import (
    SecureCredentialValidator,
    SecureSecretManager,
    CredentialRotationManager,
    SecurityAuditor,
    load_redis_config,
)


def test_credential_validation():
    """Test credential validation."""
    print("=" * 60)
    print("Testing Credential Validation")
    print("=" * 60)
    
    # Valid URL
    url = "redis://localhost:6379/0"
    print(f"✓ Valid URL: {url}")
    assert SecureCredentialValidator.validate_redis_url(url)
    
    # URL with credentials - sanitize for logging
    secret_url = "redis://user:supersecret@host:6379/0"
    safe_url = SecureCredentialValidator.sanitize_for_logging(secret_url)
    print(f"✓ Sanitized URL: {safe_url}")
    assert "supersecret" not in safe_url
    assert "***" in safe_url
    
    # Invalid scheme
    try:
        SecureCredentialValidator.validate_redis_url("http://bad.com")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Rejected invalid scheme: {e}")
    
    print()


def test_secret_management():
    """Test secure secret management."""
    print("=" * 60)
    print("Testing Secret Management")
    print("=" * 60)
    
    manager = SecureSecretManager()
    
    # Store secret
    manager.store_secret("test_key", b"sensitive_data")
    print("✓ Secret stored")
    
    # Retrieve secret
    secret = manager.get_secret("test_key")
    assert secret == b"sensitive_data"
    print("✓ Secret retrieved")
    
    # Clear secret
    manager.clear_secret("test_key")
    assert manager.get_secret("test_key") == b""
    print("✓ Secret cleared from memory")
    
    # Access log
    manager.store_secret("logged", b"data")
    manager.get_secret("logged")
    log = manager.get_access_log()
    assert len(log) > 0
    print(f"✓ Access log has {len(log)} entries")
    
    manager.clear_all()
    print()


def test_rotation_manager():
    """Test credential rotation."""
    print("=" * 60)
    print("Testing Credential Rotation")
    print("=" * 60)
    
    manager = CredentialRotationManager(
        rotation_interval=timedelta(hours=24),
        warning_threshold=timedelta(hours=2)
    )
    
    # Register credential
    rotations = []
    def rotation_callback():
        rotations.append(True)
    
    manager.register_credential("api_key", rotation_callback)
    print("✓ Credential registered")
    
    # Check rotation status
    needs_rotation = manager.needs_rotation("api_key")
    print(f"✓ Needs rotation: {needs_rotation}")
    
    # Get age
    age = manager.get_credential_age("api_key")
    print(f"✓ Credential age: {age}")
    
    print()


def test_security_auditor():
    """Test security auditing."""
    print("=" * 60)
    print("Testing Security Auditor")
    print("=" * 60)
    
    # Set up test environment
    os.environ["TEST_PASSWORD"] = "weak"
    os.environ["TEST_SECRET"] = "short"
    
    auditor = SecurityAuditor()
    
    # Run audits
    env_issues = auditor.audit_environment_variables()
    print(f"✓ Environment audit found {len(env_issues)} issues")
    for issue in env_issues[:3]:  # Show first 3
        print(f"  - {issue}")
    
    redis_issues = auditor.check_redis_config_security()
    print(f"✓ Redis audit found {len(redis_issues)} issues")
    
    cache_issues = auditor.check_cache_secret_security()
    print(f"✓ Cache audit found {len(cache_issues)} issues")
    
    # Full audit
    results = auditor.run_full_audit()
    total = sum(len(issues) for issues in results.values())
    print(f"✓ Full audit found {total} total issues")
    
    # Cleanup
    os.environ.pop("TEST_PASSWORD", None)
    os.environ.pop("TEST_SECRET", None)
    
    print()


def test_redis_config():
    """Test Redis configuration loading."""
    print("=" * 60)
    print("Testing Redis Configuration")
    print("=" * 60)
    
    # Test with URL
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"
    config = load_redis_config()
    print(f"✓ Loaded from REDIS_URL: {config}")
    assert config["host"] == "localhost"
    assert config["port"] == 6379
    
    # Test with individual variables
    os.environ.pop("REDIS_URL", None)
    os.environ["REDIS_HOST"] = "testhost"
    os.environ["REDIS_PORT"] = "6380"
    os.environ["REDIS_DB"] = "1"
    config = load_redis_config()
    print(f"✓ Loaded from individual vars: {config}")
    assert config["host"] == "testhost"
    assert config["port"] == 6380
    assert config["db"] == 1
    
    # Cleanup
    os.environ.pop("REDIS_HOST", None)
    os.environ.pop("REDIS_PORT", None)
    os.environ.pop("REDIS_DB", None)
    
    print()


def main():
    """Run all integration tests."""
    print("\n")
    print("=" * 60)
    print("TNFR Secure Config Integration Tests")
    print("=" * 60)
    print()
    
    try:
        test_credential_validation()
        test_secret_management()
        test_rotation_manager()
        test_security_auditor()
        test_redis_config()
        
        print("=" * 60)
        print("All Integration Tests PASSED ✓")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
