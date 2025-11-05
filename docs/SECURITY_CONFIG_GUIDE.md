# Security Configuration Guide

## Overview

The TNFR engine provides comprehensive security features for managing credentials, secrets, and configuration. This guide covers the security enhancements implemented in `secure_config.py`.

## Security Principles

Following TNFR's structural coherence principles, the security system implements:

1. **Structural Integrity**: Credentials maintain coherence throughout their lifecycle
2. **Resonant Validation**: URL and secret validation ensures structural stability
3. **Coherent Memory Management**: Secrets are securely cleaned from memory
4. **Temporal Reorganization**: Credential rotation implements structural lifecycle management
5. **Diagnostic Coherence**: Security auditing identifies dissonances in configuration

## Features

### 1. Credential Validation

The `SecureCredentialValidator` class provides robust validation for URLs and secrets:

#### Redis URL Validation

```python
from tnfr.secure_config import SecureCredentialValidator

# Validate Redis URL
url = "redis://localhost:6379/0"
SecureCredentialValidator.validate_redis_url(url)  # Returns True

# Only secure schemes allowed (redis://, rediss://)
try:
    SecureCredentialValidator.validate_redis_url("http://evil.com")
except ValueError as e:
    print(f"Validation failed: {e}")
```

**Security checks:**
- Only `redis://` and `rediss://` schemes allowed
- Maximum URL length (512 characters) to prevent DoS
- Hostname validation
- Port range validation (1-65535)

#### Credential Sanitization for Logging

Prevent credential exposure in logs:

```python
from tnfr.secure_config import SecureCredentialValidator

url = "redis://user:secret_password@localhost:6379/0"
safe_url = SecureCredentialValidator.sanitize_for_logging(url)
print(safe_url)  # Output: redis://user:***@localhost:6379/0
```

#### Secret Strength Validation

Ensure secrets meet minimum security requirements:

```python
from tnfr.secure_config import SecureCredentialValidator

# Validate secret strength
secret = "my-strong-secret-key"
SecureCredentialValidator.validate_secret_strength(secret, min_length=8)

# Rejects weak passwords
try:
    SecureCredentialValidator.validate_secret_strength("password")
except ValueError as e:
    print(f"Weak secret: {e}")
```

### 2. Secure Secret Management

The `SecureSecretManager` provides automatic memory cleanup for secrets:

```python
from tnfr.secure_config import SecureSecretManager, get_secret_manager

# Use global instance
manager = get_secret_manager()

# Store secret
manager.store_secret("api_key", b"sensitive_data")

# Retrieve secret (returns copy, not reference)
secret = manager.get_secret("api_key")

# Clear specific secret from memory
manager.clear_secret("api_key")

# Clear all secrets
manager.clear_all()
```

**Security features:**
- Secrets stored in mutable `bytearray` for secure clearing
- Overwritten with random bytes before deletion
- Automatic cleanup on manager destruction
- Access logging for auditing
- Returns copies to prevent external mutation

### 3. Credential Rotation

The `CredentialRotationManager` implements automatic credential lifecycle management:

```python
from datetime import timedelta
from tnfr.secure_config import CredentialRotationManager, get_rotation_manager

# Create rotation manager
manager = CredentialRotationManager(
    rotation_interval=timedelta(hours=24),
    warning_threshold=timedelta(hours=2)
)

# Register credential with rotation callback
def rotate_api_key():
    print("Rotating API key...")
    # Implementation: generate new key, update systems, etc.

manager.register_credential("api_key", rotation_callback=rotate_api_key)

# Check if rotation needed
if manager.needs_rotation("api_key"):
    manager.rotate_if_needed("api_key")

# Check if warning needed (approaching expiration)
if manager.needs_warning("api_key"):
    print("Warning: API key will expire soon")

# Get credential age
age = manager.get_credential_age("api_key")
print(f"Credential age: {age}")
```

**Features:**
- Configurable rotation intervals
- Warning threshold for upcoming expirations
- Optional rotation callbacks
- Credential age tracking
- Automatic timestamp management

### 4. Security Auditing

The `SecurityAuditor` performs comprehensive security checks:

```python
from tnfr.secure_config import SecurityAuditor

auditor = SecurityAuditor()

# Audit environment variables
env_issues = auditor.audit_environment_variables()
for issue in env_issues:
    print(f"Environment issue: {issue}")

# Check Redis configuration
redis_issues = auditor.check_redis_config_security()
for issue in redis_issues:
    print(f"Redis issue: {issue}")

# Check cache secret
cache_issues = auditor.check_cache_secret_security()
for issue in cache_issues:
    print(f"Cache issue: {issue}")

# Run full audit
results = auditor.run_full_audit()
for category, issues in results.items():
    print(f"\n{category}:")
    for issue in issues:
        print(f"  - {issue}")
```

**Audit checks:**
- Weak or default passwords in environment variables
- Short secrets (< 8 characters)
- Placeholder values (e.g., "your-token", "changeme")
- Missing Redis authentication
- Disabled TLS for Redis
- Missing or weak cache secrets
- Invalid hex encoding

### 5. Enhanced Redis Configuration

The `load_redis_config()` function now supports URL-based configuration and validation:

```python
from tnfr.secure_config import load_redis_config
import os

# Option 1: Use REDIS_URL
os.environ["REDIS_URL"] = "rediss://user:pass@secure.redis.com:6380/0"
config = load_redis_config()

# Option 2: Use individual variables (existing behavior)
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6379"
os.environ["REDIS_PASSWORD"] = "secret"
os.environ["REDIS_DB"] = "0"
os.environ["REDIS_USE_TLS"] = "true"
config = load_redis_config()

# Disable validation if needed (not recommended)
config = load_redis_config(validate_url=False)
```

**Configuration variables:**
- `REDIS_URL`: Complete Redis URL (overrides individual settings)
- `REDIS_HOST`: Redis server hostname (default: "localhost")
- `REDIS_PORT`: Redis port (default: 6379)
- `REDIS_PASSWORD`: Redis password (optional)
- `REDIS_DB`: Database number (default: 0)
- `REDIS_USE_TLS`: Enable TLS (default: false)

## Best Practices

### 1. Always Use Strong Secrets

```python
import secrets

# Generate strong secret for cache
cache_secret = secrets.token_hex(32)  # 64-character hex string
print(f"TNFR_CACHE_SECRET={cache_secret}")
```

### 2. Enable TLS for Network Connections

```bash
# Use rediss:// scheme for TLS
export REDIS_URL="rediss://username:password@redis-server:6380/0"

# Or enable via environment variable
export REDIS_USE_TLS=true
```

### 3. Implement Credential Rotation

```python
from datetime import timedelta
from tnfr.secure_config import get_rotation_manager

manager = get_rotation_manager()

# Register all credentials
manager.register_credential("redis_password", rotation_callback=rotate_redis_creds)
manager.register_credential("api_key", rotation_callback=rotate_api_key)

# Check periodically
for cred in ["redis_password", "api_key"]:
    if manager.needs_rotation(cred):
        print(f"Rotating {cred}...")
        manager.rotate_if_needed(cred)
```

### 4. Run Security Audits Regularly

```python
from tnfr.secure_config import SecurityAuditor

def periodic_security_audit():
    auditor = SecurityAuditor()
    results = auditor.run_full_audit()
    
    total_issues = sum(len(issues) for issues in results.values())
    if total_issues > 0:
        print(f"Security audit found {total_issues} issues:")
        for category, issues in results.items():
            if issues:
                print(f"\n{category}:")
                for issue in issues:
                    print(f"  - {issue}")
        return False
    return True

# Run on startup
if not periodic_security_audit():
    print("WARNING: Security issues detected")
```

### 5. Sanitize Logs

Always sanitize URLs before logging:

```python
import logging
from tnfr.secure_config import SecureCredentialValidator

logger = logging.getLogger(__name__)

redis_url = os.getenv("REDIS_URL")
safe_url = SecureCredentialValidator.sanitize_for_logging(redis_url)
logger.info(f"Connecting to Redis: {safe_url}")
```

### 6. Use Global Managers

For consistency across your application:

```python
from tnfr.secure_config import get_secret_manager, get_rotation_manager

# Use global instances
secret_mgr = get_secret_manager()
rotation_mgr = get_rotation_manager()

# All modules will share the same instances
```

## Security Checklist

- [ ] All secrets loaded from environment variables
- [ ] No hardcoded credentials in source code
- [ ] Redis authentication enabled (REDIS_PASSWORD)
- [ ] TLS enabled for network connections (REDIS_USE_TLS)
- [ ] Cache secret configured (TNFR_CACHE_SECRET)
- [ ] Cache secret is strong (32+ bytes, hex-encoded)
- [ ] Credential rotation implemented
- [ ] Security audits run periodically
- [ ] URLs sanitized in logs
- [ ] Weak passwords rejected

## TNFR Structural Alignment

These security features align with TNFR principles:

1. **Coherence**: Secrets maintain structural integrity through secure lifecycle management
2. **Resonance**: Validation ensures credentials resonate with security requirements
3. **Phase Synchrony**: Rotation manager maintains temporal coherence
4. **Dissonance Detection**: Auditor identifies configuration dissonances
5. **Self-Organization**: Automatic memory cleanup and credential rotation
6. **Operational Fractality**: Security patterns apply at all scales (development, production)

## Troubleshooting

### Issue: "Unsupported scheme" error

**Cause**: Using non-Redis URL scheme (e.g., `http://`)

**Solution**: Use `redis://` or `rediss://` (with TLS)

```python
# Wrong
url = "http://redis-server:6379"

# Correct
url = "redis://redis-server:6379"
url = "rediss://redis-server:6380"  # With TLS
```

### Issue: "Secret too short" error

**Cause**: Secret doesn't meet minimum length requirement

**Solution**: Generate stronger secret

```python
import secrets
strong_secret = secrets.token_hex(32)  # 64 hex chars = 32 bytes
```

### Issue: URLs with credentials appearing in logs

**Cause**: Not sanitizing URLs before logging

**Solution**: Always sanitize

```python
from tnfr.secure_config import SecureCredentialValidator

url = "redis://user:pass@host:6379/0"
safe_url = SecureCredentialValidator.sanitize_for_logging(url)
logger.info(f"Connecting to: {safe_url}")
```

## Additional Resources

- [SECURITY.md](../SECURITY.md) - General security policy
- [Redis Security Documentation](https://redis.io/docs/management/security/)
- [OWASP Credential Management](https://cheatsheetseries.owasp.org/cheatsheets/Credential_Storage_Cheat_Sheet.html)
