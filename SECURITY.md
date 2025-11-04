# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

The TNFR Python Engine team takes security vulnerabilities seriously. We appreciate your efforts to responsibly disclose your findings and will make every effort to acknowledge your contributions.

### How to Report a Security Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **GitHub Security Advisories**: Use the [Security Advisory](https://github.com/fermga/TNFR-Python-Engine/security/advisories/new) feature (preferred)
2. **Email**: Contact the maintainers directly (check repository for current contact)

Please include the following information in your report:

- Type of vulnerability
- Full paths of source file(s) related to the manifestation of the vulnerability
- The location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the vulnerability
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: Within 48 hours of report submission
- **Status Update**: Within 7 days with assessment and potential timeline
- **Resolution**: Varies by severity; critical issues prioritized

### Disclosure Policy

- Security vulnerabilities will be coordinated with reporters before public disclosure
- We aim to patch critical vulnerabilities within 30 days
- Public disclosure will occur after patch is released and users have time to update

## Security Best Practices for Users

### Pickle Serialization Warning

The TNFR engine uses Python's `pickle` module for caching complex TNFR structures (NetworkX graphs, EPIs, coherence states). **Pickle can execute arbitrary code during deserialization.**

**Important Security Considerations:**

1. **ShelveCacheLayer**: Only load shelf files from trusted sources
2. **RedisCacheLayer**: 
   - Use Redis authentication (AUTH command or ACL)
   - Implement network access controls
   - Use TLS for Redis connections
   - Never cache untrusted user input

### Hardened Cache Signatures

`ShelveCacheLayer` and `RedisCacheLayer` support optional payload signing to detect tampering in environments where cache files or Redis instances are not fully trusted.

- Configure a shared secret HMAC (or any signing/verification callable pair) using the new ``signer`` and ``validator`` parameters.
- Enable ``require_signature=True`` to activate hardened mode. In hardened mode the cache deletes unsigned or invalid entries and raises a :class:`tnfr.utils.SecurityError`.
- Hardened mode is opt-in; the default configuration remains backward compatible with existing caches.

**Example configuration:**

```python
import hashlib
import hmac

from tnfr.utils import RedisCacheLayer, SecurityError, ShelveCacheLayer

SECRET = b"tnfr-shared-secret"

def signer(payload: bytes) -> bytes:
    return hmac.new(SECRET, payload, hashlib.sha256).digest()

def validator(payload: bytes, signature: bytes) -> bool:
    expected = hmac.new(SECRET, payload, hashlib.sha256).digest()
    return hmac.compare_digest(expected, signature)

shelf_layer = ShelveCacheLayer(
    "cache.db",
    signer=signer,
    validator=validator,
    require_signature=True,
)

redis_layer = RedisCacheLayer(
    namespace="tnfr:cache",
    signer=signer,
    validator=validator,
    require_signature=True,
)

try:
    shelf_layer.store("alpha", {"value": 1})
except SecurityError:
    # Hardened mode rejected the payload
    ...
```

When hardened mode is active any tampered cache entry is purged and causes an immediate `SecurityError`, preventing poisoned payloads from propagating through TNFR simulations.

### Dependency Management

- All dependencies are regularly scanned with `pip-audit`
- Dependabot is enabled for automatic dependency updates
- Security updates are applied promptly

### Static Analysis

The repository uses multiple security scanning tools:

- **Bandit**: Python code security scanner
- **Semgrep**: Pattern-based code analysis
- **CodeQL**: Semantic code analysis
- **pip-audit**: Dependency vulnerability scanner

All security scans run automatically on pull requests and commits to the main branch.

## Security Features

### TNFR Structural Integrity

TNFR maintains structural fidelity through canonical invariants that prevent:

1. **Uncontrolled mutations**: EPI changes only through structural operators
2. **Non-deterministic behavior**: Reproducible simulations with seed control
3. **Phase violations**: Explicit phase verification for all couplings
4. **Frequency failures**: Validation of structural frequency (νf) in Hz_str

These invariants ensure predictable, auditable behavior across all TNFR operations.

### Domain Neutrality

The TNFR engine is domain-neutral by design, supporting:

- Multiple backend choices (NumPy, JAX, PyTorch)
- Configurable determinism for reproducibility
- Transparent structural logging for audit trails

## Known Security Considerations

### Documented Pickle Usage

The following components use pickle serialization with documented security warnings:

1. **`src/tnfr/utils/cache.py`**:
   - `ShelveCacheLayer`: Persistent file-based caching
   - `RedisCacheLayer`: Distributed Redis-based caching

2. **`src/tnfr/dynamics/dnfr.py`**:
   - Pickle serialization check for parallel processing compatibility

All pickle usage includes:
- Comprehensive security warnings in docstrings
- Clear documentation of trust requirements
- `# nosec` annotations where risk is accepted and documented

### Excluded Security Checks

The following Bandit checks are intentionally excluded in `bandit.yaml`:

- **B610**: Django QuerySet.extra - Not applicable (TNFR uses internal callables only)
- **B324**: SHA1 usage - Non-cryptographic use (topology fingerprints for logging/caching)

## Security Update Policy

1. **Critical vulnerabilities**: Patched within 48-72 hours
2. **High severity**: Patched within 1 week
3. **Medium/Low severity**: Patched in next minor release
4. **Dependency updates**: Applied automatically via Dependabot

## Attribution

We believe in responsible disclosure and will credit security researchers who report vulnerabilities (unless they prefer to remain anonymous).

## Questions?

If you have questions about this security policy, please open a [GitHub Discussion](https://github.com/fermga/TNFR-Python-Engine/discussions) or contact the maintainers.

---

**Last Updated**: November 2025
**Policy Version**: 1.0
