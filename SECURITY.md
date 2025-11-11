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

### SQL Injection Prevention

**TNFR provides proactive SQL injection prevention utilities:**

The TNFR engine currently uses in-memory NetworkX graphs and file-based persistence (JSON, YAML, Pickle). While no SQL databases are currently used, the codebase includes comprehensive SQL injection prevention utilities for future database functionality.

**Security Utilities Available:**

```python
from tnfr.security import (
    SecureQueryBuilder,
    validate_identifier,
    sanitize_string_input,
    validate_nodal_input,
)

# Always use parameterized queries
builder = SecureQueryBuilder()
query, params = builder.select("nfr_nodes", ["id", "nu_f", "phase"])\
    .where("nu_f > ?", 0.5)\
    .order_by("nu_f", "DESC")\
    .build()

# Validate all identifiers (table/column names)
table_name = validate_identifier("nfr_nodes")
column_name = validate_identifier("nu_f")

# Sanitize string inputs
user_input = sanitize_string_input(user_provided_data, max_length=1000)

# Validate TNFR structural data before persistence
node_data = validate_nodal_input({
    "nu_f": 0.75,
    "phase": 1.57,
    "coherence": 0.85,
})
```

**Security Principles:**

1. **Parameterized Queries**: Always use placeholders (?, :name) for values
2. **Identifier Validation**: Validate table/column names against whitelist pattern
3. **No String Concatenation**: Never build queries with f-strings or + operators
4. **Input Sanitization**: Validate and sanitize all user inputs
5. **TNFR Structural Validation**: Ensure structural frequency, phase, and coherence values are valid

**Example of Safe vs Unsafe Patterns:**

```python
# ❌ UNSAFE: Never do this!
# query = f"SELECT * FROM nfr_nodes WHERE id = {user_input}"

# ✓ SAFE: Use parameterized queries
builder = SecureQueryBuilder()
query, params = builder.select("nfr_nodes")\
    .where("id = ?", user_input)\
    .build()
# Execute: cursor.execute(query, params)
```

### Secret and Credential Management

**TNFR follows strict security practices to prevent hardcoded secrets:**

1. **No Hardcoded Secrets**: All API keys, passwords, tokens, and credentials are loaded from environment variables
2. **Environment Variables**: Use `.env` files for local development (never commit these!)
3. **Configuration Template**: Copy `.env.example` to `.env` and fill in your credentials
4. **Secure Defaults**: Development defaults are safe and non-functional placeholders

**Environment Configuration:**

```bash
# Copy the example configuration
cp .env.example .env

# Edit .env with your actual credentials
# This file is gitignored and will not be committed
```

**Using Configuration Utilities:**

```python
from tnfr.secure_config import (
    load_pypi_credentials,
    load_github_credentials,
    load_redis_config,
    get_cache_secret,
)

# Load credentials from environment
pypi_creds = load_pypi_credentials()
github_creds = load_github_credentials()
redis_config = load_redis_config()

# Get cache signing secret
cache_secret = get_cache_secret()
```

**Security Best Practices:**

- Use API tokens instead of passwords (e.g., PyPI tokens, GitHub tokens)
- Rotate credentials regularly
- Use different credentials for development, staging, and production
- Store production secrets in secure secret management systems (AWS Secrets Manager, HashiCorp Vault, etc.)
- Grant minimal necessary permissions to all tokens
- Revoke tokens immediately if compromised

**Automated Security Testing:**

TNFR includes automated tests that scan for hardcoded secrets:
- GitHub token detection
- PyPI token detection
- Suspicious long strings that might be secrets
- Verification of `.env` in `.gitignore`

### Pickle Serialization Warning

The TNFR engine uses Python's `pickle` module for caching complex TNFR structures (NetworkX graphs, EPIs, coherence states). **Pickle can execute arbitrary code during deserialization.**

**Important Security Considerations:**

1. **ShelveCacheLayer**: Only load shelf files from trusted sources
2. **RedisCacheLayer**: 
   - Use Redis authentication (AUTH command or ACL)
   - Implement network access controls
   - Use TLS for Redis connections
   - Never cache untrusted user input

**Security Warnings:**

Starting from version 1.x, TNFR will emit `SecurityWarning` when cache layers are created without signature validation. To suppress these warnings in trusted environments, set:

```bash
export TNFR_ALLOW_UNSIGNED_PICKLE=1
```

### Secure Cache Configuration (Recommended)

TNFR provides convenient helper functions to create secure cache layers with HMAC signature validation. **This is the recommended approach for production deployments.**

#### Quick Start with Secure Caches

```python
from tnfr.utils import create_secure_shelve_layer, create_secure_redis_layer

# Set your cache secret via environment variable (recommended)
# export TNFR_CACHE_SECRET="your-secure-random-secret-key"

# Create secure cache layers (reads secret from environment)
shelf_layer = create_secure_shelve_layer("coherence.db")
redis_layer = create_secure_redis_layer()

# Store and retrieve TNFR structures safely
shelf_layer.store("nfr_state", {"epi": [1.5, 2.3], "theta": [0.1, 0.2]})
restored = shelf_layer.load("nfr_state")
```

#### Environment Variables

- `TNFR_CACHE_SECRET`: Secret key for HMAC signature validation (required for secure layers)
- `TNFR_ALLOW_UNSIGNED_PICKLE`: Set to `1` to suppress security warnings for unsigned pickle usage

#### Manual HMAC Configuration

For more control, you can use the HMAC helper functions:

```python
from tnfr.utils import (
    create_hmac_signer,
    create_hmac_validator,
    ShelveCacheLayer,
    RedisCacheLayer,
)

# Create HMAC signer and validator
secret = b"your-secure-secret-key"
signer = create_hmac_signer(secret)
validator = create_hmac_validator(secret)

# Create cache layers with signature validation
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
```

### Hardened Cache Signatures

`ShelveCacheLayer` and `RedisCacheLayer` support payload signing to detect tampering in environments where cache files or Redis instances are not fully trusted.

- Configure a shared secret HMAC (or any signing/verification callable pair) using the ``signer`` and ``validator`` parameters.
- Enable ``require_signature=True`` to activate hardened mode. In hardened mode the cache deletes unsigned or invalid entries and raises a :class:`tnfr.utils.SecurityError`.
- **New in 1.x**: Warnings are emitted when creating cache layers without signatures. Use the secure helper functions for best practices.

**Example with custom signing:**

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
    data = shelf_layer.load("alpha")
except SecurityError:
    # Hardened mode rejected tampered payload
    ...
```

**Tamper Detection:**

When hardened mode is active, any tampered cache entry is automatically purged and causes an immediate `SecurityError`, preventing poisoned payloads from propagating through TNFR simulations.

### Migration Guide

If you're using `ShelveCacheLayer` or `RedisCacheLayer` without signatures:

1. **For trusted, local-only caches** (development):
   ```bash
   export TNFR_ALLOW_UNSIGNED_PICKLE=1
   ```

2. **For production deployments** (recommended):
   ```python
   # Before
   layer = ShelveCacheLayer("cache.db")
   
   # After (secure)
   from tnfr.utils import create_secure_shelve_layer
   layer = create_secure_shelve_layer("cache.db")
   ```

3. **Set environment variable** in production:
   ```bash
   export TNFR_CACHE_SECRET="$(openssl rand -hex 32)"
   ```

### Dependency Management

All project dependencies are continuously monitored for security vulnerabilities using automated tools and processes.

#### pip-audit: Automated Dependency Vulnerability Scanning

**What is pip-audit?**

`pip-audit` is a tool that scans Python dependencies for known security vulnerabilities by checking them against the Python Packaging Advisory Database (PyPA).

**Automated Scanning Schedule:**

- **On every push** to main/master branches
- **On every pull request** to main/master branches
- **Weekly scheduled scan** (every Monday at 5 AM UTC)

**How to Run pip-audit Locally:**

```bash
# Install pip-audit
pip install pip-audit

# Install project dependencies
pip install -e .[all]

# Scan installed packages in your environment
pip-audit

# Or scan a specific site-packages directory
SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")
pip-audit --path "$SITE_PACKAGES"
```

**Understanding pip-audit Results:**

When vulnerabilities are found, pip-audit reports:
- **Package name and version**: The vulnerable dependency
- **Vulnerability ID**: GHSA-*, PYSEC-*, or CVE identifier
- **Fix versions**: The version(s) that resolve the vulnerability
- **Severity**: Critical, High, Medium, or Low (when available)

**Security Update Process:**

When pip-audit detects vulnerabilities in the CI/CD pipeline:

1. **Automatic Detection**: The pip-audit workflow runs and uploads a JSON report as an artifact
2. **Review**: Maintainers review the `pip-audit-report` artifact from the workflow run
3. **Assessment**: Evaluate if the vulnerability affects TNFR's use case:
   - Check if the vulnerable code path is actually used by TNFR
   - Assess the severity and exploitability in TNFR's context
   - Review available fixes and their compatibility
4. **Resolution Options**:
   - **Update dependency**: Update to the fixed version in `pyproject.toml`
   - **Pin to safe version**: If latest version causes breaking changes, pin to nearest safe version
   - **Document exception**: If the vulnerability doesn't affect TNFR, document why in a security review
   - **Find alternatives**: Consider replacing the dependency if no safe version exists
5. **Testing**: Run full test suite to ensure the update doesn't break functionality
6. **Document**: Update changelog with security fix information

**Example Workflow for Fixing a Vulnerability:**

```bash
# 1. Review the pip-audit report (download from GitHub Actions artifacts)
cat pip-audit.json

# 2. Update the vulnerable dependency in pyproject.toml
# For example, if networkx 2.6 is vulnerable and 3.0 fixes it:
# Change: "networkx>=2.6,<3.0"
# To:     "networkx>=3.0,<4.0"

# 3. Update your local environment
pip install -e .[all]

# 4. Run pip-audit again to verify the fix
pip-audit

# 5. Run the test suite
pytest

# 6. Commit and create a pull request
git add pyproject.toml
git commit -m "fix: update networkx to resolve GHSA-xxxx-xxxx-xxxx"
git push origin fix/update-networkx
```

**Why pip-audit is NOT in pre-commit hooks:**

While pip-audit is valuable for CI/CD, it is **not included in pre-commit hooks** because:
- It requires network access to query the PyPA advisory database
- It can be slow (5-30 seconds depending on network and number of packages)
- It would slow down developer workflow for every commit
- The automated CI/CD scanning catches issues before merge

**Dependabot Integration:**

In addition to pip-audit, Dependabot is enabled to:
- Automatically create pull requests for dependency updates
- Monitor dependencies for security advisories
- Suggest version updates with changelogs

Both tools work together to ensure dependencies remain secure and up-to-date.

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
