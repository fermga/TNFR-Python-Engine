# Security Policy

## Supported Versions

The current package version and interpreter requirements are defined in
[pyproject.toml](pyproject.toml). This repository does not declare a supported
1.x release line or a guaranteed backport schedule. Include the exact release
or commit in a vulnerability report so maintainers can assess affected versions.
This policy guarantees no response or remediation deadline.

## Reporting a Vulnerability

Do not publish credentials, personal data or an uncoordinated exploit in an
ordinary issue. Use GitHub's
[private vulnerability reporting](https://github.com/fermga/TNFR-Python-Engine/security/advisories/new)
when available. Otherwise request a private contact channel without disclosing
the vulnerability itself.

Include the affected version, environment, minimal reproducer, required
privileges or attacker-controlled inputs, impact and any proposed fix. Remove
real secrets and sensitive data. Coordinate disclosure with maintainers once the
behavior and mitigation are understood. Repository files alone do not establish
which GitHub security settings are enabled.

## Security Best Practices for Users

### Pickle Serialization Warning

Pickle can execute code during deserialization. Treat pickle-based persistent
caches and their writers as trusted inputs. Do not load an untrusted cache merely
because its filename or structural contents look valid. Use a data-only format
and explicit schema when supported; resource limits and semantic validation
remain separate requirements.

### Secure Cache Configuration

[cache_layers.py](src/tnfr/utils/cache_layers.py) owns the implementation,
re-exported by [cache.py](src/tnfr/utils/cache.py). Secure shelve and Redis
constructors require a caller-supplied secret or `TNFR_CACHE_SECRET`; there
is no universal built-in secret. Protect the key and restrict underlying storage
access.

The signed `TNFRSIG2` envelope authenticates serialization mode and payload
before inner decoding. Signed shelve reads also restrict decoding of the outer
storage envelope. Legacy `TNFRSIG1` entries are rejected and must be rebuilt
from trusted sources. These are format-specific checks, not a general pickle
sandbox. A valid HMAC verifies authenticated bytes under the configured key;
it does not establish safe payload semantics, freshness, confidentiality or
acceptable resource use.

Unsigned cache modes remain available. `TNFR_ALLOW_UNSIGNED_PICKLE=1`
suppresses their warning; it does not enable signature checking. Inspect the
actual constructor and validation options rather than inferring protection from
a warning setting or an in-memory cache's behavior.

### Secret and Credential Management

[tnfr.config.security](src/tnfr/config/security.py) owns credential helpers;
[tnfr.secure_config](src/tnfr/secure_config.py) is a compatibility re-export.
These helpers do not automatically secure every application configuration. Keep
real credentials out of source, logs, examples, reports and artifacts. A
gitignore rule does not remove an already tracked secret; revoke exposed
credentials and address the exposure.

### SQL Injection Prevention

[tnfr.security.database](src/tnfr/security/database.py) supplies identifier
validation and parameterized query helpers. Bind untrusted values through the
database driver's parameter mechanism. SQL expression strings, including a
builder's `where(condition, ...)` condition, remain caller-supplied SQL;
do not interpolate untrusted text into them. Identifier validation and value
sanitization are not a general SQL parser or replacement for parameter binding.

### Dependency Management and Static Analysis

[TESTING.md](TESTING.md#security-checks) owns local audit commands.
[The workflow guide](.github/WORKFLOWS.md) owns CI coverage. An audit covers
installed packages and vulnerability information available at that run, not
every optional environment or all possible vulnerabilities.

[bandit.yaml](bandit.yaml) records the current B610 exception. Review exceptions
and findings against affected code. No repository-wide guarantee that every
risky serialization or subprocess path has been audited is implied.

## Security Features and Limits

Structural validation, grammar admission, coherence scores and determinism
serve mathematical or application contracts. They do not provide authentication,
authorization, confidentiality or protection from hostile code. Applications
must establish those boundaries for their deployment and data.

Security fixes should include a focused reproducer or regression where feasible,
the affected trust boundary and any compatibility or cache migration requirement.
Follow [CONTRIBUTING.md](CONTRIBUTING.md) for ordinary review and
[private reporting](#reporting-a-vulnerability) for undisclosed issues.
