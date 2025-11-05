# Command Injection Prevention - Security Implementation

## Overview

This document details the command injection prevention measures implemented in the TNFR Python Engine to ensure secure subprocess execution and prevent security vulnerabilities.

## Security Issue Addressed

**Severity**: HIGH  
**Issue**: Command injection vulnerability through unsanitized user input in subprocess execution

## Implementation Summary

### 1. Secure Subprocess Module (`src/tnfr/security/subprocess.py`)

A new security module provides:

- **Command Allowlisting**: Only pre-approved commands can be executed
- **Input Validation**: All user inputs are validated before use in commands
- **Safe Execution Wrapper**: `run_command_safely()` never uses `shell=True`
- **Path Safety**: Prevents path traversal attacks

### 2. Validation Functions

#### `validate_git_ref(ref: str) -> str`
Validates git references (branches, tags, commit SHAs) to prevent:
- Path traversal (`../`, `/`, `~`)
- Command injection (`; rm -rf /`)
- Invalid characters

```python
# Safe usage
from tnfr.security import validate_git_ref
ref = validate_git_ref("main")  # ✓ Valid
ref = validate_git_ref("feature/new-operator")  # ✓ Valid
ref = validate_git_ref("../etc/passwd")  # ✗ Raises CommandValidationError
```

#### `validate_version_string(version: str) -> str`
Validates semantic version strings:
- Must follow semver format: `\d+\.\d+\.\d+`
- Optional pre-release suffix: `-alpha`, `-beta.1`
- Prevents injection attempts

```python
from tnfr.security import validate_version_string
version = validate_version_string("1.0.0")  # ✓ Valid
version = validate_version_string("v16.2.3")  # ✓ Valid
version = validate_version_string("1.0; rm -rf /")  # ✗ Raises CommandValidationError
```

#### `validate_path_safe(path: str | Path) -> Path`
Prevents path traversal attacks:
- Rejects absolute paths in user input
- Blocks `..` in path components
- Validates safe characters only

```python
from tnfr.security import validate_path_safe
path = validate_path_safe("src/tnfr/core.py")  # ✓ Valid
path = validate_path_safe("../../../etc/passwd")  # ✗ Raises CommandValidationError
```

#### `run_command_safely(command: Sequence[str], ...) -> CompletedProcess`
Secure subprocess execution wrapper:
- **NEVER uses `shell=True`** - Critical security requirement
- Validates command is in allowlist
- Ensures all arguments are strings
- Provides timeout protection
- Captures output safely

```python
from tnfr.security import run_command_safely

# Safe command execution
result = run_command_safely(["git", "status"])
result = run_command_safely(["python", "-m", "pytest"])

# Blocked: command not in allowlist
result = run_command_safely(["curl", "http://evil.com"])  # ✗ Raises CommandValidationError
```

### 3. Command Allowlist

Only these commands are permitted:
- `git` - Version control operations
- `python` / `python3` - Python interpreter
- `pip` - Package management
- `twine` - PyPI operations
- `stubgen` - Type stub generation
- `gh` - GitHub CLI

Any attempt to execute other commands will raise `CommandValidationError`.

## Updated Scripts

The following scripts were updated to use secure subprocess execution:

1. **`scripts/check_changelog.py`**
   - Validates git references before use
   - Uses `run_command_safely()` for git operations

2. **`scripts/check_language.py`**
   - Uses `run_command_safely()` for git ls-files

3. **`scripts/generate_stubs.py`**
   - Uses `run_command_safely()` for stubgen execution

4. **`scripts/rollback_release.py`**
   - Validates version strings and git tags
   - Uses `run_command_safely()` for all subprocess calls

5. **`scripts/run_reproducible_benchmarks.py`**
   - Validates benchmark names
   - Uses `run_command_safely()` for benchmark execution

## Security Best Practices Implemented

### ✅ Never Use `shell=True`
All subprocess calls explicitly use `shell=False` (the default for `run_command_safely`).

**Why**: Using `shell=True` allows shell metacharacters like `;`, `|`, `&`, `$()` to execute arbitrary commands.

```python
# UNSAFE - Don't do this
subprocess.run(f"git checkout {user_input}", shell=True)  # ✗ VULNERABLE

# SAFE - Always do this
run_command_safely(["git", "checkout", user_input])  # ✓ SECURE
```

### ✅ Command as List of Strings
Always pass commands as lists, not strings.

```python
# SAFE
run_command_safely(["git", "log", "-n", "1"])  # ✓

# UNSAFE
run_command_safely("git log -n 1")  # ✗ (Also won't work with our validator)
```

### ✅ Validate All User Input
Every user-provided value used in commands must be validated.

```python
# Before
result = subprocess.run(["git", "checkout", args.branch])  # ✗ Unvalidated

# After
validated_branch = validate_git_ref(args.branch)  # ✓ Validated
result = run_command_safely(["git", "checkout", validated_branch])
```

### ✅ Command Allowlisting
Only known-safe commands can be executed.

```python
# Allowed
run_command_safely(["git", "status"])  # ✓
run_command_safely(["python", "-m", "pytest"])  # ✓

# Blocked
run_command_safely(["rm", "-rf", "/"])  # ✗ Not in allowlist
run_command_safely(["curl", "http://attacker.com"])  # ✗ Not in allowlist
```

### ✅ Timeout Protection
All command executions have optional timeout protection.

```python
result = run_command_safely(
    ["python", "long_script.py"],
    timeout=300  # 5 minutes max
)
```

## Testing

### Unit Tests (`tests/unit/test_security.py`)

29 comprehensive tests covering:
- Git reference validation (7 tests)
- Version string validation (3 tests)  
- Path safety validation (5 tests)
- Command execution security (11 tests)
- Integration scenarios (3 tests)

Run tests:
```bash
pytest tests/unit/test_security.py -v
```

### Integration Tests

Existing integration tests pass with secure subprocess:
```bash
pytest tests/integration/test_reproducibility.py -v
```

## TNFR Structural Context

These security utilities maintain TNFR coherence by:

1. **Preserving Operational Integrity**: External commands execute without compromising the TNFR computational environment
2. **Maintaining Structural Boundaries**: Clear separation between trusted code and untrusted input
3. **Enforcing Coherence Constraints**: Input validation respects TNFR structural requirements
4. **Ensuring Reproducibility**: Deterministic command execution supports TNFR's reproducibility requirements

## Migration Guide

To update existing code to use secure subprocess execution:

### Before (Unsafe)
```python
import subprocess

# Direct subprocess call
result = subprocess.run(["git", "status"], capture_output=True, text=True)

# With user input (VULNERABLE)
result = subprocess.run(["git", "checkout", user_branch], shell=False)
```

### After (Secure)
```python
from tnfr.security import run_command_safely, validate_git_ref

# Secure subprocess call
result = run_command_safely(["git", "status"])

# With validated user input (SECURE)
validated_branch = validate_git_ref(user_branch)
result = run_command_safely(["git", "checkout", validated_branch])
```

## Error Handling

All validation failures raise `CommandValidationError` with descriptive messages:

```python
from tnfr.security import CommandValidationError, validate_git_ref

try:
    ref = validate_git_ref(user_input)
except CommandValidationError as e:
    print(f"Invalid input: {e}")
    # Handle error appropriately
```

## References

- **CWE-78**: Improper Neutralization of Special Elements used in an OS Command
- **OWASP**: Command Injection Prevention Cheat Sheet
- **NIST**: Secure Coding Practices

## Security Contact

For security issues, please follow the guidelines in `SECURITY.md`.
