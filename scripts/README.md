# TNFR Scripts

This directory contains utility scripts for maintaining the TNFR Python Engine codebase.

## Stub Generation (`generate_stubs.py`)

Automated `.pyi` type stub generation and synchronization using `mypy.stubgen`.

### Purpose

This script helps prevent drift between Python implementations (`.py`) and type stubs (`.pyi`) by:
- Detecting missing stub files
- Identifying outdated stubs (when `.py` is newer than `.pyi`)
- Automatically regenerating stubs using `stubgen`
- Providing CI-friendly check modes

### Usage

#### Generate missing stubs

```bash
python scripts/generate_stubs.py
# or
make stubs
```

Generates `.pyi` stub files for any `.py` modules that don't have them.

#### Check for missing stubs (CI mode)

```bash
python scripts/generate_stubs.py --check
# or
make stubs-check
```

Returns exit code 1 if any stubs are missing. Useful in CI pipelines.

#### Check for outdated stubs

```bash
python scripts/generate_stubs.py --check-sync
# or
make stubs-check-sync
```

Returns exit code 1 if any `.py` files are newer than their corresponding `.pyi` stubs.

#### Regenerate outdated stubs

```bash
python scripts/generate_stubs.py --sync
# or
make stubs-sync
```

Regenerates stub files for any modules where the `.py` file is newer than the `.pyi` file.

#### Dry run mode

```bash
python scripts/generate_stubs.py --dry-run
```

Shows what would be done without making changes.

### Integration

#### Makefile Targets

Run `make help` to see all available targets:

- `make stubs` - Generate missing stub files
- `make stubs-check` - Check for missing stubs (used in pre-commit)
- `make stubs-check-sync` - Check for outdated stubs (used in CI)
- `make stubs-sync` - Regenerate outdated stubs
- `make help` - Display all available Make targets

#### Pre-commit Hook

The `.pre-commit-config.yaml` includes a hook that runs `--check` before commits to ensure no stubs are missing.

#### CI/CD

The type-check workflow includes two validation steps:

```yaml
- name: Check stub files exist
  run: python scripts/generate_stubs.py --check

- name: Check stub file synchronization
  run: python scripts/generate_stubs.py --check-sync
```

This dual-layer check ensures:
1. All Python modules have stub files
2. No stubs are outdated (preventing drift)

### How Drift Detection Works

The script compares file modification times:

1. For each `.py` file in `src/tnfr/`
2. Check if a corresponding `.pyi` file exists
3. If it exists, compare `mtime` (modification time)
4. If `.py` mtime > `.pyi` mtime, the stub is outdated

This ensures that whenever a Python file is modified, its stub can be regenerated to stay in sync.

### Factory Pattern Compliance

When writing factory functions, follow these steps to maintain stub synchronization:

1. **Write the implementation** in `.py` with full type annotations
2. **Run stub generation**: `make stubs-sync`
3. **Verify the stub**: Check that `.pyi` matches your signatures
4. **Commit both files**: `.py` and `.pyi` together

See [docs/FACTORY_PATTERNS.md](../docs/FACTORY_PATTERNS.md) for factory function guidelines.

### Troubleshooting

#### Stub generation fails

If `stubgen` fails for a module:
- Ensure the module can be imported (check dependencies)
- Check for syntax errors in the `.py` file
- Verify `mypy` is installed: `pip install mypy`

#### Stub is empty or has generic types

If the generated stub contains only `Any` types:
- Make sure the source has explicit type annotations
- Consider manually refining the stub after generation
- Run `stubgen` with `--verbose` for more details

#### False positives in drift detection

If a file is reported as outdated but hasn't changed:
- This can happen with `git` operations that update mtimes
- Run `make stubs-sync` to regenerate and resolve

### Related Documentation

- [FACTORY_PATTERNS.md](../docs/FACTORY_PATTERNS.md) - Factory function guidelines
- [FACTORY_INVENTORY_2025.md](../docs/FACTORY_INVENTORY_2025.md) - Complete factory audit
- [mypy stubgen documentation](https://mypy.readthedocs.io/en/stable/stubgen.html)

## Other Scripts

### Reproducibility and Benchmarking

- `run_reproducible_benchmarks.py` - Run benchmarks with deterministic seeds and generate checksums for artifacts

This script ensures reproducibility by:
- Setting global seeds for all benchmarks
- Running selected benchmarks with consistent parameters
- Generating SHA256 checksums for all output artifacts
- Creating a manifest file with checksums for verification

Usage:
```bash
# Run all benchmarks with default seed
python scripts/run_reproducible_benchmarks.py

# Run specific benchmarks with custom seed
python scripts/run_reproducible_benchmarks.py \
  --benchmarks comprehensive_cache_profiler full_pipeline_profile \
  --seed 123 \
  --output-dir artifacts

# Verify checksums against existing manifest
python scripts/run_reproducible_benchmarks.py --verify artifacts/manifest.json

# Use make targets
make reproduce           # Run benchmarks with deterministic seeds
make reproduce-verify    # Verify checksums
```

### Documentation and Reference Verification

- `verify_internal_references.py` - Verify internal markdown links in docs, notebooks, and scripts

### Security and Release Management

- `run_pip_audit.sh` - Run pip-audit to scan for dependency vulnerabilities
- `check_changelog.py` - Validate changelog format
- `rollback_release.py` - Rollback failed releases
- `post_security_summary.py` - Post security scan results
- `generate_security_dashboard.py` - Generate security reports

#### Dependency Vulnerability Scanning (`run_pip_audit.sh`)

Scans installed Python dependencies for known security vulnerabilities using pip-audit.

**Purpose:**
- Detect vulnerable dependencies before they reach production
- Provide early feedback to developers during local development
- Match the behavior of the CI/CD pipeline's security scanning

**Usage:**

```bash
# Run basic audit
./scripts/run_pip_audit.sh

# Install pip-audit and run audit
./scripts/run_pip_audit.sh --install

# Generate JSON report for detailed analysis
./scripts/run_pip_audit.sh --json

# Show help
./scripts/run_pip_audit.sh --help
```

**When to Use:**
- Before submitting a pull request that updates dependencies
- After installing new dependencies locally
- When investigating security alerts from CI/CD
- During security review processes

**Integration:**
- Mimics the `.github/workflows/pip-audit.yml` workflow behavior
- Scans the same site-packages directory as the CI/CD pipeline
- Provides both human-readable and JSON output formats

See [SECURITY.md](../SECURITY.md) for the complete security update process.

### Code Quality

- `check_language.py` - Enforce language policy
- `language_policy_data.py` - Language policy configuration
- `verify_consolidation.py` - Verify code consolidation

For more information on individual scripts, see their inline documentation.
