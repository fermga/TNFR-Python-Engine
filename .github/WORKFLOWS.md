# GitHub Actions Workflows

This document describes the CI/CD workflows configured for this repository.

## Active Workflows

### Core Development Workflows

#### 1. CI (`ci.yml`)
**Triggers:** Push and PR to main/master branches  
**Purpose:** Core continuous integration testing and quality checks  
**Jobs:**
- **Format check**: Runs pre-commit hooks (black, isort, pydocstyle)
- **Type check and static analysis**: Runs flake8, mypy, pyright, vulture, and language policy enforcement
- **Changelog fragments**: Enforces changelog fragments for PRs
- **Tests**: Runs pytest across Python 3.9-3.13 with coverage reporting

**Status:** ✅ Working (has legitimate doctest failures in codebase, not workflow issues)

#### 2. Docs (`docs.yml`)
**Triggers:** Push and PR to main branch  
**Purpose:** Build and validate documentation  
**Jobs:**
- Build MkDocs site with strict mode
- Run Sphinx doctests
- Perform link checking

**Status:** ✅ Working (has legitimate doctest failures in codebase, not workflow issues)

### Security Workflows

#### 3. CodeQL Analysis (`codeql-analysis.yml`)
**Triggers:** Push/PR to main/master, weekly schedule (Monday 3 AM UTC)  
**Purpose:** Advanced security scanning using GitHub CodeQL  
**Jobs:**
- Initialize CodeQL for Python
- Autobuild and analyze
- Upload SARIF results to GitHub Security tab

**Status:** ✅ Working correctly

#### 4. SAST Lint (`sast-lint.yml`)
**Triggers:** Push to main, PRs, manual dispatch  
**Purpose:** Static Application Security Testing  
**Jobs:**
- **Bandit scan**: Security vulnerability detection in Python code
- **Semgrep scan**: Pattern-based security and bug detection

**Status:** ✅ Working correctly

#### 5. Dependency Vulnerability Audit (`pip-audit.yml`)
**Triggers:** Push/PR to main/master, weekly schedule (Monday 5 AM UTC)  
**Purpose:** Scan Python dependencies for known vulnerabilities  
**Jobs:**
- Install all project dependencies
- Run pip-audit against installed packages
- Upload JSON report artifact
- Fail build if vulnerabilities detected

**Status:** ✅ Working correctly

### Quality Assurance Workflows

#### 6. Reproducibility Check (`reproducibility.yml`)
**Triggers:** Push/PR to main/master, manual dispatch  
**Purpose:** Verify benchmark reproducibility (TNFR canonical requirement)  
**Jobs:**
- Run benchmarks twice with same seed
- Compare manifest checksums
- Verify identical results

**Status:** ✅ Working correctly

#### 7. Performance Regression (`performance-regression.yml`)
**Triggers:** PRs to main/master, manual dispatch  
**Purpose:** Detect performance degradations  
**Jobs:**
- Run performance regression test suite
- Mark slow tests appropriately

**Status:** ✅ Working correctly

#### 8. Verify Internal References (`verify-references.yml`)
**Triggers:** Push/PR affecting markdown/notebook files  
**Purpose:** Validate internal documentation links  
**Jobs:**
- Scan 985+ internal references
- Report broken links

**Status:** ⚠️ Working but reports 75 broken links (codebase issue, not workflow)

### Infrastructure Workflows

#### 9. Lint Workflows (`lint-workflows.yml`)
**Triggers:** Push/PR affecting workflows, tools, or scripts  
**Purpose:** Validate workflow configurations  
**Jobs:**
- Check for invalid Bandit format usage (SARIF not supported)
- Verify bandit_to_sarif.py converter exists

**Status:** ✅ Working correctly

#### 10. Release (`release.yml`)
**Triggers:** Push to main, manual dispatch  
**Purpose:** Automated semantic release process  
**Jobs:**
- **Prepare**: Detect next version, compile changelog, apply tags
- **Publish**: Build distributions, sign with GPG, publish to PyPI, create GitHub release

**Status:** ✅ Working correctly (intentionally skips when no release needed)

## Removed Workflows

The following workflows were removed during cleanup:

### 1. `bandit.yml` (Removed)
**Reason:** Duplicate functionality - Bandit already runs in `sast-lint.yml`

### 2. `black-duck-security-scan-ci.yml` (Removed)
**Reason:** Requires paid Black Duck service with configuration not available in repository

### 3. `security-dashboard.yml` (Removed)
**Reason:** Optional aggregator workflow with configuration issues; individual security scans (CodeQL, SAST, pip-audit) already provide comprehensive coverage

## Workflow Best Practices

1. **Concurrency Control**: All workflows use `concurrency` groups with `cancel-in-progress: true` to save CI resources
2. **Caching**: Python dependencies and pip are cached to speed up builds
3. **Matrix Testing**: CI tests across Python 3.9-3.13 to ensure compatibility
4. **SARIF Integration**: Security workflows upload results to GitHub Security tab
5. **Artifact Preservation**: Important outputs (reports, logs) are uploaded as artifacts

## Known Issues

These issues exist in the codebase, not in workflow configurations:

1. **Doctest Failures**: 
   - Files: `docs/source/api/api_mapping.rst`, `docs/source/how_to_reproduce_results.rst`
   - Cause: Unexpected warnings and validation errors
   - Impact: CI and Docs workflows report failures

2. **Broken Documentation Links**: 
   - 75 broken internal references
   - Common patterns: missing API pages, moved/renamed files
   - Impact: Verify Internal References workflow reports failures

## Maintenance Notes

- All workflows follow TNFR canonical requirements (reproducibility, traceability, English-only)
- Security workflows run on schedule to catch newly disclosed vulnerabilities
- Release workflow requires semantic commit messages for version bumping
- No workflow credentials are committed to the repository (use GitHub Secrets)
