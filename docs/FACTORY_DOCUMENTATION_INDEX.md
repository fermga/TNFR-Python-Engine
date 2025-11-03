# Factory and Type System Documentation Index

> **Navigation guide for all factory pattern and type stub documentation**

This directory contains comprehensive documentation for TNFR factory patterns and the automated type stub generation system. Use this index to find the right document for your needs.

---

## Quick Start

**New to factories?** Start here:
- [Factory Quick Reference](FACTORY_QUICK_REFERENCE.md) - Templates, examples, and common patterns

**Creating a new factory?** Follow these guides in order:
1. [Factory Quick Reference](FACTORY_QUICK_REFERENCE.md) - Get the template
2. [Factory Patterns Guide](FACTORY_PATTERNS.md) - Understand the patterns
3. [Stub Automation Workflow](STUB_AUTOMATION.md) - Generate type stubs

**Troubleshooting?** Check:
- [Stub Automation Workflow](STUB_AUTOMATION.md#troubleshooting) - Common errors and solutions
- [Factory Audit](FACTORY_AUDIT_2025.md) - Known issues and status

---

## Documentation by Purpose

### For Contributors

| Document | Purpose | When to Use |
|----------|---------|-------------|
| [Factory Quick Reference](FACTORY_QUICK_REFERENCE.md) | Templates and cheat sheet | Creating or modifying factories |
| [Stub Automation Workflow](STUB_AUTOMATION.md) | Type stub generation guide | Working with .pyi files |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | General contribution guidelines | First contribution or general questions |

### For Maintainers

| Document | Purpose | When to Use |
|----------|---------|-------------|
| [Factory Patterns Guide](FACTORY_PATTERNS.md) | Comprehensive design patterns | Reviewing PRs, architectural decisions |
| [Factory Inventory](FACTORY_INVENTORY_2025.md) | Complete factory catalog | Auditing compliance, planning refactors |
| [Factory Audit](FACTORY_AUDIT_2025.md) | Audit report with findings | Understanding current state, planning work |
| [Factory Homogenization Summary](FACTORY_HOMOGENIZATION_SUMMARY.md) | Implementation history | Understanding past changes |

### For Advanced Users

| Document | Purpose | When to Use |
|----------|---------|-------------|
| [Mathematics Module README](../src/tnfr/mathematics/README.md) | Module-specific patterns | Working with mathematics factories |
| [Stub Automation Workflow](STUB_AUTOMATION.md#advanced-usage) | Advanced stub techniques | Custom stub needs, automation customization |

---

## Document Descriptions

### Primary Guides

#### [Factory Quick Reference](FACTORY_QUICK_REFERENCE.md)

**Quick access guide** with templates and common patterns.

**Contents:**
- Naming convention table (make_*, build_*, create_*)
- Complete factory templates ready to copy
- Validation checklist
- Common patterns and examples
- Testing templates
- Stub generation quick commands
- Troubleshooting tips

**Best for:** Day-to-day factory development

#### [Factory Patterns Guide](FACTORY_PATTERNS.md)

**Comprehensive design patterns** for all factory types.

**Contents:**
- Core principles (TNFR fidelity, validation, type safety)
- Detailed naming conventions with rationale
- Standard factory structure (validation → construction → verification)
- Backend integration patterns
- Common patterns with explanations
- Testing requirements and examples
- Migration checklist for existing code

**Best for:** Understanding the "why" behind patterns, reviewing code

#### [Stub Automation Workflow](STUB_AUTOMATION.md)

**Complete automation guide** for type stub generation.

**Contents:**
- How the automation works (scripts, hooks, CI)
- Workflow scenarios with step-by-step examples
- Stub generation details and examples
- Synchronization logic explanation
- Comprehensive troubleshooting section
- Best practices and anti-patterns
- Advanced usage techniques

**Best for:** Working with type stubs, setting up automation, debugging issues

### Reference Documents

#### [Factory Inventory](FACTORY_INVENTORY_2025.md)

**Complete catalog** of all factory functions in the codebase.

**Contents:**
- Full inventory by category (operators, generators, higher-order)
- Compliance status for each factory
- Detailed analysis of each function
- Stub file locations
- Test coverage information

**Best for:** Finding existing factories, auditing compliance

#### [Factory Audit](FACTORY_AUDIT_2025.md)

**Audit report** documenting factory pattern compliance.

**Contents:**
- Factory inventory with compliance status
- Type stub synchronization issues (historical)
- Automation improvements made
- CI/CD integration details

**Best for:** Understanding what was fixed, historical context

#### [Factory Homogenization Summary](FACTORY_HOMOGENIZATION_SUMMARY.md)

**Implementation summary** of factory standardization work.

**Contents:**
- Completed work (stub generation, pre-commit, docs, tests)
- Current state assessment
- Verification results
- Files changed
- Metrics (stubs generated, tests added)

**Best for:** Understanding what was done and why

### Module-Specific Guides

#### [Mathematics Module README](../src/tnfr/mathematics/README.md)

**Module organization** and factory patterns for mathematics.

**Contents:**
- Module overview and organization
- Backend abstraction usage
- Operator factory examples
- Generator construction examples
- Transform contracts
- Structural invariants
- Complete usage examples

**Best for:** Working with mathematics module factories

---

## Workflows by Task

### Creating a New Factory

1. Read [Factory Quick Reference](FACTORY_QUICK_REFERENCE.md) - Get template
2. Copy appropriate template (make_*, build_*, create_*)
3. Implement with validation and verification
4. Generate stub: `make stubs`
5. Add tests following test template
6. Review checklist in Quick Reference

### Modifying an Existing Factory

1. Make changes to `.py` file
2. Check if stub needs update: `make stubs-check-sync`
3. Regenerate if needed: `make stubs-sync`
4. Run tests: `pytest tests/`
5. Commit both `.py` and `.pyi` together

### Reviewing Factory Code

1. Check [Factory Patterns Guide](FACTORY_PATTERNS.md) for patterns
2. Verify compliance with [Factory Inventory](FACTORY_INVENTORY_2025.md)
3. Ensure tests cover validation checklist
4. Verify stub synchronization: `make stubs-check-sync`

### Debugging Stub Issues

1. Check [Stub Automation Workflow](STUB_AUTOMATION.md#troubleshooting)
2. Verify installation: `pip install -e .[typecheck]`
3. Try regeneration: `make stubs-sync`
4. Check CI logs if problem persists

### Auditing Compliance

1. Review [Factory Inventory](FACTORY_INVENTORY_2025.md)
2. Check status in [Factory Audit](FACTORY_AUDIT_2025.md)
3. Run validation: `make stubs-check && make stubs-check-sync`
4. Run tests: `pytest tests/mathematics/test_factory_patterns.py`

---

## Key Concepts

### Factory Naming

| Prefix | Returns | Purpose | Example |
|--------|---------|---------|---------|
| `make_*` | Objects | Create validated operator instances | `make_coherence_operator` |
| `build_*` | Arrays/Data | Construct generators and data structures | `build_delta_nfr` |
| `create_*` | Nodes/Factories | Create TNFR nodes or factory functions | `create_nfr` |

### Type Stub Workflow

```
1. Write/modify .py → 2. Generate .pyi → 3. Review → 4. Commit both
        ↓                     ↓                ↓            ↓
   Implementation    make stubs-sync    git diff     git add .py .pyi
```

### Automation Layers

```
Pre-commit Hook → CI Check (missing) → CI Check (sync) → Mypy Validation
     (local)           (GitHub)             (GitHub)         (local/CI)
```

---

## Quick Command Reference

```bash
# Factory development
make help                  # Show all available commands

# Stub generation
make stubs                 # Generate missing stubs
make stubs-check           # Check for missing stubs (CI)
make stubs-check-sync      # Check if stubs outdated (CI)
make stubs-sync            # Regenerate outdated stubs

# Testing
pytest tests/mathematics/test_factory_patterns.py  # Run factory tests
mypy src/tnfr                                      # Type check

# Pre-commit
pre-commit install         # Install hooks
pre-commit run --all-files # Run all hooks manually
```

---

## Related Documentation

- [AGENTS.md](../AGENTS.md) - TNFR paradigm and structural invariants
- [CONTRIBUTING.md](../CONTRIBUTING.md) - General contribution guidelines
- [ARCHITECTURE.md](../ARCHITECTURE.md) - Overall project architecture
- [README.md](../README.md) - Project overview and getting started

---

## Getting Help

If you can't find what you need:

1. **Search this index** for the right document
2. **Check the document's table of contents** (most have detailed TOC)
3. **Try the Quick Reference** for common patterns
4. **Review the Troubleshooting section** in Stub Automation
5. **Check GitHub issues** for known problems
6. **Open a new issue** with your question

---

## Document History

| Date | Event |
|------|-------|
| 2025-11-03 | Factory pattern documentation completed |
| 2025-11-03 | Stub automation system documented |
| 2025-11-03 | Quick reference guide added |
| 2025-11-03 | This index created |

---

## Maintenance

These documents are living documentation:

- Update when patterns change
- Add examples as they're discovered
- Keep troubleshooting sections current
- Review quarterly for accuracy
- Archive obsolete content to history sections

**Last reviewed:** 2025-11-03
