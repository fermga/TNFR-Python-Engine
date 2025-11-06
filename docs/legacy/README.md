# Legacy Documentation Files

This directory contains reference documentation that provides detailed implementation notes, audits, and historical context. These files are preserved for completeness but are **not part of the main documentation navigation**.

## Status

These files are **reference material only**. The main documentation has been reorganized into a coherent structure:

- **Getting Started**: `docs/source/getting-started/`
- **User Guide**: `docs/source/user-guide/`
- **Advanced Topics**: `docs/source/advanced/`
- **API Reference**: `docs/source/api/`
- **Examples**: `docs/source/examples/`

## What's Here

This directory contains technical deep-dives and implementation details that are:
- **Too detailed** for general users
- **Implementation-specific** (internal patterns, audits, inventories)
- **Historical** (phase summaries, migration records)
- **Specialized** (factory patterns, caching strategies, dependency analysis)

## Main Documentation

For current, organized documentation, see:

📖 **Main Index**: [`docs/source/index.rst`](../source/index.rst)

🚀 **Getting Started**: [`docs/source/getting-started/README.md`](../source/getting-started/README.md)

📚 **Documentation Index**: [`docs/DOCUMENTATION_INDEX.md`](../DOCUMENTATION_INDEX.md)

## File Categories in Parent Directory

The following files in `docs/` are **retained as detailed references**:

### Factory System Documentation
- `FACTORY_AUDIT_2025.md` - Factory implementation audit
- `FACTORY_DOCUMENTATION_INDEX.md` - Factory documentation index
- `FACTORY_HOMOGENIZATION_SUMMARY.md` - Factory homogenization summary
- `FACTORY_INVENTORY_2025.md` - Complete factory inventory
- `FACTORY_PATTERNS.md` - Detailed factory patterns guide
- `FACTORY_QUICK_REFERENCE.md` - Factory quick reference

**Summary in main docs**: [`advanced/PERFORMANCE_OPTIMIZATION.md`](../source/advanced/PERFORMANCE_OPTIMIZATION.md) (§3 Factory Patterns)

### Cache Optimization Documentation
- `CACHE_OPTIMIZATION.md` - Cache optimization guide
- `CACHE_OPTIMIZATION_ANALYSIS.md` - Cache optimization analysis
- `CACHING_STRATEGY.md` - Caching strategy document

**Summary in main docs**: [`advanced/PERFORMANCE_OPTIMIZATION.md`](../source/advanced/PERFORMANCE_OPTIMIZATION.md) (§2 Caching Strategies)

### Dependency and Module Analysis
- `DEPENDENCY_ANALYSIS.md` - Dependency analysis
- `MODULE_DEPENDENCY_ANALYSIS.md` - Module dependency analysis

**Summary in main docs**: [`advanced/PERFORMANCE_OPTIMIZATION.md`](../source/advanced/PERFORMANCE_OPTIMIZATION.md) (§7 Dependency Management)

### Implementation Summaries
- `CONSOLIDATION_AUDIT.md` - Consolidation audit
- `ISSUE_RESOLUTION_SUMMARY.md` - Issue resolution summary
- `MIGRATION_OPTIMIZATION.md` - Migration optimization notes
- `STUB_AUTOMATION.md` - Stub automation documentation
- `TEST_OPTIMIZATION.md` - Test optimization guide
- `TESTING_COMPATIBILITY.md` - Testing compatibility notes
- `UTILITY_MIGRATION.md` - Utility migration documentation

### Other Technical Documentation
- `API_CONTRACTS.md` - API contract specifications
- `SECURITY_CONFIG_GUIDE.md` - Security configuration guide
- `SCALABILITY.md` - Scalability considerations
- `REPRODUCIBILITY.md` - Reproducibility infrastructure
- `OPTIMIZATION_GUIDE.md` - General optimization guide
- `backends.md` - Backend implementation details
- `ci.md` - CI/CD documentation
- `utils_reference.md` - Utilities reference

### Archive
- `archive/` - Historical phase summaries and implementation records

## When to Reference These Files

### If you're...

**A user wanting to learn TNFR**: 
→ Use the main documentation starting at [`getting-started/README.md`](../source/getting-started/README.md)

**Troubleshooting a problem**:
→ Start with [`user-guide/TROUBLESHOOTING.md`](../source/user-guide/TROUBLESHOOTING.md)

**Optimizing performance**:
→ See [`advanced/PERFORMANCE_OPTIMIZATION.md`](../source/advanced/PERFORMANCE_OPTIMIZATION.md), then reference factory/cache docs if needed

**Contributing to TNFR**:
→ Read [`CONTRIBUTING.md`](../../CONTRIBUTING.md) and [`AGENTS.md`](../../AGENTS.md)

**Researching implementation details**:
→ **Then** consult these detailed technical documents

**Working on factory system**:
→ Reference `FACTORY_*.md` files for complete patterns and audits

**Investigating cache behavior**:
→ Reference `CACHE_*.md` files for implementation details

## Migration Status

The reorganization follows TNFR principles:

- ✅ **Coherence over fragmentation**: New structure is organized by user journey
- ✅ **Minimal perturbation**: Existing files preserved as references
- ✅ **Operational fractality**: Same organizational pattern at all scales
- ✅ **Complete traceability**: All original information retained

## Navigation

From here:
- **Up**: [`docs/`](../) - Parent documentation directory
- **Main Index**: [`docs/source/index.rst`](../source/index.rst)
- **Getting Started**: [`docs/source/getting-started/README.md`](../source/getting-started/README.md)
- **Documentation Index**: [`docs/DOCUMENTATION_INDEX.md`](../DOCUMENTATION_INDEX.md)

---

**Note**: This directory may be renamed or relocated in future versions. The main navigation will always be at `docs/source/index.rst`.
