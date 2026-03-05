# TNFR Documentation Hub

## 📚 Documentation Structure

The `docs/` folder contains specialized technical documentation for TNFR theory and implementation:

### Core Documentation Files

| Document | Status | Purpose |
|----------|--------|---------|
| **[AGENTS.md](../AGENTS.md)** | ✅ **CANONICAL** | Primary reference for TNFR theory |
| **[STRUCTURAL_FIELDS_TETRAD.md](STRUCTURAL_FIELDS_TETRAD.md)** | ✅ **CANONICAL** | Formal field definitions (Φ_s, \|∇φ\|, K_φ, ξ_C) |
| **[CANONICAL_OZ_SEQUENCES.md](CANONICAL_OZ_SEQUENCES.md)** | ✅ Active | Guide to dissonance-based operator sequences |
| **[API_CONTRACTS.md](API_CONTRACTS.md)** | ✅ Active | Formal contracts for 13 canonical operators |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | ✅ Active | Solutions for operator sequence validation |
| **[TNFR_FORCES_EMERGENCE.md](TNFR_FORCES_EMERGENCE.md)** | 🔬 Research | Emergent force-like interactions (non-canonical) |

### Utility Documentation

- **[TNFR_MATHEMATICS_REFERENCE.md](TNFR_MATHEMATICS_REFERENCE.md)**: Mathematical foundations aggregator
- **[STRUCTURAL_HEALTH.md](STRUCTURAL_HEALTH.md)**: Validation and health assessment
- **[SCALABILITY.md](SCALABILITY.md)**: Multi-scale hierarchical networks
- **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**: Reproducibility infrastructure
- **[SECURITY_CONFIG_GUIDE.md](SECURITY_CONFIG_GUIDE.md)**: Security configuration
- **[OPERATOR_COMPLETENESS.md](OPERATOR_COMPLETENESS.md)**: Operator coverage analysis
- **[factorization-lab/README.md](../factorization-lab/README.md)**: Spectral Paley factorization lab and certificate workflows (canonical API `tnfr.factorization.factorize()`)
- **[FACTORIZATION_SCALING_PLAN.md](FACTORIZATION_SCALING_PLAN.md)**: Roadmap for partitioning, distributed FFT backends, and operator streaming

### Factorization Program

The canonical Paley spectral factorization entry point (`tnfr.factorization.factorize`) lives in the main
package and bootstraps the lab implementation automatically. See the
[`factorization-lab/README.md`](../factorization-lab/README.md) for CLI usage, certificate formats, and
self-optimization guidelines that now apply to all factorization runs triggered through the
core package.

### Obsolete Files

- **[INTERACTIONS_GUIDE.md](INTERACTIONS_GUIDE.md)**: ⚠️ **OBSOLETE** - References non-existent assets (see file for migration path)

---

## 📋 Quick Navigation

### For Newcomers

1. Start with **[AGENTS.md](../AGENTS.md)** - Complete TNFR theory
2. Explore **[examples/](../examples/)** - Sequential tutorials 01-10
3. Practice with **[SDK](../src/tnfr/sdk/simple.py)** - Simplified API

### For Developers

1. **[API_CONTRACTS.md](API_CONTRACTS.md)** - Operator specifications
2. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions
3. **[STRUCTURAL_FIELDS_TETRAD.md](STRUCTURAL_FIELDS_TETRAD.md)** - Mathematical foundations

### For Factorization Users

1. Start with [`tnfr.factorization.factorize()`](../src/tnfr/factorization/__init__.py) for the canonical API
2. Read [`factorization-lab/README.md`](../factorization-lab/README.md) for telemetry and certificate details
3. Inspect [`factorization-lab/tests/`](../factorization-lab/tests/) for reference workflows
4. Track scaling efforts in [`FACTORIZATION_SCALING_PLAN.md`](FACTORIZATION_SCALING_PLAN.md)

### For Researchers

1. **[TNFR_FORCES_EMERGENCE.md](TNFR_FORCES_EMERGENCE.md)** - Force-like interactions
2. **[CANONICAL_OZ_SEQUENCES.md](CANONICAL_OZ_SEQUENCES.md)** - Dissonance patterns
3. **[benchmarks/](../benchmarks/)** - Experimental validation

---

## 📊 Documentation Status Summary

| Category | Count | Status |
|----------|-------|---------|
| **Canonical** | 2 | ✅ AGENTS.md + STRUCTURAL_FIELDS_TETRAD.md |
| **Active** | 8 | Current and maintained |
| **Research** | 1 | Exploratory (TNFR_FORCES_EMERGENCE.md) |
| **Obsolete** | 1 | Asset references broken (INTERACTIONS_GUIDE.md) |

**Total Active Documentation**: 11 files providing complete TNFR coverage
