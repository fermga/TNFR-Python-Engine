# TNFR Documentation Hub

## Documentation Structure

The `docs/` folder contains specialized technical documentation for TNFR theory and implementation:

### Core Documentation Files

| Document | Status | Purpose |
| ---------- | -------- | --------- |
| **[AGENTS.md](../AGENTS.md)** | **CANONICAL** | Primary reference for TNFR theory |
| **[STRUCTURAL_FIELDS_TETRAD.md](STRUCTURAL_FIELDS_TETRAD.md)** | **CANONICAL** | Formal field definitions (Φ_s, \|∇φ\|, K_φ, ξ_C) |
| **[API_CONTRACTS.md](API_CONTRACTS.md)** | Active | Formal contracts for 13 canonical operators |
| **[CANONICAL_OZ_SEQUENCES.md](CANONICAL_OZ_SEQUENCES.md)** | Active | Guide to dissonance-based operator sequences |
| **[STRUCTURAL_INTERFACE_THEORY.md](STRUCTURAL_INTERFACE_THEORY.md)** | Active | Structural-interface programme: pipelines, fair benchmarks, validated results, limitations |
| **[STRUCTURAL_INTERFACE_THEORY_PLAN.md](STRUCTURAL_INTERFACE_THEORY_PLAN.md)** | Planning | Roadmap for structural-interface diagnostics, benchmarks, and TNFR prescriptions |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | Active | Solutions for operator sequence validation |
| **[grammar/PHYSICS_VERIFICATION.md](grammar/PHYSICS_VERIFICATION.md)** | **CANONICAL** | U1-U6 mathematical proofs from nodal equation |

### Implementation & Operations

| Document | Status | Purpose |
| ---------- | -------- | --------- |
| **[SELF_OPTIMIZATION_INTEGRATION.md](SELF_OPTIMIZATION_INTEGRATION.md)** | Active | Self-optimization pipeline, CLI workflow, promotion guidance |
| **[FACTORIZATION_SCALING_PLAN.md](FACTORIZATION_SCALING_PLAN.md)** | Active | Partitioning, distributed FFT, operator streaming roadmap |
| **[FACTOR_REPLAY_GUIDE.md](FACTOR_REPLAY_GUIDE.md)** | Active | Manifest replay workflow |
| **[STRUCTURAL_HEALTH.md](STRUCTURAL_HEALTH.md)** | Active | Validation and health assessment |
| **[SCALABILITY.md](SCALABILITY.md)** | Active | Multi-scale hierarchical networks |
| **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)** | Active | Reproducibility infrastructure |
| **[OPERATOR_COMPLETENESS.md](OPERATOR_COMPLETENESS.md)** | Active | Operator coverage analysis |
| **[STRUCTURAL_INTERFACE_THEORY_PLAN.md](STRUCTURAL_INTERFACE_THEORY_PLAN.md)** | Planning | Cross-domain benchmark plan for structural-interface analysis |

### External

- **[factorization-lab/README.md](../factorization-lab/README.md)**: Spectral Paley factorization lab and certificate workflows

### Factorization Program

The canonical Paley spectral factorization entry point (`tnfr.factorization.factorize`) lives in the main
package and bootstraps the lab implementation automatically. See the
[`factorization-lab/README.md`](../factorization-lab/README.md) for CLI usage, certificate formats, and
self-optimization guidelines.

---

## Quick Navigation

### For Newcomers

1. Start with **[AGENTS.md](../AGENTS.md)** - Complete TNFR theory
2. Explore **[examples/](../examples/)** - Sequential tutorials
3. Practice with **[SDK](../src/tnfr/sdk/simple.py)** - Simplified API

### For Developers

1. **[API_CONTRACTS.md](API_CONTRACTS.md)** - Operator specifications
2. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions
3. **[STRUCTURAL_FIELDS_TETRAD.md](STRUCTURAL_FIELDS_TETRAD.md)** - Mathematical foundations

### For Factorization Users

1. Start with [`tnfr.factorization.factorize()`](../src/tnfr/factorization/__init__.py) for the canonical API
2. Read [`factorization-lab/README.md`](../factorization-lab/README.md) for telemetry and certificate details
3. Track scaling efforts in [`FACTORIZATION_SCALING_PLAN.md`](FACTORIZATION_SCALING_PLAN.md)

### For Researchers

1. **[STRUCTURAL_FIELDS_TETRAD.md](STRUCTURAL_FIELDS_TETRAD.md)** - Structural field validation
2. **[CANONICAL_OZ_SEQUENCES.md](CANONICAL_OZ_SEQUENCES.md)** - Dissonance patterns
3. **[STRUCTURAL_INTERFACE_THEORY.md](STRUCTURAL_INTERFACE_THEORY.md)** - Structural-interface programme (results + limitations)
4. **[STRUCTURAL_INTERFACE_THEORY_PLAN.md](STRUCTURAL_INTERFACE_THEORY_PLAN.md)** - Structural-interface programme roadmap
5. **[benchmarks/](../benchmarks/)** - Experimental validation

---

## Documentation Status Summary

| Category | Count | Details |
| ---------- | ------- | --------- |
| **Canonical** | 3 | AGENTS.md + STRUCTURAL_FIELDS_TETRAD.md + PHYSICS_VERIFICATION.md |
| **Active** | 11 | Current and maintained |
| **Planning** | 1 | Structural Interface Theory roadmap |

**Total Documentation**: 15 files providing complete TNFR coverage and roadmap planning
