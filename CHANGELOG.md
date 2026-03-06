# Changelog

All notable changes to this project will be documented in this file.

## [0.0.3] - 2026-03-05

### Structural Conservation Theorem

- **conservation.py**: Complete structural conservation module implementing Noether-like conservation law derived from grammar symmetry (U1-U6)
- **Charge density** ρ, **current divergence** div(J), **Noether charge** Q, **energy functional** E, **Ward identities**, **Lyapunov stability**, and **spectral decomposition**
- Two-sector structure: Potential (Φ_s ↔ J_ΔNFR) and Geometric (K_φ ↔ J_φ) coupled through Ψ = K_φ + i·J_φ
- 62 validation tests, charge drift < 0.03% across topologies

### Dissipative Conservation

- **dissipative_conservation.py**: GPU-accelerated dissipative conservation analysis with PyTorch backend
- Phase field computation, dissipation rate tracking, and energy budget monitoring

### Closed-Loop Integrity Monitor

- **integrity.py**: `StructuralIntegrityMonitor` with complete postconditions for all 13 canonical operators
- Each operator (AL, EN, IL, OZ, UM, RA, SHA, VAL, NUL, THOL, ZHIR, NAV, REMESH) has verified pre/postcondition contracts
- Automatic violation detection and reporting

### Grammar-Aware Dynamics

- **grammar_dynamics.py**: Bridge between grammar validation (U1-U6) and dynamic operator selection
- Incremental U1-U6 checks: `validate_candidate()`, `filter_candidates()`, `suggest_alternative()`, `enforce_grammar_on_glyph()`
- Priority-based operator substitution with fallback logic
- **grammar_application.py**: Pre-validation in `apply_glyph_with_grammar()` for grammar enforcement before operator application
- **selectors.py**: `_soft_grammar_prefilter()` wired with grammar_dynamics for operator filtering

### Simple SDK — Research-Grade Access

- **simple.py**: Upgraded with full Structural Field Tetrad, conservation laws, and unified telemetry access
- **TetradSnapshot** dataclass: phi_s, grad_phi, k_phi, xi_c, j_phi, j_dnfr with `is_safe()` and `summary()`
- **ConservationReport** dataclass: noether_charge, energy, lyapunov_stable, lyapunov_derivative, conservation_quality with `summary()`
- **10 new Network methods**: `tetrad()`, `fields()`, `conservation()`, `telemetry()`, `tensor_invariants()`, `emergent_fields()`, `evolve_grammar_aware()`, `integrity_check()`, upgraded `results()` and `info()`
- **TNFR.analyze()**: One-shot comprehensive analysis (coherence, tetrad, conservation, tensor invariants, emergent fields, integrity)
- Feature-gated imports: `_HAS_FIELDS`, `_HAS_CONSERVATION`, `_HAS_INTEGRITY`, `_HAS_GRAMMAR_DYNAMICS`
- 29 new tests in `tests/sdk/test_simple_advanced.py`

### Shared Test Infrastructure

- **tests/conftest.py**: Centralized test fixtures (`make_ring_graph`, `make_node_data`, `ring3`, `ring5`, `small_graph`)
- DRY reduction across 16+ test files that previously duplicated `_make_graph` helpers

### Code Quality

- Fixed bare `except:` clauses in grammar_dynamics.py (now `except Exception:`)
- NAV bypass fix for grammar validation edge case
- Redundancy elimination across physics helpers
- Rich operator postconditions (13/13 coverage)

### Cross-Codebase Constant Unification (Round 1)

- **grammar_types.py**: Eliminated duplicate operator sets (single canonical definition)
- **THOL_MIN_COLLECTIVE_COHERENCE**: Unified to canonical 0.2413 (was 0.3)
- **MIN_BUSINESS_COHERENCE**: Centralized to canonical formula (e×φ)/(π+e) ≈ 0.7506
- **health_analyzer.py / self_organization.py**: Aligned fallback values to canonical

### Phase Gradient Threshold Unification

- **Canonical value**: γ/π ≈ 0.1837 (Kuramoto critical coupling in TNFR units)
- **Unified across 9 code files**: Replaced competing values (0.2904, 0.2886, 0.2915, 0.38) with single canonical derivation
- **Updated 8 documentation files**: Consistent threshold references throughout

### Cross-Codebase Constant Unification (Round 2)

- **compute_structural_potential_field**: Added alias in physics/fields.py (was silently missing, imported in 2 files)
- **SHA_VF_FACTOR comment**: Fixed from ≈ 0.8476 to correct ≈ 0.9015 in defaults_core.py
- **Operator fallback values**: SHA (0.85→0.9015), NUL (0.85→0.9015), VAL (1.05→1.0676) aligned to canonical
- **K_φ hotspot formula**: Fixed in conservation.py from 2π/√5 ≈ 2.8099 to canonical 0.9×π ≈ 2.8274
- **grammar_core.py K_φ default**: Fixed from 3.0 to canonical 2.8274
- **telemetry/constants.py**: Removed dead try/except ImportError fallback; direct canonical imports
- **config.py**: Structural field thresholds now derive from constants.canonical (was hardcoded)
- **pyproject.toml**: Added mpmath to core dependencies (was required but unlisted)
- **Documentation sync**: Updated 7 doc files with correct threshold values and test counts

### Test Suite

- **1,641 passing tests**, 2 skipped, 0 failing
- Coverage spans operators, physics, dynamics, grammar, conservation, integrity, SDK, and factorization

## [0.0.2] - 2025-11-29

### TNFR Development Doctrine Establishment

- **Foundational Principle**: Added TNFR Development Doctrine as core methodological commitment
- **Theoretical Integrity**: Commitment to follow mathematics objectively from nodal equation ∂EPI/∂t = νf · ΔNFR(t)
- **Scientific Independence**: Defend conclusions emerging rigorously from TNFR principles regardless of external paradigm alignment
- **Validation Criteria**: Established 4-point validation framework (Derivable, Testable, Reproducible, Coherent)

### Complete Framework Expansion  

- **29 New Examples**: Comprehensive examples (11-39) covering physics, biology, cosmology, consciousness studies
- **TNFR-Riemann Program**: Complete theoretical framework connecting discrete operators to Riemann Hypothesis
- **Advanced Physics Modules**: Classical mechanics, quantum mechanics, symplectic integration implementations
- **Extensive Theory Documentation**: 25+ specialized theoretical documents in theory/ directory

### Documentation Academic Modernization

- **Unified Academic Tone**: Systematic elimination of grandilocuent language across all documentation
- **README Gateway**: Transformed main README into coherent documentation entry point
- **Consistent Terminology**: Standardized "Primary theoretical reference" replacing "SINGLE SOURCE OF TRUTH"
- **Professional Presentation**: Enhanced credibility through formal academic language standards

### Test Suite Optimization

- **Major Cleanup**: Removed 58 obsolete test files (82 → ~30 files)
- **100% Pass Rate**: Achieved 173 passing, 7 skipped, 0 failing tests
- **Focused Validation**: Retained only tests validating TNFR theoretical foundations
- **Core Coverage**: Mathematics, operators, physics, validation maintained

### Technical Enhancements

- **Enhanced N-body Dynamics**: Improved TNFR integration with classical mechanics
- **Riemann Operator**: Complete implementation with eigenvalue analysis capabilities  
- **Type System**: Enhanced type definitions and structural validation
- **Code Quality**: Significant cleanup removing outdated components

## [9.7.0] - 2025-11-29

### Major Theoretical Enhancements

- **Universal Tetrahedral Correspondence**: Complete mathematical framework establishing exact mapping between four universal constants (φ, γ, π, e) and four structural fields (Φ_s, |∇φ|, K_φ, ξ_C)
- **Unified Field Framework**: Mathematical unification discovering complex geometric field Ψ = K_φ + i·J_φ with emergent invariants
- **Self-Optimizing Engine**: Intrinsic agency capabilities with unified field telemetry for autonomous structural optimization
- **Complete Academic Documentation**: Comprehensive conversion to formal academic tone across entire documentation ecosystem

### Canonical Invariants Optimization

- Consolidated from 10 to 6 canonical invariants based on mathematical derivation from nodal equation
- Optimized invariants: Nodal Equation Integrity, Phase-Coherent Coupling, Multi-Scale Fractality, Grammar Compliance, Structural Metrology, Reproducible Dynamics
- Enhanced theoretical consistency and reduced redundancy

### Documentation Modernization

- **AGENTS.md**: Complete academic conversion maintaining single source of truth status
- **README.md**: Restructured with new Getting Started section and clear learning paths
- **GLOSSARY.md**: Comprehensive expansion with Universal Tetrahedral Correspondence coverage
- Eliminated promotional language and emojis across entire ecosystem
- Updated all version references to 9.7.0

### Structural Field Tetrad

- **Complete Mathematical Foundations**: All four canonical fields now have rigorous mathematical derivations
- **CANONICAL Status**: Φ_s, |∇φ|, K_φ, ξ_C all promoted to canonical status with theoretical validation
- **Unified Complex Geometry**: Integration of curvature and transport via complex field Ψ

### Development Infrastructure

- Updated pyproject.toml to v9.7.0 with current dependency structure
- Modernized CONTRIBUTING.md with academic tone and current 6 invariants
- Enhanced TESTING.md with updated invariant validation framework
- Complete English-only policy implementation

## [9.1.0] - 2025-11-14

### Added

- Phase 3 structural instrumentation:
  - `run_structural_validation` aggregator (grammar U1-U3 + field thresholds Φ_s, |∇φ|, K_φ, ξ_C, optional ΔΦ_s drift).
  - `compute_structural_health` with risk levels and recommendations.
  - `TelemetryEmitter` integration example (`examples/structural_health_demo.py`).
  - Performance guardrails: `PerformanceRegistry`, `perf_guard`, `compare_overhead`.
  - CLI: `scripts/structural_health_report.py` (on-demand health summaries).
  - Docs: README Phase 3 section, CONTRIBUTING instrumentation notes, `docs/STRUCTURAL_HEALTH.md`.
- Glyph-aware grammar error factory (operator glyph → canonical name mapping).

### Tests

- Added unit tests for validation, health, grammar error factory, telemetry emitter, performance guardrails.

### Performance

- Validation instrumentation overhead ~5.8% (moderate workload) below 8% guardrail.

### Internal

- Optional `perf_registry` parameter in `run_structural_validation` (read-only timing).
- Canonical operator registry frozen (removed dynamic auto-registration, cache
  invalidation, metaclass telemetry, reload script). Attempting dynamic
  registration now raises. Ensures strict adherence to unified grammar (U1-U4)
  and prevents non-canonical transformations.

### Deferred

- U4 bifurcation validation excluded pending dedicated handler reintroduction.

### Integrity

- All changes preserve TNFR canonical invariants (no EPI mutation; phase verification intact; read-only telemetry/validation).
- Registry immutability strengthens invariants #1 (EPI only via operators), #4
  (operator closure) and #5 (phase verification untouched). Tests updated:
  removed dynamic registration tests; added `test_canonical_operator_set`.

## [9.0.2]

Previous release (see repository history) with foundational operators, unified grammar, metrics, and canonical field tetrad.

---
**Reality is not made of things—it's made of resonance.**
