# Changelog

All notable changes to this project will be documented in this file.

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
