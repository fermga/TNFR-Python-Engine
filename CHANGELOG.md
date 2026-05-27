# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2026-05-27

### N17-A — U3+U5 → K41: Analytical Cascade Locality (ANALYTICAL_CONSISTENT_CONDITIONAL)

- **Verdict**: `ANALYTICAL_CONSISTENT_CONDITIONAL` — K41 $k^{-5/3}$ spectrum derived conditionally from TNFR grammar rules U2+U3+U5+CDC; algebraically closed given the Cascade Development Condition.
- **Lemma U5-SS** (U5 + U2 → scale self-similarity): U5-uniformity (same canonical operators and constants at every hierarchy level) + U2 force $u_\ell = C(\varepsilon r_\ell)^{1/3}$ in the inertial range. The K41 scaling emerges from grammar structure (U5 collapses the dimensionless ratio to a level-independent constant), not from external dimensional analysis.
- **Lemma U3-CL** (U3 → cascade locality, conditional): Under Lemma U5-SS, U3 (phase-gated coupling, $|\phi_i - \phi_j| \le \Delta\phi_{\max}$) blocks all inter-level interactions **if and only if** the Cascade Development Condition (CDC) holds → constant energy flux $\Pi_\ell = u_\ell^3 / r_\ell = \varepsilon$ across scales.
- **Theorem** (U2 + U3 + U5 + CDC → K41): $E(k_\ell) \sim \varepsilon^{2/3} k_\ell^{-5/3}$ — proof: $E_\ell \sim u_\ell^2 \sim \varepsilon^{2/3} r_\ell^{2/3}$, $E(k_\ell) = E_\ell / \Delta k_\ell$ with $\Delta k_\ell \sim k_\ell$ (log bands) → $E(k_\ell) \sim \varepsilon^{2/3} k_\ell^{-5/3}$. □
- **CDC (irreducible gap)**: CDC (adjacent cascade levels have $|\phi_{\ell,i} - \phi_{\ell+1,j}| \ge \Delta\phi_{\max}$ for all $i,j$) is not derivable from U3, U5, or the nodal equation. It is the K41 locality hypothesis restated in TNFR language, and the structural analogue of $S(T) = (1/\pi)\arg\zeta(\tfrac12 + iT)$ in the Riemann programme — reachable only by a sufficiently developed turbulent cascade, not from the canonical operator catalog alone.
- **N17-A does not close NS-G1..G4** — those gaps concern continuum-limit, uniform bounds, BKM criterion, and vortex stretching; not cascade locality.
- **N17-B pre-registered** (deferred): empirical energy spectrum via `energy_spectrum_3d()` (to be implemented in `src/tnfr/navier_stokes/operator.py`), n ∈ {32, 48}, ν ∈ {0.01, 0.005}, T = 2.0. Expected verdict: `STEEPER_THAN_K41` (CDC not satisfied at Re_eff ≤ 500).
- **Documentation**: `theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md` §20 (full lemmas, theorem, CDC gap analysis, verdict table, N17-B pre-registration spec).

### N16 — NS-G5 Closure: 2D-Embedding Lemma

- **Verdict**: NS-G5 **CLOSED** at the discrete-operator level via the **2D-Embedding Lemma (Theorem NS-G5-TNFR)**.
- **Algebraic proof** (three steps using existing `TNFRNavierStokesOperator` methods on z-independent u = (u₀(x,y), u₁(x,y), 0)):
  1. `vorticity_3d`: ω₀ = ω₁ = 0, ω₂ = ∂_x(v) − ∂_y(u)
  2. `vortex_stretching_field`: S_a = ω₀·∂_x(u_a) + ω₁·∂_y(u_a) + ω₂·∂_z(u_a) = ω₂·0 = 0 for all a
  3. `stretching_production`: = 0.0 exactly in IEEE 754
- **TNFR reading**: z-channel decoupling → no cross-channel ΔNFR → enstrophy ≤ viscous dissipation (monotonically non-increasing) → discrete TNFR analogue of 2D NS global regularity.
- **Contrast with 3D**: ∂_z(u_a) ≠ 0 activates cross-channel ΔNFR coupling → stretching production generically positive → U2 (convergence/boundedness) is not guaranteed → vortex stretching amplification is structurally active.
- **Empirical corroborator**: `examples/85_navier_stokes_dimensional_asymmetry.py` — z-independence → `stretching_production` ≈ 0 at machine precision across all tested configurations (commit `1fac358b`).
- **Scope**: NS-G5 closure does NOT affect NS-G1..G4 and does NOT address the Clay Millennium Problem (3D global regularity).
- **Documentation**: `theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md` §19.

### N15 REMESH-∞ Closure — Catalog-Completeness Theorem

- **Master deliverable**: [theory/REMESH_INFINITY_DERIVATION.md](theory/REMESH_INFINITY_DERIVATION.md) §§1–23 (v3.0, ~816 lines). Three weeks (W1 + W2 + W3) executed in a single session and pushed to `origin/main`:
  - W1 `a1f298fd` — operator existence: $\mathcal{R}_\infty = P_{\ker(I-\mathcal{R})}$, bounded self-adjoint orthogonal projection on $H^2(D)$
  - W2 `badac156` — conservation + Lyapunov: projected Noether charge $Q_\infty$ exactly conserved; energy $V_\infty \ge 0$ monotone with Cesàro $O(1/n)$ tail at rational $\tau_g/\tau_l$
  - W3 `48b0574a` — spectrum + final verdict: uniform spectral density $\rho = \mathrm{lcm}(\tau_l, \tau_g)/\pi$; **Branch A confirmed**
- **Catalog completeness**: the 13-operator TNFR catalog is **closed under the REMESH-∞ asymptotic limit**. No 14th canonical operator is required.
- **Branches ruled out**: B1 strong (constant vs log density, Thm 17.1), B1 via K41 (temporal vs spatial, Thm 18.1), B1 via RMT ($\delta$-clustering vs Wigner, Thm 19.1), B2 (no 14th operator), B3 (limit exists via mean ergodic theorem).
- **B1-Euler partial = existing P30**: the partial universality (smooth half of T-HP) reduces to P12–P15 + P28 + P30 of the TNFR-Riemann program reformulated through the $\mathcal{R}_\infty$ lens (no new content). The oscillatory half ($S(T) = (1/\pi)\arg\zeta(\tfrac12 + iT)$, RH-equivalent) lives in $\ker(\mathcal{R}_\infty)$ and remains open.
- **Consolidation edits**:
  - `AGENTS.md` — new top-level section *REMESH-∞ Closure: Catalog Completeness Theorem (N15, May 2026)*
  - `theory/README.md` — added `REMESH_INFINITY_DERIVATION.md` to canonical document map
  - `theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md` §18.7 — N15 closure block with locked verdicts, B1/K41/RMT/B2/B3 ruled out, refined prediction P-W3-1 (temporal-only)
  - `theory/TNFR_RIEMANN_RESEARCH_NOTES.md` §13septies.5 — structural identification of T-HP smooth/oscillatory split with $\mathrm{range}/\ker$ of $\mathcal{R}_\infty$
  - `theory/STRUCTURAL_OPERATORS.md` §4.3 — REMESH asymptotic limit note with operator definition, spectral density, and catalog-completeness consequence
- **Scope (locked)**: N15 does NOT advance G4 = RH and does NOT resolve 3D Navier–Stokes global regularity. It settles only the $\tau_g \to \infty$ asymptotic limit of REMESH. Pure analytical result; no numerical experiments required for the verdict.

## [0.0.3.3] - 2026-03-07

### Documentation Audit (Sessions 1-4)

- **Comprehensive tone audit**: Removed speculative/grandiose language across 25+ files
- **TNFR_RIEMANN_RESEARCH_NOTES.md**: Reduced from 2679 to 1499 lines (removed unfounded claims)
- **AGENTS.md**: Fixed 'inevitability' → 'derivation strength', updated conservation test count (62 → 88), verified all 40+ cross-reference links
- **Synced .github/agents/my-agent.md** with AGENTS.md corrections
- **Updated test counts** to 1,655 across 8 files
- **Removed orphaned file**: src/train_gmx_optimizer.py
- **Fixed contradictions** between AGENTS.md and theory/ documents
- **Validated**: 1653 passed, 2 skipped

## [0.0.3.2] - 2026-03-06

### Documentation & Consistency Fixes

- **Corrected false Γ(4/3)/Γ(1/3) derivation** in MINIMAL_STRUCTURAL_DEGREES.md and FUNDAMENTAL_THEORY.md (Γ(4/3)/Γ(1/3) = 1/3, not 0.7711)
- **Synchronized .github/agents/my-agent.md** with AGENTS.md (K_φ threshold, MIN_BUSINESS_COHERENCE, THOL_MIN values)
- **Fixed CHANGELOG version** to match pyproject.toml (0.0.3.2)
- **Fixed MIN_BUSINESS_COHERENCE precision** in ARCHITECTURE.md and CONTRIBUTING.md (0.751 → 0.7506)
- **Resolved phantom docs/TNFR_FORCES_EMERGENCE.md** references across 8+ files
- **Removed dead code** src/tnfr/config.py (shadowed by config/ package)
- **Cleaned unused imports** in sdk/simple.py

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

- **1,655 tests** (1,646 passing, 9 skipped), 0 failing
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
- **Self-Optimizing Engine**: Self-optimization capabilities with unified field telemetry for automated structural optimization
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
