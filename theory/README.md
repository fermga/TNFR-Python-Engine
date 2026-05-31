# TNFR Theory Documentation Hub

Theoretical documentation for Resonant Fractal Nature Theory (TNFR), organized around the nodal equation $\partial\mathrm{EPI}/\partial t = \nu_f \, \Delta\mathrm{NFR}(t)$ and the structural field tetrad $(\Phi_s, |\nabla\phi|, K_\phi, \xi_C)$.

---

## Theory ↔ Examples ↔ SDK Cross-Reference

Every theory document maps to executable examples and SDK entry points:

| Theory Document | Examples | SDK Entry Point |
|-----------------|----------|-----------------|
| [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) | [01](../examples/01_hello_world.py)–[03](../examples/03_network_formation.py), [05](../examples/05_coherence_evolution.py), [06](../examples/06_network_topologies.py), [08](../examples/08_emergent_phenomena.py), [09](../examples/09_visualization_suite.py), [37](../examples/37_operator_tetrad_synergy.py), [39](../examples/39_nodal_equation_decomposition.py) | `TNFR.create()`, `.evolve()`, `.tetrad()`, `.telemetry()` |
| [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md) | [05](../examples/05_coherence_evolution.py), [10](../examples/10_simplified_sdk_showcase.py), [35](../examples/35_tetrad_irreducibility.py), [39](../examples/39_nodal_equation_decomposition.py) | `.tetrad()`, `.fields()`, `.conservation()` |
| [MATHEMATICAL_DYNAMICS_BASIS.md](MATHEMATICAL_DYNAMICS_BASIS.md) | [05](../examples/05_coherence_evolution.py), [10](../examples/10_simplified_sdk_showcase.py), [31](../examples/31_mathematical_constants_basis.py) | `.tetrad()`, `.fields()` |
| [SPIRAL_ATTRACTORS_AND_LOGARITHMIC_DYNAMICS.md](SPIRAL_ATTRACTORS_AND_LOGARITHMIC_DYNAMICS.md) | [08](../examples/08_emergent_phenomena.py), [10](../examples/10_simplified_sdk_showcase.py), [32](../examples/32_spiral_attractors_demo.py) | `.evolve()`, `.tetrad()` |
| [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) | [04](../examples/04_operator_sequences.py), [07](../examples/07_phase_transitions.py), [36](../examples/36_grammar_violation_detector.py), [38](../examples/38_grammar_energy_landscape.py) | `.evolve_grammar_aware()`, `GrammarAwareDynamics` |
| [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) | [17](../examples/17_conservation_law_demo.py), [24](../examples/24_spectral_conservation_demo.py), [34](../examples/34_conservation_protocol_suite.py), [36](../examples/36_grammar_violation_detector.py), [37](../examples/37_operator_tetrad_synergy.py), [38](../examples/38_grammar_energy_landscape.py) | `.conservation()`, `ConservationReport` |
| [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md) | [27](../examples/27_variational_principle_demo.py) | `tnfr.physics.variational` |
| [DISSIPATIVE_AND_OPEN_SYSTEMS.md](DISSIPATIVE_AND_OPEN_SYSTEMS.md) | [28](../examples/28_dissipative_systems_demo.py) | `tnfr.physics.dissipative_conservation` |
| [STRUCTURAL_STABILITY_AND_DYNAMICS.md](STRUCTURAL_STABILITY_AND_DYNAMICS.md) | [29](../examples/29_lyapunov_stability_demo.py), [38](../examples/38_grammar_energy_landscape.py), [39](../examples/39_nodal_equation_decomposition.py) | `.integrity_check()`, `StructuralIntegrityMonitor` |
| [GAUGE_SYMMETRY_AND_UNIFICATION.md](GAUGE_SYMMETRY_AND_UNIFICATION.md) | [26](../examples/26_gauge_structure_demo.py) | `tnfr.physics.gauge` |
| [TNFR_YANG_MILLS_RESEARCH_NOTES.md](TNFR_YANG_MILLS_RESEARCH_NOTES.md) | Y1 finite gauge gap, Y2 U6 sweep, Y3 non-Abelian audit, Y4 finite scaling, Y5 Branch-B closure classification | `tnfr.yang_mills`, `tnfr.physics.gauge`, `tnfr.physics.conservation_gauge_unification` |
| [PHYSICAL_REGIME_CORRESPONDENCES.md](PHYSICAL_REGIME_CORRESPONDENCES.md) | [11](../examples/11_classical_limit_comparison.py)–[15](../examples/15_train_crossing_demo.py) | `tnfr.physics.classical_mechanics`, `tnfr.physics.quantum_mechanics` |
| [EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md](EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md) | [20](../examples/20_eigenmode_tetrad.py), [33](../examples/33_complex_field_unification.py), [unified_fields](../examples/unified_fields_showcase.py) | `.tensor_invariants()`, `.emergent_fields()` |
| [APPLIED_STRUCTURAL_ANALYSIS.md](APPLIED_STRUCTURAL_ANALYSIS.md) | [16](../examples/16_riemann_operator_demo.py)–[25](../examples/25_analytical_convergence_demo.py) | `tnfr.riemann` |
| [TNFR_RIEMANN_RESEARCH_NOTES.md](TNFR_RIEMANN_RESEARCH_NOTES.md) | [16](../examples/16_riemann_operator_demo.py), [18](../examples/18_riemann_convergence_proof.py)–[23](../examples/23_random_ensemble_rmt_demo.py), [25](../examples/25_analytical_convergence_demo.py), 41–76 (P12–P49 ζ-track + χ-twisted L-track), 78–89 (Type-Hygiene B0–B11) | `tnfr.riemann` |
| [TNFR_NAVIER_STOKES_RESEARCH_NOTES.md](TNFR_NAVIER_STOKES_RESEARCH_NOTES.md) | 77–86 NS-series (Taylor–Green, energy inequality, BKM criterion, vortex stretching, Reynolds sweep) | `tnfr.navier_stokes` |
| [CATALOG_TYPE_HYGIENE_PROGRAMME.md](CATALOG_TYPE_HYGIENE_PROGRAMME.md) | 78–89 (B0–B11 type-signature demos) | `tnfr.riemann` (type-signature modules) |
| [STRUCTURAL_OPERATORS.md](STRUCTURAL_OPERATORS.md) | [04](../examples/04_operator_sequences.py), [10](../examples/10_simplified_sdk_showcase.py), [29](../examples/29_lyapunov_stability_demo.py), [36](../examples/36_grammar_violation_detector.py), [37](../examples/37_operator_tetrad_synergy.py), [38](../examples/38_grammar_energy_landscape.py), [39](../examples/39_nodal_equation_decomposition.py) | `.evolve_grammar_aware()`, `.integrity_check()`, `Operator.__call__()` |
| [TNFR_NUMBER_THEORY.md](TNFR_NUMBER_THEORY.md) | [16](../examples/16_riemann_operator_demo.py), [40](../examples/40_arithmetic_number_theory.py) | `tnfr.mathematics.number_theory`, `tnfr.riemann` |
| [REMESH_INFINITY_DERIVATION.md](REMESH_INFINITY_DERIVATION.md) | [77 (REMESH-∞)](../examples/77_remesh_infinity_residue_split_demo.py) | `tnfr.operators.remesh`, `tnfr.physics.conservation` |
| [NUCLEUS_A_PRIME_LADDER_ATLAS.md](NUCLEUS_A_PRIME_LADDER_ATLAS.md) | 41–44 (P12–P15 pipeline) | `tnfr.riemann.von_mangoldt`, `tnfr.riemann.prime_ladder_hamiltonian` |
| [NUCLEUS_B_EQUIVARIANCE_OBSTRUCTIONS.md](NUCLEUS_B_EQUIVARIANCE_OBSTRUCTIONS.md) | 78–89 (Type-Hygiene B0–B11) | `tnfr.riemann` (equivariance diagnostics) |
| [GLOSSARY.md](GLOSSARY.md) | All | All (terminology reference) |

**SDK quick start**: `from tnfr.sdk import TNFR` — see [SDK README](../src/tnfr/sdk/README.md) and [example 10](../examples/10_simplified_sdk_showcase.py).

---

## Document Map

### Canonical References (unchanged, self-contained)

| Document | Scope |
|----------|-------|
| [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) | U1–U6 derivations with canonicity proofs |
| [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md) | Why exactly four structural fields: minimality, completeness, variational confirmation |
| [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) | Noether-like conservation laws: continuity, Ward identities, Lyapunov stability |
| [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md) | Lagrangian/Hamiltonian formulation of structural dynamics |
| [TNFR_RIEMANN_RESEARCH_NOTES.md](TNFR_RIEMANN_RESEARCH_NOTES.md) | TNFR–Riemann program (18 sections + 11 appendices) |
| [TNFR_NAVIER_STOKES_RESEARCH_NOTES.md](TNFR_NAVIER_STOKES_RESEARCH_NOTES.md) | TNFR–Navier–Stokes program (N12–N13 K_φ cascade, Taylor–Green experiments, N15 linkage) |
| [TNFR_YANG_MILLS_RESEARCH_NOTES.md](TNFR_YANG_MILLS_RESEARCH_NOTES.md) | TNFR–Yang–Mills structural gap programme (Y1 finite nodal gauge gap + Y2 U6 sweep + Y3 non-Abelian derivability audit + Y4 finite scaling + Y5 Branch-B closure classification before any Clay-strength claim) |
| [REMESH_INFINITY_DERIVATION.md](REMESH_INFINITY_DERIVATION.md) | **N15 catalog-completeness theorem** (May 2026, COMPLETE): asymptotic projection $\mathcal{R}_\infty$ on $H^2(D)$, projected Noether/energy ($Q_\infty$, $V_\infty$), uniform spectral density, Branch A verdict; 13-op TNFR catalog closed under the REMESH-∞ limit |
| [CATALOG_TYPE_HYGIENE_PROGRAMME.md](CATALOG_TYPE_HYGIENE_PROGRAMME.md) | **Catalog Type-Hygiene Programme** (CLOSED 2026-05-27): twelve orthogonal CDMs B0–B11, all NEGATIVE; composite meta-minimality theorem — 13-operator catalog types minimal and complete; twelve non-canonical research envelopes classified |
| [GLOSSARY.md](GLOSSARY.md) | Operational definitions and terminology |
| [TNFR.pdf](TNFR.pdf) | Historical foundational document |

### Consolidated References

| Document | Scope |
|----------|-------|
| [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) | Nodal equation, structural field tetrad, Universal Tetrahedral Correspondence, emergent invariants, multiscale domain mapping |
| [PHYSICAL_REGIME_CORRESPONDENCES.md](PHYSICAL_REGIME_CORRESPONDENCES.md) | Classical mechanics limit, inertial regime, quantum regime, structural uncertainty, thermodynamic demonstration |
| [APPLIED_STRUCTURAL_ANALYSIS.md](APPLIED_STRUCTURAL_ANALYSIS.md) | Spectral factorization (verified), demonstration-level particle collisions and elemental structure |

### Extended Theoretical Development

| Document | Scope |
|----------|-------|
| [GAUGE_SYMMETRY_AND_UNIFICATION.md](GAUGE_SYMMETRY_AND_UNIFICATION.md) | U(1) gauge symmetry of Ψ, gauge-invariant observables, conservation-gauge unification, spectral conservation |
| [TNFR_YANG_MILLS_RESEARCH_NOTES.md](TNFR_YANG_MILLS_RESEARCH_NOTES.md) | TNFR-native Yang–Mills / mass-gap attack surface: no separate quantum ontology, Y1–Y5 diagnostics implemented, Branch B obstruction classified, non-Abelian derivability and continuum limits open |
| [EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md](EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md) | Extended fields (J_φ, J_ΔNFR), complex geometric field Ψ, derived invariants (χ, 𝒮, 𝒞, ℰ, 𝒬), energy decomposition |
| [DISSIPATIVE_AND_OPEN_SYSTEMS.md](DISSIPATIVE_AND_OPEN_SYSTEMS.md) | Lindblad dissipator, dissipative continuity equation, purity decay, entropy production, dissipation tiers |
| [STRUCTURAL_STABILITY_AND_DYNAMICS.md](STRUCTURAL_STABILITY_AND_DYNAMICS.md) | Lyapunov per-operator bounds, phase transitions, life/autopoiesis, node lifecycle, Hamiltonian, integrity monitor |
| [MATHEMATICAL_DYNAMICS_BASIS.md](MATHEMATICAL_DYNAMICS_BASIS.md) | Four constants (φ, γ, π, e) as minimal basis of mathematical dynamics; classification completeness; inter-constant relations; coherence threshold derivation; Mertens connection |
| [SPIRAL_ATTRACTORS_AND_LOGARITHMIC_DYNAMICS.md](SPIRAL_ATTRACTORS_AND_LOGARITHMIC_DYNAMICS.md) | Logarithmic spiral from nodal equation; golden spiral condition; tetrad signatures; φ as structural attractor; anti-resonance property |
| [STRUCTURAL_OPERATORS.md](STRUCTURAL_OPERATORS.md) | Complete specification of 13 canonical operators: physics, transformations, constants, energy bounds, postconditions, compositions |
| [TNFR_NUMBER_THEORY.md](TNFR_NUMBER_THEORY.md) | Arithmetic TNFR: primality as ΔNFR = 0, canonical constants, pressure components, dual-lever decomposition, spectral factorization, Riemann connection |
| [NUCLEUS_A_PRIME_LADDER_ATLAS.md](NUCLEUS_A_PRIME_LADDER_ATLAS.md) | Canonical prime-ladder construction atlas: P12–P15 pipeline, self-adjoint Hamiltonian, Weil–Guinand verification |
| [NUCLEUS_B_EQUIVARIANCE_OBSTRUCTIONS.md](NUCLEUS_B_EQUIVARIANCE_OBSTRUCTIONS.md) | Equivariance obstructions on G_P14: CCET, Prime-Cancellation Lemma, Euler-Orthogonality Lemma, B1 closure |

### Hierarchy

1. [../AGENTS.md](../AGENTS.md) — Primary theoretical reference (complete framework overview)
2. **This directory** — Specialized derivations and domain applications
3. [../docs/](../docs/) — Implementation specifications
4. [../examples/](../examples/) — Executable demonstrations (01–76 sequential; NS-series 77–86; Type-Hygiene series 77–89; plus `pytorch_cuda_demo.py`, `unified_fields_showcase.py`)
5. [../src/tnfr/sdk/](../src/tnfr/sdk/) — Simplified SDK API

---

## Quick Reference

### Nodal Equation

$$
\frac{\partial \mathrm{EPI}}{\partial t} = \nu_f \cdot \Delta\mathrm{NFR}(t)
$$

### Universal Tetrahedral Correspondence

| Constant | Field | Threshold |
|----------|-------|-----------|
| $\varphi$ (Golden Ratio) | $\Phi_s$ (Structural Potential) | $\Delta\Phi_s < \varphi \approx 1.618$ |
| $\gamma$ (Euler Constant) | $|\nabla\phi|$ (Phase Gradient) | $|\nabla\phi| < \gamma/\pi \approx 0.184$ |
| $\pi$ | $K_\phi$ (Phase Curvature) | $|K_\phi| < 0.9\pi \approx 2.827$ |
| $e$ (Euler Number) | $\xi_C$ (Coherence Length) | $C(r) \sim e^{-r/\xi_C}$ |

### 13 Canonical Operators

AL, EN, IL, OZ, UM, RA, SHA, VAL, NUL, THOL, ZHIR, NAV, REMESH

### Unified Grammar (U1–U6)

- **U1**: Structural Initiation & Closure
- **U2**: Convergence & Boundedness
- **U3**: Resonant Coupling
- **U4**: Bifurcation Dynamics
- **U5**: Multi-Scale Coherence
- **U6**: Structural Potential Confinement

---

## Reading Pathways

### Newcomer (2–4 hours)

1. [../AGENTS.md](../AGENTS.md) — Paradigm and complete overview
2. [GLOSSARY.md](GLOSSARY.md) — Terminology
3. [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) — Core mathematics
4. **Try**: [examples/01_hello_world.py](../examples/01_hello_world.py), [examples/10_simplified_sdk_showcase.py](../examples/10_simplified_sdk_showcase.py)

### Researcher (1–2 weeks)

1. [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) — Grammar derivations → [example 04](../examples/04_operator_sequences.py)
2. [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) — Conservation laws → [example 17](../examples/17_conservation_law_demo.py)
3. [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md) — Lagrangian formulation → [example 27](../examples/27_variational_principle_demo.py)
4. [TNFR_RIEMANN_RESEARCH_NOTES.md](TNFR_RIEMANN_RESEARCH_NOTES.md) — Riemann program → [example 16](../examples/16_riemann_operator_demo.py)
5. [TNFR.pdf](TNFR.pdf) — Historical derivations

### Domain Specialist

- **Physics**: [PHYSICAL_REGIME_CORRESPONDENCES.md](PHYSICAL_REGIME_CORRESPONDENCES.md) → [examples 11–15](../examples/11_classical_limit_comparison.py)
- **Number Theory**: [TNFR_NUMBER_THEORY.md](TNFR_NUMBER_THEORY.md) → [example 40](../examples/40_arithmetic_number_theory.py)
- **Factorization**: [APPLIED_STRUCTURAL_ANALYSIS.md](APPLIED_STRUCTURAL_ANALYSIS.md) → [examples 16–25](../examples/16_riemann_operator_demo.py)
- **Gauge Theory / Symmetry**: [GAUGE_SYMMETRY_AND_UNIFICATION.md](GAUGE_SYMMETRY_AND_UNIFICATION.md) → [example 26](../examples/26_gauge_structure_demo.py)
- **Yang–Mills / Structural Gap**: [TNFR_YANG_MILLS_RESEARCH_NOTES.md](TNFR_YANG_MILLS_RESEARCH_NOTES.md) → Y1–Y5 diagnostics using `tnfr.yang_mills`
- **Field Extensions / Invariants**: [EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md](EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md) → [unified_fields_showcase](../examples/unified_fields_showcase.py)
- **Open Systems / Decoherence**: [DISSIPATIVE_AND_OPEN_SYSTEMS.md](DISSIPATIVE_AND_OPEN_SYSTEMS.md) → [example 28](../examples/28_dissipative_systems_demo.py)
- **Stability / Lifecycle**: [STRUCTURAL_STABILITY_AND_DYNAMICS.md](STRUCTURAL_STABILITY_AND_DYNAMICS.md) → [example 29](../examples/29_lyapunov_stability_demo.py)
- **Self-Optimization**: [../AGENTS.md § Self-Optimizing Dynamics](../AGENTS.md) → [example 30](../examples/30_self_optimization_demo.py)

### Implementation

1. [../src/tnfr/sdk/](../src/tnfr/sdk/) — SDK API (`from tnfr.sdk import TNFR`)
2. [../examples/](../examples/) — Tutorials (01–30 + 2 utility)
3. [../src/tnfr/](../src/tnfr/) — Source code
4. [../tests/](../tests/) — Validation suite
5. [../docs/](../docs/) — Technical specifications

---

## Navigation

- [Repository root](../README.md)
- [Architecture](../ARCHITECTURE.md)
- [Contributing](../CONTRIBUTING.md)
- [Technical docs](../docs/)

---

**Version**: 0.0.3.3 | **Documents**: 24 | **Examples**: 101 | **Authority**: [../AGENTS.md](../AGENTS.md)
