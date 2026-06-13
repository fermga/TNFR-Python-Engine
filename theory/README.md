# TNFR Theory Documentation Hub

Theoretical documentation for Resonant Fractal Nature Theory (TNFR), organized around the nodal equation $\partial\mathrm{EPI}/\partial t = \nu_f \, \Delta\mathrm{NFR}(t)$ and the structural field tetrad $(\Phi_s, |\nabla\phi|, K_\phi, \xi_C)$.

---

## Theory ↔ Examples ↔ SDK Cross-Reference

Every theory document maps to executable examples and SDK entry points:

| Theory Document | Examples | SDK Entry Point |
|-----------------|----------|-----------------|
| [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) | [01](../examples/01_foundations/01_hello_world.py)–[03](../examples/01_foundations/03_network_formation.py), [05](../examples/01_foundations/05_coherence_evolution.py), [06](../examples/01_foundations/06_network_topologies.py), [08](../examples/01_foundations/08_emergent_phenomena.py), [09](../examples/01_foundations/09_visualization_suite.py), [37](../examples/02_physics_regimes/37_operator_tetrad_synergy.py), [39](../examples/02_physics_regimes/39_nodal_equation_decomposition.py) | `TNFR.create()`, `.evolve()`, `.tetrad()`, `.telemetry()` |
| [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md) | [05](../examples/01_foundations/05_coherence_evolution.py), [10](../examples/01_foundations/10_simplified_sdk_showcase.py), [35](../examples/02_physics_regimes/35_tetrad_irreducibility.py), [39](../examples/02_physics_regimes/39_nodal_equation_decomposition.py) | `.tetrad()`, `.fields()`, `.conservation()` |
| [MATHEMATICAL_DYNAMICS_BASIS.md](MATHEMATICAL_DYNAMICS_BASIS.md) | [05](../examples/01_foundations/05_coherence_evolution.py), [10](../examples/01_foundations/10_simplified_sdk_showcase.py), [31](../examples/02_physics_regimes/31_mathematical_constants_basis.py) | `.tetrad()`, `.fields()` |
| [SPIRAL_ATTRACTORS_AND_LOGARITHMIC_DYNAMICS.md](SPIRAL_ATTRACTORS_AND_LOGARITHMIC_DYNAMICS.md) | [08](../examples/01_foundations/08_emergent_phenomena.py), [10](../examples/01_foundations/10_simplified_sdk_showcase.py), [32](../examples/02_physics_regimes/32_spiral_attractors_demo.py) | `.evolve()`, `.tetrad()` |
| [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) | [04](../examples/01_foundations/04_operator_sequences.py), [07](../examples/01_foundations/07_phase_transitions.py), [36](../examples/02_physics_regimes/36_grammar_violation_detector.py), [38](../examples/02_physics_regimes/38_grammar_energy_landscape.py) | `.evolve_grammar_aware()`, `GrammarAwareDynamics` |
| [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) | [17](../examples/02_physics_regimes/17_conservation_law_demo.py), [24](../examples/03_riemann_zeta/24_spectral_conservation_demo.py), [34](../examples/02_physics_regimes/34_conservation_protocol_suite.py), [36](../examples/02_physics_regimes/36_grammar_violation_detector.py), [37](../examples/02_physics_regimes/37_operator_tetrad_synergy.py), [38](../examples/02_physics_regimes/38_grammar_energy_landscape.py) | `.conservation()`, `ConservationReport` |
| [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md) | [27](../examples/02_physics_regimes/27_variational_principle_demo.py) | `tnfr.physics.variational` |
| [DISSIPATIVE_AND_OPEN_SYSTEMS.md](DISSIPATIVE_AND_OPEN_SYSTEMS.md) | [28](../examples/02_physics_regimes/28_dissipative_systems_demo.py) | `tnfr.physics.dissipative_conservation` |
| [STRUCTURAL_STABILITY_AND_DYNAMICS.md](STRUCTURAL_STABILITY_AND_DYNAMICS.md) | [29](../examples/02_physics_regimes/29_lyapunov_stability_demo.py), [38](../examples/02_physics_regimes/38_grammar_energy_landscape.py), [39](../examples/02_physics_regimes/39_nodal_equation_decomposition.py) | `.integrity_check()`, `StructuralIntegrityMonitor` |
| [GAUGE_SYMMETRY_AND_UNIFICATION.md](GAUGE_SYMMETRY_AND_UNIFICATION.md) | [26](../examples/02_physics_regimes/26_gauge_structure_demo.py) | `tnfr.physics.gauge` |
| [TNFR_YANG_MILLS_RESEARCH_NOTES.md](TNFR_YANG_MILLS_RESEARCH_NOTES.md) | Y1 finite gauge gap, Y2 U6 sweep, Y3 non-Abelian audit, Y4 finite scaling, Y5 Branch-B closure classification | `tnfr.yang_mills`, `tnfr.physics.gauge`, `tnfr.physics.conservation_gauge_unification` |
| [PHYSICAL_REGIME_CORRESPONDENCES.md](PHYSICAL_REGIME_CORRESPONDENCES.md) | [11](../examples/02_physics_regimes/11_classical_limit_comparison.py)–[15](../examples/02_physics_regimes/15_train_crossing_demo.py) | `tnfr.physics.classical_mechanics`, `tnfr.physics.quantum_mechanics` |
| [EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md](EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md) | [20](../examples/03_riemann_zeta/20_eigenmode_tetrad.py), [33](../examples/02_physics_regimes/33_complex_field_unification.py), [unified_fields](../examples/08_emergent_geometry/unified_fields_showcase.py) | `.tensor_invariants()`, `.emergent_fields()` |
| [APPLIED_STRUCTURAL_ANALYSIS.md](APPLIED_STRUCTURAL_ANALYSIS.md) | [16](../examples/03_riemann_zeta/16_riemann_operator_demo.py)–[25](../examples/03_riemann_zeta/25_analytical_convergence_demo.py) | `tnfr.riemann` |
| [TNFR_RIEMANN_RESEARCH_NOTES.md](TNFR_RIEMANN_RESEARCH_NOTES.md) | [16](../examples/03_riemann_zeta/16_riemann_operator_demo.py), [18](../examples/03_riemann_zeta/18_riemann_convergence_proof.py)–[23](../examples/03_riemann_zeta/23_random_ensemble_rmt_demo.py), [25](../examples/03_riemann_zeta/25_analytical_convergence_demo.py), 41–76 (P12–P49 ζ-track + χ-twisted L-track), 78–89 (Type-Hygiene B0–B11) | `tnfr.riemann` |
| [TNFR_NAVIER_STOKES_RESEARCH_NOTES.md](TNFR_NAVIER_STOKES_RESEARCH_NOTES.md) | 77–86 NS-series (Taylor–Green, energy inequality, BKM criterion, vortex stretching, Reynolds sweep) | `tnfr.navier_stokes` |
| [CATALOG_TYPE_HYGIENE_PROGRAMME.md](CATALOG_TYPE_HYGIENE_PROGRAMME.md) | 78–89 (B0–B11 type-signature demos) | `tnfr.riemann` (type-signature modules) |
| [STRUCTURAL_OPERATORS.md](STRUCTURAL_OPERATORS.md) | [04](../examples/01_foundations/04_operator_sequences.py), [10](../examples/01_foundations/10_simplified_sdk_showcase.py), [29](../examples/02_physics_regimes/29_lyapunov_stability_demo.py), [36](../examples/02_physics_regimes/36_grammar_violation_detector.py), [37](../examples/02_physics_regimes/37_operator_tetrad_synergy.py), [38](../examples/02_physics_regimes/38_grammar_energy_landscape.py), [39](../examples/02_physics_regimes/39_nodal_equation_decomposition.py) | `.evolve_grammar_aware()`, `.integrity_check()`, `Operator.__call__()` |
| [TNFR_NUMBER_THEORY.md](TNFR_NUMBER_THEORY.md) | [16](../examples/03_riemann_zeta/16_riemann_operator_demo.py), [40](../examples/07_number_theory/40_arithmetic_number_theory.py), [94](../examples/07_number_theory/94_generative_number_construction.py)–[97](../examples/07_number_theory/97_goldbach_additive_multiplicative.py), [100](../examples/07_number_theory/100_prime_families_orbits.py)–[102](../examples/07_number_theory/102_nodal_flow_primes_equilibria.py) | `tnfr.mathematics.number_theory`, `tnfr.riemann` |
| [REMESH_INFINITY_DERIVATION.md](REMESH_INFINITY_DERIVATION.md) | [77 (REMESH-∞)](../examples/05_type_hygiene/77_remesh_infinity_residue_split_demo.py) | `tnfr.operators.remesh`, `tnfr.physics.conservation` |
| [NUCLEUS_A_PRIME_LADDER_ATLAS.md](NUCLEUS_A_PRIME_LADDER_ATLAS.md) | 41–44 (P12–P15 pipeline) | `tnfr.riemann.von_mangoldt`, `tnfr.riemann.prime_ladder_hamiltonian` |
| [NUCLEUS_B_EQUIVARIANCE_OBSTRUCTIONS.md](NUCLEUS_B_EQUIVARIANCE_OBSTRUCTIONS.md) | 78–89 (Type-Hygiene B0–B11) | `tnfr.riemann` (equivariance diagnostics) |
| Emergent geometry (AGENTS.md § Symplectic Substrate / Structural Diffusion) | [98](../examples/08_emergent_geometry/98_emergent_symplectic_substrate.py), [99](../examples/08_emergent_geometry/99_structural_diffusion.py), [103](../examples/08_emergent_geometry/103_emergent_substrate_meets_riemann.py)–[108](../examples/08_emergent_geometry/108_emergent_field_generating_structure.py), [112](../examples/08_emergent_geometry/112_structure_predicts_coherence_flow.py) | `tnfr.physics.symplectic_substrate`, `tnfr.physics.structural_diffusion` |
| [TNFR_P_VS_NP_RESEARCH_NOTES.md](TNFR_P_VS_NP_RESEARCH_NOTES.md) | [109](../examples/09_millennium/109_p_vs_np_coherence_synthesis.py) | `tnfr` (gradient-flow relaxation) |
| [TNFR_BSD_RESEARCH_NOTES.md](TNFR_BSD_RESEARCH_NOTES.md) | [110](../examples/09_millennium/110_bsd_rank_structural_pressure.py) | `tnfr` (arithmetic structural pressure) |
| [TNFR_HODGE_RESEARCH_NOTES.md](TNFR_HODGE_RESEARCH_NOTES.md) | [111](../examples/09_millennium/111_hodge_discrete_and_honest_gap.py) | `tnfr` (discrete Hodge theory) |
| [GLOSSARY.md](GLOSSARY.md) | All | All (terminology reference) |

**SDK quick start**: `from tnfr.sdk import TNFR` — see [SDK README](../src/tnfr/sdk/README.md) and [example 10](../examples/01_foundations/10_simplified_sdk_showcase.py).

---

## Document Map

### Canonical References (unchanged, self-contained)

| Document | Scope |
|----------|-------|
| [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) | U1–U6 derivations with canonicity proofs |
| [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md) | Why exactly four structural fields: minimality, completeness, variational confirmation |
| [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) | Noether-like conservation laws: continuity, Ward identities, Lyapunov stability |
| [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md) | Lagrangian/Hamiltonian formulation of structural dynamics |
| [TNFR_RIEMANN_RESEARCH_NOTES.md](TNFR_RIEMANN_RESEARCH_NOTES.md) | TNFR–Riemann program: P12–P49 ζ-track + χ-twisted L-track attack surface, §13 obstruction analysis (T-HP, B1/B2/B3 branches), §19 milestone map |
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

### Millennium Problem Programs (TNFR-native reformulations)

Each program restates a Clay Millennium Problem in TNFR-native terms and classifies its obstruction. **None claims a solution** — every entry carries an explicit honest-scope statement (open gap or Branch B/B3 negative). See each document's status section for the precise boundary.

| Document | Problem | Status |
|----------|---------|--------|
| [TNFR_RIEMANN_RESEARCH_NOTES.md](TNFR_RIEMANN_RESEARCH_NOTES.md) | Riemann Hypothesis | Prime-ladder Hamiltonian + Weil–Guinand verified; smooth/oscillatory rescaling split. **G4 = RH open** at the T-HP boundary |
| [TNFR_NAVIER_STOKES_RESEARCH_NOTES.md](TNFR_NAVIER_STOKES_RESEARCH_NOTES.md) | Navier–Stokes regularity | $K_\phi$ cascade, enstrophy budget, native $\nu\lambda_k$ dissipation; global regularity **open** |
| [TNFR_YANG_MILLS_RESEARCH_NOTES.md](TNFR_YANG_MILLS_RESEARCH_NOTES.md) | Yang–Mills mass gap | Finite nodal gauge gap (Y1–Y5); Branch-B obstruction, non-Abelian + continuum limit **open** |
| [TNFR_P_VS_NP_RESEARCH_NOTES.md](TNFR_P_VS_NP_RESEARCH_NOTES.md) | P vs NP | Coherence verification $O(|E|)$ vs synthesis trapping in dissonance basins; mirrors P≠NP, **not a proof** (Branch B) |
| [TNFR_BSD_RESEARCH_NOTES.md](TNFR_BSD_RESEARCH_NOTES.md) | Birch–Swinnerton-Dyer | $a_p$ as structural pressure, rank-separation reproduced; GL(1)→GL(2) gap **open** (Branch B) |
| [TNFR_HODGE_RESEARCH_NOTES.md](TNFR_HODGE_RESEARCH_NOTES.md) | Hodge conjecture | Discrete Hodge theory (harmonic = Betti, Eckmann); strong negative — blind to (p,p) bigrading + algebraicity (Branch B3-leaning) |

### Hierarchy

1. [../AGENTS.md](../AGENTS.md) — Primary theoretical reference (complete framework overview)
2. **This directory** — Specialized derivations and domain applications
3. [../docs/](../docs/) — Implementation specifications
4. [../examples/](../examples/) — 125 executable demonstrations across 10 thematic subfolders (`01_foundations`, `02_physics_regimes`, `03_riemann_zeta`, `04_riemann_L_twisted`, `05_type_hygiene`, `06_navier_stokes`, `07_number_theory`, `08_emergent_geometry`, `09_millennium`, `10_applications`). Each file keeps a stable global number as identifier; the folder gives the theme.
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

1. [../AGENTS.md](../AGENTS.md) — Paradigm and complete framework overview
2. [GLOSSARY.md](GLOSSARY.md) — Terminology and operational definitions
3. [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) — Nodal equation, tetrad, Universal Tetrahedral Correspondence
4. **Try**: [01_foundations/01_hello_world.py](../examples/01_foundations/01_hello_world.py), [01_foundations/10_simplified_sdk_showcase.py](../examples/01_foundations/10_simplified_sdk_showcase.py)

### Intermediate (1–2 days)

1. [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md) — Why exactly four structural fields
2. [STRUCTURAL_OPERATORS.md](STRUCTURAL_OPERATORS.md) — Full specification of the 13 canonical operators
3. [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) — U1–U6 derivations with canonicity proofs
4. [MATHEMATICAL_DYNAMICS_BASIS.md](MATHEMATICAL_DYNAMICS_BASIS.md) — The four constants (φ, γ, π, e) as minimal basis
5. **Try**: [01_foundations/04_operator_sequences.py](../examples/01_foundations/04_operator_sequences.py), [02_physics_regimes/37_operator_tetrad_synergy.py](../examples/02_physics_regimes/37_operator_tetrad_synergy.py)

### Advanced (physics & conservation)

1. [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) — Noether-like conservation laws
2. [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md) — Lagrangian/Hamiltonian formulation
3. [EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md](EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md) — Complex field Ψ and derived invariants
4. [GAUGE_SYMMETRY_AND_UNIFICATION.md](GAUGE_SYMMETRY_AND_UNIFICATION.md) — U(1) gauge structure
5. [PHYSICAL_REGIME_CORRESPONDENCES.md](PHYSICAL_REGIME_CORRESPONDENCES.md) — Classical/quantum regime correspondences
6. **Try**: [02_physics_regimes/17_conservation_law_demo.py](../examples/02_physics_regimes/17_conservation_law_demo.py), [08_emergent_geometry/98_emergent_symplectic_substrate.py](../examples/08_emergent_geometry/98_emergent_symplectic_substrate.py)

### Researcher (Millennium programs)

1. [TNFR_NUMBER_THEORY.md](TNFR_NUMBER_THEORY.md) — Arithmetic TNFR foundations
2. [NUCLEUS_A_PRIME_LADDER_ATLAS.md](NUCLEUS_A_PRIME_LADDER_ATLAS.md) + [NUCLEUS_B_EQUIVARIANCE_OBSTRUCTIONS.md](NUCLEUS_B_EQUIVARIANCE_OBSTRUCTIONS.md) — Prime-ladder construction and obstructions
3. [TNFR_RIEMANN_RESEARCH_NOTES.md](TNFR_RIEMANN_RESEARCH_NOTES.md) — The full Riemann program (read §19 first for the milestone map)
4. [REMESH_INFINITY_DERIVATION.md](REMESH_INFINITY_DERIVATION.md) + [CATALOG_TYPE_HYGIENE_PROGRAMME.md](CATALOG_TYPE_HYGIENE_PROGRAMME.md) — Catalog-completeness theorems
5. The other Millennium programs: [Navier–Stokes](TNFR_NAVIER_STOKES_RESEARCH_NOTES.md), [Yang–Mills](TNFR_YANG_MILLS_RESEARCH_NOTES.md), [P vs NP](TNFR_P_VS_NP_RESEARCH_NOTES.md), [BSD](TNFR_BSD_RESEARCH_NOTES.md), [Hodge](TNFR_HODGE_RESEARCH_NOTES.md)

**Honest-scope reminder**: the Millennium programs are TNFR-native *reformulations* with classified obstructions, not solutions. Each document states its exact open gap.

### Researcher (1–2 weeks)

1. [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) — Grammar derivations → [example 04](../examples/01_foundations/04_operator_sequences.py)
2. [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) — Conservation laws → [example 17](../examples/02_physics_regimes/17_conservation_law_demo.py)
3. [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md) — Lagrangian formulation → [example 27](../examples/02_physics_regimes/27_variational_principle_demo.py)
4. [TNFR_RIEMANN_RESEARCH_NOTES.md](TNFR_RIEMANN_RESEARCH_NOTES.md) — Riemann program → [example 16](../examples/03_riemann_zeta/16_riemann_operator_demo.py)
5. [TNFR.pdf](TNFR.pdf) — Historical derivations

### Domain Specialist

- **Physics**: [PHYSICAL_REGIME_CORRESPONDENCES.md](PHYSICAL_REGIME_CORRESPONDENCES.md) → [examples 11–15](../examples/02_physics_regimes/11_classical_limit_comparison.py)
- **Number Theory**: [TNFR_NUMBER_THEORY.md](TNFR_NUMBER_THEORY.md) → [example 40](../examples/07_number_theory/40_arithmetic_number_theory.py)
- **Factorization**: [APPLIED_STRUCTURAL_ANALYSIS.md](APPLIED_STRUCTURAL_ANALYSIS.md) → [examples 16–25](../examples/03_riemann_zeta/16_riemann_operator_demo.py)
- **Gauge Theory / Symmetry**: [GAUGE_SYMMETRY_AND_UNIFICATION.md](GAUGE_SYMMETRY_AND_UNIFICATION.md) → [example 26](../examples/02_physics_regimes/26_gauge_structure_demo.py)
- **Yang–Mills / Structural Gap**: [TNFR_YANG_MILLS_RESEARCH_NOTES.md](TNFR_YANG_MILLS_RESEARCH_NOTES.md) → Y1–Y5 diagnostics using `tnfr.yang_mills`
- **Field Extensions / Invariants**: [EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md](EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md) → [unified_fields_showcase](../examples/08_emergent_geometry/unified_fields_showcase.py)
- **Open Systems / Decoherence**: [DISSIPATIVE_AND_OPEN_SYSTEMS.md](DISSIPATIVE_AND_OPEN_SYSTEMS.md) → [example 28](../examples/02_physics_regimes/28_dissipative_systems_demo.py)
- **Stability / Lifecycle**: [STRUCTURAL_STABILITY_AND_DYNAMICS.md](STRUCTURAL_STABILITY_AND_DYNAMICS.md) → [example 29](../examples/02_physics_regimes/29_lyapunov_stability_demo.py)
- **Self-Optimization**: [../AGENTS.md § Self-Optimizing Dynamics](../AGENTS.md) → [example 30](../examples/02_physics_regimes/30_self_optimization_demo.py)

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
