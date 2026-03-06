# TNFR Examples — Theory-Linked Tutorial Suite

**Purpose**: Executable demonstrations of TNFR theory, each linked to specific documents in [`theory/`](../theory/).

All dynamics derive from the nodal equation `∂EPI/∂t = νf · ΔNFR(t)`.

---

## Suite Overview (01–30 + 2 utility)

### Foundational Dynamics — [`FUNDAMENTAL_THEORY.md`](../theory/FUNDAMENTAL_THEORY.md)

**SDK**: `TNFR.create(n).ring()`, `.tetrad()`, `.telemetry()`, `TNFR.analyze(net)`

| # | File | Concept |
|---|------|---------|
| 01 | `01_hello_world.py` | Create network, assign EPI/νf/θ, compute C(t) |
| 02 | `02_musical_resonance.py` | Phase synchronization via harmonic coupling |
| 03 | `03_network_formation.py` | Network building strategies and coherence emergence |
| 05 | `05_coherence_evolution.py` | Coherence trajectories under nodal evolution |
| 06 | `06_network_topologies.py` | Topology-dependent dynamics (8+ types) |
| 08 | `08_emergent_phenomena.py` | Collective behaviour from individual nodal equations |
| 09 | `09_visualization_suite.py` | Visual representation of structural dynamics |

### Grammar & Operators — [`UNIFIED_GRAMMAR_RULES.md`](../theory/UNIFIED_GRAMMAR_RULES.md)

**SDK**: `net.evolve_grammar_aware(steps)`, `net.integrity_check()`

| # | File | Concept |
|---|------|---------|
| 04 | `04_operator_sequences.py` | U1–U6 validation: valid vs invalid sequences |
| 07 | `07_phase_transitions.py` | Bifurcation dynamics (U4), critical thresholds |

### SDK & Telemetry — [`FUNDAMENTAL_THEORY.md`](../theory/FUNDAMENTAL_THEORY.md) + [`GLOSSARY.md`](../theory/GLOSSARY.md)

**SDK**: `from tnfr.sdk import TNFR` — primary entry point for all examples below

| # | File | Concept |
|---|------|---------|
| 10 | `10_simplified_sdk_showcase.py` | `TNFR.create()`, `.tetrad()`, `.conservation()`, `.evolve_grammar_aware()` |

### Classical & Quantum Regimes — [`PHYSICAL_REGIME_CORRESPONDENCES.md`](../theory/PHYSICAL_REGIME_CORRESPONDENCES.md)

**SDK**: `net.telemetry()` (coherence regime classification), `net.tetrad()` (field monitoring)

| # | File | Concept |
|---|------|---------|
| 11 | `11_classical_limit_comparison.py` | TNFR vs classical N-body comparison |
| 12 | `12_classical_mechanics_demo.py` | Keplerian orbits from symplectic integrator |
| 13 | `13_quantum_mechanics_demo.py` | Emergent quantization from resonant standing waves |
| 14 | `14_uncertainty_and_interference.py` | Structural uncertainty (ΔForm·Δνf ≥ K), double slit |
| 15 | `15_train_crossing_demo.py` | Free-particle classical kinematics |

### TNFR-Riemann Program — [`TNFR_RIEMANN_RESEARCH_NOTES.md`](../theory/TNFR_RIEMANN_RESEARCH_NOTES.md)

**SDK**: Uses `src/tnfr/riemann/` modules directly (specialised domain, not wrapped in Simple SDK)

| # | File | Concept |
|---|------|---------|
| 16 | `16_riemann_operator_demo.py` | Discrete TNFR-Riemann eigenvalues, critical parameter |
| 18 | `18_riemann_convergence_proof.py` | Spectral convergence σ_c → 1/2 (four lines of attack) |
| 19 | `19_topology_comparison.py` | Cross-topology universality (path/cycle/star/complete/tree/ER) |
| 20 | `20_eigenmode_tetrad.py` | Per-eigenmode structural field tetrad on H^(k)(σ) |
| 21 | `21_complex_extension_demo.py` | Non-Hermitian operator, complex s, pseudo-spectrum |
| 22 | `22_spectral_zeta_demo.py` | Spectral zeta ζ_H(σ,u), heat kernel, Mellin bridge |
| 23 | `23_random_ensemble_rmt_demo.py` | Random matrix ensembles on prime graphs (GOE/GUE/Poisson) |
| 25 | `25_analytical_convergence_demo.py` | Analytical proof σ* → 1/2 via PNT + telescoping identity |

### Conservation & Gauge — [`STRUCTURAL_CONSERVATION_THEOREM.md`](../theory/STRUCTURAL_CONSERVATION_THEOREM.md) + [`GAUGE_SYMMETRY_AND_UNIFICATION.md`](../theory/GAUGE_SYMMETRY_AND_UNIFICATION.md)

**SDK**: `net.conservation()` → `ConservationReport`, `net.tensor_invariants()`

| # | File | Concept |
|---|------|---------|
| 17 | `17_conservation_law_demo.py` | Noether charge, energy, Lyapunov, Ward identities |
| 24 | `24_spectral_conservation_demo.py` | Spectral conservation + grammar compliance at σ = 1/2 |
| 26 | `26_gauge_structure_demo.py` | U(1) gauge symmetry of Ψ, connections, curvature, Wilson loops |

### Variational Mechanics — [`TNFR_VARIATIONAL_PRINCIPLE.md`](../theory/TNFR_VARIATIONAL_PRINCIPLE.md)

**SDK**: Uses `src/tnfr/physics/variational.py` directly

| # | File | Concept |
|---|------|---------|
| 27 | `27_variational_principle_demo.py` | Lagrangian/Hamiltonian, Euler-Lagrange residual, conjugate pairs, symplectic preservation, action functional |

### Dissipative & Open Systems — [`DISSIPATIVE_AND_OPEN_SYSTEMS.md`](../theory/DISSIPATIVE_AND_OPEN_SYSTEMS.md)

**SDK**: Uses `src/tnfr/physics/dissipative_conservation.py` directly

| # | File | Concept |
|---|------|---------|
| 28 | `28_dissipative_systems_demo.py` | Lindblad decoherence, purity decay, entropy production, dissipative regime classification, grammar violations as collapse operators |

### Stability & Emergence — [`STRUCTURAL_STABILITY_AND_DYNAMICS.md`](../theory/STRUCTURAL_STABILITY_AND_DYNAMICS.md)

**SDK**: `net.integrity_check()` → `IntegrityReport`, `net.conservation()` (Lyapunov stability)

| # | File | Concept |
|---|------|---------|
| 29 | `29_lyapunov_stability_demo.py` | All 13 operator Lyapunov bounds, energy class taxonomy, U2 net-contractivity proof, spectral gap, life/autopoiesis emergence |

### Self-Optimisation Dynamics — [AGENTS.md § Self-Optimizing Dynamics](../AGENTS.md)

**SDK**: `TNFRNetwork(G).focus(n).auto_optimize().execute()`, `from tnfr.engines.self_optimization import TNFRSelfOptimizingEngine`

| # | File | Concept |
|---|------|---------|
| 30 | `30_self_optimization_demo.py` | Optimisation landscape, strategy recommendation, experience-based learning, dry-run analysis, conservation feedback loop |

### Extended Fields — [`EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md`](../theory/EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md)

**SDK**: `net.tensor_invariants()`, `net.emergent_fields()`

| # | File | Concept |
|---|------|---------|
| — | `unified_fields_showcase.py` | Ψ = K_φ + i·J_φ, emergent fields χ/𝒮/𝒞, tensor invariants |

### Computational Infrastructure

| # | File | Concept |
|---|------|---------|
| — | `pytorch_cuda_demo.py` | PyTorch GPU acceleration for large-scale field computation |

---

## Learning Paths

### Beginner (15 min)
```bash
python 01_hello_world.py
python 02_musical_resonance.py
python 03_network_formation.py
python 04_operator_sequences.py
```

### Intermediate (1 hr)
```bash
python 05_coherence_evolution.py
python 07_phase_transitions.py
python 10_simplified_sdk_showcase.py
python 12_classical_mechanics_demo.py
python 13_quantum_mechanics_demo.py
python 27_variational_principle_demo.py
```

### Advanced — Riemann Program (2+ hrs)
```bash
python 16_riemann_operator_demo.py
python 18_riemann_convergence_proof.py
python 19_topology_comparison.py
python 20_eigenmode_tetrad.py
python 26_gauge_structure_demo.py
```

### Advanced — Stability & Self-Optimisation (1 hr)
```bash
python 28_dissipative_systems_demo.py
python 29_lyapunov_stability_demo.py
python 30_self_optimization_demo.py
```

---

## Physics Standards

Every example maintains:

1. **Nodal equation traceability**: all dynamics from `∂EPI/∂t = νf · ΔNFR(t)`
2. **Grammar compliance**: operator sequences satisfy U1-U6
3. **Operator semantics**: only the 13 canonical operators modify EPI
4. **Field monitoring**: tetrad (Φ_s, |∇φ|, K_φ, ξ_C) tracked where applicable
5. **Reproducibility**: fixed seeds, deterministic trajectories
6. **Units**: νf in Hz_str, ΔNFR as structural pressure

---

## Documentation Links

- **Theory hub**: [theory/README.md](../theory/README.md)
- **Primary reference**: [AGENTS.md](../AGENTS.md)
- **Grammar rules**: [theory/UNIFIED_GRAMMAR_RULES.md](../theory/UNIFIED_GRAMMAR_RULES.md)
- **Glossary**: [theory/GLOSSARY.md](../theory/GLOSSARY.md)
- **SDK API**: [src/tnfr/sdk/](../src/tnfr/sdk/)
