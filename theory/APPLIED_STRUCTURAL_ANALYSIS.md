# Applied Structural Analysis

**Status**: Technical reference
**Version**: 0.0.3.3
**Date**: March 2026

---

## 1. Scope

This document describes domain-specific applications of the nodal equation $\partial\mathrm{EPI}/\partial t = \nu_f \, \Delta\mathrm{NFR}(t)$ that are **implemented, tested, and computationally verified** in this repository. The primary application is spectral factorization via Paley–Jacobi graphs, which has a dedicated implementation with 10 test modules, structural verification criteria, and reproducible certificates.

Demonstration-level examples (particle collisions, elemental structure) are listed in §5 with their actual verification status.

---

## 2. Spectral Factorization

### 2.1 Statement

For a composite integer $n > 1$, the Paley–Jacobi graph $G(m)$ constructed from quadratic residues exhibits spectral coherence structure under the canonical decoder sequence $[\mathrm{UM}, \mathrm{RA}, \mathrm{IL}, \mathrm{THOL}]$. The resulting partitions expose periodicities corresponding to the prime factors of $n$.

### 2.2 Algorithm

1. **Graph construction**: Build Paley graph from modulus $m$ using quadratic residues. Maximum graph size: 4097 nodes (configurable via `TNFR_PARTITION_TARGET_SIZE`).
2. **Spectral analysis**: Compute Laplacian eigenvalues $\lambda_2, \ldots, \lambda_k$ and eigenvectors.
3. **Tetrad projection**: Translate eigendata into structural fields ($\Phi_s$, $|\nabla\phi|$, $K_\phi$, $\xi_C$).
4. **Partition planning**: Divide graph into coherent sub-regions (default target size 256, boundary overlap 4 nodes). Per-partition telemetry computed via coherence and sense index.
5. **Nodal decoding**: Apply operator sequence [UM, RA, IL, THOL] per partition. UM/RA restrict coupling to phase-compatible residues (U3); THOL preserves multi-scale identity (U5).
6. **Factor inference**: Detect periodicities in stabilized partitions via structural consistency.
7. **Certification**: Verify factor candidates against 8 structural criteria (§2.3).

### 2.3 Verification Criteria

A factor candidate is TNFR-certified when $\geq 4$ of 8 structural criteria hold and $\geq 50\%$ of partition endorsements are positive:

| Criterion | Threshold | Basis |
|-----------|-----------|-------|
| $\Delta\mathrm{NFR}$ gain | $\geq 0.15$ drop | Nodal equation convergence |
| Coherence ratio | $0.72 \leq r \leq 1.38$ | Structural similarity bounds |
| $\Phi_s$ delta | $\leq 0.35$ | Tetrahedral correspondence confinement |
| Gradient delta | $\leq 0.40$ | Phase desynchronization limits |
| Curvature delta | $\leq 0.45$ | Geometric torsion stability |
| Periodicity confidence | $\geq 0.55$ | Structural mode certification |
| Stabilized fraction | $\geq 0.30$ | Multi-scale coherence (U5) |
| Coverage fraction | $\geq 0.15$ | Spatial completeness |

### 2.4 Confidence Metric

Structural periodicity confidence:

$$
C = \sigma \cdot \delta, \qquad \sigma = \exp\!\bigl(-\mathrm{std\_spacing}/\mathrm{mean\_spacing}\bigr), \quad \delta = \mathrm{mode\_count}/\mathrm{total\_diffs}
$$

### 2.5 Pure Mode

When `TNFR_PURE_MODE=1`, factor certification relies exclusively on structural confidence ($\geq 0.6$) without arithmetic divisibility checks. This isolates the TNFR-specific signal from classical number-theoretic shortcuts.

### 2.6 Output Artifacts

| Artifact | Format | Location |
|----------|--------|----------|
| Operator certificates | JSON | `results/certificates/` |
| Partition manifests | JSON (gzip above threshold) | `results/certificates/partitioned/` |
| Factor telemetry | dict with $\Phi_s$, $|\nabla\phi|$, $K_\phi$, $\xi_C$, $C(t)$, $\Delta\mathrm{NFR}$ | `FactorizationResult.telemetry` |
| Failure diagnostics | JSON | Configurable via `TNFR_FAILURE_TELEMETRY` |

All artifacts include SHA256 hashes and deterministic seeds for reproducibility.

---

## 3. Implementation

| Component | Path |
|-----------|------|
| High-level API | `factorization-lab/tnfr_factorization/api.py` |
| Spectral Paley decoder | `factorization-lab/tnfr_factorization/spectral_paley.py` |
| Partition planner | `factorization-lab/tnfr_factorization/partitioning.py` |
| Seed management | `factorization-lab/tnfr_factorization/seed_management.py` |
| Self-optimization support | `factorization-lab/tnfr_factorization/self_opt_support.py` |
| Feedback integration | `factorization-lab/tnfr_factorization/feedback_integration.py` |
| Snapshot/replay system | `factorization-lab/tnfr_factorization/snapshot_system.py` |
| CLI entry point | `factorization-lab/tnfr_factorization/cli.py` |

### 3.1 Public API

```python
from tnfr_factorization.api import factorize

result = factorize(221, pure=True, trace=True)
# result.candidate_factors -> [13, 17]
# result.tnfr_certified_factors -> [13, 17]
# result.telemetry -> {phi_s, phase_gradient, ...}
# result.certificate_path -> path to operator certificate
```

### 3.2 Integration with TNFR Core

- `TNFRAdvancedFFTEngine` for spectral computation
- `TNFRSelfOptimizingEngine` for operator sequence recommendation
- Operator grammar validation (U1–U6) via `validate_sequence()`
- Strategy registry for resource estimation
- `ArithmeticTNFRParameters` and `ArithmeticStructuralTerms` for number-theoretic integration

---

## 4. Test Coverage

| Test module | Scope |
|-------------|-------|
| `test_spectral_paley.py` | Nodal decoder derives partition factors; FFT integration; certificate grammar compliance |
| `test_verification_robustness.py` | Verification criteria ranges; false-positive resistance parameters; boundary conditions |
| `test_false_positive_verifier.py` | False-positive resistance testing |
| `test_false_positive_methodology.py` | Simulated verification methodology |
| `test_partitioning.py` | Partition planner correctness |
| `test_feedback_integration.py` | Feedback loop integration |
| `test_self_opt_support.py` | Self-optimization engine support |
| `test_seed_management.py` | Deterministic seed management |
| `test_snapshot_system.py` | Snapshot/replay system integrity |
| `test_cli.py` | Command-line interface |

Key verified properties:
- Nodal sequence [UM, RA, IL, THOL] surfaces factors via periodicity (e.g., $221 = 13 \times 17$ yields both certified factors).
- Certificate emission obeys U1–U6 grammar.
- Verification criteria are balanced (strictness score $\in [0.3, 0.8]$).
- False-positive resistance: `min_partition_flags` $\geq 3$, `dnfr_gain_min` $\geq 0.15$, `periodicity_confidence_min` $\geq 0.5$.
- Deterministic seeding ensures reproducibility across runs.

---

## Implementation & Examples

### Executable Demonstrations

| Example | Concept from this document |
|---------|---------------------------|
| [16_riemann_operator_demo.py](../examples/16_riemann_operator_demo.py) | Discrete TNFR-Riemann eigenvalues, critical parameter |
| [18_riemann_convergence_proof.py](../examples/18_riemann_convergence_proof.py) | Spectral convergence σ_c → 1/2 |
| [19_topology_comparison.py](../examples/19_topology_comparison.py) | Cross-topology universality |
| [21_complex_extension_demo.py](../examples/21_complex_extension_demo.py) | Non-Hermitian operator, complex s |
| [22_spectral_zeta_demo.py](../examples/22_spectral_zeta_demo.py) | Spectral zeta, heat kernel, Mellin bridge |
| [23_random_ensemble_rmt_demo.py](../examples/23_random_ensemble_rmt_demo.py) | Random matrix ensembles on prime graphs |
| [25_analytical_convergence_demo.py](../examples/25_analytical_convergence_demo.py) | Analytical proof via PNT + telescoping identity |

### Key Source Modules

- `src/tnfr/riemann/operator.py` — Discrete TNFR-Riemann operators
- `src/tnfr/riemann/spectral_proof.py` — Spectral convergence proofs
- `src/tnfr/riemann/topology.py` — Topology comparison analysis

---

## 6. References

- [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) — Nodal equation and structural fields
- [PHYSICAL_REGIME_CORRESPONDENCES.md](PHYSICAL_REGIME_CORRESPONDENCES.md) — Quantization mechanism
- [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) — U1–U6 derivations
- [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) — Conservation laws
- [GLOSSARY.md](GLOSSARY.md) — Operational definitions
