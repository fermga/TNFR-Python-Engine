# Spectral Route: Paley Gap + TNFR

## Background

- **Paley graphs** \(P(n)\) are defined for prime powers \(n \equiv 1 \pmod 4\) by
  connecting nodes when their difference is a quadratic residue modulo \(n\).
- Empirical results show that the **second Laplacian eigenvalue** \(\lambda_2\) is extremely
  rigid when \(n\) is prime. Composite \(n\) introduce noticeable drifts and noisy
  eigenvectors.
- TNFR interprets eigenvectors as **coherence modes**. When \(n\) is prime, the graph forms
  a single coherent attractor (ΔNFR = 0). Composite \(n\) break the attractor into
  substructures associated with the prime factors.

## TNFR Translation

| Paley Concept | TNFR Interpretation |
|---------------|---------------------|
| Laplacian eigenvalues | Structural frequencies $\nu_f$ of the graph-as-network |
| Eigenvectors | EPI configurations and phase patterns |
| Spectral gap ($\lambda_2$) | Minimum dissonance required to leave the prime attractor |
| Clusters / communities | Sub-EPIs aligned with factors $p$, $q$, ... |
| Operator sequence analysis | Dual-lever decomposition — capacity (νf) vs pressure (ΔNFR) levers (§8) |
| Eigenvalue drift magnitude | Arithmetic-recalibrated tetrad thresholds: Φ_s<0.7452, \|∇φ\|<0.2591, K_φ<3.2275 (§7.5) |

## Algorithm Sketch

1. **Construct Graph**: Build Paley / quadratic residue graph for modulus \(n\). When
   Paley is not defined, use TNFR-compatible Cayley graphs built from quadratic residues.
2. **Compute Spectrum**: Obtain Laplacian or normalized Laplacian eigenpairs. Focus on
   \(\lambda_2\), \(\lambda_3\) and corresponding eigenvectors.
3. **Map to TNFR Fields**:
   - Treat eigenvectors as phase assignments; compute |∇φ| and K_φ over graph edges.
   - Use ΔNFR-style pressure derived from spectral energy localized on subsets.
   - Track Φ_s and ξ_C to see if coherence length matches full graph diameter (prime) or
     collapses (composite).
4. **Detect Factor Signatures**:
   - For \(n = pq\), expect eigenvector support concentrated on cosets modulo \(p\) or \(q\).
   - Identify repeating intervals or modular periodicities using TNFR operator sequences
     (UM + RA for coupling, THOL for nested EPIs).
5. **Recover Factors**:
   - Convert detected periodicity/cluster size into candidate factors via gcd operations.
   - Validate using TNFR coherence (ΔNFR cluster ≈ 0) before confirming.

## Open Research Questions

1. **Generalized Paley Graphs**: For composite $n$ or $n \not\equiv 1 \pmod 4$, which
   graph families preserve the spectral rigidity effect?
2. **Noise Control**: How to distinguish factor-induced spectral drift from measurement
   noise? Candidate: use TNFR stabilizers (IL) + arithmetic-recalibrated thresholds (§7.5).
3. **Higher-Order Factors**: Identify whether clusters can separate $n = pqr$ or prime
   powers $p^k$. The pressure decomposition (§6) may distinguish these cases via the
   factorization_pressure vs divisor_pressure ratio.
4. **Complex Field Ψ**: Can the unified field $Ψ = K_φ + i·J_φ$ provide a clearer
   signature of factor multiplicity?
5. **Full Conservation Integration**: Bridge Paley graph node attributes to the canonical
   `compute_noether_charge()` / `compute_energy_functional()` from `conservation.py`,
   replacing the current proxy values (Q = Φ_s + K_φ, E = 0.5·‖tetrad‖²).

## Immediate Experiments

- Build notebooks that compare \(\lambda_2\) for Paley graphs at prime vs composite \(n\).
- Benchmark TNFR coherence metrics on eigenvectors to confirm that Φ_s collapses when
  factors exist.
- Prototype clustering stage using TNFR operators to see if eigenvector-supported nodes
  align with multiples of \(p\) or \(q\).
