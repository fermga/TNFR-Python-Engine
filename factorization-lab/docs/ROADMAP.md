# TNFR Factorization Roadmap

## Phase 0 — Research Capture (complete)

- Summarize Paley spectral gap findings and tie them to TNFR coherence metrics.
- Define deliverables and repository scaffolding (README, docs, package skeleton).

## Phase 1 — Spectral Instrumentation

- Implement Paley graph builder with safe fallbacks when \(n\) is not prime power.
- Integrate Laplacian computation (scipy/networkx) plus TNFR telemetry adapters.
- Produce baseline plots of \(\lambda_2\) vs \(n\) (prime vs composite).

## Phase 2 — TNFR Coherence Mapping

- Convert eigenvectors to TNFR structural fields (Φ_s, |∇φ|, K_φ, ξ_C, Si).
- Detect coherence clusters using canonical operator sequences `[UM, RA, IL]`.
- Record telemetry episodes showing how ΔNFR behaves when clusters align with factors.

## Phase 3 — Factor Recovery Logic (in progress)

- Map cluster periodicity to candidate factors using modular arithmetic.
- Design dissonance-driven search (OZ, ZHIR) to refine ambiguous cluster sizes.
- Output factorization certificates: TNFR metrics + gcd validations.

### Phase 3 Milestone — TNFR Certification (November 2025)

- **Completeness proof**: Formalize that the nodal decoder `[UM, RA, IL, THOL]` converges to a unique partition-periodicity pattern for every Paley/Jacobi configuration (see `theory/TNFR_NODAL_FACTOR_COMPLETENESS.md`).
- **Deterministic verification**: Replace the gcd "proof" step with TNFR-native invariants (ΔNFR attenuation, Φ_s confinement, coherence ratios). Certificates now include `tnfr_verification_snapshot` blocks showing which factors are certified solely via nodal evidence.
- **Operational tooling**: `SpectralAnalysisResult` exposes `tnfr_certified_factors`, and the full-spectrum benchmark records verification metadata for each run.

**Status**: gcd checks remain available as optional backstops, but publication artifacts (certificates, playbook) now treat TNFR verification as the canonical proof of factor identity.

### Current canonical flow

1. **Input** `n`
2. **Construct** Paley/Jacobi graph for lifted modulus
3. **Partition** graph deterministically with telemetry per block
4. **Execute** nodal sequence `[UM, RA, IL, THOL]` per partition
5. **Record** operator-strategy plan + nodal decoding results
6. **Emit** certificates with per-partition before/after TNFR fields + invariant report
7. **Verify** factors optionally via arithmetic (gcd)

## Phase 4 — Benchmarks and Validation

- Build benchmark suite mirroring `primality-test/benchmarks/` (random semiprimes,
  RSA-style composites, smooth numbers).
- Compare against classical algorithms (trial division, Pollard-ρ, QS) for reference.
- Document success rates, runtimes, and TNFR telemetry plots.
- **Range coverage (new)**: `full_spectrum_factorization.py` now sweeps contiguous ranges (150–170 by default, plus CLI-configurable spans) and emits certificates per `n`, establishing empirical coverage beyond curated samples.

## Phase 5 — Packaging & Publication

- Finalize API (`tnfr_factorization.factorize(n)`), CLI, and notebooks.
- Prepare Zenodo metadata, release notes, and integration guide for main TNFR repo.
- Share results with TNFR-Riemann program to explore cross-links with spectral operators.
