# TNFR Number Theory Guide

Consolidated reference for applying TNFR physics to arithmetic structures. This guide documents the ΔNFR prime criterion, operator mapping on the arithmetic graph (UM/RA), telemetry fields, and comparisons with classical number-theoretic signals.


## 1. Arithmetic TNFR model

- Nodes: natural numbers n ∈ ℕ, 2 ≤ n ≤ N.
- Per node attributes: (EPI_n, νf_n, ΔNFR_n, phase φ_n).
- Edges: arithmetic neighborhood (multiplicative and additive motifs; implementation details in `src/tnfr/mathematics/number_theory.py`).

EPI is the coherent structural form; νf is the structural frequency (Hz_str); φ is the phase; ΔNFR is the nodal reorganization gradient (structural pressure). All changes must occur via TNFR operators and respect the unified grammar (U1–U6 in `AGENTS.md`).


## 2. ΔNFR: the prime criterion

We use the canonical arithmetic decomposition of reorganization pressure:

ΔNFR(n) = ζ·(ω(n) − 1) + η·(τ(n) − 2) + θ·(σ(n)/n − (1 + 1/n))

- ω(n): number of prime factors with multiplicity (Ω). 
- τ(n): divisor count. 
- σ(n): sum of divisors. 
- ζ, η, θ: positive structural weights; the canonical choice used in this repo is ζ=1.0, η=0.8, θ=0.6.

For primes p:
- ω(p) = 1,
- τ(p) = 2,
- σ(p) = 1 + p.

Therefore ΔNFR(p) = 0 exactly. This yields the TNFR prime criterion:

Prime if and only if ΔNFR(n) = 0.

Empirically, ROC/AUC calibration up to N=10000 and N=100000 shows AUC=1.0 with the decision rule “prime ⇔ ΔNFR ≤ 0”. See `benchmarks/arith_delta_nfr_roc.py` and the notebook `examples/tnfr_prime_checker.ipynb`.

Local coherence is c_n = 1/(1 + |ΔNFR_n|), which equals 1 for primes and <1 for composites.

### 2.1 Structural terms and prime certificates

`src/tnfr/mathematics/number_theory.py` now exposes canonical dataclasses so downstream code can reason about TNFR arithmetic without duplicating formulas:

- `ArithmeticStructuralTerms`: encapsulates τ(n), σ(n), ω(n). Retrieve via `net.get_structural_terms(n)` and convert to dict with `.as_dict()` when exporting telemetry.
- `PrimeCertificate`: immutable proof object produced by `net.get_prime_certificate(n)` or `net.detect_prime_candidates(..., return_certificates=True)`. It stores ΔNFR, tolerance, the structural terms, and the component-level pressures (factorization, divisor, sigma) that sum to ΔNFR.
- `ArithmeticTNFRFormalism`: static helpers for EPI, νf, ΔNFR, component breakdowns, local coherence, and symbolic expressions. Methods are shared between runtime code and documentation so the physics is written exactly once.

Example usage:

```python
from tnfr.mathematics import ArithmeticTNFRNetwork

net = ArithmeticTNFRNetwork(max_number=50)
certificate = net.get_prime_certificate(29)
assert certificate.structural_prime
print(certificate.components)  # {'factorization_pressure': 0.0, 'divisor_pressure': 0.0, ...}
```

Certificates can be generated for arbitrary subsets with `net.generate_prime_certificates(numbers=[...])`, enabling JSONL exports or notebook tables that include ΔNFR components without recalculating the physics.


## 3. Operators on the arithmetic graph (UM/RA)

TNFR requires that EPI changes only via operators and that coupling respects phase physics (U3). We map arithmetic dynamics to two canonical operators:

- UM (Coupling): creates phase-verified structural links. Contract: allow exchange only if |φ_i − φ_j| ≤ Δφ_max. This enforces resonant coupling (U3) and prevents destructive interference.
- RA (Resonance): propagates activation coherently across coupled links. Contract: propagate pattern identity (no uncontrolled mutation), often modeled as a gain–decay update with normalization.

Minimal sequence for propagation: [UM, RA, IL]. After coupling and resonance, apply Coherence (IL) to reduce |ΔNFR| and satisfy boundedness (U2). If any destabilizers are introduced (e.g., Expansion or Dissonance for exploration), follow U4a/U4b with handlers and stabilizers (THOL/IL) to avoid chaotic bifurcation.

Implementation highlights (see `src/tnfr/mathematics/number_theory.py`):
- `apply_coupling(delta_phi_max)`: marks edges as coupled (UM) if the phase difference passes the U3 check.
- `resonance_step(activation, gain, decay, delta_phi_max, normalize)`: one step of RA across UM links.
- `resonance_from_primes(...)`: seeds activation on primes and runs multiple RA steps; exposes telemetry (means, fractions ≥ threshold, correlation with prime indicator).


## 4. Telemetry: structural field tetrad

Monitor the CANONICAL fields for safety and insight (see `AGENTS.md`):
- Structural potential Φ_s (global)
- Phase gradient |∇φ| (local desynchronization)
- Phase curvature K_φ (geometric confinement)
- Coherence length ξ_C (spatial correlation scale)

Typical usage in arithmetic:
- High |∇φ| pinpoints local stress regions.
- |K_φ| ≥ 3.0 flags confinement or mutation-prone loci.
- Φ_s and ξ_C complement global/spatial views of reorganization.


## 5. Comparisons with classical signals

- Möbius μ(n): oscillatory indicator of square-freeness and parity of prime factors; correlates with structural desynchronization but does not directly pin ΔNFR=0.
- Von Mangoldt Λ(n): isolates prime powers; useful for periodic/spike analysis but not a direct structural pressure.
- τ(n), σ(n): components of ΔNFR; alone, they blur prime identity; combined with ω(n) in the TNFR equation, they cancel exactly for primes.
- Euler’s φ(n): measures totatives; interesting for coupling motifs but not a direct zero-pressure indicator.

Key distinction: ΔNFR integrates factorization pressure (ω−1), divisor pressure (τ−2), and sigma normalization (σ/n − (1 + 1/n)) so that primes are fixed points (ΔNFR=0). Classical functions highlight features but do not provide this exact cancellation.


## 6. Reproducibility and workflows

- Prime checking (equations only): see `examples/tnfr_prime_checker.ipynb` and `scripts/tnfr_is_prime.py`; installable console entry `tnfr-is-prime` is configured in `pyproject.toml`.
- ROC/AUC calibration: run `benchmarks/arith_delta_nfr_roc.py --N 10000 --folds 5 --out benchmarks/results/roc_10k.json` and similarly for 100k. The notebook loads these JSON summaries.
- Propagation study (UM/RA): `benchmarks/arith_um_ra_propagation.py` seeds primes and measures activation propagation under RA.

All experiments log seeds and parameters; adhere to the unified grammar (U1–U6) and preserve the canonical invariants listed in `AGENTS.md`.


## 7. Edge cases and safety

- n < 2: treated as non-prime; ΔNFR set to +∞ (no evolution from EPI=0 without generators; see U1a).
- Phase checks: never couple without verifying |Δφ| ≤ Δφ_max (U3).
- Bifurcation control: if using destabilizers, include handlers (THOL) and stabilizers (IL) within the sequence window (U4a/U4b).
- Multi-scale: for nested EPIs, include stabilizers at each level (U5).


## 8. Summary

- ΔNFR(n) = 0 is an exact prime criterion arising from TNFR physics.
- UM and RA apply naturally on the arithmetic graph to study propagation from prime seeds.
- The structural field tetrad provides a multi-scale view of stability and stress.
- Empirical ROC shows perfect separation up to at least N=100000 with threshold at 0.
