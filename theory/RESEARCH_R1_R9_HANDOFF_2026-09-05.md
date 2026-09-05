# TNFR Research Program R1–R9 — Implementation & Status Handoff

**Date**: 2026-09-05 · **Branch**: `main` · **Repo**: fermga/TNFR-Python-Engine
**Scope**: full implementation of the revised consolidation + research program from
`TNFR_revised_handoff_packet_2026-09-04` (consolidation C0–C5, already done earlier
this session; research lines R1–R9, this handoff).
**Audience**: the next agent (human or AI) deciding how to proceed on each line.

> **How to read this.** Every claim is tagged with its honest status. Nothing here
> proves an open problem. Positive results are exact/measured internal theorems;
> several lines are deliberate **constructive negatives** (they sharpen the TNFR
> boundary). Each line ends with a **Decision point** telling you what is left open
> and the natural next move.

---

## 0. Status legend

| Tag | Meaning |
|-----|---------|
| **PROVED** | Classical theorem, re-expressed in TNFR framing |
| **DERIVED** | Follows exactly from the nodal equation / linear algebra (residual 0 over ℚ) |
| **MEASURED** | Numerically verified on a stated domain (with derived tolerances) |
| **CONJECTURAL** | Supported by evidence but unproven; explicitly gated |
| **OPEN** | Not established; may carry negative evidence |
| **NEGATIVE** | A claim was tested and found false / no-excess (a useful boundary result) |

All work obeys: English-only, no magic constants (tolerances derived from `√ε·‖L‖`),
exact rational (`fractions.Fraction`) for small theorem cases, numpy-only core
(SciPy optional and gated), `uses_known_factors` flags + circularity audits on every
arithmetic experiment, relabel/basis-invariance guards on every spectral metric.

---

## 1. Executive summary

Nine research lines shipped as nine small PRs (one per line), each with: a `src/`
module (numpy-only or exact-rational), a test suite, a `theory/` note with an honest
claim ledger, and a `benchmarks/` script emitting a C5 reproducibility manifest.

| PR | Line | Claim | Headline result | Status |
|----|------|-------|-----------------|--------|
| PR-06 | **R1** symmetry-sector observability | NT-P01 | diffusion sector + all 13 operators equivariant (under isolation) | **DERIVED + MEASURED** |
| PR-07 | **R2** arithmetic pulse recurrence | NT-P02 | Hankel = Krylov = #eigs = `gcd(k,p−1)+1` for primes | **DERIVED + MEASURED** |
| PR-08 | **R3** CRT as U5 fractality | NT-P03 | `L_ab = I−(I−L_a)⊗(I−L_b)` exact; `λ+μ−λμ` | **DERIVED + MEASURED** |
| PR-09 | **R4** projective p-adic tower | NT-P04 | projective transport exact; **REMESH not claimed** | transport **DERIVED**; REMESH **CONJECTURAL** |
| PR-10 | **R5** finite / algebraic fields | NT-P05 | prime regression exact; extension collisions; k=2 split/inert/ramified | regression **DERIVED**; detection **CONJECTURAL** |
| PR-11 | **R6** additive-character theory | NT-P06 | additive reading = Fourier; **no TNFR excess** | reduction **DERIVED**; excess **NEGATIVE / OPEN** |
| PR-12 | **R7** arithmetic-pressure audit | NT-P07 | sufficiency proved; **redundant** (not minimal); independent; completeness open | mixed (see below) |
| PR-13 | **R8** operator certification | NT-P08 | 2 certified, 4 rejected (boundary) | certificates **MEASURED**; classification **OPEN** |
| PR-14 | **R9** directed non-normal dynamics | NT-P09 | stable spectrum + transient amplification | dynamics **DERIVED**; U2 bound **OPEN** |

**Commit range (this session, R-lines)**: `90e203b8 → f22ce64c` on `main`, all pushed.
Consolidation C0–C5 landed earlier at `ed4b289f → 90e203b8`.

**Net honesty**: 4 lines are clean positive internal theorems (R1, R2, R3, and the
regression halves of R5); 2 are explicit "not yet / not this way" gates (R4 REMESH,
R5 detection); 1 is a constructive negative (R6); 1 is a scope-correction audit (R7);
1 is a boundary catalogue with mostly negatives (R8); 1 opens a real gap in the U2
reading (R9). **No open problem is claimed solved.**

---

## 2. Commit ledger (all pushed to `main`)

```
f22ce64c  R9  research: directed non-normal structural dynamics
e99c2b41  R8  research: certify arithmetic transformations against operator contracts
aafd0b33  R7  research: audit arithmetic pressure independence and completeness
ccc7951e  R6  research: controlled additive-character number theory
053a93f3  R5  research: extend residue networks to finite fields
ad4ac26a  R4  research: add projective p-adic network tower
e5dcb7fa  R3  feat: model CRT product structure as multiscale arithmetic NFRs
34ee8403  R2  R2: arithmetic pulse recurrence on pointed residue networks
33adbd72  R1  R1: symmetry-sector observability - diffusion sector + per-operator audit
90e203b8  C5  research claim/manifest/certificate/circularity infrastructure
f2ed537b  C4  basis-invariant spectral observables
ed4b289f      Docs: align documentation with code + canon (doc-audit)
4fdfe7ce  C0-C3 Fix canonicity audit: directed/weighted dNFR, hard U3, single xi_C, phase sector
```

---

## 3. Consolidation C0–C5 (context — already merged)

The research lines sit on this base; do not re-patch it.

- **C1** transport identity — the EPI channel of ΔNFR is the OUTGOING random-walk
  Laplacian `L_rw = I − D⁻¹W` (directed successor convention). Audit residual `4.4e-16`.
- **C2** U3 hard — phase-coherent coupling is a hard invariant (`test_u3_hard_invariant`).
- **C3** ξ_C — single canonical coherence-length definition (`ξ_C ∝ 1/√λ₂`).
- **C4** basis-invariant spectral — [spectral_projectors.py](../src/tnfr/physics/spectral_projectors.py)
  + [spectral_certificates.py](../src/tnfr/physics/spectral_certificates.py): projectors
  `Π_λ = QQᴴ` are basis-free; **non-normal operators are rejected** (that rejection is
  what R9 later handles).
- **C5** research infrastructure — [src/tnfr/research/](../src/tnfr/research/):
  `ClaimStatus` (strengthen-only transitions), `ExperimentManifest` (`uses_known_factors`,
  `input_bits`), `NumericalCertificate`, `CircularityAudit` (STRUCTURAL / DESCRIPTIVE /
  CIRCULAR). Every R-line benchmark emits a manifest through this.

---

## 4. Per-line detail & decision points

### R1 — Symmetry-sector observability · `NT-P01` · **DERIVED + MEASURED**
- **Built**: [symmetry_sectors.py](../src/tnfr/physics/symmetry_sectors.py) (automorphisms via
  VF2, Reynolds projector `Q_Γ`, orbit decomposition), [equivariance.py](../src/tnfr/physics/equivariance.py),
  [operator_equivariance.py](../src/tnfr/physics/operator_equivariance.py).
- **Result**: the diffusion operator is exactly Γ-equivariant (`spec(L_e) ⊆ ...`, Schur);
  **all 13 operators** are equivariant on `Fix(Γ)` seeds (residual `<1e-6`, exactly `0`).
- **Cache leak — ROOT-CAUSED & FIXED (N01)**: TNFR content-keyed caches in `G.graph`
  (`_dnfr_prep_cache`, node-set checksum, cache managers) survive `G.copy()` by **shared
  reference** and are keyed on a **label-independent** checksum, so the isomorphic copies
  `A` and `B = σ(A)` collided and `B` read `A`'s prep — spurious `~1e-3` batch residuals.
  `_isolate_graph_caches` drops those caches per copy (per-experiment isolation); the batch
  audit is now exactly `0`, independent of cache warmth / order / repeated runs
  ([test_equivariance_cache_isolation.py](../tests/physics/test_equivariance_cache_isolation.py)).
- **Decision point**: the *general* theorem "every equivariant TNFR word preserves the
  sectors" still needs a **formal composition proof** and a treatment of
  **non-equivariant selectors** (specific-node operators = pointed). Prove by stages (N06→N07).

### R2 — Arithmetic pulse recurrence · `NT-P02` · **DERIVED + MEASURED**
- **Built**: [krylov.py](../src/tnfr/mathematics/krylov.py) (exact rational Hankel/Krylov rank),
  [arithmetic_pulse.py](../src/tnfr/mathematics/arithmetic_pulse.py).
- **Result**: on the pointed residue network `(G_{p,k}, 0)`, `Hankel rank = Krylov dim =
  #distinct eigenvalues = gcd(k, p−1)+1` for primes (35/35 cases exact over ℚ). Composites
  deviate (control). Turns the static cyclotomy rank into the order of the temporal pulse.
- **Honest scope**: a recurrence-order identity, **not** a primality test (building the
  `p`-node network is exponential in `log₂ p`).
- **Decision point**: essentially closed. Optional extension: a closed-form for the pulse
  **amplitudes** `a_j` (not just the count), or the composite-`n` structure.

### R3 — CRT as U5 fractality · `NT-P03` · **DERIVED + MEASURED**
- **Built**: [crt_multiscale.py](../src/tnfr/mathematics/crt_multiscale.py); new
  `unit_power_residue_set` in [number_theory.py](../src/tnfr/mathematics/number_theory.py)
  (existing `power_residue_set` untouched).
- **Result**: for coprime `a,b` and the **unit** power-residue set, `L_ab = I −
  (I−L_a)⊗(I−L_b)` **exactly** (residual 0 over ℚ, up to the CRT permutation), with parent
  spectrum `λ+μ−λμ` and the exact U5 bound `λ₂(ab) ≤ min(λ₂(a),λ₂(b))`.
- **Correction made mid-line**: the tempting `λ₂(ab) = min(...)` equality is **false** for
  the non-symmetric (complex-spectrum) unit set; only the **bound** holds. Documented.
- **Honest scope**: a **synthesis** theorem — it *uses the known factors* to assemble the
  whole (`CircularityAudit.graph_construction_requires_answer=True`, discovery forbidden).
  Not a factoring algorithm.
- **Decision point**: closed as a forward (synthesis) theorem. The inverse (recover unknown
  factors) is explicitly **out of scope** and should not be attempted through this door.

### R4 — Projective p-adic tower · `NT-P04` · transport **DERIVED**, REMESH **CONJECTURAL**
- **Built**: [padic_tower.py](../src/tnfr/mathematics/padic_tower.py).
- **Result (exact over ℚ)**: for the reduction-compatible family, `R_e P_{e+1} = P_e R_e`
  (projective transport), `P_{e+1} Lift = Lift P_e` (intertwining ⇒ `spec(L_e) ⊆
  spec(L_{e+1})`, coarse modes survive), `R_e Lift = I`. Circulant/units connect to R3.
- **Key discipline**: the scale map is named `projective_scale_map`, **not REMESH**. The
  `RemeshContractAudit` records all four REMESH conditions as **unverified**
  (`realizes_remesh == False`).
- **Decision point**: to earn the REMESH name, an agent must verify the four contract
  conditions (recursive EPI echo, NETWORK scale, preserved identity, U5) *independently*.
  Until then `NT-P04` stays **CONJECTURAL**. Do **not** rename the map without that proof.

### R5 — Finite & algebraic number fields · `NT-P05` · regression **DERIVED**, detection **CONJECTURAL**
- **Built**: [finite_fields.py](../src/tnfr/mathematics/finite_fields.py) (F_q via trace
  additive characters, no external field package), [algebraic_residue_networks.py](../src/tnfr/mathematics/algebraic_residue_networks.py)
  (Gaussian integers ℤ[i]/(p)).
- **Result**: prime fields reproduce R2 exactly (`gcd(k,p−1)+1`); **extensions collide**
  (`distinct ≤ gcd(k,q−1)+1`, strict — trace is many-to-one, so the independence argument
  does **not** transfer). ℤ[i]/(p) at `k=2` separates ramified(2)/inert(3)/split(6).
- **Honest scope**: the Gaussian detector uses the classical type only as a **ground-truth
  label** (DESCRIPTIVE), is **k-sensitive** (k=3,4 don't separate), tested on small `p`.
- **Decision point**: two open sub-questions — (a) a closed formula for the extension
  collision count (when is it strict?); (b) whether *any* `k`-independent observable detects
  the decomposition type. Until then `NT-P05` stays **CONJECTURAL**.

### R6 — Controlled additive-character theory · `NT-P06` · reduction **DERIVED**, excess **NEGATIVE / OPEN**
- **Built**: [additive_resonance.py](../src/tnfr/mathematics/additive_resonance.py) (local
  prime sieve, Cramér / shuffle / matched-random / constellation controls).
- **Result**: the cyclic Goldbach convolution equals `IFFT(1̂_ℙ²)` exactly — **the additive
  reading *is* Fourier**, so every linear additive observable is a Fourier function. The
  candidate non-linear TNFR observable (phase-curvature energy) shows **no discriminating
  excess** over the classical power spectrum under density-matched controls (`|z|<1` vs
  classical `|z|≈150-190`), robustly across all control kinds.
- **This is the intended constructive negative.** `NT-P06` is **OPEN with negative evidence**.
- **Decision point**: to revive `NT-P06`, an agent must exhibit a *specific* non-linear
  phase observable that provably exceeds the singular series under the same controls — a high
  bar. Recommended default: **treat R6 as settled-negative** and do not claim TNFR additive
  structure beyond Fourier without clearing that bar.

### R7 — Arithmetic-pressure audit · `NT-P07` · mixed (see ledger)
- **Built**: [arithmetic_pressure.py](../src/tnfr/mathematics/arithmetic_pressure.py) (exact
  Ω/τ/σ channels of the canonical `delta_nfr_value`; 0 mismatches with the existing
  realization).
- **Result — three notions separated**: (1) **primality sufficiency PROVED** (each channel
  is 0 iff prime); (2) **minimality-for-primality NEGATIVE** — the set is *redundant*, one
  channel suffices; (3) **linear independence MEASURED** — rank 3, no affine relation,
  despite `r≈0.9` correlation (correlation ≠ dependence); (4) **completeness OPEN** —
  `completeness_proven()==False`, the fourth-channel gate is closed.
- **Honest scope**: computing a channel requires factoring `n`, so `ΔNFR=0` is **not** a
  primality algorithm (circularity **CIRCULAR**). "minimal and complete" is **downgraded** to
  "linearly independent but primality-redundant; completeness unproven."
- **Decision point**: to restore "complete", prove no fourth independent pressure degree is
  relevant (hard) — or admit a fourth channel only through the six-criterion gate. To restore
  "minimal", pick the **one** channel with the best *structural* (not primality) justification
  and demote the other two to derived diagnostics.

### R8 — Operator certification · `NT-P08` · certificates **MEASURED**, classification **OPEN**
- **Built**: [operator_certificates.py](../src/tnfr/mathematics/operator_certificates.py)
  (`ArithmeticOperatorCertificate`; reuses `operator_contracts`, R3, R4).
- **Result**: 2 certified (emission-at-zero → **Emission**; residue-edge propagation →
  **Resonance**, with valid grammar words and 0 contract residuals); **4 rejected** (CRT
  projection = relabeling; p-adic lift = REMESH-unverified, reusing R4; affine = automorphism;
  power map = endomorphism). **No fourteenth operator invented.**
- **This is the intended boundary result** — the negatives sharpen the 13-operator catalogue.
- **Decision point**: the general classification of arithmetic maps into operators is **OPEN**.
  A next agent could (a) audit more candidates (e.g. Frobenius, Hecke-like maps), or (b) attempt
  the REMESH contract proof from R4 to convert the p-adic-lift rejection into a certificate.

### R9 — Directed non-normal dynamics · `NT-P09` · dynamics **DERIVED**, U2 bound **OPEN**
- **Built**: extended [spectral_projectors.py](../src/tnfr/physics/spectral_projectors.py)
  (`spectral_abscissa`, `transient_gain`, `pseudospectral_bound` = Kreiss LB, `schur_residual`
  SciPy-gated, `matrix_exponential` numpy-only), new [directed_diffusion.py](../src/tnfr/physics/directed_diffusion.py).
- **Result**: directed **circulants** (R2 residue digraphs) are normal → transient gain 1
  (contraction); general **directed** graphs are non-normal → **stable spectrum yet transient
  amplification** (`gain 1.03–1.99 > 1`), with `Kreiss bound ≤ gain`. C4's `spectral_clusters`
  correctly **rejects** these; R9 is their Schur/transient home. **No `eigh` on non-symmetric.**
- **Honest scope (the real gap)**: the convergence reading `r_c = ν_f λ₂` is **asymptotic
  only**. A positive transient gain means `C(t)` can dip before relaxing (a U6 temporal risk).
- **Decision point (highest-value open item)**: derive how the **transient** enters the U2
  integral convergence `∫ ν_f ΔNFR dt < ∞` and its interaction with U6 potential confinement.
  This is the one line that opens a genuine question in the core grammar; `NT-P09` is **OPEN**.

---

## 5. Consolidated claim ledger (Appendix B extended)

| Claim | Statement | Status now |
|-------|-----------|------------|
| NT-P01 | symmetry sectors invariant under equivariant TNFR words | **DERIVED+MEASURED** (diffusion + 13 ops isolated); general proof staged |
| NT-P02 | pointed-pulse Hankel rank = `gcd(k,p−1)+1` | **DERIVED+MEASURED** |
| NT-P03 | CRT realizes U5 for unit networks | **DERIVED+MEASURED** (synthesis; not factoring) |
| NT-P04 | compatible p-adic tower realizes REMESH | transport **DERIVED**; REMESH **CONJECTURAL** (gate closed) |
| NT-P05 | pulse detects split/inert/ramified | regression **DERIVED**; detection **CONJECTURAL** (k-sensitive) |
| NT-P06 | TNFR additive phase exceeds Fourier | **NEGATIVE / OPEN** (reduction exact; no excess) |
| NT-P07 | 3-channel pressure minimal & complete | split (N02): `a` sufficiency **PROVED**; `b` functional independence **PROVED** (exact witness proof over ℚ); `c` minimal-for-primality **NEGATIVE** (redundant); `d` completeness **OPEN** (task-scoped); `e` primality-as-algorithm **CIRCULAR** |
| NT-P08 | arithmetic maps ↔ operators | 2 **MEASURED** certificates, 4 rejected; full classification **OPEN** |
| NT-P09 | generalized U2 bound (non-normal transients) | Euclidean transient amplification **DERIVED+MEASURED**; stationary `L²(π)` contraction **DERIVED+MEASURED** (N03: gain `=1`, Euclidean transient is metric-dependent); scalar-`ν_f` structural-time theorem **DERIVED+MEASURED** (N04: `x=e^{−s(t)L}x₀`, clock-invariant reorganization, finite `J ≤ M‖LQ‖‖x₀‖/ω`, `M=1` normal / `>1` non-normal); **per-node-energy contraction MEASURED** (N05: `peak=1`, symmetric part of `L_sub` `≻0` over 2·10⁵ + in-hub, ambient `>1` is exactly `‖Q‖`; general PSD **CONJECTURAL**); canonical U2 metric/integral **OPEN** (gate not met, U2/U6 unmodified) |

---

## 6. How to reproduce

```powershell
# from repo root, PowerShell
$env:PYTHONPATH=(Resolve-Path ./src).Path
$py = "C:/Program Files/Python313/python.exe"

# full research + math + physics suites (all green)
& $py -m pytest tests/mathematics tests/research tests/physics -q

# per-line benchmarks (each prints its result table + C5 manifest, exits 0)
& $py benchmarks/arithmetic_pulse_recurrence.py        # R2
& $py benchmarks/crt_multiscale_composition.py         # R3
& $py benchmarks/padic_scale_consistency.py            # R4
& $py benchmarks/algebraic_field_signatures.py         # R5
& $py benchmarks/additive_resonance_controls.py        # R6
& $py benchmarks/arithmetic_pressure_audit.py          # R7
& $py benchmarks/arithmetic_operator_certification.py  # R8
& $py benchmarks/directed_nonnormal_dynamics.py        # R9

# lint (authoritative limit is 88)
& $py -m flake8 src/tnfr/mathematics src/tnfr/physics --max-line-length=88
```

**Test counts added** (per line): R1 26+27, R2 84, R3 191, R4 83, R5 134, R6 24, R7 47,
R8 16, R9 19 — plus C4 19, C5 18. All green; flake8 clean at the project limit.

---

## 7. Theory notes (the authoritative per-line write-ups)

| Line | Note |
|------|------|
| R1 | [TNFR_STRUCTURAL_OBSERVABILITY.md](TNFR_STRUCTURAL_OBSERVABILITY.md) |
| R2 | [TNFR_ARITHMETIC_DYNAMICS.md](TNFR_ARITHMETIC_DYNAMICS.md) |
| R3 | [TNFR_CRT_FRACTALITY.md](TNFR_CRT_FRACTALITY.md) |
| R4 | [TNFR_PADIC_DYNAMICS.md](TNFR_PADIC_DYNAMICS.md) |
| R5 | [TNFR_ALGEBRAIC_NUMBER_FIELDS.md](TNFR_ALGEBRAIC_NUMBER_FIELDS.md) |
| R6 | [TNFR_ADDITIVE_DYNAMICS.md](TNFR_ADDITIVE_DYNAMICS.md) |
| R7 | [TNFR_ARITHMETIC_PRESSURE.md](TNFR_ARITHMETIC_PRESSURE.md) |
| R8 | [TNFR_ARITHMETIC_OPERATORS.md](TNFR_ARITHMETIC_OPERATORS.md) |
| R9 | [TNFR_DIRECTED_NONNORMAL_DYNAMICS.md](TNFR_DIRECTED_NONNORMAL_DYNAMICS.md) |

---

## 8. Recommended priority for the next agent

Ranked by structural value of the open question:

1. **R9 / NT-P09** — derive the transient's role in U2 convergence. This is the only line
   that opens a real gap in the *core grammar*; highest value.
2. **R4 / NT-P04** — attempt the REMESH contract proof (would also upgrade the R8 p-adic-lift
   rejection). Well-scoped, four concrete conditions.
3. **R1 / NT-P01** — the staged general equivariance proof + non-equivariant-selector treatment.
4. **R5 / NT-P05** — closed form for extension collisions; search for a `k`-independent detector.
5. **R7 / NT-P07** — decide the honest scope of "minimal/complete" (prove completeness or pick a
   minimal structural channel).
6. **R6 / NT-P06** — treat as settled-negative unless a specific super-Fourier observable is found.
7. **R2 / R3** — essentially closed; extend only with a new structural idea (pulse amplitudes;
   CRT beyond the synthesis direction).

**Housekeeping not yet done** (safe, low-risk): index these 9 theory notes in
[theory/README.md](README.md) and add an R1–R9 row set to the research-programs table in
[AGENTS.md](../AGENTS.md) §12. No source changes required.

---

**Bottom line for the next agent.** The program is fully implemented, tested, and pushed.
Every line has an explicit, honest status; the open items are real and well-scoped. Start
from the physics (the nodal equation and the U2/U6 grammar) and pick a Decision point above —
do not extend any diagnostic surface without a new structural idea, and never upgrade a
CONJECTURAL/OPEN/NEGATIVE claim without clearing the bar stated in its line.
