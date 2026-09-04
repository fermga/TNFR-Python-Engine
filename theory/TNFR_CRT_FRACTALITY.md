# TNFR CRT Fractality — the Multiscale Composition Theorem (R3)

**Status**: DERIVED (CRT unit bijection + Kronecker product) + MEASURED (exact
rational residual `0`; eigenvalue composition to `~4e-15`). A **structural
synthesis** result: it uses the known factors `a, b` to assemble a network from
its sub-networks. It is **not** a factoring or discovery algorithm and makes no
complexity, cryptographic, or Millennium claim.
**Module**: [src/tnfr/mathematics/crt_multiscale.py](../src/tnfr/mathematics/crt_multiscale.py) ·
**Tests**: [tests/mathematics/test_crt_multiscale.py](../tests/mathematics/test_crt_multiscale.py) ·
**Benchmark**: [benchmarks/crt_multiscale_composition.py](../benchmarks/crt_multiscale_composition.py) ·
**Depends on**: R2 (residue Cayley operator, exact rational linear algebra),
C4 (spectral tolerance), C5 (claim manifest, circularity audit).

## 1. The multiscale object

TNFR axiom **U5** (multi-scale coherence, [AGENTS.md](../AGENTS.md) §6) requires
that nested EPIs compose without identity loss. The Chinese Remainder Theorem is
the arithmetic realisation of exactly that: for coprime moduli `a, b` the
additive group splits,

$$\mathbb{Z}/ab\mathbb{Z} \;\cong\; \mathbb{Z}/a\mathbb{Z}\times\mathbb{Z}/b\mathbb{Z},$$

and the unit group splits multiplicatively,
$(\mathbb{Z}/ab\mathbb{Z})^\ast \cong (\mathbb{Z}/a\mathbb{Z})^\ast\times(\mathbb{Z}/b\mathbb{Z})^\ast$.
The parent network `ℤ/abℤ` is the whole; the two coprime factor networks are its
sub-EPIs (finer scales).

## 2. The unit restriction (why it must be the units)

Write `S_m = ` the connection set of the residue Cayley digraph
`Cay(ℤ/mℤ, S_m)`, and `L_m = I − (1/|S_m|) W_m` for its random-walk Laplacian
(R2). The theorem needs the connection set to CRT-factor **exactly**, and that
holds **iff** the set is the k-th powers of the **units**:

$$S_m = \{\, u^k \bmod m : \gcd(u, m) = 1 \,\} = \texttt{unit\_power\_residue\_set}(m, k).$$

Under the CRT bijection of unit groups, `S_ab` maps onto the full product
`S_a × S_b`. For a **prime** `m` every non-zero residue is a unit, so this
coincides with the R2 set `power_residue_set`; for composite `m` it is a **proper
subset** (`unit_power_residue_set(15, 2) = {1, 4}` vs
`power_residue_set(15, 2) = {1, 4, 6, 9, 10}`). The unrestricted set includes
non-units and does **not** factor under CRT — it is the non-factorizing control
(§5).

## 3. The Kronecker identity (DERIVED, exact over ℚ)

Let `σ` be the CRT permutation (node `r ↦ (r mod a, r mod b)`,
`crt_ordering`). Because `S_ab` CRT-factors as `S_a × S_b`, the adjacency,
transition and Laplacian operators are exact Kronecker products (up to `σ`):

$$A_{ab} = A_a \otimes A_b, \qquad
P_{ab} = P_a \otimes P_b, \qquad
\boxed{\,L_{ab} = I - (I - L_a)\otimes(I - L_b)\,}.$$

The last identity holds because `P_m = I − L_m` and
`P_{ab} = P_a ⊗ P_b = (I−L_a)⊗(I−L_b)`. It is verified **exactly over ℚ** using
`fractions.Fraction`: `crt_kronecker_residual(a, b, k) == 0` for every tested
coprime pair and power (no floating point).

## 4. Eigenvalue composition and the U5 gap bound (DERIVED + MEASURED)

Kronecker structure fixes the parent spectrum as the **child eigenvalue
composition**. If `λ ∈ spec(L_a)` and `μ ∈ spec(L_b)` then

$$\lambda_{\text{parent}} = \lambda + \mu - \lambda\mu = 1 - (1-\lambda)(1-\mu).$$

The random-walk Laplacian always has the constant mode `λ = 0` (the trivial /
neutral EPI), so setting `μ = 0` gives `λ` unchanged: **every child eigenvalue
embeds in the parent spectrum**. Hence the parent's non-zero spectrum is the
union of the child non-zero spectra **plus** genuinely multiscale cross modes
`λ + μ − λμ` (`λ, μ ≠ 0`), and the spectral gap obeys the exact U5 bound

$$\lambda_2(ab) = \min\!\big(\lambda_2(a),\, \lambda_2(b),\,
\min_{\lambda,\mu\neq 0}|\lambda + \mu - \lambda\mu|\big)
\;\le\; \min\!\big(\lambda_2(a), \lambda_2(b)\big).$$

**Reading (U5 telemetry).** Composing scales never *speeds up* the slowest
sub-EPI — it can only add slower cross-scale modes. The composite's coherence
timescale is at least as long as its slowest factor's. Equality holds when the
child spectra are real in `[0, 1]` (symmetric connection); when the unit set is
not symmetric the spectrum is complex and a cross mode can be strictly smaller
(e.g. `4 × 9, k = 2`: `λ₂ = 0.518 < min = 1.000`). The numerical eigenvalue
composition matches to `~4e-15`, below the derived tolerance `√ε·‖L‖₂` (C4).

## 5. Non-factorizing control (the unit restriction is necessary)

The unrestricted set is the control. `residue_set_factors(a, b, k, unit=False)`
returns `False` — even for coprime primes (`3 × 5`), because the non-unit powers
`{6, 9, 10}` of `ℤ/15ℤ` have no CRT product preimage in
`S_3 × S_5 = {1} × {1, 4}`. Consequently the Kronecker identity **fails** for the
unrestricted operator (`full_power_residue_laplacian`): the measured residual is
`0.30, 0.09, …` — non-zero. This isolates the unit group as the exact carrier of
the multiscale product structure.

## 6. Honest scope — synthesis, not factoring

This is explicitly a **structural (synthesis) branch**: the composition **uses
the known factors** `a, b` to build the parent from its children. The C5
circularity audit therefore records
`graph_construction_requires_answer = True`, yielding verdict `CIRCULAR` and
`permits_discovery_claim = False`, and the experiment manifest sets
`uses_known_factors = True` (claim `NT-P03`). The algebraic product theorem
(Kronecker identity + eigenvalue law, §3–4) stands on its own and is tested
**separately** from any inverse use: recognising a *given* `L_ab` **as** a
product `L_a ⊗ L_b` requires already knowing `a, b`, so nothing here is, or may
be presented as, a factoring algorithm. The forward direction — assembling the
whole from known parts — is legitimate multiscale (U5) structure; the inverse
direction (recovering unknown factors) is **not** claimed and is out of scope.

## 7. Claim ledger

| Claim | Basis | Status |
|-------|-------|--------|
| `L_ab = I − (I−L_a)⊗(I−L_b)` (unit set, coprime) | CRT unit bijection + Kronecker | **DERIVED** + MEASURED (exact `0` over ℚ) |
| `λ_parent = λ + μ − λμ` | Kronecker eigenvalues | **DERIVED** + MEASURED (`~4e-15`) |
| `λ₂(ab) ≤ min(λ₂(a), λ₂(b))` | trivial-mode embedding | **DERIVED** + MEASURED |
| unit restriction necessary | control residual `≠ 0` | **MEASURED** |
| factoring / discovery | — | **NEGATIVE** (synthesis only; `NT-P03` forbids discovery claim) |

**Bottom line.** R3 makes CRT an explicit U5 multiscale model on a family of unit
residue networks: the whole operator, its spectrum, and its slowest coherence
timescale are exactly composed from the sub-EPIs. It closes no open problem and
provides no factoring capability.
