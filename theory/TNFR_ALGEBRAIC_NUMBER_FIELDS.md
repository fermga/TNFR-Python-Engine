# TNFR Algebraic Number Fields — Finite Fields & Gaussian Signatures (R5)

**Status**: prime-field regression **DERIVED** (R2 recovered) + extension
behaviour **MEASURED** (exact counts; trace collisions); Gaussian decomposition
detection **CONJECTURAL** (`NT-P05`), a *descriptive* study using the classical
type as a ground-truth label. No factoring, complexity, cryptographic, or
Millennium claim; no external field package required.
**Modules**: [src/tnfr/mathematics/finite_fields.py](../src/tnfr/mathematics/finite_fields.py),
[src/tnfr/mathematics/algebraic_residue_networks.py](../src/tnfr/mathematics/algebraic_residue_networks.py) ·
**Tests**: [tests/mathematics/test_finite_fields.py](../tests/mathematics/test_finite_fields.py),
[tests/mathematics/test_algebraic_residue_networks.py](../tests/mathematics/test_algebraic_residue_networks.py) ·
**Benchmark**: [benchmarks/algebraic_field_signatures.py](../benchmarks/algebraic_field_signatures.py) ·
**Depends on**: R2 (pulse rank), C5 (claim manifest, circularity audit).

## 1. Finite fields via the trace character

R2 established that on the k-th power residue Cayley network over the **prime**
field `F_p` the pulse rank equals the cyclotomy count `gcd(k, p−1) + 1`. R5
lifts the construction to a general finite field `F_q` (`q = p^f`) using the
additive characters built from the field **trace**

$$\psi_a(x) = \exp\!\Big(\tfrac{2\pi i}{p}\,
\operatorname{Tr}_{F_q/F_p}(a x)\Big),\qquad
\operatorname{Tr}(y) = y + y^p + \cdots + y^{p^{f-1}} \in F_p.$$

The eigenvalues of the additive Cayley graph `Cay(F_q, S)` with `S = ` the
non-zero k-th powers are exactly the **normalised Gauss periods**

$$\eta_a = \frac1{|S|}\sum_{s\in S}\psi_a(s),$$

constant on cosets of the k-th power subgroup `H` (since `\psi_{ah}=\psi_a` for
`h\in H`). There are `gcd(k, q−1)` such cosets plus `\eta_0 = 1`, so the distinct
count is **at most** `gcd(k, q−1) + 1`.

## 2. Prime regression and extension collisions (DERIVED + MEASURED)

**Prime fields (`f = 1`, DERIVED).** The trace is the identity, `\psi_a(x) =
\zeta_p^{a x}`, and the construction is exactly R2. Every tested case satisfies

$$\#\{\eta_a\} = \gcd(k, p-1) + 1 \qquad (`prime_field_matches_cyclotomy`).$$

**Extensions (`f ≥ 2`, MEASURED).** Here the trace is `p^{f−1}`-to-one, so distinct
cosets of `H` can share a period value and the bound is **not** tight. Measured
distinct counts (exact, and cross-checked against the explicit graph spectrum):

| field | `q` | `k` | `gcd(k,q−1)+1` | distinct | |
|-------|-----|-----|----------------|----------|---|
| `F_4` | 4 | 3 | 4 | **2** | collision |
| `F_9` | 9 | 4 | 5 | **2** | collision |
| `F_25` | 25 | 3 | 4 | **3** | collision |
| `F_49` | 49 | 4 | 5 | **3** | collision |
| `F_27` | 27 | 4 | 3 | 3 | match |

So the prime-field independence argument (distinct cosets ⇒ distinct periods)
does **not** transfer to extensions: `distinct = gcd(k, q−1) + 1` becomes an
**upper bound**, strict whenever the trace identifies cosets. The trace-character
count and the explicit additive Cayley spectrum agree in every case
(`period_and_explicit_spectrum_agree`), an internal consistency control.

## 3. Gaussian integers: split / inert / ramified (MEASURED, CONJECTURAL)

For a rational prime `p` the quotient `ℤ[i]/(p)` has three classical shapes fixed
by `p mod 4`:

- `p = 2` — **ramified**, `(2) = −i(1+i)^2`, a local ring with nilpotent `1+i`;
- `p ≡ 1 (mod 4)` — **split**, `ℤ[i]/(p) ≅ F_p × F_p`;
- `p ≡ 3 (mod 4)` — **inert**, `ℤ[i]/(p) ≅ F_{p^2}`.

Building the additive Cayley network on `(ℤ[i]/(p), +)` (`p^2` nodes) with
connection set the non-zero **unit** k-th powers, the distinct-eigenvalue count at
`k = 2` separates the three types on the tested primes:

$$\text{ramified} \to 2, \qquad \text{inert} \to 3, \qquad \text{split} \to 6.$$

The inert count agrees with the `F_{p^2}` finite-field result of §2 (`k=2` → 3),
as it must — the inert quotient *is* `F_{p^2}`. The split count `6` reflects the
product structure `F_p × F_p` (a CRT product network, cf. R3).

**Honest scope.** The classical decomposition type is used only as a *ground-truth
label* for scoring, never as an input to the observable — a **descriptive** study
in the C5 sense (`CircularityAudit(factors_used_only_for_scoring=True)`), so it
may not be presented as a discovery algorithm. The separation is **k-sensitive**:
at `k = 3, 4` the counts no longer separate the three types
(`signature_separates_types(..., 3) == False`). It is tested only on small `p`
and has no proof. The claim *"the pulse detects the decomposition type"*
(`NT-P05`) is therefore **CONJECTURAL**, with the `k = 2` signature as supporting
measured evidence.

## 4. Claim ledger

| Claim | Basis | Status |
|-------|-------|--------|
| `F_p`: `#{η_a} = gcd(k, p−1) + 1` | trace = identity (R2) | **DERIVED** + MEASURED |
| `F_q`: `#{η_a} ≤ gcd(k, q−1) + 1`, strict for some | trace many-to-one | **MEASURED** (exact counts) |
| trace-character count = explicit spectrum | character theory | **MEASURED** (agree) |
| `k=2` count separates ramified/inert/split | Cayley spectrum | **MEASURED** (small `p`) |
| pulse detects decomposition type | — | **CONJECTURAL** (`NT-P05`; k-sensitive, descriptive) |

**Bottom line.** R5 carries the arithmetic pulse from prime fields to finite
fields and Gaussian quotients. It recovers R2 exactly on `F_p`, quantifies the
trace collisions that break the formula on extensions, and exhibits a `k = 2`
spectral signature that separates the three Gaussian decomposition types. The
decomposition detector is a descriptive, k-sensitive, small-`p` observation — not
a proof and not an algorithm.
