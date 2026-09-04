# TNFR Arithmetic Pressure — Independence & Completeness Audit (R7)

**Status**: primality sufficiency **PROVED** (classical); channel linear
independence **MEASURED** (rank 3); minimality-for-primality **NEGATIVE**
(redundant); structural completeness **OPEN / CONJECTURAL** (`NT-P07`). The
"minimal and complete" language is downgraded to explicit scope. Arithmetic
pressure is a *structural descriptor*, computed from the factorisation — **not** a
primality-discovery algorithm.
**Module**: [src/tnfr/mathematics/arithmetic_pressure.py](../src/tnfr/mathematics/arithmetic_pressure.py) ·
**Tests**: [tests/mathematics/test_arithmetic_pressure.py](../tests/mathematics/test_arithmetic_pressure.py) ·
**Benchmark**: [benchmarks/arithmetic_pressure_audit.py](../benchmarks/arithmetic_pressure_audit.py) ·
**Audits**: [TNFR_NUMBER_THEORY.md](TNFR_NUMBER_THEORY.md) `NT-C01` · **Depends on**: C5.

## 1. The three-channel pressure

The canonical arithmetic realisation of the nodal gradient
([number_theory.py](../src/tnfr/mathematics/number_theory.py),
`ArithmeticTNFRFormalism.delta_nfr_value`, unit coefficients) is

$$\Delta\mathrm{NFR}(n) = \underbrace{(\Omega(n) - 1)}_{c_1}
+ \underbrace{(\tau(n) - 2)}_{c_2}
+ \underbrace{\Big(\tfrac{\sigma(n)}{n} - \big(1 + \tfrac1n\big)\Big)}_{c_3},$$

with `Ω` the prime-factor count with multiplicity, `τ` the divisor count, and `σ`
the divisor sum. This module recomputes every channel **exactly over ℚ**
(`fractions.Fraction`) and agrees with the canonical realisation to `0` mismatch
on `[2, 300)`. Three claims that are easy to conflate are audited separately.

## 2. Primality sufficiency — and redundancy (PROVED / NEGATIVE)

Each channel is non-negative for `n ≥ 2` and vanishes **exactly** at the primes:

$$c_1(n) = 0 \iff \Omega(n) = 1,\quad
c_2(n) = 0 \iff \tau(n) = 2,\quad
c_3(n) = 0 \iff \sigma(n) = n + 1,$$

and all three conditions are equivalent to "`n` prime". Verified on `[2, 1000]`
(`all_channels_sufficient`, `channels_nonnegative`, `pressure_zero_iff_prime`).

A direct **consequence** is that the three-channel set is **redundant** for
primality: a *single* channel already characterises the primes, so
`minimal_channels_for_primality = 1` and every ablation still detects the primes
(`ablation_detects_primes` for all six non-empty subsets). Hence the realisation
is **not minimal for prime detection** — the "minimal" claim, read as *minimality
for primality*, is **false** (a MEASURED negative). The channels are not there to
detect primes more than once; they are there to carry *distinct structure*.

## 3. Linear independence vs correlation (MEASURED)

As real functions on the audited range the channels are **linearly independent**:

$$\operatorname{rank}[c_1\; c_2\; c_3] = 3,\qquad
\operatorname{rank}[c_1\; c_2\; c_3\; \mathbf 1] = 4,$$

so no channel is a linear — or even affine — combination of the others
(`channel_rank`, `has_linear_relation == False`). Yet they are strongly
**correlated**, all rising with compositeness:

| pair | Pearson `r` on `[2, 1000]` |
|------|----------------------------|
| `c1`–`c2` | 0.887 |
| `c1`–`c3` | 0.877 |
| `c2`–`c3` | 0.931 |

**Correlation is not dependence.** High `r` (≈ 0.9) coexists with full rank: each
channel carries independent structural information (factor multiplicity, divisor
count, abundance), even though they move together on composites.

## 4. Class-conditioned pressure (MEASURED)

Conditioning the total pressure on the factor class (`[2, 1000]`) separates the
regimes cleanly — primes at exact zero, the composite classes strictly positive:

| class | count | mean `ΔNFR` | range |
|-------|-------|-------------|-------|
| prime | 168 | 0.000 | [0, 0] |
| semiprime | 288 | 3.318 | [3.07, 3.83] |
| prime power | 25 | 5.786 | [2.03, 17.0] |
| composite (other) | 518 | 12.515 | [6.22, 37.4] |

The tight semiprime band and the wide "other" band show the pressure grades
compositeness, not merely primality. (`pressure_by_class`; `abundance_class`
recovers the perfect numbers `6, 28, 496` on `[2, 1000]`.)

## 5. Structural completeness (OPEN) and the fourth-channel gate

Whether a **fourth** independent pressure degree is relevant is **not proven**.
`completeness_proven()` returns `False`, and a candidate fourth channel is
admitted only through the explicit gate `admits_fourth_channel`, which requires
**all six** conditions: independent structural meaning; not a function of the
existing three on the declared domain; derived from the model (not an accuracy
search); a documented change in diagnostic capability; preservation of the
`ΔNFR = 0` prime set; and a full contract with tests. The default (empty) criteria
are inadmissible, so the three-channel set is **not** asserted complete.

## 6. Honest scope and claim ledger

Computing any channel requires the factorisation of `n` (via `Ω, τ, σ`), so
`ΔNFR(n) = 0` is **not** a fast primality test — it is a structural descriptor
read *from* the factorisation. The C5 circularity audit therefore records
`uses_factorization_in_features = True` (verdict **CIRCULAR** for discovery): the
pressure may **not** be presented as a primality/factoring algorithm.

| Claim | Basis | Status |
|-------|-------|--------|
| each channel `0` iff prime; sum `0` iff prime | classical | **PROVED** |
| minimal for primality | single channel suffices | **NEGATIVE** (redundant) |
| channels linearly independent | rank 3, no affine relation | **MEASURED** |
| channels correlated but not dependent | `r ≈ 0.9`, rank 3 | **MEASURED** |
| pressure grades compositeness by class | class-conditioned means | **MEASURED** |
| three channels structurally complete | no proof | **OPEN / CONJECTURAL** (`NT-P07`) |

**Bottom line.** The three-channel arithmetic pressure is a linearly-independent
set of structural descriptors whose common (and individual) zero set is exactly
the primes. It is **redundant** for primality (not minimal), **correlated but
independent** as structure, and of **unproven completeness**. The "minimal and
complete" description is retained only under this explicit, restricted scope.
