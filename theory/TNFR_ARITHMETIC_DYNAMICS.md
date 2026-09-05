# TNFR Arithmetic Dynamics — the Pulse Recurrence Theorem (R2)

**Status**: DERIVED (Kronecker + circulant + cyclotomy) + MEASURED (exact
rational, zero discrepancies on the tested domain). Not a proof of any open
problem and **not** a primality test.
**Modules**: [src/tnfr/mathematics/arithmetic_pulse.py](../src/tnfr/mathematics/arithmetic_pulse.py),
[src/tnfr/mathematics/krylov.py](../src/tnfr/mathematics/krylov.py) ·
**Tests**: [tests/mathematics/test_arithmetic_pulse.py](../tests/mathematics/test_arithmetic_pulse.py) ·
**Depends on**: R1 (pointed graphs, §4), C4 (spectral rank), C5 (claim manifest).

## 1. The pointed object

The static cyclotomy law ([TNFR_NUMBER_THEORY.md](TNFR_NUMBER_THEORY.md)) states
that the k-th power residue Cayley operator ``L_rw = I − D⁻¹W`` on ``ℤ/pℤ``
(``p`` prime) has exactly ``s_k(p) = gcd(k, p−1) + 1`` distinct eigenvalues.
R2 gives that number a **dynamic** reading through the **pointed** network
``(G_{p,k}, 0)`` seeded at the additive identity

$$e_0 = (1, 0, \dots, 0)^\top .$$

The point is the neutral element ``0``, so it is chosen **without using any
factor** — the translation symmetry is broken by declaration (R1 §4, ADR-006),
not clandestinely.

## 2. The recurrence (DERIVED)

The pulse moments $\mu_m = e_0^\top L_{rw}^m e_0$ obey a linear recurrence.
Three classical facts chain together:

$$\underbrace{\operatorname{rank}[\mu_{i+j}]}_{\text{Hankel}}
\;=\;
\underbrace{\dim\operatorname{span}\{e_0, L e_0, L^2 e_0, \dots\}}_{\text{Krylov}}
\;=\;
\underbrace{\#\{\text{distinct eigenvalues}\}}_{\text{spectral}}
\;=\;
\underbrace{\gcd(k, p-1) + 1}_{\text{cyclotomy}} .$$

- **Hankel = Krylov** is Kronecker's theorem (holds for *any* square ``L``).
- **Krylov = #distinct eigenvalues** holds because ``L_rw`` is a **circulant**,
  so ``e_0`` has a non-zero projection on every Fourier mode and the cyclic
  subspace it generates picks up exactly one dimension per distinct eigenvalue.
- **= gcd(k, p−1) + 1** for primes is the cyclotomy law (Gauss periods).

This turns the static spectral rank into the order of the **temporal pulse**

$$h(t) = e_0^\top e^{-\nu_f L t} e_0 = \sum_{j=1}^{s} a_j\, e^{-\nu_f \lambda_j t},
\qquad s = \gcd(k, p-1) + 1 .$$

## 3. Verification (MEASURED, exact)

Everything is computed over ℚ with :class:`fractions.Fraction` (no floating-point
ambiguity). For every prime ``p ∈ {5,…,23}`` and ``k ∈ {1,2,3,4,6}`` the three
computable ranks agree and equal ``gcd(k, p−1) + 1`` (35/35). The Krylov
dimension matches the numerical distinct-eigenvalue count, and the pointed-pulse
rank is **independent of the chosen point** (translation is an automorphism of
the Cayley graph).

**Composite controls (outside the theorem).** Hankel = Krylov still holds for
composites (Kronecker is universal), but the cyclotomy value ``gcd(k, n−1) + 1``
does **not** — e.g. ``n = 15, k = 2`` gives rank ``9 ≠ 3``. The identity is
prime-specific; an occasional coincidental composite match (``n=15, k=3``) is why
this is a **control**, not a primality test.

### 3b. The amplitudes (DERIVED, N12)

The recurrence order fixes *how many* tones the pulse has; the **amplitudes**
``a_j`` fix *how loud* each is
([pulse_amplitudes.py](../src/tnfr/mathematics/pulse_amplitudes.py)). Because
``L_rw`` is a **circulant** it is diagonalized by the Fourier basis
``f_j[x] = ω^{jx}/√p``, and the pointed seed has **uniform** Fourier weight,
``|⟨e_0, f_j⟩|² = |f_j[0]|² = 1/p`` for every ``j``. Grouping the ``p`` modes by
distinct eigenvalue ``λ`` of multiplicity ``m_λ``,

$$h(t) = \sum_{j} \tfrac1p\, e^{-t\lambda_j} = \sum_{\lambda} \frac{m_\lambda}{p}\, e^{-t\lambda},
\qquad a_\lambda = \frac{m_\lambda}{p}.$$

**Theorem (`NT-P02b`).** The pointed-circulant pulse amplitudes are the
**normalized spectral multiplicities** ``a_λ = m_λ/n`` (``n = p``) — exact
rationals summing to ``1``. Two independent confirmations: the orthogonal
spectral projector gives ``e_0^* P_λ e_0 = m_λ/p`` (**basis-invariant** — a random
unitary rotation inside a degenerate eigenspace leaves it unchanged,
``~1e-16``), and ``Σ_λ (m_λ/p)\,λ^m`` reconstructs the **exact rational** moments
``μ_m = e_0^\top L^m e_0`` (``~1e-13``). Example ``(p,k)=(11,2)``: the spectrum is
``a = 1/11`` at ``λ=0`` plus ``5/11`` on each of the two conjugate eigenvalues
``1.10 ∓ 0.332i`` (real pulse, ``1/11 + 5/11 + 5/11 = 1``). The number of tones is
the R2 rank ``gcd(k,p−1)+1``, so the amplitude spectrum refines the rank without
changing it.

## 4. Two faces (do not mix)

- **Dissipative face**: the heat semigroup ``e^{−ν_f L t}`` — valid with complex
  eigenvalues (the directed, ``p ≡ 3 (mod 4)`` case) under the overdamped
  reading. The recurrence order ``s`` is face-independent.
- **Conservative face**: frequencies ``ω = √λ`` — only for real / self-adjoint
  operators (or with an explicit conservative-substrate derivation). Do not call
  the root of a complex eigenvalue a "tone" without that derivation.

## 5. Honest scope

This is a **recurrence-order identity** connecting the pointed pulse to the
cyclotomy law — a structural bridge, not an algorithm. Building and analysing the
``p``-node network is ``poly(p) = poly(2^{L})``, i.e. **exponential in the input
size** ``L = log₂ p`` bits; there is no factoring or primality speedup and no
cryptographic consequence. The result is DERIVED (from Kronecker, the circulant
structure and cyclotomy) and MEASURED exactly on the stated domain.
