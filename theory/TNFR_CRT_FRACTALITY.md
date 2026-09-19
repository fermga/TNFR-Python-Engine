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

For coprime moduli `a, b`, the Chinese Remainder Theorem gives the declared
arithmetic product carrier

$$\mathbb{Z}/ab\mathbb{Z} \;\cong\; \mathbb{Z}/a\mathbb{Z}\times\mathbb{Z}/b\mathbb{Z},$$

and the unit group splits multiplicatively,
$(\mathbb{Z}/ab\mathbb{Z})^\ast \cong (\mathbb{Z}/a\mathbb{Z})^\ast\times(\mathbb{Z}/b\mathbb{Z})^\ast$.
The two factors parameterize the parent network. This exact product
construction is useful for scale comparisons; it does not instantiate the
engine's nested-EPI hierarchy, execute REMESH, or prove its U5 contract.

## 2. A sufficient product-compatible connection family

Write `S_m = ` the connection set of the residue Cayley digraph
`Cay(ℤ/mℤ, S_m)`, and `L_m = I − (1/|S_m|) W_m` for its random-walk Laplacian
(R2). The theorem needs the connection set to CRT-factor **exactly**, and that
is guaranteed for the k-th powers of the **units**:

$$S_m = \{\, u^k \bmod m : \gcd(u, m) = 1 \,\} = \texttt{unit\_power\_residue\_set}(m, k).$$

Under the CRT bijection of unit groups, `S_ab` maps onto the full product
`S_a × S_b`. For a **prime** `m` every non-zero residue is a unit, so this
coincides with the R2 set `power_residue_set`; for composite `m` it is a **proper
subset** (`unit_power_residue_set(15, 2) = {1, 4}` vs
`power_residue_set(15, 2) = {1, 4, 6, 9, 10}`). The unrestricted set includes
non-units and does **not** factor under CRT — it is the non-factorizing control
(§5). Product factorization is the actual hypothesis, not a characterization
of unit sets: any prescribed nonempty `T_a` and `T_b` with parent connection
set `CRT^-1(T_a × T_b)` gives the same product identity, including suitable
non-unit sets.

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

## 4. Eigenvalue composition and the nonzero-modulus bound

Kronecker structure fixes the parent spectrum as the **child eigenvalue
composition**. If `λ ∈ spec(L_a)` and `μ ∈ spec(L_b)` then

$$\lambda_{\text{parent}} = \lambda + \mu - \lambda\mu = 1 - (1-\lambda)(1-\mu).$$

The random-walk Laplacian always has the constant mode `λ = 0` (the trivial /
neutral EPI), so setting `μ = 0` gives `λ` unchanged: **every child eigenvalue
embeds in the parent spectrum**. Hence the parent's non-zero spectrum is the
union of the child non-zero spectra and the nonzero cross modes
`λ + μ − λμ`. Define `g(L)=min{|lambda|:lambda in spec(L),lambda!=0}`
when that set is nonempty. The exact modulus bound is

$$g(L_{ab})=\min\!\left(g(L_a),g(L_b),
\min_{\substack{\lambda,\mu\ne0\\\lambda+\mu-\lambda\mu\ne0}}
|\lambda+\mu-\lambda\mu|\right)
\le\min(g(L_a),g(L_b)).$$

An empty cross-mode set contributes no minimum. Excluding zero cross modes
matters: additional stationary modes can appear in a product, and the smallest
nonzero modulus does not certify connectedness or convergence to one constant.
Equality holds if both child spectra lie in `[0,1]`, because
`lambda+mu-lambda*mu >= max(lambda,mu)` there. Symmetric connection alone only
places the random-walk Laplacian spectrum in `[0,2]`; it does not imply that
stronger hypothesis.

The implementation's compatibility-named `spectral_gap` uses eigenvalue
moduli above a numerical tolerance. For directed heat flow, decay depends on
real parts, so that value cannot generally be interpreted as a coherence
timescale. The recorded `4 × 9, k=2` modulus is about `0.518` versus child
moduli `1.000`; it is a finite spectral comparison. Numerical eigenvalue
composition matches to about `4e-15` on the reported cases. Neither this
thresholded statistic nor the exact product theorem certifies U5.

## 5. Non-factorizing control for the unrestricted power-residue family

The unrestricted set is the control. `residue_set_factors(a, b, k, unit=False)`
returns `False` — even for coprime primes (`3 × 5`), because the non-unit powers
`{6, 9, 10}` of `ℤ/15ℤ` have no CRT product preimage in
`S_3 × S_5 = {1} × {1, 4}`. Consequently the Kronecker identity **fails** for the
unrestricted operator (`full_power_residue_laplacian`): the measured residual is
`0.30, 0.09, …` — non-zero. These controls distinguish the two implemented
families; they do not make unit membership necessary for every product graph.

## 6. Honest scope — synthesis, not factoring

This is explicitly a **structural (synthesis) branch**: the composition **uses
the known factors** `a, b` to build the parent from its children. The C5
circularity audit therefore records
`graph_construction_requires_answer = True`, yielding verdict `CIRCULAR` and
`permits_discovery_claim = False`, and the experiment manifest sets
`uses_known_factors = True` (claim `NT-P03`). The algebraic product theorem
(Kronecker identity + eigenvalue law, §3–4) stands on its own and is tested
**separately** from any inverse use: recognising a *given* `L_ab` **as** a
product `I-(I-L_a) ⊗ (I-L_b)` is not an inverse operation implemented here.
This code is supplied `a,b`; the statement is an implementation dependency,
not a lower bound on every possible inverse algorithm. The forward direction
assembles a declared product transport model; the inverse
direction (recovering unknown factors) is **not** claimed and is out of scope.

## 7. Claim ledger

| Claim | Basis | Status |
|-------|-------|--------|
| `L_ab = I − (I−L_a)⊗(I−L_b)` (unit set, coprime) | CRT unit bijection + Kronecker | **DERIVED** + MEASURED (exact `0` over ℚ) |
| `λ_parent = λ + μ − λμ` | Kronecker eigenvalues | **DERIVED** + MEASURED (`~4e-15`) |
| `g(L_ab) ≤ min(g(L_a), g(L_b))` | trivial-mode embedding, exact nonzero moduli | **DERIVED**; finite thresholded comparisons are separate |
| unrestricted family can fail to factor | control residual `≠ 0` | **MEASURED**; not a universal necessity claim |
| factoring / discovery | — | **NEGATIVE** (synthesis only; `NT-P03` forbids discovery claim) |

R3 supplies an exact finite product identity for a declared family of unit
residue networks. The operator and spectrum compose as shown; joint nodal
emergence, full-tetrad transport, U5 and generic decay-time claims do not follow.
It closes no open problem and provides no factoring capability.
