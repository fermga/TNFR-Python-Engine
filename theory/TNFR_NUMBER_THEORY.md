# TNFR Number Theory: Arithmetic Constructions and Structural Read-outs

**Status**: Arithmetic construction reference; exact restricted identities and finite diagnostics, not a derived physical or autonomous nodal model
**Version**: 0.0.3.5
**Date**: March 2026

> **Scope correction (2026-09-18).** Arithmetic functions, modular arithmetic,
> graph constructors and assigned logarithmic frequencies are declared inputs.
> Conditional spectral consequences can be derived from those inputs; this is
> not a derivation of physical particles or of the arithmetic carrier itself
> from the nodal equation. The historical Riemann family in §§10.1–10.4 is
> superseded. General symmetry-complement and REMESH-infinity claims below are
> restricted by the current
> [Riemann program memo](TNFR_RIEMANN_RESEARCH_NOTES.md): no analytic location
> of `S(T)`, universal operator equivariance or RH result has been established.
> Historical numerical results are retained, not re-executed by this correction.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Arithmetic Networks as TNFR Systems](#2-arithmetic-networks-as-tnfr-systems)
3. [The Arithmetic Structural Triad](#3-the-arithmetic-structural-triad)
4. [Primality as Structural Equilibrium](#4-primality-as-structural-equilibrium)
5. [Canonical Arithmetic Constants](#5-canonical-arithmetic-constants)
6. [Pressure Component Analysis](#6-pressure-component-analysis)
7. [The Arithmetic Tetrad](#7-the-arithmetic-tetrad)
8. [Dual-Lever Decomposition](#8-dual-lever-decomposition)
9. [Factorization as Spectral Decoding](#9-factorization-as-spectral-decoding)
10. [Prime Path Graphs and the TNFR-Riemann Connection](#10-prime-path-graphs-and-the-tnfr-riemann-connection)
11. [Worked Examples](#11-worked-examples)
12. [Implementation Map](#12-implementation-map)
13. [Reference Questions and Research Boundaries](#13-reference-questions-and-research-boundaries)
14. [References](#14-references)

---

## 1. Introduction

Arithmetic networks can be assigned TNFR-compatible coordinates and read-outs.
Their relation to the nodal equation

$$\frac{\partial\mathrm{EPI}}{\partial t} = \nu_f \cdot \Delta\mathrm{NFR}(t)$$

requires a separately declared evolution: assigning arithmetic functions to
nodes and edges alone supplies a static state, not a trajectory. In the
construction studied here:

- **Primes are the arithmetic pressure zero set**: $\Delta\mathrm{NFR}(p) = 0$ for all primes $p$.
- **Composites carry structural pressure**: $\Delta\mathrm{NFR}(n) > 0$ whenever $n$ is composite, with a magnitude determined by the three selected arithmetic descriptors.
- **Factorization as spectral decoding**: discovering the factors of a composite can be framed as resolving the coherent sub-modes of its structural pressure field.

This document formalizes these observations, expresses the arithmetic constants as canonical units (only $\pi$ is a genuine structural scale), and maps the theory to its implementations in the repository.

### Scope

| Layer | Description | Source |
|-------|-------------|--------|
| **Primality** | Deterministic prime detection via $\Delta\mathrm{NFR}=0$ | `primality-test/`, `src/tnfr/mathematics/number_theory.py` |
| **Factorization** | Spectral factor discovery via Paley-Jacobi graphs | `factorization-lab/` |
| **Riemann program** | Declared arithmetic traces and finite pulse comparisons; historical prime-path prototype superseded | `src/tnfr/riemann/` |

These layers reuse selected arithmetic functions, graph diagnostics and
configured coefficients. This reuse does not establish a shared autonomous
evolution or U1-U6 compliance. Grammar claims require an actual declared
operator sequence and its checks; static primality and spectral calculations
do not supply such a history.

---

## 2. Arithmetic Networks as TNFR Systems

### 2.1 Network Construction

A TNFR arithmetic network $G = (V, E)$ is a directed graph where:

- **Nodes** $V = \{2, 3, \ldots, N\}$ are natural numbers.
- **Edges** encode two types of structural relationship:
  - **Divisibility edges**: $(d, n)$ for each divisor $d \mid n$ with $d < n$.
  - **GCD coupling edges**: $(a, b)$ when $\gcd(a, b) > 1$, weighted by $\gcd(a, b) / \max(a, b)$.

Each node $n$ is assigned the structural triad (EPI, $\nu_f$, $\phi$)
from chosen arithmetic functions, together with a separate arithmetic pressure
read-out $\Delta\mathrm{NFR}(n)$. These assignments do not derive a capacity
law, phase law or event schedule.

### 2.2 Sieve-Based Computation

Efficient computation uses a Lowest Prime Factor (LPF) sieve:

$$\text{lpf}[n] = \min\{p \text{ prime} : p \mid n\}$$

From the LPF array, factorization of any $n \leq N$ is $O(\log n)$, enabling computation of all arithmetic functions ($\Omega$, $\tau$, $\sigma$) for the entire network in $O(N \log \log N)$ sieve time plus $O(N \log N)$ factorization time.

### 2.3 Phase Assignment

Phase is a declared read-out policy. `ArithmeticTNFRNetwork.compute_phase`
defaults to the angle of a two-dimensional spectral layout, falling back to
`logn`; it also supports capacity normalization (`nuf`). The `logn` policy is

$$\phi_n=2\pi\,\operatorname{frac}\!\left(\frac{\log n}{\log(N+1)}\right).$$

The conservation/substrate adapters default to `logn`. A spectral-layout
orientation is not an independently derived physical phase, and none of these
static assignments supplies a phase evolution law. The former linear `n/N`
formula did not describe this implementation.

Coupling operations must separately check the wrapped U3 condition
$|\operatorname{wrap}(\phi_i-\phi_j)|\leq\Delta\phi_{\max}$. The arithmetic
graph construction and phase assignment alone do not guarantee that every
arithmetic edge satisfies that condition.

---

## 3. The Arithmetic Structural Triad

The assigned triad is (EPI, $\nu_f$, $\phi$), with phase defined in §2.3.
The form and capacity assignments below are followed by the separate pressure
and coherence read-outs. For a fixed integer label these are static functions.

### 3.1 Form: EPI(n)

The Primary Information Structure of a natural number measures its overall arithmetic complexity:

$$\mathrm{EPI}(n) = 1 + \alpha \cdot \Omega(n) + \beta \cdot \ln(\tau(n)) + \gamma_{\mathrm{epi}} \cdot \left(\frac{\sigma(n)}{n} - 1\right)$$

where:
- $\alpha = 1$ — factorization complexity weight (canonical unit, §5)
- $\beta = 1$ — divisor complexity weight (canonical unit, §5)
- $\gamma_{\mathrm{epi}} = 1$ — abundance deviation weight (canonical unit, §5)

**Model interpretation**: EPI(n) is a chosen scalar arithmetic-complexity
coordinate. Calling it form does not derive an oscillator or a physical state
space from the nodal equation.

### 3.2 Frequency: $\nu_f(n)$

The reorganization capacity of a number measures how rapidly its structural form could evolve:

$$\nu_f(n) = \nu_0 \cdot \left(1 + \delta \cdot \frac{\tau(n)}{n} + \varepsilon \cdot \frac{\Omega(n)}{\ln(n)}\right)$$

where:
- $\nu_0 = 1$ — base frequency (canonical unit, §5)
- $\delta = 1$ — divisor density modulation (canonical unit, §5)
- $\varepsilon = 1$ — factorization complexity modulation (canonical unit, §5)

**Model interpretation**: This positive arithmetic function supplies a candidate
capacity if an evolution is declared. Its definition alone specifies neither
an observed reorganization rate nor a phase angular velocity.

### 3.3 Pressure: $\Delta\mathrm{NFR}(n)$

The structural pressure equation is the central result of arithmetic TNFR:

$$\boxed{\Delta\mathrm{NFR}(n) = \zeta \cdot (\Omega(n) - 1) + \eta \cdot (\tau(n) - 2) + \theta \cdot \left(\frac{\sigma(n)}{n} - \left(1 + \frac{1}{n}\right)\right)}$$

where $\Omega(n)$ is the prime factor count with multiplicity, $\tau(n)$ the divisor count, and $\sigma(n)$ the divisor sum. The coefficients are canonically $\zeta = \eta = \theta = 1$ (unit weights; see §5).

**Model interpretation**: This nonnegative arithmetic pressure vanishes exactly
at primes under the stated positive coefficients. It is a selected measure of
compositeness, not a derived metric distance or a signed restoring response to
an EPI perturbation.

### 3.4 Local Coherence

From the pressure, local coherence is derived:

$$C_{\text{local}}(n) = \frac{1}{1 + |\Delta\mathrm{NFR}(n)|}$$

Primes have $C_{\text{local}} = 1$ (perfect coherence); composites have $C_{\text{local}} < 1$, decreasing with structural complexity.

---

## 4. Primality as Structural Equilibrium

### 4.1 The Fundamental Theorem

**Theorem (TNFR Primality Criterion)**: For any integer $n \geq 2$:

$$n \text{ is prime} \iff \Delta\mathrm{NFR}(n) = 0$$

**Proof**: Each pressure component vanishes independently for primes and is strictly positive for composites:

1. **Factorization component**: $\Omega(p) = 1$ for all primes $p$, so $\zeta \cdot (\Omega(p) - 1) = 0$. For composites, $\Omega(n) \geq 2$, giving $\zeta \cdot (\Omega(n) - 1) \geq \zeta > 0$.

2. **Divisor component**: $\tau(p) = 2$ for all primes (divisors: 1 and $p$), so $\eta \cdot (\tau(p) - 2) = 0$. For composites, $\tau(n) \geq 3$, giving $\eta \cdot (\tau(n) - 2) \geq \eta > 0$.

3. **Abundance component**: For primes, $\sigma(p) = 1 + p$, so $\sigma(p)/p = 1 + 1/p$, making $\theta \cdot (\sigma(p)/p - (1 + 1/p)) = 0$. For composites $n$ with a proper divisor $d \notin \{1, n\}$, $\sigma(n) > 1 + n$, giving $\sigma(n)/n > 1 + 1/n$ and thus $\theta \cdot (\sigma(n)/n - (1 + 1/n)) > 0$.

Since all three terms vanish iff $n$ is prime, and all are non-negative, the equivalence holds. $\square$

**Structural interpretation**: The theorem identifies the pressure zero set
with the prime labels. If this arithmetic pressure is held fixed in the
unforced scalar nodal equation, it gives zero EPI rate at a prime for any finite
capacity. It does not establish a fixed point of phase, capacity, support or
the full operator dynamics. At a fixed composite label the static EPI(n)
assignment has zero time derivative, whereas the assigned positive capacity
and pressure have a positive product. An evolved EPI coordinate therefore
cannot remain identical to that static arithmetic formula; §8 states the
separate held-pressure model explicitly.

### 4.2 Coefficient Independence

The primality criterion $\Delta\mathrm{NFR}(n) = 0$ is **independent of the coefficient values** $(\zeta, \eta, \theta)$, provided all three are positive. Each component vanishes individually for primes. The coefficients affect composite pressure magnitudes, not the arithmetic zero set.

### 4.3 Computational Properties

| Property | Value |
|----------|-------|
| **Determinism** | 100% (no probabilistic component) |
| **Time complexity** | $O(\sqrt{n})$ per number (trial division for $\Omega$, $\tau$, $\sigma$) |
| **Space complexity** | $O(1)$ basic; $O(\text{cache})$ with memoization |
| **Sieve mode** | $O(N \log \log N)$ for all primes up to $N$ |
| **Verified range** | $[2, 10^4]$ exhaustive, $[10^4, 10^{10}]$ selective |
| **Accuracy** | 100% — 0 false positives, 0 false negatives |

---

## 5. Canonical Arithmetic Constants

### 5.1 Canonical Coefficients Are Unity

Per AGENTS.md §3 the only genuine structural constant is $\pi$; $\varphi$, $\gamma$ and $e$ are not structural scales. Earlier versions wrote the triad weights as $(\varphi, \gamma, \pi, e)$ combinations, but that was a *post-hoc notational overlay* fitted to approximate empirical values ($\zeta = 1.0$, $\eta = 0.8$, $\theta = 0.6$) — not a derivation.

By the Coefficient Independence theorem (§4.2), the primality zero set is the
same for **any** positive coefficients. That theorem does not select their
magnitudes. The implementation chooses unit coefficients as the canonical
parameter-free convention, so the arithmetic pressure introduces no additional
scale; the structural content lives in $(\Omega, \tau, \sigma, n)$.

### 5.2 Pressure Coefficients

$$\boxed{\zeta = \eta = \theta = 1}$$

The three pressure channels weigh equally: the factorization excess $\Omega - 1$, the divisor excess $\tau - 2$ and the abundance excess $\sigma/n - (1 + 1/n)$ each contribute on the same unit scale. This is the configured unit-weight convention. Section 4.2 proves that the zero set does not depend on the choice of positive coefficients; it does **not** prove uniqueness of the unit convention.

### 5.3 EPI Parameters

| Parameter | Value | Model meaning |
|-----------|-------|-----------------|
| $\alpha$ | $1$ | Factorization complexity weight |
| $\beta$ | $1$ | Divisor logarithmic weight |
| $\gamma_{\mathrm{epi}}$ | $1$ | Abundance deviation weight |

### 5.4 Frequency Parameters

| Parameter | Value | Model meaning |
|-----------|-------|-----------------|
| $\nu_0$ | $1$ | Base structural frequency |
| $\delta$ | $1$ | Divisor density modulation |
| $\varepsilon$ | $1$ | Factorization modulation |

### 5.5 Detection Thresholds

Primality is detected by the **exact** criterion $\Delta\mathrm{NFR}(n) = 0$ (§4.1); the only threshold is the floating-point zero tolerance.

| Threshold | Value | Purpose |
|-----------|-------|---------|
| Primality tolerance | $10^{-10}$ | Floating-point zero detection of $\Delta\mathrm{NFR} = 0$ |

Any wider "significance band" is an operational convenience, not a structural constant — only $\pi$ is a genuine structural scale (§5.1).

### 5.6 Derivation Status

The 9 dynamical arithmetic parameters (3 pressure + 3 EPI + 3 frequency) are positive operational weights applied to arithmetic functions (canonical units; the prime ⟺ ΔNFR = 0 criterion is coefficient-independent, §4.2). The arithmetic network uses the same tetrad implementation and selected warning policies as other TNFR graphs; these policies are not graph-independent bounds. An earlier φ/γ/e "arithmetic recalibration" was removed; no domain-specific threshold tuning remains.

---

## 6. Pressure Component Analysis

### 6.1 Three Independent Pressure Channels

The $\Delta\mathrm{NFR}$ equation decomposes structural pressure into three independent channels, each measuring a distinct aspect of compositeness:

#### Factorization Pressure: $P_{\Omega} = \zeta \cdot (\Omega(n) - 1)$

Measures the **total prime factor count with multiplicity**. This is the most direct measure of compositeness: primes have $\Omega = 1$, semiprimes have $\Omega = 2$, prime powers $p^k$ have $\Omega = k$.

| $n$ | Factorization | $\Omega(n)$ | $P_\Omega$ |
|-----|---------------|-------------|------------|
| 7 (prime) | $7$ | 1 | 0 |
| 15 | $3 \times 5$ | 2 | 1 |
| 8 | $2^3$ | 3 | 2 |
| 30 | $2 \times 3 \times 5$ | 3 | 2 |
| 360 | $2^3 \times 3^2 \times 5$ | 6 | 5 |

#### Divisor Pressure: $P_{\tau} = \eta \cdot (\tau(n) - 2)$

Measures the **richness of the divisor lattice**. Primes have exactly 2 divisors; highly composite numbers have many.

| $n$ | $\tau(n)$ | $P_\tau$ |
|-----|-----------|---------|
| 7 (prime) | 2 | 0 |
| 15 | 4 | 2 |
| 8 | 4 | 2 |
| 30 | 8 | 6 |
| 360 | 24 | 22 |

#### Abundance Pressure: $P_{\sigma} = \theta \cdot (\sigma(n)/n - (1+1/n))$

Measures the **deviation of the divisor sum ratio from the prime pattern**. This is the most sensitive to the internal structure of divisors.

| $n$ | $\sigma(n)/n$ | $1+1/n$ | $P_\sigma$ |
|-----|---------------|---------|-----------|
| 7 (prime) | $8/7 \approx 1.143$ | $8/7$ | 0 |
| 15 | $24/15 = 1.600$ | $16/15 \approx 1.067$ | 0.533 |
| 8 | $15/8 = 1.875$ | $9/8 = 1.125$ | 0.750 |
| 30 | $72/30 = 2.400$ | $31/30 \approx 1.033$ | 1.367 |

### 6.2 Component Independence

The three pressure channels are **functionally linearly independent** — no linear (or affine) combination reproduces a third for all $n$. This is proved **exactly over $\mathbb{Q}$** from the witness points $p^2 \mapsto (1, 1, 1/p)$ and $pq \mapsto (1, 2, 1/p + 1/q)$ (`prove_functional_independence`; the $3\times 3$ witness matrix has exact rank 3). They therefore form a **linearly-independent diagnostic profile** of compositeness through $\Omega$, $\tau$, $\sigma$ — **not** a minimal or complete basis: the set is *redundant* for primality (each channel alone is zero iff $n$ is prime, so one channel already suffices), and structural completeness (that no fourth independent pressure degree is relevant) is **unproven**. See [TNFR_ARITHMETIC_PRESSURE.md](TNFR_ARITHMETIC_PRESSURE.md) (R7, NT-P07) for the independence proof, the redundancy result, and the closed fourth-channel gate.

### 6.3 Structural Pressure Landscape

As $n$ grows, the expected pressure for a "random" composite scales as:

$$\mathbb{E}[\Delta\mathrm{NFR}(n)] \sim \ln\ln n + (\ln n)^{\ln 2} + \text{(abundance deviation)}$$

by the Erdős-Kac theorem ($\Omega(n) \sim \ln\ln n$) and divisor function asymptotics. Primes remain at exactly zero regardless of magnitude.

---

## 7. The Arithmetic Tetrad

When the arithmetic network $G$ is constructed, the structural field tetrad ($\Phi_s$, $|\nabla\phi|$, $K_\phi$, $\xi_C$) can be computed using the centralized physics modules:

### 7.1 Structural Potential: $\Phi_s$

$$\Phi_s(n) = \sum_{m \neq n} \frac{\Delta\mathrm{NFR}(m)}{d(n, m)^2}$$

where $d(n,m)$ uses the graph supplied to the field adapter and its declared
path-length policy. Self, unreachable and zero-distance pairs contribute zero
under the shared kernel convention. Explicit edge length, transport weight
fallback and unit fallback must not be conflated. A prime
contributes zero as a pressure **source**, but its potential value still
aggregates pressure from other nodes. This static field does not make primes
dynamical sinks or attractors.

**Warning policy**: $|\Phi_s| < \pi/4 \approx 0.785$ is the shared selected per-node policy; it is not derived from arithmetic or phase wrapping.

### 7.2 Phase Gradient: $|\nabla\phi|$

$$|\nabla\phi|(n) = \frac{1}{|\mathcal{N}(n)|} \sum_{m \in \mathcal{N}(n)} |\operatorname{wrap}(\phi_n - \phi_m)|$$

where $\mathcal{N}(n)$ is the neighborhood used by the field adapter, with zero
at isolates. It measures the supplied phase differences; an arithmetic class
interpretation requires evidence under that phase policy.

**Scale**: $|\nabla\phi| \leq \pi$ is the exact wrapped-angle bound. Any
early-warning level is experiment-dependent; the measured synchronization
onset near $0.29$ is not a universal stability threshold.

### 7.3 Phase Curvature: $K_\phi$

$$K_\phi(n) = \text{wrap\_angle}\!\left(\phi_n - \overline{\phi}_{\mathcal{N}(n)}\right)$$

where $\overline{\phi}_{\mathcal{N}(n)}$ is the circular neighbor mean when
available. Exact represented phasor cancellation gives unavailable curvature;
the strict numeric owner raises rather than inventing a value. No generic
prime/composite boundary interpretation follows from this definition.

**Threshold**: $|K_\phi| < 0.9\pi \approx 2.827$; this is an operational margin inside the exact wrapped bound.

### 7.4 Coherence Length: $\xi_C$

$$C(r) \approx A \cdot e^{-r/\xi_C}$$

The coherence length estimates how far the configured structural-coherence
field correlates across the arithmetic network. Its value is state- and
estimator-dependent; no divergence at twin primes, prime gaps, or an arithmetic
critical point is established here.

### 7.5 Tetrad thresholds on the arithmetic network

The arithmetic network uses the same tetrad implementation and selected
warning policies as any TNFR network:

| Field | Threshold | Source |
|-------|-----------|--------|
| $\Phi_s$ | π/4 ≈ 0.785 (per-node), π/2 ≈ 1.571 (drift), both selected policies | `PHI_S_VON_KOCH_THRESHOLD`, `U6_STRUCTURAL_POTENTIAL_LIMIT` |
| $\lvert\nabla\phi\rvert$ | ≤ π (phase wrap); measured onset ≈0.29 is experiment-dependent | canonical wrapping / measured protocol |
| $K_\phi$ | < 0.9·π ≈ 2.827 (phase-wrap safety) | `K_PHI_CANONICAL_THRESHOLD` |
| $\xi_C$ | path-unit static coherence-product fit; separate dimensionless spectral fallback from the smallest computed eigenvalue above `1e-9` | Estimator provenance required; selected value need not be the true `λ₂` |

An earlier "arithmetic recalibration" introduced topology-specific thresholds
expressed as φ/γ/e combinations (e.g. a $K_\phi$ threshold of 3.2275 that
*exceeded* the π phase-wrap bound and was therefore unreachable). Those values
were not structural scales and have been removed (audit 2026): the arithmetic
network is governed by the same π-bounded phase sector as every other TNFR
network.

### 7.6 The Arithmetic NFR and its Emergent Geometry

The constructed arithmetic network supports a joint **Fractal-Resonant Node**
(NFR; TNFR.pdf §1.4.1) read-out through `ArithmeticTNFRNetwork.nfr()`.
Its following diagnostic facets do not prove spontaneous generation,
self-similarity or persistent identity under an autonomous evolution:

- **Resonant.** By the §4.1 primality theorem the pressure-equilibrium set
  $\{n : \Delta\mathrm{NFR}(n) = 0\}$ is *exactly* the primes.
  `equilibrium_fraction` is the prime density. The canonical static aggregate
  `coherence` is
  $1/(1+\operatorname{mean}|\Delta\mathrm{NFR}|)$ under the static
  $d\mathrm{EPI}=0$ convention. `mean_local_coherence` retains the distinct
  descriptive average
  $\operatorname{mean}_i[1/(1+|\Delta\mathrm{NFR}_i|)]$; the two generally
  differ because the coherence kernel is nonlinear. The arithmetic pressure is
  independent of EPI, so this identifies the arithmetic zero set, not full-state
  fixed points, restoring attractors or basins. An empty domain reports these aggregates
  as unavailable rather than assigning it zero coherence.
- **Geometric.** `classify_nodal_topology` assigns a radial, annular or
  multinodal diagnostic from the constructed potential. Its result depends
  on the graph, pressure and classification policy; no universal arithmetic
  center classification is established here.
- **Scale read-out.** The available $\xi_C$ estimate is a correlation or
  spectral statistic with recorded provenance, not proof of fractality or a
  uniquely determined NFR region size.

The arithmetic network exposes structural read-outs through `conservation()`
and initializes the separate auxiliary harmonic model through
`symplectic_substrate()`.  These diagnostics do not show that the arithmetic
nodal dynamics generates a symplectic geometry:

- a **Noether-like charge diagnostic** $Q = \sum_i (\Phi_s(i) + K_\phi(i))$ and
  the nonnegative structural **energy candidate**
  $E = \tfrac12 \sum_i (\Phi_s^2 + |\nabla\phi|^2 + K_\phi^2 + J_\phi^2 +
  J_{\Delta\mathrm{NFR}}^2)$, with the potential sector $\Phi_s$ sourced by the
  arithmetic $\Delta\mathrm{NFR}$ (the genuine invariants $\Omega, \tau, \sigma$);
- an auxiliary **symplectic substrate** of dimension $4N$ with conjugate pairs
  $(K_\phi, J_\phi)$ and $(\Phi_s, J_{\Delta\mathrm{NFR}})$. The geometric sector
  is populated by the *size / capacity* phase $\phi(n) \propto \log n$ (the monoid
  homomorphism $(\mathbb{N},\times) \to (\mathbb{R},+)$), which is non-degenerate
  on the dense divisibility graph. These extracted coordinates initialize the
  harmonic substrate model; they do not prove that the arithmetic graph
  trajectory remains in its realizable image.

The pressure contributes to potential and the chosen phase contributes to the
phase sector. Relative numerical magnitudes depend on input scales and policy;
no universal potential-dominance theorem follows.

---

## 8. Dual-Lever Decomposition

### 8.1 The Nodal Equation in Arithmetic

For a separately declared evolution initialized from the arithmetic assignment,
write its evolving EPI coordinate as $x_n(t)$:

$$\dot x_n(t) = \nu_f(n)\,\Delta\mathrm{NFR}(n),\qquad x_n(0)=\mathrm{EPI}(n).$$

Holding those arithmetic capacity and pressure values fixed is an additional
model assumption. The channel terminology then distinguishes:

- **Capacity lever** ($\nu_f$): How fast the number *can* reorganize. Depends on divisor structure and factorization complexity. Its primary operators are SHA, VAL and NUL.
- **Pressure lever** ($\Delta\mathrm{NFR}$): How much reorganization is *demanded*. Zero for primes, positive for composites. Its primary operators are IL, OZ, THOL and NAV.
- **Phase channel** ($\theta$): UM and ZHIR act primarily on phase.
- **Form channel** (EPI): AL, EN, RA and REMESH write or transport form.

### 8.2 Fixed Point Analysis

For primes: $\Delta\mathrm{NFR}(p) = 0 \Rightarrow \dot x_p = 0$ for any finite $\nu_f(p)$ in this unforced model.

This is stationarity of the unforced EPI coordinate under the held-pressure
model: perturbations to $\nu_f$ alone do not change its zero derivative. It is
not a full-state fixed-point, stability or attraction theorem.

For composites with the assigned positive capacity:
$\Delta\mathrm{NFR}(n) > 0 \Rightarrow \dot x_n > 0$.

The evolved scalar $x_n(t)$ increases under this fixed positive
pressure. Its sign does not prove simplification, attraction toward a prime,
or stability. If capacity and arithmetic pressure remain fixed and positive,
the nodal law gives linear drift rather than relaxation (§13.3).

### 8.3 Experimental Confirmation

Operator-tetrad synergy experiments (examples 37-39) confirmed:

1. $\Phi_s$ responds **linearly** to $\Delta\mathrm{NFR}$ perturbations with $|r| = 1.000$ (perfect correlation), confirming the pressure lever's direct coupling to the structural potential field.
2. Declared operator changes can affect capacity, pressure and the resulting
   EPI rate. Phase and support also enter the tetrad directly; these dependencies
   do not establish a closed causal chain through EPI rate alone.
3. Finite operator trajectories expose measured changes in the tetrad energy
  candidate. Grammar validity and the nominal multiplier $\Pi$ do not by
  themselves prove $dE/dt \leq 0$.

---

## 9. Factorization as Spectral Decoding

### 9.1 The Factorization Problem in TNFR Terms

Given a composite $n$ with $\Delta\mathrm{NFR}(n) > 0$, factorization is the process of decomposing the structural pressure into coherent sub-modes, each corresponding to a prime factor.

**Physical analogy**: A composite number is like a coupled oscillator system with multiple resonant frequencies. Factorization identifies the individual frequencies (prime factors) from the combined signal.

### 9.2 Spectral Paley-Jacobi Method

The implementation uses Paley graphs — algebraic constructions from quadratic residues:

1. **Graph construction**: For modulus $m$ (chosen near $n$), build the Paley graph $G(m)$ where nodes are $\{0, \ldots, m-1\}$ and edges connect quadratic residues.

2. **Spectral decomposition**: Compute the spectrum of the **emergent structural-diffusion operator** $L_{rw} = I - D^{-1}W$ (the canonical ΔNFR EPI channel; `_laplacian_eigenvalues` routes through `structural_diffusion_operator`). On the residue/Paley graph, which is **regular**, $L_{rw}$ shares eigenvectors with the classical Laplacian and the eigenvalues differ only by the degree ($\lambda_{\text{classical}}=d\cdot\lambda_{rw}$), so the Fiedler-gap → prime-size map (a Paley Gauss-sum fact) is preserved while the operator provenance is the emergent TNFR transport operator.

3. **Tetrad proxies** (HONEST SCOPE): the factorizer operates on the spectrum, not on a node-level ΔNFR field, so it uses **scalar proxies** of the tetrad — $\Phi_s\approx$ normalized edge density, $\xi_C\approx 1/(\nu_f\lambda_2)$ (the emergent diffusion relaxation time). These are labelled proxies in code (`_structural_potential`, `_coherence_length`); in the symmetric-seed fixtures of example 117 the genuine per-node tetrad (`tnfr.physics.canonical`) does not distinguish the factor cosets. This is a result for that state/observer pair, not unconditional per-node blindness.

4. **Operator sequence**: Apply the canonical decoder $[\mathrm{UM}, \mathrm{RA}, \mathrm{IL}, \mathrm{THOL}]$ per partition:
   - **UM** (Coupling): Phase-gated coupling between quadratic residues (U3 verified)
   - **RA** (Resonance): Amplify coherent periodicity patterns
   - **IL** (Coherence): Stabilize the partitioned structure
   - **THOL** (Self-organization): Preserve multi-scale identity (U5)

5. **Factor inference**: Detect periodicities in the stabilized partitions that correspond to $n/p$ for candidate factors $p$.

6. **TNFR certification**: Verify each candidate against 8 structural criteria (§9.3).

### 9.3 Structural Verification Criteria

A factor candidate is TNFR-certified when $\geq 4$ of 8 criteria hold and $\geq 50\%$ of partition endorsements are positive:

| Criterion | Threshold | Configured feature interpretation |
|-----------|-----------|---------------|
| $\Delta\mathrm{NFR}$ gain | $\geq 0.15$ drop | Selected heuristic drop, not a convergence theorem |
| Coherence ratio | $0.72 \leq r \leq 1.38$ | Structural similarity |
| $\Phi_s$ delta | $\leq 0.35$ | Selected factor-certification feature; not the canonical U6 $\pi/2$ drift policy |
| Gradient delta | $\leq 0.40$ | Phase desynchronization limit |
| Curvature delta | $\leq 0.45$ | Geometric stability |
| Periodicity confidence | $\geq 0.55$ | Structural mode certainty |
| Stabilized fraction | $\geq 0.30$ | Selected heuristic fraction; not a U5 proof |
| Coverage fraction | $\geq 0.15$ | Spatial completeness |

### 9.4 Pure Mode

Setting `TNFR_PURE_MODE=1` requests the factorizer's structural-confidence policy. A positive heuristic label without arithmetic divisibility verification is not a mathematical factor certificate or proof of a TNFR-specific mechanism.

### 9.5 Three Sectors of Primality (Unification — MEASURED)

The factorization machinery (§9.1–9.4), arithmetic pressure criterion (§4) and
residue-graph diagnostics reuse selected arithmetic structures, but they have
different inputs and observations. Example
[117_emergent_geometry_residue_graph.py](../examples/08_emergent_geometry/117_emergent_geometry_residue_graph.py)
and `benchmarks/primes_as_consequence.py` compare them. Using the same
`L_rw` owner does not identify their full dynamics or prove one common carrier:

| Sector | Method | Input | Emergent? |
|--------|--------|-------|-----------|
| **A — Arithmetic** | $\Delta\mathrm{NFR}(n)=0$ (§4) | $\Omega, \tau, \sigma$ (the factorization) | **re-expression** (primes-IN; exact but circular as a derivation) |
| **B — Spectral** | Paley eigenvalue comparison with explicit normalization: $d\lambda_2(L_{rw})$ versus $(n-\sqrt n)/2$ on the regular self-adjoint prime case | prescribed modular arithmetic and residue graph | Spectral diagnostic without factorization input; scope and prime-power controls in §§9.6–9.7 |
| **C — Representation** | irreducibility (Schur $\langle\chi,\chi\rangle=1$) | a finite group | **refuted** (the dim-4 mode of $K_5$ is irreducible yet $4=2\cdot 2$) |

**The unification, stated honestly:**

1. **Sector B avoids factorization input.** The Paley comparison reads the
   spectrum of a prescribed residue graph rather than supplying
   $\Omega,\tau,\sigma$. Its prime and prime-power distinctions must retain
   the stated operator and finite controls (§9.6). Modular arithmetic and
   the graph construction are still inputs; neither their physical realization
   nor a generic mechanism generating primes follows from this diagnostic.

2. **The measured factor signal is spectral in these fixtures.** For a semiprime $n=p\cdot q$ the factor $p$ appears as an **exact Fourier/coset mode** of the emergent diffusion spectrum ($\eta^2_{\text{coset}}\to 1$, collapsing under a node-label shuffle — example 117 Q2). The residue graph is **regular/circulant**, so the random-walk operator and classical Laplacian share eigenvectors: this is CRT structure re-expressed. For the symmetric-seed per-node substrate fixture, the tested fields do not distinguish the cosets ($\eta^2\approx0$). Other non-invariant states or observers require separate tests.

3. **The obstructions are not identified with one another.** Directed spectra
   retain information discarded by the tested self-adjoint comparison. Their
   Gauss-sum phases have not been identified with analytic
   $S(T)=\pi^{-1}\arg\zeta(1/2+iT)$. The former equalities with a
   REMESH-infinity kernel or a finite symmetry complement are withdrawn by
   the Riemann memo's scope correction. Examples
   [94](../examples/07_number_theory/94_generative_number_construction.py) and
   [97](../examples/07_number_theory/97_goldbach_additive_multiplicative.py)
   remain declared arithmetic constructions; they do not establish that all
   factorization, additive or zeta questions share one missing operator.

**Net:** sector B supplies conditional spectral information without factoring
the input first. The per-node blindness measured for particular symmetric
seeds does not exclude information in other states or observers. No physical
entity-generation mechanism or analytic zeta bridge has been derived here.

### 9.6 The Phase Sector — Sector B Extended to All Odd Primes (MEASURED)

The "partial" limitation of sector B (only $n\equiv 1\pmod 4$, §9.5) is **not** a wall of TNFR — it is an artefact of restricting to the *real/self-adjoint* spectrum. Example [119_phase_sector_directed_residue.py](../examples/08_emergent_geometry/119_phase_sector_directed_residue.py) crosses it using the **same canonical emergent operator** on the **directed** residue graph.

**The structural reason for the prime mod-4 split.** For an odd prime
$p\equiv1\pmod4$, $-1$ is a quadratic residue and the regular residue graph
is symmetric, with self-adjoint $L_{rw}$. For an odd prime
$p\equiv3\pmod4$, the residue digraph is a Paley tournament: its circulant
$L_{rw}$ is normal but not self-adjoint and has complex eigenvalues. These
prime-specific statements do not follow from the congruence alone for
arbitrary composite moduli. An eigenvalue's imaginary part is not itself the
engine's nodal phase variable.

**Scope.** This is `structural_diffusion_operator` applied directly to a
`networkx.DiGraph`, verified against a hand-built matrix
($\max|\Delta|=0$). Its complex spectrum is a diagnostic of that directed
linear transport operator; it is not the auxiliary symplectic substrate or a
complete emergent geometry. The arithmetic input is $x^2\bmod n$.

**Measured (all reproducible in example 119):**

1. **Finite primality comparison.** A three-distinct-eigenvalue rule matches
   the prime labels in **58/58** tested odd integers in `[5,119]`. The
   forward odd-prime count follows from §9.11; the finite comparison does
   not prove the converse for all composite integers.

2. **Prime powers resolved.** The directed operator gives **4+** distinct eigenvalues for $9=3^2$, $25=5^2$, $49=7^2$, $121=11^2$ — it **separates** primes from prime powers, which the real symmetric operator of example 117 could **not** ($49$ was rigid there, the honest §9.5 caveat). The phase channel removes that caveat.

3. **The phase encodes $\sqrt n$.** For $n\equiv 3\pmod 4$ primes the imaginary spectrum is the Paley-tournament eigenvalue structure $(-1\pm i\sqrt n)/2$ on the adjacency; the diffusion operator's $\max|\mathrm{Im}(\lambda)|=\sqrt n/(n-1)$ **exactly** (ratio $1.000$) — a Gauss-sum fact carried in the **phase**.

**Honest scope.** The finite comparisons improve the tested prime-power
distinction without factorization input. The odd-prime rank is the classical
Gauss-period result proved in §9.11; the finite converse tests are not a proof
for every composite. These spectra have not been identified with the tetrad
field $\Psi=K_\phi+iJ_\phi$, an analytic symmetry complement or the zeta
argument. No factoring speedup, particle mechanism or open-problem result follows.

### 9.7 The Symmetry Wall — Why the Substrate Is Blind and the Spectrum Is Not (MEASURED)

§9.6 detects primality in a **global diffusion spectrum**; §9.5 and examples
103/116 found the extracted per-node fields used to initialize the auxiliary
substrate blind to arithmetic. These are different summaries of the same
residue digraph, so their information content need not agree. Example
[120_symmetry_wall_substrate_vs_spectrum.py](../examples/08_emergent_geometry/120_symmetry_wall_substrate_vs_spectrum.py)
locates the finite observation loss at **vertex-transitivity**.

**The mechanism.** The residue digraph is a **Cayley digraph** of $\mathbb{Z}_n$ with connection set $=$ the quadratic residues. Translation preserves the difference $j-i$ and acts transitively. Consequently an equivariant per-node observer applied to an invariant input must be orbit-constant; this does not constrain an arbitrary symmetry-breaking input.

- In the symmetric-seed protocol, the per-node substrate output lies in the fixed sector and cannot distinguish nodes within the single orbit.
- The tested arithmetic distinction appears in the global spectrum (group-character / Gauss-sum eigenvalues). A scalar spectral rank is an invariant, not itself a vector in $\mathrm{Fix}(G_{\mathrm{aut}})^\perp$.

**The double dissociation (measured).** Compare the Paley residue digraph (QR structure) against a **random regular tournament** of the same out-degree, both seeded identically and evolved by the canonical nodal equation $\partial\mathrm{EPI}/\partial t=\nu_f\cdot\Delta\mathrm{NFR}$. Table regenerated after the 2026-09-04 canonicity audit fixed the directed ΔNFR orientation (default path previously computed $L_{\mathrm{in}}$ instead of the canonical $L_{\mathrm{out}}$ on digraphs; the eigenvalue columns are orientation-invariant and unchanged, only $\sigma(\Phi_s)$ shifted):

| $n$ | Paley distinct eig. | random distinct eig. | Paley $\sigma(\Phi_s)$ | random $\sigma(\Phi_s)$ |
|----:|--------------------:|---------------------:|------------------------:|-------------------------:|
| 11 | **3** | 11.0 | 0.376 | 0.356 |
| 23 | **3** | 23.0 | 0.559 | 0.533 |
| 47 | **3** | 47.0 | 0.978 | 0.968 |

- **Spectrum SEES the arithmetic**: Paley is rigidly 3 distinct eigenvalues (the §9.6 prime signature); the random tournament has $\sim n$. Swapping the QR structure for a random tournament changes the spectrum completely.
- **The tested substrate readout does not discriminate**: the per-node $\Phi_s$ dispersion is similar for Paley and the random tournament in the recorded cases. This finite comparison is not an all-state impossibility theorem.

Across odd $n$ the spectral test "$3$ distinct $\iff$ prime" is **18/18 correct**, while $\sigma(\Phi_s)$ grows monotonically with $n$ (graph size) and composites can exceed primes (e.g. $25$ vs $29$) — the substrate tracks size, not primality.

**Cross-program comparison.** Vertex transitivity explains the fixed-sector limitation of the invariant-input residue fixtures. The paused TNFR-Riemann program has its own separately defined $S_n$ representation and oscillatory residue. The two settings share representation-theoretic language, but they are not thereby one obstruction or one state space.

**Honest scope.** This explains the recorded fixed-sector result for invariant
inputs by vertex transitivity. It does not prove that every per-node observer or
perturbed state is blind, does not identify the Riemann obstruction with this
finite graph result, and closes no open problem.

### 9.8 Symmetry-breaking controls on the declared residue fixtures

Example [121](../examples/08_emergent_geometry/121_canonical_symmetry_break_negative.py)
compares a symmetric seed, capacities assigned from selected graph invariants,
and an arithmetic capacity assignment with a shuffled control. It is a finite
comparison on the chosen vertex-transitive residue graphs.

The channel-weight dictionary is graph-level configuration. This does not mean
the nodal equation has no local dependence: capacity, phase, EPI and edge
conductance are node/edge data, and the pressure depends on them. A missing
configuration slot is an implementation fact, not a physical impossibility
theorem or proof that every possible local law needs an external axiom.

| Lever | Historical observation | Scope |
|---|---|---|
| Symmetric seed | Potential dispersion near numerical zero | Consistent with an invariant state and equivariant read-out |
| Degree/triangle-derived capacity | Uniform capacity on these graphs | A function of invariant bare-graph data is orbit-constant |
| Arithmetic capacity and shuffle | Dispersion ratio about 0.96–1.05 | The selected statistic did not separate those assignments in this sample |

The exact symmetry statement is conditional: a deterministic equivariant
evolution with invariant full initial state cannot select a non-invariant state
while its uniqueness/domain assumptions hold. It does not exclude symmetry
breaking from an admitted non-invariant state, instability under perturbations,
support changes, a different observable or a finer TNFR description. A uniform
EPI field alone is not sufficient for zero multichannel pressure if phase,
capacity or topology contributions differ.

The previous universal conclusion “no canonical per-node lever exists” is
withdrawn. These controls do not settle every endogenous selection mechanism
and do not identify the Riemann obstruction with a finite fixed-sector result.

### 9.9 Phase-Sector Periodicity — a Basis-Invariant Read of the CRT Factor Coset (MEASURED)

§9.6 (Reading B, real sector) recovered the factor coset $(i\bmod p)$ of a semiprime $n=p\cdot q$ as a Fourier mode of the emergent diffusion spectrum. Example [122_factorization_phase_sector.py](../examples/08_emergent_geometry/122_factorization_phase_sector.py) reads that periodic structure with a **basis-invariant** observable and a **derived-tolerance certificate** — replacing an earlier, non-canonical single-eigenvector $\eta^2>0.9$ decision rule (2026-09-04 canonicity audit).

**The structural fact (CRT, present in both sectors).** For $n=p\cdot q$ the factor coset $(i\bmod p)$ corresponds to the Fourier frequencies $k=$ multiples of the cofactor $q$. A pure Fourier mode $\exp(2\pi i k j/n)$ with $k$ a multiple of $q$ is **constant within each coset** $(i\bmod p)$, hence an **exact eigenvector** of the emergent operator (a circulant / Cayley digraph) — verified to machine precision (eigenvector residual $\sim 10^{-14}$) for BOTH the undirected (real) and directed (complex) residue operator. The factor coset is CRT/circulant structure ($\mathbb{Z}_n\cong\mathbb{Z}_p\times\mathbb{Z}_q$), present in both spectra.

**Why a single-eigenvector $\eta^2$ is not canonical.** A degenerate eigenvalue defines an **eigenspace**, not a privileged eigenvector: if $Q_\lambda$ spans it, so does $Q_\lambda U$ for any unitary $U$. Any quantity read off individual eigenvector columns — e.g. the maximum $\eta^2$ of a single column, thresholded at $0.9$ — can change under that rotation while the operator is unchanged; it is an artefact of the eigensolver's basis. The audit exhibited explicit counterexamples ($n=209,253,299$) where the $0.9$ threshold selects a **false** divisor.

**The canonical, basis-invariant observable + certificate.** Let $C_d$ be the subspace of vectors constant on classes $(i\bmod d)$ with the constant direction removed, $P_d$ its projector, and $\Pi_\lambda$ the spectral projector of a (possibly degenerate) eigen-cluster. The score
$$\mathrm{score}(d,\lambda)=\lVert P_d\,\Pi_\lambda\rVert_2^2$$
depends only on the two subspaces, hence is invariant under
$Q_\lambda\to Q_\lambda U$. The exact algebraic condition is
`r(d)=0` if and only if the chosen subspace is invariant under `L`.
The implementation instead accepts the **numerical** tolerance test
$$r(d)=\lVert (I-P_d)\,L\,Q_d\rVert_2 < \tau,\qquad \tau=\sqrt{\varepsilon}\,\lVert L\rVert_2$$
(a selected square-root-machine-epsilon scale times the operator norm).
This is approximate invariant-subspace evidence. It is not an exact rational
certificate or a universal equivalence between small residual and an integer
being a divisor. A separate arithmetic check is needed for an actual factor.

**Measured (seed-free; floating linear algebra).**

| $n$ | true $p$ | false $d$ | $\mathrm{score}(p)$ | $\mathrm{score}(d)$ | $r(p)$ | $r(d)$ |
|---|---|---|---|---|---|---|
| 209 | 11 | 3 | 1.0000 | 0.9162 | $1.6\times10^{-15}$ | $3.9\times10^{-2}$ |
| 253 | 11 | 7 | 1.0000 | 0.9369 | $1.1\times10^{-15}$ | $5.3\times10^{-2}$ |
| 299 | 13 | 7 | 1.0000 | 0.9373 | $1.2\times10^{-15}$ | $4.8\times10^{-2}$ |

The false candidate scores **above $0.9$** (so the withdrawn threshold mis-fires) yet its residual $r(d)\gg\tau$ **rejects** it; the true factor certifies ($r(p)\sim10^{-15}<\tau$). The projector score is invariant under a unitary rotation of a degenerate eigenspace ($\Delta\sim10^{-16}$). A label shuffle drives $r(p)$ from $\sim10^{-15}$ to $O(1)$ — the signal is CRT/circulant structure, not an artefact.

**Honest scope.** This is a **periodicity diagnostic** of CRT structure in the canonical emergent spectrum — **not** a factoring algorithm. The residue (di)graph has $n$ nodes; building and diagonalizing it is $\mathrm{poly}(n)=\mathrm{poly}(2^L)$, i.e. **exponential in the input size** $L=\log_2 n$ bits, and the candidate scan is $O(\sqrt n)$ prime divisors (the order of trial division). There is **no speedup** and **no cryptographic consequence**. The correction relative to earlier drafts: the read is expressed with a basis-invariant subspace score and a derived-tolerance certificate; the non-canonical $\eta^2>0.9$ rule is withdrawn.

### 9.10 The Symmetry-Sector Decomposition — the General Principle Behind the Whole Arc (MEASURED, CAPSTONE)

§9.7 located the residue-digraph wall at vertex-transitivity. Example [123_symmetry_sector_decomposition.py](../examples/08_emergent_geometry/123_symmetry_sector_decomposition.py) shows that is a **special case** of a general representation-theoretic principle of the canonical emergent operator — the single structure behind every wall in the §9.5–§9.9 arc (and the Riemann residual).

**The principle (Schur, applied to the canonical emergent operator).** For **any** graph $G$ with automorphism group $\mathrm{Aut}(G)$, the canonical emergent operator $L_{rw}=I-D^{-1}W$ is **equivariant**: it commutes with the permutation representation of every automorphism, $P_\sigma L_{rw}=L_{rw}P_\sigma$ for all $\sigma\in\mathrm{Aut}(G)$. By Schur's lemma an equivariant operator block-diagonalizes by the isotypic components (irreps) of $\mathrm{Aut}(G)$. The coarsest split is

$$\mathbb{R}^N=\mathrm{Fix}(G)\ \oplus\ \mathrm{Fix}(G)^\perp,$$

where $\mathrm{Fix}(G)=\{\text{functions constant on the orbits of }\mathrm{Aut}(G)\}$ is the trivial isotypic component and $\dim\mathrm{Fix}(G)=$ the number of vertex orbits. $L_{rw}$ preserves each block. An equivariant per-node map sends an invariant input to $\mathrm{Fix}(G)$ and is then constant within each orbit. Arbitrary inputs can have nontrivial components; global spectral invariants and nontrivial eigenspaces are separate objects.

**Measured (five symmetry groups — cyclic, full-symmetric, star, path, product).**

| Graph | $\lvert\mathrm{Aut}\rvert$ | orbits | $\dim\mathrm{Fix}(G)$ | equivariance | $L_{rw}$ preserves $\mathrm{Fix}(G)$ |
|---|---:|---:|---:|---:|---:|
| cycle $C_8$ ($D_8$) | 16 | 1 | 1 | $0$ | $0$ |
| complete $K_6$ ($S_6$) | 720 | 1 | 1 | $0$ | $\sim10^{-17}$ |
| star $K_{1,5}$ ($S_5$) | 120 | 2 | 2 | $0$ | $\sim10^{-17}$ |
| path $P_6$ ($\mathbb{Z}_2$) | 2 | 3 | 3 | $0$ | $0$ |
| torus $C_3\square C_3$ | 72 | 1 | 1 | $0$ | $0$ |

- **M1**: equivariance $\lVert P_\sigma L_{rw}-L_{rw}P_\sigma\rVert=0$ (machine zero) for **every** automorphism.
- **M2**: $\mathrm{rank}(P_{\mathrm{triv}})=$ #orbits exactly ($P_{\mathrm{triv}}=$ mean of $P_\sigma$).
- **M3**: $L_{rw}$ preserves $\mathrm{Fix}(G)$ ($\sim10^{-17}$): block-diagonal.
- **M4**: the extracted per-node field vector used to initialize the auxiliary
  substrate satisfies $P_{\mathrm{triv}}v=v$ for the symmetric seed
  (orbit-constant); vertex-transitive $\Rightarrow$ $\sigma(\Phi_s)=0$ in this
  declared construction.
- **M5**: on vertex-transitive cases, the fixed sector consists only of
  constants. On the star and path it has several dimensions, so nonconstant
  orbit-constant modes also belong to that sector. Commuting projectors permit
  a sector-adapted eigenbasis; an arbitrary basis in a degenerate eigenspace
  need not consist of pure-sector vectors.

**The comparison.** Residue-graph symmetry, the measured substrate limitation,
spectral primality and the Riemann programme can each be organized using a
fixed/nontrivial-sector decomposition after their respective group actions are
defined. This is a common method, not proof that their obstructions are the
same object. On invariant inputs, star and path examples resolve orbit classes;
arbitrary perturbed inputs fall outside that conclusion.

**Honest scope.** This is the representation theory of graph automorphisms (Schur's lemma applied to an equivariant operator) re-expressed in the canonical emergent operator. It **explains and unifies** the arc's walls; it is not new mathematics and closes no open problem.

### 9.11 The Cyclotomy Law — Proof via Gauss Periods (PROVED)

§9.6 established the **measured** signature "$3$ distinct eigenvalues $\iff$ odd prime" for the quadratic-residue ($k=2$) digraph. Example [153_structural_frequency_rank_cyclotomy.py](../examples/07_number_theory/153_structural_frequency_rank_cyclotomy.py) generalizes it to the $k$-th power residue network and measures the **cyclotomy law** $s_k(p)=\gcd(k,p-1)+1$. Unlike §9.5–§9.10 (all measured), this law is a **theorem** — it follows from classical Gauss-period theory, here proved for **all** $k$ and every odd prime $p$.

**Setup.** Fix an odd prime $p$ and an integer $k\ge 1$; let $d=\gcd(k,p-1)$ and $\zeta=e^{2\pi i/p}$. Because $(\mathbb{Z}/p\mathbb{Z})^\times$ is **cyclic** of order $p-1$, the nonzero $k$-th power residues $R_k=\{x^k\bmod p\}$ form the unique subgroup $H\le(\mathbb{Z}/p\mathbb{Z})^\times$ of index $d$ (the $d$-th powers), with $\lvert H\rvert=(p-1)/d=:f$. The structural rank $s_k(p)$ is the number of distinct eigenvalues of the canonical $L_{rw}$ on $\mathrm{Cay}(\mathbb{Z}/p\mathbb{Z},R_k)$; since $L_{rw}=I-A/f$ is an affine image of the circulant adjacency $A$, $s_k(p)=\#\{\lambda(t):t\in\mathbb{Z}/p\mathbb{Z}\}$ with
$$\lambda(t)=\sum_{r\in H}\zeta^{tr}.$$

**Theorem (cyclotomy law).** $\;s_k(p)=d+1=\gcd(k,p-1)+1$.

**Proof.**

1. *Coset invariance.* For $t\ne 0$, $\lambda(t)$ depends only on the coset $tH$: if $t'=th$ with $h\in H$ then $\{hr:r\in H\}=H$ (group closure), so $\lambda(t')=\sum_{r\in H}\zeta^{t(hr)}=\lambda(t)$. The cosets partition $(\mathbb{Z}/p\mathbb{Z})^\times$ into $d$ classes, so over $t\ne 0$ the value $\lambda(t)$ takes the $d$ **Gauss periods** $\eta_0,\dots,\eta_{d-1}$ (one per coset); the remaining value is $\lambda(0)=\lvert H\rvert=f$. Hence $s_k(p)\le d+1$.

2. *The $d$ periods are distinct.* The only $\mathbb{Z}$-linear relation among $\{\zeta^i\}_{i=0}^{p-1}$ is $\sum_{i=0}^{p-1}\zeta^i=0$; restricted to two $0/1$-supported sums on $\{1,\dots,p-1\}$, $\sum_{i\in S}\zeta^i=\sum_{i\in S'}\zeta^i\iff S=S'$. Therefore $\sigma_a:\zeta\mapsto\zeta^a$ fixes $\eta_0=\sum_{r\in H}\zeta^r$ iff $aH=H$ iff $a\in H$. So the $\mathrm{Gal}(\mathbb{Q}(\zeta)/\mathbb{Q})\cong(\mathbb{Z}/p\mathbb{Z})^\times$ stabilizer of $\eta_0$ is exactly $H$: $\eta_0$ generates the unique degree-$d$ subfield $K_d=\mathbb{Q}(\zeta)^H$, and its $d$ Galois conjugates $\eta_0,\dots,\eta_{d-1}$ are **distinct**.

3. *No period equals the rational $\lambda(0)$.* For $d\ge 2$: the Galois group permutes $\{\eta_j\}$ **transitively** (through $(\mathbb{Z}/p\mathbb{Z})^\times/H\cong\mathbb{Z}/d\mathbb{Z}$); if some $\eta_{j_0}=f\in\mathbb{Q}$ then every conjugate would equal $f$ (Galois fixes $\mathbb{Q}$), contradicting distinctness. For $d=1$: $H=(\mathbb{Z}/p\mathbb{Z})^\times$, so $\eta_0=\sum_{i=1}^{p-1}\zeta^i=-1\ne p-1=f$. Either way no $\eta_j$ equals $f$.

4. *Conclusion.* The distinct values are exactly $\{f,\eta_0,\dots,\eta_{d-1}\}$ — $(d+1)$ of them — so $s_k(p)=d+1=\gcd(k,p-1)+1$. $\blacksquare$

**Reading of the law.** $s_k(p)-1=\gcd(k,p-1)=[(\mathbb{Z}/p\mathbb{Z})^\times:H]=$ the number of $k$-th power classes $=[K_d:\mathbb{Q}]$, the degree of the cyclotomic subfield carrying the periods. The maximal rank $k+1$ is attained $\iff d=k\iff k\mid p-1\iff p\equiv 1\pmod k\iff p$ splits completely in $\mathbb{Q}(\zeta_k)$. The quadratic case is $k=2$ ($\gcd(2,p-1)=2$ for every odd $p$ $\Rightarrow$ the **uniform rank 3** of §9.6); the extreme $d=p-1$ ($R_k=\{1\}$) is the directed $p$-cycle, all $p$ characters distinct, $s=p=(p-1)+1$.

**Prime versus prime-power scope.** The prime-field proof above also works
at `p=2`: the nonzero power-residue set is `{1}`, the two-node Laplacian
has eigenvalues `0,2`, and `gcd(k,1)+1=2`. It is therefore not an odd-only
prime theorem. The separate conductor-annotated quadratic product formula
over odd prime powers requires its own arithmetic stratification; cyclicity
of the unit group alone does not extend the prime-field independence proof
to higher powers. At `2^e`, the unit group is noncyclic for `e>=3` and the
quadratic strata differ. The recorded annotated counts `2,4,8,10,14,16,20`
for `e=1,...,7` do not follow the odd-prime-power expression
`e+ceil(e/2)+1`. That distinguishes the annotated prime-power formula from
the prime-field cyclotomy theorem; it does not invalidate the latter at two.

**Honest scope.** The cyclotomy law is classical Gauss-period / cyclotomy theory (the $k$-th power Cayley eigenvalues are Gauss periods of degree $\gcd(k,p-1)$); the contribution is the **TNFR structural-diffusion framing** and the closed-form `power_residue_rank` — now a **proved** canonical fact, not a measured pattern. Verified computationally for $k\le 40$ across the primes $p<64$ (680 cases, 0 failures) and proved for all $k$. It detects primality/cyclotomy structurally; it does not factor, does not reach the continuous arg-$\zeta$ phase, and closes no open problem.

### 9.12 The Ontological Position of a Number (the emergent ladder)

§9.5–9.7 compare which finite diagnostics distinguish primality. This
subsection assembles them into a **diagnostic profile** for an integer, measured
in example [155_ontological_position_of_numbers.py](../examples/08_emergent_geometry/155_ontological_position_of_numbers.py).

| Layer | What $n$ **is** | Mechanism | Emergent? |
|-------|-----------------|-----------|-----------|
| 0 Substrate | — | $\mathbb{R}$ continuum + $\pi$ (the one structural scale) | assumed |
| 1 Cardinal | a degeneracy $=\dim$ irrep of $\mathrm{Aut}(G)$ | Laplacian multiplicity | ✅ |
| 2 Operations | $+, \times$ | graph products ($\square\!\to\!\sum$ spectra, $\otimes\!\to\!\prod$ spectra) | ✅ |
| 3 Primality | $\rho(n)=3$ | directed residue operator (§9.6) | ✅ (Sector B) |
| 3′ Arithmetic | the factorization type ($\Omega, \tau$) | the multiplicative rank $\rho(n)$ | ✅ (this §) |
| 4 Missing bridge | prime identities and analytic $\arg\zeta$ | No established common symmetry-complement representation | Open, not an identified wall |

**The annotated rank and sampled scalar coincidences.** On the declared
**odd-modulus** domain, §9.7 proves the conductor-annotated product law
$A(m)=\prod_{p^e\|m}(e+\lceil e/2\rceil+1)$, with local factors
$3,4,6,7,9,\ldots$. It is distinct from the unannotated scalar spectral rank
$\rho$. The historical small examples reported $\rho(p)=3$,
$\rho(p^2)=4$, $\rho(p^3)=6$, $\rho(pq)=9$, $\rho(p^2q)=12$, and no
multiplicativity exceptions over that demo range. These are useful finite
type distinctions, not a globally invertible code for $\Omega$ or $\tau$.
The formula implementation `quadratic_residue_annotated_rank` itself factors
its argument; the separate graph-spectrum route constructs modular residues
without supplying those factors. The two provenance paths must not be
conflated when claiming a factorization-free diagnostic.

**Diagnostic limits.** The reported small examples distinguish certain types,
not a complete factorization: $\rho(15)=\rho(35)=9$, and $\rho=36$ is shared
by $p^3q^3$ and $p^2qr$. Thus scalar rank does not determine even the type
globally. The unannotated rank also aliases at high prime powers: §9.7 /
example 154 reports scalar $191$ versus annotated product $192$ for
$3^7\!\cdot5^2\!\cdot41^2$. The table is an organizing comparison, not a
complete ontology theorem. The underlying integer/modular carrier, selected
graph products and observation rules are supplied. No common obstruction
identifies these information losses with analytic $\arg\zeta$, and no
physical particles have been generated or identified by these diagnostics.

### 9.13 The Arithmetic Pulse — two declared linear responses

The same finite residue operator supports different **declared** models.
The pure-EPI heat response is
$h(t)=e_o^*e^{-\nu_f L t}e_o$. On pointed circulants its visible recurrence
order equals the number of distinct eigenvalues because every spectral
projector has weight $m_\lambda/n$. This exact scope is developed in
[TNFR_ARITHMETIC_DYNAMICS.md](TNFR_ARITHMETIC_DYNAMICS.md).

The separately chosen conservative graph wave uses $\omega_k=\sqrt{\lambda_k}$
only on its nonnegative self-adjoint domain. Its second-order law is not
derived from the first-order diffusion identity. A directed complex spectrum
does not supply a set of real conservative frequencies.

For quadratic residues at an odd prime $p\equiv1\pmod4$, the real Paley operator
has one zero eigenvalue and two distinct positive eigenvalues, each nonzero
one with multiplicity $(p-1)/2$. The wave consequently has a stationary mode
and two nonzero frequencies. The zero mode must be distinguished from an
oscillatory tone. The general prime cyclotomy count in §9.11 is a spectral
count even when no conservative-wave interpretation is admitted.

[emergent_arithmetic_pulse.py](../benchmarks/emergent_arithmetic_pulse.py)
records finite comparisons; its historical name does not derive an
autonomous arithmetic oscillator. Composite counts and collisions in §9.12
remain fixture-dependent diagnostics, not a complete factorization-type
decoder or a proof that primes minimize every possible spectrum complexity.

The graph, arithmetic carrier, initial excitation and chosen evolution are
supplied. These results do not derive their occurrence, sustained maintenance,
particle identity or an analytic zeta bridge.


---

## 10. Prime Path Graphs and the TNFR-Riemann Connection

### 10.1 The Discrete TNFR-Riemann Operator

**Historical, superseded construction.** Sections 10.1–10.4 retain the
eliminated prime-path prototype for traceability. They are not the current
program or evidence for an emergent critical line. The former family was

$$H^{(k)}_{\mathrm{TNFR}}(\sigma) = L_k + V_\sigma$$

where:
- $L_k$ is the graph Laplacian of the **prime path graph** $G_k$ (first $k$ primes $p_1, p_2, \ldots, p_k$ connected sequentially)
- $V_\sigma$ is a **structural potential** parametrized by $\sigma \in \mathbb{R}$:

$$V_\sigma(i) = (\sigma - \tfrac{1}{2}) \log(p_i)$$

### 10.2 Critical Parameter Convergence

The historical report described the lowest-eigenvalue sign change as

$$\sigma_c^{(k)} = \frac{1}{2} + O\!\left(\frac{1}{\log k}\right) \quad \text{as } k \to \infty$$

For the displayed construction with a symmetric positive-semidefinite graph
Laplacian $L_k\mathbf1=0$ and $p_i\ge2$, the sign change is already **exactly
at the inserted value $1/2$ for every finite size**. At $\sigma=1/2$ the
potential vanishes. Above it, adding the positive diagonal makes $H$ positive
definite. Below it, the Rayleigh quotient of $\mathbf1$ is negative. This
requires no prime-distribution theorem and works for any positive diagonal
in place of $\log p_i$. The old asymptotic wording therefore supplies no
independent critical-line evidence or generative phase-transition result.

### 10.3 Connection to the Riemann Hypothesis

At $\sigma=1/2$ this potential vanishes by definition; the transition does
not constrain zeros of analytic $\zeta$. The current
[Riemann program memo](TNFR_RIEMANN_RESEARCH_NOTES.md) supersedes this
prototype and its former bridge interpretation. G4 = RH remains open.

### 10.4 Tetrad Fields on the Prime Path

The historical prototype used the following eigenvector-variation diagnostics
from $(\lambda_j,\phi_j)$. Here $\phi_j$ is an eigenvector, not an independently
defined circular nodal phase; these expressions are not the canonical wrapped
tetrad kernels:

**Phase gradient** (discrete):
$$|\nabla\phi|^{(j)} = \frac{1}{k-1}\sum_{i=1}^{k-1}|\phi_j(p_{i+1}) - \phi_j(p_i)|$$

**Phase curvature** (discrete):
$$K_\phi^{(j)} = \frac{1}{k-2}\sum_{i=2}^{k-1}|\phi_j(p_{i+1}) - 2\phi_j(p_i) + \phi_j(p_{i-1})|$$

**Coherence length** (from correlation decay):
$$C_j(r) \approx A_j \cdot e^{-r/\xi_C^{(j)}}$$

These retained formulas record the old comparison only. They do not provide
a state map, an operator correspondence or a physical particle prediction.

### 10.5 Refactoring the Riemann Attack — From the Self-Adjoint Prime-Ladder to the Non-Self-Adjoint Phase Operator (MEASURED)

The prime-ladder P14 construction explicitly assigns prime labels and
$\nu_{f,(p,k)}=k\log p$. Its diagonal spectrum is an encoding of those
inputs. Selected finite symmetry tests establish the conditional implication
$[L,P]=0\Rightarrow[f(L),P]=0$. They do not establish that every canonical
operator commutes with prime relabeling, that a scalar spectrum is a vector
in a fixed sector, or that analytic $S(T)$ lies in its orthogonal complement.
The current Riemann memo explicitly withdraws those historical promotions.

The number-theory reframe (§9.6, §9.8) supplies a **structurally different object** for the same residue: the **directed quadratic-residue diffusion operator** $L_{rw}=I-D^{-1}W$ on the Paley tournament ($n\equiv 3\pmod 4$). It is

- **non-self-adjoint but normal**, since it is circulant; its complex
  eigenvalues contain classical Gauss-sum information;
- associated with a separately specified modular graph and symmetry action.
  A symmetry restriction proved for P14 cannot simply be transferred to it.

Complex eigenvalues alone do not identify zeta ordinates, nodal phase
evolution or a physical oscillatory entity.

So the natural question is whether the attack should pivot from "build a *self-adjoint* operator with spectrum $\{\gamma_n\}$" to "read the residue off the *non-self-adjoint* phase operator".

**The pre-registered falsifier (MEASURED).** `benchmarks/residue_phase_vs_riemann.py` tests it on primes $p\equiv 3\pmod 4$:

- **F-GAUSS** — $\max|\mathrm{Im}(\lambda)|(p)=\sqrt p/(p-1)$ **exactly** (ratio $1.000000$, 15/15 primes): the phase content is the **Paley Gauss-sum eigenvalue**, a classical fact.
- **F-ALIGN** — $\mathrm{Pearson}\big(\max|\mathrm{Im}|(p_n),\,\gamma_n\big)=\mathbf{-0.9068}$: the residue phase content **decreases** like $1/\sqrt p$ while the zeros $\gamma_n$ **increase** — opposite trends.
- **Verdict:** `GAUSS_CONFIRMED_RIEMANN_REFUTED`.

**Honest net.** The retained result rejects the tested direct alignment of
Gauss-sum imaginary parts with zeta ordinates. It does not locate a universal
symmetry obstruction, exclude other representations or prove a physical
phase mechanism. G4 = RH remains open.

The current finite pulse helper instead evaluates
$P_N(T)=\sum_{n=1}^N n^{-1/2}e^{-iT\log n}$ with prescribed amplitudes,
logarithmic frequencies and truncation. This is not an identity for analytic
$\zeta(1/2+iT)$: the ordinary infinite Dirichlet representation applies to
$\mathrm{Re}(s)>1$, with analytic continuation elsewhere
([DLMF §25.2](https://dlmf.nist.gov/25.2)). The functional-equation reflection
axis ([DLMF §25.4](https://dlmf.nist.gov/25.4)) is not a derived
$\Delta\mathrm{NFR}=0$ locus. Finite nearest-dip matches are comparisons
against known ordinates, not autonomous nodal generation or RH certificates.

---

## 11. Worked Examples

### 11.1 Prime Detection: $n = 17$

$$\Omega(17) = 1, \quad \tau(17) = 2, \quad \sigma(17) = 18$$

$$\Delta\mathrm{NFR}(17) = 1 \times (1-1) + 1 \times (2-2) + 1 \times \left(\frac{18}{17} - \frac{18}{17}\right) = 0$$

Assigned coordinates and read-out: $\mathrm{EPI}(17) \approx 2.75$, $\nu_f(17) \approx 1.47$, $C_{\text{local}} = 1.0$.

**Interpretation**: Zero arithmetic pressure and unit pressure-only coherence;
the held-pressure unforced scalar model has zero EPI rate at this label.

### 11.2 Semiprime: $n = 15 = 3 \times 5$

$$\Omega(15) = 2, \quad \tau(15) = 4, \quad \sigma(15) = 24$$

| Component | Calculation | Value |
|-----------|------------|-------|
| Factorization | $1 \times (2-1)$ | 1 |
| Divisor | $1 \times (4-2)$ | 2 |
| Abundance | $1 \times (24/15 - 16/15)$ | 0.533 |
| **Total** | | **3.533** |

$C_{\text{local}} = 1/(1+3.533) \approx 0.221$.

### 11.3 Prime Power: $n = 8 = 2^3$

$$\Omega(8) = 3, \quad \tau(8) = 4, \quad \sigma(8) = 15$$

| Component | Calculation | Value |
|-----------|------------|-------|
| Factorization | $1 \times (3-1)$ | 2 |
| Divisor | $1 \times (4-2)$ | 2 |
| Abundance | $1 \times (15/8 - 9/8)$ | 0.750 |
| **Total** | | **4.750** |

Using $\Omega$ (with multiplicity) rather than $\omega$ (distinct primes) gives prime powers a strong pressure signal: $2^3$ registers $\Omega = 3$, not $\omega = 1$.

### 11.4 Squarefree Composite: $n = 30 = 2 \times 3 \times 5$

$$\Omega(30) = 3, \quad \tau(30) = 8, \quad \sigma(30) = 72$$

| Component | Calculation | Value |
|-----------|------------|-------|
| Factorization | $1 \times (3-1)$ | 2 |
| Divisor | $1 \times (8-2)$ | 6 |
| Abundance | $1 \times (72/30 - 31/30)$ | 1.367 |
| **Total** | | **9.367** |

Structural triad: $\mathrm{EPI}(30) \approx 7.48$, $\nu_f(30) \approx 2.15$, $C_{\text{local}} \approx 0.097$.

---

## 12. Implementation Map

### 12.1 Source Modules

| Module | Path | Scope |
|--------|------|-------|
| **Arithmetic network** | `src/tnfr/mathematics/number_theory.py` | `ArithmeticTNFRNetwork`, `ArithmeticTNFRFormalism`, `PrimeCertificate` |
| **Primality testing** | `primality-test/tnfr_primality/core.py` | Standalone ΔNFR computation, validation |
| **Canonical constants** | `primality-test/tnfr_primality/constants.py` | Arithmetic pressure coefficients (separate subproject) |
| **Advanced integration** | `primality-test/tnfr_primality/advanced_core.py` | Full repo infrastructure bridge |
| **Optimized batch** | `primality-test/tnfr_primality/optimized.py` | Caching, benchmarking, batch processing |
| **Spectral factorization** | `factorization-lab/tnfr_factorization/spectral_paley.py` | Paley-Jacobi spectral decoder |
| **Factorization API** | `factorization-lab/tnfr_factorization/api.py` | High-level `factorize()` function |
| **Nodal-pulse foundation** | `src/tnfr/riemann/nodal_pulse.py` | Emergent prime-NFR nodal pulse ($\nu_f = \log n$; zeros as destructive interference) |
| **Prime-ladder Hamiltonian** | `src/tnfr/riemann/prime_ladder_hamiltonian.py` | Canonical $\nu_f$ prime-ladder (P14) |
| **Canonical constants (repo)** | `src/tnfr/constants/canonical.py` | Repository-wide canonical constant definitions |

### 12.2 Executable Demonstrations

| Example | Concept |
|---------|---------|
| [41_von_mangoldt_zeta_demo.py](../examples/03_riemann_zeta/41_von_mangoldt_zeta_demo.py) | Prime-ladder von Mangoldt series (P12) |
| [42_riemann_zeros_as_resonances.py](../examples/03_riemann_zeta/42_riemann_zeros_as_resonances.py) | Riemann zeros as resonance poles (P13) |
| [43_prime_ladder_hamiltonian_demo.py](../examples/03_riemann_zeta/43_prime_ladder_hamiltonian_demo.py) | Canonical νf prime-ladder Hamiltonian (P14) |
| [31_mathematical_constants_basis.py](../examples/02_physics_regimes/31_mathematical_constants_basis.py) | The structural scale π and the mathematical-constant basis |
| [40_arithmetic_number_theory.py](../examples/07_number_theory/40_arithmetic_number_theory.py) | Primality, triad, component analysis |
| [94_generative_number_construction.py](../examples/07_number_theory/94_generative_number_construction.py) | Compositional generation from prime atoms; U5 fractality; grammar certification |
| [95_primes_from_spectral_waves.py](../examples/07_number_theory/95_primes_from_spectral_waves.py) | Prime staircase ψ(x) as spectral-wave superposition; spectral coherence ⟺ RH (honest scope) |
| [96_spectral_vibration_of_coherence.py](../examples/07_number_theory/96_spectral_vibration_of_coherence.py) | Oscillatory residue S(T) as prime-ladder vibration {k·log p}; why aggregate C(t) is blind (honest scope) |
| [97_goldbach_additive_multiplicative.py](../examples/07_number_theory/97_goldbach_additive_multiplicative.py) | Goldbach phase-matching: negative structural result; additive/multiplicative orthogonality; B2/B3 ontological note |
| [100_prime_families_orbits.py](../examples/07_number_theory/100_prime_families_orbits.py) | Special prime families (twin, cousin, sexy, Sophie Germain, safe, Cunningham, Mersenne, constellations) as orbits and level-sets of arithmetic maps on the zero-pressure fixed-point set $Z=\{\Delta\mathrm{NFR}=0\}$; three generator classes; detection exact, infinitude open (honest scope) |
| [101_numbers_as_coupled_network.py](../examples/07_number_theory/101_numbers_as_coupled_network.py) | Numbers as a coupled TNFR network: $\Omega(n)$ grades both the per-node pressure $\Delta\mathrm{NFR}$ and the divisibility/GCD transport centrality ($r\approx 0.8$–$0.9$); primes ($\Omega{=}1$, $\Delta\mathrm{NFR}{=}0$) are the transport periphery, large primes isolated; correspondence-through-$\Omega$ not identity, not scale-free (honest scope) |
| [102_nodal_flow_primes_equilibria.py](../examples/07_number_theory/102_nodal_flow_primes_equilibria.py) | The actual nodal flow $\partial\mathrm{EPI}/\partial t=\nu_f\Delta\mathrm{NFR}$ on numbers: primes are EXACTLY the equilibria (§4 theorem in motion, frozen) while composites drift $\Omega$-graded; refines §7.1 — primes are static low-$\Phi_s$ sinks but NOT dynamical attractors (diffusion flow pulls primes UP toward the composite bulk) |
| [146_primality_grammatical_inertness.py](../examples/07_number_theory/146_primality_grammatical_inertness.py) | Tests the restricted scalar flow $\mathrm{EPI}_{k+1}=\mathrm{EPI}_k+dt\,\nu_f\Delta\mathrm{NFR}_{\rm arith}$ with arithmetic pressure held fixed. It verifies prime $\iff\Delta\mathrm{NFR}_{\rm arith}=0\iff C=1$, zero prime drift for sampled positive capacities, and exact capacity scaling of composite drift. Because no canonical graph operator or grammar history is applied, it does not prove grammatical inertness. Fixed positive composite pressure also gives an unbounded infinite-time integral, so the fixture is not a U2-convergence result or attraction toward primality. |
| [147_numbers_as_free_monoid_words.py](../examples/07_number_theory/147_numbers_as_free_monoid_words.py) | Compares the classical free commutative monoid on the primes with the unit-coefficient arithmetic pressure. The exact content is the additive law for $\Omega$, the coprime multiplicative laws for $\tau$ and $\sigma$, and $\Omega(n)=1\iff n$ prime for $n\ge2$. Count $\Omega$ and log-size give a declared analogy with pressure and capacity; integers are not operator-grammar words, and prime multiplication is not a canonical destabilizer. |
| [148_capacity_arm_carries_von_mangoldt.py](../examples/07_number_theory/148_capacity_arm_carries_von_mangoldt.py) | Uses the classical identities `log n = sum_(d divides n) Lambda(d)` and Mobius inversion. Assigned logarithmic capacity is an arithmetic coordinate. The continued `-zeta_prime/zeta` has residue minus the multiplicity at a zeta zero; no global pressure-versus-capacity observability dichotomy or analytic symmetry-complement result follows. |
| [149_p14_is_the_capacity_arm_operator.py](../examples/07_number_theory/149_p14_is_the_capacity_arm_operator.py) | Reads back supplied `nu_f=k*log(p)` entries in the decoupled P14 matrix. Its weighted finite trace agrees with the matching finite von Mangoldt sum; convergence to the classical infinite identity has domain `Re(s)>1`. This is a declared arithmetic construction, not a derivation of prime labels, REMESH occurrence or a zero-spectrum Hamiltonian. |
| [153_structural_frequency_rank_cyclotomy.py](../examples/07_number_theory/153_structural_frequency_rank_cyclotomy.py) | Applies the structural-diffusion spectrum to power-residue Cayley graphs. The exact cyclotomy law $s_k(p)=\gcd(k,p-1)+1$ is proved for the declared odd-prime construction from classical Gauss-period theory and checked on a finite grid. Other correlations and squarefree grading observations in the example remain finite measurements; no restoring primality dynamics or open-problem result follows. |

### 12.3 Test Coverage

| Test area | Location |
|-----------|----------|
| Primality validation (10k range) | `primality-test/test_installation.py` |
| Arithmetic network construction | `tests/` (number_theory tests) |
| Factorization spectral decoder | `factorization-lab/tests/test_spectral_paley.py` |
| Factorization verification | `factorization-lab/tests/test_verification_robustness.py` |
| Riemann operator spectral | `tests/` (riemann tests) |

---

## 13. Reference questions and research boundaries

These questions are a secondary inventory, not an active task queue. The
[FIVE_STAGE_EXECUTION_PLAN.md](research/FIVE_STAGE_EXECUTION_PLAN.md) owns priorities;
arithmetic reuse does not replace the current joint nodal emergence objective.

### 13.1 Computational

- **Sub-$O(\sqrt{n})$ primality**: Can spectral methods on arithmetic networks detect primes faster than trial division?
- **Sieve optimization**: Can the TNFR pressure landscape guide more efficient sieve algorithms?
- **Large-number factorization**: Scaling the spectral Paley-Jacobi method to numbers beyond current computational limits.

### 13.2 Theoretical

- **TNFR-Riemann bridge**: Can a declared nodal construction support an
  independently proved analytic correspondence? The historical $H^{(k)}$
  family is superseded (§10); no G4/RH bridge is established by its sign change.
- **Pressure distribution**: What is the exact probability distribution of $\Delta\mathrm{NFR}(n)$ for "random" composites?
- **Goldbach connection**: Can the additive decomposition of even numbers be formulated as a phase-matching problem ($|\phi_p + \phi_q - \phi_{2n}| \leq \Delta\phi_{\max}$)?
- **Arithmetic coherence length**: How does $\xi_C$ in the arithmetic network relate to the distribution of prime gaps?

### 13.3 Structural

- **Special prime families** (PARTIALLY ADDRESSED — [100_prime_families_orbits.py](../examples/07_number_theory/100_prime_families_orbits.py)): twin, cousin, sexy, Sophie Germain, safe, Cunningham, Mersenne, and constellation families are organized as **structured subsets of the zero-pressure fixed-point set** $Z=\{n\ge 2:\Delta\mathrm{NFR}(n)=0\}$ (the primes), carved out by three classes of arithmetic map: additive-gap level-sets ($S_g(p)=p+g$), affine-recurrence orbits ($T(p)=2p+1$: Sophie Germain, safe, Cunningham chains), and exponential-form images ($M(p)=2^p-1$: Mersenne). Detection/generation is exact via the verified $\Delta\mathrm{NFR}=0$ theorem; **infinitude conjectures** (twin-prime, Sophie Germain, Mersenne) remain OPEN — the same honest stance as Goldbach (§13.2). The witness pressure signatures (e.g. the twin witness $p+1$ divisible by 6) are faithful TNFR restatements of classical divisibility facts.
- **Arithmetic network as a coupled system** (MEASURED — [101_numbers_as_coupled_network.py](../examples/07_number_theory/101_numbers_as_coupled_network.py)): on the divisibility/GCD network the prime-factor count $\Omega(n)$ is a **common structural coordinate** that grades both the per-node arithmetic pressure $\Delta\mathrm{NFR}$ ($r(\Omega,\Delta\mathrm{NFR})\approx 0.94$) and the network-transport centrality ($r(\Omega,\deg)\approx 0.75$), so the two pictures are linked ($r(\Delta\mathrm{NFR},\deg)\approx 0.81$). Primes ($\Omega{=}1$, $\Delta\mathrm{NFR}{=}0$) form the **transport periphery** (≈ 0.18× the composite stationary mass, ≈ 2.4× effective resistance, and large primes $p>N/2$ are literally isolated). Honest scope: a **correspondence through $\Omega$, not a dynamical identity** — the per-node $\Delta\mathrm{NFR}$ is not the graph-diffusion Laplacian; the network is **not scale-free**; "primes peripheral" restates the classical $\gcd(p,m)>1\iff p\mid m$ in transport language.
- **Held arithmetic-pressure flow** ([example 102](../examples/07_number_theory/102_nodal_flow_primes_equilibria.py)):
  positive fixed capacity and the assigned arithmetic pressure give zero EPI
  rate at primes and constant positive drift at composites. There is no
  restoring response because that pressure does not depend on evolving EPI.
  A separate fixed-graph diffusion comparison relaxes toward its weighted
  mean under its own hypotheses. Neither model proves that primes are
  universal potential minima, attractors, or stable full-state NFRs.
- **Higher-order pressure**: Are there fourth or fifth pressure components (beyond $\Omega$, $\tau$, $\sigma$) that provide additional structural information?
- **Algebraic number fields**: Extension of the arithmetic triad to Gaussian integers, Eisenstein integers, or general number fields.
- **p-adic structure**: Connection between the arithmetic tetrad and p-adic analysis.

---

## 14. References

### Internal

- [AGENTS.md](../AGENTS.md) — Primary theoretical reference (TNFR framework)
- [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) — Nodal equation and structural field tetrad
- [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) — U1-U6 grammar derivations
- [STRUCTURAL_OPERATORS.md](STRUCTURAL_OPERATORS.md) — 13 canonical operators with tetrad synergies
- [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) — Conservation laws
- [APPLIED_STRUCTURAL_ANALYSIS.md](APPLIED_STRUCTURAL_ANALYSIS.md) — Spectral factorization verification
- [TNFR_RIEMANN_RESEARCH_NOTES.md](TNFR_RIEMANN_RESEARCH_NOTES.md) — TNFR-Riemann program (18 sections + 11 appendices)
- [MATHEMATICAL_DYNAMICS_BASIS.md](MATHEMATICAL_DYNAMICS_BASIS.md) — The structural-field tetrad; the one structural scale (π)
- [GLOSSARY.md](GLOSSARY.md) — Operational definitions

### External

- Hardy, G.H. & Wright, E.M. — *An Introduction to the Theory of Numbers* (arithmetic functions)
- Erdős, P. & Kac, M. — "The Gaussian Law of Errors in the Theory of Additive Number Theoretic Functions" (1940)
- Kuramoto, Y. — *Chemical Oscillations, Waves, and Turbulence* (phase synchronization)

---

**Version**: 0.0.3.5 | **Status**: Canonical | **Authority**: [AGENTS.md](../AGENTS.md)
