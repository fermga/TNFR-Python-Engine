# TNFR Number Theory: Arithmetic Emergence from Structural Dynamics

**Status**: Canonical theoretical reference
**Version**: 0.0.3.3
**Date**: March 2026

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
13. [Open Questions and Research Directions](#13-open-questions-and-research-directions)
14. [References](#14-references)

---

## 1. Introduction

Number theory can be formulated within the TNFR framework when the nodal equation

$$\frac{\partial\mathrm{EPI}}{\partial t} = \nu_f \cdot \Delta\mathrm{NFR}(t)$$

is applied to a network whose nodes are natural numbers and whose edges encode arithmetic relationships (divisibility, common factors). In this setting:

- **Primes are zero-pressure fixed points**: $\Delta\mathrm{NFR}(p) = 0$ for all primes $p$.
- **Composites carry structural pressure**: $\Delta\mathrm{NFR}(n) > 0$ whenever $n$ is composite, with magnitude proportional to factorization complexity.
- **Factorization as spectral decoding**: discovering the factors of a composite can be framed as resolving the coherent sub-modes of its structural pressure field.

This document formalizes these observations, expresses the arithmetic constants as canonical units (only $\pi$ is a genuine structural scale), and maps the theory to its implementations in the repository.

### Scope

| Layer | Description | Source |
|-------|-------------|--------|
| **Primality** | Deterministic prime detection via $\Delta\mathrm{NFR}=0$ | `primality-test/`, `src/tnfr/mathematics/number_theory.py` |
| **Factorization** | Spectral factor discovery via Paley-Jacobi graphs | `factorization-lab/` |
| **Riemann program** | Prime path spectral operators and critical parameter convergence | `src/tnfr/riemann/` |

All three layers share the same canonical constants, structural fields, and grammar constraints (U1-U6).

---

## 2. Arithmetic Networks as TNFR Systems

### 2.1 Network Construction

A TNFR arithmetic network $G = (V, E)$ is a directed graph where:

- **Nodes** $V = \{2, 3, \ldots, N\}$ are natural numbers.
- **Edges** encode two types of structural relationship:
  - **Divisibility edges**: $(d, n)$ for each divisor $d \mid n$ with $d < n$.
  - **GCD coupling edges**: $(a, b)$ when $\gcd(a, b) > 1$, weighted by $\gcd(a, b) / \max(a, b)$.

Each node $n$ is assigned the structural triad (EPI, $\nu_f$, $\Delta\mathrm{NFR}$) and a phase $\phi_n$ derived from its arithmetic properties.

### 2.2 Sieve-Based Computation

Efficient computation uses a Lowest Prime Factor (LPF) sieve:

$$\text{lpf}[n] = \min\{p \text{ prime} : p \mid n\}$$

From the LPF array, factorization of any $n \leq N$ is $O(\log n)$, enabling computation of all arithmetic functions ($\Omega$, $\tau$, $\sigma$) for the entire network in $O(N \log \log N)$ sieve time plus $O(N \log N)$ factorization time.

### 2.3 Phase Assignment

Each node receives a phase derived from its position in the arithmetic structure:

$$\phi_n = 2\pi \cdot \frac{n}{N} \pmod{2\pi}$$

Phase compatibility ($|\phi_i - \phi_j| \leq \Delta\phi_{\max}$) governs coupling operations (U3), ensuring that arithmetic relationships respect the resonant coupling constraint.

---

## 3. The Arithmetic Structural Triad

The structural triad specializes the general TNFR triad (EPI, $\nu_f$, $\Delta\mathrm{NFR}$) to arithmetic:

### 3.1 Form: EPI(n)

The Primary Information Structure of a natural number measures its overall arithmetic complexity:

$$\mathrm{EPI}(n) = 1 + \alpha \cdot \Omega(n) + \beta \cdot \ln(\tau(n)) + \gamma_{\mathrm{epi}} \cdot \left(\frac{\sigma(n)}{n} - 1\right)$$

where:
- $\alpha = 1$ — factorization complexity weight (canonical unit, §5)
- $\beta = 1$ — divisor complexity weight (canonical unit, §5)
- $\gamma_{\mathrm{epi}} = 1$ — abundance deviation weight (canonical unit, §5)

**Physical interpretation**: EPI(n) is the structural form of the number, analogous to the configuration of an oscillator. Primes have the simplest forms; highly composite numbers have the richest.

### 3.2 Frequency: $\nu_f(n)$

The reorganization capacity of a number measures how rapidly its structural form could evolve:

$$\nu_f(n) = \nu_0 \cdot \left(1 + \delta \cdot \frac{\tau(n)}{n} + \varepsilon \cdot \frac{\Omega(n)}{\ln(n)}\right)$$

where:
- $\nu_0 = 1$ — base frequency (canonical unit, §5)
- $\delta = 1$ — divisor density modulation (canonical unit, §5)
- $\varepsilon = 1$ — factorization complexity modulation (canonical unit, §5)

**Physical interpretation**: $\nu_f$ is the capacity lever in the nodal equation. Numbers with rich divisor structures have slightly higher reorganization capacity, but this is irrelevant for primes because the pressure lever vanishes.

### 3.3 Pressure: $\Delta\mathrm{NFR}(n)$

The structural pressure equation is the central result of arithmetic TNFR:

$$\boxed{\Delta\mathrm{NFR}(n) = \zeta \cdot (\Omega(n) - 1) + \eta \cdot (\tau(n) - 2) + \theta \cdot \left(\frac{\sigma(n)}{n} - \left(1 + \frac{1}{n}\right)\right)}$$

where $\Omega(n)$ is the prime factor count with multiplicity, $\tau(n)$ the divisor count, and $\sigma(n)$ the divisor sum. The coefficients are canonically $\zeta = \eta = \theta = 1$ (unit weights; see §5).

**Physical interpretation**: $\Delta\mathrm{NFR}(n)$ is the pressure lever — how much reorganization the arithmetic structure of $n$ demands. It quantifies the structural distance from primality.

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

**Structural interpretation**: The theorem states that primes are the unique **zero-pressure fixed points** of the arithmetic structural manifold. Under the nodal equation, $\partial\mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR} = 0$ at primes, regardless of $\nu_f$. Primes are structurally inert — they require no reorganization.

### 4.2 Coefficient Independence

The primality criterion $\Delta\mathrm{NFR}(n) = 0$ is **independent of the coefficient values** $(\zeta, \eta, \theta)$, provided all three are positive. Each component vanishes individually for primes. The coefficients affect only the relative weighting of pressure components for composites — the landscape of the structural manifold, not its fixed points.

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

By the Coefficient Independence theorem (§4.2) the primality criterion $\Delta\mathrm{NFR}(n) = 0$ holds for **any** positive coefficients. The arithmetic pressures carry no phase/geometric content, so even $\pi$ has no role; the canonical choice therefore introduces **no constant at all** — all weights are **unity**, and the structural content lives entirely in the arithmetic invariants $(\Omega, \tau, \sigma, n)$.

### 5.2 Pressure Coefficients

$$\boxed{\zeta = \eta = \theta = 1}$$

The three pressure channels weigh equally: the factorization excess $\Omega - 1$, the divisor excess $\tau - 2$ and the abundance excess $\sigma/n - (1 + 1/n)$ each contribute on the same unit scale. This is the canonical, parameter-free form — every coefficient is forced by the §4.2 coefficient-independence theorem rather than fitted.

### 5.3 EPI Parameters

| Parameter | Value | Physical meaning |
|-----------|-------|-----------------|
| $\alpha$ | $1$ | Factorization complexity weight |
| $\beta$ | $1$ | Divisor logarithmic weight |
| $\gamma_{\mathrm{epi}}$ | $1$ | Abundance deviation weight |

### 5.4 Frequency Parameters

| Parameter | Value | Physical meaning |
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

The 9 dynamical arithmetic parameters (3 pressure + 3 EPI + 3 frequency) are positive operational weights applied to arithmetic functions (canonical units; the prime ⟺ ΔNFR = 0 criterion is coefficient-independent, §4.2). The structural-field thresholds are the **same canonical π-derived bounds as any TNFR network** — only π is a genuine structural scale (per-node $|\Phi_s| < \pi/4$, drift $\Delta\Phi_s < \pi/2$; see §7.5 and [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) §4). An earlier φ/γ/e "arithmetic recalibration" was removed (audit 2026); no domain-specific tuning remains.

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

The three pressure channels are algebraically independent — no linear combination of two can reproduce the third for all $n$. This makes the decomposition **minimal and complete** for characterizing compositeness through the three canonical arithmetic functions ($\Omega$, $\tau$, $\sigma$).

### 6.3 Structural Pressure Landscape

As $n$ grows, the expected pressure for a "random" composite scales as:

$$\mathbb{E}[\Delta\mathrm{NFR}(n)] \sim \ln\ln n + (\ln n)^{\ln 2} + \text{(abundance deviation)}$$

by the Erdős-Kac theorem ($\Omega(n) \sim \ln\ln n$) and divisor function asymptotics. Primes remain at exactly zero regardless of magnitude.

---

## 7. The Arithmetic Tetrad

When the arithmetic network $G$ is constructed, the structural field tetrad ($\Phi_s$, $|\nabla\phi|$, $K_\phi$, $\xi_C$) can be computed using the centralized physics modules:

### 7.1 Structural Potential: $\Phi_s$

$$\Phi_s(n) = \sum_{m \neq n} \frac{\Delta\mathrm{NFR}(m)}{d(n, m)^2}$$

where $d(n, m)$ is the graph distance in the arithmetic network. Primes, being zero-pressure nodes, act as **sinks** in the potential field — they attract nearby composites toward equilibrium.

**Threshold**: $|\Phi_s| < \pi/4 \approx 0.785$ (π-derived, quarter phase-wrap — see [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) §4); the arithmetic network uses this same canonical π-derived bound (§7.5).

### 7.2 Phase Gradient: $|\nabla\phi|$

$$|\nabla\phi|(n) = \frac{1}{|\mathcal{N}(n)|} \sum_{m \in \mathcal{N}(n)} |\phi_n - \phi_m|$$

where $\mathcal{N}(n)$ are the neighbors of $n$ in the arithmetic network. High phase gradient indicates local desynchronization — composites with many diverse factors show elevated gradients.

**Threshold**: $|\nabla\phi| \lesssim \pi/16 \approx 0.196$ (heuristic early-warning, kinematic bound $\pi$) for stable operation; arithmetic recalibration gives $0.2591$.

### 7.3 Phase Curvature: $K_\phi$

$$K_\phi(n) = \text{wrap\_angle}\!\left(\phi_n - \overline{\phi}_{\mathcal{N}(n)}\right)$$

where $\overline{\phi}_{\mathcal{N}(n)}$ is the circular mean of neighbor phases. Elevated curvature flags numbers at structural boundaries — e.g., the transition between prime-rich and composite-rich regions.

**Threshold**: $|K_\phi| < 0.9\pi \approx 2.827$; arithmetic recalibration gives $3.2275$.

### 7.4 Coherence Length: $\xi_C$

$$C(r) \approx A \cdot e^{-r/\xi_C}$$

The coherence length measures how far structural correlations propagate through the arithmetic network. Near critical points (e.g., twin primes, prime gaps), $\xi_C$ diverges — a signature of long-range correlation in the prime distribution.

### 7.5 Tetrad thresholds on the arithmetic network

The arithmetic network uses the same canonical, π-derived structural-field
tetrad thresholds as any TNFR network — only π is a genuine structural scale:

| Field | Threshold | Source |
|-------|-----------|--------|
| $\Phi_s$ | π/4 ≈ 0.785 (per-node), π/2 ≈ 1.571 (drift) | `PHI_S_VON_KOCH_THRESHOLD`, `U6_STRUCTURAL_POTENTIAL_LIMIT` |
| $|\nabla\phi|$ | ≤ π (phase wrap); π/16 ≈ 0.196 heuristic early-warning | `GRAD_PHI_CANONICAL_THRESHOLD` |
| $K_\phi$ | < 0.9·π ≈ 2.827 (phase-wrap safety) | `K_PHI_CANONICAL_THRESHOLD` |
| $\xi_C$ | spectral gap (ξ_C ∝ 1/√λ₂) | Computed per network |

An earlier "arithmetic recalibration" introduced topology-specific thresholds
expressed as φ/γ/e combinations (e.g. a $K_\phi$ threshold of 3.2275 that
*exceeded* the π phase-wrap bound and was therefore unreachable). Those values
were not structural scales and have been removed (audit 2026): the arithmetic
network is governed by the same π-bounded phase sector as every other TNFR
network.

### 7.6 The Arithmetic NFR and its Emergent Geometry

The arithmetic network is itself a **Fractal-Resonant Node** (NFR; TNFR.pdf
§1.4.1) — a region of structural coherence coupled by divisibility/GCD.
`ArithmeticTNFRNetwork.nfr()` surfaces the joint read-out of its three emergent
facets:

- **Resonant.** By the §4.1 primality theorem the equilibrium set
  $\{n : \Delta\mathrm{NFR}(n) = 0\}$ is *exactly* the primes, so the resonant-
  coherence attractors of the arithmetic NFR are the prime numbers;
  `equilibrium_fraction` is the prime density and the mean per-node coherence
  $C = 1/(1+|\Delta\mathrm{NFR}|)$ measures distance from this attractor.
- **Geometric.** The nodal topology (radial / annular / multinodal), read by
  `classify_nodal_topology` from the structural-potential geometry, is
  **multinodal** — its centers are the highly-composite / abundant numbers
  (6, 12, 24, 30, 36, …), the hubs of the divisibility lattice.
- **Fractal.** The coherence length $\xi_C$ sets the region scale.

The same nodal dynamics generates an **emergent geometry** (AGENTS.md §4),
exposed by `conservation()` and `symplectic_substrate()`, which delegate to the
canonical Structural Conservation Theorem and symplectic-substrate machinery
applied to the divisibility network:

- a conserved **Noether charge** $Q = \sum_i (\Phi_s(i) + K_\phi(i))$ and the
  structural **energy functional**
  $E = \tfrac12 \sum_i (\Phi_s^2 + |\nabla\phi|^2 + K_\phi^2 + J_\phi^2 +
  J_{\Delta\mathrm{NFR}}^2)$, with the potential sector $\Phi_s$ sourced by the
  arithmetic $\Delta\mathrm{NFR}$ (the genuine invariants $\Omega, \tau, \sigma$);
- a valid **symplectic substrate** of dimension $4N$ with conjugate pairs
  $(K_\phi, J_\phi)$ and $(\Phi_s, J_{\Delta\mathrm{NFR}})$. The geometric sector
  is populated by the *size / capacity* phase $\phi(n) \propto \log n$ (the monoid
  homomorphism $(\mathbb{N},\times) \to (\mathbb{R},+)$), which is non-degenerate
  on the dense divisibility graph.

The emergent geometry is thus **potential-dominated**: the arithmetic structure
(factorization pressure) drives the structural-potential geometry, while phase is
the secondary size grading.

---

## 8. Dual-Lever Decomposition

### 8.1 The Nodal Equation in Arithmetic

Applying the nodal equation to the arithmetic network:

$$\frac{\partial\mathrm{EPI}(n)}{\partial t} = \nu_f(n) \cdot \Delta\mathrm{NFR}(n)$$

This decomposes structural evolution into two independent levers:

- **Capacity lever** ($\nu_f$): How fast the number *can* reorganize. Depends on divisor structure and factorization complexity. Modulated by operators UM, SHA, VAL, NUL.
- **Pressure lever** ($\Delta\mathrm{NFR}$): How much reorganization is *demanded*. Zero for primes, positive for composites. Modulated by operators IL, OZ, THOL, ZHIR, NAV.

### 8.2 Fixed Point Analysis

For primes: $\Delta\mathrm{NFR}(p) = 0 \Rightarrow \partial\mathrm{EPI}/\partial t = 0$ regardless of $\nu_f(p)$.

This is a **structurally stable fixed point**: perturbations to $\nu_f$ do not affect the equilibrium. The prime's structural form is frozen by the absence of pressure, not by the absence of capacity.

For composites: $\Delta\mathrm{NFR}(n) > 0 \Rightarrow \partial\mathrm{EPI}/\partial t > 0$.

The composite's structure is under active reorganization pressure. The rate depends on $\nu_f$, but the direction (toward simpler structure) is determined by the positive pressure.

### 8.3 Experimental Confirmation

Operator-tetrad synergy experiments (examples 37-39) confirmed:

1. $\Phi_s$ responds **linearly** to $\Delta\mathrm{NFR}$ perturbations with $|r| = 1.000$ (perfect correlation), confirming the pressure lever's direct coupling to the structural potential field.
2. The **complete causal chain** is: Operator $\to$ ($\nu_f$, $\Delta\mathrm{NFR}$) $\to$ $\partial\mathrm{EPI}/\partial t$ $\to$ Tetrad $\to$ ($\mathcal{E}$, $\mathcal{Q}$).
3. Grammar-compliant operator sequences maintain Lyapunov descent ($dE/dt \leq 0$) even when the contractivity ratio $\Pi > 1$.

---

## 9. Factorization as Spectral Decoding

### 9.1 The Factorization Problem in TNFR Terms

Given a composite $n$ with $\Delta\mathrm{NFR}(n) > 0$, factorization is the process of decomposing the structural pressure into coherent sub-modes, each corresponding to a prime factor.

**Physical analogy**: A composite number is like a coupled oscillator system with multiple resonant frequencies. Factorization identifies the individual frequencies (prime factors) from the combined signal.

### 9.2 Spectral Paley-Jacobi Method

The implementation uses Paley graphs — algebraic constructions from quadratic residues:

1. **Graph construction**: For modulus $m$ (chosen near $n$), build the Paley graph $G(m)$ where nodes are $\{0, \ldots, m-1\}$ and edges connect quadratic residues.

2. **Spectral decomposition**: Compute the spectrum of the **emergent structural-diffusion operator** $L_{rw} = I - D^{-1}W$ (the canonical ΔNFR EPI channel; `_laplacian_eigenvalues` routes through `structural_diffusion_operator`). On the residue/Paley graph, which is **regular**, $L_{rw}$ shares eigenvectors with the classical Laplacian and the eigenvalues differ only by the degree ($\lambda_{\text{classical}}=d\cdot\lambda_{rw}$), so the Fiedler-gap → prime-size map (a Paley Gauss-sum fact) is preserved while the operator provenance is the emergent TNFR transport operator.

3. **Tetrad proxies** (HONEST SCOPE): the factorizer operates on the spectrum, not on a node-level ΔNFR field, so it uses **scalar proxies** of the tetrad — $\Phi_s\approx$ normalized edge density, $\xi_C\approx 1/(\nu_f\lambda_2)$ (the emergent diffusion relaxation time). These are labelled proxies in code (`_structural_potential`, `_coherence_length`); the genuine per-node tetrad (`tnfr.physics.canonical`) is measured by example 117 and is **blind to the factor cosets** (§9.5) — the factor signal lives in the spectrum, which the proxies summarize.

4. **Operator sequence**: Apply the canonical decoder $[\mathrm{UM}, \mathrm{RA}, \mathrm{IL}, \mathrm{THOL}]$ per partition:
   - **UM** (Coupling): Phase-gated coupling between quadratic residues (U3 verified)
   - **RA** (Resonance): Amplify coherent periodicity patterns
   - **IL** (Coherence): Stabilize the partitioned structure
   - **THOL** (Self-organization): Preserve multi-scale identity (U5)

5. **Factor inference**: Detect periodicities in the stabilized partitions that correspond to $n/p$ for candidate factors $p$.

6. **TNFR certification**: Verify each candidate against 8 structural criteria (§9.3).

### 9.3 Structural Verification Criteria

A factor candidate is TNFR-certified when $\geq 4$ of 8 criteria hold and $\geq 50\%$ of partition endorsements are positive:

| Criterion | Threshold | Physical basis |
|-----------|-----------|---------------|
| $\Delta\mathrm{NFR}$ gain | $\geq 0.15$ drop | Nodal equation convergence |
| Coherence ratio | $0.72 \leq r \leq 1.38$ | Structural similarity |
| $\Phi_s$ delta | $\leq 0.35$ | U6 structural-potential confinement |
| Gradient delta | $\leq 0.40$ | Phase desynchronization limit |
| Curvature delta | $\leq 0.45$ | Geometric stability |
| Periodicity confidence | $\geq 0.55$ | Structural mode certainty |
| Stabilized fraction | $\geq 0.30$ | Multi-scale coherence (U5) |
| Coverage fraction | $\geq 0.15$ | Spatial completeness |

### 9.4 Pure Mode

Setting `TNFR_PURE_MODE=1` restricts factor certification to structural confidence ($\geq 0.6$) without arithmetic divisibility checks, isolating the TNFR-specific signal from classical shortcuts.

### 9.5 Three Sectors of Primality (Unification — MEASURED)

The factorization machinery (§9.1–9.4), the arithmetic primality criterion (§4), and the emergent-geometry program are **one structure read in three sectors**, not independent projects. Example [117_emergent_geometry_residue_graph.py](../examples/08_emergent_geometry/117_emergent_geometry_residue_graph.py) measures all three with the **emergent geometry used for everything** (the structural-diffusion operator $L_{rw} = I - D^{-1}W$ is *exactly* the canonical ΔNFR EPI channel), and `benchmarks/primes_as_consequence.py` (Camino 11) frames the trichotomy:

| Sector | Method | Input | Emergent? |
|--------|--------|-------|-----------|
| **A — Arithmetic** | $\Delta\mathrm{NFR}(n)=0$ (§4) | $\Omega, \tau, \sigma$ (the factorization) | **re-expression** (primes-IN; exact but circular as a derivation) |
| **B — Spectral** | $g(n)=\lvert\lambda_2(\text{residue circulant}) - \tfrac{n-\sqrt n}{2}\rvert = 0$ | only $x^2 \bmod n$ | **genuinely emergent** (primes-OUT; non-circular) |
| **C — Representation** | irreducibility (Schur $\langle\chi,\chi\rangle=1$) | a finite group | **refuted** (the dim-4 mode of $K_5$ is irreducible yet $4=2\cdot 2$) |

**The unification, stated honestly:**

1. **Sector B is the genuine emergence.** The Paley gap $g(n)=0$ selects the primes $n\equiv 1\pmod 4$ from the **self-adjoint spectrum of the quadratic-residue graph alone** — it never computes $n\bmod k$. Primality is, in part, a *consequence* of self-adjoint structure, not a primitive. This is the non-circular core that the arithmetic sector A (which consumes $\Omega,\tau,\sigma$) cannot claim.

2. **The factor signal is spectral, not substrate.** For a semiprime $n=p\cdot q$ the factor $p$ appears as an **exact Fourier/coset mode** of the emergent diffusion spectrum ($\eta^2_{\text{coset}}\to 1$, collapsing under a node-label shuffle — example 117 Q2). But the residue graph is **regular/circulant**, so the emergent random-walk operator and the classical Laplacian **share eigenvectors**: the coset signal is the residue-graph (CRT) structure re-expressed, *not* something the emergent framing adds. The genuinely-emergent per-node symplectic substrate ($\Phi_s, K_\phi, J_{\Delta\mathrm{NFR}}$) is **BLIND** to the cosets ($\eta^2\approx 0$ — example 117 Q3), exactly as on the arithmetic network (examples 101/103/116). The substrate **re-expresses** what lives in the spectrum; it does not independently discover the factor.

3. **Both walls coincide.** Sector B is **partial**: in the **real/self-adjoint** spectrum it detects only $n\equiv 1\pmod 4$ (misses $2$ and many $n\equiv 3\pmod 4$) — it reaches the support/scale, never the *phase* (§9.6 crosses precisely this restriction by going to the **directed** operator's complex spectrum, extending detection to all odd primes; the *continuous* arg-$\zeta$ phase still remains beyond reach). The residual is the same $e$–$\pi$ / $\mathrm{Fix}(G)^\perp$ obstruction as the paused TNFR-Riemann program ($S(T)=\tfrac1\pi\arg\zeta(\tfrac12+iT)=\ker(\mathcal R_\infty)$; §10, TNFR_RIEMANN_RESEARCH_NOTES §13septies). Multiplication is the Fundamental Theorem re-expressed via UM/REMESH ([94](../examples/07_number_theory/94_generative_number_construction.py), $\nu_f=\log p$ additive-in-log); addition (Goldbach) is **orthogonal** to this multiplicative coherence ([97](../examples/07_number_theory/97_goldbach_additive_multiplicative.py)) and would need a branch-B2 additive operator. The three number-theory questions (primality, factorization, the Riemann zeros) hit one obstruction, located precisely, not three.

**Net:** the optic-shift converts the imposed arithmetic carrier (sector A) into a *partially emergent* one (sector B) and pins the residual at the phase / the $\not\equiv 1\pmod 4$ class. It SHARPENS the unification; it does not dissolve the wall. Genuine non-circular emergence exists in TNFR — but partial, spectral, and never in the per-node emergent substrate.

### 9.6 The Phase Sector — Sector B Extended to All Odd Primes (MEASURED)

The "partial" limitation of sector B (only $n\equiv 1\pmod 4$, §9.5) is **not** a wall of TNFR — it is an artefact of restricting to the *real/self-adjoint* spectrum. Example [119_phase_sector_directed_residue.py](../examples/08_emergent_geometry/119_phase_sector_directed_residue.py) crosses it using the **same canonical emergent operator** on the **directed** residue graph.

**The structural reason for the mod-4 split.** For $n\equiv 1\pmod 4$, $-1$ is a quadratic residue, so the residue graph is **symmetric**: the canonical operator $L_{rw}=I-D^{-1}W$ is self-adjoint and its spectrum is **real**. For $n\equiv 3\pmod 4$, $-1$ is **not** a residue, so the residue digraph is a **Paley tournament** (one directed edge per pair); the canonical operator is **non-self-adjoint** (a non-symmetric circulant, hence still *normal*) and its spectrum is **complex** — the arithmetic content lives in the **phase** (the imaginary part), which the real spectrum discards.

**Doctrine compliance.** This is the *same* `structural_diffusion_operator` (the literal ΔNFR EPI channel) applied directly to a `networkx.DiGraph` — verified identical to a hand-built operator ($\max|\Delta|=0$). Nothing ad-hoc; the complex spectrum **is** the canonical emergent geometry on a directed graph. Only arithmetic input: $x^2\bmod n$.

**Measured (all reproducible in example 119):**

1. **Unified primality.** "The directed emergent operator has exactly **3 distinct (complex) eigenvalues**" $\iff n$ is an odd prime — **58/58 correct** over odd $n\in[5,119]$, zero mismatches. This extends Reading B from $n\equiv 1\pmod 4$ to **all odd primes** via the phase sector.

2. **Prime powers resolved.** The directed operator gives **4+** distinct eigenvalues for $9=3^2$, $25=5^2$, $49=7^2$, $121=11^2$ — it **separates** primes from prime powers, which the real symmetric operator of example 117 could **not** ($49$ was rigid there, the honest §9.5 caveat). The phase channel removes that caveat.

3. **The phase encodes $\sqrt n$.** For $n\equiv 3\pmod 4$ primes the imaginary spectrum is the Paley-tournament eigenvalue structure $(-1\pm i\sqrt n)/2$ on the adjacency; the diffusion operator's $\max|\mathrm{Im}(\lambda)|=\sqrt n/(n-1)$ **exactly** (ratio $1.000$) — a Gauss-sum fact carried in the **phase**.

**Honest scope.** A genuine, non-circular extension of Reading B to all odd primes (input only $x^2\bmod n$), removing the prime-power caveat — a real improvement over §9.5. But it remains **spectral** and bounded by the same $e$–$\pi$ / $\mathrm{Fix}(G)^\perp$ wall: it detects primality structurally, it does **not** factor, does **not** reach the continuous phase $S(T)=\tfrac1\pi\arg\zeta(\tfrac12+iT)$, and closes no open problem. The "3 distinct eigenvalues" rigidity is the doubly-regular-tournament signature (a known algebraic-graph fact), recovered as the canonical emergent operator's complex spectrum. The lesson: **the phase sector is reachable — the complex field $\Psi=K_\phi+i\,J_\phi$ is the right object** (AGENTS.md "Regime Correspondences") — but the *continuous* arg-$\zeta$ phase of the Riemann residual still lies beyond this discrete-spectrum reach.

### 9.7 The Symmetry Wall — Why the Substrate Is Blind and the Spectrum Is Not (MEASURED)

§9.6 detects primality in the **global spectrum**; §9.5 and examples 103/116 found the **per-node symplectic substrate** ($\Phi_s$, $K_\phi$, $J_{\Delta\mathrm{NFR}}$) blind to arithmetic. These look contradictory — the *same* canonical emergent operator on the *same* residue digraph. Example [120_symmetry_wall_substrate_vs_spectrum.py](../examples/08_emergent_geometry/120_symmetry_wall_substrate_vs_spectrum.py) resolves the contradiction and unifies the arc with **one** structural mechanism: **vertex-transitivity**.

**The mechanism.** The residue digraph is a **Cayley digraph** of $\mathbb{Z}_n$ with connection set $=$ the quadratic residues. The translation $\sigma:i\mapsto i+1\pmod n$ preserves the difference $j-i$, hence the QR edge set: it is **always** a graph automorphism (every $n$, verified). The automorphism group acts **transitively** on nodes — every node is structurally equivalent. Consequence: the graph's arithmetic (which differences are QRs) is a property of the *edge* structure invariant under the *node* automorphism; it **cannot label any individual node**. So:

- Any per-node substrate variation comes from the (arithmetic-neutral) **seed**, never from the arithmetic — the substrate lives in the symmetric / fixed sector $\mathrm{Fix}(G_{\mathrm{aut}})$, **blind** to the connection set.
- The arithmetic appears only in a **global** invariant sensitive to the connection set — the **spectrum** (eigenvalues $=$ group-character / Gauss sums) $=$ the complement $\mathrm{Fix}(G_{\mathrm{aut}})^\perp$.

**The double dissociation (measured).** Compare the Paley residue digraph (QR structure) against a **random regular tournament** of the same out-degree, both seeded identically and evolved by the canonical nodal equation $\partial\mathrm{EPI}/\partial t=\nu_f\cdot\Delta\mathrm{NFR}$:

| $n$ | Paley distinct eig. | random distinct eig. | Paley $\sigma(\Phi_s)$ | random $\sigma(\Phi_s)$ |
|----:|--------------------:|---------------------:|------------------------:|-------------------------:|
| 11 | **3** | 11.0 | 0.400 | 0.366 |
| 23 | **3** | 23.0 | 0.587 | 0.618 |
| 47 | **3** | 47.0 | 0.998 | 0.938 |

- **Spectrum SEES the arithmetic**: Paley is rigidly 3 distinct eigenvalues (the §9.6 prime signature); the random tournament has $\sim n$. Swapping the QR structure for a random tournament changes the spectrum completely.
- **Substrate is BLIND**: the per-node $\Phi_s$ dispersion is statistically **identical** for Paley and the random tournament. The substrate cannot tell the QR arithmetic from a random tournament of the same degree.

Across odd $n$ the spectral test "$3$ distinct $\iff$ prime" is **18/18 correct**, while $\sigma(\Phi_s)$ grows monotonically with $n$ (graph size) and composites can exceed primes (e.g. $25$ vs $29$) — the substrate tracks size, not primality.

**The unification (one wall, four domains).** Vertex-transitivity confines arithmetic to the spectral / group-representation sector $\mathrm{Fix}(G_{\mathrm{aut}})^\perp$ and leaves the per-node substrate in the symmetric sector $\mathrm{Fix}(G_{\mathrm{aut}})$, blind. This is the **same** structure as the paused TNFR-Riemann program, where the oscillatory residue $S(T)=\tfrac1\pi\arg\zeta(\tfrac12+iT)$ lives in $\ker(\mathcal R_\infty)\cap\mathrm{Fix}(S_n)^\perp$, unreachable by symmetric ($\mathrm{Fix}$-trapped) constructions (AGENTS.md "REMESH-∞ Closure"; TNFR_RIEMANN_RESEARCH_NOTES §13septies, §13sexagesima-octava Tetrad-$\mathrm{Fix}(S_n)$ Lemma). Physics (the symplectic substrate), number theory (Gauss sums, primality), emergent geometry (the canonical operator) and the Riemann residual hit **one symmetry wall**, located precisely: arithmetic is in the spectrum, the per-node substrate is in the fixed sector.

**Honest scope.** This **explains** the $e$–$\pi$ / $\mathrm{Fix}(G)^\perp$ wall structurally (vertex-transitivity / representation theory); it does **not** cross it and closes no open problem. It confirms, with a measured double dissociation and an arithmetic-neutral control, that running the directed dynamics does **not** let the per-node substrate see arithmetic — the blindness is a **symmetry constraint**, not a dynamics artefact. The arithmetic remains spectral, bounded by the same wall as the paused Riemann program.

### 9.8 Can a Canonical Symmetry-Break Cross the Wall? — The B2-P2 Lever, Measured (NEGATIVE)

§9.7 located the wall at vertex-transitivity. The obvious next move is to **break** that symmetry canonically — the TNFR-Riemann program calls this candidate **B2-P2 (NodeIndexedCouplingWeights)**. The analytical verdict is on record: AGENTS.md "B0★-β-P2 FAILS" (§13sexagesima-sexta) closes P2 at the **slot level** — the nodal equation $\partial\mathrm{EPI}/\partial t=\nu_f\cdot\Delta\mathrm{NFR}$ has no per-node-weight slot; weights enter only as graph-level **channel** scalars ($\texttt{DNFR\_WEIGHTS}=\{\text{phase},\text{epi},\text{vf},\text{topo}\}$), so any per-node law needs an external rule-selection axiom not derivable from the catalog. Example [121_canonical_symmetry_break_negative.py](../examples/08_emergent_geometry/121_canonical_symmetry_break_negative.py) **measures** this closure at the number-theory level.

**The code fact (the missing slot).** `tnfr.dynamics.dnfr._configure_dnfr_weights` produces ONE graph-level dict of channel weights, normalized once and reused for every node. There is no per-node weight in the canonical machinery; the only per-node levers are (a) the initial seed and (b) the per-node $\nu_f$. Both are tested.

**The three levers (measured).**

| Lever | Result | Reading |
|---|---|---|
| **D1** symmetric seed | $\sigma(\Phi_s)\sim 10^{-32}$ (machine zero), prime & composite alike | the canonical dynamics ALONE makes zero per-node structure ($\Delta\mathrm{NFR}=0$ on a uniform field); all of §9.7's variation came from the random seed |
| **D2** structure-derived $\nu_f$ | in/out-degree, triangle counts: $\sigma=0$ exactly | every per-node structural invariant is constant on the vertex-transitive graph → any canonical structure-derived $\nu_f$ is uniform → no break |
| **D3** arithmetic-injected $\nu_f$ | $\sigma_{\mathrm{arith}}/\sigma_{\mathrm{shuffled}}\approx 1$ (0.96–1.05) | injecting $\nu_f=1+[\,i\in\mathrm{QR}\,]$ does break uniformity, but a shuffled control with the same $\nu_f$ multiset (QR labels destroyed) gives identical dispersion → the substrate echoes the injected **multiset**, not the arithmetic (§9.5/ex 116 mechanism), and is circular |

**Conclusion.** There is **no canonical (non-circular) per-node lever** that breaks vertex-transitivity: the nodal equation has no per-node weight slot (code fact); structure-derived levers are uniform (D2); the symmetric dynamics makes no structure (D1); and the only lever that does break uniformity is an external arithmetic injection that the shuffled control reveals as echo (D3). This is the **empirical, number-theory-level confirmation** of the analytical B2-P2 closure. The wall of §9.7 is **structural**, not an artefact of which canonical knob was turned.

**Honest scope.** A clean **measured negative**: it confirms the analytical closure, it does **not** break the wall, and it closes no open problem. It is a re-expression of two known structural facts — no canonical per-node observable exists on a homogeneous (vertex-transitive) graph, and the nodal equation carries no per-node weight slot — measured here in TNFR's own substrate.

### 9.9 Factorization in the Phase Sector — the Complex Spectrum Completes the Recovery (MEASURED)

§9.6 (Reading B, real sector) recovered the factor coset $(i\bmod p)$ of a semiprime $n=p\cdot q$ as an $\eta^2\to 1$ Fourier mode of the emergent diffusion spectrum, but only **partially**: it missed the high-frequency factor modes of $51=3\cdot 17$ and $91=7\cdot 13$. §9.6/§9.7 showed the missing content lives in the **phase** (the complex spectrum of the directed residue digraph). Example [122_factorization_phase_sector.py](../examples/08_emergent_geometry/122_factorization_phase_sector.py) uses that phase sector to **complete** the factor-coset recovery.

**The structural fact (CRT, present in both sectors).** For $n=p\cdot q$ the factor coset $(i\bmod p)$ corresponds to the Fourier frequencies $k=$ multiples of the cofactor $q$. A pure Fourier mode $\exp(2\pi i k j/n)$ with $k$ a multiple of $q$ is **constant within each coset** $(i\bmod p)$, hence an **exact eigenvector** of the emergent operator (a circulant / Cayley digraph) — verified to machine precision (eigenvector residual $\sim 10^{-14}$) for BOTH the undirected (real) and directed (complex) residue operator. The factor coset is CRT/circulant structure ($\mathbb{Z}_n\cong\mathbb{Z}_p\times\mathbb{Z}_q$), present in both spectra.

**Why the real sector misses 51 and 91.** The undirected residue operator is **symmetric**: its eigenvalues come in degenerate pairs ($\lambda_k=\lambda_{n-k}$). When the factor-coset frequency lands in a degenerate eigenspace, the eigensolver returns an arbitrary real combination that **scrambles** the coset structure, so the $\eta^2$ test fails (51, 91). The directed operator is **non-symmetric** (a non-self-adjoint circulant): its eigenvalues are complex Gauss sums, which are **less degenerate** and **isolate** the factor-coset mode — so the complex spectrum exposes precisely the modes the real sector loses.

**Measured (the correct complex-mean $\eta^2$).** *(One must use complex means: a Fourier mode has flat magnitude, so a magnitude-$\eta^2$ is blind to the coset.)*

| Sector | Factor recovery (scan prime $d\le\sqrt n$, smallest with best $\eta^2>0.9$) |
|---|---|
| Real undirected | **8/10** — fails on exactly 51 and 91 (the §9.6 caveat) |
| Complex directed | **10/10** — recovers 51 and 91 via the phase |

The shuffle control collapses the factor-coset $\eta^2$ from $1.0$ to the random baseline ($\sim 0.1$–$0.2$): the signal is the CRT/circulant structure, not an artefact.

**Honest scope.** This re-expresses CRT / Fourier **period** structure (the content Shor's algorithm exploits via period finding) in the canonical emergent spectrum. It is **not** a new or fast factoring algorithm: the candidate scan is $O(\sqrt n)$ prime divisors — the same order as trial division, **no speedup**, and **no cryptographic threat**. The complex sector's only advantage is reduced eigenvalue degeneracy (a linear-algebra fact about circulant / Cayley digraphs), which isolates the factor-coset mode. The result **completes** §9.6's partial real-sector recovery ($8/10\to 10/10$) via the phase sector; it closes no open problem.

### 9.10 The Symmetry-Sector Decomposition — the General Principle Behind the Whole Arc (MEASURED, CAPSTONE)

§9.7 located the residue-digraph wall at vertex-transitivity. Example [123_symmetry_sector_decomposition.py](../examples/08_emergent_geometry/123_symmetry_sector_decomposition.py) shows that is a **special case** of a general representation-theoretic principle of the canonical emergent operator — the single structure behind every wall in the §9.5–§9.9 arc (and the Riemann residual).

**The principle (Schur, applied to the canonical emergent operator).** For **any** graph $G$ with automorphism group $\mathrm{Aut}(G)$, the canonical emergent operator $L_{rw}=I-D^{-1}W$ is **equivariant**: it commutes with the permutation representation of every automorphism, $P_\sigma L_{rw}=L_{rw}P_\sigma$ for all $\sigma\in\mathrm{Aut}(G)$. By Schur's lemma an equivariant operator block-diagonalizes by the isotypic components (irreps) of $\mathrm{Aut}(G)$. The coarsest split is

$$\mathbb{R}^N=\mathrm{Fix}(G)\ \oplus\ \mathrm{Fix}(G)^\perp,$$

where $\mathrm{Fix}(G)=\{\text{functions constant on the orbits of }\mathrm{Aut}(G)\}$ is the trivial isotypic component and $\dim\mathrm{Fix}(G)=$ the number of vertex orbits. $L_{rw}$ preserves each block. Consequently any canonical **per-node** observable that is itself $\mathrm{Aut}(G)$-invariant lands in $\mathrm{Fix}(G)$ — constant **within** each orbit, never resolving node-from-node inside an orbit — while all **discriminating** information lives in $\mathrm{Fix}(G)^\perp$, the spectrum.

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
- **M4**: the canonical per-node symplectic substrate from a symmetric seed satisfies $P_{\mathrm{triv}}v=v$ exactly (orbit-constant); vertex-transitive $\Rightarrow$ $\sigma(\Phi_s)=0$ — the §9.7 blindness, now a **corollary**.
- **M5**: only the constant eigenmode has $\lVert P_{\mathrm{triv}}v\rVert=1$ (it **is** $\mathrm{Fix}(G)$); every node-separating mode has $\lVert P_{\mathrm{triv}}v\rVert=0$ ($\mathrm{Fix}(G)^\perp$).

**The unification.** The residue-digraph wall (§9.7), the substrate blindness (§9.5/ex 103/116), the spectral primality (§9.6), and the Riemann oscillatory residue $S(T)\in\ker(\mathcal R_\infty)\cap\mathrm{Fix}(S_n)^\perp$ are the **same** $\mathrm{Fix}(G)/\mathrm{Fix}(G)^\perp$ split for different symmetry groups. The star and path (non-vertex-transitive) sharpen the binary blind/sees of §9.7 to the full orbit structure: the substrate resolves the orbit partition and no finer.

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

**The even-modulus boundary (PROVED).** The proof uses cyclicity of $(\mathbb{Z}/p\mathbb{Z})^\times$ at exactly one point — that the $k$-th powers form a single index-$d$ subgroup. This holds at every **odd** prime power $p^e$ (where $(\mathbb{Z}/p^e\mathbb{Z})^\times$ is cyclic), giving the local quadratic factor $f(e)=e+\lceil e/2\rceil+1$ and the conductor-annotated product theorem over odd moduli. It **fails at the prime $2$**: $(\mathbb{Z}/2^e\mathbb{Z})^\times$ is **non-cyclic** for $e\ge 3$ ($\cong\mathbb{Z}/2\times\mathbb{Z}/2^{e-2}$), so the squares form an index-$4$ (not index-$2$) subgroup and the Gauss-sum stratification differs; already $2^1$ is degenerate (the sole unit is $1$). Measured (example 154 machinery): the conductor-annotated count at $2^e$ is $2,4,8,10,14,16,20$ for $e=1,\dots,7$ versus $f(e)=3,4,6,7,9,10,12$ — agreeing only at $e=2$ by coincidence, while every odd prime power matches $f(e)$ exactly. Hence the conductor-annotated product theorem and the cyclotomy law are genuinely **odd-only**, and $A(2^e)$ is the arithmetic continuation, not a spectral count.

**Honest scope.** The cyclotomy law is classical Gauss-period / cyclotomy theory (the $k$-th power Cayley eigenvalues are Gauss periods of degree $\gcd(k,p-1)$); the contribution is the **TNFR structural-diffusion framing** and the closed-form `power_residue_rank` — now a **proved** canonical fact, not a measured pattern. Verified computationally for $k\le 40$ across the primes $p<64$ (680 cases, 0 failures) and proved for all $k$. It detects primality/cyclotomy structurally; it does not factor, does not reach the continuous arg-$\zeta$ phase, and closes no open problem.

### 9.8 The Ontological Position of a Number (the emergent ladder)

§9.5–9.7 answer "is primality emergent?" sector by sector. This subsection assembles them — together with the cardinal/operation emergence of the emergent-number arc — into the **ontological position** of a number: a ladder whose every rung is read from canonical TNFR structure/dynamics, measured in example [155_ontological_position_of_numbers.py](../examples/08_emergent_geometry/155_ontological_position_of_numbers.py).

| Layer | What $n$ **is** | Mechanism | Emergent? |
|-------|-----------------|-----------|-----------|
| 0 Substrate | — | $\mathbb{R}$ continuum + $\pi$ (the one structural scale) | assumed |
| 1 Cardinal | a degeneracy $=\dim$ irrep of $\mathrm{Aut}(G)$ | Laplacian multiplicity | ✅ |
| 2 Operations | $+, \times$ | graph products ($\square\!\to\!\sum$ spectra, $\otimes\!\to\!\prod$ spectra) | ✅ |
| 3 Primality | $\rho(n)=3$ | directed residue operator (§9.6) | ✅ (Sector B) |
| 3′ Arithmetic | the factorization type ($\Omega, \tau$) | the multiplicative rank $\rho(n)$ | ✅ (this §) |
| 4 The wall | prime **identities** / $\arg\zeta$ phase | $S(T)\in\mathrm{Fix}(S_n)^\perp$ | ❌ (open) |

**The multiplicative spectral rank (Layer 3′, realizes the §9.7 PROVED law).** The quadratic-residue spectral rank $\rho(n)$ (the §9.6/§9.7 count of distinct diffusion eigenvalues) extends from primes to all $n$ via the **proved** conductor-annotated product law of §9.7: $A(m)=\prod_{p^e\|m}(e+\lceil e/2\rceil+1)$ is multiplicative with prime-power factor $f(e)=e+\lceil e/2\rceil+1$ depending **only on the exponent** ($f=3,4,6,7,9,\dots$). The *unannotated* scalar rank realizes it at small exponents — $\rho(p)=3$ (cyclotomy $k=2$), $\rho(p^2)=4$, $\rho(p^3)=6$ — and is multiplicative there ($\rho(mn)=\rho(m)\rho(n)$, 0 exceptions over the demo range), **faithfully encoding the factorization type**: $\rho=3\leftrightarrow$ prime, $4\leftrightarrow p^2$, $6\leftrightarrow p^3$, $9\leftrightarrow pq$, $12\leftrightarrow p^2q$. The factorization and divisor channels of the $\Delta\mathrm{NFR}$ triad ($\Omega=\sum_i a_i$, $\tau=\prod_i(a_i+1)$) are therefore **read off the spectrum** — from $x^2\bmod n$, never trial division. The arithmetic that sector A *consumes* (§9.5) genuinely *emerges* here.

**The wall, located on the ladder.** $\rho$ fixes the **type**, never the prime **identities** ($\rho(15)=\rho(35)=9$); it is not globally injective on types ($\rho=36$ is shared by $p^3q^3$ and $p^2qr$ — a type collision) and the unannotated scalar rank **aliases** at high prime powers (the §9.7 / example 154 scalar CRT wall: $3^7\!\cdot5^2\!\cdot41^2$ gives scalar $191$ vs product $192$). Recovering the identities is the same $e$–$\pi$ / $\mathrm{Fix}(S_n)^\perp$ residue (the continuous $\arg\zeta$ phase, §10) as every other sector. **Net:** the emergent ontology positions a number completely **up to** the prime-identity / phase wall — cardinal, operations, primality and factorization type all derive from structure; only the identities and the continuous phase remain. This is the precise sense in which "the arithmetic emerges from the canonical TNFR structure and dynamics."

### 9.12 The Arithmetic Pulse — the Cyclotomy Law as the Prime's Chord (MEASURED)

The *pulse* read-out (the **conservative** face of the nodal dynamics,
[EMERGENT_ONTOLOGY.md §5.5](EMERGENT_ONTOLOGY.md)) reads the resonant spectrum
$\omega_k=\sqrt{\lambda_k}$ of the canonical $L_{rw}$. Applied to the arithmetic
NFR — the residue Cayley network $\mathrm{Cay}(\mathbb{Z}/n,R_k)$ — its **tone
structure is exactly the PROVED cyclotomy law** of §9.11.
[benchmarks/emergent_arithmetic_pulse.py](../benchmarks/emergent_arithmetic_pulse.py)
measures it.

**The pulse tone-count is the cyclotomy law.** The number of *distinct* resonant
tones of the residue-NFR pulse is `structural_frequency_rank` (the distinct
eigenvalues of $L_{rw}$), and on a prime this is

$$\#\{\text{distinct tones}\} = s_k(p) = \gcd(k,p-1)+1 \quad (\text{§9.11, PROVED}).$$

Measured exactly for $k=2,3,4,5$ across the primes (0 mismatches): the **arithmetic
pulse IS the cyclotomy law** — the harmonic structure of a number's vibration is
its cyclotomy degree.

**A prime is the most degenerate chord.** For $p\equiv1\pmod4$ the real Paley-NFR
pulse is the **silent mode** ($\lambda=0$) plus exactly **two resonant tones**
$(\omega_-,\omega_+)$, each with multiplicity $(p-1)/2$ — the pulse's own
`spectral_multiplicity` field reads $(p-1)/2$ exactly. A prime vibrates in the
simplest chord the arithmetic NFR allows, at any size; **composites split the
chord into more tones, multiplicatively** ($15\to9=3\times3$, $45\to12=4\times3$),
so the tone-count encodes the **factorization type** — the multiplicative spectral
rank of the §9.8 ladder, now read as the chord size.

**The pulse splits across the symmetry wall.** The two scales of the pulse land on
the two sides of the §9.7/§9.10 $\mathrm{Fix}(G)\oplus\mathrm{Fix}(G)^\perp$ split:
the **per-NFR** pulse is **blind** (the residue graph is vertex-transitive, so the
per-node substrate is in $\mathrm{Fix}(G)$), while the **collective** pulse — the
spectrum — carries the cyclotomy ($\mathrm{Fix}(G)^\perp$). The real/phase split of
§9.6 is inherited: the *real* conservative pulse reads the cyclotomy on the
symmetric NFR ($p\equiv1\pmod4$); the *complex* directed pulse extends it to all
odd primes.

**Honest scope.** The tone-count *is* `structural_frequency_rank` (already the
documented cyclotomy diagnostic), and $s_k(p)=\gcd(k,p-1)+1$ is the PROVED
classical Gauss-period fact of §9.11. The contribution is the conservative-**pulse**
reading — those distinct eigenvalues are the distinct resonant **tones** of the
arithmetic vibration, so a prime is a maximally-degenerate chord and the
factorization type is the chord size. It detects primality / factorization
**type** structurally; it does **not** factor, does **not** reach the prime
**identities** or the continuous $\arg\zeta$ phase (the same $\mathrm{Fix}(S_n)^\perp$
wall, §10), and closes no open problem.

---

## 10. Prime Path Graphs and the TNFR-Riemann Connection

### 10.1 The Discrete TNFR-Riemann Operator

The TNFR-Riemann program constructs a family of operators on prime path graphs:

$$H^{(k)}_{\mathrm{TNFR}}(\sigma) = L_k + V_\sigma$$

where:
- $L_k$ is the graph Laplacian of the **prime path graph** $G_k$ (first $k$ primes $p_1, p_2, \ldots, p_k$ connected sequentially)
- $V_\sigma$ is a **structural potential** parametrized by $\sigma \in \mathbb{R}$:

$$V_\sigma(i) = (\sigma - \tfrac{1}{2}) \log(p_i)$$

### 10.2 Critical Parameter Convergence

The **critical parameter** $\sigma_c^{(k)}$ is the value of $\sigma$ at which the smallest eigenvalue of $H^{(k)}_{\mathrm{TNFR}}(\sigma)$ changes sign (spectral phase transition).

**Main numerical result**:

$$\sigma_c^{(k)} = \frac{1}{2} + O\!\left(\frac{1}{\log k}\right) \quad \text{as } k \to \infty$$

This convergence is:
- **Numerically verified** across multiple topologies and parameter ranges
- **Analytically bounded** using the Prime Number Theorem and telescoping identities
- **Universal** — independent of graph construction details

### 10.3 Connection to the Riemann Hypothesis

At $\sigma = 1/2$, the potential $V_{1/2}$ vanishes and $H^{(k)}_{\mathrm{TNFR}}$ reduces to the pure graph Laplacian. The spectral transition at $\sigma_c \to 1/2$ provides **structural coherence evidence** for the critical line of the Riemann zeta function $\zeta(s)$.

**Status**: The bridge from the discrete TNFR operator result to the classical Riemann Hypothesis remains an **open conjecture** (Conjecture 10.1 in the Riemann Research Notes). The framework constitutes a research program, not a closed proof.

### 10.4 Tetrad Fields on the Prime Path

From eigenpairs $(\lambda_j, \phi_j)$ of $H^{(k)}_{\mathrm{TNFR}}$:

**Phase gradient** (discrete):
$$|\nabla\phi|^{(j)} = \frac{1}{k-1}\sum_{i=1}^{k-1}|\phi_j(p_{i+1}) - \phi_j(p_i)|$$

**Phase curvature** (discrete):
$$K_\phi^{(j)} = \frac{1}{k-2}\sum_{i=2}^{k-1}|\phi_j(p_{i+1}) - 2\phi_j(p_i) + \phi_j(p_{i-1})|$$

**Coherence length** (from correlation decay):
$$C_j(r) \approx A_j \cdot e^{-r/\xi_C^{(j)}}$$

These tetrad fields on the prime path link the arithmetic distribution of primes to the structural field theory.

### 10.5 Refactoring the Riemann Attack — From the Self-Adjoint Prime-Ladder to the Non-Self-Adjoint Phase Operator (MEASURED)

The TNFR-Riemann program is paused at the **Tetrad-Hilbert-Pólya (T-HP)** conjecture on the **self-adjoint** prime-ladder operator P14 (`src/tnfr/riemann/prime_ladder_hamiltonian.py`). That route is walled by the **Euler-Orthogonality Lemma** (TNFR_RIEMANN_RESEARCH_NOTES §13vicies-novies.11): on the prime-ladder graph every canonical operator **commutes with the $S_n$ prime-relabelling**, so the spectrum lives in $\mathrm{Fix}(S_n)$ and is structurally blind to the Riemann residue $S(T)=\tfrac1\pi\arg\zeta(\tfrac12+iT)\in\mathrm{Fix}(S_n)^\perp$.

The number-theory reframe (§9.6, §9.8) supplies a **structurally different object** for the same residue: the **directed quadratic-residue diffusion operator** $L_{rw}=I-D^{-1}W$ on the Paley tournament ($n\equiv 3\pmod 4$). It is

- **non-self-adjoint** (a non-symmetric circulant — hence *normal*; its complex spectrum is the $\mathbb{Z}/n$ character / Gauss-sum eigenbasis) → its spectrum is **complex**, carrying the arithmetic in the **phase** (imaginary part) — structurally aligned with the fact that the Riemann zeros are **imaginary parts** $\{\gamma_n\}$, whereas the self-adjoint Hilbert-Pólya framing seeks a **real** spectrum; and
- symmetric only under the **affine group of $\mathbb{Z}/n$**, **not** the $S_n$ prime-relabelling — so it is **not subject to the Euler-Orthogonality Lemma**, and it already reaches **all odd primes** (§9.6), past the self-adjoint mod-4 restriction.

So the natural question is whether the attack should pivot from "build a *self-adjoint* operator with spectrum $\{\gamma_n\}$" to "read the residue off the *non-self-adjoint* phase operator".

**The pre-registered falsifier (MEASURED).** `benchmarks/residue_phase_vs_riemann.py` tests it on primes $p\equiv 3\pmod 4$:

- **F-GAUSS** — $\max|\mathrm{Im}(\lambda)|(p)=\sqrt p/(p-1)$ **exactly** (ratio $1.000000$, 15/15 primes): the phase content is the **Paley Gauss-sum eigenvalue**, a classical fact.
- **F-ALIGN** — $\mathrm{Pearson}\big(\max|\mathrm{Im}|(p_n),\,\gamma_n\big)=\mathbf{-0.9068}$: the residue phase content **decreases** like $1/\sqrt p$ while the zeros $\gamma_n$ **increase** — opposite trends.
- **Verdict:** `GAUSS_CONFIRMED_RIEMANN_REFUTED`.

**Honest net.** The non-self-adjoint phase operator **does** evade the Euler-Orthogonality wall and reaches arithmetic in the phase — a genuine structural advance and a more natural arena than the self-adjoint prime-ladder — but its phase content is $\sqrt p$ **Gauss sums**, not $\{\gamma_n\}$. This is the §9.5 "**both walls coincide**" statement made operator-explicit: the residue-phase $\to$ $\zeta$-zeros bridge is the **same** $e$–$\pi$ / $\mathrm{Fix}(S_n)^\perp$ residue. The reframe **relocates and sharpens** the obstruction — from "find a self-adjoint $F$ with $\mathrm{spec}=\{\gamma_n\}$" to "connect the non-self-adjoint Gauss-sum phase ($\sqrt p$) to the $\zeta$-zero phase ($S(T)$)" — but **does not dissolve it**. The program stays paused at T-HP; **G4 = RH remains OPEN**; this closes no open problem.

---

## 11. Worked Examples

### 11.1 Prime Detection: $n = 17$

$$\Omega(17) = 1, \quad \tau(17) = 2, \quad \sigma(17) = 18$$

$$\Delta\mathrm{NFR}(17) = 1 \times (1-1) + 1 \times (2-2) + 1 \times \left(\frac{18}{17} - \frac{18}{17}\right) = 0$$

Structural triad: $\mathrm{EPI}(17) \approx 2.75$, $\nu_f(17) \approx 1.47$, $C_{\text{local}} = 1.0$.

**Interpretation**: Zero pressure, perfect coherence, structural fixed point.

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

### 11.4 Highly Composite: $n = 30 = 2 \times 3 \times 5$

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
| **Riemann operators** | `src/tnfr/riemann/operator.py` | Discrete TNFR-Riemann spectral operators |
| **Spectral convergence** | `src/tnfr/riemann/spectral_proof.py` | $\sigma_c \to 1/2$ convergence proofs |
| **Canonical constants (repo)** | `src/tnfr/constants/canonical.py` | Repository-wide canonical constant definitions |

### 12.2 Executable Demonstrations

| Example | Concept |
|---------|---------|
| [16_riemann_operator_demo.py](../examples/03_riemann_zeta/16_riemann_operator_demo.py) | Critical parameter eigenvalue analysis |
| [18_riemann_convergence_proof.py](../examples/03_riemann_zeta/18_riemann_convergence_proof.py) | Spectral convergence $\sigma_c \to 1/2$ |
| [19_topology_comparison.py](../examples/03_riemann_zeta/19_topology_comparison.py) | Cross-topology universality |
| [21_complex_extension_demo.py](../examples/03_riemann_zeta/21_complex_extension_demo.py) | Complex plane extensions |
| [22_spectral_zeta_demo.py](../examples/03_riemann_zeta/22_spectral_zeta_demo.py) | Spectral zeta and Mellin bridge |
| [23_random_ensemble_rmt_demo.py](../examples/03_riemann_zeta/23_random_ensemble_rmt_demo.py) | Random matrix theory on prime graphs |
| [25_analytical_convergence_demo.py](../examples/03_riemann_zeta/25_analytical_convergence_demo.py) | PNT-based analytical bounds |
| [31_mathematical_constants_basis.py](../examples/02_physics_regimes/31_mathematical_constants_basis.py) | The structural scale π and the mathematical-constant basis |
| [40_arithmetic_number_theory.py](../examples/07_number_theory/40_arithmetic_number_theory.py) | Primality, triad, component analysis |
| [94_generative_number_construction.py](../examples/07_number_theory/94_generative_number_construction.py) | Compositional generation from prime atoms; U5 fractality; grammar certification |
| [95_primes_from_spectral_waves.py](../examples/07_number_theory/95_primes_from_spectral_waves.py) | Prime staircase ψ(x) as spectral-wave superposition; spectral coherence ⟺ RH (honest scope) |
| [96_spectral_vibration_of_coherence.py](../examples/07_number_theory/96_spectral_vibration_of_coherence.py) | Oscillatory residue S(T) as prime-ladder vibration {k·log p}; why aggregate C(t) is blind (honest scope) |
| [97_goldbach_additive_multiplicative.py](../examples/07_number_theory/97_goldbach_additive_multiplicative.py) | Goldbach phase-matching: negative structural result; additive/multiplicative orthogonality; B2/B3 ontological note |
| [100_prime_families_orbits.py](../examples/07_number_theory/100_prime_families_orbits.py) | Special prime families (twin, cousin, sexy, Sophie Germain, safe, Cunningham, Mersenne, constellations) as orbits and level-sets of arithmetic maps on the zero-pressure fixed-point set $Z=\{\Delta\mathrm{NFR}=0\}$; three generator classes; detection exact, infinitude open (honest scope) |
| [101_numbers_as_coupled_network.py](../examples/07_number_theory/101_numbers_as_coupled_network.py) | Numbers as a coupled TNFR network: $\Omega(n)$ grades both the per-node pressure $\Delta\mathrm{NFR}$ and the divisibility/GCD transport centrality ($r\approx 0.8$–$0.9$); primes ($\Omega{=}1$, $\Delta\mathrm{NFR}{=}0$) are the transport periphery, large primes isolated; correspondence-through-$\Omega$ not identity, not scale-free (honest scope) |
| [102_nodal_flow_primes_equilibria.py](../examples/07_number_theory/102_nodal_flow_primes_equilibria.py) | The actual nodal flow $\partial\mathrm{EPI}/\partial t=\nu_f\Delta\mathrm{NFR}$ on numbers: primes are EXACTLY the equilibria (§4 theorem in motion, frozen) while composites drift $\Omega$-graded; refines §7.1 — primes are static low-$\Phi_s$ sinks but NOT dynamical attractors (diffusion flow pulls primes UP toward the composite bulk) |
| [146_primality_grammatical_inertness.py](../examples/07_number_theory/146_primality_grammatical_inertness.py) | Bridges the operator-grammar thread (examples 139-145) to number theory: every operator acts through the single nodal rule $\partial\mathrm{EPI}/\partial t=\nu_f\Delta\mathrm{NFR}$, so on arithmetic nodes (where $\Delta\mathrm{NFR}$ is the §4 primality field) primes are the KERNEL of the capacity ($\nu_f$) lever — frozen under every grammatical program (the dual-lever, ex 37/130). prime $\iff \Delta\mathrm{NFR}{=}0 \iff C{=}1$ (maximal coherence); composite drift $=(\nu_f\text{ gain})\times$ pressure exactly; the U2 convergence target $\Delta\mathrm{NFR}\to 0$ IS primality, $C$ decreasing monotonically with $\Omega$ (coherence debt $=$ factorization complexity); a prime needs the EMPTY word (the identity of the star-free syntactic monoid, ex 145) — primality is grammatical inertness. Restates the §4 theorem through the grammar dynamics (grammar-lens reading); not new number theory (honest scope) |
| [147_numbers_as_free_monoid_words.py](../examples/07_number_theory/147_numbers_as_free_monoid_words.py) | Deepens 146 to its algebraic core, uniting physics + grammar + number theory. By the FTA the multiplicative monoid $(\mathbb{N},\times)$ is the FREE COMMUTATIVE MONOID on the primes — numbers are words (primes = letters, $1$ = empty word, $\Omega$ = word length, $\times$ = concatenation). The coherence debt $\Delta\mathrm{NFR}$ splits by COMPOSITION LAW: the factorization channel $\zeta(\Omega-1)$ is ADDITIVE (a monoid homomorphism, $P_\Omega(mn)=P_\Omega(m)+P_\Omega(n)+\zeta$ exact — the free-monoid backbone), while the divisor $\eta(\tau-2)$ and abundance $\theta(\sigma/n-\ldots)$ channels are MULTIPLICATIVE (the divisor lattice). Multiplying by a prime is the unit destabilizer ($+\zeta$ per letter); the additive channel ALONE detects primality ($\Omega=1\iff$ prime). The DUAL-LEVER (ex 37/130) restricted to arithmetic IS the two canonical additive gradings of the free monoid: count $\Omega$ ($\to\Delta\mathrm{NFR}$ pressure) and size $\log n$ ($\to\nu_f$ capacity, ex 94). Fixes the dictionary physics dual-lever $\leftrightarrow$ free-monoid gradings $\leftrightarrow$ primality; classical multiplicative number theory restated through the lens (honest scope) |
| [148_capacity_arm_carries_von_mangoldt.py](../examples/07_number_theory/148_capacity_arm_carries_von_mangoldt.py) | Answers which dual-lever arm carries the Riemann difficulty (and why the substrate is blind). The CAPACITY arm $\log n = \sum_{d\mid n}\Lambda(d)$ exactly (Möbius-inverse $\Lambda=\mu*\log$), so von Mangoldt — and $\psi(x)=\sum\Lambda$, the Chebyshev staircase carrying $S(T)$ (ex 96) — sits on the capacity ($\nu_f$, ex 147) arm. The Riemann ZEROS are the POLES of the capacity Dirichlet series $-\zeta'/\zeta(s)=\sum\Lambda(n)n^{-s}$ (P12; measured simple pole residue 1 at $\rho_1=\tfrac12+14.1347i$), while $\sum\Omega(n)n^{-s}=\zeta(s)P(s)$ has $\zeta$ in the numerator (zeros invisible to the PRESSURE arm). The pressure arm $\Omega$ is smooth (Erdős–Kac Gaussian CLT); the per-node substrate encodes pressure ($\Phi_s\leftarrow\Delta\mathrm{NFR}\leftarrow\Omega$), so it is structurally BLIND to the capacity/von-Mangoldt arm where the zeros live — the $\mathrm{Fix}(G)^\perp$ blindness of ex 103/116/120, now located on the dual-lever axis. Classical identities read through the lens; does not advance RH (G4 open, program paused at T-HP) (honest scope) |
| [149_p14_is_the_capacity_arm_operator.py](../examples/07_number_theory/149_p14_is_the_capacity_arm_operator.py) | Identifies the canonical TNFR-Riemann Hamiltonian P14 as EXACTLY the capacity-arm operator of the dual-lever — the structural reason it sees the primes while the pressure substrate is blind (closes the loop of ex 148). Every P14 node $(p,k)$ carries $\nu_f = k\log p$ (CAPACITY, 20/20 exact) and $\Delta\mathrm{NFR}=0$ (PRESSURE neutral), so P14 puts all structural information on the capacity lever — the axis (log $=\nu_f$) carrying von Mangoldt + the zeros (ex 148). Inter-prime orthogonality (disconnected ladders, independent invariant subspaces) IS the Euler product at the operator level $=$ the free-monoid freedom (ex 147). The weighted trace reproduces $Z_{vM}(s)=-\zeta'/\zeta(s)$ to machine precision (certificate $\mathrm{overall\_ok}$), and the zeros are its poles. Unifies physics $\nu_f$-capacity $\leftrightarrow$ free-monoid size-grading $\leftrightarrow$ the prime-ladder Hamiltonian; no new operator, does not advance RH (G4 open, program paused at T-HP) (honest scope) |
| [153_structural_frequency_rank_cyclotomy.py](../examples/07_number_theory/153_structural_frequency_rank_cyclotomy.py) | The canonical structural-diffusion operator (the $\Delta\mathrm{NFR}$ EPI channel, `structural_diffusion_operator`) on arithmetic Cayley networks has a structural-frequency RANK that unifies three modules. (M1) TWO-ARM PRIMALITY: primality is a simultaneous fixed point of BOTH dual-lever arms — per-node pressure $\Delta\mathrm{NFR}(n)=0$ (§4) AND global spectral rank $s_{QR}(m)=3$ (ex 119), 0 disagreements; both GROW with factorization complexity (corr$(\Delta\mathrm{NFR},\log A)=0.93$), bridging §4 $\leftrightarrow$ ex 119. (M2) CYCLOTOMY LAW: $s_k(p)=\gcd(k,p-1)+1$ (measured, 0 fails $k\le10$, $p<60$); the maximal rank $k+1$ is reached $\iff p\equiv1\pmod k \iff p$ splits completely in $\mathbb{Q}(\zeta_k)$ (0 mismatches); QR is the $k=2$ case. (M3) FREE-MONOID EXPONENTIAL GRADING: on squarefree $m$ the rank is (per-prime rank)$^\omega$ — QR $3^\omega$, unitary/Ramanujan $2^\omega$ — the EXPONENTIAL reading of the word length $\omega$ (ex 147) whose pressure counterpart is the LINEAR $\zeta(\omega-1)$. Underneath = classical Gauss periods/cyclotomy; NEW = the unified TNFR structural-diffusion framing; no open problem advanced (honest scope) |

### 12.3 Test Coverage

| Test area | Location |
|-----------|----------|
| Primality validation (10k range) | `primality-test/test_installation.py` |
| Arithmetic network construction | `tests/` (number_theory tests) |
| Factorization spectral decoder | `factorization-lab/tests/test_spectral_paley.py` |
| Factorization verification | `factorization-lab/tests/test_verification_robustness.py` |
| Riemann operator spectral | `tests/` (riemann tests) |

---

## 13. Open Questions and Research Directions

### 13.1 Computational

- **Sub-$O(\sqrt{n})$ primality**: Can spectral methods on arithmetic networks detect primes faster than trial division?
- **Sieve optimization**: Can the TNFR pressure landscape guide more efficient sieve algorithms?
- **Large-number factorization**: Scaling the spectral Paley-Jacobi method to numbers beyond current computational limits.

### 13.2 Theoretical

- **Conjecture 10.1 (TNFR-Riemann bridge)**: Does the spectral determinant of $H^{(k)}_{\mathrm{TNFR}}$ analytically continue to $\zeta(s)$?
- **Pressure distribution**: What is the exact probability distribution of $\Delta\mathrm{NFR}(n)$ for "random" composites?
- **Goldbach connection**: Can the additive decomposition of even numbers be formulated as a phase-matching problem ($|\phi_p + \phi_q - \phi_{2n}| \leq \Delta\phi_{\max}$)?
- **Arithmetic coherence length**: How does $\xi_C$ in the arithmetic network relate to the distribution of prime gaps?

### 13.3 Structural

- **Special prime families** (PARTIALLY ADDRESSED — [100_prime_families_orbits.py](../examples/07_number_theory/100_prime_families_orbits.py)): twin, cousin, sexy, Sophie Germain, safe, Cunningham, Mersenne, and constellation families are organized as **structured subsets of the zero-pressure fixed-point set** $Z=\{n\ge 2:\Delta\mathrm{NFR}(n)=0\}$ (the primes), carved out by three classes of arithmetic map: additive-gap level-sets ($S_g(p)=p+g$), affine-recurrence orbits ($T(p)=2p+1$: Sophie Germain, safe, Cunningham chains), and exponential-form images ($M(p)=2^p-1$: Mersenne). Detection/generation is exact via the verified $\Delta\mathrm{NFR}=0$ theorem; **infinitude conjectures** (twin-prime, Sophie Germain, Mersenne) remain OPEN — the same honest stance as Goldbach (§13.2). The witness pressure signatures (e.g. the twin witness $p+1$ divisible by 6) are faithful TNFR restatements of classical divisibility facts.
- **Arithmetic network as a coupled system** (MEASURED — [101_numbers_as_coupled_network.py](../examples/07_number_theory/101_numbers_as_coupled_network.py)): on the divisibility/GCD network the prime-factor count $\Omega(n)$ is a **common structural coordinate** that grades both the per-node arithmetic pressure $\Delta\mathrm{NFR}$ ($r(\Omega,\Delta\mathrm{NFR})\approx 0.94$) and the network-transport centrality ($r(\Omega,\deg)\approx 0.75$), so the two pictures are linked ($r(\Delta\mathrm{NFR},\deg)\approx 0.81$). Primes ($\Omega{=}1$, $\Delta\mathrm{NFR}{=}0$) form the **transport periphery** (≈ 0.18× the composite stationary mass, ≈ 2.4× effective resistance, and large primes $p>N/2$ are literally isolated). Honest scope: a **correspondence through $\Omega$, not a dynamical identity** — the per-node $\Delta\mathrm{NFR}$ is not the graph-diffusion Laplacian; the network is **not scale-free**; "primes peripheral" restates the classical $\gcd(p,m)>1\iff p\mid m$ in transport language.
- **The nodal flow on numbers** (MEASURED — [102_nodal_flow_primes_equilibria.py](../examples/07_number_theory/102_nodal_flow_primes_equilibria.py)): running the actual nodal equation $\partial\mathrm{EPI}/\partial t=\nu_f\Delta\mathrm{NFR}$ settles the §7.1 "primes attract composites" question. **Positive result**: primes are EXACTLY the equilibria of the arithmetic flow (the §4 theorem in motion — $\Delta\mathrm{NFR}{=}0\iff\partial\mathrm{EPI}/\partial t{=}0$; 34/34 primes frozen, composite drift is $\Omega$-graded $r\approx 0.93$). **Refinement of §7.1**: those equilibria are NOT attractors — they are marginal (no restoring force, $\Delta\mathrm{NFR}_{\mathrm{arith}}$ independent of EPI), and the canonical diffusion flow instead relaxes to the degree-weighted (composite) bulk, pulling primes UP toward composites (the opposite of "attract"). The §7.1 STATIC half (primes at low $\Phi_s$) is correct; the DYNAMICAL "attract" half is not realized — a dynamical extension of the Example 101 inversion.
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

**Version**: 0.0.3.3 | **Status**: Canonical | **Authority**: [AGENTS.md](../AGENTS.md)
