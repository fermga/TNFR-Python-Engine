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

This document formalizes these observations, derives every constant from the Universal Tetrahedral Correspondence ($\varphi$, $\gamma$, $\pi$, $e$), and maps the theory to its implementations in the repository.

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
- $\alpha = 1/\varphi \approx 0.6180$ — factorization complexity weight
- $\beta = \gamma/(\pi + \gamma) \approx 0.1552$ — divisor complexity weight
- $\gamma_{\mathrm{epi}} = \gamma/\pi \approx 0.1837$ — abundance deviation weight

**Physical interpretation**: EPI(n) is the structural form of the number, analogous to the configuration of an oscillator. Primes have the simplest forms; highly composite numbers have the richest.

### 3.2 Frequency: $\nu_f(n)$

The reorganization capacity of a number measures how rapidly its structural form could evolve:

$$\nu_f(n) = \nu_0 \cdot \left(1 + \delta \cdot \frac{\tau(n)}{n} + \varepsilon \cdot \frac{\Omega(n)}{\ln(n)}\right)$$

where:
- $\nu_0 = (\varphi/\gamma)/\pi \approx 0.8923$ — base frequency
- $\delta = \gamma/(\varphi \cdot \pi) \approx 0.1136$ — divisor density modulation
- $\varepsilon = e^{-\pi} \approx 0.0432$ — factorization complexity modulation

**Physical interpretation**: $\nu_f$ is the capacity lever in the nodal equation. Numbers with rich divisor structures have slightly higher reorganization capacity, but this is irrelevant for primes because the pressure lever vanishes.

### 3.3 Pressure: $\Delta\mathrm{NFR}(n)$

The structural pressure equation is the central result of arithmetic TNFR:

$$\boxed{\Delta\mathrm{NFR}(n) = \zeta \cdot (\Omega(n) - 1) + \eta \cdot (\tau(n) - 2) + \theta \cdot \left(\frac{\sigma(n)}{n} - \left(1 + \frac{1}{n}\right)\right)}$$

where $\Omega(n)$ is the prime factor count with multiplicity, $\tau(n)$ the divisor count, and $\sigma(n)$ the divisor sum. The coefficients $(\zeta, \eta, \theta)$ are derived in §5.

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

### 5.1 The Universal Tetrahedral Correspondence

All arithmetic TNFR constants derive from four universal mathematical constants via the Universal Tetrahedral Correspondence:

| Constant | Value | Mathematical Role |
|----------|-------|-------------------|
| $\varphi$ (Golden Ratio) | $1.618034\ldots$ | Harmonic proportion |
| $\gamma$ (Euler-Mascheroni) | $0.577216\ldots$ | Harmonic growth rate |
| $\pi$ | $3.141593\ldots$ | Geometric relations |
| $e$ (Euler number) | $2.718282\ldots$ | Exponential base |

### 5.2 Pressure Coefficients

The three coefficients in the $\Delta\mathrm{NFR}$ equation are:

$$\zeta = \varphi \times \gamma \approx 0.9340$$

**Derivation**: Links the golden ratio (optimal harmonic proportion in self-similar structures) with the Euler constant (growth rate of harmonic series $H_n \sim \ln n + \gamma$). Since $\Omega(n)$ counts prime factors — an additive function that grows like $\ln \ln n$ on average by the Erdős-Kac theorem — the natural weight combines the harmonic proportion constant with the harmonic growth rate constant.

$$\eta = \frac{\gamma}{\varphi} \times \pi \approx 1.1207$$

**Derivation**: The ratio $\gamma/\varphi \approx 0.3567$ is the Kuramoto coupling ratio in TNFR units (phase gradient threshold normalized by harmonic proportion). Multiplying by $\pi$ (the geometric constant governing circular phase space) gives the natural weight for divisor count $\tau(n)$, which measures the geometric complexity of the divisor lattice.

$$\theta = \frac{1}{\varphi} = \varphi - 1 \approx 0.6180$$

**Derivation**: The reciprocal of the golden ratio is the unique positive number satisfying $\theta^2 + \theta = 1$, the recursive self-similarity equation. This weights the abundance ratio $\sigma(n)/n$, which measures how the divisor sum scales relative to the number itself — a ratio whose deviation from the prime value $(1 + 1/n)$ captures the self-similar structure of the divisor lattice.

### 5.3 EPI Parameters

| Parameter | Expression | Value | Physical meaning |
|-----------|------------|-------|-----------------|
| $\alpha$ | $1/\varphi$ | $\approx 0.6180$ | Factorization complexity weight |
| $\beta$ | $\gamma/(\pi+\gamma)$ | $\approx 0.1552$ | Divisor logarithmic weight |
| $\gamma_{\mathrm{epi}}$ | $\gamma/\pi$ | $\approx 0.1837$ | Abundance deviation weight |

### 5.4 Frequency Parameters

| Parameter | Expression | Value | Physical meaning |
|-----------|------------|-------|-----------------|
| $\nu_0$ | $(\varphi/\gamma)/\pi$ | $\approx 0.8923$ | Base structural frequency |
| $\delta$ | $\gamma/(\varphi\pi)$ | $\approx 0.1136$ | Divisor density modulation |
| $\varepsilon$ | $e^{-\pi}$ | $\approx 0.0432$ | Factorization modulation |

### 5.5 Detection Thresholds

| Threshold | Expression | Value | Purpose |
|-----------|------------|-------|---------|
| Structural significance | $\gamma/(e\pi)$ | $\approx 0.0676$ | Minimum $\Delta\mathrm{NFR}$ for structural relevance |
| Primality tolerance | $10^{-10}$ | $10^{-10}$ | Floating-point zero detection |
| 2× structural | $2\gamma/(e\pi)$ | $\approx 0.1352$ | Extended significance band |

### 5.6 Derivation Status

The 9 dynamical arithmetic parameters (3 pressure + 3 EPI + 3 frequency) are written as expressions of $(\varphi, \gamma, \pi, e)$ from the Universal Tetrahedral Correspondence applied to arithmetic functions. The 3 arithmetic **thresholds** are *empirically recalibrated* values (e.g. $\Phi_s < 0.7452$ vs the general 0.7711), not closed-form derivations — the general per-node $\Phi_s$ threshold on which they are based is itself empirically validated without a derivation (see [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) §4.3). The "zero empirical fitting" characterization applies to the dynamical parameters, not to the thresholds.

---

## 6. Pressure Component Analysis

### 6.1 Three Independent Pressure Channels

The $\Delta\mathrm{NFR}$ equation decomposes structural pressure into three independent channels, each measuring a distinct aspect of compositeness:

#### Factorization Pressure: $P_{\Omega} = \zeta \cdot (\Omega(n) - 1)$

Measures the **total prime factor count with multiplicity**. This is the most direct measure of compositeness: primes have $\Omega = 1$, semiprimes have $\Omega = 2$, prime powers $p^k$ have $\Omega = k$.

| $n$ | Factorization | $\Omega(n)$ | $P_\Omega$ |
|-----|---------------|-------------|------------|
| 7 (prime) | $7$ | 1 | 0 |
| 15 | $3 \times 5$ | 2 | 0.934 |
| 8 | $2^3$ | 3 | 1.868 |
| 30 | $2 \times 3 \times 5$ | 3 | 1.868 |
| 360 | $2^3 \times 3^2 \times 5$ | 6 | 4.670 |

#### Divisor Pressure: $P_{\tau} = \eta \cdot (\tau(n) - 2)$

Measures the **richness of the divisor lattice**. Primes have exactly 2 divisors; highly composite numbers have many.

| $n$ | $\tau(n)$ | $P_\tau$ |
|-----|-----------|---------|
| 7 (prime) | 2 | 0 |
| 15 | 4 | 2.241 |
| 8 | 4 | 2.241 |
| 30 | 8 | 6.724 |
| 360 | 24 | 24.656 |

#### Abundance Pressure: $P_{\sigma} = \theta \cdot (\sigma(n)/n - (1+1/n))$

Measures the **deviation of the divisor sum ratio from the prime pattern**. This is the most sensitive to the internal structure of divisors.

| $n$ | $\sigma(n)/n$ | $1+1/n$ | $P_\sigma$ |
|-----|---------------|---------|-----------|
| 7 (prime) | $8/7 \approx 1.143$ | $8/7$ | 0 |
| 15 | $24/15 = 1.600$ | $16/15 \approx 1.067$ | 0.330 |
| 8 | $15/8 = 1.875$ | $9/8 = 1.125$ | 0.464 |
| 30 | $72/30 = 2.400$ | $31/30 \approx 1.033$ | 0.844 |

### 6.2 Component Independence

The three pressure channels are algebraically independent — no linear combination of two can reproduce the third for all $n$. This makes the decomposition **minimal and complete** for characterizing compositeness through the three canonical arithmetic functions ($\Omega$, $\tau$, $\sigma$).

### 6.3 Structural Pressure Landscape

As $n$ grows, the expected pressure for a "random" composite scales as:

$$\mathbb{E}[\Delta\mathrm{NFR}(n)] \sim \zeta \cdot \ln\ln n + \eta \cdot (\ln n)^{\ln 2} + \theta \cdot \text{(abundance deviation)}$$

by the Erdős-Kac theorem ($\Omega(n) \sim \ln\ln n$) and divisor function asymptotics. Primes remain at exactly zero regardless of magnitude.

---

## 7. The Arithmetic Tetrad

When the arithmetic network $G$ is constructed, the structural field tetrad ($\Phi_s$, $|\nabla\phi|$, $K_\phi$, $\xi_C$) can be computed using the centralized physics modules:

### 7.1 Structural Potential: $\Phi_s$

$$\Phi_s(n) = \sum_{m \neq n} \frac{\Delta\mathrm{NFR}(m)}{d(n, m)^2}$$

where $d(n, m)$ is the graph distance in the arithmetic network. Primes, being zero-pressure nodes, act as **sinks** in the potential field — they attract nearby composites toward equilibrium.

**Threshold**: $|\Phi_s| < 0.7711$ (empirically validated; no closed-form derivation — see [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) §4.3) in the general tetrad; arithmetic-specific recalibration gives $\Phi_s < 0.7452$.

### 7.2 Phase Gradient: $|\nabla\phi|$

$$|\nabla\phi|(n) = \frac{1}{|\mathcal{N}(n)|} \sum_{m \in \mathcal{N}(n)} |\phi_n - \phi_m|$$

where $\mathcal{N}(n)$ are the neighbors of $n$ in the arithmetic network. High phase gradient indicates local desynchronization — composites with many diverse factors show elevated gradients.

**Threshold**: $|\nabla\phi| < \gamma/\pi \approx 0.1837$ for stable operation; arithmetic recalibration gives $0.2591$.

### 7.3 Phase Curvature: $K_\phi$

$$K_\phi(n) = \text{wrap\_angle}\!\left(\phi_n - \overline{\phi}_{\mathcal{N}(n)}\right)$$

where $\overline{\phi}_{\mathcal{N}(n)}$ is the circular mean of neighbor phases. Elevated curvature flags numbers at structural boundaries — e.g., the transition between prime-rich and composite-rich regions.

**Threshold**: $|K_\phi| < 0.9\pi \approx 2.827$; arithmetic recalibration gives $3.2275$.

### 7.4 Coherence Length: $\xi_C$

$$C(r) \approx A \cdot e^{-r/\xi_C}$$

The coherence length measures how far structural correlations propagate through the arithmetic network. Near critical points (e.g., twin primes, prime gaps), $\xi_C$ diverges — a signature of long-range correlation in the prime distribution.

### 7.5 Arithmetic Recalibration

The arithmetic network has distinct topology from general TNFR networks (divisibility graphs are highly structured, not random). Arithmetic-specific tetrad thresholds are derived from validated experiments:

| Field | General threshold | Arithmetic threshold | Source |
|-------|-------------------|---------------------|--------|
| $\Phi_s$ | 0.7711 | 0.7452 | `PHI_S_THRESHOLD` |
| $|\nabla\phi|$ | 0.1837 | 0.2591 | `GRAD_PHI_THRESHOLD` |
| $K_\phi$ | 2.8274 | 3.2275 | `K_PHI_THRESHOLD` |
| $\xi_C$ | (topology-dependent) | (topology-dependent) | Computed per network |

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
| $\Phi_s$ delta | $\leq 0.35$ | Tetrahedral confinement (U6) |
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

**The structural reason for the mod-4 split.** For $n\equiv 1\pmod 4$, $-1$ is a quadratic residue, so the residue graph is **symmetric**: the canonical operator $L_{rw}=I-D^{-1}W$ is self-adjoint and its spectrum is **real**. For $n\equiv 3\pmod 4$, $-1$ is **not** a residue, so the residue digraph is a **Paley tournament** (one directed edge per pair); the canonical operator is non-normal and its spectrum is **complex** — the arithmetic content lives in the **phase** (the imaginary part), which the real spectrum discards.

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

---

## 11. Worked Examples

### 11.1 Prime Detection: $n = 17$

$$\Omega(17) = 1, \quad \tau(17) = 2, \quad \sigma(17) = 18$$

$$\Delta\mathrm{NFR}(17) = 0.9340 \times (1-1) + 1.1207 \times (2-2) + 0.6180 \times \left(\frac{18}{17} - \frac{18}{17}\right) = 0$$

Structural triad: $\mathrm{EPI}(17) \approx 1.73$, $\nu_f(17) \approx 0.90$, $C_{\text{local}} = 1.0$.

**Interpretation**: Zero pressure, perfect coherence, structural fixed point.

### 11.2 Semiprime: $n = 15 = 3 \times 5$

$$\Omega(15) = 2, \quad \tau(15) = 4, \quad \sigma(15) = 24$$

| Component | Calculation | Value |
|-----------|------------|-------|
| Factorization | $0.9340 \times (2-1)$ | 0.934 |
| Divisor | $1.1207 \times (4-2)$ | 2.241 |
| Abundance | $0.6180 \times (24/15 - 16/15)$ | 0.330 |
| **Total** | | **3.505** |

$C_{\text{local}} = 1/(1+3.505) \approx 0.222$.

### 11.3 Prime Power: $n = 8 = 2^3$

$$\Omega(8) = 3, \quad \tau(8) = 4, \quad \sigma(8) = 15$$

| Component | Calculation | Value |
|-----------|------------|-------|
| Factorization | $0.9340 \times (3-1)$ | 1.868 |
| Divisor | $1.1207 \times (4-2)$ | 2.241 |
| Abundance | $0.6180 \times (15/8 - 9/8)$ | 0.464 |
| **Total** | | **4.573** |

Using $\Omega$ (with multiplicity) rather than $\omega$ (distinct primes) gives prime powers a strong pressure signal: $2^3$ registers $\Omega = 3$, not $\omega = 1$.

### 11.4 Highly Composite: $n = 30 = 2 \times 3 \times 5$

$$\Omega(30) = 3, \quad \tau(30) = 8, \quad \sigma(30) = 72$$

| Component | Calculation | Value |
|-----------|------------|-------|
| Factorization | $0.9340 \times (3-1)$ | 1.868 |
| Divisor | $1.1207 \times (8-2)$ | 6.724 |
| Abundance | $0.6180 \times (72/30 - 31/30)$ | 0.845 |
| **Total** | | **9.437** |

Structural triad: $\mathrm{EPI}(30) \approx 3.43$, $\nu_f(30) \approx 0.95$, $C_{\text{local}} \approx 0.096$.

---

## 12. Implementation Map

### 12.1 Source Modules

| Module | Path | Scope |
|--------|------|-------|
| **Arithmetic network** | `src/tnfr/mathematics/number_theory.py` | `ArithmeticTNFRNetwork`, `ArithmeticTNFRFormalism`, `PrimeCertificate` |
| **Primality testing** | `primality-test/tnfr_primality/core.py` | Standalone ΔNFR computation, validation |
| **Canonical constants** | `primality-test/tnfr_primality/constants.py` | All (φ,γ,π,e)-derived coefficients |
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
| [31_mathematical_constants_basis.py](../examples/02_physics_regimes/31_mathematical_constants_basis.py) | Role of φ, γ, π, e |
| [40_arithmetic_number_theory.py](../examples/07_number_theory/40_arithmetic_number_theory.py) | Primality, triad, component analysis |
| [94_generative_number_construction.py](../examples/07_number_theory/94_generative_number_construction.py) | Compositional generation from prime atoms; U5 fractality; grammar certification |
| [95_primes_from_spectral_waves.py](../examples/07_number_theory/95_primes_from_spectral_waves.py) | Prime staircase ψ(x) as spectral-wave superposition; spectral coherence ⟺ RH (honest scope) |
| [96_spectral_vibration_of_coherence.py](../examples/07_number_theory/96_spectral_vibration_of_coherence.py) | Oscillatory residue S(T) as prime-ladder vibration {k·log p}; why aggregate C(t) is blind (honest scope) |
| [97_goldbach_additive_multiplicative.py](../examples/07_number_theory/97_goldbach_additive_multiplicative.py) | Goldbach phase-matching: negative structural result; additive/multiplicative orthogonality; B2/B3 ontological note |
| [100_prime_families_orbits.py](../examples/07_number_theory/100_prime_families_orbits.py) | Special prime families (twin, cousin, sexy, Sophie Germain, safe, Cunningham, Mersenne, constellations) as orbits and level-sets of arithmetic maps on the zero-pressure fixed-point set $Z=\{\Delta\mathrm{NFR}=0\}$; three generator classes; detection exact, infinitude open (honest scope) |
| [101_numbers_as_coupled_network.py](../examples/07_number_theory/101_numbers_as_coupled_network.py) | Numbers as a coupled TNFR network: $\Omega(n)$ grades both the per-node pressure $\Delta\mathrm{NFR}$ and the divisibility/GCD transport centrality ($r\approx 0.8$–$0.9$); primes ($\Omega{=}1$, $\Delta\mathrm{NFR}{=}0$) are the transport periphery, large primes isolated; correspondence-through-$\Omega$ not identity, not scale-free (honest scope) |
| [102_nodal_flow_primes_equilibria.py](../examples/07_number_theory/102_nodal_flow_primes_equilibria.py) | The actual nodal flow $\partial\mathrm{EPI}/\partial t=\nu_f\Delta\mathrm{NFR}$ on numbers: primes are EXACTLY the equilibria (§4 theorem in motion, frozen) while composites drift $\Omega$-graded; refines §7.1 — primes are static low-$\Phi_s$ sinks but NOT dynamical attractors (diffusion flow pulls primes UP toward the composite bulk) |

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
- [MATHEMATICAL_DYNAMICS_BASIS.md](MATHEMATICAL_DYNAMICS_BASIS.md) — Role of (φ, γ, π, e)
- [GLOSSARY.md](GLOSSARY.md) — Operational definitions

### External

- Hardy, G.H. & Wright, E.M. — *An Introduction to the Theory of Numbers* (arithmetic functions)
- Erdős, P. & Kac, M. — "The Gaussian Law of Errors in the Theory of Additive Number Theoretic Functions" (1940)
- Kuramoto, Y. — *Chemical Oscillations, Waves, and Turbulence* (phase synchronization)

---

**Version**: 0.0.3.3 | **Status**: Canonical | **Authority**: [AGENTS.md](../AGENTS.md)
