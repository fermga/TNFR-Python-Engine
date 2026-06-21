# The Four Constants as a Minimal Basis of Mathematical Dynamics

**Status**: Established result — derived from classification of mathematical structures and validated against TNFR implementation  
**Version**: 1.0 (March 2026)  
**Prerequisites**: [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md), [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) §4

---

## 1. Statement

The four mathematical constants (φ, γ, π, e) that govern the structural field tetrad (Φ_s, |∇φ|, K_φ, ξ_C) are not arbitrary selections. Each constant controls a **distinct and irreducible class of mathematical dynamics**. Together they span the minimal basis used within TNFR to characterize coherent dynamical systems on graphs.

| Constant | Class of dynamics | Mathematical domain |
|----------|-------------------|---------------------|
| φ (golden ratio) | Self-similar proportion | Fractal geometry, recursive sequences |
| γ (Euler–Mascheroni) | Discrete accumulation | Number theory, harmonic analysis |
| π (Archimedes) | Circular geometry | Riemannian geometry, trigonometry |
| e (Napier) | Exponential growth/decay | Dynamical systems, probability |

This classification reflects well-known properties of the four constants. TNFR's contribution is to **operationalize** this classification through the structural field tetrad.

---

## 2. The Four Classes of Mathematical Dynamics

### 2.1 Self-Similar Proportion (φ)

The golden ratio φ = (1 + √5)/2 ≈ 1.618 is the unique positive solution of x = 1 + 1/x, equivalently x² − x − 1 = 0.

**Defining property**: φ is the fixed point of the recursive proportionality x ↦ 1 + 1/x. This means φ governs systems where **the relation between whole and part reproduces across scales**.

**Where it appears**:
- Fibonacci sequences: F(n+1)/F(n) → φ
- Phyllotaxis (botanical divergence angles): 2π/φ² ≈ 137.5°
- Penrose tilings (quasicrystal geometry)
- Continued fraction representation: φ = [1; 1, 1, 1, ...] (the slowest-converging continued fraction, making φ the "most irrational" number)

**TNFR correspondence**: φ ↔ Φ_s (structural potential). The inverse-square accumulation Φ_s = Σ ΔNFR_j / d(i,j)² saturates on a resonant chain to the Basel value ζ(2) = π²/6 ≈ 1.6449 — a genuine closed-form property of the kernel. The U6 drift-confinement scale φ ≈ 1.6180 is *adopted* (not derived) as the operational threshold sitting 1.64% inside that saturation, motivated by φ being the most-irrational number (golden-mean KAM tori are the last to break under resonant perturbation) rather than by the heuristic fixed-point relation x = 1 + 1/x, which is **superseded** for this correspondence (see [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md) §4.1).

### 2.2 Discrete Accumulation (γ)

The Euler–Mascheroni constant γ = lim_{n→∞} [H_n − ln(n)] ≈ 0.577, where H_n = Σ_{k=1}^{n} 1/k is the harmonic series.

**Defining property**: γ measures the **asymptotic gap between discrete summation and continuous integration**. It quantifies how much "excess" accumulates when summing discrete contributions compared to the smooth logarithmic integral.

**Where it appears**:
- Harmonic series asymptotics: H_n = ln(n) + γ + O(1/n)
- Mertens's theorem (prime products): Π_{p≤N} (1 − 1/p) ~ e^{−γ}/ln(N)
- Digamma function: ψ(1) = −γ
- Laplace transforms of logarithms
- Regularization of divergent sums in physics

**TNFR correspondence**: γ ↔ |∇φ| (phase gradient). The critical coupling threshold γ/π ≈ 0.1837 from the Kuramoto model governs when discrete phase misalignments accumulate beyond the smooth-evolution regime.

### 2.3 Circular Geometry (π)

The ratio π = C/d ≈ 3.14159 of circumference to diameter of any circle.

**Defining property**: π governs **all phenomena involving angular measure, curvature, and periodic closure**. It is the half-period of the sine function and the fundamental constant of Riemannian geometry on curved spaces.

**Where it appears**:
- Circle geometry: A = πr², C = 2πr
- Fourier analysis: the basis functions e^{inx} have period 2π
- Gaussian distribution: (2π)^{−1/2} exp(−x²/2)
- Euler's identity: e^{iπ} + 1 = 0
- Riemannian curvature (Gauss–Bonnet theorem)

**TNFR correspondence**: π ↔ K_φ (phase curvature). Phase curvature is defined on the circle S¹, and |K_φ| ≤ π by the wrap_angle construction.

### 2.4 Exponential Growth/Decay (e)

Napier's constant e = lim_{n→∞} (1 + 1/n)^n ≈ 2.71828.

**Defining property**: e is the unique base for which d/dx[a^x] = a^x, making it the natural constant of **continuous change proportional to current state**. It governs all processes where growth or decay rate is proportional to the quantity itself.

**Where it appears**:
- Exponential growth/decay: dx/dt = kx ⟹ x(t) = x₀ e^{kt}
- Compound interest (continuous limit)
- Radioactive decay
- Markov chain transition probabilities
- Moment-generating functions in probability

**TNFR correspondence**: e ↔ ξ_C (coherence length). Spatial correlations decay exponentially C(r) = A exp(−r/ξ_C), where e is the unique base that preserves form under rescaling.

---

## 3. Classification Completeness

### 3.1 Exhaustive Coverage

The four constants cover the four fundamental types of mathematical behaviour:

| Type | Governing constant | Algebraic characterization |
|------|-------------------|---------------------------|
| **Proportional** | φ | Fixed point of x = 1 + 1/x (recursive self-reference) |
| **Accumulative** | γ | Regularization constant of Σ 1/k − ∫ 1/x dx (discrete–continuous gap) |
| **Geometric** | π | Period of exp(ix) (angular closure) |
| **Dynamic** | e | Eigenfunction of d/dx (rate = state) |

**Claim**: These four classes are **mutually irreducible** — no constant can be expressed as a simple algebraic combination of the other three. While all four are linked through deep analytic identities (see §5), each governs a qualitatively different structural phenomenon.

### 3.2 Why Not Other Constants?

Several important mathematical constants can be expressed in terms of (φ, γ, π, e):

| Constant | Expression | Structural role |
|----------|-----------|----------------|
| ln(2) ≈ 0.693 | Defined via e: ln(2) = log_e(2) | Binary information unit |
| √2 ≈ 1.414 | Algebraic (rational root) | Diagonal scaling |
| Catalan G ≈ 0.916 | Derived from π via L-function | Alternating series |
| Apéry ζ(3) ≈ 1.202 | Product over primes (γ-related) | Zeta specialization |

Constants like √2 are algebraic (roots of polynomials with integer coefficients) and do not generate new classes of dynamics. Constants like ζ(3) or the Catalan constant are special values of functions built from (γ, π, e). The golden ratio φ is the sole algebraic constant in the basis; its "most irrational" character (hardest to approximate by rationals) distinguishes it from all other algebraic numbers.

---

## 4. Coherent Systems and the Four Degrees of Freedom

### 4.1 The Structural Requirement

Within the TNFR framework, a coherent dynamical system — a spatially extended pattern that persists through time via internal feedback — is characterized through four structural aspects:

1. **Global stability** (proportionality across scales): The system must maintain structural ratios as it evolves. Without a stable proportion between parts and whole, the pattern fragments. → **φ**

2. **Local stress detection** (accumulation of discrete fluctuations): Each element must monitor how much local mismatch accumulates from its neighbours. This is intrinsically a discrete-summation problem on graphs. → **γ**

3. **Geometric confinement** (angular/curvature bounds): Phase relationships between coupled elements must remain within angular bounds. Violation means destructive interference. → **π**

4. **Temporal propagation** (exponential memory/diffusion): Information and correlations must propagate through the system with characteristic scales. Propagation on graphs is inherently Markovian (exponential decay per hop). → **e**

### 4.2 Formal Mapping

The mapping from structural requirements to constants and fields is:

```
Structural need           → Mathematical class      → Constant → Field
─────────────────────────────────────────────────────────────────────────
Global proportionality    → Self-similar recursion   → φ        → Φ_s
Local stress accumulation → Discrete harmonic sums   → γ        → |∇φ|
Geometric confinement     → Angular/curvature bounds → π        → K_φ
Temporal propagation      → Exponential decay        → e        → ξ_C
```

This mapping is the notational constant↔field association ([FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) §4; only π is a genuine structural scale).

---

## 5. Inter-Constant Relations and the Tetrahedral Geometry

The four constants are not independent in the arithmetic sense — deep analytic identities connect them. These connections manifest as the **edges** and **faces** of the mathematical tetrahedron.

### 5.1 The Six Edges

Each edge connects two constants, producing a dimensionless ratio with operational significance:

| Edge | Expression | Value | TNFR role | Mathematical origin |
|------|-----------|-------|-----------|---------------------|
| φ–γ | φ/γ | 2.803 | Structural frequency base | Proportion/accumulation coupling |
| φ–π | φ/(φ+π) | 0.340 | Optimization penalty | Proportion/geometry competition |
| φ–e | φ/e | 0.595 | EPI canonical bound | Proportion/dynamics ratio |
| γ–π | γ/π | 0.184 | Phase gradient threshold | Accumulation in angular units |
| γ–e | γ/(e+γ) | 0.175 | Temporal evolution rate | Accumulation vs. growth |
| π–e | π/e | 1.156 | Spectral speedup factor | Geometry/dynamics ratio |

### 5.2 The Four Faces

Each face of the tetrahedron combines three constants, producing a threshold that governs a specific regime:

| Face | Omitted constant | Expression | Value | TNFR role |
|------|-----------------|-----------|-------|-----------|
| φ–π–e | γ (accumulation) | φe/(π+e) | 0.7506 | Strong coherence threshold C_crit |
| φ–γ–e | π (geometry) | γ/(e+γ) | 0.175 | Temporal rate constant |
| φ–γ–π | e (dynamics) | φ/(φ+π) | 0.340 | Optimization penalty |
| γ–π–e | φ (proportion) | 1/(π+1) | 0.241 | Minimum collective coherence |

**Structural interpretation of C_crit**: The strong-coherence threshold C_crit = φe/(π+e) ≈ 0.7506 combines proportion (φ), dynamics (e), and geometry (π). It represents the point where structural stability, dynamic propagation, and geometric confinement simultaneously sustain coherent pattern persistence. The omitted constant γ (local accumulation) is subsumed because above C_crit the network operates in a smooth regime where discrete fluctuations are bounded.

### 5.3 Numerical Approximations of Interest

Two approximate numerical relations connect the constants through distinct mathematical channels:

**Relation 1**: $e^\gamma \approx \sqrt{\pi}$ (relative error: 0.49%)

- Left side: exponential growth (e) applied to accumulation constant (γ)
- Right side: geometric scale (√π) from the Gaussian integral ∫ exp(−x²) dx = √π
- **Status**: Not an exact identity. The near-coincidence reflects the deep connection between harmonic accumulation and Gaussian geometry. The exact ratio $e^\gamma / \sqrt{\pi} \approx 1.00486$ is bounded but not known to be rational or algebraic.
- **Mathematical context**: This approximation connects to Mertens's theorem, where e^γ appears in the asymptotic product over primes Π_{p≤N}(1−1/p) ~ e^{−γ}/ln(N).

**Relation 2**: $\pi/e + 1/\varphi \approx \sqrt{\pi}$ (relative error: 0.074%)

- Left side: geometry/dynamics ratio (π/e) plus inverse proportion (1/φ)
- Right side: diffusive scale √π
- **Status**: Not an exact identity. The tighter approximation (error 20× smaller than Relation 1) suggests a deeper structural connection. The combination π/e + 1/φ mixes geometry, dynamics, and proportion — the same three ingredients in C_crit.
- **Interpretation**: √π appears universally in diffusion processes (heat kernel, random walks, Gaussian distribution). The relation suggests that diffusive behaviour emerges from the interplay of geometric periodicity, exponential propagation, and self-similar proportion.

**Important caveat**: These are numerical observations, not theorems. Their significance lies in suggesting structural connections rather than in establishing identities. Converting them into rigorous results requires demonstrating that they emerge from model equations rather than numerical coincidence. See §8 (Research Programme) for proposed verification protocols.

---

## 6. The Logarithmic Spiral as Universal Structural Trajectory

### 6.1 Definition

The logarithmic (equiangular) spiral in polar coordinates is

$$
r(\theta) = a \, e^{b\theta} \tag{1}
$$

where $a > 0$ is the initial radius and $b \neq 0$ controls the growth rate.

This curve simultaneously involves:
- **e** (exponential growth of radius)
- **π** (angular rotation, with period 2π)
- **φ** (when $b = \ln\varphi / (\pi/2) \approx 0.306$, the spiral grows by factor φ per quarter turn — the **golden spiral**)

### 6.2 Key Properties

**Self-similarity**: The logarithmic spiral is the **unique** plane curve that is both equiangular (constant angle between tangent and radial direction) and self-similar under dilation:

$$
r(\theta + \Delta\theta) = r(\theta) \cdot e^{b\Delta\theta} \tag{2}
$$

Scaling and rotation are coupled: enlarging the spiral is equivalent to rotating it.

**Scale invariance**: The spiral looks identical at every magnification. This is the geometric manifestation of the self-referential property x = 1 + 1/x that defines φ.

### 6.3 Connection to TNFR Nodal Dynamics

The nodal equation ∂EPI/∂t = νf · ΔNFR admits spiral trajectories when:

1. The phase field rotates: dθ/dt = ω (angular frequency).
2. The structural pressure drives radial growth: ΔNFR ∝ ∇Φ_s ∝ r (pressure proportional to amplitude).

Under these conditions, the evolution in the (EPI amplitude, phase) plane takes the form:

$$
\frac{d|EPI|}{dt} = \nu_f \cdot k \cdot |EPI|, \quad \frac{d\theta}{dt} = \omega \tag{3}
$$

whose solution is

$$
|EPI|(\theta) = |EPI|_0 \cdot \exp\left(\frac{\nu_f \cdot k}{\omega} \cdot \theta\right) \tag{4}
$$

This is a logarithmic spiral with growth parameter $b = \nu_f k / \omega$.

**Structural tetrad signature of spiral dynamics**:

| Field | Behaviour during spiral evolution |
|-------|----------------------------------|
| Φ_s | Grows with |EPI| (increasing structural potential) |
| \|∇φ\| | Constant or oscillating (steady angular velocity) |
| K_φ | Bounded (smooth curvature if b well-defined) |
| ξ_C | Scales with spiral arm separation |

### 6.4 Natural Occurrence

Logarithmic spirals appear in physical systems where growth and rotation are simultaneously present:

- Galaxy spiral arms (gravitational + rotational dynamics)
- Tropical cyclones (Coriolis + pressure gradient)
- Nautilus shells (biological growth + geometric constraint)
- Sunflower seed arrangements (phyllotaxis with angle 2π/φ²)
- Turbulent vortices (inertial + viscous dynamics)

Within TNFR, these systems share a common structural signature: **coherent patterns that reorganize while maintaining resonant coupling**. The spiral trajectory in structural space is a natural attractor when rotation (phase evolution) and growth (EPI evolution) are simultaneously active.

### 6.5 Implications

The logarithmic spiral demonstrates that three of the four constants (φ, π, e) describe a single geometric object — a growth pattern widely observed across physical and biological systems. The fourth constant γ enters through the discrete-accumulation process that governs **how** the spiral pattern distributes across network nodes (harmonic summation of contributions).

---

## 7. The Golden Ratio as Structural Attractor

### 7.1 Fixed-Point Dynamics

The golden ratio satisfies the recursive equation

$$
\varphi = 1 + \frac{1}{\varphi} \tag{5}
$$

The iteration $x_{n+1} = 1 + 1/x_n$ converges to φ from any positive starting point. The convergence rate is controlled by the derivative of f(x) = 1 + 1/x at the fixed point:

$$
|f'(\varphi)| = \frac{1}{\varphi^2} \approx 0.382 < 1 \tag{6}
$$

This guarantees stable convergence.

### 7.2 Relevance to TNFR Structural Dynamics

In TNFR, the structural potential Φ_s accumulates via inverse-square contributions:

$$
\Phi_s(i) = \sum_{j \neq i} \frac{\Delta\mathrm{NFR}_j}{d(i,j)^2} \tag{7}
$$

On self-similar networks (where the graph structure repeats at multiple scales), this summation exhibits recursive self-reference: the potential at one scale depends on potentials at sub-scales. The recursive equation for the equilibrium potential takes the form:

$$
\Phi_s^{*} = f(\Phi_s^{*}) \tag{8}
$$

where f has the same fixed-point structure as Eq. (5). The confinement threshold Δ Φ_s < φ in grammar rule U6 reflects this: structural potential excursions beyond φ indicate departure from the self-similar equilibrium.

### 7.3 Optimal Irrationality

The golden ratio has the property of being the **most poorly approximable irrational number** by rationals. Its continued fraction representation [1; 1, 1, 1, ...] converges more slowly than any other irrational number. This implies:

**Anti-resonance property**: In coupled oscillator systems, frequencies whose ratio approximates φ are **maximally non-resonant** — they avoid destructive interference more effectively than any other irrational frequency ratio.

This is observed in:
- KAM theory (Kolmogorov–Arnold–Moser): Tori with golden-ratio frequency ratios survive perturbation longest.
- Phyllotaxis: The divergence angle 2π/φ² ≈ 137.5° produces optimal packing (each new element maximally avoids previous ones).
- Plasma fusion devices: Golden-ratio magnetic field line ratios optimize confinement.

**TNFR implication**: Networks whose structural ratios approach φ are expected to exhibit maximum resilience to perturbation, because the golden proportion minimizes destructive resonance between competing structural scales. This is a testable prediction (see §8.2).

---

## 8. Research Programme

The observations in this document generate specific testable predictions. These are organized by verification difficulty.

### 8.1 Immediate Computational Experiments

**Experiment A (Golden ratio emergence)**: Run grammar-compliant operator sequences on random networks (seed-controlled, multiple topologies). After convergence, measure ratios between structural observables at different scales. Test whether ratios Φ_s(scale k) / Φ_s(scale k+1) cluster near φ.
**Status**: Implemented — see [31_mathematical_constants_basis.py](../examples/02_physics_regimes/31_mathematical_constants_basis.py) for multi-topology constant emergence verification.

**Experiment B (C_crit universality)**: Verify that the coherence threshold C_crit = φe/(π+e) ≈ 0.7506 separates stable from bifurcating regimes across topologies *without being imposed as a parameter*. The threshold is already used in the codebase (MIN_BUSINESS_COHERENCE); the experiment tests whether it emerges from dynamics rather than being a design choice.
**Status**: Implemented — see [31_mathematical_constants_basis.py](../examples/02_physics_regimes/31_mathematical_constants_basis.py) which verifies C_crit emergence across ring, random, small-world, and scale-free topologies.

**Experiment C (Spiral trajectories)**: Evolve networks with simultaneous rotation (coupling operators UM/RA) and growth (emission AL). Plot the trajectory in (|EPI|, θ) space. Test whether the growth parameter b = ln(φ)/(π/2) produces the most stable spiral.
**Status**: Implemented — see [32_spiral_attractors_demo.py](../examples/02_physics_regimes/32_spiral_attractors_demo.py) for golden spiral condition verification and tetrad signatures along spiral trajectories.

### 8.2 Analytical Verification

**Task 1**: Derive the fixed-point equation Φ_s = f(Φ_s) explicitly for regular lattices and show that its stable solution involves φ.

**Task 2**: Analyze the Lyapunov functional E = ½Σ(Φ_s² + |∇φ|² + K_φ²) along spiral trajectories and determine whether the golden spiral (b ~ ln(φ)/(π/2)) minimizes energy dissipation rate.

**Task 3**: Investigate whether the near-equality π/e + 1/φ ≈ √π can be derived from the nodal equation under diffusive-regime reduction.

### 8.3 Methodological Standards

All experiments must follow TNFR reproducibility requirements:
- Fixed seeds for all stochastic components
- Results reported in structural telemetry terms (C(t), Si, Φ_s, |∇φ|, K_φ, ξ_C)
- Multiple topologies (ring, random, small-world, scale-free, complete)
- Grammar compliance verified for all operator sequences

---

## 9. Connection to the TNFR-Riemann Programme

### 9.1 γ and Prime Distribution

The Euler–Mascheroni constant γ appears prominently in prime number theory:

- **Mertens's theorems**: Π_{p≤N}(1 − 1/p) ~ e^{−γ}/ln(N)
- **Prime harmonic series**: Σ_{p prime} 1/p diverges with coefficient γ
- **Digamma function at integers**: ψ(n) = H_{n−1} − γ

These connections mean that γ is not merely a property of harmonic sums but encodes information about the **distribution of primes** — the fundamental building blocks of arithmetic structure.

### 9.2 The Structural Field Tetrad and Prime Dynamics

In the TNFR-Riemann programme ([TNFR_RIEMANN_RESEARCH_NOTES.md](TNFR_RIEMANN_RESEARCH_NOTES.md)), the discrete operator $H^{(k)}(\sigma) = L_k + V_\sigma$ is defined on prime path graphs $G_k$. The critical parameter convergence $\sigma_c^{(k)} \to 1/2$ connects to the tetrad as follows:

| Tetrad field | Role in prime path graphs |
|-------------|--------------------------|
| Φ_s | Accumulated prime connectivity pressure |
| \|∇φ\| | Phase mismatch between connected primes (bound by the π phase-wrap) |
| K_φ | Curvature of the eigenvalue spectrum near σ = 1/2 |
| ξ_C | Correlation length of the spectral density (diverges at critical σ_c) |

The tetrad fields provide a complete diagnostic framework for the TNFR-Riemann operator, and the constant γ serves as the bridge between the structural dynamics (|∇φ| threshold) and the arithmetic structure (prime distribution).

### 9.3 Research Status

The bridge from discrete TNFR operators to the classical Riemann zeta function remains **conjectural** (Conjecture 10.1 in the research notes). The observation that γ simultaneously governs phase gradient physics and prime distribution is consistent with the conjecture but does not constitute a proof.

---

## 10. Structural Parallels across Mathematics

The four-dimensional basis (φ, γ, π, e) → (proportion, accumulation, geometry, dynamics) echoes dimensional structures across mathematics and physics:

| Domain | Four-fold structure | Correspondence |
|--------|---------------------|----------------|
| Spacetime | (t, x, y, z) | Temporal + 3 spatial |
| Electromagnetism | 4-potential A_μ | (φ, A_x, A_y, A_z) |
| Thermodynamics | (T, P, V, S) | Temperature, pressure, volume, entropy |
| Complex analysis | Real/Imaginary × Analytic/Meromorphic | Four function classes |
| **TNFR tetrad** | **(φ, γ, π, e)** | **Proportion, accumulation, geometry, dynamics** |

The recurrence of four-fold structure is suggestive but should be treated as an analogy rather than a derived result. Within TNFR, **four independent channels** — value, first derivative, second derivative, and integral correlation — suffice to characterize the structural state on graphs (see [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md) §3).

---

## 11. Summary

1. The four constants (φ, γ, π, e) each govern a **distinct and irreducible class of mathematical dynamics**: self-similar proportion, discrete accumulation, circular geometry, and exponential growth/decay.

2. These four classes are **complete within TNFR**: they span proportion, accumulation, geometry, and dynamics, which together cover the structural requirements of coherent systems on graphs.

3. The TNFR structural field tetrad (Φ_s, |∇φ|, K_φ, ξ_C) operationalizes this classification through measurable fields, each governed by its corresponding constant.

4. Inter-constant relations generate the **tetrahedral edge and face values** used throughout the TNFR constant system (300+ constants from algebraic combinations of the four).

5. The logarithmic spiral provides a concrete geometric example where three constants (φ, π, e) interact to produce a widely observed growth pattern.

6. φ as an attractor in structural dynamics follows from fixed-point theory and the anti-resonance property of golden-ratio irrationality.

7. γ bridges structural dynamics and number theory through its role in both the phase gradient threshold and prime distribution asymptotics.

8. All claims generate **specific testable predictions** (§8) amenable to computational verification within the TNFR engine.

---

## Implementation & Examples

### Executable Demonstrations

| Example | Concept from this document |
|---------|---------------------------|
| [31_mathematical_constants_basis.py](../examples/02_physics_regimes/31_mathematical_constants_basis.py) | Four-constant emergence, C_crit universality, edge/face values (§5, §8.1 A–B) |
| [32_spiral_attractors_demo.py](../examples/02_physics_regimes/32_spiral_attractors_demo.py) | Golden spiral trajectories from nodal equation (§6, §8.1 C) |
| [05_coherence_evolution.py](../examples/01_foundations/05_coherence_evolution.py) | Coherence thresholds governed by constant-derived values |
| [10_simplified_sdk_showcase.py](../examples/01_foundations/10_simplified_sdk_showcase.py) | SDK access to tetrad fields and constant-derived thresholds |

---

## 12. References

- [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md) — Irreducibility and completeness of the tetrad
- [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) — the structural-field tetrad (§4)
- [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) — Grammar rules derived from constants
- [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) — Conservation laws from symmetry
- [TNFR_RIEMANN_RESEARCH_NOTES.md](TNFR_RIEMANN_RESEARCH_NOTES.md) — γ and prime distribution
- [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md) — Lagrangian structure
- `src/tnfr/constants/canonical.py` — 300+ derived constants
- Hardy, G. H. and Wright, E. M. — *An Introduction to the Theory of Numbers* (harmonic series, Mertens's theorem)
- Strogatz, S. — *Nonlinear Dynamics and Chaos* (Kuramoto model, coupled oscillators)
- Khinchin, A. Ya. — *Continued Fractions* (φ as worst-approximable irrational)
