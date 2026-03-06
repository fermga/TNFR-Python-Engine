# Structural Stability and Dynamics

This document collects the stability analysis, phase transition theory, lifecycle dynamics, and integrity monitoring that emerge from the nodal equation $\partial\mathrm{EPI}/\partial t = \nu_f \cdot \Delta\mathrm{NFR}(t)$. Each section corresponds to a verified implementation in the codebase.

**Status**: CANONICAL — All results derived from the nodal equation and validated computationally.

---

## 1. Lyapunov Stability Analysis

### 1.1 Energy Functional

The structural energy functional serves as a Lyapunov candidate:

$$
E[G] = \frac{1}{2}\sum_i \left[\Phi_s(i)^2 + |\nabla\phi|(i)^2 + K_\phi(i)^2 + J_\phi(i)^2 + J_{\Delta\mathrm{NFR}}(i)^2\right]
$$

For grammar-compliant evolution: $dE/dt \le 0$ (Lyapunov stability).

### 1.2 Per-Operator Energy Bounds

Each canonical operator changes $E$ by a bounded amount $\Delta E$ whose sign and magnitude depend on the operator's glyph factor. The glyph factors are derived from the canonical constants $\varphi$, $\gamma$, $\pi$, $e$.

#### Stabilisers ($\Delta E \le 0$)

| Operator | Glyph factor | Contraction rate $\rho$ | Derivation |
|----------|-------------|------------------------|-----------|
| **IL** (Coherence) | $f = \varphi/(\varphi + \gamma) \approx 0.737$ | $\rho = 1 - f^2 \approx 0.457$ | $\Delta E = \frac{1}{2}(f^2 - 1)\sum\Delta\mathrm{NFR}^2$ |
| **EN** (Reception) | $m = 1/(\pi + 1) \approx 0.241$ | $\rho = m(1-m) \approx 0.183$ | Mixing contraction: $E' = m\,E_{\text{in}} + (1-m)\,E$ |
| **UM** (Coupling) | $\Delta\mathrm{NFR}_{\text{red}} = 0.15$ | $\rho \ge 0.15$ | Phase sync reduces local $\Delta\mathrm{NFR}$ by fraction |
| **THOL** (Self-org) | accel $= 0.10$ | $\rho \approx 0.10$ | Autopoietic organisation reduces net energy |
| **NAV** (Transition) | $\eta = 0.5$, jitter $= 0.05$ | $\rho \approx 0.499$ | Weighted interpolation: $E' = \eta\,E_t + (1-\eta)\,E$ |

#### Destabilisers ($\Delta E \le C_{\text{op}} \cdot E$)

| Operator | Glyph factor | Expansion rate $\kappa$ | Derivation |
|----------|-------------|----------------------|-----------|
| **OZ** (Dissonance) | $f = \varphi/\gamma \approx 2.803$ | $\kappa = f^2 - 1 \approx 6.857$ | $\Delta E = \frac{1}{2}(f^2 - 1)\sum\Delta\mathrm{NFR}^2$ |
| **VAL** (Expansion) | $f = 1 + \gamma/(\pi \cdot e) \approx 1.067$ | $\kappa = f^2 - 1 \approx 0.139$ | Dimensional scaling $E' = f^2 \cdot E$ |
| **AL** (Emission) | $b = 1/(\pi \cdot e) \approx 0.117$ | $\kappa = b^2 \approx 0.014$ per node | Creation from vacuum: $\Delta E = b^2$ |
| **RA** (Resonance) | $a = 0.05$ (amplification) | $\kappa = (1+a)^2 - 1 \approx 0.103$ | Amplification: $E' = (1+a)^2 \cdot E$ |

#### Neutral / Quasi-Isometric

| Operator | Bound | Note |
|----------|-------|------|
| **SHA** (Silence) | $|\Delta E| \le 0.187$ | Freezes $\nu_f$ via factor $(1 - \gamma/(\pi+e))$ |
| **ZHIR** (Mutation) | $|\Delta E| \le 0.056$ per node | Phase shift by $0.3 \cdot \Delta\mathrm{NFR}$ |
| **REMESH** (Recursivity) | $\Delta E = 0$ (exact) | $\alpha = 0.5$ weighted average preserves energy exactly |

#### Mixed

| Operator | Worst case | Best case |
|----------|-----------|-----------|
| **NUL** (Contraction) | $\kappa \approx 6.854$ (densification) | $\rho \approx 0.187$ (simple contraction) |

**Dual-Lever Interpretation**: The energy bounds above classify operators by
their *energy effect*, but the underlying mechanism is the **dual-lever
structure** of the nodal equation: each operator acts through the capacity
lever ($\nu_f$: UM, SHA, VAL, NUL), the pressure lever ($\Delta$NFR: IL, OZ,
THOL, ZHIR, NAV), both, or neither (AL, EN, RA, REMESH). The energy class
(stabiliser/destabiliser/neutral/mixed) is the net consequence of which
lever(s) the operator engages. See [STRUCTURAL_OPERATORS.md
§17.1](STRUCTURAL_OPERATORS.md) and [example 39](../examples/39_nodal_equation_decomposition.py).

### 1.3 Grammar U2 Lyapunov Theorem

Grammar rule U2 (CONVERGENCE & BOUNDEDNESS) requires that every destabiliser be accompanied by a stabiliser. The formal proof shows that the **net** energy change across a grammar-compliant sequence is non-positive:

$$
\sum_{\text{ops}} \Delta E_{\text{op}} \le 0 \quad\text{(for any U2-compliant sequence)}
$$

This confirms Lyapunov stability for the full 13-operator algebra.

**Refinement**: The formal bound $\sum \Delta E_{\text{op}} \le 0$ is
*sufficient* but not *necessary* for energy descent. Experimental
observation ([example 38](../examples/38_grammar_energy_landscape.py)) shows
grammar-compliant sequences with cumulative Lyapunov product $\Pi \approx
1.288$ (formally non-contractive) that still achieve net energy decrease
($\Delta E = -9.59$). The multiplicative bound is conservative because
operator interactions on the shared graph state are nonlinear.

### 1.4 Spectral Gap Characterisation

The algebraic connectivity $\lambda_1$ of the graph Laplacian controls the relaxation time:

| Quantity | Expression | Physical meaning |
|----------|-----------|------------------|
| Relaxation time | $\tau = 1/\lambda_1$ | Time for diffusive equilibration |
| Mixing time | $t_{\text{mix}} \sim \ln(N)/\lambda_1$ | Time to reach near-equilibrium |
| Cheeger bound | $h^2/(2d_{\max}) \le \lambda_1$ | Lower bound from expansion |

**Implementation**: `src/tnfr/physics/lyapunov.py` — `OperatorStabilityClass`, `OperatorEnergyBound`, `LyapunovPerOperator`, `analyze_spectral_gap()`.

---

## 2. Phase Transitions

### 2.1 Order Parameter

The symmetry breaking field $\mathcal{S}$ serves as the order parameter for phase transitions:

$$
\mathcal{S}(i) = \left(|\nabla\phi|^2 - K_\phi^2\right) + \left(J_\phi^2 - J_{\Delta\mathrm{NFR}}^2\right)
$$

### 2.2 Phase Classification

| Phase | Condition | Physical meaning |
|-------|-----------|------------------|
| **NON_LIFE** | $\langle\mathcal{S}\rangle < \gamma_c^2$ | Below noise floor; no emergent organisation |
| **CRITICAL** | $\gamma_c^2 \le \langle\mathcal{S}\rangle < \gamma_c$ | Near critical point; susceptibility diverges |
| **LIFE** | $\langle\mathcal{S}\rangle \ge \gamma_c$ | Emergent self-organisation |

### 2.3 Critical Exponent

The universal critical exponent emerges from the Euler constant / $\pi$ ratio:

$$
\gamma_c = \gamma/\pi \approx 0.1837
$$

Near the critical point:

$$
|\langle\mathcal{S}\rangle| \sim |p - p_c|^{\gamma_c}
$$

### 2.4 Constants

| Constant | Value | Derivation |
|----------|-------|-----------|
| Critical exponent $\gamma_c$ | $\gamma/\pi \approx 0.1837$ | Universal Tetrahedral Correspondence |
| Noise floor | $\gamma_c^2 \approx 0.0337$ | Squared critical exponent |
| Chirality threshold | $\gamma/(\pi + \gamma) \approx 0.155$ | Homochirality significance boundary |

### 2.5 Susceptibility

The structural susceptibility diverges at the critical point:

$$
\chi_{\mathcal{S}}(t) = N \cdot \operatorname{Var}(\mathcal{S})
$$

### 2.6 Critical Exponent Fitting

For systems near the transition, the critical exponent can be fit from the scaling law $|\langle\mathcal{S}\rangle| \sim |p - p_c|^{\gamma_{\text{fit}}}$. The theoretical prediction $\gamma_{\text{fit}} \to \gamma/\pi$ serves as validation.

**Implementation**: `src/tnfr/physics/phase_transition.py` — `Phase` enum, `PhaseTransitionTelemetry`, `PhaseSnapshot`, `compute_order_parameter()`, `classify_phase()`, `detect_phase_transition()`, `fit_critical_exponent()`.

---

## 3. Life and Autopoiesis

### 3.1 Autopoietic Coefficient

The autopoietic coefficient measures a system's capacity for self-generation relative to external driving:

$$
A(t) = \frac{\langle G(\mathrm{EPI}) \cdot \partial\mathrm{EPI}/\partial t\rangle}{\langle|\Delta\mathrm{NFR}_{\text{ext}}|^2\rangle}
$$

where the self-generation function follows logistic growth:

$$
G(\mathrm{EPI}) = \gamma\,\|\mathrm{EPI}\|\left(1 - \frac{\|\mathrm{EPI}\|}{\mathrm{EPI}_{\max}}\right)
$$

### 3.2 Life Threshold

$$
A(t) > 1.0 \implies \text{Life emergence}
$$

When $A > 1$, the system generates more structural change through self-organisation than through external forcing — the hallmark of autopoiesis.

### 3.3 Auxiliary Indices

| Index | Definition | Interpretation |
|-------|-----------|---------------|
| Vitality $V_i$ | $\gamma\,\|\mathrm{EPI}\|(1 - \|\mathrm{EPI}\|/\mathrm{EPI}_{\max})$ | Self-generation capacity |
| Self-Organisation $S$ | $\varepsilon\,|\partial G/\partial\|\mathrm{EPI}\|| / (|\partial\Delta\mathrm{NFR}_{\text{ext}}/\partial t| + \delta)$ | Sensitivity of self-generation to reorganisation |
| Stability Margin $M$ | $(\|\mathrm{EPI}\| - \mathrm{EPI}_{\max}/2)/\mathrm{EPI}_{\max}$ | Position relative to carrying capacity |

### 3.4 Life Emergence Detection

The life threshold time $t_{\text{life}}$ is found by interpolation at the $A(t) = 1.0$ crossing. The `LifeTelemetry` dataclass records the complete trajectory $(V_i(t), A(t), S(t), M(t))$.

**Implementation**: `src/tnfr/physics/life.py` — `detect_life_emergence()`, `LifeTelemetry` dataclass.

---

## 4. Node Lifecycle

### 4.1 Lifecycle States

Each TNFR node passes through a sequence of canonical states determined by its structural attributes:

| State | Condition | Physical meaning |
|-------|-----------|------------------|
| **DORMANT** | $\nu_f < \text{activation threshold}$ | Below activation energy |
| **ACTIVATION** | $\nu_f$ increasing, $\Delta\mathrm{NFR}$ growing | Energy accumulation |
| **STABILIZATION** | High $C(t)$, low $|\Delta\mathrm{NFR}|$ | Coherent attractor reached |
| **PROPAGATION** | High phase coupling | Pattern spreading via UM/RA |
| **MUTATION** | High $|\Delta\mathrm{NFR}|$, phase shifts | Qualitative state change |
| **COLLAPSING** | Losing coherence | Approaching dissolution |
| **COLLAPSED** | $\nu_f \to 0$, EPI dissolved | Terminal state |

Priority order for classification: mutation > propagation > stabilization > activation > dormant. Collapse is checked first.

### 4.2 Collapse Conditions

Four canonical collapse reasons, checked in priority order:

| Collapse reason | Condition | Physical basis |
|----------------|-----------|---------------|
| **Frequency failure** | $\nu_f < \text{collapse threshold}$ | Fundamental reorganisation capacity lost |
| **Extreme dissonance** | $|\Delta\mathrm{NFR}| > \text{bifurcation threshold}$ | Structural instability |
| **Network decoupling** | Phase coherence below minimum | Loss of resonance with neighbours |
| **EPI dissolution** | $\mathrm{EPI} \to 0$ | Form completely degraded |

### 4.3 Default Thresholds

| Parameter | Default | Source |
|-----------|---------|--------|
| Activation threshold | $0.1$ (min $\nu_f$) | Operational |
| Collapse threshold | $0.01$ (min $\nu_f$) | Operational |
| Bifurcation threshold | $10.0$ (max $|\Delta\mathrm{NFR}|$) | Operational |
| Stabilization $\Delta\mathrm{NFR}$ | $1.0$ | Operational |
| Stabilization coherence | $0.8$ | Operational |
| Propagation coupling | $0.7$ | Operational |
| Mutation $\Delta\mathrm{NFR}$ | $\varphi \times \pi \approx 5.083$ | Canonical |

**Implementation**: `src/tnfr/operators/lifecycle.py` — `LifecycleState`, `CollapseReason`, `get_lifecycle_state()`, `check_collapse_conditions()`.

---

## 5. Internal Hamiltonian Construction

### 5.1 Definition

The internal Hamiltonian governs structural evolution:

$$
\hat{H}_{\text{int}} = \hat{H}_{\text{coh}} + \hat{H}_{\text{freq}} + \hat{H}_{\text{coupling}}
$$

### 5.2 Components

**Coherence potential** (attractive interaction):

$$
\hat{H}_{\text{coh}} = -C_0 \sum_{i,j} w_{ij}\,|i\rangle\langle j|
$$

where $w_{ij}$ is the coherence weight from structural similarity and $C_0 = -1.0$ (attractive).

**Frequency operator** (diagonal):

$$
\hat{H}_{\text{freq}} = \sum_i \nu_{f,i}\,|i\rangle\langle i|
$$

Each node's $\nu_f$ becomes its diagonal energy.

**Coupling Hamiltonian** (topology):

$$
\hat{H}_{\text{coupling}} = J_0 \sum_{(i,j) \in E} \left(|i\rangle\langle j| + |j\rangle\langle i|\right)
$$

All components are $N \times N$ Hermitian matrices ($N$ = number of nodes).

### 5.3 Time Evolution

The unitary time evolution operator:

$$
U(t) = \exp\left(-i\,\hat{H}_{\text{int}}\,t\,/\,\hbar_{\text{str}}\right)
$$

Propagates states: $|\psi(t)\rangle = U(t)|\psi(0)\rangle$.

### 5.4 Energy Spectrum

The eigenvalue equation:

$$
\hat{H}_{\text{int}}|\phi_n\rangle = E_n|\phi_n\rangle
$$

gives stationary states $|\phi_n\rangle$ with energies $E_n$ (maximally stable configurations).

### 5.5 ΔNFR from Hamiltonian

The $\Delta\mathrm{NFR}$ operator follows from the Hamiltonian commutator:

$$
\Delta\mathrm{NFR} = \frac{i}{\hbar_{\text{str}}}\,\hat{H}_{\text{int}}
$$

Per-node: $\Delta\mathrm{NFR}_n = (i/\hbar_{\text{str}})\langle n|[\hat{H}_{\text{int}}, \rho_n]|n\rangle$ where $\rho_n = |n\rangle\langle n|$.

**Implementation**: `src/tnfr/operators/hamiltonian.py` — `InternalHamiltonian` class with `get_spectrum()`, `time_evolution_operator()`, `compute_delta_nfr_operator()`.

---

## 6. Structural Integrity Monitor

### 6.1 Purpose

The integrity monitor verifies **postconditions** of all 13 canonical operators after each application. This closes the loop between theoretical contracts and runtime behaviour.

### 6.2 Postconditions (13/13 Operators)

| Operator | Contract verified |
|----------|------------------|
| **AL** (Emission) | $\mathrm{EPI} > 0$ and $\nu_f$ increased |
| **EN** (Reception) | $C(t)$ not decreased |
| **IL** (Coherence) | $C(t)$ not decreased (outside dissonance test) |
| **OZ** (Dissonance) | $|\Delta\mathrm{NFR}|$ increased |
| **UM** (Coupling) | Phase compatibility $|\phi_i - \phi_j| \le \Delta\phi_{\max}$ |
| **RA** (Resonance) | EPI identity preserved; phase sync not decreased |
| **SHA** (Silence) | EPI unchanged; $\nu_f \to 0$ |
| **VAL** (Expansion) | $\dim(\mathrm{EPI})$ increased |
| **NUL** (Contraction) | $\dim(\mathrm{EPI})$ decreased |
| **THOL** (Self-org) | Global form preserved; sub-EPIs created |
| **ZHIR** (Mutation) | Phase $\theta$ changed when $\Delta\mathrm{EPI}/\Delta t > \xi$ |
| **NAV** (Transition) | Controlled trajectory; no coherence collapse |
| **REMESH** (Recursivity) | Nested structure maintained; parent identity preserved |

### 6.3 Monitor Modes

| Mode | Behaviour |
|------|-----------|
| **OFF** | No checking (production performance) |
| **OBSERVE** | Log violations without blocking |
| **ENFORCE** | Raise `StructuralIntegrityViolation` on failure |

### 6.4 Corrective Suggestions

When a violation is detected in OBSERVE or ENFORCE mode, the monitor provides corrective suggestions (e.g. "apply IL after OZ to restore convergence").

**Implementation**: `src/tnfr/physics/integrity.py` — `IntegrityReport`, `IntegritySummary`, `MonitorMode`, `StructuralIntegrityViolation`, `POSTCONDITIONS` registry.

**Tests**: `tests/test_integrity.py`

---

## Implementation Reference

| Module | Content |
|--------|---------|
| `src/tnfr/physics/lyapunov.py` | Per-operator energy bounds, spectral gap analysis |
| `src/tnfr/physics/phase_transition.py` | Order parameter, phase classification, critical exponent |
| `src/tnfr/physics/life.py` | Autopoietic coefficient, life emergence detection |
| `src/tnfr/operators/lifecycle.py` | Node states, collapse conditions |
| `src/tnfr/operators/hamiltonian.py` | Internal Hamiltonian, time evolution, spectrum |
| `src/tnfr/physics/integrity.py` | 13/13 postconditions, monitor modes |

---

## Implementation & Examples

### SDK Entry Points

```python
from tnfr.sdk import TNFR

net = TNFR.create(20).ring().evolve(5)
report = net.integrity_check()    # IntegrityReport (13/13 operators)
```

### Executable Demonstrations

| Example | Concept from this document |
|---------|---------------------------|
| [29_lyapunov_stability_demo.py](../examples/29_lyapunov_stability_demo.py) | All 13 operator Lyapunov bounds, energy class taxonomy, U2 net-contractivity proof, spectral gap, life/autopoiesis emergence |

### Key Source Modules

- `src/tnfr/physics/integrity.py` — Structural integrity monitor (13/13 operator postconditions)
- `src/tnfr/physics/conservation.py` — Energy functional (Lyapunov candidate)
- `src/tnfr/physics/phase_transition.py` — Phase transition detection
- `src/tnfr/operators/lifecycle.py` — Node lifecycle management

---

## Cross-References

- Lyapunov energy in conservation: [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) §8
- Grammar U2 (convergence): [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)
- Hamiltonian/Lagrangian formulation: [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md)
- Order parameter $\mathcal{S}$: [EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md](EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md) §3.2
- Dissipative extensions: [DISSIPATIVE_AND_OPEN_SYSTEMS.md](DISSIPATIVE_AND_OPEN_SYSTEMS.md)
- Gauge structure: [GAUGE_SYMMETRY_AND_UNIFICATION.md](GAUGE_SYMMETRY_AND_UNIFICATION.md)
