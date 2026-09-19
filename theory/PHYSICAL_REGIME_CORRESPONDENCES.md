# Physical Regime Correspondences

**Status**: Technical reference
**Version**: 0.0.3.5
**Date**: March 2026

---

## 1. Scope

This document records five scoped comparisons between TNFR observables and
separately declared physical models. Each comparison states its assumptions
and available evidence; proposed comparisons without a retained driver are
identified explicitly. The nodal equation
$\partial\mathrm{EPI}/\partial t = \nu_f\,\Delta\mathrm{NFR}(t)$ directly gives a
first-order structural drift law. It does not, by itself, derive Newtonian,
quantum or thermodynamic dynamics. Those labels apply only to the adapters or
auxiliary models named below.

### Verification Status

| Comparison | Implementation | External reference | Test coverage | Status |
|------------|----------------|-------------------|---------------|--------|
| Classical adapter | `classical_mechanics.py` | Harmonic and circular-orbit closure | `test_classical_mechanics.py` | Finite regression |
| Kinematic adapter | `classical_mechanics.py` | Two-train analytical | Embedded in example | Demonstrated |
| Finite spectral comparison | `quantum_mechanics.py` | Supplied mode labels | None | Legacy adapter; normalization unresolved |
| Fourier/interference comparison | No retained driver | Declared Fourier convention required | None | Proposed correspondence only |
| Coupled-oscillator thermal proxy | No retained driver | Newton cooling comparison proposed | None | Proposed correspondence only |

## 2. Classical Adapter and Overdamped Limit

### 2.1 Regime Conditions

A high-coherence structural snapshot requires small aggregate pressure and EPI
rate:

$$
\operatorname{mean}|\Delta\mathrm{NFR}| \ll 1, \qquad
\operatorname{mean}|d\mathrm{EPI}/dt| \ll 1.
$$

Under these measured assumptions the canonical read-out
$C(t)=1/(1+\operatorname{mean}|\Delta\mathrm{NFR}|+
\operatorname{mean}|d\mathrm{EPI}/dt|)$ is close to one. Phase spread and a
constant capacity may be assumptions of a selected mechanical comparison, but
neither quantity enters the constitutive kernel directly. A small phase
gradient alone does not establish high $C(t)$.

### 2.2 Observable Mapping

| Model quantity | TNFR comparison | Scope |
|----------------|-----------------|-------|
| Position $q$ | Selected spatial component of EPI | Adapter convention |
| Velocity $\dot q$ | Selected flow component | Adapter convention |
| Mobility | $\nu_f$ | Exact coefficient in the first-order nodal law |
| Inertial mass $m$ | Explicit adapter parameter | No identity $m=1/\nu_f$ follows from TNFR |
| Force slot $F$ | Optional $\Delta\mathrm{NFR}$ bridge | Must be declared and dimensionally specified |
| Potential comparison | $\Phi_s$ | Structural source aggregation, not mechanical potential energy |
| Action comparison | Optional phase accumulation | Mapper does not construct this bridge |

### 2.3 First-order law and second-order adapter

Reading an EPI coordinate as $q$ and pressure as $F$ gives the exact algebraic
form

$$
\dot q = \nu_f F,
$$

so $\nu_f$ is a mobility. Newtonian motion instead requires an additional state
and a separately chosen law,

$$
\dot q=v,\qquad \dot v=F/m.
$$

The classical examples implement this second-order adapter. Their inertial mass
is an adapter parameter; substituting $\nu_f=1/m$ is an optional dimensional
identification, not a consequence of the nodal equation. A damped graph wave has
a restricted pure-EPI diffusion limit under its own stated hypotheses. Neither
construction derives the full inertial regime from the isotropic auxiliary
substrate.

### 2.4 Force Interpretation

| Adapter mechanism | Directly available telemetry | Optional TNFR comparison |
|-------------------|------------------------------|--------------------------|
| Externally supplied pair force | $q$, $p$, force/acceleration and classical energy | $\Delta\mathrm{NFR}$ and $\Phi_s$ only after an explicit scalar pressure bridge |
| Configured damping | Adapter state and dissipation rate | $C(t)$ only after pressure and EPI-rate channels are materialized |
| Configured restoring force | Adapter state and force | Phase-gradient and curvature only from a separately constructed graph state |

These rows specify possible comparisons. The mapper and N-body solver do not
materialize canonical pressure, $C(t)$ or the tetrad, and the tetrad does not
generate the adapter's force law.

### 2.5 Integration Scheme

The Verlet/Yoshida integrators in `src/tnfr/dynamics/symplectic.py` preserve the
symplectic structure of the declared Hamiltonian map up to numerical error. This
property does not certify all TNFR structural invariants or any arbitrary
operator schedule.

Workflow:

1. Select and record the adapter, force law, units and integrator order.
2. Export $q$, $p$, force/acceleration and adapter energy with numerical
   tolerances.
3. If a graph comparison is required, declare how adapter state maps to scalar
   EPI and $\Delta\mathrm{NFR}$, materialize $d\mathrm{EPI}/dt$, and record its
   provenance before computing $C(t)$ or the tetrad.
4. Compare the adapter trajectory against its stated reference.

### 2.6 Validation

The focused tests check one-period harmonic and circular-orbit closure under
explicitly supplied force evaluators. The Kepler demonstration
(`examples/02_physics_regimes/12_classical_mechanics_demo.py`) plots a selected
inverse-square central-force trajectory. These finite checks validate the
declared force/integration paths at their tolerances; they do not derive gravity
from the nodal equation.

## 3. Zero-pressure and Kinematic Comparisons

### 3.1 Canonical zero-pressure statement

$$
\Delta\mathrm{NFR}=0 \quad\Rightarrow\quad
\partial\mathrm{EPI}/\partial t=0.
$$

This gives zero instantaneous unforced EPI rate. A fixed EPI trajectory needs
pressure to remain zero; changing phase, capacity or support can generate later
pressure. Constant
translation does not follow unless a separate kinematic adapter stores velocity
and advances an external position coordinate.

A reproducible zero-pressure experiment should record the pressure and EPI-rate
residuals, fixed/moving coordinate convention, operator schedule and structural
telemetry. Excluding named destabilizers is insufficient: any custom pressure
law or initial state can still carry nonzero pressure.

### 3.2 Two-train adapter

`examples/02_physics_regimes/15_train_crossing_demo.py` uses prescribed constant
velocities:

| Parameter | Train A | Train B |
|-----------|---------|---------|
| Initial position | $x=0$ km | $x=600$ km |
| Velocity | $+300$ km/h | $-250$ km/h |

Its analytical crossing is

$$
t_c=\frac{600}{300+250}\approx1.0909\ \mathrm{h},\qquad
x_c=300t_c\approx327.27\ \mathrm{km}.
$$

Numerical agreement checks the kinematic adapter. It does not turn the
zero-pressure fixed EPI chart into a derivation of Newton's first law.

## 4. Finite Spectral and Wave Correspondence

### 4.1 Scope

Finite graph or cavity operators possess discrete modes because their state
space and boundary-value problem are finite. Large wrapped phase gradients can
be useful stress diagnostics, but they do not by themselves select a quantum
model or cause quantization.

### 4.2 Observable Mapping

| Auxiliary-model quantity | TNFR comparison | Boundary |
|--------------------------|-----------------|----------|
| Complex wave field $\psi$ | $\Psi=K_\phi+iJ_\phi$ | Auxiliary geometric sector |
| Modal energy/frequency | Eigenvalue-derived model value | Not identical to nodal $\nu_f$ |
| Potential $V(x)$ | Declared function compared with $\Phi_s(x)$ | Matching must be specified |
| Mode index $n$ | Eigenmode or winding label | Depends on boundary operator |
| Damping/selection | Explicit dissipative adapter | IL/SHA labels alone do not define measurement collapse |

### 4.3 Mode-selection mechanism

A finite spectral experiment must specify its wave operator, boundary
conditions, initial field, normalization and damping rule. Those additional
assumptions produce standing modes and their eigenvalues. The nodal law can
supply structural telemetry along the experiment, but it does not supply the
wave equation, superposition principle or a measurement rule on its own.

### 4.4 Validation boundary

The retired example 13 adjusted an energy array toward an imposed phase
reticulum using a handwritten noisy update. It did not construct a cavity
operator or derive its spectrum; it is not retained as spectral evidence.

The legacy `QuantumMechanicsMapper` remains an auxiliary adapter with unresolved
conventions. `calculate_theoretical_levels(L)` currently returns `n**2` and
ignores `L`, despite its docstring's `n**2/(8*L**2)` formula. Its reverse mapper
constructs `abs(EPI)*exp(i*phase)` rather than inverting the separate assignment
of complex input to `(K_phi, J_phi)`. These are not validated inverse mappings
or a derived wave dynamics. A quantitative spectral comparison would need a
declared operator, normalization and dedicated tests before supplying evidence.

## 5. Fourier Width and Interference Correspondence

### 5.1 Fourier-width product

An analytical Fourier-width bound requires specified normalization, integrable
signals and a compatible second-moment definition. A finite sampled window
does not automatically inherit that bound. Every reported product must name
its width estimator, Fourier convention and sampling window; no fixed TNFR
constant has been derived by this comparison.

### 5.2 Two-path interference

A two-path auxiliary wave model can compare detector intensity with wrapped
phase separation. Constructive and destructive bands change phase-order and
phase-gradient diagnostics. They change structural $C(t)$ only when the model
also specifies how phase affects $\Delta\mathrm{NFR}$ or $d\mathrm{EPI}/dt$ and
the recorded channels confirm the resulting change. Phase separation is not a
direct argument of the constitutive $C(t)$ kernel.

### 5.3 Validation boundary

The retired example 14 combined a raw-array Fourier-width calculation with a
separately forced wave-grid model. Its magnitude-weighted, truncated frequency
read-out did not validate a stated analytical uncertainty bound; its wave
intensity was not canonical structural coherence. No quantitative validation
or canonical pressure-to-phase bridge is retained for this proposed comparison.

## 6. Coupled-oscillator Thermal Proxy

This section describes a proposed Kuramoto-style comparison with thermal labels
for selected diagnostics. The previously cited `17_thermodynamics_demo.py` is
not a retained entry point; example 17 concerns conservation instead. There is
no dedicated quantitative thermal validation here and no derivation of the
laws of thermodynamics from TNFR.

### 6.1 Declared proxy mapping

| Demonstration quantity | Chosen structural proxy | Status |
|------------------------|-------------------------|--------|
| Heat/noise | Phase dispersion $\sigma_\phi$ | Model convention |
| Temperature | Local phase-gradient variance | Model convention |
| Entropy | Separately defined monotone diagnostic | No identity $S\propto1/C(t)$ |
| Synchronized state | Small pairwise wrapped separation | Phase condition, not full structural equilibrium |

### 6.2 Pairwise U3 compatibility is not transitive

U3 checks each active pair independently. If A and B are each within
$\Delta\phi_{\max}$ of C, their mutual separation may reach
$2\Delta\phi_{\max}$ and fail the gate. Therefore a synchronized cluster needs
explicit pairwise or edgewise evidence; compatibility through a shared neighbor
is insufficient.

### 6.3 Balance boundary

Conservation of a structural current requires a stated symmetry and model. The
auxiliary symplectic substrate conserves its declared charges under its exact
flow. A general engine operator sequence, Kuramoto experiment or sum of
$J_\phi$ values has no automatic conservation law; its residual must be measured
along the actual trajectory.

### 6.4 Stochastic desynchronization boundary

Random perturbations can increase phase spread in a specified stochastic
schedule, while sufficiently strong coupling can also resynchronize the same
network without invoking IL or THOL. The direction of $C(t)$ depends on the
observed pressure and EPI-rate channels. No universal time arrow follows from
omitting named stabilizers or closure operators.

### 6.5 Cooling comparison

A proposed coffee-cup proxy would couple a central random-phase patch to an
outer aligned ring under a supplied oscillator law. A Newton-cooling comparison would fit

$$
T(t)=T_{\mathrm{env}}+(T_0-T_{\mathrm{env}})e^{-kt}
$$

to a predeclared temperature proxy and report residuals and uncertainty. The
comparison has no retained validated driver or extracted time constant;
exponential cooling remains a hypothesis outside the active research gate.

## 7. Regime Transition Summary

The comparisons can be organized by their declared pressure and phase
conditions. This table is a model index, not a phase diagram derived from TNFR:

| Comparison | $\Delta\mathrm{NFR}$ | $\lvert\nabla\phi\rvert$ | Primary telemetry | Declared model |
|--------|---------------------|----------------|-------------------|-------------------|
| Zero-pressure chart | $=0$ | Measured separately | EPI rate, $C(t)$ | Fixed EPI |
| Classical adapter | External force value; graph bridge optional | Optional low-spread regime | Adapter $q,p,F$; tetrad only through a declared bridge | Explicit $F=ma$ adapter |
| Thermal proxy | Model-dependent | Distributed | Phase variance, pressure/rate | Coupled oscillators |
| Finite spectral model | Model-dependent | Model-dependent | Eigenpairs, $\Psi$ | Declared boundary operator |
| Fourier-width model | Not required | Broadband | Width estimators | Declared Fourier convention |

These regimes organize several implemented TNFR read-outs and declared auxiliary
models. Only reductions with explicit hypotheses, such as fixed-graph EPI
diffusion, follow from the nodal equation. The classical, thermodynamic, and
quantum-labelled constructions are scoped correspondences or adapters; their
presence in one package does not derive those physical theories from the nodal
equation.

---

## 8. Implementation Reference

| Component | Location |
|-----------|----------|
| Classical mechanics mapper | `src/tnfr/physics/classical_mechanics.py` |
| Quantum mechanics module | `src/tnfr/physics/quantum_mechanics.py` |
| Symplectic integrators | `src/tnfr/dynamics/symplectic.py` |
| Structural field computation | `src/tnfr/physics/fields.py` |
| Central-force demonstration | `examples/02_physics_regimes/12_classical_mechanics_demo.py` |
| Two-train kinematics | `examples/02_physics_regimes/15_train_crossing_demo.py` |

---

## Implementation & Examples

### TNFR ↔ Classical Mechanics Dictionary

> Originally `docs/TNFR_CLASSICAL_MAPPING.md`. Consolidated here as the canonical mapping reference.

**Scope**: This is an adapter dictionary. `dynamics/nbody.py` assumes the
Newtonian potential, while `dynamics/nbody_tnfr.py` assumes a different
regularized phase-coupled pair law. Neither model derives Newtonian mechanics
from coherence or from the nodal equation.

| TNFR Quantity | Definition (TNFR) | Classical Analog | Notes |
|---------------|-------------------|------------------|-------|
| **EPI** | Canonical coherent form | Adapter payload for $q$ and $\dot q$ | The payload convention does not redefine EPI generally |
| **νf** | Reorganization rate (Hz_str) | Mobility in the overdamped projection | High νf → faster drift for fixed pressure |
| **ΔNFR** | Scalar structural pressure in the engine | Optional force/acceleration bridge | The Newtonian solver returns acceleration but does not materialize canonical graph pressure |
| **Φ_s** | Inverse-square ΔNFR accumulation | Potential-like readout | U6 monitors drift between declared snapshots; a well analogy adds no bound |
| **\|∇φ\|** | Local desynchronization | Optional stress comparison | No mechanical stress or force identity is implied |
| **K_φ** | Phase torsion read-out | Curvature comparison | Does not generate the adapter force |
| **ξ_C** | Static product-fit length or separate spectral fallback | Correlation-range comparison only on the fit branch | Does not set either N-body pair law; estimator provenance is required |
| **Ψ = K_φ + i·J_φ** | Complex auxiliary geometric field | Phase-space comparison | No Hamilton-Jacobi identity is established |
| **Operator sequences** | Canonical engine transformations | Work/impulse comparison | No equivalence between grammar validity and mechanical admissibility is established |

**Structural Triad ↔ Phase Space comparison**:
- Form (EPI) → configuration coordinate in the overdamped projection
- Frequency (νf) → mobility; inverse inertial mass is only an assignment in a
  separately specified second-order adapter or model
- Phase (φ/θ) → oscillator phase; action-angle status requires the auxiliary
  Hamiltonian construction

**Field Tetrad ↔ Energetics**:
- Φ_s → source-aggregation diagnostic; mean $|\Delta\Phi_s| < \pi/2$ is a
  selected before/after policy, not a potential-energy bound
- |∇φ| → local phase-stress comparison
- K_φ → phase-curvature comparison; no centripetal/Coriolis force follows from it
- ξ_C → fitted correlation-range comparison, with the dimensionless spectral
  fallback reported separately; the N-body force laws do not read it

The [tetrad guide](../docs/STRUCTURAL_FIELDS_TETRAD.md#coherence-length)
owns the fit, fallback, distance and sampling conventions. A comparison table
does not override those observational dependencies or provide a force law.

### Executable Demonstrations

| Example | Concept from this document |
|---------|---------------------------|
| [11_classical_limit_comparison.py](../examples/02_physics_regimes/11_classical_limit_comparison.py) | Finite comparison of Newtonian and declared phase-coupled N-body adapters |
| [12_classical_mechanics_demo.py](../examples/02_physics_regimes/12_classical_mechanics_demo.py) | Declared inverse-square central-force trajectory |
| [15_train_crossing_demo.py](../examples/02_physics_regimes/15_train_crossing_demo.py) | Prescribed constant-velocity kinematic adapter |

### Key Source Modules

- `src/tnfr/physics/classical_mechanics.py` — Explicit classical adapter and diagnostics
- `src/tnfr/physics/quantum_mechanics.py` — Legacy mappings and supplied mode labels; limitations in §4.4

---

## 9. References

- [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) — the structural-field tetrad
- [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) — U1–U6 derivations
- [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) — Conservation laws
- [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md) — Lagrangian formulation
- [GLOSSARY.md](GLOSSARY.md) — Operational definitions
