# Physical phenomena atlas and regime correspondences

**Status**: Technical reference
**Version source**: [pyproject.toml](../pyproject.toml)
**Reviewed**: 2026-09-26

---

## 1. Scope

This document records implemented graph-transport and auxiliary-wave results,
and separately declared classical adapters. Each comparison retains its
assumptions and evidence. The nodal equation
$\partial\mathrm{EPI}/\partial t = \nu_f\,\Delta\mathrm{NFR}(t)$ directly gives a
first-order structural drift law. It does not, by itself, derive Newtonian,
quantum or thermodynamic dynamics. Those labels apply only to the adapters or
auxiliary models named below. Retired supplied-target demonstrations and
unimplemented thermal proxies are not research routes or validation evidence.

This is also the single owner of the physical-emergence atlas: correspondence
cards, reuse and evidence boundaries. The
[execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md#physical-atlas-selection-and-dependencies)
alone selects work and its order. The atlas does not replace the generative
objective or establish that physical entities emerge from TNFR.

### Verification Status

| Comparison | Implementation | External reference | Test coverage | Status |
|------------|----------------|-------------------|---------------|--------|
| Classical adapter | `classical_mechanics.py` | Harmonic and circular-orbit closure | `test_classical_mechanics.py` | Finite regression |
| Kinematic adapter | `classical_mechanics.py` | Two-train analytical | Embedded in example | Demonstrated |
| Finite graph spectra and auxiliary waves | `structural_diffusion.py` | Declared graph generator and wave equation | `test_structural_diffusion.py` | Conditional algebra and finite numerical controls |
| Sampled ring modes and winding | `winding_certificates.py`, `structural_diffusion.py` | Declared cycle and phase field | `test_emergent_wave_particle_scope.py` | Aliasing, branch and spectral controls; no particle identification |

### Physical emergence atlas

The useful question is which known mechanisms admit a justified reduction,
controlled approximation or falsifiable comparison with a specified nodal law.
Writing a known vector field as `nu_f*DeltaNFR` by assignment establishes a
representation, not an independently derived mechanism. Replace intuitive
ratings such as "very high correspondence" with the evidence states below.
No entry currently has complete physical admission under the P2 protocol.

#### Six fields for each correspondence card

1. **Phenomenon:** precise regime and evidence status, rather than a whole
   physical theory claimed from one resemblance.
2. **Reference equation:** source, full state, constitutive assumptions, units,
   boundaries and scale/size limit. External equations are comparisons, not
   undeclared TNFR laws.
3. **Observables:** instruments or simulated observations, preparation, clock,
   uncertainty and accessible inputs. Identify hidden state explicitly.
4. **TNFR dictionary:** the plan's
   [definition/identification/evolution contract](research/FIVE_STAGE_EXECUTION_PLAN.md#variable-definition-identification-and-evolution),
   with each quantity marked primitive, derived, supplied or diagnostic.
5. **Reduction:** state/observation map and matching of the full vector field
   on the stated domain, or a controlled approximation with an error bound.
   In one declared clock, a closed observation `H` requires
   `DH(z) F(z)=F_ref(H(z))`; matching one trajectory or one sine term is weaker.
6. **Prediction and decision:** predeclared response, fair comparator, reserved
   evidence and rejection/abstention criterion. If the models are equivalent
   in scope, record "no discriminating prediction in this regime"; identify
   an additional justified premise before claiming predictive novelty.

Proofs and raw evidence stay in their existing owners. The table is an index
of cards and prerequisites, not thirteen independently active campaigns.

#### Current evidence map

| Phenomenon | Present TNFR evidence and reusable owner | Missing bridge or limiting condition |
| --- | --- | --- |
| Diffusion / relaxation | **Exact conditional reduction:** isolated EPI transport and its [stability theorem](TNFR_DIFFUSION_STABILITY_THEOREM.md); separate [TCLab exploratory response](research/TCLAB_EXPLORATORY_PROTOCOL.md) | Measurement/clock and constitutive admission remain open; driven thermal input is not unforced diffusion. |
| Synchronization / Kuramoto-type dynamics | **Exact conditional observed-phase reduction and mode selection** in [derived form phase](nodal/DERIVED_FORM_PHASE.md#unequal-capacity-mode-selection-and-a-finite-phase-only-discriminator); supplied primitive-phase models are separate | Physical observation/clock map remains open; the admitted passive capacity family locks generically while fading, with no positive-capacity-ratio threshold. |
| Josephson / superconductivity | **Constitutive comparison candidate:** sine interaction and phase-current readers are reusable algebra | Electrical current/voltage, gauge-invariant junction phase, storage/dissipation and a superconducting bridge are absent. |
| Dynamical phase transitions | **Open model-specific test:** existing [restoration/response-slope criteria](CAPACITY_LOCALIZATION_BALANCE.md#8-a-local-form-capacity-relation-dissipation-and-restoration-criteria) | Select a complete law, control parameter and order parameter; distinguish bifurcation, finite crossover and thermodynamic transition. |
| Kibble–Zurek | **Prerequisites missing:** phase, memory and winding readers may support a future admitted model | Establish critical relaxation/correlation scaling, quench protocol, order-parameter manifold and defects first. |
| Turing / reaction–diffusion | **Conditional mechanisms and obstructions reusable:** pure diffusion versus the same [multichannel capacity closure](CAPACITY_LOCALIZATION_BALANCE.md#8-a-local-form-capacity-relation-dissipation-and-restoration-criteria) | A justified local reaction/feedback law and a spatial-mode instability test; nonuniform appearance alone is insufficient. |
| Critical phenomena | **No identified physical critical observable:** [field scope](../docs/STRUCTURAL_FIELDS_TETRAD.md#coherence-length) and spectral relaxation remain useful controls | Physical connected correlations, ensemble/dynamics, finite-size protocol and uncertainties; no critical exponent follows from the name `xi_C`. |
| Vortices / topological defects | **Conditional discrete winding evidence:** [retention and branch loss](COUPLING_WINDING_PERSISTENCE.md#6-canonical-loss-branch-crossings-and-observation-limits) | Spatial/cell realization, interpolation, physical order parameter and defect core; graph winding alone is not a physical vortex. |
| Solitons / localized coherent structures | **Deferred candidate:** [conditional identity and recovery](NODAL_RESEARCH_STRATEGY.md#closure-audit-equations-events-and-identity) | A justified supporting law and the claimed propagation/stability or collision property; persistence alone does not identify a soliton. |
| BEC / superfluidity | **Deferred physical bridge:** classical phase observations are available | Quantum state/statistics, observables and constitutive bridge; classical phase order does not establish condensation or superfluidity. |
| Renormalization / universality | **Exact restricted coarse reductions and memory:** [scale bridge](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md), [elimination](DERIVED_EPI_MEMORY.md) | Declared scale/time/field transformations and effective-model flow/fixed points; REMESH and nesting are not automatically RG. |
| Emergent geometry | **Conditional inherited geometry:** the same scale and observation owners | Demonstrate realizability and retained dynamics; a tetrad readout does not derive a fundamental physical geometry. |
| Gravity / fundamental quantum theory | **Parked strong interpretation:** existing adapters retain their limited roles below | No derivation or physical admission; this atlas does not reopen particle, cosmology or substrate-origin campaigns. |

#### Interaction and musical analogies: questions, not identifications

Particle collisions motivate an **input-pattern to output-pattern** question:
which identities survive, transform, split or appear, with what measured
response and balances? Collider experiments actually reconstruct particle
products, momenta and energies through detectors; see the
[CMS detector description](https://cms.cern/detector).
TNFR currently has no physical collision energy, particle species, cross
sections or quantum outcome rule identified by its contact study. Its
[prepared contact and formation boundary](COHERENT_PATTERN_CONTACT.md)
can inform the mathematical interaction question without being called a
particle-collision explanation. This analogy opens no collider-data campaign
and does not change the workstation/ordinary-laboratory empirical scope.

Music suggests complementary observations: frequency relationships, beating,
phase locking, rhythm and the response to a perturbation. Acoustic structure
and perceived consonance require separate explanations. Experimental work
distinguishes harmonicity from beating in
[consonance perception](https://mcdermottlab.mit.edu/papers/Cousineau_McDermott_Peretz_2012_amusia_consonance.pdf),
while [cross-cultural measurements](https://mcdermottlab.mit.edu/papers/McDermott_etal_2016_consonance.pdf)
show that consonance preferences are not uniform across populations.
An appealing frequency ratio or musical pattern therefore does not establish
one universal structural preference or a shared particle-generating law.

For TNFR, reuse the existing modal, derived-phase and synchronization owners
before making an acoustic correspondence. A mode of a supplied Laplacian,
a damped form oscillation and a maintained audible tone have different
dynamics and measurement requirements. A future comparison needs a declared
output signal, clock and prediction; sonifying telemetry does not validate
the dynamics. These are supporting prompts for the sole research queue,
not two new experimental programmes or a reason to select pleasing results.

#### First card: derived and supplied phase dynamics

The reference comparison is a declared Kuramoto-type law
`theta_dot_i=omega_i+sum_j K_ij*sin(theta_j-theta_i)`. Coupling normalization,
frequency distribution, support and finite/continuum limit are part of that
model. Transition type is not universal: the cited
[frequency-dependent Kuramoto study](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.106.044310)
derives and numerically studies different transitions for specified choices.
It is not experimental validation of TNFR.

The existing coupled directed triangles supply a stronger starting point than
assigning a sine law. With equally oriented unit triangles, reciprocal unit
links between corresponding vertices, common fixed capacity `nu>0` and pure
EPI diffusion, the [exact reduction](nodal/DERIVED_FORM_PHASE.md#212-a-coupled-amplitude-and-phase-law-derived-from-fine-diffusion)
gives, for positive contrast amplitudes,

\[
\dot\psi_a=-\frac{\sqrt3\nu}{4}
 +\frac\nu2\frac{r_b}{r_a}\sin(\psi_b-\psi_a).
\]

Equal amplitudes form a restricted invariant family with
`delta_dot=-nu*sin(delta)`. This is a Kuramoto-type two-phase reduction in a
rotating frame, with coefficients inherited from the supplied fine graph.
General angular prediction needs the amplitude ratio; absolute form needs
amplitudes and means, and zero contrast needs Cartesian continuation. Total
contrast strictly decays. This provides an emergent observed interaction,
not sustained oscillators, a many-body threshold or primitive-phase identity.

| Mechanism | Actual role and reusable owner | Correspondence verdict |
| --- | --- | --- |
| Derived form phase | Regional contrasts observed from the fine diffusion generator; [existing controls](../tests/physics/test_coupled_directed_form_phase.py) | Exact conditional amplitude/phase reduction; the two observed phases close without amplitude information only on restricted invariant preparations. |
| Supplied sine evolution | [`propose_u3_gated_phase_step`](../src/tnfr/dynamics/phase_evolution.py); [P2 locking](FORCED_SUPPORT_BALANCE.md#26-conditional-phase-locking-and-form-restoration-on-fixed-p2), [K3 reduction](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#joint-evolving-phaseform-reduction-on-fixed-k3) | Capacity-as-angular-rate, gain, admitted-neighbor averaging and U3 gate are supplied premises; not unrestricted Kuramoto. |
| Native phase coordination | [Per-call circular-mean relaxation](FORCED_SUPPORT_BALANCE.md#34-native-runtime-admission-uses-relaxation-not-the-supplied-sine-clock) | Different execution law; it has no supplied physical `dt` or free angular advance. |
| Phase contribution to EPI pressure | Arg of a neighbor resultant, through [configured pressure](NODAL_PARAMETER_FOUNDATIONS.md#21-the-implemented-pressure-is-a-specified-relational-map) | Not the sine phase row or an electrical current. The [existing Arg/current discriminator](nodal/PRESSURE_CONSTITUTIVE_SCOPE.md) must not be repeated as a new discovery. |
| Named UM/RA events and cycle readouts | [Operator contracts](STRUCTURAL_OPERATORS.md) and [winding certificates](../src/tnfr/physics/winding_certificates.py) | U3 compatibility is admission; RA primarily blends EPI. A name or gate does not select an oscillator law. |

The observed quantities are phase separation, amplitude ratio and full contrast
budget, not automatically `C` or the tetrad. Physical instrument mapping is
open. The existing phase-only, radial-response and branch-loss witnesses are
negative controls; the [source-matching owner](nodal/PHASE_FORM_EXCHANGE.md)
defines the full-state obligation before feeding observed phase into pressure.
Exact agreement on the restricted reduction is an equivalence result, not
an exclusive physical prediction. Further contrasts require a stated new
premise and prospective response; this card does not authorize a large sweep.

The [completed unequal-capacity extension](nodal/DERIVED_FORM_PHASE.md#unequal-capacity-mode-selection-and-a-finite-phase-only-discriminator)
now provides that bounded contrast. For fixed positive capacity within each
triangle, generic contrast directions approach a predicted capacity-dependent
lag bounded by `atan(sqrt(3)/5)`; an exceptional fast eigenline is retained.
All amplitudes still decay. Starting with equal amplitudes does not justify
holding them equal: at relative phase `pi/2`, the full relative-phase
acceleration is `-5*(nu_0-nu_1)^2/8`, while the frozen-ratio approximation
predicts zero despite matching the initial relative angular velocity. The retained
finite control separates these predictions without fitting a phase gain.
This closes the first mathematical card. The
[conditional measurement contract](research/PHASE_AMPLITUDE_MEASUREMENT_PROTOCOL.md)
now specifies the spatial observation and uncertainty test, but admits none
of the existing public sources for this directed law. The subsequent
[six-study review](research/PASSIVE_TRANSPORT_PROTOCOL.md#directed-source-decision-2026-09-26)
also admits none; the exact-model physical test is parked. MOSTR is a deferred
three-tank transport lead, not evidence of this phase mechanism. The primary
[collective-emergence objective](EMERGENT_ONTOLOGY.md#research-target-physical-properties-of-collective-patterns)
does not require one-to-one sensor mappings for primitive EPI. Reciprocal scalar
RC/thermal dynamics cannot be identified with its complex contrast spectrum.
The independent physical bridge remains open. A synchronization threshold
or KZ interpretation would be unsupported for this particular family.

#### Corrections that govern later cards

**Josephson:** periodicity and exchange antisymmetry allow more than one
harmonic; they do not single out sine. The [cited junction review](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.76.411)
specifically discusses departures from sinusoidal current–phase behavior;
a [primary calculation](https://arxiv.org/abs/cond-mat/0504682) retains first and
second harmonics. A physical card must address voltage–phase evolution as well
as current–phase behavior and independent electrical units. Even the measured
[Josephson voltage/frequency relation](https://www.ptb.de/cms/en/ptb/fachabteilungen/abt2/abt2-josephson.html)
is additional physics, not implied by a graph sine term. Josephson is a
separate comparison, not a prerequisite for Kibble–Zurek.

**Correlation and relaxation:** current `xi_C` is an uncentered static
coherence-product fit with a separately tagged spectral fallback. Neither
branch is automatically a connected physical correlation length. Keep that
diagnostic's definition unchanged; a critical observable needs its own admitted
observation map. Likewise configured `C` is not automatically an order parameter.
Even common-capacity diffusion has modal relaxation times
`tau_k=1/(nu_f*lambda_k)` for positive modes, not simply `1/nu_f`.
The relevant full generator/Jacobian and observation determine relaxation in
other models. A finite plot or fitted power law does not establish universality;
the [critical-phenomena reference](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.71.S358)
provides physical context, not a TNFR bridge.

**Discrete winding:** on a declared cycle, the wrapped edge sum divided by
`2*pi` is invariant along continuous trajectories that avoid antipodal edges.
A branch crossing can change it with fixed support and well-defined node
phases. The [existing C8 counterexample](COUPLING_WINDING_PERSISTENCE.md#6-canonical-loss-branch-crossings-and-observation-limits)
already demonstrates this. Physical vortex charge needs an admitted spatial
order parameter and interpolation; net winding is not a count of all defects.
Retained winding, autonomous defect formation and NFR identity are separate.

**Turing:** in pure EPI diffusion `xdot=-diag(nu(x))*L_rw*x`, positive smooth
state-dependent mobility on fixed reciprocal nonnegative support does not
destabilize uniform form. At `x=a*1`, its derivative multiplies `L_rw*x=0`,
so the Jacobian is `-diag(nu(a*1))*L_rw`. Reuse the diffusion obstruction.
The existing multichannel `nu=g(x)` model additionally places capacity in
pressure and can have a destabilizing combined slope; it is a different
constitutive class, already classified in [capacity closure](CAPACITY_LOCALIZATION_BALANCE.md#8-a-local-form-capacity-relation-dissipation-and-restoration-criteria).
A Turing claim requires a stable local homogeneous system and a nonuniform
mode destabilized by transport, not merely contrast growth or threshold
crossing. [Turing's original model](https://fab.cba.mit.edu/classes/862.22/notes/computation/Turing-1952.pdf)
supplies coupled reaction and diffusion laws. Any new TNFR closure must justify
its counterpart before searching for patterns.

**Kibble–Zurek:** first establish the model's transition, order-parameter
manifold, spatial correlation and relaxation laws. Under the conventional
algebraic scaling assumptions and a linear quench, the freeze-out argument
gives `xi_hat proportional to tau_Q^(nu_crit/(1+z*nu_crit))`; defect density
additionally depends on codimension and the measurement/coarsening protocol.
Here `nu_crit` is a critical exponent, not capacity `nu_f`. These are conditional
reference relations from the [KZ review](https://arxiv.org/abs/1310.1600), not
TNFR predictions. Size, noise/preparation, sampling time and defect definition
must be frozen; a different transition class need not obey this power law.
The [2023 Ising-domain experiment](https://www.nature.com/articles/s41567-023-02112-5)
studies binary structural domains, not automatically circular-phase vortices.
Its public raw-data suitability remains unassessed. The cited
[2026 holographic disk study](https://www.nature.com/articles/s41467-026-69940-w)
is a numerical gravitational-dual model, not laboratory observations; it is
excluded from this project's experimental shortlist.

**Scale and stronger interpretations:** an RG card needs a coarse map,
space/time/field rescaling, controlled closure and a demonstrated effective-law
flow or fixed point. Existing memory and quotient results supply tools, not
that conclusion. Soliton, BEC and superfluid cards likewise need their defining
state and dynamical observables; the [BEC reference](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.71.463)
does not make a classical phase reader a quantum condensate. Retain these as
deferred entries under the same evidence contract.

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

Finite symmetric graph operators possess a discrete orthonormal eigenbasis.
The choice of graph and operator remains an input. Large wrapped phase gradients can
be useful stress diagnostics, but they do not by themselves select a quantum
model or cause quantization.

### 4.2 Observable Mapping

| Auxiliary-model quantity | TNFR comparison | Boundary |
|--------------------------|-----------------|----------|
| Complex wave field $\psi$ | Separate from $\Psi=K_\phi+iJ_\phi$ | A state map and dynamical bridge would be required |
| Modal energy/frequency | Eigenvalue-derived model value | Not identical to nodal $\nu_f$ |
| Potential $V(x)$ | Declared function compared with $\Phi_s(x)$ | Matching must be specified |
| Mode index $n$ | Eigenmode or winding label | Depends on boundary operator |
| Damping/selection | Explicit dissipative adapter | IL/SHA labels alone do not define measurement collapse |

### 4.3 Retained graph-mode calculation

For fixed reciprocal nonnegative conductance and common positive capacity,
the pure-EPI generator has modes `exp(-nu_f*lambda_k*t)`. The separately
supplied graph wave has frequencies `sqrt(lambda_k)`; this wave equation is
not the first-order nodal law. Reuse
[structural_diffusion.py](../src/tnfr/physics/structural_diffusion.py) and its
[stability scope](TNFR_DIFFUSION_STABILITY_THEOREM.md), rather than imposing an
array of desired levels. Boundary conditions, initial state, clock and damping
remain declared inputs.

On a unit cycle, the sampled phasor `exp(2*pi*i*k*j/n)` is a normalized
Laplacian eigenvector with eigenvalue `1-cos(2*pi*k/n)`. Sampling aliases k
modulo n; the winding reader separately checks the wrap branch and U3 domain.
This exact graph identity does not turn the phasor into the geometric field
`Psi`: at uniform phase the phasor is one while `Psi` is zero. The retained
[ring-mode controls](../tests/research/test_emergent_wave_particle_scope.py)
exercise this distinction without claiming particle creation or quantum duality.

### 4.4 Validation boundary

The retired example 13 adjusted an energy array toward an imposed phase
reticulum using a handwritten noisy update. It did not construct a cavity
operator or derive its spectrum; it is not retained as spectral evidence.

The supplied-target quantum mapper has also been retired: its level generator
ignored the declared length and its two state mappings were not inverses.
There is no replacement physical quantum API. Graph spectra, winding and
auxiliary wave calculations retain their own mathematical contracts; they
do not establish probability amplitudes, detector statistics or physical
energy levels.

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

## 6. Phase and balance boundaries

Phase dispersion, structural coherence and the field energies are distinct
read-outs. None is a physical temperature or thermodynamic entropy without
an independently justified measurement and evolution model. The former
thermal-proxy proposal had no retained validated driver and is not maintained
as a research route.

### 6.1 Pairwise U3 compatibility is not transitive

U3 checks each active pair independently. If A and B are each within
$\Delta\phi_{\max}$ of C, their mutual separation may reach
$2\Delta\phi_{\max}$ and fail the gate. Therefore a synchronized cluster needs
explicit pairwise or edgewise evidence; compatibility through a shared neighbor
is insufficient.

### 6.2 Balance boundary

Conservation of a structural current requires a stated symmetry and model. The
auxiliary symplectic substrate conserves its declared charges under its exact
flow. A general engine operator sequence, Kuramoto experiment or sum of
$J_\phi$ values has no automatic conservation law; its residual must be measured
along the actual trajectory.

### 6.3 Stochastic desynchronization boundary

Random perturbations can increase phase spread in a specified stochastic
schedule, while sufficiently strong coupling can also resynchronize the same
network without invoking IL or THOL. The direction of $C(t)$ depends on the
observed pressure and EPI-rate channels. No universal time arrow follows from
omitting named stabilizers or closure operators.

## 7. Regime Transition Summary

The comparisons can be organized by their declared pressure and phase
conditions. This table is a model index, not a phase diagram derived from TNFR:

| Comparison | $\Delta\mathrm{NFR}$ | $\lvert\nabla\phi\rvert$ | Primary telemetry | Declared model |
|--------|---------------------|----------------|-------------------|-------------------|
| Zero-pressure chart | $=0$ | Measured separately | EPI rate, $C(t)$ | Fixed EPI |
| Classical adapter | External force value; graph bridge optional | Optional low-spread regime | Adapter $q,p,F$; tetrad only through a declared bridge | Explicit $F=ma$ adapter |
| Finite spectral model | Model-dependent | Model-dependent | Eigenpairs and declared wave coordinates | Declared boundary operator |

These regimes organize several implemented TNFR read-outs and declared auxiliary
models. Only reductions with explicit hypotheses, such as fixed-graph EPI
diffusion under its pressure realization, follow from the stated nodal model.
Classical adapters and auxiliary waves supply extra laws. Their presence in
one package does not derive the corresponding physical theories.

---

## 8. Implementation Reference

| Component | Location |
|-----------|----------|
| Classical mechanics mapper | `src/tnfr/physics/classical_mechanics.py` |
| Graph transport and auxiliary waves | `src/tnfr/physics/structural_diffusion.py` |
| Declared-cycle winding | `src/tnfr/physics/winding_certificates.py` |
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
- `src/tnfr/physics/structural_diffusion.py` — Declared diffusion generators, spectra and auxiliary waves
- `src/tnfr/physics/winding_certificates.py` — Cycle, branch and winding evidence

---

## 9. References

- [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) — the structural-field tetrad
- [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) — U1–U6 contracts and mathematical scope
- [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) — Conservation laws
- [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md) — Lagrangian formulation
- [GLOSSARY.md](GLOSSARY.md) — Operational definitions
