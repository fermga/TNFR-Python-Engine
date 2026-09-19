# TNFR Fundamental Theory

**Status**: Canonical reference
**Version**: 0.0.3.5
**Origin**: March 2026
**Scope review**: September 19, 2026

---

## 1. Scope

This document describes the nodal equation, canonical structural diagnostics,
operator policies, and explicitly specified model correspondences. The tetrad
is a canonical selection of diagnostic channels; complete state reconstruction
and universal minimality are not established. Mathematical hypotheses and
counterexamples are centralized in
[DIAGNOSTIC_AND_GRAMMAR_SCOPE.md](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md), which governs
the interpretation of grammar calibration and field thresholds here.


---

## 2. Governing Dynamics

### 2.1 Nodal Equation

On a continuous segment in a declared structural chart, nodal evolution satisfies

$$
\frac{\partial \mathrm{EPI}}{\partial t} = \nu_f(t) \, \Delta \mathrm{NFR}(t) \tag{1}
$$

where:

| Symbol | Definition | Units |
|--------|-----------|-------|
| EPI | Primary Information Structure — structural configuration in a declared chart | Declared form unit X |
| $\nu_f$ | Structural frequency — reorganization capacity | Hz_str |
| $\Delta\mathrm{NFR}$ | Nodal field response — directed structural pressure | X when capacity has inverse-time units |

### 2.2 Structural Triad

The engine's structural triad distinguishes three attributes:

1. **Form (EPI)**: coherent structural configuration in a declared state space. A Banach-space formulation and the engine's signed scalar chart are representations whose scope must be stated; the nodal equation alone does not choose their dimension. Named operators use their jump contracts; declared solvers use the shared nodal integrator.
2. **Frequency ($\nu_f$)**: nonnegative reorganization capacity; $\nu_f=0$ suppresses continuous EPI flow, without necessarily erasing stored form.
3. **Phase ($\phi$ or $\theta$)**: circular synchronization coordinate; coupling uses $|\mathrm{wrap}(\phi_i-\phi_j)|\leq\Delta\phi_{\max}$. The EPI equation alone supplies no phase clock.

### 2.3 Integrated Form and Stability Criterion

On one fixed vector chart without jumps, integrating Eq. (1) over $[t_0,t_f]$ gives:

$$
\mathrm{EPI}(t_f) = \mathrm{EPI}(t_0) + \int_{t_0}^{t_f} \nu_f(\tau) \, \Delta\mathrm{NFR}(\tau) \, d\tau \tag{2}
$$

On an infinite horizon, a sufficient condition for convergence is absolute
integrability of the structural velocity:

$$
\int_{t_0}^{\infty} \|\nu_f(\tau) \, \Delta\mathrm{NFR}(\tau)\| \, d\tau < \infty \tag{3}
$$

Boundedness only requires bounded partial integrals and does not imply (3) or
convergence to a limit. Local integrability of a supplied time-dependent
velocity defines its finite-horizon accumulated change; it does not prove
existence or uniqueness for an unspecified state-dependent law. See the
[integrability distinctions](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#1-existence-boundedness-and-convergence).
U2 prescribes stabilization and a two-unit prefix-debt policy;
it motivates control of accumulated change but does not independently prove
an infinite-horizon estimate. Its numerical calibration and the three-operation
U4b window use a mean-rate surrogate, not every graph mode's decay time.

For locally finite operator events, add their actual EPI jumps to (2).
Changing node support or structural chart requires an explicit identification
map. The integral is a Banach-valued integral when EPI is Banach-valued;
neither subtraction nor integration is defined by a bare set of forms alone.

### 2.4 Physical concepts, mathematical types and implementation

The research starting point is coherent form and its reorganization through
the nodal law. A historical formula or a Python type is evidence of a proposed
formalization, not an additional axiom that makes that formalization necessary.
The [joint closure review](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#14-constitutive-closure-audit-from-the-nodal-law)
uses the following distinctions before selecting any new dynamics.

| Quantity | Minimum mathematical meaning | What remains to be established |
|----------|------------------------------|--------------------------------|
| EPI, denoted x | A structural configuration with specified equivalence of forms and a chart or vector-space realization. | Which independent coordinates retain the intended structure and its dynamics; what observations identify them. |
| DeltaNFR, denoted p | A directed structural response in the tangent space of x, or its vector-chart representative. | A constitutive map from the joint nodal/network state to that response. A scalar distance alone gives no direction. |
| Capacity nu | A nonnegative multiplier converting structural response into a rate relative to the declared time. | Its independent state dependence, evolution and physical calibration. It is not automatically angular frequency. |
| Time t | A common ordered evolution parameter with explicit units and event conventions. | Its relation to laboratory time or a derived internal clock. A simulation step index is not such a derivation. |
| Phase theta | A point on the circle with an explicitly defined relation to structural configuration. | Whether it is independent, a derived coordinate or a history observable; and its evolution in that realization. |
| Support, conductance W and metric length | Distinct structural data specifying interaction, normalized transport and distance. | Which data emerge from form and which are separate state; admissible changes and their law. |

A scalar value `EPI=0` is not absence of the graph node. Its capacity, phase
and incident support still exist in the current engine and can affect other
nodes. Consequently a representation that discards phase at zero EPI needs
a projectability proof. Calling zero form a "vacuum" or invoking Emission
does not derive creation of the substrate or of its phase/capacity law.

If x lies in a differentiable manifold M, the type-correct statement is
`x_dot=nu*p`, with `p in T_x M`. In a Banach vector chart this tangent space
can be identified with the ambient Banach space. A metric space supplies
distances, but not by itself the tangent vector appearing in this equation.
If p denotes an operator instead of its evaluated vector, its argument and
action must be specified, for example `p=P(z)`, where z retains the relevant
joint state and history. Neither interpretation determines P uniquely.

The current graph engine supports a signed real scalar EPI coordinate and
its exact uniform-real `BEPIElement` embedding. The richer implementation
stores finite complex continuous-field samples and a finite coefficient
array; it is not an exact representation of every element of
`C^0([0,1]) direct-sum ell^2`. The supplied grid need not be [0,1], and no
interpolation or infinite tail is inferred. Its maximum-component scalar read-out loses
information outside the uniform-real embedding. Its historical `direct_sum`
method adds aligned components; it does not enlarge the state-space dimension.
Its regularity functional is not the canonical coherence or a Banach norm.
See [epi.py](../src/tnfr/mathematics/epi.py),
[spaces.py](../src/tnfr/mathematics/spaces.py) and
[the shared chart boundary](../src/tnfr/types.py).

Scalar numerical solvers use that same finite signed reader: uniform-real
BEPI is supported, while nonuniform or complex live/serialized forms are
rejected before solver-owned writes instead of being projected to a magnitude.
This is a representation contract, not a derivation of rich-EPI dynamics.
See [solver boundary controls](../tests/test_nodal_solver_epi_scope.py).

Consequently neither the existence of BEPI storage nor the use of floats
proves the physically necessary EPI type. Temporal spectral entropy of a
scalar series likewise proves neither a vector-valued state nor equivalence
to a complete spatial/modal representation. Adequacy requires a specified
observation/reduction map and closed dynamics or explicit missing-state
memory. The existing [quotient and memory derivations](DERIVED_EPI_MEMORY.md)
provide that test; storage labels alone do not.

A faithful chart of a full structural state and a closed scalar observable
are different claims. For an observation h and a specified full law z'=F(z),
autonomous observable evolution requires `Dh(z)F(z)` to agree for all states
with the same h(z); events additionally require `h(J(z))=J_bar(h(z))`.
An injective chart is sufficient to retain state but is not necessary for
such a closed observable. Capacity, phase, support and history remain inputs
unless their projected evolution also closes. These conditions test a proposed
representation; they do not supply the missing full law by assuming it.

### 2.5 Dimensional consistency and structural activity

Write `[x]=X`, `[t]=T`, `[nu]=T^-1`. The nodal product then requires `[p]=X`.
For normalized dimensionless x, p is dimensionless and x_dot has units T^-1.
This does not identify x with energy, probability or the diagnostic C(t).

The preserved [original source](TNFR.pdf), printed/physical pages 212-219,
contains incompatible alternatives that must not be silently combined:

- Pages 212 and 216 define DeltaNFR as a time derivative of internal structure.
  Page 213 instead uses a mean metric displacement between consecutive forms.
  The latter is nonnegative and cannot alone be a signed/tangent pressure:
  positive capacity would then prohibit decreasing scalar EPI, contrary to
  the explicit P2 diffusion control.
- Page 218 assigns both `[nu]=T^-1` and `[DeltaNFR]=X/T`. Their product is
  X/T^2, not X/T. The stated substitution `lambda=DeltaNFR/nu` leaves
  `nu^2*lambda` with the same wrong time dimension. It renames the expression
  without repairing the balance. An explicit normalization or a different
  constitutive convention is necessary; the source supplies no unique one.
- Pages 212-213 define capacity by counts of reorganizations per interval.
  For literal locally finite discrete events, the interval count divided by
  interval length tends to zero away from events as the interval shrinks to
  the observation time, and may be singular at an event.
  A smooth nonzero capacity requires a declared averaged intensity, continuum
  limit or independent continuous definition; the count formula alone is
  insufficient. Potentially possible events and realized events also differ.

The engine's directed pressure convention is dimensionally usable in its
normalized chart; it does not resolve the source's interpretation by fiat.
Adding phase, EPI, capacity and topology gradients requires compatible units
or explicitly normalized component scales. If capacities retain units T^-1,
the coefficient multiplying a capacity difference must supply the missing
X*T scale. Normalizing four numeric weights to sum to one does not establish
that physical normalization. This remains a constitutive/metrology obligation.

One useful consequence does follow without a new force. On a continuous
interval with positive capacity, define each node's accumulated activity

```text
s_i(t) = integral nu_i(tau) d tau,     dx_i/ds_i = p_i.
```

Thus pressure is change of structural form per unit accumulated activity in
this representation, not a second time derivative or an unsigned distance.
This is a reparameterization of a supplied trajectory, not a definition of
the unknown capacity law. Zero capacity makes the local clock singular;
heterogeneous nodes generally do not share one removable clock. See the
[existing derivation](FORCED_SUPPORT_BALANCE.md#23-capacity-exposure-does-not-determine-a-phase-clock).
The [Hz bridge](../src/tnfr/units.py) is an explicit conversion convention;
its default value does not independently calibrate laboratory time.

### 2.6 A structural coordinate must transform its pressure

For a nodewise differentiable invertible chart `y=f(x)` with the same time and capacity,
the chain rule requires `p_y=Df(x)*p_x`. Keeping only the written form
`y_dot=nu*p_y` does not justify reusing an unchanged coordinate formula for p.
For a chart mixing several nodes with unequal positive capacities, the
network formula instead is `p_y=D_nu^-1 Df D_nu p_x` if those same capacities
are retained. The simpler pushforward requires commutation with D_nu.

On unit P2 with x=(1,2), capacity one and pure EPI diffusion, p=(1,-1).
The positive-chart transformation y=x^2 requires p_y=(2,-4), whereas
recomputing the same Laplacian formula at y=(1,4) gives (3,-3). More generally,
preserving the pure scalar neighbor-difference formula on every pair requires
`f(b)-f(a)=f'(a)*(b-a)`, hence a common affine f. This conditional result
does not make the chosen affine chart physically unique. In the full pressure
`p=-e L_W x+F`, an affine rescaling y=a*x+b also needs F_y=a*F; unchanged
phase/capacity source coefficients do not automatically satisfy that.

The [portable foundation controls](../tests/physics/test_nodal_foundation_scope.py)
exercise the actual pressure owner, signed scalar embedding and lossy rich-EPI
projection. They do not select a new state space or evolve a new model.

---

### 2.7 All-channel parameter foundations

The [parameter foundation audit](NODAL_PARAMETER_FOUNDATIONS.md) extends these
types to capacity, phase, time, pressure coefficients, support, conductance,
distance, all four tetrad fields, diagnostics, energy, history and operators.
It owns the joint form/time covariance law and the conditional derivation of
a local diffusive generator. That derivation requires explicit locality,
linearity, shift, equilibrium and maximum-principle premises; reciprocity
and graph symmetry supply further restrictions rather than following from
the nodal product alone. Numeric defaults and telemetry remain distinct from
autonomous constitutive laws.

### 2.8 Representation dimension does not select a maintaining mechanism

Scalar-EPI exclusion results must retain their scalar-chart hypotheses.
They do not rule out every manifold-valued TNFR realization. However, merely
adding components to the same reciprocal diffusive pressure does not remove
its dissipation. On a common finite-dimensional real inner-product fiber,
let the rows of `X` be nodal forms, `B=D-W` for fixed symmetric nonnegative
conductance, and `M=diag(nu_i/d_i)` with zero mobility at isolates. The
componentwise extension of the declared pure-EPI model gives

\[
\dot X=-MBX,\qquad
E_D=\tfrac12\operatorname{tr}(X^TBX)
=\tfrac14\sum_{ij}W_{ij}\|X_i-X_j\|^2,
\]
\[
\dot E_D=-\operatorname{tr}((BX)^TM(BX))\le0.
\]

This follows by differentiating the quadratic and using symmetry of `B`;
it needs no assumption about the number of components. It is a conditional
mathematical extension, not an implemented rich-EPI engine law. A faithful
smooth chart change transports an existing Lyapunov function by composition
with its inverse. Consequently a richer chart alone cannot convert those
same dissipative trajectories into recurrent ones. A new connection,
nonlinear response, internal generator or coupled channel would be new
dynamics whose origin must be justified separately. Rich storage and
auxiliary Hilbert operators do not provide that justification by themselves.

### 2.9 Assumed substrate and emergence between scales

A primitive of one description is not thereby external to TNFR or fundamental
at every scale. The present engine takes a nodal state and its admitted laws
as inputs. A coarser nodal state could instead be an observation of finer TNFR
structure. The restricted [EPI quotient](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#1-pure-epi-coarse-graining)
already derives macro form, conductance and capacity from that structure.
This does not derive the existence of the fine substrate or the selection of
the observed partition. Conversely, taking a fine substrate as a premise does
not assert a substance outside the framework. Structural scale is not an
assertion of an earlier physical time.

For a declared fine state `Z` with a complete law `Z_dot=F(Z)`, let `Y=Pi(Z)`
be a proposed effective nodal description. On a smooth fixed-support domain,
an autonomous effective law `Y_dot=G(Y)` requires

\[
D\Pi(Z)F(Z)=G(\Pi(Z)).
\]

Thus two admitted fine states with the same `Y` must give the same observed
derivative. A time-dependent observation additionally contributes
`partial_t Pi`; an admitted event map `J` requires the separate relation
`Pi_after(J(Z))=J_bar(Pi_before(Z))`. Neither event occurrence nor a moving
partition follows from the continuous identity. Memory-dependent models must
retain the required history, not silently claim instantaneous closure.

Writing `Y_dot=nu_bar*p_bar` by defining pressure from the observed derivative
would not pass this test. The effective pressure, capacity, phase, support
and their laws must be constructed independently from the fine structure.
The [scale bridge](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md) owns the restricted
positive result and the phase, topology and field-inheritance obstructions;
[derived memory](DERIVED_EPI_MEMORY.md) supplies exact lost-state accounting
in its declared linear domain. A failed reduction can require an inherited
coordinate or memory rather than a new force. It does not, by itself, prove
that TNFR needs a different underlying theory. A claimed emergent NFR must
also establish formation and identity: closure of an analyst-selected
partition alone is insufficient.

The hypothesis that the present nodal description is an effective part of a
more complete TNFR formulation is therefore open and mathematically
testable in scoped models. It is not a demonstrated identification of all
physical entities with one substrate. Repeating a nodal description at a
finer scale also leaves that finer state/law as premises; it is not an
explanation of existence from no assumptions.

**Initial activity is not a sustained pulse.** Positive capacity with zero
pressure gives zero EPI rate. Nonzero `nu_i*p_i` at an initial state gives
activity, but need not give oscillation: pure-EPI diffusion provides active
relaxation. An auxiliary wave spectrum or a configured phase clock supplies
a different claim. The [pulse audit](NODAL_RESEARCH_STRATEGY.md#resonant-persistence-and-pulse-audit)
and [clock controls](../tests/physics/test_structural_clock_scope.py) keep these
distinctions executable. An assumed initial state, even an active one, does
not select the missing phase/capacity/support evolution or provide permanent
renewal. “Primordial pulse” is not an additional model variable or a proved
mechanism here.

## 3. Structural Field Tetrad

TNFR exposes four canonical diagnostic channels. They complement the structural
triad, frequency, and graph state; they do not determine the full dynamics.
Their computation and storage depend on the selected telemetry path.

### 3.1 Structural Potential ($\Phi_s$)

$$
\Phi_s(i) = \sum_{j \neq i} \frac{\Delta\mathrm{NFR}_j}{d(i,j)^2} \tag{4}
$$

Aggregates surrounding pressure using the selected inverse-square kernel.
Explicit edge `length` defines distance; absent length, `weight` is the
compatibility fallback, then unit length. The sum includes only reachable
distinct nodes at strictly positive finite distance; zero-distance pairs are
omitted under the shared read-out convention. U6 monitors its change under a declared policy; the
aggregation alone does not prove stability.

### 3.2 Phase Gradient ($|\nabla\phi|$)

$$
\lvert\nabla\phi\rvert(i) = \frac{1}{|\mathcal{N}(i)|}\sum_{j\in\mathcal{N}(i)} \big|\mathrm{wrap}(\theta_j - \theta_i)\big| \tag{5}
$$

Quantifies local desynchronization between a node and its neighborhood;
isolates return zero. A large value is a diagnostic observation, not a law
selecting a Coherence event.

### 3.3 Phase Curvature ($K_\phi$)

$$
K_\phi(i) = \mathrm{wrap\_angle}\big(\theta_i - \mathrm{circular\_mean}(\theta_{\mathcal{N}(i)})\big) \tag{6}
$$

Measures circular phase curvature, with $|K_\phi| \leq \pi$ by construction.
The displayed circular mean requires a nonzero resultant. The shared read-out
now distinguishes nonzero represented direction from exact represented
cancellation, with explicit unavailable evidence or a numeric API error; it
does not invent an arithmetic phase direction. Exact sums of retained phasor
components do not certify exact trigonometry. Curvature alone does not certify
a bifurcation. See the [API domain](../docs/STRUCTURAL_FIELDS_TETRAD.md#phase-curvature).

### 3.4 Coherence Length ($\xi_C$)

Estimated from uncentered products of static pressure-only coherence
$c_i=1/(1+|\Delta\mathrm{NFR}_i|)$, grouped by structural path distance:

$$
q(r)=\operatorname{mean}_{d(i,j)=r}(c_i c_j)\approx A \exp(-r / \xi_C) \tag{7}
$$

Both backends now share the distance/pair/fit definition. This is not connected
covariance and has no goodness-of-fit acceptance test. If unsuitable, fitting
yields to a spectral fallback: $1/\sqrt{\lambda_2}$ on a connected undirected
normalized Laplacian is a dimensionless mode scale, whereas a successful fit
has path-distance units. A large length is not a proof of criticality. See
[estimator scope](NODAL_PARAMETER_FOUNDATIONS.md#52-one-coherence-fit-definition-across-implementations).

### 3.5 Complex Geometric Field ($\Psi$)

Phase curvature and phase current unify into a single complex field:

$$
\Psi = K_\phi + i \cdot J_\phi \tag{8}
$$

This complex packaging retains the two real coordinates. Correlation measured
on particular trajectories does not reduce their algebraic degrees of freedom
or establish completeness of the field representation.

### 3.6 Emergent Invariants

The following algebraic read-outs combine the fields. Their conventional names
do not prove conservation, quantization or physical dimensional compatibility:

| Conventional diagnostic name | Definition | Supported interpretation |
|-----------|-----------|--------------|
| Energy density $\mathcal{E}$ | $\Phi_s^2 + \lvert\nabla\phi\rvert^2 + K_\phi^2 + J_\phi^2 + J_{\Delta\mathrm{NFR}}^2$ | Nonnegative diagnostic functional; decay needs evidence |
| Topological charge $\mathcal{Q}$ | $\lvert\nabla\phi\rvert \cdot J_\phi - K_\phi \cdot J_{\Delta\mathrm{NFR}}$ | Bilinear diagnostic; no general integer or conserved-charge theorem |
| Chirality $\chi$ | $\lvert\nabla\phi\rvert \cdot K_\phi - J_\phi \cdot J_{\Delta\mathrm{NFR}}$ | Signed channel contrast; geometric handedness needs an explicit symmetry action |
| Symmetry breaking $\mathcal{S}$ | $(\lvert\nabla\phi\rvert^2 - K_\phi^2) + (J_\phi^2 - J_{\Delta\mathrm{NFR}}^2)$ | Channel contrast; transition interpretation needs a protocol |
| Coherence coupling $\mathcal{C}$ | $\Phi_s \cdot \lvert\Psi\rvert$ | Product of potential and geometric-field magnitude; no cross-scale closure theorem |

---

## 4. The Structural-Field Tetrad

### 4.1 Statement

The four fields organize source aggregation, local phase mismatch, circular
curvature, and correlation. Higher derivative operators can be formed by
composition, but this does not prove that their outputs can be recovered from
four lossy diagnostics. Universal minimality and complete reconstruction remain
open under a specified state space and equivalence relation. π gives the exact
phase-wrap maximum; writing other values as π-fractions does not prove their
physical necessity.

| Field | Symbol | Operational limit | Structural scale |
|-------|--------|-------------------|------------------|
| Structural potential | $\Phi_s$ | Drift policy $\pi/2$; per-node policy $\pi/4$ | Pressure and graph-kernel dependent |
| Phase gradient | $\lvert\nabla\phi\rvert$ | Warning $\pi/16$ | Exact maximum $\pi$ |
| Phase curvature | $K_\phi$ | Warning $0.9\pi$ | Exact absolute maximum $\pi$; Laplacian only after linearization |
| Coherence length | $\xi_C$ | Configured length comparisons | Correlation fit or spectral reference $1/\sqrt{\lambda_2}$ on connected undirected graphs |

The phase bounds are kinematic identities. The warning margins and potential
thresholds are unchanged engine policies. Fitted correlation lengths need not
equal the graph-spectral reference for every state.

### 4.2 The four fields

The tetrad groups four selected diagnostic channels by their construction:

```text
        Φ_s (0th — global aggregation)
             /|\
            / | \
           /  |  \
  |∇φ| ------+------ K_φ
  (1st)      |  (2nd; π-bounded)
          \   |   /
           \  |  /
            \|/
          ξ_C (non-local — product fit / spectral fallback)
```

### 4.3 Derivation Outline

The detailed arguments have one owner in
[Minimal Structural Degrees, section 4](MINIMAL_STRUCTURAL_DEGREES.md#4-field-scales-and-selected-thresholds):

| Field | Established fact | Remaining distinction |
| --- | --- | --- |
| Potential | Fixed-kernel linearity and norm bound | Neither a universal pi-fraction bound nor unique selection of exponent 2 |
| Phase gradient | Mean absolute wrapped mismatch is at most pi | Warning margin and measured synchronization onset are separate |
| Phase curvature | Wrapped bound where the circular mean is available | Represented cancellation is an availability condition, not invented zero curvature |
| Coherence length | A static product fit with a separately identified spectral fallback | No universal decay law or criticality follows from either estimate |

This summary does not replace the linked graph, distance, representation and
estimator hypotheses.

### 4.4 Grammar Integration

Grammar obligations and field readouts have distinct enforcement paths:

| Rule | Primary fields | Enforcement |
|------|---------------|-------------|
| U1 (Initiation/Closure) | Context and endpoint roles | Supported generator and closure contracts |
| U2 (Convergence policy) | Operator roles and debt | Stabilizer presence and prefix debt at most 2 |
| U3 (Resonant Coupling) | Actual wrapped phase mismatch | Phase alignment verified before UM/RA |
| U4 (Bifurcation Control) | Operator context | Handlers, recent destabilizer, prior IL for Mutation |
| U5 (Multi-scale Coherence) | Declared hierarchy | Deep Recursivity requires nearby scale stabilization |
| U6 (Structural Confinement) | $\Phi_s$ | Read-only drift policy $\pi/2$ |

---

## 5. Core Structural Metrics

### 5.1 Total Coherence $C(t)$

The shared numeric kernel is $C(p,r)=1/(1+|p|+|r|)$; the network read-out
applies it to mean pressure and rate magnitudes. It differs from mean local
coherence and assumes declared input scales. Stored inputs need not describe
a fresh simultaneous nodal rate. The selected cuts $\pi/(\pi+1)$ and
$1/(\pi+1)$ label diagnostic bands; neither cut is a stability theorem.

### 5.2 Sense Index $Si$

Configured, clipped combination of relative capacity, phase dispersion and
relative pressure. Both numerical backends normalize against live maxima.
The score is telemetry; thresholds and its use in a selector or adaptation
loop are additional policies. It is not an independent nodal primitive or
proof of future stability. See the
[telemetry and time audit](NODAL_PARAMETER_FOUNDATIONS.md#6-telemetry-time-and-energy-are-not-interchangeable).

---

## 6. Multiscale Domain Mapping

Domain-correspondence studies use a specified pressure law and reduction
procedure. A derivation must state the assumptions at each step:

### 6.1 Reduction Procedure

1. **Declare the fine law**: Specify form coordinates, pressure dependencies,
   capacity, phase, support and time. A diffusive or solenoidal decomposition
   requires its own operator, metric and domain assumptions.
2. **Specify the observation**: Choose the coarse variables and prove that
   equal observations have equal projected rates, or retain the required
   hidden coordinates and memory. A graph quotient is not automatically a PDE.
3. **Separate flows and events**: Derive the actual effect of each supplied
   operator map. An EPI jump or stored-pressure edit is not automatically a
   continuous source or damping term.
4. **Define the read-outs**: Compute the tetrad where available and state which
   information it discards. Use the
   [scale and geometry contracts](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md) and
   [derived memory](DERIVED_EPI_MEMORY.md) rather than assuming tetrad closure.

### 6.2 Regime Summary

Model comparisons and research applications (each retains its own assumptions):

| Domain | Regime condition | Telemetry priorities | Governing reduction | Verification |
|--------|-----------------|---------------------|-------------------|-------------|
| Overdamped drift | Specified restoring pressure and frequency | Structural velocity and pressure | First-order mobility law; frequency is not inverse mass | Requires the stated pressure law |
| Instantaneous EPI stationarity | $\Delta\mathrm{NFR}=0$ | Structural velocity | Zero unforced EPI derivative; full equilibrium also needs the other state laws | Direct nodal identity |
| Discrete-mode analogy | Bounded graph with specified boundary conditions | Graph spectrum | Standing graph modes; no quantum-state identification | Spectral calculations |
| Spectral factorization | Specified Paley/residue graph construction | Graph-spectral and available field diagnostics | Partitioned periodicity detection under that construction | [Number-theory scope](TNFR_NUMBER_THEORY.md) |

### 6.3 Tetrad Requirements per Domain

Every domain study must report the four structural-field channels, including
explicit unavailability when its state or estimator does not support one:

- **$\Phi_s$**: Report distributions and gradients; compare against the selected $\pi/2 \approx 1.571$ drift policy with its baseline and aggregation.
- **$|\nabla\phi|$**: Monitor the heuristic early-warning level ($\approx \pi/16 \approx 0.196$; not derived — the kinematic bound is $\pi$).
- **$K_\phi$**: Report available curvature and the selected $0.9\pi$ warning;
  a crossing neither selects Mutation nor proves a transition.
- **$\xi_C$**: Report fit quality, path units and estimator/fallback provenance;
  a critical-scaling interpretation requires a separate finite-size protocol.

---

## 7. Emergent Geometry from the Nodal Equation

The following constructions have different coordinates and evolution laws.
Their certificates apply to those specified models; identifying them with the
full four-channel nodal dynamics requires an explicit mathematical bridge.

### 7.1 Transport Layer (Structural Diffusion)

For the pure EPI channel on a graph with the stated neighbor-weight convention,

    ΔNFR_epi = −L_rw EPI,
    EPI' = −diag(ν_f) L_rw EPI.

On a fixed undirected graph with positive homogeneous frequency, Laplacian
modes decay as exp(−ν_f λ_k t), and equilibrium is constant on each connected
component. Degree-weighted total EPI is conserved. For fixed positive
heterogeneous frequencies the invariant weights are degree_i/ν_f_i and modal
rates come from diag(ν_f)L_rw. Isolated nodes have no diffusive coupling.

The Dirichlet energy ½ EPIᵀ(D−W)EPI has degree-metric gradient L_rw EPI.
This supplies an exact gradient-flow identity for the isolated EPI channel.
It is a different functional from the sum of squared tetrad fields and does
not prove a variational identity for the full four-channel pressure.

**Implementation:** [structural_diffusion.py](../src/tnfr/physics/structural_diffusion.py)
and [variational.py](../src/tnfr/physics/variational.py).
The connected homogeneous formulas require their stated assumptions; the
spectral stability of nonstationary modes does not alone settle stationary
sources, nonlinear operator gains, or general U2 compliance.

### 7.2 Auxiliary Symplectic Substrate

The substrate implementation specifies an ambient phase space with pairs
(K_φ, J_φ) and (Φ_s, J_ΔNFR). Its isotropic Hamiltonian is

    H_sub = ½ Σ_i (K_φ² + J_φ² + Φ_s² + J_ΔNFR²),

with the phase-gradient term treated as a fixed background in the corresponding
readout. The canonical symplectic form and harmonic flow are well-defined on
these independent ambient coordinates. Hamiltonian flow preserves symplectic
form and phase volume; that theorem does not certify all 13 implemented
operator maps.

The isotropic model has its specified U(1)/U(2) invariances and oscillator
charges. Action-angle coordinates apply away from zero actions; singular
levels and global quotient topology need separate treatment. Extracting
coordinates from a graph can impose dependencies that an ambient-coordinate
certificate does not remove.

**Implementation:** [symplectic_substrate.py](../src/tnfr/physics/symplectic_substrate.py).
See its current certificate assumptions and
[the variational note](TNFR_VARIATIONAL_PRINCIPLE.md).

### 7.3 Orthogonal Structure and the Overdamped Projection

A separately specified damped graph wave with stiffness L_rw has an
overdamped diffusion limit under its damping, time-scale, and coordinate
assumptions. The isotropic substrate evolves with identity stiffness, so the
graph-wave calculation is not a derivation of the full nodal equation from
that substrate.

Likewise an orthogonal decomposition of selected graph currents is a result
in its specified graph metric. It does not establish a universal orthogonal
decomposition of every four-channel nonlinear evolution. The exact EPI
Dirichlet gradient flow, the graph-wave approximation, and the independent
harmonic substrate should be reported separately.

---
## 8. Empirical Validation

Field and operator experiments test defined protocols, with topology, weights,
initial state, gains, time steps, and seeds recorded. A correlation between
potential drift and coherence loss is evidence for that protocol, not a
universal upper bound or a proof of state reconstruction.

The exact definition-level facts are pressure linearity and the π phase-wrap
bounds. Potential policies π/4 and π/2, curvature margin 0.9π, and phase-gradient
warning π/16 retain their current values. The finite-graph witnesses and the
distinctions they require are recorded in
[DIAGNOSTIC_AND_GRAMMAR_SCOPE.md](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md).
A test count measures tested behavior; it does not establish an open theorem.

---
## 9. Practical Guidance

1. **Monitoring**: Export available tetrad fields and explicit failures after
   each operator batch. Threshold crossings are policy flags; any resulting
   action needs its own declared controller and grammar admission.
2. **Operator design**: Specify a proposed map's state changes, domain and
   postconditions, then verify its field response. A field interpretation does
   not establish a new canonical operator or its autonomous selection.
3. **Model calibration**: Compare like units and declared observation scales.
   Phase ratios such as $|\nabla\phi|/\pi$ are dimensionless; dividing
   dimensional $\Phi_s$ by the number $\pi/2$ alone is not. A potential ratio
   requires a reference with the same pressure/path units. See the
   [parameter foundations](NODAL_PARAMETER_FOUNDATIONS.md#3-joint-changes-of-form-and-time-units).
4. **Correlation diagnostics**: A large $\xi_C$ warrants checking fit quality,
   spectral fallback, and finite-size effects before interpreting a critical
   regime. Any subsequent operators must satisfy their grammar and contracts.

---

## 10. Implementation Reference

| Component | Location |
|-----------|----------|
| Structural field computation | `src/tnfr/physics/fields.py` |
| Grammar validation (U1–U5 context; U6 is a separate observer) | `src/tnfr/operators/grammar.py` |
| Structural balance diagnostics | `src/tnfr/physics/conservation.py` |
| Integrity monitor | `src/tnfr/physics/integrity.py` |
| Canonical constants | `src/tnfr/constants/canonical.py` |
| SDK access (tetrad, conservation) | `src/tnfr/sdk/simple.py` |
| Auxiliary symplectic substrate | `src/tnfr/physics/symplectic_substrate.py` |
| Structural diffusion (transport) | `src/tnfr/physics/structural_diffusion.py` |
| Test suite | `tests/` (current executable verification; counts are obtained from pytest) |

---

## 11. Implementation & Examples

### SDK Entry Points

```python
from tnfr.sdk import TNFR

net = TNFR.create(20).ring().evolve(5)    # Configured engine evolution
tetrad = net.tetrad()                      # Structural Field Tetrad
telem = net.telemetry()                    # C(t), Si, phase, νf
analysis = TNFR.analyze(net)               # Comprehensive analysis
```

### Executable Demonstrations

| Example | Concept from this document |
|---------|---------------------------|
| [01_hello_world.py](../examples/01_foundations/01_hello_world.py) | SDK initialization, one requested operator word and canonical C/tetrad observations |
| [04_operator_sequences.py](../examples/01_foundations/04_operator_sequences.py) | Flat grammar admission versus an independent illustrative pressure proxy |
| [07_phase_transitions.py](../examples/01_foundations/07_phase_transitions.py) | Prepared state ensembles and descriptive finite-size diagnostics |
| [10_simplified_sdk_showcase.py](../examples/01_foundations/10_simplified_sdk_showcase.py) | SDK topology builders, configured evolution and comparison read-outs |
| [99_structural_diffusion.py](../examples/08_emergent_geometry/99_structural_diffusion.py) | Declared diffusion and auxiliary graph-model correspondences |
| [179_phase_form_driven_response.py](../examples/08_emergent_geometry/179_phase_form_driven_response.py) | Prescribed phase contrast and derived EPI response; no autonomous maintenance claim |

### Key Source Modules

- `src/tnfr/physics/fields.py` — Structural Field Tetrad computation
- `src/tnfr/operators/definitions.py` — 13 canonical operator implementations
- `src/tnfr/operators/nodal_equation.py` — Nodal equation `∂EPI/∂t = νf·ΔNFR(t)`
- `src/tnfr/sdk/simple.py` — Simplified SDK with `TetradSnapshot`

---

## 12. References

- [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) — U1–U6 derivations
- [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md) — Four diagnostic channels and open reconstruction/minimality questions
- [DIAGNOSTIC_AND_GRAMMAR_SCOPE.md](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md) — Exact hypotheses, numerical policies, and finite-graph witnesses
- [MATHEMATICAL_DYNAMICS_BASIS.md](MATHEMATICAL_DYNAMICS_BASIS.md) — Broader derivative-tower context, read with the scope distinctions above
- [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) — Balance diagnostics and restricted exact conservation results
- [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md) — Lagrangian formulation
- [GLOSSARY.md](GLOSSARY.md) — Operational definitions
- [TNFR.pdf](TNFR.pdf) — Original theoretical derivations
- [AGENTS.md](../AGENTS.md) — Primary repository reference
