# Nodal parameter foundations and constitutive scope

Reviewed 2026-10-06. This reference owns the parameter ledger, units and
cross-channel dependencies. Detailed proofs have one topical owner in the
reading map below. The [execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone defines active research; these notes are not parallel task queues.

Form, tangent pressure and chart definitions belong to
[Fundamental Theory](FUNDAMENTAL_THEORY.md#24-physical-concepts-mathematical-types-and-implementation).
The [constitutive audit](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#14-constitutive-closure-audit-from-the-nodal-law)
owns the auxiliary-law inventory; [API contracts](../docs/API_CONTRACTS.md) and
the [field guide](../docs/STRUCTURAL_FIELDS_TETRAD.md) own runtime and diagnostic
details. Definitions, conditional derivations, configured policies and physical
identification remain distinct. An observed response does not select its law.

## Reading map

Read the ledger below for definitions, then the note answering the current
question. Section numbers remain stable across this family so earlier
mathematical references can still be located. No proof is duplicated here.

| Question | Detailed owner | Sections |
| --- | --- | --- |
| What is a connection, what vanishes when it is absent, and what must a birth law specify? | [Relation foundations](nodal/RELATION_FOUNDATIONS.md) | Support, conductance, metric and causal distinctions; zero-relation admission and remaining constitutive freedom |
| Locality, covariance, phase-domain boundaries and the prospective pressure comparison. | [Pressure premises and constitutive response](nodal/PRESSURE_CONSTITUTIVE_SCOPE.md) | 4 |
| Signed representation, pressure derivatives and finite phase/capacity compatibility. | [Joint form, phase and capacity response](nodal/JOINT_PARAMETER_RESPONSE.md) | 9–11 |
| Fine-to-coarse form, inherited pressure and metric, tetrad reconstruction and changing support. | [Inherited form dynamics and observation](nodal/INHERITED_FORM_DYNAMICS.md) | 12–14 |
| Causal source admission, regional frames, directed exchange and the prescribed-input response. | [Phase-to-form source matching and exchange](nodal/PHASE_FORM_EXCHANGE.md) | 15–16 |
| Symmetry obstructions, phase-law/cutoff premises, reciprocal response, source work and actual event occurrence. | [Primitive phase closure and source selection](nodal/PRIMITIVE_PHASE_CLOSURE.md) | 17–20; [premise review](nodal/PRIMITIVE_PHASE_CLOSURE.md#phase-premise-review) |
| Directed diffusion, sufficient amplitude/phase observations, wave-coordinate scope and regular continuation. | [Derived form phase and inherited closure](nodal/DERIVED_FORM_PHASE.md) | 21 |
| A declared phase/form closure, finite predictions, inherited exchange scale and integration boundaries. | [Conditional cotangent exchange](TNFR_VARIATIONAL_PRINCIPLE.md#cotangent-phase-exchange) | 13.21-13.26; added premise, held capacity; [integration audit](TNFR_VARIATIONAL_PRINCIPLE.md#cotangent-integration-audit), unequal blocks and same-tetrad future witness |
| A relational storage premise compatible with native pressure, its local exchange law and an independent spectral discriminator. | [Relational exchange admission](nodal/RELATIONAL_EXCHANGE_ADMISSION.md) | Conditional alternative; full mean/source balance, origin/scale admission and the arbitrary-storage P2 momentum obstruction |
| When do reciprocal dynamics permit oscillation, recurrence or a frequency-selective response? | [Resonance foundations](nodal/RESONANCE_FOUNDATIONS.md) | Complete sine law; positive-loss obstruction, conservative pulse and almost-everywhere recurrence have different hypotheses |
| Which internal coordinates, periods and stability properties survive grouping? | [Nonlinear replica state](nodal/SINE_PAIR_STATE.md#sine-replica-unordered-state), [joint persistence](nodal/SINE_REPLICA_PULSE.md#sine-replica-joint-persistence), [capacity asymmetry](nodal/SINE_PAIR_STATE.md#sine-replica-capacity-asymmetry) | Exact finite-symmetry quotient with capacity-state correlations; geometric trapping survives unequal positive held capacities, while internal circulation needs its symmetry premises |
| Can phase identify constituent pairs without a supplied partition? | [Phase-only pairing](nodal/SINE_PAIR_GROUPING.md#sine-phase-pairing) | Strict mutual nearest partners inside the protected tube; independent support and complete-law admission, with no claim of formation |
| Which equilibria does the complete law permit without choosing a target winding? | [Fully acute critical geometries](nodal/SINE_PAIR_GROUPING.md#sine-replica-acute-critical) | Exhaustive doubled-C5 classification with arbitrary positive held capacities: uniform form and consensus or uniform phase winding of either sign; represented-state admission is separate |

<a id="derived-information-ledger"></a>
## Derived information and what it adds

The purpose of retaining another quantity is to preserve information consumed
by the declared dynamics. A function of the full state is not an additional
independent primitive, although it can reveal information lost by a collective
observation. More fitted coefficients do not by themselves improve a model.
The following ledger connects existing owners without introducing another
fundamental law or research queue.

| Quantity or mechanism | Information retained and established use | Limit and owner |
| --- | --- | --- |
| First/third relative phase moments `Z1,Z3` | Distinguish the phase information consumed by supplied sine and cubic currents and their potentials | Instantaneous incident sums, not a complete future state. [Classification](nodal/SINE_CONSTITUTIVE_INFORMATION.md#first-phase-moment-sufficiency) |
| Rate-weighted moments `M1,M3` | Retain which relative velocity belongs to which phase; derive moment motion, current derivatives and incident work | Generic evolution needs further correlations. The [regular star chart](nodal/SINE_PAIR_STATE.md#sine-star-moment-chart) instead recovers its existing four-dimensional relative state, with explicit singular obstructions. [Motion and full-star witness](nodal/SINE_CONSTITUTIVE_INFORMATION.md#phase-motion-information) |
| Pair phase product `P`, internal form square `U` and association `W` | Together with mean form `X` and mean phasor `Z`, retain the full unordered pair even at cancellation | [Global pair state](nodal/SINE_PAIR_STATE.md#sine-global-pair-state) on the fixed doubled C5. Constrained derived coordinates preserve four continuous degrees of freedom per pair. Second-moment state information does not add a second-harmonic pressure law |
| Phase-dependent stiffness | The full interaction Hessian connects local geometric protection with coupled form/phase response | Law- and state-dependent; neither the bare support Laplacian nor tetrad curvature replaces it. [Storage-family response](nodal/RESONANCE_FOUNDATIONS.md#storage-family-pattern-robustness) |
| Observable dimension and retained hidden state | Exact row-space growth identifies information needed for a selected linear response; hidden initialization supplies a causal source | A linear observability dimension is not a nonlinear closure theorem. [Shared linear observation](../src/tnfr/mathematics/linear_observation.py), [bridge memory](nodal/RESONANCE_FOUNDATIONS.md#sine-bridge-causal-memory) |
| Memory kernel and reactive response matrix | Derived elimination connects visible motion to its retained environment; the low-frequency matrix can encode a stationary hidden-section metric | A supplied projection and complete fine law remain necessary. Neither an independent primitive capacity nor local irreversible damping follows. [Memory owner](nodal/RESONANCE_FOUNDATIONS.md#sine-bridge-causal-memory) |
| Contact phase moments and relative rates | Explain exchange between a collective pulse and internal deviations while all constituents keep moving | Mean amplitude or common rhythm alone does not close dynamics or establish acquisition. The [pair interaction contract](nodal/SINE_PAIR_INTERACTION.md#sine-moving-pattern-interface) separates present current, retained state and hidden-environment memory. [Collective feedback](nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-collective-pulse-transfer) |
| Normalized contact counts and phase-offset moments | Specify when grouping can inherit an exact nodal description while retaining constituent structure | Exact prepared closure differs from spontaneous grouping and transverse stability. [Partition owner](nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-phase-offset-partition) |
| Winding, sector margins and own-law storage budget | Separate a declared identity, its local maintenance and energetic accessibility | Existing geometry need not be reachable from a given source. [Formation and constitutive scope](nodal/SINE_CONSERVATIVE_PREPARATION.md#sine-constitutive-robustness) |

For a proposed collective state, first test whether states with the same
observation have the same observed derivative. If not, retain the missing
association, derive memory under an admitted elimination, or state the closure
obstruction. If they do, check finite evolution, domain boundaries and events
before asserting exact closure. The tetrad remains a diagnostic observation;
it neither replaces these tests nor selects the consumed constitutive premises.

## 1. Start from a typed law, not from its defaults

Write the unforced nodal equation as

\[
\dot x_i=\nu_i p_i,\qquad p_i=P_i(z),\qquad
z=(x,\nu,\theta,G,W,\ell,h).
\]

Here `h` denotes whatever history the declared model actually needs. It is not
an assertion that all these coordinates are independent physical primitives.
Some could be observations of a richer form; that requires an explicit map
and a closure proof. Conversely, naming them parts of one structure does not
already provide their equations of motion.

For a chosen real EPI chart, let `[x]=X`, `[t]=T`, `[nu]=T^-1`. Then `[p]=X`.
The phase coordinate is an angle on the circle, expressed in radians; an
explicit structural distance can have a separate unit `L`. All code values
may be nondimensional, but the reference scales and time convention must then
be declared. A finite real number does not certify dimensional consistency.

Three levels must stay distinct:

1. **Definition or identity:** for example, wrapped angles have magnitude at
   most pi in the radian chart.
2. **Conditional consequence:** diffusion dissipates its Dirichlet energy
   under the stated nonnegative reciprocal conductance and mobility premises.
3. **Model or numerical choice:** the pressure mixture, phase/capacity update,
   operator word, clipping interval, threshold, seed or sampling protocol.

An identity involving a chosen coefficient proves properties of that choice;
it does not derive why nature or the nodal law must choose that coefficient.
The naming of `constants.canonical` is not an epistemic classification.

### 1.1 Thresholds, constants and numerical settings have different duties

The active configuration audit distinguishes the following categories. A
formula containing pi, a historical `CANONICAL` suffix or a passing regression
test does not move a value from a selected policy to a derived physical law.

| Category | Examples and owner | Meaning and limitation |
| --- | --- | --- |
| Mathematical identity / representation | The radian wrap bound pi; binary64 `PI` in [constants](../src/tnfr/constants/canonical.py) | The geometric bound is exact; the stored floating value approximates it. `sin(pi/6)` is a dimensionless score, not an angular tolerance of 30 degrees. |
| Conditional model quantity | Spectral gaps, diffusion rates, Dirichlet bounds, cycle restoring margins | Derived from the specified graph, law and domain; generally state or configuration dependent. They are not universal fixed TNFR numbers. |
| Constitutive / control policy | Pressure weights, U3 ceiling, capacity averaging, grammar windows and scalar rails | They select a model, allowed event or admissible chart. Structural constraints may restrict their range without selecting a unique value. |
| Diagnostic / classification policy | Si mixture, tetrad warnings, R-squared fit cut and comparison exponents | These classify observations. Consuming them as feedback is an additional controller; a fit-quality cut is not a statistical significance test. |
| Numerical method / error budget | RK coefficients, time step, residual tolerances and represented arithmetic | Method coefficients belong to a specified numerical scheme. A tolerance needs units and a scale; it cannot establish infinite-time convergence or define a new physical law. |
| Implementation setting | Cache capacities, worker counts and tuning scores in [operational constants](../src/tnfr/constants/operational.py) | These organize computation, not nodal dynamics. The compatibility suffix does not give them physical meaning. |

The live owners, rather than equal-looking numbers, determine what can be
unified:

| Configuration / observation | Actual role | Single owner / distinction |
| --- | --- | --- |
| `EPS_DNFR_STABLE=1e-3` | Absolute stored-pressure admission for capacity adaptation; equality is admitted | [Adaptation](../src/tnfr/dynamics/adaptation.py), with the value in `RemeshDefaults`. Its caller must refresh pressure when fresh admission is claimed. This is not the exact equilibrium equation `p=0`. |
| `VF_ADAPT_MU=0.1`, `VF_ADAPT_TAU=5` | Snapshot-average fraction and consecutive qualifying invocation count | [Core defaults](../src/tnfr/config/defaults_core.py). Five calls are not a physical duration without a declared call cadence; successful writes do not reset the count. |
| `SELECTOR_THRESHOLDS` | Cuts for Si, network-normalized absolute pressure and acceleration | The [shared resolver](../src/tnfr/config/selector_thresholds.py) serves selection, adaptation and Si aggregation. `GLYPH_THRESHOLDS` is a different policy, not another source of Si defaults. |
| `PHASE_ADAPT` | Global/local phase controller settings | Partial mappings inherit the same core defaults as full mappings; they must not activate an obsolete private fallback table. |
| `DELTA_PHI_MAX`, `UM_MAX_PHASE_DIFF` | Selected U3 ceiling, with an optional tighter UM limit | The [phase gate resolver](../src/tnfr/operators/_phase_gate.py) owns execution and diagnostic interpretation. The pi/2 admission ceiling is distinct from the pi wrap bound. |
| Potential, gradient and curvature warnings | Selected pi/4, pi/16 and 0.9*pi comparisons | [Field constants](../src/tnfr/constants/canonical.py); U6's pi/2 reference-relative potential drift is a separate observable. Equal units or related formulas do not make the policies interchangeable. |
| Legacy precondition names | Some retained exports no longer drive execution | [Precondition defaults](../src/tnfr/config/thresholds.py) identify active fallbacks and inert compatibility values. The old one-radian RA phase warning does not override the live U3 gate. |

In particular, selector pressure ratios and the absolute adaptation threshold
must not be collapsed into one constant. Their denominators, scales and
purposes differ. Nor should unrelated gains be coupled merely because their
current numeric values match. Centralization means one owner for one semantic
quantity, with explicit compatibility aliases where needed.

#### Unit changes also transform admission policies

Under Section 3's positive chart/time change `y=a*x+b`, `tau=c*t`, with
consistently transformed pressure `p'=a*p` and capacity `nu'=nu/c`, equivalent
absolute-pressure admission requires `epsilon_p'=a*epsilon_p`. A signed
form-rate threshold transforms by `a/c`, an acceleration threshold by `a/c^2`,
capacity rails by `1/c`, form rails by `a*x_bound+b`, and a time step by c.
Dimensionless blend fractions and invocation counts do not rescale, although
changing the invocation schedule changes a hybrid model. The unchanged radian
chart leaves phase-angle gates unchanged.

This proves why a bare positive pressure cut cannot be a chart-independent
constant selected by the nodal product. It does not choose a replacement
threshold or prove complete runtime covariance: the configured normalized
pressure mixture and auxiliary laws retain Section 3's separate obligations.
The existing covariance controls already test that boundary; there is no
second numerical campaign here.

#### Corrected implementation boundaries

The audit retains complete default values while removing inconsistent
interpretations. Partial selector configurations now use the shared selector
defaults everywhere; omitted keys no longer fall back to a different glyph
policy in adaptation or Si aggregation. Nonfinite, boolean or out-of-range
selector thresholds are rejected instead of clamped or silently compared.
Partial phase-adaptation configurations inherit the current default policy.
U3 diagnostics share the live uppercase configuration keys and validation;
invalid admission inputs cannot produce a successful structural check.

Malformed consumed selector, capacity, phase and numerical policies reject
under the [API admission contracts](../docs/API_CONTRACTS.md); custom solvers
and callbacks retain their own scope. Missing Si or pressure is unavailable
capacity-gate evidence, not an observed zero; missing capacity cannot populate
the consumed neighbor snapshot. The
[capacity-law admission](CAPACITY_LOCALIZATION_BALANCE.md#capacity-law-admission)
separates these invocation-based events from a supplied continuous relation.
Clip-aware tolerance has EPI units;
the legacy unclipped rate comparison has EPI/time units. A bare tolerance
cannot be carried between these modes or treated as causal execution evidence.

Field threshold diagnostics retain observation availability. Unavailable
coherence length is not a successful zero, fitted length is compared in its
own path geometry, and spectral fallback retains separate units/provenance.
A satisfactory curvature fit cannot override observed variance violations.
Precision modes do not change intended mathematical definitions, but can
change rounded decisions near a cut; a mode name supplies no certified error
bound. The auxiliary pulse comparison likewise enforces its requested error
tolerance and requires an independent oracle before reporting success.

Importing binary64 constants no longer changes the caller's mpmath precision.
Cold-import controls at 15 and 80 decimal digits preserve that context and
the previous binary64 pi/log(2) values. Numerical precision is part of the
experiment's provenance, not a hidden side effect of configuration import.
Legacy unused tolerance names do not certify integral convergence and the
unused one-degree phase tolerance does not relax U3 execution.

The [grammar calibration helpers](../src/tnfr/config/physics_derivation.py)
require finite positive nonboolean capacity and step with represented product
`0 < nu_f*dt <= 1`. Their scalar recurrence is evaluated exactly on those
represented coefficients; invalid, underflowed or oscillatory inputs do not
produce a spurious calibration. The 64-position U4 cap remains a policy and
need not achieve the surrogate's target. U2 uses the exact reciprocal floor
of the represented rate, avoiding overflow and integer-boundary rounding.
Neither helper establishes graph-wide relaxation or event occurrence.

## 2. Parameter and dependency ledger

| Family | Meaning and owner | Established boundary |
| --- | --- | --- |
| EPI `x` | Coherent form in a declared chart; [foundation types](FUNDAMENTAL_THEORY.md#24-physical-concepts-mathematical-types-and-implementation), [scalarization](../src/tnfr/mathematics/epi.py) | Signed real storage and finite BEPI storage exist. Neither storage complexity nor temporal entropy selects the necessary physical state space. Differential equivalence needs a regular chart and transformed pressure/mobility; injective storage alone is insufficient. |
| Capacity `nu` | Nonnegative local mobility/rate in `xdot=nu*p`; [adaptation](../src/tnfr/dynamics/adaptation.py) | Zero freezes this unforced EPI channel; it need not destroy stored form or freeze all other channels. Held capacity is not a measured oscillation frequency (section 3.4). `nu<=2*pi` is a configured rail, not a phase theorem. No unique autonomous capacity law follows from the product; even a fixed pressure law can have distinct capacities with the same instantaneous EPI rate (section 3.2). |
| Pressure `p` | Directed tangent response, evaluated from a declared constitutive map; [dnfr](../src/tnfr/dynamics/dnfr.py), [support transport](../src/tnfr/physics/support_transport.py) | The EPI channel is a graph difference. The full pressure does not by itself specify a joint potential or joint dynamics. Its sign is chart-dependent and does not name an operator. Stored pressure need not be freshly evaluated pressure. |
| Phase `theta` | Circle coordinate and wrapped neighbor separation; [phase response](../src/tnfr/physics/phase_response.py) | A common rotation is a symmetry of the regular difference/phasor formulas. Relative phase is independent of scalar EPI in the present representation. The nodal product does not imply `theta_dot=nu` or any synchronization law. |
| Clock `t`, `dt` | Declared time coordinate and numerical increment; [integrator](../src/tnfr/dynamics/integrators.py), [directed structural time](../src/tnfr/physics/directed_diffusion.py) | Physical time, operator position, invocation count and history index are different. A step size alone supplies no stability guarantee. A changing clock rate transforms the capacity law too; synchronization need not be a monotone clock (section 3.1). |
| Channel coefficients | `p=w_phi*g_phi+w_epi*g_epi+w_vf*g_vf+w_topo*g_topo`; [defaults](../src/tnfr/config/defaults_core.py) | Numeric sum one is a selected normalization, not physical dimensional analysis. Priority phase/EPI/capacity is configured. See section 3. |
| Support `G` | Declared neighbor relation, including zero-conductance support edges; [relation foundations](nodal/RELATION_FOUNDATIONS.md) | Phase and capacity support need not equal positive EPI transport support. Isolates, self-loops, multiple edges and directionality require explicit conventions. Support availability, instantaneous response and emergent effective interaction are distinct. |
| Conductance `W` | Nonnegative form-transport weight; [support transport](../src/tnfr/physics/support_transport.py) | Common positive rescaling leaves row-normalized EPI transport unchanged. It is therefore not an absolute interaction rate at held capacity. The [zero-relation boundary](nodal/RELATION_FOUNDATIONS.md#zero-relation-boundary) differs between a first neighbor and a new bridge between already active regions. |
| Distance `ell`, kernel exponent | Structural path-distance read-out; [edge semantics](../src/tnfr/physics/_edge_semantics.py) | Explicit `length` wins; absent length, `weight` is a compatibility fallback, then unit length. Undirected distances can be a pseudometric; distinct-node zero distances are omitted from the potential and coherence fit. Parallel lengths combine by minimum; directed distances follow outgoing arcs and can be asymmetric. No equation here derives distance from conductance or selects inverse-square exponent 2 uniquely. |
| Tetrad | Potential, phase gradient, phase curvature, coherence length; [fields](../src/tnfr/physics/fields.py) | Required complementary diagnostics, not a proved closed state or a four-dimensional complete basis. Section 5 states their different units and domains. |
| Coherence `C`, Sense Index `Si` | Shared diagnostic conventions; [metrics common](../src/tnfr/metrics/common.py), [sense index](../src/tnfr/metrics/sense_index.py) | Normalized read-outs are not new dynamical laws. Their use as feedback is a separately configured controller, including where an old runtime already does so. |
| `dEPI`, acceleration | Stored rate or declared event secant, then rate difference per time | A stored pre-projection rate may differ from the realized clipped increment. An event secant is not automatically the continuous derivative; acceleration inherits both timestamps and rate provenance. |
| Currents, energies, charges | Graph contractions and auxiliary models; [conservation](../src/tnfr/physics/conservation.py), [variational principle](TNFR_VARIATIONAL_PRINCIPLE.md) | Positivity of a sum of squares does not prove decay. A charge label does not prove conservation or quantization. Units and a symmetry/action must be supplied for a physical Noether claim. |
| Memory and scale | Hidden-coordinate elimination; [EPI memory](../src/tnfr/physics/epi_memory.py); declared jumps in [REMESH](../src/tnfr/operators/_delayed_remesh_kernel.py) | Exact projected memory is model-dependent. REMESH mixing factors and integer history delays are selected maps, not the automatically derived kernel. Nesting alone is not self-similarity or an autonomous macro-NFR. |
| Operators and grammar | Named transformation contracts, phase gates, word admission and runtime evidence; [contracts](../src/tnfr/operators/operator_contracts.py), [grammar bases](../src/tnfr/operators/grammar_canon.py) | Channel/sign contracts do not uniquely derive magnitudes, order or occurrence. U1-U6 combine definitions, conditions and policies. Completeness of the 13 transformations remains open. |
| Numerical and statistical controls | Clipping, tolerances, discretization, regression bins, sampling and random seeds | They define an experiment or implementation. Reproducibility is not physical necessity; small residuals need an error scale and cannot replace a proof. |

### Variable properties and unresolved foundations

The ledger defines quantities within models; it does not settle every property
needed for a generative interpretation. A variable is admitted by its type,
dependencies and allowed operations, not by its name or by a floating-point
storage slot. For each realization, record:

- Independent state, derived observation, held premise, input, diagnostic or
  numerical policy. An independent coordinate may also be held by a stated law.
- Domain, units, chosen frame/chart, admissible transformations and the meaning
  of equality. Equality of observations need not mean equality of fine states.
- Zero and singular boundaries, positivity/realizability constraints, and which
  combinations of quantities can coexist in one state.
- The prospective evolution or algebraic map, its premises, and which properties
  are inherited under composition or a change of scale.

The outstanding foundational questions are narrower than "all variables are
undefined":

| Foundation question | What can already be used | What remains to justify |
| --- | --- | --- |
| What structure does EPI retain? | A declared signed scalar chart, regular chart changes and exact collective observations | Adequacy of a selected state space, dimension, metric and equivalence of forms for the proposed substrate; EPI is not defined as an amount of coherence |
| What fixes capacity? | Nonnegative mobility relative to a common clock, with explicit zero and positivity boundaries | Its independent structural dependence and evolution, or a proved effective reduction; an observed angular speed or fitted product is not automatically capacity |
| What determines pressure? | A specified tangent response evaluated from joint state, including the complete implemented mixture | Which constitutive premises select that response and its scales; an evaluated `p` is distinct from the map `P` and from a stored event write |
| Which phase is being used? | Primitive `theta` on the circle and separately derived contrast orientation `psi` | Whether a chosen model retains primitive phase, derives it or omits it; their equality requires a map and matching induced dynamics, including zero-form boundaries |
| Where do time and relations come from? | A declared clock, support, transport conductance and separate metric length | Their structural origin and evolution; synchronization alone does not supply a monotone clock, and transport weight does not fix distance |
| Which collective properties can coexist? | Means, amplitudes and relational correlations constructed from fine state | Joint realizability, retained orientation/history and observation-specific closure; independently plausible pairwise values need not define one global state |

The first five rows retain the proofs in the ledger's existing owners; they
are not new failed tests or laws to fill in arbitrarily. In particular,
[signed form and phase](nodal/JOINT_PARAMETER_RESPONSE.md#9-signed-epi-and-phase-an-explicit-representation-test)
already show why primitive phase cannot generally be discarded at zero EPI.
The [inherited metric](nodal/INHERITED_FORM_DYNAMICS.md#132-inherited-metric-restoring-pressure-and-coordinate-dependence)
already distinguishes coordinate mobility from primitive capacity. Reusing
these results prevents another investigation based on an unproved identification.

The derived-form realization has its own compact
[variable contract](nodal/DERIVED_FORM_PHASE.md#variable-contract-for-the-derived-form-family)
and [joint-realizability result](nodal/DERIVED_FORM_PHASE.md#joint-realizability-of-collective-relations).
It is a conditional mathematical realization: its fine scalar form evolves,
capacity/support are held, pressure is evaluated prospectively, primitive
phase is unused, and the regional phase/correlations are derived. The
[reciprocal sine realization](nodal/RESONANCE_FOUNDATIONS.md#reciprocal-exchange)
instead retains primitive phase and its feedback into form; its pulse results
cannot be transferred to the phase-unused realization. Neither set of
choices selects a fundamental physical substrate. Resolving a variable's
mathematical admission and justifying its ontological role are distinct tasks.

### 2.1 The implemented pressure is a specified relational map

Let `N_i` be the unique outgoing neighbors, `k_i=|N_i|`, and let `W` be
the EPI conductance. Define the support and transport differences by

\[
(G_Uv)_i=\frac1{k_i}\sum_{j\in N_i}(v_j-v_i),\qquad
(G_Wx)_i=\frac1{d_i}\sum_jW_{ij}(x_j-x_i),\quad d_i=\sum_jW_{ij}.
\]

An empty support row or zero transport strength gives the corresponding
zero row. On a regular phase branch, with nonzero neighbor resultant,

\[
S_i=\sum_{j\in N_i}e^{\mathrm i\theta_j},\qquad
g_{\phi,i}=\frac{\operatorname{wrap}(\arg S_i-\theta_i)}{\pi},\qquad
p=a g_\phi+eG_Wx+bG_U\nu+tG_Uk.
\]

Here `(a,e,b,t)` are the effective phase/EPI/capacity/topology coefficients
of the declared configuration. The default recipe is
`(pi/(pi+1), pi/(pi+1)^2, 1/(pi+1)^2, 0)`; executable coefficients retain
their represented normalization. Those numbers are configured choices, not
consequences of the nodal product. Section 3 gives their different units.

Only the EPI difference reads transport conductance. Phase, capacity and
degree contrast use unique support neighbors, including zero-conductance
edges. A self-loop is one such neighbor; parallel edges aggregate EPI
conductance without repeating that neighbor. U3 restricts particular
operators, not the neighborhood used by this pressure reading. The
[pressure owner](../src/tnfr/dynamics/dnfr.py) and
[support transport](../src/tnfr/physics/support_transport.py) implement these
conventions. Phase-resultant cancellation, wrap boundaries and represented
arithmetic retain their execution-path-specific availability and numerical
scope; this formula does not assert exact parity between all backends.

The cached dense support adjacency is binary: parallel-edge multiplicity must
not weight the phase or topology mean. The shared
[neighbor-difference reducer](../src/tnfr/mathematics/_neighbor_differences.py)
also evaluates coefficient-weighted linear channels before judging their
representability. A finite final pressure can exist even when an unscaled
difference or normalized intermediate weight overflows or underflows. Graph
and dense optimization adapters reuse this realization; multiplying a rounded
Laplacian is a distinct floating-point calculation, despite the same exact-real
model. Matrix certificates retain the representation stated in their hypotheses.

Capacity has **two separate roles**: `G_U nu` contributes a spatial pressure
contrast, while `diag(nu)` multiplies the complete pressure in the EPI rate.
The former is not a capacity evolution equation. Likewise, `t=0` removes
only the explicit degree-contrast channel: the other channels still depend
on the graph, and EPI transport still depends on its conductance. Degree
contrast is not a general curvature or a law for evolving topology.

Writing `p=eG_Wx+F`, where `F=a g_phi+bG_U nu+tG_U k`, isolates a source
relative to EPI diffusion. It need not be external forcing: all its readings
can belong to the same nodal state. Computing `F` still does not determine
the evolution of its phase, capacity or support. For fixed reciprocal
positive transport strength and held `F`, the EPI-sector potential
`V=e*x^T(D-W)*x/2-x^T*D*F` gives
`xdot=-diag(nu/d)*grad_x(V)`. This conditional representation does not
derive a joint potential or autonomous source law; the
[variational closure audit](TNFR_VARIATIONAL_PRINCIPLE.md#13-forced-potential-family-and-reciprocal-closure-constraints)
owns those additional obligations.

The engine assumes a support and initial nodal state on which these differences
can be read. An isolated row has no neighbor-pressure contribution; an empty
graph supplies no nodal equation from which to instantiate its first node.
Emission acts on an existing node, and nested birth starts from an existing
parent. Their names do not derive an initial substrate. If a larger whole is
modeled as one NFR, its internal regions can supply relational pressure without
an assumed external environment. Identifying that whole with physical reality,
deriving its initial support or placing this structure before an independently
defined physical clock remains a hypothesis. A prerequisite of the current
model is not evidence for an additional substance beyond coherence.

### 2.2 Pressure execution and representation contract

The default scalar pressure rejects nonuniform or complex BEPI rather than
projecting it to a magnitude; uniform-real embeddings keep their sign. The
shared neighbor-difference kernels preserve representable linear differences
and validate active conductances. Pressure is computed before node pressure
writes, so a nonfinite assembled vector is rejected without partial pressure
assignment. This is not a transaction over every preparation cache.

The phase path uses the signed `atan2(sin(delta),cos(delta))` convention,
retaining tiny displacements and signed antipodal ties. Centers outside the
principal chart are read through their materialized phasors before subtraction;
this prevents a huge unreduced coordinate from erasing the neighbor direction.
The certified acute two-neighbor midpoint remains authoritative when available.
Otherwise the consumed floating neighbor sums determine the argument. Exactly
zero sums have the explicit computational extension of zero phase pressure;
small nonzero sums are not suppressed by an undeclared epsilon. Neither this
extension nor a tiny represented resultant derives a unique physical direction
at an ideal cancellation. Reduction order and transcendental rounding can still
matter; there is no claim of bitwise backend equivalence or exact runtime mean
conservation. JIT execution disables fast-math reassociation for this reason.

Public `DNFR_WEIGHTS` updates are observed at the next default refresh, including
in-place edits. The engine retains a detached configuration snapshot next to
its effective mix; detached forcing observers use the same resolution without
writing caches. A legacy explicitly supplied `_dnfr_weights` mix is retained
until the public configuration changes after its first preparation. This
compatibility override is not the recommended configuration interface.
`_DNFR_META.weights_effective` records the executed coefficients;
`weights_norm` only reports normalized proportions. The default pressure and
selector score policies share strict nonnegative represented-real coefficient
admission. Malformed mappings, Boolean/text coefficients, negative values and
nonzero inputs lost during materialization reject before use. Finite large
weights normalize in scaled coordinates when their sum overflows; they do not
disable a valid mixture. A zero total retains the explicit uniform-mix policy.
Setting every public coefficient to zero therefore does not disable pressure.
These are configuration policies, not constitutive deductions.

Optional hooks must be named as different models where their formulas differ:

| Entry point | Actual contract |
| --- | --- |
| `default_compute_delta_nfr` | Configured four-channel map above, with weighted EPI transport. |
| `dnfr_phase_only` | Neighbor-argument phase channel with the same local midpoint and represented-zero rules; not a pairwise sine oscillator law. |
| `dnfr_epi_vf_mixed` | Unweighted unique-support EPI/capacity differences with fixed half coefficients. |
| `dnfr_laplacian` | Unweighted unique-support EPI/capacity differences with raw public coefficients, without renormalization. |
| `compute_delta_nfr_hamiltonian` | Auxiliary projector-diagonal commutator readout, identically zero; see [its algebraic scope](STRUCTURAL_STABILITY_AND_DYNAMICS.md#53-compatibility-helpers-and-sign-scope). It is not a more fundamental generating pressure law. |

NumPy, optimized NumPy, Torch and JAX graph-pressure adapters delegate to the
default CPU owner; an adapter name does not certify GPU pressure or a distinct
constitutive law. Backend, configuration, scalar-domain and failure controls live
in [the pressure read tests](../tests/core_physics/test_pressure_read_contract.py),
[phase path controls](../tests/test_dnfr_fallback_parity.py) and the existing
[linear pressure suite](../tests/core_physics/test_stable_neighbor_pressure.py).
The actual Numba compilation checks explicitly skip where Numba is unavailable;
executing the Python kernel body is a separate validation level.

## 3. Joint changes of form and time units

The type audit imposes a useful positive constraint. Hold support fixed and
write the configured channel readings as

\[
p=w_\phi g_\phi+w_e g_e+w_\nu g_\nu+w_Tg_T.
\]

Here `g_phi` is a wrapped angular displacement divided by pi, `g_e` is the
weighted neighbor EPI difference, `g_nu` is the unweighted neighbor capacity
difference, and `g_T` is the declared dimensionless topology reading. Thus

\[
[w_\phi]=[w_T]=X,\quad [w_e]=1,\quad [w_\nu]=XT.
\]

For `a,c>0`, choose a common affine form chart and a constant time-unit change:

\[
y=ax+b\mathbf1,\quad \tau=ct,\quad \nu_\tau=\nu/c.
\]

Then `g_e` scales by `a`, `g_nu` by `1/c`, and the phase/topology readings
are unchanged. Coefficientwise covariance of this constitutive family uses

\[
(w_\phi',w_e',w_\nu',w_T')=
(a w_\phi,w_e,ac w_\nu,a w_T).
\]

This gives `p'=a*p` and `dy/dtau=(a/c)*dx/dt`, as required. Keeping all
numeric coefficients fixed fails in general, even for a change of time unit
alone. Renormalizing the transformed coefficients to sum one also changes
the rate by `1/S`, where `S` is their transformed sum, unless a compensating
factor is explicitly introduced. The production normalized-weight interface
therefore is not automatically covariant under physical unit changes.

The tests in
[test_nodal_parameter_covariance.py](../tests/physics/test_nodal_parameter_covariance.py)
exercise the existing support/forcing owners with exact rational arithmetic
on their retained finite coefficients. They include wrong-fixed-coefficient
and renormalization counterexamples; they do not claim exact transcendental
phase evaluation or derive new dynamics. Bounds, thresholds, external inputs
and every auxiliary law would also need transformation to establish covariance
of a complete runtime.

The same distinction applies to phase: `theta_dot=nu` treats the numeric
capacity as an angular rate. If `nu` instead counts cycles per time, the
conversion is `theta_dot=2*pi*nu`. Neither choice follows from
`xdot=nu*p`; their relationship requires a stated constitutive premise.
Writing angles in radians gives the exact pi bound, not a universal ceiling
on a rate or a unique coefficient for other channels.

<a id="dimensionless-relaxation-invariant"></a>

**A dimensionless invariant of the selected two-channel family.** With joint
storage `E_D+beta*V_phi`, form storage has units `X^2` and phase cost is
dimensionless, hence `[beta]=X^2`. The
[coefficient audit](nodal/RELATIONAL_RESPONSE_IDENTIFICATION.md#coefficient-synergy-audit)
derives `chi=beta*(e/w)^2`, invariant under the declared form/time changes,
including compensated engine weight normalization. This does not reduce the
full four-channel pressure or all preparation/geometry data to one number.

At uniform form and phase consensus, the selected law has real, critical or
complex nonzero-mode poles separated by `chi=4/pi^2`. This is conditional on
the phase-row and `Arg/pi` normalization, not a universal physical constant
forced by the circle alone. Real poles do not imply monotone observations;
nonconsensus phase geometry can change the boundary. Capacity separability,
storage balance, clock covariance and complete synchronized replicas leave
different chi values admissible.

The same owner's [memory and pole identity](nodal/RELATIONAL_RESPONSE_IDENTIFICATION.md#coefficient-memory-identification)
connects chi to the hidden-phase memory kernel and two temporal poles of one
resolved spatial mode. This can identify a parameter within an admitted model,
not derive its universal value. Independent calibration and reserved evaluation
remain distinct; no coefficient fit or physical realization is supplied here.

### 3.1 Capacity, clock and positivity require compatible laws

A common smooth clock change `tau=T(t)`, `alpha=T'(t)>0`, preserves the
unforced nodal row when `nu_tau=nu/alpha` and pressure is evaluated at the
corresponding original state. If the pressure itself depends on capacity,
its constitutive map must also be transformed; substituting `nu_tau` into
the old formula is not that transformation. For a twice differentiable T,

```text
d nu_tau / d tau = nu_dot/alpha^2 - nu*alpha_dot/alpha^3.
```

Thus a nonconstant clock change generally changes even the form of a supplied
capacity law. The constant-unit result above is its special case. Phase laws,
event times and memory arguments must transform as well. The existing
[capacity-exposure result](nodal/FORCED_SOURCE_AND_CLOCK.md#23-capacity-exposure-does-not-determine-a-phase-clock)
and [time-varying diffusion theorem](TNFR_DIFFUSION_STABILITY_THEOREM.md)
already delimit when a common activity clock is removable. Independent local
activity parameters do not eliminate the need to specify which neighbor state
is read at the shared time.

Synchronization alone cannot be a universal clock: the existing positive
P2 diffusion solution with equal fixed phases has phase order R=1 at every
time while its EPI contrast changes. A clock reconstructed from a structural
observable requires a specified domain and a strictly monotone reading there;
an equilibrium or repeated reading does not supply elapsed time. This excludes
that universal identification, not the possibility of restricted internal clocks.

Nonnegative capacity is a state-domain premise. A supplied locally Lipschitz
continuous-time law `nu_dot=F(z)` must point inward, `F_i>=0` at `nu_i=0`,
to preserve the nonnegative orthant while its solution exists. Nonnegative
initial data alone do not impose this condition. Multiplicative laws such as
`nu_dot=nu*g(z)` with finite integrated g preserve positivity, but do not
uniquely follow from the nodal equation. A clamp is a configured projection;
positivity of a continuous flow also does not guarantee positivity of an
arbitrarily large explicit Euler step. Neither a positive upper rail nor a
phase-to-capacity conversion is selected by these domain requirements.

### 3.2 An EPI rate need not identify capacity, even with a fixed pressure law

On unit P2 with consensus phase, let `delta=x_1-x_0>0`, `e=w_epi>0`,
`b=w_vf>0`, and `f=e*delta`. Both nodes have the same degree, so the actual
default topology and phase channels vanish. For capacities `nu=(2u,u)`, the
declared exact pressure and rate are

```text
p = (f-b*u, -f+b*u),
x_dot = (2u*(f-b*u), -u*(f-b*u)).
```

The distinct positive choices `u=f/(4b)` and `u'=3f/(4b)` give identical
nonzero EPI rates with threefold capacity and one-third pressure. This is an
ambiguity inside one fixed prospective pressure law, not a retrospective
definition `p=x_dot/nu` or freedom to refit its coefficients. It concerns one
instantaneous response, not equality of complete trajectories.

There is an independent prospective discriminator. Hold each supplied
capacity profile and phase fixed. Since the initial EPI rates coincide,
`p_dot=e*G_U*x_dot` coincides and is nonzero. The nodal product then gives
`x_ddot=diag(nu)*p_dot`, so the second profile has threefold acceleration.
The shared joint-response owner checks this consequence before any trajectory
or calibration. Its exact controls treat the materialized e,b as rational
coefficients and declare rational capacities; they do not assert exact
equality after arbitrary binary64 materialization.

Support conventions introduce another, narrower ambiguity. Pure EPI diffusion
on unit P2 with capacities `(1/2,1/4)` equals the EPI generator on the same
off-diagonal conductance with capacities `(1,1)` and self-loop weights `(1,3)`.
In both, mobility `nu_i/d_i=(1/2,1/4)`, the off-diagonal Dirichlet form and its
energy rate coincide. Primitive capacity is therefore not identified by that
transport budget if holding conductances are unrestricted. With the same
consensus phases, the full default law distinguishes these states through
its capacity-gradient channel; pure
transport equivalence is not equivalence of the complete nodal state.
Controls are centralized in
[constitutive capacity scope](../tests/physics/test_constitutive_capacity_scope.py).

A supplied local relation `nu=g(EPI)` is a different, constrained state model.
The [capacity-balance owner](CAPACITY_LOCALIZATION_BALANCE.md#8-a-local-form-capacity-relation-dissipation-and-restoration-criteria)
derives its conserved coordinate, response-slope boundary and conditional
restoration criterion. These results test a proposed relation; they do not
select it or transfer to independently prepared capacities.

<a id="pressure-clock-full-state-closure"></a>
### 3.3 Pressure, clock and capacity charts must transform the whole law

On fixed support and a smooth admitted chart, write a declared full model as
`z_dot=F(z)`, with rows `x_dot=nu*P(z)`, `theta_dot=Omega(z)` and
`nu_dot=A(z)`. The product is componentwise. A change
`d tau/dt=alpha(z)>0` and a capacity chart `kappa_i=q_i(z)*nu_i` preserve
this model only when the coordinate map is regular and invertible and

\[
P_{{\rm new},i}=\frac{P_i}{\alpha q_i},\qquad
\Omega_{\rm new}=\frac{\Omega}{\alpha},\qquad
\frac{d\kappa_i}{d\tau}
=\frac{q_i A_i+\nu_i\mathcal L_Fq_i}{\alpha}.
\]

Here `L_F q_i=Dq_i*F`; all old-state dependencies must be evaluated through
the coordinate inverse. Positive `q_i` alone does not ensure invertibility:
`q_i=1/nu_i` on a positive-capacity domain collapses that coordinate to one.
The pressure-preserving choice `q_i=1/alpha` recovers section 3.1, replacing
`alpha_dot` by `L_F alpha` for a state-dependent clock. Such a choice is a
valid state chart only where its coordinate map is regular; orbit time
reparameterization itself does not require changing the capacity coordinate.

The product's apparent factorization freedom therefore does not supply a
missing capacity or phase law. Even if `A=0`, transformed capacity need not
be held. For example, pure P2 contrast `a_dot=-2*nu*a`, `nu_dot=0`, under
`q=1+a^2`, has `kappa=q*nu`, pressure `-2*a/q` and necessarily
`kappa_dot=-4*a^2*kappa^2/q^2`. At `a=nu=1`, the correct `a_ddot` is 4;
incorrectly holding transformed `kappa=2` gives zero. This is coordinate
covariance, not a new physical capacity mechanism.

The existing [joint-response owner](../src/tnfr/physics/phase_response.py)
already differentiates the implemented pressure along supplied form, phase,
capacity and conductance rates and computes
`x_ddot=nu_dot*P+nu*P_dot`. Its pressure derivatives do not select those rates.
A new law, such as the cotangent correction, needs its own differentiated
term; the old pressure derivative cannot silently stand for it. Support
events and nonsmooth branch changes require separate event/domain handling.

Common phase rotation is also invisible to the configured pressure and
tetrad: on their regular charts `D_theta P * 1=0`. Adding a common phase
velocity is not identifiable from pressure differentiation alone. A relative
phase quotient may be declared; an absolute lifted phase clock needs a
reference and temporal evidence. Conversely, a genuine difference of joint
orbit direction cannot be hidden in a common clock, as the
[exact exchange-family discriminator](TNFR_VARIATIONAL_PRINCIPLE.md#cotangent-integration-audit)
shows. The [same-tetrad future witness](TNFR_VARIATIONAL_PRINCIPLE.md#cotangent-tetrad-future-witness)
then separates clock/law ambiguity from lost state under one fixed law.

<a id="derived-pulse-scales"></a>
### 3.4 Derived pulse scales and observation periods

A pulse is a property of a complete trajectory, not a fourth primitive
coordinate added to the form/capacity/phase triad. The
[finite-amplitude doubled-C5 pulse](nodal/SINE_REPLICA_PULSE.md#sine-replica-internal-pulse)
is derived from the existing reciprocal sine rows with held common positive
capacity, fixed support and zero loss. Its frequency depends on geometry,
amplitude, capacity and the declared exchange/storage coefficients. Capacity
scales time in that family, but neither `theta_dot=nu` nor `frequency=nu`
follows. A physical frequency additionally needs an independent clock bridge.

The observation must be specified before assigning a period. The full labeled
fine state has period T; its exact unordered-pair state has minimal period
T/2 because a half-cycle exchanges the two constituents. A mean observation
can remain constant throughout the same motion. These are different retained
descriptions of one trajectory, not different microscopic clocks. A common
clock change transforms all rows and periods; neither a periodic read-out
nor synchronization alone gives a globally monotone time coordinate.

Existence and robustness are separate properties. The
[full transverse calculation](nodal/SINE_REPLICA_PULSE.md#sine-replica-pulse-splitting)
retains internal and collective feedback and proves instability of the
prepared waveform for sufficiently small nonzero amplitude, without a
numerical amplitude radius. Changing common capacity or time units cannot
remove that dimensionless instability at fixed normalized amplitude. The
[positive-loss obstruction and reversible alternatives](nodal/RESONANCE_FOUNDATIONS.md#permanent-pulse-admission)
also preclude treating permanent oscillation as a consequence of positive
capacity alone. Zero loss remains a declared law premise, not a measured or
uniquely derived microscopic property.

## 4. What locality and symmetry can derive

The [pressure note](nodal/PRESSURE_CONSTITUTIVE_SCOPE.md) owns the
derivations, assumptions, counterexamples and finite-response comparison.
Locality, covariance and regularity restrict a declared response family;
they do not uniquely select the four-channel mixture or primitive phase law.

## 5. Geometry and the whole tetrad

### 5.1 Potential and local phase fields

At fixed metric kernel, `Phi_s=B_G*p` is linear in pressure. For inverse-square
distance, `[Phi_s]=X/L^2` if distances carry length units. Rescaling all distances
by `k>0` scales potential by `k^-2`; independent conductance rescaling does not
change it when every edge has an explicit fixed length. With the old weight
fallback those two changes are coupled by representation, not by a theorem.

`|grad phi|` is a mean absolute wrapped neighbor separation, not a derivative
per unit metric length. `K_phi=wrap(theta_i-Arg sum_j exp(i theta_j))` is a
circular discrepancy; both are bounded by pi in the radian chart. Interpreting
them as spatial differential operators requires a length convention and a
limit. A nonzero resultant and a regular branch are needed for the displayed
phase-response derivative. At very small resultants the current curvature
read-out must distinguish ill-conditioning from an undefined direction.
The shared `observe_phase_curvature` interface now sums its materialized
binary64 phasor components exactly and records that numerical scope. Every
nonzero represented resultant has a reported numerical direction; the former
`1e-9` arithmetic-angle fallback is removed. At exact represented joint zero,
the evidence marks curvature unavailable and numeric curvature/full-field
adapters raise `UndefinedPhaseCurvatureError`. The gradient remains available;
an isolated node has the explicit empty-neighborhood zero convention.

Exact summation of materialized components is not exact transcendental
evaluation or a conditioning guarantee. For example, represented phases
`(0,0,+float(pi),-float(pi))` can have exactly cancelling represented phasors
although their exact-real trigonometric resultant is nonzero. Unavailability
describes the chosen numerical realization; it does not prove a physical
singularity. Pressure, IL and coordination dynamics retain their separately
declared kernels. This correction selects no new phase evolution law.

### 5.2 One coherence-fit definition across implementations

The scalar and vector implementations now share
[_coherence_fit.py](../src/tnfr/physics/_coherence_fit.py). The fitted field is
the static pressure-only read-out `c_i=1/(1+|p_i|)`, with zero rate argument;
it is not generally the full runtime coherence. The fitted statistic is

\[
q(r)=\operatorname{mean}_{d(i,j)=r}(c_i c_j),\qquad
\log q(r)\approx \log A-r/\xi_C.
\]

This is an **uncentered product fit**, not a connected covariance or a
universal correlation-decay theorem. The shared distance channel uses explicit
`length`, then compatibility `weight`, then one, taking the minimum of parallel
lengths and following outgoing arcs. Undirected graphs use unordered pairs;
directed graphs use ordered reachable pairs. Off-diagonal zero-distance pairs
are excluded: undirected zero-length paths give a pseudometric rather than a
separating metric, and directed distances need not be symmetric.
Negative or nonfinite edge lengths and unrepresentable reachable distances
raise rather than silently selecting the spectral fallback. A caller-supplied
distance matrix must satisfy the explicit numeric domain, zero diagonal and
undirected symmetry; it remains declared data, not authenticated shortest paths.

The [field guide](../docs/STRUCTURAL_FIELDS_TETRAD.md) owns fit acceptance,
sampling, supplied-distance admission and cache dependencies. These are
estimator and implementation policies, not physical thresholds.

The previous implementations used hop distances versus conductance distances,
and the vector path discarded one orientation on directed graphs. The common
owner removes this disagreement. Tests use a weighted star with
`c_center=1`, `c_leaf=exp(-r_leaf/xi)`: every pair product is exactly the
target exponential in real arithmetic. This checks a known fit, metric scaling,
explicit-length independence, directionality and cache refresh without inferring
a physical transition from synthetic data.

The spectral fallback remains a different diagnostic:
`1/sqrt(lambda_positive)` from a normalized graph Laplacian is dimensionless.
It does not scale with explicit metric length and is not interchangeable with
the fitted length. The legacy scalar return alone does not attest which
estimator succeeded; research must retain the estimator path and conditions.
Disconnected/directed graphs need their stated implementation scope rather
than an automatic connected symmetric `lambda_2` interpretation.

## 6. Telemetry, time and energy are not interchangeable

The shared coherence map `C(p,r)=1/(1+|p|+|r|)` is a chosen normalized numeric
diagnostic. Before nondimensionalization, `p` and `r=xdot` have different units;
their raw sum is not a unit-invariant physical observable. Declaring scales
for the two inputs would make the definition meaningful under unit changes,
but selecting those scales is still a model/measurement decision.

Network coherence applies this kernel to mean magnitudes. It generally differs
from mean per-node coherence; a projected parent has another pressure again.
Freshness also matters: stored pressure, stored rate and a newly recomputed
pressure are not automatically simultaneous. The zero-pressure/rate predicate
is not a certificate that phase, capacity and topology are at a fixed point.
The shared P2 joint-response control starts with zero pressure and EPI rate,
yet a supplied relative phase velocity gives nonzero pressure rate and EPI
acceleration. It is an explicit counterexample to promoting the instantaneous
predicate into joint equilibrium, not an autonomous evolution law.
Primary coherence and dispersion now share strict authoritative-alias reads:
invalid provided pressure/rate values raise instead of becoming a later alias
or zero; genuinely missing values retain the public zero default.

The [field guide](../docs/STRUCTURAL_FIELDS_TETRAD.md) owns authoritative
source admission, detached cache reads and partial-field availability.
Invalid or unrepresentable inputs cannot establish equilibrium.

Numerical realization must also be controlled before interpreting weak fields
as emergence. Shared circular differences retain tiny nonzero separations;
structural source reductions retain finite inverse-square responses across
large and small represented distances. SDK pressure summaries use the same
stable linear-reduction and dispersion owners. Backend-dependent zeros,
infinities or inconsistent branch signs cannot serve as evidence for a new
phase transition. These controls support the prospective phase/form comparison
without selecting either candidate law.

The separate dispersion diagnostic `1-std(p)/max|p|` is scale-invariant in
exact arithmetic. Its duplicated implementations previously squared raw
pressures and overflowed or underflowed: `(m,2m)` produced `0.75`, `0`, or
`1` as the numeric scale changed. One shared kernel now normalizes before
computing variance. The same fixture retains `0.75` at extreme and subnormal
represented scales, with sign and backend controls. This repairs a numerical
identity; it does not turn dispersion into the primary coherence or a law.

Si combines relative capacity, phase dispersion and relative pressure with
configured weights and clipping. It changes when the comparison population or
normalizers change. The corrected Python path refreshes the same live capacity
and pressure maxima as NumPy instead of trusting stale maxima. This repairs
backend-dependent values without promoting Si to an autonomous mechanism.
Existing selectors/adaptation that read Si remain declared feedback policies.

The default integrator holds supplied pressure and capacity fixed within a
call. Its `rk4` option integrates the supported time forcing, not a newly
evaluated nonlinear pressure at every RK stage. Clipping is a separate map;
an unconstrained rate is not the clipped step secant. With the optional
additive Gamma source the model is `xdot=nu*p+Gamma`: at positive capacity it
can be rewritten using `p_eff=p+Gamma/nu`, but that rewriting is singular
at zero capacity. Nonzero Gamma can move a zero-capacity node. The four
[forcing-scope controls](../tests/test_nodal_forcing_scope.py) verify the
distinction through both integration backends. Unforced proofs require Gamma
to vanish; source residuals cannot be concealed by reconstructing pressure.

The [API contracts](../docs/API_CONTRACTS.md) own registry-based Gamma
evaluation, cache/dispatch consistency and callback/quadrature boundaries.

The [solver contract](../docs/API_CONTRACTS.md#nodal-solver-input-clock-and-output-boundaries)
owns capacity/rate admission, advancing represented clocks and restoration
on numerical failure. Those safeguards do not derive a physical clock or
capacity law.

Capacity-rate telemetry now uses the shared
[timestamped observer](../src/tnfr/metrics/capacity_rates.py). For advancing
recorded times, two capacity samples define the interval secant
`r_n=(nu_n-nu_(n-1))/(t_n-t_(n-1))`. Three define
`B_n=2*(r_n-r_(n-1))/(t_n-t_(n-2))`, using the separation of the two interval
midpoints. Exact rational arithmetic on represented samples avoids inventing
a duration from configured `DT` or rounding midpoint timestamps together.
Finite public results remain approximations to those rational values.
With `[nu]=T^-1`, these diagnostics have units `T^-2` and `T^-3`.
Under `tau=c*t`, `nu_tau=nu/c`, their values scale by `c^-2` and `c^-3`,
respectively; the irregular-sample controls check that covariance.

Each node retains at most three samples and a `capacity_rate_diagnostic`
payload with status, availability and source times. Missing time/capacity,
insufficient samples and unrepresentable results are explicit unavailable
values (`None`). A duplicate time creates no interval; changed capacity at
the same time restarts history at the right-hand value. Backward time or
invalid input is rejected before capacity diagnostics are committed. The
metrics callback preflights capacity/time chronology before recording its
history. It records an initial sample on attachment only when runtime time
is already present. Legacy untimestamped derivative fields are not evidence.

`history['B']` is unavailable unless every current node supplies a representable
second difference; `capacity_rate_coverage` reports the coverage. A genuine
measured zero remains distinct from unavailable. `delta_Si` remains an
increment, not a time derivative. The repaired secants can still contain
unobserved events between endpoints, so they do not prove a smooth capacity
law or exact pointwise derivatives. Event resolution and physical clock
calibration remain separate obligations.

Likewise `E_D=1/2*x^T(D-W)x` is a conditional transport energy, while the
tetrad sum of squares and auxiliary substrate Hamiltonian are other functionals.
Adding quantities with different raw units requires scales/metric coefficients.
Packing curvature and current into a complex number supplies neither their
unit conversion nor a physical quantum state. The existing variational and
conservation owners already separate trajectory residuals, auxiliary flows
and restricted dissipation; reuse those distinctions for every new claim.

The [joint sine storage](nodal/RESONANCE_FOUNDATIONS.md#reciprocal-exchange)
is different from an auxiliary Hamiltonian: its balance follows from the
complete form/phase rows themselves. Zero loss conserves that storage while
permitting internal exchange; positive loss has its stated dissipation term.
Conservation alone gives neither a particular waveform nor its stability,
and does not identify this structural storage with physical energy.

## 7. Memory, operators and scale

Eliminating hidden coordinates in a declared linear system produces an exact
memory kernel and a hidden-initial-state term. These are derived from that
system and projection; no new sustaining mechanism has been discovered by
renaming the kernel. The existing
[derived-memory analysis](DERIVED_EPI_MEMORY.md) and `epi_memory` owner test
continuous closure and minimal realizations. Event closure additionally
requires the existing event-intertwining check.

Keeping internal state can instead give an exact nonlinear collective law.
In the [unordered replica description](nodal/SINE_PAIR_STATE.md#sine-replica-unordered-state),
`R=cos(delta)`, `U=u^2` and `Q=u*sin(delta)` retain internal phase dispersion,
form contrast and their correlation. R enters the inherited phase-to-form
coupling through the derived product `R_i*R_j`; its dynamical role follows
from the fine law, not from a controller reading a coherence score. Q is not
nodal pressure. This constrained state removes only pair labels, retains the
continuous degrees of freedom and requires its local phase chart; R or the
means alone do not close the dynamics.

REMESH instead references retained pre-jump snapshots at integer positions
`history[-(tau+1)]`. A delay in samples becomes a fixed physical delay only
under a declared uniform sampling convention. Variable cycle durations cannot
be ignored. The mixing coefficient, history support, clipping and metric are
part of its map; existing finite/conditional stability certificates already
record those premises. They do not derive REMESH from every projected kernel.

The 13 operators specify allowed named transformations. They coexist with
declared shared nodal solvers. Reciprocal scalar IL/OZ defaults do not make
the full operators inverse or isometric. Reciprocal NUL capacity/pressure
factors preserve their product only for the ideal two scalar rescalings;
capacity is not a geometric volume. Generator/closure labels, operation-count
debts and phase gates supply admission contracts, not spontaneous event timing.

Noise parameters, finite event counts and seeds likewise require a stated
discrete or stochastic model. A fixed per-call perturbation is not a
time-step-independent continuum noise law. A tolerance is not an exact zero,
and clipping-generated persistence is not evidence for unconstrained stability.

## 8. Reuse and consequence for the generative objective

The review preserves a useful core: typed nodal rates, reversible transport,
phase geometry, exact reduction/memory and finite event evidence. The new
locality and covariance results constrain proposed completions without choosing
one to manufacture a pattern. The repaired telemetry permits consistent tests
of those completions; it does not supply the missing laws.

Any further proposed representation must declare its state/equivalence, explain
which quantities are coordinates and which are observations, and test its
directed response with the existing closure and phase-response owners. A
nonuniform finite-amplitude pattern maintained indefinitely cannot be inferred
from fixed positive-capacity pure diffusion on fixed connected
positive-conductance support, which relaxes spatial disagreement. This does
not exclude a declared finite-lived identity. Sources,
finite accumulated activity, memory, changing geometry or another justified
channel can alter that conclusion, but their origin and work balance must be
part of the same model. This is the link to the original generative objective,
not permission to add a tuned stabilizer or a desired attractor.

The [execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md) alone records the
current gate and next action. A universally selected multichannel closure, a unique
selection of all numerical coefficients, and emergence of laboratory particles
or the observable world remain unproved. The audit covers foundational
parameter families and their principal owners; it is not an exhaustive proof
of every implementation or historical document in this repository.


## Earlier section links

<details markdown="1">
<summary>Stable destinations for links published before modularization</summary>

Current repository links point directly to the proof owner. These compact
locators preserve earlier release, external and archived references.

- [4. What locality and symmetry can derive](nodal/PRESSURE_CONSTITUTIVE_SCOPE.md#4-what-locality-and-symmetry-can-derive)
- <a id="41-exact-low-degree-reduction-and-its-nonlinear-boundary"></a>[4.1 Exact low-degree reduction and its nonlinear boundary](nodal/PRESSURE_CONSTITUTIVE_SCOPE.md#41-exact-low-degree-reduction-and-its-nonlinear-boundary)
- <a id="42-when-scale-covariance-and-regularity-force-linear-form-response"></a>[4.2 When scale covariance and regularity force linear form response](nodal/PRESSURE_CONSTITUTIVE_SCOPE.md#42-when-scale-covariance-and-regularity-force-linear-form-response)
- <a id="43-phase-domain-orientation-and-a-discriminating-structural-response"></a>[4.3 Phase domain, orientation and a discriminating structural response](nodal/PRESSURE_CONSTITUTIVE_SCOPE.md#43-phase-domain-orientation-and-a-discriminating-structural-response)
- <a id="prospective-finite-response-discriminator"></a>[Prospective finite-response discriminator](nodal/PRESSURE_CONSTITUTIVE_SCOPE.md#prospective-finite-response-discriminator)
- <a id="44-constitutive-admission-ledger-and-remaining-choices"></a>[4.4 Constitutive admission ledger and remaining choices](nodal/PRESSURE_CONSTITUTIVE_SCOPE.md#44-constitutive-admission-ledger-and-remaining-choices)
- <a id="9-signed-epi-and-phase-an-explicit-representation-test"></a>[9. Signed EPI and phase: an explicit representation test](nodal/JOINT_PARAMETER_RESPONSE.md#9-signed-epi-and-phase-an-explicit-representation-test)
- <a id="91-a-sign-ambiguity-with-different-radial-responses"></a>[9.1 A sign ambiguity with different radial responses](nodal/JOINT_PARAMETER_RESPONSE.md#91-a-sign-ambiguity-with-different-radial-responses)
- <a id="92-zero-form-does-not-erase-a-nodes-phase"></a>[9.2 Zero form does not erase a node's phase](nodal/JOINT_PARAMETER_RESPONSE.md#92-zero-form-does-not-erase-a-nodes-phase)
- <a id="93-a-faithful-encoding-without-a-new-law"></a>[9.3 A faithful encoding, without a new law](nodal/JOINT_PARAMETER_RESPONSE.md#93-a-faithful-encoding-without-a-new-law)
- <a id="10-joint-pressure-response-and-the-capacity-product-rule"></a>[10. Joint pressure response and the capacity product rule](nodal/JOINT_PARAMETER_RESPONSE.md#10-joint-pressure-response-and-the-capacity-product-rule)
- <a id="101-one-identity-with-three-distinct-neighborhood-responses"></a>[10.1 One identity, with three distinct neighborhood responses](nodal/JOINT_PARAMETER_RESPONSE.md#101-one-identity-with-three-distinct-neighborhood-responses)
- <a id="102-pressure-invisible-motion-can-change-form-acceleration"></a>[10.2 Pressure-invisible motion can change form acceleration](nodal/JOINT_PARAMETER_RESPONSE.md#102-pressure-invisible-motion-can-change-form-acceleration)
- <a id="103-which-phase-source-motions-can-capacity-compensate"></a>[10.3 Which phase-source motions can capacity compensate?](nodal/JOINT_PARAMETER_RESPONSE.md#103-which-phase-source-motions-can-capacity-compensate)
- <a id="104-full-tetrad-and-the-remaining-closure-obligation"></a>[10.4 Full tetrad and the remaining closure obligation](nodal/JOINT_PARAMETER_RESPONSE.md#104-full-tetrad-and-the-remaining-closure-obligation)
- <a id="105-geometry-response-is-an-input-to-closure-not-its-selection-law"></a>[10.5 Geometry response is an input to closure, not its selection law](nodal/JOINT_PARAMETER_RESPONSE.md#105-geometry-response-is-an-input-to-closure-not-its-selection-law)
- <a id="11-finite-joint-phase-capacity-source-compatibility"></a>[11. Finite joint phase-capacity source compatibility](nodal/JOINT_PARAMETER_RESPONSE.md#11-finite-joint-phase-capacity-source-compatibility)
- <a id="111-necessary-and-sufficient-condition-and-all-capacities"></a>[11.1 Necessary and sufficient condition and all capacities](nodal/JOINT_PARAMETER_RESPONSE.md#111-necessary-and-sufficient-condition-and-all-capacities)
- <a id="112-positive-capacity-and-declared-bands"></a>[11.2 Positive capacity and declared bands](nodal/JOINT_PARAMETER_RESPONSE.md#112-positive-capacity-and-declared-bands)
- <a id="113-exact-compatible-p2-family-and-strict-u3-obstruction"></a>[11.3 Exact compatible P2 family and strict-U3 obstruction](nodal/JOINT_PARAMETER_RESPONSE.md#113-exact-compatible-p2-family-and-strict-u3-obstruction)
- <a id="114-shared-implementation-tetrad-and-remaining-freedom"></a>[11.4 Shared implementation, tetrad and remaining freedom](nodal/JOINT_PARAMETER_RESPONSE.md#114-shared-implementation-tetrad-and-remaining-freedom)
- <a id="12-intrinsic-response-from-a-closed-fine-nodal-model"></a>[12. Intrinsic response from a closed fine nodal model](nodal/INHERITED_FORM_DYNAMICS.md#12-intrinsic-response-from-a-closed-fine-nodal-model)
- <a id="121-mechanism-inventory-and-reuse-decision"></a>[12.1 Mechanism inventory and reuse decision](nodal/INHERITED_FORM_DYNAMICS.md#121-mechanism-inventory-and-reuse-decision)
- <a id="122-shape-and-rate-are-induced-by-the-nodal-generator"></a>[12.2 Shape and rate are induced by the nodal generator](nodal/INHERITED_FORM_DYNAMICS.md#122-shape-and-rate-are-induced-by-the-nodal-generator)
- <a id="123-an-exact-induced-angular-response-and-closed-rate-law"></a>[12.3 An exact induced angular response and closed rate law](nodal/INHERITED_FORM_DYNAMICS.md#123-an-exact-induced-angular-response-and-closed-rate-law)
- <a id="124-canonical-identification-and-observation-scope"></a>[12.4 Canonical identification and observation scope](nodal/INHERITED_FORM_DYNAMICS.md#124-canonical-identification-and-observation-scope)
- <a id="125-neighbor-coupled-phase-and-internal-pressure-from-scalar-epi"></a>[12.5 Neighbor-coupled phase and internal pressure from scalar EPI](nodal/INHERITED_FORM_DYNAMICS.md#125-neighbor-coupled-phase-and-internal-pressure-from-scalar-epi)
- <a id="126-what-this-changes-in-the-research-question"></a>[12.6 What this changes in the research question](nodal/INHERITED_FORM_DYNAMICS.md#126-what-this-changes-in-the-research-question)
- <a id="13-faithful-macro-state-and-tetrad-inheritance-on-the-retained-prism"></a>[13. Faithful macro state and tetrad inheritance on the retained prism](nodal/INHERITED_FORM_DYNAMICS.md#13-faithful-macro-state-and-tetrad-inheritance-on-the-retained-prism)
- <a id="131-four-internal-coordinates-close-but-omit-a-pressure-direction"></a>[13.1 Four internal coordinates close, but omit a pressure direction](nodal/INHERITED_FORM_DYNAMICS.md#131-four-internal-coordinates-close-but-omit-a-pressure-direction)
- <a id="132-inherited-metric-restoring-pressure-and-coordinate-dependence"></a>[13.2 Inherited metric, restoring pressure and coordinate dependence](nodal/INHERITED_FORM_DYNAMICS.md#132-inherited-metric-restoring-pressure-and-coordinate-dependence)
- <a id="133-the-potential-kernel-must-be-inherited-too"></a>[13.3 The potential kernel must be inherited too](nodal/INHERITED_FORM_DYNAMICS.md#133-the-potential-kernel-must-be-inherited-too)
- <a id="134-full-tetrad-dependency-and-representation-boundary"></a>[13.4 Full tetrad dependency and representation boundary](nodal/INHERITED_FORM_DYNAMICS.md#134-full-tetrad-dependency-and-representation-boundary)
- <a id="14-causal-support-versus-changing-geometry"></a>[14. Causal support versus changing geometry](nodal/INHERITED_FORM_DYNAMICS.md#14-causal-support-versus-changing-geometry)
- <a id="141-positive-geometry-alone-cannot-supply-internal-amplitude-in-this-family"></a>[14.1 Positive geometry alone cannot supply internal amplitude in this family](nodal/INHERITED_FORM_DYNAMICS.md#141-positive-geometry-alone-cannot-supply-internal-amplitude-in-this-family)
- <a id="142-geometry-work-can-increase-energy-while-the-form-shrinks"></a>[14.2 Geometry work can increase energy while the form shrinks](nodal/INHERITED_FORM_DYNAMICS.md#142-geometry-work-can-increase-energy-while-the-form-shrinks)
- <a id="143-canonical-phase-pressure-can-compensate-the-loss-instantaneously"></a>[14.3 Canonical phase pressure can compensate the loss instantaneously](nodal/INHERITED_FORM_DYNAMICS.md#143-canonical-phase-pressure-can-compensate-the-loss-instantaneously)
- <a id="144-capacity-and-causal-closure-remain-explicit-dependencies"></a>[14.4 Capacity and causal closure remain explicit dependencies](nodal/INHERITED_FORM_DYNAMICS.md#144-capacity-and-causal-closure-remain-explicit-dependencies)
- <a id="15-when-an-observed-phase-can-be-a-causal-source"></a>[15. When an observed phase can be a causal source](nodal/PHASE_FORM_EXCHANGE.md#15-when-an-observed-phase-can-be-a-causal-source)
- <a id="151-source-matching-precedes-phase-tangency"></a>[15.1 Source matching precedes phase tangency](nodal/PHASE_FORM_EXCHANGE.md#151-source-matching-precedes-phase-tangency)
- <a id="152-regular-observation-and-the-zero-amplitude-boundary"></a>[15.2 Regular observation and the zero-amplitude boundary](nodal/PHASE_FORM_EXCHANGE.md#152-regular-observation-and-the-zero-amplitude-boundary)
- <a id="153-the-current-supporting-phase-has-only-common-rotation-freedom"></a>[15.3 The current supporting phase has only common-rotation freedom](nodal/PHASE_FORM_EXCHANGE.md#153-the-current-supporting-phase-has-only-common-rotation-freedom)
- <a id="154-persistent-support-requires-persistent-directed-work"></a>[15.4 Persistent support requires persistent directed work](nodal/PHASE_FORM_EXCHANGE.md#154-persistent-support-requires-persistent-directed-work)
- <a id="16-phase-and-form-directed-exchange-frames-and-the-moving-mean"></a>[16. Phase and form: directed exchange, frames and the moving mean](nodal/PHASE_FORM_EXCHANGE.md#16-phase-and-form-directed-exchange-frames-and-the-moving-mean)
- <a id="161-radial-and-angular-effects-of-an-actual-phase-source"></a>[16.1 Radial and angular effects of an actual phase source](nodal/PHASE_FORM_EXCHANGE.md#161-radial-and-angular-effects-of-an-actual-phase-source)
- <a id="162-internal-phase-requires-a-frame-and-transported-comparisons"></a>[16.2 Internal phase requires a frame and transported comparisons](nodal/PHASE_FORM_EXCHANGE.md#162-internal-phase-requires-a-frame-and-transported-comparisons)
- <a id="163-exact-primitive-phase-to-internal-form-coupling-on-repeated-triples"></a>[16.3 Exact primitive-phase to internal-form coupling on repeated triples](nodal/PHASE_FORM_EXCHANGE.md#163-exact-primitive-phase-to-internal-form-coupling-on-repeated-triples)
- <a id="164-the-common-source-reveals-nonlinear-threefold-geometry"></a>[16.4 The common source reveals nonlinear threefold geometry](nodal/PHASE_FORM_EXCHANGE.md#164-the-common-source-reveals-nonlinear-threefold-geometry)
- <a id="165-prescribed-rotating-phase-contrast-periodic-particular-response"></a>[16.5 Prescribed rotating phase contrast: periodic particular response](nodal/PHASE_FORM_EXCHANGE.md#165-prescribed-rotating-phase-contrast-periodic-particular-response)
- <a id="166-what-the-result-does-and-does-not-close"></a>[16.6 What the result does and does not close](nodal/PHASE_FORM_EXCHANGE.md#166-what-the-result-does-and-does-not-close)
- <a id="167-same-input-response-theorem-and-an-all-time-band-bound"></a>[16.7 Same-input response theorem and an all-time band bound](nodal/PHASE_FORM_EXCHANGE.md#167-same-input-response-theorem-and-an-all-time-band-bound)
- <a id="168-reproduce-and-interpret-the-documented-result"></a>[16.8 Reproduce and interpret the documented result](nodal/PHASE_FORM_EXCHANGE.md#168-reproduce-and-interpret-the-documented-result)
- <a id="17-primitive-phase-origin-symmetry-retained-state-and-the-missing-row"></a>[17. Primitive phase origin: symmetry, retained state and the missing row](nodal/PRIMITIVE_PHASE_CLOSURE.md#17-primitive-phase-origin-symmetry-retained-state-and-the-missing-row)
- <a id="171-existing-phase-writers-do-not-select-the-supplied-rotating-contrast"></a>[17.1 Existing phase writers do not select the supplied rotating contrast](nodal/PRIMITIVE_PHASE_CLOSURE.md#171-existing-phase-writers-do-not-select-the-supplied-rotating-contrast)
- <a id="172-a-form-only-equivariant-phase-map-cannot-wind-around-the-origin"></a>[17.2 A form-only equivariant phase map cannot wind around the origin](nodal/PRIMITIVE_PHASE_CLOSURE.md#172-a-form-only-equivariant-phase-map-cannot-wind-around-the-origin)
- <a id="173-relative-oriented-area-is-already-available-in-the-joint-state"></a>[17.3 Relative oriented area is already available in the joint state](nodal/PRIMITIVE_PHASE_CLOSURE.md#173-relative-oriented-area-is-already-available-in-the-joint-state)
- <a id="174-admission-conditions-for-a-joint-linear-response"></a>[17.4 Admission conditions for a joint linear response](nodal/PRIMITIVE_PHASE_CLOSURE.md#174-admission-conditions-for-a-joint-linear-response)
- <a id="175-memory-preserves-omitted-phase-information-it-does-not-derive-its-law"></a>[17.5 Memory preserves omitted phase information; it does not derive its law](nodal/PRIMITIVE_PHASE_CLOSURE.md#175-memory-preserves-omitted-phase-information-it-does-not-derive-its-law)
- <a id="176-consequence-for-the-single-research-queue"></a>[17.6 Consequence for the single research queue](nodal/PRIMITIVE_PHASE_CLOSURE.md#176-consequence-for-the-single-research-queue)
- <a id="18-nonlinear-phase-response-oriented-area-and-tetrad-reuse"></a>[18. Nonlinear phase response, oriented area and tetrad reuse](nodal/PRIMITIVE_PHASE_CLOSURE.md#18-nonlinear-phase-response-oriented-area-and-tetrad-reuse)
- <a id="181-the-complete-nonlinear-projection-retains-the-common-pressure"></a>[18.1 The complete nonlinear projection retains the common pressure](nodal/PRIMITIVE_PHASE_CLOSURE.md#181-the-complete-nonlinear-projection-retains-the-common-pressure)
- <a id="182-nonzero-oriented-area-production-from-initially-uniform-phase"></a>[18.2 Nonzero oriented-area production from initially uniform phase](nodal/PRIMITIVE_PHASE_CLOSURE.md#182-nonzero-oriented-area-production-from-initially-uniform-phase)
- <a id="183-the-same-oriented-area-is-readable-through-the-tetrad"></a>[18.3 The same oriented area is readable through the tetrad](nodal/PRIMITIVE_PHASE_CLOSURE.md#183-the-same-oriented-area-is-readable-through-the-tetrad)
- <a id="184-production-of-orientation-is-not-sustained-identity"></a>[18.4 Production of orientation is not sustained identity](nodal/PRIMITIVE_PHASE_CLOSURE.md#184-production-of-orientation-is-not-sustained-identity)
- <a id="19-nonrepeated-neighbors-and-local-oriented-transfer"></a>[19. Nonrepeated neighbors and local oriented transfer](nodal/PRIMITIVE_PHASE_CLOSURE.md#19-nonrepeated-neighbors-and-local-oriented-transfer)
- <a id="20-phase-reset-source-work-and-actual-occurrence"></a>[20. Phase-reset source work and actual occurrence](nodal/PRIMITIVE_PHASE_CLOSURE.md#20-phase-reset-source-work-and-actual-occurrence)
- <a id="201-exact-source-work-criterion-for-a-finite-phase-proposal"></a>[20.1 Exact source-work criterion for a finite phase proposal](nodal/PRIMITIVE_PHASE_CLOSURE.md#201-exact-source-work-criterion-for-a-finite-phase-proposal)
- <a id="202-eligibility-selection-and-the-missing-sustaining-law"></a>[20.2 Eligibility, selection and the missing sustaining law](nodal/PRIMITIVE_PHASE_CLOSURE.md#202-eligibility-selection-and-the-missing-sustaining-law)
- <a id="203-geometric-admissibility-does-not-require-a-reset"></a>[20.3 Geometric admissibility does not require a reset](nodal/PRIMITIVE_PHASE_CLOSURE.md#203-geometric-admissibility-does-not-require-a-reset)
- <a id="21-derived-form-phase-wave-coordinates-and-genuine-continuation"></a>[21. Derived form phase, wave coordinates and genuine continuation](nodal/DERIVED_FORM_PHASE.md#21-derived-form-phase-wave-coordinates-and-genuine-continuation)
- <a id="211-directed-form-rotation-and-exact-eliminated-state-memory"></a>[21.1 Directed form rotation and exact eliminated-state memory](nodal/DERIVED_FORM_PHASE.md#211-directed-form-rotation-and-exact-eliminated-state-memory)
- <a id="212-a-coupled-amplitude-and-phase-law-derived-from-fine-diffusion"></a>[21.2 A coupled amplitude and phase law derived from fine diffusion](nodal/DERIVED_FORM_PHASE.md#212-a-coupled-amplitude-and-phase-law-derived-from-fine-diffusion)
- <a id="inherited-observation-identity-and-source-work"></a>[Inherited observation identity and source work](nodal/DERIVED_FORM_PHASE.md#inherited-observation-identity-and-source-work)
- <a id="213-canonical-phase-geometry-has-directed-sensitivity-on-reciprocal-support"></a>[21.3 Canonical phase geometry has directed sensitivity on reciprocal support](nodal/DERIVED_FORM_PHASE.md#213-canonical-phase-geometry-has-directed-sensitivity-on-reciprocal-support)
- <a id="214-local-wave-realizability-does-not-select-a-wave-law"></a>[21.4 Local wave realizability does not select a wave law](nodal/DERIVED_FORM_PHASE.md#214-local-wave-realizability-does-not-select-a-wave-law)
- <a id="215-a-form-coordinate-boundary-need-not-require-an-operator"></a>[21.5 A form-coordinate boundary need not require an operator](nodal/DERIVED_FORM_PHASE.md#215-a-form-coordinate-boundary-need-not-require-an-operator)
- <a id="native-phase-contrast-budget"></a>[native phase contrast budget](nodal/PRIMITIVE_PHASE_CLOSURE.md#native-phase-contrast-budget)
- <a id="native-phase-writer-closure"></a>[native phase writer closure](nodal/PRIMITIVE_PHASE_CLOSURE.md#native-phase-writer-closure)

</details>
