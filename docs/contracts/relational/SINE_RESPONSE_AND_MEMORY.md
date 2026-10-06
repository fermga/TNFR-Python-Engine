# Sine response, constitutive comparisons and hidden memory

Declared input/output resonance, bridge-law and clock discrimination, return-path geometry, causal hidden memory, autonomous pulses and recurrence distinctions.

Part of [Relational dynamics contract index](../RELATIONAL_DYNAMICS.md). Section links remain stable; hypotheses and model changes remain local to each result.

<a id="sine-bridge-channels"></a>

### Conservative form/phase response at an aligned bridge

`tnfr.physics.relational_sine_resonance.assess_sine_bridge_channels(source, *,
target_phase_turns, bridge)` returns `SineBridgeChannelAssessment`. It re-admits
the primitive `SineExchangeComparison`, then admits an exact rational-turn
critical target on its connected simple unit support. The target must be
strictly acute, capacities held and positive, exchange/storage coefficients
positive and loss zero. The selected edge must be an actual cut edge with
aligned target phases. A positive-loss recovery certificate is not required
and cannot supply attraction to this conservative calculation.

The observation is the bridge form contrast and `sqrt(beta)` times the
bridge phase perturbation, in clock `tau=w*t/pi`. It retains the nodal
cut-space constraints and uses transpose preparation in the edge energy
chart. `nodal_bridge_lift` has unit oriented bridge contrast and zero weighted
mean: use it for form preparation, and divide it by `sqrt(beta)` for phase
preparation. Origins are retained separately; the homogeneous jets use zero
hidden state. General hidden initialization remains a separate source.

The report keeps admitted source and target separately, exact mobility,
storage scale, `clock_rate_pi_numerator=w`, bridge index/orientation, lift,
incidence row and outward cosine bounds. `first_jet_bounds` and
`second_jet_diagonal_bounds` enclose response derivatives, not Taylor
coefficients. `channel_difference_bounds` encloses form minus phase second
derivative. Its nonnegative exact sign is positive precisely when an edge
incident to a bridge endpoint has a nonzero target gap; the independent
`channel_difference_is_positive` preserves this fact even if interval
resolution includes zero. A zero difference does not certify equality at
later orders. Cached rates, storage and verdicts do not determine the result.

The [proof](../../../theory/nodal/RESONANCE_FOUNDATIONS.md#structural-channel-distinction),
[executable usage](../../guides/relational/SINE_RESPONSE_AND_MEMORY.md#compare-sine-bridge-channels)
and [full-row derivative controls](../../../tests/physics/test_sine_bridge_channels.py)
share this domain. Direct `to_dict()` uses schema
`tnfr.relational-sine-bridge-channels.v1`; `relational_report_to_dict` supports
the generic SDK envelope. Both retain exact fractions and target provenance.
No graph mutation, loss law, finite-amplitude approximation, autonomous
connection or physical identification is derived.

<a id="bridge-storage-family"></a>

### Pattern and response under a declared phase-storage family

`assess_bridge_storage_family(source, *, left_cycle, right_cycle,
target_phase_turns, epsilon)` in the
[resonance owner](../../../src/tnfr/physics/relational_sine_resonance.py) returns
`BridgeStorageFamilyAssessment`. The two disjoint ordered C6 rings cover the
full unit support, with their first nodes joined by the sole bridge. Held
capacities, storage scale and normalized exchange equal one, loss is zero,
and the clock is `tau=t/pi`. Each internal exact target gap has absolute turn
`1/6`; the bridge is aligned. Both winding signs are admitted. Shared source,
target and support admission is also used by the sine bridge-memory reader.

The required nonnegative `epsilon` declares the static alternative
`U_epsilon=1-cos(delta)+epsilon*(2/3-cos(delta)+cos(delta)^3/3)`.
Shared exact-or-represented scalar admission rejects Booleans and nonfinite
values, retaining rational coefficients exactly. `law` identifies this
reciprocal cubic-sine family. The normalized sine `source` retains primitive
support, capacity and reference provenance; its cached sine rates and verdicts
are not rates or certificates of the alternative. The supplied exact target
is distinct from the captured source state. No executor or default changes.

The report retains full node order, target and bridge orientation, exact
`form_laplacian`, `phase_hessian`, `full_tangent_generator` and
`full_energy_metric`. Full matrices order form then phase perturbations.
`edge_phase_curvatures` follows the retained target geometry's edge order.
`bridge_observation_rows` and `nodal_bridge_preparation` retain the same raw
bridge contrasts and nodal preparation for every coefficient. These maps
are orthonormal in each law's tangent energy metric because the bridge
curvature remains one. Equal finite perturbations need not have equal
nonlinear storage.

`first_jet`, `second_jet` and `channel_difference` are exact derivatives in
the declared clock; the difference is the form diagonal minus the phase
diagonal of the second jet. It can be positive, zero or negative. A zero
difference does not generally imply identical future responses. For the
specific value `epsilon=4/9`, the complete phase Hessian equals the form
Laplacian; the [proof](../../../theory/nodal/RESONANCE_FOUNDATIONS.md#storage-family-pattern-robustness)
establishes tangent channel symmetry, separately from nonlinear dynamics.

`target_storage` is the potential at the declared target. The local theorem
retains `local_phase_radius_turns=1/24`, an enclosure of
`m=cos(5*pi/12)` in `local_curvature_lower_bounds`, and an enclosure of
`m*pi**2/288` in `local_excess_storage_threshold_bounds`. Protection requires
the initial lifted edge perturbations to lie strictly inside that radius and
the **same law's full excess storage above the target** to lie strictly below
the threshold. Comparing a separately justified upper excess bound with the
threshold enclosure's lower endpoint is sufficient; its upper endpoint is
not a safe cutoff. Excess storage is not globally nonnegative. The reader
does not evaluate these initial-state premises or certify that the captured
source lies in the protected neighborhood. No global basin, attraction,
formation or physical identification follows.

Direct `to_dict()` and the generic SDK envelope preserve exact fractions,
target association and scope through the shared atomic JSON writer. See the
[usage](../../guides/relational/SINE_RESPONSE_AND_MEMORY.md#compare-bridge-storage-family),
[admission tests](../../../tests/physics/test_bridge_storage_family.py) and
[independent complete-law checks](../../../tests/physics/test_bridge_storage_family_dynamics.py).

<a id="finite-bridge-law-discrimination"></a>
### Finite nonlinear bridge-law discrimination

`assess_bridge_finite_law_discrimination(source, *, left_cycle, right_cycle,
target_phase_turns, amplitude, duration, preparation_error, observation_error)`
in [`relational_bridge_discrimination.py`](../../../src/tnfr/physics/relational_bridge_discrimination.py)
returns `BridgeFiniteLawDiscrimination`. Shared two-C6 bridge admission rebuilds
the complete support, unit capacities, target and common preparation/readout
for the fixed comparison `eta=0` versus `eta=1/100`. The captured source anchors
support and reference-law metadata; the two declared finite preparations are
separate states, with opposite form rows and the same exact rational-turn target.

Each actual preparation may differ by `preparation_error` in every one of its
twenty-four form and lifted-phase coordinates. `observation_error` bounds each
complete bridge-form reading, not each individual sensor. The statistic is
`C=(2*a-u_plus(h)+u_minus(h))/(a*h**2)` from two positive-time observations.
Amplitude and duration are declared in the normalized model and `tau=t/pi`;
clock uncertainty is not included in either error parameter.

The [proof](../../../theory/nodal/RESONANCE_FOUNDATIONS.md#finite-bridge-law-discrimination)
uses full-law reversibility and the exact finite-amplitude initial curvature.
It separately bounds finite-time remainder, preparation propagation and reading
error. `curvature_prediction_bounds` retain the resulting outward intervals;
`separation_margin_lower_bound>0` and `phase_chamber_certified` are both needed
for `discrimination_certified`. An unavailable margin does not imply identical
laws. The reader computes no trajectory and consumes no measured response.

Require positive amplitude and duration, nonnegative error budgets, and
`2*duration<=1` for the shared exponential enclosure. Invalid inputs reject;
failed sufficient inequalities return explicit reasons. All derivatives and
uncertainty bounds retain the full nonlinear state. Symbolic target angles are
not converted into supposedly exact binary64 phases. The projected initial
boxes can include additional outward rounding width; this does not enlarge
the declared admissible preparation error. Any implemented angle approximation
must independently fit within that error.

The direct schema is `tnfr.bridge-finite-law-discrimination.v1`; generic SDK
exports use the shared report wrapper. This is a conditional analytical
prediction under a declared clock, not physical law selection, formation,
clock identification or an alternative runtime installation. See the
[finite-budget example](../../guides/relational/SINE_RESPONSE_AND_MEMORY.md#finite-bridge-law-discrimination).

<a id="finite-bridge-clock-law-discrimination"></a>
### Finite bridge-law discrimination with a bounded common clock

`assess_bridge_clock_law_discrimination` in the same shared owner returns
`BridgeClockLawDiscrimination`. It retains the preceding support, exact target,
two complete laws and opposite form preparations, and adds a third preparation
with zero form and phase `theta_target + amplitude*p`. This uses the existing
independent phase column; none of the twenty-four coordinates is discarded.

All three observations occur at the same declared `sampling_duration=s`.
Their actual structural duration is `h=k*s`, where one unknown **constant**
positive `k` is shared by every evolution row and every experiment and lies
within independently declared `clock_scale_bounds`. This is not a bound on
arbitrary clock drift or independently reset clocks.

The observable is the raw ratio
`Q=(2*a-u_plus(h)+u_minus(h))/(2*(a-v(h)))`. Here `v` is the **signed** bridge
phase gap relative to its exact aligned target lift: any integer target-turn
offset must be removed. The certified chamber keeps this branch near zero.
An unsigned phase distance is not this observable. Each of the three initial
states has its own full-coordinate preparation error cube; the common reading
error bounds each complete gap in its declared normalized coordinate units.
An actual experiment must independently admit both form and phase readouts.

The [proof](../../../theory/nodal/RESONANCE_FOUNDATIONS.md#finite-bridge-clock-law-discrimination)
divides both raw responses by the same unknown `a*h**2` as an algebraic device,
then cancels it in the quotient. These normalized responses are proof
coordinates, not separately measured values. The exact finite-amplitude phase
coefficient contains `sin(a)/a`; replacing it by its tangent is unsupported.
Uniform errors retain the dependence on the actual time: with fixed growth and
derivative bounds at the longest duration, each complete error budget is
`A*h**2+B/h**2`. Convexity in `h**2` makes its maximum the larger **complete**
endpoint total. Taking separate maxima of its components is unnecessarily
weaker. Clock uncertainty does not disappear from these remainders.

Both normalized phase prediction intervals must have strictly positive lower
bounds before quotient formation. An unresolved denominator produces no ratio
bounds or separation margin. Certification additionally requires the phase
chamber and strict separation of the resulting ratio intervals. Overlap or a
failed sufficient bound is unavailable evidence, not equivalence of the laws.
No observed response is consumed and no selected law is installed.

The numerical domain is `0<amplitude<=1`, positive sampling duration, ordered
positive clock bounds, nonnegative preparation/reading errors, and
`2*sampling_duration*clock_scale_bounds[1]<=1`. Invalid inputs reject. Outward
initial boxes do not enlarge the admitted preparation cubes. The direct schema
is `tnfr.bridge-clock-law-discrimination.v1`; generic SDK export uses the shared
wrapper. The [example](../../guides/relational/SINE_RESPONSE_AND_MEMORY.md#finite-bridge-clock-law-discrimination)
keeps the previous amplitude, sampling time and errors and adds ten-percent
clock uncertainty. This is conditional discrimination between two supplied
models, not clock identification, a unique pressure law or physical validation.

<a id="sine-bridge-memory"></a>

### Exact two-cycle bridge memory and the local-loss boundary

`tnfr.physics.relational_sine_bridge_memory.assess_sine_bridge_memory(source,
*, left_cycle, right_cycle, target_phase_turns)` returns the immutable
`SineBridgeMemoryAssessment`. It reuses the preceding bridge admission and
restricts its scope to two ordered, disjoint C6 cycles covering the full
support, with their first nodes joined by the sole bridge. Every internal
target edge has absolute turn `1/6`. Capacities, storage scale and effective
exchange are one, and loss is zero. Either winding sign, cycle reversal,
common target-phase shifts and node relabeling preserve the calculation.
Unsupported geometry or primitive state raises; cached source verdicts do
not provide the tangent law.

`projection_rows` maps a supplied full nodal tangent vector, ordered as all
form perturbations then all phase perturbations, to eight rational shell
coordinates. The coordinate order is four form increments followed by four
phase increments. `nodal_lift` is an invariant section, not reconstruction
of the complete nodal state. Exact intertwining and energy orthogonality
authenticate the reduction; the discarded symmetry sectors are unobservable
at this bridge in this tangent law. They need not be absent from the network.

The report retains `full_tangent_generator`, `coordinate_generator` and
`energy_metric`. Shared `derive_linear_observation` checks that all eight
coordinates are required for this all-state bridge observation. Shared
`derive_coordinate_memory` selects visible indices `(0,4)` and retains the
other six coordinates, in original complementary order. Its blocks `A,B,C,D`
give the exact causal identity, in clock `tau=t/pi`,

`y' = A*y + B*exp(D*tau)*h0 + integral B*exp(D*(tau-s))*C*y(s) ds`.

`hidden_initial_projection_rows` maps a separately declared nodal tangent
initial state to `h0`; `initial_source_rows` gives its source at time zero.
The captured source state is neither a target perturbation nor a zero-hidden
preparation. No initial perturbation is invented or exponential evaluated.
The report also retains the zero-lag kernel through `coordinate_memory`,
its first two derivatives and the exact matrices
`static_visible_generator=A-B*D^-1*C` and
`low_frequency_rate_matrix=I+B*D^-2*C`.

Those last matrices are resolvent coefficients. The static generator is
`(2/13)*R`, with `R=((0,-1),(1,0))`; the rate matrix is
`diag(309/169,449/169)`. They do not replace the causal convolution or supply
a memoryless executor. The [proof](../../../theory/nodal/RESONANCE_FOUNDATIONS.md#sine-bridge-causal-memory)
establishes a nondecaying three-frequency kernel, a visible slow hidden mode
and the absence of symmetric damping in the zero-frequency coefficient.
Time-domain approximation needs separate preparation, time/frequency and
error premises. No nonlinear formation, physical mass or irreversible-loss
claim follows from the retained rate matrix.

Direct export uses schema `tnfr.relational-sine-bridge-memory.v1`; the generic
SDK envelope and atomic JSON writer also support the report. Exact fractions,
target provenance, matrix order and scope are preserved. See the
[guide](../../guides/relational/SINE_RESPONSE_AND_MEMORY.md#retain-sine-bridge-memory),
[admission controls](../../../tests/physics/test_sine_bridge_memory.py) and
[independent full-law kernel checks](../../../tests/physics/test_sine_bridge_memory_kernel.py).

<a id="return-path-storage-geometry"></a>

### Constitutive dependence of an implicit return-path equilibrium

`assess_return_path_storage_geometry(source, *, left_cycle, right_cycle,
mediator, epsilon, refinements=40)` in the
[phase-cycle owner](../../../src/tnfr/physics/phase_cycle_geometry.py) returns
`ReturnPathStorageGeometryAssessment`. The ordered disjoint C5 rings and
the distinct mediator cover the full eleven-node support. Their first nodes
are linked through the mediator, and their second nodes have a direct return
edge. No additional edge is admitted. Shared source admission retains unit
conductance support, unit held capacities, zero loss, `w=beta=1` and clock
`tau=t/pi`; the sine comparison is a primitive reference association, not
an observation of the alternative law.

The required nonnegative `epsilon` declares the same cubic-sine potential as
the [bridge comparison](#bridge-storage-family). Exact-or-represented scalar
admission rejects Boolean and nonfinite inputs. `refinements` is an integer
from 1 through 64; this is a numerical work budget. The same strict-sign
bisection kernel also serves the retained native return-geometry instrument,
whose original sine bracket and default preparation remain distinct.

The target has constant zero form and prescribed named periods `(1,-1,0)`.
Its special edge angle `t` is defined by the unique root of
`F_epsilon(t)=j_epsilon(pi/2-t/4)-j_epsilon(t)-j_epsilon(2*t/3)`.
`root_turn_bracket` encloses `s=t/(2*pi)` between strict certified endpoint
signs, starting in `(1/12,1/5)`. Its residuals enclose `F_epsilon/(1+epsilon)`;
this positive normalization improves arithmetic and changes neither roots
nor the declared dynamics or clock. Unresolved interval signs reject a
refinement instead of selecting a half by rounded values or a tolerance.

`nodal_turn_affine_coefficients` and `edge_turn_affine_coefficients` store
`(intercept,slope)` in the **same** unknown root `s`, preserving correlations.
Node lifts follow `source.nodes`; oriented edge gaps follow `geometry.edges`.
`edge_integer_offsets` retain the circular reconstruction, and
`named_cycle_periods` are exact affine identities. The corresponding bounds
enclose one implicitly defined equilibrium; arbitrary choices from their
Cartesian product need not satisfy its periods or current balance. A root
midpoint is an approximation and is not certified as an exact critical state.

The [proof](../../../theory/nodal/RELATIONAL_RETURN_PATH_GEOMETRY.md#return-path-storage-dependence)
establishes existence, uniqueness throughout the acute period cell and
constitutive motion of the geometry. `nodal_root_residual_multipliers` keep
the complete nodal current identities, so criticality does not follow merely
from small `nodal_current_residual_bounds`. `edge_current_bounds`,
`edge_curvature_bounds` and `target_phase_storage_bounds` refer to the original
unscaled potential. `minimum_acute_margin_turns_bounds` retains the distance
from every target gap to a quarter turn. The positive Hessian supports
conditional local stability under this conservative law; it supplies no
capture radius, attraction, trajectory or spontaneous winding formation.

Direct schema `tnfr.return-path-storage-geometry.v1` and generic SDK projections
preserve the law, exact affine data, implicit root, source association and
unavailable claims. Export is detached
and does not authenticate provenance. See the
[usage](../../guides/relational/SINE_RESPONSE_AND_MEMORY.md#compare-return-path-storage-geometry),
[admission controls](../../../tests/physics/test_return_path_storage_geometry.py)
and [complete-nodal checks](../../../tests/physics/test_return_path_storage_dynamics.py).

<a id="return-path-geometry-response"></a>

### Infer a constitutive interval and a separate nodal response

`assess_return_path_geometry_response(source, *, left_cycle, right_cycle,
mediator, special_turn_bounds, form_direction, observation_origin)` returns
`ReturnPathGeometryResponseAssessment` from the same
[phase-cycle owner](../../../src/tnfr/physics/phase_cycle_geometry.py).
It shares the preceding reader's complete support, coefficient-independent
geometry admission and unit-capacity, zero-loss, `w=beta=1`, `tau=t/pi` premises.
It does not run a forward coefficient root to construct its own observation.

`special_turn_bounds` contains two ordered exact-or-represented real endpoints
for the special angle divided by mathematical `2*pi`. `form_direction`
contains eleven admitted signed real values in source-node order. Boolean,
nonfinite and malformed inputs reject before the calculation.
`observation_origin` is explicitly `supplied_mathematical_interval` or
`measured_interval`; the latter declaration authenticates neither measurement
nor physical correspondence. Stored source phases, cached rates and storage
are not observations of the inferred equilibrium.

The [inverse theorem](../../../theory/nodal/RELATIONAL_RETURN_PATH_GEOMETRY.md#return-path-geometry-response)
identifies the admissible half-open branch from the sine root through, but
excluding, the pure-cubic limiting root. The coefficient is `-F0/F3` on that
branch. Strict outward endpoint signs yield the following statuses:

| `coefficient_status` | Meaning and availability |
| --- | --- |
| `bounded` | A compatible finite coefficient enclosure; both bounds and the full initial acceleration enclosure are available |
| `unbounded_above` | The compatible angle interval reaches the limiting geometry; `coefficient_upper` and finite response bounds are `None` |
| `incompatible` | The interval misses every finite nonnegative member; coefficient and response bounds are `None` |
| `unresolved_interval_arithmetic` | Numerical signs cannot certify classification; bounds remain unavailable rather than implying incompatibility |

The reader retains original observation bounds, endpoint residual enclosures,
`geometric_outer_turn_bounds`, exact nodal/edge affine correlations and named
periods. Outer angle/coefficient boxes enclose candidates; their Cartesian
product is not a set of independent equilibria. Geometric error can make
coefficient error unbounded and does not identify the dynamic rate scale.

For a bounded case, `phase_velocity_direction=K L h` and
`response_acceleration_bounds` encloses `-K H_epsilon K L h` for all compatible
equilibria. `H_epsilon` is the full phase Hessian, evaluated with outward
trigonometric intervals. This uses a prescribed initial form direction and
zero initial phase perturbation. `response_status` is
`bounded_initial_tangent_acceleration` or `unavailable`; the known phase
velocity direction remains available even when the acceleration is not.
This conservative availability rule also withholds a response for unbounded
coefficients when a specially chosen null direction could have zero response.
No positive-time trajectory, event, runtime installation or formation claim
follows from an initial derivative.

Direct schema `tnfr.return-path-geometry-response.v1` and the generic SDK
envelope preserve exact fractions, unavailable values and declared origins.
The [guide](../../guides/relational/SINE_RESPONSE_AND_MEMORY.md#infer-return-path-response),
[admission tests](../../../tests/physics/test_return_path_geometry_inference.py)
and [independent full-row controls](../../../tests/physics/test_return_path_geometry_prediction.py)
exercise this boundary. The
[staged producer](../../../benchmarks/return_path_geometry_response.py) consumes
a fixed known-source protocol, saves its prediction before evaluation and
checks saved bounds against rebuilt geometric premises before using them.

<a id="sine-resonance"></a>
### Sine-law resonance and declared input/output response

[`relational_sine_resonance.py`](../../../src/tnfr/physics/relational_sine_resonance.py)
owns static tangent assessments for the complete normalized-sine law.
`assess_sine_cycle_resonance(model=..., node_count=..., mode_index=...,
capacity=..., winding=0)` declares a complete unit cycle with uniform form,
exact target phase turns `winding*j/node_count`, common held positive capacity
and an explicit regular reference model with positive EPI loss weight.
The reference model supplies normalized coefficients, not native Arg rates.

Counts, winding and mode must be exact nonboolean integers, with `n>=3`,
`4*abs(winding)<n` and `1<=mode_index<n`. Shared represented-real admission
retains exact positive rational capacity. The analytic template allocates no
graph and performs no mode search or floating eigensolve. Its modal rows
differentiate the shared sine field; bounds use mathematical pi and outward
interval arithmetic. This is not admission of a live observed cycle.

`SineCycleResonance` retains the template, coefficient ratio
`beta*(e/w)^2`, modal matrix, damping ratio and separate pole/phase-peak
classifications. Its input is a unit Euclidean real Fourier mode added to
the form-rate row. Form amplitude has transfer `s/D(s)`, phase amplitude
has `b*r/D(s)`, with `D=s^2+e*r*s+a*b*cos(alpha)*r^2` and
`r=capacity*lambda/2`. The work-conjugate output is `lambda*X`.
Natural frequency equals the form/work gain peak; it is not necessarily
an observable free-oscillation frequency. The exact work peak gain is
`2/(e*capacity)`. Positive-frequency phase gain and complex poles require
different inequalities.

`report.gain(angular_frequency)` admits a finite nonnegative exact or represented
real frequency in the model clock and encloses all three gains separately.
Under [chained-report admission](SINE_COMPARISON_AND_INFERENCE.md#sine-chained-report-admission), it rebuilds
and retains the mode from its law, model, cycle size, winding, mode index
and capacity.
Unresolved denominators or classification signs retain unavailable reasons;
an interval containing zero does not establish a zero gain, critical damping
or absent susceptibility. At zero frequency the form/work transfer is zero.
Invalid domains reject rather than being coerced into an admissible mode.

`certify_sine_recovery_resonance(recovery, port=(i,j))` reuses a shared
full-support sine pattern/cycle recovery report through
[chained-report admission](SINE_COMPARISON_AND_INFERENCE.md#sine-chained-report-admission).
The target must meet that owner's sufficient acute criticality, positive
capacity, support and recovery hypotheses. With `K=diag(nu_i/d_i)` and
`p=e_i-e_j`, the declared probe is `f=K*p`, the input term is `f*u(t)`
and the conjugate output is `y=f^T*L*delta_x`. The theorem establishes an
interior gain maximum and the bound `(K_ii+K_jj)/e`, without selecting its
frequency. All original environmental coordinates remain in the target;
the statement concerns its tangent, not an observed transient's transfer.
Uncertain held capacities describe a family of systems. If a selected port's
capacity is uncertain, the corresponding probes/readouts also form a family,
rather than one calibrated fixed physical port.

Reports use exact shared JSON projection through `to_dict()` and the SDK
atomic writer for export. `relational_report_to_dict` also accepts the
resonance report types and delegates projection to this owner while preserving
the generic SDK envelope. Export does not authenticate a manually constructed
dataclass. No report installs forcing, executes RA, adds an edge, advances a
solver or demonstrates nonlinear formation, autonomous excitation or physical
resonance. The [foundation](../../../theory/nodal/RESONANCE_FOUNDATIONS.md)
owns the full-network proof, gain ceiling, counterexamples and clock scope.

<a id="sine-mediated-response"></a>
### Remote sine response through a retained intermediary

`assess_sine_mediated_response(recovery, donor_cycle=..., receiver_cycle=...,
mediator=...)` in the same [resonance owner](../../../src/tnfr/physics/relational_sine_resonance.py)
uses [chained-report admission](SINE_COMPARISON_AND_INFERENCE.md#sine-chained-report-admission) for its supplied
recovery and requires exactly two disjoint ordered
five-node cycles and one intermediary. Each cycle's first node is its sole
contact with the intermediary. All eleven nodes and twelve edges must be
present, with no additional edges. Capacities must be exact held positive
scalars; uncertain capacity families are outside this adapter.

Shared recovery admits the exact acute critical target, independently of the
nominal observed perturbation. Both bridge target gaps must be zero modulo a
turn. Those hypotheses force uniform acute cycle twists; winding -1, 0 and 1
are permitted separately on each cycle. No graph mutation or target fitting
occurs. The declared clock and normalized coefficients are retained.

The donor direction is `f=K*p_D`, where `p_D` is one at the donor port,
minus one quarter at its other four nodes, and zero elsewhere. The receiver
contrast `r_R` has the same weights on the receiver. It observes phase or
form relative to the other receiver nodes, using target-relative phase lifts.
It is not a structural-work-conjugate readout. The report concerns the
exact target's tangent, not a time-invariant approximation inferred from the
observed transient.

`SineMediatedResponse` retains `form_markov_bounds` and `phase_markov_bounds`
for derivative orders 0, 1 and 2, and the corresponding full-state derivative
vectors. These are `C*J**j*(f,0)`, without Taylor factorials. They describe
both the autonomous tangent response to initial form `epsilon*f` and the
impulse kernel for hypothetical form-rate input `f*u(t)`. A constant input
switched on from zero has its corresponding first phase effect at order 3,
not order 2. No impulse or other forcing is installed.

Static bridge balance gives exact zero DC contrasts. The phase second
derivative is `phase_order_two_pi_numerator/pi`, which is strictly negative
by exact positive-capacity/coefficient admission. This sign remains distinct
from an outward enclosure that touches zero. The form coefficient can cancel.
`positive_frequency_phase_peak_certified` and
`phase_impulse_sign_reversal_certified` apply the
[remote-response proof](../../../theory/nodal/RESONANCE_FOUNDATIONS.md#mediated-resonance):
there is a finite-frequency phase-gain maximum and the autonomous tangent
phase contrast becomes positive after its initial negative response.
Peak frequency, reversal time and a quantitative nonlinear amplitude bound
remain unavailable. Neither positive-realness nor the collocated gain ceiling
is assigned to this remote phase output.

The retained 2-by-2 contact blocks use form-then-phase order. They define the
intermediary generator and the donor-to-intermediary and intermediary-to-receiver
couplings. `memory_kernel_at_zero_bounds` encloses their product. The complete
kernel also depends on the intermediary exponential, hidden initial state and
visible history; no instantaneous direct edge replaces it.

The two `*_silent` flags certify sufficient tangent reflection symmetries by
checking support, exact target cosine geometry and equal capacities on reflected
pairs. A false flag means this proof is unavailable, not that transmission has
been demonstrated. Finitely many zero Markov entries never prove permanent
silence. Invalid support, incompatible recovery or inexact capacities reject.
`to_dict()` uses schema `tnfr.relational-sine-mediated-response.v1`; the generic
SDK export delegates to it and retains the generic envelope. This is a static
theorem report, not a trajectory, formation event or physical observation.

<a id="sine-pair-pulse"></a>
### Autonomous nonlinear pulse on a complete sine pair

`assess_sine_pair_pulse(graph, reference_model=...)` in the shared
[resonance owner](../../../src/tnfr/physics/relational_sine_resonance.py)
reuses sine state admission, rates and work. It requires the complete two-node,
single-unit-edge support, held nonnegative capacities and absent inputs/events.
The model supplies the coefficients, storage scale and clock. The reader
never changes a positive loss coefficient to zero. It does not execute the
native argument-pressure runtime or install a new pressure law.

In captured node order, `u=x0-x1`, `delta=theta1-theta0` and
`sigma=nu0+nu1`. Storage is `E=u**2/2+beta*(1-cos(delta))`;
`normalized_energy_bounds` encloses `E/beta`. Equal captured phase values
also give `exact_energy`, preserving exact rational decisions at zero energy
and the separatrix. Other energies use outward enclosures; unresolved signs
remain unavailable rather than being rounded into a certificate.

`SinePairPulseAssessment` separates the following cases:

- `libration_certified`: supplied `e=0`, `sigma>0` and certified
  `0<E<2*beta`. The full nonlinear continuous state is periodic, including
  a pair with exactly one zero capacity.
- `dissipative_recurrence_excluded`: `e>0`. Nonstationary recurrence is
  excluded even if the initial loss happens to be zero.
- `frozen` or `equilibrium`: respectively both capacities vanish, or the
  captured forms and phases coincide. These checks certify stationary cases;
  they do not enumerate every circular representation of an equilibrium.
- `separatrix_out_of_scope`, `rotation_out_of_scope` or
  `energy_classification_unresolved`: no nonstationary libration certificate.

`small_amplitude_period_bounds` encloses `T0=2*pi**2*sqrt(beta)/(w*sigma)`
when `e=0` and `sigma>0`. For certified libration, `period_bounds` uses
the exact elliptic-integral inequalities `T0<T<=T0/sqrt(1-E/(2*beta))`.
The period refers to the model clock, not an identified laboratory unit.
A certificate of period existence can remain true while its numerical bound
is unavailable near the separatrix; `period_unavailable_reasons` retains
that distinction. Relative-rate bounds and `initial_continuous_loss` come
from the shared complete-law reader, not a fitted oscillation.

`to_dict()` uses schema `tnfr.relational-sine-pair-pulse.v1`; the generic
SDK report export delegates to the same projection. The detached assessment
does not advance time, modify a graph, certify an Euler trajectory, select
an attracting amplitude or generate motion from equilibrium. The
[proof](../../../theory/nodal/RESONANCE_FOUNDATIONS.md#permanent-pulse-admission)
also establishes the broader finite closed positive-loss sine obstruction.
Zero loss and a nonzero initial preparation remain explicit premises,
not a derived fundamental selection or physical identification.

<a id="sine-conservative-path-memory"></a>
### Conservative path memory and nonlinear regional transfer

`assess_sine_path_memory(graph, reference_model=..., mediator=...,
consensus_form=0, consensus_phase=0)` uses the same
[resonance owner](../../../src/tnfr/physics/relational_sine_resonance.py).
It requires a complete unit P3 path, the correct middle node, common positive
held capacity and an explicitly zero EPI loss weight. Shared capture owns
scalar/alias, support and absent-source admission. Positive loss, unequal or
zero capacity and extra support reject; this adapter never changes the law.

The `SinePathMemoryAssessment` report separates two uses of the captured state:

- `nonlinear_edge_storage_bounds` and `nonlinear_edge_rate_bounds` describe
  actual incident-edge storage and its instantaneous work in the original
  structural clock. Endpoint order follows graph capture; `path` and
  `path_indices` retain left, middle, right. `nonlinear_transfer_bounds`
  encloses the left edge's rate, whose exact negative is the right edge's
  rate. The two `nonlinear_edge_rate_residual_bounds` compare these formulas
  with the independent chain-rule work. Total continuous loss is exactly zero.
- `coordinate_memory` and `linear_observation` concern the separately
  declared uniform-form/uniform-phase reference tangent. The supplied origins
  are admitted through shared exact-or-represented-real admission. The report
  does not infer this target from the graph or linearize about a moving state.
  `initial_tangent_state` contains all form differences from the reference,
  followed by all chosen real phase-lift differences, in captured node order.
  No error estimate identifies its linear evolution with the nonlinear future.

The exact clock is `tau=w*t/pi`; `clock_rate_pi_numerator` and
`clock_rate_bounds` retain that conversion. In this clock, the consensus
generator is `[[0,-M],[M/beta,0]]`, where `M=nu*D^-1*L` uses the complete
path degrees. Its entries are exact rationals for admitted captured values;
mathematical pi has not been replaced by a floating-point coefficient.
The existing coordinate-memory owner partitions endpoint forms then endpoint
phases as visible, retaining middle form and phase as hidden.

With `omega_squared=hidden_frequency_squared=nu**2/beta`, the exact kernel is
`cos(omega*tau)*K0+sin(omega*tau)*K1/omega`. `K0` belongs to
`coordinate_memory.kernel_at_zero`; `K1=kernel_derivative_at_zero` is its
normalized-time derivative. Hidden initialization contributes the analogous
combination of `hidden_initial_forcing=B*h0` and
`hidden_initial_forcing_derivative=B*D*h0`. Omitting that term changes the
preparation. No kernel, exponential or memory convolution is numerically
propagated by this static report.

The shared minimal-observation construction requires all six tangent
coordinates to close the four endpoint observations. This is an all-state
linear statement, not a physical count of hidden constituents. The full
tangent has a return period enclosed by `full_tangent_period_bounds`,
`2*pi**2*sqrt(beta)/(w*nu)` in structural time. A particular state can be
stationary or have a shorter fundamental period; the exact
`tangent_motion_nonstationary` flag checks its generator action. Neither
that flag nor the period bound certifies a nonlinear pulse.

`to_dict()` uses schema `tnfr.relational-sine-path-memory.v1`; generic SDK
export delegates to this same projection. The
[proof](../../../theory/nodal/RESONANCE_FOUNDATIONS.md#finite-conservative-memory)
derives regional transfer, the periodic memory and the obstruction to exact
irreversible finite tangent reduction. A finite-window damping approximation,
microscopic zero-loss selection and universal fractality remain separate
questions. No graph mutation, external drive or runtime-default change occurs.

<a id="sine-nonlinear-recurrence"></a>
### Full nonlinear recurrence: family admission and individual limits

`assess_sine_recurrence(graph, reference_model=..., energy_ceiling=...,
form_mean_bounds=(lower, upper))` uses the shared
[resonance owner](../../../src/tnfr/physics/relational_sine_resonance.py).
It admits complete finite connected simple unit support with at least two
nodes, strictly positive held capacities, explicit zero EPI loss weight and
the shared absent-input/event premises. It applies to the full nonlinear
sine law, with real form and circular phase, without an equilibrium target
or small-amplitude assumption.

The caller supplies a strictly positive energy ceiling and a strictly
positive-width mean interval. Shared exact-or-represented-real admission
rejects Booleans, nonfinite values and unordered/invalid bounds. These values
select the analyzed family; they do not add parameters to the dynamics.
The family contains every state with `E<=energy_ceiling` and weighted form
mean in that interval, at the captured support, capacity and law coefficients.
It is not a Cartesian product of independently bounded phase and form data.

`SineRecurrenceAssessment` retains exact weights `rho_i=d_i/nu_i`, their
normalization and `weighted_form_mean`. Its
`weighted_mean_rate_residual_bounds` is computed from the shared form rates,
and `comparison.storage_rate` retains the computed work enclosure. The exact
divergence identity is `-e*sum(nu_i)`, hence zero in the admitted law. Interval
residuals checking zero do not by themselves prove either invariant.

The report uses `path_length_upper_bound=n-1` to enclose
`R=sqrt(2*(n-1)*energy_ceiling)` in `form_radius_bounds` and supplies the
family coordinate box `[lower-R,upper+R]` in `form_coordinate_bounds`.
This conservative connected-path bound needs no spectral computation or
all-pairs distance search. Compactness includes the phase torus, not arbitrary
unwrapped real phase lifts. Tiny exact positive capacities and slab widths
remain positive in admission even if an outward projection touches zero.

The report keeps three conclusions separate:

- `almost_everywhere_recurrence_certified` and
  `finite_positive_family_volume_certified` apply to the declared family in
  ambient form Lebesgue measure times phase-circle Haar measure.
- `snapshot_family_membership` is `inside`, `outside` or `unresolved`, with
  explicit reasons. Exact mean comparison and, for identical captured phases,
  `captured_energy_exact` can resolve boundaries that projected energy cannot.
  Otherwise overlapping energy bounds remain unresolved. An outside snapshot
  does not invalidate the separately admitted family theorem or inherit its
  form box.
- `snapshot_motion_status` is `stationary`, `nonstationary` or `unresolved`.
  A nonzero exact form Laplacian or a sign-resolved sine current proves motion.
  Uniform captured phases and zero form Laplacian certify a stationary case;
  these are sufficient checks, not an exhaustive equilibrium classifier.
  `individual_recurrence_status` is `trivial_stationary_recurrence` for that
  stationary case and `unavailable_for_chosen_state` otherwise.

In particular, confirmed family membership and known motion do not imply an
individual recurrence certificate. The
[proof and exact separatrix counterexample](../../../theory/nodal/RESONANCE_FOUNDATIONS.md#nonlinear-recurrence)
explain why this distinction is necessary. No return deadline, exact period,
attracting amplitude or preserved pattern identity is reported. A finite
numeric grid or seed distribution does not automatically inherit a theorem
for absolutely continuous ambient preparations; fixed-energy and fixed-mean
surfaces need their own measure argument.

The [joint-episode theorem](../../../theory/nodal/SINE_PAIR_GROUPING.md#sine-joint-recurrent-episodes)
combines this family result with a separately proved open acquisition region
on the same doubled-C5 law. It establishes repeated finite matching episodes
for almost every preparation there. This reader alone does not admit that
region, infer an identity predicate, or certify any captured state's episodes.
Leaving a collective-coordinate chart is not itself loss of joint matching.

`to_dict()` uses schema `tnfr.relational-sine-recurrence.v1`; generic SDK
export uses the same projection. The reader executes no trajectory, changes
no runtime law and does not reinterpret a dissipative recovery report.
