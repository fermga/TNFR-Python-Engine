# Conservative sine saddle preparations and certificates

Signed-involution preparation, the C5/private-leaf saddle, directed passage, same-orbit retention, sensitivity, retained-metric forecasts and constitutive controls.

Part of [Relational dynamics contract index](../RELATIONAL_DYNAMICS.md). Section links remain stable; hypotheses and model changes remain local to each result.

<a id="sine-directed-saddle-corridor"></a>
### Directed nonlinear passage through the principal phase seam

`assess_sine_saddle_corridor(source, *, cycle, lower_phase, upper_phase)` in
`physics/relational_sine_corridor.py` returns `SineSaddleCorridor`. It reads
the complete conservative unit-capacity C5/private-leaf source and reuses
exact signed-involution admission. The ordered source must belong to the
real-lift family `(a,b,0,-b,-a)` and `(u,v,0,-v,-u)`, with the matched leaf
rows retained. The proof uses the scaled clock `tau=t/pi`.

The declared strip has
`-2*pi/3 < lower_phase < u < -pi/2 < upper_phase < 0`. A source with winding `+1`
and four nonclosing receiver gaps strictly inside their principal branches
can receive a directed-exit certificate when the rebuilt full storage is
below four, the energy-controlled conjugate force is strictly positive,
and its initial momentum exceeds the lower-face exit allowance. The owner
uses exact represented-real admission and outward rational intervals; a
failed sufficient comparison remains explicitly unavailable.

The momentum `6*a+3*b+2*c+d` includes the leaf forms. It is not just the
closing phase rate. The shared full storage bounds both this momentum and
the transverse motion, giving a finite residence-time upper bound. The
lower-face condition rules out the wrong exit; the remaining exit is beyond
the principal seam with winding zero. The proof does not assume monotone
closing phase, aligned contacts throughout the motion, or a tangent law.

The source, phase coordinates, energy, momentum, force bound, exit margin
and time bounds remain in the report. Cached input bounds are rebuilt from
primitive state and law. The direct schema is `tnfr.sine-saddle-corridor.v1`;
the SDK delegates projection and atomic export to the shared owners.

This is an exact-family finite passage theorem, with no trajectory evaluated
and no independent full-state uncertainty box certified. Its initial
unit-winding state is outside the wider protected region and is nonacute.
Reversing the motion supplies a zero-to-unit passage with the same limitation;
it does not alone establish acute retention. The same-orbit
[formation theorem](../../../theory/nodal/SINE_CONSERVATIVE_PREPARATION.md#sine-conservative-formation-retention)
adds the required inner passage and retained band. See the
[proof](../../../theory/nodal/SINE_CONSERVATIVE_PREPARATION.md#sine-directed-saddle-corridor)
and [workflow](../../guides/relational/SINE_SADDLE_CERTIFICATES.md#sine-directed-saddle-corridor).

<a id="sine-conservative-formation-retention"></a>
### Same-orbit formation and finite retention

`assess_sine_saddle_retention_band(source, *, cycle)` and
`assess_sine_saddle_formation(source, *, cycle, epsilon=Fraction(1, 2**32))`
in `physics/relational_sine_corridor.py` return `SineSaddleRetentionBand`
and `SineSaddleFormation`. Both re-admit the complete conservative unit law
and C5/private-leaf support. Here the supplied comparison **anchors law and
support only**: its captured state is not asserted to be the mathematical
preparation, a hitting state or a target.

The band reader verifies a sufficient implication from an actual deep hit
`u=-5/2` on the admitted signed family, with conserved storage at most
`35001/10000` and retained path branches. The band `[-13/5,-12/5]` gives a
uniform acute margin greater than `1/12000` and a phase-coordinate speed
below `1/7`. From the deep hit, either band exit takes more than `7/10`
scaled time units in either direction. The target checkpoint is one half-unit
before the deep hit; its next full unit remains acute. Independent form and
phase errors of radius `2^-18` at every fine node use the complete-field
Lipschitz bound, including perturbations that break symmetry.

The formation reader admits `0 < epsilon <= 2^-32` through shared exact
scalar admission. Epsilon specifies a favorable preparation, not a new law
parameter. It rebuilds the saddle and band and checks the correlated recipe
`z_saddle + epsilon*(3*X_lambda, -v_lambda)`. Its exact eigenvector components
share one algebraic root; interval components cannot be selected independently.
Bounds at scaled times `+3` and `-3` place actual states of this one orbit in
both nonlinear corridors. Reversal connects the outer zero-winding endpoint
to the inner band under the same complete law.

`formation_existence_certified` and
`independent_source_ball_existence_certified` concern this analytical family.
The report retains energy and nonlinear error bounds, both directional gates,
connection-time ceiling `64/epsilon`, target radius and duration. The source
radius is represented exactly by `source_radius_prefactor` and
`source_radius_exponent`, meaning `prefactor * exp(-exponent)`, namely
`2^-19 * exp(-128/epsilon)`. It is positive but generally far too small to
materialize in a floating scalar. Do not replace it with zero, a rounded
positive floor or a practical tolerance.

The nominal source is the reversed outer hitting state; the checkpoint is
the same reversed orbit's deep-hit state evaluated one half-unit earlier.
Their numerical coordinates and hitting times are not returned. Accordingly,
`numerical_source_center_available`, `captured_source_formation_certified`
and `captured_source_retention_certified` remain false. A certified theorem
report is not a concrete preparation, source checkpoint or numerical forecast.
No trajectory is run and no frozen response is modified.

Direct schemas are `tnfr.sine-saddle-retention-band.v1` and
`tnfr.sine-saddle-formation.v1`. Shared SDK projection retains the exact
radius expression and validates nested labels; export does not authenticate
or turn the analytical recipe into observed state. The
[proof](../../../theory/nodal/SINE_CONSERVATIVE_PREPARATION.md#sine-conservative-formation-retention)
closes conditional finite formation/retention existence under supplied law
and support, with autonomous preparation, physical identification and an
operational source still separate. See the
[workflow](../../guides/relational/SINE_SADDLE_CERTIFICATES.md#sine-conservative-formation-retention).

<a id="sine-operational-saddle-preparation"></a>
### Rational preparation and conditional full-state sensitivity

`prepare_sine_saddle_state(source, *, cycle, epsilon=Fraction(1, 2**32))`
in `physics/relational_sine_corridor.py` returns `SineSaddlePreparation`.
The source anchors the admitted unit conservative law and C5/private-leaf
support. `prepared_state` contains a new exact rational form/phase state,
with every fine node retained and rates in original structural time.
The correlated algebraic recipe is enclosed, materialized with exact signed
symmetry, and compared to the actual rational center. The representation
error is propagated into both local corridor gates. Actual full-edge energy
and initial phase branches are recomputed; cached certificates cannot
substitute for these checks. Full and regional sine storage share the
128-bit interval cosine kernel, resolving the nearcritical energy excess
without changing the pressure or evolution law.

`preparation_certified` means the actual rational intermediate state passes
these sufficient conditions. It has winding one; the zero-winding source
coordinates and hitting times remain unavailable. No trajectory is evaluated.
An unresolved energy or local margin returns `unavailable`, even when the
ideal analytical family remains certified. Computational representation error
is not a physical preparation tolerance.

`assess_sine_saddle_sensitivity(source, *, cycle, phase_radius=Fraction(1, 1000))`
in `physics/relational_sine_sensitivity.py` returns `SineSaddleSensitivity`.
It rebuilds the exact saddle and a positive rational metric on all twenty
form/phase coordinates, including both conserved origins and perturbations
outside the signed family. Exact matrix inequalities certify tangent growth
at most `1/3` in both time directions. The nonlinear bound is
`1/3 + 28*phase_radius` in scaled time `tau=t/pi`, conditional on both
complete trajectories remaining within that primitive-phase radius of the
exact saddle throughout the compared interval. Radius admission is shared,
finite and nonnegative, excluding Booleans. This radius is a proof domain,
not a constitutive parameter.

The source again anchors law/support only. In particular,
`captured_source_flow_bound_certified` remains false: static metric admission
does not establish whole-time containment, handle a numerical residual or
advance a validated forecast. Norm conversions are supplied, but repeated
conversion to independent coordinate boxes loses the retained correlations.
The metric is an analysis norm, not a new physical storage law.

Direct schemas are `tnfr.sine-saddle-preparation.v1` and
`tnfr.sine-saddle-sensitivity.v1`. Shared SDK projection preserves exact
values and validates nested labels without authenticating a checkpoint.
The [proof](../../../theory/nodal/SINE_CONSERVATIVE_PREPARATION.md#sine-operational-saddle-preparation)
and [workflow](../../guides/relational/SINE_SADDLE_CERTIFICATES.md#sine-operational-saddle-preparation)
separate rational preparation, conditional error bounds and an actual forecast.

<a id="sine-saddle-metric-forecast"></a>
### Retained-metric full-state saddle forecast

`forecast_sine_saddle_metric(source, *, cycle, duration, time_step,
initial_coordinate_radius=0, phase_radius=Fraction(1, 1000),
growth_rate_bounds=None, growth_bisections=8, direction=1,
order=12, max_steps=256)` in `physics/relational_sine_metric_forecast.py`
returns `SineSaddleMetricForecast`. Here the re-admitted source is the actual
initial state, not just a support anchor. The law remains conservative unit
sine exchange on the complete C5/private-leaf support, with held unit
capacities and no forcing or events. The sensitivity metric and all theorem
premises are rebuilt from primitive state and law.

The source's twenty exact form/phase coordinates center an independent
coordinate-error cube of the supplied nonnegative radius. The cube is
enclosed in a positive-metric ball **once**. The shared numerical owner
`mathematics/_validated_metric.py` carries that ball's center and radius
between steps. Its coordinate projection serves strict Picard inclusion,
whole-time phase admission, Taylor remainders and output observations.
An output box never becomes the next metric radius. Complete center-solution
rounding and truncation error are added in the same metric, using a scaled
norm calculation to avoid losing small squared errors to the interval grid.

Both `duration` and `time_step` use **original structural time**. Exact
integer `direction=+1` or `-1` selects the forward or reversed autonomous
field, with positive elapsed duration. Every phase in every Picard tube
must remain strictly within `phase_radius` of the exact saddle phase lift;
neither wrapping nor a rounded saddle midpoint replaces that comparison.
The original-clock growth bound is `(1/3+28*phase_radius)/pi_lower`.

With `phase_radius=None`, the alternative declared numerical mode instead
rebuilds the full interval Jacobian on every admitted Picard tube. It retains
the same metric and complete sine law, which is smooth on all real phase
lifts. For `S=J.T*W+W*J`, the midpoint plus the diagonal row sums of interval
radii dominates every actual symmetric Jacobian. An exact positive-semidefinite
test of `2*gamma*W-majorant` certifies the whole-tube rate. The rational bracket
defaults to `(0, 1)`; `growth_bisections` admits 0--16 fixed refinements.
An insufficient upper endpoint returns unavailable, without bracket expansion.
These rates already use the original clock and receive no further pi scaling.
Supplying a growth bracket together with a finite phase radius is rejected.

`growth_mode` identifies the chosen method. Computed-mode steps retain their
Jacobian enclosures, rational majorant, declared bracket, selected rate and
matrix certificate. Its forecast-level `growth_rate_upper_bound` is the
maximum certified step rate, or unavailable if no step passed. This replaces
the small-phase-domain growth premise with a fresh whole-tube proof; it does
not select or alter the evolution law.

Positive finite time arguments, nonnegative coordinate error, Taylor order
1--16 and a positive integer step budget no larger than 256 are admitted
before execution. The shared scalar exponential additionally requires
`abs(growth_rate*time_step) <= 1` for each actual step. Numerical work limits
are not physical restrictions. A failed tube returns `unavailable` with the
last certified center/radius, `validated_duration`, failed enclosure and
reason. There is no automatic retry, radius enlargement or changed law.

Each step retains its Picard/domain evidence, propagated initial radius,
center polynomial and remainder, local metric error and endpoint ball.
`status="admitted"` certifies enclosure through the requested horizon; it
does not itself certify formation, winding change or retention. The distinct
schema `tnfr.sine-saddle-metric-forecast.v1` and SDK report type preserve this
scope. Existing consumers of the coordinate-box `SineForecast` do not accept
this report without a separate admission contract. Export is not checkpoint
authentication. See the [step proof](../../../theory/nodal/SINE_CONSERVATIVE_PREPARATION.md#sine-operational-saddle-preparation)
and [manufactured example](../../guides/relational/SINE_SADDLE_CERTIFICATES.md#sine-saddle-metric-forecast).

<a id="sine-metric-connection"></a>
### Two-direction same-orbit connection evaluation

`forecast_sine_metric_connection(source, *, cycle, duration, time_step,
initial_coordinate_radius, growth_rate_bounds=(0, 1), growth_bisections=8,
order=16, max_steps=256)` in `physics/relational_sine_metric_connection.py`
re-admits one complete source and runs two fresh metric forecasts from it,
with directions `+1` and `-1`. It uses whole-tube Jacobian growth admission,
not a phase cutoff. The normalized bracket and all preparation, clock and
work parameters are shared by both calls. A caller-supplied forecast or
favorable cached verdict cannot replace either computation.

The reader selects the first forward endpoint whose complete enclosure has
unique principal branches and winding zero. On the backward leg it selects
the first contiguous sequence of complete Picard tubes with strict acute
receiver gaps and fixed winding one, spanning at least `pi_upper` in original
time. Endpoint samples do not establish that continuous interval. Reversibility
then links the reflected forward state to the reflected backward window on
the same orbit; it changes forms' signs and leaves phases unchanged.

`SineMetricConnection` keeps both complete or partial forecasts, the selected
times, phase margins and independent horizon-completion flags. A valid prefix
can certify `same_orbit_connection_certified` even if a later step fails;
`declared_horizons_complete` still reports that failure. Missing observations
on complete horizons mean `no_certificate_on_declared_horizons`, not a global
formation obstruction. Incomplete coverage without the observations remains
`unavailable`.

The forward endpoint enclosure contains the image of the initial family,
but need not itself lie inside that image. Consequently,
`independent_zero_winding_source_ball_certified` remains false. Neither a
new independent source preparation nor an infinite lifetime follows from
these two forecasts. Direct schema `tnfr.sine-metric-connection.v1` and the
shared SDK retain these distinctions. The
[frozen-trial instrument](../../../benchmarks/sine_metric_connection.py) separates
preparation from evaluation and additionally requires both complete horizons
for its study-level pass. See the
[proof and source boundary](../../../theory/nodal/SINE_CONSERVATIVE_PREPARATION.md#sine-operational-saddle-preparation).

<a id="phase-storage-discriminator"></a>
### Local agreement and a different global phase barrier

`assess_saddle_storage_discriminator(source, *, cycle,
initial_coordinate_radius, epsilon=Fraction(1, 2**32))` in
`physics/relational_phase_storage.py` returns `SaddleStorageDiscriminator`.
The source anchors the admitted conservative unit C5/private-leaf support.
The reader freshly rebuilds the named rational saddle preparation; it does
not advance the captured source or accept a cached preparation verdict.

The report compares sine with one explicitly supplied smooth cutoff storage
law. Its pressure and phase storage change together; its distinct law tag
never installs that law in the native or sine runtime. The
[complete-law proof](../../../theory/nodal/SINE_CONSTITUTIVE_INFORMATION.md#local-phase-storage-nonselection)
owns its potential, companion rows, conserved storage and regularity.
The cutoff law is infinitely differentiable but not real analytic.

The nonnegative independent radius applies to every form and phase coordinate.
Shared circular admission bounds every initial support edge, and shared sine
storage bounds apply to the alternative only when the whole initial box lies
in the region where both potentials coincide. Outside that certified region,
alternative initial storage is unavailable. Invalid primitive declarations
reject before calculation; unresolved sufficient inequalities remain explicit.

The whole local window uses the bound
`b=729*(3*epsilon+preparation_error+initial_coordinate_radius)` and the strict
edge margin `pi_lower/12-2*b`. A positive margin proves that the two complete
flows agree for `|tau|<=3`, including every admitted perturbation. The two
directional gate flags concern the nominal signed preparation; arbitrary
perturbations need not preserve that symmetry or inherit its gate verdicts.

If the complete initial storage is strictly below the alternative seam cost
`4`, and the initial cycle branches are unique, conservation excludes a
winding change for both time directions. This is a conditional all-time
topological obstruction, not a trajectory, an all-time acute-identity theorem
or a statement that the alternative admits no formation from other sources.
The corresponding sine passage belongs to its unchanged retained evidence;
this reader neither reruns nor authenticates that response.

Exact report projection uses `tnfr.saddle-storage-discriminator.v1` through
`to_dict()` and the shared SDK. `sine_full_cube_passage_certified` remains
false: the nominal sine gates do not establish passage for every perturbed
member, and this reader does not consume the separate frozen sine response.
Source, prepared state, uncertainty, margins and separate claim flags remain
visible. An export is not a checkpoint or evidence authentication. See the
[static example](../../guides/relational/SINE_SADDLE_CERTIFICATES.md#phase-storage-discriminator).

<a id="sine-constitutive-robustness"></a>
### Same-source cubic-law acquisition obstruction

`assess_sine_constitutive_robustness(evidence_directory)` in
[`research/sine_constitutive_robustness.py`](../../../src/tnfr/research/sine_constitutive_robustness.py)
reads the fixed two-direction evidence bundle. It verifies retained file
identity, re-admits the consumed complete-law/support/state premises and
rebuilds the source winding, reference retention and storage observations.
It does not run a producer, reconstruct an alternative source or integrate
the changed law. File hashes establish byte identity, not mathematical truth
or authentication of the historical execution; validated-reference soundness
remains an explicit premise.

The comparison coefficient is fixed at `eta=1/100` for
`j_eta(delta)=sin(delta)+eta*sin(delta)^3`, with its own reciprocal storage,
unchanged phase row, held unit capacities and all twenty coordinates. The
source is exactly `R Phi_sine(237,z)`, with `R` reversing form. Its enclosing
endpoint box is used only to bound observations of that mapped family;
it is not an independently prepared rectangular source set.

The original initial enclosure bounds conserved sine storage `H0(z)`.
The retained source phases bound the added potential `A` on all ten edges.
Their sum `H0+eta*A` bounds the changed law's storage at the same source.
The [conditional barrier theorem](../../../theory/nodal/SINE_CONSERVATIVE_PREPARATION.md#sine-constitutive-robustness)
gives `B_eta=7/2+47*eta/24`: strict `H_eta<B_eta` excludes entry into either
acute unit-winding sector at any time from the source's zero winding.
This is not permanent zero winding, loss of an already formed identity,
nonformation from every source, or a physical selection of either law.

`SineConstitutiveRobustness.to_dict()` uses
`tnfr.sine-constitutive-robustness.v1`; the generic SDK exporter preserves
its exact rational evidence. Missing or inconsistent evidence cannot become
a passing zero observation. The original frozen `passed=false` is preserved
independently of the new obstruction. See the
[read-only example](../../guides/relational/SINE_SADDLE_CERTIFICATES.md#sine-constitutive-robustness).

<a id="sine-involution-reduction"></a>
### Exact signed involution reduction

`assess_sine_involution_reduction(source, *, permutation_indices, sign=-1)`
in [`relational_sine_symmetry.py`](../../../src/tnfr/physics/relational_sine_symmetry.py)
returns `SineInvolutionReduction`. It shares complete source, support,
capacity and permutation admission with the cycle-symmetry reader below.
The permutation must additionally be an involution and `sign` the exact
integer `-1` or `+1`, excluding Booleans. No group search is performed.

The fixed family obeys `x[p(i)]=sign*x[i]` and
`theta[p(i)]=sign*theta[i]` in the declared real phase lifts. The negative
sign reverses both state rows together; capacities keep their original sign.
Every two-node orbit retains one coordinate per row. For sign `-1`, fixed
nodes have zero form and phase; for sign `+1`, their coordinates remain free.
The report exposes representatives, the exact reconstruction matrix and
captured-source residuals. It does not implicitly recenter or wrap phases.

`family_invariance_certified` requires a full-support automorphism and
preserved held capacities. `source_membership_certified` independently checks
the actual captured form and phase. Both are needed for
`source_trajectory_reduction_certified`; `status="certified"` alone describes
the family, not membership. Nonpreserved support or capacity makes the family
unavailable. Malformed primitives, signs or noninvolutive permutations reject.

`report.evaluate(form_coordinates, phase_coordinates)` rebuilds the report's
primitive law and symmetry before reconstructing a complete detached state.
It returns `SineInvolutionState`, retaining the shared full comparison,
storage and representative rates in **original structural time**. It does
not evolve or project the captured source. Cached favorable flags, matrices
and residuals cannot replace admission. Reported residual intervals check
arithmetic; exact row equality follows from equivariance, not interval overlap.

An arbitrary independent uncertainty box around a symmetric center need not
lie in this family. No transverse stability, attraction or formation is
implied. In particular, combined sign/reflection preserves winding and does
not inherit the pure-reflection zero-winding obstruction below. This symmetry
is distinct from conservative time reversal, which reverses form alone.
Direct schemas are `tnfr.sine-involution-reduction.v1` and
`tnfr.sine-involution-state.v1`; shared SDK projection and atomic export apply.
See the [proof](../../../theory/nodal/SINE_CONSERVATIVE_PREPARATION.md#sine-involution-saddle-reduction)
and [workflow](../../guides/relational/SINE_SADDLE_CERTIFICATES.md#signed-involution-reduction).

<a id="sine-c5-leaf-saddle"></a>
### Conservative C5 saddle and its full tangent

`assess_sine_c5_leaf_saddle(source, *, cycle)` in
[`relational_sine_resonance.py`](../../../src/tnfr/physics/relational_sine_resonance.py)
returns `SineC5LeafSaddle`. It reuses complete private-leaf admission:
an ordered C5, exactly one private leaf per receiver, unit capacities,
zero loss and unit exchange/storage coefficients. Extra edges, a different
law or omitted environment reject. The captured source anchors support and
law; it is not asserted to occupy the hypothetical saddle.

The target has zero form and receiver/leaf phases `(i-2)/6` exact turns.
Exact circular reconstruction checks its sine balance. Four cycle curvatures
are `1/2`, the closing curvature is `-1/2`, and contacts have curvature one.
The full Laplacian, phase Hessian, tangent and quadratic energy use the same
shared assembly as existing bridge assessments. Their clock is `tau=t/pi`.

The signed involution supplies an invariant odd reconstruction. This adapter
orders its four coordinates positively at cycle positions 0 and 1 and their
leaves, independently of source node order. It derives the reduced blocks
`theta'=A*x`, `x'=-B*theta` from full matrices and verifies their intertwining
identities. Full relative phase inertia is `(8 positive, 1 negative, 0 zero)`;
the conservative tangent has one growing/decaying real pair, eight oscillatory
pairs and two full-state origin modes. Positive-loss stability verdicts are
not reused at zero loss.

The characteristic polynomial is recomputed from the exact reduced matrix.
Its sign variation and exact rational endpoint signs isolate the unique
positive root; `growth_rate_bounds` bounds its square root. No floating
eigensolver supplies a certificate. `unstable_direction_bounds` and its
cycle-gap image enclose one correlated algebraic eigenvector defined by
that root, not every independently selected member of the displayed intervals.
The target remains at winding `+1`: crossing its `2*pi/3` protected-region
boundary is distinct from reaching winding zero or an acute target.

The report is a static conditional assessment, not a solver, event or
formation certificate. Schema `tnfr.sine-c5-leaf-saddle.v1` supports shared
SDK projection and atomic export. The
[proof and nonlinear remainder](../../../theory/nodal/SINE_CONSERVATIVE_PREPARATION.md#sine-involution-saddle-reduction)
state how far a tangent comparison can legitimately be used.
