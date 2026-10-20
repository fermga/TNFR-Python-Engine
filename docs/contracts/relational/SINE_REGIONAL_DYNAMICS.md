# Conservative sine regional and collective dynamics

Regional storage and currents, conservative winding acquisition and retention, compatible collective families, actual contact feedback and forecast-derived organization.

Part of [Relational dynamics contract index](../RELATIONAL_DYNAMICS.md). Section links remain stable; hypotheses and model changes remain local to each result.

<a id="sine-regional-storage-balance"></a>

### Regional storage and two-channel boundary work

`comparison.regional_storage_balance(region=...)` returns
`SineRegionalStorageBalance`. Its shared owner re-admits primitive support,
model, phase, form and held capacity, then rebuilds the entire comparison.
Cached rates, gradients, resultants, dissipation and storage cannot substitute
for those premises. It admits nonnegative capacities without division by them.

The ordered nonempty region contains distinct source nodes. Internal region,
internal complement and cross-boundary edge sets partition storage exactly
once. `*_form_storage` are exact fractions; `*_phase_storage` enclose the unit
cosine potential; `*_storage` includes the model's `beta` multiplier.
Boundary edges are oriented `(region,complement)` in source-index coordinates.
A full-support region has an empty complement and cut.

All rates use complete-network degrees, gradients and the original clock
`clock="structural_t"`. The regional identity is
`dE_R/dt=regional_boundary_work-regional_loss`. Work retains both form and
phase channels. The complement has its own identity; cross-edge storage
changes by the negative sum of the two inward boundary powers. Consequently
those powers need not be opposites. `direct_*_storage_rate` are computed from
edge derivatives independently of the boundary-work sums. Balance and
partition residual intervals expose arithmetic agreement, not a new theorem.

This storage ledger differs from the following weighted-form transfer.
A positive instantaneous power is not accumulated work, monotonic regional
storage, an external drive or evidence of pattern formation. The
[derivation](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#sine-regional-storage-balance)
and [controls](../../../tests/physics/test_sine_regional_storage_balance.py)
retain that distinction. Direct schema
`tnfr.relational-sine-regional-storage-balance.v1` and the generic SDK envelope
preserve the rebuilt comparison, edge partition, exact fractions and intervals.

`regional_channels` and `complement_channels` add the separate storage rows
under the same admitted general coefficients. Their internal conversion `J`
is positive from phase to form. `boundary_form_input` and
`boundary_phase_input` are the inputs to the respective internal stores;
they are **not** the existing conjugate `*_boundary_form_work` and
`*_boundary_phase_work`. `signed_form_loss` may be negative and is distinct
from the nonnegative full-node loss. Phase rates of storage include `beta`.
The direct rates satisfy `F_dot=J+B_F-D_F` and `(beta*V)_dot=-J+B_V`;
independently reconstructed residuals retain interval uncertainty.
Nested gradient/current/rate rows follow their `region_indices` order.
`flat_phase_storage_acceleration` is available when each internal edge has
an exactly zero represented phase gap. It encloses
`beta*sum_internal(theta_dot_j-theta_dot_i)^2` in original time; otherwise
it is `None`. This conservative flatness admission does not infer equality
modulo an unrepresented exact multiple of pi or discard uncertain angles.
The curvature describes an initial response, not finite acquisition. The
[channel derivation](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#sine-regional-channel-accessibility)
and [independent controls](../../../tests/physics/test_sine_regional_channel_balance.py)
own the coefficient and sign checks. The direct schema remains v1 with
additive nested fields; manually constructed older reports may retain `None`
there and supply no channel evidence.

<a id="sine-regional-transfer"></a>
### Regional form transfer and supplied increments

`comparison.regional_transfer(region=...)` reuses a captured
`SineExchangeComparison`, without a graph reread. All held capacities must be
strictly positive. The fixed-support weights are `rho_i=d_i/nu_i`; the report
uses weighted form sums, not arithmetic means or physical mass. The
[boundary identity](../../../theory/nodal/SINE_PAIR_INTERACTION.md#sine-autonomous-regional-transfer)
holds for both zero and positive form loss in this law. It does not transfer
to native Arg pressure or state-dependent reciprocal mobility.
Both readers use [chained-report admission](SINE_COMPARISON_AND_INFERENCE.md#sine-chained-report-admission).
Both readers rebuild their complete comparison from admitted primitive state
before retaining it as nested evidence. An equivalent original comparison is
kept only after calculation on normalized values; altered cached rates or
storage cannot accompany a fresh ledger as if they were consistent evidence.
Regional accounting consumes rebuilt rates and currents. The increment
identity itself needs only forms, weights and supplied signed increments;
the additional rebuild keeps its attached full-state report consistent.

`SineRegionalTransfer` retains the source, supplied region and its complement,
weights, weighted form sums and the inward cut current. The diffusion and sine
contributions are separate. Internal edges cancel algebraically; each cut
edge contributes `e*(x_out-x_in)+(w/pi)*sin(theta_out-theta_in)` to the region.
Direct weighted sums of the rebuilt nodal rates and their cut residuals are
computed independently. `global_weighted_form_conserved` records the conditional
identity; an interval residual containing zero is a numerical consistency
check, not its proof. A regional current does not close the region's evolution
without the consumed boundary state or certify persistent identity.

No pair midpoint or nonzero regional resultant is required by this cut
reader. The [zero-pair-resultant theorem](../../../theory/nodal/SINE_PAIR_INTERACTION.md#sine-zero-resultant-restoration)
shows why cancelled collective form transport can coexist with internal
phase motion. Its exact antipodal preparation is symbolic; a graph storing
represented radians remains its own source. The cut reader reports current,
not a future activation time. With the theorem's zero loss, node-relative
`resultant_kinematics()` bounds nodal form acceleration after the declared
capacity/degree and phase-weight scaling; it does not
report that pair's phasor derivative or supply a midpoint at cancellation.
The separately admitted [global pair state](SINE_PAIR_DYNAMICS.md#sine-global-pair-state) retains
the complete unordered state through cancellation on the fixed unit doubled
C5; it does not broaden this comparison reader's model or input format.

`comparison.assess_form_increment(increments=...)` accepts a complete ordered
vector of finite signed increments in captured node order. It does not infer
unobserved changes or silently fill missing nodes. `SineFormIncrementAssessment`
retains exact weights, increments and before/after weighted form sums.
`closed_flow_endpoint_obstructed` is true precisely when the weighted change
is nonzero. Zero change leaves reachability unresolved;
`endpoint_reachability_certified` remains false. No phase endpoint, storage
budget, runtime event or trajectory is supplied by this necessary condition.
The test concerns a full endpoint in the same form chart; it does not establish
an obstruction on a quotient that discards common form origins.

The pair-Emission report's `form_increment(outcome=...)` accepts
`first_member`, `second_member` or `whole_pair` and delegates those actual
clipped/rounded proposals to the same increment owner. Other nodes have zero
increment because these three declared actions leave them unchanged. A
positive AL-only increment fails the global invariant; an increase in one
region with compensating change elsewhere need not fail it. Saturated or
rounded no-ops are not reported as positive injections.

Owner schemas are `tnfr.relational-sine-regional-transfer.v1` and
`tnfr.relational-sine-form-increment.v1`. Generic SDK export delegates their
payloads through `tnfr.relational-report.v1`. Source provenance and label
validation are retained. No observation mutates the graph, installs an
activation rule or identifies form with an independently measured quantity.

<a id="sine-conservative-winding-entry"></a>

### Finite conservative regional winding acquisition

`certify_sine_conservative_winding_entry(source, *, cycle, scaled_window,
edge_turn_offsets, source_error_bound=0)` in
[relational_sine_entry.py](../../../src/tnfr/physics/relational_sine_entry.py)
returns `SineConservativeWindingEntry`. This separate theorem requires an
exact `SineExchangeComparison` as its nominal center, common nominal phase
lifts, constant nominal form on the selected receiver cycle, unit capacities,
zero loss, `w=beta=1` and no forcing or events. All source nodes and edges
remain present; the receiver's initial phase velocities follow from the
environmental form and full degrees. `source_error_bound=epsilon` admits
independent initial errors of at most that nonnegative size in every form
coordinate and phase lift, including the environment. Phase errors use radians;
the same numerical bound applies in the declared form units. Capacity and
support remain fixed. A nonuniform nominal receiver is outside this contract;
receiver contrasts within the supplied uncertainty box are admitted.

`cycle` is an ordered simple cycle of source labels. `scaled_window=(a,b)`
contains admitted exact-or-represented real values with `0<a<b` in
`tau=t/pi`. `original_time_window_bounds` enclose `pi*a` and `pi*b` in the
source clock. Each supplied edge offset is a nonboolean integer in cycle
traversal order; it is a declared observation branch, not an executed event.

The full normalized law has `abs(x_i')<=1` and `abs(theta_i'')<=2`.
`initial_phase_velocity` is the freshly computed exact `K L x(0)`.
`whole_window_form_bounds` and `whole_window_phase_bounds` enclose the full
nonlinear solution; `cycle_raw_gap_bounds` use the tighter direct edge
expression with remainder `2*(b^2+epsilon*(1+2*b))`. True lifted cycle gaps telescope to
zero; independent points selected from these outer boxes need not do so.

After subtracting each declared `2*pi` multiple, strict whole-window
inclusion in `(-pi,pi)` certifies `certified_winding=-sum(edge_turn_offsets)`.
Initial winding is zero throughout the source box when `2*epsilon<pi`
is strictly certified; otherwise `initial_winding=None` and acquisition is
unavailable. Only certified initial zero winding and a certified nonzero final period set
`acquisition_certified=True` and `status="certified_finite_acquisition"`.
A certified zero period remains distinct from acquisition. Failed sufficient
branch bounds yield `status="unavailable"` and explicit reasons; they do not
prove that the actual trajectory fails. Malformed or unsupported premises
reject before certification.

`acute_margin_lower_bound` additionally bounds the distance of every
principal gap from `+/-pi/2`. `acute_acquisition_certified` requires both
nonzero certified winding and a strictly positive whole-window acute margin.
These additive fields do not strengthen a nonacute certificate or change
the existing branch status. An unavailable bound is not a failed trajectory.

The same reader exposes an explicit full-state box at time `a` through
`entry_form_bounds` and `entry_phase_bounds`. Its centers are `x(0)` and
`theta(0)+a*K*L*x(0)`; its analytic radii are `a+epsilon` and
`a^2+epsilon*(1+2*a)`, retained as `analytic_entry_form_radius` and
`analytic_entry_phase_radius`. Interval materialization can round endpoints
outward. `entry_form_radius` and `entry_phase_radius` therefore measure the
actual stored box around its exact centers. `entry_box_contains_source_flow` certifies containment
of every actual source member's endpoint. These centers are enclosure
references, not substituted actual states or held-pressure dynamics.
Every point of this independent box has continuation bounds for
`scaled_retention_duration=h=b-a` under the same law. They use
`entry_box_form_remainder_bound=entry_form_radius+h` and
`entry_box_phase_remainder_bound=entry_phase_radius+2*h*entry_form_radius+h^2`.
Its own `entry_box_cycle_principal_gap_bounds`, branch margin and acute
margin retain that materialization allowance. `entry_box_acute_retention_certified`
requires a nonzero period and strict margins from these bounds. The original
`whole_window_*` and `node_*` fields still bound actual source trajectories;
they must not silently substitute for this independently rounded target box.
Continuation is also separate from proving that the source initially lacked
the identity. The dyadic documented witness requires no entry-box rounding.

`initial_total_storage` retains the nominal-center value for compatibility;
`initial_storage_bounds` encloses the entire uncertain source. It is not an
energy bound for every member of the later rectangular entry box, which
can include unreachable states. Each actual trajectory retains its own
conserved origins and storage. No midpoint, shared-energy assumption or
cached source rate replaces these premises.

The [proof and frozen witness](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#conservative-regional-winding-entry)
certify nonacute winding on one finite interval. This does not transfer
positive-loss capture, acute equilibrium, attraction, indefinite maintenance
or complete NFR identity. The
[robust short-passage result](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#robust-conservative-passage)
uses the same owner to certify acute identity with nonzero source and target
widths. It does not establish entry into the separate invariant-family tube
or retention beyond its declared short interval. The
[regional storage ledger](#sine-regional-storage-balance) keeps its own
original-clock rates; a snapshot power is not the theorem's integrated work.
Direct schema `tnfr.sine-conservative-winding-entry.v1` and generic SDK export
retain both clock conventions, branch evidence, source and scope. See
[admission tests](../../../tests/physics/test_sine_conservative_winding_entry.py),
[independent full-law controls](../../../tests/physics/test_sine_conservative_winding_dynamics.py)
and the [usage](../../guides/relational/SINE_REGIONAL_DYNAMICS.md#conservative-regional-winding).

<a id="sine-conservative-source-geometry"></a>

### Environmental source geometry and acute entry

`analyze_sine_conservative_source_geometry(source, *, receiver)` returns
`SineConservativeSourceGeometry` from the same entry owner. It shares the
conservative common-phase admission above and requires uniform initial form
on the nonempty receiver. The receiver need not be a cycle. All environmental
nodes remain present, including those without direct receiver contact.

The exact map `T=-D_R^-1 A_RQ` uses full degrees and maps environmental form
contrasts relative to the receiver's common form to initial receiver phase
velocities. The report retains the primitive source, receiver/environment
order, map, relative map, exact ranks, reconstructed velocities, residuals
and identical-row groups. Relative rows subtract the first receiver row.
Empty environment and one-node receivers have explicit zero-dimensional
maps. Cached source rates or storage are not input evidence.

Full relative rank suffices to prescribe every initial relative velocity;
it is not necessary for one acute configuration. Identical rows establish
equal initial velocities, not an invariant partition or a later obstruction.
The [source-image theorem](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#sine-conservative-source-geometry)
combines a declared interior phase cell with the existing global nonlinear
remainder to certify finite acute entry. It supplies no autonomous source
preparation or indefinite maintenance. Direct schema
`tnfr.sine-conservative-source-geometry.v1` and the generic SDK exporter
retain the exact map and its scope. See the
[map controls](../../../tests/physics/test_sine_conservative_source_geometry.py),
[analytic protocol](../../../tests/physics/test_conservative_source_geometry_protocol.py)
and [usage](../../guides/relational/SINE_REGIONAL_DYNAMICS.md#conservative-source-geometry).

<a id="sine-conservative-handoff"></a>

### Acquired low storage and subsequent acute exit

`assess_sine_conservative_handoff(source, *, cycle)` returns
`SineConservativeHandoff` from the shared entry owner. Its admitted source has
common phase, uniform receiver form, the unchanged conservative unit-capacity
law and an induced C5 receiver. The actual full-degree phase velocities on
the ordered receiver must form an arithmetic progression with nonzero signed
step. Its magnitude is `omega` and its sign is `orientation`. This is an
exact restriction on a source class, not a fitted frequency or new phase law.

The reader bounds the actual acquired state at
`tau_star=(2*pi/5)/omega` using the global nonlinear remainder. Correlated
cycle errors telescope, giving the tighter phase-storage upper bound
`V5+10*tau_star^4`, separately from the form bound `10*tau_star^2`.
It retains strict acute-entry and regional-subbarrier predicates separately.
`entry_relative_phase_rate_bounds` enclose the actual five edge rates at
entry in `tau`, using their complete initial velocities and the global
acceleration bound. These rates can be large even with small internal storage.
At `tau_out=5/(3*omega)`, a lower bound on four oriented gaps can prove acute
exit. `handoff_obstruction_certified` requires all these predicates: acquired
low regional storage has not supplied retention through this deadline.

`unavoidable_boundary_work_lower_bound`, when available, bounds work reached
by the **first acute exit after entry**. It is not net work at the declared
endpoint; later return remains possible. The report retains both clocks,
exact interval evidence, source and unavailable reasons. Failed sufficient
inequalities do not establish failed formation or successful maintenance.
Sources outside the declared law, cycle or ramp class reject. Re-admission
rebuilds consumed state and rates instead of trusting cached source fields.

Direct schema `tnfr.sine-conservative-handoff.v1` and generic SDK export
preserve the conditional obstruction. The reader runs no trajectory, changes
no support and selects no source. See the
[proof](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#sine-conservative-handoff-obstruction),
[independent and adversarial controls](../../../tests/physics/test_sine_conservative_handoff.py)
and [usage](../../guides/relational/SINE_REGIONAL_DYNAMICS.md#conservative-handoff).

<a id="sine-phase-offset-partition"></a>

### Compatible phase-offset collective families

`assess_sine_phase_offset_partition(source, *, blocks, phase_offset_turns)`
returns `SinePhaseOffsetPartition` from the
[shared partition owner](../../../src/tnfr/physics/relational_sine_partition.py).
The source supplies an admitted full-support `SineExchangeComparison` under
the unit-capacity conservative law `e=0`, `w=beta=1`, with no forcing or
events. Its captured form and phase are a **law/support anchor**, not evidence
that the source already belongs to the proposed family. `blocks` declares
a partition of all fine nodes, with the shared software admission requiring
`2 <= number of blocks < number of nodes`. The mathematical theorem also
allows trivial partitions, which this API does not admit.
`phase_offset_turns` independently declares
the fixed offsets in source node order using integers or `Fraction` values.
Floating turns or rounded radians do not stand for these exact offsets.

The proposed family has block-constant form and phase equal to a free
collective phase plus its supplied fine offset. Admission compares external
normalized neighbor counts and complex phase sums within each block, and
requires zero internal sine current. Equality of internal cosine sums is
not required. These are joint full-law conditions for every collective
state, not observed agreement of rates at one snapshot.

`status` is `certified`, `excluded` or `unavailable`.
`invariance_certified` is true only for the first status. Exact recognized
cancellations can prove equalities; an interval excluding zero can prove
inequality. Neither a nonempty symbolic residual nor an interval containing
zero proves that an equality fails. An undecided transcendental identity
therefore stays unavailable. The report retains normalized counts, row
sine/cosine expressions and their bounds. Quotient coefficients are
available only for a certified family. Their form and phase couplings
remain separate; phase cancellation need not preserve the bare coarse
normalized-sine formula.

`partition.evaluate(collective_form, collective_phase_turns)` re-admits the
primitive source, partition and offsets and rebuilds the required family
evidence. A certified family then yields a detached `SinePhaseOffsetState`
for the supplied collective coordinates. `block_form_rates` and
`full_form_rates` retain rigorous enclosures; `block_phase_rates` and
`full_phase_rates` retain exact represented-real coefficients. Rates use
`tau=t/pi`; phase rates are radian derivatives, while collective phase inputs
are integer or `Fraction` turns. Evaluation reconstructs
the fine state, inherited collective response and storage; it does not
advance a solver, mutate a network or trust cached certification fields.
Zero-containing rate residuals do not replace the invariant-family proof.
Direct exports use `tnfr.sine-phase-offset-partition.v1` and
`tnfr.sine-phase-offset-state.v1`; generic SDK export preserves the same
premises and availability.

Internal regional storage and net internal regional power are constant and
zero-rate, respectively, along the exact family. Collective motion and
cross-edge interaction can still be nonzero. Invariance supplies neither
autonomous formation into the family nor robustness to states outside it;
smooth uniqueness excludes finite-time exact entry from outside under the
same unchanged autonomous law. See the
[theorem](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-phase-offset-partition)
and [standalone usage](../../guides/relational/SINE_REGIONAL_DYNAMICS.md#phase-offset-partition).

<a id="sine-weighted-phase-offset-partition"></a>

### Weighted collective families and capacity resets

`assess_sine_weighted_phase_offset_partition(source, *, blocks,
phase_offset_turns)` in the
[weighted partition owner](../../../src/tnfr/physics/relational_sine_weighted_partition.py)
accepts a `SineSupportExchangeComparison` with zero loss, unit exchange and
beta, strictly positive conductances and strictly positive held capacities.
Its support is finite, connected, simple and undirected. This separate
extension does not broaden the unit-law reader above or install a runtime.
The partition and exact rational-turn offsets retain the same ordered
admission rules. Captured source coordinates anchor the complete law;
membership in the proposed family is a separate assertion.

The criterion compares external weighted sums multiplied by `nu_i/d_i`
for the phase row, and external bare complex phase moments multiplied by
the same factor for the form row. Internal sine currents must vanish;
internal real moments need not agree. Positive unequal capacities can
compensate unequal bare degrees. The original structural clock `t` is used
for every returned rate, including the `1/pi` factor in both rows. Offsets
and the original `evaluate` phase inputs are turns; angular rates are radian
derivatives. The separate zero-offset radian interface below retains real lifts.

The returned `SineWeightedPhaseOffsetPartition` retains separate form and
phase coupling data, reciprocal-capacity masses and inherited storage.
Recognized symbolic identities certify equality, an excluded zero proves
inequality, and undecided transcendental identities remain unavailable.
This numerical admission is not a complete decision procedure for the
mathematical necessary-and-sufficient criterion.
`partition.evaluate(collective_form, collective_phase_turns)` reconstructs
both fine and collective rows, storage and means after re-admitting the
primitive source and rebuilding its criterion. Cached derived fields and
verdicts cannot stand in for those premises. No time step is taken.

<a id="sine-weighted-radian-state"></a>
`partition.evaluate_radians(collective_form, collective_phase_radians)`
returns `SineWeightedRadianState` for a certified family with every phase
offset exactly zero. It admits each real form and phase before arithmetic;
phase inputs remain real radian lifts. Nonzero offsets, including integer
turn offsets, reject instead of silently changing a lift or its mean. The
existing rational-turn evaluator remains available for its own domain.

`partition.observe_radian_state()` additionally requires the captured
source to have exactly equal forms and equal radian lifts within each block.
It does not average or project an off-family state. Both methods re-admit
the primitive source and rebuild the exact partition criterion. The explicit
evaluator can construct a different hypothetical member; its
`source_membership_certified` flag is true only when that member's complete
form and phase coordinates equal the admitted captured source. This is
coordinate agreement, not provenance authentication or evidence of acquisition.

The radian state retains the fresh `partition`, collective coordinates and
`full_source`. The latter uses the general weighted-support owner to rebuild
every fine row and storage value on the lifted coordinates. Independent
collective rows use the inherited cross-edge counts `N_ab`, conductance sums
`C_ab` and block weights `M_a=sum(d_i/nu_i)`:
`X_dot_a=sum_b N_ab*sin(Theta_b-Theta_a)/(pi*M_a)` and
`Theta_dot_a=sum_b C_ab*(X_a-X_b)/(pi*M_a)`.
`quotient_bare_rows` retains the first channel's coefficients. Replacing
`N_ab` by simple adjacency generally changes the law; changing one scalar
capacity per block cannot compensate unequal incident multiplicities.

Block rates, exact phase-rate numerators/residuals, form-rate residual
intervals, quotient storage and weighted form/radian means are retained.
`full_row_equality_certified` follows from the exact all-state criterion;
zero-containing numerical residuals are arithmetic checks, not its proof.
`full_source` retains fine node order, support, capacities and the original
clock. Its identity and finite-response consumers keep their separate domains.
Direct schema `tnfr.sine-weighted-radian-state.v1` and generic SDK export
rebuild from the partition and collective coordinates, discarding cached
membership, rows, storage and means. See the
[canonical contact result](../../../theory/nodal/SINE_WEIGHTED_STAR_CANONICAL_CONTACT.md#weighted-star-collective-state-result)
and [radian state controls](../../../tests/physics/test_sine_weighted_radian_state.py).

### Primary capacity resets of a weighted collective family

`assess_sine_weighted_capacity_reset(partition, *, capacity_after)` consumes
positive capacities that are exactly representable as binary64 outputs,
in source node order. It rebuilds the pre-reset family and post-reset law
and compares the actual capacity ratios within each block. It does not
execute SHA or infer its configured factor. A common ideal factor can
produce unequal represented ratios, so ideal real multiplication cannot
replace the supplied outputs. Every capacity-only reset leaves the current
form/phase family membership unchanged; preservation of its subsequent
complete field is a separate condition.

These report types support direct and generic SDK export through their
owner's rebuilding path. A primary capacity-map certificate does not admit
a complete grammatical event: target-associated histories, debt, live
preconditions, lifecycle records, configuration and execution path must also
be supplied. Neither exact collective closure nor zero event storage work
establishes attraction, formation, autonomous occurrence or physical identity.
See the [proof and action boundary](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-weighted-phase-offset-partition)
and [standalone example](../../guides/relational/SINE_REGIONAL_DYNAMICS.md#weighted-phase-offset-partition).

<a id="sine-weighted-star-transverse-band"></a>

### Linear transverse stability of the supplied weighted star

`assess_sine_weighted_star_transverse_band(*, capacity_scale_bounds,
amplitude_bound)` in the
[transverse owner](../../../src/tnfr/physics/relational_sine_weighted_transverse.py)
returns `SineWeightedStarTransverseBand`. This is an analytical family reader,
not admission of an arbitrary graph or a claim about the captured source in
the preceding example. The fixed five-node support, internal conductance 3,
cross conductance 2, zero offsets, conservative law and capacities
`(4s,2s,2s,2s,8)` are part of its model. It accepts ordered positive scale
endpoints and `0 <= amplitude_bound < 1`, the upper bound on
`k=sin(Delta_max/2)` for collective librations about consensus. Rational
inputs remain exact; other real inputs use shared represented-real admission.
Booleans, nonfinite values, unordered or nonpositive scales and separatrix
or rotation amplitudes reject before interval calculation.

The default band is `23/25 <= s <= 1`, `0 <= k <= 1/4`. Two reported modes
have internal-star eigenvalues 1 and 4, with spatial multiplicities 2 and 1:
they exhaust the six transverse dimensions. The four collective dimensions,
including conserved means and possible phase drift between neighboring
collective orbits, are separate. Every angular rate and clock change follows
the complete two-row linearization. Period bounds use
`tau=sqrt(2)*(s+8)*t/pi` and also retain the original structural clock `t`.
At zero amplitude these are limiting periods, not measured periods of an
equilibrium; positive constant oscillators separately prove its transverse
stability.

The reader bounds each full-period half trace by a constant-rotation
comparison plus a rigorous exponential error. `rotation_error_upper_bound`
bounds the operator 2-norm difference in coordinates `(r*y,y')`, where
`r=sqrt(alpha*(lambda+1))` and the derivative uses normalized time; it also
bounds the half-trace difference. Unit monodromy determinant is
derived analytically. A positive `strict_trace_margin=1-abs_max(half_trace)`
certifies bounded infinitesimal transverse motion. A wider valid band can
return `unavailable`; unresolved trace bounds do not prove instability.
The fixed outward precision can also leave a positive denominator or
frequency unresolved, and the exponential calculation has a declared
unit-argument work domain. These conditions remain explicit abstentions.
At `amplitude_bound=0`, the separate constant-oscillator result can certify
stability even when the return-trace enclosure is inconclusive.

Direct schema `tnfr.sine-weighted-star-transverse-band.v1` and generic SDK
export rebuild every field from the admitted scale endpoints and amplitude
bound. No supplied cached mode, support, margin or verdict is trusted.
The interval certificate can be wider than the proof's independent rational
inequalities. No trajectory, monodromy integration or SHA event is performed;
no finite error radius, nonlinear stability, attraction, formation or physical
identification is established. See the
[proof](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-weighted-star-transverse-band),
[controls](../../../tests/physics/test_sine_weighted_star_transverse.py) and
[usage](../../guides/relational/SINE_REGIONAL_DYNAMICS.md#weighted-star-transverse-band).

<a id="sine-weighted-star-state"></a>

### Sufficient continuous state of a marked organization and its boundary

`assess_sine_weighted_star_state(*, form, phase_turns, capacity_scale=1)`
in the [state owner](../../../src/tnfr/physics/relational_sine_weighted_state.py)
returns `SineWeightedStarState`. It admits five signed real forms, five
exact rational-turn phase lifts and a positive held capacity scale. The
fixed model is the same weighted star: marked group `A=(0,1,2,3)`, center
0, and dynamic boundary node `B=4`. Its support, weights and capacities
are shared with the transverse owner. This mathematical state is neither
an imported graph snapshot nor a newly formed organization.

The reader computes two form means and two lifted phase means, plus four
zero-sum internal deviations in each channel. These are ten independent
coordinates: four collective and six internal. The complete nonlinear
field is reconstructed from them and checked against independently
assembled fine-edge rows. `reconstruct_sine_weighted_star_state(*,
collective_form, collective_phase_turns, internal_form, internal_phase_turns,
capacity_scale=1)` inverts this chart. Its internal entries must sum exactly
to zero; invalid entries reject rather than being silently recentered.
Both functions retain rational form inputs and use shared represented-real
admission otherwise. Complex, Boolean and nonfinite scalars reject; phase
lifts require integers or Fractions, not rounded floating turns.

All pressure and rate bounds use original structural time `t`. Phase rates
are radians per unit `t`; turn rates would require another division by
`2*pi`. `pressure_numerator_expressions`, `form_rate_numerator_expressions`
and `phase_rate_numerators` omit the common `1/pi` factor. Pressure is
computed from bare neighbor sine sums and bare degree independently of the
form response. Collective form rates average the nodal products; they are
not generally the product of mean capacity and mean pressure.

The moment is `M=mean(exp(i*eta))` for the internal radian phase deviations.
Its real/imaginary parts and derivatives remain defined at `M=0`. The latter
consume mixed internal form/phase information. Neither collective means
and full storage, nor those quantities augmented by `M`, are sufficient
state in general; the [matched-state proofs](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-weighted-star-state-sufficiency)
exhibit exact derivative disagreements arbitrarily near the coherent family.
The explicit internal coordinates supply a sufficient extension without
claiming a minimal reduced state or deriving a nonlinear memory kernel.

Full storage and weighted means use every fine node. Their instantaneous
balance identities are checked symbolically. `collective_reference_storage_bounds`
is the coherent-family expression evaluated at the actual means; it need
not be conserved off that family. `storage_remainder_bounds` is its signed,
lift-dependent difference from full storage, not a nonnegative reservoir.
The reference-storage and remainder-rate bounds retain their opposite
instantaneous exchange under the conserved full storage.
Individual full-turn changes of phase representatives can change the means
and moment; their joint transformation preserves the fine field and storage.
No principal-angle replacement or resultant angle is inferred.

Direct schema `tnfr.sine-weighted-star-state.v1` and generic SDK export
re-admit primitive scale, forms and phase lifts and rebuild all derived
fields. Exact row reconstruction establishes the implemented chart identity,
not finite-error retention. The report does not transfer the transverse
linear certificate to arbitrary finite states, infer amplitude from full
storage off the family, or admit an operator. Actual histories, target
associations, stored/refreshed pressure, grammar, lifecycle, configuration
and execution path retain their [existing owners](../OPERATOR_EVENTS.md).
See the [example](../../guides/relational/SINE_REGIONAL_DYNAMICS.md#weighted-star-state)
and [independent controls](../../../tests/physics/test_sine_weighted_star_state.py).

<a id="sine-weighted-star-retention"></a>

### Finite-amplitude retention of the marked internal identity

`assess_sine_weighted_star_retention(state)` from the same state owner returns
`SineWeightedStarRetention`. It re-admits the primitive state and rebuilds
its full storage; cached margins or derived fields are never premises.
The chosen identity requires each center-leaf form contrast to be strictly
less than `1/2` and each internal principal circular phase gap to be strictly
less than `1/4` turn. These margins define this study's identity, not universal
NFR thresholds. The environment node remains dynamic.

The exact minimum form storage at a center-leaf contrast `d` is
`35*d^2/16`; the form boundary therefore costs `35/64`. Every phase boundary
costs at least one in the full nonnegative phase storage. A strict outward
upper bound `H < 35/64` certifies the identity for all real times of the
unchanged complete conservative law. Global existence and exact storage
conservation are theorem premises established for that law, not extrapolated
from samples. This is finite-amplitude identity retention, distinct from
linear stability or tracking one reference orbit.

`source_identity_admitted` independently checks the supplied current
contrasts. `storage_margin` is `35/64-H_upper`. Nonpositive or unresolved
margins give `unavailable`, even when the current identity is admitted.
The report also bounds each internal form contrast by `sqrt(16*H_upper/35)`
and each internal phase cosine below by `1-H_upper`. Full-turn changes of
phase representatives preserve this circular certificate. A negative
reference-storage remainder is not a rejection or a usable extra reserve.

The [proof](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-weighted-star-identity-retention)
gives a uniform nonzero neighborhood of every coherent reference with
`H <= 1/2`: errors at every fine node of at most `1/256` in form and
`1/2048` turn in phase remain inside the certified storage sublevel.
Those preparation-error bounds are an analytical theorem; this point-state
reader does not infer that a supplied uncertain preparation satisfies them.
The same proof separates a capacity-only primary reset's zero storage change
from actual operator admission and from preservation of the exact family.

Direct schema `tnfr.sine-weighted-star-retention.v1` and generic SDK export
rebuild the nested state and certificate. The calculation executes no event
or trajectory, establishes no previous acquisition or attraction, and
supplies no stability certificate after a changed support, conductance,
forcing or loss law. Threshold equality is not strict retention. The
boundary minimizer establishes sharpness of the energy-sublevel implication,
not a prediction that a particular trajectory escapes at that energy.

<a id="sine-weighted-star-contact"></a>

### Compose two maintained identities under a supplied boundary contact

`assess_sine_weighted_star_contact(left, right)` from
[the contact owner](../../../src/tnfr/physics/relational_sine_weighted_contact.py)
returns `SineWeightedStarContact`. Both arguments must be
`SineWeightedStarState` instances with admitted `capacity_scale=1`. The
reader rebuilds their primitive forms and declared phase lifts before
constructing two disjoint copies of the fixed star and a unit-conductance
edge between their boundary nodes. The report retains all twenty continuous
coordinates. It installs no support in a live graph and executes no UM.

The joined node order is the five left nodes followed by the five right
nodes; the bridge is `(4,9)`. Capacities remain unchanged. Both boundary
degrees become five, so both of their original-time rows change. The
post-contact pressure is independently rebuilt from bare sine sums and
the new degrees. Conductance enters the form gradient in the phase row.
`left` and `right` remain rebuilt isolated-reference reports: their old
rates are not the joined rates. Read the latter from the contact report's
`pressure_bounds`, `form_rate_bounds` and `phase_rate_bounds`; phase rates
are radians per original structural time.

`support_work_bounds` accounts for the added bridge's nonnegative form
and phase storage at unchanged nodal coordinates. `component_storage_bounds`
retains both old edge sets inside the joined model. Their derivatives
generally differ from zero; `component_storage_rate_bounds` and
`bridge_storage_rate_bounds` sum to the identically zero full storage
derivative. This instantaneous exchange is distinct from a signed transfer
through an arbitrarily long window. Post-contact weighted means use the
new degree/capacity ratios; the old two separately conserved means do not
transfer as conserved quantities.

`source_individual_retention_certified` records the two isolated-law
certificates. `all_time_joint_identity_retention_certified` instead needs
strict joined full storage below `35/64`, including the accounted bridge
work. This guarantees both marked center-leaf identities under the fixed
joined conservative law. The barrier is sharp for this storage-sublevel
inclusion. Equality and unresolved upper bounds give `unavailable`, which
does not predict identity loss. Two passing isolated certificates alone
do not imply a passing joint certificate.

Direct schema `tnfr.sine-weighted-star-contact.v1` and generic SDK export
rebuild both primitive states and every consumed derived field. The
[proof](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-weighted-star-maintenance-composition)
exhibits a full-dimensional preparation neighborhood with nonzero initial
exchange and all-time identity retention. This point reader does not
authenticate an uncertain preparation against that neighborhood. Supplied
support, storage work and held law do not establish event occurrence,
formation of a larger NFR or physical identification. See the
[example](../../guides/relational/SINE_REGIONAL_DYNAMICS.md#weighted-star-contact)
and [independent controls](../../../tests/physics/test_sine_weighted_star_contact.py).

<a id="sine-weighted-star-support-retention"></a>

### Retain marked stars on an actual weighted successor support

`assess_sine_weighted_star_support_retention(source, *, stars)` from
[the protected-star owner](../../../src/tnfr/physics/relational_sine_star_retention.py)
returns `SineWeightedStarSupportRetention`. It consumes a general
`SineSupportExchangeComparison`, rebuilding its primitive state, support,
conductances and model. `stars` declares two disjoint ordered five-node
tuples `(center, leaf1, leaf2, leaf3, boundary)` covering the ten source
nodes. The source must declare the regular conservative weighted-form /
bare-sine law with unit exchange and phase-storage coefficients and
strictly positive held capacities.

Every original center-leaf conductance must be at least three and each
original member-boundary conductance at least two. Missing or weakened
protected edges are outside this theorem's domain and reject admission.
Additional nonnegative-conductance edges are retained with their bare
phase support. Actual capacities need not fit the original star's
one-parameter family; actual degrees and both complete rates remain those
of the rebuilt source. No input is replaced by the earlier unit-bridge
model or its rates.

The full storage dominates each protected star's original Dirichlet and
nonnegative phase storage. A strict upper enclosure below `35/64` thus
certifies both marked center-leaf identities for all time under the held
successor law. Larger storage gives an unavailable sufficient certificate,
not a loss verdict. This reusable lower barrier need not be sharp when
other support or stronger protected conductances raise the actual minimum.

The report retains the protected edges and their admitted weights, exact
form contrasts and cosine enclosures for the internal phase gaps. Its
current identity status distinguishes admitted, excluded and unavailable
phase/form observations. Circular identity is read from cosines of actual
radian differences; no rational-turn conversion or primitive phase angle
is invented. Full storage, margin and quantitative contrast bounds are
rebuilt on export, including through the generic SDK.

Direct schema `tnfr.sine-weighted-star-support-retention.v1` exposes a
detached continuation certificate. It does not execute an operator,
authenticate an event record or establish that a history and grammar
context was admitted. The
[canonical-contact study](../../../theory/nodal/SINE_WEIGHTED_STAR_CANONICAL_CONTACT.md)
uses the actual `AL,UM,SHA` successor, whose five nonunit cross edges and
changed capacities require this reader rather than the earlier fixed
unit-bridge report. The
[usage example](../../guides/relational/SINE_REGIONAL_DYNAMICS.md#weighted-star-canonical-contact)
keeps the event, stored pressure and detached continuation law separate.
The [finite-form response contract](SINE_COMPARISON_AND_INFERENCE.md#sine-finite-form-response)
adds a separately justified finite endpoint enclosure under the same complete
held law. Retention and a nonzero response are distinct certificates.

<a id="sine-moving-pattern-window"></a>

### Full-state uncertainty around a compatible moving pattern

`assess_sine_moving_pattern_window(reference, *, form_error_bounds,
phase_error_bounds, scaled_horizon, receiver_phase_radius,
contact_phase_radius, reference_contact_bound)` returns
`SineMovingPatternWindow` from the
[partition owner](../../../src/tnfr/physics/relational_sine_partition.py).
The input is a `SinePhaseOffsetState`; its primitive source, partition,
offsets and collective coordinates are rebuilt before deriving a reference.
Cached fields or verdicts cannot substitute for those inputs.

This reader admits the first block as an ordered induced C5 and the second
as its five private leaves, with unit conservative law, uniform signed
fifth-turn receiver gaps and one common contact phase. Every fine node
continues to evolve. The supplied nodal form/phase error bounds are
nonnegative and independently cover all ten nodes. Phase errors and the
three phase bounds are **radians**; `scaled_horizon` is a strictly positive
duration in `tau=t/pi`. The three phase bounds are also strictly positive.
Reference coordinates retain the preceding API's exact turn convention.

Separate checks admit the initial phase-error chart and a reference contact
libration envelope. The relative Bregman storage uses all fine form and
phase edge errors. A strict first-exit inequality combines
`initial_relative_storage_upper_bound`, `relative_storage_barrier` and
`growth_rate_upper_bound`; it preserves receiver and contact margins
separately. `whole_window_retention_certified` requires every premise.
Successful reports retain propagated storage, edge error bounds, receiver
storage excess and conserved weighted-mean error bounds. The edge estimates
do not bound absolute coordinates without those retained origins.

`status` is `certified` or `unavailable`; an unavailable sufficient estimate
does not exclude maintenance. Contact convexity is sufficient for this
certificate, not necessary for identity or for the exact compatible family.
The proof covers both forward and backward intervals of the declared
duration. It therefore does not establish first formation at the central
reference state or attraction toward that state.

`initial_full_storage_bounds` is independently rebuilt from primitive
full-edge uncertainty boxes, even if retention is unavailable. When the
acute unit-winding receiver chart is admitted, its lower bound also retains
the exact cycle minimum `V_R>=V_5`, together with all edge-form and contact
phase lower bounds. Independent edge intervals alone can forget that
constraint and suggest a spurious budget overlap. If the
receiver chart is admitted and that storage upper bound is below `7/2`,
`phase_flat_acquisition_status="excluded"` retains the existing phase-flat
acquisition obstruction. Otherwise the value is `not_excluded`, which is
not evidence of successful acquisition. A proposed earlier source must
also have a conserved storage compatible with this interval; overlap alone
does not provide entry. These checks concern full storage, not just a
regional excess or an untracked environmental reservoir.

The direct schema is `tnfr.sine-moving-pattern-window.v1`; generic SDK export
preserves its primitive reference, independent error budget, unavailable
fields and scoped conclusions. No trajectory is run or modified. See the
[moving-reference proof](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-moving-pattern-window)
and [example](../../guides/relational/SINE_REGIONAL_DYNAMICS.md#moving-pattern-window).

<a id="sine-collective-pulse-transfer"></a>

### Actual contact-pulse feedback outside an invariant family

`observe_sine_collective_pulse(source, *, cycle, contact_turn_offsets)`
returns `SineCollectivePulseBalance` from the
[partition owner](../../../src/tnfr/physics/relational_sine_partition.py).
It re-admits the actual comparison's full primitive state, conservative
unit-capacity law and C5/private-leaf support. Membership in a phase-offset
family is not assumed. `cycle` supplies the five receiver labels in traversal
order; its private leaves are matched from the unchanged support.

The five `contact_turn_offsets` must be exact integers, excluding Booleans.
In cycle order they declare the retained lifts
`phi_i=theta_leaf-theta_receiver-2*pi*k_i`. No principal-branch wrapping or
discarded history is inferred. The contact mean, its deviations and the
collective storage depend on this declaration; the full nodal storage
does not. Phase values are radians and all derivatives use `tau=t/pi`.

The report retains both block form means, their `form_gap`, full contact
rates, contact phase/deviation bounds and the complex deviation resultant.
`collective_energy_bounds` encloses `h=u^2/2+1-cos(mean(phi))`;
`collective_storage_bounds` encloses `5h`.
`collective_energy_rate_bounds` and `feedback_energy_rate_bounds` evaluate
the same exact mean-contact transfer identity in two forms. A zero-containing
factorization residual records numerical consistency, not its proof.
`remainder_storage_bounds=H-5h` is signed; its rate is `-5h'`.
Nonnegativity requires extra contact-convexity premises and is not presumed
by this reader.

The same report resolves receiver phase motion into
`receiver_internal_phase_rates=L*q/3` and
`receiver_contact_phase_rates=P*r/3`, where `q` is centered receiver form,
`r=x_R-x_Q`, `P` subtracts the receiver mean and `L` is the cycle Laplacian.
Their exact sum is `receiver_relative_phase_rates`. The fields
`receiver_relative_phase_acceleration_bounds` and
`receiver_relative_phase_jerk_bounds` enclose its first and second derivatives
from the full nodal rows. All five tuples use the declared cycle order;
their derivative units are radians per corresponding power of `tau`.
They are independent of the declared integer contact lifts and common origins.
They consume the actual contact phases and velocities, not an imposed pulse.
Cached source rates, gradients and storage do not supply these values.

These are instantaneous derivatives, not retention verdicts. Even exact zero
relative velocity and acceleration can coexist with nonzero third phase
derivative. Interval overlap with zero proves neither exact cancellation nor
rigidity. The [relative-feedback theorem](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-relative-phase-feedback)
classifies exactly rigid acute trajectories separately from finite identity
whose internal gaps are allowed to change.

`flat_phase_jet_available` requires equal exact source phase lifts at every
node and zero declared offsets. Only then are
`first_three_energy_derivatives=(0,0,0)` and the exact
`fourth_energy_derivative` available. The latter uses the actual contact-rate
variance and third central moment. For another preparation these derivatives
are `None`, not zero. A nonzero first available derivative fixes the initial
direction of transfer, but supplies no certified numerical time window,
later entry or retention.

The direct export schema is `tnfr.sine-collective-pulse-balance.v1`.
The report observes an actual state without installing a law, advancing time
or certifying formation. The [derivation](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-collective-pulse-transfer)
and [standalone example](../../guides/relational/SINE_REGIONAL_DYNAMICS.md#collective-pulse-balance)
keep the conserved-budget and phase-lift limitations explicit.

<a id="sine-cycle-sector-barrier"></a>

### Conservative cycle-sector source admission

`assess_sine_cycle_barrier(source, *, cycle)` returns `SineCycleBarrier`
from the [regional owner](../../../src/tnfr/physics/relational_sine_regional.py).
It re-admits a complete `SineExchangeComparison`, zero loss, unit exchange,
unit storage coefficient and held unit capacities. Five distinct source
labels must trace a C5. Additional nodes, contacts and chords remain in the
full law and rebuilt storage; this check does not require an induced cycle.

The report independently assesses the two wider sectors with winding
`-1,+1` and all principal gaps strictly below `2*pi/3` in magnitude.
`sectors` records each membership, exclusion and invariance verdict. If
`full_storage_bounds.hi <= energy_barrier`, with the shared derived barrier
`7/2`, an outside source cannot reach any acute state of that orientation;
an inside source cannot leave that wider sector in either time direction.
The latter does **not** certify strict acuteness, a moving-reference error
box or an attracting waveform. No initially flat phase is assumed.

`finite_time_energy_bound_certified` records this closed upper-bound test.
The existing `strict_energy_bound_certified` retains its strict `<` meaning.
At boundary equality, nonnegative full storage forces uniform form, aligned
extra-edge phases and balanced cycle currents. The full field is therefore
zero, and smooth uniqueness prevents a distinct orbit from reaching this
boundary equilibrium in finite time. This does not assert that a source
with energy `7/2` is itself stationary, nor exclude asymptotic approach to
a boundary equilibrium.

Principal-branch ambiguity leaves `initial_winding` unavailable. A separately
proved edge outside the wider chart can still establish nonmembership without
assigning a winding. An upper storage endpoint equal to the barrier suffices
for the finite-time test; an interval extending above it does not. Unresolved
membership never supplies a sector verdict. Cached storage and rates
are ignored. Source association and JSON projection do not authenticate a
preparation. Direct schema: `tnfr.sine-cycle-barrier.v1`.

The moving-pattern window also exposes `zero_winding_acquisition_status`:
a certified acute target box entirely below the barrier cannot be reached
from any zero-winding source under the same law. Its existing
`phase_flat_acquisition_status` retains the restricted compatible conclusion.
`not_excluded` is not an entry claim. See the
[proof](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#sine-cycle-sector-barrier).

<a id="sine-cycle-retention"></a>

### Finite acute retention from conserved full storage

`assess_sine_cycle_retention(source, *, cycle, scaled_duration,
source_error_bound=0)` returns `SineCycleRetention` from the
[regional owner](../../../src/tnfr/physics/relational_sine_regional.py).
Its complete law and ordered C5 admission are shared with the cycle barrier:
zero loss, unit exchange/storage coefficients, held unit capacities and full
simple unit support, including every environmental node and additional edge.

The positive duration uses `tau=t/pi`. The nonnegative error bound declares
independent errors of at most that amount in every full-source form and
continuous phase coordinate, in their declared model units. The mathematical
set is defined by the exact admitted center and radius. Displayed gap intervals
are outward bounds; they do not replace that set with an independently enlarged
full-state box. Boolean, textual, negative or nonfinite budgets reject.

The reader rebuilds nominal storage and both a direct edge-box bound and a
bound from the full storage gradients and the global Taylor remainder. It
intersects their upper bounds in `full_storage_bounds`; no cached source
storage, torque or phase rate supplies evidence. Nominal phase torques remain
present even when a represented phase is close to an ideal critical twist.

`initial_acute_unit_winding_certified` requires every state in the declared
box to have the same acute winding `-1` or `+1`. Under that premise, the
phase-storage floor is `V5=5*(1-cos(2*pi/5))`. Each oriented receiver edge
has `edge_cauchy_factors = 1/d_i + 1/d_j + 2/(d_i*d_j)`, using actual full
degrees. Its phase speed is bounded by the square root of twice this factor
times `full_storage_bounds.hi - phase_storage_floor_bounds.lo`.
`edge_phase_transport_upper_bounds` multiplies these speed bounds by the
duration. Subtracting them from the initial edge margins gives
`edge_retention_margin_lower_bounds`.

Only a strictly positive minimum margin certifies
`whole_window_retention_certified`, valid throughout both time directions
up to the declared duration. Missing initial geometry or a failed sufficient
budget yields `unavailable`, not a prediction of failure. The speed bounds
come from a first-exit proof; the reader does not extrapolate an instantaneous
velocity or assume future acuteness. It installs no law and runs no solver.

This is prepared maintenance. An above-barrier storage interval does not prove
entry from zero winding, and a zero-winding source must precede the protected
backward interval. No exact invariant reference, rigid receiver geometry or
acute contact phases are required. Schema: `tnfr.sine-cycle-retention.v1`;
shared SDK projection and atomic export preserve the primitive source,
uncertainty and availability. See the
[proof](../../../theory/nodal/SINE_CONSERVATIVE_PREPARATION.md#sine-energy-speed-retention)
and [example](../../guides/relational/SINE_REGIONAL_DYNAMICS.md#energy-speed-retention).

<a id="sine-reversible-preparation"></a>

### Reversible preparation into a retained full-state region

`assess_sine_reversible_preparation(forecast, checkpoint, *, cycle,
target_error_bound, source_error_bound, scaled_retention_duration=1)` returns
`SineReversiblePreparation` from the same regional owner. Both radii and the
scaled duration must be positive admitted real scalars. It re-admits the
complete conservative unit-law forecast and checkpoint, including support,
node order, unit capacities and every form/phase coordinate. The forecast
starts at time zero without prior conditioning or a frozen hidden node; its
initial enclosure must contain `R(checkpoint)`, where `R(x,theta)=(-x,theta)`.

The reader rebuilds the checkpoint's energy-speed retention certificate.
At each retained grid endpoint at original time `A`, it reverses the exact
rational midpoint to obtain a source center `c`, retaining the largest
endpoint halfwidth `eta`. The independent source set is the exact coordinate
ball `B_delta(c)`, not the larger backward enclosure. The shared field has
global sup-norm Lipschitz bound `2/pi` in this clock. An outward exponential
bound `E >= exp(2*A/pi)` therefore proves forward inclusion in the retained
target of radius `epsilon` when `E*(eta+delta) < epsilon` strictly.

Certification additionally requires whole-source winding zero, overlapping
source/target conserved-storage bounds, and `A > pi*T`, where `T` is the
scaled retention duration. Storage overlap alone does not prove entry.
Every endpoint retains its exact center, uncertainty, amplification, winding,
energy and failed predicates. The first passing endpoint is selected. A
certified prefix survives an unavailable suffix; without success, incomplete
coverage yields `unavailable`, and complete coverage yields
`no_certificate_on_declared_grid`. Neither means that formation is impossible.

The reader checks primitive admission, the forecast chain and Picard
inclusions; it does not replay or authenticate the supplied Taylor endpoint
proofs. Its conclusion is conditional on that numerical evidence. Schema
`tnfr.sine-reversible-preparation.v1` is supported by shared SDK projection
and atomic export. This constructs mathematical preparations, without
autonomous selection or physical identification. See the
[proof](../../../theory/nodal/SINE_CONSERVATIVE_PREPARATION.md#sine-reversible-preparation)
and [workflow](../../guides/relational/SINE_REGIONAL_DYNAMICS.md#reversible-preparation).

<a id="sine-contact-averaging"></a>

### Finite exclusion from rapid contact motion

`assess_sine_contact_averaging(source, *, cycle, scaled_horizon)` returns
`SineContactAveraging` from the
[partition owner](../../../src/tnfr/physics/relational_sine_partition.py).
It uses the same complete unit conservative law but requires exactly the
C5/private-leaf support, uniform initial form within each block and equal
represented initial receiver phase lifts. Initial leaf phases are arbitrary.
The finite positive horizon `T` uses `tau=t/pi`; Boolean, textual and
nonfinite scalars reject through shared admission.

For the signed initial form gap `u`, the report bounds each contact's speed
away from zero by `M=4*abs(u)/3-4*T`. When `M>0`, it computes the whole-window
current-integral bound `epsilon=2/M+4*T/M**2` and receiver phase-storage bound
`640*T**2*epsilon**2/81`. If that bound is strictly below `7/2`,
`whole_window_acute_winding_excluded` holds on the entire declared forward
interval. This proof can apply when full storage exceeds the cycle barrier.
It uses exact transformed full-law rows and integration by parts, not a
truncated tangent model, sampled trajectory or an installed averaged law.

If speed separation or the final strict test fails, status is `unavailable`;
there is no prediction of formation. The conclusion does not extend to later
times or nonuniform block forms, including the separate distributed-ramp
entry witness. Direct schema: `tnfr.sine-contact-averaging.v1`. Both source
readers support the shared SDK report projection and atomic export. See the
[proof](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#sine-contact-averaging)
and [usage](../../guides/relational/SINE_REGIONAL_DYNAMICS.md#conservative-source-admission).

<a id="sine-conservative-phase-transport"></a>

### Conservative phase transport from a flat preparation

`analyze_sine_conservative_phase_transport(source, *, contrasts)` returns
`SineConservativePhaseTransport` from the shared
[entry owner](../../../src/tnfr/physics/relational_sine_entry.py). It requires an
exact comparison, globally common initial phase lifts, zero loss, unit held
capacities and `w=beta=1`. Its clock is `tau=t/pi`. Initial form may be
nonuniform; unlike environmental winding entry, no receiver cycle is required.
Primitive source admission and exact graph rows are shared with winding entry;
cached source gradients, rates and storage are not derivative evidence.

Each supplied contrast is a nonzero row in source-node order, with exact zero
sum. Coefficients use shared exact-or-represented real admission; Boolean and
nonfinite values reject. These rows describe relative phase observations, not
new coupling weights or additional dynamics.

Writing `A=K L`, the report retains exact initial derivatives
`initial_phase_velocity=A*x`, `initial_phase_acceleration=0`,
`initial_form_acceleration=-A^2*x` and `initial_phase_jerk=-A^3*x`.
The jerk is a third derivative at the initial instant. It cannot be
extrapolated into a finite-time trajectory without a remainder certificate.

For each contrast `c`, the reader also derives current weights `c*A*K`,
oriented edge-current coefficients, and the global acceleration bound
`B=sum_edges abs((c*A*K)_i-(c*A*K)_j)`. Thus for every nonnegative `tau`,
`abs(c*theta(tau)-c*theta(0)-tau*c*A*x(0)) <= B*tau^2/2`.
`contrast_quadratic_remainder_coefficients` retains `B/2` separately from
`contrast_initial_jerk`. Correlated paths can cancel shared nodes before
bounding; summing independent edge boxes would lose that information.

The [regional derivation](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#conservative-regional-phase-transport)
uses these quantities to identify delayed internal propagation and a necessary
acute-entry condition. Neither a nonzero jerk nor an upper acceleration bound
certifies acquisition, future work, capture or permanent identity. Direct
schema `tnfr.sine-conservative-phase-transport.v1` and generic SDK export retain
the source, contrasts, derivatives, clock and scope. See the
[independent controls](../../../tests/physics/test_sine_conservative_phase_transport.py).

<a id="sine-regional-organization"></a>

### Continuous regional organization from a complete forecast

`assess_sine_regional_organization(forecast, *, cycle_indices,
minimum_duration, acute_margin)` in
[relational_sine_regional.py](../../../src/tnfr/physics/relational_sine_regional.py)
returns `SineRegionalOrganization`. It observes a declared induced C5 inside
the complete support of a `SineForecast`, under zero loss, unit held capacities
and `w=beta=1`. Cycle entries are distinct source indices in traversal order.
The strictly positive duration uses original structural `t`; the nonnegative
requested acute margin uses radians. Both are observation policies, not
evolution laws.

The reader re-admits the law, support, state intervals, time grid and every
consumed enclosure. Consecutive steps must cover the actual validated prefix,
retain their input and endpoint boxes, and preserve the augmented held
capacity. Whole-time Picard inclusion is recomputed from full tube rates.
Derived winding, storage, rates and work are rebuilt rather than copied from
cached verdicts. This checks the consistency of supplied Taylor certificates;
it does not authenticate their production or replay the full derivative proof.
Retain the shared solver's actual evidence and source provenance.

Each whole-time tube is checked for fixed integer branches, nonzero cycle
winding and a strict margin above the requested acute margin. A contiguous
interval must keep the same offsets and meet `minimum_duration` before the outcome becomes
`finite_acute_retention_certified`. Samples or an acute endpoint are
insufficient. Observations retain internal form and phase storage, full-network
phase and relative phase rates, and net boundary power computed from both
conjugate channels. The separate [channel reader](#sine-regional-channel-history)
resolves internal conversion and inputs to each regional store.

Conservative regional balance gives net work from the initial time as
`E_R(t)-E_R(initial)`. Subtracting interval enclosures bounds this quantity;
it does not establish the instantaneous sign, monotonic supply or a fitted
damping term. Internal regional storage includes form as well as phase.
Acute phase retention alone does not certify the stronger low-storage
protection theorem or indefinite maintenance.

`acute_winding_excluded_on_horizon` requires complete declared coverage and
an independently recomputed exclusion on every tube. A nonacute edge,
nonnegative cosine across a two-edge path, or certified zero winding can
exclude an acute unit-winding C5 under their stated interval signs. Failure
to certify the requested margin/duration is not itself such an exclusion.
Otherwise the outcome is `unresolved`, retaining partial coverage and
numerical reasons. A certified interval inside a validated prefix can remain
valid even when a later step is unavailable; `horizon_complete` keeps that
distinction explicit.

Direct schema `tnfr.sine-regional-organization.v1` and generic SDK export retain
the full forecast and observer evidence. The
[control](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#conservative-regional-organization-control)
freezes a particular source and budget separately from this reusable reader.

<a id="sine-regional-channel-history"></a>

### Cumulative regional channel observations and phase accessibility

`assess_sine_regional_channels(forecast, cycle_indices=...)` reuses the
[complete conservative forecast admission](#sine-regional-organization)
and the [regional channel kernel](#sine-regional-storage-balance). It admits
the same induced C5, unit capacities and zero-loss `w=beta=1` law. Every
step retains the whole-time channel rates and cumulative interval integrals
of internal conversion and both boundary inputs in original structural time.
Positive conversion means phase-to-form transfer. Summing rate intervals
times full step durations bounds the integrals; endpoint samples are not
quadrature evidence. Endpoint storage changes must be compatible with both
integrated balances or the reader rejects the supplied evidence.

`phase_storage_upper_bound` is the running bound from primitive whole-time
boxes. `running_phase_work_upper_bound` independently bounds the running
integral of `-J+B_V`, including intermediate times inside each step.
`initial_phase_flat_certified` requires exactly equal point-valued receiver
phase lifts; false means this premise was not certified. Under that premise,
either a strict phase-storage bound below `acute_accessibility_barrier=7/2`
or the corresponding integrated-work bound excludes any fully acute unit
winding entry **on the validated prefix**. Reaching or failing to resolve
the barrier is not entry. A prepared acute state or arbitrary nonacute
winding is outside this exclusion's conclusion.

`horizon_complete` separately reports coverage. Even an empty validated
prefix cannot establish absence later in the requested horizon. The observer
re-admits primitives and recomputes Picard inclusion; it does not replay
Taylor endpoint proofs, authenticate a producer or infer new dynamics.
Schema `tnfr.sine-regional-channel-history.v1` and the generic SDK envelope
retain normalized full forecast evidence and all cumulative bounds.

The benchmark's `--analyze-channels SOURCE_RESPONSE --output PATH` mode
strictly decodes its retained response schema, binds it to the saved protocol
and archived declaration, and checks original file/manifest hashes. It writes
a separately identified retrospective analysis and current-source archive,
leaving the source response intact. Its hashes establish file associations,
not external provenance authentication; historical source need not equal
current analysis source. The result is not a new reserved prediction.
The [proof and scope](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#sine-regional-channel-accessibility)
remain independent of the transport and serialization checks.
