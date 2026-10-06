# Native relational capture, memory and continuous proofs

Native capture and exclusions, cycle/contact memory, transit, reflected proofs, regularity and retained robustness evidence.

Part of [Relational dynamics contract index](../RELATIONAL_DYNAMICS.md). Section links remain stable; hypotheses and model changes remain local to each result.

## Native relational capture, memory and proof tools

These certificates retain the native law and the preparation/domain specified
by each API. An observed state, a proposed event and a validated trajectory
are not interchangeable evidence.

### Conditional relational capture

[`certify_relational_capture`](../../../src/tnfr/physics/relational_capture.py) and
`Network.relational_capture(model, *, cycles)` apply the
[protected-capture theorem](../../../theory/nodal/RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-protected-capture)
to one fresh detached field. They add no solver, projection or controller.
The existing model/graph/phase-domain admission runs first. Supply exactly
two ordered five-node cycles covering the full graph; its only other edges
must join matching positions zero and one. Unsupported topology or malformed
inputs raise. An admitted engine snapshot need not satisfy this theorem.

Each ring must have exactly identical form and phase coordinates
`x=m+(A,-A,-B,0,B)` and `theta=c+(a,-a,-b,0,b)`, with held unit capacities
and positive EPI, phase and storage coefficients. Fractions of the captured
values retain both offsets and every copy/reflection defect. Nonzero defects
are not rounded away, even when small. The theorem's exact symmetry is a
sufficient premise; its failure is not proof that recovery is impossible.

Mathematical-pi enclosures test three disjoint open rectangles in `(a,b)`:

| Rectangle | Phase coordinates | Ideal limiting phase sector |
| --- | --- | --- |
| Positive twist | `(2*pi/3,pi) x (0,pi/2)` | `+1` |
| Consensus | `(-2*pi/3,2*pi/3) x (-pi/2,pi/2)` | `0` |
| Negative twist | `(-pi,-2*pi/3) x (-pi/2,0)` | `-1` |

The same rational cosine owner encloses full-support phase storage at the
exact represented raw radians. `storage_bounds` adds exact form storage and
the declared beta factor; its upper bound must be strictly below `7*beta`.
This threshold is derived for this support, symmetry and storage model, not a
universal physical constant or a configurable acceptance tolerance. The
field's floating `phase_storage` is retained separately and is not used as
an exact energy enclosure. Chosen real phase lifts are not silently replaced.

`RelationalCaptureCertificate` retains all candidate rectangle margins, the
selected `rectangle_kind`, exact symmetry defects, coefficient/capacity
admission, storage bounds and enclosure provenance. It sets `target_sector`
only when every premise passes; otherwise `status="unavailable"`,
`target_sector=None` and explicit `unavailable_reasons` are retained.
Its `admitted` property concerns the ideal continuous conditional law from
this exact represented initial state. Current winding is computed separately
by the shared observer and need not match the predicted limiting sector.

The result does not certify future Euler steps, authenticate a caller-created
dataclass, establish an error bound for a numerical trajectory, or demonstrate
formation from winding zero. Exact capture implies an open full-state basin
qualitatively, but no asymmetry tolerance is computed or admitted here.
The shared `relational_report_to_dict` exporter preserves the exact evidence.
[Usage](../../guides/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#check-a-protected-relational-basin) and
[routine controls](../../../tests/test_relational_capture.py) exercise the same owner.

`certify_relational_local_capture(graph, *, model, cycles, target_sector=1)`
and `Network.relational_local_capture` apply the existing full-state local
energy theorem to the same supplied support. They require unit capacity and
storage scale, positive coefficients and a declared sector in `{-1,0,1}`.
They do not require reflection symmetry. The ideal reference in each ring is
`sector*pi*(4/5,-4/5,-2/5,0,2/5)`; common form and phase offsets are removed
exactly, without changing individual phase lifts. Rational pi and cosine
enclosures must prove squared quotient distance below `9/800` and excess
storage above the target below `1/100000`. These conservative constants come
from the local energy barrier, not fitting the evaluated response.

`RelationalLocalCaptureCertificate` retains the declared target, affine-pi
phase errors, distance/energy bounds and individual admission reasons. Its
target is available only when all premises hold. A negative lower bound on
the enclosed excess is allowed: interval uncertainty can straddle zero,
while the admitted local theorem supplies nonnegativity of the exact excess.
This certifies ideal continuation from the precise represented snapshot.
Applying it to a numerical endpoint supplies no error enclosure connecting
that endpoint to the original continuous initial-value problem. Both capture
reports use the same exact exporter; neither controls the dynamics.

`observe_relational_sector_geometry(graph, *, storage_scale, cycles,
target_sector=1)` and `Network.relational_sector_geometry` expose the shared
geometric evidence without a dynamics model. They consume the supplied simple
unit support, signed form, primitive phase and positive storage scale. The
support must be the two five-node rings with matching bridges at positions
zero and one. Capacity, pressure, forcing and reference rates are not consumed.
The detached `RelationalSectorGeometry` retains nodes, edges, coordinates,
exact form storage, gap lifts, cycle periods and rigorous energy/barrier bounds.
Its `admitted` property means that acute geometry, the declared sector and the
strict energy barrier all pass; it makes no claim about a future trajectory.

Its `sublevel_*` bounds concern states in the same sector with storage no
greater than the reported upper bound. They are unavailable if any geometric
premise fails. The report contains no `future_*` bounds or limiting-target
prediction. Turning sublevel evidence into a future guarantee requires a
separate complete-law proof, such as the
[strict-loss class theorem](../../../theory/nodal/RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-sector-law-class).
That theorem covers specified autonomous smooth phase laws with the native
form row, positive held capacities/coefficient scales, common-offset symmetry,
rest and a uniform pointwise strict form-loss bound. Its conclusions include
sector retention, recovery of the twist and eventual exponential convergence;
the rates remain law-dependent. A snapshot residual or supplied Boolean flag
cannot admit a law to that class. Geometry can be read even when a graph has
forcing or lacks capacity; this does not certify that graph's execution.

`certify_relational_sector_capture(graph, *, model, cycles, target_sector=1)`
and `Network.relational_sector_capture` apply the
[full-state acute-sector theorem](../../../theory/nodal/RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-acute-sector-capture).
On the same support they require strictly positive held capacities, positive
storage scale and coefficients, strictly acute gaps and both ring periods equal to the declared
`+1` or `-1` sector. Exact affine-pi inequalities validate integer full-turn
candidates for each raw edge difference. The three independent cycle periods
include the four-edge bridge cycle, which must have period zero. Floating
remainders and numerical winding telemetry do not establish these premises.

`RelationalSectorCaptureCertificate` retains these gap margins and periods,
full energy bounds and a certified interval for the sufficient barrier
`beta*[10-5*cos(2*pi/5)-4*cos(3*pi/8)]`. The energy upper bound must be strictly below
the barrier lower bound. Exact reflection and a small Euclidean distance to a
target are unnecessary. The limiting target is circular, modulo a common
offset and fixed nodewise full-turn representatives. This is a sufficient
barrier, not a claimed sharp bound or a numerical tolerance. Unresolved
premises remain unavailable with evidence. As with the other certificates,
admission concerns ideal continuation from the captured state; applying a
new theorem after an experiment does not revise its frozen decision criteria.

`geometric_barrier_bounds` encloses the unscaled geometric cost and
`capture_barrier_bounds` includes beta. The retained `unit_capacity` and
`unit_storage_scale` flags are descriptive; `positive_capacity` supplies
the capacity admission. Zero capacity is not admitted by this recovery theorem.
For full admission, `normalized_energy_margin_lower_bound=eta_lower>0`
supplies three conditional all-future bounds: acute edge margin `13*eta_lower`,
nodewise resultant real part `26*degree*eta_lower/pi_upper`, and phase metric
`26*degree*eta_lower`. The resultants/metrics follow `field.nodes`. All three
future-bound fields are `None` when any premise fails, even if energy alone
passes. They describe the ideal continuation, not Euler error or a solver step.
The [derivation](../../../theory/nodal/RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-sector-consolidation)
connects those margins to the same Jensen barrier without adding a threshold policy.

| Certificate | Distinguishing sufficient premises | Scope |
| --- | --- | --- |
| `relational_capture` | Exact reflected state, unit capacity, positive beta, strict rectangle and energy below `7*beta` | Can admit nonacute states and consensus |
| `relational_consensus_capture` | Exact reflected form, equal initial phases, unit capacity/beta, effective `e=w=1/2`, form storage at most 9 | Analytic entry into consensus by time one, then convergence; no evaluated endpoint |
| `relational_consensus_formation_obstruction` | Arbitrary full form on the same two-ring support, equal initial phases, unit capacity/beta, effective `e=w=1/2`, form storage at most 9 | Excludes aligned acute unit winding throughout regular existence; no global continuation or consensus verdict |
| `relational_local_capture` | Unit capacity/beta, small full-state quotient distance and excess energy | No reflection; declared target `-1`, `0` or `1` |
| `relational_sector_capture` | Positive held capacities/beta, exact acute sector and scaled barrier | No reflection or local-radius gate; targets `-1` or `1`; quantitative regularity |

These independent sufficient theorems share one capture owner and exact
exporter. The sector certificate uses the same geometric kernel as the
law-neutral observer while preserving its reference-law admission and flat
report fields. The rho and eta comparison laws are mathematical comparisons,
not runtime modes of this certificate. No ordering of verdicts or automatic
policy is implied.

<a id="phase-consensus-capture"></a>
### Analytic capture from phase consensus

`certify_relational_consensus_capture(graph, *, model, cycles)` and
`Network.relational_consensus_capture(model, *, cycles)` apply the
[phase-consensus obstruction](../../../theory/nodal/RELATIONAL_FORMATION_CONTROLS.md#relational-consensus-preparation-obstruction).
They reuse one existing native capture, including full field and support
admission. Both rings have the same exact represented form
`m+(A,-A,-B,0,B)` and every captured raw phase equals one common value.
Held capacities and storage scale are one; normalized effective weights
are exactly `e=w=1/2`. The sufficient initial form budget is
`F=6*A**2-4*A*B+4*B**2<=9`. These fixed constants belong to this theorem
and structural clock, not a configurable policy or physical scale.

`RelationalConsensusCaptureCertificate.initial` retains the ordinary capture
report unchanged. Its initial `energy_admitted` may be false: the new result
proves a later entry, rather than relaxing that report's strict `E<7` test.
The new report retains exact `initial_form_storage`, `form_storage_margin`,
phase/coefficient admission and separate analytic bounds:

- Through `bootstrap_time=1`, the reflected phase coordinates obey
  `|a|<=sqrt(3*F/2)/(3*pi)` and `|b|<=sqrt(F)/(2*pi)`, with strict
  positive margins below `1/2` and outward rational enclosures.
- At time one, phase storage is at most `F/4` and total storage at most
  `2*F/3<=6`. The report retains consensus rectangle margins and
  `capture_margin=7-2*F/3`. These certify subsequent ideal consensus.

On full admission, `target_sector=0` and `status="admitted"`. Missing theorem
premises give explicit reasons, `target_sector=None` and unavailable dynamic
estimates; they do not prove that formation or instability occurs. Malformed
inputs or unsupported native support still raise through shared admission.
The analytic bounds are not an endpoint state, a sampled trajectory or a
guarantee that future finite steps pass the configured phase-domain guard.
The conclusion excludes maintained unit winding only in this stated class.
The shared exporter validates the nested initial labels and preserves exact
fractions and unavailable values. See the
[usage example](../../guides/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#check-relaxation-from-phase-consensus).

<a id="full-form-consensus-obstruction"></a>
### Full-form phase-consensus obstruction

`certify_relational_consensus_formation_obstruction(graph, *, model, cycles)`
and `Network.relational_consensus_formation_obstruction` reuse one fresh native
field and the existing exact two-C5/adjacent-matching-bridge support admission.
They retain arbitrary form, including all bridge costs, without imposing or
projecting copy/reflection symmetry. Exactly equal captured raw phases,
unit held capacities and beta, effective `e=w=1/2`, and `F(0)<=9` admit the
[nonlinear theorem](../../../theory/nodal/RELATIONAL_FORMATION_CONTROLS.md#relational-full-consensus-formation-obstruction).

The report exposes `all_regular_time_phase_storage_upper_bound=7*F(0)/10`,
the outward target-storage bounds and their strictly positive exclusion margin.
`excluded_target_sectors=(-1,1)` excludes entry into the aligned acute
unit-winding targets or convergence to them on the ideal regular path. It does
not forbid every transient winding change or classify all other outcomes.
`continuation_status="not_certified"` remains explicit: this theorem proves
neither indefinite regular existence nor convergence to consensus.

Failed theorem premises return `status="unavailable"`, named reasons and no
prospective bounds or excluded sectors. Invalid live state/support still
rejects through shared admission. The detached observer executes no trajectory,
installs no law and does not revise frozen evidence. Export preserves exact
rationals, full source state, cycles and unavailable fields, validating labels
before projection. Publicly constructing a report does not authenticate it.
The earlier reflected capture is retained because it proves the stronger
consensus conclusion on its smaller class.

<a id="relational-detachment-observation"></a>
### Isolated-cycle capture and hypothetical detachment

`certify_relational_cycle_capture(graph, *, model, cycle, target_sector=1)`
and `Network.relational_cycle_capture` apply the
[isolated-ring theorem](../../../theory/nodal/RELATIONAL_EFFECTIVE_CONNECTIONS.md#relational-pattern-detachment)
through the shared capture owner. The graph must be exactly the supplied
ordered unit C5. Native connected-field admission runs first. The sufficient
capture conditions require positive held capacities and coefficients, strictly
acute true-pi gaps, oriented winding equal to the declared `+1` or `-1`, and
mathematical storage strictly below `beta*(5-4*cos(3*pi/8))`.

`RelationalCycleCaptureCertificate` retains the fresh `field`, `cycle`, exact
gap/period evidence, phase and total `storage_bounds`, geometric and scaled
barrier bounds, and `storage_margin_lower_bound`. The gap evidence follows
`field.edges`; `cycle_winding` follows the declared orientation. `admitted`
and `target_sector` require every premise. An unmet sufficient condition
returns `status="unavailable"`, `target_sector=None` and reasons; invalid
field, topology or input admission raises. No reflection or equal-capacity
premise is imposed. Recovery means ideal convergence to uniform form and
the cycle twist modulo this component's own limiting offsets.

`observe_relational_detachment(graph, *, model, cycles, target_sector=1)`
and `Network.relational_detachment` compare removal of the two matching bridges
at positions zero and one of the supplied C5 rings. The original graph must
be exactly this connected unit support. The observer retains form, phase and
capacity and recomputes each component's degree, pressure, metric and rates.
It evaluates the two post-cut connected components separately. A disconnected
union is used only for shared reset accounting, never relational execution.
Live nodal attributes, graph metadata, histories and support remain unchanged.

`RelationalDetachmentObservation` retains:

- `before`, the connected field; `components`, the two full C5 capture
  certificates; and `removed_bridges`, the supplied matching pairs.
- `reset`, the shared `RelationalResetObservation`. Its pressure snapshots
  contain freshly evaluated pre/post-cut pressure, not stale stored pressure.
  It owns represented event storage and `assess_supply`; continuous loss is
  not an event reserve.
- Exact represented `form_rate_change`, `phase_rate_change`, `pressure_change`
  and `phase_metric_change`, each after minus before in `before.nodes` order.
- `field_form_storage_change`, `field_phase_storage_change` and
  `field_storage_change`, the sums of component field values minus the
  corresponding original field value. Phase storage is unscaled.
- `storage_reconciliation_residual`, equal to `field_storage_change` minus
  `reset.storage_change`; `capture_admitted` is the conjunction of the two
  component admissions.

The reset uses wrapped represented half-sine costs; a positive-resultant
native field uses raw represented gaps. Their floating arithmetic need not
agree, particularly for large phase representatives. Neither substitutes for
the rational cosine enclosures used by the capture certificate. The exporter
retains all three evidence types separately, exact fractions, nested labels
and the derived `capture_admitted` flag.

The joined acute-sector theorem implies both ideal component barriers;
direct component checks can also succeed outside that sufficient joined
barrier. Zero bridge storage does not imply unchanged rates or safe timing.
The early formation seed has a regular post-cut field but a topological
obstruction to the target twist; formed-state convergence establishes a
sufficiently late window, without certifying a particular event time.
No observer installs a removal event, selects its time, certifies prior
formation error or widens connected-support execution. A different phase law
requires its own post-cut admission to the theorem; the SDK certifies the
declared native reference model. See the
[usage](../../guides/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#inspect-hypothetical-pattern-detachment),
[engine controls](../../../tests/test_relational_detachment.py) and
[SDK/export controls](../../../tests/sdk/test_relational_detachment.py).

<a id="relational-seeded-formation-obstruction"></a>
### Storage obstruction for a uniform receiver

[`certify_relational_seeded_formation_obstruction()`](../../../src/tnfr/physics/relational_capture.py)
evaluates the [fixed source/receiver storage theorem](../../../theory/nodal/RELATIONAL_NATIVE_FORMATION.md#induced-formation-storage-obstruction).
The source is an exact unit-C5 winding-one twist; the receiver has uniform
phase on its own supplied C5. Both have the same constant form, held unit
capacity, `beta=1` and `e=w=1/2`. The two cases add either the position-zero
bridge or both matching bridges at positions zero and one. Subsequent support
is held fixed without forcing or additional events.

`RelationalSeededFormationObstruction` encloses the source storage and the
necessary two-pattern target storage. Each `RelationalSeededFormationCase`
retains its matching port pairs, bridge-cost ceiling, strict initial-storage
upper bound and `additional_storage_gap_lower_bound`. It covers every relative
phase admitted by the **initial ideal positive-resultant chamber**, without
sampling that phase or evaluating a graph. It does not assert that every
represented approximation will pass the engine's numerical admission.

Writing `c=cos(2*pi/5)`, the source storage is `V*=5*(1-c)`. Positive real
source-port resultants imply each added bridge costs less than `1+2*c`.
Both acute winding-one rings require total storage at least `2*V*`, so the
one- and two-bridge deficits exceed `4-7*c` and `3-9*c`, respectively. The
shared exact cosine geometry certifies positive lower bounds on both deficits.
The target is two acute winding-one sectors or their recovered twists;
ordinary winding one in a nonacute configuration is insufficient for this
storage lower bound.

`status="obstructed"` and `obstruction_certified=True` mean the specified target
is excluded under storage-nonincreasing continuation. There is no `admitted`
alias and no failed numerical trajectory. If the bounds cannot separate,
`status="unavailable"` retains the intervals and reasons. The deficit is only
a necessary missing storage amount; supplying a work number without changing
the state or law cannot repair it, and exceeding it would not prove formation.

Initial states in the wider mathematical regular domain are outside this
certificate. The same theorem documents a two-port state with negative real
but nonzero imaginary source resultants, positive native metrics and enough
total storage to avoid this particular obstruction. The positive-resultant
option rejects that state; the regular option admits its represented
preparation when the resultant rectangles, branch and materialization checks
pass. This changes executable coverage, not the obstruction's hypotheses.
Initial regularity and sufficient storage do not establish a regular future
route, target reachability or eventual formation.

The separate [geometric and initial-rate audit](../../../theory/nodal/RELATIONAL_NATIVE_FORMATION.md#regular-seeded-reachability-audit)
provides a regular common-sublevel path, an affordable internal cancellation
crossing and an exact eight-coordinate reflection restriction. These do not
alter this obstruction certificate's scope or admit the existing copied-ring
transit solver. The native full field and uniform tangent supply its static
controls; no new event or autonomous selection API is required.

The [guide](../../guides/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#check-a-uniform-receivers-formation-budget),
[capture controls](../../../tests/test_relational_capture.py),
[independent proof/domain controls](../../../tests/physics/test_relational_seeded_formation.py)
and [SDK export controls](../../../tests/sdk/test_relational_reports.py) distinguish
this conditional exclusion from successful capture and physical evidence.

<a id="relational-cycle-memory"></a>
### Asymptotic collective memory on an isolated cycle

`bound_relational_cycle_memory(*, model, form_direction, phase_direction,
capacity, amplitude_radius, target_sector=1)` in the
[cycle-memory owner](../../../src/tnfr/physics/relational_cycle_memory.py)
encloses the quadratic coefficient of a limiting common-phase shift. Its
[theorem](../../../theory/nodal/RELATIONAL_PHASE_MEMORY.md#relational-retained-phase-memory)
concerns the ideal native law on a unit C5 with positive **homogeneous held**
capacity, `e,w,beta>0`, and no inputs or further support events.

The supplied ordered directions `u,v` each contain exactly five finite real
coordinates with exact sum zero. They specify the ideal family
`x=epsilon*u`, `theta_i=target_sector*2*pi*i/5+epsilon*v_i`, in a common
declared lift, with dimensionless epsilon. Directions carry their nodal
coordinate units. Common initial offsets can be supplied identically without
changing the coefficient. Exact rational inputs are preserved; other reals
use shared represented-real admission. Booleans, nonfinite values, unordered
containers, wrong cardinality and noncentered directions reject. The model
uses its admitted effective coefficients; zero EPI weight, nonpositive capacity
or radius, and sectors other than nonboolean integer `+1` or `-1` reject.
This is an ideal family declaration, not admission of a materialized graph.

For `D*v = v_(i+1)-v_(i-1)` and `kappa=target_sector*2*pi/5`, the exact
`cyclic_pairing=u.T*D*v` supplies
`C=w*sin(kappa)*cyclic_pairing/(10*e*beta*pi*cos(kappa)^2)`.
The limiting lifted phase shift is `C*epsilon^2+O(epsilon^3)`.
The positive common capacity cancels from C; it still affects the clock
and the response to an added connection. A zero pairing makes this quadratic
term zero, without proving the absence of all higher-order memory.

The immutable `RelationalCycleMemoryBounds` retains:

- The model, directions, common capacity, declared radius and sector;
  exact `cyclic_pairing` and `quadratic_phase_shift_coefficient_bounds`.
- `quadratic_left_port_form_rate_coefficient_bounds`: C multiplied by
  `-capacity*w/[pi*(1+2*cos(kappa))]`. This is the leading initial form-rate
  response after hypothetical one-port contact with an untouched, aligned,
  equal-form and equal-capacity reference C5. It is not a measured response.
- True-pi/trigonometric bounds, `reference_storage_bounds`,
  `storage_upper_bound`, `capture_barrier_bounds`, and strict lower bounds
  on the family acute and storage margins. These establish a sufficient
  recovery basin for every `abs(epsilon)<=amplitude_radius` when
  `basin_admitted` is true; failures retain `unavailable_reasons`.
- `remainder_order=3` and **`remainder_bound=None`**. The coefficient remains
  available when a requested family radius fails the sufficient basin test,
  since it concerns the local limit as epsilon tends to zero.

The family storage bound uses the existing cycle Dirichlet owner and cosine
barrier, with no trajectory or fitted response. Certified coefficient bounds
and an admitted recovery radius do **not** enclose the finite-amplitude final
offset or certify its sign at a chosen nonzero amplitude. The missing remainder
must not be replaced by zero. Exact scaling preserves tiny nonzero input
directions and model denominators without first rounding them into the
interval grid.

The observable distinction requires a retained reference and a declared
readout; an isolated common phase rotation is a symmetry. Hypothetical contact
cost is even in that phase difference, while the signed form-rate response
distinguishes its sign. Contact can require supplied work and change the future
state. Neither an actual event, a nondestructive measurement nor a new clock
is installed. The SDK exporter retains this static report; graph-dependent
mean/covariance and contact readouts remain with `relational_pattern` and
`relational_attachment`, rather than a duplicate `Network` execution method.
See [usage](../../guides/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#bound-an-ideal-pattern-memory-family),
[coefficient admission](../../../tests/test_relational_cycle_memory.py),
[independent mechanism controls](../../../tests/physics/test_relational_cycle_memory_mechanism.py)
and [SDK export](../../../tests/sdk/test_relational_cycle_memory.py).

<a id="relational-finite-memory"></a>
### Fixed finite-amplitude limiting memory

`certify_relational_cycle_memory(*, amplitude=Fraction(1, 2**20))` reuses the
general coefficient/capture owner with the narrower
[integrated error proof](../../../theory/nodal/RELATIONAL_PHASE_MEMORY.md#relational-finite-phase-memory).
Its premises are fixed: ideal unit C5, winding +1, `e=w=1/2`, `beta=nu=1`,
zero means, `x(0)=+/-epsilon*(1,-1,0,0,0)` and
`theta(0)=theta_*+epsilon*(0,1,-1,0,0)`. The supplied law remains unforced,
with fixed support/capacity throughout recovery. An untouched aligned C5
supplies the reference for hypothetical matching-port contact.

Shared exact/represented admission requires `0 < amplitude <= 1/1024`;
invalid inputs raise. The immutable `RelationalFiniteMemoryCertificate`
retains the preparation, radius `1/100`, strict first-exit storage margin,
state/integral bounds and checked constants. Its
`phase_remainder_bound=2**15*amplitude**3` bounds the discrepancy between
the actual limiting phase and its enclosed quadratic approximation.
Each `RelationalFiniteMemoryCase` retains its asymptotic report and the
full `limiting_phase_shift_bounds`.

The nonlinear hypothetical readout is evaluated directly over those bounds:
the right-port form rate is
`atan(sin(delta)/(2*cos(2*pi/5)+cos(delta)))/(2*pi)` and the left rate is
its negative. `contact_storage_bounds` enclose `2*sin(delta/2)**2`, avoiding
subtraction near zero. This work is separate from continuous dissipation;
no reservoir, passive event or actual attachment is supplied.

`phase_signs_separated` and `contact_signs_separated` retain independent
verdicts; `status="admitted"` and `admitted=True` require both. At the
default amplitude, the positive-arm phase is between approximately
`5.48241e-13` and `6.05085e-13`, and its left readout lies between
`-5.95181e-14` and `-5.39267e-14`. The negative arm has the opposite signs.
The exact rational interval endpoints in the report are authoritative;
symmetric bounds do not prove exact nonlinear antisymmetry of the two arms.
The error bound is exactly `2**-45` at this amplitude.

An inconclusive sign bound returns `status="unavailable"` with explicit
reasons. For example, `1/1024` admits recovery but not phase-sign separation;
at `2**-200`, exact scaled phase signs survive while fixed interval precision
cannot resolve the contact signs. Neither result becomes an exact zero.
Failure of the fixed proof constants raises `ArithmeticError`.

This is a limiting-state certificate without a trajectory, finite recovery
time, graph initialization error bound or physical measurement. The generic
family report's `remainder_bound=None` remains unchanged. The shared SDK
exporter preserves all bounds, nested scopes and the `admitted` property.
See [usage](../../guides/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#certify-one-finite-memory-preparation),
[admission/readout tests](../../../tests/test_relational_finite_memory.py),
[independent proof controls](../../../tests/physics/test_relational_finite_memory_proof.py)
and [export tests](../../../tests/sdk/test_relational_cycle_memory.py).

<a id="relational-memory-readout"></a>
### Finite-clock memory readout

`certify_relational_cycle_memory_readout(*, decay_blocks=40)` reuses the fixed
amplitude `2**-20` certificate and the
[continuous decay proof](../../../theory/nodal/RELATIONAL_PHASE_MEMORY.md#relational-finite-time-memory).
Require a nonnegative integer number of blocks; booleans and nonintegers
raise `TypeError`, negative integers raise `ValueError`. Each block is 4128
units of the declared structural clock. This is a conservative proof schedule,
not laboratory seconds or an earliest recovery time.

The `RelationalMemoryReadoutCertificate` retains the nested limiting-memory
certificate, exact horizon, modified-storage equivalence/initial bounds,
centered-state radius `R=8*epsilon/2**decay_blocks` and mean-phase tail
`Q=4096*R**2`. The auxiliary modified storage proves decay; it does not
replace the native storage or install a new dynamics. At the default horizon
165120, `R=2**-57` and `Q=2**-102`.

Each `RelationalMemoryReadoutCase` includes mean and port-phase intervals,
acute margins, resultant real-part bounds and three separate rate pairs:

- `*_no_contact_form_rate_bounds` enclose the residual isolated-ring motion;
  only the untouched right reference has an exact zero background.
- `*_contact_form_rate_bounds` evaluate the full proposed degree-three port
  law, including residual form and phase. Left and right rates need not be
  exact negatives, and immediate phase rates need not vanish.
- `*_contact_form_rate_change_bounds` subtract the no-contact baseline,
  enclosing the change caused by the proposed connection at the same state.

`contact_storage_bounds` enclose the independent event work
`u[0]**2/2 + 2*sin((m+v[0])/2)**2`. The default bounds are strictly positive;
continuous loss is not credited as a reservoir. The separate analytical
`contact_rate_error_upper_bound=2*R+Q` and
`rate_change_error_upper_bound=4*R+Q` compare the unknown actual finite-time
rates/increments with the unknown actual limiting readout. They are not
error bars around an arbitrary midpoint: the limiting-memory interval has
its own retained uncertainty.

The three phase/contact/increment separation flags remain independent.
`admitted` requires all three, with per-case and overall unavailable reasons.
At 40 blocks both preparations have opposite resolved signs at both ports;
at zero blocks conservative bounds are unavailable, not a zero response.
Resultant or acute-domain arithmetic failures raise `ArithmeticError`.
Intervals deliberately discard correlations and may be wider than necessary.

This reader performs no trajectory, contact or subsequent evolution. Its
certificate concerns a hypothetical one-sided response at a finite clock time
under the fixed ideal preparation, not a binary64 initialization guarantee or
a finite accumulated measurement. The shared SDK exporter preserves exact
time, nested evidence, all three observations and the `admitted` property.
See [usage](../../guides/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#bound-memory-readout-at-a-finite-time),
[API tests](../../../tests/test_relational_memory_readout.py),
[proof controls](../../../tests/physics/test_relational_finite_memory_proof.py)
and [native mechanism controls](../../../tests/physics/test_relational_cycle_memory_mechanism.py).

<a id="relational-memory-contact"></a>
### Accumulated response during a finite contact

[`certify_relational_memory_contact(*, duration=Fraction(1, 4096))`](../../../src/tnfr/physics/relational_memory_contact.py)
consumes the fixed 40-block readout and the
[continuous short-contact proof](../../../theory/nodal/RELATIONAL_PHASE_MEMORY.md#relational-finite-contact-memory).
It assumes one matching-port unit bridge is inserted at structural time
165120 and held throughout the declared duration, with the same native law,
unit capacities and preparation. The alternative keeps both rings separate
and continues their recovery. This reader executes neither alternative.

Shared exact/represented admission requires `0 < duration <= 1/3`; invalid
values raise, including booleans. This domain keeps the shared exact
exponential enclosure within its interval of admission; it is not a universal
maximum connection lifetime. Exact rational durations are not rounded to
binary64 before evaluation.

`RelationalMemoryContactCertificate` retains the nested readout, exact
duration/end time, neighborhood, proof constants, exponential bounds and
per-arm `RelationalMemoryContactCase` results. If the initial distance from
the aligned joined equilibrium is bounded by `rho`, the whole contact state
is bounded by `rho*exp(3*h)` and every coordinate's change by
`rho*(exp(3*h)-1)`. The accumulated form error about `h*initial_rate` is
`rho*(exp(3*h)-1-3*h)`, enclosed with the shared rational series. These are
continuous bounds, not an Euler step, a fitted acceleration or a zero remainder.

Each case preserves three different accumulated observations:

- `*_contact_form_change_bounds`: final-minus-initial port form on the
  joined support, including the continuous remainder.
- `*_no_contact_form_change_bounds`: independently evolving isolated control.
  The left ring retains a bounded residual change; the right reference stays
  exactly at its zero-form equilibrium.
- `*_contact_induced_form_change_bounds`: the difference between the two
  evolved solutions at the same elapsed time. It is not a subtraction of
  two initial rates extrapolated without error.

Positive acute margins apply over the whole interval, preserving both ring
windings. This does not claim an unchanged shape or arbitrary future lifetime.
The inherited `contact_storage_bounds` give the insertion cost;
`required_event_work_upper_bound` is an upper bound sufficient to cover either
arm, not a minimal necessary supply or evidence that a reservoir exists.
`continuous_loss_upper_bound` is a separate bound on subsequent dissipation.
No removal, event selector or credit from earlier loss is supplied.

At the default duration `1/4096`, all accumulated contact and contact-induced
signs separate, and both windings persist. For the positive preparation, the
right reference port change is approximately between `1.30029e-17` and
`1.46936e-17`; exact endpoints in the report are authoritative. At duration
`1/3`, the state/winding bounds still pass while the response signs are
unavailable. All intervals and reasons are retained; an inconclusive bound
does not demonstrate a lost or zero signal. Fixed proof/containment failures
raise `ArithmeticError`. `admitted` requires both sign verdicts and winding
preservation; the SDK exporter retains that property and all nested evidence.

The [guide](../../guides/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#certify-accumulated-response-during-contact),
[API controls](../../../tests/test_relational_memory_contact.py),
[proof controls](../../../tests/physics/test_relational_finite_memory_proof.py)
and [native field controls](../../../tests/physics/test_relational_cycle_memory_mechanism.py)
retain this fixed ideal-law scope. Physical clock/measurement, numerical
initialization error and lasting memory after detachment are separate duties;
the fixed ideal retention certificate below addresses only the last of these.

<a id="relational-memory-retention"></a>
### Retained regional record after contact removal

[`certify_relational_memory_retention()`](../../../src/tnfr/physics/relational_memory_contact.py)
closes the fixed default contact with a supplied state-preserving removal of
its one bridge at `T+h`, where `T=165120` and `h=1/4096`. It consumes the
existing contact enclosure and the
[regional retention proof](../../../theory/nodal/RELATIONAL_PHASE_MEMORY.md#relational-retained-receiver-record),
without evaluating a trajectory or constructing a binary64 surrogate for its
ideal interval state. There are no new configurable preparation or event laws.

`RelationalMemoryRetentionCertificate` retains the nested contact, reference
storage and isolated capture barrier, separate receiver-mean and component
capture verdicts, and per-preparation `RelationalMemoryRetentionCase` bounds.
The untouched receiver starts with zero mean. Initially only its contact port
has a nonzero form rate, so its final mean lies in `h*right_initial_rate/5 +/- B`,
where `B` is the existing per-node accumulated form remainder. Averaging five
independent remainders costs `B`, not `B/5`. A single-port change is not the
regional mean, and the joined support does not supply an equal-and-opposite
transfer law.

After removal each unit C5 retains its acute winding. Its storage is at most
the exact twist storage plus `20*rho_tube**2`, using the whole-contact state
tube and cancellation of the phase potential's first variation. A strict
margin below the inherited isolated capture barrier admits recovery of both
rings. With the declared common held capacities, each isolated arithmetic
form mean is conserved exactly. Thus `receiver_persistent_mean_bounds` also
encloses the receiver's eventual uniform form and remains valid at every time
after the cut. Its no-contact control has mean exactly zero.

The two preparation means are separated: approximately
`+[2.47071e-18, 3.06859e-18]` and `-[2.47071e-18, 3.06859e-18]`, with exact report
endpoints authoritative. These symmetric bounds do not prove exact
antisymmetry of the actual outcomes. The record is a change of mean relative
to the declared baseline, not a new winding, intrinsic shape label or physical
particle property.

The endpoint bridge form/phase bounds enclose its storage before removal.
`removal_storage_change_bounds` is the negative of that cost; it is separate
from insertion work and continuous loss. A negative jump neither identifies
where energy goes nor installs a reservoir. The report preserves availability
reasons and requires regional sign separation, both component captures and
nonincreasing removal storage for `admitted`. A successful contact port sign
alone cannot establish retention. No disconnected union is passed to the
connected native executor; future recovery is a componentwise theorem.

The [guide](../../guides/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#certify-a-retained-receiver-record),
[API controls](../../../tests/test_relational_memory_contact.py),
[proof controls](../../../tests/physics/test_relational_finite_memory_proof.py),
[native mean controls](../../../tests/physics/test_relational_cycle_memory_mechanism.py)
and [SDK export](../../../tests/sdk/test_relational_cycle_memory.py) retain this
conditional scope. Neither event occurrence, laboratory precision nor physical
identification is certified.

### Validated conditional relational transit

[`certify_relational_transit_capture`](../../../src/tnfr/physics/relational_transit.py)
and `Network.relational_transit_capture(*, model, cycles, horizon, time_step,
order=12, requested_sector=1)` perform a read-only proof computation for the
ideal continuous law.
The [validated-transit derivation](../../../theory/nodal/RELATIONAL_FORMATION_CONTROLS.md#relational-validated-transit)
owns the reduction and enclosure argument. This API does not advance the
graph, execute a numerical engine step, select operators or alter the supplied
preparation. A represented Euler chord is not used as an exact ODE tube.

The initial state must have the same exact copied/reflected two-ring support
and coordinates as `relational_capture`, with held unit capacities, positive
coefficients and `model.phase_domain="positive_resultant"`. Existing graph,
scalar and cycle admission remains active. Reflection defects are never
projected away. The initial snapshot need not already pass a protected-basin
rectangle or storage test: its complete detached `RelationalCaptureCertificate`
is retained as `initial`, independently of the transit verdict.
The regular engine option does not broaden this proof: its interval system
still requires the positive-resultant chamber throughout every accepted tube.

`horizon` and `time_step` must be strictly positive exact `Fraction` or integer
values; floats and Booleans are rejected. These times are relative structural
durations from the supplied state, with no interpretation of a stored graph
clock or laboratory units. `order` must be an integer from 4 through 16.
The work policy admits at most 4096 declared proof steps. The last interval is
shortened to reach the exact horizon; there is no automatic retry, preparation
change, adaptive step selection or horizon extension.

Interval coordinates are `(q,r,a,b)`, where `q=3*A-B` and `r=2*B-A`. Every
accepted `TransitStep` retains a strict whole-time Picard inclusion, positive
relative-resultant bounds for all distinct node rows including the central
row, and a rational endpoint enclosure. Centered interval Taylor expansion
bounds analytic truncation on that whole tube. A Metzler matrix comparison
propagates the entering coordinate uncertainty using enclosed Jacobian bounds;
128-bit outward dyadic arithmetic retains rounding uncertainty. Whole-time
regularity, local truncation and propagated error are separate obligations.
The analytic `atan(u)/u` enclosure retains its zero-safe series for constant
intervals inside `[-1/2,1/2]`. Outside that interval, Taylor evaluation requires
the constant interval to exclude zero and encloses the same function through
`atan(u)` and interval division. An interval outside the series domain that
also contains zero is unavailable. This extends the proof arithmetic's
admitted domain without changing the relational pressure law, phase chamber
or evolution equation. It introduces no additional evolution. Frozen source
and result bundles retain their original provenance.

`RelationalTransitCertificate.admitted` requires the entire requested horizon
to be validated and the entire final box to lie in a selected protected
rectangle, with joint-storage upper bound strictly below `7*beta`. The
`requested_sector` policy accepts integer `1`, `0`, `-1` or `None`; Booleans,
floats and other values are rejected. Its default `1` requires the positive
rectangle `(2*pi/3,pi) x (0,pi/2)`. Zero requests the consensus rectangle
`(-2*pi/3,2*pi/3) x (-pi/2,pi/2)`, and `-1` requests the reflection of the
positive rectangle. `None` accepts any one of these disjoint rectangles.
The point and interval capture owners share their affine margin definitions.

This sufficient endpoint gate invokes the existing protected-basin theorem.
Only full admission sets `target_sector` to the limiting sector actually
certified: consensus is the valid integer `0`, not an unavailable result.
`requested_sector` is a caller's admission policy; it neither changes the
trajectory nor establishes its current winding or limiting target. Endpoint
centers, sampled storage or floating winding alone cannot pass the gate.
`initial_winding_zero` separately
certifies a sufficient zero-winding condition on the original lifted phase
state; it is not a gate for a general transit certificate. A claim of entry
from winding zero requires both this flag and complete transit admission.

An unresolved proof step stops the calculation and retains the accepted
`steps`, `validated_horizon`, their final `endpoint`, the attempted
`failed_tube` and explicit `unavailable_reasons`. A fully validated horizon
can also return unavailable when the whole endpoint fails the sufficient
capture gate. Unavailable means the requested proof was not obtained, not that
the ideal trajectory fails to recover. No limiting target is assigned then.
The report preserves `initial_box`, requested clock/order policy,
`endpoint_storage` and enclosure-method scope. All three candidates' interval
margins appear in `candidate_rectangle_margin_bounds`, in sector order
`1,0,-1`. A matching strict rectangle supplies `rectangle_kind` and
`rectangle_margin_bounds`, even if another obligation leaves the overall
report unavailable; without a match they are `None` and empty respectively.
`positive_rectangle_margins` remains the positive candidate's margins for
compatibility, including when consensus or negative capture is requested.
These geometric observations do not override horizon, storage or initial-law
admission failures, which always leave `target_sector=None`.

`relational_report_to_dict` exports the full nested evidence with exact
fraction endpoints and validates node/cycle labels through `initial`.
Projection is detached evidence, not a restart format or provenance
authentication. Successful admission concerns this conditional ideal ODE;
it supplies no guarantee for subsequent finite engine execution and leaves
every original frozen experimental verdict unchanged. See the
[SDK usage](../../guides/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#validate-continuous-transit-to-a-protected-basin).

### Unequal reflected source/receiver continuous proof

`certify_relational_reflected_transit(initial_box, model=..., horizon=...,
time_step=..., order=...)` in
[`relational_reflected_transit.py`](../../../src/tnfr/physics/relational_reflected_transit.py)
accepts exact interval coordinates `(p,r,P,R,a,b,A,B)`. It declares two unit
C5 rings with matching adjacent bridges, unit held capacities, exact joint
reflection and no forcing or events. It requires the explicit regular model.
The source and receiver remain independent; no live graph is inspected,
projected onto symmetry or advanced. The
[derivation](../../../theory/nodal/RELATIONAL_NATIVE_FORMATION.md#regular-seeded-continuous-response)
owns reconstruction, the complete law and the ideal initial-value interpretation.

The private reflected-flow owner evaluates all six independent nodal
resultants, including the central rows, with shared interval/jet principal
arguments. Positive-real resultants use the removable-axis `atan(u)/u`
metric formula; sign-separated imaginary charts support negative real parts.
Zero/branch ambiguity rejects the box. A failed sufficient enclosure is not
a proved singularity. The shared Picard/Taylor/comparison kernel retains whole
tubes, domain margins, propagated initial uncertainty and Taylor remainders.
The existing copied-ring certificate retains its separate four-coordinate,
positive-resultant contract.

The certificate's `admitted` flag means only that the requested finite horizon
is enclosed. It separately retains winding flags, integrated loss, storage and
remaining budget relative to the two acute twists. A positive remaining budget
does not admit capture. A negative upper budget excludes that future target
under the same storage-loss law. Interval loss integration and direct storage
are intersected; inconsistent enclosures reject rather than being clamped.
The first unresolved tube is retained without automatic step refinement.
`to_dict()` reuses exact SDK dataclass projection, preserving rational bounds;
it does not authenticate publicly constructed certificates.

The [frozen producer](../../../benchmarks/relational_seeded_response.py) separates
preparation from response, retains source archives and never overwrites a
record. A separately identified numerical correction retains its prior-record
hash and cannot be described as an independent blind replication. This proof
API is not a new operator, event law or SDK graph-projection workflow.

Its `target-budget` study retains the same ideal initial state and law, starting
at zero with default total horizon one. An explicit exact rational `horizon`
can declare a separate bounded continuation (at most 256 steps); it must match
at preparation and evaluation. The short study's horizon is fixed.
The protocol declares the strict sign test
before evaluation. `passed` means full-horizon admission, the declared endpoint
width and a resolved sign, whether positive or negative; it is not a formation
verdict. `target_budget_verdict` describes the declared endpoint, while
`budget_sign_at_validated_horizon` preserves partial evidence. The first
certified negative budget remains a scoped future exclusion even if a later
tube or endpoint is unresolved. False winding flags mean preservation was not
certified; they do not prove a change. No retry or horizon extension is implicit.

`certify_relational_reflected_barrier(initial_box, model=...)` in the same
proof owner is a separate static observer. It admits the regular reflected
lift, computes storage from the state and checks the sharp separating surface
`a+A=4*pi/3`, whose storage is at least `7*beta`. Strictly lower storage on
the lower side excludes subsequent two-acute-winding-`+1` entry under the
same fixed-support unforced law. Its `obstructed` flag is a sufficient
conditional theorem; `unavailable` does not establish reachability. It exposes
energy and separator margins, domain evidence and exact JSON projection.
The [collective energy proof](../../../theory/nodal/RELATIONAL_NATIVE_FORMATION.md#reflected-collective-energy-barrier)
distinguishes this transition barrier from the smaller target minimum. The
observer neither changes live state nor supplies convergence or future
regularity. Applying it retrospectively does not rewrite a frozen verdict.

### Ideal reflected equilibria and full-network stability

`certify_relational_reflected_equilibrium(family, model=..., orientation=1)` in
[`relational_reflected_equilibria.py`](../../../src/tnfr/physics/relational_reflected_equilibria.py)
encloses one named analytic state: `consensus`, `aligned_twist`,
`aligned_saddle` or `opposite_twist`. The latter three have orientations `+1`
and `-1`; consensus has one representative. This stability certificate requires
the explicit regular model and positive EPI weight, phase weight and storage
scale. Zero dissipation is outside its recovery contract.

Coordinates enclose the named ideal point, not a box whose every state is an
equilibrium. The opposite branch uses the unique cubic root in `(1/8,1/7)`
and bounded cosine-based angle isolation. Source rates enclosing zero are a
consistency check; the [analytic classification](../../../theory/nodal/RELATIONAL_NATIVE_FORMATION.md#reflected-regular-equilibria)
establishes existence and completeness within the stated reflection lift.
No graph is recognized, rounded or projected into an equilibrium.

The report includes exact-coordinate bounds, storage, three edge-stiffness
classes, winding, domain margins, full phase-Hessian inertia and joint quotient
mode counts. Inertia is ordered positive/negative/zero, and quotient modes
stable/unstable/neutral. The full ten-node phase space has one common phase
zero mode; the full twenty-coordinate state has two common-offset zero modes.
The quotient counts exclude those two offsets. Stability is proved using the
full Hessian and symmetric quadratic pencil, including perturbations outside
reflection. `to_dict()` preserves exact bounds through shared SDK projection.
Existence and local recovery do not certify formation or global regularity.

### Global regularity evidence and boundary-access limits

`observe_relational_regularity(graph, model=...)` in
[`relational_regularity.py`](../../../src/tnfr/physics/relational_regularity.py)
requires an explicit regular model and captures one detached admitted field.
It encloses storage from exact represented forms and certified trigonometry.
If its upper bound is strictly below `beta*max(2,minimum_degree)`, the
[continuous theorem](../../../theory/nodal/RELATIONAL_DOMAIN_AND_CAPTURE.md#regular-domain-continuation-and-boundary-access)
guarantees future regularity under held unit support, capacities and model,
without forcing/events. The report gives conservative positive metric and
excluded-ray-distance lower bounds. Otherwise status is
`storage_test_unresolved` and those future margins are `None`; this does not
establish a singularity, instability or failure of a different certificate.

The report separately bounds instantaneous resultant derivatives using the
captured numerical phase rates. Those are exact supplied directions with
outward trigonometric evaluation, not enclosures of ideal ODE rates. The
instantaneous speed bounds cannot be multiplied by an arbitrary duration to
certify a future trajectory. Global regular continuation itself promises
neither convergence, continued acute admission nor finite Euler stability.

`certify_relational_reflected_boundary_exit(model=..., form_amplitude=1)` in
[`relational_reflected_boundary.py`](../../../src/tnfr/physics/relational_reflected_boundary.py)
instead evaluates a named ideal boundary family. The amplitude is an exact
positive integer or Fraction; the model is explicitly regular. Its endpoint
is not admitted for native execution. The report encloses limiting consumed
rates, resultants and derivatives, storage/loss, the exact `7*beta` gap and
the path-dependent full-state phase-rate limit. A positive gap proves the
counterexample also lies below that barrier.

The [local backward-flow argument](../../../theory/nodal/RELATIONAL_NATIVE_FORMATION.md#reflected-boundary-exit)
proves existence of nearby acute initial states reaching zero in finite time.
It supplies no chosen initial sample, exit-time estimate or prediction for
the frozen formation experiment. The smooth auxiliary reflected field is
not a full-state continuation rule. Six-resultant admission remains enforced;
the full vector field has no continuous extension at this endpoint.
Both reports use shared exact SDK projection through `to_dict()` and execute
no trajectory, event, reset or model change.

### Retained formation robustness

[`audit_relational_formation_robustness`](../../../src/tnfr/research/relational_formation_robustness.py)
reads the original continuous-transit report and its sibling protocol/source
archive. It checks their retained digests and archived sources before using
the whole-time tubes. It does not run the producer or evolve a graph.
The physical-flow formulas retain their archived syntax. An audit-only
four-coordinate comparison oracle is also checked against the archived syntax;
the current shared comparison kernel must match its exact matrix on every
consumed inflated tube. A refactored delegate can therefore retain the frozen
premise, while a changed comparison formula is inconsistent. The audit records
the shared comparison owner's source digest without changing the archive.

The [formation-robustness theorem](../../../theory/nodal/RELATIONAL_FORMATION_CONTROLS.md#relational-formation-law-robustness)
uses the exact same preparation, support, held unit capacities, coefficients
and structural clock. Each retained four-coordinate tube is enlarged by
`1/1024`. Shared interval geometry and reference Jacobians establish a regular
corridor, a cumulative logarithmic norm bound and a protected endpoint. The
comparison controls a changed law throughout the corridor, rather than
relabelling the original tube as its trajectory enclosure.

`RelationalFormationRobustnessAudit` retains reference provenance, exact
arithmetic evidence and separate `RelationalFormationLawBound` records for
the rho and eta comparisons. Its `admitted` property requires all evidence
checks to pass. `to_dict()` preserves exact fractions; it does not authenticate
a publicly constructed report. Missing records and unresolved proof bounds
remain unavailable; inconsistent record identities reject the evidence before
derivative calculation. Neither case reports failed physical formation.

The two parameter caps apply separately, with the other correction absent.
They are conservative sufficient mathematical bounds, not optimal tolerances,
physical constants or new modes of `RelationalExchangeModel`. Capture uses
the reflected `R+` region with storage below `7*beta`; the saved endpoint is
not admitted to the stricter full-state acute-energy region. Exact symmetry
preservation by the compared laws is essential to this quantitative result.
The original reference certificate and all frozen verdicts remain unchanged.
