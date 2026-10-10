# TNFR Testing Guide

This page owns local validation instructions. [pyproject.toml](pyproject.toml)
defines tools and dependencies; [the workflow guide](.github/WORKFLOWS.md)
describes CI, whose executable commands belong to the linked workflow YAML.
Expected mathematics and contracts follow [AGENTS.md](AGENTS.md)
and the source being tested. A passing test establishes its asserted behavior
and domain, not a general physical theorem.

## Run the repository tests

Run commands from the repository root with the interpreter of a separate
environment. Install the editable project and the dependency groups used by
the main CI test job:

```sh
python -m pip install -e ".[test,numpy,yaml,orjson]"
python -m pytest
```

`test` is the compatibility alias for `test-all`; NumPy is already a core
dependency. There are no `dev` or `all` extras. Smaller extras such as
`test-unit` install only their declared tools and may not support collection
of the whole repository.
Aggregate extras reference the smaller groups so their dependency bounds have
one owner; the compatibility alias does not maintain another dependency list.
The maintained benchmark instruments are standalone CLIs, so these groups do
not install `pytest-benchmark`. The `test-performance` extra remains available
as an explicit optional plugin for downstream tests; it does not define a
benchmark-directory test suite.

The default `testpaths` in [pyproject.toml](pyproject.toml) selects the routine
engine gate: nodal execution, operator/grammar contracts, numerical admission,
public diagnostics, caches, CLI and SDK. It includes the complete operator and
SDK directories and the listed production-field owners. The broad historical
research/certificate directories are not part of every routine run.
`pythonpath = ["src"]` imports the working tree, and `addopts = "-m 'not slow'"`
excludes marked slow cases. No custom collection plugin or second test registry
is involved. The test dependency requires pytest 7.4 or newer, supporting the
[standard testpaths wildcard configuration](https://docs.pytest.org/en/7.4.x/reference/reference.html#confval-testpaths).

Use the explicit root path for the complete retained inventory, or a research
module/directory when its owner changes:

```sh
python -m pytest tests --collect-only -q
python -m pytest tests -q
python -m pytest tests/physics/test_phase_alignment_metric.py -q
python -m pytest tests/mathematics -q
```

Explicit paths take precedence over `testpaths`. Research tests remain
executable evidence for retained conditional results; they have not been
declared obsolete merely because they are outside the routine gate. Add a new
production-field regression to `testpaths` when it belongs in that gate. A
routine pass is not a claim that every mathematical campaign was replayed.

Choose affected mechanisms through the [theory-to-execution map](theory/README.md#theory-to-execution),
then inspect their source and tests. That map links shared implementations,
representative controls and theorem owners; this guide does not maintain a
second inventory of research results or individual test cases.

## Shared evidence maintenance

The [evidence workflow](docs/guides/RESEARCH_EVIDENCE.md) separates maintained
artifact mechanics, model-specific audits and immutable archived programs.
For changes to their shared owners, select exact-record, archive and restoration
controls separately from scientific producers:

```sh
python -m pytest tests/research/test_artifact_io.py tests/research/test_frozen_source.py tests/scripts/test_restore_frozen_source.py tests/mathematics/test_validated_taylor_arithmetic.py tests/mathematics/test_validated_box_taylor.py tests/physics/test_sine_flow_kernel.py -q
```

Exercise duplicate/nonfinite JSON, exact tags before scalar comparison, unsafe or
duplicate archive names, expanded-byte limits, altered hashes and refusal to
replace an existing record. Restoration controls use disposable synthetic Git
repositories and read-only inspection of the committed freeze. Verify the full
pinned base, admitted supplemental files and destination rejection; do not
create a real reserved workspace or invoke an evaluator during regression.

Retained Taylor reconstruction checks primitive shape/source admission and
Horner/remainder/endpoint arithmetic. It does not prove stored derivative
enclosures or replace a consumer's source, law, event or strict Picard checks.
The shared sine field still needs independent edge-sum and jet controls for its
layouts and held coefficients. Select changed consumers and existing read-only
evidence audits in addition to these shared tests.

Reuse expensive parsed records and archive bytes through module-scoped fixtures;
copy a fixture before a mutation test. Pure test helpers must not import other
test modules' autouse fixtures. Keep no-execution guards scoped and reversible,
so an audit cannot leak patched producers or subprocess calls into unrelated
tests. Shared arithmetic must not collapse a producer and its independent
mathematical expectation into the same implementation. Current-code regression
does not update frozen evidence or authorize the pending research response.

## Select research checks by contract

Select the changed mathematical owner and its consuming APIs through the
[theory-to-execution map](theory/README.md#theory-to-execution). The map owns
individual module/test links; proofs own model-specific hypotheses, constants
and frozen preparations. This guide groups the obligations needed to choose
coverage, rather than repeating each research result.

For the [constitutive foundation audit](theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#justified-law-admission),
select the countermodels and complete-law controls whose premises are being
reassessed. Check pressure selection, capacity and support freedom, auxiliary
variational premises, phase closure, structural clock and event occurrence
separately; no one channel's success supplies the other laws.

```sh
python -m pytest tests/physics/test_pressure_constitutive_scope.py tests/physics/test_constitutive_capacity_scope.py tests/physics/test_constitutive_support_scope.py tests/physics/test_constitutive_variational_scope.py tests/physics/test_phase_exchange_constitutive_freedom.py tests/physics/test_structural_clock_scope.py tests/physics/test_selection_birth_closure.py tests/physics/test_relational_sine_comparison.py tests/research/test_sine_constitutive_robustness.py -q
```

These existing controls preserve scoped alternatives and known nonselection
results. They do not evaluate a new alternative at the currently acquired
objects or close a new research gate. Formation, response and storage evidence
remain associated with their own complete laws. Read retained evidence without
replaying its producers; test each new constitutive consequence independently
when its mathematical owner changes.

For claimed NFR representations and collective actions, also select
`tests/physics/test_epi_phase_representation_scope.py` and
`tests/physics/test_operator_quotient.py`; the routine gate covers registered
operator and grammar contracts. A new constitutive comparison must test its
actual full field, balance, matching order and law-specific derivative/source
bounds. Existing sine controls do not certify a different current, even when
target storage and the full tangent agree. Keep local-onset evidence separate
from finite-horizon discrimination and from acquisition under the new law.

For the [target-matched organized-pattern comparison](theory/nodal/ORGANIZED_PATTERN_CONSTITUTIVE_FREEDOM.md),
select `tests/physics/test_sine_class_constitutive_comparison.py` together with
`tests/physics/test_sine_class_neighbor_nonadditivity.py`. Check the alternative's
own potential and oriented edge derivatives, both complete rate rows, storage
loss and exact matching order. Rebuild the mixed onset from all incident edges,
including internal mediator edges; test both signs and symmetry nulls. Retain
exact phasor/scalar admission and independent derivative/identity bounds.
These are static algebra and local-onset controls. For the separate
[same-source acquisition proof](theory/nodal/ORGANIZED_PATTERN_CONSTITUTIVE_FREEDOM.md#organized-pattern-same-source-acquisition),
also select `tests/physics/test_sine_class_constitutive_acquisition.py` and
`tests/physics/test_sine_formed_class_pair.py`. Independently check full-family
transit, source cost, own-law Hessian and trapping margins. The endpoint
deformation gives the worst bounds for the whole declared parameter interval;
this is an analytic monotonicity argument, not a parameter sweep. Missing
positive stiffness or insufficient margins must remain unavailable. These
checks do not extend later dwell, contact or finite-response certificates.
The frozen sine producers remain unchanged.

For [finite constitutive discrimination of actually acquired objects](theory/nodal/ORGANIZED_PATTERN_CONSTITUTIVE_FREEDOM.md#organized-pattern-finite-carried-response),
select `tests/physics/test_sine_constitutive_handoff.py` and
`tests/physics/test_sine_constitutive_discrimination.py` with the acquisition,
onset and `test_sine_lyapunov.py` owners. Upper-Hessian changes also require the
existing maintenance, dipole, repeated-interaction and formed-contact consumers.
Check actual nonlinear stiffness, preserved source means, exact subgrid return
bounds, support/input work and all four branch identities. Derive the real
variation/remainder constants and static onset independently; propagate arbitrary
carried-state errors and all four raw-reading errors. A failed sufficient bound
is unavailability, not an observational collision. The observation allowances
are mathematical premises, not demonstrated physical calibration or resolution.

For [common scalar-sensor calibration](theory/nodal/ORGANIZED_PATTERN_CONSTITUTIVE_FREEDOM.md#organized-pattern-calibrated-observation),
select `tests/physics/test_sine_constitutive_sensor.py` with the carried-state
discrimination and handoff owners. Rebuild the absolute operating range and
single-input increments from the actual source and complete-law bounds;
propagate gain, curvature and four errors in sensor units. Distinguish strict
mixed-threshold separation, opposite signs and fixed-common-sensor full-record
separation. Exercise a common nonlinear sensor that changes the signs without
collapsing the records, insufficient calibration and invalid primitive inputs.
Re-admit source/calibration primitives rather than consuming cached verdicts,
and retain exact subgrid arithmetic. The full-record theorem quantifies one
shared sensor; it does not certify separation across independently varying
unknown sensor maps. Reuse module reports and prohibit scientific regeneration.

For [constitutive inheritance through organized interfaces](theory/nodal/ORGANIZED_PATTERN_CONSTITUTIVE_INHERITANCE.md),
select `tests/physics/test_sine_constitutive_inheritance.py`,
`tests/physics/test_sine_constitutive_inheritance_algebra.py` and
`tests/physics/test_sine_constitutive_recovery.py` with the existing
handoff, acquisition, discrimination and sensor owners when the shared
`_sine_constitutive_recovery.py` changes. Rebuild both fine supports and edge
sums independently, including degrees, first/higher harmonic factors and
registered phase offsets. Check both collective rows and the joint form,
clock, reciprocal-coefficient and constitutive-parameter map; a current-only
match does not establish complete-law equivalence. Test pulled-back storage,
its additive support baseline, inputs and original-coordinate observations.
Keep exact positive subgrid attachment costs, source mean constraints, own-law
Hessian/gap margins and the finite full-state defect. Exercise loss of
registration in ninth-moment-only observations and unavailable source/budget
premises. Algebraic parameter extension does not transfer formation, and
invariance of the organized family does not prove its generic attraction.

For [composition-compatible constitutive laws](theory/nodal/COMPOSITION_COMPATIBLE_CONSTITUTIVE_LAWS.md),
select `tests/physics/test_composition_compatible_constitutive.py` together
with `tests/physics/test_sine_constitutive_inheritance_algebra.py`. Check
harmonic support and target moments independently, including controls that
satisfy those linear restrictions but fail potential nonnegativity. Derive
the counterfamily's potential/current relation and target derivatives from
its explicit formula. Reconstruct the two full fine supports and compare
both collective rows, storage, loss and the complete chart/clock/reciprocal
map; epsilon must remain unchanged. Exercise exact signed parameter endpoints,
invalid domains and signed form/phasor admission. These detached controls
establish static algebra, not acquisition, recovery, contact identity or
autonomous composition. Do not rerun frozen producers or borrow their
source certificates for the new law.

For [oriented-interface profile inheritance](theory/nodal/ORIENTED_INTERFACE_PROFILE_INHERITANCE.md),
select `tests/physics/test_oriented_interface_constitutive.py` with the
composition-compatible and inherited-algebra owners when shared harmonic
evaluation changes. Independently reconstruct weighted undirected incidences,
degree normalization, full fine rows, storage, loss and positive support work.
Check both evolution rows under the clock/form/reciprocal/phase-origin map,
including inputs and raw-coordinate observations. Exercise the explicit
nonsine family's interior obstruction and endpoint compatibility without
mistaking that family-specific result for exclusion of every law at one
weight. Retain the distinct midpoint counterfamily and the theorem's open
interval quantifier. Test invalid exact scalars and signed form/phase
admission before materialization. These static controls establish neither
acquired-source identity nor event occurrence; frozen producers stay unchanged.

For [network-wide organized compatibility](theory/nodal/ORGANIZED_NETWORK_COMPATIBILITY.md),
select `tests/physics/test_organized_network_compatibility.py` with
`tests/physics/test_oriented_interface_constitutive.py` when their shared
contact factors change. Independently reconstruct all 27 nodes, weighted
reverse incidences and degree-four normalization. Compare both fine and
collective rows, full storage and loss, including cross terms missing from
isolated-pair loss addition. Test the phase-cycle and amplitude conditions
separately, the fixed equal/unequal controls and the compatible endpoint.
Retain joint phasor consistency, marked-origin transformations, raw
observations, input work and supplied support costs. Check full-node
stationarity directly; exact closure and a nominal equilibrium are not
acquired-source capture or transverse stability certificates. Reject invalid
primitive scalars and phasors before deriving contact reports, and preserve
the previous pair APIs and frozen evidence.

For [capture of the acquired three-object assembly](theory/nodal/ACQUIRED_ORGANIZED_NETWORK_CAPTURE.md),
select `tests/physics/test_acquired_organized_network_capture.py` with
`tests/physics/test_organized_network_compatibility.py` and
`tests/physics/test_sine_constitutive_recovery.py`. Independently reconstruct
the complete weighted support and target fine-node balances for both fixed
controls. Keep the implicit unequal-control target separate from a rounded
phase assignment and the stationary equal-control reference. Check the
full-state storage remainder, sector boundary and source entry, including
all original residuals and their mean correlations. The containing sector
allows nonacute transient edges; target acuteness cannot replace its actual
proof. Retain attachment work separately from continuous loss and source
excess below the interval grid. Reject invalid preparation inputs before
source assessment, and exercise unavailable recovery rather than treating
it as instability. These controls run no trajectory or frozen producer.

For [predictive repeated composition](theory/nodal/REPEATED_ORGANIZED_COMPOSITION.md),
select `tests/physics/test_repeated_organized_composition.py` with
`tests/physics/test_sine_constitutive_inheritance.py` and
`tests/physics/test_sine_constitutive_recovery.py`. Independently reconstruct
the 36-node weighted rows, both grouping maps, storage and outer attachment
work. Verify the ordered-port counterexample against equal parent means and
internal storage, and retain the difference between exact family restriction
and the finite full-transverse approximation. Check strict bootstrap margins,
declared parent/child identity, response resolution and unavailable source
admission. Rebuild both original parent sources from primitive allowances;
their relative origins are additional preparation information. Degree-three
capture does not supply degree-four response or event work. These static
analytic tests evaluate no trajectory or frozen producer.

For [acquired parent registration](theory/nodal/ORGANIZED_PARENT_ACQUISITION.md),
select `tests/physics/test_organized_parent_acquisition.py` with the repeated
composition, isolated recovery and source-acquisition controls. Independently
check both separated modes against the complete 36-node rows, both clock
transformations, full-source error and strict entry/readout inequalities.
Keep actual reached forms and phase means in the predictor; the generic
parent residual ball does not imply the longer source-specific forecast.
Exercise all eight primitive allowances before recovery, unavailable sources,
exact residual work and the correlated antipodal symmetry without replacing
pi by a represented angle. These are static analytic controls; no trajectory
or frozen producer is regenerated.

For [acquired-parent restoring response](theory/nodal/ORGANIZED_PARENT_RESTORING_RESPONSE.md),
select `tests/physics/test_organized_parent_restoring_response.py` together with
the acquired-parent, repeated-composition and original child-source controls
above. Independently reconstruct the supplied phase jump, all three relative
modes, event work and both clock transformations. Verify unchanged parent means
at the event, distinct finite parent responses, carried fine errors and strict
recovery margins. Preserve nonzero reached forms; the post-event source is not
a fresh zero-velocity preparation. Exercise invalid primitives before source
work, unavailable sources and all fixed error/work budgets. Shared heat-bound
arithmetic must preserve the previous acquisition certificate exactly. These
are analytic and static tests, without replaying frozen trajectories.

For [continuous phase-event actuation](theory/nodal/ORGANIZED_PARENT_CONTINUOUS_ACTUATION.md),
select `tests/physics/test_organized_parent_continuous_actuation.py` with the
restoring-response, acquired-parent and original child-source controls above.
Independently check the forced complete-row storage balance, input and clock
transformations, exact endpoint displacement and integrated-loss lower bound.
Keep signed work distinct from input magnitude or positive supplied work.
Test strict budget obstruction and unavailable source admission separately;
an unproved realization is not a certified control. The new result must not
depend on later unforced restoration or readout verdicts. Reusing signed event
work must preserve the previous restoring-response report exactly. No driven
trajectory or frozen producer is evaluated by these analytic controls.

For [constructive phase-event realization](theory/nodal/ORGANIZED_PARENT_CONSTRUCTIVE_ACTUATION.md),
select `tests/physics/test_organized_parent_constructive_actuation.py` with
the continuous-actuation, restoring-response, acquired-parent, repeated-
composition and original source controls above. Independently verify the cubic
and both endpoint derivatives, the fixed inverse-Laplacian displacement,
complete forced rows and clock changes, and the exact integrated polynomial
loss. Check every path identity margin and the original fine uncertainty;
nominal endpoints alone do not establish full-source reachability. Verify
release separately from path admission, including its shifted clock and source
association. The detached path evaluator authenticates no preparation and
executes no solver or graph mutation. Shared interval field extraction must
preserve previous observations exactly; retain malformed-input and subgrid
controls. Do not replay frozen source producers.

For [collective actuation](theory/nodal/ORGANIZED_PARENT_COLLECTIVE_ACTUATION.md),
add `tests/physics/test_organized_parent_collective_actuation.py` to that
selection. Verify identical child input channels against independent complete
rows, including the inherited denominator and all clock conversions. The
input and predictor consume only the eight initial means. Keep the actual
fine state, compare both flows under the same input, and include both the
initial discarded residual and the final error in the endpoint budget.
Check actual work beyond reference work, every forecast coordinate, and
recovery after propagating endpoint mismatch. Old aggregate verdicts or an
outer-gap-only forecast cannot replace these premises. No release observation,
fine-state feedback, source refinement or frozen producer is admitted.

For [autonomous reservoir exchange](theory/nodal/ORGANIZED_PARENT_RESERVOIR_EXCHANGE.md),
select `tests/physics/test_organized_parent_reservoir_exchange.py` with the
acquisition, repeated-composition and original source controls above. Verify
both exchange signs, the factor-nine reservoir reaction, all three rows and
clocks, and the full joint storage derivative. Zero reservoir state is not a
disconnected port. Exercise the joint-error comparison with distinct reservoir
responses and check the cooperative supersolutions without replacing the
finite-inertia law by its singular limit. Retain the entire reservoir interval
and every original source allowance. Distinguish a proved failed readout,
an unavailable estimate and registration at the later deadline; the latter
does not prove a loss-and-return cycle. No frozen producer is replayed.

For [reservoir memory and equilibrium](theory/nodal/ORGANIZED_PARENT_RESERVOIR_MEMORY.md),
select `tests/physics/test_organized_parent_reservoir_memory.py` together with
the reservoir-exchange and repeated-composition owners. Check the retained
quantity against all complete rows, its leakage-dependent derivative, the
fixed phase lift, implicit equilibrium enclosure and the full fine-state
potential Hessian. Exercise strict mean, chart, phase-radius and storage-trap
conditions separately from equilibrium registration. Stability on a fixed
information leaf does not imply attraction to one point across different
leaves. An unresolved root or margin remains unavailable, and a local trap
does not certify capture of the previously acquired source.

For the [organized source port](theory/nodal/ORGANIZED_SOURCE_PORT.md), select
`tests/physics/test_organized_source_port.py` with the repeated-composition,
parent-acquisition and isolated-recovery owners. Independently sum the actual
45-node support, retain uniform normalization despite differing degrees, and
check both port channels, donor reaction, regional storage and contact work.
Distinguish exact carrier closure with retained conserved totals from a
scalar-source obstruction and fine-state mean nonclosure. Re-admit all ten
preparation allowances before rebuilding either source; transform both donor
rows and its complete formation/recovery clock. Initial winding and bounded
attachment work certify neither future retention nor a selected event.

For [organized-source capture](theory/nodal/ORGANIZED_SOURCE_CAPTURE.md),
select `tests/physics/test_organized_source_capture.py` with the source-port,
parent-acquisition and isolated-recovery owners. Check the full fine target,
both signed cycle-boundary faces and matching-edge faces, and the actual
reached-source excess rather than the earlier assembly storage or a carrier
projection. Retain exact fine errors below the trigonometric interval grid.
The same-source disconnected control must preserve its separate component
means; distinguish its obstruction from a simulated endpoint. Failed source,
chart or strict barrier premises yield unavailable capture. No finite settling
deadline or continuous parent-registration certificate follows.

For [acquired outer winding](theory/nodal/ORGANIZED_OUTER_WINDING.md), select
`tests/physics/test_organized_outer_winding.py` with isolated recovery,
prepared-entry and shared preparation contracts. Check every fine row of the
product law, the factor-two carrier clock, all 80 original relative source
directions, exact attachment work and full endpoint cross terms. All five
internal and nine outer winding observations are required. Cover all 180
signed boundary constraints, including the empty negative outer faces;
Jensen bounds cannot consume a nonacute proposed minimizer. The paired
actual-residual control may evolve and must retain outer winding zero.
Failed sufficient premises remain unavailable. Analytic-only checks must
exclude the prepared-entry trajectory regressions
`test_complete_nonlinear_response_is_inside_the_analytic_endpoint` and
`test_nonflat_corner_response_respects_mean_centered_and_correlated_edge_bounds`; do not
regenerate frozen response producers for this proof-only owner.

For the [outer-organization response](theory/nodal/ORGANIZED_OUTER_RESPONSE.md),
select `tests/physics/test_organized_outer_response.py` with the outer-winding,
isolated-recovery and shared preparation owners. Check the complete phase
jump, its zero global mean, all unchanged internal differences and both
signed boundary-work contributions. The control reaches time 200 with its
actual residuals; it is not reset to its nominal equilibrium. Verify the
finite nonlinear increment remainder, background motion, paired-source
association and all four raw observation errors. Work may be negative; the
absolute budget and each sector's full post-jump storage margin are separate
checks. Every original allowance must be admitted before source rebuilding;
failed premises remain unavailable. Keep trajectory and frozen-producer
guards active for these analytic contracts.

For [unknown-origin outer observation](theory/nodal/ORGANIZED_OUTER_OBSERVATION.md),
select `tests/physics/test_organized_outer_observation.py` with outer-response,
outer-winding and shared preparation owners. Verify the reusable finite
increment bound preserves the earlier protocol exactly. The inverse receives
only 15 primitive mean-form readings; opposite probes share one baseline,
which cancels in the odd combination and appears twice in the even one.
Check both quadratures, fine-angle allowances, every interface and the full
cycle sum. Necessary candidate intersections are not joint realizability
witnesses. Keep inconsistent records, unsupported premises and conditional
identification distinct. Independent static record enclosures must not be
described as measured finite trajectories. Probe work is per branch, and
the broader state class has only the declared finite identity horizon.

For [identity-preserving assembly work](theory/nodal/ORGANIZED_ASSEMBLY_WORK.md),
select `tests/physics/test_organized_assembly_work.py` with
`tests/physics/test_acquired_organized_network_capture.py`,
`tests/physics/test_organized_network_compatibility.py` and
`tests/physics/test_sine_constitutive_recovery.py` when shared geometry or
source admission changes. Check the internal acute winding-one constraint,
the full weighted contact cost and the outward required-work floor for both
fixed controls against independent endpoint calculations. Preserve arbitrary
fine form, internal phase deformation and relative origins; cross-contact
acuteness is not a premise. Retain the exact recovered-source excess below
the interval grid, and distinguish unavailable source recovery from a passive
counterexample. The positive floor is not asserted to be a sharp minimum.
Check primitive admission before source assessment and retain the earlier
capture controls; arbitrary reset endpoints inherit no capture verdict.
These static controls install no event and rerun no frozen producer.

For [joint acquisition on preexisting fine support](theory/nodal/ORGANIZED_JOINT_ACQUISITION.md),
select `tests/physics/test_organized_joint_acquisition.py` with
`tests/physics/test_acquired_organized_network_capture.py` and
`tests/physics/test_organized_network_compatibility.py`. Independently rebuild
the connected primitive source, actual weighted Laplacian, spectral decay,
both complete evolution rows and initial contact storage. Retain every
coordinate error and the separate initial zero sums; only global means are
conserved during connected evolution. Check winding-zero preparation,
time-200 winding-one entry and the full reached form/phase storage against
the shared target/barrier owner. Neither the old isolated recovery radius
nor its source verdict may supply this handoff. Exercise invalid primitives
before assessment and unavailable strict margins without inferring failure
of the actual flow. Preserve the earlier capture and work contracts; these
static checks rerun no trajectory or frozen producer.

For the [locally unprepared receiver mechanism](theory/nodal/ORGANIZED_RECEIVER_MECHANISM.md),
select `tests/physics/test_organized_receiver_mechanism.py` with
`tests/physics/test_organized_joint_acquisition.py` and
`tests/physics/test_organized_network_compatibility.py`. Rebuild the two-ramp
source on the complete equal-weight support, including receiver contact
storage, original coordinate errors and conserved global means. Check the
time-200 donor/receiver winding distinction, complete initial-layer heat
bound and strict receiver exclusion interval, actual regional boundary work
and mixed-coordinate identity against independent fine-node calculations;
check the slow comparison's heat and Lipschitz bounds. Preserve the distinction
between a necessary delay and eventual nonformation; neither the shared
target nor a phase-only approximation supplies full-state entry. Exercise
primitive admission and unavailable bounds before interpreting a report.
These analytic controls generate no trajectory or reserved response.

For the [slow receiver admission](theory/nodal/ORGANIZED_RECEIVER_SLOW_ADMISSION.md),
select `tests/physics/test_organized_receiver_slow_admission.py` with
`tests/physics/test_organized_receiver_mechanism.py`,
`tests/physics/test_organized_joint_acquisition.py` and
`tests/physics/test_acquired_organized_network_capture.py`. Check the exact
second-order phase equation and its energy against both complete fine rows;
the inverse Laplacian acts only on the conserved global mean-zero leaf.
Independently verify the source-inclusive, uniform first-order defect bound,
the time-400 phase-rate signs and the donor/receiver winding distinction.
Retain every original error and distinguish actual phase motion from the
earlier mixed-coordinate current. Exercise invalid primitive admission and
unavailable inequalities; no cached report, symmetric nominal source or
small defect supplies a whole-family capture verdict. These analytic controls
evaluate neither a reduced trajectory nor a frozen response producer.

For the [source-preserving receiver enclosure](theory/nodal/ORGANIZED_RECEIVER_ENCLOSURE.md),
select `tests/physics/test_organized_receiver_enclosure_geometry.py`,
`tests/research/test_organized_receiver_enclosure.py`,
`tests/research/test_organized_receiver_enclosure_lifecycle.py` and
`tests/research/test_organized_receiver_enclosure_evidence.py` with the shared
metric Taylor, frozen-source and exact-artifact controls. Use synthetic reference
states; never replay the reserved two-ramp trajectory during regression.
Check both mean-zero reflection sectors against an independent full-space
Jacobian certificate, fixed seams, signed scalar growth, endpoint recentering,
and whole-step donor margins. An unavailable donor corridor must not prevent
later certified endpoint loss. Audit retained records without regenerating
them; preserve first attempts, failed-step records and every valid prefix.

For [regional work and winding retention](theory/nodal/ORGANIZED_REGIONAL_RETENTION.md),
select `tests/physics/test_organized_regional_retention.py` with the receiver
mechanism and slow-admission controls above. Rebuild release bounds from
primitive source errors; a supplied all-prefix work budget tests a conditional
inequality and never becomes observed history. Exercise missing budgets,
strict equality, failed source admission and representation boundaries.
Check regional and contact storage, powers and both actual evolution rows
against independent fine-edge calculations. Include a state with increasing
regional storage despite complete loss, and opposite form states with opposite
normal phase rates. The proof owns the first-exit barrier; static examples
alone cannot establish it. Select the read-only enclosure evidence audit when
checking the inherited donor-loss consequence. No control regenerates a frozen
response or substitutes endpoint work for an all-prefix bound.

For [constitutive identity barriers and work](theory/nodal/CONSTITUTIVE_IDENTITY_BARRIERS.md),
select `tests/physics/test_constitutive_identity_barriers.py`,
`tests/physics/test_organized_regional_retention.py` and
`tests/physics/test_composition_compatible_constitutive.py`. Independently
evaluate the factored potential, the first-boundary minimizing configurations
and the strict exclusion/convexity margins used by the global proof. A few
sampled configurations are not a proof of minimality. Retain the narrower
barrier domain separately from the valid constitutive domain, leaving absent
minima unavailable. Check actual equal-network fine-edge target jets and
same-state regional powers, both complete rows, contact storage and loss under
each law. Exact quarter-turn phasors and represented phase lifts are distinct
inputs; rounded pi cannot establish an exact witness identity. Preserve the
zero-deformation observation and the sine-only source assessment. Select the
inherited-interface algebra and oriented-interface controls when changing the
shared edge evaluator. These checks run no trajectory or frozen producer.

For [constitutive receiver source and comparison](theory/nodal/CONSTITUTIVE_RECEIVER_COMPARISON.md),
select the source and comparison controls in the theory-to-execution map,
`tests/test_comparison_flow.py` and the existing receiver geometry and
read-only evidence checks. Admit the original errors and every law in the
fixed interval, then independently rebuild source storage, transport and
actual velocity-defect inequalities. Check the factored current residual,
forward/backward cone intersections, fresh expanded-domain growth, full
mean-zero error directions, signed scalar growth and one recentering per
original step. The module-scoped full-prefix fixture reads only the pinned
response archive and blocks trajectory/Taylor regeneration. It rebuilds
consumed endpoint arithmetic; stored Picard and derivative enclosures remain
declared numerical premises. Cached sine verdicts and radii must not enter
the changed-law decision. Test receiver retention separately from donor
endpoint availability; a failed donor margin is not a changed outcome.

For [preparation information and acquired modes](theory/nodal/SINE_PREPARATION_MODE_SELECTION.md),
select `tests/physics/test_sine_preparation_mode_selection.py` with the shared
preparation and isolated-class controls above. Reconstruct the Laplacian energy
isometry independently and test equality for paired perturbed states, not just
centers or overlapping energy intervals. Retain mapped coordinate-error bounds,
zero sums and unwrapped target lifts; verify winding in circular coordinates.
Check complete-family transit and capture against an independent calculation,
including strict-budget and unavailable cases. This source pairing is not a
symmetry of the dynamics and does not transfer a later interaction certificate.

For [source-path capture and phase passage](theory/nodal/SINE_PREPARATION_PATH_SELECTION.md),
also select `tests/physics/test_sine_preparation_path_selection.py` and
`tests/physics/test_relational_sine_sector_capture.py`. Check the exact
Cayley isometry independently, including its changed endpoint residual map.
Admit parameter intervals before materialization, retain uniform source
transport and correlated ramp-potential extrema, and compare the shared
all-face boundary bounds with the independent C9 formula. Distinguish
acute-cell capture from the old small-ball certificate. The critical-energy
obstruction proves later exclusion of winding two without a final-mode or
crossing-time prediction; unresolved bounds must not become nonformation.

For [terminal restrictions on those same sources](theory/nodal/SINE_PREPARATION_PATH_LIMITS.md),
also select `tests/physics/test_sine_preparation_path_limits.py` and
`tests/physics/test_odd_cycle_sine_geometry.py`. Check complete critical closure,
independent signed-Hessian inertia and the full two-row Jacobian. Energy tests
must retain equality and unresolved comparisons; an antipodal critical state
has no certified winding. Generic exclusion of saddle sources requires both
full-dimensional error rows at each fixed preparation parameter and never
selects a nominal source's basin. Changes to the shared odd-cycle algebra also
require the C5 critical-set and resonance consumers; preserve their exact
ordering, output, zero-current branches and existing proof scope.

For the [source-specific capture selector](theory/nodal/SINE_PREPARATION_PATH_CAPTURE.md),
select `tests/physics/test_sine_preparation_path_capture.py` together with the
path-source and acute-sector owners. Independently check the exact mixed-state
initialization, uniform hidden-form remainder, cone boundary derivatives,
finite passage budget, acute tube and complete storage. Retain the distinction
between each member's hitting/capture time and the common deadline justified
by subsequent trapping. Failed initial cones or strict margins stay unavailable;
capture does not assign the same deadline to a smaller recovery radius.

For [constitutive robustness of that selector](theory/nodal/SINE_PREPARATION_PATH_ROBUSTNESS.md),
select `tests/physics/test_sine_constitutive_path_capture.py`, the source-path
and sine-capture controls, and the constitutive comparison/acquisition owners.
Check the law-independent source extraction against its existing consumers.
Independently verify the deformed closing current `j(pi-q)`, the endpoint
minimum cancellation, own-law transit and exact quartic source correction,
finite contraction, full-cell convexity and storage trapping. The deformation
endpoint bounds apply to the whole interval by proved monotonicity, not sampling.
Reject malformed intervals before materialization; failed strict margins remain
unavailable. No sine classification or frozen response may supply a verdict.

For [class-mediated collective response](theory/nodal/SINE_CLASS_MEDIATED_RESPONSE.md),
select both the primitive/report controls and independent full-support
algebra. Rebuild the 27-node support, its `(3,4,3)` central degrees, common
early derivatives and the first class-dependent receiver term. Retain
paired probe/unprobed histories, all source errors, the complete phase-blind
postcontact control, exact event work and actual conserved-mean shifts.
Reflection and absent-probe controls must not invent a distinction that the
interface hides. A tangent coefficient alone cannot replace a finite
nonlinear remainder, fresh formation handoff or full-family identity.

```sh
python -m pytest tests/physics/test_sine_class_mediation.py tests/physics/test_sine_class_mediation_algebra.py tests/physics/test_sine_class_mediation_evidence.py -q
```

Before the protocol freeze, these controls must not evaluate an admitted
reserved source. After a retained outcome exists, audit its primitives and
arithmetic separately from report wiring; keep old formation/contact
evidence unchanged and do not rerun frozen producers for unrelated changes.
The [read-only saved-evidence suite](tests/physics/test_sine_class_mediation_evidence.py)
checks the archive/protocol association, preserved prospective proof,
original source and endpoint budgets, exact interval inflation, work and
identity margins, and all declared stopping predicates. It neither calls
the assessor nor executes its archived producer.

For [acquired-mediator effective memory](theory/nodal/SINE_CLASS_MEDIATED_MEMORY.md),
also select the memory reader/bound controls and independent full-support
algebra. Check the 36-visible/18-hidden partition, rational spatial factors
with separate ideal scalar enclosures, class-independent zero-lag kernel
and class-dependent first derivative. Retain the hidden initial source;
test its cancellation only in a paired tangent difference with the same
full initialization. Verify the nonlinear residual bound against both
probe and baseline, including exact zero budgets and failed sufficient
separation without erasing available finite bounds.

```sh
python -m pytest tests/physics/test_sine_class_memory.py tests/physics/test_sine_class_memory_algebra.py tests/physics/test_sine_class_mediation_evidence.py -q
```

The stationary-map controls must keep visible mobility and distinguish
linear paired cancellation from the nonlinear comparator's uncertain
visible sources. They establish no invariant midpoint section or fast
limit. The same-charge hidden perturbation checks the exact visible
acceleration difference; the proof separately supplies existence within
the finite formation image, without a measured reachable radius. Changes
to shared response coefficients must preserve the frozen source/report
arithmetic. These controls need no new acquisition or reserved campaign.

For [two-probe nonlinear superposition](theory/nodal/SINE_CLASS_NONLINEAR_SUPERPOSITION.md),
rebuild four continuations of one complete initial state with a common
final readout. The delayed-only control carries its unprobed state to the
second event. Check exact tangent cancellation, full-law reflection/sign
equivariance, the carried bridge-gap curvature identity and the ideal
cubic-amplitude, fourth-joint-time coefficient. Arbitrary source residuals
must remain admitted rather than being projected onto a symmetric subset.

```sh
python -m pytest tests/physics/test_sine_class_superposition.py tests/physics/test_sine_class_superposition_algebra.py tests/physics/test_sine_class_mediation_evidence.py -q
```

Keep scalar-statistic cancellation distinct from intersection of all four
endpoint record sets; neither proves continuous-history equivalence or
that every record is shared. Test equality at each sufficient noise
threshold, failed bounds without a fabricated positive response, and
signed/zero impulses and boundary event times. Independent work and
identity checks retain the actual preevent state, both conserved-mean
changes and the separate radius/storage guards. These are conditional
arithmetic and implementation controls, not a new acquired response or
permission to replay the frozen class-mediation assessment.

For the [finite nonlinear protocol](theory/nodal/SINE_CLASS_NONLINEAR_PROTOCOL.md),
test the fixed degree-32 heat polynomial as an analytic coefficient,
including rational history cancellation, actual-degree normalization,
three cosine channels and the contraction tail. The full-law remainder
must retain the odd internal corrections and the source bound valid on
the longer horizon. Use independent coefficient or algebra controls;
an unvalidated trajectory cannot replace the claimed enclosure.

```sh
python -m pytest tests/physics/test_sine_class_nonlinear_protocol.py tests/physics/test_sine_class_nonlinear_protocol_algebra.py tests/physics/test_sine_class_superposition.py tests/physics/test_sine_class_mediation_evidence.py -q
```

Distinguish true sign, recorded sign above `4*delta` and disjoint
nonlinear/tangent four-record sets above `8*delta`, including equality
and unavailable-sign cases. Check exact rational endpoints separately
from outward display intervals, signed impulses and exact null histories.
The new horizon admission must not enlarge the old superposition API's
domain. Reused work/identity ledgers retain both preevent states and
independent policy guards. These controls acquire no source or reserved
response and must not rerun a frozen producer.

For [organization-dependent nonlinear interaction](theory/nodal/SINE_CLASS_NONLINEAR_ORGANIZATION.md),
test exact common-channel cancellation before transcendental enclosure,
the mediator-channel heat tail, two complete nonlinear remainders and
independent source errors between classes. Keep the eight-reading recorded
contrast budget separate from the sixteen-error comparison against an
independently noisy zero-contrast alternative. Exercise exact nulls,
strict boundaries, unavailable signs and independent work/identity guards.

```sh
python -m pytest tests/physics/test_sine_class_nonlinear_organization.py tests/physics/test_sine_class_nonlinear_organization_algebra.py tests/physics/test_sine_class_nonlinear_protocol.py tests/physics/test_sine_class_nonlinear_protocol_algebra.py tests/physics/test_sine_class_mediation_evidence.py -q
```

The unchanged design must retain its unresolved full-law verdict despite
the signed heat contribution. Check that the independent nonlinear-error
estimate still straddles zero with source and sensor errors removed;
this proves a limitation of that estimate, not actual class equivalence
or overlap of the original reading vectors. Independent edge algebra and
heat quadrature are coefficient controls, not complete sine trajectories.
The tests acquire no source and replay no reserved response. SDK wiring
uses `test_reduced_port_sdk_wiring_does_not_evaluate_research` separately.

For the [independent class comparison protocol](theory/nodal/SINE_CLASS_COMPARISON_PROTOCOL.md),
use unrelated polynomial controls for the two-source composition. Both sources
must be admitted before either runs, and the first incomplete child must stop
later execution. Reconstruct complete law, source, event carry, step arithmetic,
counts and observations before consuming a child report. A saved band or status
cannot replace that evidence; derivative generation remains an explicit premise.

```sh
python -m pytest tests/physics/test_sine_class_comparison_readout.py tests/physics/test_sine_class_comparison_protocol_algebra.py tests/physics/test_sine_class_readout.py tests/physics/test_sine_class_four_history_readout_algebra.py tests/research/test_frozen_source.py tests/scripts/test_restore_frozen_source.py -q
```

Keep nominal reference covers separate from the original correlated acquired
families and their Cartesian outer covers. Check that source transport is added
once, while the cubic prediction never narrows a forward interval. Test both
open-prediction consistency checks, the transported width postcondition and the
distinct strict eight/sixteen-error margins. A declared step/order policy does
not guarantee the requested numerical width. Preserve unavailable and first
failure evidence; these tests execute no selected reference or reserved response.

The comparison freeze has a separate read-only association and synthetic-helper
gate: `python -m pytest tests/physics/test_sine_class_comparison_freeze.py -q`.
Reuse the shared source inspector for archive/base checks. Admit both source
recipes, fixed predictions, version policy and prospective proof prefix;
compile only the named pure helpers for serialization/error-retention controls.
Never execute an archived main or producer to validate the freeze.

The first retained comparison has its own read-only response gate:
`python -m pytest tests/physics/test_sine_class_comparison_evidence.py -q`.
Parse the large response once, reconstruct both complete branch trees with the
shared evidence reader, then independently rebuild the cross-class contrast,
source transfer and all observation criteria. Retained derivatives and Picard
generation remain execution premises. This gate must not rerun the producer,
regenerate a coefficient or replace missing response evidence.

For [repeated joined interaction](theory/nodal/SINE_CLASS_REPEATED_INTERACTION.md),
check the full joined degree metric, its exact spectral bounds, the distinction
between Euclidean and degree-weighted common-origin projections, and the
preserved means of all eight branches. Rebuild recurrent work and storage from
the new relative source family; it is not the original formation image.
Test exact return bounds below the interval grid and the independent tangent
residual allowance, including the common-mean counterexample and strict noise
boundary. Reuse the comparison audit's module fixture to associate the nominal
interval only after reconstructing its retained full response.

```sh
python -m pytest tests/physics/test_sine_class_repeated_interaction.py tests/physics/test_sine_lyapunov.py tests/physics/test_sine_class_comparison_evidence.py -q
```

These are conditional arithmetic and read-only evidence checks. They neither
simulate the repeated schedule nor replay acquisition or a frozen producer.
The quadratic tangent return and the nonlinear acute trapping proof retain
different premises despite sharing the modified-energy kernel.

For the [nonlinear collective interface](theory/nodal/SINE_CLASS_COLLECTIVE_INTERFACE.md),
check the six-port/hidden-state partition, retained source term, reflection-odd
quadratic feedback, and central pressure observers independently of the
coefficient producer. The finite input family and its approximation error do
not establish identity or work admission for arbitrary schedules. Exercise
separate nominal amplitude tails and nonlinear initialization defects without
assigning nominal parity to actual residuals. Simultaneous port jumps retain
their quadratic cross terms.

```sh
python -m pytest tests/physics/test_sine_class_collective_interface.py tests/physics/test_sine_class_cubic_evidence.py tests/physics/test_sine_class_repeated_interaction.py -q
```

The cubic evidence module reconstructs the supplied comparison premise once
before applying the interface certificate. It must not regenerate coefficients
or a nonlinear response. The modified unit-delay pressure helper also requires
the amplitude, superposition and repeated-interaction contract tests; it does
not merge their distinct initial storage, contact or mean hypotheses.

For [causal component composition](theory/nodal/SINE_CLASS_INTERFACE_COMPOSITION.md),
check both complete boundary rows against independent edge equations with the
actual joined degrees. Exercise component charge exchange, internal and bridge
storage, simultaneous signed event work, hidden initialization and primitive
full-lift residual admission. The explicit cubic residual includes hidden
quadratic feedback, nonlinear source interactions and omitted bridge terms.

```sh
python -m pytest tests/physics/test_sine_class_interface_composition.py tests/physics/test_sine_class_collective_interface.py tests/physics/test_sine_class_port_readout_evidence.py -q
```

The retained-response audit reuses its module fixture and reconstructed gap
to check that this substitution preserves the existing memory distinction.
It subtracts the new approximation defect once; actual-source transport and
both recording errors are already present. No coefficient or response is
regenerated. A tolerance for one impulse is not a uniform two-channel input-map
certificate, and output closeness alone proves neither feedback existence
nor exact storage, identity or work properties for an arbitrary substitute.

For [distinct-neighbor nonadditivity](theory/nodal/SINE_CLASS_NEIGHBOR_NONADDITIVITY.md),
derive the donor/receiver mixed onset from independent incident-edge algebra.
Reconstruct all normalized rows used by the finite phase and cubic-current
bounds; the mediator's internal edges contribute alongside its two contacts.
Check exact agreement with the collective descriptor while preventing the
onset calculation from constructing unconsumed coordinate-memory partitions.
Retain hidden quadratic recoupling, cubic phase feedback, amplitude tails and
the four nonlinear initialization defects. The exact common linear source
cancels only because all four histories start from the same complete state.

```sh
python -m pytest tests/physics/test_sine_class_neighbor_nonadditivity.py tests/physics/test_sine_class_collective_interface.py tests/physics/test_sine_class_superposition_algebra.py tests/physics/test_sine_class_interface_composition.py -q
```

Check the exact zero-input identities and preserve the scope of nominal
opposite-input symmetry. Exercise the earlier nonlinear observation budget,
strict independent-comparator threshold and separate work/identity admission.
The reused event ledger receives an explicit pressure bound for the commuting
distinct-port jumps; its old delayed donor default is not applicable. These
controls check the implementation of the conditional analytic distinction
using static algebra and rational envelopes. They neither evaluate selected time coefficients or
full-law responses nor establish the mathematical proof or exclusion of all
pairwise dynamical laws.

For the [independent distinct-neighbor protocol](theory/nodal/SINE_CLASS_NEIGHBOR_FORWARD_PROTOCOL.md),
check optional event/readout selector admission, legacy defaults, complete
state carry and retained arithmetic. Independent polynomial fixtures must
exercise simultaneous and delayed distinct-port events, observed-node jumps,
shared-prefix cancellation and first-failure behavior. Research controls
rebuild the signed absolute-phase cover and the four nonlinear source defects;
they keep numerical width, closed prediction overlap and four/eight-reading
error decisions separate.

```bash
python -m pytest tests/physics/test_sine_class_readout.py tests/physics/test_sine_class_comparison_readout.py tests/research/test_sine_class_neighbor_forward.py tests/research/test_sine_class_neighbor_forward_evaluator.py tests/research/test_frozen_source.py tests/scripts/test_restore_frozen_source.py -q
```

The new research policy tests use the shared no-regeneration guard. Evaluator
wiring uses unrelated temporary records and mocked scientific entry points;
its attempt, exception, export-failure and no-retry controls do not execute the
selected four-history response. Existing frozen default reports remain subject
to their read-only reconstruction audits. A successful implementation test is
not an independent full-law evaluation or a proof of source acquisition.

After freezing, run the separate read-only association gate:
`python -m pytest tests/research/test_sine_class_neighbor_forward_freeze.py -q`.
It pins archive, protocol, receipt and Git base, then checks primitive policy
and dormant evaluator structure under the no-regeneration guard. The evaluator
wiring test admits pinned archive/member bytes before compiling only function
definitions; it cannot silently fall back to a mutable build script. Freeze
inspection never invokes the reserved experiment or authorizes a retry.

For [nonlinear excess storage and loss](theory/nodal/SINE_CLASS_STORAGE_EXCESS.md),
run the response-free mathematical gate:

```bash
python -m pytest tests/physics/test_sine_class_storage_excess.py tests/physics/test_sine_class_neighbor_nonadditivity.py tests/physics/test_sine_class_collective_interface.py tests/physics/test_sine_class_interface_composition.py -q
```

Check full/tangent balances using their own original-coordinate states,
simultaneous event work, nonzero ordinary tangent cross-energy, exact onset
and static adjoint cones. Rebuild quartic, amplitude-tail and common-source
errors independently, including the zero-input source defect. Exercise exact
nulls, unresolved bounds, primitive admission and the separately supplied
coordinate-to-storage observation allowance. The shared no-regeneration guard
must block response and time-coefficient execution. A finite static sign is
an analytic prediction, not retained storage-response evidence.

For the [prepared passive-loss observer](theory/nodal/SINE_CLASS_STORAGE_EXCESS.md#sine-storage-prepared-observer),
check the complete full/tangent fields, original phase units, carried passive
loss, simultaneous events, global work limits and first-failure evidence.

```sh
python -m pytest tests/physics/test_sine_class_storage_readout.py tests/physics/test_sine_class_storage_excess.py tests/mathematics/test_validated_taylor_arithmetic.py -q
```

Synthetic polynomial fields test wiring and arithmetic reconstruction; an
unrelated small full-law reference checks the actual fields. Guards reject
the selected storage word and historical scientific producers. Re-admit
source, target, law and every consumed Taylor step before checking cached
losses. Derivative and strict Picard generation remain execution premises.
These controls prepare an instrument, not a frozen storage comparison or
an independent evaluation of the selected storage prediction.

The [first retained neighbor response](theory/nodal/SINE_CLASS_NEIGHBOR_FORWARD_PROTOCOL.md#sine-neighbor-forward-reserved-result)
has a separate read-only gate:
`python -m pytest tests/research/test_sine_class_neighbor_forward_evidence.py -q`.
Admit exact attempt, protocol, archive and sole outcome-member bytes before
decoding. Reuse one module-scoped report reconstruction to check all 64 full54
steps and distinct-port events. Rebuild source, width, prediction and recorded
separation decisions independently; cached endpoints/verdicts are checked,
never used as their own premises. Selective corruptions exercise selector,
carry and Taylor-arithmetic rejection. Shared guards block all scientific
regeneration; derivative and strict Picard generation remain retained
execution premises rather than authenticated by this arithmetic audit.

For the [changed-input collective prediction](theory/nodal/SINE_CLASS_CHANGED_INPUT_PREDICTION.md),
check grounded-path kernel coefficients, exact causal convolution weights,
hidden quadratic feedback and independent linear initialization. Compare low
orders with independently assembled edge equations; do not execute the old
full-state coefficient producer as a substitute for this causal implementation.
The new single mediator impulse needs degree-four work and its own mean shift.

```sh
python -m pytest tests/physics/test_sine_class_port_prediction.py tests/research/test_sine_class_collective_protocol.py tests/physics/test_sine_class_port_prediction_evidence.py -q
```

The evidence audit reconstructs retained endpoints and all consumed error
allowances without rerunning the selected prediction or a complete-law response.
Time-series error, interval rounding, physical source width, amplitude omission
and recording errors remain separate. Success against the grounded tangent
alternative does not prove nonlinear necessity, class discrimination or a
reserved forward comparison.

For the [independent changed-input full-law protocol](theory/nodal/SINE_CLASS_COLLECTIVE_FORWARD_PROTOCOL.md),
select the complete 54-coordinate producer, its retained-evidence reader and
the research policy together. Check both primitive sources before any field
execution, absolute phase lifts, the single mediator form event, all-coordinate
carry and the global first-failure attempt budget. Rebuild retained Taylor
endpoints before consuming readings; stored derivative and strict Picard
generation remain separate execution premises.

```sh
python -m pytest tests/physics/test_sine_class_port_readout.py tests/research/test_sine_class_collective_forward.py tests/research/test_sine_class_collective_forward_freeze.py tests/research/test_sine_class_collective_forward_evaluator.py tests/mathematics/test_validated_taylor_arithmetic.py -q
```

The policy controls reconstruct the already retained causal prediction without
regenerating its coefficients. They independently check target-plus-residual
source association, one actual-family source allowance, closed nominal
prediction overlap, the declared agreement tolerance and strict two-error
recorded separation. Amplitude and nonlinear-initialization errors belong to
the inspected reduced prediction, not the full-law forward enclosure.

This selection admits the producer and prospective freeze; it does not execute
the selected forward response. Freeze controls inspect source, protocol and
attempt boundaries. The [evaluator wiring controls](tests/research/test_sine_class_collective_forward_evaluator.py)
admit pinned archive/member bytes before compiling function definitions and
exercise only temporary records with mocked scientific calls and Git inspection.
They can be selected independently of the freeze audit. They check preflight rejection,
exclusive attempts and retained projection/export failures without executing
the selected producer or restoring its workspace. Numerical completion,
resolution, prediction consistency and discrimination remain separate verdicts;
an earlier passing prediction cannot narrow or substitute for an independent forward
interval.

For the [retained first full-law comparison](theory/nodal/SINE_CLASS_COLLECTIVE_FORWARD_PROTOCOL.md#sine-collective-forward-reserved-result),
select the [read-only response audit](tests/physics/test_sine_class_port_readout_evidence.py):

```sh
python -m pytest tests/physics/test_sine_class_port_readout_evidence.py -q
```

The audit reads the canonical response ZIP and preserves its exact uncompressed
record and attempt association. It re-admits both absolute full-state sources,
the form event, complete law and fixed schedule, then reconstructs retained
Taylor endpoints before applying the source allowance once. Prediction overlap,
the declared resolution tolerance and noisy-comparator separation are checked
independently. Producer and coefficient regeneration remain prohibited; stored
derivative enclosures and Picard generation remain explicit execution premises.
This finite comparison does not establish physical identification or nonlinear
necessity for the changed-input observation.

The forward-policy, forward-freeze and retained-prediction audits share the
lazy [execution guard](tests/sine_evidence_helpers.py). Each module opts in
separately: scientific generators and their imported aliases, acquisition and
restoration are blocked until fixture teardown. Subprocesses are prohibited
except for the freeze audit's exact pinned read-only Git commands. Independent
mathematical assertions and artifact hashes remain with each audit owner.
The [guard controls](tests/research/test_sine_evidence_helpers.py) check lazy
imports, alias coverage and restoration after failure or nested use:

```sh
python -m pytest tests/research/test_sine_evidence_helpers.py -q
```

For [common-scale amplitude feasibility](theory/nodal/SINE_CLASS_AMPLITUDE_FEASIBILITY.md),
check whole-interval coefficient homogeneity and higher-amplitude bounds,
the complex-domain boundary, the fixed source/noise allowances and the
separate work and identity verdicts. The supplied base coefficient is a
conditional premise of the public calculator; finite scalar admission
alone does not establish its association with the model.

```sh
python -m pytest tests/physics/test_sine_class_amplitude_feasibility.py tests/physics/test_sine_class_amplitude_feasibility_algebra.py tests/physics/test_sine_class_amplitude_feasibility_evidence.py tests/physics/test_sine_class_cubic_evidence.py -q
```

Independent algebra checks cover carried amplitude levels, weighted heat
symmetry and the donor Laplacian in the exact jump work. Reconstruct the
retained coefficient endpoints and tails before applying the conditional
calculator; do not regenerate coefficients or a nonlinear response.
Upper-scale work and radius bounds cover the interval by monotonicity,
whereas conserved means require their own interval bounds. If the shared
event ledger changes, also select the superposition, nonlinear protocol,
organization, cubic and spatial contract owners. Preserve their default
conservative bounds and frozen evidence.

For the [spatial class observation](theory/nodal/SINE_CLASS_SPATIAL_OBSERVATION.md),
test the two-node observation, receiver-odd locality, complete quadratic
phase feedback and the heat-only class-blind control. Exact low-order jets
check the first class-sensitive time coefficient independently of the
finite assessment. The higher-amplitude bound already compares classes;
sum its three nonzero-history contributions only once.

```sh
python -m pytest tests/physics/test_sine_class_spatial_observation.py tests/physics/test_sine_class_spatial_observation_algebra.py tests/physics/test_sine_class_contrast.py -q
```

Preserve arbitrary source residuals and sixteen independent nodal reading
errors; the separately noisy null comparison needs thirty-two. Reuse the
full-coordinate coefficient owner instead of a second solver. Previously
retained coordinates are prior information, not unseen responses. Test
the original eight-reading decisions and cubic/heat contracts when their
shared projection or decision helpers change.

The compact spatial assessment references the preceding complete
coefficient bundle. Its [read-only audit](tests/physics/test_sine_class_spatial_evidence.py)
reuses the full coefficient-evidence checks and reconstructs the new
two-node quadratic projection, higher-order bound and source/noise
decisions. Select both evidence owners together:

```sh
python -m pytest tests/physics/test_sine_class_spatial_evidence.py tests/physics/test_sine_class_cubic_evidence.py -q
```

The reference prevents duplicating complete series; it does not authenticate
their generation. No audit may execute a coefficient or response producer.

For the [complete cubic-amplitude response](theory/nodal/SINE_CLASS_CUBIC_RESPONSE.md),
check all three full-coordinate variation levels, the gamma scalings,
oriented-edge signs, quadratic odd-mode feedback and exact jump ancestry.
Time-polynomial truncation and the higher-amplitude Cauchy remainder need
independent controls. Fixed order 64 uses exponential-tail starts 65, 64
and 63, with each endpoint enclosure carried into its suffix. This analytic
coefficient recurrence does not change the full-flow integrator's limits.

```sh
python -m pytest tests/physics/test_sine_class_cubic_response.py tests/physics/test_sine_class_cubic_response_algebra.py tests/physics/test_sine_class_contrast.py tests/physics/test_sine_class_nonlinear_organization.py tests/physics/test_sine_class_nonlinear_organization_algebra.py -q
```

Use module fixtures for the complete finite coefficient; admission and
decision controls reuse it or use explicitly synthetic coefficients.
Nominal amplitude parity cannot remove arbitrary actual-source errors.
An unavailable complex-amplitude domain must withhold the full-response
bound, while exact event-pairing identities remain separate. A true sign
can coexist with a certified scalar noise-cancellation witness; neither
implies that the complete recorded vectors overlap. Preserve the original
heat method's unresolved outcome after shared decision refactoring. These
controls evaluate no acquired source or reserved nonlinear response.

The [retained coefficient evidence](docs/assets/sine_formed_classes/class-cubic-response-v1.evidence.zip)
has a separate read-only audit:

`python -m pytest tests/physics/test_sine_class_cubic_evidence.py -q`

It verifies bounded archive/source associations and re-admits the primitive
policy and all retained intervals. Rebuild the three-level event carry,
time-polynomial endpoints and tails, class subtraction and complete
source/noise/work decisions. Coefficient generation remains the archived
execution premise; the audit neither regenerates the coefficients nor
replays a nonlinear producer. Preserve the recorded packaging failure
before the numerical attempt and the subsequent complete source archive.

For the [four-history observation producer](theory/nodal/SINE_CLASS_NONLINEAR_PROTOCOL.md#sine-nonlinear-protocol-validated-readout),
select its full-state/event controls and the shared source-box kernel.
Use unrelated sources: check both rows from independent edge sums,
stationary states, exact donor-only jumps, all 54 carried coordinates,
shared-prefix ancestry and the delayed-only unprobed control. Zero delay,
final-time jumps and zero duration must preserve their declared events.

```sh
python -m pytest tests/physics/test_sine_class_readout.py tests/physics/test_sine_class_four_history_readout_algebra.py tests/mathematics/test_validated_box_taylor.py tests/physics/test_sine_two_port_readout.py tests/physics/test_sine_aperture_readout.py -q
```

The [independent field/source/prefix controls](tests/physics/test_sine_class_four_history_readout_algebra.py)
rebuild both nodal rows and retained first derivatives, distinguish an
acquired subset from its Cartesian cover, and reconstruct suffix receiver
increments under nonzero source width. Raw endpoint subtraction is a
separate outer bound. Exercise
failure inside a segment, budget exhaustion before an event and explicit
unattempted branches. All-four availability must not be inferred from a
completed prefix or an individual reading. Domain guards establish smooth
flow, not acquisition or identity. Interval widths contain source and
numerical enclosure effects; no sensor error is supplied by the producer.
Do not evaluate the prospective class-two source to tune this admission;
retain every earlier frozen source and response unchanged.

For the [matched four-history freeze](theory/nodal/SINE_CLASS_NONLINEAR_PROTOCOL.md#sine-nonlinear-protocol-frozen-evaluation),
audit immutable protocol/archive/receipt associations and the canonical
source cover from primitive arithmetic. Check the acquired-family
assumptions separately from its Cartesian cover, fixed event ancestry,
the unique-step budget and the isolated forward argument list. Read-only
controls must reject altered bytes, undeclared inputs and any response or
attempt claim at the freeze stage; they must not call acquisition,
prediction, the reserved producer or an archived worker. Test open-band
overlap/containment and strict zero/four-error/eight-error classification
using unrelated synthetic intervals, including equality and unavailable
evidence. Hash agreement establishes association, not authentication or
the truth of retained derivative enclosures.

```sh
python -m pytest tests/physics/test_sine_class_readout_freeze.py tests/physics/test_sine_class_readout_protocol_algebra.py -q
```

The [independent protocol arithmetic](tests/physics/test_sine_class_readout_protocol_algebra.py)
checks the fixed branch budget, source cover and strict observation
criteria without executing a reserved flow. The
[freeze audit](tests/physics/test_sine_class_readout_freeze.py)
checks the archived execution boundary and immutable byte associations;
it is not a retained-response audit.

For the [retained first outcome](theory/nodal/SINE_CLASS_NONLINEAR_PROTOCOL.md#sine-nonlinear-protocol-reserved-result),
select the [read-only response audit](tests/physics/test_sine_class_readout_evidence.py):

```sh
python -m pytest tests/physics/test_sine_class_readout_evidence.py -q
```

Its module fixture reads the canonical response ZIP once, verifies the
unchanged outer and uncompressed bytes, and checks the pinned
source/protocol/attempt association. It reconstructs every retained Taylor
increment and endpoint, full-state handoffs, exact events, raw endpoint
and shared-prefix mixed bounds, and all five declared predicates. Guards
prevent producer, field, jet, Picard and subprocess execution. Retained
derivative enclosures and their generation remain numerical premises;
arithmetic reconstruction is not independent derivative validation. The
source cover does not erase the acquired family's correlation requirements.

For a physical-source admission, test deductions from the declared source
law separately from manufacturer specifications and measured responses.
The [digital-PLL audit](theory/research/DPLL_PHYSICAL_ADMISSION.md) uses
[`test_dpll_physical_admission.py`](tests/research/test_dpll_physical_admission.py)
to derive the detector from exact XOR overlap, construct distinct delayed
histories with identical current state, and check the scoped current-state
and phase-map obstructions. These algebraic controls neither establish
experimental reachability of their supplied history family nor calibrate
an instrument. They require no response download, fitting or frozen replay.

```sh
python -m pytest tests/research/test_dpll_physical_admission.py -q
```

Shared admission changes can cross the routine/research selection boundary.
The default gate includes native relational execution through `tests/test_*.py`
and the listed conservation-diagnostic tests. It does **not** select
[`test_sine_admission.py`](tests/physics/test_sine_admission.py), the
`tests/physics/test_relational_sine_*.py` family, or the regional-transfer
research modules. For a change to the common stored-coefficient boundary,
exercise both native execution and its sine adapter explicitly:

```sh
python -m pytest tests/test_relational_exchange_execution.py tests/test_relational_regular_execution.py tests/physics/test_sine_admission.py -q
```

Then select the affected consumers from the theory-to-execution map. For a
chained report, cover the producer's primitive admission and the downstream
calculation: sampling smoothness to sample budgets, cycle modes to gains,
visible evidence to hidden-state/capacity inference and prior forecasts, or
captured states to regional currents and supplied increments. An SDK pass
checks its adapters and projection; it does not replace those mathematical
controls. A shared primitive-source change also needs the relative-pattern
and forecast-endpoint association controls, including unchanged held capacities
and partial validated horizons.

| Changed contract | Required independent controls |
| --- | --- |
| Complete law, state and coordinates | Differentiate both full nodal rows; retain support, degree normalization, capacity, clock and inputs. Compare native Arg, baseline sine and alternative-mobility laws only under their actual premises. Test moving references, common origins, zero coordinates and same-observation/different-response counterexamples. |
| Primitive and numerical admission | Reject malformed support, law identifiers, authoritative aliases, nonfinite values, Boolean physical scalars, invalid radii and missing consumed coordinates before arithmetic. Rebuild consumed geometry and gradients; cached flags or displayed intervals cannot widen exact admission. Preserve each consumer's zero-capacity/loss and work-limit domain. |
| Algebra, derivatives and storage | Use independent edge sums, high-precision evaluations, exact matrix identities or analytic solutions. Check signed work and loss, all field directions, endpoint events and strict boundaries. Symbolic irrational targets, rational probes and rounded graph states provide different evidence. |
| Relative state, uncertainty and memory | Retain hidden initialization, every environmental node, original port degrees, actual observation times and each member's conserved means. Distinguish unknown common origins, independent nodal residuals and correlated relative coordinates. Inference compatibility is not existence, identifiability or a replayed response. |
| Forecasts and continuous enclosures | Exercise strict Picard inclusion, Taylor remainders, whole-time tubes, endpoint chains and held-parameter semantics against separate analytic fixtures. Retain partial-horizon status and actual validated time. Samples, Euler chords and held-pressure integration cannot replace a coupled continuous enclosure. |
| Geometry, equilibrium and recovery | Check exact circular reconstruction, full sine-current cancellation, weighted gaps and Hessian inertia with independent full-node equations. Geometric feasibility is not equilibrium. Local recovery, all-sector capture and conservative trapping have different initial sets and loss premises. Failed sufficient inequalities remain unavailable. |
| Formation, preparation and budgets | Separate identity absent initially from a supplied target. Keep the original form cost, phase uncertainty and complete law in positive and negative controls. Distinguish scalar storage, directional loss, transient barrier passage, maintained endpoints and accumulated port work; an instantaneous sign is not an integrated supply. |
| Symmetry and composition | Test whole-support/state/capacity equivariance, cycle orientation, relabeling and genuine quotient reconstruction. Ordinary reflection, combined sign/reflection and set-invariant uncertainty boxes have different consequences. Retain external attachments, relative zero modes and noncommuting weighted operators. |
| Collective observations and structural events | Check all competitors, mutuality, unresolved ties and state/support admission separately. Compare quotient and inherited rates with the full fine field. Hypothetical bridges, cuts, AL proposals and resets retain their work budgets and read-only scope; conservation or eligible contact does not prove occurrence. |
| Resonance, pulse and recurrence | Retain the selected port/readout, full tangent pencil and actual coefficient domain. Distinguish gain peaks from complex poles, exact periodicity from orbital stability, and family recurrence from a chosen-state verdict. Independent quadrature or static variation controls do not prove a nonlinear infinite-time response. |
| SDK, CLI and evidence projection | Check delegation, one-source capture, immutable observations, unavailable values, supported node labels and exact fraction export. Direct report schemas and the generic SDK envelope are separate contracts. Malformed reports and source changes must not authenticate themselves through serialization. |

For local-response geometry inference, select the inverse owner and SDK
projection through the theory-to-execution map. Use independent full-node
edge sums and graph-distance cancellations to check the noncritical affine
family and its remote-current remainder. Synthetic observation bands test
monotone exclusion, calibration/noise propagation and unresolved arithmetic;
they are not reserved responses. Keep the nominal-family parameter separate
from the actual arc mean under phase residuals. An enclosing interval cannot
serve as an existence or non-identifiability witness. Retained dipole and
capture evidence can be audited without rerunning their frozen producers.

For successive-input geometry/gain inference, also select
[`test_sine_two_pulse_inference.py`](tests/physics/test_sine_two_pulse_inference.py).
Check the complete carried state, the factored leading determinant,
degenerate and unresolved rank, and all three reading-error coefficients.
Positive factors can certify rank even when their determinant product rounds
across zero; test factor-cancelled inversion and unresolved individual factors
separately, including horizons below the product's resolution floor.
A shared middle reading must retain its incidence coefficients; for this
inverse's opposite-sign rows the marginal sensor bounds coincide with the
conservative separate-increment bounds. Test exact large-offset cancellation and
distinguish interval width from sensor error. Synthetic leading responses
exercise the inverse but are not reserved complete-flow observations;
necessary marginal intervals prove neither joint realizability nor exact
full-state identifiability.

For the bounded observation-clock adapter, also select
[`test_sine_clock_inference.py`](tests/physics/test_sine_clock_inference.py)
and the shared joint-inverse suite. Check constant-rate reduction,
observation-time versus structural-time duration, maximum-duration error
admission, effective gain `J=G*rho`, exact positive-prior projections and
the shared three-reading error map. Include rational scale extremes and
rejection at invalid clock/horizon domains. The nested auxiliary envelope
must not be treated as an actual maximum-clock trajectory. Keep the exact
all-row-rate/clock equivalence distinct from an ideal-source curvature
control disproving exact gain/clock equivalence. Neither synthetic control
is a reserved response or uniform noisy clock-identification theorem.

For four-reading finite-curvature inference, also select
[`test_sine_curvature_inference.py`](tests/physics/test_sine_curvature_inference.py)
with the clock and joint-inverse suites. Independently check the raw
finite-difference coefficients, original-source and third-derivative
errors, positive divisions, necessary clock/gain projection and unchanged
actual-angle meaning. All four reading pairs require admission, including
a malformed half-window value absent from the coarse child. Preserve one
middle-reading history and exact offset cancellation; curvature is not a
new independent sensor error. Test unavailable refinement with a retained
coarse report, strict exclusions, touching constraints and informative as
well as nonimproving outer bounds. Exact local information results for an
ideal fixed-source subfamily do not imply noisy global identification.

For clock drift, select
[`test_sine_clock_drift_inference.py`](tests/physics/test_sine_clock_drift_inference.py)
with the curvature and clock suites. Check exact equal-exposure ambiguity,
positive global rate and derivative-bound admission, endpoint matching,
individual reading-transfer bounds and the unchanged sensor-error allowance.
Include zero drift and singleton-prior reduction, all four primitive pairs,
necessary mean-rate/gain projections and unavailable reference arithmetic.
Use an independent varying-clock full-state control for finite transfer;
this is an implementation check, not a reserved response or proof of
instantaneous-profile recovery. Do not treat the widened reference readings
as physical observations or a new independently sampled noise channel.

For finite-aperture observation, select
[`test_sine_aperture_inference.py`](tests/physics/test_sine_aperture_inference.py)
with the clock-drift and curvature suites. Reconstruct the fixed boxcar
moments and all four virtual-reading rows independently; test constant,
linear and quadratic exactness, cubic remainders and the separate
post-event second-window error. Keep all four primitive averaged pairs,
their original sensor-error map, clock transfer and numerical widths
distinct. The curvature child must receive zero additional sensor error.
Check source-gated bounds, zero-drift aperture error, necessary marginal
projection and method abstention. Independent full-state integration of
the averaging law is an implementation control, not a reserved response
or a calibration of the sensor kernel. Point-sample exposure equivalence
alone must not be reused as equality of interval averages.

For the [response-free resolution budget](theory/nodal/SINE_APERTURE_RESOLUTION.md),
select [`test_sine_aperture_budget.py`](tests/physics/test_sine_aperture_budget.py)
with the aperture inverse and SDK suites. Independently reconstruct moment
row norms, separate sensor/enclosure/clock/reconstruction errors, strict
source and quotient guards, and conditional width targets. Exercise exact
boundary equality, large and subgrid rational inputs, and failed sufficient
budgets without inventing unavailable width certificates. Rebuild the
noise-overlap threshold from the two complete-history remainder bounds;
check that only additive sensor error admits that existential witness.
Numerical halfwidth alone must not trigger it, and an inadmissible whole-ball
sufficient guard must not erase the valid zero-residual witness. Guard
these controls against producer, inverse and frozen-worker execution.
No selected readings or reserved responses are required for this theorem.

When changing the independent full-state response generator, select the
[direct source-box Taylor suite](tests/mathematics/test_validated_box_taylor.py)
and [two-port readout suite](tests/physics/test_sine_two_port_readout.py).
The former checks independent linear, coupled oscillator and nonlinear
solutions, uncertain-baseline cancellation, thirty-six and sixty-four
coordinates, domain failures and the fixed Picard budget. The latter
rebuilds both fast-clock sine rows and the actual support, retains all state
coordinates and tests complete-flow enclosures on unrelated synthetic
preparations. Its numerical cross-check is not a validated response or a
reserved experiment. Changes to the common Picard or jet owners also need
their existing comparison and retained-metric consumer suites.

For the [validated finite-aperture producer](theory/nodal/SINE_APERTURE_INFERENCE.md#sine-aperture-validated-producer),
also select [`test_sine_aperture_readout.py`](tests/physics/test_sine_aperture_readout.py).
Check affine endpoint-rate positivity separately from the smooth extended
field, both nodal clock-scaled rows, the observation-time integral row
and complete 38-coordinate carry. Verify all four exact aperture widths,
phase-only events and cumulative integral increments without resets or
unrelated endpoint subtraction. Independent stationary, exact-integral
and coupled numerical controls must not substitute the inverse's response
approximation for the forward law. Exercise a later-window failure and
retain its successful prefix, failed tube and absent complete result.
These controls admit an implementation; they do not execute a reserved
response or calibrate an averaging sensor.

```sh
python -m pytest tests/mathematics/test_validated_box_taylor.py tests/physics/test_sine_two_port_readout.py tests/physics/test_sine_two_port_inference.py tests/physics/test_sine_two_pulse_inference.py -q
python -m pytest tests/physics/test_sine_clock_inference.py tests/physics/test_sine_two_pulse_inference.py tests/sdk/test_relational_reports.py -q
python -m pytest tests/physics/test_sine_curvature_inference.py tests/physics/test_sine_clock_inference.py tests/physics/test_sine_two_pulse_inference.py -q
python -m pytest tests/physics/test_sine_clock_drift_inference.py tests/physics/test_sine_curvature_inference.py tests/physics/test_sine_clock_inference.py tests/sdk/test_relational_reports.py -q
python -m pytest tests/physics/test_sine_aperture_inference.py tests/physics/test_sine_clock_drift_inference.py tests/physics/test_sine_curvature_inference.py tests/sdk/test_relational_reports.py -q
python -m pytest tests/physics/test_sine_aperture_budget.py tests/physics/test_sine_aperture_inference.py tests/sdk/test_relational_reports.py -q
python -m pytest tests/physics/test_sine_aperture_readout.py tests/mathematics/test_validated_box_taylor.py tests/physics/test_sine_two_port_readout.py tests/sdk/test_relational_reports.py -q
python -m pytest tests/physics/test_sine_formed_evidence.py tests/physics/test_sine_curvature_evidence.py tests/physics/test_sine_clock_drift_evidence.py tests/physics/test_sine_aperture_evidence.py -q
```

The [reserved inference protocol](theory/nodal/SINE_TWO_PORT_INFERENCE.md#sine-two-port-inference-protocol)
separates the full response producer from the inverse. Verify the actual
inverse keyword allowlist: no hidden state, true angle, exact sensor gain,
realized error or producer remainder may enter. Retain the two scalar
readings and baseline correlation, independent calibration readings,
numerical interval width and per-reading error as distinct evidence.
Run retained-record audits after a first evaluation; tests must not
regenerate that reserved response or silently retune a failed budget.

The [reserved two-input protocol](theory/nodal/SINE_TWO_PULSE_INFERENCE.md#sine-two-pulse-inference-protocol)
uses one held but uncalibrated gain per trajectory and three readings.
The [retained-record suite](tests/physics/test_sine_formed_evidence.py)
checks both complete thirty-six-coordinate Taylor certificates, exact
endpoint/event carry, the one middle reading, public-only inverse packets,
original angle/gain coverage, marginal widths and the fixed controls.
Rebuild consumed arithmetic from saved primitives; cached success flags
are not the evidence. These audits execute neither frozen producer nor
inverse. Preserve a false optional `whole_window_acute_certified` flag
without treating it as a failed inference criterion or a proved trajectory
event.

The [retained four-reading protocol and result](theory/nodal/SINE_FINITE_CURVATURE_INFERENCE.md#sine-curvature-inference-result)
add a passive half-time observation and one held unknown clock per case.
Select the same shared Taylor/readout suites for the zero-jump continuation;
check all three segments and both complete endpoint handoffs, distinguishing
observed from structural segment times. The [dedicated read-only audit](tests/physics/test_sine_curvature_evidence.py)
checks the retained four-reading record and public ten-key packets without
replaying its producer or inverse. Rebuild finite-curvature, source, coverage, width
and fixed-control criteria from primitives, including available coarse
children under false clock/gain priors. One half-time reading and its error
must be reused consistently; a separate fitted curvature is not evidence.
Preserve first failure, numerical abstention and optional acute flags.

The [reserved nonconstant-clock result](theory/nodal/SINE_CLOCK_DRIFT_INFERENCE.md#sine-clock-drift-result)
is audited by [`test_sine_clock_drift_evidence.py`](tests/physics/test_sine_clock_drift_evidence.py).
Rebuild the primitive source, linear profile exposures, held sensor and
all nine full-state certificates, then the recorded differences and necessary
inverse bounds. Check the prospective sign/error margins, both endpoint
handoffs per history, exact eleven-key packets, independent public transfer
allowances, width/coverage and every fixed stopping predicate. The companion
uses an exact equal-exposure association to the positive history; it is not
a duplicate response. Reconstruct its derivative/rate admission and zero
integral correction without executing an archived helper, producer or inverse.
Saved derivative enclosures remain a premise, and hash/attempt receipts do
not independently authenticate chronology or physical acquisition.

The [retained finite-aperture result](theory/nodal/SINE_APERTURE_INFERENCE.md#sine-aperture-result)
is checked by [`test_sine_aperture_evidence.py`](tests/physics/test_sine_aperture_evidence.py).
Rebuild all twelve 38-coordinate step certificates, affine-clock provenance,
full endpoint carry, passive cumulative integral and exact observed-width
normalization. Reconstruct the raw averages, held sensor/errors, eleven-key
public packets, moment/error transfer, nested inverse and every frozen stop.
Check the signed-average/curvature budgets independently of the saved verdict.
Only the 36 nodal coordinates share the exact first-window exposure equality;
the rate and cumulative integral need not match the reference. No point-sample
companion association transfers automatically to these averages. Namespace
imports of shared audit helpers must not import their autouse fixtures into
another module. Keep producer/inverse/worker guards and frozen-byte checks;
stored derivative enclosures remain premises, with independently assembled
first-rate consistency and whole-tube inclusion checks.

### Boundaries that need explicit regression coverage

- **Chained evidence is rebuilt from its premises.** Change primitive inputs
  independently of cached fields, and check that downstream bounds are
  recomputed or unsupported declarations reject. Retain valid stored
  coefficients without renormalizing them. Test stale favorable flags, zeroed
  derivative bounds and inconsistent state/support associations with independent
  expected results. Invalid source declarations must fail before a solver runs
  or independent one-shot samples are consumed; a supplied signed endpoint
  increment must not be replaced by an evaluated form rate.
- **Shared admission does not mean shared theorem domains.** Specialized
  sine-cycle recovery retains its larger support domain (including the
  51-node control); general phase geometry has separate work caps. The
  validated comparison and retained-metric Taylor owners keep their
  24-coordinate limit, with boundary refusal and independent controls.
  Hidden-capacity sine forecasts use `2*n+1` coordinates. The separate
  direct source-box Taylor path admits at most 64 coordinates, without
  changing that comparison cap or introducing a Jacobian propagation.
  Test both policies and the full 36-coordinate two-port source; dimension
  admission alone certifies no response.
- **Metric uncertainty remains correlated.** Compare retained-ball propagation
  with independent analytic rotating, contracting and nonlinear flows, including
  boundary points outside coordinate axes. Check original-clock growth bounds,
  both directions, every domain margin and partial coverage. Include tiny local
  errors whose squares fall below the interval grid; normalized norm arithmetic
  must not create an artificial square-root precision floor. A coordinate
  projection is not the ball consumed by the next step, and a manufactured
  solver control is not a formation experiment.
- **Exact and represented geometry remain distinct.** A rounded `pi`, a
  small field residual or overlapping intervals cannot establish an exact
  antipodal state, equilibrium or symmetry. Test zero resultants, branch
  limits, unresolved denominators and strict equality boundaries explicitly.
  An admitted native proposal chord is not a continuous trajectory proof.
  Exact `Fraction` input to a graph may pass through binary64 capture;
  distinguish that source from an explicitly supplied rational preparation.
  An unrecognized symbolic cancellation with a zero-containing interval
  must remain unavailable, not become either equality or a counterexample.
- **Preparation and endpoint sets retain their provenance.** A tighter
  correlated source bound cannot replace a forecast's larger Cartesian
  endpoint enclosure. Partial forecasts keep their validated time and
  original requested-horizon status. Full-state capture must include remaining
  form storage, every sector face and the actual phase uncertainty; favorable
  phase geometry alone is insufficient.
- **Reduction retains information and scale.** Memory/second-order identities
  keep initial form, the weighted constant mode and moving history. Slow-phase
  controls retain the fast transient, memberwise references, original form
  remainder and preparation cost. Test zero horizon, tiny positive feedback,
  large growth and monotone exponential tails without silently clipping time.
  For collective contact observations, independently vary internal deviations
  at fixed means and verify the complete nodal response. Declared contact
  lifts can change a collective energy split while full storage is unchanged;
  test its signed remainder and exact local derivatives without treating them
  as a finite-horizon prediction. Exact-family membership, finite-width
  maintenance and acquisition into that neighborhood need separate controls.
  For reduced component ports, independently verify projection/lift identities,
  orbit multiplicities, the nonlinear bridge and its changed degree weights.
  Include odd initial errors and nonlinearly generated discarded modes; linear
  parity invariance is not an exact nonlinear quotient. Transfer the unchanged
  component law to the held-out receiver and compare total prediction error
  against the final outward full-response lower bound, retaining preparation
  and readout errors. Prove actual joined identity from the full law, not from
  the surrogate's stability or coordinate count.
  For network assembly, derive central mobility from every live contact
  degree and check exact charge/storage identities against the fine graph.
  Include a multiply connected port that rejects unchanged one-contact
  normalization, cycles with consistent component origins, and disconnected
  states whose instantaneous rows do not establish global capture. A uniform
  whole-state approximation bound must retain original preparation errors
  and generated odd modes; it is not a receiver-contrast or sensor budget.
  For all-time tracking, independently admit both invariant charts and the
  actual degree-weighted spectral gap. Preserve initial-error transients,
  generated odd modes and both conserved global mean floors; convergence to
  a target alone is not cross-flow contraction. Test separate form and phase
  resolution margins, including a valid envelope that resolves only one
  channel. Failure of a sufficient resolution bound is not trajectory error
  evidence, and a phase pass must not hide a failed natural form scale.
  A sharper form comparison must preserve operator order in its heat
  convolution, include the initial phase-to-form contribution and nonlinear
  bridge variation, and strictly admit both parity feedback gains. Exercise
  noncommuting operators and dyadic gain boundaries independently. Keep the
  previous partial artifact unchanged; a new method is separate evidence,
  not permission to enlarge the old allowance or replace its verdict.
  A two-port equilibrium admission must reconstruct every fine edge and
  cycle period, distinguish balanced circulation from nonzero nodal rates,
  and retain the matched-class zero-current control. Enclosed root existence
  and strict acute margins, not small residuals, justify the equilibrium;
  local recovery does not establish capture of a supplied initial family.
  For a storage handoff obstruction, reconstruct the complete boundary
  witness and its integer periods independently of the equilibrium root.
  Check the source lower bound across relative origins, the nonnegative
  full-form contribution and the radian phase-error allowance. Exercise the
  exact strict-margin boundary and reject Boolean or nonfinite radii. A
  lower-storage boundary witness excludes the scalar certificate, not capture
  by the full dynamics; current-source checks must leave frozen equilibrium
  evidence untouched.
  A finite two-port transit certificate must retain both initial form and
  phase errors, actual degrees and conserved means, every consumed clock
  factor and the full-state reconstruction. Verify its reference derivative
  independently on all eighteen nodes. Its acute-chart comparison needs a
  closed first-exit argument, not an assumed reference domain. Test actual
  endpoint changes against each member's own initial short gaps, with the
  initial error counted as well as the endpoint error. Whole-window acuteness
  and finite directional change do not establish eventual capture. Keep
  nonsymmetric error families and explicit unavailable margins in coverage.
  A longer capture gate must distinguish a validated nominal gradient
  reference from the full uncertain form-phase family. Check the folded
  metric and full reconstruction, whole-time tube admission, exact completed
  horizon and retained numerical radius. Every reference target, rate and
  chart prerequisite must be rebuilt from primitives. The analytic final
  interval continues the same full flow and needs its own original-form
  bound; a small phase distance alone is insufficient. Keep incomplete
  prefixes and failed capture margins unavailable, and preserve the first
  fixed-budget response even when its sufficient criteria fail.
  A supplied probe must retain the actual pre-event form and phase residuals,
  the exact jump, its storage work and the changed conserved-mean leaf.
  Reconstruct the receiver observation on each support: the common rule
  uses receiver degree masses twenty and eighteen, not identical coefficient
  vectors. Check the unjoined invariant directly, independent readout errors,
  complete-law finite response remainders and strict post-event trapping.
  An endpoint-ball assessor cannot certify the source's acquisition; test
  the separate finite source handoff without consuming cached verdicts.
  Keep a pure-heat transmission countermodel so the response is not reported
  as winding-specific evidence or selection of the sine constitutive law.
  An interior phase dipole needs the identical local coefficient vector in
  both models, with scalar observation errors distinguished from per-node
  errors. Verify its exact local sine response before bounding both full
  propagators; matching local adjacency does not prove finite response
  equality. Preserve the correlated target-angle contrast and every nonlinear
  remainder term. A degree-metric warmup must remove each connected
  component's own means, retain the combined control norm and prove strict
  original-form and phase budgets through the shared modified-energy
  kernels. Its starting source is already trapped; two norm caps alone
  cannot grant acquisition or trapping. Test missing warmup and target
  prerequisites so endpoint candidates never become delivered response,
  work or identity evidence. The phase-blind control needs its own complete
  rows and original-source heat bound: zero causal phase-pulse response
  does not erase raw background relaxation. Preserve both sine-law storage
  margins, signed jump work and all four scalar readout errors.
- **All-time and symmetry verdicts have different scope.** Fixed-budget
  consensus tests need an independent mixed-Lyapunov derivative from both
  full rows, zero-budget and failed-premise controls. Reflection excludes
  winding only at nonantipodal observations; it proves neither antipodal
  avoidance nor consensus. Equal-budget sources, tiny exact asymmetries and
  combined sign/reflection provide discriminating controls.
- **Classification is not a numerical case survey.** Exact factorized
  critical sets, full-support Hessian congruences and mode counts need
  independent witnesses and invalid-domain controls. Do not replace their
  completeness proofs by enumerating thousands of trajectories or transfer
  a positive-loss convergence result to zero loss or frozen capacities.

## Current checks and retained evidence

Distinguish three kinds of validation when a research owner changes:

- **Current-source contracts:** exercise the shared engine, SDK or CLI, including
  numerical admission, unsupported domains, read-only behavior and atomicity.
- **Mathematical controls:** check independent identities, counterexamples or
  exact coefficient probes under their stated hypotheses. Rational probes do
  not replace the actual irrational constants in an ideal theorem.
- **Retained evidence audits:** validate frozen inputs, source fingerprints,
  saved intervals/checkpoints and original verdicts without rerunning a producer.
  A current-source regression is separate from authenticating historical evidence.

For example, the [composition](theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md)
and [memory](theory/nodal/RELATIONAL_PATTERN_MEMORY.md) owners identify their
static controls and retained finite responses. The
[relational admission owner](theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md)
identifies recovery, formation and capture obligations. A successful current
snapshot or a passing subset must not rewrite a stopped or negative historical
verdict. Check the selected fixture before claiming that a run has no evolution.

The [benchmark guide](benchmarks/README.md#running-and-reporting) owns producer
invocation, freezing and artifact lifecycle. Reuse the appropriate saved evidence
or shared fixture; do not regenerate completed responses for unrelated changes.
A new reserved prediction needs its own declared inputs and prospective protocol.

The [formed C9 bundle audit](tests/physics/test_sine_formed_evidence.py) checks
the committed artifact hashes, archived source inventories, protocol associations
and the distinct reduced-model stopping rules without calling an assessor or producer:
the two-component error fraction uses its final outward receiver-gap lower
endpoint, while network composition uses an absolute whole-window full-state
allowance plus the exact degree/charge/storage control. Neither rule substitutes
for its saved actual-family identity and supplied-work obligations.
The all-time relaxation audit separately retains the phase and form resolution
decisions, including any qualified partial outcome; it must not reinterpret
a failed sufficient form margin as an observed tracking failure.
The separate ordered-heat form certificate preserves that earlier report body
as its baseline and applies the same primitive inputs and channel allowances.
Its audit checks both new strict margins without relabeling the original
`phase_only` result or its false joint stopping rule.
The two-port compatibility bundle has a different static criterion: primary
and matched-control equilibrium admission, full cycle/current consistency,
strict acute and Hessian bounds, and nonzero versus zero interface current.
It certifies neither a source trajectory nor capture after an attachment.
The separate two-port handoff obstruction is an exact analytic result with
current-source boundary-witness controls in
[the handoff suite](tests/physics/test_sine_two_port_handoff_obstruction.py),
not another frozen experiment. Select it together with
[the compatibility suite](tests/physics/test_sine_two_port_compatibility.py)
when changing their shared owner. Its failed scalar admission does not
revise the compatibility bundle or assert a failed trajectory.
The two-port capture bundle retains all 4,096 successful nominal-reference
steps separately from its analytic complete-state handoff. Its audit must
preserve the original thirty-six-coordinate preparation, clocks, root
prerequisites, strict tube margins, endpoint radius and final storage margin.
Read the saved first response and archived source; do not rerun its producer
to inspect the result or substitute the folded reference for the full family.
The supplied two-port probe has a separate conditional endpoint suite,
[response and maintenance controls](tests/physics/test_sine_two_port_probe.py).
Its source-evidence audit must match the original capture primitives and
rebuild every consumed handoff bound. Artifact hashes and rational chain
consistency associate the retained record with its declared source; they do
not independently revalidate omitted Taylor calculations or authenticate
execution. The probe protocol freezes its own assessment without replaying
the capture producer or altering its saved response. Its original attempt
failed during auxiliary control export; the retained response comes from
a separately archived deterministic export-recovery recomputation with
unchanged scientific inputs and runtime. Check the original archive, failure
record, recovery wrapper and explicit evaluation history separately. A
successful retained recovery must not overwrite the failure or be labeled
as a successful first attempt.
Select the [source-handoff suite](tests/research/test_sine_two_port_handoff.py)
with the conditional endpoint suite when changing this acquisition-to-probe
chain. Exercise altered primitive laws, lost metric correlations, malformed
or incomplete evidence and cached verdict changes independently; a cached
flag must neither grant nor remove an otherwise justified handoff. The
research audit's direct schema and explicit retained-execution premise are
separate from the generic SDK projection of the probe report.
The [interior-dipole suite](tests/physics/test_sine_two_port_dipole.py) adds
finite warmup, exact local response, full-flow remainders, phase-blind
background exclusion and separate work/identity admissions. Select it with
the [modified-energy kernel suite](tests/physics/test_sine_lyapunov.py)
and the source-handoff suite for changes to that continuation. The initial
capture evidence remains fixed; a later analytic warmup neither replays its
validated reference nor authenticates its retained execution. The saved
dipole response passes its thirteen conditions in the first frozen
assessment. Audit its original protocol/source association, source handoffs,
warmup, finite contrast, heat exclusion and both work/identity channels
separately from the earlier form-probe export recovery.

```sh
python -m pytest tests/physics/test_sine_formed_evidence.py -q
```

Missing committed artifacts fail this check. Historical source need not match
the current implementation; current-source regression remains separate. The
original pair/probe records still lack an evaluation-time source snapshot.
Content consistency neither authenticates chronology nor proves the mathematics.

Optional retained-record audits skip explicitly when local evidence is absent;
they must not recreate a producer or count missing evidence as a passed response.
Synthetic intervals, step records and stubbed producers test consumer logic,
not historical execution. Preserve original source archives, failures and
inconclusive verdicts; corrections require separately identified evidence.
Forecast readers recompute their consumed admission and Picard inclusion, but
do not replay the full Taylor proof or authenticate record production. Test
that boundary explicitly: partial coverage remains partial, and failure to
certify a requested duration or margin is not an exclusion of actual identity.
Retrospective channel integration from saved whole-time tubes must preserve
the original verdict and receive its own analysis provenance.

Local execution is serial by default. To use the main CI scheduling policy,
run `python -m pytest -n 2 --dist loadfile`; this keeps the same `not slow`
selection and assertions. Each file stays on one worker to reuse its module
fixtures. Workers isolate Python state and prepare session fixtures independently.
Pytest-benchmark does not collect timing measurements
under xdist; performance measurements need a separate serial run.

Choose an affected directory, module or test for bounded validation:

```sh
python -m pytest tests/core_physics -q
python -m pytest tests/operators/test_u3_hard_invariant.py -q
python -m pytest tests/sdk -q
python -m pytest tests/cli -q
python -m pytest tests/sdk --collect-only -q
```

To include all retained slow tests, use `python -m pytest tests -o addopts=""`.
To select only marked slow tests, use `python -m pytest -m slow` with those
paths. Inspect `python -m pytest --markers` and collection before treating
a marker-selected run as coverage: a registered marker can select no tests.

Standalone scripts do not inherit pytest's source-path configuration. Use the
editable installation above or explicitly set the shell's `PYTHONPATH` to
`src` so a previously installed release cannot replace the working source.

## Organization

[tests/](tests) contains top-level modules and subject directories. Search for
the affected API rather than maintaining another inventory of individual tests.
Core nodal behavior is in [core_physics/](tests/core_physics), operators in
[operators/](tests/operators), specialized certificates in
[physics/](tests/physics) and public network usage in [sdk/](tests/sdk).
[CLI integration](tests/cli) checks the same study recipe through Python and
the module entry point, including malformed input, diagnostic availability,
output replacement and logging isolation. Catalog checks do not execute a study.
[conftest.py](tests/conftest.py) and [utils.py](tests/utils.py) own shared helpers.
The [core scope map](tests/core_physics/README.md) identifies the engine tests
that replace retired self-contained illustrations.

Consolidation removes work, not just collected test identifiers. Test each
invalid boundary at its shared owner, then retain distinct consumer-wiring,
rollback and representation cases instead of repeating the entire Cartesian
product at every wrapper. Reuse an expensive producer only when its results
are read without mutation; keep independent numerical oracles and negative
controls. Do not turn removed parameter grids into hidden loops or replace
meaningful assertions with finiteness checks. Operator labels alone do not
justify energy-sign or conservation assertions with arbitrary tolerances.

The separately packaged arithmetic applications have their own test paths:

```sh
python -m pytest applications/factorization-lab/tests -q
python -m pytest applications/factorization-lab/benchmarks/test_benchmark_suite.py -q
python -m pytest applications/primality-test/tests -q
```

Run these as separate invocations: their local import roots differ from the core
suite. The root default selection does not include them. See the corresponding
[factorization guide](applications/factorization-lab/README.md) and
[primality guide](applications/primality-test/README.md) for application setup and scope.

Research producers under [benchmarks/](benchmarks/README.md) have declared entry
points and provenance requirements; a default pytest run does not implicitly
cover them. Do not regenerate retained evidence for an unrelated change.

For examples with `run_protocol` and `build_report`, reuse the real report through
a module fixture for numerical and scope checks. The shared
[example helper](tests/example_protocol_helpers.py) checks `main` separately with
a sentinel protocol and synthetic report, without rerunning a scientific producer.

Reusable independent oracles belong in test support modules rather than in
another collected test suite. The [pair oracle helper](tests/physics/_sine_pair_oracles.py)
shares fine support, exact jets and high-precision reductions without importing
production certificates or pytest fixtures. Each consuming suite retains its
own complete-law assumptions, preparation budgets and module-scoped reports.
When optimizing a shared certificate kernel, compare complete exact projections
on representative available and unavailable cases as well as these independent
controls; timing differences alone cannot establish unchanged evidence.

[Runtime facade tests](tests/physics/test_runtime_facade_imports.py) own the two
cold import orders for the P2/REMESH example families; those checks still use
fresh processes and resolve the actual public APIs.

Cold subprocesses that test this checkout explicitly request the
`source_tree_environment` fixture from [conftest.py](tests/conftest.py) and pass
it as `subprocess.run(..., env=source_tree_environment)`. It copies the process
environment and prepends pytest's configured source paths to `PYTHONPATH`,
without changing the parent's environment. Pytest's `pythonpath` setting only
updates its own interpreter; a fresh Python process can otherwise import a
different installed package. Keep the fixture opt-in: tests of installed-package
discovery or intentional import isolation retain their declared environments.
Working directories, timeouts and optimized-mode flags remain test-specific.

The [Makefile](Makefile) target `make test` uses the routine engine selection;
`make test-all` includes the retained research suites, still excluding `slow`.
`make dev-test` adds coverage over `src`, including compatibility shims. Research
producers use the individually documented entry points in the
[benchmark guide](benchmarks/README.md), not a directory-wide pytest campaign.
`make validate` checks importability, references, documentation integrity and
SDK tests.

## Structural regression evidence

Assert the contract of the changed path. Record representation, graph/support,
coefficients, primitive triad, input history, step size and operator sequence as
relevant. An operator jump and a continuous solver step need different expected
results. For a full grammar word, verify initiation, closure and contextual
admission; distinguish fragments explicitly.

Do not assume all operators increase coherence or that SHA freezes every later
EPI update. Its capacity attenuation, pure-EPI diffusion and a multichannel
trajectory have different scopes. Inspect the
[operator contracts](src/tnfr/operators/operator_contracts.py) and
[grammar scope](theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md) before asserting an invariant.
Initialization may set fixture state directly; test subsequent evolution through
the API under review.

When stochastic behavior changes, control the RNG and record its seed, initial
state and execution order. A seed alone does not fix timestamps, external data
or every backend. When relevant, compare available tetrad fields with estimator
provenance and unavailable-field reasons. Diagnostic scores are not acceptance
thresholds unless that specific policy is being tested.

## Backends and optional dependencies

[tests/conftest.py](tests/conftest.py) accepts `--math-backend` and
`TNFR_TEST_MATH_BACKEND`. The command-line option takes precedence; the
selected value sets `TNFR_MATH_BACKEND` and clears the backend cache before
collection.

```sh
python -m pytest tests/mathematics/test_backends.py --math-backend=numpy -q -rs
```

Install `compute-jax` or `compute-torch` before requesting those optional
adapters. Explicit per-test backend requests are not replaced by the session
setting. Review skip reasons with `-rs` and report adapters actually exercised.
NumPy is required by the shared conftest; a missing installation is not evidence
of successful NumPy-free execution.

## Code quality and documentation checks

Install tools needed for the check; options are configured in
[pyproject.toml](pyproject.toml):

```sh
python -m pip install -e ".[dev-minimal,test-quality,typecheck]"
python -m black --check src/tnfr
python -m flake8 src
python -m pydocstyle src/tnfr
python -m mypy src/tnfr
python -m pyright src/tnfr
```

Pass affected files for a focused edit where supported. Black and isort use
88 columns; pydocstyle uses NumPy conventions. CI's advisory checks are listed
in [the workflow guide](.github/WORKFLOWS.md). `make format` modifies files;
it is not a read-only validation step.

The lazy `physics` namespace has one maintained export map and a generated
typing facade. After changing that map, run
`python scripts/generate_physics_stub.py --write`; validate it with
`python scripts/generate_physics_stub.py --check`. The generator reads the
source AST without importing TNFR, and the
[facade tests](tests/physics/test_physics_facade_imports.py) check typed exports,
runtime object identity and cold import boundaries.

For shared relational report export, select
`tests/sdk/test_relational_reports.py`. Preserve every registered owner-managed
type, actual-class and subclass admission, native label validation, exact
fractions and unavailable fields. Cold-process controls must show that export
does not import unrelated research owners. Use guarded zero-horizon reports
or synthetic wiring for producer-backed types; an SDK export test must not
regenerate a selected scientific response.

Optional pre-commit setup requires installing `pre-commit` separately before
`pre-commit install`. [.pre-commit-config.yaml](.pre-commit-config.yaml)
configures Black, isort, pydocstyle and a local Bandit-command guard. That guard
uses Bash, which must be available on Windows. The hooks do not perform a code
review; CI's formatting gate checks all tracked files selected by those hooks.

For Markdown-only changes, use the relevant reference check. For site or
documentation-infrastructure changes, also run integrity and build checks:

```sh
python scripts/verify_internal_references.py --ci
python scripts/check_documentation.py
python -m pip install -e ".[docs]"
python scripts/prepare_docs.py
python -m mkdocs build --strict
```

The reference checker accepts `--dirs` followed by files or directories to
bound the check. The staging script copies repository owners into
`build/docs-source/`; MkDocs renders that generated source into `site/`.
Edit the repository owners, never either generated tree. The documentation
integrity check also validates the theory catalog, its generated navigation,
operator contracts, grammar roles and glossary cards/index. Regenerate views with
`python scripts/check_documentation.py --write-generated` after changing their
source declarations, then rerun the read-only check.
It executes the Python block in the README's Quick start section and compares
its captured output with that section's declared output. Publication metadata
uses the same strict JSON reader as the engine, including duplicate-key rejection.

## Security checks

Follow [SECURITY.md](SECURITY.md) for reporting and trust boundaries:

```sh
python -m pip install -e ".[security]"
python -m pip_audit
python -m bandit -r src -c bandit.yaml
```

The dependency audit covers installed packages, including extras actually
installed; it does not automatically cover every optional dependency. Bandit's
configured exception is recorded in [bandit.yaml](bandit.yaml). A clean report
does not guarantee that dependencies or code contain no vulnerabilities.

## Validation workflow and reporting

1. Identify the affected contract and existing owners. Reproduce a suspected bug
   before the fix when feasible; distinguish a new counterexample from a measured
   baseline.
2. Add meaningful regression coverage when behavior or an important boundary
   changes. A small documentation correction need not add numerical tests.
3. Run affected checks. Broaden to dependents, optional backends or the default
   suite when changed scope or a failure justifies it. Do not rerun unrelated
   expensive research producers as a routine checklist item.
4. Report actual commands, interpreter and relevant dependency versions,
   pass/fail/skip counts, warnings and untested scope. Do not label a failure
   pre-existing without baseline evidence.

The `structural_rng` fixture supplies NumPy's generator with seed zero.
`structural_tolerances` supplies atol=1e-12 and rtol=1e-10; these are not
exact-theorem eligibility gates. Choose scale-appropriate numerical tolerances
and keep them distinct from represented exact equality. The autouse cleanup
resets selected global state; restore additional state your test changes.

Coverage is feedback, not proof of a physical invariant. With test dependencies
installed, a selected run can generate a report:

```sh
python -m pytest tests/sdk --cov=tnfr --cov-report=term-missing
```

The current project configuration does not enforce a coverage percentage.
The Python 3.11 CI job instead uses `--cov=src` to retain the whole source-directory
scope, with `-n 2 --dist loadfile`. Pytest-cov combines the workers' data and
produces terminal and XML reports. Do not wrap distributed pytest with
`coverage run`: that alone measures the controller instead of combining worker
execution. Coverage of Python subprocesses launched inside tests is a separate
configuration and is not enabled here.

## Retained C6 campaign reproduction

The union/history report tests that reconstruct the retained B54/B55 lineage
are marked `slow`, including tests whose shared fixtures perform that work.
During the 2026-09-19 release check, one union-report fixture consumed more
than 20 minutes of CPU time. This observed cost is not a runtime ceiling.
Lightweight lineage and output-overwrite rejection tests remain available when
their research paths are selected; the finite-state union oracles run
independently of the retained campaign. C6 is outside the routine engine gate.

Select the expensive report checks explicitly:

```sh
python -m pytest -m slow tests/physics/test_c6_winding_union_report.py tests/physics/test_c6_winding_history_report.py -q
```

For the independent small-state controls:

```sh
python -m pytest tests/physics/test_c6_carried_return_unions.py -q
```

This classification changes selection by cost, not assertions or scientific
scope. Report the expensive campaign as unrun when omitted, and retain failures
from earlier attempts. A passing reduced suite does not certify the historical
report or close C6 global stability.
