# Cycle winding under Coupling: scope and retained evidence

Conditional cycle transport and winding protection remain reusable. The C6 global-stability campaign is parked: 41 of 56 exit labels are excluded and 15 remain open. Its numerical studies are archived evidence, not the current research queue.

## Chapters and scope

| Chapter | Responsibility |
| --- | --- |
| [C6 response, contraction and represented rounding](research/archive/c6/RESPONSE_AND_ROUNDING.md) | Parked C6 response domains, numerical reserves and rounding-cell restrictions. |
| [C6 carried flow and pressure cells](research/archive/c6/CARRIED_FLOW.md) | Exact carry, live-pressure and finite itinerary results under their numerical model. |
| [C6 pressure excursions and compensation](research/archive/c6/PRESSURE_EXCURSIONS.md) | Finite pressure, mean-budget and compensation obstructions; no global closure. |
| [C6 relational regions and exit exclusions](research/archive/c6/RELATIONAL_REGIONS.md) | Correlated regions, histories and regional exclusion proofs. |
| [C6 history refinement and retained exclusions](research/archive/c6/HISTORY_EXCLUSIONS.md) | Exact union, safe-memory and refinement evidence with its source-bound budgets. |
| [C6 remaining verification obligations](research/archive/c6/REMAINING_OBLIGATIONS.md) | Retained bounded-search failures and unresolved certification conditions. |

Section numbers remain stable across this collection. The [execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) alone assigns research work.

## 1. The structural state and the actual operator realization

Let `phi_i` be the phase at node `i` of an oriented cycle `C_n`, `n>=3`, and
let its oriented shortest-arc gaps be

$$
d_i=\operatorname{wrap}(\phi_{i+1}-\phi_i),\qquad
-\pi<d_i<\pi,\qquad
\sum_i d_i=2\pi W.
$$

Indices are cyclic and `W` is the integer winding of this declared cycle.
The final identity follows from a consistent closed phase loop, not from
arbitrarily supplied real gap coordinates. Reversing orientation changes its
sign. Retaining a positive wrap-branch margin makes the observation defined;
it does not by itself prove that a future operation preserves that margin.

The following theorem uses the existing canonical Coupling implementation
in [_coupling_stage_kernel.py](../src/tnfr/operators/_coupling_stage_kernel.py),
with `UM_BIDIRECTIONAL=False` and `UM_FUNCTIONAL_LINKS=False`. The first
configuration makes each target average only its neighbors and write only
its own phase. The second keeps the simple cycle support fixed. Other
preconditions and grammar admission remain necessary at execution time.

Let `g` be the effective UM gate, including any tightening of the hard U3
limit, and suppose every gap lies in one interval

$$
-g<m\leq d_i\leq M<g,\qquad 0<g\leq\pi/2.
$$

The interval may be positive, negative or contain zero. Both cycle neighbors
are therefore compatible with every target. Their separation in the local
lift is `d_(i-1)+d_i`, whose absolute value is strictly less than `pi`.
The neighbor phasor sum is nonzero, so the circular midpoint is unambiguous.
The UM phase factor `eta=UM_theta_push` is the already configured operator
coefficient, with `0<eta<=1`. Its repository default is `1/(pi+1)` as
materialized by `UM_THETA_PUSH`. The theorem covers the declared factor
interval; it does not derive that particular default uniquely from the nodal
equation or add an independent phase-evolution law.

The default bidirectional UM configuration has a different effect: it includes
the target in the phasor mean and writes neighboring phases too. It is outside
this theorem. Even on a uniform twisted cycle, one bidirectional target shrinks
the two inner gaps by `1-eta` and expands the adjacent outer gaps by `1+eta`.
The interval invariance proved below cannot simply be transferred to it.

## 2. The midpoint identity and its canonical pressure connection

In the target's local lift, its neighbor phases are
`phi_i-d_(i-1)` and `phi_i+d_i`. Their phasor sum is

$$
2\cos\frac{d_{i-1}+d_i}{2}\;
\exp\left(i\left[\phi_i+\frac{d_i-d_{i-1}}2\right]\right).
$$

The cosine is positive under the stated strict gate. Consequently the actual
target-only UM shortest-arc update has displacement

$$
u_i=\frac\eta2(d_i-d_{i-1}).
$$

This identity connects directly to the existing canonical phase-pressure
channel in [dnfr.py](../src/tnfr/dynamics/dnfr.py). On this same simple cycle,
its unweighted neighbor-phasor mean is the midpoint above, hence

$$
\partial\phi_i
=-\frac1\pi\operatorname{wrap}(\phi_i-\bar\phi_i)
=\frac{d_i-d_{i-1}}{2\pi},\qquad
u_i=\eta\pi\,\partial\phi_i.
$$

The configured phase-channel weight belongs to the aggregation of
`DeltaNFR`; it is separate from the unweighted channel identity here. The
nodal integrator subsequently uses `nu_f*DeltaNFR` to evolve EPI. Reading the
same structural gradient in UM does not identify a phase jump with that EPI
evolution or introduce a second continuous-time law. The identity excludes
vanishing resultants, arbitrary neighborhood profiles and wrap-crossing
configurations. Binary64 phasor evaluation retains its separate residual.

## 3. One target: conserved circulation and interval protection

Updating target `i` changes just its incident gaps. Put
`a=d_(i-1)`, `b=d_i`. Then

$$
\begin{pmatrix}a'\\b'\end{pmatrix}
=\begin{pmatrix}1-\eta/2&\eta/2\\\eta/2&1-\eta/2\end{pmatrix}
\begin{pmatrix}a\\b\end{pmatrix}.
$$

Each new gap is a convex combination of the old pair, and their sum is
unchanged. Every gap remains in `[m,M]`, strictly inside U3 and away from the
wrap branch. The same is true along the affine shortest-arc completion of
the declared phase jump. Thus the closed cycle retains `W`. This completion
is a mathematical path witness; the engine event itself is a discrete update.

For the gap mean `d_bar=sum(d)/n`, define the diagnostic squared spread
`V(d)=sum((d_i-d_bar)^2)/2`. Direct expansion gives

$$
V(d)-V(d')=\frac{\eta(2-\eta)}4(a-b)^2\geq0.
$$

Every finite sequence of these admitted target-only updates preserves the
same interval and winding, even if its target order varies. Arbitrary target
selection does not imply convergence to uniform gaps: a sequence can keep
updating an already equal pair while leaving other differences untouched.

## 4. A simultaneous stage is the canonical cycle diffusion map

An immutable all-target UM stage proposes the displacement in section 2 for
every node. Since target-only UM has no overlapping neighbor writes, the
new edge gaps are

$$
d_i'=d_i+u_{i+1}-u_i
=(1-\eta)d_i+\frac\eta2(d_{i-1}+d_{i+1}).
$$

Equivalently,

$$
d'=(I-\eta L_{\rm rw,C_n})d.
$$

This is precisely the normalized cycle Laplacian already used by the canonical
pure-EPI diffusion channel. Here it acts on an oriented gap observation under
a UM stage; the index counts operator stages, not elapsed physical time.
Its nonnegative, doubly stochastic matrix preserves the gap sum, the initial
gap interval, U3 compatibility and winding in the same exact model.

The exact nonnegative spread drop is

$$
V(d)-V(d')=
\frac{\eta(1-\eta)}2\sum_i(d_{i+1}-d_i)^2
+\frac{\eta^2}{8}\sum_i(d_{i+1}-d_{i-1})^2.
$$

The Fourier eigenvalues are
`1-eta+eta*cos(2*pi*k/n)`. For fixed `0<eta<1`, their absolute values are
strictly below one for all `k!=0`, so repeated exact all-target stages converge
to `d_bar=2*pi*W/n`. At `eta=1`, an even cycle has an alternating gap mode with
eigenvalue `-1`; its spread is preserved, although winding and interval
protection still hold. An odd cycle has no such nonconstant mode. These are
exact-model statements, not asymptotic binary64 execution certificates.

For same-sign gaps and nonzero total `S=sum(d)=2*pi*W`, circulation concentration
can be read without another dynamical parameter:

$$
I(d)=\sum_i\left(\frac{d_i}{S}\right)^2
=\frac1n+\frac{2V(d)}{S^2}.
$$

It cannot increase under the protected maps. Under the strictly contracting
all-target regime it tends to `1/n`. The preserved winding therefore becomes
an evenly distributed twist; this class supplies no mechanism for sustained
localization of circulation. It neither constructs a localized physical
entity nor supplies a restoring law for localized EPI or capacity support.

## 5. Nonzero winding can coexist with zero canonical pressure

A uniform twist `phi_i=2*pi*W*i/n` with `|2*pi*W/n|<g` is fixed by target-only
UM. Its phase-pressure channel vanishes because its two neighbor phasors
have their mean at the target phase. With uniform EPI, uniform positive
capacity and the uniform degree of the simple cycle, the other three
canonical gradient channels also vanish. Therefore the complete canonical
`DeltaNFR` is zero in this exact restricted state, despite nonzero `W`.

Pressure equilibrium thus need not mean phase consensus. A positive-winding
example with the strict canonical gate requires `n>4*W`. The spectral
convergence above concerns uniformity of gaps, not uniformity of phases.
It supplies neither attraction for arbitrary multichannel trajectories nor
a dynamic preparation of nonzero winding from a zero-winding state. The
protected UM class cannot perform that preparation because it conserves `W`.

## 6. Canonical loss, branch crossings and observation limits

Outside the protected class, canonical events can change winding. For the
regular unit-winding eight-cycle, change only the phase at node zero from its
initial value zero to `theta`. The edge from node seven to node zero reaches
the wrap branch when `theta=3*pi/4`. Immediately below this value the winding
is one; immediately above it, before the next incident branch crossing, it
is zero. The cycle support can remain unchanged throughout.

The production Transition (`NAV`) word supplies a finite instance. On the
prepared regular eight-cycle with repository default regime policy and seed
17, repeated admitted single-Transition words give `theta_0=2.35` after 13
steps and `theta_0=2.5500000000000003` after 14. Their winding observations are
respectively one and zero. The reported minimum branch margins are
approximately `0.006194490192344748` and `0.19380550980765499`. The actual
EPI, capacity and pressure effects of Transition remain part of those events;
the example does not impose an auxiliary phase trajectory. Its endpoint
change requires a branch crossing in any continuous fixed-cycle completion,
but does not certify an unobserved continuous path inside a discrete event.

U3 admission applies to the operator's required compatible relations; it is
not a universal promise that every cycle edge stays admissible under every
canonical operator. Equal endpoint winding likewise cannot establish a
protected intervening path: opposite edge slips can cancel. Neither a
phase-independent EPI operation nor a static snapshot is a proof about all
future phase writes. Cycle deletion, changed support, absent phase values,
branch hits and nonzero signed branch slips must remain distinguishable.

## 7. Exact companion, runtime checks and remaining work

[`coupling_winding.py`](../src/tnfr/physics/coupling_winding.py) implements the
single-target and all-target exact gap maps with conserved sum, interval and
spread identities. It reuses the shared exact-or-represented real reader:
integer and rational inputs remain exact, including NumPy integers promoted
to Python integers; other real values retain their binary64 rational value.
The observer accepts signed gap coordinates and does not round their sum
into a purported winding integer. A separate declared-cycle phase observation
must establish closure and winding. Its factor, state and gate hypotheses
also do not admit a live operator word by themselves.

[Exact tests](../tests/physics/test_coupling_winding.py) cover convex transport,
Jensen dissipation, finite compositions, the even-cycle boundary and the
shared pure-EPI pressure map.
[Pressure tests](../tests/physics/test_coupling_pressure_bridge.py) compare the
actual scalar and vectorized canonical phase channel with the midpoint
identity, then use the shared nodal integrator for EPI evolution. Their
regular twisted controls bound observed binary64 pressure residuals; a
selected finite tolerance is not a rigorous bound on every libm evaluation.

[Runtime tests](../tests/physics/test_coupling_winding_runtime.py) and the
[benchmark](../benchmarks/canonical_winding_persistence.py) compare actual
canonical UM histories with the exact gap companion and record endpoint
residuals, branch/U3 margins, winding and actual operator histories. They
use eight UM/SHA pairs on perturbed C8 and C16 with W=0 and W=1. SHA leaves
phase unchanged and attenuates capacity by its existing default factor;
the word uses no phase-factor override. Both sequence validators and live
operator requirements remain active. Separate DERIVED and MEASURED manifests
record the working-source digest and the corresponding evidence boundary.
The word observer now uses live primary-EPI admission, shared adjacency
validation, runtime U3 limits and node/edge identity comparisons; its
[integrity tests](../tests/physics/test_winding_word_integrity.py) cover these
boundaries without changing committed-prefix behavior after later failure.
The production controls
also exercise canonical winding loss through Transition. These finite
checks are separate from the exact repeated-map theorem and do not prove
future binary64 protection, arbitrary grammar stability or a physical
memory experiment.

The [capacity-localization study](CAPACITY_LOCALIZATION_BALANCE.md) now tests
physical nodal flow in addition to the discrete phase words above. It finds
EPI diffusion under uniform capacity and a conditional nonuniform equilibrium
when a retained capacity profile balances that diffusion. The UM/SHA word
above has no physical EPI-flow interval, so retention of an EPI shape under
that word alone could not establish resistance to diffusion. The remaining
question is joint maintenance when canonical operators evolve the supporting
phase and capacity fields. That requires their complete effects and cannot
be inferred from integer winding alone.

## Section link directory

These aliases route existing citations to their substantive owner.

- <a id="canonical-coupling-preserves-winding-in-a-restricted-cycle-regime"></a>[Canonical Coupling preserves winding in a restricted cycle regime](#canonical-coupling-preserves-winding-in-a-restricted-cycle-regime)

- <a id="8-one-shared-local-response-for-circular-means"></a>[8. One shared local response for circular means](research/archive/c6/RESPONSE_AND_ROUNDING.md#8-one-shared-local-response-for-circular-means)

- <a id="9-default-all-target-c6-response-and-a-phase-neighborhood"></a>[9. Default all-target C6 response and a phase neighborhood](research/archive/c6/RESPONSE_AND_ROUNDING.md#9-default-all-target-c6-response-and-a-phase-neighborhood)

- <a id="a-sufficient-nonlinear-phase-map-invariant-box"></a>[A sufficient nonlinear phase-map invariant box](research/archive/c6/RESPONSE_AND_ROUNDING.md#a-sufficient-nonlinear-phase-map-invariant-box)

- <a id="connection-to-nodal-pressure"></a>[Connection to nodal pressure](research/archive/c6/RESPONSE_AND_ROUNDING.md#connection-to-nodal-pressure)

- <a id="10-finite-default-c6-controls-and-remaining-scope"></a>[10. Finite default C6 controls and remaining scope](research/archive/c6/RESPONSE_AND_ROUNDING.md#10-finite-default-c6-controls-and-remaining-scope)

- <a id="11-nonlinear-contraction-and-a-joint-phasecapacityepi-domain"></a>[11. Nonlinear contraction and a joint phase/capacity/EPI domain](research/archive/c6/RESPONSE_AND_ROUNDING.md#11-nonlinear-contraction-and-a-joint-phasecapacityepi-domain)

- <a id="a-uniform-nonlinear-oscillation-bound"></a>[A uniform nonlinear oscillation bound](research/archive/c6/RESPONSE_AND_ROUNDING.md#a-uniform-nonlinear-oscillation-bound)

- <a id="from-phase-pressure-to-a-preserved-epi-reserve"></a>[From phase pressure to a preserved EPI reserve](research/archive/c6/RESPONSE_AND_ROUNDING.md#from-phase-pressure-to-a-preserved-epi-reserve)

- <a id="what-tends-to-persist-in-this-restricted-model"></a>[What tends to persist in this restricted model](research/archive/c6/RESPONSE_AND_ROUNDING.md#what-tends-to-persist-in-this-restricted-model)

- <a id="executable-algebra-and-its-boundary"></a>[Executable algebra and its boundary](research/archive/c6/RESPONSE_AND_ROUNDING.md#executable-algebra-and-its-boundary)

- <a id="12-two-cycle-admission-and-the-represented-null-boundary"></a>[12. Two-cycle admission and the represented null boundary](research/archive/c6/RESPONSE_AND_ROUNDING.md#12-two-cycle-admission-and-the-represented-null-boundary)

- <a id="13-additive-defects-finite-reserves-and-the-neutral-epi-mean"></a>[13. Additive defects, finite reserves and the neutral EPI mean](research/archive/c6/RESPONSE_AND_ROUNDING.md#13-additive-defects-finite-reserves-and-the-neutral-epi-mean)

- <a id="14-a-conditional-invariant-bound-with-persistent-numerical-defects"></a>[14. A conditional invariant bound with persistent numerical defects](research/archive/c6/RESPONSE_AND_ROUNDING.md#14-a-conditional-invariant-bound-with-persistent-numerical-defects)

- <a id="15-offline-audit-of-the-retained-numerical-boundary"></a>[15. Offline audit of the retained numerical boundary](research/archive/c6/RESPONSE_AND_ROUNDING.md#15-offline-audit-of-the-retained-numerical-boundary)

- <a id="16-held-nodal-rounding-cells-and-the-remaining-phasemean-boundary"></a>[16. Held nodal rounding cells and the remaining phase/mean boundary](research/archive/c6/RESPONSE_AND_ROUNDING.md#16-held-nodal-rounding-cells-and-the-remaining-phasemean-boundary)

- <a id="the-actual-quarter-flow-arithmetic"></a>[The actual quarter-flow arithmetic](research/archive/c6/RESPONSE_AND_ROUNDING.md#the-actual-quarter-flow-arithmetic)

- <a id="a-local-error-bound-and-a-zero-sum-counterexample"></a>[A local error bound and a zero-sum counterexample](research/archive/c6/RESPONSE_AND_ROUNDING.md#a-local-error-bound-and-a-zero-sum-counterexample)

- <a id="retained-observations-and-their-provenance"></a>[Retained observations and their provenance](research/archive/c6/RESPONSE_AND_ROUNDING.md#retained-observations-and-their-provenance)

- <a id="phase-evaluation-has-additional-distinct-numerical-contracts"></a>[Phase evaluation has additional, distinct numerical contracts](research/archive/c6/RESPONSE_AND_ROUNDING.md#phase-evaluation-has-additional-distinct-numerical-contracts)

- <a id="17-opposite-pair-closure-and-a-restricted-mean-preserving-arithmetic-class"></a>[17. Opposite-pair closure and a restricted mean-preserving arithmetic class](research/archive/c6/RESPONSE_AND_ROUNDING.md#17-opposite-pair-closure-and-a-restricted-mean-preserving-arithmetic-class)

- <a id="an-exact-partial-observable-of-the-nonlinear-phase-map"></a>[An exact partial observable of the nonlinear phase map](research/archive/c6/RESPONSE_AND_ROUNDING.md#an-exact-partial-observable-of-the-nonlinear-phase-map)

- <a id="the-nodal-equation-closes-the-corresponding-epi-observable"></a>[The nodal equation closes the corresponding EPI observable](research/archive/c6/RESPONSE_AND_ROUNDING.md#the-nodal-equation-closes-the-corresponding-epi-observable)

- <a id="a-closed-binary64-class-for-the-isolated-epi-channel"></a>[A closed binary64 class for the isolated EPI channel](research/archive/c6/RESPONSE_AND_ROUNDING.md#a-closed-binary64-class-for-the-isolated-epi-channel)

- <a id="what-the-retained-default-execution-actually-preserves"></a>[What the retained default execution actually preserves](research/archive/c6/RESPONSE_AND_ROUNDING.md#what-the-retained-default-execution-actually-preserves)

- <a id="18-certified-two-neighbor-phase-realization"></a>[18. Certified two-neighbor phase realization](research/archive/c6/CARRIED_FLOW.md#18-certified-two-neighbor-phase-realization)

- <a id="one-geometric-kernel-for-il-and-phase-pressure"></a>[One geometric kernel for IL and phase pressure](research/archive/c6/CARRIED_FLOW.md#one-geometric-kernel-for-il-and-phase-pressure)

- <a id="a-local-cancellation-bound-with-explicit-remaining-errors"></a>[A local cancellation bound, with explicit remaining errors](research/archive/c6/CARRIED_FLOW.md#a-local-cancellation-bound-with-explicit-remaining-errors)

- <a id="fixed-finite-comparison"></a>[Fixed finite comparison](research/archive/c6/CARRIED_FLOW.md#fixed-finite-comparison)

- <a id="19-pressure-boxes-and-the-two-binade-mean-bias-law"></a>[19. Pressure boxes and the two-binade mean-bias law](research/archive/c6/CARRIED_FLOW.md#19-pressure-boxes-and-the-two-binade-mean-bias-law)

- <a id="inverting-the-shared-nodal-arithmetic"></a>[Inverting the shared nodal arithmetic](research/archive/c6/CARRIED_FLOW.md#inverting-the-shared-nodal-arithmetic)

- <a id="why-exactly-opposite-pressures-can-still-bias-epi"></a>[Why exactly opposite pressures can still bias EPI](research/archive/c6/CARRIED_FLOW.md#why-exactly-opposite-pressures-can-still-bias-epi)

- <a id="binding-the-obstruction-to-the-corrected-c6-records"></a>[Binding the obstruction to the corrected C6 records](research/archive/c6/CARRIED_FLOW.md#binding-the-obstruction-to-the-corrected-c6-records)

- <a id="20-exact-nodal-area-with-a-carried-numerical-remainder"></a>[20. Exact nodal area with a carried numerical remainder](research/archive/c6/CARRIED_FLOW.md#20-exact-nodal-area-with-a-carried-numerical-remainder)

- <a id="state-and-exact-nodal-balance"></a>[State and exact nodal balance](research/archive/c6/CARRIED_FLOW.md#state-and-exact-nodal-balance)

- <a id="the-signed-mean-budget-and-its-remaining-source"></a>[The signed mean budget and its remaining source](research/archive/c6/CARRIED_FLOW.md#the-signed-mean-budget-and-its-remaining-source)

- <a id="retained-pressure-comparison"></a>[Retained-pressure comparison](research/archive/c6/CARRIED_FLOW.md#retained-pressure-comparison)

- <a id="gate-before-graph-integration"></a>[Gate before graph integration](research/archive/c6/CARRIED_FLOW.md#gate-before-graph-integration)

- <a id="21-live-pressure-and-an-explicit-carried-flowevent-contract"></a>[21. Live pressure and an explicit carried flow/event contract](research/archive/c6/CARRIED_FLOW.md#21-live-pressure-and-an-explicit-carried-flowevent-contract)

- <a id="numerical-state-pressure-and-event-ownership"></a>[Numerical state, pressure and event ownership](research/archive/c6/CARRIED_FLOW.md#numerical-state-pressure-and-event-ownership)

- <a id="exact-readout-difference-in-the-epi-pressure-channel"></a>[Exact readout difference in the EPI pressure channel](research/archive/c6/CARRIED_FLOW.md#exact-readout-difference-in-the-epi-pressure-channel)

- <a id="matched-finite-runtime-comparison"></a>[Matched finite runtime comparison](research/archive/c6/CARRIED_FLOW.md#matched-finite-runtime-comparison)

- <a id="22-generated-pressure-can-exclude-a-fixed-phase-numerical-equilibrium"></a>[22. Generated pressure can exclude a fixed-phase numerical equilibrium](research/archive/c6/CARRIED_FLOW.md#22-generated-pressure-can-exclude-a-fixed-phase-numerical-equilibrium)

- <a id="a-necessary-cancellation-condition-over-the-complete-epi-band"></a>[A necessary cancellation condition over the complete EPI band](research/archive/c6/CARRIED_FLOW.md#a-necessary-cancellation-condition-over-the-complete-epi-band)

- <a id="exact-duration-of-an-unchanged-visible-state"></a>[Exact duration of an unchanged visible state](research/archive/c6/CARRIED_FLOW.md#exact-duration-of-an-unchanged-visible-state)

- <a id="evidence-and-the-next-gate"></a>[evidence-and-the-next-gate](research/archive/c6/CARRIED_FLOW.md#evidence-and-the-next-gate)

- <a id="evidence-and-extension-conditions"></a>[Evidence and extension conditions](research/archive/c6/CARRIED_FLOW.md#evidence-and-extension-conditions)

- <a id="23-exact-closure-of-the-generated-umil-phase-component"></a>[23. Exact closure of the generated UM/IL phase component](research/archive/c6/CARRIED_FLOW.md#23-exact-closure-of-the-generated-umil-phase-component)

- <a id="projection-support-and-exact-state-identity"></a>[Projection, support and exact state identity](research/archive/c6/CARRIED_FLOW.md#projection-support-and-exact-state-identity)

- <a id="the-inherited-null-reaches-a-phase-fixed-point"></a>[The inherited null reaches a phase fixed point](research/archive/c6/CARRIED_FLOW.md#the-inherited-null-reaches-a-phase-fixed-point)

- <a id="a-periodic-source-is-not-automatically-a-bounded-source"></a>[A periodic source is not automatically a bounded source](research/archive/c6/CARRIED_FLOW.md#a-periodic-source-is-not-automatically-a-bounded-source)

- <a id="evidence-and-next-dependency"></a>[Evidence and next dependency](research/archive/c6/CARRIED_FLOW.md#evidence-and-next-dependency)

- <a id="24-local-pressure-lattice-product-trap-obstruction-and-finite-compensation"></a>[24. Local pressure lattice, product-trap obstruction and finite compensation](research/archive/c6/CARRIED_FLOW.md#24-local-pressure-lattice-product-trap-obstruction-and-finite-compensation)

- <a id="the-local-discrete-laplacian-is-exact"></a>[The local discrete Laplacian is exact](research/archive/c6/CARRIED_FLOW.md#the-local-discrete-laplacian-is-exact)

- <a id="a-cartesian-product-cannot-supply-the-trapping-certificate-here"></a>[A Cartesian product cannot supply the trapping certificate here](research/archive/c6/CARRIED_FLOW.md#a-cartesian-product-cannot-supply-the-trapping-certificate-here)

- <a id="actual-compensation-occurs-then-fails-at-the-first-cell-boundary"></a>[Actual compensation occurs, then fails at the first cell boundary](research/archive/c6/CARRIED_FLOW.md#actual-compensation-occurs-then-fails-at-the-first-cell-boundary)

- <a id="the-complete-13-state-class-cannot-retain-a-bounded-carried-orbit"></a>[The complete 13-state class cannot retain a bounded carried orbit](research/archive/c6/CARRIED_FLOW.md#the-complete-13-state-class-cannot-retain-a-bounded-carried-orbit)

- <a id="ownership-and-next-gate"></a>[ownership-and-next-gate](research/archive/c6/CARRIED_FLOW.md#ownership-and-next-gate)

- <a id="ownership-and-scope"></a>[Ownership and scope](research/archive/c6/CARRIED_FLOW.md#ownership-and-scope)

- <a id="25-exact-carry-compatible-itineraries-across-pressure-cell-boundaries"></a>[25. Exact carry-compatible itineraries across pressure-cell boundaries](research/archive/c6/CARRIED_FLOW.md#25-exact-carry-compatible-itineraries-across-pressure-cell-boundaries)

- <a id="translated-cells-retain-the-complete-accumulated-nodal-area"></a>[Translated cells retain the complete accumulated nodal area](research/archive/c6/CARRIED_FLOW.md#translated-cells-retain-the-complete-accumulated-nodal-area)

- <a id="visible-recurrence-is-weaker-than-recurrence-of-the-carried-state"></a>[Visible recurrence is weaker than recurrence of the carried state](research/archive/c6/CARRIED_FLOW.md#visible-recurrence-is-weaker-than-recurrence-of-the-carried-state)

- <a id="two-derived-boundaries-reveal-a-source-sign-reversal"></a>[Two derived boundaries reveal a source-sign reversal](research/archive/c6/CARRIED_FLOW.md#two-derived-boundaries-reveal-a-source-sign-reversal)

- <a id="one-nodal-direction-still-excludes-the-four-observed-states-as-a-cycle"></a>[One nodal direction still excludes the four observed states as a cycle](research/archive/c6/CARRIED_FLOW.md#one-nodal-direction-still-excludes-the-four-observed-states-as-a-cycle)

- <a id="next-gate-close-a-relational-class-or-exclude-it-structurally"></a>[next-gate-close-a-relational-class-or-exclude-it-structurally](research/archive/c6/CARRIED_FLOW.md#next-gate-close-a-relational-class-or-exclude-it-structurally)

- <a id="requirements-for-closure-or-structural-exclusion-of-a-relational-class"></a>[Requirements for closure or structural exclusion of a relational class](research/archive/c6/CARRIED_FLOW.md#requirements-for-closure-or-structural-exclusion-of-a-relational-class)

- <a id="26-pressure-sign-sectors-and-the-carried-curvature-budget"></a>[26. Pressure-sign sectors and the carried curvature budget](research/archive/c6/PRESSURE_EXCURSIONS.md#26-pressure-sign-sectors-and-the-carried-curvature-budget)

- <a id="a-sign-sector-has-a-uniform-nodal-drift-bound"></a>[A sign sector has a uniform nodal drift bound](research/archive/c6/PRESSURE_EXCURSIONS.md#a-sign-sector-has-a-uniform-nodal-drift-bound)

- <a id="the-visible-sign-index-combines-exact-evolution-and-carry-transfer"></a>[The visible sign index combines exact evolution and carry transfer](research/archive/c6/PRESSURE_EXCURSIONS.md#the-visible-sign-index-combines-exact-evolution-and-carry-transfer)

- <a id="one-shared-bounded-cell-exit-replay"></a>[One shared bounded cell-exit replay](research/archive/c6/PRESSURE_EXCURSIONS.md#one-shared-bounded-cell-exit-replay)

- <a id="the-inherited-continuation-reaches-positive-node-1-pressure"></a>[The inherited continuation reaches positive node-1 pressure](research/archive/c6/PRESSURE_EXCURSIONS.md#the-inherited-continuation-reaches-positive-node-1-pressure)

- <a id="sign-recovery-has-not-yet-repaid-accumulated-evolution"></a>[Sign recovery has not yet repaid accumulated evolution](research/archive/c6/PRESSURE_EXCURSIONS.md#sign-recovery-has-not-yet-repaid-accumulated-evolution)

- <a id="27-finite-positive-pressure-repayment-and-exact-return-obstructions"></a>[27. Finite positive-pressure repayment and exact return obstructions](research/archive/c6/PRESSURE_EXCURSIONS.md#27-finite-positive-pressure-repayment-and-exact-return-obstructions)

- <a id="an-exact-affine-budget-separates-repayment-crossing-and-overshoot"></a>[An exact affine budget separates repayment, crossing and overshoot](research/archive/c6/PRESSURE_EXCURSIONS.md#an-exact-affine-budget-separates-repayment-crossing-and-overshoot)

- <a id="the-positive-episode-ends-before-the-proposed-repayment-prefix"></a>[The positive episode ends before the proposed repayment prefix](research/archive/c6/PRESSURE_EXCURSIONS.md#the-positive-episode-ends-before-the-proposed-repayment-prefix)

- <a id="two-pressure-levels-impose-a-separate-integer-period-constraint"></a>[Two pressure levels impose a separate integer period constraint](research/archive/c6/PRESSURE_EXCURSIONS.md#two-pressure-levels-impose-a-separate-integer-period-constraint)

- <a id="evidence-owner-and-next-gate"></a>[evidence-owner-and-next-gate](research/archive/c6/PRESSURE_EXCURSIONS.md#evidence-owner-and-next-gate)

- <a id="evidence-owner-and-extension-conditions"></a>[Evidence owner and extension conditions](research/archive/c6/PRESSURE_EXCURSIONS.md#evidence-owner-and-extension-conditions)

- <a id="28-a-third-pressure-level-and-a-transverse-finite-class-drift"></a>[28. A third pressure level and a transverse finite-class drift](research/archive/c6/PRESSURE_EXCURSIONS.md#28-a-third-pressure-level-and-a-transverse-finite-class-drift)

- <a id="finite-level-arithmetic-reuses-the-accumulated-nodal-equation"></a>[Finite-level arithmetic reuses the accumulated nodal equation](research/archive/c6/PRESSURE_EXCURSIONS.md#finite-level-arithmetic-reuses-the-accumulated-nodal-equation)

- <a id="a-new-level-is-reached-after-actual-node-1-overshoot"></a>[A new level is reached, after actual node-1 overshoot](research/archive/c6/PRESSURE_EXCURSIONS.md#a-new-level-is-reached-after-actual-node-1-overshoot)

- <a id="a-different-node-excludes-confinement-to-the-three-visible-states"></a>[A different node excludes confinement to the three visible states](research/archive/c6/PRESSURE_EXCURSIONS.md#a-different-node-excludes-confinement-to-the-three-visible-states)

- <a id="evidence-and-the-next-structural-gate"></a>[Evidence and the next structural gate](research/archive/c6/PRESSURE_EXCURSIONS.md#evidence-and-the-next-structural-gate)

- <a id="29-frozen-neighborhood-bounds-and-a-censored-node-4-sign-test"></a>[29. Frozen-neighborhood bounds and a censored node-4 sign test](research/archive/c6/PRESSURE_EXCURSIONS.md#29-frozen-neighborhood-bounds-and-a-censored-node-4-sign-test)

- <a id="local-pressure-constancy-does-not-require-a-frozen-whole-graph"></a>[Local pressure constancy does not require a frozen whole graph](research/archive/c6/PRESSURE_EXCURSIONS.md#local-pressure-constancy-does-not-require-a-frozen-whole-graph)

- <a id="the-neighbor-changes-first-without-reversing-the-pressure"></a>[The neighbor changes first, without reversing the pressure](research/archive/c6/PRESSURE_EXCURSIONS.md#the-neighbor-changes-first-without-reversing-the-pressure)

- <a id="the-sign-threshold-constrains-the-coupled-neighborhood-budget"></a>[The sign threshold constrains the coupled neighborhood budget](research/archive/c6/PRESSURE_EXCURSIONS.md#the-sign-threshold-constrains-the-coupled-neighborhood-budget)

- <a id="evidence-and-next-gate"></a>[evidence-and-next-gate](research/archive/c6/PRESSURE_EXCURSIONS.md#evidence-and-next-gate)

- <a id="evidence-and-remaining-requirements"></a>[Evidence and remaining requirements](research/archive/c6/PRESSURE_EXCURSIONS.md#evidence-and-remaining-requirements)

- <a id="30-a-coupled-profile-disagreement-tube-and-finite-numerical-band-horizon"></a>[30. A coupled profile, disagreement tube and finite numerical-band horizon](research/archive/c6/PRESSURE_EXCURSIONS.md#30-a-coupled-profile-disagreement-tube-and-finite-numerical-band-horizon)

- <a id="b32-the-actual-source-determines-a-centered-relative-profile"></a>[B32: the actual source determines a centered relative profile](research/archive/c6/PRESSURE_EXCURSIONS.md#b32-the-actual-source-determines-a-centered-relative-profile)

- <a id="b33-exact-carried-evolution-separates-shape-rounding-and-mean"></a>[B33: exact carried evolution separates shape, rounding and mean](research/archive/c6/PRESSURE_EXCURSIONS.md#b33-exact-carried-evolution-separates-shape-rounding-and-mean)

- <a id="b34-exact-c6-contraction-distinguishes-norm-and-energy-gains"></a>[B34: exact C6 contraction distinguishes norm and energy gains](research/archive/c6/PRESSURE_EXCURSIONS.md#b34-exact-c6-contraction-distinguishes-norm-and-energy-gains)

- <a id="b35-uniform-numerical-bounds-give-a-conditional-disagreement-tube"></a>[B35: uniform numerical bounds give a conditional disagreement tube](research/archive/c6/PRESSURE_EXCURSIONS.md#b35-uniform-numerical-bounds-give-a-conditional-disagreement-tube)

- <a id="b36-close-a-finite-band-premise-and-assess-the-remaining-cut"></a>[B36: close a finite band premise and assess the remaining cut](research/archive/c6/PRESSURE_EXCURSIONS.md#b36-close-a-finite-band-premise-and-assess-the-remaining-cut)

- <a id="31-b37-closing-the-spatial-and-rounding-bounds-on-each-other"></a>[31. B37: closing the spatial and rounding bounds on each other](research/archive/c6/PRESSURE_EXCURSIONS.md#31-b37-closing-the-spatial-and-rounding-bounds-on-each-other)

- <a id="32-b38-static-compensation-refutes-a-class-wide-linear-drift-argument"></a>[32. B38: static compensation refutes a class-wide linear drift argument](research/archive/c6/PRESSURE_EXCURSIONS.md#32-b38-static-compensation-refutes-a-class-wide-linear-drift-argument)

- <a id="33-b39-a-finite-first-passage-theorem-for-the-pressure-cut"></a>[33. B39: a finite first-passage theorem for the pressure cut](research/archive/c6/PRESSURE_EXCURSIONS.md#33-b39-a-finite-first-passage-theorem-for-the-pressure-cut)

- <a id="34-b40-the-realized-first-passage-and-a-finite-mean-budget-reversal"></a>[34. B40: the realized first passage and a finite mean-budget reversal](research/archive/c6/PRESSURE_EXCURSIONS.md#34-b40-the-realized-first-passage-and-a-finite-mean-budget-reversal)

- <a id="35-b41-complete-carry-cells-cannot-provide-invariant-trapping"></a>[35. B41: complete carry cells cannot provide invariant trapping](research/archive/c6/PRESSURE_EXCURSIONS.md#35-b41-complete-carry-cells-cannot-provide-invariant-trapping)

- <a id="36-b42-every-carry-leaves-the-seven-static-compensation-cells"></a>[36. B42: every carry leaves the seven static compensation cells](research/archive/c6/PRESSURE_EXCURSIONS.md#36-b42-every-carry-leaves-the-seven-static-compensation-cells)

- <a id="37-b43-a-finite-live-bridge-from-the-original-winding-preparation"></a>[37. B43: a finite live bridge from the original winding preparation](research/archive/c6/PRESSURE_EXCURSIONS.md#37-b43-a-finite-live-bridge-from-the-original-winding-preparation)

- <a id="actual-phase-evidence-accompanies-each-finite-nodal-record"></a>[Actual phase evidence accompanies each finite nodal record](research/archive/c6/PRESSURE_EXCURSIONS.md#actual-phase-evidence-accompanies-each-finite-nodal-record)

- <a id="completed-finite-capture-and-remaining-scope"></a>[Completed finite capture and remaining scope](research/archive/c6/PRESSURE_EXCURSIONS.md#completed-finite-capture-and-remaining-scope)

- <a id="38-b44-a-bounded-mean-interval-does-not-close-the-centered-energy-tube"></a>[38. B44: a bounded mean interval does not close the centered-energy tube](research/archive/c6/PRESSURE_EXCURSIONS.md#38-b44-a-bounded-mean-interval-does-not-close-the-centered-energy-tube)

- <a id="candidate-region-and-actual-starting-state-membership"></a>[Candidate region and actual starting-state membership](research/archive/c6/PRESSURE_EXCURSIONS.md#candidate-region-and-actual-starting-state-membership)

- <a id="exact-common-translation-window"></a>[Exact common translation window](research/archive/c6/PRESSURE_EXCURSIONS.md#exact-common-translation-window)

- <a id="outward-witnesses-for-arbitrary-rational-mean-endpoints"></a>[Outward witnesses for arbitrary rational mean endpoints](research/archive/c6/PRESSURE_EXCURSIONS.md#outward-witnesses-for-arbitrary-rational-mean-endpoints)

- <a id="scope-of-the-obstruction-and-the-next-dependency"></a>[Scope of the obstruction and the next dependency](research/archive/c6/PRESSURE_EXCURSIONS.md#scope-of-the-obstruction-and-the-next-dependency)

- <a id="39-b45-the-coordinate-arithmetic-class-does-not-rescue-local-mean-confinement"></a>[39. B45: the coordinate arithmetic class does not rescue local mean confinement](research/archive/c6/RELATIONAL_REGIONS.md#39-b45-the-coordinate-arithmetic-class-does-not-rescue-local-mean-confinement)

- <a id="coordinate-congruences-derived-from-the-allowed-nodal-areas"></a>[Coordinate congruences derived from the allowed nodal areas](research/archive/c6/RELATIONAL_REGIONS.md#coordinate-congruences-derived-from-the-allowed-nodal-areas)

- <a id="a-mean-preserving-lift-onto-all-six-coordinate-classes"></a>[A mean-preserving lift onto all six coordinate classes](research/archive/c6/RELATIONAL_REGIONS.md#a-mean-preserving-lift-onto-all-six-coordinate-classes)

- <a id="uniform-energy-control-without-enumerating-lifted-states"></a>[Uniform energy control without enumerating lifted states](research/archive/c6/RELATIONAL_REGIONS.md#uniform-energy-control-without-enumerating-lifted-states)

- <a id="the-independent-mean-boundary-still-has-an-outward-witness"></a>[The independent mean boundary still has an outward witness](research/archive/c6/RELATIONAL_REGIONS.md#the-independent-mean-boundary-still-has-an-outward-witness)

- <a id="40-b46-opposite-node-relay-strips-and-a-signed-local-budget"></a>[40. B46: opposite-node relay strips and a signed local budget](research/archive/c6/RELATIONAL_REGIONS.md#40-b46-opposite-node-relay-strips-and-a-signed-local-budget)

- <a id="two-coupled-cell-coordinates-with-independent-switches"></a>[Two coupled cell coordinates with independent switches](research/archive/c6/RELATIONAL_REGIONS.md#two-coupled-cell-coordinates-with-independent-switches)

- <a id="removing-the-oscillatory-contribution-exposes-a-signed-drift"></a>[Removing the oscillatory contribution exposes a signed drift](research/archive/c6/RELATIONAL_REGIONS.md#removing-the-oscillatory-contribution-exposes-a-signed-drift)

- <a id="result-and-next-boundary"></a>[Result and next boundary](research/archive/c6/RELATIONAL_REGIONS.md#result-and-next-boundary)

- <a id="41-b47-a-local-relay-budget-survives-the-interacting-boundary"></a>[41. B47: a local relay budget survives the interacting boundary](research/archive/c6/RELATIONAL_REGIONS.md#41-b47-a-local-relay-budget-survives-the-interacting-boundary)

- <a id="the-third-switch-cannot-be-treated-as-another-independent-relay"></a>[The third switch cannot be treated as another independent relay](research/archive/c6/RELATIONAL_REGIONS.md#the-third-switch-cannot-be-treated-as-another-independent-relay)

- <a id="local-support-supplies-a-common-corrected-coordinate"></a>[Local support supplies a common corrected coordinate](research/archive/c6/RELATIONAL_REGIONS.md#local-support-supplies-a-common-corrected-coordinate)

- <a id="analytic-deadline-and-first-held-neighborhood-exit"></a>[Analytic deadline and first held-neighborhood exit](research/archive/c6/RELATIONAL_REGIONS.md#analytic-deadline-and-first-held-neighborhood-exit)

- <a id="42-b48-exact-correlated-set-viability-and-the-remaining-global-gap"></a>[42. B48: exact correlated-set viability and the remaining global gap](research/archive/c6/RELATIONAL_REGIONS.md#42-b48-exact-correlated-set-viability-and-the-remaining-global-gap)

- <a id="universal-preimages-rather-than-an-observed-return"></a>[Universal preimages, rather than an observed return](research/archive/c6/RELATIONAL_REGIONS.md#universal-preimages-rather-than-an-observed-return)

- <a id="the-actual-profile-candidate"></a>[The actual profile candidate](research/archive/c6/RELATIONAL_REGIONS.md#the-actual-profile-candidate)

- <a id="exact-scope-of-the-quadratic-controls"></a>[Exact scope of the quadratic controls](research/archive/c6/RELATIONAL_REGIONS.md#exact-scope-of-the-quadratic-controls)

- <a id="reproduction-and-unresolved-target"></a>[Reproduction and unresolved target](research/archive/c6/RELATIONAL_REGIONS.md#reproduction-and-unresolved-target)

- <a id="43-b49-protected-pair-contrasts-and-relational-past-envelopes"></a>[43. B49: protected pair contrasts and relational past envelopes](research/archive/c6/RELATIONAL_REGIONS.md#43-b49-protected-pair-contrasts-and-relational-past-envelopes)

- <a id="conditional-contrast-strips-from-the-nodal-map"></a>[Conditional contrast strips from the nodal map](research/archive/c6/RELATIONAL_REGIONS.md#conditional-contrast-strips-from-the-nodal-map)

- <a id="a-compact-envelope-of-states-with-a-compatible-past"></a>[A compact envelope of states with a compatible past](research/archive/c6/RELATIONAL_REGIONS.md#a-compact-envelope-of-states-with-a-compatible-past)

- <a id="current-result-and-reproduction"></a>[Current result and reproduction](research/archive/c6/RELATIONAL_REGIONS.md#current-result-and-reproduction)

- <a id="44-b50-exact-point-predecessors-and-the-temporal-correlation-gap"></a>[44. B50: exact point predecessors and the temporal-correlation gap](research/archive/c6/RELATIONAL_REGIONS.md#44-b50-exact-point-predecessors-and-the-temporal-correlation-gap)

- <a id="exact-predecessor-sets-not-a-graph-of-visible-labels"></a>[Exact predecessor sets, not a graph of visible labels](research/archive/c6/RELATIONAL_REGIONS.md#exact-predecessor-sets-not-a-graph-of-visible-labels)

- <a id="current-branch-and-pointwise-first-exit-exclusions"></a>[Current branch and pointwise first-exit exclusions](research/archive/c6/RELATIONAL_REGIONS.md#current-branch-and-pointwise-first-exit-exclusions)

- <a id="why-retaining-temporal-labels-helps-and-what-remains-open"></a>[Why retaining temporal labels helps, and what remains open](research/archive/c6/RELATIONAL_REGIONS.md#why-retaining-temporal-labels-helps-and-what-remains-open)

- <a id="45-b51-whole-outgoing-regions-excluded-by-their-complete-pasts"></a>[45. B51: whole outgoing regions excluded by their complete pasts](research/archive/c6/RELATIONAL_REGIONS.md#45-b51-whole-outgoing-regions-excluded-by-their-complete-pasts)

- <a id="region-targets-and-the-complete-past-argument"></a>[Region targets and the complete-past argument](research/archive/c6/RELATIONAL_REGIONS.md#region-targets-and-the-complete-past-argument)

- <a id="current-c6-result"></a>[Current C6 result](research/archive/c6/RELATIONAL_REGIONS.md#current-c6-result)

- <a id="complementary-probes-and-the-next-boundary"></a>[Complementary probes and the next boundary](research/archive/c6/RELATIONAL_REGIONS.md#complementary-probes-and-the-next-boundary)

- <a id="independent-validation-and-retained-source"></a>[Independent validation and retained source](research/archive/c6/RELATIONAL_REGIONS.md#independent-validation-and-retained-source)

- <a id="46-b52-a-nodal-excursion-budget-excludes-the-node-2-lower-boundary"></a>[46. B52: a nodal excursion budget excludes the node-2 lower boundary](research/archive/c6/RELATIONAL_REGIONS.md#46-b52-a-nodal-excursion-budget-excludes-the-node-2-lower-boundary)

- <a id="the-infinite-claim-reduces-to-one-derived-finite-gate"></a>[The infinite claim reduces to one derived finite gate](research/archive/c6/RELATIONAL_REGIONS.md#the-infinite-claim-reduces-to-one-derived-finite-gate)

- <a id="shared-implementation-and-present-coverage"></a>[Shared implementation and present coverage](research/archive/c6/RELATIONAL_REGIONS.md#shared-implementation-and-present-coverage)

- <a id="complementary-controls-and-next-proof-target"></a>[Complementary controls and next proof target](research/archive/c6/RELATIONAL_REGIONS.md#complementary-controls-and-next-proof-target)

- <a id="47-b53-separate-cell-offset-budgets-exclude-four-further-regions"></a>[47. B53: separate cell-offset budgets exclude four further regions](research/archive/c6/RELATIONAL_REGIONS.md#47-b53-separate-cell-offset-budgets-exclude-four-further-regions)

- <a id="a-shared-origin-containing-forward-envelope"></a>[A shared origin-containing forward envelope](research/archive/c6/RELATIONAL_REGIONS.md#a-shared-origin-containing-forward-envelope)

- <a id="preserve-each-targets-own-proof-coordinate"></a>[Preserve each target's own proof coordinate](research/archive/c6/RELATIONAL_REGIONS.md#preserve-each-targets-own-proof-coordinate)

- <a id="remaining-gap-and-useful-negative-controls"></a>[Remaining gap and useful negative controls](research/archive/c6/RELATIONAL_REGIONS.md#remaining-gap-and-useful-negative-controls)

- <a id="validated-return-reduction-and-stopped-refinements"></a>[Validated return reduction and stopped refinements](research/archive/c6/RELATIONAL_REGIONS.md#validated-return-reduction-and-stopped-refinements)

- <a id="48-b54-complete-return-guards-exclude-two-node-0-lower-regions"></a>[48. B54: complete return guards exclude two node-0 lower regions](research/archive/c6/RELATIONAL_REGIONS.md#48-b54-complete-return-guards-exclude-two-node-0-lower-regions)

- <a id="preserve-the-intermediate-state-in-one-shared-relation"></a>[Preserve the intermediate state in one shared relation](research/archive/c6/RELATIONAL_REGIONS.md#preserve-the-intermediate-state-in-one-shared-relation)

- <a id="backward-exclusion-covers-every-possible-visit-length"></a>[Backward exclusion covers every possible visit length](research/archive/c6/RELATIONAL_REGIONS.md#backward-exclusion-covers-every-possible-visit-length)

- <a id="a-tested-stronger-temporal-refinement"></a>[A tested stronger temporal refinement](research/archive/c6/RELATIONAL_REGIONS.md#a-tested-stronger-temporal-refinement)

- <a id="49-b55-exact-unions-exclude-another-complete-first-exit-region"></a>[49. B55: exact unions exclude another complete first-exit region](research/archive/c6/RELATIONAL_REGIONS.md#49-b55-exact-unions-exclude-another-complete-first-exit-region)

- <a id="bounded-complementary-controls-and-reuse"></a>[Bounded complementary controls and reuse](research/archive/c6/RELATIONAL_REGIONS.md#bounded-complementary-controls-and-reuse)

- <a id="50-b56-global-counts-lose-chronology-while-word-budgets-recover-it"></a>[50. B56: global counts lose chronology while word budgets recover it](research/archive/c6/HISTORY_EXCLUSIONS.md#50-b56-global-counts-lose-chronology-while-word-budgets-recover-it)

- <a id="an-exact-obstruction-to-the-global-count-relaxation"></a>[An exact obstruction to the global count relaxation](research/archive/c6/HISTORY_EXCLUSIONS.md#an-exact-obstruction-to-the-global-count-relaxation)

- <a id="joint-word-guards-give-exact-repetition-limits"></a>[Joint word guards give exact repetition limits](research/archive/c6/HISTORY_EXCLUSIONS.md#joint-word-guards-give-exact-repetition-limits)

- <a id="integration-tested-boundaries-and-the-remaining-proof"></a>[Integration, tested boundaries and the remaining proof](research/archive/c6/HISTORY_EXCLUSIONS.md#integration-tested-boundaries-and-the-remaining-proof)

- <a id="51-b57-joint-last-return-memory-excludes-a-further-whole-exit-slab"></a>[51. B57: joint last-return memory excludes a further whole exit slab](research/archive/c6/HISTORY_EXCLUSIONS.md#51-b57-joint-last-return-memory-excludes-a-further-whole-exit-slab)

- <a id="a-descending-cover-with-an-explicit-origin-sentinel"></a>[A descending cover with an explicit origin sentinel](research/archive/c6/HISTORY_EXCLUSIONS.md#a-descending-cover-with-an-explicit-origin-sentinel)

- <a id="whole-target-exclusion-preserves-terminal-transient-visits"></a>[Whole-target exclusion preserves terminal transient visits](research/archive/c6/HISTORY_EXCLUSIONS.md#whole-target-exclusion-preserves-terminal-transient-visits)

- <a id="reuse-and-boundaries-of-the-method"></a>[Reuse and boundaries of the method](research/archive/c6/HISTORY_EXCLUSIONS.md#reuse-and-boundaries-of-the-method)

- <a id="52-b58-exact-redundancy-and-selective-safe-memory-partitions"></a>[52. B58: exact redundancy and selective safe-memory partitions](research/archive/c6/HISTORY_EXCLUSIONS.md#52-b58-exact-redundancy-and-selective-safe-memory-partitions)

- <a id="propagated-scalar-exclusions-leave-the-residual-operator-unchanged"></a>[Propagated scalar exclusions leave the residual operator unchanged](research/archive/c6/HISTORY_EXCLUSIONS.md#propagated-scalar-exclusions-leave-the-residual-operator-unchanged)

- <a id="preserve-the-exact-excluded-pieces-instead-of-their-hulls"></a>[Preserve the exact excluded pieces instead of their hulls](research/archive/c6/HISTORY_EXCLUSIONS.md#preserve-the-exact-excluded-pieces-instead-of-their-hulls)

- <a id="bounded-results-and-the-next-gate"></a>[bounded-results-and-the-next-gate](research/archive/c6/HISTORY_EXCLUSIONS.md#bounded-results-and-the-next-gate)

- <a id="bounded-results-and-unresolved-scope"></a>[Bounded results and unresolved scope](research/archive/c6/HISTORY_EXCLUSIONS.md#bounded-results-and-unresolved-scope)

- <a id="53-b59-canonical-fixed-safe-piece-memory-certificates"></a>[53. B59: canonical fixed-safe-piece memory certificates](research/archive/c6/HISTORY_EXCLUSIONS.md#53-b59-canonical-fixed-safe-piece-memory-certificates)

- <a id="proof-derived-partitions-of-the-unchanged-nodal-state"></a>[Proof-derived partitions of the unchanged nodal state](research/archive/c6/HISTORY_EXCLUSIONS.md#proof-derived-partitions-of-the-unchanged-nodal-state)

- <a id="complete-graph-construction-and-persistent-source-coverage"></a>[Complete graph construction and persistent source coverage](research/archive/c6/HISTORY_EXCLUSIONS.md#complete-graph-construction-and-persistent-source-coverage)

- <a id="canonical-campaign-and-next-refinement"></a>[Canonical campaign and next refinement](research/archive/c6/HISTORY_EXCLUSIONS.md#canonical-campaign-and-next-refinement)

- <a id="54-b60-exact-predecessor-cuts-and-their-computational-cost"></a>[54. B60: exact predecessor cuts and their computational cost](research/archive/c6/HISTORY_EXCLUSIONS.md#54-b60-exact-predecessor-cuts-and-their-computational-cost)

- <a id="guarded-paths-justify-each-additional-removed-set"></a>[Guarded paths justify each additional removed set](research/archive/c6/HISTORY_EXCLUSIONS.md#guarded-paths-justify-each-additional-removed-set)

- <a id="representation-strength-and-work-budget-are-distinct-controls"></a>[Representation strength and work budget are distinct controls](research/archive/c6/HISTORY_EXCLUSIONS.md#representation-strength-and-work-budget-are-distinct-controls)

- <a id="canonical-integration-and-verified-campaign"></a>[Canonical integration and verified campaign](research/archive/c6/HISTORY_EXCLUSIONS.md#canonical-integration-and-verified-campaign)

- <a id="next-bounded-discriminator"></a>[Next bounded discriminator](research/archive/c6/HISTORY_EXCLUSIONS.md#next-bounded-discriminator)

- <a id="55-b61-priority-refinement-excludes-another-original-exit-label"></a>[55. B61: priority refinement excludes another original exit label](research/archive/c6/HISTORY_EXCLUSIONS.md#55-b61-priority-refinement-excludes-another-original-exit-label)

- <a id="count-and-coverage-premises"></a>[Count and coverage premises](research/archive/c6/HISTORY_EXCLUSIONS.md#count-and-coverage-premises)

- <a id="advisory-priority-preserves-the-complete-history-cover"></a>[Advisory priority preserves the complete history cover](research/archive/c6/HISTORY_EXCLUSIONS.md#advisory-priority-preserves-the-complete-history-cover)

- <a id="complete-target-observation-and-independent-verification"></a>[Complete target observation and independent verification](research/archive/c6/HISTORY_EXCLUSIONS.md#complete-target-observation-and-independent-verification)

- <a id="56-b62-canonical-synergies-and-proof-ownership"></a>[56. B62: canonical synergies and proof ownership](research/archive/c6/HISTORY_EXCLUSIONS.md#56-b62-canonical-synergies-and-proof-ownership)

- <a id="what-belongs-to-the-nodal-model-and-what-belongs-to-its-representation"></a>[What belongs to the nodal model and what belongs to its representation](research/archive/c6/HISTORY_EXCLUSIONS.md#what-belongs-to-the-nodal-model-and-what-belongs-to-its-representation)

- <a id="a-reusable-certificate-for-retained-history-covers"></a>[A reusable certificate for retained history covers](research/archive/c6/HISTORY_EXCLUSIONS.md#a-reusable-certificate-for-retained-history-covers)

- <a id="reuse-priorities-and-mathematical-discriminators"></a>[Reuse priorities and mathematical discriminators](research/archive/c6/HISTORY_EXCLUSIONS.md#reuse-priorities-and-mathematical-discriminators)

- <a id="bounded-cover-controls-and-decisions"></a>[Bounded cover controls and decisions](research/archive/c6/HISTORY_EXCLUSIONS.md#bounded-cover-controls-and-decisions)

- <a id="57-b63-shared-priority-discovery-and-retained-cover-verification"></a>[57. B63: shared priority discovery and retained-cover verification](research/archive/c6/HISTORY_EXCLUSIONS.md#57-b63-shared-priority-discovery-and-retained-cover-verification)

- <a id="primitive-reconstruction-is-the-admission-boundary"></a>[Primitive reconstruction is the admission boundary](research/archive/c6/HISTORY_EXCLUSIONS.md#primitive-reconstruction-is-the-admission-boundary)

- <a id="advisory-scheduling-and-complete-accounting"></a>[Advisory scheduling and complete accounting](research/archive/c6/HISTORY_EXCLUSIONS.md#advisory-scheduling-and-complete-accounting)

- <a id="canonical-campaign-and-regression-boundary"></a>[Canonical campaign and regression boundary](research/archive/c6/HISTORY_EXCLUSIONS.md#canonical-campaign-and-regression-boundary)

- <a id="58-b64-exact-exclusion-transfer-and-local-chronology"></a>[58. B64: exact exclusion transfer and local chronology](research/archive/c6/HISTORY_EXCLUSIONS.md#58-b64-exact-exclusion-transfer-and-local-chronology)

- <a id="new-exact-pieces-and-their-limits"></a>[New exact pieces and their limits](research/archive/c6/HISTORY_EXCLUSIONS.md#new-exact-pieces-and-their-limits)

- <a id="matched-transfer-control"></a>[Matched transfer control](research/archive/c6/HISTORY_EXCLUSIONS.md#matched-transfer-control)

- <a id="exhausting-the-local-node-0-hull-calculation"></a>[Exhausting the local node-0 hull calculation](research/archive/c6/HISTORY_EXCLUSIONS.md#exhausting-the-local-node-0-hull-calculation)

- <a id="separate-ingress-paths-retain-additional-correlations"></a>[Separate ingress paths retain additional correlations](research/archive/c6/HISTORY_EXCLUSIONS.md#separate-ingress-paths-retain-additional-correlations)

- <a id="59-b65-a-complete-mask-8-prehistory-layer"></a>[59. B65: a complete mask-8 prehistory layer](research/archive/c6/HISTORY_EXCLUSIONS.md#59-b65-a-complete-mask-8-prehistory-layer)

- <a id="common-nodal-content-distinct-full-histories"></a>[Common nodal content, distinct full histories](research/archive/c6/HISTORY_EXCLUSIONS.md#common-nodal-content-distinct-full-histories)

- <a id="60-b66-exact-geometric-gain-and-reuse-of-local-histories"></a>[60. B66: exact geometric gain and reuse of local histories](research/archive/c6/HISTORY_EXCLUSIONS.md#60-b66-exact-geometric-gain-and-reuse-of-local-histories)

- <a id="a-complete-selected-family-with-measured-gain"></a>[A complete selected family, with measured gain](research/archive/c6/HISTORY_EXCLUSIONS.md#a-complete-selected-family-with-measured-gain)

- <a id="reusing-an-admitted-cover-reveals-information-that-a-hull-loses"></a>[Reusing an admitted cover reveals information that a hull loses](research/archive/c6/HISTORY_EXCLUSIONS.md#reusing-an-admitted-cover-reveals-information-that-a-hull-loses)

- <a id="research-value-and-the-next-decision"></a>[Research value and the next decision](research/archive/c6/HISTORY_EXCLUSIONS.md#research-value-and-the-next-decision)

- <a id="61-b67-the-second-mask8-family-and-a-shared-refinement-kernel"></a>[61. B67: the second mask8 family and a shared refinement kernel](research/archive/c6/HISTORY_EXCLUSIONS.md#61-b67-the-second-mask8-family-and-a-shared-refinement-kernel)

- <a id="complete-incoming-histories-and-reuse-of-known-restrictions"></a>[Complete incoming histories and reuse of known restrictions](research/archive/c6/HISTORY_EXCLUSIONS.md#complete-incoming-histories-and-reuse-of-known-restrictions)

- <a id="complete-observation-gain-without-a-hull-gain"></a>[Complete-observation gain, without a hull gain](research/archive/c6/HISTORY_EXCLUSIONS.md#complete-observation-gain-without-a-hull-gain)

- <a id="smaller-observations-complete-histories-one-shared-next-gate"></a>[smaller-observations-complete-histories-one-shared-next-gate](research/archive/c6/HISTORY_EXCLUSIONS.md#smaller-observations-complete-histories-one-shared-next-gate)

- <a id="smaller-observations-and-complete-histories"></a>[Smaller observations and complete histories](research/archive/c6/HISTORY_EXCLUSIONS.md#smaller-observations-and-complete-histories)

- <a id="62-b68-a-shared-source-control-and-the-boundary-of-local-progress"></a>[62. B68: a shared-source control and the boundary of local progress](research/archive/c6/REMAINING_OBLIGATIONS.md#62-b68-a-shared-source-control-and-the-boundary-of-local-progress)

- <a id="one-positive-branch-and-one-negative-control"></a>[One positive branch and one negative control](research/archive/c6/REMAINING_OBLIGATIONS.md#one-positive-branch-and-one-negative-control)

- <a id="coordinate-coset-relevance-real-geometric-gain-no-modular-closure"></a>[Coordinate-coset relevance: real geometric gain, no modular closure](research/archive/c6/REMAINING_OBLIGATIONS.md#coordinate-coset-relevance-real-geometric-gain-no-modular-closure)

- <a id="change-the-next-gate-preserve-the-unresolved-objective"></a>[change-the-next-gate-preserve-the-unresolved-objective](research/archive/c6/REMAINING_OBLIGATIONS.md#change-the-next-gate-preserve-the-unresolved-objective)

- <a id="limits-of-refinement-and-the-retained-separator-audit"></a>[Limits of refinement and the retained separator audit](research/archive/c6/REMAINING_OBLIGATIONS.md#limits-of-refinement-and-the-retained-separator-audit)

- <a id="63-b69-direct-dual-transfer-fails-and-a-short-entry-policy-obstruction"></a>[63. B69: direct dual transfer fails, and a short entry-policy obstruction](research/archive/c6/REMAINING_OBLIGATIONS.md#63-b69-direct-dual-transfer-fails-and-a-short-entry-policy-obstruction)

- <a id="direct-applicability-to-the-current-histories"></a>[Direct applicability to the current histories](research/archive/c6/REMAINING_OBLIGATIONS.md#direct-applicability-to-the-current-histories)

- <a id="five-inequalities-expose-the-cost-of-blanket-entry-positivity"></a>[Five inequalities expose the cost of blanket entry positivity](research/archive/c6/REMAINING_OBLIGATIONS.md#five-inequalities-expose-the-cost-of-blanket-entry-positivity)

- <a id="root-conditioned-continuation"></a>[Root-conditioned continuation](research/archive/c6/REMAINING_OBLIGATIONS.md#root-conditioned-continuation)

- <a id="64-b70-executable-history-affine-obligations-and-exact-verification"></a>[64. B70: executable history-affine obligations and exact verification](research/archive/c6/REMAINING_OBLIGATIONS.md#64-b70-executable-history-affine-obligations-and-exact-verification)

- <a id="exact-sparse-certificate-complete-history-checks"></a>[Exact sparse certificate, complete history checks](research/archive/c6/REMAINING_OBLIGATIONS.md#exact-sparse-certificate-complete-history-checks)

- <a id="equivalent-geometry-and-an-existing-exact-extremum-owner"></a>[Equivalent geometry and an existing exact extremum owner](research/archive/c6/REMAINING_OBLIGATIONS.md#equivalent-geometry-and-an-existing-exact-extremum-owner)

- <a id="65-b71-mechanism-reuse-and-explicit-verification-boundaries"></a>[65. B71: mechanism reuse and explicit verification boundaries](research/archive/c6/REMAINING_OBLIGATIONS.md#65-b71-mechanism-reuse-and-explicit-verification-boundaries)

- <a id="66-b72-shared-exact-verification-and-the-first-bounded-lp"></a>[66. B72: shared exact verification and the first bounded LP](research/archive/c6/REMAINING_OBLIGATIONS.md#66-b72-shared-exact-verification-and-the-first-bounded-lp)

- <a id="67-b73-feasibility-only-control-and-equivalent-proof-coordinates"></a>[67. B73: feasibility-only control and equivalent proof coordinates](research/archive/c6/REMAINING_OBLIGATIONS.md#67-b73-feasibility-only-control-and-equivalent-proof-coordinates)

- <a id="68-b74-scaled-proposal-full-guarded-rejection-and-retained-witnesses"></a>[68. B74: scaled proposal, full guarded rejection and retained witnesses](research/archive/c6/REMAINING_OBLIGATIONS.md#68-b74-scaled-proposal-full-guarded-rejection-and-retained-witnesses)

- <a id="69-b75-verified-witness-refinement-and-inconclusive-bounded-search"></a>[69. B75: verified witness refinement and inconclusive bounded search](research/archive/c6/REMAINING_OBLIGATIONS.md#69-b75-verified-witness-refinement-and-inconclusive-bounded-search)
