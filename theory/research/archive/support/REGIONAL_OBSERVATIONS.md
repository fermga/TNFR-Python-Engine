# Retained forced-support regional observations

Frozen finite regional comparisons and their original preparation and numerical limitations.

Part of [Forced support balance and model boundaries](../../../FORCED_SUPPORT_BALANCE.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

**Archived research record.** This preserves conditional derivations and source-bound evidence. Its local next-step language is historical and does not schedule work.

## 6. Bounded executed comparison

[`forced_support_balance.py`](../../../../benchmarks/forced_support_balance.py)
executes three cases with the shared nodal Euler integrator and duration
`dt=1/4` per segment. The causal case begins at time `0.5` with the
actually born and connected child, then executes 24 segments while
holding its support, capacity and phase. The prepared compatible
control executes 12 segments; the prepared clipping control executes
two. The latter starts with uniform EPI `3.99` inside `[-4,4]`, using
the captured heterogeneous capacities and phases.

Every one of these 38 segments records the before-state, raw integrator
endpoint, explicit full pressure refresh, independent forcing capture
and finite error budgets. The campaign verifies unchanged nodes,
conductance, support, capacities, phases, effective weights and forcing.
The three reused two-segment birth preparations are recorded separately;
the complete artifact contains 44 executed Euler segments.
The final parent SHA in the causal case follows the measured endpoint.
Each executor invocation owns its transaction; the outer campaign is
an observed finite orchestration, not a new changing-node executor.

The recorded finite results are summarized below. Displayed decimals
approximate the exact rational read-outs of represented states.

| Case | Elapsed time | Observed mean change | Clipped segments |
|------|--------------|----------------------|------------------|
| Causal attachment | `6` | `-0.00010809228696952129` | `0/24` |
| Prepared compatible | `3` | `-1.1533598133637477e-16` | `0/12` |
| Prepared near-bound | `0.5` | `-0.0015017826861904956` | `2/2` |

The causal reference has compatibility residual approximately
`-0.0003191519727040398` and mean drift `-1.8015381161585726e-5`.
Its relative `H`-variance decreases from `3.687922367686489` to
`0.06603591731695654`, approximately `1.79%` of its initial value;
the Dirichlet error decreases from `7.192753706434506` to
`0.08734582300741148`. Both remain nonzero at the measured endpoint.
The largest captured pressure defect is approximately `4.48e-17` and
the largest held-input endpoint defect approximately `1.12e-16`.

The compatible control has exactly zero forcing, drift and relative
profile in the exact reference. Its observed mean change is retained
as a finite numerical defect, while its `H`-variance decreases from
`3.534687651989093` to `0.3587332981472503`.

The near-bound control clips in both segments despite a negative
reference mean drift. Its aggregate mean endpoint-defect contribution
is approximately `-0.0014927749956097026`, with a largest component
endpoint defect of approximately `0.010852609921791328`. Local upper
bound contact therefore materially changes the mean budget; it cannot
be explained by the small held-model drift alone.

Every captured mean identity, centered recurrence, raw energy budget
and relative error energy budget has exactly zero rational residual.
The complete data are generated as
`artifacts/research/forced_support_balance.json` by

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/forced_support_balance.py
```

The finite decay supports the held-support interpretation. It does not
establish the measured runtime's infinite-time limit, self-restoration
under later operators or empirical correspondence.

The 29 independent exact controls in
[`test_forced_support.py`](../../../../tests/physics/test_forced_support.py)
include a hand-solvable heterogeneous pair, arbitrary initial means,
the profile's uniform nonzero rate, cancellation between mean defects,
clipping, an excessive Euler step and tampered public caches. The
nonlinear forcing capture has 16 independent checks in
[`test_forcing_realization.py`](../../../../tests/physics/test_forcing_realization.py).
The 13 bounded campaign checks are in the
[runtime test suite](../../../../tests/physics/test_forced_support_balance_runtime.py).

The [child-target Coupling follow-up](../../../CHILD_COUPLING_FEEDBACK.md) adds exact
same-EPI reset budgets when capacity, conductance or forcing changes. It keeps
the original derived profile as a separate fixed comparison and continues
from the retained attached endpoint before its terminal Silence action.

## 8. Retained THOL regional audit

[`thol_regional_balance_audit.py`](../../../../benchmarks/thol_regional_balance_audit.py)
applies section 7 to **one** authenticated sixteen-node snapshot at `t=0.5`,
immediately after the original all-parent UM attachment and pressure refresh.
Before computation it fixes eight actual parent-child pairs, in retained
birth order, plus the complete actual-child cohort. It validates their birth
receipts, hierarchy, parent pointers, source state, forcing decomposition and
original model digest. There is no partition search, graph reconstruction,
new native call or trajectory. Nine regional observations are not nine
independent experiments or nine certified NFRs.

The full graph has 24 undirected positive-conductance edges. Each ancestry
pair has one internal edge and four cut edges. The child cohort has sixteen
cut edges and no internal edges. The normalized EPI coefficient is retained
as approximately `0.18315345241335917`; it is not refitted. The following
decimals summarize exact rational rates in the declared structural time.
Grouping by parity is only a presentation of the eight preselected pairs.

| Region | Negative internal term | Boundary term | Non-EPI source term | Model variance rate |
| --- | ---: | ---: | ---: | ---: |
| Pairs 0, 2, 4, 6 | `-0.336177322` | `-0.377518225` | `-0.007512998` | `-0.721208545` |
| Pairs 1, 3, 5, 7 | `-0.008502957` | `+0.015068434` | `-0.000818827` | `+0.005746650` |
| All actual children | `0` | `-0.005023392` | `0` | `-0.005023392` |

Four pairs instantaneously reduce their internal contrast and four increase
it. Thus a whole-network attenuation score would conceal different regional
responses. For the child cohort, the modeled variance decrease comes entirely
from the parent boundary. Its weighted-total influx is approximately
`1.3525044630156273`, with an additional `0.039791667697099464` from the
capacity channel; phase and topology contribute exactly zero to this
cohort's weighted-total rate. The capacity source has zero centered-variance
contribution in this particular cohort. Zero aggregate contribution does not
imply that a channel is absent from the complete network dynamics.

The child cohort's fresh-kernel variance defect is approximately
`8.351053868395752e-19`, and its stored-minus-fresh residual is exactly zero.
All nine observations close all four exact model/stored identities. The
conditional child rates are approximately `0.1739957797926912`; held-parent
profiles alternate approximately `1.0665389696346133` and
`0.6469908610321591`. Actual parents also evolve, so those conditional values
are neither observed equilibria nor a new frozen target for later scoring.
The audit identifies a boundary-supported response, not autonomous regional
maintenance or its absence.

Reproduction uses the original retained artifact without replaying it:

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/thol_regional_balance_audit.py
```

Input: `artifacts/research/thol_native_runtime_response_2026_09_18.json`,
SHA-256 `71252d116d8933d15a797707ed9f44ed865406422da6db492b48f6989d739c95`.
Output: `artifacts/research/thol_regional_balance_audit_2026_09_18.json`,
SHA-256 `a1cbda157979dfc85e49aed747ab68d1758ab543ab70588ea0060f33536f1a84`.
The output records its working-source scope and digest
`sha256:21d3bb97569d2ca6dbce5c206df2732397a038135990fc91baf70c4106f51921`.
Ignored local artifacts are not promised in a fresh clone; the tracked
derivation, source, portable tests and input binding retain the result's scope.

The regional owner has 33 new tests and the
[`retained audit`](../../../../tests/physics/test_thol_regional_balance_audit.py) has 19.
A combined 313-test validation covers those tests, the reused full-support,
forcing/target owners, grammar/contract/sequence witnesses and selector
symmetry. It passes with zero failures. No full-repository, laboratory,
infinite-time or autonomous-emergence verification is claimed.

## 10. Retained temporal regional identity audit

[`thol_regional_identity_audit.py`](../../../../benchmarks/thol_regional_identity_audit.py)
applies section 9 to the original control interval `t=1.5` to `1.75`.
It keeps the same eight ancestry pairs and complete child cohort as section 8.
The identity contract retains ordered EPI, its full-metric regional mean and
centered form, represented phase, capacity, membership and boundary support.
Only the explicitly labeled relative-EPI comparison factors out a uniform
regional EPI translation. Neither vertex survival nor exact relative-form
inequality decides whether a reorganizing region is a persistent NFR.

The offline reader authenticates the retained input, ancestry, original
model, source captures, full support and metric, clocks, ordered IL writes,
Euler entry/exit and later phase-normalization/coordination boundaries.
Complete retained JSON references preserve the available bounded histories,
selector configuration and counters. They do not expose opaque resource
contents or recover an unrecorded past. No live graph is reconstructed,
pressure kernel replayed, native step executed or new trajectory generated.

All nine regions retain their membership, support and capacity; all nine
change both their weighted mean and centered EPI. Their represented phase
arrays also change. A represented-angle difference alone is not proof of
a changed circular relative pattern; normalization and coordination remain
separate stages. The following decimals summarize exact finite endpoint
differences, not instantaneous derivatives or integrated physical fluxes.

| Region | Initial variance | Final variance | Variance change | Weighted-mean change |
| --- | ---: | ---: | ---: | ---: |
| Pairs 0, 2, 4, 6 | `0.902258620` | `0.821047349` | `-0.081211271` | `-0.009957785` |
| Pairs 1, 3, 5, 7 | `0.024179569` | `0.024454198` | `+0.000274629` | `+0.009676222` |
| All actual children | `0.002657877` | `0.003925631` | `+0.001267755` | `+0.015229068` |

Parity groups only summarize the preselected pairs; the artifact preserves
their individual exact values. In the child cohort, the finite variance
change is approximately `0.0012677546577278922`, with this decomposition:

| Contribution to child variance change | Value |
| --- | ---: |
| Initial internal term multiplied by `dt` | `0` |
| Initial boundary term multiplied by `dt` | `+0.0015088569377293777` |
| Initial centered phase/capacity/topology source terms multiplied by `dt` | `0` |
| Generated-pressure realization discrepancy multiplied by `dt` | `-6.412689936119145e-19` |
| Observed IL pressure-write term multiplied by `dt` | `-0.000364318044755452` |
| Euler quadratic term | `+0.00012321576475397104` |
| Combined endpoint-defect terms | `-4.001406686548848e-18` |

The parent boundary supplies the growing child contrast, while IL reduces
part of that first-order drive. Section 8's earlier negative instantaneous
rate and this later positive finite change are different observations;
neither alone establishes instability, recovery or loss of coherence.
There are still no internal child-child edges. Boundary-supported contrast
is compatible with the conditional nodal response in section 9, without
making its held-parent target an actual maintained equilibrium.

The captured phase source changes after integration: the largest absolute
component difference is approximately `0.0317983299409307`. Capacity and
topology source differences are exactly zero; normalized channel weights
are unchanged. The endpoint phase source is therefore not the source consumed
by the preceding EPI integration. No generation-time forcing capture exists
in this historical schema: reusing the entry capture at generation is a
conditional identification, checked against equal EPI, phase, capacity,
support, ordered neighbors and source configuration. Historical IL rows
also lack a resolved retention factor. The artifact records their observed
pressure writes and explicitly leaves `IL_factor_certified=False`.

All finite weighted-total and variance identities close exactly. The
[`portable audit tests`](../../../../tests/physics/test_thol_regional_identity_audit.py)
cover independent arithmetic, identity distinctions, 28 malformed-record
controls and a read-only retained-input smoke test when the local input is
available. Together with the finite observer and reused regional/support/
forcing tests, **156 tests pass**; all four changed Python files pass flake8.
The result is finite retained-record accounting. It supplies neither a
causal execution seal nor autonomous source maintenance, future stability
or a physical-emergence result. The known phase-enumeration sensitivity
remains an explicit boundary for subsequent source-maintenance claims.

Reproduction consumes the same pinned input as section 8:

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/thol_regional_identity_audit.py
```

Output: `artifacts/research/thol_regional_identity_audit_2026_09_18.json`,
SHA-256 `57cc9774c0645779d6e54df5fff2e0f1f33fe4bc7390bce5c90c04a869ab3745`.
Its declared working-source digest is
`sha256:f1e3ad6b0a8e6b30b12019469ea0f654b3432fd397b314b91173fdf815a2e775`.
The local validation checkpoint is
`artifacts/research/regional_identity_validation_2026_09_18.json`.
Missing ignored input is an explicit reproduction limitation; it must not
trigger an automatic rerun of its historical producer.

## 13. Versioned phase correction at the retained regional state

[`thol_exact_phase_source_comparison.py`](../../../../benchmarks/thol_exact_phase_source_comparison.py)
performs the predeclared comparison at `t=1.75`: one legacy replay matching
the archive, one `exact_components_v1` call and its node-order reversal.
All three retain the same primitive components, local neighbor order and
archived effective gains. The two exact calls have identical aligned raw
proposals and realized phases. Only the reference and new source-order
phase outputs receive fresh forcing captures, using the same endpoint
EPI, capacity, conductance and original enumeration. The existing source
audit and regional observer perform all nine exact balances.

Compared with the legacy output, the new phase differs by an almost common
rotation of approximately `0.001213708823288334` radians. The maximum
anchor-relative pattern residual is `4.440892098500626e-16`. The maximum
phase-source change is `1.072266750968715e-16`; the maximum fresh-kernel
pressure change is `1.1102230246251565e-16`, affecting five of sixteen nodes.
Kernel-defect differences are retained separately. None of the nine model
variance-rate signs changes. Pair 3 remains positive, approximately
`0.0007715771064528146`; the child cohort remains positive, approximately
`0.005813679258034855`. Historical stored nodal rates do not change.

The distinction from section 11 is substantive: that earlier comparison
used a different archived enumeration output, which changed a regional
rate's sign. The present alternative is the explicitly selected exact
version, not that archived vector relabeled. With unchanged local targets
and a common global gain, changing the global target within the same
wrapped-displacement branches produces a common rotation in exact
unrounded update arithmetic. This explains why absolute phase motion need
not change relative phase pressure; the observed binary64 residuals are
still recorded rather than declared zero by a tolerance.

This closes the bounded numerical gate, not arbitrary runtime invariance,
transcendental accuracy or regional maintenance. The native default remains
legacy. Section 14 uses already retained paired endpoints to separate
regional perturbation recovery from loss of the control's form.
No additional numeric-policy sweep is required by this result.

Local output: `artifacts/research/thol_exact_phase_source_comparison_2026_09_18.json`,
SHA-256 `1a59e2a2f0f4634dfc393096fab94550d2feeb1f119df3dfa90e8c33e1ee246c`.
The study used exactly three detached coordination calls and two forcing
captures, with no native step or historical producer rerun. Its ten new
portable tests and the reused source-audit tests pass **53 cases**.

## 14. Regional recovery versus loss of form in the retained paired window

[`thol_regional_recovery_audit.py`](../../../../benchmarks/thol_regional_recovery_audit.py)
reads the authenticated control/child-Emission window at six predeclared
times: `1.75, 2, 2.25, 2.5, 2.75, 3`. It uses all eight actual ancestry
pairs and the child cohort. It executes no dynamics, coordination or pressure
capture. These observations retain the historical **legacy** phase policy;
they are not a continuation of the exact-version comparison in section 13.

For each region restrict the original full-graph metric `h_i=d_i/nu_i`,
checking unchanged support and capacity at all twelve branch endpoints.
For paired difference `delta=x_perturbed-x_control`, let `P_R` subtract the
regional H-weighted mean. The existing paired-distance kernel gives

`E_delta = (1/2) sum_R h_i (P_R delta)_i^2`,
`V_control = (1/2) sum_R h_i (P_R x_control)_i^2`.

The readout retains the separate mean offset, control centered vector and
its drift from the initial control. `E_delta/V_control` is available only
for positive control variance. The evolving reference is fixed by the
archived control branch, not fitted to the perturbed outcome. These are
EPI response diagnostics, not a complete definition of NFR identity.

The following endpoint ratios compare `t=3` with `t=1.75`; values are
decimal displays of exact represented rational calculations. A ratio below
one means that quantity decreased. All 54 regional endpoint observations are
retained in the artifact.

| Region | Paired centered error, end/start | Control variance, end/start | Error/control ratio improves? |
| --- | ---: | ---: | --- |
| Pair 0 | 0.287757 | 0.262526 | No |
| Pair 1 | 0.215629 | 0.932912 | Yes |
| Pair 2 | 0.278658 | 0.262344 | No |
| Pair 3 | 0.215169 | 0.700275 | Yes |
| Pair 4 | 0.278660 | 0.263815 | No |
| Pair 5 | 0.215161 | 0.732370 | Yes |
| Pair 6 | 0.278949 | 0.264487 | No |
| Pair 7 | 0.212958 | 0.689536 | Yes |
| All actual children | 416.642313 | 4.091503 | No |

All eight pairs reduce their centered response error at every recorded
step, with endpoint reductions of about 71.22%-78.70%. But the even pairs'
control contrast decreases faster than the response error: smaller absolute
error there is not improved fidelity relative to the remaining pattern.
Only the four odd pairs improve both raw and contrast-normalized error.
This does not retrospectively select those four as confirmed NFRs.

The child cohort's centered error grows from `1.697128915806442e-9` to
`7.070957167203885e-7`. Its factor of 416.64 starts from a very small
denominator; the final error/control ratio remains only about
`4.402362158591030e-5`. Its mean offset decreases from about `0.07059424`
to `0.04870253`, so mean relaxation coexists with increasing spatial error.
Every control region retains positive variance at all six sampled endpoints,
but every control centered vector changes. Neither exact original-form
maintenance nor full engine-state return is established.

The result refutes a uniform regional-recovery interpretation of the earlier
whole-network attenuation score. It does not refute TNFR or demonstrate
unbounded instability. The recorded IL/EN/AL policy, changing control forms
and shared transport remain part of the explanation. The largest child-error
increment occurs over `2.25 -> 2.5`, which contains Reception and subsequent
integration: its error changes from `1.162021485807489e-8` to
`4.894373864559578e-7`. Isolating those already recorded stages is the next
direct mechanism question; temporal coincidence alone does not attribute
the increment to Reception.

Local output: `artifacts/research/thol_regional_recovery_audit_2026_09_18.json`,
SHA-256 `5a926d82a5cbb2d2ea8f51fd1fbfe9d033aa5a59caff797f465ed0efdaeec8b0`.
Fifteen new tests and the reused regional-identity tests pass **49 cases**.
An independent standard-library recount verifies 559 exact checks over
the 54 regional endpoints, using the original full-graph degrees/capacities.
