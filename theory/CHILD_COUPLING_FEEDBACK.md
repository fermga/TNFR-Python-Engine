# Child Coupling and changes of the structural reference

**Status:** Exact event/target balances, restricted capacity/profile and local
phase-response identities, bounded canonical response controls, and a
self-consistent spatial envelope, finite pressure first-passage result and
observed finite mean-deficit repayment for fixed-source carried C6 evolution;
complete-carry-cell trapping obstruction, finite seven-cell escape proof,
and a finite graph-owned bridge from the original preparation to its phase tail.
Recovery of an earlier relative pattern requires a fixed comparison target.
**Research links:** B2.d.8-B2.d.45/Q3/O3.a, S1, S2, S3, S8, S9, S10 and S16.

## 1. Closing an existing structural feedback path

[THOL birth and transport](THOL_BIRTH_AND_TRANSPORT.md) established an
actual child node followed by an admitted parent-child connection.
[Forced support balance](FORCED_SUPPORT_BALANCE.md) then separated the
retained graph's relative profile from its mean drift during physical
EPI evolution. A child-target Coupling event now tests another existing
path: the evolved EPI enters the compatibility score, actual links
change the pressure neighborhoods, and refreshed nodal flow changes EPI.

The child must already have a U3-compatible neighbor. Equal phase alone
does not admit Coupling on an isolate. The functional candidate sample,
phase gate, EPI/sense compatibility and actual edge commit must be
captured. The child-target selector is a declared policy; the study does
not derive autonomous target selection from the resulting profile.

The
[canonical Coupling kernel](../src/tnfr/operators/_coupling_stage_kernel.py)
preserves EPI during its event. Its capacity synchronization uses the
old compatible neighbors, before functional edges are added. Default
bidirectional phase updates also use that old neighborhood. New links
can subsequently change the phase pressure even when the stored phases
are unchanged, because the neighbor phasor sets have changed.

## 2. What singleton capacity synchronization predicts

Suppose the child's only old compatible neighbor is its parent, with
parent capacity one and child capacity `nu_c=1-delta`, `0<delta<1`.
The existing default synchronization coefficient is
`s=COUPLING_GENTLE`, whose ideal expression is `1/(4*pi)`. The canonical
capacity proposal is

$$
\nu_c^+=\nu_c+s(1-\nu_c),\qquad
\delta^+=(1-s)\delta.
$$

This describes the exact companion using the represented coefficient;
the actual binary64 write has its own observable arithmetic residual.
The initial production capacity `0.95` likewise denotes its represented
value, rather than the exact rational `19/20`.

If the original cycle and single child edge of weight `omega` remain
unchanged, all other capacities remain one, and no phase contribution
changes, the weighted capacity forcing is

$$
b_{\nu}=\frac23w_{\nu}(\omega-1)\delta,
\qquad b_{\nu}^+=(1-s)b_{\nu}.
$$

The total forcing numerator scales this way only when the other
weighted channel contributions vanish. Even in that case the mean
drift changes with the metric mass:

$$
\frac{\bar v^+}{\bar v^-}
=(1-s)\frac{Z^-}{Z^+},\qquad
Z=16+\omega+\frac{\omega}{\nu_c}
$$

for the C8 with one child edge. A smaller capacity mismatch therefore
does not by itself establish the same reduction factor for mean drift.

The default functional-link branch can leave this singleton geometry.
If the child ends with `k` edges into otherwise unit-capacity cycle
vertices, with positive weights `omega_j` and no other added edges,
then the exact unweighted capacity gradients give

$$
g_{\nu,c}=\delta,\qquad
g_{\nu,j}=-\delta/3\ \text{at each adjacent cycle vertex},
$$

$$
\sum_i d_i g_{\nu,i}
=\frac{2\delta}{3}\left(\sum_{j=1}^k\omega_j-k\right).
$$

Adding links can increase the magnitude of this negative contribution
despite a smaller deficit. Unequal link weights at reflected cycle
vertices can also remove weighted phase cancellation while preserving
the same stored phases. The actual refreshed channel decomposition is
needed to interpret a changed total drift.

The reflected preparation gives a concrete nonlinear phase identity.
Let `eta=UM_theta_push`, with ideal existing phases
`theta1=(1-eta)*pi/4`, `theta7=-theta1` and child/parent phase zero.
After child edges to vertices `1` and `7` are added, vertex `1` has
neighbor phasor sum `2+i`, hence

$$
g_{\phi,1}=\frac{\arctan(1/2)}{\pi}-\frac{1-\eta}{4},
\qquad g_{\phi,7}=-g_{\phi,1}.
$$

The default coefficient makes `g_phi,1<0`. The other equal-strength
reflected pairs cancel in the weighted sum, while parent and child
phase pressure vanish. Thus

$$
b_{\phi}=w_{\phi}(\omega_1-\omega_7)g_{\phi,1}.
$$

For `omega1<omega7`, this term is positive and opposes the negative
capacity contribution. The cause is a changed neighbor-phasor geometry;
no independent phase evolution has been introduced. These expressions
are the ideal chart identities. Actual stored phases and represented
phasor arithmetic must be checked against the captured production
kernel before attributing the measured cancellation to them.

Direct UM pressure attenuation is a separate operator write. The
postevent held reference uses independently captured canonical forcing
and a refreshed full pressure; an attenuated stored pressure must not
be substituted for that reference.

## 3. Same-EPI events change both the model and its coordinates

Let two valid forced-support references describe the states before and
after one event, on the same ordered node set. Each has connected
symmetric positive conductance, positive capacity and the profile
construction of the preceding note. Index their conductances,
Laplacians, metric matrices, forcing, drift and centered relative
profiles by `0` and `1`.

At a canonical UM event the actual EPI field `x` is unchanged. Define

$$
m_j=\frac{h_j^\top x}{Z_j},\qquad
u_j=x-m_j\mathbf1-z_j,\qquad j\in\{0,1\}.
$$

The reference-induced coordinate change is exactly

$$
\delta u=u_1-u_0=(m_0-m_1)\mathbf1+z_0-z_1.
$$

The change `m1-m0` is a reweighting of the same EPI values. It is not
an accumulated physical change of EPI during the event. Likewise the
profile shift `z1-z0` changes the comparison target. Neither is an
additional term in the nodal evolution law.

The old reference may have been derived before a preceding physical
flow interval. Its source EPI is not automatically the current event
EPI. Reset observations therefore require explicit actual before/after
snapshots, validate each under its own reference and enforce identical
node order and EPI across the event.

## 4. Exact variance and Dirichlet error resets

For `V_j=u_j^T H_j u_j/2`, direct expansion gives

$$
\begin{aligned}
V_1-V_0
={}&\tfrac12u_0^\top(H_1-H_0)u_0\\
 &+u_0^\top H_1\delta u
 +\tfrac12\delta u^\top H_1\delta u.
\end{aligned}
$$

These are the metric change, the signed interaction with the changed
reference, and its quadratic term. A decrease of `V` need not describe
any improved physical EPI field: that field is identical at the two
event endpoints.

For the relative Dirichlet error `E_j=u_j^T B_j u_j/2`, the corresponding
identity is

$$
\begin{aligned}
E_1-E_0
={}&\tfrac12u_0^\top(B_1-B_0)u_0\\
 &+u_0^\top B_1\delta u
 +\tfrac12\delta u^\top B_1\delta u.
\end{aligned}
$$

The metric term reuses the existing conductance-reset identity by
holding the error field `u0` fixed across the old and new supports.
The same helper applied to the actual EPI field instead gives

$$
\tfrac12x^\top(B_1-B_0)x,
$$

the raw Dirichlet reset. Raw structural energy and error relative to a
profile are distinct observables. The error coordinates are detached
data; they are not written into live nodes or treated as new nodes.

Each subsequent held-support flow interval has the separate mean and
Euler defect budgets from the preceding block. A finite event/flow
sequence combines those interval budgets with the explicit resets
above. Its changing weighted mean additionally includes each
`m1-m0` reweighting. Telescoping such terms is exact finite accounting;
it does not prove nonincrease under arbitrary later events.

## 5. A fixed old-profile observable prevents a moving-target conclusion

Retain the original metric `H0`, projection
`P0=I-1*h0^T/Z0`, and original relative profile `z0`. Define

$$
R_0(x)=\tfrac12(P_0x-z_0)^\top H_0(P_0x-z_0).
$$

The target here is specifically the old **derived relative profile**,
not the finite observed EPI field at an earlier time. It also ignores
uniform EPI shifts by its declared projection. This fixed observable
can be evaluated after capacity or conductance changes without
asserting that the old dynamics still govern the graph.

Because the UM event preserves `x`, it preserves `R0(x)` exactly.
Meanwhile its new-model error `V1` can rise or fall due to the reset
terms. Improvement of a changing-reference score alone therefore
cannot establish recovery toward the old target.

Even exact future relaxation under the new held model has a distinct
limit for this comparison. If its coefficients remain fixed and its
unrestricted-chart error decays, then

$$
R_0(x(t))\longrightarrow
\tfrac12(P_0z_1-z_0)^\top H_0(P_0z_1-z_0).
$$

This value is generally nonzero even though the new-model error tends
to zero. Actual finite trajectories must report both observables and
their mean/pressure defects. Clipping and later structural events
invalidate a direct promotion to that asymptotic reference.

## 6. Shared observations and the bounded comparison

The existing
[`forced_support.py`](../src/tnfr/physics/forced_support.py) now centralizes
the profile coordinates in both old state observations and the new
`observe_forced_support_pattern` read-out. That read-out accepts only
the ordered node identifiers and EPI alongside the retained reference.
It keeps the original profile and metrics fixed without certifying a
current pressure law.

`observe_forced_support_reset` takes both references and explicit event
snapshots. It rebuilds their public cached fields, validates each
snapshot under its own held model, and reuses
[`support_transport.py`](../src/tnfr/physics/support_transport.py)
for the raw and error-coordinate conductance resets. Its signed
metric-first energy decomposition carries a checked exact residual.
These detached observations do not establish operator admission,
pressure refresh or a causal link between arbitrary supplied records.

The 19 independent controls in
[`test_forced_support_reset.py`](../tests/physics/test_forced_support_reset.py)
include references predating the event, separate metric/profile changes,
fixed old-profile comparisons after dynamics changes, and a direct
counterexample to a moving-target recovery claim. In that example,
unchanged EPI has positive old-profile error and exactly zero error
under a newly chosen reference; the reset terms fully account for
the apparent improvement.

The executed comparison in
[`child_coupling_feedback.py`](../benchmarks/child_coupling_feedback.py)
retains the actual born and connected graph after the preceding 24
physical segments, at time `6.5`. One case applies public child-target
UM; the matched case takes no extra event. Each then executes 12
additional refreshed Euler intervals. The comparison retains both
the old fixed-profile observable and each case's current-model error.
Its independent runtime checks are in the
[runtime test suite](../tests/physics/test_child_coupling_feedback_runtime.py).

The actual child-target event preserves every EPI and phase value.
Its child capacity changes from `0.95` to `0.9539788735772974` and it
adds child edges to cycle vertices `1` and `7`, with respective weights
`0.8791579029688148` and `0.8981657798702134`. The old parent-child edge
remains. The captured vertex-1 phase matches `(1-eta)*pi/4`; the wrapped
vertex-7 reflection residual is approximately `-2.45e-16`. This finite
check supports the ideal chart calculation above without certifying
exact transcendental arithmetic.

The materialized weighted forcing contributions are approximately:

| Channel | Before child UM | After child UM |
|---------|-----------------|----------------|
| Phase | `0` | `+0.0006063375069147582` |
| Capacity | `-0.0003191519727040398` | `-0.0006920500301710027` |

The new phase contribution opposes a larger negative capacity term.
Including the new metric mass, the reference mean drift changes from
`-1.8015381161585726e-5` to `-4.014215906537859e-6`.
This is a combined structural effect, rather than the singleton
capacity prediction alone.

At unchanged EPI the current-model variance jumps from
`0.06603591731695654` to `0.9294668175027274`. Its reset terms are
approximately `+0.03686666543873662` from the metric,
`-0.3938392738714086` from the reference interaction, and
`+1.2204035086184428` from the quadratic reference shift.
Their exact rational identity residual is zero. The raw EPI Dirichlet
reset is approximately `+0.029393575677005273`; the relative Dirichlet
error reset is `+0.46220825274344396`. They also have exact zero budget
residuals. The weighted mean reweights by approximately
`-0.018281219168014524` despite no EPI change during the event.

Both cases then execute 12 intervals of duration `1/4`, reaching time
`9.5`. They share the initial fixed old-profile variance
`R0=0.06603591731695654` and give:

| Continuation | Fixed old-profile variance at `9.5` |
|--------------|-------------------------------------|
| Child UM | `0.14196498883949968` |
| No extra event | `0.021532984486475837` |

In the child-UM case, the new-regime variance nevertheless decreases
from `0.9294668175027274` to `0.5219712930713153`. Thus a reduced
reference drift and relaxation toward the new profile coexist with
increased distance from the old target over this measured horizon.
The finite comparison does not establish recovery toward that target.

The full represented states, reset terms and later interval budgets
are generated as `artifacts/research/child_coupling_feedback.json` by

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/child_coupling_feedback.py
```

This tests one finite structural action and its subsequent evolution.
The conditional singleton calculation is not silently substituted for
the default functional-link branch or treated as an additional run.
The operator policy and all measured intervals remain explicit.

## 7. Exact full-event accounting on the same ordered nodes

The new `observe_forced_support_event` in the shared
[`forced_support.py`](../src/tnfr/physics/forced_support.py) separates an
actual EPI jump from accompanying reference and metric changes. Each
endpoint must satisfy its own reference's fixed-model domain: symmetric
nonnegative conductance with connected positive-conductance support,
positive weighted degrees, positive capacities and positive EPI coefficient.
Node count and order agree across the event. The existing same-EPI observer
retains its stricter EPI-equality contract.

For actual endpoint fields `x0`, `x1`, let

$$
\delta=x_1-x_0,\qquad
a=\frac{h_0^\top\delta}{Z_0},\qquad
v=\delta-a\mathbf1=P_0\delta.
$$

In the original reference, `u0=P0*x0-z0` and the algebraic midpoint error is
`u_mid=P0*x1-z0=u0+v`. Direct expansion gives the EPI-jump contributions

$$
J_H=u_0^\top H_0v+\tfrac12v^\top H_0v,\qquad
J_B=(B_0u_0)^\top v+\tfrac12v^\top B_0v.
$$

The first term in each is signed; positive quadratic terms do not imply a
positive total. Apply the same-EPI reset of section 4 at field `x1`, changing
from the old to the new reference. With its reset differences `R_H`, `R_B`
and mean reweighting `r_m`, the complete endpoint identities are

$$
V_1(x_1)-V_0(x_0)=J_H+R_H,\qquad
E_1(x_1)-E_0(x_0)=J_B+R_B,
$$

$$
m_1(x_1)-m_0(x_0)=a+r_m.
$$

The complete relative-error change likewise equals the centered EPI jump
`v` plus the reset's reference-error shift. Totals telescope along compatible
endpoint/reference chains. The decomposition's separate cross, quadratic
and metric terms need not cancel under reversal.

The midpoint is an algebraic evaluation of post-event EPI in old coordinates,
not an observed intermediate graph or an internal operator write order. The
public `ForcedSupportEvent` exposes it only as a `ForcedSupportPattern`;
the copied pressure used internally to validate the reused reset is not
exposed as observed midpoint pressure. Both actual endpoints and references
are rebuilt and validated; reference source EPI may predate the event.
The observer neither evolves EPI nor certifies elapsed flow, operator
admission, pressure refresh, causal provenance or nonincrease.

Independent exact controls are in
[`test_forced_support_event.py`](../tests/physics/test_forced_support_event.py).
For P2 with old `w=1`, `nu=(1,2)`, `e=1/2`, `F=(1/4,1/2)` and `x0=(1,0)`,
the original metric is `H0=diag(1,1/2)` and profile `z0=(-1/6,1/3)`.
The pure jump to `x1=(0,1)` has H cross/quadratic terms `-1,2/3` and B
terms `-3,2`. Simultaneously changing to `w=2,nu=(1,1)` with the same
forcing adds reset changes `23/96,7/16`; complete changes are `-3/32,-9/16`.
These hand-solvable balances do not require a canonical runtime campaign.

A separate exact control sets `x1=(2,0)` and a new reference with
`w=1,nu=(1,1),F=(1,-1)`. Its new-profile error is zero while the original
H-error increases from `3/8` to `25/24`. Even with a real EPI jump, a new
zero-error reference does not establish recovery of an older target. The
event's pre-reference and a study's fixed target are separate inputs to
different existing observers; the event API does not silently choose a target.

## 8. Matched perturbation and response comparison

The B2.d.9b campaign independently reaches the actual child-UM graph at
`t=9.5` in each branch, before either pending SHA closure. The target remains
the original B2.d.8 `H0,z0` derived at `t=6.5`, not the current post-UM
reference. The three branches use unchanged defaults and seed `17`:

- U: no additional event.
- P: child VAL followed by IL, then canonical pressure refresh.
- F: the identical VAL/IL prefix, then child UM and pressure refresh.

No pressure refresh is inserted between VAL and IL. Both word validators,
actual retained per-target histories and live operator gates admit the
declared words. Materialized initial states, controls and RNG provenance
agree across all branches. P/F additionally have identical VAL/IL records
and complete materialized states at the pre-feedback boundary. Every branch
then runs twelve shared Euler intervals of `0.25` to `t=12.5`; child and
parent SHA closures execute separately after measurement. All closures are
admitted, and no clipping occurs in these retained runs.

Default VAL changes child EPI from `1.140013776631149` to
`1.2307331905030023` and capacity from `0.9539788735772974` to
`1.0298941002448299`. IL changes stored pressure but preserves EPI and phase
in this particular symmetric preparation; its general phase-update path
is not absent from the operator. The following refresh is captured separately.
Feedback UM changes child capacity to `1.0275152033332051` and neighboring
phases at nodes `1,7` from approximately `(0.595761,5.687424)` to
`(0.451913,5.831272)`. It adds no edges: all branches retain eleven edges.
Neither a capacity-only perturbation nor an edge-addition explanation matches
these actual writes.

The common initial original-profile error is `0.14196498883949968`.
The matched results and the conditional exact-model limits derived in
section 9 are:

| Branch | Original-profile error at `12.5` | Held exact-model limiting error |
|--------|----------------------------------|---------------------------------|
| U | `0.2564525415209221` | `0.9461915126089225` |
| P | `0.2530099369965697` | `0.9461612626392493` |
| F | `0.17618454861095278` | `0.5998144725644323` |

With `D=R_P-R_U`, `B=R_P-R_F` and `G=R_F-R_U`, exact arithmetic on the
represented endpoints gives

```text
D = -0.0034426045243524145
B = +0.07682538838561694
G = -0.08026799290996935
D - B - G = 0 exactly (displayed decimal values are rounded).
```

Since `D<0`, the predeclared classification is **no damage in the fixed
observable**. UM has a positive finite response benefit relative to P and U;
this is not a damage/repair result, and `B/D` is deliberately undefined.
All three final errors remain above their common initial error. A lower
matched endpoint error does not establish attraction or full-state recovery.

VAL alone changes `R0` by `-0.0047656984792713`; IL and UM have zero immediate
effect on `R0` because they leave EPI fixed. Subsequent flow accounts for the
remaining change. The original-target and current-regime H-variance event/flow
telescopes both close exactly. Current-regime Dirichlet budgets remain separate
from H variance. The largest captured pressure and Euler endpoint defects are
approximately `5.47e-18` and `1.11e-16`; these measurements are not solver-error
bounds or evidence about an infinite trajectory.

The campaign reuses `prepare_child_feedback_endpoint` and the shared interval
helper in the earlier benchmark; its two old case payloads remain exactly
unchanged. Reproduction:

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/structural_perturbation_response.py
```

The script writes the local ignored artifact
`artifacts/research/structural_perturbation_response.json`, with actual source
provenance, full endpoint captures and budgets. The
[runtime controls](../tests/physics/test_structural_perturbation_response_runtime.py)
cover target continuity, matched preparation, actual writes, independent energy
identities, executor/history bindings, refusal and classification boundaries.
This finite result closes B2.d.9b; it does not justify a factor or horizon search
until recovery appears.

## 9. Frozen-model compatibility with the original target

The existing references also give a mechanism discriminator without another
trajectory or a new observable. Let `j` denote a post-event held model within
the positive-capacity, connected reversible domain of
[the forced-support balance](FORCED_SUPPORT_BALANCE.md). Its exact continuous
solution has the form

$$
x_j(t)=(m_j(0)+b_jt)\mathbf1+z_j+\exp(-A_jt)u_j(0),
\qquad u_j(0)=P_jx_j(0)-z_j.
$$

Here `t` is elapsed time in the held post-event regime, `b_j` is its
`mean_drift`, `A_j=e_j diag(nu_j) L_rw,j`, and `u_j(0)` is centered in the current
metric `H_j`. Reversible connected diffusion with positive capacity makes
this centered term tend to zero. The original projector `P0` annihilates
the uniform drift, so the previously fixed observable has the limit

$$
R_{*,j}=\lim_{t\to\infty}R_0(x_j(t))
=\tfrac12\|P_0z_j-z_0\|_{H_0}^2.
$$

Compute this quantity by passing the current reference's `relative_profile`
to `observe_forced_support_pattern(original_target, nodes=..., epi=...)`.
The profile need not have zero mean in `H0`; the shared pattern observer
performs the original projection. No second profile solver is needed.

A positive `R_*,j` excludes asymptotic restoration of the original profile
under that unchanged exact model. It does not exclude transient passage
through the target or later recovery under a different admitted model.
A zero value establishes target compatibility only. Neither value proves
the asymptotics of clipped binary64 execution, future operator admission,
full-state restoration or autonomous policy selection.

Finite increments require a separate reading. For an actual represented EPI
jump `d=x_after-x_before`, put `u=P0*x_before-z0` and `v=P0*d`. Then

$$
R_0(x_{\rm after})-R_0(x_{\rm before})
=\langle u,v\rangle_{H_0}+\tfrac12\|v\|_{H_0}^2.
$$

This same-target identity attributes an immediate change to the actual EPI
write. Phase, capacity, conductance or pressure changes that leave EPI fixed
have zero immediate contribution; they affect the following evolution.
Read those following flow defects under each interval's current model using
the existing step observer. Substituting the older recovery target as the
current flow model would confuse two different contracts.

If only child `c` changes EPI by `delta`, the same identity simplifies to

$$
\Delta R_0=h_cu_c\delta+
\tfrac12h_c\left(1-\frac{h_c}{Z_0}\right)\delta^2,
\qquad Z_0=\sum_i h_i.
$$

The actual jump's alignment with the pre-event error determines its sign.
An operator classified as a grammar destabilizer need not increase this
particular target error. The named perturbation must therefore be measured
before it can be classified as damage. This exact sign criterion is an
observation, not a rule permitting ad hoc EPI writes or selection of a new
operator factor after seeing the outcome.

For the retained VAL event, `u_c=-0.10281937232096414` and
`delta=0.09071941387185345`. Its exact H cross and quadratic terms are
approximately `-0.008206125390920804` and `+0.003440426911649504`, summing
to the recorded negative jump. With this fixed pre-event error and metric,
positive single-child jumps below `-2*u_c/(1-h_c/Z0)`, approximately
`0.21638445016881427`, reduce the target error. The actual default VAL jump
lies inside that interval. This explains its immediate direction without
choosing a new factor or claiming that the interval is a policy invariant.

The next structural question is compatibility of the target with a held
post-event model. Equivalently to `P0*z_j=z0`, require

$$
P_0\operatorname{diag}(\nu_j)
\left[-e_jL_{{\rm rw},j}z_0+F_j\right]=0.
$$

Connected positive-capacity diffusion then identifies the unique relative
profile modulo a uniform field. Every observed `R_*,j` above is positive:
UM reduces the mismatch but does not remove it. A candidate restoring rule
must explain how the canonical channels could satisfy this compatibility
condition and control the old-target error. A repeated-policy theorem further
requires an admitted full-state domain, its preservation and controlled
execution defects. None follows from extending the observed word or horizon.

## 10. Compatibility, channel cancellation and the old metric

The shared [`observe_forced_support_target`](../src/tnfr/physics/forced_support.py)
now evaluates that compatibility condition and the instantaneous signed balance
at an explicit actual snapshot. Both references are rebuilt, the snapshot must
match the current model, and all node orders must agree. Let

$$
A=e\operatorname{diag}(\nu)D^{-1}B,\quad b=\operatorname{diag}(\nu)F,\quad
r=P_0(b-Az_0),\quad m=P_0z-z_0.
$$

The current profile obeys `-A*z+b=mean_drift*1`. Since `A*1=0`,

$$
r=P_0Am.
$$

This identity gives `r=0` if and only if `m=0`: if `P0*A*m=0`, then
`A*m=c*1`; multiplying by the current metric weights `h^T` gives `c=0`
because `h^T*A=0`. Connectivity makes `m` uniform, and its old-metric
centering makes it zero. This proves relative-shape compatibility. It does
not require zero mean drift. For example, P2 with `nu=(1,2)`, `e=1/2` and
`F=(1,1/2)` preserves the zero relative profile while every EPI coordinate
drifts at rate one. The existing scalar zero-pressure compatibility test and
this vector target test answer different questions.

For an actual old-target error `u=P0*x-z0`, the held model gives

$$
\dot u=-P_0Au+r,\qquad
\dot R_0=-u^\top H_0Au+u^\top H_0r.
$$

The first term is signed; an old metric need not make it nonpositive. A
hand-solvable P3 counterexample has unit edge conductance, old capacities
`(1,2,1)`, hence `H0=I`, current capacities `(1/8,1/8,4)`, `e=1`, and
held `F=0` in both models. Both profiles are zero. At actual
`x=(1/8,3/4,5/8)`, `u=(-3/8,1/4,1/8)`, compatibility holds but
`dR0/dt=11/512>0`. This is a detached coefficient control: its zero forcing
is not the canonical default forcing for heterogeneous capacities.
If instead `H0=c*H_current`, the same identity reduces to
`dR0/dt=-c*e*u^T*B*u+u^T*H0*r`. With `r=0` it is strictly negative away
from the target in this connected centered domain. Neither sufficient
condition establishes future runtime admission or controls clipping.

[`decompose_non_epi_forcing`](../src/tnfr/physics/forcing_realization.py)
reuses the capture owner's exact phase/vf/topology products. It rebuilds
support-gradient caches and checks that their sum equals the supplied forcing;
it does not rerun or authenticate the nonlinear phase kernel. Project each
target-pressure component through `P0*diag(nu)` to obtain `r_a`, including the
EPI component. Then

$$
r=\sum_a r_a,\quad G_{ab}=r_a^\top H_0r_b,\quad
J_r=\tfrac12\|r\|_{H_0}^2
=\tfrac12\sum_a G_{aa}+\sum_{a<b}G_{ab}.
$$

Cross terms can cancel. Nonzero channel magnitudes alone do not establish
incompatibility, and summing only their squares is incorrect. `J_r` is a
derived diagnostic, not a new physical energy. In the hypothetical exact
model initialized at `u=0`, `R0(t)=J_r*t^2+O(t^3)`, so zero instantaneous
`dR0/dt` there does not imply a compatible target. No graph is assigned that
hypothetical state by this observer.

At the supplied actual EPI, the observer also separates the stored-pressure
contribution `u^T*H0*diag(nu)*(p_stored-p_model)`. A kernel defect captured
at another EPI cannot be reused as the defect at `z0`. Independent exact
checks, including signed cancellation and stale-reference controls, are in
[`test_forced_support_target.py`](../tests/physics/test_forced_support_target.py).

## 11. What the retained U/P/F channels explain

The offline analysis in
[`structural_target_compatibility.py`](../benchmarks/structural_target_compatibility.py)
reads the B2.d.9b captures without new trajectories or operator choices.
It binds each current reference's EPI, capacity and stored pressure to the
recorded actual post-event endpoint. U's first later forcing capture supplies
only its checked held phase/capacity/support coefficients; its later EPI and
kernel defect do not replace the earlier state. Parent evidence bytes and
producer metadata are retained separately from the new analysis manifest.

At the three post-event states at `t=9.5`, the exact-model read-outs are:

| Case | `J_r` | Homogeneous term | Target-source term | Total `dR0/dt` |
|------|-------|------------------|--------------------|----------------|
| U | `0.0120728383` | `-0.0396520771` | `0.0761747684` | `0.0365226913` |
| P | `0.0120906598` | `-0.0389605486` | `0.0759599729` | `0.0369994243` |
| F | `0.0043113892` | `-0.0389606073` | `0.0468645978` | `0.0079039905` |

All targets remain incompatible, and all three instantaneous old-target
rates remain positive. UM reduces the outward rate in this comparison;
it does not reverse it. The stored-pressure contributions are approximately
`2.25e-18`, `1.84e-18` and `9.55e-19`, recorded separately rather than
silently included in the exact-model terms.

The phase channel's own squared magnitude actually grows from about
`0.0030834147` in P to `0.0032578705` in F. Nevertheless its cross term with
the EPI channel changes from `+0.0037467097` to `-0.0042059349`.
Better cancellation, not a decrease of every individual channel, explains
the smaller combined incompatibility. These are projected target-rate
channels, not an identity for every phase-gradient or tetrad diagnostic.

For a symmetric accounting of the P-to-F change, define

$$
\Delta J_a=\tfrac12(r_P+r_F)^\top H_0(r_{F,a}-r_{P,a}),\qquad
J_F-J_P=\sum_a\Delta J_a.
$$

The phase, capacity and EPI allocations are approximately
`-0.00777818875`, `-0.000001046589` and `-0.0000000352605`; topology is zero.
Their exact sum equals the observed residual-energy change. This is a signed
algebraic allocation of simultaneous writes, not a set of independently
executed channel ablations or a universal causal ranking.

Reproduce the detached analysis after generating or retaining its input:

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/structural_target_compatibility.py
```

The output `artifacts/research/structural_target_compatibility.json` is local
and ignored. It records the input SHA-256, historical producer manifest and
current analysis-source digest; it does not relabel the old trajectory as
new execution. Its
[schema controls](../tests/physics/test_structural_target_compatibility_report.py)
reject stale endpoint sources, unmatched coefficients and overwritten input.

## 12. A canonical capacity-profile family and its obstruction

A useful exact family requires connected simple undirected support with one
common positive edge conductance, positive capacities, all represented phases
equal to zero, and fixed channel coefficients `e>0`, `w_vf>0`, `w_topo=0`.
These are declared restrictions; the zero topology weight and channel ratio
are read from the existing defaults in the implementation controls. Equal
conductance makes the unweighted capacity neighborhood agree with `L_rw`:

$$
F=-w_{\nu}L_{\rm rw}\nu,\quad k=\frac{w_\nu}{e},\quad
\dot x=-e\operatorname{diag}(\nu)L_{\rm rw}(x+k\nu).
$$

Here `k` is the ratio of existing channel coefficients, not a fitted factor.
The common phase has zero phase forcing, and `d^T*F=0`. Therefore the held
profile and mean drift are exactly

$$
z=-kP_H\nu,\qquad \bar v=0.
$$

For original/current states inside this family with the same ratio,

$$
P_0z-z_0=-kP_0(\nu-\nu_0).
$$

The original relative target is compatible exactly when the capacity
increment is spatially uniform. Nonuniform capacities can support a
nonuniform profile, but this identity alone does not explain how the
capacity pattern is generated or maintained. Heterogeneous conductance or
nonzero phase invalidates the simple formula; explicit counterexamples are
retained in the controls below. The U/P/F preparation has nonuniform phases
and unequal conductances, so this theorem is not substituted for section 11.

The same formula exposes a reuse path: the detached coordinate `y=x+k*nu`
has centered value `P_H*y=P_H*x-z`, exactly the existing forced-profile
error. While capacity is held, `y` follows the same pure EPI diffusion.
Across a capacity event, `Delta y=Delta x+k*Delta nu`. This is a change of
analysis coordinates, not an extra engine state or an authorized EPI write.
The coupled capacity/EPI model below reuses this coordinate and the existing
event, metric and diffusion budgets; no parallel relaxation solver is needed.

On P2, let the original capacity be `(c,c)`, and the actual new capacity
be `(c+a,c)`, `a>0`. With unit edge conductance,

$$
m=(-ka/2,ka/2),\quad
r=\frac{w_\nu a(2c+a)}2(-1,1),\quad
R_* = \frac{k^2a^2}{4c}.
$$

A single default VAL with a positive capacity increment breaks compatibility.
Under zero phase, IL preserves capacity/support and cannot remove that
incompatibility after pressure refresh; its immediate stored-pressure
contraction is a separate operator effect. Target-only UM has no new edge
available on P2. Its ideal affine capacity rule reduces the gap by
`a'=(1-gamma)*a`; binary64 execution must use its actually observed `a'`.
For those represented endpoints, the exact-model limiting-error ratio is
`(a'/a)^2`. A smaller positive gap gives a smaller positive mismatch, not
complete restoration.

The [capacity-family controls](../tests/physics/test_capacity_target_compatibility.py)
include exact P2/P3 preparations and one actual default `VAL -> IL -> UM -> SHA`
word, with both word validators, live gates, histories and pressure refreshes.
VAL breaks compatibility, IL leaves it broken, and UM reduces but does not
eliminate the observed capacity gap. SHA is a separately checked closure;
its EPI preservation is not recovery. No physical flow interval, repeated
word, infinite-time binary64 theorem or invariant full-runtime domain is
asserted by this finite control.

B2.d.10 closes the target-residual decomposition and this restricted
mechanism/obstruction. The coupled model below uses the same coordinate and
event/flow owners. Admission, history, phase, support, clipping and numerical
defects remain separate obligations for a complete runtime policy.

## 13. Joint capacity and form evolution on synchronized P2

The conditional exact model in
[`capacity_feedback.py`](../src/tnfr/physics/capacity_feedback.py) retains the
ordered target/neighbor support of section 12. Let their capacities be
`(c+a,c)`, with `c>0`, `0<=a<=A`, and set

$$
k=w_\nu/e,\quad \rho=1-\gamma,\quad s=he,\quad
d=x_0-x_1,\quad q=d+ka.
$$

Here `e>0`, `w_nu>=0` and `0<gamma<1` are declared pressure/operator
coefficients; the runtime study below reads their existing defaults. The
duration `h>=0` and bounds `A,ell,U` specify numerical and state domains,
not extra physical parameters. The zero-capacity case, changing support and
nonzero phase forcing are outside this model.

Target-only UM preserves EPI and changes `a` to `a'=rho*a`. Refreshed nodal
pressure is then `p=(-e*(d+ka'), e*(d+ka'))`. A held-input Euler interval gives

$$
a_{n+1}=\rho a_n,\qquad
q_{n+1}=[1-s(2c+\rho a_n)](q_n-k\gamma a_n),\qquad
d_{n+1}=q_{n+1}-ka_{n+1}.
$$

This is a coupled recurrence: treating the new capacity as merely a faster
clock would omit its change to the forcing profile. The API evaluates
detached rational data only; graph EPI still advances through the shared
nodal integrator. Multiple exact internal Euler substeps with the same held
pressure/capacity have this same total-duration endpoint. Their binary64
rounding and any pressure refresh inside the interval require separate checks.

A sufficient invariant domain is

$$
0\le a\le A,\quad
\ell\le x_0\le U-ka,\quad \ell+ka\le x_1\le U,
\quad 0<\ell\le U,\quad kA\le U-\ell,\quad
s(c+\rho A)\le1.
$$

In lifted coordinates both components lie in
`[ell+k(c+a), U+kc]`. The UM event subtracts `k*gamma*a` from `y0`
and reduces the required lower bound by that amount; `y1` stays fixed.
During flow the lifted update is row-stochastic, with mixing coefficients
`s(c+a')` and `s*c`, so both coordinates remain in the interval. Converting
back gives the same domain at `a'`. This proves exact-model positivity and
bounded EPI over arbitrary finite repetitions. If the interval lies inside
the configured hard clip range, exact clipping is inactive. Fitting this
positive domain above the configured UM EPI/frequency thresholds is a
sufficient scalar precondition, not a proof of full grammar/history admission.

The box can be constructed from initial data without searching:
`A=a0`, `ell=min(x0,x1-k*a0)`, `U=max(x0+k*a0,x1)`, provided the resulting
lower bound is positive and the timestep/clip conditions hold. Mere convexity
of the lifted flow is insufficient for arbitrary positive EPI: with
`c=a=1`, `k=gamma=s=1/2`, initial `x=(3/16,3/16)` gives
`x'=(0,5/16)`. These detached rational coefficients are a domain
counterexample, not the default runtime fixture or a parameter fit.

Define the uniform multiplier bound

$$
\beta=\max\{|1-2sc|,\ |1-s(2c+\rho A)|\}\le1.
$$

The executable finite envelope is

$$
|q_n|\le\beta^n|q_0|+
k\gamma a_0\beta\sum_{j=0}^{n-1}\beta^{n-1-j}\rho^j,
\qquad |d_n|\le |q_n|+ka_0\rho^n.
$$

The sum is `(beta^n-rho^n)/(beta-rho)` when the factors differ and
`n*beta^(n-1)` when equal; zero cycles are handled separately. `beta<1`
implies decay of both capacity gap and spatial EPI disagreement. At `h=0`
or the equal-capacity full-swap boundary the uniform bound can be one;
neither boundary supplies that contraction conclusion.

The mean is also accounted for. For `M=(x0+x1)/2`,

$$
M_{n+1}-M_n=-\frac{s a_{n+1}}2(d_n+ka_{n+1}),\qquad
|M_\infty-M_n|\le
\frac{s(U-\ell)a_0\rho^{n+1}}{2\gamma}.
$$

The tail follows by summing the box bound on the lifted difference. Thus the
arithmetic mean has a limit, but need not equal its initial value. For the
current metric mean `mu=[c*x0+(c+a)*x1]/(2c+a)`, the same cycle gives

$$
\mu_{n+1}-\mu_n=
\frac{c\gamma a_n d_n}{(2c+a_n)(2c+\rho a_n)}.
$$

This is exactly a same-EPI metric reset followed by zero H-mean model drift,
already owned by `observe_forced_support_reset` and
`observe_forced_support_step`. With `beta<1`, both EPI coordinates tend to
the common arithmetic-mean limit; their endpoint error is bounded by half
the disagreement envelope plus the mean-tail bound. This is an exact
conditional relaxation theorem, not a formation mechanism for a persistent
differentiated pattern. The
[exact controls](../tests/physics/test_capacity_feedback.py) retain mean
identities, boundary cases, the domain obstruction and invalid-input checks.

## 14. Finite admission and the represented capacity obstruction

The supplied word `VAL IL UM UM ... SHA` fails the string-based canonical
transition validator at `UM -> UM`; the separate `ValidatedSequence` check
alone accepts it. Both results are retained. No transition policy is changed
to obtain a run. The declared finite study instead uses `VAL IL`, then sixteen
`UM IL / refresh / Euler` blocks, and finally SHA after measurement. Within
this zero-phase P2 fixture, each separator IL leaves EPI, capacity and support
unchanged. Its raw pressure reduction is recorded, and fresh canonical
pressure matches the pre-IL pressure exactly. This verified identity explains
why the same coupled recurrence can be used as the block's exact reference.

[`benchmarks/capacity_feedback.py`](../benchmarks/capacity_feedback.py) independently
prepares two unit P2 graphs with EPI `(0.5,0.5)`, capacity `(1,1)`, zero phase
and seed 17. All operator factors are the defaults. One branch applies those
sixteen blocks; the other keeps the post-VAL/IL capacity during sixteen flow
intervals. Both end at engine time `4`, with `h=0.25`, before their separate
admitted SHA closures. This is a new restricted comparison, not another
continuation of the nonuniform-phase U/P/F graph. No factor or horizon search
is performed, and the original uniform profile and metric stay fixed.

The data-derived positive box is approximately `[0.4746697041,0.5651190317]`,
with `A=0.07957747155`; its additional capacity-dependent margins are checked
at every recorded boundary. The exact reference has `beta=0.9084232738` and
`rho=0.9204225285`. Both branches start after preparation at
`R0=0.0003957858736`. Their pre-closure read-outs are:

| Branch | Actual capacity gap | Actual `R0` | Conditional held-profile `R_*` |
|--------|---------------------|-------------|---------------------------------|
| UM/IL blocks | `0.02111475158` | `1.397878163e-7` | `1.129307508e-5` |
| Held capacity | `0.07957747155` | `3.718826587e-5` | `1.604059727e-4` |

The lower final error is a finite response benefit, not monotonic recovery:
the coupled branch's sixteenth interval increases `R0` from approximately
`4.802676566e-9` to `1.397878163e-7`. In both branches the final EPI lies
closer to the original uniform target than the current conditional frozen
profile does. A transient crossing must not be reported as a maintained
equilibrium. The arithmetic mean changes from `0.5198943679` to
`0.5193772453` in the coupled branch and `0.5188997311` in the control.

Each flow uses the existing executor with four default internal Euler
substeps and held pressure/capacity. The report binds actual endpoints to
that executor's evidence, retains full materialized event records, checks
the P2 family and domain at boundaries, and records diagnostics. It separates
three signed endpoint contributions:

1. The actual represented UM capacity gap versus `rho*a`, propagated through
   the exact nodal endpoint map.
2. Fresh binary64 pressure versus the exact modeled pressure at the actual
   post-event capacity and EPI.
3. The integrator endpoint versus the exact held stored-pressure update.

Their exact sum is the actual endpoint minus the ideal cycle endpoint.
All three contributions are nonzero in this run; none is rounded away to
claim exact realization. Existing event and forced-step balances retain
metric changes and pressure/execution effects on the mean. The exact-model
repetition envelope is reported separately; no uniform runtime error bound
is inferred from these finite defects or from finite domain membership.

A decisive production-kernel control prevents that promotion. Let
`nu0=1+2^-52`, `nu1=1`. The default `0<gamma<1/4` computes
`nu0 + gamma*(nu1-nu0)` as the same binary64 `nu0`. The positive capacity gap
is a represented fixed point, whereas the exact model would reduce it by
`gamma*2^-52`. Default UM is actually admitted and executed in this control,
and the capacity-supported relative profile remains nonzero. A positive
exact contraction coefficient therefore cannot be transferred to the
represented capacity gap. The control's actual EPI is still uniform: a
nonzero conditional frozen-profile mismatch is not proof of a positive
runtime EPI error floor. Small Euler increments can also round away. This
obstruction concerns the capacity map; it does not assert a complete fixed
EPI/phase/history runtime orbit.

The [runtime controls](../tests/physics/test_capacity_feedback_runtime.py) cover
matched preparations, the direct-word refusal, every UM/IL admission,
refreshed separator identity, executor inputs, finite box membership, all
three error sources, closure and the represented gap fixed point.
Reproduce the finite record with:

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/capacity_feedback.py
```

Its local ignored artifact is `artifacts/research/capacity_feedback.json`.
It carries its own source digest and runtime manifest; the earlier U/P/F
artifacts retain their historical source identities. The SHA metric
`time_to_collapse=+infinity` has an explicit `numeric_kind: positive_infinity`
JSON tag; nonfinite structural state or defects still fail strict encoding.
B2.d.11 closes this conditional joint recurrence, exact invariant box and
finite comparison. The following restricted arithmetic result resolves the
unit-capacity boundary without another long word. Full-state admission and
action selection remain open. No laboratory correspondence has been tested.

## 15. The represented capacity class on the unit interval

The same [capacity-feedback owner](../src/tnfr/physics/capacity_feedback.py)
now supplies `derive_p2_binary64_coupling_lattice` and its detached endpoint
observer. The production arithmetic has one shared owner,
`coupling_capacity_blend` in the
[Coupling kernel](../src/tnfr/operators/_coupling_stage_kernel.py). Extraction
preserves the existing neighbor mean, subtraction, multiplication and sum
in their original order. The
[shared platform probe](../src/tnfr/_binary64.py) also retains REMESH's former
private alias; basic format probes are not an exhaustive hardware proof.

Assume IEEE binary64 nearest-even arithmetic with separate operations, fixed
neighbor capacity `c=1`, target capacity `1<=v<=2`, and a fixed represented
factor `g`. Write

$$
\delta=2^{-52},\quad u=2^{-53},\quad v=1+j\delta,
\quad 0\le j\le2^{52},\qquad
T(v)=\operatorname{RN}\!\left(v+
\operatorname{RN}\!\left(g\operatorname{RN}(1-v)\right)\right).
$$

The executable reference verifies the strict inequalities

$$
6g(1+u)<\tfrac12,\qquad 7g(1-u)>\tfrac12,\qquad
g<\tfrac18,\qquad g(1-u)>\tfrac5{64}.
$$

The actual default factor
`g=5734161139222659/72057594037927936` satisfies them. They define a proved
coefficient class containing that default; they do not cover every allowed
UM factor. `5/64` is a rational enclosure used in the proof, not a new
coupling setting. The fixed unit interval is a numerical domain restriction,
not a physical frequency scale or a scaling theorem for subnormal capacities.

Subtraction is exact on this interval, and every positive product is normal.
With `m=RN(g*j*delta)/delta`, the relative product bound gives
`g(1-u)j<=m<=g(1+u)j`. Final rounding changes the real index `j-m` by at most
one half. Hence indices `0,...,6` are fixed, and every index at least seven
strictly decreases. To rule out skipping the terminal index, `g<1/8` and
representability of `j*delta/8` give `m<=j/8` by monotonic rounding.
For `j>=8` the output index is at least seven; `j=7` maps to six. These
arguments include the upper endpoint `v=2`. Halfway cases round the final
capacity to even; rounding a decrement independently is not equivalent.

The map preserves `[1,v]`. Its terminal index is therefore

$$
j_\infty=\min(j_0,6).
$$

There is also a finite analytic bound. For `r=59/64`,

$$
j_n\le\frac{32}{5}+r^n\left(j_0-\frac{32}{5}\right),\qquad
59^{512}(5\cdot2^{52}-32)<3\cdot64^{512}.
$$

The integer inequality puts the upper envelope below seven after 512
numeric updates for every initial index in the class. Together with the
no-skipping result, this proves arrival at six from above. It does not
execute 512 graph cycles or admit a repeated UM word; the grammar refusal
from section 14 remains unchanged. The result concerns only this fixed
numeric capacity map. The
[42 lattice controls](../tests/physics/test_binary64_capacity_feedback.py)
cover independent endpoints, both directions of final halfway rounding,
factor/platform restrictions, cache rebuilding and fabricated endpoint refusal.

In the section 12 capacity-forcing family with independent uniform comparison
metric `H0=I`, the terminal exact frozen-profile mismatch is

$$
R_*^{\rm terminal}=\frac{k^2j_\infty^2\delta^2}{4}
\le9k^2\delta^2.
$$

Equality holds for initial indices above six. This is a conditional profile
calculation after holding the terminal capacities, not a positive lower bound
on the EPI that the binary64 integrator will actually store.

## 16. Uniform stored EPI despite a nonzero exact profile

The [finite boundary benchmark](../benchmarks/binary64_capacity_feedback.py)
prepares five independent unit P2 graphs with zero phase, EPI `(0.5,0.5)` and
capacity `(1+j*2^-52,1)`, for the declared indices `0,1,6,7,2^52`. Each graph
executes one admitted default UM, one canonically refreshed Euler interval
of duration `0.25` with four held-input internal substeps, and SHA after
measurement. The uniform comparison reference is prepared independently;
it is not an invented ancestor or evidence of a prior perturbation.

| Input index | Observed post-UM index | Exact held-profile `R_*` | Observed post-flow `R0` |
|-------------|-----------------------|--------------------------|-------------------------|
| `0` | `0` | `0` | `0` |
| `1` | `1` | `1.248880010e-33` | `0` |
| `6` | `6` | `4.495968036e-32` | `0` |
| `7` | `6` | `4.495968036e-32` | `0` |
| `2^52` | `4145214556169080` | `0.02145926008` | `0.0003837216997` |

For indices one and six, UM leaves a positive gap fixed. From seven it
reaches six in the observed event. All three then retain EPI exactly
`(0.5,0.5)` through the actual interval, although their refreshed nodal
pressures and exact frozen profiles are nonzero. At index six the exact
model predicts a total EPI change of about `(-1.94e-17,+1.94e-17)`.
The represented pressure contribution plus the integrator endpoint defect
cancel that model change exactly. These errors are kept as signed rational
values, not discarded with a tolerance. Four smaller stored additions also
remain below the local EPI rounding thresholds.

There is no conflict between the two observations: the exact pressure model
and finite-precision state storage are different maps. An unchanged EPI
array is insufficient to infer zero structural pressure. Nor does unchanged
primary EPI/capacity certify a full fixed runtime state: time, derivatives,
histories and the later SHA closure have their own effects. The upper-boundary
control does change EPI, ruling out a blanket classification of every case
as numerical stasis. All cases retain actual gates, factor values, family
checks, executor endpoint bindings, diagnostics and separate mean/defect budgets.

The [15 runtime controls](../tests/physics/test_binary64_capacity_feedback_runtime.py)
also verify strict JSON retention of the tiny nonzero profile mismatch.
Generate the local artifact with:

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/binary64_capacity_feedback.py
```

`artifacts/research/binary64_capacity_feedback.json` has its own source
manifest. B2.d.11 and earlier artifacts remain historical; their source
digests are not rewritten after the shared-kernel extraction. B2.d.12 closes
this bounded numeric class and its finite EPI distinction, without generalizing
to other capacity scales, kernels or complete-runtime invariant domains.

## 17. Compatible capacity support and differentiated prepared regions

B2.d.13 addresses that structural question using the existing U3 gate and
pressure law. Let `C` be the simple, reciprocal graph of neighbors actually
selected by UM at the start of one simultaneous all-target stage. Every row
must be nonempty. With its unweighted degree matrix `D_C`, unit-conductance
Laplacian `B_C` and fixed factor `0<gamma<1`, the exact capacity proposal is

```text
nu' = nu - gamma*D_C^-1*B_C*nu.
```

Thus `nu'=nu` if and only if capacity is constant on each connected component
of `C`. Distinct components can retain distinct positive constants. Each
component preserves its own `D_C`-weighted capacity mean. These are properties
of the configured UM map; the factorization of the nodal equation alone does
not select that map or its coefficient. The result concerns simultaneous
snapshot proposals, not a sequence of single-target updates.

There is a strict exact energy balance away from that componentwise fixed set.
For `E_C(nu)=nu^T*B_C*nu/2`, `p=D_C^-1*B_C*nu` and
`S=(B_C*nu)^T*D_C^-1*(B_C*nu)`,

```text
E_C(nu') - E_C(nu) = -gamma*S + gamma^2*E_C(p)
                   <= -gamma*(1-gamma)*S.
```

The inequality follows edgewise from `(p_i-p_j)^2<=2*(p_i^2+p_j^2)`, which
gives `E_C(p)<=p^T*D_C*p=S`. Positivity is preserved by convex averaging.
On a fixed finite `C`, nonconstant normalized-Laplacian modes have multipliers
`1-gamma*lambda`, of magnitude below one. Conditional repeated exact capacity
updates therefore approach each component's conserved mean. Changing phases,
new links, represented arithmetic and live grammar are outside that argument.

The common readout is
[`observe_coupling_support`](../src/tnfr/physics/coupling_support.py); its detached
`derive_compatible_capacity_balance` reuses the existing transport energy and
Laplacian kernels. The live reader uses the production phase selector and
resolved factors, retains actual neighbor order, and checks reciprocity of the
materialized rows. It includes zero-weight support edges in U3 selection even
though they carry no EPI conductance. Empty selected rows are recorded as
blocked targets with no all-target balance, never as admitted identity updates.
Full operator admission remains a separate obligation.

### A connected pressure graph with two compatible components

The finite control prepares unit triangles `(0,1,2)` and `(3,4,5)` joined by
edge `(2,3)`. EPI is initially `0.5` everywhere and capacities are
`(1,1,1,2,2,2)`. In the split case, the first triangle has phase zero and the
second the binary64 value `math.pi`. These phases and capacities are declared
preparation; their formation has not been derived.

The actual U3 selection excludes the bridge and all candidate cross-links.
Both selected components are complete triangles, so no compatible missing
links remain. One actual default all-target UM preserves phase, capacity,
EPI and support. Its raw stored-pressure attenuation is recorded separately;
canonical refresh restores the same pressure at unchanged primary inputs.
The capacity energy on `C` is zero while that on the connected full pressure
graph is `1/2`. Using one graph for both operations would miss this distinction.

The captured canonical phasor channel is exactly zero at these represented
inputs on the tested platform. This is verified from its actual realization,
not inferred merely from regional synchronization or a symbolic `pi` label.
The capacity gradient is `(0,0,1/3,-1/3,0,0)` and the topology weight is zero.
With the captured `k=w_vf/e`, the existing held-profile theorem gives

```text
H = diag(2,2,3,3/2,1,1),     mean_H(nu) = 4/3,
z = (k/3,k/3,k/3,-2*k/3,-2*k/3,-2*k/3),
x_infinity = 1/2 + z,       mean drift = 0.
```

For fixed coefficients, `y=x+k*nu` obeys EPI diffusion. The initial range
`[1/2+k,1/2+2*k]` is convex-invariant under refreshed exact Euler steps
`h<=1/(e*max(nu))`. It yields `x_A in [1/2,1/2+k]` and
`x_B in [1/2-k,1/2]`, inside the configured scalar interval at default `k`.
These are conditional exact-model bounds, not a binary64 runtime invariant
class or a guarantee of future word admission.

### Finite runtime discriminator

[`compatible_capacity_regions.py`](../benchmarks/compatible_capacity_regions.py)
executes one all-target UM, one physical interval `h=0.25` with the shared
default integrator's four held-pressure substeps, and an all-target SHA closure
after measurement. Both full-word validators, actual target admissions,
two-phase stage results, histories, raw/ refreshed pressure, event reference
reset, signed flow defects and executor endpoints are retained. An independently
reproduced UM proposal is explicitly labeled as a prediction and checked against
the actual writes; the executor does not retain those UM proposals itself.

| Prepared case | New links | Post-flow EPI, rounded for display | Original-profile error, before -> after |
|---------------|-----------|------------------------------------|----------------------------------------|
| Split `0/math.pi` | 0 | `(0.5,0.5,0.504858,0.490283,0.5,0.5)` | `0.118208 -> 0.113675` |
| Aligned zero | 8 | `(0.508745,0.508745,0.508501,0.483658,0.482510,0.482510)` | `0.118208 -> 0.099933` |

The split case's conditional exact limit has EPI approximately `0.606103`
in the first region and `0.287793` in the second. Only the displayed finite
interval is executed; no numerical approach to that limit is claimed.

The aligned control starts with the identical graph, EPI and capacity but
phases zero everywhere. Its first capacity proposals use the original
neighbors: only bridge endpoints change, with exact neighbor means `4/3`
and `5/3` before binary64 rounding. Default functional links then add all eight
missing edges, producing `K6`. This is a phase-controlled complete operator
comparison with an observed topology change, not a fixed-topology ablation or
evidence that one capacity mechanism alone causes the error difference.
Both runs keep the original comparison profile; a changed current reference
is separately accounted for.

The mathematical and finite runtime controls live in
[`test_coupling_support.py`](../tests/physics/test_coupling_support.py) and
[`test_compatible_capacity_regions_runtime.py`](../tests/physics/test_compatible_capacity_regions_runtime.py).
Generate the separately manifested local artifact with:

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/compatible_capacity_regions.py
```

This closes a scoped maintenance discriminator: prepared phase compatibility
can protect a capacity contrast while the connected nodal pressure law drives
a differentiated EPI profile. Autonomous preparation, repeated complete-runtime
maintenance and physical correspondence remain open. In particular, the P2
UM/IL separator result cannot be transferred here: IL's phase-lock neighborhood
is not UM's U3-selected neighborhood and includes the bridge. The next dependency
is to derive which phase/support conditions survive a complete admitted policy,
then connect its preparation to existing birth/attachment or winding mechanisms.
No longer horizon, grammar change or fitted restoring pressure supplies that proof.

## 18. Antipodal phase response under simultaneous UM and IL

B2.d.14 tests that next dependency. The exact phase fixed configuration from
section 17 has a locally expanding mode when followed by the candidate IL
separator. This is a limitation of this proposed maintenance mechanism, not a
violation of IL's pressure-contraction contract or a universal impossibility
result for differentiated TNFR structures.

### Exact local composition

Use lifted coordinates `(a,a,b,pi-b,pi-a,pi-a)` on the same two triangles and
bridge. The base is the mathematical antipodal state `a=b=0`. In a sufficiently
small neighborhood, U3 selects only the two triangles, their internal links
already exist, and every missing cross-link remains incompatible. All relevant
phasor resultants are nonzero. Reflection together with phase conjugation and
the within-region interior swaps preserve this two-coordinate chart locally;
this symmetry does not prove an invariant neighborhood under repetition.

Let `t=UM_theta_push` and `alpha` be IL's phase-lock coefficient. Every
bidirectional UM target reads the entire compatible triangle, including itself.
Its exact phasor mean is `C=Arg(2*exp(i*a)+exp(i*b))`. Each receiving node gets
three identical proposals, whose shortest-arc displacements are **averaged**.
The derivative is consequently

```text
       [ 1-t/3     t/3   ]
U(t) = [                 ].
       [ 2*t/3   1-2*t/3 ]
```

IL includes the bridge in its full neighborhood. The interior phasor mean has
linear part `(a+b)/2`. The first bridge vertex sees
`2*exp(i*a)-exp(-i*b)`, whose argument has linear part `2*a+b`. Therefore

```text
           [ 1-alpha/2   alpha/2 ]
M(alpha) = [                    ],       P = M(alpha)*U(t).
           [ 2*alpha        1    ]

det(I-P) = -alpha^2*(1-t) - 2*alpha*t/3.
```

For every `0<alpha<=1` and `0<=t<=1`, this determinant is negative. The real
monic characteristic polynomial evaluated at one is negative and tends to
positive infinity, so `P` has a real eigenvalue strictly above one. Thus even
the preceding UM averaging cannot make this exact local phase fixed point
attracting. This is an analytic characteristic-polynomial certificate, not an
eigenvalue estimated from the finite probes. At `alpha=0`, the constant
two-coordinate direction is neutral. The algebraic control `t=0` is excluded
by canonical UM admission; no runtime coefficient is changed to obtain it.

For the predeclared direction `a=b=epsilon`, UM is exact-real identity and
the first-order IL response is `(epsilon,(1+2*alpha)*epsilon)`. The quadratic
readout `Q=| (a,a,b,-b,-a,-a) |^2/2 = 2*a^2+b^2` therefore has linearized gain
`1+4*alpha*(1+alpha)/3`, about `1.52` at the default coefficient. `Q` measures
phase deviation from this base; it is neither EPI energy nor a claimed global
Lyapunov function. Individual other directions can contract despite the
existence of an expanding mode.

The exact reference and tangent observer extend the existing
[`coupling_support.py`](../src/tnfr/physics/coupling_support.py) owner. They rebuild
public reference caches, preserve exact or represented coefficient values and
infer no graph or execution provenance. The unchanged IL default `0.3` now has
one owner in
[`_coherence_stage_kernel.py`](../src/tnfr/operators/_coherence_stage_kernel.py),
shared by direct validation, direct proposals and simultaneous stages. This
removes repeated defaults without claiming a new derivation of that coefficient.

### Finite controls and the pressure distinction

[`antipodal_region_phase_response.py`](../benchmarks/antipodal_region_phase_response.py)
predeclares `epsilon=0,+/-2^-12,+/-2^-16`, regional phases
`epsilon` and `pi-epsilon`, and otherwise the same preparation as section 17.
Each case executes one all-target UM, refresh, one all-target IL, refresh, one
`h=0.25` nodal interval and an all-target SHA closure after measurement. Both
word validators, live admissions, an independent strict IL readiness check,
actual two-phase stages, histories, metrics and proposal/endpoint comparisons
are retained. Independent proposals are labeled predictions, not sealed
executor-retained objects.

| Prepared epsilon | Observed post-IL bridge tangent, approximately | Observed phase Q gain |
|------------------|----------------------------------------------|-----------------------|
| `+2^-12` | `+0.000390624965076` | `1.519999847413` |
| `-2^-12` | `-0.000390624965076` | `1.519999847413` |
| `+2^-16` | `+0.000024414062492` | `1.519999999411` |
| `-2^-16` | `-0.000024414062492` | `1.519999999411` |

The exact local scalar response on this direction is
`epsilon+alpha*(atan(3*tan(epsilon))-epsilon)`. The runtime controls compare
against that independently evaluated expression as well as the linear model.
Their observed-minus-linear residual includes nonlinear, binary64 and chart
effects; two magnitudes are not a numerical proof of an asymptotic derivative.

In the zero-input control, IL writes approximately `3.67e-17` to node 2's
phase. This is a retained represented-arithmetic residue, not exact antipodal
invariance; the initial zero phase energy makes its gain undefined (`None`).
Its refreshed represented phase-pressure channel nevertheless remains zero.
The symbolic base uses mathematical `pi`, not the stored value `math.pi`.

In all four nonzero controls, IL increases canonical `C(t)` immediately by
contracting stored pressure. After canonical refresh, `C(t)` instead falls
below its pre-IL value because the changed phases produce new pressure. For
`epsilon=+2^-12`, the three values are approximately
`0.99352517 -> 0.99508085 -> 0.99351353`. This is consistent with the local
operator contract: refresh is a separate pressure realization. Neither raw
`C(t)` improvement nor preserved capacity proves preserved phase structure.
Capacities, EPI and support remain unchanged through both stages; the following
nodal interval changes EPI while holding the captured phase/capacity/support
inputs fixed. Source decomposition and signed event/flow budgets retain that
connection to `dEPI/dt=nu_f*DeltaNFR`.

The [exact controls](../tests/physics/test_coupling_support.py) independently
derive the six-node source/receiver tangent before comparing the reduction.
The [runtime controls](../tests/physics/test_antipodal_region_phase_response_runtime.py)
check both signs, scalar response, null residue, actual admission, shared
defaults, coherence scopes, flow endpoints and strict artifact serialization.
Generate the separately manifested local artifact with:

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/antipodal_region_phase_response.py
```

### Consequence for the next structural test

The mean-of-phasors derivative has entries
`R_ij=cos(theta_j-m_i)/|sum_neighbors exp(i*theta)|` for neighbors,
where `m_i` is their mean argument. Its rows sum to one but need not be
nonnegative: the antipodal bridge has coefficients `(1,1,-1)`. Treating this
response as ordinary positive diffusion would erase the expanding mode.
The existing phase-pressure, winding and coupling-support work should share
this local response description before another maintenance policy is proposed.

The uniform-winding `C6` comparator under the default all-target bidirectional
UM merge is now delivered in
[the cycle/winding owner](COUPLING_WINDING_PERSISTENCE.md#8-one-shared-local-response-for-circular-means).
The shared exact Gram/receiver response preserves the antipodal obstruction
and gives C6 strict centered tangent contraction plus a separate exact
phase-map invariant neighborhood. Its five finite default controls retain
winding and nodal response. The same owner now supplies nonlinear phase
oscillation contraction and a conditional exact joint phase/capacity/EPI
reserve. In its strict Euler regime, EPI tends to its initial mean while
phase approaches a rotated winding. Three finite two-cycle controls audit
admission; the null retains a numerical phase residue that prevents a
defect-free transfer of the homogeneous contraction bound. The additive
extension now accounts for those retained errors with finite signed reserve
and mean budgets. Its separate uniform range bound requires independently
justified future error and signed mean-prefix hypotheses; it does not turn
the six observed intervals into a runtime invariant class. The held Euler
arithmetic now has an independent local rounding bound; exact addition cells
explain the retained null EPI stasis despite nonzero pressure and changing
phase. A zero-sum held-input fixture still produces mean drift. Those results
resolve local integration arithmetic while leaving phase/pressure accuracy
and uniform signed mean control open. The new opposite-pair continuation
derives a finite nonlinear phase/EPI partial quotient and a distinct
mean-preserving binary64 pure-channel class. The retained full winding
execution fails its paired-pressure and common-binade premises, localizing
the next question to joint pressure-pair leakage and compatible rounding
cells. A certified two-neighbor midpoint now centralizes the eligible IL
and CPU phase-pressure realizations, correcting a retained phasor rounding
error. The same finite comparison improves pressure-mean cancellation
while retaining nonzero k1 EPI mean drift. Local accuracy does not supply
a future signed-mean bound. Exact inverse pressure boxes now show that
balanced pressures reproduce the same biased k1 trace, and a two-binade
law explains that bias at EPI `0.5`. This rules out a pressure-only repair
inside those boxes. A separate explicit remainder-carrying nodal reference
now bounds executor mean error by the terminal rounding remainder,
independently of the supplied finite step count. It preserves reconstructed
mean under balanced nodal area; pressure-realization error still requires
its own bound. Graph adoption requires explicit pressure-readout,
operator/reset, history and certificate contracts (section 20). The
restricted flow/event implementation supplies those contracts for
EPI-preserving UM/IL/SHA with live visible-pressure refresh, persistent carry
and whole-graph rollback (section 21). Its matched finite branch comparison
separates actual source changes from executor rounding. Broader jumps and
generated-pressure summability remain open. Section 22 now excludes an
exact binary64 zero-pressure state throughout the EPI band for one retained
fixed phase slice. Its carried-cell horizon separates a visible-state
change from a band exit. Joint UM/IL trapping and signed source bounds
remain open; fixed-phase nonconvergence does not decide them. The detailed
phase continuation in section 23 reaches an exactly closed default UM/IL
phase map from the inherited null, with a nonzero periodic source mean.
The signed compensation required from refreshed EPI arithmetic is now
explicit; phase closure alone does not provide it. The full
local analysis in section 24 gives exact generated compensation on one
finite cell prefix and its loss at the first refresh after a cell exit.
Independent-coordinate traps and permanent residence in the 13-state
control class are excluded under the declared per-step conditions;
correlated trapping and full-state reachability remain open. The
derivation is kept there rather than duplicated in this feedback note.
Section 25 adds exact whole-itinerary carry compatibility: a feasible
closed visible word is a carried cycle only when its total nodal area
vanishes coordinatewise. Two derived cell boundaries show canonical
pressure-mean sign reversals, while their carried path remains open.
The four observed states retain strict negative node-1 pressure, excluding
a carried cycle confined to them despite the changing mean source.
Section 26 reaches positive node-1 pressure through a neighboring EPI
transition with the inherited carry intact. It centralizes bounded
cell-exit replay, derives uniform strict-sector residence bounds and
separates index change into nodal area and carry transfer. The first
positive-pressure hit still has negative accumulated node-1 area;
section 27 follows its two positive steps to the next cell boundary,
where pressure becomes negative again before compensating that area.
The exact affine budget separates this observed prefix from a hypothetical
four-step crossing. A two-level integer constraint rules out short exact
return at the fixed step and capacity, without excluding a broader bounded
class. Additional canonically reachable pressure levels and correlated
trapping must pass the shared six-coordinate budget and carry checks.
Section 28 reaches a third node-1 pressure in eleven steps. Node 1
overshoots its earlier loss while the complete vector remains unbalanced.
The three-level scalar length constraint is unchanged. Independently,
node 4 retains a positive pressure across the three observed visible
states, giving a uniform 43-step residence bound with the shared finite
drift observer. A larger candidate class must change that drift or fail
its confinement test, rather than rely on the recovered node-1 balance.
Section 29 makes the frozen-stencil bound executable. A neighbor changes
at step 15, ahead of the retained-carry center deadline 21, and lowers
node 4's pressure while leaving it positive. The declared eight-boundary
sign test ends censored at eighteen steps. The remaining exact gradient
cut and the full nodal-area identity now constrain the next coupled
reachability argument. The shared gradient observer rejects uniform area
errors that would be invisible to a Laplacian-only comparison.
This supplies a return-class discriminator, not a longer live campaign
or a stability conclusion; existential carry and reachability stay distinct.

[Section 30](COUPLING_WINDING_PERSISTENCE.md#30-a-coupled-profile-disagreement-tube-and-finite-numerical-band-horizon)
completes B2.d.32-B2.d.36
through a canonical centered reference, a carried-step identity, exact
spatial contraction, a uniform numerical-defect tube and a separate finite
band budget. It reuses the forced-support Poisson solver for
`w_epi*L*z=A-mean(A)*1` and verifies
`y_next=(I-h*w_epi*L)*y+h*P*(w_epi*L*r+eta)` on all eighteen retained
steps, where `P` removes the arithmetic mean, `y=P*X-z`, `X=x+r`, and
`eta` contains the actual EPI-product and channel-assembly errors.
At the unchanged default weight and `h=1/16`,
the exact centered norm factor is `573161353023261791/2^59`; its square
is the homogeneous energy factor. The uniform rounded-defect calculation
gives the energy envelope a floor near `6.6560e-31` on `[3/8,5/8]`. Combining the
spatial envelope and a distinct absolute mean bound proves a sufficient
conditional numerical-band horizon of `196713720348826219` steps without
executing them. This is a fixed-source numerical-map result, not a
live phase/grammar or future-runtime certificate.

The retained eighteen-step mean change is exactly `-131/(3*2^114)`,
split into phase source `-3/2^113`, product/assembly rounding
`-113/(3*2^114)` and zero carry feedback. That first shape envelope
left the indefinite signed mean budget and node 4's nonpositive-pressure
cut unresolved.

B2.d.37-B2.d.40 now connect those questions through the same canonical
pressure, carried-state and forced-profile owners. The
[self-consistent envelope](COUPLING_WINDING_PERSISTENCE.md#31-b37-closing-the-spatial-and-rounding-bounds-on-each-other)
uses its own spatial restriction to close the rounding bound, giving
an energy floor approximately `2.958228394578814e-31` and a uniform
per-node product/assembly error bound `6.731922543446726e-32`.
The
[static seven-point certificate](COUPLING_WINDING_PERSISTENCE.md#32-b38-static-compensation-refutes-a-class-wide-linear-drift-argument)
admits seven states at exactly the retained endpoint's reconstructed mean
and verifies a positive rational combination of their full pressure vectors
equal to zero. This refutes a strictly signed fixed linear drift on that
enclosed class; its algebraic coefficients do not define an executable
switching rule or prove any of those points reachable.

The
[finite passage theorem](COUPLING_WINDING_PERSISTENCE.md#33-b39-a-finite-first-passage-theorem-for-the-pressure-cut)
shows that strictly positive node-4 pressure cannot persist through all
first `30255` transitions: the derived centered growth would leave the
spatial envelope while the independent finite band guarantee still holds.
Thus a nonpositive readout must occur at an index no greater than `30254`,
starting from the B31 endpoint. The
[decisive continuation](COUPLING_WINDING_PERSISTENCE.md#34-b40-the-realized-first-passage-and-a-finite-mean-budget-reversal)
with its inherited carry reaches the first hit in 118 steps and 59 visible
cell boundaries. Readouts `0..117` are positive; the endpoint has gradient
index `-22` and pressure `-187043320717485/2^105`.

The same finite run repays the B27 mean deficit by overshoot at step 59:
its net mean area changes from `-1/(3*2^114)` to `1/2^113`.
The final local mean area is `41/2^112`, split into phase source
`-59/(3*2^113)`, product/assembly rounding `305/(3*2^113)` and zero
carry feedback. Relative to B27 the final mean area is
`217/(3*2^114)`, while every net coordinate area remains nonzero.
This closes the declared finite sign and deficit-crossing questions,
without an exact vector return or an infinite compensation theorem.

B2.d.41 adds the
[complete-cell obstruction](COUPLING_WINDING_PERSISTENCE.md#35-b41-complete-carry-cells-cannot-provide-invariant-trapping):
any nonempty finite family in B37's gradient class fails forward invariance
when each visible tuple admits its entire legal carry fiber. The theorem
constructs an outward hypothetical carry; it does not continue or establish
escape of the retained state. B2.d.42's stronger claim for B38's particular
seven displayed rows comes from their
[exact temporal graph](COUPLING_WINDING_PERSISTENCE.md#36-b42-every-carry-leaves-the-seven-static-compensation-cells).
Only self-loops and the zero-based edge `5 -> 4` are feasible. Every legal
incoming carry leaves that family or fails band admission within 77 steps.
Whole-band exit, entry from B40 and a general recurrence theorem are not
implied. The static convex pressure balance therefore remains an algebraic
certificate whose seven cells cannot themselves form a closed temporal class.

B2.d.43's
[finite live bridge](COUPLING_WINDING_PERSISTENCE.md#37-b43-a-finite-live-bridge-from-the-original-winding-preparation)
now executes the original null C6 preparation through 89 UM/IL pairs and
terminal SHA, retaining 179 events and 356 pressure-refreshed carried flows.
It reaches the same fixed phase source as B40 with its own actual EPI and
carry: tail-entry offsets `(-3,0,2,-1,6,-5)` at time `22`, and offsets
`(-3,0,2,0,8,-5)` immediately before SHA at `22.25`, all in units `2^-54`
from `.5`. Every flow passes independent replay; all six nodal-balance
residuals vanish and the accumulated mean area is `3293/(3*2^114)`.
No detached remainder is imported or reset. SHA then changes capacity from
one to `0.9204225284540524`, so the observed unit-capacity premise cannot be
extended to a later live invocation. The finite bridge closes for this
declared carried solver; it does not authenticate B27-B40's different
historical detached state or prove indefinite trapping.

The next gate is forward inclusion of carry subcells restricted jointly
with visible shape and signed accumulated vector area, or a recurrent-budget
obstruction. Use B43's actual tail carry and its closure for that executed
branch, retaining the fixed-map conditions explicitly; a B40 analysis must
remain identified as a separate detached branch. Enlarging a complete-cell
family inside the same gradient class
cannot evade B41. Repeating the node-4 sign search or seeking a fixed strict
linear separator over the full certified class is also no longer the active
question. Detailed proofs and shared owners remain centralized in the winding
note; none of these blocks advances the saved B40 endpoint.
B2.d.44 now excludes the
[local energy tube with independent mean bounds](COUPLING_WINDING_PERSISTENCE.md#38-b44-a-bounded-mean-interval-does-not-close-the-centered-energy-tube)
around B43's actual pre-SHA mean. Opposite canonical mean increments and
exact carry-cell translations provide hypothetical outward points at both
boundaries. B2.d.45 preserves the
[coordinate arithmetic restriction as well](COUPLING_WINDING_PERSISTENCE.md#39-b45-the-coordinate-arithmetic-class-does-not-rescue-local-mean-confinement):
a derived affine carry lift keeps the requested mean and satisfies an analytic
energy bound in a slightly smaller local window. Neither result advances the
saved B43 state or proves escape of its actual trajectory. The next region
must couple mean boundaries to shape and individual carries; the independent
local mean interval is insufficient even after this arithmetic restriction.
The earlier target-only winding theorem remains distinct. Represented joint
invariance, autonomous preparation, physical correspondence, eventual
antipodal U3 crossing and complete-runtime long-time behavior remain open;
extending the failed nominal antipodal run is not the next test.
