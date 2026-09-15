# Child Coupling and changes of the structural reference

**Status:** Exact same-EPI reset budgets and a bounded canonical action.
Recovery of an earlier relative pattern requires a fixed comparison target.
**Research links:** B2.d.8/O3.a, S3, S8, S9, S10 and S16.

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

## 7. The next dependency is a genuine perturbation budget

A smaller current-model error does not by itself establish recovery
toward the fixed old profile. The next bounded comparison must first
separate an actual EPI perturbation from the accompanying changes of
metric, forcing and reference. The current same-EPI observer correctly
rejects endpoints with an EPI jump.

Default edge-aware Expansion can change signed EPI as well as capacity.
Default Coherence also has a phase update, while its stored-pressure
attenuation can disappear at the next canonical refresh. A proposed
VAL/IL preparation therefore requires live grammar checks and captured
writes to every affected nodal channel. It cannot be treated as a
capacity-only perturbation or a permanently retained pressure change.

A full-event budget can separate the EPI jump in the old fixed
coordinates from a change of reference at the resulting EPI. Such an
intermediate is an algebraic decomposition, not an asserted ordering
of writes inside the actual operator. Its implementation and independent
checks are a prerequisite to the next perturbation campaign.

The intended matched branches must share the admitted perturbation
prefix, pressure refreshes and elapsed time, with additional UM only
in the feedback branch and a separate unperturbed continuation. They
must retain a fixed comparison target alongside each current-model
error. This would test a finite recovery response rather than infer
one from a changing reference; it remains outside the present campaign.
