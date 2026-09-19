# Relative form and mean drift on a held canonical support

**Status:** Exact fixed-support balance and finite Euler error identities.
Runtime observations retain pressure and endpoint defects separately.
Section 7 adds exact instantaneous regional balances on the full support;
they do not assume that a selected region is already an autonomous NFR.
Sections 11-13 connect retained phase sensitivity to the source, integrate
an exact represented-component reduction and compare its regional effect.
Section 14 separates finite regional response from loss of control contrast.
**Research links:** B2.d.7/O3.a, S3, S8, S9 and S16.

The current structural results are section 22's source tangency, section 23's
capacity/phase independence and section 24's fixed-source phase geometry.
The latter separates a regular rigidity theorem from an exact flexible
family and its U3 boundary. Section 21's fixed-point theorem remains
conditional on its specified relaxation policies.

**Research status:** This note retains mathematical dependencies and historical
finite studies, not an execution queue. Prepared regional-response work is
parked; only the [current G3 gate](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
assigns the next scientific task.

## 1. The reference comes from the existing nodal channels

The connection studied in
[THOL birth and transport](THOL_BIRTH_AND_TRANSPORT.md) changes both the
weighted EPI walk and the unweighted capacity/phase neighborhoods.
Its next evolution can be analyzed by retaining that connected support,
positive capacities and phases, then refreshing the canonical pressure
before each declared physical step.

Let `W` be fixed symmetric nonnegative conductance, connected through its
positive off-diagonal entries. Let `d_i=sum_j W_ij>0`, `D=diag(d_i)`,
`B=D-W`, and `nu_i>0`. Loops follow the shared conductance convention and
zero-weight support edges contribute no EPI transport. The existing
normalized EPI channel coefficient is `e>0`. Write the held remaining
canonical channels as

$$
F=w_{\nu}g_{\nu}+w_{\phi}g_{\phi}
  +w_{\mathrm{topo}}g_{\mathrm{topo}}.
$$

Here `g_nu` and `g_topo` use the actual unique-neighbor support; `g_phi`
uses the canonical neighbor-phasor argument. Holding these source fields
and their channel coefficients makes `F` constant. It is a decomposition
of the existing pressure, with no additional feedback coefficient.
Its phase component need not be a linear phase Laplacian.

The nodal equation on this retained state is therefore

$$
\dot x=\operatorname{diag}(\nu)(-eD^{-1}Bx+F).
$$

An exact rational companion can treat finite materialized coefficients
as exact real inputs. A live pressure refresh generally differs from
this reference by a recorded numerical residual. Neither a stored
pressure value nor an operator label establishes that the other
channels have been independently identified or held fixed.

## 2. Compatibility, mean drift and the unique relative profile

Define the positive metric vector `h_i=d_i/nu_i`, its diagonal matrix
`H=diag(h_i)`, total weight `Z=sum_i h_i`, and weighted mean

$$
m(x)=\frac{h^\top x}{Z}.
$$

Symmetry gives `1^T B=0`, hence

$$
\frac{d}{dt}(h^\top x)=\mathbf1^\top DF=:b,
\qquad \dot m=\bar v:=b/Z.
$$

The initial mean is unrestricted. Its exact reference trajectory is
`m(t)=m(x0)+bar_v*t`; the mean is conserved only when `b=0`.

A zero-pressure stationary EPI exists precisely when

$$
b=\sum_i d_iF_i=0.
$$

Indeed its equation is `e*B*x=D*F`, and the range of a connected
symmetric Laplacian is the subspace orthogonal to `1`. Positive capacity
makes zero rate equivalent to zero pressure in this fixed model.
Zero capacity, disconnected positive conductance and changing channels
require different conclusions.

For every `F`, including `b!=0`, there is a unique centered relative
profile `z` satisfying

$$
eBz=DF-\bar v h,\qquad h^\top z=0.
$$

The right-hand side sums to zero by the definition of `bar_v`, so a
solution exists and is unique after fixing its weighted mean. A useful
exact computation is the nonsingular rational system

$$
(eB+hh^\top)z=DF-\bar v h.
$$

Multiplication by `1^T` enforces `h^T z=0`; the remaining equation is the
original profile equation. The rank-one term enforces the algebraic
choice of mean and is not part of the nodal pressure or evolution.
The existing exact matrix inverse suffices for this bounded solve.

Let

$$
A=e\operatorname{diag}(\nu_i/d_i)B,
\qquad u=x-m(x)\mathbf1-z.
$$

Then `h^T u=0` and direct substitution gives

$$
\dot u=-Au,
\qquad
x(t)=[m(x_0)+\bar v t]\mathbf1+z+e^{-At}u(0).
$$

This separates a retained spatial form, uniform mean drift and relaxing
deviations without adding an independent dynamical law. A nonzero `z`
alone does not establish localization or a zero-pressure equilibrium.
On the exact drifting profile the pressure is `bar_v/nu_i`, while
the raw Dirichlet energy stays at `z^T*B*z/2` because `B*1=0`.
A constant spatial energy therefore does not establish stationarity.

## 3. Exact relaxation and the finite-chart boundary

The identity `H*A=e*B` makes `A` self-adjoint in the positive `H` metric.
Its only zero mode on connected conductance is the uniform vector.
The centered error variance and Dirichlet error satisfy

$$
V(u)=\tfrac12u^\top Hu,
\qquad \dot V=-e\,u^\top Bu\leq0,
$$

$$
E(u)=\tfrac12u^\top Bu,
\qquad
\dot E=-e(Bu)^\top\operatorname{diag}(\nu_i/d_i)(Bu)\leq0.
$$

The positive nonuniform spectrum implies `u(t)->0` in the exact held
model. Thus the relative profile attracts every initial field after
matching its initial weighted mean. When `b!=0`, the complete EPI field
continues to drift and does not converge to a stationary state.

This asymptotic statement assumes an unrestricted scalar chart with the
same fixed coefficients for all times. If every EPI coordinate must stay
within finite bounds `[l,r]`, its positively weighted mean must also stay
in that interval. The ideal affine mean therefore cannot remain inside
the chart beyond

$$
t>\frac{r-m(x_0)}{\bar v}\quad(\bar v>0),
\qquad
t>\frac{m(x_0)-l}{|\bar v|}\quad(\bar v<0).
$$

Individual nodes may reach a bound earlier; the mean does not control
every coordinate. Even `b=0` does not ensure that the limiting profile
lies inside the configured chart. Clipping changes the endpoint map
and contributes a measurable defect. A clipped stationary endpoint
need not have zero canonical pressure.

## 4. Finite refreshed Euler observations and their exact defects

For one declared duration `dt>=0`, let `p` be the captured pressure
before integration and `x_plus` the observed endpoint. Define

$$
p_*(x)=-eD^{-1}Bx+F,
\qquad \varepsilon_p=p-p_*(x),
$$

$$
\delta_x=x_+-x-dt\operatorname{diag}(\nu)p,
\qquad
\xi=dt\operatorname{diag}(\nu)\varepsilon_p+\delta_x.
$$

The first defect measures pressure realization relative to the declared
held model. The second measures the implemented endpoint relative to
its actual held pressure, including arithmetic, clipping or a different
solver. These definitions give the exact finite identity

$$
x_+=(I-dtA)x+dt\operatorname{diag}(\nu)F+\xi.
$$

Consequently the mean budget is

$$
m(x_+)-m(x)-dt\bar v
=\frac{dt\,d^\top\varepsilon_p+h^\top\delta_x}{Z}.
$$

Neither defect is inferred from the other. Their signed mean
contributions can cancel, so a small mean error alone is insufficient
evidence of an accurate pressure realization or endpoint.

Let `P_H=I-1*h^T/Z`, `Q=I-dt*A`, and `chi=P_H*xi`. Centering the observed
endpoint by its own weighted mean gives

$$
u_+=Qu+\chi,
\qquad
\chi=dtP_H\operatorname{diag}(\nu)\varepsilon_p+P_H\delta_x.
$$

The exact Dirichlet error budget is

$$
\begin{aligned}
E(u_+)-E(u)
={}&-dt\,e(Bu)^\top\operatorname{diag}(\nu_i/d_i)(Bu)\\
 &+\tfrac12dt^2(Au)^\top B(Au)\\
 &+(BQu)^\top\chi+\tfrac12\chi^\top B\chi.
\end{aligned}
$$

The first term is the dissipative nodal contribution. The next is the
finite Euler correction. The remaining terms retain the signed
interaction with the observed defect and its quadratic contribution.
They must not be discarded merely because the underlying continuous
reference is dissipative.

Likewise the weighted variance has the exact identity

$$
V(u_+)-V(u)
=-dt\,e\,u^\top Bu+\tfrac12dt^2(Au)^\top H(Au)
 +(Qu)^\top H\chi+\tfrac12\chi^\top H\chi.
$$

Over a finite sequence with the same held model, each mean identity
and each energy identity telescopes. Centered errors also propagate
through the complete matrices `Q_k`, so defects need not stay in one
spatial eigenmode. These are finite observations; a future error bound
or a runtime asymptotic theorem needs additional uniform hypotheses.

For the ideal Euler map, the sufficient step condition
`dt*e*max(nu)<=1` places the spectrum of `dt*A` in `[0,2]`. Hence neither
`V` nor `E` increases when `chi=0`. Equality can preserve an alternating
mode on an even cycle; nonincrease is not automatically strict decay.
The observed defect terms remain necessary under this step condition.

## 5. Reuse, realization and the evidence boundary

[`forced_support.py`](../src/tnfr/physics/forced_support.py) supplies
`derive_forced_support_balance`, `observe_forced_support_state` and
`observe_forced_support_step`. They rebuild public cached fields before
using them and check the held node order, conductance, unique support
and capacities. Their detached records contain no execution seal.
The profile computation reuses
[`_exact_linear_algebra.py`](../src/tnfr/physics/_exact_linear_algebra.py).
It introduces no alternative EPI integrator. The finite error budget
reuses
[`support_transport.py`](../src/tnfr/physics/support_transport.py)
on detached error coordinates with the same conductance and capacity.
Its reference error pressure is `-e*D^-1*B*u`, and its endpoint defect is
exactly `chi`. This is an algebraic observation of transformed data;
the error coordinate is not written into live nodes or interpreted as
a newly created physical node.

The held non-EPI source must be obtained from the actual canonical
channels independently of a fitted pressure residual. In particular,
the nonlinear phase contribution must use the shared phasor kernel,
retain its represented coefficients and distinguish its realization
from the subsequent full multichannel pressure arithmetic. A direct
operator pressure write or a stale stored value belongs in the
pressure defect and must not silently redefine `F`.

[`capture_non_epi_forcing`](../src/tnfr/physics/forcing_realization.py)
materializes the unit phase channel with the existing fused NumPy
kernel, then combines its represented values with the exact support
capacity/topology gradients and the effective channel coefficients.
It reads the engine's actual cached weight mix without normalizing it
again. The fresh full-kernel pressure defect and the stored-minus-fresh
pressure residual are reported separately. Their sum is
`epsilon_p` in the finite budget above.

This bridge is restricted to the default NumPy, non-JIT pressure branch
with at most 100 directed unique-support entries. It rejects a disabled
NumPy branch or a custom pressure callback. Neighbor insertion order
and zero-weight phase-support edges are preserved. The materialized
phase values define exact reference coefficients; the observer does
not prove exact transcendental evaluation, identify another backend
or validate future graph states.

The bounded experiment begins with the causally born and canonically
connected child from the preceding block. It retains the resulting
support, capacities and phases during explicitly refreshed Euler
segments. A separately prepared compatible control uses uniform
capacity and aligned phase on the same support. A separate prepared
boundary control starts near a scalar EPI bound. These comparisons
distinguish nonzero forcing, homogeneous relaxation and clipping
without changing the birth history of the causal case.

The tests must reject a claimed zero-pressure profile when `b!=0`,
verify the profile equation and its centered gauge, and account for
both pressure and endpoint defects at every captured step. They must
also keep a clipped mean change distinct from `dt*bar_v` and preserve
all fixed-support assumptions. Agreement over the measured horizon
establishes neither spontaneous selection of the held fields nor
restoration under later structural events. Those require an admitted
policy that actually changes the support, capacities or phases.

## 6. Bounded executed comparison

[`forced_support_balance.py`](../benchmarks/forced_support_balance.py)
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
[`test_forced_support.py`](../tests/physics/test_forced_support.py)
include a hand-solvable heterogeneous pair, arbitrary initial means,
the profile's uniform nonzero rate, cancellation between mean defects,
clipping, an excessive Euler step and tampered public caches. The
nonlinear forcing capture has 16 independent checks in
[`test_forcing_realization.py`](../tests/physics/test_forcing_realization.py).
The 13 bounded campaign checks are in the
[runtime test suite](../tests/physics/test_forced_support_balance_runtime.py).

The [child-target Coupling follow-up](CHILD_COUPLING_FEEDBACK.md) adds exact
same-EPI reset budgets when capacity, conductance or forcing changes. It keeps
the original derived profile as a separate fixed comparison and continues
from the retained attached endpoint before its terminal Silence action.

## 7. A region and its environment on the same nodal support

The full-network balance does not by itself identify which mechanism supports
a particular region. For a fixed proper nonempty node set `R`, retain the
**full-graph** conductance, strengths `d_i`, positive capacities and `e>0`
from section 1. Connectivity is unnecessary for the identities below, but
every full strength must be positive. Removing the environment and
renormalizing an induced graph changes the model and is not this observation.

Let `h_i=d_i/nu_i`, `Z_R=sum_R h_i`,

$$
M_R=\sum_{i\in R}h_i x_i,\qquad m_R=M_R/Z_R,\qquad
z_i=x_i-m_R,\qquad V_R=\frac12\sum_{i\in R}h_i z_i^2.
$$

`M_R` is a weighted EPI total, with no identification as physical mass.
`V_R` measures internal EPI contrast about the region's own weighted mean;
neither quantity alone defines coherence, identity or biological activity.
For an edge oriented from the region to its complement, define the outward
EPI current `J_ij=W_ij*(x_i-x_j)`. Direct substitution of the nodal equation
and cancellation of opposite internal currents give

$$
\dot M_R=-e\sum_{i\in R,j\notin R}J_{ij}
              +\sum_{i\in R}d_iF_i.
$$

The complementary region receives the opposite cut current. Since
`sum_R h_i*z_i=0`, differentiating the regional mean contributes zero to
the fixed-metric variance derivative. Pairing internal edges yields

$$
\dot V_R=
-e\sum_{\{i,j\}\subset R}W_{ij}(x_i-x_j)^2
-e\sum_{i\in R,j\notin R}W_{ij}z_i(x_i-x_j)
+\sum_{i\in R}d_i z_iF_i.
$$

The first sum counts each internal undirected edge once. It is nonpositive.
The boundary and non-EPI source terms are signed: either can supply or remove
regional contrast. Loops contribute to `d_i` but carry zero EPI current.
Zero-weight support edges contribute no EPI current; their influence on the
other canonical channels remains in the independently captured `F`.

For stored pressure `p`, retain
`epsilon_i=p_i-(-e*(Bx)_i/d_i+F_i)`. The stored-rate balances add
`sum_R d_i*epsilon_i` and `sum_R d_i*z_i*epsilon_i`, respectively.
When a forcing observation is available, split this discrepancy into its
fresh-kernel arithmetic defect and stored-minus-fresh pressure residual.
Do not redefine `F` from the observed derivative to remove either term.
Linearity also separates the phase, capacity and topology contributions.

These identities use one fixed region and metric. If support or capacity
varies continuously, differentiation includes metric-variation terms; birth,
membership changes and discrete events require their own endpoint budgets.
An instantaneous equality is not a finite-time restoration or invariance
theorem. Even exactly balanced source and loss at one instant do not prove
that the conditions supplying that balance persist.

The single executable owner is
[`observe_regional_support_balance`](../src/tnfr/physics/support_transport.py).
It rebuilds snapshot caches, preserves the full source node order and accepts
an explicit ordered region plus an independently supplied source vector.
The observer checks four exact identities: model and stored-pressure rates
for both weighted total and variance. It does not evolve the graph or choose
its region. Public detached records carry no causal execution seal.

The independent
[`regional tests`](../tests/physics/test_regional_support_balance.py) include
a three-node path with `x=(0,1,5)`, unit edges, capacities `(1,2,1)`, `e=1`,
`F=0` and `R={0,1}`. Here `h_R=(1,1)`, `m_R=1/2`, internal dissipation is `1`,
boundary work is `2` and `dot V_R=1`: internal diffusion can coexist with
growing regional contrast. Complementary total fluxes cancel. This control
demonstrates why whole-network smoothing cannot replace a regional budget.

### Conditional child response is not an autonomous child region

For a selected child set `B` with no internal edges or self-loops, fixed
external EPI and the same held canonical source give, for each child,

$$
\dot x_i=-e\nu_i(x_i-x_i^*),\qquad
x_i^*=\frac{\sum_{j\notin B}W_{ij}x_j}{d_i}+\frac{F_i}{e}.
$$

The conditional relaxation rate is `e*nu_i`. This is the diagonal block of
the existing nodal generator, without an added restoring force. If the
environment changes, this conditional profile changes too and is not a
fixed future target. Dependence on the environment is consistent with a
relational NFR; the open question is how the coupled system generates and
maintains the relevant region and its supporting boundary conditions.

## 8. Retained THOL regional audit

[`thol_regional_balance_audit.py`](../benchmarks/thol_regional_balance_audit.py)
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
[`retained audit`](../tests/physics/test_thol_regional_balance_audit.py) has 19.
A combined 313-test validation covers those tests, the reused full-support,
forcing/target owners, grammar/contract/sequence witnesses and selector
symmetry. It passes with zero failures. No full-repository, laboratory,
infinite-time or autonomous-emergence verification is claimed.

## 9. Finite regional observation with a held nodal rate

On the same full support and positive metric as section 7, hold the **stored**
nodal rate `r_i=nu_i*p_i` for one declared duration `dt>=0`. For an observed
endpoint `x_plus`, let `y=x+dt*r` and `delta=x_plus-y`. Both are detached
algebraic coordinates; constructing them does not execute a nodal step.
Write `mean_R(v)=sum_R h_i*v_i/Z_R` and `v_c=v-mean_R(v)` on the region.
The exact finite identities are

$$
M_R(x_+)-M_R(x)=dt\,\dot M_R(x)+\sum_{i\in R}h_i\delta_i,
$$

$$
V_R(x_+)-V_R(x)=dt\,\dot V_R(x)
+\frac{dt^2}{2}\sum_{i\in R}h_i(r_c)_i^2
+\sum_{i\in R}h_i(y_c)_i(\delta_c)_i
+\frac12\sum_{i\in R}h_i(\delta_c)_i^2.
$$

The dotted terms use the stored-pressure balances of section 7, including
the independently identified canonical source and stored-minus-model
pressure discrepancy. That discrepancy may contain an intentional IL
pressure contraction; it is not automatically floating-point error.
The quadratic rate term is nonnegative. Endpoint discrepancy can contain
rounding, clipping or a different supplied endpoint and remains signed in
its linear cross term. Uniform endpoint shifts change the weighted total
but not centered variance. Changed capacity, support or region invalidates
this fixed-metric identity and requires additional budgets.

This identity supplies finite accounting, not solver authentication or a
future stability theorem. An offline runtime application must separately
bind the starting pressure, actual duration, integration boundary and later
events. In particular, a phase update after integration can change the
future canonical source while leaving the just-completed EPI increment
unchanged; it must not be retroactively substituted into the consumed rate.
The interruption checkpoint and frozen regional observation contract are in
the [single execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md).

The shared executable owner is
[`observe_regional_support_euler`](../src/tnfr/physics/support_transport.py).
It reuses the instantaneous regional observer, rebuilds both snapshots and
rejects changed node order, full conductance, support or capacity. Its
[`24 focused tests`](../tests/physics/test_regional_support_euler.py) cover
direct heterogeneous examples, a mean-only uniform shift, clipping, zero
duration, invalid metric/support changes and detached cache tampering.
Together with the 79 existing support/forcing tests, 103 tests pass.

### Regional contrast is not a universal coherence score

`V_R` measures dispersion about a region's current weighted mean. It is not
the canonical `C=1/(1+|DeltaNFR|+|dEPI|)`, nor error relative to a predeclared
nonuniform pattern. Boundary forcing can create a structured difference
between children while diffusion dissipates internal differences. Increasing
`V_R` therefore need not be instability, just as decreasing it need not be
restoration of a particular form. A finite signed budget identifies those
mechanisms without defining an NFR by one favorable scalar trend.

For example, when two independent children have the same fixed capacity,
their conditional equation from section 7 gives
`dot(x_i-x_j)=-e*nu*((x_i-x_j)-(x_i_star-x_j_star))` under held external
inputs. A nonzero target contrast can cause the squared contrast first to
decrease and then increase when the signed difference crosses zero. This
is an algebraic property of the stated stable scalar response, not a proof
that the actual moving environment follows the held-input model. Actual
IL pressure writes and finite-step defects must still be retained.

## 10. Retained temporal regional identity audit

[`thol_regional_identity_audit.py`](../benchmarks/thol_regional_identity_audit.py)
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
[`portable audit tests`](../tests/physics/test_thol_regional_identity_audit.py)
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

## 11. Phase-source relevance at a fixed regional state

To distinguish a phase-readout difference from a difference in the nodal
drive, hold `x`, `nu`, `W`, `e`, the channel weights and region fixed. Let two
declared phase arrays produce captured canonical sources `F_a` and `F_b`.
The exact source components from the shared forcing owner use represented
unit phase gradients; their trigonometric evaluation is not replaced by a
linearized phase model. With `Delta F=F_b-F_a`, the model pressure and nodal
rate differences are

$$
\Delta p_{\mathrm{model}}=\Delta F,\qquad
\Delta\dot x_{\mathrm{model}}=\operatorname{diag}(\nu)\Delta F.
$$

Since the state, region and metric have not changed, neither have their
mean, internal dissipation or boundary work. Section 7 immediately gives

$$
\Delta\dot M_R=\sum_{i\in R}d_i\Delta F_i,\qquad
\Delta\dot V_R=\sum_{i\in R}d_i z_i\Delta F_i.
$$

When only phase changes, `Delta F=Delta F_phase`; capacity and topology
source differences vanish. These identities reuse the full-graph regional
owner and the independently captured source. They do not reconstruct a
force from an endpoint derivative or introduce a new evolution law.

The fresh represented kernel also has a discrepancy `epsilon_kernel`
relative to this model, so `Delta p_fresh=Delta F+Delta epsilon_kernel`.
If the recorded stored pressure is retained, its actual nodal rate remains
unchanged in both detached readings. Its stored-minus-fresh residual changes
by `-Delta p_fresh`. Thus a changed candidate fresh pressure is not a changed
historical increment or an observed alternative future. A regional sum can
also vanish despite a nonzero pointwise source difference; retain both.

The [execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md) fixes two already
archived phase alternatives, the original endpoint and nine existing regions.
The baseline fresh capture must first reproduce the archived capture exactly.
The test concerns the relevance of an established enumeration ambiguity,
not another phase-coordination experiment or a maintenance theorem. An
exact matching baseline validates that retained input on the current numeric
path; it does not make a two-input comparison a uniform robustness bound.

### Retained two-phase comparison

[`thol_phase_source_relevance.py`](../benchmarks/thol_phase_source_relevance.py)
implements that fixed protocol. It authenticates the native, phase-audit and
regional-identity inputs, binds the complete original coordination boundary
and checks the saved label/enumeration projections. Saved output phases are
already aligned to original node identities. Both fresh captures use the
original endpoint node and neighbor order, with the retained cached channel
weights and admitted default NumPy pressure branch. The full baseline
capture reproduces the archive exactly before the alternative is evaluated.

The alternative phase array changes the phase-source component at seven of
sixteen nodes. Its largest absolute difference is approximately
`0.04914323880773078`; the largest difference in the fresh-kernel arithmetic
discrepancy is approximately `2.610120553818296e-17`. These are separate
observed quantities, not a universal floating-point error bound. Capacity
and topology contributions, regional metrics, boundary work and internal
dissipation remain fixed. All nine source-difference identities close exactly.

| Fixed region | Baseline model variance rate | Alternative model variance rate | Difference |
| --- | ---: | ---: | ---: |
| Pair 0 | `-0.399116639750073` | `-0.399116639750073` | exactly `0` |
| Pair 1 | `+0.007133454707669` | `+0.000771577106453` | `-0.006361877601216` |
| Pair 2 | `-0.400949399259614` | `-0.397283880240531` | `+0.003665519019083` |
| Pair 3 | `+0.000771577106453` | `-0.005590300494763` | `-0.006361877601216` |
| Pair 4 | `-0.399116639750073` | `-0.399116639750073` | `+7.997873694454377e-18` |
| Pair 5 | `+0.000771577106453` | `+0.000771577106453` | exactly `0` |
| Pair 6 | `-0.399116639750073` | `-0.399116639750073` | exactly `0` |
| Pair 7 | `+0.000771577106453` | `+0.000771577106453` | `-2.14527927667545e-17` |
| All actual children | `+0.005813679258035` | `+0.007568908697630` | `+0.001755229439595` |

The pair-3 model rate changes sign. The child cohort's model weighted-total
rate also changes, from approximately `1.0523570656281966` to
`1.1735385705832913`. Tiny nonzero differences are retained rather than
classified by a fitted tolerance. The complete captures preserve the other
weighted-total rates and every exact coefficient.

Consequently, source invariance under these two previously archived phase
outputs is **rejected in scope**. The known enumeration ambiguity can change
the modeled regional drive, not only its tetrad display. This comparison
does not rerun the coordination algorithm: it connects the earlier finite
ordering witness to fresh pressure at one fixed endpoint. All historical
stored nodal rates remain exactly unchanged, including the pair-3 stored
variance rate; the sign change above concerns the candidate fresh model.
It is not a new trajectory, an observed future transition or a verdict on
regional coherence. Source reproducibility remains an unmet prerequisite
for the proposed maintenance interpretation.

Validation passes **151 tests**: 43 new
[`source-relevance tests`](../tests/physics/test_thol_phase_source_relevance.py)
plus the reused forcing, regional, identity and phase tests. An independent
standard-library recount checks the saved source decomposition and all nine
regional rate pairs directly from primitive retained state and conductance;
175 exact checks pass without another kernel call. Both new Python files
pass flake8. The study itself uses exactly two forcing-capture calls, zero
coordination calls and zero native calls. It changes no engine evolution.

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/thol_phase_source_relevance.py
```

The three pinned input identities are retained in the output and the
[execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md). Output:
`artifacts/research/thol_phase_source_relevance_2026_09_18.json`, SHA-256
`62c4fb07068980acebc75b264f63d715e3a0d28bef6fcc5628441937c01ffb86`.
Working-source digest:
`sha256:f3f8609af8ff7033a98c9f0ac449ca4a1343ffece1aa73620340ab98e31e918c`.
Local validation: `artifacts/research/phase_source_relevance_validation_2026_09_18.json`.
The ignored inputs remain required for the retained run; portable tests use
declared synthetic records and do not regenerate the historical studies.

## 12. Exact reduction of represented phase components

The source sensitivity in section 11 motivates a shared arithmetic contract
before changing production phase coordination. Fix a finite nonempty list
of materialized binary64 component pairs `(c_j,s_j)`. These are supplied
numeric values, not claims that either component is an exact transcendental
cosine or sine. Define

$$
a=\sum_j c_j,\qquad b=\sum_j s_j,
$$

with exact dyadic arithmetic. The existing
[`exact_weighted_sum_ratio`](../src/tnfr/mathematics/_exact_weighted.py)
computes each sum with unit weights: binary64 denominators are powers of
two, so alignment to their largest denominator and integer addition preserve
every bit. Addition of integers is associative and commutative. Hence `a`,
`b` and their exact joint-zero predicate are independent of input permutation.
This is an algebraic argument for every admitted finite list; finite
permutation tests validate its implementation rather than prove it by sampling.

If `a=b=0`, the resultant has no direction. The readout must return an
explicit unavailable angle rather than call `atan2(0,0)`. An empty input is
a separate invalid request, not evidence of cancellation. For a nonzero
pair, set `m=max(|a|,|b|)>0` and retain the exact scaled pair `(a/m,b/m)`.
This common positive scaling preserves the exact direction. Both components
lie in `[-1,1]` and at least one equals `+1` or `-1`. Materializing each once
therefore avoids overflow and joint-zero underflow even when the unscaled
sum is outside the finite floating-point range.

Let `(u,v)` be the resulting binary64 pair and retain the exact defects
`u-a/m` and `v-b/m`. The minor component can underflow to signed zero;
that loss must remain visible. A numerical `atan2(v,u)` is now defined and
uses the same inputs under every permutation of the fixed component list.
That determinism is conditional on the same numeric runtime. It is not a
certificate of correctly rounded transcendental angle, a cross-library
identity, global gauge covariance or continuity near a vanishing resultant.
Canonicalizing an exactly zero component to positive zero also specifies
the numeric display at the negative-real-axis branch; it does not choose
a direction for an exactly zero pair.

There are two distinct zero questions. The sum of supplied rounded
components can vanish even when the exact-real sum
`sum exp(i*represented_theta_j)` does not, or conversely differ from the
ideal phase preparation. The existing rational resultant enclosures address
the represented-angle transcendental question separately. The exact
two-neighbor midpoint contract also keeps its own strict chart hypotheses;
it is not replaced by this finite component reduction.

The reducer is a mathematical readout. It introduces no phase gain,
alignment epsilon, external direction, pressure law or EPI update. Its
production use is a separate caller contract, documented below.
The [single execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md) owns the
implementation checkpoint and the subsequent integration boundary.

The implementation is
[`reduce_phasor_components`](../src/tnfr/mathematics/phasor_resultant.py),
returning a frozen `RepresentedPhasorResultant`. Its retained `components`
follow input order, so the complete record is not invariant under a
permutation; the derived sums, scale, materialized pair, defects and angle
are. Nonempty finite outer iterables are required, and generators are
consumed once. The shared finite-real materialization rejects booleans,
nonfinite values, overflow and nonzero input underflow; input signed zeros
are canonicalized to positive zero. The binary64 rounding precondition is
checked before consuming the input. No unit-norm assumption or trigonometric
evaluation is introduced. At cancellation, `joint_zero=True` and all
direction/materialization fields are `None`. This is distinct from the
exception for an empty or malformed input.

The [`52 independent reducer tests`](../tests/test_phasor_resultant.py)
compare with a separate `Fraction.from_float` summation oracle. They cover
all permutations of a cancellation-sensitive four-pair control, exact
cancellation without an `atan2` call, input generators, signed zeros,
axes, subnormals, sums above the finite float range and a minor scaled
component that underflows while the exact pair remains nonzero. They also
check the explicit input and IEEE-rounding preconditions. With the reused
phase-midpoint, unified numerical, circular-semantics and stable-pressure
regressions, **225 tests pass**. Both new Python files pass flake8.
This validated the standalone reducer contract before production integration;
no retained research trajectory was rerun. Local validation is retained at
`artifacts/research/phasor_resultant_validation_2026_09_18.json`.

### Caller policies remain separate from arithmetic

The existing callers do not have one common degenerate-result contract.
Replacing their sums without reviewing these policies would not complete
the phase repair.

| Existing caller | Current reduction | Empty or degenerate behavior |
| --- | --- | --- |
| [Legacy global coordination](../src/tnfr/dynamics/coordination.py) | NumPy means or scalar `fsum` | Nonempty zero rounded resultant reaches `atan2`; empty graph returns after earlier history/gain bookkeeping |
| [Local phase list mean](../src/tnfr/metrics/trig.py) | NumPy means or compensated fallback | No usable neighbors return the caller's fallback; nonempty cancellation reaches `atan2` |
| [Bulk local mean](../src/tnfr/metrics/trig.py) | `bincount` and division by neighbor count | Isolates retain node phase; nonempty cancellation reaches `atan2` |
| [Unified circular mean](../src/tnfr/mathematics/unified_numerical.py) | NumPy means or scalar `fsum` | Empty input rejects; mean resultant norm at or below the configured numeric tolerance rejects |
| [Default nonfused pressure](../src/tnfr/dynamics/dnfr.py) | Neighbor component averages | The existing `1e-12` small-resultant policy uses the node's phase, giving zero phase pressure |
| [Fused pressure](../src/tnfr/dynamics/fused_dnfr.py) | Indexed component addition | Isolates have zero gradient; nonempty cancellation reaches `atan2`, except rows handled by the separate certified midpoint path |

Those thresholds and fallbacks describe existing code, not derivations from
the new reducer. Pairwise U3 phase admissibility uses wrapped separation
and needs no resultant mean. The first repair target is global phase
coordination, where the retained ordering witness was observed; the public
circular mean, local means and pressure branches need their own caller
contracts before integration. The reducer alone changes none of them.

### Versioned global-coordination integration

[`coordinate_global_local_phase`](../src/tnfr/dynamics/coordination.py) now
accepts `global_reduction="exact_components_v1"`. The default `"legacy"`
retains its previous reduction and returns `None`. Both paths share the
adaptive-gain and local-proposal implementation. The selected version changes
the global reduction and its domain policy; it is not a physical coupling
coefficient or a new pressure law. The native runtime still uses the default.

For the opt-in path, the global target comes from the exact sum of the
materialized cached components. Local means, their neighbor order, phase
displacements and phase writes retain their existing owners. The cases are:

| Condition | Selected behavior |
| --- | --- |
| Nonempty graph, effective `kG != 0`, nonzero exact resultant | Use the reducer's materialized direction in the existing update |
| Nonempty graph, effective `kG != 0`, exact joint-zero resultant | Raise `UndefinedGlobalPhaseError` and restore graph-owned state |
| Effective `kG == 0` | Apply zero global displacement without choosing a global target; local evolution remains active |
| Empty graph | Return a distinct `empty_graph` record after the existing history/gain bookkeeping; no resultant is requested |

An exact resultant can be tiny and nonzero. No alignment tolerance turns it
into cancellation, and this arithmetic decision does not certify its angle
as an approximation to the true transcendental sum of the represented phases.
At inactive coupling the resultant is still retained as a diagnostic; its
available angle, if any, is not selected as the update's global target.

The exact invocation captures the existing
[`GraphTransactionSnapshot`](../src/tnfr/operators/network_stage.py) before
caller gain/job conversion, history changes or cache writes. Failures during
admission, proposal computation, phase writes or evidence construction trigger
the shared restoration path, retaining the primary exception. The opt-in
reader rejects nonfinite gains, components and proposals; failed job-count
conversion also propagates rather than taking the legacy fallback. Restoration
covers graph-owned state and aliases admitted by that transaction owner, not
external I/O or all independently retained Python references. The coordinator
does not introduce a second rollback implementation.

The returned frozen `GlobalPhaseCoordinationEvidence` records the numerical
version, empty/applied status, node and neighbor order, primitive phases,
materialized-component reducer record, requested/effective gains and their
mode, local targets, raw proposals, realized phases and execution branch.
It is a public completed-call readout; it is not a sealed causal certificate,
and arbitrary hashable node identifiers are not deep-copied into immutable
identities. Its numeric evidence concerns that call, not future execution.

Permutation invariance holds for the derived global resultant with the same
materialized component multiset. Whole-coordinator invariance would additionally
require appropriate invariance of local reductions, adaptive gains and input
materialization. Their policies, and the pressure/U3 paths, remain separate.
The [execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md) owns validation and
the next frozen comparison; no historical source output is overwritten.

## 13. Versioned phase correction at the retained regional state

[`thol_exact_phase_source_comparison.py`](../benchmarks/thol_exact_phase_source_comparison.py)
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

[`thol_regional_recovery_audit.py`](../benchmarks/thol_regional_recovery_audit.py)
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


## 15. Child-cohort distortion and regional mean-to-shape transfer

[`thol_child_distortion_audit.py`](../benchmarks/thol_child_distortion_audit.py)
reuses the two authenticated saved reset and regional-response reports for
exactly `t=2.25 -> 2.5`, the actual eight children, and the same two archived
branches. It neither advances a graph nor evaluates a reset, pressure or
phase kernel. The fixed metric is the restriction of the original full
`H=diag(d/nu)`; parent coordinates and their boundary remain in the model.

### Staged nodal accounting

Write `delta_0`, `delta_g`, `delta_i`, `delta_f` for the paired EPI difference
before Reception, at integrator entry, after integration and at the endpoint.
The stored pressure was generated at `delta_0`. With admitted common reset
map/offset `(S,c)` and generation coefficients `(A,b)`,

`delta_g = S delta_0 + r_reset`,

`delta_f = (S-hA) delta_0 + r_reset + r_pressure + r_integrator + r_post`.

The common source and affine offset cancel only after exact equality checks.
The map is `S-hA`; replacing it by `(I-hA)S` would change pressure timing.
At the integrator entry the stored-minus-current-model rate splits as

`A(delta_g-delta_0) + diag(nu) (epsilon_kernel + epsilon_stored)`.

Here the first term is the intentional held-input lag, while the other terms
are differences of the originally captured generation defects. No forcing
is inferred from the endpoint derivative. Source components and parent-cut
work remain available through the existing regional observer.

For child-centered restriction `C_B` and increment `q`, the same observer
at zero elapsed time gives the exact reset budget

`E_B(delta+q)-E_B(delta) = <C_B delta,C_B q>_H + ||C_B q||_H^2/2`.

All sixteen paired local writes are retained, followed by the finite Euler
budget at `h=1/4` and the postintegration jump. Signed cross terms are essential:
a per-node positive quadratic term alone cannot explain the energy change.
The later phase update is not assigned to an earlier EPI difference.

### A connection to regional identity

Let `m_B,m_P` be the child and parent H-weighted means of `delta_0`.
The one retained map `T=S-hA` acts on the exact four-part decomposition

`delta_0 = m_P 1 + (m_B-m_P) 1_B + z_B + z_P`,

where the two residuals are separately centered and supported on their own
cohort. The audit retains each `C_B T` image, all centered runtime residuals,
and the complete Gram energy. These are contributions to one recorded map,
not four interventions; their energies are not additive without cross terms.
It also retains `C_B T 1`, so constant preservation is checked rather than
assumed for represented coefficients.

This gives a precise mechanism test: a nonzero `C_B T 1_B` converts a
child-parent mean contrast into within-child spatial contrast, even if
both initial cohorts are internally uniform and the global constant image
vanishes. On a support with no internal child conductance and common child
capacity `nu_B`, `(A 1_B)_B = e nu_B 1_B`, so `C_B A 1_B=0`. Any nonzero
mean-to-child-shape transfer in this held-pressure step then lies in the
admitted Reception map. This is a statement about the combined recorded
support and sequential map; it does not isolate update order as a cause.

The transfer complements the earlier family-mean closure obstruction: it
measures cohort means feeding unresolved shape on a two-cohort partition.
The earlier obstruction measured hidden coordinates feeding eight family
means. Neither direction makes an arbitrary subset an autonomous NFR.


### Retained result and maintenance boundary

The independent represented-rational recount agrees with every stage:

| Paired child state | Centered error energy | H-weighted mean offset |
| --- | ---: | ---: |
| Before Reception | 1.162021485807489e-8 | 0.06635321519 |
| After Reception | 4.174981357577767e-7 | 0.05390852580 |
| After nodal integration | 4.894373864559578e-7 | 0.05132075833 |
| Final endpoint | 4.894373864559578e-7 | 0.05132075833 |

Reception contributes `4.058779208997018e-7`, about **84.9442%** of the
observed increase. Integration contributes `7.193925069818111e-8`, the
remaining **15.0558%**; the postintegration EPI contribution is exactly zero.
These percentages describe signed stage increments in this interval, not
universal operator gains. Parent writes do not immediately change the
child-only score, but change the inputs later consumed by child writes.
Individual child-write increments have both signs and largely cancel;
their complete ordered sum, rather than the largest intermediate score,
is the Reception contribution.

The integration's first-order parent-boundary work is
`+9.735871208336425e-8`. Its held-generation lag contributes
`-2.830200479482354e-8`, and the generation-kernel discrepancy contributes
about `+4.447225349707089e-21`. Internal child dissipation, paired forcing
work and stored-minus-fresh generation work are zero here. The Euler
quadratic term is `+2.882543409599146e-9`; the endpoint-defect linear and
quadratic terms are about `3.68103e-20` and `6.16782e-33`. Thus the held-input
lag partly **attenuates** the increase; neither that lag nor numerical
residuals explains the dominant growth. These are finite Euler accounting
terms, not time-integrated flux measurements.

The mean-to-shape mechanism is present. The initial child-parent mean
contrast is about `0.05994453363`. The eight entries of `C_B S 1_B` are
nonzero, ranging from about `-0.00194634` to `0.00426771`, while
`C_B A 1_B` is **exactly zero**. The isolated self-energy of the mapped mean
contrast is `1.272946951102293e-7`; the mapped parent-centered and
child-centered self-energies are about `7.19564e-8` and `5.97976e-9`.
Their three cross terms are positive and together contribute about
`2.84206e-7`. All other terms remain in the exact Gram ledger, including
the tiny nonzero represented global-constant image. The self-energies alone
must not be presented as additive percentages of the endpoint error.

This accounts for the distortion through the existing nodal transport,
recorded Reception map and actual boundary. It establishes neither a new
physical force nor autonomous regional restoration. Mean relaxation can
feed spatial distortion in a region whose environment is coupled differently
to its members. That is the useful connection to the NFR identity problem:
a maintenance test must retain regional form, mean contrast and environmental
response together instead of relying on one global attenuation score.

The historical follow-up criterion, evaluated in section 16, required a
restoring response after **actual
regional form damage** under a predeclared canonical perturbation and
continuation. Its control must retain measurable nonuniform form, with
control drift reported; reduced error caused only by control flattening is
not sufficient. Keep the mean-to-shape and boundary contributions visible,
and distinguish an existing state-dependent restoring mechanism from supplied
operator timing or an imposed target. A failed damage, control or mechanism
gate is a reported negative/inconclusive result, not a reason to tune the
pressure or select a favorable region. This remains a criterion for that
study, not a current assignment to repeat it; the execution plan owns resumption.

Local output: `artifacts/research/thol_child_distortion_audit_2026_09_18.json`,
SHA-256 `f3ba30169c969e2b39a0958f3437ce0c6822a3bd7c52006942f184398c1b66b3`.
The producer ran once on the two saved reports, with zero kernel, pressure
capture or native calls. Validation includes 27 new portable tests, 40
existing regional/reset tests, and a separate standard-library/Fraction
recount of **277 exact checks**. The existing reset suite also invoked its
optional historical read-only producer test once; that verification is
separate from this producer's budget and did not extend or rewrite evidence.


## 16. Localized regional form damage and finite configured restoration

[`thol_regional_restoration.py`](../benchmarks/thol_regional_restoration.py)
uses one complete authenticated control replay to `t=1.5`, then two detached
copies of that live source. The shared copy owner rebuilds runtime caches;
the study explicitly preserves local neighbor insertion order and the known
last-operator instance marker, checks scientific source equality, and checks
that execution leaves the opposite branch and retained source unchanged.
This avoids interpreting the archival readout's opaque resource descriptions
as a complete runtime checkpoint. It does not claim independence of arbitrary
external resources or callbacks.

The predeclared intervention is one public default Emission on the first
actual child in the retained birth receipt. This is a supplied preparation
mark, not an endogenous target-selection law. Both strict read-only admission
and the native stage apply. The region stays all eight children, with the
original full metric `H=diag(d/nu)` restricted to them. No factor, target,
pressure law or region is fitted to the response.

For a localized EPI jump `J` at child `j`, subtracting the regional H-weighted
mean gives the exact initial form-damage energy

`E0 = (H_j/2) (1-H_j/H_B) J^2`, where `H_B=sum_B H_i`.

Thus a nonzero local jump on a proper multi-node region changes its centered
form as well as its mean. The control at that instant fixes the reference
independently of the later outcome; the continuation compares the same two
branches at seven fixed times from `1.5` to `3`.

### Shared accounting and decision scope

[`thol_regional_restoration_accounting.py`](../benchmarks/thol_regional_restoration_accounting.py)
centralizes the pure reader for this test. It reuses the regional Euler,
paired-distance, source-decomposition and record-admission owners. It keeps
pre-generation, pre-integration, integration and endpoint EPI budgets,
per-glyph EPI and pressure writes, and any intervening residual. Captured
pressure generation precedes named glyphs and integration; clocks and
full support/capacity are checked at the observed boundaries.

At integrator entry let `delta_g` be the current paired field, `delta_0`
its pressure-generation value and `delta F_g,delta F_0` the independently
captured source differences. The stored-minus-current-model pressure splits
exactly into generation kernel discrepancy, generation stored-minus-fresh
discrepancy, observed intervening pressure writes, and

`e [gradient(delta_0)-gradient(delta_g)] + delta F_0-delta F_g`.

The last two terms are held-EPI and held-source lags. The source difference
need not vanish in a localized experiment. Each term is evaluated in the
same integration-entry regional energy balance; its work is a signed
finite Euler contribution, not a measured integrated flux.

For `z_i=delta_i-m_B`, parent-boundary work is further resolved as

`e sum_cut w_ij z_i(delta_j-delta_i)`
`= -e sum_cut w_ij z_i^2 + e sum_cut w_ij z_i(delta_j-m_B)`.

The first term is nonpositive ordinary relaxation through the boundary;
the second retains the incoming parent field. This algebraic split adds no
new feedback law. A negative self term alone is not evidence that a regional
entity autonomously maintains itself.

The declared finite criterion requires actual positive initial form damage,
positive control variance at all seven boundaries, decreased endpoint error,
decreased error/control-variance ratio, and nondecreased endpoint control
variance. The last condition is a conservative sufficient policy against
flattening, not a general definition of pattern identity. Control centered
vector drift and mean offsets remain visible. Passing the criterion supports
finite configured tracking of an evolving control; it does not establish
exact original-form return, erasure of operator history, endogenous scheduling
or autonomous long-term maintenance.


### Measured result

The retained invocation completes the twelve continuation calls and 26
post-preparation forcing-capture API calls. The single target is actual
child `0_sub_0`; its native AL jump is
`44798133900177/562949953421312`. All six predefined gates pass. Both
branches execute `IL/IL/IL/EN/IL/AL`, on unchanged support and capacity.

| Time | Centered paired error | Control variance | Error / control variance |
| --- | ---: | ---: | ---: |
| 1.50, after localized AL | 0.006734007748 | 0.002657876774 | 2.533604196 |
| 1.75 | 0.006296948745 | 0.003925631432 | 1.604060099 |
| 2.00 | 0.005890587815 | 0.005124599118 | 1.149472901 |
| 2.25 | 0.005512655227 | 0.006256037468 | 0.881173627 |
| 2.50 | 0.002939941119 | 0.015159875045 | 0.193929113 |
| 2.75 | 0.002766568872 | 0.015512596781 | 0.178343375 |
| 3.00 | 0.002554142631 | 0.016061734388 | 0.159020350 |

The error energy decreases at every sampled boundary and by **62.07099%**
over the whole window. Its ratio to control variance decreases by
**93.72355%**, while control variance grows by a factor **6.04307**. The
normalization therefore benefits from growing control contrast as well as
falling raw error; both are reported separately. The paired mean offset
falls from about `0.01437493` to `0.00937148`, but does not vanish. The
control's centered-vector drift energy at the endpoint is about
`0.00625356`: the control itself evolves substantially.

The exact aggregate ledger is:

| Contribution | Child-error energy change / Euler work |
| --- | ---: |
| Reception EPI writes | -0.002218738764485463 |
| Actual integration, all six steps | -0.001961126353158593 |
| Pre-generation and postintegration EPI changes | 0 exactly |
| Boundary self-relaxation, within integration | -0.002429152538002955 |
| Incoming parent field, within integration | +0.000079302599110432 |
| IL pressure-write correction, within integration | +0.000451285357921915 |
| Held-EPI lag, within integration | -0.000099945741002708 |
| Euler quadratic term, within integration | +0.000037383968814722 |

The first three rows telescope the measured endpoint energy; the later
rows decompose integration and must not be added a second time. Paired
non-EPI source-channel vectors, internal child dissipation, held-source lag
and stored-minus-fresh generation discrepancies are exactly zero. Generation
kernel work is about `5.42265e-19`, and combined integration endpoint-defect
work about `-1.41588e-18`; they remain explicit. Every observed intervening
pressure write is allocated to its glyph receipt, with zero unassigned
remainder. IL and common AL make no immediate paired EPI-energy change.

The positive IL correction is relative to the captured unattenuated-pressure
Euler reference; it is not an executed no-IL experiment or a claim that IL
violates its nodal coherence contract. Decreasing stored pressure and
reducing paired form error are different observables. The negative boundary
self term is ordinary coupled transport, while the prescribed EN event
supplies the large direct EPI-error reduction.

This gives a useful comparison with section 15: Reception increased the
child-cohort error for the earlier all-child perturbation but decreases it
for this localized perturbation. An operator label alone therefore does
not determine the sign of a regional form-error budget. Sections 17-18
characterize that direction and boundary dependence from the existing maps,
without another trajectory sweep.

The scientific outcome is **supported_in_scope** for finite configured
recovery toward the evolving control. It does not establish autonomous NFR
maintenance: the source preparation, marked intervention and operator timing
remain supplied, and control-form drift is nonzero. No physical emergence
claim follows.

Retained output: `artifacts/research/thol_regional_restoration_2026_09_18.json`,
SHA-256 `7ebf3b9cd372f70a5b5d170b16e50fae22ec2a56595c73adb3ce82866c596259`.
The `.completed.json` checkpoint has identical bytes. Validation includes
58 new portable tests (25 producer/CLI and 33 accounting), 51 reused observer
tests, and **289 independent exact checks** using only the standard library.
Two earlier invocations lost their results to output-format failures;
normalization and complete manifest validation now precede execution, and a
durable completed-result checkpoint precedes final provenance/output checks.
Those technical repetitions are recorded separately in the execution plan
and do not count as independent scientific replications.

## 17. Conditional regional response and environmental input

### A condition derived from the admitted nodal map

Fix a positive full-support metric `H=diag(d_i/nu_i)` and a nonempty proper
region `B`. Let `C` restrict to `B` and subtract its H-weighted mean,
`H_B=diag(H_i:i in B)`, and `M=C^T H_B C`. The regional form-error energy is
`E_B(delta)=delta^T M delta/2`. It does not measure the regional mean,
phase, capacity, operator history or the whole identity of an NFR.

For two trajectories sharing admitted reset coefficients and generation
coefficients, with equal affine offsets and non-EPI sources, the held-pressure
interval of section 15 has

`delta_after = T delta_before + r`, with `T=S-hA`.

The matrices come from the retained operator/support coefficients, not a fit
to the endpoint. The residual `r` retains reset realization, pressure
realization, integration and postintegration changes separately. Common-source
cancellation is a checked premise of each pair; the source need not have the
same value in two different experiments. Replacing `T` by `(I-hA)S` would
refresh pressure at a different stage and would change the declared dynamics.

The exact quadratic response form is

`Q=T^T M T-M`,

`E_B(delta_after)-E_B(delta_before)`
`= delta_before^T Q delta_before/2`
`  + <CT delta_before,Cr>_H + ||Cr||_H^2/2`.

For the declared exact map (`r=0`), the sign is determined by the initial
difference and admitted coefficients. With a measured residual it is a
retained-runtime verification. A prospective binary64 guarantee additionally
needs a residual bound supplied independently of the endpoint being predicted.

### Existing shape versus incoming mean and parent differences

Write the initial difference in the same four-part decomposition as section 15:

`delta=z_B + m_P 1 + (m_B-m_P)1_B + z_P = z_B+w`.

Here `z_B` and `z_P` are centered on their respective cohorts, extended by zero
outside them; `Cw=0`. The component `m_P 1` remains explicit even when its
represented image is very small: represented constant preservation must be
checked, not assumed. Define

`v=CTz_B`, `u=CTw+Cr`,

`Z=||C delta||_H^2`, `A_shape=||v||_H^2`, `U=||u||_H^2`.

The regional error does not increase **if and only if**

`2<v,u>_H + U <= Z-A_shape`.

The right side is the attenuation available from the initial regional shape;
the left side is the signed effect of mean contrast, parent shape and the
realization residual. Either side may have an unfavorable sign. Parent
contributions and their cross terms cannot be replaced by an operator label
or treated as nonnegative additive percentages of the final error.

A direction-independent sufficient condition for every arbitrary centered
input with norm at most `sqrt(U)` is

`Z >= A_shape+U` and `(Z-A_shape-U)^2 >= 4 A_shape U`.

This is exactly `sqrt(A_shape)+sqrt(U)<=sqrt(Z)`, expressed using rational
arithmetic. The triangle inequality proves sufficiency; an input aligned with
`v` attains the upper bound. It is not necessary for a particular input
direction, because a negative cross term can produce cancellation. Neither
the norm nor the signed test derives an autonomous bound on the parent input.

### Why a region-only gain can fail

A finite constant `q` with `E_B(T delta)<=q E_B(delta)` for every real `delta`
can exist only if `T(ker C)` is contained in `ker C`. Conversely, preserving
that kernel induces a linear map on the finite-dimensional regional quotient,
which has a finite norm. This converse supplies neither `q<=1` nor strict
contraction. A basis of `ker C` is `1_B` together with the individual outside
coordinate vectors; their centered images provide a finite exact test.

In particular, `CT1_B!=0` maps zero initial regional shape error to positive
shape error. Section 15 already supplies this obstruction for its declared
map: `CA1_B=0` but `CS1_B!=0`. Thus a regional variance alone cannot give an
unconditional response guarantee for that map. The obstruction is algebraic;
it does not assert that every basis vector is a grammar-admitted runtime
intervention. Environmental relations or additional state can restrict the
admissible inputs, but those restrictions must be derived and checked.

The reusable implementation is
[regional_response.py](../src/tnfr/physics/regional_response.py). It reuses the
exact-coordinate and matrix-product owners, retains both the quadratic form
and the signed input ledger, and performs no evolution or operator selection.

### Matched retained witnesses

The [admission adapter](../benchmarks/thol_regional_response_admission.py)
authenticates the earlier reset report and the localized-restoration report.
At `t=9/4 -> 5/2` all four records share exactly the same `S`, `A`, zero
affine offset, original full H metric, support/capacities and local EN
configuration/order. The sources cancel within each pair; their values also
agree across the experiments. These are common **declared** coefficients:
matching configuration does not prove identity of historical implementation
sources or a general binary64 kernel theorem. The later record's local reset
defect is retained in full without inventing a rounding/clipping split.

The [criterion study](../benchmarks/thol_regional_response_criterion.py)
applies the same form to Reception alone and to the complete held-pressure
interval. It first reconstructs every full paired endpoint from the admitted
map and its separate residual components. For the complete interval:

| Read-out | Earlier all-child perturbation | Localized perturbation |
| --- | ---: | ---: |
| Initial regional error energy | 1.162021486e-8 | 0.005512655227 |
| Final regional error energy | 4.894373865e-7 | 0.002939941119 |
| Isolated child-shape image ratio `A_shape/Z` | 0.5145995631 | 0.5275976258 |
| Available squared-norm drop `Z-A_shape` | 1.128091474e-8 | 0.005208382835 |
| Signed input work `2<v,u>+U` | 9.669152579e-7 | 0.000062954619 |
| Actual energy change | +4.778171716e-7 | -0.002572714108 |
| Combined realization correction to ideal energy change | -3.177557986e-20 | -9.007103052e-18 |
| Sufficient input-ball condition | Fails | Passes |

The initial perturbation magnitudes differ. The isolated ratios describe
images of the two actual child-shape directions, not an all-direction gain
bound or separately executed interventions. Both images attenuate, but the
mean/parent input overwhelms the first witness's available margin. The
localized witness has ample margin and also passes the conservative
direction-independent test at its **observed** input radius. The unknown
future environment has not been bounded by that observation. Reception alone
has the same opposing signs, and the measured realization corrections do not
reverse any of these four signs.

For both `S` and `T`, the exact image of `1_B` has centered energy about
`3.542510399e-5`, while `CA1_B=0`. The same-map obstruction therefore excludes
a finite unrestricted regional-shape gain. This identifies an environmental
closure obligation, not a contradiction of conditional recovery or of the
operator's nodal coherence contract. The child cohort still has no internal
conductance and is defined by recorded lineage; this calculation does not
promote it to an independently emerged coherent region.

This completes the conditional-criterion delivery with no new graph,
trajectory, scalar kernel or forcing capture. The retained result is
`artifacts/research/thol_regional_response_criterion_2026_09_18.json`.
There are 82 new portable tests, 84 reused passing observer tests, and 239
independent exact checks using only standard-library arithmetic. The latter
reconstruct local reset composition, the nodal generator, source/pressure
defects, response matrices and both sign criteria from the saved evidence.

## 18. Environmental-input geometry and protected read-outs

### Image and annihilator in the original metric

Keep the same region `B`, centering `C`, positive metric `H_B` and declared
transition `T`. Let the columns of `N` be `1_B` followed by every outside
coordinate vector. They form a basis of `ker C`. The exact input map is

`G=CTN : regional-mean/outside differences -> centered regional shape`.

Its columns belong to the `(size(B)-1)`-dimensional H-centered shape space.
A centered vector `y` represents the scalar read-out `y^T H_B z`. That
read-out is insensitive to every declared environmental input precisely when

`G^T H_B y=0` and `1^T H_B y=0`.

These are arbitrary independent algebraic inputs. Their individual or joint
reachability under the native policy, phase, grammar and source constraints is
not inferred from this map. Likewise, input insensitivity alone would not
make a read-out constant under the region's self-dynamics or immune to the
separately measured binary64 realization residual.

For independent columns `V` of `G`, their Gram matrix `K=V^T H_B V` is
positive definite. The image projector is

`P_image=V K^-1 V^T H_B`.

With local centering `P_B=I-1 h_B^T/sum(h_B)`, the complementary projector is
`P_protected=P_B-P_image`. Both are H-self-adjoint and idempotent. Its image
gives every protected centered read-out, with dimension
`size(B)-1-rank(G)`. Exact column reconstruction `G=V K^-1 V^T H_B G`
provides a certificate of the image, rather than relying on a numerical
singular-value threshold.

The existing [regional response owner](../src/tnfr/physics/regional_response.py)
now shares one validated domain, centering and input basis between the response
criterion and this geometry. It reuses the [exact Krylov rank](../src/tnfr/mathematics/krylov.py)
and [matrix inverse](../src/tnfr/physics/_exact_linear_algebra.py) implementations.
No new elimination algorithm, physical parameter or evolution path is added.

### Exact classification of the retained child cohort

The [retained geometry study](../benchmarks/thol_regional_input_geometry.py)
authenticates section 17's result, computes each of `S` and `S-hA` once,
and matches the centering, full metric and input images of both retained
perturbations. It does not replay their admission or trajectories.

| Declared step | Full input rank | Parent-only input rank | Child-shape dimension | Protected dimension |
| --- | ---: | ---: | ---: | ---: |
| Reception `S` | 7 | 7 | 7 | 0 |
| Reception plus held-pressure Euler `S-hA` | 7 | 7 | 7 | 0 |

A separate standard-library proof reconstructs `CTN` directly and proves the
rank using a nonzero `7 x 7` minor from child rows 0--6 and parent columns
0--6. The determinant is approximately `5.34828347915e-9` for Reception and
`2.46679963704e-8` for the complete interval; exact rational determinants are
retained in the artifact. H-centering supplies the independent rank upper
bound seven. The weighted annihilator has only the zero vector, and
`P_image=P_B`, `P_protected=0` exactly for both maps.

The parent-only witnesses establish this result without relying on the tiny
represented global-constant image. The change of input geometry between
the stages is exactly `G_T-G_S=-h C A N`; it changes coefficients but does
not reduce the image dimension. The regional-indicator column is unchanged
because `CA1_B=0` in this support/capacity configuration.

Thus every nonzero centered linear child-shape read-out can be affected by
some declared parent input. This excludes absolute input protection for
this particular formed support and lineage-defined region. It neither
excludes the conditional recovery of section 17 nor rules out protection
under a genuinely derived restriction on environmental inputs. It says
nothing by itself about a different geometry, nonlinear native reachability,
repeated stability, spontaneous formation or physical particles.
Full rank also does not measure the magnitude of sensitivity: weak response
and finite conditional robustness are separate from exact input immunity.

Retained output:
`artifacts/research/thol_regional_input_geometry_2026_09_18.json`.
Independent reconstruction, exact minors and annihilator certificate:
`artifacts/research/regional_input_geometry_crosscheck_2026_09_18.json`.
Validation: 76 new portable tests, 82 reused passing tests, and 100 independent
exact checks, including the final report's projectors, Gram inverse, image
reconstruction and source/output hashes. All trajectory, kernel and forcing
capture counts remain zero.

## 19. Support symmetry versus the admitted nodal and reset maps

### Separate mathematical objects

Let `Gamma` be the complete automorphism group of the original weighted
support, including any zero-conductance support edges. Retain its subgroup
that preserves the child set, positive capacity and full metric. For a
permutation with `P[permuted(i),i]=1`, a declared matrix `M` respects the
permutation exactly when `PM=MP`, equivalently
`M[permuted(i),permuted(j)]-M[i,j]=0` at every entry. A vector field is
invariant when `v[permuted(i)]-v[i]=0` at every node.

These are different checks. For an affine map `Mx+a`, both matrix
commutation and `Pa=a` are needed. The paired difference map can cancel a
common `a` even when each individual affine evolution lacks that symmetry.
Here `b` is the captured contribution of the canonical phase, capacity and
topology pressure channels multiplied by capacity; `c` is the Reception
offset. The individual held-pressure interval has offset `c+h b`.

The [exact equivariance owner](../src/tnfr/physics/equivariance.py) now
centralizes the exact matrix/field checks alongside the existing numerical
diffusion diagnostics. It delegates complete enumeration and orbit extraction
to [symmetry sectors](../src/tnfr/physics/symmetry_sectors.py). The explicit
cap of 2,000 is an operational limit: exceeding it raises instead of returning
a partial group. The owner creates a detached support graph for enumeration,
not an evolved TNFR state, and uses no numerical commutation tolerance.

For the common matrix subgroup `Gamma_0`, the environmental space justified
by an **assumed invariant input** is `Fix(Gamma_0) intersect ker(C)`. Since
the group preserves the region, a basis consists of `1_B` and the indicator
of each outside orbit. This characterizes a conditional input space; group
commutation does not itself establish that a particular initial state,
source, boundary history or native input belongs to it.

### Complete retained classification

The [retained symmetry study](../benchmarks/thol_regional_map_symmetry.py)
authenticates the prior response report and preserves its original support,
region, metric and coefficients. Exact checks rederive `H=d/nu`, `A`, the
captured source, and the ordered local-row composition of `S`. The weighted
support has 16 nodes and 24 edges, with no zero-conductance edges in this
witness. Its complete group has eight elements. Every one preserves the
child set, capacities and full metric.

| Object checked within that group | Preserving permutations |
| --- | ---: |
| Weighted support, child set, capacity and H | 8 |
| Nodal generator `A` | 8 |
| Ordered Reception map `S` | 1, identity only |
| Held-pressure map `T=S-hA` | 1, identity only |
| Zero Reception offset `c` | 8 |
| Captured canonical source `b` | 1, identity only |
| Captured generation EPI and phase in each paired record | 1, identity only |

The eight support actions have four orbits: even parents, odd parents, even
children and odd children. The common subgroup for the three declared
matrices is the identity, whose sixteen orbits are singletons. Its fixed
environmental space therefore still has nine independent inputs: the
regional mean and each of the eight parent coordinates. Their image retains
rank seven. No nontrivial restriction on those inputs follows from a common
symmetry of these matrices.

A compact obstruction uses the cyclic shift by two positions on both parent
and child index sets. It preserves the weighted support and commutes with `A`,
but

`S[0,0]-S[2,2]`
`= -630642904502747332437308389707 / 162259276829213363391578010288128`
`~= -0.003886636972791`.

The same entry difference occurs in `T` because `A` commutes. Thus this is
not solely the tiny represented constant-mode leakage of the previous
study. In every retained branch the local EN row family itself satisfies
`row_permuted(i)[permuted(j)]=row_i[j]` for all eight symmetries. Its ordered
composition does not. This identifies fixed sequential composition as the
obstruction at the declared affine-map level. No alternative order or
simultaneous runtime event has been executed by this calculation, and it
does not establish a counterfactual binary64 trajectory.

Source and state asymmetry remain distinct. The four recorded source vectors
are equal across the experiments but invariant only under the identity;
paired cancellation is still valid. Phase invariance here concerns the saved
represented coordinates, not equivalence under a global phase gauge.
Likewise, matrix commutation does not certify complete history or runtime
equivariance. Simultaneous stage semantics cannot be substituted for the
captured sequential map without a separate derivation and admission.

The completed result obstructs a symmetry-based input restriction for this
particular map and cohort. It does not rule out conditional recovery,
correlated inputs derived by another mechanism, or other configurations.
In particular it does not decide whether a Platonic graph forms or persists
under TNFR dynamics. It establishes the check required before transferring
geometric symmetry to a dynamical claim.

Retained report: `artifacts/research/thol_regional_map_symmetry_2026_09_18.json`.
Independent exact enumeration, composition and symmetry witnesses:
`artifacts/research/map_symmetry_crosscheck_2026_09_18.json`.
Validation: 113 new portable tests, 140 reused passing tests and 487
independent exact checks, including final report/source identities, signed
defects, fixed-input images and scope flags. All native, kernel, forcing
capture and historical-admission counts are zero.

## 20. Same-snapshot Reception and the limit of geometric protection

### Comparison using the existing coefficient kernel

For the same retained neighborhood and represented mix, let J contain the
local Reception coefficient row of each target. The sequential map S is
their chronological single-target composition; J reads every row from one
common input. Define U=J-hA with the same pre-generated nodal pressure timing
as T=S-hA. The existing [comparison reader](../benchmarks/thol_regional_map_symmetry.py)
now exposes this comparison through `--snapshot-comparison`, reusing the
exact symmetry and regional-input owners rather than adding another engine.

All 64 saved rows (16 nodes, two paired experiments) match the
[shared represented Reception row](../src/tnfr/operators/_neighbor_epi_kernel.py).
Its coefficients depend on mix and input support, not earlier EPI writes.
Reception uses an unweighted neighbor mean; A retains the actual weighted
transport. The comparison checks configured mix, neighbor membership and
the common hard interval. J is a declared unclipped real-affine map:
binary64 mean/blend evaluation, clipping, semantic kinds and auxiliary writes
remain distinct. Sequential defects and outcomes are not transferred to J.

### Exact result and the unmet input condition

| Quantity | Sequential S / T | Same-snapshot J / U |
| --- | ---: | ---: |
| Common map subgroup order, including A | 1 | 8 |
| Fixed environmental input dimension | 9 | 3 |
| Centered image rank for those fixed inputs | 7 | 1 |
| Orthogonal centered read-out dimension for those fixed inputs | 0 | 6 |
| Centered image rank for unrestricted inputs | 7 | 7 |
| Image of regional-mean contrast `CM 1_B` | Nonzero | Zero |

The three conditional inputs are the regional constant and the even-parent
and odd-parent orbit indicators. Their centered output spans one direction,
constant on each child orbit with opposite signs and the ratio required by
the original H metric. Thus six independent centered linear read-outs
annihilate this restricted input image. This is an exact one-step conditional
input-protection result, not a persistence or repeated self-dynamics theorem.

The condition is not met by either recorded experiment. For each actual
paired difference delta, decompose delta=z_B+w as in section 17: w retains
the parent differences and the H-weighted regional mean. Both retained w
vectors preserve only the identity, as do the saved source, EPI and phase.
No averaging is applied. Consequently the six-direction result cannot be
promoted to either retained environment; unrestricted input rank stays seven.
The common-source cancellation still holds, separately from these symmetries.

### Runtime boundary and disposition

The criterion report is a projection of historical evidence, not a complete
graph-owned stage checkpoint. It lacks replayable grammar histories/debt,
EPI kinds and complete graph/source/runtime configuration. Older trace
boundaries additionally contain explicitly unavailable resources and named
callables, not restorable live owners. The existing
[two-phase dispatcher](../src/tnfr/operators/network_stage.py) requires those
preflight and commit conditions. No synthetic history or defaults are inserted
to present this map calculation as authenticated runtime execution.

This closes the ordering/symmetry comparison. Same-snapshot semantics remove
the exhibited ordering obstruction at the declared-map level, while the
actual input restriction remains unjustified. Neither symmetry nor regular
polyhedral shape is established as necessary for an NFR. The broader open
question concerns a closed endogenous feedback mechanism for differentiated
coherence, using the existing nodal channels and explicit state/history;
additional symmetry sweeps and historical-state restoration are parked.

Retained report:
`artifacts/research/thol_snapshot_reception_comparison_2026_09_18.json`.
Independent arithmetic:
`artifacts/research/snapshot_map_independent_2026_09_18.json`.
This delivery performs 64 coefficient-builder calls, zero scalar evolution
kernel calls, zero native steps and zero live stage calls. No historical
trajectory is rerun or extended.
Validation: 14 new portable tests, 149 distinct reused passing tests and 274
independent exact checks, including the final report and working-source
identities. The previous symmetry/input reports retain their original bytes.

## 21. Closing the relaxed phase-capacity source

### Exact model and common fixed fields

Write x=EPI, let L_W be the weighted EPI random-walk Laplacian and L_U
the unweighted unique-neighbor support Laplacian. The actual canonical
pressure decomposition is

`p = -e L_W x + w_phi g_phi - w_vf L_U nu - w_topo L_U k`,

where k is the support-degree vector. These two Laplacians generally differ.
This section studies fixed physical fields of the existing phase and capacity
updates, together with zero fresh pressure. It does not assume that the EPI
nodal equation uniquely determines those configured update policies.

**Telemetry versus dynamics.** Si is a derived diagnostic of capacity, phase
and pressure, not an additional nodal force or a fundamental evolution law.
Computing/storing Si does not itself reorganize the triad. The implementation
chooses to consume Si in a threshold-based capacity-adaptation policy. That
policy has a dynamical effect because it writes capacity; its use of a
diagnostic is an extra operational assumption, not a derivation from
`dEPI/dt=nu_f*DeltaNFR`. Every result below about that gate is conditional on
this implemented policy. It cannot exclude persistent differentiated forms
under other structurally justified TNFR dynamics. The configured phase and
averaging laws likewise remain explicit premises.

Assume finite undirected support, connected positive symmetric EPI conductance,
positive capacity, e>0 and w_topo=0, as in the retained channel configuration.
The phase support is connected as well. Use exact-real means and trigonometry,
no scalar clipping, no named operator/reset interventions and no external
changes to support or fields. Required diagnostics refer to these same fields.

The [coordination owner](../src/tnfr/dynamics/coordination.py) has the ideal map

`theta_i^+ = theta_i + k_G wrap(m_G-theta_i) + k_L wrap(m_i-theta_i)`,

with global and neighbor phasor arguments m_G and m_i. Suppose all phases
admit a common real lift in an interval of width less than pi, gains are
nonnegative and at least one is positive. Require a fixed point of this
lifted update, excluding nonzero multiples of 2*pi hidden by normalization.
Every phasor mean lies between its participating minimum and maximum, with
a strict interior value unless those phases agree. At a maximum phase both
increments are nonpositive. If k_G>0, a nonconstant state gives a strictly
negative global increment. If k_G=0 and k_L>0, a fixed maximum forces every
neighbor to share that maximum; connectedness propagates equality. Therefore
the phase fixed fields in this chart are exactly consensus phases. In
particular the ideal local phase-pressure channel g_phi vanishes.

For an active [capacity update](../src/tnfr/dynamics/adaptation.py),

`nu_i^+ = (1-mu) nu_i + mu mean_{j in N(i)} nu_j`, with `0<mu<=1`.

If every node is eligible, a fixed capacity satisfies L_U nu=0. The same
maximum principle makes nu constant. With phase consensus and w_topo=0,
the non-EPI source vanishes. Positive capacity then makes zero nodal rate
equivalent to p=0, and e L_W x=0 forces uniform EPI on connected positive
conductance. Conversely constant positive capacity, constant EPI and phase
consensus leave these physical fields fixed. This classifies common fixed
fields of the selected substeps; it does not classify cancellations between
substeps of an arbitrary composite runtime cycle.

### Conditional consequence of the Sense Index gate policy

Within that adaptation policy, the all-eligible premise can be weakened.
At zero fresh pressure and phase
consensus, the [Sense Index](../src/tnfr/metrics/sense_index.py) reduces to

`Si_i = clamp01(alpha*nu_i/nu_max + beta + gamma)`.

Assume its normalization uses the current positive nu_max and that its
nonnegative weights satisfy
`s_max=clamp01(alpha+beta+gamma) >= si_hi`. Normalized exact weights give
s_max=1; represented weights need the displayed inequality, not an assumed
exact sum of one. Every capacity maximum meets the Si threshold and the
zero-pressure part of the stability gate. If unchanged fields are repeatedly
evaluated without external counter resets, its finite stable_count reaches
VF_ADAPT_TAU. At a boundary of a nonconstant maximum plateau, positive-mu
averaging would strictly decrease capacity. Such a plateau cannot persist.
Connectedness therefore forces uniform capacity even if some lower-capacity
nodes are initially ineligible.

The existing held-capacity profile `x=c*1-(w_vf/e)*nu`, when L_W=L_U,
remains a valid [conditional balance](../src/tnfr/physics/capacity_localization.py).
It is not a self-consistent heterogeneous equilibrium of this fresh-diagnostic
relaxation policy. Freezing Si or disabling its refresh supplies a different
closure. The native runtime refreshes pressure and optional Si before later
substeps; that timing agrees at a common equilibrium but cannot be ignored
along a changing trajectory. Stable counters and histories may grow even
when physical fields are fixed.

### What can and cannot be inferred

| Channel or mechanism | Exact conclusion in this branch | Remaining sustaining condition |
| --- | --- | --- |
| Attractive phase coordination | Consensus at a lifted fixed point in the common semicircle | A phase pattern outside that chart, nonstationary motion or another admitted phase feedback needs its own proof |
| Implemented capacity policy consuming fresh Si | Under this gate and averaging law, a persistent zero-pressure equilibrium cannot hide heterogeneous maxima behind inactive gates | This conditional obstruction does not constrain a different capacity law derived from nodal structure |
| Pure EPI transport after source loss | Only uniform zero-pressure form | A maintained canonical source or a different notion of identity is required for differentiation |
| Derived partial-observation memory | Reexpresses existing full dynamics | A memory kernel does not create a new sustaining source |
| Topology pressure | Absent when w_topo=0 | Nonzero topology weight changes the retained configuration and still requires weighted compatibility |

This is not a convergence theorem. On P2, local-only phase gain k_L=1
swaps two unequal phases within the semicircle; capacity gain mu=1 likewise
permits a swap. With k_L=4 and phases (0,pi/2), increments (+2*pi,-2*pi)
also show why wrapped stationarity is weaker than a lifted fixed point.
Binary64 stalled gaps, clipping and represented phasor roundoff remain outside
the exact theorem. Positive-conductance connectivity cannot be replaced by
mere support connectivity with zero-weight links.

Uniform EPI is not the same as an unstructured complete triad. For local-only
phase coordination on C_n, n>=5, the regular winding theta_j=2*pi*j/n has
neighbor resultant `2*cos(2*pi/n)*exp(i*theta_j)` and is nonconstant but fixed.
Its local phase pressure is zero: it preserves phase structure without
supplying differentiated stationary EPI. Existing winding studies retain
their scope; this observation does not reopen the archived C6 campaign.

A different conditional escape is already present in the pressure algebra.
On a unit-conductance nonregular graph with consensus phase, constant positive
capacity and symbolic w_topo>0, `x=c*1-(w_topo/e)*k` has zero pressure.
The three-node path gives a nonuniform example without a new force term.
However it holds the irregular support and changes the retained zero topology
coefficient. On the actual retained weighted support, the independently read
compatibility scalar is
`d^T g_topo = -2170365805794839/4222124650659840`, which is nonzero.
Activating topology pressure alone there would cause mean drift under
consensus phase and uniform capacity, not a stationary pattern. No coefficient
is changed or fitted in this delivery.

**Disposition.** The stationary branch of these configured relaxation laws
is classified; stationary TNFR mechanisms in general remain open. Neither
fixed-point uniqueness nor a supplied profile establishes autonomous NFR
formation. The full triad, support and any genuine endogenous history must
carry the identity being tested. A new sustaining closure must be explicit;
configured lag schedules, REMESH echoes and target-based controllers cannot
be promoted silently to a law derived from the nodal equation.

The general result is the analytic proof above, independently reviewed against
the phase, adaptation and Sense Index owners. All 108 existing targeted phase,
adaptation, capacity-balance and forced-support tests pass; this is regression
evidence, not a numerical proof of the general theorem. Finite exact controls
and the authenticated weighted-topology calculation are retained in
`artifacts/research/source_closure_analytic_checks_2026_09_18.json`, with the
script `artifacts/research/validate_source_closure_2026_09_18.py`. This analytic
delivery changes no engine code, policy parameters or historical artifacts
and executes no graph, pressure kernel or trajectory.

## 22. Source tangency without a telemetry controller

The [diagnostic scope](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#7-derived-observables-and-dynamical-closure)
separates a derived observable from a justified causal law. The following
result advances the primary closure question directly from the existing
pressure channels; it neither uses Si nor supplies a new evolution policy.

### Regular fixed-support identity

Keep fixed symmetric conductance W, finite undirected unique-neighbor support U, constant
channel coefficients e>0, w_phi>=0, v=w_vf>=0 and w_topo=0. Assume positive
capacity and differentiable exact-real fields on an interval without events,
clipping, zero neighbor resultants or phase-wrap crossings. Write

`p=-e L_W x+F`, `F=w_phi*g(theta)-v L_U nu`, `x_dot=diag(nu)*p`.

These are the same pressure conventions as section 21, not a fitted source.
The phase support is unweighted even when W is weighted. For
`S_i=sum_{j in N(i)} exp(i*theta_j)`, its mean-response matrix is

`R_ij=1[j in N(i)] Re(exp(i*theta_j)/S_i)`.

Differentiating Arg(S_i) gives R. Its rows sum to one; they need not be
nonnegative. On the declared regular branch,

`g_dot=(R-I)*theta_dot/pi`.

The exact coefficient owner already exists:
[derive_phase_response](../src/tnfr/physics/phase_response.py) computes
`mean_response=R` from admitted unit planar cosine Gram data. Use that field,
not its separately composed operator-stage Jacobian. The support gradients
remain owned by [support_transport](../src/tnfr/physics/support_transport.py),
and [forcing_realization](../src/tnfr/physics/forcing_realization.py) owns the
captured channel split. Neither a Gram table nor a captured binary64
pressure is a proof that a live continuous chart is admissible.

Differentiating the declared pressure, rather than reconstructing it from
measured EPI motion, now gives the identity

`p_dot=-e L_W diag(nu)*p + (w_phi/pi)*(R-I)*theta_dot - v L_U nu_dot`.

In particular, at p=0,

`p_dot=(w_phi/pi)*(R-I)*theta_dot-v L_U nu_dot`,

`x_ddot=diag(nu)*p_dot`.

This is the **source-tangency condition**: the phase-source and capacity-source
changes must cancel if EPI is to remain at zero pressure. The direct
capacity-rate term in x_ddot is `diag(nu_dot)*p`, which vanishes here;
capacity still acts through its contribution to pressure. The identity
does not determine theta_dot or nu_dot, nor a time for any operator to act.

At phase consensus, R=I-L_U. On connected support the condition is equivalent
to spatial constancy of

`(w_phi/pi)*theta_dot + v*nu_dot`.

For fixed phase and v>0, this requires **uniform capacity increments/rates**,
not uniform initial capacity. Thus a heterogeneous capacity-supported profile
from section 21 is not excluded by the nodal equation itself. If v=0, capacity
has no pressure-source constraint of this type; its positive mobility still
sets EPI response away from p=0.

### Finite events and what an instantaneous test cannot prove

Between two supplied fields on the same support with fixed x and channel
weights, the exact finite counterpart is

`p_plus-p=w_phi*(g_plus-g)-v L_U*(nu_plus-nu)`.

For a capacity-only change with fixed phase, v>0 and connected U, zero pressure
is preserved exactly when `nu_plus-nu` is uniform. A real event that also
writes EPI must include `-e L_W*(x_plus-x)`; changing support, coefficients,
clipping or stored operator pressure needs its own terms. This detached
identity is not grammar admission, event activation or runtime provenance.

A concrete algebraic control uses unit P2, consensus phase, e=v=1/2,
nu=(1/2,1) and x=(1,1/2). Its canonical pressure is exactly zero. A supplied
capacity increment (1/4,0), with x fixed, gives pressure (-1/8,1/8) and
post-change nodal rate (-3/32,1/8). The uniform increment (1/4,1/4) preserves
zero pressure. These are detached states checked against the existing
channel owners; no capacity law or graph evolution is introduced.

Pointwise tangency is only necessary. For a counterexample, allow the
explicitly *free, adversarial completion* nu(t)=(1+t^2,1) on P2, phase zero,
x(0)=0, e>0 and v>0; let x solve the original nodal equation. At t=0,
p=p_dot=x_ddot=0, but `p_ddot=x'''=(-2v,2v)`. This proves insufficiency of
one instantaneous test, not a proposed TNFR capacity law. Conversely, if
F stays constant throughout an interval and p(0)=0, the pressure equation
is homogeneous and uniqueness gives p(t)=0 there. This all-time condition
still supplies neither a law maintaining F nor stability after perturbation.

### Consequence for the primary research question

This identifies a missing relation rather than replacing it with a controller:
the nodal EPI equation and current pressure decomposition do not uniquely
specify capacity, phase, support or event activation. Uniform phase rotation
already lies in ker(R-I), and constant capacity shifts lie in ker(L_U).
Multiple mathematical completions are therefore possible; their existence
does not make them canonical TNFR mechanisms.

Geometry can preserve a phase pattern without sourcing differentiated EPI.
A stationary lifted local-only phase update with nonzero local gain requires
`wrap(Arg(S_i)-theta_i)=0`, so g_i=0 even outside a shared semicircle.
Regular winding does not evade this condition. Prescribed winding or an
imported particle label therefore cannot fill the missing source law.

This completes the bounded tangency calculation. Any proposed law for the
missing channel rates/activation must be independently justified and checked
against this identity. That requirement does not assign a new law-search
task; the [execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the current gate.
No controller experiment, C6 restart, topology sweep or new public physics
claim follows. The finite exact controls reuse existing algebra and are
recorded in `artifacts/research/source_tangency_checks_2026_09_18.json` with
`artifacts/research/validate_source_tangency_2026_09_18.py`. The symbolic proof
above, not a finite test count, establishes the stated conditional identity.

## 23. Capacity exposure does not determine a phase clock

### Source and implementation boundary

The original [TNFR source](TNFR.pdf) distinguishes structural frequency from
periodic oscillation. Its physical PDF pages equal the printed page numbers:

- Section 1.4.10, pages 48-49, presents a classical dynamical-system
  correspondence, including a Kuramoto model with omega identified with nu_f.
- Page 211 says structural frequency "no representa una oscilación periódica
  convencional"; pages 212-213 describe coherent-reconfiguration counts per
  structural-time interval.
- Page 224 qualifies the Kuramoto analogy: nu_f "mide reorganización, no ritmo".
  Page 218 describes structural time as an internal process coordinate.

These are compatible when the oscillator identification is a particular
constitutive correspondence, not a universal deduction. The audited PDF has
378 pages and SHA256
`5ba0f4a2da2d01e550e7004c3c09c6927e6620826052df84cd28babbcbd34fc3`.

The [shared oscillator proposal](../src/tnfr/dynamics/phase_evolution.py) uses

`theta_i_plus=wrap(theta_i+dt*nu_i+dt*K*mean_U3 sin(theta_j-theta_i))`.

The nodal optimizer and FFT engine use this additional phase model beside
their EPI diffusion. It numerically interprets the supplied frequency as an
angular rate in radians per supplied time unit. By contrast, the ordinary
runtime calls [phase coordination](../src/tnfr/dynamics/coordination.py), a
per-invocation global/local mean relaxation with no dt or free nu_i advance.
Neither path establishes that the other is a derived limit. The optional
phase-transport formula in [canonical.py](../src/tnfr/dynamics/canonical.py)
also declares operational sensitivities; its name is not a uniqueness proof.

Multiplying by 2*pi would convert cycles to radians only after identifying
one counted event with a complete cycle. That identification is absent from
the nodal EPI law. The [Hz bridge](../src/tnfr/units.py) is already explicitly
configured and does not supply it. No frequency factor, phase law, pressure
weight or runtime behavior is changed by this result.

### Exact independence, including a relative-phase witness

The EPI law and current pressure channels do not determine phase speed.
On unit P2, let x=c*1, nu=nu_0*1>0 and theta=a(t)*1. Every pressure channel
vanishes for every differentiable a(t). The same initial triad therefore
admits both constant phase and advancing common phase, with unchanged x and
zero phase separation. This is common-phase freedom only; it does not by
itself show that relative phases affect a prediction.

For that stronger distinction, take the existing zero-pressure family on P2,

`nu=(nu_0,nu_1)>0`, `nu_0!=nu_1`, `x_i(0)=c-(v/e)*nu_i`, `theta(0)=0`,

with held capacities and existing coefficients e>0, v>=0, w=w_phi>0.
There are two mathematical completions of these same nodal data:

1. Keep phases fixed. The canonical source is held, p=0 and x remains fixed.
2. Let theta_i(t)=nu_i*t and solve x_dot=diag(nu)*p with the same canonical
   pressure. This is an independence witness, not a proposed physical law
   or an executed operator sequence. Restrict time so the relative phase
   stays inside the chosen regular chart and U3 bound.

Set d=nu_1-nu_0 and a=e*(nu_0+nu_1)>0. In the second completion,
g=(d*t/pi,-d*t/pi), and pressure is (q,-q), where

`q_dot=-a*q+w*d/pi`, `q(0)=0`,

`q(t)=(w*d/(pi*a))*(1-exp(-a*t))`.

Consequently `x_0(t)=x_0(0)+nu_0*integral(q)` and
`x_1(t)=x_1(0)-nu_1*integral(q)` obey the original nodal law, with

`integral_0^t(q)=(w*d/(pi*a))*(t-(1-exp(-a*t))/a)`.

The EPI futures differ whenever d and w are nonzero. No pressure was solved
backwards from an observed derivative. These are analytic alternatives within
the incompletely closed equations, not a claim that both are admitted full
runtime trajectories. Thus phase closure changes structural predictions;
the ambiguity is not only a global phase convention.

The section 22 tangent gives the same initial result on connected support:

`p_dot(0)=-(w/pi)*L_U*nu`,
`x_ddot(0)=-(w/pi)*diag(nu)*L_U*nu`.

For nu=(1/2,1), their coefficients of w/pi are respectively (1/2,-1/2)
and (1/4,-1/2). The existing sine-coupling proposal has zero interaction at
initial phase consensus, so its initial free-advance consequence does not
depend on selecting K. Its later coupled path is not identified with the
uncoupled analytic completion above. The failed implication is universal
phase speed from the EPI equation; configured oscillator models are not
thereby mathematically forbidden.

### What the nodal equation does derive: accumulated capacity

On a regular interval with nu_i(t)>0, define

`s_i(t)=integral_t0^t nu_i(u) du`.

Changing parameter along that same trajectory gives exactly

`dx_i/ds_i=p_i(t_i(s_i))`.

No conversion factor or additional driving term is introduced. This is
accumulated capacity, not circular phase or a measured physical clock.
Neighbors in p_i must still be evaluated at the shared time t_i(s_i); separate
local clocks cannot be treated as simultaneous independent network clocks.
A common clock removes every nodal prefactor only when the capacities agree
at each time; a common factor in heterogeneous capacities leaves their
relative factors. Zero capacity stalls the clock and invalidates this regular
inverse. At p=0, accumulated capacity may advance while EPI stays fixed.

The pure-EPI common-clock solution and heterogeneous boundary already belong
to [directed transport section 6](TNFR_DIRECTED_NONNORMAL_DYNAMICS.md#6-structural-time-and-heterogeneous-capacity).
The finite-exposure retention result in
[cycle support section 4](CYCLE_SUPPORT_DYNAMICS.md#4-default-attenuation-can-retain-epi-as-capacity-tends-to-zero)
is reused, not rerun. Capacity also enters the full pressure through
`-v L_U nu`; a time-coordinate change must not be confused with rescaling
capacity while silently holding that source unchanged.

**Disposition.** The capacity-to-phase gate is closed with an independence
result and a derived reparameterization. A fundamental law for relative
phase/capacity evolution remains missing. Geometry can constrain compatible
motions before a speed is assigned; the execution plan owns that next bounded
question. No oscillator sweep, telemetry controller or physical-particle
claim follows from this result. The finite controls and source provenance
are retained in `artifacts/research/capacity_phase_checks_2026_09_18.json`
and `artifacts/research/validate_capacity_phase_2026_09_18.py`.

## 24. Rigidity and flexibility of a held phase source

### Regular rigidity from the shared mean derivative

On fixed finite support with nonempty neighborhoods, use section 22's
`S_i=sum_j exp(i*theta_j)` and `R_ij=1[j in N(i)] Re(exp(i*theta_j)/S_i)`.
Assume nonzero resultants and a regular center-to-mean wrap chart. Then
`Dg=(R-I)/pi` and `R*1=1`. This statement refers to the canonical full-support
phase channel, not a new phase-update equation or a U3-filtered mean.

If R is nonnegative and its positive-entry directed graph is strongly
connected, its fixed-vector space is exactly the common-rotation line.
Indeed, a maximal component of a vector satisfying R*h=h is a convex mean
of its neighbors. Every neighbor with a positive coefficient must share that
maximum; connectivity propagates it to every node. Thus

`ker(R-I)=span{1}`, `rank(R-I)=n-1`.

A differentiable constant-g path staying in this domain satisfies
`(R-I)*theta_dot=0`, hence `theta_dot=a(t)*1` and
`theta(t)=theta(t0)+c(t)*1`. Common rotation conversely preserves g. Its
speed remains unspecified. This is a compatibility theorem, not relaxation,
attraction, self-maintenance or a stability theorem for the full triad.

There is also local uniqueness modulo rotation: fix one phase coordinate.
The remaining derivative has rank n-1, so n-1 independent output coordinates
give a locally invertible map. Nearby states with the same full g differ
only by rotation. This does not prove global reconstruction or connectedness
of a level set. The already recorded consensus and regular winding on a cycle
can both have g=0 and individually rigid derivatives while belonging to
different local branches; no winding campaign is repeated here.

A sufficient geometric domain on connected undirected support is that every
neighbor phasor has positive projection onto its row resultant:

`Re(exp(i*theta_j)*conj(S_i))>0` for every support neighbor j.

This makes R positive on all support edges. A common phase lift of width
strictly below pi/2 guarantees it, since every numerator is a sum of positive
pairwise cosines; it also ensures regular resultants and wraps. The rowwise
projection criterion is wider and must retain its separate wrap premise.

For **zero phase source**, a further useful corollary uses the actual U3
scale. If g_i=0 and every edge separation is strictly below pi/2, S_i points
along theta_i and

`R_ij=1[j in N(i)] cos(theta_j-theta_i)/|S_i|>0`.

Connected support then has the same rigidity. This does not automatically
extend to nonzero g: the row resultant need not point along its center.
The closed pi/2 gate permits zero projections and needs separate treatment.

### Connected support alone is insufficient: a cube family

Use the unit-conductance cube Q3=C4 x P2, with nodes (j,l), j modulo four,
l in {0,1}, horizontal neighbors (j-1,l),(j+1,l) and partner (j,1-l).
Assign the same four phases to both layers:

`theta_(j,l)=(a,b,a+pi,b+pi)_j`.

Each horizontal pair is antipodal and cancels exactly. The partner has the
center's phase, so `S_i=exp(i*theta_i)` and `g_i=0` for every a,b. Resultants
have squared magnitude one and the center-to-mean displacement is zero.
Varying b-a therefore gives a genuine finite source-preserving deformation,
not merely an extra direction of a pointwise derivative. Here a,b are phase
coordinates, not new force coefficients or a proposed activation law.

At quadrature b-a=pi/2, R is the vertical-partner permutation. It is
nonnegative but reducible into four two-node classes despite connected
graph support. Its rank(R-I)=4 and tangent dimension is four. At the exact
phasor point cos(b-a)=3/5, sin(b-a)=4/5, the horizontal derivatives are
signed and the rank is six, giving tangent dimension two. With c=cos(b-a),
the spectra of R-I split by layer parity:

`layer-symmetric: (0,0,2c,-2c)`,
`layer-antisymmetric: (-2,-2,-2+2c,-2-2c)`.

The displayed family supplies two finite phase parameters (one common and
one relative); tangent dimension four at quadrature does not prove that
all four directions integrate into a four-dimensional finite level set.

**U3 boundary.** The horizontal circular separations are |wrap(b-a)| and
pi-|wrap(b-a)|. Requiring every edge to obey the default pi/2 bound forces
quadrature. Any relative-angle departure violates one horizontal edge.
This family is therefore not an all-edge U3-compatible Coupling orbit.
The existing pressure channel reads all support neighbors; silently dropping
U3-incompatible neighbors changes its definition and destroys this argument.
No graph creation mechanism, admitted event sequence or phase-speed law is
supplied by the geometric family, and the cube is not a selected emergent
polyhedron or a particle identification.

The EPI consequence still follows exactly in its declared mathematical
scope. The unit cube is regular, so L_W=L_U=L and topology pressure is zero.
For held positive capacities, the existing family

`x=c0*1-(v/e)*nu`, `e>0`, `v=w_vf>=0`,

has `p=-L(e*x+v*nu)=0` throughout the phase deformation. It is differentiated
when v>0 and nu is nonconstant. This reuses the capacity-supported balance;
it derives neither the capacity distribution nor a law maintaining it.

### Reusable exact observation and claim boundary

[observe_phase_source_geometry](../src/tnfr/physics/phase_response.py) rebuilds
the existing PhaseResponseReference from its primitive Gram/incidence data
and observes rank(R-I) with the shared exact-rank owner. It returns the
scaled source derivative, tangent dimension and mean-response sign. The
merged operator-stage Jacobian is deliberately separate: at phase_factor=0
that Jacobian is the identity even when the source remains locally rigid.
An exact rank n-1 also certifies conditional local rigidity for some signed
R, without needing the sufficient nonnegative proof.

For example, K4 with phasors (1,0) on three nodes and (-3/5,4/5) on the
fourth has negative mean entries -1/13 but rank(R-I)=3. Negative entries
alone do not imply geometric freedom. Conversely the quadrature cube is
nonnegative yet has more than the rotation line. Zero resultants are rejected;
a live phase chart, U3 admission, causal execution and temporal stability are
not established by an exact Gram or rank calculation.

The [portable controls](../tests/physics/test_phase_source_geometry.py) cover
these independent matrices, exact cube cancellation, disconnected support,
permutation covariance and rejection of tampered cached references. The
general rigidity and finite-family proofs are analytic, not inferred from
the number of passing controls. This closes the planned geometric gate:
regular rigid regions and flexible boundary families both exist. The sole
execution plan owns the continuation; no controller or new evolution law
follows from this result. The subsequent
[grammar audit](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#10-u3-exact-geometric-content-and-a-strict-gate-counterexample)
settles the strict-U3 nonzero-source rank question: a connected double-star
has a two-dimensional tangent kernel strictly inside the gate, but its extra
direction is obstructed at second order. Its local finite level set has only
common rotation. The remaining question concerns finite geometry, not another
rank calculation. The proof and portable fixture have one owner in that audit.

## 25. Relational time and synchronization are separate claims

The original source describes Reception in terms of shared/relational time
and synchronization (PDF page 83), and internal process time on
pages 218-219. This motivates a relational-clock hypothesis; it is not a
derivation that global synchronization equals elapsed time. Section 23 owns
the existing accumulated-capacity identity and its source/units audit.

**Alignment need not advance, even while form evolves.** On pure-EPI unit P2,
positive equal capacities and phase consensus give `R=1`. Holding those phases
fixed is compatible with the EPI equation while a nonuniform form relaxes:
for capacity one and initial EPI `(1,0)`,
`x(t)=((1+exp(-2t))/2,(1-exp(-2t))/2)`. Synchronization remains complete while
form and its accumulated capacity change. Conversely, uniform EPI/capacity
permit any differentiable common phase rotation under that same EPI identity.
The alignment statistic does not identify its speed. Neither control selects
a complete physical phase law.

On the retained prism with phases `(-a,a,0)` in both triangles, `|a|<pi/4`,
the same statistic is `R=(1+2*cos(a))/3`. It is even in `a` and has zero
derivative at `a=0`. Along the supplied control `a=A*cos(chi)`, it repeats
after half the full phase cycle. It therefore loses orientation and cannot
serve as a globally invertible clock for that motion. This concerns phase
alignment; a broader notion of coordination involving form, capacity and
history must supply its own observation and evolution, not inherit this
statistic's name.

### A local state clock requires an already specified tangent

For a complete autonomous state law `z_dot=V(z)` and a scalar observation
`tau=T(z)`, a regular local time coordinate requires
`h=dT[V]>0`. Then `dz/dtau=V/h`. If the relevant components of `V` are missing,
the chain rule does not generate them. A single-valued real state function
cannot increase strictly around a closed orbit: its endpoint difference is
zero, whereas the integral of a strictly positive rate would be positive.
An unwrapped angular clock needs a chart/history or cycle count, as well as
a law for its advance; a circular phase alone supplies neither.

A regular positive reparameterization preserves the oriented path, and an
onto unbounded time change preserves recurrence. Finite accumulated exposure
can instead map infinite original time to a finite internal-time endpoint;
that retention mechanism is already covered by the capacity results and is
not a proof of continuing active oscillation. Relabeling time does not turn
the pure gradient trajectories of variational section 13.10 into recurrent
ones.

### Curve admission can determine a speed without selecting the curve

For a declared full-state curve `z(chi)`, let `v=dx/dchi` be its EPI tangent
and let `b=diag(nu)*p` be the nodal rate evaluated independently from that
state and the existing pressure law. A regular positive scalar clock must
satisfy

\[
b=h v,\qquad h=d\chi/dt>0.
\]

For `v!=0`, this is equivalent to collinearity and `v^T b>0`; the only
possible speed is `h=(v^T b)/(v^T v)`. The inner product merely computes the
unique proportionality coefficient and introduces no physical metric or
force. Every component must agree. If exactly one of `b,v` vanishes there
is no regular positive clock; if both vanish the EPI equation leaves the
clock unconstrained at that point. At an EPI turning point, phase or other
coordinates may still move, so failure to identify the clock there must
not be confused with complete-state stationarity.

This is an EPI admission condition for a supplied curve, not its generation
mechanism or a complete phase/capacity law. The pressure must not be obtained
retrospectively as `h*v/nu`. It provides a useful rejection test: changing a
clock cannot repair a source tangent pointing in the wrong direction.

### Reuse and implementation scope

The existing `structural_time` reader numerically accumulates supplied
capacity over the supplied grid, starting at its first point. Its trapezoidal
value is an estimate, not an exact integral for an arbitrary capacity
function. `certify_structural_time` now uses that accumulated exposure as its
finite structural observation window, including zero exposure; it previously
used the final input timestamp instead. It remains a finite numerical
diagnostic with an unassessed tail, not a derived physical clock or a general
infinite-time certificate.

Controls: [clock scope](../tests/physics/test_structural_clock_scope.py) and
[existing structural-time implementation](../tests/physics/test_structural_time.py).
The complementary [oriented source/form work identity](TNFR_VARIATIONAL_PRINCIPLE.md#1312-oriented-sourceform-work-without-a-selected-clock)
allows a proposed loop to be rejected before choosing its speed. The
[single execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
uses these conditions within G3; no synchrony statistic is promoted to a
controller or a fundamental time law.
