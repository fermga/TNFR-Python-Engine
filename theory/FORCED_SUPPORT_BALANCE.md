# Relative form and mean drift on a held canonical support

**Status:** Exact fixed-support balance and finite Euler error identities.
Runtime observations retain pressure and endpoint defects separately.
Section 7 adds exact instantaneous regional balances on the full support;
they do not assume that a selected region is already an autonomous NFR.
Sections 11-13 connect retained phase sensitivity to the source, integrate
an exact represented-component reduction and compare its regional effect.
Section 14 separates finite regional response from loss of control contrast.
**Research links:** B2.d.7/O3.a, S3, S8, S9 and S16.

Sections 22-25 establish source/phase/capacity boundaries. Sections 26-32
connect conditional joint restoration, regional locking, cycle identity,
support events and post-event relaxation. Section 33 distinguishes maintained
winding from its formation: common-chart Coupling/sine compositions cannot
create it, and capacity contrast has a necessary chart-escape budget. Each
result retains its declared model. Section 34 connects the existing native
relaxation and writer proofs to complete-step execution, including represented
fixed-point repairs. Section 35 resolves heterogeneous Mutation admission
and retains one generated native prefix separately from outward phase action.
Section 36 resolves that trace's phase-action and held-flow energy budget.
Section 37 separates exact reference clipping from the remaining endpoint
defect; the ideal bounded held flow retains the observed increase. Section 38
proves a conditional capacity-recovery obstruction for the admitted default
writer class and repairs a represented adaptation range defect. Section 39
shows why keeping the same stationary form and phase geometry still constrains
capacity differences, even if capacity is omitted from the identity definition.
Section 40 distinguishes feasible static phase compensation from the native
writer's opposing direction and closes a conditional same-P3 source class.
Section 41 separates retained geometric identity from global phase order and
proves a symmetry obstruction to selecting one global phase target.
Section 42 completes the conditional geometric-identity contract on the actual
post-UM weighted C5 and separates its rotation, deformation and form mean.
None supplies an autonomous substrate law.

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

## 26. Conditional phase locking and form restoration on fixed P2

### Declared composition and complete state

Compose the existing [U3-gated phase proposal](../src/tnfr/dynamics/phase_evolution.py)
with fresh [canonical multichannel pressure](../src/tnfr/dynamics/dnfr.py)
on the fixed undirected unit edge `0--1`. This explicitly named composition
is separate from the optimizer/FFT routes, whose phase updates do not feed
back into their pure-EPI pressure. No new oscillator formula is introduced.

The continuous comparison below supplies constant positive capacities
`nu0,nu1`, fixed support, coupling `K>0`, and effective pressure coefficients
`e>0,w>=0,v>=0` for EPI, phase and capacity. The topology gradient is zero
on P2. The equations `nu_dot=0` and `support_dot=0` are premises. Capacity
as angular speed and the sine interaction are the existing configured model,
not deductions from `x_dot=nu*p`. With a continuous phase lift satisfying
`|theta1-theta0|<gamma`, where `0<gamma<=pi/2` is the effective U3 limit,
the complete varying state obeys

```text
theta0_dot=nu0+K*sin(theta1-theta0),
theta1_dot=nu1+K*sin(theta0-theta1),
p0=-e*(x0-x1)+w*(theta1-theta0)/pi+v*(nu1-nu0),  p1=-p0,
x0_dot=nu0*p0,                                      x1_dot=nu1*p1.
```

Pressure is evaluated from the current state, never evolved independently
or reconstructed to fit a desired form. These fixed equations define an
autonomous conditional model, without a time-dependent external signal or
operator dispatch. They do not explain the origin of their supplied
capacities, support or constitutive identification. Coupling is **phase to
EPI**, not bidirectional: EPI does not enter either phase equation.

### Locked target, conserved mean and local response

Define `sigma=nu0+nu1`, `d=nu1-nu0`, `delta=theta1-theta0`, `q=x0-x1` and
`b=(theta0+theta1)/2`. Direct addition and subtraction give

```text
delta_dot=d-2*K*sin(delta),
q_dot=sigma*(-e*q+w*delta/pi+v*d),
b_dot=sigma/2,
m=(nu1*x0+nu0*x1)/sigma,             m_dot=0.
```

The weighted mean is generally not the arithmetic EPI mean. Reconstruction
is `x0=m+nu0*q/sigma`, `x1=m-nu1*q/sigma`. There is exactly one strictly
admitted phase lock if and only if

```text
|d|<2*K*sin(gamma),
delta_star=asin(d/(2*K)),
q_star=(w*delta_star/pi+v*d)/e.
```

At the lock both phases advance at `sigma/2`, while EPI is stationary at
the reconstructed target with its initial m. In particular, maintained
contrast does not imply nonzero instantaneous EPI rate or growing pressure.
The phase and form contrast linearization is triangular, with eigenvalues
`-2*K*cos(delta_star)<0` and `-e*sigma<0`. This proves local attraction of
the relative state. The conserved m and common rotation leave neutral
directions; no single absolute full-state point attracts all preparations.

Equality in the gate condition places the lock on its excluded boundary.
At `|delta|=pi/2` the phase linearization also loses strict contraction.
For `K=0`, a nonzero d produces relative drift, while `d=0` leaves arbitrary
relative phase with no restoring response. Those are not admitted strict
locking cases. When capacities agree, the admitted target has
`delta_star=q_star=0`: common phase rotation alone supplies no contrast.

The fresh-pressure chain rule is a further independent consistency check:

```text
p0_dot=-e*sigma*p0+(w/pi)*(d-2*K*sin(delta)),  p1_dot=-p0_dot.
```

Thus zero pressure away from the phase lock need not remain zero. This
identity follows from the same constitutive map; it does not authorize an
independently prescribed pressure evolution.

### An invariant chart and explicit error bounds

Choose a proof interval `[-rho,rho]` with `0<rho<gamma`,
`|delta(0)|<=rho` and `|d|<=2*K*sin(rho)`. The phase vector field points
inward at its endpoints. The interval is therefore forward invariant, with
nonzero phasor resultants and strict U3 throughout the ideal evolution.
It is an admission region, not a new dynamical coefficient. Put

```text
mu=2*K*cos(rho)>0,       A=e*sigma>0,       C=sigma*w/pi,
D0=|delta(0)-delta_star|, E0=|q(0)-q_star|.
```

The sine secant on the interval is at least `cos(rho)`. Consequently, for
every `t>=0`,

```text
|delta(t)-delta_star| <= D0*exp(-mu*t),
|q(t)-q_star| <= E0*exp(-A*t)+C*D0*I(A,mu,t),
I(A,mu,t)=(exp(-mu*t)-exp(-A*t))/(A-mu)       if A!=mu,
I(A,A,t)=t*exp(-A*t).
```

The nonnegative I is the convolution of the two decaying exponentials;
the second bound follows by variation of constants. Pressure satisfies
`|p0|=|p1|<=e*|q-q_star|+(w/pi)*|delta-delta_star|`. Errors of the individual
EPI coordinates relative to the same-m target are respectively `nu0/sigma`
and `nu1/sigma` times the contrast error. A perturbation changing m changes
the final mean rather than being restored to the old absolute target.

There is also a bounded form interval. Write `Q(delta)=(w*delta/pi+v*d)/e`
and set `q_lo=min(q(0),Q(-rho))`, `q_hi=max(q(0),Q(rho))`. The q vector
field points inward at these endpoints. Its reconstruction gives bounds
for both node forms, which can verify that configured clipping remains
inactive. A clipped trajectory is not automatically this smooth model.

For the **ideal simultaneous Euler map** with a fixed step h, the sufficient
conditions `h>0`, `2*h*K<=1` and `h*A<=1` preserve both intervals. Indeed,
the phase map is nondecreasing there and maps its endpoints inward, while
the form map is a convex combination of q and Q(delta). With
`alpha=1-h*mu` and `beta=1-h*A`, both in `[0,1)`, the discrete errors obey

```text
D_n <= alpha^n*D0,
E_n <= beta^n*E0+h*C*D0*sum(beta^(n-1-j)*alpha^j, j=0,...,n-1).
```

The sum is empty at n=0. These are conservative sufficient conditions,
not a claim that every larger step is unstable. They refer to exact Euler
arithmetic, separately from the continuous theorem and represented execution.
Signed numerical defects must be accounted for and chart membership checked
before any corresponding finite binary64 claim; no infinite runtime
convergence follows merely from these formulas.

### Inactive-channel control and implementation boundary

The prospective `w=0` control must hold **effective** e and v fixed. Then
phase evolution is unchanged, but `q_star=v*d/e` and the phase-to-form term
vanishes. With v also zero the target contrast vanishes; with v nonzero the
old capacity-supported contrast remains. Conversely, v=0 with w>0 and d!=0
isolates the contribution supplied by the phase lock.

The pressure owner normalizes configured weights. Simply deleting the raw
phase weight would generally change e and v and invalidate that comparison.
On this P2 only, moving the removed phase weight to the inactive topology
channel preserves the other effective coefficients; the captured normalized
values must still be checked. This uses a zero existing channel, not an
additional force or an altered nodal equation.

The explicit [P2 composition owner](../src/tnfr/physics/p2_phase_form.py)
exports `derive_p2_phase_form_model` and `propose_p2_phase_form_step` through
`tnfr.physics`. The first returns a frozen `P2PhaseFormModel`: represented
capacities and normalized coefficients, an exact rational locking ratio and
classification for the ideal real half-pi gate, binary64 target estimates,
and the sufficient monotone step ceiling. The displayed phase decay rate is
the least binary64 upper enclosure of the exact rational radicand's square
root `sqrt((2K-d)*(2K+d))`; it is not a lower contraction certificate. This
calculation avoids intermediate overflow and underflow near the lock boundary.
Its default K reuses the existing
operational coupling setting, not a newly derived structural constant.
The theorem above allows general `gamma<=pi/2`; this implementation uses
the half-pi case and separately checks the represented strict gate.

`propose_p2_phase_form_step` creates its own detached fixed P2 and returns
a frozen `P2PhaseFormStep`. It admits canonical phases in `[0,2*pi)`, checks
the strict represented half-pi gap before and after, and requires initial
and proposed EPI in `[-1,1]` with clipping inactive. The shared phase kernel
and `DefaultIntegrator` use one initial state and one unsplit Euler segment;
pressure is freshly evaluated before consumption and again at the endpoint.
`capture_non_epi_forcing` and `SupportTransportEuler` retain the source and
Euler residuals, alongside measured weighted-mean drift. No caller graph or
complete runtime invocation is certified. A single proposal is reproducible
with the declared recipe:

```python
from tnfr.physics import derive_p2_phase_form_model, propose_p2_phase_form_step
weights = {"phase": 1, "epi": 1, "vf": 0, "topo": 0}
model = derive_p2_phase_form_model((0.95, 1.05), pressure_weights=weights)
step = propose_p2_phase_form_step(model, (0.0, 0.0), (0.25, 0.75), dt=0.25)
print(model.lock_status, model.locked_contrast_estimate)
print(tuple(map(float, step.after_epi)), step.after_phase_gap)
```

Target estimates are not transcendental interval certificates, and a finite
accepted proposal does not prove repeated binary64 stability. The
[portable controls](../tests/physics/test_p2_phase_form.py) compare prospective
targets, perturbations and the inactive phase channel within this scope.

The result establishes conditional restoration of a differentiated form
under a specified source-state law. It does not derive the supporting
capacity distribution, full substrate evolution, autonomous NFR creation,
a physical clock or a particle interpretation. The
[single G3 plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) owns
any subsequent admission or experiment; existing passive and independent-
pressure results are not rerun or promoted by this composition.

## 27. Capacity adaptation moves the conditional P2 target

The [existing capacity adaptation owner](../src/tnfr/dynamics/adaptation.py)
moves an eligible node toward the immutable-snapshot mean of its neighbors.
Its eligibility gate consumes stored pressure, stored Si and a consecutive
evaluation counter. It is a configured controller, not a capacity law derived
from the nodal equation. The following one-event calculation tests whether
this controller preserves the supporting capacities of Section 26.

### Exact event calculation and its limits

Retain the positive P2 capacities and definitions of Section 26. Let
`a0,a1` be the eligibility indicators at one event, and let `0<=mu<=1` be
the configured adaptation fraction. In exact arithmetic, with clipping
inactive, the simultaneous capacity proposals are

```text
nu0_plus=nu0+mu*a0*d,       nu1_plus=nu1-mu*a1*d,
d_plus=(1-mu*(a0+a1))*d,
sigma_plus=sigma+mu*(a0-a1)*d.
```

Positive capacities remain positive because the active proposals are convex
combinations of the two positive inputs. Moreover, `|d_plus|<=|d|`, so a
strictly admitted fixed-capacity phase lock remains available after this
event, at the same K and phase gate. Availability of a new lock does not mean
that the previous locked state remains a lock.

When both nodes are eligible, sigma is unchanged and d is multiplied by
`1-2*mu`. Its magnitude strictly decreases for `0<mu<1` and `d!=0`.
The endpoint `mu=1` swaps capacities and reverses d; it is not a strict
contraction. If just one node is eligible, d is multiplied by `1-mu`,
while sigma increases or decreases by `mu*d`. With neither node eligible,
or with `mu=0`, the capacities do not change. These alternatives matter:
partial eligibility also changes the form decay rate `e*sigma_plus` and
the sufficient Euler step ceiling. They must be checked for the next step.

The post-event comparison target follows from the same constitutive law:

```text
delta_star_plus=asin(d_plus/(2*K)),
q_star_plus=(w*delta_star_plus/pi+v*d_plus)/e.
```

The old target remains the reference for the preservation test. Computing
the new target explains its displacement; it does not replace the old
reference or fit a trajectory. For `w>0`, `v>=0` and `d!=0`, reducing
`|d|` reduces the target's magnitude. In the phase-only source witness
`v=0`, eliminating the capacity contrast eliminates the predicted
phase-supported form contrast. Indefinitely repeated eligibility would
need additional evidence; the one-event identity does not establish it.

### Immediate pressure and subsequent form response

The event changes capacity, but leaves EPI and phase unchanged. At the same
form and phase, freshly evaluated pressure therefore changes by

```text
p0_plus-p0=v*(d_plus-d),       p1_plus-p1=-(p0_plus-p0).
```

In particular, with `v=0` an immediate pressure check misses the displaced
target: pressure is unchanged even though the relative phase derivative
changes by `d_plus-d`. Starting at the ideal old lock and subsequently
holding the new capacities fixed gives

```text
delta_dot(0_plus)=d_plus-d,
p0(0_plus)=0,
p0_dot(0_plus)=(w/pi)*(d_plus-d),
q_dot(0_plus)=0,
q_ddot(0_plus)=sigma_plus*(w/pi)*(d_plus-d).
```

This is the existing phase-to-pressure-to-form path. No independent pressure
evolution is introduced. An ideal simultaneous Euler step first changes the
phase gap by `h*(d_plus-d)` while leaving q unchanged; the second changes q
by `h^2*sigma_plus*(w/pi)*(d_plus-d)`. Actual binary64 residuals and chart
admission remain separate obligations.

The conserved mean of Section 26 is also specific to fixed capacities.
Writing its post-event value using unchanged x gives the exact jump

```text
m_plus-m=-mu*d*q*(a1*nu0+a0*nu1)/(sigma*sigma_plus).
```

Subsequent fixed-capacity segments conserve the new mean. They do not
restore the old absolute target simply because each segment has its own
weighted-mean conservation law. There is no EPI jump in this capacity event;
the mean changes because its weights change.

### Finite default-policy witness

The [P2 regression controls](../tests/physics/test_p2_phase_form.py) prepare
the prospective Section 26 lock for capacities `(0.95,1.05)`, raw pressure
weights `phase=epi=1`, `vf=topo=0`, and the existing coupling `K=0.1`.
They compare an active capacity writer with an inactive-writer control,
preserving the old target in both branches. Fresh pressure and fresh Si
are computed through their shared owners before the gate is evaluated.

For the specified binary64 preparation and current shared phase-pressure
kernel, pressure is `(0,0)` and Si is approximately
`(0.8972318538363662,0.9694744245977734)`. Both nodes qualify under the existing defaults:
`EPS_DNFR_STABLE=0.001`, `si_hi=0.5`, `VF_ADAPT_TAU=5` and
`VF_ADAPT_MU=0.1`. Beginning with zero counters, the fifth qualifying
evaluation changes capacities to `(0.96,1.04)`. These are five gate
evaluations on a prepared state, not five elapsed units of physical time.

The old target estimates are `delta_star=0.5235987755982994` and
`q_star=0.16666666666666682`. Reapplying the unchanged model to the observed
new capacities gives `delta_star_plus=0.4115168460674884` and
`q_star_plus=0.13098988043445475`. The new target is displaced, although
strict locking remains available. With only node 0 or only node 1 eligible,
the respective ideal proposals are `(0.96,1.05)` or `(0.95,1.04)`; both
give a phase target near `0.46676533904729683` and a form contrast near
`0.1485760219466835`, with different total capacities.
The unchanged high-threshold regression declares `si_hi=0.875` before
execution. Both nodes now qualify. The earlier pressure arithmetic produced
`(-2.7755575615628914e-17,2.7755575615628914e-17)` and Si approximately
`(0.8389323,0.91117487)`, admitting only the higher-capacity node. The current
shared principal-phase difference and division by represented pi instead
cancel the prepared form gradient exactly in this witness. We retain the
threshold and record the changed outcome rather than retuning it to restore
partial eligibility. The single-node proposals above remain conditional
blend identities, not the outcome of this current runtime control. This is a
separate configured policy control, not another canonical threshold.

After the writer window is closed, two existing P2 proposals with `h=0.25`
expose the delayed form response. In the active branch the first phase-gap
change is approximately `-0.005`; the second contrast change is
approximately `-0.00039788735772973427`. The inactive-writer branch retains
the old contrast within the existing `1e-14` regression criterion in this
finite comparison. These decimal
values are represented observations or target estimates, not exact
transcendental certificates or infinite-time runtime bounds.

The adaptation owner itself refreshes neither pressure nor Si. Si normalizes
pressure by the current maximum magnitude, so an arbitrarily small nonzero
antisymmetric pressure can lose the full pressure contribution to Si compared
with exact zero. The audit retains the actual fresh values instead of
silently substituting their ideal equilibrium limits. Counters count
qualifying calls, reset on a failed gate and are not reset by an adaptation
event; continued eligibility can therefore cause a write on the next call.

This detached, finite policy audit rejects preservation of this nonzero
fixed-capacity target by the existing default writer. It does not establish
instability of the new lock, a complete default runtime trajectory, permanent
eligibility, or an emergent law for Si, capacity or substrate formation.

## 28. Geometry-limited regional phase locking on the unit barbell

### Declared model and the bridge constraint

Use triangles `(0,1,2)` and `(3,4,5)`, joined only by edge `(2,3)`, with
unit symmetric conductance and degrees `d=(2,2,3,3,2,2)`. Capacities are
constant within each region: `nu_L=kappa-epsilon`, `nu_R=kappa+epsilon`,
with `kappa>|epsilon|`. These positive supplied capacities are also the
free angular rates in the existing averaged-sine phase law. Set the
effective pressure coefficients to `e>0,w>0` for form and phase, with
capacity and topology coefficients zero. The topology channel is disabled,
not identically zero on this irregular graph. The ideal equations are

```text
theta_dot_i = nu_i + (K/d_i) sum_(j~i) sin(theta_j-theta_i),  K>0,
x_dot = diag(nu) [-e L_rw x + w g(theta)],
g_i(theta) = Arg(sum_(j~i) exp(i(theta_j-theta_i))) / pi.
```

All edges must remain U3-admitted. This composition excludes Gamma,
events, clipping, capacity adaptation and other controllers. The scalar
form chart is signed real; zero capacity is outside this result. These
assumptions specify an additional phase/capacity model, not a consequence
of the nodal product alone.

Multiplication of the phase row by `d_i` cancels the sine interaction on
each undirected edge. Any common locked angular rate must therefore be
`omega=sum(d_i nu_i)/14=kappa`. Summing only the left region gives the
necessary cut balance `K sin(theta_3-theta_2)=7 epsilon`: its volume is
seven, and its sole outward edge carries all of its phase imbalance.
Consequently no lock under this model exists when `|7 epsilon|>K`.

For `|7 epsilon|<K`, let the principal angles and lock offsets be

```text
alpha = asin(2 epsilon/K),       beta = asin(7 epsilon/K),
psi = (-alpha-beta/2, -alpha-beta/2, -beta/2,
        beta/2,        alpha+beta/2, alpha+beta/2).
```

Substitution in all six rows gives `theta(t)=psi+kappa*t*1`. The interior
and bridge loads are different: `sin(alpha)=2 epsilon/K` and
`sin(beta)=7 epsilon/K`. A two-region average erases that distinction.
With the half-pi U3 gate the strict threshold is `|epsilon|<K/7`;
a narrower configured gate additionally requires `|beta|` to lie strictly
inside that gate. At equality `|7 epsilon|=K`, the bridge cosine vanishes
and strict relative restoring stability is lost. The overloaded result
uses the cut equation and does not depend on imposing equal interior phases.

### Full six-node local stability

The phase Jacobian at the lock is `-K D^(-1) B_cos`, where `B_cos` is the
weighted graph Laplacian with edge weights `cos(psi_j-psi_i)`. On the
strict branch every weight is positive. Similarity by `D^(1/2)` gives
one neutral common-rotation mode and five strictly negative modes. This
includes perturbations that split either interior pair; it is not merely
stability within the four-fiber reduction.

A finite neighborhood follows without linearizing. For a fixed common chart
offset `c0`, write `y=theta-psi-(c0+kappa*t)*1` in a continuous lift.
For any edge its nonlinear
error coupling equals `a_ij(y)(y_j-y_i)`, where

```text
a_ij(y) = integral_0^1 cos(psi_j-psi_i+s*(y_j-y_i)) ds.
```

If `|beta|+osc(y(0))<pi/2` and is strictly inside the configured gate,
the coefficients are symmetric and positive. At a largest error the
derivative is nonpositive; at a smallest error it is nonnegative. Thus
the error range is invariant and the degree-weighted error mean is
conserved. With that mean set to zero,

```text
||y(t)||_D <= exp(-K*cos(|beta|+osc(y(0)))*lambda_2*t) ||y(0)||_D,
lambda_2 = (11-sqrt(73))/12.
```

The same range argument holds for ideal simultaneous Euler when `h*K<=1`.
For the norm bound it suffices to require `h*K<=1/2`; the symmetric error
map then has nonnegative eigenvalues and contracts by at most
`1-h*K*cos(|beta|+osc(y(0)))*lambda_2` on the centered subspace.

### The actual phasor source and its compatible form

The phase source is not the sine interaction or a global Laplacian
substitute. At the lock define

```text
gamma = atan2(3 epsilon/K, 2*cos(alpha)+cos(beta)),
g_star = (alpha/2, alpha/2, gamma, -gamma, -alpha/2, -alpha/2) / pi.
```

The two-neighbor rows are phasor midpoints; a bridge row instead has the
resultant `2 exp(-i alpha)+exp(i beta)`. Its imaginary part is
`3 epsilon/K` and its real part is positive on the strict branch.
Reflection makes `sum(d_i*g_star_i)=0` exactly, so the source is
compatible with a stationary form. With `s=w/(e*pi)`, an independent
solution of `L_rw*z=(w/e)*g_star` is

```text
F = 2*alpha+3*gamma,
z = s*(F/2+alpha, F/2+alpha, F/2, -F/2, -F/2-alpha, -F/2-alpha).
```

The complete stationary family is `x_star=z+c*1`. Its physically relevant
mean for this declared nodal model uses `H=diag(d_i/nu_i)`, not uniform
weights and not `D` alone. To specify initial weighted mean `m`, set

```text
m_H(z) = s*(epsilon/kappa)*((11/7)*alpha+(3/2)*gamma),
x_star = z + (m-m_H(z))*1.
```

The [target evaluator](../src/tnfr/physics/regional_phase_lock.py)
`derive_regional_phase_lock` reuses `derive_forced_support_balance` for
the represented source and mean constraint. The analytic formula above
provides an independent reference. Exact rational cut loads, binary64
inverse-trigonometric estimates, the exactly odd represented source and
the actual production phasor source are distinct evidence. In particular,
enforcing reflection in a comparison source is not permission to replace
or silently correct the production pressure.

### Mean conservation is conditional on the phase source

In general the full form mean obeys

```text
d(m_H(x))/dt = w*sum(d_i*g_i(theta)) / sum(d_i/nu_i).
```

Canonical phasor directions do not generally cancel in that sum. Thus an
arbitrary phase perturbation can move the absolute form target even with
fixed capacities. The phase degree mean, whose sine terms do cancel,
does not supply a form conservation law.

There is a useful invariant restricted preparation. Let reversal exchange
`0<->5`, `1<->4`, `2<->3`, and require the rotating phase offsets to be
odd under that reversal. The supplied detuning and graph share this
symmetry, so both the ideal continuous phase law and its Euler map
preserve it. The fresh phase source remains odd, its degree sum vanishes,
and `m_H(x)` is conserved for any initial form. This symmetry permits a
fixed full form reference during recovery without repeatedly recentering
the observed trajectory. General preparations retain the measured drift.

### Prospective finite recovery protocol

The following preparation and criteria precede runtime observations:
`kappa=1`, `K=1/2`, `epsilon=1/32`, `e=w=1/2`, `h=1/8`, and 256
simultaneous phase/form Euler steps. Fix `m=0`, use `x_star` above, and
initialize

```text
theta(0) = psi + a common canonical chart offset
           + (2,-1,1,-1,1,-2)/128,
x(0) = x_star + (31,31,31,-33,-33,-33)/256.
```

The phase perturbation splits both interior pairs and preserves reflection.
It has degree mean zero and `Y0^2=||y(0)||_D^2=13/8192`. The form
perturbation is H-centered with `X0^2=||x(0)-x_star||_H^2=7/32`.
Neither input nor target is fitted to an observed endpoint.

Elementary bounds give `alpha<1/7`, `beta<1/2`, total phase width below
one radian, edge separation at most `17/32`, and `lambda_2>1/5`.
The following rational lower rates suffice:

```text
lambda_p = 1759/20480,       lambda_x = 31/320,
a = 1-h*lambda_p,            b = 1-h*lambda_x.
```

For the form rate, `e*x^T B*x >= e*nu_min*lambda_2*||x||_H^2`
on the H-centered subspace follows by comparing the minimizing constant
in the D and H variances. The ideal Euler form map has nonnegative
spectrum under `h*e*nu_max<=1/2`, satisfied here.

To bound the changing source, on a common lift of width below one the
phasor-direction derivative is `(R-I)/pi`, with nonnegative entries
`R_ij<=2/d_i` on support. Hence `||R||_D<=2` and the phase-source
Lipschitz bound is strictly below one, since `pi>3`. This yields

```text
Y_n <= a^n*Y0,
X_n <= b^n*X0
       + w*sqrt(nu_max)*Y0*(a^n-b^n)/(lambda_x-lambda_p).
```

Here `w*sqrt(nu_max)*Y0/X0=sqrt(429/229376)<1/23`. Exact rational
evaluation at `n=256` therefore verifies, before execution,

```text
a^256 < 1/10,
b^256 + (a^256-b^256)/(23*(lambda_x-lambda_p)) < 1/5.
```

The selected acceptance is a phase error below one tenth and a full fixed
mean form error below one fifth of their initial norms. These are scoped
ideal-Euler predictions, not a posteriori choices of a favorable time.
The [finite execution controls](../tests/physics/test_regional_phase_lock.py)
use the existing phase proposal, fresh canonical pressure and shared nodal
integrator. They separately retain source and Euler defects, non-symmetric
mean drift, overload and configured-gate controls. Common rotation crosses
the canonical phase wrap during this experiment; relative circular errors
must be reconstructed against the fixed rotating reference, rather than
mistaking a wrap jump for physical phase recovery or instability.

No ideal estimate above certifies infinite binary64 stability or a complete
runtime containing other writers. This mechanism conditionally maintains
regional differentiation while its supplied capacity contrast and phase law
persist. It neither selects those inputs autonomously nor establishes a
physical NFR, particle or endogenous substrate.

### Finite outcome with the frozen target

The production control passes both reserved recovery criteria after 256 steps
(`t=32` in the supplied clock). The target estimates are
`alpha=0.1253278311680654`, `beta=0.4528165947449256` and

```text
x_star = (0.10786151777372711, 0.10786151777372711, 0.06796843009895889,
         -0.07382420301501220, -0.11371729068978044, -0.11371729068978044).
```

| Error norm | Initial | Final | Observed ratio | Precomputed ideal upper ratio |
| --- | --- | --- | --- | --- |
| Phase, D norm | 0.0398361 | 0.000883519 | 0.0221789 | 0.0630834 |
| Full form, H norm | 0.467707 | 0.0160276 | 0.0342685 | 0.118917 |

No target is recentered or fitted after execution. The reflected preparation's
largest observed weighted-mean drift is below `2.77e-16`; its signed Euler
defects are below `1.42e-17`. A separate unperturbed target control keeps form
unchanged at its represented coordinates and has final phase error below
`5.48e-15`. Numerical comparisons permit a separately declared `1e-11`
allowance; neither that allowance nor these measured residuals certifies
asymptotic binary64 accuracy.

The controls also retain a genuinely nonconserving mean response to an
asymmetric phase perturbation, failure to preserve the old form when phase
pressure is disabled, exact overload classification and tighter-gate rejection
of an otherwise strict ideal lock. The target is independently checked against
80-digit phasor and closed Poisson formulas. These results support conditional
regional maintenance and recovery, while the source of the supplied capacity
contrast and phase law remains open.

## 29. Graph-independent locking and phase-source constraints

### Exact model: phase support is not transport conductance

Let the phase support be a fixed connected, simple undirected graph with at
least two nodes, unique neighbor sets `N_i` and counts `d_i=|N_i|`. Capacities
`nu_i>0` and the shared coupling `K>0` are fixed. Every support edge is admitted
by the U3 gate. The supplied phase law is the same averaged-sine law as in
Sections 26 and 28:

```text
v_i = theta_dot_i = nu_i + (K/d_i) sum_(j in N_i) sin(theta_j-theta_i),
Z_i = sum_(j in N_i) exp(i*(theta_j-theta_i)),
g_i = Arg(Z_i)/pi, when Z_i != 0.
```

Here `g` is the ideal canonical neighbor-phasor pressure channel, not the sine
coupling itself. The formulas use circular edge displacements; they do not
require all phases to lie in one global semicircle. The identification of
capacity with free angular rate and the value of K remain constitutive inputs.

Transport may separately use symmetric nonnegative conductances `W_ij` with
strengths `s_i=sum_j W_ij` and `L_W=I-diag(s)^(-1)W`. Phase counts `d_i`
include every support neighbor, even one with zero transport conductance.
Replacing `d_i` by `s_i` in the phase law changes the model. Statements about
stationary form below require the positive-conductance transport graph to be
connected, so every `s_i>0`.

### Locked rate, cut load and the sign of the phase source

At a common-rate lock `theta_i(t)=psi_i+omega*t`, multiply each phase row by
`d_i`. Contributions from the two orientations of an edge cancel exactly.
Therefore

```text
omega = sum_i d_i*nu_i / sum_i d_i,
Im(Z_i) = d_i*(omega-nu_i)/K.
```

More generally, summing only over a vertex set S gives the necessary cut balance

```text
sum_(i in S) d_i*(omega-nu_i)
    = K*sum_(i in S, j outside S, i~j) sin(psi_j-psi_i).
```

For strictly acute edge gaps, its absolute value is less than
`K*|boundary(S)|` for every nonempty proper S. If the gap magnitudes are at
most `rho<pi/2`, the sharper upper bound is `K*|boundary(S)|*sin(rho)`.
The one-bridge load in Section 28 is one instance. Cut inequalities are
necessary, not sufficient: compatible phase differences on cycles and the
actual gate must still be satisfied. No enumeration of graph cuts is required
to use the identity for a specified region.

Now assume every circular edge gap has magnitude strictly below `pi/2`.
Every cosine is positive, so define

```text
r_i = Re(Z_i)/d_i > 0,
g_i = atan((omega-nu_i)/(K*r_i))/pi.
```

The resultant lies in the right half-plane, removing both the zero-resultant
and argument-branch ambiguities. Consequently

```text
sign(g_i) = sign(omega-nu_i),
g_i = 0  if and only if  nu_i = omega.
```

For an edge bound `rho<pi/2`, `cos(rho)<=r_i<=1`, hence

```text
atan(|omega-nu_i|/K)/pi
    <= |g_i|
    <= atan(|omega-nu_i|/(K*cos(rho)))/pi
    <= |omega-nu_i|/(pi*K*cos(rho)).
```

The nodal capacity imbalance fixes the source's sign and constrains its
magnitude. It does not reconstruct the source without the retained resultant
geometry `r_i`, nor prove that a lock exists for the proposed capacities.

### What equal capacities obstruct, and what they do not

With common capacity `nu_i=kappa>0`, the locked rate is `omega=kappa` and
the strictly acute, fully admitted lock has `g=0`. For the unforced form row

```text
x_dot = diag(nu) [-e*L_W*x + w*g],   e>0, w>=0,
```

all stationary forms therefore satisfy `L_W*x=0`. Connected nonnegative
reciprocal transport makes x constant. This conclusion requires the capacity
and topology pressure contributions to be absent or zero, Gamma to be zero,
and no other event, history, clipping or controller term to alter the row.
Equal capacity itself makes the capacity-gradient channel zero; it does not
make topology pressure vanish on an irregular graph.

This is an obstruction to differentiated stationary **scalar EPI through this
phase-source mechanism**. It is not a prohibition of nonuniform phase, a
nontrivial triad geometry, moving patterns, or every possible NFR identity.
In particular, on a unit cycle C_n with `n>=5`, set

```text
psi_j = 2*pi*j/n  modulo 2*pi.
```

Each node has two relative neighbor angles `+2*pi/n` and `-2*pi/n`.
Their sine sum is zero and `Z_i=2*cos(2*pi/n)>0`. Common capacity therefore
gives an exact rotating phase lock with `g=0`, despite winding once around the
circle and lacking a narrow global phase chart. The local absolute phase
gradient is `2*pi/n`, not zero; for C6 it is `pi/3`. The pattern retains
nonuniform phase while its phase channel supplies no differentiated stationary
form. This identity alone asserts neither perturbation stability nor
autonomous formation of that winding state.
The existing [winding owner](COUPLING_WINDING_PERSISTENCE.md#5-nonzero-winding-can-coexist-with-zero-canonical-pressure)
already records this pressure/phase distinction; its additional UM/IL
preservation and recovery theorems retain their own execution hypotheses.

Zero capacity, disconnected positive-conductance transport and other source
channels are different mechanisms. Zero capacity can freeze nonzero pressure;
disconnected transport permits componentwise constants. They are not
counterexamples to the positive-capacity connected-transport statement.

### A residual bound away from exact locking

Use the same degree-weighted capacity mean omega at an arbitrary admitted
strictly acute state, and define the exact phase-rate residual
`epsilon_i=v_i-omega`. Edge cancellation still gives `sum_i d_i*epsilon_i=0`
and

```text
Im(Z_i) = d_i*(omega-nu_i+epsilon_i)/K,
g_i = atan((omega-nu_i+epsilon_i)/(K*r_i))/pi.
```

For common capacity and the edge bound rho this yields

```text
|g_i| <= atan(|epsilon_i|/(K*cos(rho)))/pi
       <= |epsilon_i|/(pi*K*cos(rho)).
```

Thus a nonzero observed phase-rate residual can support a nonzero transient
source even when all capacities agree. A small residual is not an exact lock,
and this algebra does not prove the residual will decay. Computed sine sums,
resultants, production pressure and binary64 phase-step residuals retain their
separate numerical errors; a numerical value near zero is not a symbolic proof.

### Capacity contrast is necessary here, not sufficient for stationary form

For any fixed positive capacities, multiplication of the full form row by
`H_i=s_i/nu_i` gives the mean balance

```text
d(m_H(x))/dt = w*sum_i s_i*g_i / sum_i s_i/nu_i.
```

A stationary form with `w>0` therefore requires `sum_i s_i*g_i=0`.
For this fixed connected transport, this is also the Poisson compatibility
condition for the declared source. It is not implied by the phase identity
`sum_i d_i*(omega-nu_i)=0`: taking phasor arguments is nonlinear, and transport
strengths need not equal phase-support counts. The existing
[`derive_forced_support_balance`](../src/tnfr/physics/forced_support.py) owns
the compatible-profile and drift calculations.

An exact countercontrol uses the unit star K1,3 with center first. Declare
`K=1` and rational capacities

```text
nu = (5/6, 3/2, 3/2, 1/2).
```

The common rate is one. The principal leaf equations give the strictly acute
lock `psi=(0,pi/6,pi/6,-pi/6)`. Set
`gamma=atan(1/(3*sqrt(3)))`. Its canonical source and compatibility sum are

```text
g = (gamma/pi, -1/6, -1/6, 1/6),
sum_i d_i*g_i = 3*gamma/pi - 1/6 > 0.
```

For the strict inequality, `0<gamma<pi/6` and
`tan(3*gamma)=10/(9*sqrt(3))>tan(pi/6)`; both compared angles lie in
`(0,pi/2)`. Thus the lock's nonzero source has nonzero mean drive. No stationary
EPI solves this phase/form model, although the forced-support owner can still
describe its relative profile and common drift. The capacities precede the
phase and form calculation; no desired EPI target defines them retrospectively.
Section 28's reflection symmetry supplies compatibility for its barbell;
heterogeneity by itself does not.

### Admission boundaries and represented observations

- **Missing gated edges:** phase motion averages only admitted neighbors, while
  canonical pressure reads all support neighbors. On P2, a gap `1/2` with an
  effective gate `1/4` removes both phase interactions. Equal capacities then
  give common free advance while full-support pressure has
  `g=(1/(2*pi),-1/(2*pi))`. This is outside full-support admission even though
  the gap is acute. The full-support identities cannot silently use the
  smaller admitted-neighbor denominator.
- **Half-pi boundary and zero resultants:** ideal C4 quarter-turn winding has
  opposite neighbor phasors and `Z_i=0`; its direction is undefined. Binary64
  trigonometric evaluation can leave a tiny nonzero resultant, which must not
  be mistaken for an exact nonzero ideal direction or exact degeneracy.
- **Nonacute branches:** an ungated antipodal P2 has zero sine sum and a negative
  real resultant, so equal capacity does not imply zero phasor pressure. This
  is a mathematical boundary control, not an admitted production U3 branch:
  the engine's hard gate cannot exceed `pi/2`, and drops those interactions.
- **Nonreciprocal support:** a directed C6 with one outgoing neighbor per node
  and winding one has common rate `kappa+K*sin(pi/3)` and `g=1/3`. The reciprocal
  cancellation and degree-weighted rate formula do not apply.
- **Other dynamics:** capacity evolution, changes of support, different phase
  laws, additional pressure channels and Gamma require their actual balances.
  The theorem does not authenticate an executor or admit all configured graph
  writers merely because one snapshot satisfies the geometric assumptions.

The detached [`observe_phase_lock_source`](../src/tnfr/physics/phase_response.py)
read-out keeps the support, capacity, actual phase-pressure capture and
numerically evaluated lock/source relations together. It reports numerical
residuals, not an `is_lock` decision or an exact trigonometric certificate.
The [independent controls](../tests/physics/test_phase_lock_source.py) exercise
the production owners and the symbolic identities and boundaries above.
Exact theorem hypotheses, represented coefficients, finite numerical evidence
and autonomous-law selection remain distinct.

## 30. Acute phase locks are circulation states with integral cycle periods

### Fixed support, common capacity and the declared phase law

Retain Section 29's connected simple undirected phase support, full U3
admission, common fixed capacity `nu_i=kappa>0` and supplied coupling `K>0`.
The phase law is

```text
theta_dot_i = kappa + (K/d_i)*sum_(j in N_i) sin(theta_j-theta_i).
```

Every oriented shortest-arc edge gap is strictly inside `(-pi/2,pi/2)`.
This is a local condition; no common real-valued phase chart is assumed.
The theorem concerns exact common-rate solutions of this supplied phase law.
It does not identify continuous sine evolution with UM events, establish
stability, or derive a phase or support law from the nodal EPI equation.

Choose an arbitrary orientation of each of the `M` support edges. Let B be
the `N` by `M` incidence matrix with `-1` at the tail and `+1` at the head.
Write

```text
delta_e = wrap(psi_head(e)-psi_tail(e)),
f_e = sin(delta_e),
-pi/2 < delta_e < pi/2.
```

The neighbor sine sum at node i is `-(B*f)_i`. Section 29 fixes the locked
rate to kappa, so the equal-capacity locking equations are exactly

```text
B*f = 0.
```

Thus f is an edge circulation with no nodal sources or sinks. This is an
algebraic observation derived from the existing phase coordinates, not a new
primitive or an independently prescribed physical current. Transport
conductances do not enter this unweighted phase balance.

### Integral periods and reconstruction

Choose a spanning tree T. For each edge outside T, orient the corresponding
fundamental cycle so that this chord has coefficient `+1`; complete the cycle
with its unique return path in T. The resulting matrix C has one signed
cycle column per chord. It has `b=M-N+1` columns, `B*C=0`, and its chord rows
form the identity. These columns span all real circulations: subtracting the
combination given by a circulation's chord entries leaves a circulation
supported on a tree, which must vanish successively at leaves. The same
argument for integer entries shows that C is also an integer lattice basis.

For a circle-valued phase assignment, each closed cycle returns to its
starting phase, hence

```text
C^T*delta = 2*pi*w,       w in Z^b.
```

Conversely, these fundamental-cycle period conditions suffice. Fix a root
phase and integrate delta along the tree to assign real lifts to every
vertex. The period condition for each chord says that its supplied gap
differs from the difference of those lifts by an integer multiple of `2*pi`.
Reducing all lifts modulo `2*pi` therefore reconstructs the required edge
gaps. Strict acuity makes each supplied gap the unique shortest-arc value.
Connectedness makes the reconstruction unique modulo one common rotation.

The integer-lattice qualification matters. An arbitrary real cycle-space
basis cannot be used with an unmodified integer-period test. Even an integer
basis need not generate the whole integer lattice: on C5, replacing its
oriented cycle c by `2*c` would incorrectly admit constant gaps `pi/5`, since
`(2*c)^T*delta=2*pi` while `c^T*delta=pi`. Those gaps cannot come from closed
circle-valued node phases. Fundamental cycles prevent this false admission.
Another integer lattice basis changes winding coordinates by an invertible
integer matrix; the underlying phase configuration does not change.

### Necessary and sufficient locking equations, and sector uniqueness

Since sine is one-to-one in the acute interval, every lock has a unique
circulation coordinate z and satisfies

```text
f = C*z,
abs((C*z)_e) < 1                         for every edge,
C^T*asin(C*z) = 2*pi*w,                   w in Z^b.
```

Conversely, a solution of these equations supplies
`delta=asin(C*z)`. The period conditions reconstruct phases, the circulation
condition supplies zero neighbor sine sums, and common rotation at rate
kappa gives the required exact lock. Actual configured U3 limits must also
admit every reconstructed gap; the nonlinear equations do not override a
tighter gate. The criterion is an existence equivalence, not an assertion
that every integer vector w has a solution.

There is at most one strictly acute lock per winding sector w, up to common
rotation. To prove this, suppose `f=C*z` and `f'=C*z'` solve the same sector.
Then

```text
sum_e (f_e-f'_e)*(asin(f_e)-asin(f'_e))
    = (z-z')^T*C^T*(asin(f)-asin(f'))
    = 0.
```

Each summand is nonnegative and is strictly positive when `f_e!=f'_e`.
Consequently `f=f'`, then `delta=delta'`, and the reconstruction differs only
by common rotation. In particular, the zero-winding sector contains only
phase consensus, since zero gaps supply one solution in that sector.
The result gives uniqueness of a locked geometry, not convergence toward it
or invariance of the acute domain under future evolution.

An equivalent variational formulation makes the same distinction. On the
open convex domain `abs(C*z)<1`, define

```text
F_w(z) = sum_e [f_e*asin(f_e) + sqrt(1-f_e^2) - 1]
         - 2*pi*w^T*z,       f=C*z.
```

Its gradient is `C^T*asin(C*z)-2*pi*w`, and its Hessian is
`C^T*diag(1/sqrt(1-f_e^2))*C`, positive definite when `b>0`.
This provides the same at-most-one interior critical point. It does not
guarantee an interior critical point or equate this auxiliary circulation
functional with the engine's EPI pressure or dynamical Lyapunov function.

### What graph structure permits or excludes

- **Trees:** `b=0`, so every circulation is zero. Every acute equal-capacity
  lock is phase consensus.
- **Bridges:** sum `B*f=0` over either side of a bridge. Its single cut term
  forces `f_e=0`, hence `delta_e=0`. Distinct cycle-bearing regions can carry
  their own allowed circulation, but the bridge endpoints have equal phase
  in a lock. A bridge does not generate a capacity-free phase source.
- **Short cycles:** a simple cycle of length l has
  `abs(sum_cycle delta)<l*pi/2`. For `l<=4`, its integral winding must be zero.
  If cycles of length at most four span the real cycle space, all acute
  locks are consensus. Indeed, delta is then orthogonal to every circulation,
  so `delta^T*f=0`; every term `delta_e*sin(delta_e)` is positive unless
  `delta_e=0`. Here real spanning is sufficient because the periods already
  vanish, unlike the integer reconstruction problem above.
- **Long cycles are necessary somewhere, not sufficient by themselves:** a
  nonzero acute lock must have nonzero winding on some simple cycle, which
  must have at least five edges. The mere presence of a long cycle does not
  imply a nonzero lock. K5 contains a five-cycle, but its cycle space is
  generated by triangles, excluding any nonconsensus acute lock.
- **A single cycle with attached trees:** bridges have zero gap and circulation
  is constant around the unique n-cycle. The complete acute lock family is
  `delta_cycle=2*pi*w/n`, with integer `abs(w)<n/4`, additionally restricted
  by the actual U3 gate. Tree vertices share the phase of their attachment
  vertex. Thus the unit windings of C5 and C6 are the first nonzero members,
  while C4's unit winding lies on the excluded half-pi boundary.

For the third and fourth statements, if every simple-cycle period were zero,
the fundamental-cycle periods would be zero and sector uniqueness would give
consensus. No enumeration of every graph or a new numerical recovery sweep
is needed to obtain these obstructions.

At each admitted acute lock, `B*f=0` and positive neighbor cosines imply zero
canonical phase pressure, as proved in Section 29. Nonzero winding still
gives nonzero local absolute phase gradients on its nonzero-gap edges. A
unique phase geometry in a sector can therefore retain information that a
zero-pressure label alone does not describe. This classifies candidate phase
identity; it does not prove differentiated stationary EPI, autonomous NFR
formation, physical identity or a complete state description by winding alone.

### Persistence of a sector is not its formation

For a continuous circle-valued phase trajectory on fixed support, retain the
same oriented fundamental cycles. If no edge reaches antipodal separation,
its wrapped gap varies continuously inside `(-pi,pi)`. Every cycle period is
both continuous and an integer multiple of `2*pi`; its winding is therefore
constant on that time interval. This argument does not require the sine law,
equal capacities, locking, or even acute gaps.

The U3 boundary and the winding boundary are different. Crossing a configured
gate at or below `pi/2` can change which neighbors the phase law uses, without
changing a support-cycle winding. A winding change on fixed support requires
an antipodal branch encounter at `abs(delta)=pi` in any continuous phase
history. Loss of U3 admission is therefore not evidence of a winding slip.
Conversely, gate admission at finitely sampled endpoints does not certify
the unobserved path or exclude cancelling slips. Discrete operator events
need their own path witness or endpoint interpretation.

Changing support can create or remove the cycle whose winding is being
classified. For example, prepare a five-vertex path with consecutive phase
gaps `2*pi/5`, then add the edge joining its endpoints. All five resulting
cycle gaps are acute and the new cycle has winding one, with no phase change
or branch encounter. The prepared phase arrangement and edge addition are
inputs in this example; it is not a derivation of their endogenous selection.
Comparing sectors across a support change therefore requires identifying
which old cycles survive and which cycle classes are new.

This separates two research obligations: maintaining a phase configuration
within an existing sector, and explaining how the retained configuration and
its support sector are produced. The protected UM results in the
[winding owner](COUPLING_WINDING_PERSISTENCE.md) address specified preservation
and recovery maps. They neither replace the present continuous sine law nor
supply an autonomous mechanism that selects nonzero winding from a
zero-winding, fixed-support, branch-safe preparation.

### Exact implementation boundary

[`phase_cycle_geometry.py`](../src/tnfr/physics/phase_cycle_geometry.py)
centralizes oriented support and the fundamental integer cycle basis. Its
exact rational-turn reconstruction uses `a_e=delta_e/(2*pi)`, requires
`abs(a_e)<1/4`, and tests that `C^T*a` has integer entries. This certifies
declared edge-phase compatibility without converting rounded trigonometric
residuals into an exact lock claim. The represented live phases and actual
U3 execution still require their separate observations and admission.

The implemented symbolic odd-sine cancellation test groups equal absolute
turns with opposite incidence signs at each node. A successful exact
cancellation supplies a sufficient locking witness; failure to cancel this
way is inconclusive, since more general trigonometric identities may also
balance a node. The implementation does not solve the full nonlinear
`C^T*asin(C*z)=2*pi*w` existence problem. The
[independent controls](../tests/physics/test_phase_cycle_geometry.py) retain
these topology, reconstruction and sufficient-witness boundaries separately
from finite checks against the production phase and pressure owners.

One control makes this incompleteness explicit: three internally disjoint
paths between the same endpoints have lengths `(10,6,10)` and constant
oriented turns `(1/20,1/12,-3/20)`. Their endpoint differences agree modulo
one turn. Internal nodes cancel pairwise, while the endpoints use
`sin(pi/10)+sin(pi/6)=sin(3*pi/10)`, verified by the exact coefficients of
`1` and `sqrt(5)`. This is an acute lock under the supplied equal-capacity
law, but the oddness-only checker correctly reports `unresolved`. A failed
sufficient symbolic check must therefore not be presented as nonexistence.

## 31. One added chord extends the cycle lattice, not a phase-generation law

### Retained support and the exact lattice identity

Let `G=(V,E)` be a connected simple undirected phase support. Add exactly one
missing edge `e_*=(u,v)` between its existing vertices, retaining every old
edge and vertex. Orient the new edge from u to v and retain the orientations
of the old edges. No conductance, capacity, phase law or locking assumption
is needed for the following topology identity. Connecting distinct components,
deleting edges, changing node identity, loops and parallel edges are different
operations outside this one-chord result.

Retain an old spanning tree T. Write R for the old fundamental cycle basis
in **rows**, so `B*R^T=0` for Section 30's oriented incidence B. Each row has
coefficient `+1` on its own non-tree edge. Embed R in the new edge space by
inserting zero in the added-edge column. Let r be the signed old-edge vector
of the unique tree path from v back to u. Appending the row

```text
c_* = (r, 1)
```

gives the retained-tree basis `R_ext` of the new graph. Here the displayed
last coordinate denotes the added edge; implementations may retain their
lexicographic edge order instead. The edge e_* followed by this tree path
is a closed oriented cycle, so `B_+*c_*^T=0`.

The new cycle rank is `b_+=b+1`. More strongly, these rows generate the full
**integer** cycle lattice. For any integer circulation h on the new support,
subtract its added-edge coefficient times c_*. The remainder has zero on
the added edge and is an integer circulation on G, hence an integer
combination of R's rows. Independence follows first from the added-edge
coordinate and then from independence of R. Thus

```text
Z_1(G_+; Z) = embedded Z_1(G; Z) + Z*c_*
```

is a direct sum for this chosen tree, and the quotient by the embedded old
lattice has rank one. This identifies the surviving old cycles before any
comparison of winding coordinates.

A fresh deterministic spanning-tree calculation can produce a different
post-event basis `R_+`. Both are integer lattice bases, so

```text
R_ext = U*R_+,            U and U^(-1) have integer entries,
det(U) = +1 or -1.
```

One can compute U without a real-valued inverse: its columns are the
entries of `R_ext` on the non-tree edge columns of `R_+`, whose corresponding
submatrix is the identity. The reciprocal construction supplies the integer
inverse. Basis changes therefore cannot create or erase a winding; they
change its coordinates. A comparison that simply subtracts two independently
chosen basis-coordinate lists need not compare the same cycles.

### The new period comes from the actual endpoint gap

Express a circle-valued phase field in turns, with oriented edge gaps
`a_e=wrap_turn(phi_head-phi_tail)` in `(-1/2,1/2)`. This statement excludes
antipodal edge pairs so that the shortest-arc gap is unambiguous. It does not
require strict acuity or U3 admission. For the final field on the new support,
let `a_old^+` be its gaps restricted to old edges and `a_*^+` its new-edge gap.
Then

```text
w_old^+ = R*a_old^+,
w_*^+ = r*a_old^+ + a_*^+,
w_+ = U^(-1) * (w_old^+, w_*^+).
```

Every displayed cycle period is an integer. The return-path sum differs
from `phi_u-phi_v` by an integer, while the new-edge gap differs from
`phi_v-phi_u` by an integer. Their sum is therefore integral. No desired
period supplies the gap: the existing path and actual endpoint phases
determine it. Conversely, appending an arbitrary prescribed edge gap without
checking this relation can fail circular reconstruction.

The tree is a coordinate choice. Replacing its return path by another old
return path changes c_* by an old integer cycle, so

```text
w_*' = w_* + z^T*w_old              for some integer vector z.
```

The quotient lattice has a single generator up to orientation, but a winding
functional descends to that quotient only when it vanishes on all old cycles.
In general, a particular numeric "new winding" is therefore a retained-path
coordinate, not an intrinsic new scalar charge. The complete embedded old
periods together with the chosen new period are unambiguous data.

### Separate a support extension from phase writes at the same event

For branch-safe initial and final phase fields, the old cycles have endpoint
change

```text
Delta_w_old = R*(a_old^+ - a_old^-).
```

If the support-only operation leaves the phase gaps unchanged, this is zero.
The new cycle can nevertheless have nonzero winding. For example, prepare
a five-vertex path with ascending edge turns `1/5` and add the endpoint edge.
With the implementation's orientation `0 -> 4`, its gap is `-1/5` and the
tree return path has sum `-4/5`, giving `w_*=-1`. Traversing the same cycle
as `0 -> 1 -> 2 -> 3 -> 4 -> 0` gives `+1`. All gaps are strictly acute;
there is no phase change or pre-existing cycle whose winding slipped.
The initial nonuniform path field and the added edge remain supplied inputs.

When an actual event also writes phase, compute the final new period from
the **final** path gaps and endpoint gap. A useful counterfactual is to add
the same edge to the old phase field. With `S^-=r*a_old^-`, it has

```text
a_*^0 = wrap_turn(-S^-),
w_*^0 = S^- + a_*^0,
```

provided the old endpoints are not antipodal. Its gap can be nonacute even
when both actual endpoint states satisfy their strict-acute domains. If the
counterfactual endpoints are antipodal, retain its unavailability instead of
choosing a winding by a rounding convention. When defined, `w_*^+-w_*^0`
records an endpoint difference relative to that old-phase extension.

This decomposition does not claim that the counterfactual stage actually
executed, or impose an order on simultaneous phase and support writes.
An executor's event provenance and before/after fields are separate evidence.
Neither equal endpoint periods nor a changed endpoint period describes an
unobserved continuous phase path. On fixed support, an actual continuous
winding change requires an antipodal branch encounter; a discrete operator
jump instead supplies an endpoint change unless a path is also specified.
Crossing the U3 gate alone is still not a winding slip.

### A larger cycle lattice does not guarantee a maintained lock

The rank increase is topological. It neither supplies sine balance nor
preserves the set of acute locks. Adding a diagonal to C5, for example,
produces independent triangle and square cycles spanning its two-dimensional
cycle space. Section 30 then excludes every nonconsensus acute equal-capacity
lock under the fully admitted supplied sine law. The original C5 unit winding
has a nonacute gap across either such diagonal, consistently failing the
new full-support acute premise. Adding a connection can thus remove a
previously available lock class while increasing the number of cycle
coordinates.

A topology-only path closure, an operator event that closes the path while
writing phase, and subsequent relaxation to a lock are three distinct
claims. This support-reset identity establishes the first comparison and
accounts for the second's observed endpoint phases. It does not establish
the third, derive candidate selection, or show autonomous generation of a
coherence pattern or a physical entity.

### Shared implementation and production boundary

[`phase_cycle_geometry.py`](../src/tnfr/physics/phase_cycle_geometry.py)
owns the one-chord support extension and exact endpoint reset alongside its
existing fundamental-cycle basis. The implementation requires the same
ordered nodes and exactly one added edge, rederives supplied geometry/state
data, and retains both integer basis-coordinate maps. Its exact phase reset
uses the existing rational-turn `PhaseCycleState` domain: both actual
endpoint fields are strictly acute and reconstructible. The topology and
branch-safe mathematical identities above have broader scope; this finite
implementation does not silently certify that larger domain.

The counterfactual old-phase extension is reported separately, including
antipodal unavailability. Exact turns are declared angles divided by
mathematical `2*pi`; dividing binary64 runtime radians by represented tau
does not turn an observed event into an exact-angle certificate. Actual
operator endpoints instead retain the shared branch-aware winding observer
and their numerical residuals. The
[exact reset controls](../tests/physics/test_phase_chord_reset.py) and
[Coupling event control](../tests/physics/test_coupling_sector_birth.py)
exercise those different evidence paths. Neither the graph-theoretic result
nor a finite observed sector birth derives the Coupling writer's configured
candidate, threshold or Sense Index policy.

### Prospective default-Coupling event and policy control

The finite control initializes the five-node path with `theta_j=2*pi*j/5`,
`EPI=1/8`, `nu=1`, Si `=0.8`, unit old conductances and refreshed pressure.
Its four path gaps span `4/5` of a turn; there is no cycle or initial winding.
It uses the existing atomic two-phase Coupling executor on target 0 with
declared history `AL`, seed 17, all eligible candidates, and the default UM
factors, bidirectional phase writes, capacity synchronization, pressure
stabilization and functional-link threshold. No factor or threshold is tuned.
The pressure observation uses effective phase/EPI weights `(1/2,1/2)` and
zero capacity/topology weights, without Gamma.

Write `f=1/(pi+1)` for the default phase push. Before execution, the kernel
predicts node 0's displacement `+f*pi/5` and node 1's `-f*pi/5`; other phases
are unchanged. Only nonneighbor 4 is initially phase-compatible. The final
closing gap and maximum old-edge gap are `(2+f)*pi/5<pi/2`. The resulting
compatibility score for equal form and Si is

```text
a = 1-(2+f)/10 = 0.7758546992994776,
threshold = pi/(pi+1) = 0.7585469929947761.
```

The executed stage adds exactly edge `(0,4)` with represented weight a.
Its observed winding is `+1` around `(0,1,2,3,4)`, or `-1` in the exact
reset owner's added-chord orientation. The finite U3 margin is approximately
`0.16244986676002404` radians. Actual phase writes agree with the prediction;
EPI and capacity stay unchanged. The retained stage result and histories
identify this finite invocation, without asserting an unobserved future.

The final state is not a sine lock. On its acute cycle, the ideal phasor
source is

```text
g = (f/10)*(-3,3,-1,0,1).
```

The runtime source matches this within the test's separate binary64 allowance.
Coupling's target-local pressure reduction leaves a positive stored pressure
at node 0, while a canonical refresh gives negative pressure there. A direct
operator write is therefore not identified with freshly computed pressure.
The read-out retains both fields and their difference.

The closing edge is not unit conductance: transport strengths are
`s=(1+a,2,2,2,1+a)`, whereas phase-neighbor counts are all two. With
`F=g/2`, the initial source compatibility sum is

```text
sum_i s_i*F_i = f*(1-a)/10 > 0
               approximately 0.005412055686023124.
```

The shared forced-support observer confirms its positive represented value.
Thus the subsequent weighted EPI mean cannot simply be assumed conserved,
even though the unweighted phase-source sum is zero. Replacing the new edge
by a unit weight would change this actual event's form dynamics.

Two controls delimit the result. Changing only candidate 4's Si from `0.8`
to `0.0` lowers compatibility by `0.2` and suppresses edge creation. Disabling
functional links also prevents closure. Both preserve the same primary
phase, EPI, capacity, stored-pressure and history writes; the differing
support changes the subsequent pressure refresh. This is evidence of the
existing writer's policy dependency, not a proposal to use Si as a fundamental
support law. The event is admitted as finite, policy-selected sector birth;
autonomous selection remains open. The event alone does not establish
subsequent maintenance; Section 32 supplies its conditional phase/form result.

## 32. Acute-cycle relaxation retains phase winding while form relaxes

### The supplied joint law and the support it actually uses

Fix a simple undirected phase-support cycle with cyclically ordered vertices
`i=0,...,n-1`, `n>=3`, and positive common capacity `kappa`. Let every support
edge remain U3-admitted. The supplied averaged-sine phase law is

```text
theta_dot_i = kappa + (K/2)*(sin(delta_i)-sin(delta_(i-1))),    K>0,
delta_i = wrap(theta_(i+1)-theta_i).
```

Indices are cyclic. The phase denominator is two independently of transport
conductance. Fix symmetric nonnegative conductances W on that support whose
positive part is connected. Write `s_i=sum_j W_ij`, `S=sum_i s_i`,
`B=diag(s)-W` and `L_rw=diag(s)^(-1)*B`. The form law considered here is

```text
x_dot = kappa*(-e*L_rw*x + w*g),             e>0, w>=0,
g_i = Arg(exp(i*delta_i)+exp(-i*delta_(i-1)))/pi.
```

The phase and EPI coefficients `w,e` are supplied effective pressure weights.
The full configured mixture is admissible: common fixed capacity makes its
capacity-gradient channel zero, and every node of the simple cycle has unique
support degree two, making its topology-gradient channel zero. Their
coefficients need not be disabled or renormalized, even with unequal transport
conductances. Gamma, further operator events, support changes and evolving
capacities are absent. Any clipping is inactive under
the chart condition below. This is a conditional continuous model of the
declared channels; the nodal product alone does not derive this phase law
or the selection of its support.

### Acute gaps form an invariant region

Assume all initial oriented gaps lie in `[m,M]` with
`-pi/2<m<=M<pi/2`. On the corresponding continuous lifts, differentiating
the neighboring phase equations gives the exact gap equation

```text
delta_dot_i = (K/2)*(sin(delta_(i+1))-2*sin(delta_i)
                    +sin(delta_(i-1)))
            = -(K/2)*(L_cycle*sin(delta))_i.
```

At a maximal gap its derivative is nonpositive; at a minimal gap it is
nonnegative, since sine is increasing on this interval. Thus `[m,M]^n` is
forward invariant. This also supplies continuation inside the branch-safe
acute chart. If the configured U3 gate contains that interval, no edge loses
admission along this exact trajectory. A gate narrower than the initial gaps
does not meet the premise.

The gap sum has zero derivative. Since it is initially `2*pi*ell` for the
cycle winding integer ell, let

```text
delta_bar = 2*pi*ell/n,
u = delta-delta_bar*1,
rho = max(abs(m),abs(M)) < pi/2,
lambda_C = lambda_2(L_cycle) = 2-2*cos(2*pi/n).
```

Then `sum u_i=0`. Multiplication of the gap equation by u gives

```text
(1/2)*d||u||_2^2/dt
 = -(K/2)*sum_i (u_(i+1)-u_i)
                  *(sin(delta_(i+1))-sin(delta_i))
 <= -(K*cos(rho)/2)*u^T*L_cycle*u
 <= -gamma*||u||_2^2,

gamma = K*cos(rho)*lambda_C/2 > 0.
```

Consequently `||u(t)||_2<=Q*exp(-gamma*t)`, where `Q=||u(0)||_2`.
This is convergence to uniform **oriented gaps**, not to equal phases when
`ell!=0`. The phase-rate error also satisfies
`||theta_dot-kappa*1||_2<=K*||u||_2`: the cycle incidence has norm at most
two and sine is 1-Lipschitz. Its integrability implies that each lifted
`theta_i(t)-kappa*t` has a limit, with a rate-error tail bounded by
`K*Q*exp(-gamma*t)/gamma`. The common rotation is therefore separated from
the convergent relative configuration. No global semicircle containing all
node phases is required.

### The canonical phase source vanishes, but phase geometry remains

Both neighbor displacements lie in the center's open half-pi interval.
Their resultant is

```text
exp(i*delta_i)+exp(-i*delta_(i-1))
 = 2*cos((delta_i+delta_(i-1))/2)
       *exp(i*(delta_i-delta_(i-1))/2).
```

Its displayed cosine is positive. The exact phasor direction is therefore
the midpoint and

```text
g_i = (delta_i-delta_(i-1))/(2*pi),
sum_i g_i = 0,
||g(t)||_2 <= (Q/pi)*exp(-gamma*t).
```

This local identity uses the actual neighbor-phasor pressure, not a proposed
global replacement of that pressure by a Laplacian. It ceases to be justified
outside its chart/resultant premises. At the limiting winding configuration
the pressure source is zero, whereas the tetrad's local absolute phase
gradient is `abs(delta_bar)=2*pi*abs(ell)/n`. Thus vanishing phase pressure
does not erase nonuniform phase geometry. This maintained relative-phase
configuration does not, on its own, identify an autonomous NFR or a particle.

### One existing margin controls availability and conditional stiffness

The same geometry supplies more information without another fitted parameter.
Let `r_i=|sum_(j in N(i)) exp(i*theta_j)|/2`. The preceding exact identity gives

```text
r_i = cos((delta_i+delta_(i-1))/2) >= cos(rho) > 0.
```

Thus the preserved acute interval certifies availability of every local phasor
direction for this ideal trajectory, even when the global resultant is zero.
The existing `cosine_lower_bound` in `CycleRelaxationEnvelope` is an outward
rational lower bound for this same quantity; a separate tunable availability
threshold is unnecessary. The certificate does not assert zero rounding error
or that an arbitrary native phase writer preserves the interval.

It also controls the Hessian of the supplied sine law's alignment potential.
For an oriented cycle incidence matrix A and
`V(theta)=sum_edges(1-cos(delta_e))`,

```text
Hessian(V) = A diag(cos(delta_e)) A^T >= cos(rho)*L_cycle.
```

After removing the common rotation, its smallest eigenvalue is at least
`cos(rho)*lambda_2(L_cycle)`. Multiplication by the declared phase mobility
`K/2` gives the existing decay bound. This is a conditional restoring margin,
not a new primitive or a phase law derived from the nodal product.
The phase-pressure response is a different derivative: within these fixed
two-neighbor branches, `D_theta g=-L_cycle/(2*pi)`, independent of the cosine
weights. Confusing these two Jacobians would silently substitute the supplied
sine evolution for the actual Arg-based pressure. Existing source, response
and cycle owners already contain the necessary quantities.

### The weighted form response and its nonconserved mean

Let `m_s=(sum_i s_i*x_i)/S`, `y=x-m_s*1`, and
`||z||_s^2=sum_i s_i*z_i^2`. Fixed reversible transport is self-adjoint in
this metric. Let

```text
lambda = kappa*e*lambda_2(L_rw) > 0,
D0 = ||y(0)||_s^2,
A = Q/pi,
C = kappa*w*sqrt(max_i s_i)*A.
```

Projection onto the s-centered subspace is an orthogonal contraction.
Variation of constants and the preceding source bound yield

```text
||y(t)||_s <= exp(-lambda*t)*sqrt(D0) + C*J(lambda,gamma,t),

J(lambda,gamma,t)
 = integral_0^t exp(-lambda*(t-v))*exp(-gamma*v) dv
 = (exp(-gamma*t)-exp(-lambda*t))/(lambda-gamma), lambda!=gamma,
 = t*exp(-lambda*t),                                  lambda=gamma.
```

Both terms tend to zero. The equality case is retained instead of dividing
by a zero rate difference. Conservative positive lower bounds on either
rate remain valid because the defining integral decreases as either rate
increases. Squared upper bounds can be formed from these nonnegative terms
without treating rounded square roots as exact coefficients.

The s-weighted mean need not be conserved. Diffusion makes no contribution
to its derivative, but the actual phase source gives

```text
m_s_dot = (kappa*w/S)*sum_i s_i*g_i
        = (kappa*w/(2*pi*S))*sum_i (s_i-s_(i+1))*u_i.
```

Define

```text
M = kappa*w*Q*sqrt(sum_i (s_i-s_(i+1))^2)/(2*pi*S).
```

Then `abs(m_s_dot)<=M*exp(-gamma*t)`. Hence a finite limit `m_infinity`
exists and

```text
abs(m_s(t)-m_s(0)) <= M*(1-exp(-gamma*t))/gamma,
abs(m_infinity-m_s(t)) <= M*exp(-gamma*t)/gamma.
```

Every EPI coordinate converges to this same limiting mean. The theorem does
not assign the limit to the initial mean unless the weighted source actually
vanishes. Equal transport strengths make M zero; unequal strengths require
the retained source integral or its bound. The mean-tail estimate refers to
the exact modeled trajectory at time t, not automatically to a numerically
computed endpoint with an unmeasured solver error.

For a sufficient all-time form-chart check, `J<=1/lambda`. A conservative
global disagreement bound is

```text
D_global = C^2/lambda^2,                    D0=0,
           D0,                             C=0,
           2*D0+2*C^2/lambda^2,             otherwise.
```

Since `abs(y_i)<=sqrt(D_global/min_i s_i)`, the entire exact trajectory lies
inside the interval centered at `m_s(0)` with radius
`M/gamma+sqrt(D_global/min_i s_i)`. If this interval lies in the configured
hard EPI chart, clipping stays inactive and the preceding linear form
equation applies throughout. This sufficient interval can be conservative;
failure to fit is not evidence that clipping actually occurs.

### Apply the theorem to the actual closing-edge conductance

For the ideal arithmetic of Section 31's default Coupling event, put
`f=1/(pi+1)` and retain its five cyclic gaps

```text
delta(0) = (pi/5)*(2-2*f, 2+f, 2, 2, 2+f).
```

Their mean is `2*pi/5`, their winding is one, and
`Q=f*pi*sqrt(6)/5`. Their maximum `(2+f)*pi/5` is strictly below `pi/2`.
The theorem therefore predicts convergence toward uniform winding-one
gaps under the supplied phase law. The predicted nonzero winding is retained,
while the initially nonzero phase pressure decays.

The EPI graph must retain the closing conductance
`a=1-(2+f)/10`, not replace it by one. Its strengths and total are
`s=(1+a,2,2,2,1+a)` and `S=8+2*a`. In particular,

```text
sum_i s_i*g_i = (a-1)*(delta_0-delta_3)/(2*pi),
sum_i s_i*w*g_i at t=0 = w*f*(1-a)/5,
M = kappa*w*(1-a)*Q/(sqrt(2)*pi*S).
```

For `w=1/2` this recovers Section 31's positive initial compatibility
`f*(1-a)/10`. Uniform initial EPI therefore starts with a positive weighted
mean derivative, even though the unweighted phase-source sum is zero.
The asymptotic result is uniform EPI at a finite, potentially shifted value,
alongside nonuniform winding-one phase geometry. It is not a maintained
nonuniform scalar EPI profile.

The ideal expression for f and the captured binary64 event endpoint are
separate initial data. The implemented endpoint certificate starts from
the **actual** materialized phases, capacities, conductances and EPI, rather
than silently replacing them with the ideal formulas above. Exact real
arithmetic then defines the conditional reference model from those inputs.

### Reuse and numerical evidence boundaries

[`cycle_relaxation.py`](../src/tnfr/physics/cycle_relaxation.py) bounds this
conditional model without advancing a graph. It reuses the shared detached
pressure/transport capture, exact rational-pi branch enclosures and reversible
spectral-gap owners. An actual materialized oriented gap is represented as
`r_i+2*k_i*pi`, with rational r_i and a verified branch integer k_i. Its
winding and deviations from the mean retain mathematical pi; a binary64
gap sum is not rounded to an integer and promoted to a proof.

Rational enclosures bound the initial Q, the acute radius and the phase and
transport gaps. For `0<=rho<pi/2`, the concavity bound
`cos(rho)>=1-2*rho/pi` supplies a conservative positive phase coefficient.
Shared exact exponential enclosures, with outward exponent rounding on a
2^-64 arithmetic grid and the shared exponent work limit 4096, bound the decay, Duhamel response,
mean drift/tail and all-time chart. These are exact conditional inequalities
for the reference model, not floating-point asymptotic convergence results.

Production execution instead refreshes its represented pressure and uses
the shared phase proposal and nodal integrator. Trigonometric reduction,
source realization, represented coefficients, Euler truncation, phase writes
and possible clipping retain their own numerical discrepancies. Finite
production observations can check a declared horizon against the prospective
envelopes; they do not remove those discrepancies or certify arbitrary future
binary64 execution. The preceding Coupling event remains policy-selected,
and this maintenance theorem does not derive autonomous support selection,
formation from a uniform preparation, or physical identification.

### Frozen finite production continuation

The [post-event controls](../tests/physics/test_cycle_postevent_relaxation.py)
reuse the same default UM event through the
[shared preparation](../tests/joint_phase_helpers.py). Before continuing,
the evaluator fixes envelopes at `t=0,8,16,32` from the actual captured
endpoint, with `K=1/2`. The finite execution uses `h=1/8` for 256 steps,
with fresh canonical pressure, the shared phase proposal and the existing
nodal Euler integrator. There are no further operators, support/capacity
changes, controller updates or fitted final targets.

The conservative represented-state reference rates are approximately
`gamma_lower=0.03573031566296992` and
`lambda_lower=0.32231115638261676`. Its all-time sufficient EPI interval is
approximately `[-0.09720168,0.34720168]`, strictly inside the hard chart
`[-1,1]`. These are rounded displays of rational model bounds.

The retained finite observations are:

| Read-out | At event endpoint | At t=32 |
| --- | ---: | ---: |
| Gap deviation norm from winding-one twist | 0.3716106 | 0.00342818 |
| Canonical phase-source norm | 0.1079811 | 0.000641409 |
| Strength-metric EPI disagreement norm | 0 | 0.00172119 |
| Weighted EPI mean | 0.125 | 0.1280933944 |

The form disagreement initially rises in response to the source and then
falls; its observed peak norm is approximately `0.05042945`. All four
reserved observations satisfy the prospective envelopes. Each finite step
retains winding one, a positive U3 margin and interior EPI, while the exact
Euler energy accounting has zero identity residual. The maximum retained
state-rounding defect is approximately `1.39e-17`; fresh-pressure assembly
defect is at most approximately `2.00e-18`. These do not bound accumulated
phase or Euler truncation error or prove numerical convergence.

Independent 110-digit comparisons check affine-pi initial gaps, source/mean
identities, the rational spectral lower bound and the exponential convolution.
Reversed orientation, node relabeling, zero source, equal decay rates, unit
transport weights and rejected branch/gate/capacity/support domains delimit
the implementation. Replacing the closing weight by one makes the ideal
mean-drive bound zero, demonstrating why the actual weight cannot be dropped.
The raw operator pressure and refreshed pressure produce the same conditional
reference but retain different reported stored-pressure residuals.

## 33. A common phase semicircle obstructs winding generation by the existing maps

### The question is generation, not maintenance of a prepared sector

Sections 31 and 32 distinguish a support event that exposes a prepared phase
arrangement as a new cycle period from subsequent conditional maintenance.
They do not show that those mechanisms can generate the phase arrangement
from an initially common semicircle. The following obstruction addresses that
specific missing step. It concerns cycle winding, not every possible form of
coherence, nonuniform EPI or NFR identity.

Let a finite set of existing nodes have real phase lifts `q_i` in a common
interval `I=[a,b]` of width `b-a<pi`, with physical phases `q_i mod 2*pi`.
Such a closed interval is contained in an open semicircle. The interval can
cross the displayed zero-phase cut, and need not have width below the U3
gate `pi/2`. In particular, this is a **global chart** hypothesis, not the
weaker condition that every existing support edge has an acute gap.

All angles, circular means and arithmetic in the proof initially mean exact
real operations. The implemented binary64 maps and detached chart observation
have separate boundaries below. Support may change among these same nodes.
No assumption about EPI equality, transport conductance or Sense Index is
needed for the phase-chart statement.

### Actual Coupling proposal and merge formulas preserve that chart

The target kernel first selects its U3-compatible existing neighbors. Its
phase consensus is the circular mean of either those neighbors alone or,
when bidirectional writes are enabled, the target together with those
neighbors. An admitted target has a nonempty compatible-neighbor set.
Every selected input is in I. After rotation by `-(a+b)/2`, each input
phasor has strictly positive real part. Their sum cannot be zero, and its
direction has a unique lift c between the minimum and maximum input lifts.
Thus `c in I` for either bidirectional setting.

For the declared push `f in [0,1]`, each proposed phase write has the form

```text
p_i = q_i + f*wrap(c-q_i) = (1-f)*q_i+f*c in I.
```

The second equality follows from `abs(c-q_i)<=b-a<pi`; the shortest arc is
the ordinary difference in this common lift. Reduction modulo `2*pi` is a
representation change, not motion out of the chart. This argument covers
the target and every optional bidirectional neighbor write. It does not
require the default factor specifically; the kernel's entire admitted
interval `[0,1]` is convex.

The simultaneous stage can give one node several proposals from different
targets. All proposals read the same snapshot and use the same original
`q_i`. If they are `p_i^(1),...,p_i^(r)`, the actual stage's idealized
shortest-arc displacement merge is

```text
q_i^+ = q_i + (1/r)*sum_alpha wrap(p_i^(alpha)-q_i)
      = (1/r)*sum_alpha p_i^(alpha) in I.
```

Each difference has magnitude below pi, so the second equality does not
identify distinct circle branches. Untouched nodes remain in I. Source-rank
ordering is relevant to represented summation but not to this exact convex
identity. The post-merge U3 validation remains a separate requirement: the
chart result neither promises that every requested stage is admitted nor
weakens the configured gate.

Functional-link candidates and their eventual weights are chosen separately
from these phase proposals. Adding edges between existing nodes whose final
phases stay in I cannot invalidate the common lift. This conclusion holds
regardless of whether EPI, Sense Index, candidate sampling or a threshold
selects a particular link. It establishes a phase constraint on the existing
selection policy, not a derivation or endorsement of that policy as an
emergent physical law. The fixed-node premise excludes importing a new node
with an independently prescribed phase outside I.

### Equal-capacity sine evolution preserves the moving chart

Between events consider the supplied averaged-sine law with one common
capacity/free angular rate `kappa>=0`, coupling `K>=0`, and whatever neighbor
subset `A_i(t)` the U3 gate currently admits:

```text
theta_dot_i = kappa + (K/d_i)*sum_(j in A_i) sin(theta_j-theta_i),
d_i = |A_i|,
```

with zero coupling contribution when `d_i=0`. In a frame rotating at kappa,
the same formula acts on lifted `q_i=theta_i-kappa*t` without the free term.
While the range is below pi, a maximal q_i has only nonpositive sine
contributions, and a minimal q_i has only nonnegative contributions.
Consequently the maximum cannot increase and the minimum cannot decrease.
The chart I is invariant in the rotating frame, and its width remains
strictly below pi.

This extremum argument does not require reciprocity, a fixed neighbor count
or a conserved phase mean. It remains valid when the gate changes the
admitted subsets, or when the support changes without a phase write. For
a discontinuous gate it states invariance for every absolutely continuous
trajectory satisfying the supplied equation almost everywhere; it does
not assert a new existence or uniqueness theorem for arbitrary switching
fields. Grammar and operator events retain their own admission conditions.

The shared helper also has a directly applicable exact-Euler statement. For
one simultaneous step with `h>0` and `h*K<=1`, put

```text
a_ij = sin(q_j-q_i)/(q_j-q_i),         q_j!=q_i,
a_ij = 1,                            q_j=q_i.
```

Within the common chart `0<a_ij<=1`. For a nonempty admitted set the ideal
update, after removing the common rotation `h*kappa`, is

```text
q_i^+ = (1-(h*K/d_i)*sum_j a_ij)*q_i
        + (h*K/d_i)*sum_j a_ij*q_j.
```

All coefficients are nonnegative and sum to one. An empty admitted set leaves
q_i unchanged in this rotating frame. Thus the Euler map preserves I for
arbitrary simultaneous admitted subsets under this step bound. The bound
is sufficient, not a declaration that larger steps always lose the chart.
The helper does not enforce `h*K<=1` for all callers; a composition invoking
this result must establish it separately. Sine of the raw displayed phase
difference equals sine of the lifted difference only in this exact-real
identity; represented argument reduction remains numerical evidence.

### Why neither new edges nor finite compositions can create a winding

For any oriented edge between nodes in I, its unique principal gap is
`q_head-q_tail`, since its magnitude is below pi. For every closed support
walk these differences telescope:

```text
sum_(edges in a closed walk) wrap(theta_head-theta_tail) = 0.
```

Every cycle period therefore vanishes, independently of the cycle basis.
This includes a cycle created by the next admissible link addition. Existing
cycles, new cycles and a different choice of spanning tree have the same
zero-period conclusion.

Induction now applies to any finite admitted composition consisting only of
the stated exact Coupling stages, common-capacity sine-flow intervals or
Euler steps satisfying their step condition, and support changes among the
same nodes that leave phases unchanged. There is a common moving phase
chart throughout, so none of these compositions generates a nonzero cycle
winding from the stipulated preparation. This conditional statement does
not require choosing a particular sequence of functional-link decisions.
It is not an unrestricted claim about all 13 operators or complete runtime
execution.

Section 31's initially prepared five-node path has phases
`0,2*pi/5,4*pi/5,6*pi/5,8*pi/5`. Its shortest containing circular arc has
width `8*pi/5`, exceeding pi. It is already outside this invariant chart
class, even though each consecutive path gap is acute. Before closure the
path has no cycle winding to measure. Closure makes that supplied,
non-semicircle phase arrangement observable as a new nonzero period; it
does not demonstrate its generation from a common-chart preparation.

### Common capacity is an ideal identity with an explicit numerical safeguard

The Coupling capacity proposal writes only requested targets. Bidirectional
phase writes do not themselves give every affected neighbor a capacity write.
Its declared blend, with `r in [0,1]`, is

```text
nu_target^+ = nu_target+r*(mean(nu_compatible_neighbors)-nu_target).
```

If every node has the same capacity kappa, this formula preserves kappa
exactly. Disabling capacity synchronization also preserves that supplied
common value. Hence Coupling does not create the heterogeneous free rates
that would escape the preceding **exact-model** premise.

The represented implementation requires an explicit equal-input identity.
Unguarded evaluation of `fsum(neighbors)/len(neighbors)` is not always
idempotent: three binary64 values `0.1` have the represented mean
`0.10000000000000002`. With blend factor one, a target-only write would
create a capacity difference from equal represented inputs. Summation can
also overflow for several equal large finite capacities even when their
mathematical mean is that same finite input.

The shared `coupling_capacity_blend` therefore returns the target value
directly when all compatible-neighbor capacities equal it. This is the
declared blend's equal-input identity, not a new capacity-selection law or
a change to nonuniform-input arithmetic. Both the direct and staged owners
reuse this numerical kernel. A runtime study must still retain the observed
capacities: this narrow safeguard does not certify unrelated capacity writes,
phase arithmetic or future invocations.

### A necessary capacity-contrast budget for leaving the common chart

Allow the supplied capacities in the same continuous sine law to vary by
node and time. Assume locally integrable finite rates and an absolutely
continuous lifted trajectory satisfying that law almost everywhere between
finitely many admitted Coupling events. Use unrotated real lifts for this
estimate. Up to the first chart boundary, let

```text
D(t) = max_i q_i(t)-min_i q_i(t) < pi,
osc(nu(t)) = max_i nu_i(t)-min_i nu_i(t).
```

The coupling contribution still points inward at each phase extremum.
Consequently the derivative of the maximum is at most `max_i nu_i(t)` and
that of the minimum is at least `min_i nu_i(t)` almost everywhere. It follows
that

```text
D'(t) <= osc(nu(t)),
D(t) <= D(0)+integral_0^t osc(nu(v)) dv.
```

Each exact Coupling phase event has `D(t^+)<=D(t^-)`, by the convex proof
above, so the same bound survives their insertion. Support-only changes
among the existing nodes do not change D. These statements concern every
trajectory satisfying the premises; they add no existence or uniqueness
claim for the discontinuous gate.

If `t_*` is the first boundary with `D(t_*)=pi`, a necessary condition is

```text
integral_0^t_* osc(nu(v)) dv >= pi-D(0).
```

If a uniform bound `osc(nu)<=R`, `R>0`, is supplied, this also gives
`t_* >= (pi-D(0))/R`. Zero contrast recovers the invariant common-chart
result. A capacity update can change the subsequent rate contrast without
an instantaneous phase jump; it contributes through the integral, not as
an invented phase displacement.

This budget is necessary, not sufficient. Inward coupling can prevent chart
escape even when the bound allows it. Losing the global common chart is
itself weaker than creating a nonzero cycle period or a maintained pattern;
the first limiting antipodal pair need not be a support edge. The budget
does not derive the supplied capacities or their evolution. It supplies a
checkable requirement for a proposed escape mechanism, not that mechanism.

### Exact chart observation, finite execution and the remaining obligation

[`phase_chart.py`](../src/tnfr/physics/phase_chart.py) observes whether the
captured phase set admits a common open semicircle. It interprets represented
radians as exact real inputs, retains mathematical pi through the shared
rational enclosures, and uses the equivalent criterion that the largest
empty circular gap exceed pi. Successful admission provides consistent lifts
and zero periods in the shared support-cycle coordinates. Exclusion of the
chart and an unresolved enclosure are separate outcomes; neither is a
positive winding or a successful formation result.

This detached observer does not establish how its state was produced, seal
a trajectory or prove that a future runtime event retains the same chart.
Its connected 2-to-32-node, at-most-50-edge implementation budget is narrower
than the chart theorem. The actual arithmetic owners are
[`_coupling_stage_kernel.py`](../src/tnfr/operators/_coupling_stage_kernel.py)
and [`phase_evolution.py`](../src/tnfr/dynamics/phase_evolution.py); the
[formation controls](../tests/physics/test_phase_chart_formation.py) exercise
their finite behavior separately from the exact composition argument.
Phasor evaluation, shortest-arc subtraction, displacement averaging, modulo
writes and Euler rounding must not be silently identified with exact convex
maps. In particular, the equal-capacity safeguard alone gives no binary64
phase-chart theorem for arbitrary repetition.

Generating nonzero winding from the invariant preparation requires leaving
at least one of its premises. Heterogeneous free rates can remove the common
rotation; a phase-changing map outside the proved convex family can remove
the chart invariant; and an introduced or prepared phase outside the chart
changes the initial-data hypothesis. None of these possibilities is by
itself a sufficient generative mechanism. Topology changes alone among the
same common-chart phases cannot supply the missing period. A large numerical
step or accumulated rounding discrepancy is also not a derived physical
source. Any proposed escape must identify its actual operator/state
preconditions or constitutive law and retain the independent evidence for
formation, maintenance and selection.

### Finite controls for overlapping writes and the excluded preparations

The production control fixes a five-node path with displayed signed phases
`(-0.30,-0.12,0.04,0.16,0.29)` represented modulo tau, common capacity one,
EPI `1/8`, Si `0.8`, seed 17 and declared initial AL histories. One actual
default simultaneous Coupling stage targets `(1,3)` with all eligible
candidates; no factor or compatibility threshold is tuned. Node 2 receives
overlapping phase proposals. The stage adds `(0,3)`, `(1,3)` and `(1,4)`:
cycle rank rises from zero to three while every fundamental period is zero.
The exact endpoint observer proves both common charts; their diameters shrink.

Independent 80-digit circular-mean and convex-merge calculations agree with
the represented writes within `2e-15` radians while retaining a nonzero
arithmetic difference. Reversing target order preserves primary endpoints.
Four subsequent shared joint steps at `h=1/8`, `K=1/2` retain the observed
common chart, common capacity and zero cycle periods. These finite checks
neither seal future execution nor turn the exact model into a binary64 theorem.

A separate three-node path starts with equal zero phase and supplied positive
capacities `(1,2,3)`. One phase proposal at `h=2`, `K=1/2` gives `(2,4,6)`
radians, outside every common open semicircle. This control violates the
common-capacity premise and is a tree: it demonstrates possible chart escape,
not winding birth, an exact continuous trajectory or autonomous contrast.
Section 31's prepared wound path and its executed C5 endpoint are also
correctly excluded from this obstruction's initial class.

Geometry controls retain repeated phases, node relabeling, a chart crossing
zero and transport-weight independence. The represented pair `(0,math.pi)`
is slightly narrower than a mathematical semicircle and is admitted with a
small positive exact margin. A valid deliberately coarse pi enclosure instead
returns `unresolved` where required, with unavailable periods rather than zeros.
[Direct/staged capacity regressions](../tests/operators/test_coupling_jacobi_stage.py)
confirm that the repaired shared capacity identity retains `0.1` at a
three-neighbor target, plus zero, subnormal and extreme finite kernel inputs.

## 34. Native runtime admission uses relaxation, not the supplied sine clock

### Reuse the native owners before transferring the oscillator result

The preceding sine-law results do not identify the phase evolution used by
`runtime.step`. That entry point calls
[`coordinate_global_local_phase`](../src/tnfr/dynamics/coordination.py), a
configured relaxation per invocation. It does not call the averaged-sine
proposal, add `dt*nu_f` to phase, or use U3 to filter the coordinator's local
neighbors. Capacity contrast therefore is not, by itself, a native angular
speed contrast. Section 33's accumulated capacity-contrast budget concerns
its supplied oscillator law and cannot be transferred to this native map.

The existing
[native diameter bound](NODAL_PARAMETER_FOUNDATIONS.md#native-phase-contrast-budget),
[writer audit](NODAL_PARAMETER_FOUNDATIONS.md#native-phase-writer-closure) and
[selector reachability result](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#uniform-capacity-and-default-selector-reachability)
remain the mathematical and policy owners. This section connects those
results to the complete step boundary, extends the represented common-capacity
identity beyond the earlier retained unit value, and specifies what an actual
native formation check must retain. It introduces no substitute phase law.

### The actual step order and freshness boundary

For the built-in path, [`runtime.py`](../src/tnfr/dynamics/runtime.py) performs
the following operations in order:

1. Before-step callbacks; mutation-flow boundary and node-sample bookkeeping.
2. The configured pressure refresh, then Si refresh when `use_Si=True`.
3. Selector decisions, lag overrides, grammar processing and sequential
   primitive glyph execution; another mutation-flow boundary is recorded.
4. The configured integrator, followed by EPI/capacity clamps and phase
   re-expression.
5. Global/local phase coordination, followed by the stored-pressure/Si
   capacity-adaptation gate and another mutation-flow boundary.
6. The optional auxiliary math step, delayed-EPI history, automatic REMESH,
   validators, after-step callbacks and cache telemetry.

There is no intervening pressure/Si refresh after the glyphs, EPI integration
or phase coordination. Under ordinary IL execution, the integrator consumes
the operator-contracted stored pressure. AL/EN can change form without
refreshing that pressure. The later adaptation gate reads the resulting
stored pressure together with Si computed before those state changes.
Automatic REMESH can introduce a further EPI jump after that gate. These
values identify their actual observation times; calling them fresh terminal
state pressure and Si would be incorrect.

`use_Si=False` suppresses the refresh, not consumers of stored Si.
`apply_glyphs=False` suppresses glyph execution, not subsequent coordination
and adaptation. The native integrator owns physical EPI time, whereas the
coordinator's gains act once per invocation without a `dt` argument. Changing
the number of native steps can therefore change the number of relaxation
maps even at a fixed total EPI integration duration.

### Default coordination stays in the common chart and consumes its diameter

In a common open-semicircle lift, the exact global phasor mean `b` and every
nonempty local mean `l_i` lie in the same interval; isolates use their own
phase. The actual coordinator's ideal formula is

```text
q_i^+ = (1-kG-kL)*q_i + kG*b + kL*l_i.
```

The existing diameter proof applies when `kG,kL>=0` and `kG+kL<=1`:
`D^+<=(1-kG)*D`. It is stronger than merely retaining the chart. The default
enabled adaptive policy clamps both gains after its stable, transition or
dissonant branch. Their nominal configured bounds are

```text
1/(8*pi^2) <= kG <= 1/(2*pi),
1/(8*pi)   <= kL <= 1/(2*pi).
```

For executable inequalities the endpoints are their stored binary64 values,
treated as exact coefficients. Their global lower endpoint `m` is positive
and their two upper endpoints sum to less than one. Thus every default
adaptive branch supplies the exact coefficient bound `D^+<=(1-m)*D` for
the ideal map. Kuramoto order and recent disruptive-glyph load choose the
adaptive branch; neither branch supplies an outward phase source in this
domain. The separate fallback defaults for a partially configured policy
must not be substituted for the fully injected default configuration.

These facts are already checked by the
[native diameter controls](../tests/physics/test_native_phase_diameter_budget.py).
They do not guarantee arbitrary configured overrides, disabled adaptation,
modified gain bounds, undefined represented resultants or all binary64
trajectories. Legacy native coordination uses ordinary reductions; its
optional exact-component evidence version is not silently enabled by
`runtime.step` and does not certify transcendental phase accuracy either.

### Which native glyphs and other writers preserve the premises

With fresh default Si and equal positive capacities, its capacity-normalized
term equals one and the existing selector bound gives a base IL choice.
For initialized nodes, meaning nonzero scalar EPI or valid retained nonempty
glyph history, IL is admitted by the incremental grammar: U1a no longer
requires a generator, and IL supplies stabilization rather than new debt.
The default selector has no parametric soft-repetition filter. Lag overrides
can request AL or EN; if one is rejected in this initialized class, IL is
the first admitted fallback. Thus the relevant reachable primitive set is
`{IL,AL,EN}` under those hypotheses. Outside that class, inspect the actual
applied labels and grammar outcome; a label appearing later in the fallback
list does not prove either its occurrence or its impossibility.

The native dispatcher does not invoke the public `Coherence` class or an
all-target operator stage. `_op_IL` changes only stored pressure; `_op_AL`
changes form. For a graph-bound EN call, the adapter captures a
`ReceptionReadSnapshot` and its prepared writer changes form and its kind.
These paths leave phase, capacity and support unchanged. Their history and
cache writes do not constitute a phase update. A public operator with an
additional configured phase action is a different path.

The built-in default integrator advances scalar EPI, its rate bookkeeping
and the runtime clock, holding phase and capacity fixed. The ordinary clamps
apply common capacity bounds to every node. An equal positive capacity
already inside those bounds therefore remains that same value. Exact phase
wrapping preserves the circular state; its represented implementation is
addressed below. No EPI stability or eventual EPI equilibrium follows just
from these phase/capacity facts.

Automatic network REMESH is the protected delayed-EPI map. It does not call
topological remeshing or the separate structural-memory phase/capacity
interpolation. Its transaction protects phase, capacity and support against
its `ON_REMESH` observers. The standard optional math engine advances a
separate state vector and telemetry without an inverse projection onto the
nodal phases or capacities. Arbitrary custom pressure hooks, integrators,
selectors, mutable-object implementations, validators and before/after
callbacks remain additional writers whose effects require their own audit.
The theorem excludes such unaccounted changes rather than treating a callback
name or a suppressed exception as evidence of a read-only operation.

### Preserve the declared adaptation fixed point in represented arithmetic

The adaptation owner uses a held neighbor-capacity snapshot and eligibility
from stored pressure, stored Si and a consecutive-call counter. In exact
arithmetic, a common capacity kappa is unchanged by

```text
nu_i^+ = (1-mu)*nu_i + mu*mean(nu_neighbors),      0<=mu<=1,
```

regardless of which subset qualifies. This does not require the stored gate
inputs to describe the latest phase/form state: arbitrary eligibility still
leaves that equal-capacity fixed point unchanged.

A represented implementation must preserve this identity explicitly.
For example, with target and neighbor capacities `0.104` and default
`mu=0.1`, the two-product expression can return `0.10400000000000001`.
If only that target is eligible, it creates a numerical contrast between
previously equal capacities. This is not evidence of a structurally generated
capacity mechanism. The shared adaptation proposal retains the current value
when its represented neighbor mean equals that value, avoiding the redundant
multiply/add operations. Section 38 additionally enforces the neighbor and
convex-proposal intervals for nonuniform inputs. Interior arithmetic and the
eligibility policy remain unchanged; neither guard changes a threshold or
event-selection law.

Phase clamps have a corresponding representation boundary. The native
contract re-expresses phases in `[-pi,pi)`, not necessarily `[0,2*pi)`.
Recomputing `(theta+pi) % (2*pi)-pi` for an already valid centered value can
lose small increments unnecessarily. The clamp therefore retains that valid
value directly and uses the existing re-expression for values outside the
centered range. The coordinator can subsequently store a raw unwrapped
proposal. This guard preserves the existing range contract where it applies;
it is neither a phase evolution law nor a guarantee of exact re-expression
at other branches.

The detached common-chart observer currently accepts canonical represented
inputs in `[0,math.tau)`. A centered or raw native endpoint outside that
input domain is not admitted merely because a periodic interpretation exists.
Finite controls that use this observer must state their actual input range;
they may not normalize the live graph just to manufacture an admissible
observation. A separately retained compatible lift and its representation
error are different evidence.

### Conditional whole-step conclusion and its limits

Start with initialized scalar nodes, fixed existing support, equal positive
capacity inside the configured rails and a common open-semicircle phase
chart. Retain fresh default Si, default selection and adaptive gains, the
built-in integration path and no unaccounted phase/capacity/support writer.
For each admitted **exact-model** native step, the primitive glyphs preserve
these phase/capacity conditions, adaptation preserves common capacity and
the coordinator contracts the chart diameter. For any finite number N of
such steps,

```text
D_N <= (1-m)^N*D_0 < pi.
```

Every support-cycle period remains zero. Indefinite continuation satisfying
these same ideal premises would exhaust phase contrast, rather than renew
it. This conditional conclusion is the existing native theorem with its
actual writer/admission requirements; it is not a new autonomous formation
law or an unrestricted claim about every grammar history and runtime option.

The numerical fixed-point and clamp repairs remove identifiable arithmetic
defects from this path. They do not make phasor evaluation, shortest-arc
arithmetic, normalization or complete runtime repetition exact. A finite
native experiment must retain actual glyphs, effective gains, primitive
phase/capacity endpoints, pressure/Si freshness and every intervening writer.
The existing diameter ledger then keeps phase realization and intervening
changes as explicit defects; it is not permission to assign unobserved
changes a zero value. Whether a different admitted native mechanism can
supply and maintain phase contrast remains a separate formation question.

### Finite ordinary-step control

[Native step controls](../tests/physics/test_native_step_formation.py) execute
two independently initialized unit-conductance C5 graphs with seed 17, actual
injected defaults, common capacity `0.3`, phases
`(0.10,0.13,0.11,0.17,0.14)` and EPI
`(0.125,0.128,0.121,0.127,0.123)`. The cases differ only in supplied stability
counters: all zero versus `VF_ADAPT_TAU-1`. Those initial counters do not
certify preceding stability. The default `dt=0.5`, selector, pressure reader,
Si weights, coordinator, adaptation and REMESH gate are retained. Transparent
wrappers observe the original functions without replacing their decisions.

Both executions refresh pressure and Si, apply native IL at every node and
advance EPI nontrivially through the nodal integrator. The coordinator takes
its adaptive stable branch with represented gains
`kG=0.04461342911494398`, `kL=0.140157221158957`. The observed chart diameter
falls from approximately `0.07` to `0.05566448226923737`; the exact endpoint
observer separately admits both charts and finds zero cycle period. All
observed phases remain in the observer's input range without re-expression.
Support and capacity remain unchanged. Only node 4 passes the pressure/Si
gate; only the counter-ready case reaches the capacity writer, which retains
`0.3` exactly. Its counter advances rather than being reset.

The trace also retains the timing limitations: adaptation reads the IL-scaled
stored pressure and pre-glyph Si. The detached terminal pressure comparison
has a nonzero freshness residual. Mutation-flow timestamps and delayed-EPI
history contain the actual integrated endpoints. Automatic REMESH abstains
because the required stable-fraction history is absent; its gate was executed,
but this control does not claim an applied REMESH map. These are two finite
whole-step observations, not a proof of arbitrary binary64 repetition,
unobserved prior history or autonomous pattern formation.

## 35. Heterogeneous capacity opens a conditional Mutation admission gate

Section 34 closes a uniform-capacity class; it does not establish that fresh
default selection always excludes phase writers. The question addressed here
is whether heterogeneous supplied capacities can simultaneously satisfy the
default selector, live Mutation evidence and incremental grammar. The owners
remain [native selection](../src/tnfr/dynamics/selectors.py),
[Mutation evidence](../src/tnfr/physics/mutation_trigger.py),
[its runtime adapter](../src/tnfr/operators/_mutation_gate.py) and
[incremental grammar](../src/tnfr/operators/grammar_dynamics.py). Admission
of an endpoint and production of its preceding history are separate results.

### The ideal fresh-state compatibility inequality

Let `V=max_i(nu_i)>0`, `P=max_i(abs(p_i))`, `r_i=nu_i/V`, and
`z_i=abs(p_i)/P` when `P>0`. Let `d_i` be the absolute wrapped displacement
from the target phase to its neighbor-phasor mean, divided by pi. This is
the ideal phase dispersion used in the default Si definition; an isolate
has zero dispersion. Write `a,b,c` for the normalized default Si weights,
approximately `(0.75854699,0.18315345,0.05829955)`. Then

```text
Si_i = a*r_i + b*(1-d_i) + c*(1-z_i).
```

With no lag override, the default selector proposes ZHIR exactly when
`Si_i<=lo` and `z_i<=h`, where the injected thresholds are `lo=0.25` and
`h=0.133`. Thus its ideal low-Si condition is equivalent to

```text
r_i <= [lo-b*(1-d_i)-c*(1-z_i)]/a,       z_i<=h.
```

If `P=0`, both normalization owners use their zero-pressure fallback and
`z_i=0`; no division by zero or positive-pressure premise is required.
Mutation still requires strictly positive target capacity. Its additional
configured floor `ZHIR_MIN_VF` defaults to zero; `0.1` is the signed growth
threshold `ZHIR_THRESHOLD_XI`, not a capacity floor.

In an exact common open-semicircle chart of diameter `D<pi`, every defined
neighbor mean lies in that chart, so `d_i<=D/pi`. A necessary bound is

```text
r_i <= R(D) = [lo-b*(1-D/pi)-c*(1-h)]/a.
```

For the nominal defaults, `R(0)` is approximately `0.02148955`,
`R(pi/4)` approximately `0.08185280`, and its limiting value as `D` tends
to pi approximately `0.26294256`. These are restrictions on relative
capacity, not a proof of selection: current pressure, history, grammar and
represented numerical margins still have to be checked. An additional
capacity floor larger than `V*R(D)` would exclude this chart's gate.
Executable comparisons use captured represented coefficients and thresholds;
the rounded displays above are not directed interval certificates.

For the pressure bound, use the separate effective pressure weights
`A,B,C`, whose nominal defaults coincide with `a,b,c`. On nonnegative
conductance and support with defined phase resultants, the ideal default
pressure is

```text
p_i = A*g_i + B*(weighted_neighbor_mean(x)-x_i)
              + C*(neighbor_mean(nu)-nu_i),
abs(g_i) <= D/pi,
P <= A*D/pi + B*(x_max-x_min) + C*(V-nu_min).
```

The capacity difference is raw capacity, **not** the normalized Si quantity
`r_i`. The EPI mean uses transport conductance; the phase and capacity means
use support neighbors. The default topology coefficient is zero. The bound
therefore uses the actual EPI and capacity ranges, including their units;
replacing the last term by `C` would generally be incorrect. With the default
rails it is at most `A*D/pi+2*B+C*VF_MAX`, where `VF_MAX` is the stored
binary64 value near `2*pi`. This bounds fresh current pressure only.

In particular, a small current product `nu_i*p_i` does not invalidate a
larger previously observed signed EPI secant. Mutation validates two retained
samples ending at the live EPI, with timestamped evidence authoritative.
The preceding interval may have had different pressure, phase or capacity.
Conversely, supplying a valid sample pair does not establish that native
execution generated it. EPI rails alone do not exclude the history gate:
two samples in `[-1,1]` can exceed the `0.1` threshold over an appropriately
short positive interval. No measured derivative is reconstructed from
current pressure to manufacture this evidence.

### Which additional reachable primitives can supply a phase write

For initialized nodes and fresh default selection, the base choices are
`{IL,OZ,ZHIR,NAV,RA}`. Lag overrides add AL and EN. An incremental-grammar
rejection has IL as its first admitted fallback in this class; it does not
make every later fallback label reachable. Successful native primitive
writes have the following scope:

| Reachable glyph | Direct state writes relevant here | Support change |
| --- | --- | --- |
| IL, OZ, NAV | Stored pressure | None |
| AL, EN | EPI, with EN's form-kind contract | None |
| RA | EPI, optional capacity amplification and phase alignment | None |
| ZHIR | Phase, after live evidence and grammar admission | None |

This is the primitive dispatcher audited in section 34, not the public
operator classes, arbitrary hooks or an all-target stage. In particular,
the native selector does not directly choose UM or a child-creating THOL
operation. Optional network REMESH retains its separate protected EPI scope.

The [RA primitive](../src/tnfr/operators/__init__.py) uses only its actual
U3-compatible neighbor subset. In a common chart, its ideal phase write
has the form `q_i^+=(1-eta)*q_i+eta*b_i`, where `b_i` is that subset's
phasor mean and default `eta=1/(4*pi)` lies in `[0,1]`. It therefore cannot
expand the common lift hull. This remains true for successive admitted
primitive writes; it does not remove represented mean/wrap errors.
Fresh default RA selection also requires `Si_i<0.5`. Nonnegativity of the
other Si terms gives `nu_i/V<0.5/a`. Its optional default multiplier
`1+1/(8*pi)` consequently leaves that target below approximately
`0.68538189*V`, with `V` from the selection snapshot. This limited bound
does not supply a general capacity-convergence theorem.

ZHIR supplies a different write. Its
[shared proposal](../src/tnfr/operators/_mutation_stage_kernel.py) uses the
current pressure sign and the default configured shift factor, giving a
shift near `+0.25` or `-0.25` radians. The trigger is the observed signed
EPI growth, not the sign or magnitude of this current pressure. Positive
growth, phase-shift direction and subsequent useful source work must not
be identified with one another.

### Declared-history endpoint: the admission intersection is nonempty

Consider connected unit-conductance P4 with nodes `(0,1,2,3)`, injected
defaults and the following supplied state:

```text
capacity = (0.6, 1.2, 6.0, 6.0)
phase    = (2.5, 1.0, 0.2, 2.0)
EPI      = (-0.85, 0.95, -0.9, 0.9).
```

Supply node 0's glyph history `IL,OZ` and timestamped EPI pairs
`(-0.5,-0.95),(0.0,-0.85)`, with zero lag counters before selection.
These histories are declared initial data, not a claimed native prefix.
An independent binary64 endpoint refresh gives approximately

```text
pressure = (0.00247643659, -0.12731743356, 0.50822502874, -0.76429162695)
Si       = (0.22966947381,  0.36304588940, 0.88544357994,  0.83676124714)
base     = (ZHIR, NAV, IL, IL).
```

At target 0, the endpoint pressure can also be checked directly as
`-A*1.5/pi+B*1.8+C*0.6`. The shared history gate reports an observed rate
approximately `0.2`, above `0.1`, while the current nodal product is only
approximately `0.00148586195`. The incremental grammar admits ZHIR for the
declared history. All supplied EPI and capacity values are inside the
default rails. The exact common-chart observer admits diameter `2.3<pi`.
Thus the selector, live history, capacity and grammar conditions are jointly
compatible; the equal-capacity obstruction does not extend to this endpoint.

The admitted default positive shift would move target 0 from `2.5` to
`2.75`, outside the previous lift hull `[0.2,2.5]`. The new diameter is
still `2.55<pi`: outward phase motion is not a chart-crossing claim.
P4 is a tree, so its zero support-cycle winding is vacuous. The maintained
native execution controls below use a separate declared graph; endpoint
arithmetic and read-only admission do not certify an applied P4 batch or
a complete native P4 step.

### A finite native prefix produces its own Mutation evidence

The [heterogeneous Mutation controls](../tests/physics/test_heterogeneous_mutation_admission.py)
retain one prospective six-node case: a complete graph on nodes `(0,1,2,3,4)`
with pendant node 9 joined only to node 1. All conductances are one. Inject
defaults, seed 17, initial runtime time zero, empty glyph histories and no
supplied Si. Initialize

```text
node 0:    phase=0.0, EPI= 0.6, capacity=0.8
nodes 1-4: phase=2.8, EPI=-1.0, capacity=0.1
node 9:    phase=4.7, EPI= 1.0, capacity=6.25.
```

Three ordinary native steps, each with the default physical interval `0.5`,
produce the target-0 sequence `IL,OZ,ZHIR`. The pressure reader, fresh Si,
selector, grammar, nodal integrator, coordinator and adaptation remain the
original owners; observation wrappers do not replace their decisions.
Before the three target decisions the retained fresh values are approximately

| Native decision | Target pressure | Target Si | Global maximum absolute pressure |
| --- | --- | --- | --- |
| IL | 0.34221320754 | 0.15845231195 | 1.18360987888 |
| OZ | 0.18135858879 | 0.19478282220 | 0.70716117563 |
| ZHIR | 0.04528547527 | 0.23430789739 | 0.63220290369 |

The first two low-Si base choices request OZ; the initial lack of a recent
handler makes grammar supply IL first. The next OZ is then admitted. Before
the third decision, normalized target pressure is below `0.133`, and the
actual timestamped history supplies a positive secant approximately
`0.19126945644>0.1`. Its current nodal product is only approximately
`0.03622838022`; that prediction is not substituted for the observed rate.
The actual preceding `IL,OZ` supplies the grammar context. This is a native
history-producing trace, unlike the P4 endpoint's supplied histories.

The paired control changes only all initial capacities to `0.8`; it yields
target `IL,IL,IL`. This comparison identifies a finite consequence of
supplied capacity heterogeneity under the same configured policy; it does
not establish how that initial heterogeneity arose.

The initial phases `(0,2.8,4.7)` do not admit a common open-semicircle chart.
Therefore this case is not an escape from section 34's common-chart class.
Before the third decision, however, an independent exact-pi lift check
already admits a common chart. The target phase is
`1.063472935200046`; its neighbors lie between `2.751877932766905` and
`2.8637878070029825`. The actual ZHIR jump is `+0.25`, toward every one
of those neighbors. The pendant's raw phase `-2.2705373138792972` requires
adding true `2*pi`, enclosed by the shared rational pi bounds. This check
does not pass an unsupported negative value to the detached common-chart
observer or replace true `2*pi` by represented `math.tau`.

The retained lift's diameter is approximately `2.949175<pi` before ZHIR
and decreases by `0.25` at that phase-only event. This causal occurrence
therefore supplies observed phase alignment, not outward hull renewal.
Capacity remains heterogeneous, so the event does not contradict section
34's equal-capacity selection restriction. A new cycle period, coherent
identity or continued source renewal does not follow. The ensuing
coordinator's contribution and the corresponding source-work balance
are measured in section 36.

### From admission to the retained source budget

The native trace closes finite occurrence with genuine temporal evidence;
section 36 measures whether that actual write renews source work and how the
ensuing relaxation changes it. One local distinction is already exact.
Hold a neighbor-phase center fixed, choose a consistent branch, and write
`u=center-phase` and `delta` for the admitted Mutation shift. Provided the
before/after separations remain on that branch,

```text
(u-delta)^2-u^2 = delta^2-2*u*delta.
```

For the default direction `sign(delta)=sign(p)`, opposite signs of `p` and
`u` increase this squared separation. Matching signs decrease it unless
the shift overshoots sufficiently. This is a fixed-center local identity,
not a global hull, winding, pressure-work or field-energy theorem. The P4
endpoint has `u=-1.5` and positive pressure, but a positive Mutation shift
can instead align a target whose neighbor center lies ahead. Triggering
Mutation therefore does not imply contrast renewal.

The completed budget in section 36 reuses this actual third-step case and its
uniform-capacity control, retaining phase endpoints around the glyph batch
and coordinator. It introduces no additional case or longer horizon. The
[forcing capture](../src/tnfr/physics/forcing_realization.py) reports a
pressure vector `F`, not `diag(nu)*F`. At fixed EPI, capacity and support,
the prospective refreshed Dirichlet-rate increment is
`(B*x)^T*diag(nu)*Delta_p`; a unit-capacity formula cannot be copied directly
to this heterogeneous case. The instantaneous EPI energy remains unchanged.
Moreover, ZHIR retains stored pressure: the ensuing held-pressure interval
does not automatically use that refreshed-law increment. Keep actual flow
provenance separate from the detached fresh-source comparison.
Do not replace the phase source by its linear comparison
outside an admitted chart or attribute earlier glyphs' changes to ZHIR.
The observed secant, current nodal product, operator jump and subsequent
relaxation remain separate evidence. This bounded continuation does not
authorize an open-ended search for a favorable trace.

A reachable configured mechanism does not derive the Si policy or prove
repeated renewal of a coherent pattern. The formation objective still
requires useful retained source work against relaxation and its maintenance
under the same admitted execution.
This section adds no autonomous selection law, substrate-creation result,
particle identification or new active research branch.

## 36. Native Mutation and coordination in the Dirichlet source budget

The native occurrence in section 35 is now examined through the existing
pressure and energy owners, using the same three-step heterogeneous case
and its uniform-capacity control. No additional initial condition, longer
run, substituted glyph or revised selector is introduced. The distinction
is between a phase event's effect on a prospective fresh nodal rate and
the work of the pressure actually retained by the following integrator.

### One detached channel ledger, with heterogeneous mobility retained

[Forcing realization](../src/tnfr/physics/forcing_realization.py) owns
`ForcingDirichletBalance` and `observe_forcing_dirichlet_balance`. The
observer consumes an existing `NonEpiForcingObservation`, reuses its shared
channel decomposition and rebuilds the transport snapshot. It performs no
graph evolution, pressure recapture or new phase-kernel call. The primitive
full-pressure vector remains detached data; cached gradient, energy and
pressure-defect fields are recomputed rather than trusted as evidence.

For fixed symmetric nonnegative conductance, write `B=D-W`, `x=EPI`,
`E_D=x^T*B*x/2`, `nu` for the capacity vector and `e` for the EPI pressure
weight. The existing EPI gradient is `g_epi=-D^(-1)*B*x`, with zero rows
at zero strength. Let `F` be the sum of the captured phase, capacity and
topology pressure channels. Define

```text
p_model  = e*g_epi + F
eps_k    = p_fresh - p_model
eps_s    = p_stored - p_fresh
score    = diag(nu)*B*x
R_epi    = score dot (e*g_epi)
R_source = score dot F
R_model  = R_epi + R_source
R_fresh  = R_model + score dot eps_k
R_stored = R_fresh + score dot eps_s.
```

The three source-channel works are reported separately and retain their
signs. The final rate agrees exactly with the rebuilt snapshot's stored
nodal energy rate. These are rational identities for represented input
values, not claims of exact transcendental phase evaluation, authenticated
history or universal runtime stability. The captured kernel discrepancy
is relative to the represented channel reference, not an error enclosure
against an exact-real phasor model. Zero capacity and zero-strength rows
retain their existing degenerate-mobility convention.

The EPI term satisfies

```text
R_epi = -D_loss,
D_loss = e*sum_{d_i>0}(nu_i/d_i)*(B*x)_i^2 >= 0.
```

Thus instantaneous nondecrease for the prospective refreshed nodal row
requires `R_source + score dot eps_k >= D_loss`. A positive change in
phase-source work alone is insufficient; capacity-source work, diffusion
and the declared numerical reference matter too. This criterion concerns
Dirichlet form contrast. It does not identify that contrast with coherent
identity, tetrad energy, a particle or a maintained NFR.

At fixed EPI, capacity and support, a phase-only ZHIR event has zero
instantaneous EPI-energy jump. It can nevertheless change the fresh phase
pressure at its target and at neighbors whose resultants include that
target. The channel ledger retains those nonlocal contributions instead
of inferring source work from the target's phase displacement alone.
Stored pressure remains unchanged by ZHIR, so its held nodal rate does
not automatically become the newly observed fresh rate.

### Reuse the finite Euler endpoint identity

The existing
[`observe_support_transport_euler`](../src/tnfr/physics/support_transport.py)
owns the finite energy accounting. With fixed conductance and capacity,
held `r=diag(nu)*p_stored`, interval `h`, exact reference `y=x+h*r` and
observed endpoint defect `delta=x_after-y`, it reports

```text
E_D(x_after)-E_D(x)
  = h*R_stored + h^2*r^T*B*r/2
    + (B*y) dot delta + delta^T*B*delta/2.
```

The linear drift, nonnegative Euler quadratic contribution and signed
endpoint-defect contribution remain separate. The last term can contain
rounding, clipping or other deviations from the declared reference. The
algebra alone does not identify their origin. A distinct clipping budget
would require an observed raw pre-clip endpoint; no such intermediate
endpoint is inferred from the final value. Likewise, an instantaneous
rate sampled at an endpoint is not an integral of refreshed rates along
the actual held-pressure interval.

The quadratic contribution is also present under exact continuation with
constant held rate; it is not automatically a numerical error. A
nonpositive initial energy rate therefore does not exclude positive finite
energy change. At consensus, for example, the initial rate is zero while
a nonuniform held forcing can generate positive energy quadratically.
The instantaneous source criterion above must not replace the integrated
or finite endpoint budget when assessing maintenance.

### What the retained third step shows

The enhanced
[native admission controls](../tests/physics/test_heterogeneous_mutation_admission.py)
retain the actual target event, integrator and coordinator boundaries from
section 35. The third heterogeneous batch applies
`(ZHIR,OZ,IL,IL,IL,IL)` in node order `(0,1,2,3,4,9)`. Thus the pressure
at flow start also includes the later nodes' pressure writes; it is not
attributed solely to the target Mutation. Approximate global rates are:

| Boundary | EPI term | Phase work | Capacity work | Stored-residual work | Fresh rate |
| --- | --- | --- | --- | --- | --- |
| Before target ZHIR | `-1.939783470` | `+2.456864795` | `-0.207571077` | `0` | `+0.309510247` |
| After target ZHIR | `-1.939783470` | `+2.107507269` | `-0.207571077` | `+0.349357525` | `-0.039847278` |
| Subsequent flow start | `-1.939783470` | `+2.107507269` | `-0.207571077` | `+0.326643961` | `-0.039847278` |
| Before coordinator | `-1.974221035` | `+2.138489438` | `-0.192984463` | `+0.340073217` | `-0.028716059` |
| After coordinator | `-1.974221035` | `+1.660508618` | `-0.192984463` | `+0.818054037` | `-0.506696879` |

Topology work is zero at these boundaries. The retained kernel-rate
defects have magnitude below `6e-16`; the table's decimals are rounded
displays, while each underlying rational ledger has zero identity residual.

The target ZHIR write preserves EPI, capacity, conductance and stored
pressure. Its observed phase alignment therefore changes the prospective
fresh rate in the dissipative direction while leaving instantaneous energy
and the stored rate unchanged. The following coordinator also preserves
EPI energy at its phase event and makes its own prospective fresh rate
more negative. These comparisons use each event's actual EPI endpoint;
the intervening integration changes `B*x`, so their work increments must
not be added as if they used one unchanged score.

The heterogeneous run's actual EPI-energy increase is accounted for by
its positive retained-pressure drift, Euler quadratic contribution and
endpoint defect. At flow start, `R_stored` is approximately `+0.286796684`.
The finite third-flow accounting is:

| Contribution | Heterogeneous case | Uniform-capacity control |
| --- | --- | --- |
| Held drift `h*R_stored` | `+0.14339834185` | `-0.12701644047` |
| Held-flow quadratic term | `+1.13459958299` | `+0.01902551467` |
| Combined endpoint-defect term | `-1.16456266791` | approximately `-1.38e-15` |
| Observed energy change | `+0.11343525693` | `-0.10799092580` |

Both exact endpoint identities have zero residual. The heterogeneous
endpoint-defect contribution is substantial; without an actual raw
pre-clip capture it is not assigned entirely to clipping or to rounding.
The observed increase is not evidence that the aligning Mutation or
coordinator supplied a positive fresh-rate increment. The uniform control,
which applies IL at every node in all three steps, instead loses EPI energy
over its corresponding interval. This is a finite
comparison of the supplied capacity configurations and their resulting
trajectories, not a universal claim that heterogeneous capacities increase
energy or that homogeneous capacities always decrease it.

[Detached ledger controls](../tests/physics/test_forcing_dirichlet_balance.py)
independently check heterogeneous mobility, signed channel contributions,
fresh-versus-stored pressure and reconstruction of altered caches. They
also retain the distinction between exact arithmetic over detached data
and evidence of a live kernel or causal execution.

### Interpretation boundary

This retained case demonstrates native Mutation occurrence, observed phase
alignment and a finite EPI increase under held pressure. The fresh-law and
executed-flow ledgers explain why these observations coexist. Neither
the energy increase nor the operator label establishes sustained renewal
of a coherent identity. The unresolved modeling distinction concerns how
pressure refresh and the interval law contribute to retained source work;
changing that law would be a declared model change, not an explanation
retroactively applied to this trace. The sole task queue remains the
[execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md).

## 37. Separate the frozen hard-rail reference from its represented endpoint

Section 36 leaves a substantial combined endpoint defect in the same
heterogeneous third interval. A detached exact reference can separate the
energy effect of the configured hard rails from the remaining endpoint
difference without inventing an internal clipping trace. This section uses
that interval and its existing uniform-capacity control only.

### Frozen constant-velocity hard clipping has an exact semigroup

Let the common scalar rail be `[L,U]`, with `L<=U`, and define
`C(a)=min(U,max(L,a))` componentwise. Hold the velocity vector `v` fixed,
start with every `x_i` inside the rail, and put

```text
T_t(x) = C(x+t*v),       t>=0.
```

For a coordinate with `v_i>=0`, the lower rail is inactive and

```text
T_s(T_t(x))_i
  = min(U, min(U,x_i+t*v_i)+s*v_i)
  = min(U, x_i+(s+t)*v_i)
  = T_(s+t)(x)_i.
```

If the inner value has reached U, nonnegative `s*v_i` leaves it there;
otherwise both expressions apply the same upper cap to the same sum.
For `v_i<=0`, the corresponding lower-cap calculation is

```text
T_s(T_t(x))_i
  = max(L, max(L,x_i+t*v_i)+s*v_i)
  = max(L, x_i+(s+t)*v_i).
```

Zero velocity satisfies both cases. Thus `T_0` is the identity on the rail
and `T_s o T_t=T_(s+t)` for nonnegative times. By induction, exact repeated
hard clipping of this fixed velocity over any nonnegative partition gives
the same final reference `C(x+h*v)` for the same total duration h.

The in-rail initial state and fixed velocity are essential premises.
Refreshed pressure, changing capacity, forcing, operator jumps, different
rails or a different clipping map require separate laws. The identity is
not binary64 partition invariance: multiplication, addition and time
accumulation can round differently under different partitions, even when
the exact reference agrees. Native phase coordination and adaptation also
act per invocation and are outside this frozen reference.

The hard-rail reference is a constrained map with declared rails. At a
saturated outward-moving coordinate it suppresses further displacement;
it is not an unconstrained solution of `dx/dt=nu*p` there. Its algebra does
not derive the rails or remove their role from the execution provenance.

### Common hard rails cannot increase Dirichlet energy relative to raw input

The common scalar projection is 1-Lipschitz, so for any two coordinates
`abs(C(y_i)-C(y_j))<=abs(y_i-y_j)`. On symmetric nonnegative conductance,
each edge's squared separation therefore decreases or stays fixed. Summing
the weighted terms gives

```text
E_D(C(y)) <= E_D(y).
```

This compares the projected endpoint with its raw input y, not with the
initial state x. It does not imply `E_D(C(x+h*v))<=E_D(x)`: a nonuniform
held velocity can create contrast before projection. The common hard-rail
assumption must not be replaced by different per-node rails or a configured
soft clipping function.

### An exact reference split without an internal clipping claim

For the retained held-pressure interval, set

```text
v = diag(nu)*p_stored
y = x+h*v
z = C(y).
```

Treat the captured scalar state, capacity, pressure, duration and rail
values as exact rationals. With fixed conductance and the actual captured
endpoint `x_actual`, the energy telescope is

```text
E_D(x_actual)-E_D(x)
  = [E_D(y)-E_D(x)]
    + [E_D(z)-E_D(y)]
    + [E_D(x_actual)-E_D(z)].
```

The first bracket is the existing held-flow drift plus quadratic term.
The second is a nonpositive exact hard-rail reference correction. The last
is the signed remaining energy difference between the represented actual
endpoint and that projected reference. The corresponding state difference
`x_actual-z` remains available; a small energy difference alone would not
establish a small state difference.

The [support transport owner](../src/tnfr/physics/support_transport.py)
exposes `SupportTransportClippedFlow` through
`observe_support_transport_clipped_flow(before, after, dt, lower=..., upper=...)`.
It reuses the existing Euler budget for this decomposition rather than
introducing another integrator or an inferred pressure law. The reference
y is not asserted to be a captured raw runtime array, and z is not asserted
to be an observed internal clipping output. Agreement with this reference
does not authenticate how many substeps or clipping calls occurred. Without
their trace, the last bracket is an endpoint difference, not an exclusively
rounding or exclusively clipping measurement.

### The same retained interval is close to its projected exact reference

The retained cases use the actual third interval `h=0.5` and captured
common hard rails `[-1,1]`. No new trajectory or pressure refresh is
performed. Approximate energies and signed contributions are:

| Quantity | Heterogeneous case | Uniform-capacity control |
| --- | --- | --- |
| Initial energy `E_D(x)` | `6.427820565521181` | `7.184031670285270` |
| Raw reference energy `E_D(y)` | `7.705818490365846` | `7.076040744486429` |
| Projected reference energy `E_D(z)` | `6.541255822452809` | `7.076040744486429` |
| Actual endpoint energy | `6.541255822452811` | `7.076040744486428` |
| Raw reference change | `+1.277997924844665` | `-0.107990925798841` |
| Hard-rail reference correction | `-1.164562667913037` | `0` |
| Actual-minus-projected-reference energy | approximately `+3.16e-15` | approximately `-1.38e-15` |
| Observed total energy change | `+0.113435256931631` | `-0.107990925798842` |

The exact rational telescopes have zero residual; these decimal displays
are rounded separately. In the heterogeneous reference only pendant node 9
crosses a rail: its raw value is approximately `-2.498611286112409` and
the projected value is `-1`, which also equals its observed endpoint.
Every other reference coordinate remains unprojected. The maximum absolute
actual-minus-projected-reference state difference is approximately
`4.16e-16`. In the uniform control no reference coordinate is projected,
and the maximum absolute state difference is approximately `3.51e-16`.

Thus the heterogeneous case's large combined Euler defect from section 36
is resolved, at the endpoint level, into a substantial nonpositive hard-rail
reference correction and a small remaining difference from that reference.
This does not turn the constructed y or z into observations of internal
runtime clipping. It also does not establish binary64 partition invariance.
The [reference controls](../tests/physics/test_support_transport_clipped_flow.py)
and enhanced [retained native controls](../tests/physics/test_heterogeneous_mutation_admission.py)
exercise the same shared observer; the existing
[runtime partition boundary](../tests/physics/test_runtime_flow_refinement.py)
retains the distinction between exact composition and represented execution.

### Consequence for the maintenance claim

The heterogeneous projected exact reference still gains Dirichlet energy.
The positive observed change is therefore not explained away as the small
remaining endpoint discrepancy. It belongs to this configured bounded
held-pressure evolution, whose hard projection removes part of a larger
raw increase. Its source still includes supplied heterogeneous capacity,
retained operator pressure and the declared clock and boundary policy.

Sections 35 and 36 separately show that the actual Mutation and coordinator
align phases and reduce their prospective fresh-rate work in this case.
The surviving finite form-energy increase does not contradict those results
and does not establish a restoring law, persistent differentiated identity
or autonomous NFR maintenance. No internal clipping history, new feedback
law, physical interpretation or additional research queue follows from
this endpoint decomposition.

## 38. Default capacity writers cannot restore an inward-contracted profile

The source and interval audits above establish finite native behavior;
they do not explain regeneration of the capacity contrast that supports
a differentiated profile. This section examines that missing restoration
condition directly. It uses the admitted initialized default writer class
from sections 34 and 35, without extending the prior native prefix or
introducing a new capacity law.

### Writer and timing premises

At a native decision, require freshly computed default Si and the ordinary
default selector, with its injected thresholds and operator factors. Use
initialized scalar nodes, so IL is admitted as the first grammar fallback.
Retain the complete node set and exclude custom selectors, operator hooks,
callbacks or other unaccounted capacity writers. Capacities are nonnegative
and initially inside the fixed common configured rails. The maximum M is
positive. These are execution premises, not properties inferred from a
label saying that a run is canonical.

The selection snapshot precedes the sequential primitive batch. Its base
choices are `{IL,OZ,ZHIR,NAV,RA}`; lag overrides add AL and EN. In this
initialized class a rejected candidate does not reach a later
capacity-changing fallback because IL is already admitted. The actual
IL, OZ, ZHIR, NAV, AL and EN primitives preserve capacity. RA writes only
its own target's capacity. Consequently, before a target's own RA write,
its capacity is still the value used for its selection, even if earlier
targets have already changed their own states. Public operator classes
and deliberately supplied words are different execution paths.

The built-in nodal integrator and phase coordinator preserve capacity.
The capacity adaptation owner uses a held neighbor snapshot, updating
only its eligible subset. Its pressure, Si and counter gates need not be
fresh for the interval argument below: any eligible subset has the same
convex-range property. The protected optional network REMESH operation
does not supply a hidden capacity update. The relevant source owners are
[selection](../src/tnfr/dynamics/selectors.py),
[primitive operators](../src/tnfr/operators/__init__.py),
[runtime ordering](../src/tnfr/dynamics/runtime.py) and
[adaptation](../src/tnfr/dynamics/adaptation.py).

### Exact interval invariance under those writers

Let `m=min_i(nu_i)` and `M=max_i(nu_i)` at the selection snapshot. Write
`a` for the default normalized Si capacity weight and `hi=0.5` for its
upper selection threshold. The other Si contributions are nonnegative,
so the ideal fresh definition gives `Si_i>=a*nu_i/M`. An ordinary RA
choice requires `Si_i<hi`; hence

```text
nu_i < (hi/a)*M.
```

The optional default RA amplification multiplies the target by
`gamma=1+b`, where the effective default boost b has nominal recipe
`1/(8*pi)`. The effective coefficients satisfy

```text
gamma*hi/a < 1       (approximately 0.68538189),
```

Thus even an amplified RA target remains below the selection snapshot's M.
The multiplier is at least one, so it cannot cross below m. An unamplified
target is unchanged. All other reachable primitives preserve capacity;
therefore the entire admitted sequential batch remains in `[m,M]`.
This argument uses the selection snapshot bound, not the false claim
that RA is itself a convex neighbor average.

For an eligible adaptation target, the ideal update is

```text
nu_i^+ = (1-eta)*nu_i + eta*mean(nu_neighbors),       0<=eta<=1.
```

Every neighbor value and the target lie in the current interval, so both
the mean and the convex update do too. An isolate uses its own value;
an ineligible target is unchanged. The common rails are inactive on that
already in-rail interval. Thus one admitted exact-model native step obeys

```text
min(nu_after) >= min(nu_before),
max(nu_after) <= max(nu_before).
```

Induction gives nested capacity intervals for every finite continuation
that satisfies these premises. No common semicircle, phase lock, fixed
pressure, fixed eligible subset or convergence to consensus is required.
In particular, this argument must not import the phase-chart observer's
phase representation, connected-graph or resource restrictions.

### An arbitrarily small inward perturbation prevents profile recovery

Declare a nonuniform target capacity profile `nu_star`, with extrema
`0<=m<M`, and arithmetic mean `mu` strictly between them. For any
`0<epsilon<=1`, consider the nearby inward perturbation

```text
nu_epsilon = (1-epsilon)*nu_star + epsilon*mu*1,
l = (1-epsilon)*m + epsilon*mu,
u = (1-epsilon)*M + epsilon*mu.
```

Then `m<l<=u<M`. Every future capacity profile w admitted by the nested
interval result lies in `[l,u]^n`. At target nodes attaining m and M,
comparison with the original profile immediately gives

```text
norm_inf(w-nu_star) >= max(l-m, M-u) > 0.
```

Removing a common positive capacity scale does not remove the obstruction.
For any `c>0`, the same two coordinates give

```text
norm_inf(w-c*nu_star) >= max(l-c*m, c*M-u)
                      >= (M*l-m*u)/(M+m)
                       = epsilon*mu*(M-m)/(M+m) > 0.
```

The second inequality follows by averaging the two affine lower bounds
with weights `M/(M+m)` and `m/(M+m)`, which cancels c. The bound is sharp
over this full interval box: use `c=(l+u)/(M+m)` and project `c*nu_star`
onto `[l,u]` coordinatewise. This mathematical construction proves the
bound's sharpness; it is not a proposed engine operation.

For positive m, the equivalent ratio statement is `u/l<M/m`; the
quantitative bound above also handles m equal to zero without dividing by
it. The argument uses extrema and is unchanged by a permitted relabeling
of the target nodes. It does not assume that the original target was a
fixed point: every future state in the interval box remains a positive
distance from its original positive-scaling orbit.

Because epsilon can be arbitrarily small, this rules out local asymptotic
restoration of a nonuniform capacity profile, including recovery modulo
common positive scale, within this admitted exact writer class whenever
such capacity perturbations belong to the restoration claim. It does not
rule out transient organization, a family with different capacity profiles,
or restoration of an identity definition that does not retain this profile.

### The adaptation implementation must preserve its declared convex range

Exact convexity cannot simply be attributed to rounded arithmetic. With
default `eta=0.1`, target capacity `0.104` and a singleton neighbor equal
to the immediately preceding binary64 value, the former two-product blend
can return the immediately following value above `0.104`. It therefore
exceeds both inputs. The equal-input guard from section 34 does not cover
this nonuniform case.

The shared adaptation proposal now bounds its represented stable mean by
the actual neighbor minimum and maximum, and bounds the represented blend
by the target/mean interval. These comparison guards retain the existing
arithmetic result when it is already inside the declared interval. They
change neither eligibility, counters, factors nor the intended exact
convex map. They enforce the adaptation range in represented arithmetic
without treating numerical overshoot as a structural source.

That implementation guarantee concerns adaptation. The full interval
theorem still has its stated exact-model selector and RA hypotheses;
binary64 Si normalization, threshold comparisons and multiplication require
their own evidence before upgrading it to a represented runtime theorem
for this class. No assumed future rounding margin supplies that evidence.

### Reuse and scope of the obstruction

The conditional algebra belongs with
[capacity feedback](../src/tnfr/physics/capacity_feedback.py), using the
shared exact scalar/vector readers. `derive_capacity_interval_recovery_bound`
returns a `CapacityIntervalRecoveryBound` containing the original and
perturbed intervals, their ratios, both separation bounds and a minimizing
common scale. Its executable domain is nonempty strictly positive capacity
vectors and `contraction` in `(0,1]`; a uniform vector returns zero
separation, not a false nonrecovery claim. The zero-minimum extension of
the proof above is not an additional admitted evaluator domain.
Its existing P2 Coupling envelope
and separate binary64 lattice theorem concern different supplied maps;
neither authenticates the present selector, grammar or runtime class.
The new interval separation likewise remains conditional on the writer
admission established above, rather than a seal on arbitrary executions.

[Recovery-bound controls](../tests/physics/test_capacity_recovery.py)
exercise the exact algebra and its supported domain;
[native writer controls](../tests/physics/test_native_capacity_envelope.py)
check the declared selector and primitive paths. The
[adaptation regressions](../tests/test_structural_stability_adaptation.py)
retain the adjacent-value boundary independently of the conditional
recovery theorem. These checks supply finite implementation evidence,
not admission of unobserved future choices.

The native control uses one default step on unit path4, seed 17, constant
EPI `0.125`, zero phases, capacities `(0.3,0.3,0.3,1)` and empty initial
histories. Its actual word is `(RA,RA,NAV,IL)`; the first two capacities
become approximately `0.31193662073189216`, while the global interval stays
`[0.3,1]`. All four decisions precede the first write. Nodes 1 and 2,
initially equal in capacity, become unequal: the interval theorem does
not assert that every local pairwise contrast decreases.

This result identifies a concrete missing regeneration property for the
default capacity-supported route. It is not a theorem about all thirteen
operators: explicitly chosen capacity-expanding operators, changed
selection laws, new nodes or other capacity writers lie outside its
premises. It does not identify capacity with every possible scalar-form,
phase-sector or support identity, prove physical emergence, or prescribe
a replacement feedback law. The investigation's task queue remains owned
by the [execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md).

## 39. The same stationary reduced identity constrains capacity differences

Section 38 obstructs recovery of an original capacity profile, but an
identity definition may deliberately discard capacity. The narrower
question here is whether altered capacities can still realize the same
EPI shape and phase geometry with zero fresh pressure. This is an algebraic
stationary-realizability gate, not a theorem about arbitrary convergence,
form derivatives or complete-runtime persistence.

### Fix the reduced identity and the pressure realization

Compare two states on the same connected reciprocal unique-neighbor
support, with the same transport conductances and effective pressure
coefficients. Let their EPI values differ by one common additive constant
and their exact phase configurations by one common rotation. Require a
well-defined phase source under the declared circular conventions. The
weighted EPI gradient and ideal phase source then agree; the fixed topology
channel agrees as well. Let `b>0` be the active capacity-pressure coefficient.

Write the unweighted capacity-support operator as

```text
(G*q)_i = mean(q_neighbors)-q_i.
```

This G uses unique support neighbors, including edges with zero transport
conductance. It is generally different from the weighted EPI operator
`-L_rw`. With `delta_nu=nu_after-nu_before`, the ideal fresh-pressure
difference between the two states is exactly

```text
delta_p = b*G*delta_nu.
```

All capacities must be positive to identify zero unforced nodal velocity
with zero pressure. Gamma is absent or zero, and neither clipping nor an
operator jump is being used to keep EPI at an endpoint. Under those
premises, if both states realize zero fresh pressure, `G*delta_nu=0`.

On connected reciprocal support, the kernel of G consists precisely of
constant vectors. At a maximum of a harmonic vector, its neighbor average
can equal that maximum only if every neighbor has the same value; connectivity
propagates this equality. Conversely, a constant vector has zero neighbor
difference. Therefore the necessary and sufficient capacity condition for
preserving the original zero-pressure realization, with the other declared
identity data fixed, is

```text
nu_after = nu_before + c*1,
```

for a common additive capacity offset c that preserves positive capacities.
This is not common capacity scaling: adding a constant preserves capacity
neighbor differences, whereas scaling a nonuniform profile generally does
not. Unequal transport weights do not alter this conclusion because G
belongs to the capacity channel's support convention.

### The capacity interval obstruction survives this reduced identity

Let a nonuniform original capacity profile have extrema m and M, and
contract it toward its arithmetic mean as in section 38. The perturbed
capacity difference is `epsilon*(mu*1-nu_star)`. Since G annihilates the
constant mean, its prospective ideal pressure change at the original
form/phase identity is

```text
delta_p = -epsilon*b*G*nu_star = -epsilon*F_vf_star.
```

The perturbed interval `[l,u]` has width `(1-epsilon)*(M-m)`.
Conditional on subsequent
capacities w remaining in that interval, for every real additive offset c,

```text
norm_inf(w-nu_star-c*1)
  >= max(l-m-c, M+c-u)
  >= [M-m-(u-l)]/2
   = epsilon*(M-m)/2 > 0.
```

The second inequality averages the two extreme-coordinate bounds. It
therefore holds uniformly over all future profiles in the interval box,
not merely for the initial contracted profile. The unconstrained bound is
sharp at `c=(l+u-m-M)/2` with `w` obtained by projecting `nu_star+c*1`
into `[l,u]`. That minimizing offset can make some shifted target
capacities negative; no sharpness claim is made after imposing positive
capacity on the shifted target. The positive lower bound remains valid
for every model-admitted positive shift because that is a smaller
set of comparisons.

The existing
[`CapacityIntervalRecoveryBound`](../src/tnfr/physics/capacity_feedback.py)
now also reports `additive_separation` and `minimizing_offset`. These
quantities reuse the same declared perturbation and interval geometry;
they introduce no evolution law or additional run.

Combining the connected-support identity with this interval separation
excludes an exact zero-pressure stationary realization of the original
EPI shape and phase geometry at every subsequent state satisfying the
interval and fixed-model premises. It does not establish a positive lower
bound on a trajectory's pressure, prove nonconvergence of its EPI or phase,
or rule out a different identity. Such conclusions would need additional
compactness, continuity, admission and dynamical hypotheses.

### Retain phase realization and kernel differences in represented captures

[Forcing realization](../src/tnfr/physics/forcing_realization.py) supplies
`ForcingCapacityDifference` through `observe_forcing_capacity_difference`.
The detached comparison reuses the validated channel decomposition and
rebuilt transport snapshots. It requires matching support, conductance
and coefficients, positive capacities, an exact common EPI offset and
an exact common raw phase offset in the captured coordinates.

The mathematical theorem allows a common circular rotation, whereas this
observer's raw-offset admission is a narrower represented-coordinate
condition. Independently rewrapping or rounding phases must not be treated
as proof of that condition. Even an admitted represented common offset can
change the materialized phase-kernel coefficients slightly; their observed
difference remains explicit rather than being declared zero.

Consequently, the represented pressure ledger separates

```text
fresh_pressure_change
  = capacity_pressure_change
    + phase_realization_change
    + kernel_assembly_defect_change.
```

The stored-pressure difference additionally retains the change in stored
versus fresh residual. Neither a supplied capture nor an exact arithmetic
identity authenticates a kernel call, a stationary baseline or a causal
trajectory. The observer does not infer pressure from an observed EPI
derivative or declare a numerical near-zero value to be exact stationarity.

In particular, the general nodal-rate difference is

```text
delta_rate = diag(nu_after)*delta_p
             + diag(delta_nu)*p_before.
```

The second term vanishes only for a zero-pressure baseline. Actual captured
baseline residuals must remain in that identity; a model's exact stationary
construction does not erase them.

### A prospective stationary construction and its capacity controls

The [stationary identity controls](../tests/physics/test_stationary_capacity_identity.py)
use a declared three-node path with transport weights 1 and 2, injected
default pressure coefficients, seed 17 and the default NumPy kernel. Before
any kernel observation, prescribe

```text
phase = (0, pi/8, 0),
capacity = (1/2, 1, 1/2),
x_i = 3/4 - (w_phase/w_epi)*(phase_i/pi)
            - (w_vf/w_epi)*capacity_i.
```

At this particular target the two endpoints have equal phase and capacity,
so the weighted EPI and unweighted capacity rows share the relevant
three-coordinate pattern. This special coincidence does not make the
capacity channel conductance-weighted in general. The phase source is
`(1/8,-1/8,1/8)` and the transport strengths `(1,3,2)` give the required
zero weighted sum. The prescribed exact-real target therefore has zero
pressure; no measured pressure is fitted to define its EPI.

Its materialized EPI is approximately
`(0.5908450569081046,-0.08600896788251489,0.5908450569081046)`.
The shared signed phase-pressure realization captures approximately
`(0.12500000000000003,-0.125,0.12500000000000003)`: singleton phasor rows and
the certified two-neighbor midpoint round separately. Their exact represented
degree-weighted sum is `3/2^55`, rather than the ideal zero. Baseline fresh
pressure is approximately `(+2.43e-17,+3.47e-18,+2.43e-17)`. The separately
retained phase-realization, modeled-pressure and kernel residuals account for
this nonzero result. Supplied stored zeros are initialization, not evidence
of an exact runtime stationary state or mean conservation.

Half contraction toward the capacity mean gives
`(7/12,5/6,7/12)` while holding materialized EPI and phase fixed. The shared
observer finds

```text
capacity_pressure_change = w_vf*(-1/4,+1/4,-1/4),
fresh_pressure_change = capacity_pressure_change,
```

with zero phase-realization and kernel-defect changes in this comparison.
Each nonzero component has magnitude approximately `0.014574888647966173`.
The final captured pressure also retains the small baseline residual;
it is not replaced by that difference vector. Stored pressures remain
their supplied zeros, exposing the fresh-versus-stored discrepancy rather
than concealing the loss of stationary realizability.

The complementary controls retain the theorem's boundaries:

| Declared change | Observed pressure-difference consequence |
| --- | --- |
| Add `1/4` to every capacity on the connected path | Exact fresh-pressure change is zero |
| Raise only node 0 capacity by `1/4` | Capacity contribution is `w_vf*(-1/4,1/8,0)`; its middle entry is not the transport-weighted `w_vf/12` |
| Shift capacities independently on two disconnected edges and an isolate | Component offsets `1/4`, `1/2`, `3/4` leave fresh pressure unchanged |
| Set the capacity-channel weight explicitly to zero, then contract capacities | Fresh pressure is unchanged |

The asymmetric control retains its small nonzero kernel-defect change.
The rate comparison also keeps `diag(delta_nu)*p_before` rather than
treating the numerical baseline as exactly zero. These are detached
production-kernel observations at prospectively specified inputs, not an
operator schedule, a trajectory, a perturbation-recovery experiment or an
infinite-time stability claim.

[Difference-observer controls](../tests/physics/test_forcing_capacity_difference.py)
check the shared arithmetic, admission and represented phase/assembly
contributions; [recovery-bound controls](../tests/physics/test_capacity_recovery.py)
check the additive separation without assuming that its minimizing shifted
target stays positive.

### Limits of this stationary gate

Disconnected support allows an independent constant capacity shift on
each connected component. A zero capacity-pressure coefficient makes this
channel insensitive to capacity differences. Changed phase geometry,
transport, topology or coefficients can compensate a changed capacity
source and are outside the fixed-identity comparison.

Positive capacities, no Gamma and no clipping are also substantive
conditions. Zero capacity can suppress a nonzero pressure; forcing can
cancel it; a rail can hold an endpoint under outward pressure. None is a
zero-pressure stationary realization of the declared unforced model.
Likewise, a common EPI drift may preserve differences while using nonzero
pressure, and moving or periodic patterns require their own treatment.
The present result addresses stationary pressure realizability only and
does not introduce a second task queue or a replacement constitutive law.

## 40. Static phase compensation and the native direction obstruction

Section 39 fixes both form and phase geometry. Here the phase is allowed
to change while the original EPI shape is retained. The same prospectively
declared weighted P3 supplies the comparison; no new oscillator, selected
target after observation or extended trajectory is introduced. Static
pressure compatibility, the actual native writer and complete-runtime
restoration remain different questions.

### A different phase geometry can compensate the lost capacity source

Write `a=w_phase>0`, `e=w_epi>0` and `b=w_vf>0` for the effective pressure
coefficients. In the P3 family, both endpoints have the same EPI X, phase
and capacity; the middle has EPI Y, a relative phase gap delta and a
capacity excess d. Transport edge weights remain 1 and 2. Because the
endpoint values agree, the weighted EPI row and the unweighted support
rows yield

```text
p = [e*(Y-X)+a*delta/pi+b*d]*(1,-1,1).
```

The topology contribution is zero because the injected topology weight
is zero, not because this irregular support has zero topology gradient.
The original construction has `delta_0=pi/8`, `d=1/2` and zero ideal
pressure. Inward capacity contraction reduces d to `(1-epsilon)*d`.
Keeping the same EPI shape therefore requires

```text
delta_required = delta_0 + pi*(b/a)*epsilon*d.
```

This is a positive phase-gap increase. The effective default ratio
`b/a<1/12` keeps `delta_required<pi/6` for `0<epsilon<=1`, so the required
state remains in a regular acute chart. Static feasibility is not blocked
by a phase-wrap singularity. For the retained half contraction,
`delta_required` is approximately `0.45306233345003016`, compared with
the original gap `0.39269908169872414`.

This construction preserves EPI shape through a different phase geometry;
it does not preserve the full form-and-phase identity of section 39.
Prescribing that compensating phase is initialization of a compatible
state, not evidence that the engine generates its restoring change.

### The native phase update moves in the opposite direction

In this common chart, let g be the shared global phasor target. Each
endpoint's local target is the middle phase; the middle's local target
is the common endpoint phase. The ideal formula implemented by the
[native coordinator](../src/tnfr/dynamics/coordination.py) therefore gives

```text
delta_after = (1-kG-2*kL)*delta_before.
```

The global target cancels from this difference. Unequal transport
conductances do not change the phase-neighbor convention. The injected
default adaptive gain bounds from section 34 imply
`0<1-kG-2*kL<1`, independently of which admitted adaptive branch supplies
the gains. Thus this writer decreases the positive gap, whereas lost
capacity support requires an increase.

At fixed EPI and capacity, a gap contraction by factor rho contributes

```text
delta_p_phase = -a*(1-rho)*delta_before/pi*(1,-1,1).
```

Starting from the uncompensated capacity contraction, this adds to the
existing residual `-b*epsilon*d*(1,-1,1)`. Starting instead from the
statically compensated ideal zero-pressure state, the same native phase
write produces a nonzero residual. The compatible stationary surface is
therefore not invariant under this phase map.

The default selector does not supply a compensating Mutation in either
retained segment. After half contraction, `min(nu)/max(nu)=0.7`. With
alpha denoting the Si capacity coefficient, fresh default Si is bounded
below by `0.7*alpha>0.5`; every base choice is IL. Initial lag counters
remain below their forcing thresholds, and the nonzero EPI values admit
IL in the incremental grammar. This argument concerns the actual default
selection path, not a deliberately supplied operator word.

### Two isolated native segments confirm the direction

The finite controls reuse the materialized EPI and capacities from section
39. One starts with the original phase gap; the other initializes the
analytically prescribed compensating gap before any observation. Each
segment refreshes pressure and Si, executes the actual default glyph batch
and invokes the native coordinator. No nodal integration, capacity
adaptation, full runtime step or temporal-growth history is performed.

Both batches request and apply IL at all three nodes, pass grammar and
leave the lag counters at one. The coordinator selects its stable adaptive
branch with represented gains
`kG=0.04461342911494398`, `kL=0.140157221158957`, giving
`rho=0.675072128567142`. The retained observations are:

| Initial phase choice | Gap before coordination | Gap after coordination | Maximum fresh pressure magnitude before | Maximum fresh pressure magnitude after |
| --- | --- | --- | --- | --- |
| Original, uncompensated gap | `0.39269908169872414` | `0.2651002049687197` | approximately `0.014574888647966177` | approximately `0.04538402112491463` |
| Prescribed static compensation | `0.45306233345003016` | `0.3058497538157081` | approximately `3.99e-17` | approximately `0.03554492002170308` |

Pressure magnitudes are displayed; after coordination the endpoint
pressures are negative and the middle pressure positive. The statically
compensated capture retains its small nonzero baseline rather than claiming
an exact represented equilibrium. Its pressure increase is not explained
by a large initialization residual.

EPI, capacity and support remain unchanged in both segments. IL contracts
stored pressure, and the coordinator leaves that held pressure unchanged.
The fresh pressures in the table are detached read-outs; they are not
substituted into an executed physical interval. Neither segment is evidence
of complete-runtime recovery or failure of every future trajectory.

### An exact invariant source class closes this native P3 route

No further run is needed to combine the earlier conditional results for
the uncompensated half-contraction case. Retain the initialized, fresh-Si,
default writer class of sections 34 and 38, this fixed support and these
coefficients, with no additional phase/capacity writers. The capacity
interval stays within `[l,u]=[7/12,5/6]`. Its minimum-to-maximum ratio cannot
fall below 0.7, so every future fresh base selection remains IL. Lag AL/EN
and admitted fallback IL also preserve phase and capacity. Adaptation may
change any eligible subset without leaving the interval.

The equal endpoint phases remain equal under the coordinator, and its
positive gap factor keeps `0<=delta_t<=delta_0`. These phase facts do not
require intermediate EPI to remain at its original value. At any subsequent
state whose EPI does equal the original shape modulo a common offset,
capacity differences satisfy

```text
nu_middle-nu_endpoint <= u-l = (1-epsilon)*d,
mean(nu_endpoints)-nu_middle >= -(u-l).
```

Consequently its ideal fresh pressure necessarily obeys

```text
p_endpoint <= -b*epsilon*d < 0,
p_middle   >= +b*epsilon*d > 0.
```

This is a uniform exclusion of zero-pressure stationary realizability of
that original EPI shape within the declared exact source class, including
unequal future endpoint capacities. It is conditional on the stated
writer, chart, source and capacity-envelope premises. It is not an
unrestricted binary64 continuation theorem, a statement about every actual
held pressure, or a proof that hybrid EPI jumps cannot maintain another
kind of pattern.

The separately initialized compensated gap is larger than `delta_0` and
lies outside this source-box hypothesis. Its one-invocation failure above
is retained separately; it is not retroactively included in that invariant
class. Together the results distinguish an available static compensation
from its absence as a restoring native phase action in the audited route.
They supply no replacement phase law or additional execution queue.

## 41. Geometric identity and the undefined global phase target

Recovering every coordinate of a stationary target is not the definition of
every coherent entity. Section 32 already supplies a different conditional
identity: acute cycle phases approach a uniform twist while scalar EPI
approaches consensus. The limiting phase geometry remains nonuniform. This
section checks that identity against the native global coordinator, without
changing the earlier P3 target or assuming that its supplied sine law is the
default runtime law.

### Local coherent geometry can have zero global phase order

On a simple C5, fix the ideal phases on the circle as

```text
theta_i = theta_0 + i*delta modulo 2*pi,  delta = 2*pi/5.
```

The global unweighted phasor sum is zero. Each node's two neighbors instead
have the nonzero resultant

```text
exp(i*theta_(i-1)) + exp(i*theta_(i+1))
    = 2*cos(delta)*exp(i*theta_i),  cos(delta)>0.
```

Thus its local phase target is its own phase, canonical phase pressure and
curvature are zero, and absolute local phase gradient is `delta`. All edges
are strictly inside the half-pi U3 limit; winding is one. With constant EPI,
common positive capacity and no additional forcing, the other pressure
channels also vanish on this cycle. With fresh pressure and its nodal rate,
the ideal coherence read-out is one although global Kuramoto order is zero.
This distinguishes two existing
diagnostic meanings; it does not redefine either metric.

Positive unequal transport conductances, including section 32's retained
closing-edge weight, do not change these phase facts. The local and global
phase reducers use unique neighbors and unit global weights, respectively.
Under section 32's supplied phase law this is a relative equilibrium, with
common phase advance and unchanged relative geometry. Its conditional
relaxation bounds do not transfer automatically to the native coordinator.

### A symmetric phase multiset cannot select one covariant direction

Suppose a phase-only global target `T` takes values on the circle, is
invariant under permutations of its inputs and covariant under a common
phase rotation. Let P cyclically permute the five regular phases. Then

```text
P*theta = theta + delta*1 modulo 2*pi,
T(P*theta) = T(theta),
T(theta+delta*1) = T(theta)+delta modulo 2*pi.
```

The last two conditions contradict `delta!=0 modulo 2*pi`. No deterministic
single-valued target can satisfy both symmetries at this multiset. This is
the same stabilizer logic used for parent selection in
[the THOL audit](THOL_BIRTH_AND_TRANSPORT.md#8-selection-actual-dispatch-and-the-unique-parent-obstruction),
applied to a circular output rather than a chosen vertex.

The argument concerns the phase multiset consumed by the global reducer.
It does not assume that a cycle with a distinguished conductance has a
cyclic automorphism preserving its complete weighted state. A marked node,
external direction or other additional input changes the target-selection
problem. None is supplied by the zero phasor resultant itself.

### Any selected global target changes the uniform-twist orbit

The actual [coordinator](../src/tnfr/dynamics/coordination.py) applies a
shared global gain gamma and local gain to shortest-arc displacements. At
the ideal uniform twist the local displacement is zero. If a single global
target T is nevertheless supplied, its ideal phase update is

```text
theta_i^+ = theta_i + gamma*wrap(T-theta_i) modulo 2*pi.
```

For `0<gamma<1`, these increments cannot all agree modulo `2*pi`, so the
result is not merely a common rotation. Around the oriented cycle, four
gaps and the one gap crossing the target's shortest-arc cut become

```text
four gaps:       (1-gamma)*delta,
exceptional gap: delta+gamma*(2*pi-delta).
```

An antipodal tie can move which edge is exceptional; it does not restore
equal gaps. Their unwrapped sum remains `2*pi`. Every gap remains inside
the open shortest-arc branch exactly when `gamma<3/8`; winding then remains
one despite the changed geometry. Strict acuteness additionally requires
`gamma<1/16`. At equality the exceptional gap is `pi/2`: the closed hard U3
gate may admit it, but the strictly acute theorem no longer applies. At
`gamma=3/8` that gap reaches the antipodal branch boundary.

Consequently preservation of a winding integer is weaker than invariance
of the uniform-twist orbit. This obstruction needs no EPI integration or
long trajectory. It also does not say that the resulting phase state has
lost every possible coherent identity.

### Native policy and represented arithmetic have separate scope

The default adaptive policy classifies sufficiently low global Kuramoto
order as dissonant and retains a strictly positive global gain. That policy
therefore reacts to the ideal twist's zero global order despite its regular
local geometry. It is a configured preference for global alignment, not a
consequence that local TNFR coherence or winding is absent.

The ideal regular polygon and its materialized binary64 angles are distinct
inputs. Their transcendental sums need not agree exactly; cached trigonometry
and reduction add further numerical effects. The existing
[retained phase audit](THOL_BIRTH_AND_TRANSPORT.md#global-phase-direction-near-the-symmetric-input)
already exposes that distinction and the conditioning of a small resultant.
The shared represented-component reducer sums its supplied component pairs
exactly; it does not certify the transcendental values or a rotation symmetry.
The coordinator's opt-in `exact_components_v1` rejects an exactly zero
represented resultant when the effective global term is active. A nonzero
represented sum remains a numerical direction, even if very small. The
legacy path instead retains its existing numerical `atan2` behavior.

The [static writer controls](../tests/physics/test_native_phase_writer_boundaries.py)
use unit C5, represented `theta_i=0.125+2*pi*i/5`, EPI `0.5`, capacity one,
stored pressure zero, empty glyph histories, injected defaults and seed 17.
One legacy invocation and a matched `exact_components_v1` invocation both
select the dissonant branch with `kG=0.05929429797231117`. Four observed gaps
are approximately `1.1821256490720864` and the fifth `1.5546827108912409`,
matching the derived deformation. Winding remains one and the strict U3
margin is approximately `0.0161136159036559`. The exact sum of cached
component pairs is `(-7,3)/2^55`, nonzero; this does not contradict the ideal
polygon's zero sum. Both calls preserve EPI, capacity, stored pressure,
support and clock. They execute no glyph, nodal integration or full runtime
step and establish neither future invariance nor formation of this pattern.

These results identify a model-compatibility boundary between an existing
geometric identity and an existing native policy. They select no replacement
target, zero-resultant threshold, phase law or autonomous event schedule.

## 42. A geometric-identity contract for the retained weighted C5

### Declared state, law and identity family

Keep Section 32's scalar form chart, supplied clock and continuous phase/form
law. The modeled state is `(x,theta,nu,W)`, not every engine history, cache or
controller variable. Retain the actual Section 31 post-UM support and its
materialized nonunit closing conductance. The ordered phase support is C5;
all transport conductances are fixed, positive and symmetric. Capacity is
held at its captured common positive value `kappa`, and effective pressure
weights and `K>0` are fixed. No Gamma, subsequent event, active clipping or
native controller is included. The phase law identifies kappa with radians
per supplied time unit; neither laboratory seconds nor a clock emergent from
synchronization is inferred.

The identity family is specified before evaluating a response:

```text
I(W,kappa) = { x=c*1, nu=kappa*1,
              theta_i=alpha+2*pi*i/5 modulo 2*pi, W fixed }.
```

Its free labels are common rotation alpha and uniform form c in the declared
chart. It excludes phase consensus and fixes winding `+1` in the chosen
orientation. Reversing the cycle enumeration changes the displayed winding
sign, not the physical state; conjugating the physical phases is a different
operation and is not quotiented out. The family preserves nontrivial phase
geometry even though form is uniform. It does not require nonzero scalar-form
amplitude; that would be a further identity obligation. It is not a definition
of every NFR or an identification with a physical particle.

On this family the local phasor source and EPI pressure vanish, the sine
corrections cancel, and `theta_dot=kappa*1`. Thus it is exactly invariant
under the declared continuous law: c is constant and alpha rotates. The
global Kuramoto order is zero while each normalized local resultant is
`cos(2*pi/5)>0`. The global-target obstruction in Section 41 therefore remains
relevant; invariance here is not invariance under that native coordinator.

### A lifted shape coordinate separates rotation from deformation

Admit initial oriented gaps inside one strictly acute interval `[m,M]`,
contained in the fixed U3 gate, with sum `2*pi`. Section 32 preserves this
domain. Write `delta_bar=2*pi/5`, `u_i=delta_i-delta_bar`, and define

```text
p_0=0,  p_i=sum_(j<i) u_j,
h_i=p_i-mean(p),
alpha=theta_lift_0+mean(p).
```

Then `sum h_i=0`, the cyclic difference of h equals u, and
`theta_lift_i=alpha+i*delta_bar+h_i`. The last cyclic difference is valid
because `sum u_i=0`; the phase lift itself closes after one full turn.
Under the supplied equal-capacity sine law the corrections telescope in
`sum theta_dot_i=5*kappa`. Consequently

```text
alpha(t)=alpha(0)+kappa*t,
sum_i h_i(t)^2 <= ||u(t)||_2^2 / lambda_2(L_C5)
              <= Q^2*exp(-2*gamma*t) / lambda_2(L_C5).
```

The second line is the cycle Poincare inequality on centered h. This is a
lifted shape norm. The circular orbit distance is
`inf_alpha sum_i wrap(theta_i-alpha-i*delta_bar)^2`. It is no greater than
the displayed lifted bound, because wrapping each residual and then
minimizing over a common rotation cannot increase this chosen value. No
claim is made that the lifted norm is the global circular minimum.

The offset alpha uses the ordered support and its admitted lift. It is a
coordinate, not a permutation-invariant target obtained from the phase
multiset, and it supplies no exception to Section 41's symmetry obstruction.

The existing `CycleRelaxationEnvelope` exposes this decomposition as exact
affine-pi pairs in `initial_phase_offset_affine` and
`initial_phase_shape_affine`, with shared rational enclosures in
`initial_phase_shape_enclosures`. Its
`phase_orbit_distance_squared_upper` reuses the existing sample times, gap
envelopes and certified spectral lower bound. These are derived read-outs of
one captured state/model, not new dynamic parameters, a second solver or
another evidence record. The properties also apply to the evaluator's other
admitted cycle sizes and winding sectors; the identity selected here is C5,
winding one.

### Form mean and the strength of the persistence claim

Use the actual transport strengths s, their sum S and
`m_s=s^T*x/S`. The form residual remains
`D=sum_i s_i*(x_i-m_s)^2`; phase and form residuals are reported separately
without inventing a cross-channel normalization. Section 32 supplies D's
Duhamel bound and

```text
m_s_dot = kappa*w*s^T*g/S,
|m_s(t)-m_s(0)| <= (M_source/gamma)*(1-exp(-gamma*t)),
|m_infinity-m_s(t)| <= (M_source/gamma)*exp(-gamma*t).
```

Here `M_source` is that section's mean-rate prefactor (its symbol M there),
not the upper phase-gap endpoint used above. Thus the limiting uniform form
is unknown but constrained by the captured initial mean and
`mean_limit_offset_upper`. Permitting this governed mean drift is not
permission to fit an arbitrary c(t). On the invariant family itself g=0,
so there is no mean drift. With the retained nonunit edge, a perturbed state
can have nonzero weighted drift even though `sum g_i=0`.

These estimates establish conditional attraction to the identity family.
They also give local orbital stability relative to that family: keep W,
kappa, coefficients and a common strictly acute neighborhood fixed. As
initial gap and form residuals tend to zero, their all-time upper bounds
and the mean-shift bound tend to zero. The neighborhood keeps the rates
bounded away from zero. Form disagreement need not decrease monotonically;
the retained post-event state initially develops form contrast before it
relaxes. Capacity, support, operator-policy and phase-law perturbations are
outside this stability claim.

### Evidence and remaining premises

The [existing control owner](../tests/physics/test_cycle_postevent_relaxation.py)
reuses its retained initial capture to reconstruct the phase lift independently
at high precision, check centering, affine-pi enclosures, reconstruction and
the Poincare bound. Its winding is already one and its initial EPI disagreement
is zero, but its phase deformation is strictly positive. Thus a winding label
and uniform form do not alone identify a regular geometric shape. Reversal
with relabeling retains the same nodewise shape; a consensus control has zero
shape in the separate winding-zero family. No new trajectory is generated for
these assertions. The earlier finite binary64 continuation remains evidence
only for its stated execution horizon and numerical defects.

The derived parts are phase-domain preservation, this rotation/shape split,
conditional invariance, attraction and mean bounds. Fixed support/capacity,
the supplied sine phase law, its clock interpretation, pressure realization,
coefficients and prepared sector remain explicit model premises. The native
global coordinator is incompatible with the regular orbit; support creation
still depends on the recorded UM policy. Autonomous maintenance, formation
from a different sector and empirical identification remain unresolved. The
[single G3 queue](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) owns
the next mechanism-admission task.
