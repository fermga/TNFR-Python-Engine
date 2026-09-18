# Relative form and mean drift on a held canonical support

**Status:** Exact fixed-support balance and finite Euler error identities.
Runtime observations retain pressure and endpoint defects separately.
Section 7 adds exact instantaneous regional balances on the full support;
they do not assume that a selected region is already an autonomous NFR.
**Research links:** B2.d.7/O3.a, S3, S8, S9 and S16.

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
