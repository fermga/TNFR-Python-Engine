# Collective state and a matrix-response comparison

<a id="organized-effective-law-contract"></a>

## Status and responsibility

This owner specifies one complete collective comparison for the foundational
question raised by the [physical feasibility review](../research/ORGANIZED_ELECTRICAL_FEASIBILITY.md).
The law below is a supplied constitutive choice. Its balance and coordinate
identities are conditional algebra. The [prepared-reference theorem](#prepared-matrix-response-result)
adds own-law identity retention, a finite response separation and a precise
state-reduction obstruction. Acquisition of this preparation, fine-source
inheritance and physical realization remain unestablished.
The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone selects work and acceptance conditions.

The question is how retained state, response and storage must change together
when an organized object has an additional structural coupling. It does not
assume a new primitive capacity or a uniquely selected fundamental law. The
[NFR ontology](../EMERGENT_ONTOLOGY.md#nfr-participation-in-form) requires any
claim about constituents to retain their influence in the same configuration.

## 1. Complete comparison and units

Use the fixed, unit-weight, undirected C5 Laplacian L and five signed forms U
with five primitive circular phases Phi. These are the collective reference
coordinates in the [quarter-law dictionary](ORGANIZED_ELECTRICAL_PROTOCOL.md).
They do not include the 45-node source's internal deviations. Retain the
same normalized structural clock tau=(1023/1024)t and units,
gamma=1/(1023 pi), and held unit scalar capacities. Primes below denote
derivatives in tau. Laboratory calibration is a separate obligation.
Write m for the conserved mean form; r below always denotes the response
parameter, not the mean denoted r in the earlier electrical protocol.

Let r be a supplied finite constant with r>=0, and define

\[
 W_r=I+rL^2,\qquad M_r=W_r^{-1},\qquad
 S_i(\Phi)=\sum_{j\sim i}\sin(\Phi_j-\Phi_i).
 \tag{1}
\]

The dimensionless parameter r changes the response in this declared chart;
it is not fixed by the nodal row. The complete driven equations are

\[
 W_r U'=(-LU+\gamma S(\Phi))/4+f(\tau),\qquad
 \Phi'=\gamma LU/4.
 \tag{2}
\]

Input f is an independently specified, neutral, locally integrable
five-vector; no phase feedback, support event or input selector is implicit.
There are no jumps. Thus the solved form row has input M_r f, not f.
The unforced pressure is the independently specified map
`p=M_r*(-LU+gamma*S)/4`, with `U'=p` at unit capacity. Circular differences
enter S without choosing a principal-phase average. Means of phase lifts
are used only after declaring those lifts.

W_r is symmetric positive definite, commutes with L and satisfies W_r 1=1.
The full equations preserve the form mean under neutral input; on any
continuous phase lift they also preserve its common mean. One may fix the
neutral form leaf and quotient a common phase origin, while retaining four
relative forms and four relative phases. These are symmetry reductions,
not removal of response-relevant internal state. An admitted initial set,
its lifted chart where needed and every proposed identity must still be
specified before a finite claim.

At r=0, (2) recovers the nominal collective quarter law. For r>0 it changes
only the form row in these coordinates. This differs from the
[phase-dependent diagonal reciprocal mobility](SINE_PAIR_MOBILITY.md), which
changes both rows and keeps another storage. Its proofs do not transfer.

The [electrical audit](../research/ORGANIZED_ELECTRICAL_FEASIBILITY.md#organized-electrical-physical-admission)
motivates (1) through an additional positive ring capacitance, with
r=c_J/c. That motivation does not force this law on arbitrary NFRs or
provide a fine-node electrical realization. In particular, replacing L
by L45 would be a further constitutive choice requiring its own admission.

## 2. Storage and supplied work

Let V(Phi)=sum_edges[1-cos(Phi_j-Phi_i)], so grad V=-S. The comparison's
own storage is

\[
 H_r(U,\Phi)=\tfrac12 U^TLW_rU+V(\Phi).
 \tag{3}
\]

Since L and W_r commute, LW_r is symmetric and positive semidefinite.
Differentiating along (2) gives, almost everywhere for integrable f,

\[
 H_r'=(LU)^T W_rU'-S^T\Phi'
       =-\|LU\|^2/4+(LU)^Tf.
 \tag{4}
\]

The cross terms cancel. This is a storage/work balance, not a formation,
stability, recurrence or observation theorem. The old H0 is not the storage
in (4); the difference is `r*U^T*L^3*U/2`. Consequently equal initial states
need not have equal storage across r. A comparison must declare which
preparation quantities are matched instead of assuming both equalities.

There is an explicit obstruction to reusing the old decay claim. Set f=0,
q=LU and H0=H_r at r=0. Along the positive-r law,

\[
 H0'=-q^TM_rq/4+\gamma q^T(M_r-I)S/4.
\]

Choose Phi=(0,pi/2,0,0,0), so S=(1,-2,1,0,0), and
U=-(gamma*r/2)LS. Then q=(gamma/2)(I-W_r)S and, for every r>0,

\[
 H0'=\frac{\gamma^2}{16}S^T(M_r-I)W_r(M_r-I)S>0,
 \qquad H_r'=-\|q\|^2/4<0.
\]

The strict inequalities follow because S is a nonzero neutral vector and
L is positive definite on that subspace. This example uses the full
circular-phase domain of (2); it is not an admitted acute winding source.
It disproves a global transfer of the old storage monotonicity, without
claiming loss of an acquired identity.

Because W_r is invertible, unforced equilibria of (2) agree with those of
the r=0 complete law. Their tangent rates and finite response need not agree.
Matching equilibria does not transfer attraction, an initial-state capture
bound or an observation certificate. A finite positive-r study must declare
its parameter range and horizon before requesting uniform bounds.

## 3. Coordinate, locality and memory boundaries

The invertible charge-like chart Z=W_r U changes both equations:

\[
 Z'=(-LM_rZ+\gamma S(\Phi))/4+f,\qquad
 \Phi'=\gamma LM_rZ/4,
 \tag{5}
\]

with storage `Z^T*L*M_r*Z/2+V(Phi)` and work pairing `(LM_r Z)^T f`.
Moving W_r out of the form row while retaining `Phi'=gamma*L*Z/4` would
therefore change the model. A regular chart change is distinct from
discarding state and from choosing a different pressure law.

Although W_r is positive definite, M_r is not generally entrywise
nonnegative. For example, at r=1/100 its first row is
`(2181,81,-19,-19,81)/2305`. It is not a convex averaging map or a vector
of nonnegative nodal capacities. Its coupling need not have the one-edge
locality of the original sine-pressure row; locality must be justified
for the actual map, including the phase and input channels.

This fixed matrix is instantaneous. It introduces no temporal memory by
itself. Derived memory in the cited reductions comes from eliminating
dynamical state under a specified observation, retaining hidden
initialization and forcing, as in
the [memory owner](../DERIVED_EPI_MEMORY.md). The
[inherited-metric result](INHERITED_FORM_DYNAMICS.md#132-inherited-metric-restoring-pressure-and-coordinate-dependence)
already distinguishes coordinate response factors from primitive capacity.
Neither construction justifies replacing nu_f by M_r in the native runtime.

Holding r constant is part of (2). On the neutral form leaf, a time-dependent
charge law `q proportional to W_r U` differentiates to an expression
proportional to `W_r U'+r' L^2 U`; off that leaf the charge map uses
`W_r(U-m*1)`. Substituting W_r(tau) into (2) alone would be a different model.
Parameter changes or
support events also need their storage/work and occurrence laws.

## 4. Identity, scale and implementation boundary

The state in (2) is sufficient for this declared ten-coordinate ODE. The
[finite theorem below](#prepared-matrix-response-result) uses a bounded
preparation of these same coordinates, including the relative phases.
It is not a proof that ten coordinates predict the full acquired source.
The [same-slot source remainder](ORGANIZED_ELECTRICAL_PROTOCOL.md#fixed-support-electrical-protocol),
its hidden transport, constituent identities and acquisition use another
complete law. They remain valid there. Neither r=0 recovery nor a shared
equilibrium establishes those claims at positive r.

A claim about a parent NFR must define whether it concerns this prepared
collective reference or a finer organization with retained constituent state.
The latter needs a separately declared fine law, a state/clock/input map,
exact closure or controlled error, and own-law identity bounds. Static
re-encoding or formal associativity of an exact elimination is insufficient
as a new capability; finite nonlinear prediction must retain the consumed
boundary information. Events additionally require the existing
[operator descent criterion](../STRUCTURAL_OPERATORS.md#collective-operator-descent).

This comparison installs no execution law, SDK method or operator.
The [independent algebra controls](../../tests/physics/test_organized_effective_law.py)
check (1)-(5), the r=0 boundary and deliberate wrong-chart/input controls.
They evaluate no reserved response. Exact matrix calculations reuse
[shared linear algebra](../../src/tnfr/mathematics/_exact_linear_algebra.py).
The [coordinate-memory owner](../../src/tnfr/mathematics/linear_observation.py)
can split an admitted fixed linear/tangent generator or rational spatial
blocks. It does not close the nonlinear sine law. Such reuse must retain
nonlinear residuals, hidden initialization and forcing where consumed;
rationalizing gamma would specify a different coefficient, not certify
the exact comparison above. A nonlinear reduction needs its own proof.
Generic runtime integration is justified only by a proved consumer contract.

<a id="prepared-matrix-response-declaration"></a>
## 5. Declared finite prepared-reference question

The analytical comparison fixes the following domain before evaluating any
response. It concerns (2) alone, without a fine-node extension, source replay,
physical sensor or numerical trajectory. The two models match their complete
initial coordinates and prescribed input; their initial storages need not match.

| Primitive | Declaration |
| --- | --- |
| Response | Fixed `r in [1/100,1/10]`; separate reference `r=0` |
| Prepared identity | Either C5 winding `w=+1` or `w=-1` |
| Phase chart | `Phi_i=2*pi*w*i/5+theta_i`, with a continuous perturbation lift and `sum(theta)=0`; the closing oriented edge retains its `2*pi*w` offset |
| Full preparation | `sum(U)=0`, `||U(0)||_2+||theta(0)||_2 <= delta=2^-26` |
| Input | `f=a*v` for `0<=tau<=T`, zero afterwards; `a=1/16`, `v=(1,-1,0,0,0)` |
| Clock and horizon | The existing structural tau; `T=1/1024` |
| Observation | `y=v^T*U(T)` with separate additive error bounded by `eta=2^-26` |
| Identity criterion | Every oriented phase gap stays within `rho=1/10` of `2*pi*w/5` during the pulse and its unforced continuation |
| Finite prediction criterion | Whole-preparation observed record intervals for the positive-r family and r=0 are disjoint, with separation strictly greater than `2^-21` |
| Calculation | Exact rational inequalities and certified transcendental bounds; no integration, parameter scan, precision search or response fitting |

The prediction must retain both evolution rows, the full preparation bound,
the input channel `M_r*f`, and the law's own storage and supplied work. A
separate state-sufficiency control tests whether identical forms, winding,
conserved means and storage can still conceal different form derivatives.
Neither success condition would derive acquisition of this preparation,
constitutive selection, a laboratory clock or a physical measurement map.

<a id="prepared-matrix-response-result"></a>
## 6. Own-law identity and finite prediction

Declaration revision `a205a8fe` fixes Section 5. The result is analytical:
no measured or simulated response is used to set its constants or decide
between the laws. Write `kappa=2*pi*w/5`, let D take the five oriented
cyclic differences, and use the declared closing-edge offset. All norms
below are Euclidean except the explicitly marked maximum norm.
The mixed form/phase norm uses numerical coordinates in the fixed normalized
chart of Section 1; it is not an invariant metric or a laboratory calibration.

### 6.1 Identity uses the actual storage

On the centered form leaf, define excess storage relative to the prepared
twist by

\[
 E_r=H_r-V_* =\tfrac12 U^TLW_rU+
 V(\Phi_w^*+\theta)-V_*,\qquad
 V_*=5[1-\cos(2\pi/5)].
 \tag{6}
\]

The full preparation cost includes V_*; E_r is not total storage or a
budget that supplies the external input. Consider the convex relative
phase cell `||D theta||_infinity <= rho=1/10`, `sum(theta)=0`.
Here `cos(kappa+d)>1/5` for every `|d|<=rho`: cosine is 1-Lipschitz and
`cos(2*pi/5)=(sqrt(5)-1)/4>3/10`. The first variation of V at the twist
vanishes because `sum(D theta)=0`. Taylor's integral remainder and
`LW_r>=0` therefore give

\[
 E_r\ge\tfrac1{10}\|D\theta\|^2.
 \tag{7}
\]

Every first contact with this cell's boundary requires `E_r>=1/1000`.
Also `||L||<=4`, as follows directly from the edge-square bound
`sum_edges(z_j-z_i)^2<=4||z||^2`. The initial preparation consequently obeys

\[
 E_r(0)\le2(1+16r_{\max})\|U(0)\|^2+2\|\theta(0)\|^2
 \le(4+32r_{\max})\delta^2=\tfrac{36}{5}\,2^{-52}.
 \tag{8}
\]

The bound also holds for the r=0 reference. Initial cell admission follows
from `||D theta(0)||_infinity<=2 delta<rho`. Completing the square in the
own-law balance (4), rather than transferring H0 decay, yields

\[
 E_r'=-\tfrac14\|LU-2f\|^2+\|f\|^2\le\|f\|^2.
 \tag{9}
\]

During the declared pulse, `||f||^2=2a^2`. Thus

\[
 E_r(\tau)\le\overline E=\tfrac{36}{5}\,2^{-52}+2^{-17}
 <2^{-16}<\tfrac1{1000}\quad(0\le\tau\le T).
 \tag{10}
\]

Continuity excludes a first boundary contact. After T the input is zero
and E_r is nonincreasing, so the same identity survives for all future
times. In particular the gaps remain strictly acute with the original
winding. The smooth periodic sine term and constant positive-definite
W_r give global continuation; there is no hidden event at pulse release.

On the fixed-mean lift, the intersection of `E_r<=overline E` with the
closed relative cell is compact and contained in its interior. Zero
dissipation requires `LU=0`, hence `U=0`; remaining invariant trajectories
also require `S=0`. The potential is strictly convex on this relative cell
by the cosine Hessian bound used in (7), so its only critical point is the prepared twist.
The invariant-set argument therefore gives convergence to that twist after
the pulse for each fixed r. It supplies no uniform settling time, acquisition
from another winding sector or indefinite external activity.

### 6.2 A finite response with both rows live

Let `z(tau)=||U(tau)||+||theta(tau)||`. For every phase state, the sine-source
Jacobian is an incidence matrix times a diagonal cosine matrix times its
transpose; hence `||S(Phi_w^*+theta)||<=4||theta||`. Also `||M_r||<=1` and
the exact `gamma=1/(1023*pi)<1`. Both rows of (2) give

\[
 D^+z\le2z+2a,\qquad
 \sup_{0\le\tau\le T}z(\tau)\le
 Z=\frac{\delta+2aT}{1-2T}.
 \tag{11}
\]

The rational bound follows by taking the supremum in the integrated
inequality, using `T<1/2`; it is not a numerical exponential or a frozen
phase approximation. Integrating the form row and taking its v projection
gives the finite prediction

\[
 y=aT h(r)+R,\quad h(r)=v^TM_rv,
 \quad |R|\le\mathcal R=2\delta+2TZ.
 \tag{12}
\]

The remainder retains initial form, unknown initial phase shape, damping
and the phase response throughout the pulse. Here `||v||=sqrt(2)<2` and
`||M_r(-LU+gamma S)/4||<=z` supply (12).
It is not a tangent trajectory or a finite difference at time zero.

On the neutral C5 space, `L^2-5L+5I=0`. With `d(r)=1+15r+25r^2`,

\[
 M_r=\frac{(1+20r)I-5rL}{d(r)},\qquad
 h(r)=\frac{2(1+5r)}{d(r)},\qquad
 h'(r)=-\frac{20+100r+250r^2}{d(r)^2}<0.
 \tag{13}
\]

The formula for M_r here is restricted to the neutral space; its common
mode remains one. Since `v^Tv=2` and `v^TLv=6`, it determines h exactly
throughout the parameter interval, not merely at sampled endpoints.
Write `b=mathcal R+eta`. The observed record sets are contained in

\[
 \mathcal Y_+=\left[aT\tfrac{12}{11}-b,\;
                         aT\tfrac{840}{461}+b\right],\qquad
 \mathcal Y_0=[2aT-b,\;2aT+b].
 \tag{14}
\]

For the fixed declaration, exact arithmetic gives

\[
 Z=\frac{8193}{66977792},\qquad
 \mathcal R=\frac{9215}{34292629504},\qquad
 \inf\mathcal Y_0-\sup\mathcal Y_+
 =\frac{40665805}{3952225550336}>2^{-21}.
 \tag{15}
\]

An independent, coarser spectral bound also suffices: `L>=I` on the neutral
space implies `h(r)<=2/(1+r)`, so the separation is at least
`2aT/101-2b=555365/865888894976>2^-21`. Thus the conclusion does not depend
on fitting the exact response gain or on the last digits of a trajectory.

The pulse's actual supplied work and continuous loss are separately bounded:

\[
 \int_0^T |(LU)^Tf|\,d\tau\le8aTZ
 =\frac{8193}{137170518016},\qquad
 \int_0^T\tfrac14\|LU\|^2\,d\tau\le4TZ^2.
 \tag{16}
\]

Equation (4) still retains the signed work. The `2a^2 T` allowance in (10)
is an energy-growth inequality, not an equality for delivered work or an
autonomous reservoir. No hardware or observation mechanism is inferred
from the bounded additive error in (14).

## 7. Why form, winding and storage do not close the state

This comparison also admits a constructive obstruction inside its prepared
domain. Fix `w=+1`, one node i, `s=delta/2>0`, and two states

\[
 U_+=U_-=0,\qquad \theta_\pm=\pm s(e_i-\mathbf1/5).
 \tag{17}
\]

They have the same complete form vector, both conserved means, winding and
own storage `V_*+2*cos(2*pi/5)*(1-cos(s))`. Their changed edge gaps are the
same pair `kappa-s,kappa+s` in opposite positions; all other gaps equal
kappa. Both states satisfy the full preparation radius.

At the same independently specified input, direct edge summation gives

\[
 (U_i')_\pm=(M_rf)_i\mp\tfrac\gamma2
 (M_{ii}-M_{i,i+1})\cos(2\pi/5)\sin s,\qquad
 M_{ii}-M_{i,i+1}=\frac{1+5r}{d(r)}>0.
 \tag{18}
\]

Reflection symmetry makes the two adjacent entries of M equal; their even
terms cancel in this node's response. The two rates are strictly different.
Therefore form, even augmented by winding, means and storage, is not an
exact autonomous state on this class. The finite envelope (12) remains
valid because it retains a bound on the discarded phase information.
This obstruction does not exclude a controlled approximation or a memory
description retaining hidden initialization and the supplied input.

## 8. Executable certificate and remaining scope

The [detached arithmetic owner](../../src/tnfr/physics/_organized_effective_law.py)
rebuilds (8), (10)-(16) from admitted primitive budgets. It reports identity
and record separation separately: a failed sufficient bound is not a
trajectory failure or an impossibility theorem. Its variable-budget interface
supports arithmetic controls, not a new search or evaluation campaign.
The [shared C5 preparation owner](../../src/tnfr/physics/_organized_c5_preparation.py)
evaluates (8) and supplies this collar's radius and barrier to the unloaded,
loaded-sensor and recovery consumers. It admits primitive budgets without
inferring that a supplied state belongs to the preparation.
The [independent controls](../../tests/physics/test_organized_effective_law.py)
use edge-built matrices, symbolic polynomial identities, direct sine sums,
boundary coercivity and deliberate loss of admission. They preserve the
existing frozen evidence and execute no reserved response.

This closes the declared prepared-reference identity/prediction question.
It does not select r, derive this law from the nodal identity, inherit it
from the 45-node source, or supply acquisition, operational preparation,
physical identification or a sensor/clock model. The exact state and work
requirements now constrain the separate finite-window observation question;
only the [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
activates that work.
