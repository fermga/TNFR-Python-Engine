# Observing collective winding with finite form inputs

<a id="organized-finite-observation"></a>

## Prospective protocol and decision

The [single finite pulse](ORGANIZED_ELECTRICAL_ACTUATION.md) admitted one
approximate phase intervention. This question admits a complete continuing
observation history and derives its inverse anew. Known inputs and collective
form measurements must suffice; actual phases, source labels and hypothetical
reference readings are not inverse inputs.

| Item | Fixed choice and scope |
| --- | --- |
| Source | Both actual outer-zero and outer-one acquired five-child hosts, immediately after outer formation. Retain primitive component allowances at most `1e-6`, quarter formation 200, recovery dwell `8e10`, recovered norm `q=1e-8`, decay power 80, outer formation 200, all fine state and conserved means |
| Larger inference class | Unknown initial signed forms `abs(x_i)<=X=1/100`; `theta[c,j]=psi_j+Phi_c+z[c,j]`, `psi_j=2*pi*(j-4)/9`, `sum_j z[c,j]=0`, `abs(z[c,j])<=zeta=1e-6`; all principal mean gaps have magnitude at most `A=7/5`. Initial outer winding is in `{-1,0,1}`. Inclusion of both acquired families must be proved; acquisition of every member of this larger class is not claimed |
| Support and complete law | Fixed C9-by-C5, nodes `9*c+j`, 90 unit edges, held unit capacities; `x'=(-L*x+gamma*S(theta))/4+T*f`, `theta'=gamma*L*x/4`, `gamma=1/(1023*pi)`, full-circle pairwise sine |
| Clock | Prime denotes `tau=(1023/1024)*t`. Observation begins at the actual formed source; no earlier word, donor or contact is consumed |
| Input program | Mark children `c=0,1,2,3,4` in that order. In each block interleave three unforced plateaus of duration `h=1e-5` with three cubic form pulses of duration `d=1` and phase-command increments `epsilon,-2*epsilon,epsilon`, `epsilon=1/10`. Each plateau precedes its pulse, at cumulative command `0,+epsilon,-epsilon`. All inputs are fixed before readings |
| Actual duration | One block is `3*(d+h)` and the complete word is `Tobs=15+15*h`. Release input at the final pulse endpoint, retaining the reached state |
| Readings | All five actual mean signed forms at both endpoints of every plateau: ordered shape `5 blocks x 3 plateaus x 2 endpoints x 5 children`, 150 scalar readings. No phases, derivatives, reset, copies or reference branch are observed |
| Uncertainty | Each raw scalar has arbitrary additive error at most `eta_raw=1e-14`; correlations are unrestricted. Input waveform, clock, support, gain and sampling times are ideal stipulated quantities in this mathematical protocol |
| Inference | Use the diffusion-corrected increments (8), then the two quadratures and complete cycle constraint. Identify the actual initial winding for every member of the admitted class and every allowed error, or report a scoped unavailable certificate. Necessary compatibility of arbitrary records is not their joint realizability |
| State and work | Preserve all five internal winding-one cycles, all nine respective outer windings and the within-child phase collar throughout the complete history. Bound release errors, full storage, continuous loss and positive supplied work |
| Resource decision | Peak full storage below `5e8`, integrated loss below `2e9`, absolute supplied work below `2e10`, in structural units. Report necessary cost floors separately. Old jump-work and single-pulse budgets do not transfer |
| Evidence | Freeze declaration, consuming source and driver before one rational/outward dyadic128 analytical assessment at the stated source/noise corner. No trajectory, raw physical data, adaptive input, scan or frozen-producer replay |

The supplied input is an intervention used to obtain information. Its
autonomous occurrence and a physical measurement apparatus remain open.
The question concerns collective observation of an already formed identity,
not acquisition by measurement or identification of a material particle.

## 1. One continuous history and the global phase bound

Let R repeat a child vector over nine slots (called T in the complete row
in the declaration), and use `L=I5 tensor L9+L5 tensor I9`. For marked
child c set `k_c=e_c-1/5` and `v_c=L5^+ k_c`. On a ramp with signed
increment a and local fraction `r=u/d`, the reused waveform is

\[
 s(r)=3r^2-2r^3,\quad
 z(u)=\frac{4a}{\gamma d}v_c s'(r),\quad
 f(u)=z'(u)+L_5z(u)/4.                              \tag{1}
\]

The command H evolves from its current value by `a*k_c*s`; it is constant
on the plateaus. Both z endpoints vanish. The complete H is continuous,
with `H'=gamma*L5*z/4`, starts and ends at zero, and never changes any edge
by more than epsilon. The middle increment has magnitude `2*epsilon`,
but crosses from `+epsilon` to `-epsilon`; this is not an excursion of
`2*epsilon`. The fine input Rf is zero on plateaus and after release.
Bounded switches in f jump no state. Endpoint readings are state values;
the plateau estimates use the unforced one-sided dynamics.

For the one actual solution throughout the entire program, put

\[
 y=x-Rz,\qquad e=\theta-\theta(0)-RH.
\]

Every switching boundary retains y and e continuously. The exact rows are

\[
 y'=-Ly/4+\gamma S(\theta)/4,\quad
 e'=\gamma Ly/4=-\gamma y'+\gamma^2S(\theta)/4.      \tag{2}
\]

With `g=gamma_upper`, heat contraction and `||S||infinity<=4` give

\[
 \|y(t)\|_\infty\le M=X+gT_{obs},\qquad
 e(t)=\gamma\{x(0)-y(t)\}
           +\frac{\gamma^2}{4}\int_0^t S(\theta(u))\,du,
\]
\[
 \boxed{\|e(t)\|_\infty\le E=2gX+2g^2T_{obs}}.
                                                               \tag{3}
\]

This exact integration uses the live phase row and form diffusion. It
avoids treating a transient large prescribed form offset as uncontrolled
phase drift. No stationary background, phase feedback or reset is used.
Global form and lifted-phase sums are conserved because every input is
neutral and the edge currents sum to zero. Common means need not vanish
in the larger inference class; the actual acquired hosts retain their
original zero totals.

At every sampling endpoint z vanishes, so `abs(X_c)<=M`. At final release
both z and H vanish, yielding `||x_end||infinity<=M` and
`||theta_end-theta_initial||infinity<=E`. For example
`||x_end-x_initial||infinity<=M+X`; neither state is restored exactly.
The subsequent unforced flow must start at this actual endpoint.

## 2. Complete identity and actual-source inclusion

Initial internal fine gaps are `2*pi/9` plus differences of z; outer fine
gaps are the admitted principal mean gaps plus those residual differences.
Using (3), sufficient whole-history identity margins are

\[
 \begin{split}
 m_{internal,+}&=2\pi/9-2\zeta-2E,\\
 m_{internal,acute}&=\pi/2-2\pi/9-2\zeta-2E,\\
 m_{outer,acute}&=\pi/2-A-\epsilon-2\zeta-2E.
 \end{split}                                                \tag{4}
\]

Strict positivity prevents a cut crossing on each of the five internal
and nine outer marked cycles. Every initial outer identity is retained,
including negative winding in the larger admitted class. Removing the
current child mean from e changes a within-child residual by at most 2E,
so a sufficient strict circular-collar margin is

\[
 m_{collar}=\pi/2-2\zeta-4E.                       \tag{5}
\]

The actual source is rebuilt from the [recovery and resource primitives](ORGANIZED_JOINT_RESOURCES.md)
and [outer acquisition](ORGANIZED_OUTER_WINDING.md), not from the old
observation or contact reports. If the latter proof gives nominal form
bound M*, nominal outer-gap bound A*, and full Euclidean departures R,Z,
the inclusion checks are

\[
 M_*+R<X,\qquad Z<\zeta,\qquad A_*+2Z<A.           \tag{6}
\]

Within-child mean removal is an orthogonal projection, so Z still bounds
its residual; there is no state change in this decomposition. These
bounds also cover the outer-zero control with stationary nominal carrier.
Independent child acquisition/recovery, matching law and clock, support,
capture, and historical storage/work premises remain necessary. Equality
of their five resource bounds does not equate five actual preparations.
The first plateau supplies the unforced first window; no extra prefix is
inserted before it.

## 3. Form readings isolate the interaction current

Let U be the five actual child mean forms. Internal sine currents cancel
exactly within a child, including off the carrier, leaving two external
currents per child. On an unforced plateau,

\[
 U'=-L_5U/4+\gamma\bar S/4.                         \tag{7}
\]

The observer supplies two ordered vectors P,Q of measured mean forms at
that plateau's endpoints. For the receiver `r=(c+1) mod 5`, form the exact
primitive combination

\[
 D=Q_r-P_r+\frac h8\{L_5(P+Q)\}_r.                \tag{8}
\]

All terms are in form units. The second term compensates diffusion rather
than equating the observed increment to a sine-current response. Integrating
(7) shows that its true-value counterpart equals
`gamma/4 * integral(bar S_r)` plus the trapezoid error in the diffusion
term. The fine unforced rows bound each mean's second derivative by

\[
 M_2=4M+2g+4g^2M.                                  \tag{9}
\]

Since `||L5||infinity=4`, the trapezoid error in D is at most
`M2*h^3/12`. During any plateau the command is fixed. Each of the two
external phase differences differs from its initial shifted counterpart
by at most 2E, so the mean current differs by at most 4E. The plateau
integral may therefore be compared to the *initial* shifted current
with error at most `g*h*E`, irrespective of its block's position.

The two receiver-reading coefficients in (8) are `-1+h/4` and `1+h/4`;
the four neighboring endpoint coefficients are `-h/8`. With `h<4`, their
absolute sum is exactly `2+h/2`. Correlations need not be discarded.
For every measured corrected increment,

\[
 D=\frac{\gamma h}{4}\bar S_r(\theta(0)+RH_q)+\delta,
 \quad |\delta|\le b=g h E+M_2h^3/12+(2+h/2)\eta_{raw}.
                                                               \tag{10}
\]

No derivative or phase is measured. Reads from different plateaus belong
to one carried state history; the initial state in (10) is a proof
reference, never a copied or observed branch.

## 4. Quadratures, circular geometry and the initial winding

For interface c define the initial fine edge phasor by

\[
 C_c+iS_c=\frac19\sum_j e^{i d_{c,j}},\quad
 d_{c,j}=a_c+z_{c+1,j}-z_{c,j},\quad |a_c|\le A.     \tag{11}
\]

The three plateau levels of block c affect just its receiver's boundary
with marked child c. Their initial currents satisfy

\[
 \bar S_r^{\pm}-\bar S_r^0
   =\mathord\pm\sin\epsilon\,C_c
                          +(1-\cos\epsilon)S_c.
\]

Use the three *new* corrected increments to compute

\[
 O_c=D_+-D_-,\quad V_c=D_++D_--2D_0.
\]
\[
 O_c=\frac{\gamma h\sin\epsilon}{2}C_c+\delta_C,
 \quad V_c=\frac{\gamma h(1-\cos\epsilon)}2 S_c+\delta_S,
 \quad |\delta_C|\le2b,\quad |\delta_S|\le4b.        \tag{12}
\]

The coefficients and initial class in (12) exactly match the shared
[quadrature-to-cycle inverse](ORGANIZED_OUTER_OBSERVATION.md).
Consequently its existing geometry and circular kernel apply unchanged
after the new primitive reading combinations and new complete-history
errors have been established. This transfers only that algebraic inverse,
not the earlier matched-branch or instantaneous sequential theorem.

In particular `C_c>=cos(A+2*zeta)>0`, the phasor modulus is at least
`cos(2*zeta)`, and its circular argument differs from a_c by at most
`2*zeta`. The kernel intersects those constraints, bounds each mean gap,
and tests their sum against `2*pi*w`, `w in {-1,0,1}`. It retains shared
outward rounding and the existing angle-arithmetic allowance. A uniform
returned cycle width strictly smaller than `2*pi` admits at most one
winding; an actual valid history supplies its included winding. This proves
conditional identification for every admitted state and error record.

Malformed primitive readings reject before arithmetic. Impossible raw
magnitudes, incompatible phasor annuli or absent cycle candidates remain
explicit. A candidate or even a singleton for arbitrary supplied records
does not authenticate their origin, their law, their error bounds or their
joint realizability. Noise that defeats the uniform certificate leaves
protocol admission unavailable, even if particular data look decisive.

## 5. Complete continuous resource account

The storage and balance remain

\[
 H=\tfrac12x^TLx+\sum_e(1-\cos\delta_e),\qquad
 H'=-\|Lx\|_2^2/4+(Lx)^TRf.                        \tag{13}
\]

Let `ell=gamma_lower`. The largest ramp has amplitude `2*epsilon` and
duration one. With the full-history residual M, the
[pulse extrema](ORGANIZED_ELECTRICAL_ACTUATION.md) give

\[
 Z_m=\frac{24\epsilon}{5\ell},\quad
 J=\frac{48\epsilon}{5\ell}+8M,\quad
 F_m=\frac{108\epsilon}{5\ell}.
\]
\[
 \sup H\le180(Z_m+M)^2+180,\quad
 Loss\le45T_{obs}J^2/4,\quad
 W_{abs}\le45\cdot15 JF_m.                         \tag{14}
\]

The input is present only over the fifteen ramps. The last bound also
bounds total positive supplied work, including the sum of positive
individual channel contributions. It uses absolute powers before summing,
so simultaneous supplying and absorbing channels do not conceal costs.
Storage change, loss and supplied work are distinct ledger quantities;
old event work and signed hidden storage supply no reservoir.

Returning the command to zero does not erase the cost. On each ramp,
the actual collective phase increment is its command plus at most 2E
componentwise. The marked-minus-adjacent increment has magnitude at least
`delta_a=max(0,abs(a)-4E)`. The previously proved C5 contrast inequalities
then imply, on disjoint ramp intervals,

\[
 Loss\ge\frac{90}{g^2}
             (2\delta_{\epsilon}^2+\delta_{2\epsilon}^2),\qquad
 \sup\tfrac12x^TLx\ge12\delta_{2\epsilon}^2/g^2.
                                                               \tag{15}
\]

Plateau loss is nonnegative. Since initial storage is at most `180X^2+180`,
positive supplied work is at least `max(0,Loss_lower-180X^2-180)`.
These structural bounds prove neither calibrated joules nor optimal cost.
Finite bandwidth, tracking uncertainty and physical sampling remain additional
premises before interpreting this protocol as a realizable measurement.

## 6. Instrument and handoff boundary

The [pure protocol, word and inverse](../../src/tnfr/physics/_organized_finite_observation.py)
admit only their primitive time, noise and ordered form readings. The
[source assessor](../../src/tnfr/physics/_organized_finite_observation_source.py)
separately rebuilds actual acquisition and inclusion, then assesses the
fixed corner through the [exclusive driver](../../benchmarks/organized_finite_observation.py).
The [independent controls](../../tests/physics/test_organized_finite_observation.py)
exercise synthetic algebra and source failures without generating a reserved
response. No existing frozen driver or inverse is rewritten.

The final state belongs to this new finite-input history. Neither the old
post-observation contact response nor its source-specific electrical error
certificate transfers merely because the same winding was retained. Their
full source, means, support, law and clock must match the new endpoint.
The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone selects the next dependency. No physical dataset, autonomous actuator,
laboratory calibration or material identity is admitted by the inverse.
