# A finite input on the actual five-child organization

<a id="organized-electrical-finite-actuation"></a>

## Prospective declaration

The [electrical identification owner](ORGANIZED_ELECTRICAL_IDENTIFICATION.md#organized-electrical-operational-identification)
excludes a literal instantaneous relative-phase jump with finite work in
the fixed passive reference. This construction replaces the first positive
intervention with one finite, externally supplied form input. It does not
derive the occurrence of that input from an autonomous TNFR source.

| Item | Fixed choice and scope |
| --- | --- |
| Actual source | The five independently recovered C9 children and both acquired outer-zero and outer-one hosts of the [joint resource box](ORGANIZED_JOINT_RESOURCES.md); original component allowances at most `1e-6`, quarter formation 200, recovery dwell `8e10`, recovered norm `q=1e-8`, decay power 80, and outer formation 200. Retain the reached state, all fine directions and conserved means |
| Entry history | Continue the host unforced for the original first observation window `h0=1e-5`; intervene immediately before its first `+epsilon*k0` event. No donor, later phase event, contact or old response verdict is a prerequisite |
| Support and law | C9-by-C5, nodes `9*c+j`, all 90 unit edges, held unit capacities; `x'=(-L*x+gamma*S(theta))/4+T*f`, `theta'=gamma*L*x/4`, `gamma=1/(1023*pi)`, full-circle pairwise sine |
| Clock | Prime is `tau=(1023/1024)*t`. Pulse elapsed time `u` runs from 0 to `d=1`. Physical time and energy scales remain uncalibrated |
| Input information | Known marked support, gamma, clock, amplitude and duration only. Five neutral child channels, each repeated over nine nodes. No actual form or phase reading, hidden-state feedback, reset, phase input or support change |
| Waveform | `a=1/10`, marked child zero, `k=e0-1/5`, `s(r)=3*r^2-2*r^3`, `r=u/d`; the independent form forcing is (2) below, zero outside the pulse |
| Endpoint decision | Each actual form magnitude below `1/500`; each form change from pulse entry below `1/250`; each lifted phase differs from its actual entry value plus `a*k` by less than `2e-6`. These are approximation policies, not exact form restoration |
| Identity | Preserve the five internal winding-one cycles, all nine respective outer windings and a strict within-child phase collar throughout the prefix and pulse |
| Resources | Peak full structural storage below `2e8`, integrated full dissipation below `1e9`, absolute supplied work below `3e8`. Report positive-only work bounded by that absolute work and independent necessary lower bounds. No old jump-work allowance is reused |
| Assessment | One exact rational/outward dyadic128 analytical assessment after committing declaration, source and driver. No trajectory, parameter search, actuator-error model or frozen-producer replay |
| Handoff | Release the input without a state jump and retain the actual endpoint. The original fifteen-event word, twenty-five-reading schedule and inverse are not admitted for the changed history |

These deliberately conservative finite resource ceilings define a
construction, not an optimum or laboratory feasibility claim. Source
formation and attachment costs remain in their original ledger; pulse work
is a new term. Failure of a sufficient bound leaves this candidate unavailable
and does not authorize retuning its frozen choices.

## 1. Independently supplied waveform and complete evolution

Write `T` for repetition over the nine slots and
`L=I5 tensor L9 + L5 tensor I9`. The marked vector k is neutral. On C5,

\[
 v=L_5^+k=(2/5,0,-1/5,-1/5,0)^T,\quad L_5v=k.
 \tag{1}
\]

This is the inverse on the neutral sector, with no grounded charge or
discarded common phase. Define prescribed offsets and forcing by

\[
 \begin{split}
 h(u)&=a k s(u/d),\\
 z(u)&=\frac{4a}{\gamma d}v s'(u/d),\\
 f(u)&=\frac{4a}{\gamma d^2}v s''(u/d)
           +\frac{a}{\gamma d}k s'(u/d).
 \end{split}                                             \tag{2}
\]

The primes on s denote differentiation of its dimensionless argument.
Exactly `h'=gamma*L5*z/4` and `z'=-L5*z/4+f`. Both z endpoints vanish;
h starts at zero and ends at a*k. Every vector in (2) has zero sum.
Bounded jumps of f when it is switched on or off are allowed; the states
remain continuous. Finite bandwidth and tracking error have not been admitted.

Equation (2) is an executable time function, independent of the response.
It is not asserted to solve the nonlinear unforced model. For the actual
solution define

\[
 y=x-Tz,\quad
 \eta=\theta-\theta_{\rm entry}-Th.
\]

The complete equations give, without linearizing the sine,

\[
 y'=-Ly/4+\gamma S(\theta)/4,\qquad
 \eta'=\gamma Ly/4,\qquad y(0)=x_{\rm entry},\ \eta(0)=0.
 \tag{3}
\]

The phase row is live; its background drift is part of the endpoint error.
The supplied signal preserves both global means. Smooth finite-dimensional
linear growth and bounded sine forcing give a unique solution on the whole
finite interval. No numerical integration is needed for the bounds below.

The [earlier collective construction](ORGANIZED_PARENT_COLLECTIVE_ACTUATION.md)
also used a uniform input and a same-input comparison, but consumed initial
collective observations on a different support. Here (2) consumes none of
those observations; its coefficients and source bounds are derived on C5.
This is detached continuous-law analysis, not a registered hybrid operator
or a change to the native relational runtime.

## 2. Actual source, residual state and identity

Rebuild isolated recovery from its primitive allowances and clock, then
the [outer source](ORGANIZED_OUTER_WINDING.md) from the recovered radius.
One equal recovery bound can cover the five independent sources without
equating their actual states. Retain both capture branches, acquisition
identity, complete law and attachment/storage premises. The outer theorem
supplies full Euclidean departures R,Z from its carrier and nominal form
bound M*. These cover the outer-zero control as well.

With `g=gamma_upper`, the initial unforced prefix admits

\[
 M_0=M_*+R+g h_0,
 \quad A_0,B_0=\mathcal T(R,Z,2g h_0),
 \tag{4}
\]

where the shared hidden-transport function is

\[
 \mathcal T(A,B,q)=\left(\frac{A+qB}{1-q^2},
                       \frac{B+qA}{1-q^2}\right),\qquad q<1.
 \tag{5}
\]

Its proof belongs to the [source comparison](ORGANIZED_ELECTRICAL_SOURCE.md).
Let A be an upper bound on the actual principal mean-phase gaps at pulse
entry: the largest nominal outer-one gap plus `2Z+4g*M0*h0` also bounds
the outer-zero control. Prefix internal gaps use B0 and external gaps retain
the same mean and fine errors. This does not select a nominal state in place
of an acquired one.

The maximum degree is four, `||L||_infinity<=8`, and `|S_i|<=4`.
Heat contraction applied to (3) yields

\[
 \|y(u)\|_\infty\le M=M_0+gd,\quad
 \|\eta(u)\|_\infty\le D=2gMd.                    \tag{6}
\]

Since both z endpoints vanish, the endpoint bounds are

\[
 \|x(d)\|_\infty\le M,\quad
 \|x(d)-x(0)\|_\infty\le (2M+g)d,\quad
 \|\theta(d)-\theta(0)-Ta k\|_\infty\le D.
 \tag{7}
\]

No small form bound is claimed during the pulse: z can be large.
For the within-child orthogonal projection Q, `QTf=0`, Q commutes with L,
and `QS(psi+T*Phi)=0`. Consequently (5) still transports the actual hidden
form and phase norms under the input, with

\[
 A_h,B_h=\mathcal T(A_0,B_0,2gd).                    \tag{8}
\]

Thus all internal gaps lie between `2*pi/9-2Bh` and `2*pi/9+2Bh`.
The strict collar margin is `pi/2-2Bh`. An external fine gap stays at
magnitude at most `A+|a|+2B0+2D`. Strictly positive internal gaps below
pi/2 and external magnitudes below pi/2 preserve every declared cycle
winding throughout the pulse. Winding retention is a finite identity result,
not all-future maintenance or invariance of the original observation prior.

There is a sharper comparison with the *unforced continuation of the same
actual entry state*. It is a proof device, not a measured branch required
by the actuator. Subtract that continuation shifted by `(Tz,Th)` from the
actual forced solution. A child-common phase displacement changes only two
external edges per slot, so its sine-current increment has infinity norm
at most `2|a|s`. Using `integral_0^1 s=1/2`, the sine Lipschitz bound 8
and the same domain `2gd<1`,

\[
 E_x\le\frac{g|a|d}{4(1-(2gd)^2)},\qquad
 E_\theta\le2gd E_x.                              \tag{9}
\]

At release this compares with `J_a(F_d(state_entry))`: first the real
unforced elapsed duration, then the phase translation. The order matters.
Neither (7) nor (9) identifies the endpoint with the old instantaneous
event or admits the original observation inverse.

## 3. Full storage, supplied work and unavoidable cost

The unchanged full storage is

\[
 H=\tfrac12x^TLx+\sum_{e=1}^{90}(1-\cos\delta_e),
 \quad H'=-\tfrac14\|Lx\|_2^2+(Lx)^TTf.
 \tag{10}
\]

Define `W=integral (Lx)^T Tf`, `Wabs=integral |(Lx)^T Tf|`,
`Wplus=integral max(0,(Lx)^T Tf)` and `Loss=integral ||Lx||^2/4`.
Then `H(d)-H(0)=W-Loss` and `Wplus<=Wabs`. None is paid from signed
hidden storage or the old jump-work estimate. The exact uniform port identity
`(Lx)^T Tf=9(L5 X)^T f` remains valid off the carrier; independent physical
calibration is still required to assign amperes, seconds or joules.

For `ell=gamma_lower>0`, the cubic extrema give

\[
 Z_m=\frac{12|a|}{5\ell d},\quad
 J_z=\frac{24|a|}{5\ell d},\quad
 F_m=\frac{48|a|}{5\ell d^2}+\frac{6|a|}{5\ell d},\quad
 J=J_z+8M.                                          \tag{11}
\]

These bound respectively `||z||infinity`, `||L5 z||infinity`,
`||f||infinity` and `||Lx||infinity`. Counting all nodes and edges gives

\[
 \sup H\le180(Z_m+M)^2+180,\quad
 Loss\le45dJ^2/4,\quad W_{abs}\le45dJF_m.
 \tag{12}
\]

The last bound also bounds the sum of positive supplied work over the
individual channels: it was obtained by summing absolute node powers before
integration, so it does not require cancellation between supplying and
absorbing channels.

Finite upper bounds alone could conceal a costly construction. The phase
row also forces a lower cost for the **actual** collective increment
`Delta Phi=a*k+e`, with `||e||infinity<=D`. Orthogonal decomposition gives

\[
 Loss\ge\frac{36}{\gamma^2d}\|\Delta\Phi\|_2^2,\qquad
 \sup\tfrac12x^TLx\ge
 \frac{72}{\gamma^2d^2}\Delta\Phi^TL_5^+\Delta\Phi.
 \tag{13}
\]

These are the C5 electrical pulse inequalities in structural units, with
the actual lift and all fine storage retained. Put `v=max(0,|a|-2D)`.
The marked-minus-adjacent increment has magnitude at least v;
`||e0-e1||^2=2` and `(e0-e1)^T L5 (e0-e1)=6`. Therefore

\[
 Loss\ge18v^2/(g^2d),\qquad
 \sup\tfrac12x^TLx\ge12v^2/(g^2d^2).               \tag{14}
\]

They are necessary dissipative and peak-storage costs, not independently
measured energies or an optimized input. Net work additionally depends on
the initial storage through (10); in particular
`Wplus>=max(0,Loss_lower-H_entry_upper)` if the actual entry storage upper
bound is retained. The tiny supplied gamma makes a rapid
phase translation require a large transient form; an instantaneous jump's
small storage difference cannot account for this continuous cost.

## 4. Instrument and evidence boundary

The [pulse and admission owner](../../src/tnfr/physics/_organized_electrical_actuation.py)
admits represented-real primitives before arithmetic, evaluates the neutral
waveform and bounds the full source. It reuses the original recovery,
outer-acquisition and hidden-transport owners without changing frozen
producers. The [single-assessment driver](../../benchmarks/organized_electrical_actuation.py)
uses the shared exclusive source/configuration receipt lifecycle.
The [independent controls](../../tests/physics/test_organized_electrical_actuation.py)
exercise algebra, full balances, invalid domains and synthetic source failures;
they do not evaluate the reserved source decision.

An externally prescribed input and a derived collective interaction are
different claims. A future autonomous actuator would require its own TNFR
state, preparation, coupling and joint energy balance; (2) derives none of
those. Likewise this conditional finite intervention supplies no physical
identification of electric charge or fundamental matter.

The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns continuation. A complete finite observation protocol must retain the
elapsed intervals, actual release forms, cumulative work and inference
errors. Reusing the frozen instantaneous word or its verdict would omit
the very state evolution established here.

<a id="finite-actuation-result"></a>
## 5. Frozen decision and its handoff

Declaration, consuming source and driver were committed at
`9488f528226679b649724634ff950a917c57fb5f` before the single assessment.
The preserved [attempt](../../docs/assets/organized_electrical_actuation/attempt.json)
and [result](../../docs/assets/organized_electrical_actuation/result.json)
record one evaluation and verdict `finite_organized_electrical_actuation_admitted`.
All twelve prerequisite flags, six source margins and ten decision margins
pass. No source trajectory or physical experiment was evaluated.
The [read-only evidence controls](../../tests/physics/test_organized_electrical_actuation_evidence.py)
reconstruct source receipts, support, primitive bounds and the decision
without executing the assessor again.

The following are outward decimal summaries of exact archived bounds, in
the declared structural units:

| Quantity | Certified bound |
| --- | --- |
| Actual endpoint form magnitude | less than `0.001319` |
| Form change from actual pulse entry | less than `0.002948` |
| Phase error relative to the entry-plus-command objective | less than `8.203e-7` radians |
| Hidden form / phase Euclidean norms throughout the pulse | less than `3.049e-8` / `6.004e-8` |
| Minimum outer acute-gap margin | greater than `0.1975` radians |
| Circular collar margin | greater than `1.5707962` radians |
| Form / phase departure from the shifted unforced continuation | less than `7.779e-6` / `4.841e-9` |
| Peak full storage | less than `1.071e8` |
| Full continuous loss | greater than `1.859e6` and less than `2.678e7` |
| Peak form storage, necessary lower bound | greater than `1.239e6` |
| Positive supplied work, necessary lower bound | greater than `1.858e6` |
| Absolute supplied work and sum of positive channel work | less than `2.410e8` |

The supply lower bound is a direct consequence of (10), (14) and
`H_entry<=180*M0^2+180<181`; it does not treat dissipation alone as supplied
work or spend an undeclared initial reservoir.

The source remains the reached five-child state, with its unknown means and
fine deviations. Its endpoint has the declared approximate phase translation,
all five internal and nine outer windings retained, and a finite residual
form change. The input is then released without resetting that endpoint.

This closes one finite-actuation construction. The large continuous cost is
part of the result; finite realizability does not imply efficient realization.
Neither the original jump-work budget nor its instantaneous readout inverse
is restored by this admission. The separate
[finite observation theorem](ORGANIZED_FINITE_OBSERVATION.md#finite-observation-result)
admits a complete continuing history through a new current correction and
inverse, retaining its actual state, timing and cumulative work. An autonomous
source of the input and physical identification remain separate open questions.
