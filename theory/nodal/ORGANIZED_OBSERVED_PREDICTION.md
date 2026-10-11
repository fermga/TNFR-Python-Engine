# Predictive sufficiency of finite collective form records

<a id="organized-observed-prediction-declaration"></a>
## Fixed observation-to-future comparison

The [active information gate](../research/FIVE_STAGE_EXECUTION_PLAN.md#organized-observed-prediction-admission)
removes the exact collective initialization consumed by the
[preparation-based forecast](ORGANIZED_COMPUTABLE_PREDICTION.md).
It asks whether two complete histories compatible with one existing noisy
record must have nearby future mean forms. The target is an output diameter,
not an inverse for every phase or hidden constituent coordinate.

| Item | Fixed scope |
| --- | --- |
| Complete fine law | Unit C9-by-C5 support, node order `9*c+j`, 90 unit edges, held unit capacities, full-circle pairwise sine; `x'=(-L_f*x+gamma*S(theta))/4+R*f`, `theta'=gamma*L_f*x/4`, `gamma=1/(1023*pi)` |
| Clock and horizon | Structural `tau=(1023/1024)*t`; local time `v=tau-200` runs from 0 to `T=16`. Continue the actual reached state after the word; no reset, extra event or changed support |
| Observation-start class | Original `abs(x_i)<=X=1/100`, `theta=psi+R*Phi+z`, `psi_j=2*pi*(j-4)/9`, child-zero hidden means, `abs(z_i)<=1e-6`, principal collective gaps at most `7/5`, winding in `{-1,0,1}`; additionally retain `||u||2<=a0=31/10^9`, `||z||2<=b0=601/10^10` |
| Unknown collective state | All five initial mean forms remain unknown, including their conserved common mean in `[-X,X]`. Fix only the unobservable common phase gauge by choosing initial `Phi_0=0` after a common rotation; no relative phase, winding or attachment mean is supplied to an inverse |
| Input and observations | Unchanged fifteen cubic ramps and fifteen preceding plateaus, marked children `0,1,2,3,4`, increments `1/10,-2/10,1/10`, ramp duration 1 and plateau duration `1e-5`; all five mean forms at both endpoints of every plateau, in original order, 150 raw scalars |
| Raw errors | Each scalar has arbitrary correlated additive error at most `eta=1e-14`. Input waveform, support, capacities, gain and sample clock are exact conditional premises; no apparatus error is silently inferred to vanish |
| Future target | Five mean forms at `tau=216`, or local time 16. The last reading is at `214.00015`; prediction includes the final ramp and the unforced tail, a span of `1.99985` |
| Decision | Joint overlap of the complete 150-reading record sets must imply a coordinatewise future diameter at most `2e-13`, uniformly over the declared class. This admits an abstract coordinate-midrange radius at most `1e-13`; a future reading adds its own `eta` |
| Static certificate | Fifteen formal columns: five initial forms, five scaled initial edge sines and five scaled initial edge cosines. Enclose the resulting 150-by-15 sampled heat-response matrix and 5-by-15 future matrix; evaluate no nonlinear target trajectory or raw response |
| Coefficient computation | Order-16 Taylor coefficients, shared dyadic128 interval jets, maximum cell `1/8`, separate input stages. Partition each stage from its left endpoint into full cells and a final shorter cell: 15 plateau cells, 120 ramp cells, eight tail cells. Each coefficient's propagated radius must remain at most `1e-12` |
| Weight proposal | Convert matrix midpoints to IEEE binary64, form `Mhat.T @ Mhat`, and use one NumPy solve for `K_raw=solve(Mhat.T @ Mhat, Mhat.T)`. Independently round its last ten rows and `Nhat @ K_raw` to their nearest multiples of `2^-40`, ties to even, obtaining `K_b` and W. Record the proposal backend; validate the retained rational weights against the interval matrices |
| Numerical policy | One fixed matrix assessment, no row selection, parameter scan, refinement, precision escalation, alternate inverse search or retry. Floating-point least squares proposes weights only; exact/outward residual inequalities decide admission |
| Failure and evidence | A singular/nonfinite proposal, coefficient-budget failure, nonpositive bootstrap denominator or unmet diameter is a scoped unavailable certificate, not an impossibility theorem. Freeze this owner, consuming source and driver before matrix evaluation; retain source/configuration, all matrix enclosures, rational weights, residuals and verdict. No frozen producer, source assessor or reserved response is replayed |

The class contains the previously acquired sources, but acquisition of every
member is not asserted. This is a structural information comparison with an
unchanged complete law. A numerical result cannot supply physical preparation,
calibration, a laboratory clock, a unique fundamental law or a physical identity.

## 1. Fine-source uncertainty is retained on both sides

For each valid fine history introduce the carrier with its own actual initial
means. Those means are unknown to the observer; this is a proof comparison,
not an initialization procedure. Rebuild the shared
[hidden transport](ORGANIZED_COMPRESSED_MEMORY.md#2-hidden-drift-without-a-hidden-solver)
from the declared primitive bounds, with `g=1/3000`:

\[
 s=2gT,\qquad
 Z=\frac{b_0+s a_0}{1-s^2},\qquad
 \rho_0=\frac{gZ^2}{9},\qquad
 E_0=\frac{T\rho_0}{1-(gT)^2}<2.3\cdot10^{-18}.
 \tag{1}
\]

Both denominators are positive. The projected nonlinear sine remainder and
heat contraction give this uniform Euclidean mean-form error, hence the same
coordinate error, on the full 16-unit history. It includes the hidden
initialization without identifying its variances. It is not the much smaller
preparation-specific error requiring known attachment means.

If two fine histories admit one common raw record, their true forms differ
by at most `2*eta` at every one of the 150 coordinates simultaneously.
Their two carrier histories consequently satisfy

\[
 \|u(t_j)\|_\infty\le\delta:=2(\eta+E_0)
 \quad\text{at all 30 vector sample times}.
 \tag{2}
\]

No independence or cancellation of errors is assumed. Conversely, (2) alone
does not authenticate a raw record or prove joint realizability. At the future
endpoint, a carrier-difference bound must again add `2*E0` for the two fine
histories.

The original identity proof also extends to this containing class through
local time 16. Substitute `T=16` in its full fine-state bound
`E_f=2gX+2g^2T` and retain the
[internal, outer and collar margins](ORGANIZED_FINITE_OBSERVATION.md#2-complete-identity-and-actual-source-inclusion).
The known command never shifts an edge by more than `1/10` and vanishes
after release; all those margins remain strictly positive. This is a
whole-history fine-state statement, independent of whether the new information
certificate passes. Work, storage and the additional unforced loss retain
their [separate account](ORGANIZED_COMPUTABLE_PREDICTION.md#4-identity-and-work-on-the-unforced-continuation).

## 2. A static design with unknown circular phases

Let `L=L5`, `A=L/4`, and orient edge e from child e to child `e+1 mod 5`.
Let B have tail -1 and head +1, so `L=B B^T` and
`S5(Phi)=-B sin(B^T Phi)`. Remove the exact prescribed offsets as in the
[carried forecast](ORGANIZED_COMPUTABLE_PREDICTION.md#1-exact-removal-of-the-commanded-offsets):

\[
 Y=U-z_{cmd},\qquad \Psi=\Phi-H,\qquad
 Y'=-AY+\frac\gamma4 S_5(\Psi+H),\qquad
 \Psi'=\gamma AY.
 \tag{3}
\]

At every reading and at the future endpoint, `z_cmd=0`, so `Y=U`.
The two compared histories share H and every input stage. Their difference
`u=Y^(1)-Y^(2)` equals their form difference also between readings.

For each history let `a_e=(B^T Psi(0))_e`, understood modulo `2*pi`, and put

\[
 p=\left(u(0),b_s,b_c\right),\qquad
 (b_s)_e=\frac\gamma4\left(\sin a_e^{(1)}-\sin a_e^{(2)}\right),\qquad
 (b_c)_e=\frac\gamma4\left(\cos a_e^{(1)}-\cos a_e^{(2)}\right).
 \tag{4}
\]

These fifteen formal coefficients need not be independently realizable.
Allowing arbitrary coefficients enlarges the comparison and cannot invalidate
a bound for the actual subset. Initial-edge phasors are independent of the
common phase gauge and branch representatives. No real phase difference
between the two histories, common winding or angle inverse is required.

Write `h(v)=B^T H(v)`. Define the static heat response V by

\[
 V'=-AV-B\left[b_s\mathbin\odot\cos h
                      +b_c\mathbin\odot\sin h\right],
 \qquad V(0)=u(0).
 \tag{5}
\]

It is linear in p. Let M stack its five coordinates at all thirty sample
times, and let N be its five-coordinate map at time 16. Thus
`V_samples=M p` and `V(16)=N p`. M and N depend only on the declared support,
clock and known command; computing them neither chooses an unknown state nor
evaluates a nonlinear future response.

## 3. A difference remainder rather than an absolute model bias

Use `B_* = max(||b_s||infinity,||b_c||infinity)` and
`U_* = sup_[0,T] ||u||infinity`. Since the carrier has two incident sine
currents per node, `||S5||infinity<=2`. Equation (3) and heat contraction imply

\[
 \|Y(v)\|_\infty\le X+gv/2,\qquad
 \Psi(v)-\Psi(0)=\gamma\{Y(0)-Y(v)\}
                +\frac{\gamma^2}{4}\int_0^v S_5(\Psi+H)\,ds.
\]
\[
 \|\Psi(v)-\Psi(0)\|_\infty\le2gX+g^2T,
 \qquad D:=4gX+2g^2T.
 \tag{6}
\]

Hence each endogenous edge-phase drift d has magnitude at most D. For
`q_k=exp(i*a_e^(k))`, the difference left after freezing the two initial
phasors is

\[
 q_1(e^{id_1}-1)-q_2(e^{id_2}-1)
 =(q_1-q_2)(e^{id_1}-1)+q_2(e^{id_1}-e^{id_2}).
 \tag{7}
\]

Its modulus is at most `D*abs(q1-q2)+abs(d1-d2)`.
The scaled phasor difference obeys
`(gamma/4)*abs(q1-q2)<=sqrt(2)*B_*`.
Moreover `||B^T A||infinity=2`, so the complete phase row gives
`abs(d1-d2)<=2g integral_0^v ||u(s)||infinity ds`.
There are two incident edges per node and `2*sqrt(2)<3`. Consequently the
exact pair row can be written

\[
 u'=-Au-B[b_s\odot\cos h+b_c\odot\sin h]+r,
 \quad \sum_c r_c=0,
 \quad
 \|r(v)\|_\infty
 \le3D B_*+g^2\int_0^v\|u(s)\|_\infty\,ds.
 \tag{8}
\]

This remainder vanishes with differences. Substituting an absolute
single-history phase error here would create an unrelated error floor and
would not prove a finite-noise continuity statement.

The zero-sum remainder benefits from the heat spectral gap. For zero-sum w,
`lambda2(L)=(5-sqrt(5))/2>4/3`, `sqrt(5)<9/4`, and norm conversion yield

\[
 \|e^{-vA}w\|_\infty
 \le\min\{1,(9/4)e^{-v/3}\}\,\|w\|_\infty.
\]

Split the integral at 3 and use `exp(1)>8/3`. The integral of this majorant
on the positive half-line is at most
`3+(27/4)*exp(-1)<177/32<6`. Applying it to (8) gives

\[
 R_*:=\sup_{[0,T]}\|u-V\|_\infty
 \le a B_*+\beta U_*,\qquad
 a=18D,\quad\beta=6g^2T.
 \tag{9}
\]

All sampled difference vectors satisfy (2), including local time zero.
For any time up to T, its most recent sample is at most two time units
earlier. This includes the last ramp and the release tail; there is no
unobserved two-unit interval mistakenly shortened to the tail alone.
The static force in (8) has norm at most `3*B_*`. Heat contraction from that
most recent sample, followed by (8), therefore gives

\[
 U_*\le\delta+6(1+D)B_*+2g^2T U_*.
\]
\[
 c=1-2g^2T>0,\qquad
 A_*:=a+\frac{6\beta(1+D)}c,\qquad
 B_0:=\frac\beta c,\qquad
 R_*\le A_* B_*+B_0\delta.
 \tag{10}
\]

The symbols `B_*`, B and `B_0` respectively denote a scalar coefficient
norm, the incidence matrix and a scalar remainder coefficient; none is a
hidden-state observation.

## 4. Verifiable weights and the prospective diameter inequality

Retain rational proposed matrices `K_b` of shape 10-by-150 and W of shape
5-by-150. The former proposes recovery of the ten scaled phasor coefficients.
The following nonnegative quantities
are maximum absolute row sums, with interval matrix products and upper
endpoints used where M and N are enclosed:

\[
 k=\|K_b\|_\infty,\qquad w=\|W\|_\infty,
 \qquad J=[\,0_{10\times5}\ I_{10}\,]-K_bM.
\]
\[
 e_0=\|J_{[:,0:5]}\|_\infty,\quad
 e_b=\|J_{[:,5:15]}\|_\infty,\quad
 F=N-WM,
 \quad n_0=\|F_{[:,0:5]}\|_\infty,\quad
 n_b=\|F_{[:,5:15]}\|_\infty.
 \tag{11}
\]

No inverse identity is taken on trust. Since the first sample is at zero,
`||u(0)||infinity<=delta`. Applying `K_b` to `u_samples=M p+(u-V)_samples`
and retaining the residual J gives

\[
 B_*\le k(\delta+R_*)+e_0\delta+e_b B_*.
\]

If the declared assessment proves

\[
 d:=1-e_b-k A_*>0,\qquad
 Q:=\frac{k(1+B_0)+e_0}{d},
 \tag{12}
\]

then `B_*<=Q*delta` and `R_*<=(A_* Q+B_0)*delta`.
The future identity
`u(T)=W u_samples+F p+(u-V)(T)-W(u-V)_samples`
then yields

\[
 G:=w+n_0+n_b Q+(1+w)(A_*Q+B_0),
 \qquad
 \boxed{\|U_{fine}^{(1)}(216)-U_{fine}^{(2)}(216)\|_\infty
       \le\delta G+2E_0.}
 \tag{13}
\]

The fixed decision is `delta*G+2*E0<=2e-13`, in addition to every numerical,
domain and evidence obligation. An unavailable bound is not a constructed
pair whose future separation exceeds the target.

This is a theorem about the set of future states jointly compatible with
one complete record. If nonempty, its coordinate midranges have error at
most half the proved diameter. Computing that nonlinear feasible set is a
separate operational problem. In particular, W applied to a single raw
record is not automatically such a forecast: (7)-(13) bound pair differences,
not the single-history nonlinear bias.

## 5. Fixed validated computation of the static matrices

Each of the fifteen coefficient columns solves
`q'=-A*q+f_H(v)`. A first-form column starts at one basis vector with zero
forcing. A phasor column starts at zero and has forcing equal to an incidence
column times the known sine or cosine command, so `||f_H||infinity<=1`.
All exact columns obey `||q(v)||infinity<=1+v<=17`.

For each fixed cell, form the degree-16 jet of the exact local command
polynomial using the shared
[interval Taylor owner](../../src/tnfr/mathematics/_interval_taylor.py).
Its sine and cosine jets give the forcing coefficients F_j. The normalized
solution coefficients obey the linear recurrence

\[
 q_{j+1}=\frac{-Aq_j+F_j}{j+1}.
 \tag{14}
\]

Use point-center coefficients through degree 16. For the degree-17 Taylor
remainder, let F be the maximum absolute zeroth forcing coefficient over
the entire cell. Heat contraction supplies the center-started local bound
`||q||infinity<=||center||infinity+duration*F`. Use the symmetric box from
that bound as the zeroth state in the same recurrence, and let the local
clock's zeroth coefficient equal the entire cell; its first clock coefficient
is one and the others zero. The recurrence therefore encloses the required
normalized seventeenth derivative everywhere on that local solution. This
analytically derived local tube avoids an independent existence assumption.
No scalar trigonometric error is substituted for a derivative or ODE remainder.

The degree-16 polynomial plus this remainder times the cell duration to
power 17 encloses the local endpoint. Choose the polynomial's exact rational
midpoint as the next center. Add the maximum sum of polynomial radius and
absolute remainder to the carried column radius: the homogeneous heat flow
is contractive in the infinity norm, so this retains initialization and rounding errors
without box wrapping or an uncertain auxiliary clock. Check the `1e-12`
radius cap through the final radius of each column. The propagated radii
are nondecreasing, so this checks every cell and sample endpoint as well.

Both plateau endpoints are stored. Stage boundaries preserve coefficient
centers and radii, and the future matrix follows the final unforced tail.
Record all enclosures, exact grid times and each column's polynomial,
remainder and carried radius for read-only recurrence checks. Those checks
retain the shared kernel's derivative-enclosure premises; cached residuals
alone cannot establish arbitrary supplied matrix provenance. The first five initial-form
columns carry the common form mode; no projection discards it.

The midpoint normal-equation solve only proposes fixed weights. Its rank,
conditioning, rounding and backend cannot certify (12)-(13). Those statements
follow only from the retained rational `K_b` and W, outward interval products
against all coefficient enclosures, and the exact scalar comparisons.
The interval matrices do not authenticate their source; the evidence receipt
retains the frozen implementation and configuration separately.

## 6. An unavoidable common-form floor

The common form has not been fixed to zero. Translation by a constant
multiple of the all-ones vector changes neither the Laplacian, pressure
currents nor the phase row. Take the admissible exact carrier with zero
initial mean forms and equal collective phases, and compare its common
form translations by `+eta` and `-eta`. They satisfy the same complete law,
hidden bounds and initial class. Their form difference is exactly `2*eta`
at every node and time under the same neutral word.

The midpoint of their true 150-reading vectors is one common admitted
record, with opposite errors of magnitude eta at every scalar. Their future
mean-form separation is also `2*eta`. This proves a uniform diameter floor
of `2e-14`, or a forecast-radius floor of `1e-14`, without evaluating a
trajectory. It does not decide the looser declared `2e-13` target.

All conclusions concern this law, class, word, ideal clock and finite
observation allowance. The independent physical preparation and measurement
bridge remains open. The [sole execution queue](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns any subsequent work; this owner creates no parallel campaign.

<!-- FROZEN DECLARATION AND PROOF END: append evaluated results only below. -->

<a id="organized-observed-prediction-result"></a>
## Retained static certificate

The single assessment declared at
`de5b500f70b1626d75f12484af0e6381fc9d93af` certified the prospective
diameter. Its [attempt](../../docs/assets/organized_observed_prediction/attempt.json),
[exact response archive](../../docs/assets/organized_observed_prediction/response.zip)
and [content receipt](../../docs/assets/organized_observed_prediction/receipt.json)
retain all 150-by-15 observation coefficients, the future matrix, rational
weights, 2,145 compact step witnesses and source/backend associations.
There are 143 fixed cells per column. No nonlinear target history, source
producer or physical response was evaluated.

The displayed decimals summarize retained outward/exact rational quantities:

| Certificate quantity | Result |
| --- | ---: |
| Maximum coefficient radius | `4.38345956e-20`, below `1e-12` |
| Phase-weight row norm k | `499.5434372` |
| Future-weight row norm w | `3.025122144` |
| Inverse initial / phasor defects | `1.55604e-11` / `3.48244e-10` |
| Future initial / phasor defects | `6.36647e-12` / `2.01118e-11` |
| Positive bootstrap denominator d | `0.8161673611` |
| Carrier future gain G | `3.9317888454` |
| Fine future diameter upper bound | `<7.866e-14` |
| Margin below the fixed `2e-13` target | `>1.2134e-13` |

Thus the coordinate midrange of any nonempty jointly compatible future set
has radius less than `3.933e-14`. A future sensor reading adds `eta`, giving
less than `4.933e-14` against that abstract forecast. This is a uniform
conditional statement over the declared unknown-initialization class,
not only the earlier known attachment state. It neither identifies all
hidden coordinates nor selects one constitutive law.

The certified diameter is also below `1e-13`. Consequently any *certified
member* of the true feasible-future set would itself be accurate within
that policy radius; finding its exact coordinate midrange is unnecessary.
Computing and certifying such a member from a raw record is still separate
work. A fitted phase vector or the dual expression `W*y` does not discharge
that obligation. A carrier-only candidate additionally needs explicit
source/model error and record-feasibility accounting.

Read-only evidence tests verify the source associations, fixed grid,
sample/endpoint reconstruction from compact witnesses, error recurrences
and independently rebuilt dual residuals. The Taylor derivative enclosure
continues to rely on the proved shared kernel and declared forcing law;
receipts are content associations, not execution authentication. Preserve
this response and the declaration prefix without replaying the producer.
